#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

const int NUM_PIXELS = 64;

struct Sample {
    std::vector<double> pixels;
    double target;
};

double squash(double z) {
    return 1.0 / (1.0 + std::exp(-z));
}

// -------------------------------------------------------------
// Method A: Fixed Step Size (What we used before)
// -------------------------------------------------------------
void train_fixed(std::vector<double>& dials, double& baseline, 
                 const std::vector<Sample>& dataset, int passes, double step_size) {
    for (int pass = 0; pass < passes; ++pass) {
        for (const auto& s : dataset) {
            double raw = baseline;
            for (int i = 0; i < NUM_PIXELS; ++i) raw += s.pixels[i] * dials[i];
            double diff = squash(raw) - s.target;

            baseline -= step_size * diff;
            for (int i = 0; i < NUM_PIXELS; ++i) {
                if (s.pixels[i] > 0.0) dials[i] -= step_size * diff * s.pixels[i];
            }
        }
    }
}

// -------------------------------------------------------------
// Method B: Dynamic Step Size (Decays over time)
// Starts with big steps to cover ground fast, shrinks to tiny steps for accuracy
// -------------------------------------------------------------
void train_decay(std::vector<double>& dials, double& baseline, 
                 const std::vector<Sample>& dataset, int passes, double initial_step) {
    for (int pass = 0; pass < passes; ++pass) {
        // Formula: Step size shrinks as pass number grows
        double current_step = initial_step / (1.0 + 0.05 * pass);

        for (const auto& s : dataset) {
            double raw = baseline;
            for (int i = 0; i < NUM_PIXELS; ++i) raw += s.pixels[i] * dials[i];
            double diff = squash(raw) - s.target;

            baseline -= current_step * diff;
            for (int i = 0; i < NUM_PIXELS; ++i) {
                if (s.pixels[i] > 0.0) dials[i] -= current_step * diff * s.pixels[i];
            }
        }
    }
}

// -------------------------------------------------------------
// Method C: Momentum (Rolling Bowling Ball)
// Remembers previous step velocity so it accelerates in steady directions
// -------------------------------------------------------------
void train_momentum(std::vector<double>& dials, double& baseline, 
                    const std::vector<Sample>& dataset, int passes, double step_size) {
    std::vector<double> velocity_dials(NUM_PIXELS, 0.0);
    double velocity_baseline = 0.0;
    const double friction = 0.85; // Retains 85% of previous velocity

    for (int pass = 0; pass < passes; ++pass) {
        for (const auto& s : dataset) {
            double raw = baseline;
            for (int i = 0; i < NUM_PIXELS; ++i) raw += s.pixels[i] * dials[i];
            double diff = squash(raw) - s.target;

            // Accelerate velocity
            velocity_baseline = friction * velocity_baseline - step_size * diff;
            baseline += velocity_baseline;

            for (int i = 0; i < NUM_PIXELS; ++i) {
                if (s.pixels[i] > 0.0) {
                    velocity_dials[i] = friction * velocity_dials[i] - step_size * diff * s.pixels[i];
                    dials[i] += velocity_dials[i];
                }
            }
        }
    }
}

// Helper to evaluate average error
double eval_error(const std::vector<double>& dials, double baseline, const std::vector<Sample>& dataset) {
    double total = 0.0;
    for (const auto& s : dataset) {
        double raw = baseline;
        for (int i = 0; i < NUM_PIXELS; ++i) raw += s.pixels[i] * dials[i];
        total += std::abs(squash(raw) - s.target);
    }
    return total / dataset.size();
}

int main() {
    std::cout << "=== COMPARING TRAINING METHODS (At only 50 passes!) ===\n\n";

    // Simple test dataset (vertical line vs hollow box)
    std::vector<Sample> dataset;
    // 1: vertical line
    std::vector<double> one(64, 0.0);
    for (int r = 0; r < 8; ++r) one[r * 8 + 4] = 1.0;
    dataset.push_back({one, 1.0});

    // 0: border box
    std::vector<double> zero(64, 0.0);
    for (int i = 0; i < 8; ++i) {
        zero[0 * 8 + i] = 1.0; zero[7 * 8 + i] = 1.0;
        zero[i * 8 + 0] = 1.0; zero[i * 8 + 7] = 1.0;
    }
    dataset.push_back({zero, 0.0});

    int test_passes = 50; // A small number of passes!

    // Test A: Fixed Step
    std::vector<double> dials_A(64, 0.0); double base_A = 0.0;
    train_fixed(dials_A, base_A, dataset, test_passes, 0.15);
    std::cout << "Method A [Fixed Step Size 0.15]      | Error after 50 passes: " 
              << std::fixed << std::setprecision(5) << eval_error(dials_A, base_A, dataset) << "\n";

    // Test B: Decaying Step (starts bigger, shrinks)
    std::vector<double> dials_B(64, 0.0); double base_B = 0.0;
    train_decay(dials_B, base_B, dataset, test_passes, 0.40);
    std::cout << "Method B [Decaying Step Size]        | Error after 50 passes: " 
              << std::fixed << std::setprecision(5) << eval_error(dials_B, base_B, dataset) << "\n";

    // Test C: Momentum (accelerating ball)
    std::vector<double> dials_C(64, 0.0); double base_C = 0.0;
    train_momentum(dials_C, base_C, dataset, test_passes, 0.15);
    std::cout << "Method C [Momentum (Accelerating)]   | Error after 50 passes: " 
              << std::fixed << std::setprecision(5) << eval_error(dials_C, base_C, dataset) << "\n";

    return 0;
}
