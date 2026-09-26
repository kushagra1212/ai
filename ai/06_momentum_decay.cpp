#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <string>

const int GRID_SIZE = 8;
const int NUM_PIXELS = 64;

struct Sample {
    std::vector<double> pixels;
    double target;
    std::string label;
};

double squash(double z) {
    if (z > 50.0) return 1.0;
    if (z < -50.0) return 0.0;
    return 1.0 / (1.0 + std::exp(-z));
}

// Visual bar showing accuracy (out of 20 chars)
std::string render_progress_bar(double error) {
    // 0 error = 100% accuracy = 20 hashes [####################]
    double accuracy = std::max(0.0, 1.0 - error * 2.0);
    int filled = static_cast<int>(accuracy * 20);
    if (filled > 20) filled = 20;
    std::string bar = "[";
    for (int i = 0; i < 20; ++i) {
        bar += (i < filled) ? "#" : ".";
    }
    bar += "]";
    return bar;
}

class MomentumDecayLearner {
public:
    std::vector<double> dials;
    double baseline;

    std::vector<double> velocity_dials;
    double velocity_baseline;

    MomentumDecayLearner() {
        dials = std::vector<double>(NUM_PIXELS, 0.0);
        baseline = 0.0;
        velocity_dials = std::vector<double>(NUM_PIXELS, 0.0);
        velocity_baseline = 0.0;
    }

    double predict(const std::vector<double>& pixels) const {
        double raw = baseline;
        for (int i = 0; i < NUM_PIXELS; ++i) {
            raw += pixels[i] * dials[i];
        }
        return squash(raw);
    }

    void train(const std::vector<Sample>& dataset, int total_passes, 
               double initial_step, double decay_factor, double friction) {
        
        std::cout << "------------------------------------------------------------------------------------\n";
        std::cout << " Pass | Current Step | Avg Speed | Avg Error | Accuracy Progress Bar\n";
        std::cout << "------------------------------------------------------------------------------------\n";

        for (int pass = 0; pass < total_passes; ++pass) {
            // 1. CALCULATE CURRENT STEP SIZE (Decay Formula: shrinks over time)
            double current_step = initial_step / (1.0 + decay_factor * pass);

            double total_error = 0.0;

            for (const auto& sample : dataset) {
                double guess = predict(sample.pixels);
                double diff = guess - sample.target;
                total_error += std::abs(diff);

                // 2. MOMENTUM UPDATE (Accelerating Ball Formula)
                // Update baseline speed & position
                velocity_baseline = friction * velocity_baseline - current_step * diff;
                baseline += velocity_baseline;

                // Update pixel dials speed & position
                for (int i = 0; i < NUM_PIXELS; ++i) {
                    if (sample.pixels[i] > 0.0) {
                        velocity_dials[i] = friction * velocity_dials[i] - current_step * diff * sample.pixels[i];
                        dials[i] += velocity_dials[i];
                    }
                }
            }

            double avg_error = total_error / dataset.size();

            // Calculate average speed of dials
            double total_speed = std::abs(velocity_baseline);
            for (double v : velocity_dials) total_speed += std::abs(v);
            double avg_speed = total_speed / (NUM_PIXELS + 1);

            // Log every 5 passes or on completion
            if (pass % 5 == 0 || pass == total_passes - 1 || avg_error < 0.001) {
                std::cout << "  " << std::setw(3) << pass << " | "
                          << std::fixed << std::setprecision(4) << current_step << "       | "
                          << std::fixed << std::setprecision(4) << avg_speed << "    | "
                          << std::fixed << std::setprecision(4) << avg_error << "    | "
                          << render_progress_bar(avg_error) << "\n";
            }

            // Early stop if error is near 0
            if (avg_error < 0.0005 && pass >= 15) {
                std::cout << "------------------------------------------------------------------------------------\n";
                std::cout << ">>> Converged early at Pass " << pass << "! Error reached near zero. <<<\n\n";
                break;
            }
        }
    }

    void print_dials() const {
        std::cout << "--- Visual Map of Learned Dials ---\n";
        std::cout << "Legend: [ + ] = ink here means 'YES, it is a 1'\n";
        std::cout << "        [ - ] = ink here means 'NO, it is NOT a 1'\n\n";
        for (int r = 0; r < GRID_SIZE; ++r) {
            for (int c = 0; c < GRID_SIZE; ++c) {
                double val = dials[r * GRID_SIZE + c];
                if (val > 0.8)       std::cout << " +++ ";
                else if (val > 0.2)  std::cout << "  +  ";
                else if (val < -0.8) std::cout << " --- ";
                else if (val < -0.2) std::cout << "  -  ";
                else                 std::cout << "  .  ";
            }
            std::cout << "\n";
        }
        std::cout << "Baseline value: " << baseline << "\n\n";
    }
};

int main() {
    std::cout << "====================================================================================\n";
    std::cout << " PROGRAM A: COMBINING MOMENTUM (SPEED) + DECAY (SLOW DOWN AS YOU ARRIVE)\n";
    std::cout << "====================================================================================\n\n";

    // Dataset: Vertical digits "1" vs Non-"1" shapes
    std::vector<Sample> dataset;

    // 1: Center vertical line
    std::vector<double> one_center(64, 0.0);
    for (int r = 0; r < 8; ++r) one_center[r * 8 + 3] = 1.0;
    dataset.push_back({one_center, 1.0, "1 (center line)"});

    // 1: Center vertical line with hook
    std::vector<double> one_hook = one_center;
    one_hook[0 * 8 + 2] = 1.0;
    dataset.push_back({one_hook, 1.0, "1 (with hook)"});

    // 1: Center vertical line with base
    std::vector<double> one_base = one_center;
    for (int c = 2; c <= 4; ++c) one_base[7 * 8 + c] = 1.0;
    dataset.push_back({one_base, 1.0, "1 (with base)"});

    // 0: Outer border box
    std::vector<double> zero_box(64, 0.0);
    for (int i = 0; i < 8; ++i) {
        zero_box[0 * 8 + i] = 1.0; zero_box[7 * 8 + i] = 1.0;
        zero_box[i * 8 + 0] = 1.0; zero_box[i * 8 + 7] = 1.0;
    }
    dataset.push_back({zero_box, 0.0, "0 (box)"});

    // L: Letter L
    std::vector<double> letter_L(64, 0.0);
    for (int r = 0; r < 8; ++r) letter_L[r * 8 + 1] = 1.0;
    for (int c = 1; c < 7; ++c) letter_L[7 * 8 + c] = 1.0;
    dataset.push_back({letter_L, 0.0, "Letter L"});

    // +: Plus sign
    std::vector<double> plus_sign(64, 0.0);
    for (int i = 0; i < 8; ++i) {
        plus_sign[3 * 8 + i] = 1.0;
        plus_sign[i * 8 + 3] = 1.0;
    }
    dataset.push_back({plus_sign, 0.0, "Plus sign"});

    MomentumDecayLearner learner;

    // Train with Initial Step = 0.40, Decay Factor = 0.05, Friction = 0.80
    learner.train(dataset, 60, 0.40, 0.05, 0.80);

    learner.print_dials();

    // Verify on a test sample
    std::cout << "--- TESTING SAMPLES ---\n";
    for (const auto& sample : dataset) {
        double conf = learner.predict(sample.pixels);
        std::cout << "  Sample: " << std::setw(18) << std::left << sample.label
                  << " | Target: " << sample.target
                  << " | Prediction: " << std::fixed << std::setprecision(1) << (conf * 100.0) << "% "
                  << (conf >= 0.70 ? "[Match: 1]" : "[Match: Not 1]") << "\n";
    }

    return 0;
}
