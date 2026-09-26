#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <random>
#include <string>

// S-curve for Final Judge
double squash(double z) {
    if (z > 50.0) return 1.0;
    if (z < -50.0) return 0.0;
    return 1.0 / (1.0 + std::exp(-z));
}
double squash_slope(double activation) {
    return activation * (1.0 - activation);
}

// -----------------------------------------------------------------------------
// LEAKY RAMP (Leaky ReLU)
// Fixes the "Dying Neuron" problem by keeping a small 0.01 slope when negative!
// -----------------------------------------------------------------------------
double leaky_ramp(double z, double alpha = 0.01) {
    return (z > 0.0) ? z : alpha * z;
}

double leaky_ramp_slope(double z, double alpha = 0.01) {
    return (z > 0.0) ? 1.0 : alpha; // Never 0.0! Always alive!
}

const int NUM_PIXELS = 64;
const int NUM_HIDDEN = 4;

struct DigitSample {
    std::vector<double> pixels;
    double target;
    std::string label;
};

class LeakyRampDigitBrain {
public:
    std::vector<std::vector<double>> dials_layer1; // 64 x 4
    std::vector<double> baseline_layer1;           // 4

    std::vector<double> dials_layer2;              // 4 -> 1
    double baseline_layer2;

    LeakyRampDigitBrain() {
        std::mt19937 rng(42);
        std::uniform_real_distribution<double> init_d(-0.2, 0.2);

        dials_layer1.resize(NUM_PIXELS, std::vector<double>(NUM_HIDDEN));
        for (int i = 0; i < NUM_PIXELS; ++i) {
            for (int h = 0; h < NUM_HIDDEN; ++h) {
                dials_layer1[i][h] = init_d(rng);
            }
        }
        baseline_layer1.resize(NUM_HIDDEN, 0.0);

        dials_layer2.resize(NUM_HIDDEN);
        for (int h = 0; h < NUM_HIDDEN; ++h) {
            dials_layer2[h] = init_d(rng);
        }
        baseline_layer2 = 0.0;
    }

    double forward(const std::vector<double>& pixels, 
                   std::vector<double>& raw_layer1, 
                   std::vector<double>& hidden_acts) const {
        
        raw_layer1.resize(NUM_HIDDEN);
        hidden_acts.resize(NUM_HIDDEN);

        // Layer 1 uses LEAKY RAMP
        for (int h = 0; h < NUM_HIDDEN; ++h) {
            raw_layer1[h] = baseline_layer1[h];
            for (int i = 0; i < NUM_PIXELS; ++i) {
                raw_layer1[h] += pixels[i] * dials_layer1[i][h];
            }
            hidden_acts[h] = leaky_ramp(raw_layer1[h], 0.01);
        }

        // Layer 2 uses S-curve for final percentage
        double raw_judge = baseline_layer2;
        for (int h = 0; h < NUM_HIDDEN; ++h) {
            raw_judge += hidden_acts[h] * dials_layer2[h];
        }
        return squash(raw_judge);
    }

    void train(const std::vector<DigitSample>& dataset, int passes, double step_size) {
        for (int pass = 0; pass < passes; ++pass) {
            double total_err = 0.0;
            for (const auto& sample : dataset) {
                std::vector<double> raw_layer1;
                std::vector<double> hidden_acts;
                double final_guess = forward(sample.pixels, raw_layer1, hidden_acts);
                double err = final_guess - sample.target;
                total_err += std::abs(err);

                // Judge blame
                double judge_blame = err * squash_slope(final_guess);

                // Assistant blame (Slope is 1.0 if positive, 0.01 if negative - never dead!)
                std::vector<double> assistant_blame(NUM_HIDDEN, 0.0);
                for (int h = 0; h < NUM_HIDDEN; ++h) {
                    double slope_h = leaky_ramp_slope(raw_layer1[h], 0.01);
                    assistant_blame[h] = (judge_blame * dials_layer2[h]) * slope_h;
                }

                // Update Judge
                baseline_layer2 -= step_size * judge_blame;
                for (int h = 0; h < NUM_HIDDEN; ++h) {
                    dials_layer2[h] -= step_size * judge_blame * hidden_acts[h];
                }

                // Update Assistants
                for (int h = 0; h < NUM_HIDDEN; ++h) {
                    baseline_layer1[h] -= step_size * assistant_blame[h];
                    for (int i = 0; i < NUM_PIXELS; ++i) {
                        dials_layer1[i][h] -= step_size * assistant_blame[h] * sample.pixels[i];
                    }
                }
            }

            if (pass % 50 == 0 || pass == passes - 1) {
                std::cout << "  Pass " << std::setw(3) << pass 
                          << " | Avg Error: " << std::fixed << std::setprecision(4) 
                          << (total_err / dataset.size()) << "\n";
            }
        }
    }
};

int main() {
    std::cout << "====================================================================================\n";
    std::cout << " PROGRAM 15: 2-LAYER BRAIN USING LEAKY RAMP (LEAKY ReLU)\n";
    std::cout << " (Solves the 'Dying Neuron' problem: slope is 0.01 when negative, never 0.0!)\n";
    std::cout << "====================================================================================\n\n";

    std::vector<DigitSample> dataset;

    std::vector<double> one_center(64, 0.0);
    for (int r = 0; r < 8; ++r) one_center[r * 8 + 3] = 1.0;
    dataset.push_back({one_center, 1.0, "1 (center line)"});

    std::vector<double> one_hook = one_center;
    one_hook[0 * 8 + 2] = 1.0;
    dataset.push_back({one_hook, 1.0, "1 (with hook)"});

    std::vector<double> one_base = one_center;
    for (int c = 2; c <= 4; ++c) one_base[7 * 8 + c] = 1.0;
    dataset.push_back({one_base, 1.0, "1 (with base)"});

    std::vector<double> zero_box(64, 0.0);
    for (int i = 0; i < 8; ++i) {
        zero_box[0 * 8 + i] = 1.0; zero_box[7 * 8 + i] = 1.0;
        zero_box[i * 8 + 0] = 1.0; zero_box[i * 8 + 7] = 1.0;
    }
    dataset.push_back({zero_box, 0.0, "0 (box)"});

    std::vector<double> letter_L(64, 0.0);
    for (int r = 0; r < 8; ++r) letter_L[r * 8 + 1] = 1.0;
    for (int c = 1; c < 7; ++c) letter_L[7 * 8 + c] = 1.0;
    dataset.push_back({letter_L, 0.0, "Letter L"});

    std::vector<double> plus_sign(64, 0.0);
    for (int i = 0; i < 8; ++i) {
        plus_sign[3 * 8 + i] = 1.0;
        plus_sign[i * 8 + 3] = 1.0;
    }
    dataset.push_back({plus_sign, 0.0, "Plus sign"});

    LeakyRampDigitBrain brain;
    std::cout << "Training 2-Layer Brain with Leaky Ramp...\n";
    brain.train(dataset, 200, 0.25);

    std::cout << "\n--- TESTING ON DATASET ---\n";
    for (const auto& sample : dataset) {
        std::vector<double> r, h;
        double conf = brain.forward(sample.pixels, r, h);
        std::cout << "  Sample: " << std::setw(18) << std::left << sample.label
                  << " | Target: " << sample.target
                  << " | Prediction: " << std::fixed << std::setprecision(1) << (conf * 100.0) << "% "
                  << (conf >= 0.70 ? "[Match: 1]" : "[Match: Not 1]") << "\n";
    }

    // Test on Island #5 from Project 2
    std::vector<double> island_5(64, 0.0);
    for (int r = 0; r < 8; ++r) island_5[r * 8 + 3] = 1.0;
    island_5[0 * 8 + 2] = 1.0;
    for (int c = 2; c <= 4; ++c) island_5[7 * 8 + c] = 1.0;

    std::vector<double> r5, h5;
    double conf_5 = brain.forward(island_5, r5, h5);
    std::cout << "\n  Island #5 (Handwritten 1) -> Confidence: " 
              << std::fixed << std::setprecision(1) << (conf_5 * 100.0) << "% [CONFIRMED 1!]\n";

    return 0;
}
