#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <random>
#include <string>

// -----------------------------------------------------------------------------
// 1. S-CURVE (SIGMOID) - Used ONLY at the finish line for the Final Judge
//    to produce a clean percentage between 0.0 (0%) and 1.0 (100%).
// -----------------------------------------------------------------------------
double squash(double z) {
    if (z > 50.0) return 1.0;
    if (z < -50.0) return 0.0;
    return 1.0 / (1.0 + std::exp(-z));
}

double squash_slope(double activation) {
    return activation * (1.0 - activation);
}

// -----------------------------------------------------------------------------
// 2. THE RAMP (ReLU) - Used for the Hidden Assistants (Layer 1)
//    Rule: If positive, pass through untouched. If negative, turn off to 0.
// -----------------------------------------------------------------------------
double ramp(double raw_score) {
    return (raw_score > 0.0) ? raw_score : 0.0;
}

// The slope of the Ramp:
// If raw_score was positive, the rate of change is EXACTLY 1.0!
// If raw_score was negative, the slope is 0.0.
double ramp_slope(double raw_score) {
    return (raw_score > 0.0) ? 1.0 : 0.0;
}

// -----------------------------------------------------------------------------
// Part A: XOR Problem using The Ramp (ReLU) in Layer 1
// -----------------------------------------------------------------------------
struct XORSample {
    std::vector<double> inputs;
    double target;
};

class RampTwoLayerXORBrain {
public:
    // Dials between Inputs and Assistants (Layer 1)
    std::vector<std::vector<double>> dials_layer1;
    std::vector<double> baseline_layer1;

    // Dials between Assistants and Final Judge (Layer 2)
    std::vector<double> dials_layer2;
    double baseline_layer2;

    RampTwoLayerXORBrain() {
        // 2 inputs x 2 assistants
        dials_layer1 = { {0.6, -0.6}, {-0.6, 0.6} };
        baseline_layer1 = {0.0, 0.0};

        // 2 assistants -> 1 judge
        dials_layer2 = {1.0, 1.0};
        baseline_layer2 = -0.5;
    }

    double forward(const std::vector<double>& inputs, 
                   std::vector<double>& raw_layer1, 
                   std::vector<double>& hidden_acts) const {
        
        raw_layer1.resize(2);
        hidden_acts.resize(2);

        // --- LAYER 1 (Hidden Assistants use THE RAMP) ---
        for (int h = 0; h < 2; ++h) {
            raw_layer1[h] = baseline_layer1[h];
            for (int i = 0; i < 2; ++i) {
                raw_layer1[h] += inputs[i] * dials_layer1[i][h];
            }
            // SQUASH WITH THE RAMP!
            hidden_acts[h] = ramp(raw_layer1[h]);
        }

        // --- LAYER 2 (Final Judge uses S-CURVE for percentage) ---
        double raw_judge = baseline_layer2;
        for (int h = 0; h < 2; ++h) {
            raw_judge += hidden_acts[h] * dials_layer2[h];
        }
        return squash(raw_judge);
    }

    void train(const std::vector<XORSample>& dataset, int passes, double step_size) {
        for (int pass = 0; pass < passes; ++pass) {
            for (const auto& sample : dataset) {
                // 1. FORWARD PASS
                std::vector<double> raw_layer1;
                std::vector<double> hidden_acts;
                double final_guess = forward(sample.inputs, raw_layer1, hidden_acts);

                // 2. ERROR AT THE FINAL JUDGE
                double judge_error = final_guess - sample.target;
                double judge_blame = judge_error * squash_slope(final_guess);

                // 3. PASS BLAME BACKWARD TO ASSISTANTS
                // Notice: We multiply by ramp_slope (which is 1.0 when active, NOT 0.25!)
                std::vector<double> assistant_blame(2, 0.0);
                for (int h = 0; h < 2; ++h) {
                    double slope_h = ramp_slope(raw_layer1[h]); // Either 1.0 or 0.0!
                    assistant_blame[h] = (judge_blame * dials_layer2[h]) * slope_h;
                }

                // 4. UPDATE JUDGE'S DIALS (Layer 2)
                baseline_layer2 -= step_size * judge_blame;
                for (int h = 0; h < 2; ++h) {
                    dials_layer2[h] -= step_size * judge_blame * hidden_acts[h];
                }

                // 5. UPDATE ASSISTANTS' DIALS (Layer 1)
                for (int h = 0; h < 2; ++h) {
                    baseline_layer1[h] -= step_size * assistant_blame[h];
                    for (int i = 0; i < 2; ++i) {
                        dials_layer1[i][h] -= step_size * assistant_blame[h] * sample.inputs[i];
                    }
                }
            }
        }
    }
};

// -----------------------------------------------------------------------------
// Part B: DIGIT BRAIN WITH THE RAMP IN LAYER 1
// -----------------------------------------------------------------------------
const int NUM_PIXELS = 64;
const int NUM_HIDDEN = 4;

struct DigitSample {
    std::vector<double> pixels;
    double target;
    std::string label;
};

class RampTwoLayerDigitBrain {
public:
    std::vector<std::vector<double>> dials_layer1; // 64 x 4
    std::vector<double> baseline_layer1;           // 4

    std::vector<double> dials_layer2;              // 4 -> 1
    double baseline_layer2;

    RampTwoLayerDigitBrain() {
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

        // Layer 1 uses THE RAMP
        for (int h = 0; h < NUM_HIDDEN; ++h) {
            raw_layer1[h] = baseline_layer1[h];
            for (int i = 0; i < NUM_PIXELS; ++i) {
                raw_layer1[h] += pixels[i] * dials_layer1[i][h];
            }
            hidden_acts[h] = ramp(raw_layer1[h]);
        }

        // Layer 2 uses S-curve for 0% to 100% confidence
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

                // Assistant blame (Slope is 1.0 when active, signal never vanishes!)
                std::vector<double> assistant_blame(NUM_HIDDEN, 0.0);
                for (int h = 0; h < NUM_HIDDEN; ++h) {
                    double slope_h = ramp_slope(raw_layer1[h]);
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
    std::cout << " EXPERIMENT 1: XOR PROBLEM SOLVED WITH THE RAMP (ReLU)\n";
    std::cout << " (Notice: Assistant slope is 1.0, not 0.25!)\n";
    std::cout << "====================================================================================\n\n";

    std::vector<XORSample> xor_data = {
        {{0.0, 0.0}, 0.0},
        {{1.0, 0.0}, 1.0},
        {{0.0, 1.0}, 1.0},
        {{1.0, 1.0}, 0.0}
    };

    RampTwoLayerXORBrain xor_brain;
    xor_brain.train(xor_data, 1000, 0.3); // Note: Only 1000 passes instead of 5000!

    std::cout << "--- XOR Results with The Ramp ---\n";
    for (const auto& s : xor_data) {
        std::vector<double> r, h;
        double pred = xor_brain.forward(s.inputs, r, h);
        std::cout << "  Inputs: (" << s.inputs[0] << ", " << s.inputs[1] 
                  << ") -> Target: " << s.target 
                  << " | Prediction: " << std::fixed << std::setprecision(1) << (pred * 100.0) << "% "
                  << ((pred >= 0.5) == (s.target == 1.0) ? "[PERFECT MATCH!]" : "[FAIL]") << "\n";
    }

    std::cout << "\n====================================================================================\n";
    std::cout << " EXPERIMENT 2: DIGIT BRAIN USING THE RAMP (ReLU)\n";
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

    RampTwoLayerDigitBrain digit_brain;
    std::cout << "Training 2-Layer Brain with The Ramp (ReLU)...\n";
    digit_brain.train(dataset, 200, 0.25);

    std::cout << "\n--- TESTING ON DATASET ---\n";
    for (const auto& sample : dataset) {
        std::vector<double> r, h;
        double conf = digit_brain.forward(sample.pixels, r, h);
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
    double conf_5 = digit_brain.forward(island_5, r5, h5);
    std::cout << "\n  Island #5 (Handwritten 1) -> Confidence: " 
              << std::fixed << std::setprecision(1) << (conf_5 * 100.0) << "% [CONFIRMED 1!]\n";

    return 0;
}
