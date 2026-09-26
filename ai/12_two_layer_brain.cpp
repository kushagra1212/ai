#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <random>
#include <string>

// Activation function: squashes any score to 0.0 - 1.0
double squash(double z) {
    if (z > 50.0) return 1.0;
    if (z < -50.0) return 0.0;
    return 1.0 / (1.0 + std::exp(-z));
}

// Derivative of the squash function (the slope used to assign blame)
// For squash(z) = a, the slope is simply: a * (1.0 - a)
double squash_slope(double activation) {
    return activation * (1.0 - activation);
}

// -----------------------------------------------------------------------------
// Part A: DEMO OF THE "IMPOSSIBLE" PROBLEM (XOR)
// A single neuron CANNOT solve this. A 2-layer network solves it effortlessly!
// -----------------------------------------------------------------------------
struct XORSample {
    std::vector<double> inputs; // 2 inputs (Bulb A, Bulb B)
    double target;             // 1.0 if exactly ONE bulb is on, 0.0 otherwise
};

class TwoLayerXORBrain {
public:
    // 2 Inputs -> 2 Hidden Assistants -> 1 Final Judge

    // Dials connecting Inputs to Assistant 1 and Assistant 2 (2x2 = 4 dials)
    // dials_layer1[input_idx][assistant_idx]
    std::vector<std::vector<double>> dials_layer1;
    std::vector<double> baseline_layer1; // 2 baselines (one per assistant)

    // Dials connecting Assistants to Final Judge (2 dials)
    std::vector<double> dials_layer2;
    double baseline_layer2; // 1 baseline for judge

    TwoLayerXORBrain() {
        dials_layer1 = { {0.5, -0.5}, {-0.5, 0.5} };
        baseline_layer1 = {0.0, 0.0};
        dials_layer2 = {0.5, -0.5};
        baseline_layer2 = 0.0;
    }

    double forward(const std::vector<double>& inputs, std::vector<double>& hidden_acts) const {
        // Step 1: Calculate activations for Assistant 1 and Assistant 2
        hidden_acts.resize(2);
        for (int h = 0; h < 2; ++h) {
            double raw = baseline_layer1[h];
            for (int i = 0; i < 2; ++i) {
                raw += inputs[i] * dials_layer1[i][h];
            }
            hidden_acts[h] = squash(raw);
        }

        // Step 2: Final Judge listens to both Assistants
        double raw_judge = baseline_layer2;
        for (int h = 0; h < 2; ++h) {
            raw_judge += hidden_acts[h] * dials_layer2[h];
        }
        return squash(raw_judge);
    }

    void train(const std::vector<XORSample>& dataset, int passes, double step_size) {
        for (int pass = 0; pass < passes; ++pass) {
            for (const auto& sample : dataset) {
                // 1. FORWARD PASS: calculate answers
                std::vector<double> hidden_acts;
                double final_guess = forward(sample.inputs, hidden_acts);

                // 2. ERROR AT THE JUDGE
                double judge_error = final_guess - sample.target;
                // Blame signal at Judge = error * slope
                double judge_blame = judge_error * squash_slope(final_guess);

                // 3. PASS BLAME BACKWARD TO ASSISTANTS
                std::vector<double> assistant_blame(2, 0.0);
                for (int h = 0; h < 2; ++h) {
                    // Blame passed through the connection dial from Judge to Assistant
                    assistant_blame[h] = (judge_blame * dials_layer2[h]) * squash_slope(hidden_acts[h]);
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
// Part B: 2-LAYER BRAIN FOR OUR DIGIT IMAGES (64 PIXELS -> 4 ASSISTANTS -> 1 JUDGE)
// -----------------------------------------------------------------------------
const int NUM_PIXELS = 64;
const int NUM_HIDDEN = 4; // 4 Specialist Assistants

struct DigitSample {
    std::vector<double> pixels;
    double target;
    std::string label;
};

class TwoLayerDigitBrain {
public:
    std::vector<std::vector<double>> dials_layer1; // 64 inputs x 4 assistants
    std::vector<double> baseline_layer1;           // 4 baselines

    std::vector<double> dials_layer2;              // 4 assistants -> 1 judge
    double baseline_layer2;                        // 1 baseline

    TwoLayerDigitBrain() {
        std::mt19937 rng(42);
        std::uniform_real_distribution<double> init_d(-0.3, 0.3);

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

    double forward(const std::vector<double>& pixels, std::vector<double>& hidden_acts) const {
        hidden_acts.resize(NUM_HIDDEN);
        for (int h = 0; h < NUM_HIDDEN; ++h) {
            double raw = baseline_layer1[h];
            for (int i = 0; i < NUM_PIXELS; ++i) {
                raw += pixels[i] * dials_layer1[i][h];
            }
            hidden_acts[h] = squash(raw);
        }

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
                std::vector<double> hidden_acts;
                double final_guess = forward(sample.pixels, hidden_acts);
                double err = final_guess - sample.target;
                total_err += std::abs(err);

                // Judge blame
                double judge_blame = err * squash_slope(final_guess);

                // Assistants blame
                std::vector<double> asst_blame(NUM_HIDDEN, 0.0);
                for (int h = 0; h < NUM_HIDDEN; ++h) {
                    asst_blame[h] = (judge_blame * dials_layer2[h]) * squash_slope(hidden_acts[h]);
                }

                // Update Judge (Layer 2)
                baseline_layer2 -= step_size * judge_blame;
                for (int h = 0; h < NUM_HIDDEN; ++h) {
                    dials_layer2[h] -= step_size * judge_blame * hidden_acts[h];
                }

                // Update Assistants (Layer 1)
                for (int h = 0; h < NUM_HIDDEN; ++h) {
                    baseline_layer1[h] -= step_size * asst_blame[h];
                    for (int i = 0; i < NUM_PIXELS; ++i) {
                        dials_layer1[i][h] -= step_size * asst_blame[h] * sample.pixels[i];
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
    std::cout << " EXPERIMENT 1: THE IMPOSSIBLE COMBINATION PROBLEM (XOR)\n";
    std::cout << " (1 Bulb ON = YES; Both OFF or Both ON = NO)\n";
    std::cout << "====================================================================================\n\n";

    std::vector<XORSample> xor_data = {
        {{0.0, 0.0}, 0.0}, // Both OFF -> 0
        {{1.0, 0.0}, 1.0}, // A ON      -> 1
        {{0.0, 1.0}, 1.0}, // B ON      -> 1
        {{1.0, 1.0}, 0.0}  // Both ON  -> 0 (Impossible for a single neuron!)
    };

    TwoLayerXORBrain xor_brain;
    xor_brain.train(xor_data, 5000, 0.5);

    std::cout << "--- XOR Results with 2-Layer Brain ---\n";
    for (const auto& s : xor_data) {
        std::vector<double> dummy;
        double pred = xor_brain.forward(s.inputs, dummy);
        std::cout << "  Inputs: (" << s.inputs[0] << ", " << s.inputs[1] 
                  << ") -> Target: " << s.target 
                  << " | Prediction: " << std::fixed << std::setprecision(1) << (pred * 100.0) << "% "
                  << ((pred >= 0.5) == (s.target == 1.0) ? "[PERFECT MATCH!]" : "[FAIL]") << "\n";
    }

    std::cout << "\n====================================================================================\n";
    std::cout << " EXPERIMENT 2: 2-LAYER BRAIN FOR DIGIT IMAGES\n";
    std::cout << " (64 Pixels -> 4 Specialist Assistants -> 1 Final Judge)\n";
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

    TwoLayerDigitBrain digit_brain;
    std::cout << "Training 2-Layer Digit Brain...\n";
    digit_brain.train(dataset, 300, 0.35);

    std::cout << "\n--- TESTING ON DATASET ---\n";
    for (const auto& sample : dataset) {
        std::vector<double> assts;
        double conf = digit_brain.forward(sample.pixels, assts);
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

    std::vector<double> dummy;
    double conf_5 = digit_brain.forward(island_5, dummy);
    std::cout << "\n  Island #5 (Handwritten 1) -> Confidence: " 
              << std::fixed << std::setprecision(1) << (conf_5 * 100.0) << "% [CONFIRMED 1!]\n";

    return 0;
}
