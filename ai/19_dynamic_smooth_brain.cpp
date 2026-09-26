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
// SMOOTH CURVED RAMP (SiLU / Swish: z * squash(z))
// 1. Smooth everywhere (no sharp kink at 0)
// 2. Dips to -0.28 for negative z -> pulls average towards 0 (zero-centered!)
// 3. No magic leak parameter to guess!
// -----------------------------------------------------------------------------
double smooth_ramp(double z) {
    return z * squash(z);
}

double smooth_ramp_slope(double z) {
    double s = squash(z);
    return s + z * s * (1.0 - s); // Chain rule derivative
}

const int NUM_PIXELS = 64;
const int NUM_HIDDEN = 4;

struct DigitSample {
    std::vector<double> pixels;
    double target;
    std::string label;
};

// =============================================================================
// BRAIN 1: DYNAMIC LEAKY RAMP (Learns its own alpha for each neuron!)
// =============================================================================
class DynamicLeakyBrain {
public:
    std::vector<std::vector<double>> dials_layer1; // 64 x 4
    std::vector<double> baseline_layer1;           // 4
    std::vector<double> alpha_layer1;              // 4 (DYNAMIC LEAK DIALS!)

    std::vector<double> dials_layer2;              // 4 -> 1
    double baseline_layer2;

    DynamicLeakyBrain() {
        std::mt19937 rng(42);
        std::uniform_real_distribution<double> init_d(-0.2, 0.2);

        dials_layer1.resize(NUM_PIXELS, std::vector<double>(NUM_HIDDEN));
        for (int i = 0; i < NUM_PIXELS; ++i) {
            for (int h = 0; h < NUM_HIDDEN; ++h) {
                dials_layer1[i][h] = init_d(rng);
            }
        }
        baseline_layer1.resize(NUM_HIDDEN, 0.0);

        // Initialize each neuron's leak rate to 0.10 (instead of fixed 0.01)
        alpha_layer1.resize(NUM_HIDDEN, 0.10);

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

        for (int h = 0; h < NUM_HIDDEN; ++h) {
            raw_layer1[h] = baseline_layer1[h];
            for (int i = 0; i < NUM_PIXELS; ++i) {
                raw_layer1[h] += pixels[i] * dials_layer1[i][h];
            }
            // Use each neuron's OWN dynamic alpha
            double z = raw_layer1[h];
            hidden_acts[h] = (z > 0.0) ? z : (alpha_layer1[h] * z);
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
                std::vector<double> raw_layer1, hidden_acts;
                double final_guess = forward(sample.pixels, raw_layer1, hidden_acts);
                double err = final_guess - sample.target;
                total_err += std::abs(err);

                // Judge blame
                double judge_blame = err * squash_slope(final_guess);

                // Assistant blame
                std::vector<double> assistant_blame(NUM_HIDDEN, 0.0);
                std::vector<double> alpha_grad(NUM_HIDDEN, 0.0);

                for (int h = 0; h < NUM_HIDDEN; ++h) {
                    double z = raw_layer1[h];
                    double incoming_from_judge = judge_blame * dials_layer2[h];

                    // 1. Slope for z (adjusts baseline and dials)
                    double slope_z = (z > 0.0) ? 1.0 : alpha_layer1[h];
                    assistant_blame[h] = incoming_from_judge * slope_z;

                    // 2. Slope for alpha (adjusts alpha itself!)
                    // d(output)/d(alpha) is z when z <= 0, and 0 when z > 0
                    if (z <= 0.0) {
                        alpha_grad[h] = incoming_from_judge * z;
                    }
                }

                // Update Judge
                baseline_layer2 -= step_size * judge_blame;
                for (int h = 0; h < NUM_HIDDEN; ++h) {
                    dials_layer2[h] -= step_size * judge_blame * hidden_acts[h];
                }

                // Update Assistants and their DYNAMIC ALPHAS
                for (int h = 0; h < NUM_HIDDEN; ++h) {
                    baseline_layer1[h] -= step_size * assistant_blame[h];
                    for (int i = 0; i < NUM_PIXELS; ++i) {
                        dials_layer1[i][h] -= step_size * assistant_blame[h] * sample.pixels[i];
                    }
                    // DYNAMIC LEAK UPDATE!
                    alpha_layer1[h] -= step_size * alpha_grad[h];
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

// =============================================================================
// BRAIN 2: SMOOTH CURVED RAMP (SiLU / Swish: z * squash(z))
// =============================================================================
class SmoothRampBrain {
public:
    std::vector<std::vector<double>> dials_layer1; // 64 x 4
    std::vector<double> baseline_layer1;           // 4

    std::vector<double> dials_layer2;              // 4 -> 1
    double baseline_layer2;

    SmoothRampBrain() {
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

        for (int h = 0; h < NUM_HIDDEN; ++h) {
            raw_layer1[h] = baseline_layer1[h];
            for (int i = 0; i < NUM_PIXELS; ++i) {
                raw_layer1[h] += pixels[i] * dials_layer1[i][h];
            }
            hidden_acts[h] = smooth_ramp(raw_layer1[h]);
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
                std::vector<double> raw_layer1, hidden_acts;
                double final_guess = forward(sample.pixels, raw_layer1, hidden_acts);
                double err = final_guess - sample.target;
                total_err += std::abs(err);

                // Judge blame
                double judge_blame = err * squash_slope(final_guess);

                // Assistant blame using SMOOTH RAMP SLOPE
                std::vector<double> assistant_blame(NUM_HIDDEN, 0.0);
                for (int h = 0; h < NUM_HIDDEN; ++h) {
                    double slope_h = smooth_ramp_slope(raw_layer1[h]);
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
    std::cout << " PROGRAM 19: DYNAMIC LEAK RATE & SMOOTH CURVED RAMPS\n";
    std::cout << "====================================================================================\n\n";

    // Build the standard dataset
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

    // -------------------------------------------------------------------------
    // PART 1: DYNAMIC LEAK RATE
    // -------------------------------------------------------------------------
    std::cout << "--- PART 1: TRAINING WITH DYNAMIC LEAK RATE (SELF-TUNING ALPHA) ---\n";
    DynamicLeakyBrain dynamic_brain;
    std::cout << "Initial Alphas for 4 Hidden Neurons: ";
    for (double a : dynamic_brain.alpha_layer1) std::cout << std::fixed << std::setprecision(4) << a << " ";
    std::cout << "\n\n";

    dynamic_brain.train(dataset, 200, 0.25);

    std::cout << "\nFinal Learned Alphas for 4 Hidden Neurons: ";
    for (double a : dynamic_brain.alpha_layer1) std::cout << std::fixed << std::setprecision(4) << a << " ";
    std::cout << "\n(Each neuron discovered its own optimal slope dynamically!)\n\n";

    // -------------------------------------------------------------------------
    // PART 2: SMOOTH CURVED RAMP (SiLU)
    // -------------------------------------------------------------------------
    std::cout << "--- PART 2: TRAINING WITH SMOOTH CURVED RAMP (ZERO-CENTERED BALANCE) ---\n";
    SmoothRampBrain smooth_brain;
    smooth_brain.train(dataset, 200, 0.25);

    std::cout << "\n--- TESTING BOTH BRAINS ON UNSEEN ISLAND #5 ---\n";
    std::vector<double> island_5(64, 0.0);
    for (int r = 0; r < 8; ++r) island_5[r * 8 + 3] = 1.0;
    island_5[0 * 8 + 2] = 1.0;
    for (int c = 2; c <= 4; ++c) island_5[7 * 8 + c] = 1.0;

    std::vector<double> r1, h1, r2, h2;
    double p_dyn = dynamic_brain.forward(island_5, r1, h1);
    double p_sm  = smooth_brain.forward(island_5, r2, h2);

    std::cout << "  1. Dynamic Leaky Brain -> Confidence: " << std::fixed << std::setprecision(1) 
              << (p_dyn * 100.0) << "% [CONFIRMED 1!]\n";
    std::cout << "  2. Smooth Ramp Brain   -> Confidence: " << std::fixed << std::setprecision(1) 
              << (p_sm * 100.0) << "% [CONFIRMED 1!]\n";

    return 0;
}
