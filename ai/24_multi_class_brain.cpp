#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <random>
#include <string>
#include <algorithm>

// -----------------------------------------------------------------------------
// FIRST PRINCIPLES: MULTI-JUDGE COMPETITION (SOFTMAX)
// -----------------------------------------------------------------------------
// Converts an array of raw scores into probabilities that sum to 1.0 (100%)
// Subtracts max_val first to prevent floating-point overflow!
std::vector<double> competing_probabilities(const std::vector<double>& raw_scores) {
    int n = raw_scores.size();
    std::vector<double> probs(n);

    double max_z = -1e9;
    for (double z : raw_scores) {
        if (z > max_z) max_z = z;
    }

    double sum_exp = 0.0;
    for (int i = 0; i < n; ++i) {
        probs[i] = std::exp(raw_scores[i] - max_z);
        sum_exp += probs[i];
    }

    for (int i = 0; i < n; ++i) {
        probs[i] /= sum_exp;
    }
    return probs;
}

// Leaky ramp for internal feature maps
double leaky_ramp(double z, double alpha = 0.01) {
    return (z > 0.0) ? z : alpha * z;
}
double leaky_ramp_slope(double z, double alpha = 0.01) {
    return (z > 0.0) ? 1.0 : alpha;
}

// -----------------------------------------------------------------------------
// Sample representation: 8x8 image + integer target class (0..9)
// -----------------------------------------------------------------------------
struct MultiSample {
    std::vector<std::vector<double>> image; // 8x8
    int target_class;                       // 0, 1, 2, ...
    std::string label;
};

// -----------------------------------------------------------------------------
// MULTI-CLASS CONVOLUTIONAL BRAIN
// Architecture:
// 1. Input: 8x8 Image
// 2. Convolution: 4 Learnable 3x3 Stencils (Vertical, Horizontal, Diagonal, etc.)
// 3. Global Peak-Finder: 4 peak numbers
// 4. Multi-Judge Output: 4 classes (e.g. Digits 0, 1, 2, 7) or 10 classes
// -----------------------------------------------------------------------------
const int NUM_FILTERS = 4;
const int NUM_CLASSES = 4; // Let's recognize 0, 1, 2, 7

class MultiClassCNN {
public:
    // Stencils: 4 filters, each 3x3
    std::vector<std::vector<std::vector<double>>> stencils; // [4][3][3]
    std::vector<double> filter_biases;                      // [4]

    // Final Judges: 4 judges (one per class), each has 4 dials
    std::vector<std::vector<double>> judge_dials; // [NUM_CLASSES][NUM_FILTERS]
    std::vector<double> judge_baselines;          // [NUM_CLASSES]

    MultiClassCNN() {
        std::mt19937 rng(42);
        std::uniform_real_distribution<double> init_d(-0.2, 0.2);

        stencils.resize(NUM_FILTERS, std::vector<std::vector<double>>(3, std::vector<double>(3)));
        filter_biases.assign(NUM_FILTERS, 0.0);

        for (int f = 0; f < NUM_FILTERS; ++f) {
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    stencils[f][r][c] = init_d(rng);
                }
            }
        }

        judge_dials.resize(NUM_CLASSES, std::vector<double>(NUM_FILTERS));
        judge_baselines.assign(NUM_CLASSES, 0.0);

        for (int k = 0; k < NUM_CLASSES; ++k) {
            for (int f = 0; f < NUM_FILTERS; ++f) {
                judge_dials[k][f] = init_d(rng);
            }
        }
    }

    std::vector<double> forward(const std::vector<std::vector<double>>& img,
                                std::vector<std::vector<std::vector<double>>>& raw_maps,
                                std::vector<std::vector<std::vector<double>>>& act_maps,
                                std::vector<std::pair<int, int>>& peak_coords,
                                std::vector<double>& peaks) const
    {
        raw_maps.assign(NUM_FILTERS, std::vector<std::vector<double>>(6, std::vector<double>(6, 0.0)));
        act_maps.assign(NUM_FILTERS, std::vector<std::vector<double>>(6, std::vector<double>(6, 0.0)));
        peak_coords.assign(NUM_FILTERS, {0, 0});
        peaks.assign(NUM_FILTERS, -1e9);

        // 1. Slide 4 stencils over the 8x8 image
        for (int f = 0; f < NUM_FILTERS; ++f) {
            for (int r = 0; r < 6; ++r) {
                for (int c = 0; c < 6; ++c) {
                    double sum = filter_biases[f];
                    for (int kr = 0; kr < 3; ++kr) {
                        for (int kc = 0; kc < 3; ++kc) {
                            sum += img[r + kr][c + kc] * stencils[f][kr][kc];
                        }
                    }
                    raw_maps[f][r][c] = sum;
                    double act = leaky_ramp(sum, 0.01);
                    act_maps[f][r][c] = act;

                    if (act > peaks[f]) {
                        peaks[f] = act;
                        peak_coords[f] = {r, c};
                    }
                }
            }
        }

        // 2. Each class judge computes raw score from the 4 peaks
        std::vector<double> raw_judges(NUM_CLASSES, 0.0);
        for (int k = 0; k < NUM_CLASSES; ++k) {
            raw_judges[k] = judge_baselines[k];
            for (int f = 0; f < NUM_FILTERS; ++f) {
                raw_judges[k] += peaks[f] * judge_dials[k][f];
            }
        }

        // 3. Multi-judge competition (Softmax)
        return competing_probabilities(raw_judges);
    }

    void train(const std::vector<MultiSample>& dataset, int passes, double step_size) {
        for (int pass = 0; pass < passes; ++pass) {
            double total_loss = 0.0;
            int correct_count = 0;

            for (const auto& sample : dataset) {
                std::vector<std::vector<std::vector<double>>> raw_maps, act_maps;
                std::vector<std::pair<int, int>> peak_coords;
                std::vector<double> peaks;

                std::vector<double> probs = forward(sample.image, raw_maps, act_maps, peak_coords, peaks);

                // Loss: -log(probability of correct class)
                double p_correct = std::max(1e-12, probs[sample.target_class]);
                total_loss += -std::log(p_correct);

                // Check prediction
                int best_class = 0;
                double best_prob = probs[0];
                for (int k = 1; k < NUM_CLASSES; ++k) {
                    if (probs[k] > best_prob) {
                        best_prob = probs[k];
                        best_class = k;
                    }
                }
                if (best_class == sample.target_class) correct_count++;

                // 1. Judge blame: EXACT FIRST-PRINCIPLES GRADIENT: (Prediction - Target)!
                std::vector<double> judge_blames(NUM_CLASSES, 0.0);
                for (int k = 0; k < NUM_CLASSES; ++k) {
                    double target = (k == sample.target_class) ? 1.0 : 0.0;
                    judge_blames[k] = probs[k] - target;
                }

                // 2. Blame routed to the 4 peaks
                std::vector<double> peak_blames(NUM_FILTERS, 0.0);
                for (int f = 0; f < NUM_FILTERS; ++f) {
                    for (int k = 0; k < NUM_CLASSES; ++k) {
                        peak_blames[f] += judge_blames[k] * judge_dials[k][f];
                    }
                }

                // 3. Update Judge dials
                for (int k = 0; k < NUM_CLASSES; ++k) {
                    judge_baselines[k] -= step_size * judge_blames[k];
                    for (int f = 0; f < NUM_FILTERS; ++f) {
                        judge_dials[k][f] -= step_size * judge_blames[k] * peaks[f];
                    }
                }

                // 4. Update Stencils from winning patches
                for (int f = 0; f < NUM_FILTERS; ++f) {
                    int wr = peak_coords[f].first;
                    int wc = peak_coords[f].second;
                    double z = raw_maps[f][wr][wc];
                    double delta = peak_blames[f] * leaky_ramp_slope(z, 0.01);

                    filter_biases[f] -= step_size * delta;
                    for (int kr = 0; kr < 3; ++kr) {
                        for (int kc = 0; kc < 3; ++kc) {
                            stencils[f][kr][kc] -= step_size * delta * sample.image[wr + kr][wc + kc];
                        }
                    }
                }
            }

            if (pass % 50 == 0 || pass == passes - 1) {
                std::cout << "  Pass " << std::setw(3) << pass 
                          << " | Avg Cross-Entropy Loss: " << std::fixed << std::setprecision(4) 
                          << (total_loss / dataset.size()) 
                          << " | Accuracy: " << (correct_count * 100.0 / dataset.size()) << "%\n";
            }
        }
    }
};

int main() {
    std::cout << "====================================================================================\n";
    std::cout << " PROGRAM 24: MULTI-CLASS CONVOLUTIONAL BRAIN (RECOGNIZING MULTIPLE DIGITS)\n";
    std::cout << " (Multi-Judge Competition: 100% Probability Distribution via Softmax)\n";
    std::cout << "====================================================================================\n\n";

    std::vector<MultiSample> dataset;

    // Helper to make blank grid
    auto make_grid = []() { return std::vector<std::vector<double>>(8, std::vector<double>(8, 0.0)); };

    // Class 0: Digit '0' (Box)
    auto d0_a = make_grid();
    for (int i = 1; i <= 6; ++i) { d0_a[1][i] = 1.0; d0_a[6][i] = 1.0; d0_a[i][1] = 1.0; d0_a[i][6] = 1.0; }
    dataset.push_back({d0_a, 0, "Digit 0 (Box)"});

    auto d0_b = make_grid();
    for (int i = 0; i <= 7; ++i) { d0_b[0][i] = 1.0; d0_b[7][i] = 1.0; d0_b[i][0] = 1.0; d0_b[i][7] = 1.0; }
    dataset.push_back({d0_b, 0, "Digit 0 (Big Box)"});

    // Class 1: Digit '1' (Vertical lines)
    auto d1_c = make_grid();
    for (int r = 0; r < 8; ++r) d1_c[r][3] = 1.0;
    dataset.push_back({d1_c, 1, "Digit 1 (Center)"});

    auto d1_l = make_grid();
    for (int r = 0; r < 8; ++r) d1_l[r][1] = 1.0;
    dataset.push_back({d1_l, 1, "Digit 1 (Shifted Left)"});

    auto d1_r = make_grid();
    for (int r = 0; r < 8; ++r) d1_r[r][6] = 1.0;
    dataset.push_back({d1_r, 1, "Digit 1 (Shifted Right)"});

    // Class 2: Digit '2' (Top bar, diagonal/step, bottom bar)
    auto d2_a = make_grid();
    for (int c = 1; c <= 5; ++c) d2_a[1][c] = 1.0;
    d2_a[2][5] = 1.0; d2_a[3][4] = 1.0; d2_a[4][3] = 1.0; d2_a[5][2] = 1.0;
    for (int c = 1; c <= 6; ++c) d2_a[6][c] = 1.0;
    dataset.push_back({d2_a, 2, "Digit 2 (Standard)"});

    // Class 3: Digit '7' (Top horizontal bar + right vertical line)
    auto d7_a = make_grid();
    for (int c = 1; c <= 6; ++c) d7_a[1][c] = 1.0;
    for (int r = 1; r <= 7; ++r) d7_a[r][6] = 1.0;
    dataset.push_back({d7_a, 3, "Digit 7 (Top + Right bar)"});

    auto d7_b = make_grid();
    for (int c = 0; c <= 5; ++c) d7_b[0][c] = 1.0;
    for (int r = 0; r <= 7; ++r) d7_b[r][5] = 1.0;
    dataset.push_back({d7_b, 3, "Digit 7 (Shifted Left)"});

    MultiClassCNN brain;

    std::cout << "--- 1. TRAINING THE MULTI-CLASS CONVOLUTIONAL BRAIN ---\n";
    brain.train(dataset, 600, 0.25);

    std::cout << "\n--- 2. TESTING ON TRAINING SAMPLES ---\n";
    std::vector<std::string> class_names = {"Digit 0", "Digit 1", "Digit 2", "Digit 7"};

    for (const auto& sample : dataset) {
        std::vector<std::vector<std::vector<double>>> r_maps, a_maps;
        std::vector<std::pair<int, int>> p_coords;
        std::vector<double> pks;
        std::vector<double> probs = brain.forward(sample.image, r_maps, a_maps, p_coords, pks);

        int pred_class = 0;
        double max_p = probs[0];
        for (int k = 1; k < NUM_CLASSES; ++k) {
            if (probs[k] > max_p) { max_p = probs[k]; pred_class = k; }
        }

        std::cout << "  Sample: " << std::setw(22) << std::left << sample.label
                  << " | True: " << class_names[sample.target_class]
                  << " | Pred: " << class_names[pred_class]
                  << " (" << std::fixed << std::setprecision(1) << (max_p * 100.0) << "%)\n";
    }

    // -------------------------------------------------------------------------
    // TEST ON UNSEEN DIGITS
    // -------------------------------------------------------------------------
    std::cout << "\n--- 3. TESTING ON UNSEEN SHIFTED SAMPLES ---\n";

    // Test A: Unseen Shifted '1' at Column 4
    auto unseen_1 = make_grid();
    for (int r = 0; r < 8; ++r) unseen_1[r][4] = 1.0;

    // Test B: Unseen Shifted '7' at center
    auto unseen_7 = make_grid();
    for (int c = 2; c <= 5; ++c) unseen_7[1][c] = 1.0;
    for (int r = 1; r <= 7; ++r) unseen_7[r][5] = 1.0;

    std::vector<std::vector<std::vector<double>>> r_maps, a_maps;
    std::vector<std::pair<int, int>> p_coords;
    std::vector<double> pks;

    auto p1 = brain.forward(unseen_1, r_maps, a_maps, p_coords, pks);
    auto p7 = brain.forward(unseen_7, r_maps, a_maps, p_coords, pks);

    std::cout << "  Unseen Shifted '1' -> Probabilities: ";
    for (int k = 0; k < NUM_CLASSES; ++k) std::cout << class_names[k] << ": " << std::fixed << std::setprecision(1) << (p1[k]*100) << "% | ";
    std::cout << "\n";

    std::cout << "  Unseen Shifted '7' -> Probabilities: ";
    for (int k = 0; k < NUM_CLASSES; ++k) std::cout << class_names[k] << ": " << std::fixed << std::setprecision(1) << (p7[k]*100) << "% | ";
    std::cout << "\n";

    return 0;
}
