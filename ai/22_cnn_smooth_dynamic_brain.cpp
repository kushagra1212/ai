#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <random>
#include <string>
#include <algorithm>

// -----------------------------------------------------------------------------
// S-curve for Final Judge
// -----------------------------------------------------------------------------
double squash(double z) {
    if (z > 50.0) return 1.0;
    if (z < -50.0) return 0.0;
    return 1.0 / (1.0 + std::exp(-z));
}
double squash_slope(double a) {
    return a * (1.0 - a);
}

// -----------------------------------------------------------------------------
// 1. Dynamic Leaky Ramp (Parametric)
// -----------------------------------------------------------------------------
double dynamic_leaky_act(double z, double alpha) {
    return (z > 0.0) ? z : alpha * z;
}
double dynamic_leaky_slope_z(double z, double alpha) {
    return (z > 0.0) ? 1.0 : alpha;
}

// -----------------------------------------------------------------------------
// 2. Smooth Curved Ramp (SiLU / Swish: z * squash(z))
// -----------------------------------------------------------------------------
double smooth_ramp(double z) {
    return z * squash(z);
}
double smooth_ramp_slope(double z) {
    double s = squash(z);
    return s + z * s * (1.0 - s);
}

// -----------------------------------------------------------------------------
// Dataset Sample (8x8 Image)
// -----------------------------------------------------------------------------
struct Sample {
    std::vector<std::vector<double>> image; // 8x8
    double target;                          // 1.0 for '1', 0.0 for others
    std::string label;
};

// =============================================================================
// MODEL 1: CNN WITH DYNAMIC LEAK RATE (Filters learn their own alpha dials!)
// =============================================================================
class DynamicLeakyCNN {
public:
    std::vector<std::vector<double>> stencil_0; // 3x3
    double bias_0;
    double alpha_0; // Dynamic leak rate for filter 0

    std::vector<std::vector<double>> stencil_1; // 3x3
    double bias_1;
    double alpha_1; // Dynamic leak rate for filter 1

    double judge_dial_0;
    double judge_dial_1;
    double judge_baseline;

    DynamicLeakyCNN() {
        std::mt19937 rng(42);
        std::uniform_real_distribution<double> init_d(-0.2, 0.2);

        stencil_0.assign(3, std::vector<double>(3));
        stencil_1.assign(3, std::vector<double>(3));
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                stencil_0[r][c] = init_d(rng);
                stencil_1[r][c] = init_d(rng);
            }
        }
        bias_0 = 0.0;
        bias_1 = 0.0;

        // Initialize both filters with alpha = 0.10
        alpha_0 = 0.10;
        alpha_1 = 0.10;

        judge_dial_0 = init_d(rng);
        judge_dial_1 = init_d(rng);
        judge_baseline = 0.0;
    }

    double forward(const std::vector<std::vector<double>>& img,
                   std::vector<std::vector<double>>& raw_0,
                   std::vector<std::vector<double>>& act_0,
                   std::pair<int, int>& max_pos_0,
                   std::vector<std::vector<double>>& raw_1,
                   std::vector<std::vector<double>>& act_1,
                   std::pair<int, int>& max_pos_1,
                   double& peak_0,
                   double& peak_1) const 
    {
        raw_0.assign(6, std::vector<double>(6, 0.0));
        act_0.assign(6, std::vector<double>(6, 0.0));
        raw_1.assign(6, std::vector<double>(6, 0.0));
        act_1.assign(6, std::vector<double>(6, 0.0));

        peak_0 = -1e9;
        peak_1 = -1e9;

        for (int r = 0; r < 6; ++r) {
            for (int c = 0; c < 6; ++c) {
                double sum_0 = bias_0;
                double sum_1 = bias_1;
                for (int kr = 0; kr < 3; ++kr) {
                    for (int kc = 0; kc < 3; ++kc) {
                        double px = img[r + kr][c + kc];
                        sum_0 += px * stencil_0[kr][kc];
                        sum_1 += px * stencil_1[kr][kc];
                    }
                }
                raw_0[r][c] = sum_0;
                raw_1[r][c] = sum_1;

                act_0[r][c] = dynamic_leaky_act(sum_0, alpha_0);
                act_1[r][c] = dynamic_leaky_act(sum_1, alpha_1);

                if (act_0[r][c] > peak_0) { peak_0 = act_0[r][c]; max_pos_0 = {r, c}; }
                if (act_1[r][c] > peak_1) { peak_1 = act_1[r][c]; max_pos_1 = {r, c}; }
            }
        }

        double raw_judge = judge_baseline + peak_0 * judge_dial_0 + peak_1 * judge_dial_1;
        return squash(raw_judge);
    }

    void train(const std::vector<Sample>& dataset, int passes, double step_size) {
        for (int pass = 0; pass < passes; ++pass) {
            double total_err = 0.0;

            for (const auto& sample : dataset) {
                std::vector<std::vector<double>> raw_0, act_0, raw_1, act_1;
                std::pair<int, int> max_pos_0, max_pos_1;
                double peak_0 = 0.0, peak_1 = 0.0;

                double guess = forward(sample.image, raw_0, act_0, max_pos_0, raw_1, act_1, max_pos_1, peak_0, peak_1);
                double err = guess - sample.target;
                total_err += std::abs(err);

                // Judge blame
                double judge_blame = err * squash_slope(guess);

                // Blame to peaks
                double blame_peak_0 = judge_blame * judge_dial_0;
                double blame_peak_1 = judge_blame * judge_dial_1;

                // Routing to winning peak patches
                int r0 = max_pos_0.first, c0 = max_pos_0.second;
                double z0 = raw_0[r0][c0];
                double delta_0 = blame_peak_0 * dynamic_leaky_slope_z(z0, alpha_0);

                int r1 = max_pos_1.first, c1 = max_pos_1.second;
                double z1 = raw_1[r1][c1];
                double delta_1 = blame_peak_1 * dynamic_leaky_slope_z(z1, alpha_1);

                // Dynamic Alpha gradients (d(act)/d(alpha) is z when z <= 0)
                double alpha_0_grad = (z0 <= 0.0) ? (blame_peak_0 * z0) : 0.0;
                double alpha_1_grad = (z1 <= 0.0) ? (blame_peak_1 * z1) : 0.0;

                // Update Judge
                judge_baseline -= step_size * judge_blame;
                judge_dial_0   -= step_size * judge_blame * peak_0;
                judge_dial_1   -= step_size * judge_blame * peak_1;

                // Update Stencil 0 & Alpha 0
                bias_0 -= step_size * delta_0;
                alpha_0 -= step_size * alpha_0_grad; // SELF-LEARNING LEAK RATE!
                for (int kr = 0; kr < 3; ++kr) {
                    for (int kc = 0; kc < 3; ++kc) {
                        stencil_0[kr][kc] -= step_size * delta_0 * sample.image[r0 + kr][c0 + kc];
                    }
                }

                // Update Stencil 1 & Alpha 1
                bias_1 -= step_size * delta_1;
                alpha_1 -= step_size * alpha_1_grad; // SELF-LEARNING LEAK RATE!
                for (int kr = 0; kr < 3; ++kr) {
                    for (int kc = 0; kc < 3; ++kc) {
                        stencil_1[kr][kc] -= step_size * delta_1 * sample.image[r1 + kr][c1 + kc];
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

// =============================================================================
// MODEL 2: CNN WITH SMOOTH CURVED RAMP (SiLU / Swish: z * squash(z))
// =============================================================================
class SmoothCNN {
public:
    std::vector<std::vector<double>> stencil_0;
    double bias_0;

    std::vector<std::vector<double>> stencil_1;
    double bias_1;

    double judge_dial_0;
    double judge_dial_1;
    double judge_baseline;

    SmoothCNN() {
        std::mt19937 rng(42);
        std::uniform_real_distribution<double> init_d(-0.2, 0.2);

        stencil_0.assign(3, std::vector<double>(3));
        stencil_1.assign(3, std::vector<double>(3));
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                stencil_0[r][c] = init_d(rng);
                stencil_1[r][c] = init_d(rng);
            }
        }
        bias_0 = 0.0;
        bias_1 = 0.0;

        judge_dial_0 = init_d(rng);
        judge_dial_1 = init_d(rng);
        judge_baseline = 0.0;
    }

    double forward(const std::vector<std::vector<double>>& img,
                   std::vector<std::vector<double>>& raw_0,
                   std::vector<std::vector<double>>& act_0,
                   std::pair<int, int>& max_pos_0,
                   std::vector<std::vector<double>>& raw_1,
                   std::vector<std::vector<double>>& act_1,
                   std::pair<int, int>& max_pos_1,
                   double& peak_0,
                   double& peak_1) const 
    {
        raw_0.assign(6, std::vector<double>(6, 0.0));
        act_0.assign(6, std::vector<double>(6, 0.0));
        raw_1.assign(6, std::vector<double>(6, 0.0));
        act_1.assign(6, std::vector<double>(6, 0.0));

        peak_0 = -1e9;
        peak_1 = -1e9;

        for (int r = 0; r < 6; ++r) {
            for (int c = 0; c < 6; ++c) {
                double sum_0 = bias_0;
                double sum_1 = bias_1;
                for (int kr = 0; kr < 3; ++kr) {
                    for (int kc = 0; kc < 3; ++kc) {
                        double px = img[r + kr][c + kc];
                        sum_0 += px * stencil_0[kr][kc];
                        sum_1 += px * stencil_1[kr][kc];
                    }
                }
                raw_0[r][c] = sum_0;
                raw_1[r][c] = sum_1;

                // Smooth Ramp: z * squash(z) -> zero-centered dip to -0.28!
                act_0[r][c] = smooth_ramp(sum_0);
                act_1[r][c] = smooth_ramp(sum_1);

                if (act_0[r][c] > peak_0) { peak_0 = act_0[r][c]; max_pos_0 = {r, c}; }
                if (act_1[r][c] > peak_1) { peak_1 = act_1[r][c]; max_pos_1 = {r, c}; }
            }
        }

        double raw_judge = judge_baseline + peak_0 * judge_dial_0 + peak_1 * judge_dial_1;
        return squash(raw_judge);
    }

    void train(const std::vector<Sample>& dataset, int passes, double step_size) {
        for (int pass = 0; pass < passes; ++pass) {
            double total_err = 0.0;

            for (const auto& sample : dataset) {
                std::vector<std::vector<double>> raw_0, act_0, raw_1, act_1;
                std::pair<int, int> max_pos_0, max_pos_1;
                double peak_0 = 0.0, peak_1 = 0.0;

                double guess = forward(sample.image, raw_0, act_0, max_pos_0, raw_1, act_1, max_pos_1, peak_0, peak_1);
                double err = guess - sample.target;
                total_err += std::abs(err);

                // Judge blame
                double judge_blame = err * squash_slope(guess);

                // Blame to peaks
                double blame_peak_0 = judge_blame * judge_dial_0;
                double blame_peak_1 = judge_blame * judge_dial_1;

                // Routing to winning peak patches using SMOOTH RAMP SLOPE
                int r0 = max_pos_0.first, c0 = max_pos_0.second;
                double delta_0 = blame_peak_0 * smooth_ramp_slope(raw_0[r0][c0]);

                int r1 = max_pos_1.first, c1 = max_pos_1.second;
                double delta_1 = blame_peak_1 * smooth_ramp_slope(raw_1[r1][c1]);

                // Update Judge
                judge_baseline -= step_size * judge_blame;
                judge_dial_0   -= step_size * judge_blame * peak_0;
                judge_dial_1   -= step_size * judge_blame * peak_1;

                // Update Stencil 0
                bias_0 -= step_size * delta_0;
                for (int kr = 0; kr < 3; ++kr) {
                    for (int kc = 0; kc < 3; ++kc) {
                        stencil_0[kr][kc] -= step_size * delta_0 * sample.image[r0 + kr][c0 + kc];
                    }
                }

                // Update Stencil 1
                bias_1 -= step_size * delta_1;
                for (int kr = 0; kr < 3; ++kr) {
                    for (int kc = 0; kc < 3; ++kc) {
                        stencil_1[kr][kc] -= step_size * delta_1 * sample.image[r1 + kr][c1 + kc];
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
    std::cout << " PROGRAM 22: COMBINING THE SLIDING STENCIL (CNN) WITH DYNAMIC & SMOOTH RAMPS\n";
    std::cout << "====================================================================================\n\n";

    // Dataset: Center, Shifted Left, Shifted Right, Hook, Box, Letter L, Plus, Bar
    std::vector<Sample> dataset;

    std::vector<std::vector<double>> one_c(8, std::vector<double>(8, 0.0));
    for (int r = 0; r < 8; ++r) one_c[r][3] = 1.0;
    dataset.push_back({one_c, 1.0, "1 (Center)"});

    std::vector<std::vector<double>> one_l(8, std::vector<double>(8, 0.0));
    for (int r = 0; r < 8; ++r) one_l[r][1] = 1.0;
    dataset.push_back({one_l, 1.0, "1 (Shifted Left)"});

    std::vector<std::vector<double>> one_r(8, std::vector<double>(8, 0.0));
    for (int r = 0; r < 8; ++r) one_r[r][5] = 1.0;
    dataset.push_back({one_r, 1.0, "1 (Shifted Right)"});

    auto one_hook = one_c;
    one_hook[0][2] = 1.0;
    dataset.push_back({one_hook, 1.0, "1 (With hook)"});

    std::vector<std::vector<double>> box(8, std::vector<double>(8, 0.0));
    for (int i = 0; i < 8; ++i) {
        box[0][i] = 1.0; box[7][i] = 1.0;
        box[i][0] = 1.0; box[i][7] = 1.0;
    }
    dataset.push_back({box, 0.0, "0 (Box)"});

    std::vector<std::vector<double>> l_shape(8, std::vector<double>(8, 0.0));
    for (int r = 0; r < 8; ++r) l_shape[r][1] = 1.0;
    for (int c = 1; c < 7; ++c) l_shape[7][c] = 1.0;
    dataset.push_back({l_shape, 0.0, "Letter L"});

    std::vector<std::vector<double>> plus(8, std::vector<double>(8, 0.0));
    for (int i = 0; i < 8; ++i) { plus[3][i] = 1.0; plus[i][3] = 1.0; }
    dataset.push_back({plus, 0.0, "Plus Sign"});

    std::vector<std::vector<double>> horiz(8, std::vector<double>(8, 0.0));
    for (int c = 0; c < 8; ++c) horiz[4][c] = 1.0;
    dataset.push_back({horiz, 0.0, "Horizontal Bar"});

    // -------------------------------------------------------------------------
    // 1. DYNAMIC LEAK CNN
    // -------------------------------------------------------------------------
    std::cout << "--- 1. TRAINING CNN WITH DYNAMIC SELF-TUNING LEAK (PReLU) ---\n";
    DynamicLeakyCNN dyn_cnn;
    std::cout << "Initial Alphas: Filter 0 = " << dyn_cnn.alpha_0 << ", Filter 1 = " << dyn_cnn.alpha_1 << "\n";
    dyn_cnn.train(dataset, 400, 0.20);
    std::cout << "Final Learned Alphas: Filter 0 = " << dyn_cnn.alpha_0 << ", Filter 1 = " << dyn_cnn.alpha_1 << "\n\n";

    // -------------------------------------------------------------------------
    // 2. SMOOTH CURVED RAMP CNN
    // -------------------------------------------------------------------------
    std::cout << "--- 2. TRAINING CNN WITH SMOOTH CURVED RAMP (SiLU / Swish) ---\n";
    SmoothCNN smooth_cnn;
    smooth_cnn.train(dataset, 400, 0.20);

    // -------------------------------------------------------------------------
    // 3. TESTING BOTH BRAINS ON UNSEEN SHIFTED SAMPLES
    // -------------------------------------------------------------------------
    std::cout << "\n--- 3. TESTING ON UNSEEN SHIFTED SAMPLES (COLUMN 2 & COLUMN 6) ---\n";

    std::vector<std::vector<double>> test_col2(8, std::vector<double>(8, 0.0));
    for (int r = 0; r < 8; ++r) test_col2[r][2] = 1.0;

    std::vector<std::vector<double>> test_col6(8, std::vector<double>(8, 0.0));
    for (int r = 0; r < 8; ++r) test_col6[r][6] = 1.0;

    std::vector<std::vector<double>> r0, a0, r1, a1;
    std::pair<int, int> mp0, mp1;
    double p0, p1;

    double d_col2 = dyn_cnn.forward(test_col2, r0, a0, mp0, r1, a1, mp1, p0, p1);
    double d_col6 = dyn_cnn.forward(test_col6, r0, a0, mp0, r1, a1, mp1, p0, p1);

    double s_col2 = smooth_cnn.forward(test_col2, r0, a0, mp0, r1, a1, mp1, p0, p1);
    double s_col6 = smooth_cnn.forward(test_col6, r0, a0, mp0, r1, a1, mp1, p0, p1);

    std::cout << "  [Dynamic Leaky CNN]  Digit '1' at Col 2 -> Confidence: " << std::fixed << std::setprecision(1) << (d_col2 * 100.0) << "%\n";
    std::cout << "  [Dynamic Leaky CNN]  Digit '1' at Col 6 -> Confidence: " << std::fixed << std::setprecision(1) << (d_col6 * 100.0) << "%\n\n";

    std::cout << "  [Smooth Curved CNN]  Digit '1' at Col 2 -> Confidence: " << std::fixed << std::setprecision(1) << (s_col2 * 100.0) << "%\n";
    std::cout << "  [Smooth Curved CNN]  Digit '1' at Col 6 -> Confidence: " << std::fixed << std::setprecision(1) << (s_col6 * 100.0) << "%\n";

    return 0;
}
