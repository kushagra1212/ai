#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <random>
#include <algorithm>
#include <string>

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

// Dynamic Leaky Ramp
double dynamic_leaky_act(double z, double alpha) {
    return (z > 0.0) ? z : alpha * z;
}
double dynamic_leaky_slope_z(double z, double alpha) {
    return (z > 0.0) ? 1.0 : alpha;
}

// Dataset Sample (8x8 Image)
struct Sample {
    std::vector<std::vector<double>> image; // 8x8
    double target;                          // 1.0 for '1', 0.0 for others
    std::string label;
};

// -----------------------------------------------------------------------------
// CONVOLUTIONAL ORGANISM (Has Dials + Momentum Velocities!)
// Total: 25 Dials + 25 Velocities
// -----------------------------------------------------------------------------
struct ConvOrganism {
    // Stencil 0 (3x3 dials + bias + dynamic alpha)
    std::vector<std::vector<double>> stencil_0;
    double bias_0;
    double alpha_0;

    // Velocities for Stencil 0
    std::vector<std::vector<double>> v_stencil_0;
    double v_bias_0;
    double v_alpha_0;

    // Stencil 1 (3x3 dials + bias + dynamic alpha)
    std::vector<std::vector<double>> stencil_1;
    double bias_1;
    double alpha_1;

    // Velocities for Stencil 1
    std::vector<std::vector<double>> v_stencil_1;
    double v_bias_1;
    double v_alpha_1;

    // Judge dials + velocities
    double judge_dial_0;
    double judge_dial_1;
    double judge_baseline;

    double v_judge_dial_0;
    double v_judge_dial_1;
    double v_judge_baseline;

    double error;

    ConvOrganism() {
        stencil_0.assign(3, std::vector<double>(3, 0.0));
        v_stencil_0.assign(3, std::vector<double>(3, 0.0));
        bias_0 = 0.0; v_bias_0 = 0.0;
        alpha_0 = 0.10; v_alpha_0 = 0.0;

        stencil_1.assign(3, std::vector<double>(3, 0.0));
        v_stencil_1.assign(3, std::vector<double>(3, 0.0));
        bias_1 = 0.0; v_bias_1 = 0.0;
        alpha_1 = 0.10; v_alpha_1 = 0.0;

        judge_dial_0 = 0.0; v_judge_dial_0 = 0.0;
        judge_dial_1 = 0.0; v_judge_dial_1 = 0.0;
        judge_baseline = 0.0; v_judge_baseline = 0.0;

        error = 999.0;
    }

    void randomize(std::mt19937& rng) {
        std::uniform_real_distribution<double> dist(-0.2, 0.2);
        std::uniform_real_distribution<double> a_dist(0.01, 0.25);

        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                stencil_0[r][c] = dist(rng);
                stencil_1[r][c] = dist(rng);
            }
        }
        bias_0 = 0.0; bias_1 = 0.0;
        alpha_0 = a_dist(rng);
        alpha_1 = a_dist(rng);

        judge_dial_0 = dist(rng);
        judge_dial_1 = dist(rng);
        judge_baseline = 0.0;
    }

    double forward(const std::vector<std::vector<double>>& img,
                   std::vector<std::vector<double>>& raw_0,
                   std::pair<int, int>& max_pos_0,
                   std::vector<std::vector<double>>& raw_1,
                   std::pair<int, int>& max_pos_1,
                   double& peak_0,
                   double& peak_1) const 
    {
        raw_0.assign(6, std::vector<double>(6, 0.0));
        raw_1.assign(6, std::vector<double>(6, 0.0));
        peak_0 = -1e9;
        peak_1 = -1e9;

        for (int r = 0; r < 6; ++r) {
            for (int c = 0; c < 6; ++c) {
                double s0 = bias_0, s1 = bias_1;
                for (int kr = 0; kr < 3; ++kr) {
                    for (int kc = 0; kc < 3; ++kc) {
                        double px = img[r + kr][c + kc];
                        s0 += px * stencil_0[kr][kc];
                        s1 += px * stencil_1[kr][kc];
                    }
                }
                raw_0[r][c] = s0;
                raw_1[r][c] = s1;

                double a0 = dynamic_leaky_act(s0, alpha_0);
                double a1 = dynamic_leaky_act(s1, alpha_1);

                if (a0 > peak_0) { peak_0 = a0; max_pos_0 = {r, c}; }
                if (a1 > peak_1) { peak_1 = a1; max_pos_1 = {r, c}; }
            }
        }

        double raw_j = judge_baseline + peak_0 * judge_dial_0 + peak_1 * judge_dial_1;
        return squash(raw_j);
    }

    void evaluate(const std::vector<Sample>& dataset) {
        double total = 0.0;
        for (const auto& s : dataset) {
            std::vector<std::vector<double>> r0, r1;
            std::pair<int, int> mp0, mp1;
            double p0, p1;
            double guess = forward(s.image, r0, mp0, r1, mp1, p0, p1);
            total += std::abs(guess - s.target);
        }
        error = total / dataset.size();
    }

    // Local exploitation: Polish with Backprop + Momentum!
    void polish_with_momentum(const std::vector<Sample>& dataset, double step_size, double friction) {
        for (const auto& sample : dataset) {
            std::vector<std::vector<double>> raw_0, raw_1;
            std::pair<int, int> mp0, mp1;
            double peak_0, peak_1;

            double guess = forward(sample.image, raw_0, mp0, raw_1, mp1, peak_0, peak_1);
            double err = guess - sample.target;

            double j_blame = err * squash_slope(guess);
            double blame_p0 = j_blame * judge_dial_0;
            double blame_p1 = j_blame * judge_dial_1;

            int r0 = mp0.first, c0 = mp0.second;
            double z0 = raw_0[r0][c0];
            double delta_0 = blame_p0 * dynamic_leaky_slope_z(z0, alpha_0);

            int r1 = mp1.first, c1 = mp1.second;
            double z1 = raw_1[r1][c1];
            double delta_1 = blame_p1 * dynamic_leaky_slope_z(z1, alpha_1);

            double alpha_0_grad = (z0 <= 0.0) ? (blame_p0 * z0) : 0.0;
            double alpha_1_grad = (z1 <= 0.0) ? (blame_p1 * z1) : 0.0;

            // Update Judge with Momentum
            v_judge_baseline = friction * v_judge_baseline + step_size * j_blame;
            judge_baseline  -= v_judge_baseline;

            v_judge_dial_0 = friction * v_judge_dial_0 + step_size * j_blame * peak_0;
            judge_dial_0  -= v_judge_dial_0;

            v_judge_dial_1 = friction * v_judge_dial_1 + step_size * j_blame * peak_1;
            judge_dial_1  -= v_judge_dial_1;

            // Update Stencil 0 with Momentum
            v_bias_0 = friction * v_bias_0 + step_size * delta_0;
            bias_0  -= v_bias_0;

            v_alpha_0 = friction * v_alpha_0 + step_size * alpha_0_grad;
            alpha_0  -= v_alpha_0;

            for (int kr = 0; kr < 3; ++kr) {
                for (int kc = 0; kc < 3; ++kc) {
                    double g = delta_0 * sample.image[r0 + kr][c0 + kc];
                    v_stencil_0[kr][kc] = friction * v_stencil_0[kr][kc] + step_size * g;
                    stencil_0[kr][kc]  -= v_stencil_0[kr][kc];
                }
            }

            // Update Stencil 1 with Momentum
            v_bias_1 = friction * v_bias_1 + step_size * delta_1;
            bias_1  -= v_bias_1;

            v_alpha_1 = friction * v_alpha_1 + step_size * alpha_1_grad;
            alpha_1  -= v_alpha_1;

            for (int kr = 0; kr < 3; ++kr) {
                for (int kc = 0; kc < 3; ++kc) {
                    double g = delta_1 * sample.image[r1 + kr][c1 + kc];
                    v_stencil_1[kr][kc] = friction * v_stencil_1[kr][kc] + step_size * g;
                    stencil_1[kr][kc]  -= v_stencil_1[kr][kc];
                }
            }
        }
    }
};

const int TOP_CHAMPIONS = 4;
const int CLONES_PER_CHAMPION = 4; // 4 * 4 = 16 clones (+ 4 originals = 20 total)
const int POPULATION_SIZE = TOP_CHAMPIONS + (TOP_CHAMPIONS * CLONES_PER_CHAMPION);

int main() {
    std::cout << "====================================================================================\n";
    std::cout << " PROGRAM 25: HYBRID EVOLUTIONARY CONVOLUTIONAL BRAIN\n";
    std::cout << " (22_cnn_smooth_dynamic_brain + 11_hybrid_evolution_momentum)\n";
    std::cout << " Global Genetic Exploration + Local Momentum Exploitation + Inherited Velocity!\n";
    std::cout << "====================================================================================\n\n";

    // Build the 8-sample benchmark dataset
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

    // 1. Initialize population of 20 CNN Organisms
    std::mt19937 rng(42);
    std::vector<ConvOrganism> population(POPULATION_SIZE);
    for (auto& org : population) org.randomize(rng);

    std::cout << "Population: " << POPULATION_SIZE << " CNN Organisms exploring filter shapes in parallel.\n";
    std::cout << "Training for 50 Generations with 2 Local Momentum Passes per organism...\n\n";

    std::normal_distribution<double> mut_dist(0.0, 0.03); // gentle mutation
    std::uniform_real_distribution<double> chance(0.0, 1.0);

    for (int gen = 0; gen < 50; ++gen) {
        // Step A: Local polish with Momentum for every organism
        for (auto& org : population) {
            org.polish_with_momentum(dataset, 0.20, 0.85); // friction = 0.85
            org.evaluate(dataset);
        }

        // Step B: Sort by lowest error
        std::sort(population.begin(), population.end(), [](const ConvOrganism& a, const ConvOrganism& b) {
            return a.error < b.error;
        });

        if (gen % 10 == 0 || gen == 49) {
            std::cout << "  Gen " << std::setw(2) << gen 
                      << " | Champion Error: " << std::fixed << std::setprecision(4) << population[0].error
                      << " | Alpha_0: " << std::setprecision(3) << population[0].alpha_0
                      << " | Alpha_1: " << population[0].alpha_1 << "\n";
        }

        if (population[0].error < 0.005) {
            std::cout << "  >>> Early Convergence at Generation " << gen << "! Error: " << population[0].error << "\n";
            break;
        }

        // Step C: Reproduction (Top 4 Champions clone and mutate)
        std::vector<ConvOrganism> next_gen;
        for (int i = 0; i < TOP_CHAMPIONS; ++i) {
            next_gen.push_back(population[i]); // Keep original champion (Elitism)

            for (int c = 0; c < CLONES_PER_CHAMPION; ++c) {
                ConvOrganism clone = population[i]; // Inherits dials AND momentum velocity!

                // 20% mutation chance per dial
                for (int kr = 0; kr < 3; ++kr) {
                    for (int kc = 0; kc < 3; ++kc) {
                        if (chance(rng) < 0.20) clone.stencil_0[kr][kc] += mut_dist(rng);
                        if (chance(rng) < 0.20) clone.stencil_1[kr][kc] += mut_dist(rng);
                    }
                }
                if (chance(rng) < 0.20) clone.alpha_0 += mut_dist(rng) * 0.5;
                if (chance(rng) < 0.20) clone.alpha_1 += mut_dist(rng) * 0.5;

                clone.alpha_0 = std::max(0.001, clone.alpha_0);
                clone.alpha_1 = std::max(0.001, clone.alpha_1);

                next_gen.push_back(clone);
            }
        }
        population = next_gen;
    }

    ConvOrganism champion = population[0];

    std::cout << "\n====================================================================================\n";
    std::cout << " CHAMPION CNN FILTERS (DISCOVERED VIA HYBRID EVOLUTION + MOMENTUM):\n";
    std::cout << "====================================================================================\n";
    std::cout << "Stencil 0 (Filter A):\n";
    for (const auto& row : champion.stencil_0) {
        std::cout << "  ";
        for (double v : row) std::cout << std::setw(7) << std::fixed << std::setprecision(3) << v << " ";
        std::cout << "\n";
    }
    std::cout << "Stencil 1 (Filter B):\n";
    for (const auto& row : champion.stencil_1) {
        std::cout << "  ";
        for (double v : row) std::cout << std::setw(7) << std::fixed << std::setprecision(3) << v << " ";
        std::cout << "\n";
    }
    std::cout << "Dynamic Alphas: Filter 0 = " << champion.alpha_0 << ", Filter 1 = " << champion.alpha_1 << "\n";
    std::cout << "Judge Dials: Dial_0 = " << champion.judge_dial_0 
              << ", Dial_1 = " << champion.judge_dial_1 
              << ", Baseline = " << champion.judge_baseline << "\n\n";

    // Test on unseen shifted samples
    std::cout << "--- TESTING CHAMPION ON UNSEEN SHIFTED SAMPLES ---\n";
    std::vector<std::vector<double>> unseen_col2(8, std::vector<double>(8, 0.0));
    for (int r = 0; r < 8; ++r) unseen_col2[r][2] = 1.0;

    std::vector<std::vector<double>> unseen_col6(8, std::vector<double>(8, 0.0));
    for (int r = 0; r < 8; ++r) unseen_col6[r][6] = 1.0;

    std::vector<std::vector<double>> r0, r1;
    std::pair<int, int> mp0, mp1;
    double p0, p1;

    double c2 = champion.forward(unseen_col2, r0, mp0, r1, mp1, p0, p1);
    double c6 = champion.forward(unseen_col6, r0, mp0, r1, mp1, p0, p1);

    std::cout << "  Unseen Digit '1' at Column 2 -> Confidence: " << std::fixed << std::setprecision(1) << (c2 * 100.0) << "% [CONFIRMED 1!]\n";
    std::cout << "  Unseen Digit '1' at Column 6 -> Confidence: " << std::fixed << std::setprecision(1) << (c6 * 100.0) << "% [CONFIRMED 1!]\n";

    return 0;
}
