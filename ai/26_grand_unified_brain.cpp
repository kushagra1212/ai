#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <random>
#include <algorithm>
#include <string>

// -----------------------------------------------------------------------------
// S-curve & Competing Probabilities (Softmax)
// -----------------------------------------------------------------------------
double squash(double z) {
    if (z > 50.0) return 1.0;
    if (z < -50.0) return 0.0;
    return 1.0 / (1.0 + std::exp(-z));
}

std::vector<double> competing_probabilities(const std::vector<double>& raw_scores) {
    int n = raw_scores.size();
    std::vector<double> probs(n);

    double max_z = -1e9;
    for (double z : raw_scores) if (z > max_z) max_z = z;

    double sum_exp = 0.0;
    for (int i = 0; i < n; ++i) {
        probs[i] = std::exp(raw_scores[i] - max_z);
        sum_exp += probs[i];
    }
    for (int i = 0; i < n; ++i) probs[i] /= sum_exp;
    return probs;
}

// -----------------------------------------------------------------------------
// 1. SMOOTH CURVED RAMP (SiLU / Swish: z * squash(z))
// -----------------------------------------------------------------------------
double smooth_ramp(double z) {
    return z * squash(z);
}
double smooth_ramp_slope(double z) {
    double s = squash(z);
    return s + z * s * (1.0 - s);
}

// -----------------------------------------------------------------------------
// 2. DYNAMIC LEAKY RAMP
// -----------------------------------------------------------------------------
double dynamic_leaky_act(double z, double alpha) {
    return (z > 0.0) ? z : alpha * z;
}
double dynamic_leaky_slope_z(double z, double alpha) {
    return (z > 0.0) ? 1.0 : alpha;
}

// Dataset sample
struct MultiSample {
    std::vector<std::vector<double>> image; // 8x8
    int target_class;                       // 0, 1, 2, 3
    std::string label;
};

const int NUM_FILTERS = 4;
const int NUM_CLASSES = 4; // 0, 1, 2, 7

// -----------------------------------------------------------------------------
// GRAND UNIFIED ORGANISM
// Architecture:
// 1. CNN: 4 Stencils (3x3 each) -> 6x6 feature maps -> Global Peak Pooling
// 2. Activation: Smooth Curved Ramp (SiLU) + Dynamic Leaky Ramp supported
// 3. Multi-Neuron: 4 Judges competing via Softmax (100% distribution)
// 4. Momentum + Evolution: 60 Dials + 60 Velocities inherited across generations!
// -----------------------------------------------------------------------------
struct UnifiedOrganism {
    // 1. Convolutional Layer: 4 Stencils (3x3 each) + Biases
    std::vector<std::vector<std::vector<double>>> stencils;   // [4][3][3]
    std::vector<double> biases;                               // [4]

    // Velocities for Stencils and Biases
    std::vector<std::vector<std::vector<double>>> v_stencils;
    std::vector<double> v_biases;

    // 2. Multi-Neuron Judge Layer: 4 Judges, each connected to 4 filter peaks
    std::vector<std::vector<double>> judge_dials;             // [4][4]
    std::vector<double> judge_baselines;                      // [4]

    // Velocities for Judges
    std::vector<std::vector<double>> v_judge_dials;
    std::vector<double> v_judge_baselines;

    double loss;
    double accuracy;

    UnifiedOrganism() {
        stencils.assign(NUM_FILTERS, std::vector<std::vector<double>>(3, std::vector<double>(3, 0.0)));
        v_stencils.assign(NUM_FILTERS, std::vector<std::vector<double>>(3, std::vector<double>(3, 0.0)));

        biases.assign(NUM_FILTERS, 0.0);
        v_biases.assign(NUM_FILTERS, 0.0);

        judge_dials.assign(NUM_CLASSES, std::vector<double>(NUM_FILTERS, 0.0));
        v_judge_dials.assign(NUM_CLASSES, std::vector<double>(NUM_FILTERS, 0.0));

        judge_baselines.assign(NUM_CLASSES, 0.0);
        v_judge_baselines.assign(NUM_CLASSES, 0.0);

        loss = 999.0;
        accuracy = 0.0;
    }

    void randomize(std::mt19937& rng) {
        std::uniform_real_distribution<double> dist(-0.2, 0.2);

        for (int f = 0; f < NUM_FILTERS; ++f) {
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    stencils[f][r][c] = dist(rng);
                }
            }
            biases[f] = 0.0;
        }

        for (int k = 0; k < NUM_CLASSES; ++k) {
            judge_baselines[k] = 0.0;
            for (int f = 0; f < NUM_FILTERS; ++f) {
                judge_dials[k][f] = dist(rng);
            }
        }
    }

    std::vector<double> forward(const std::vector<std::vector<double>>& img,
                                std::vector<std::vector<std::vector<double>>>& raw_maps,
                                std::vector<std::pair<int, int>>& peak_coords,
                                std::vector<double>& peaks) const
    {
        raw_maps.assign(NUM_FILTERS, std::vector<std::vector<double>>(6, std::vector<double>(6, 0.0)));
        peak_coords.assign(NUM_FILTERS, {0, 0});
        peaks.assign(NUM_FILTERS, -1e9);

        // Convolution with Smooth Curved Ramp (SiLU)
        for (int f = 0; f < NUM_FILTERS; ++f) {
            for (int r = 0; r < 6; ++r) {
                for (int c = 0; c < 6; ++c) {
                    double sum = biases[f];
                    for (int kr = 0; kr < 3; ++kr) {
                        for (int kc = 0; kc < 3; ++kc) {
                            sum += img[r + kr][c + kc] * stencils[f][kr][kc];
                        }
                    }
                    raw_maps[f][r][c] = sum;
                    double act = smooth_ramp(sum); // Smooth curved activation!

                    if (act > peaks[f]) {
                        peaks[f] = act;
                        peak_coords[f] = {r, c};
                    }
                }
            }
        }

        // Multi-Neuron Judge Layer
        std::vector<double> raw_judges(NUM_CLASSES, 0.0);
        for (int k = 0; k < NUM_CLASSES; ++k) {
            raw_judges[k] = judge_baselines[k];
            for (int f = 0; f < NUM_FILTERS; ++f) {
                raw_judges[k] += peaks[f] * judge_dials[k][f];
            }
        }

        return competing_probabilities(raw_judges);
    }

    void evaluate(const std::vector<MultiSample>& dataset) {
        double total_loss = 0.0;
        int correct = 0;

        for (const auto& s : dataset) {
            std::vector<std::vector<std::vector<double>>> rm;
            std::vector<std::pair<int, int>> pc;
            std::vector<double> pks;
            std::vector<double> probs = forward(s.image, rm, pc, pks);

            total_loss += -std::log(std::max(1e-12, probs[s.target_class]));

            int best = 0;
            for (int k = 1; k < NUM_CLASSES; ++k) {
                if (probs[k] > probs[best]) best = k;
            }
            if (best == s.target_class) correct++;
        }

        loss = total_loss / dataset.size();
        accuracy = (correct * 100.0) / dataset.size();
    }

    // Local exploitation: Polish with Backpropagation + Momentum
    void polish_with_momentum(const std::vector<MultiSample>& dataset, double step_size, double friction) {
        for (const auto& sample : dataset) {
            std::vector<std::vector<std::vector<double>>> raw_maps;
            std::vector<std::pair<int, int>> peak_coords;
            std::vector<double> peaks;

            std::vector<double> probs = forward(sample.image, raw_maps, peak_coords, peaks);

            // 1. Multi-Neuron Judge blame: (Prediction - Target)
            std::vector<double> judge_blames(NUM_CLASSES, 0.0);
            for (int k = 0; k < NUM_CLASSES; ++k) {
                double target = (k == sample.target_class) ? 1.0 : 0.0;
                judge_blames[k] = probs[k] - target;
            }

            // 2. Blame routed to the 4 filter peaks
            std::vector<double> peak_blames(NUM_FILTERS, 0.0);
            for (int f = 0; f < NUM_FILTERS; ++f) {
                for (int k = 0; k < NUM_CLASSES; ++k) {
                    peak_blames[f] += judge_blames[k] * judge_dials[k][f];
                }
            }

            // 3. Update Judges with Momentum
            for (int k = 0; k < NUM_CLASSES; ++k) {
                v_judge_baselines[k] = friction * v_judge_baselines[k] + step_size * judge_blames[k];
                judge_baselines[k]  -= v_judge_baselines[k];

                for (int f = 0; f < NUM_FILTERS; ++f) {
                    v_judge_dials[k][f] = friction * v_judge_dials[k][f] + step_size * judge_blames[k] * peaks[f];
                    judge_dials[k][f]  -= v_judge_dials[k][f];
                }
            }

            // 4. Update Stencils with Momentum using Smooth Ramp Slope
            for (int f = 0; f < NUM_FILTERS; ++f) {
                int wr = peak_coords[f].first;
                int wc = peak_coords[f].second;
                double z = raw_maps[f][wr][wc];

                // Derivative via smooth_ramp_slope!
                double delta = peak_blames[f] * smooth_ramp_slope(z);

                v_biases[f] = friction * v_biases[f] + step_size * delta;
                biases[f]  -= v_biases[f];

                for (int kr = 0; kr < 3; ++kr) {
                    for (int kc = 0; kc < 3; ++kc) {
                        double g = delta * sample.image[wr + kr][wc + kc];
                        v_stencils[f][kr][kc] = friction * v_stencils[f][kr][kc] + step_size * g;
                        stencils[f][kr][kc]  -= v_stencils[f][kr][kc];
                    }
                }
            }
        }
    }
};

const int TOP_CHAMPIONS = 4;
const int CLONES_PER_CHAMPION = 3; // 4 + 12 = 16 organisms total
const int POPULATION_SIZE = TOP_CHAMPIONS + (TOP_CHAMPIONS * CLONES_PER_CHAMPION);

int main() {
    std::cout << "====================================================================================\n";
    std::cout << " PROGRAM 26: GRAND UNIFIED BRAIN (ALL CONCEPTS IN ONE!)\n";
    std::cout << " [CNN + Smooth Curved Ramp + Multi-Neuron Softmax + Hybrid Momentum Evolution]\n";
    std::cout << "====================================================================================\n\n";

    // Dataset: Digits 0, 1, 2, 7 with shifted variations
    std::vector<MultiSample> dataset;
    auto make_grid = []() { return std::vector<std::vector<double>>(8, std::vector<double>(8, 0.0)); };

    // Class 0: Box
    auto d0_a = make_grid();
    for (int i = 1; i <= 6; ++i) { d0_a[1][i] = 1.0; d0_a[6][i] = 1.0; d0_a[i][1] = 1.0; d0_a[i][6] = 1.0; }
    dataset.push_back({d0_a, 0, "Digit 0 (Box)"});

    auto d0_b = make_grid();
    for (int i = 0; i <= 7; ++i) { d0_b[0][i] = 1.0; d0_b[7][i] = 1.0; d0_b[i][0] = 1.0; d0_b[i][7] = 1.0; }
    dataset.push_back({d0_b, 0, "Digit 0 (Big Box)"});

    // Class 1: Vertical line
    auto d1_c = make_grid();
    for (int r = 0; r < 8; ++r) d1_c[r][3] = 1.0;
    dataset.push_back({d1_c, 1, "Digit 1 (Center)"});

    auto d1_l = make_grid();
    for (int r = 0; r < 8; ++r) d1_l[r][1] = 1.0;
    dataset.push_back({d1_l, 1, "Digit 1 (Shifted Left)"});

    auto d1_r = make_grid();
    for (int r = 0; r < 8; ++r) d1_r[r][6] = 1.0;
    dataset.push_back({d1_r, 1, "Digit 1 (Shifted Right)"});

    // Class 2: Digit 2
    auto d2 = make_grid();
    for (int c = 1; c <= 5; ++c) d2[1][c] = 1.0;
    d2[2][5] = 1.0; d2[3][4] = 1.0; d2[4][3] = 1.0; d2[5][2] = 1.0;
    for (int c = 1; c <= 6; ++c) d2[6][c] = 1.0;
    dataset.push_back({d2, 2, "Digit 2 (Standard)"});

    // Class 3: Digit 7
    auto d7_a = make_grid();
    for (int c = 1; c <= 6; ++c) d7_a[1][c] = 1.0;
    for (int r = 1; r <= 7; ++r) d7_a[r][6] = 1.0;
    dataset.push_back({d7_a, 3, "Digit 7 (Top + Right)"});

    auto d7_b = make_grid();
    for (int c = 0; c <= 5; ++c) d7_b[0][c] = 1.0;
    for (int r = 0; r <= 7; ++r) d7_b[r][5] = 1.0;
    dataset.push_back({d7_b, 3, "Digit 7 (Shifted)"});

    std::mt19937 rng(42);
    std::vector<UnifiedOrganism> population(POPULATION_SIZE);
    for (auto& org : population) org.randomize(rng);

    std::cout << "Population: " << POPULATION_SIZE << " Multi-Class CNN Organisms.\n";
    std::cout << "Training for 30 Generations with Local Momentum Backprop (friction = 0.30)...\n\n";

    std::normal_distribution<double> mut_dist(0.0, 0.02);
    std::uniform_real_distribution<double> chance(0.0, 1.0);

    for (int gen = 0; gen < 30; ++gen) {
        // Step A: Local polish with Momentum (3 passes per generation)
        for (auto& org : population) {
            for (int pass = 0; pass < 3; ++pass) {
                org.polish_with_momentum(dataset, 0.20, 0.30); // friction = 0.30
            }
            org.evaluate(dataset);
        }

        // Step B: Sort by lowest loss
        std::sort(population.begin(), population.end(), [](const UnifiedOrganism& a, const UnifiedOrganism& b) {
            return a.loss < b.loss;
        });

        if (gen % 5 == 0 || gen == 29) {
            std::cout << "  Gen " << std::setw(2) << gen 
                      << " | Champion Loss: " << std::fixed << std::setprecision(4) << population[0].loss
                      << " | Accuracy: " << std::setprecision(1) << population[0].accuracy << "%\n";
        }

        if (population[0].loss < 0.005) {
            std::cout << "  >>> Early Convergence at Generation " << gen << "! Loss: " << population[0].loss << "\n";
            break;
        }

        // Step C: Reproduction + Mutation (Inherits Dials AND Momentum Velocities!)
        std::vector<UnifiedOrganism> next_gen;
        for (int i = 0; i < TOP_CHAMPIONS; ++i) {
            next_gen.push_back(population[i]); // Keep Champion (Elitism)

            for (int c = 0; c < CLONES_PER_CHAMPION; ++c) {
                UnifiedOrganism clone = population[i]; // Inherits dials + velocities!

                for (int f = 0; f < NUM_FILTERS; ++f) {
                    for (int kr = 0; kr < 3; ++kr) {
                        for (int kc = 0; kc < 3; ++kc) {
                            if (chance(rng) < 0.10) clone.stencils[f][kr][kc] += mut_dist(rng);
                        }
                    }
                }
                next_gen.push_back(clone);
            }
        }
        population = next_gen;
    }

    UnifiedOrganism champ = population[0];

    std::vector<std::string> names = {"Digit 0", "Digit 1", "Digit 2", "Digit 7"};

    std::cout << "\n--- TESTING CHAMPION ON ALL DATASET SAMPLES ---\n";
    for (const auto& s : dataset) {
        std::vector<std::vector<std::vector<double>>> rm;
        std::vector<std::pair<int, int>> pc;
        std::vector<double> pks;
        auto probs = champ.forward(s.image, rm, pc, pks);

        int best = 0;
        for (int k = 1; k < NUM_CLASSES; ++k) if (probs[k] > probs[best]) best = k;

        std::cout << "  Sample: " << std::setw(22) << std::left << s.label
                  << " | True: " << names[s.target_class]
                  << " | Pred: " << names[best]
                  << " (" << std::fixed << std::setprecision(1) << (probs[best]*100) << "%)\n";
    }

    // Test on Unseen Shifted Digits
    std::cout << "\n--- TESTING CHAMPION ON UNSEEN SHIFTED DIGITS ---\n";

    // Unseen Shifted 1 at Column 4
    auto test_1 = make_grid();
    for (int r = 0; r < 8; ++r) test_1[r][4] = 1.0;

    // Unseen Shifted 7
    auto test_7 = make_grid();
    for (int c = 2; c <= 5; ++c) test_7[1][c] = 1.0;
    for (int r = 1; r <= 7; ++r) test_7[r][5] = 1.0;

    std::vector<std::vector<std::vector<double>>> rm;
    std::vector<std::pair<int, int>> pc;
    std::vector<double> pks;

    auto p1 = champ.forward(test_1, rm, pc, pks);
    auto p7 = champ.forward(test_7, rm, pc, pks);

    std::cout << "  Unseen Shifted '1' -> Probabilities: ";
    for (int k = 0; k < NUM_CLASSES; ++k) std::cout << names[k] << ": " << std::fixed << std::setprecision(1) << (p1[k]*100) << "% | ";
    std::cout << "\n";

    std::cout << "  Unseen Shifted '7' -> Probabilities: ";
    for (int k = 0; k < NUM_CLASSES; ++k) std::cout << names[k] << ": " << std::fixed << std::setprecision(1) << (p7[k]*100) << "% | ";
    std::cout << "\n";

    return 0;
}
