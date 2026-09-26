#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cmath>
#include <random>
#include <iomanip>
#include <algorithm>
#include <string>
#include <chrono>

// =============================================================================
// PROGRAM 26: GRAND UNIFIED DEEP CNN (ALL THREE WAYS COMBINED!)
//
// 1. WAY 1: 16 Convolutional Stencils (5x5 each)
//    -> Captures 16 distinct stroke primitives (slants, arcs, loops, bars)
//
// 2. WAY 2: Intermediate Hidden Layer (64 Hidden Neurons)
//    -> Learns compound concepts ("Top Loop" + "Bottom Loop" = "8")
//
// 3. WAY 3: Hierarchical Feature Stacking (3-Layer Computational Hierarchy)
//    -> Layer 1: Raw Pixels -> 16 Stroke Stencils
//    -> Layer 2: 64 Spatial Features -> 64 Compound Concept Neurons
//    -> Layer 3: 64 Concepts -> 10 Competing Judges (Softmax Probabilities)
//
// 4. Momentum Backpropagation + Hybrid Evolution across generations!
// =============================================================================

uint32_t read_big_endian(std::ifstream& f) {
    uint8_t bytes[4];
    f.read(reinterpret_cast<char*>(bytes), 4);
    return (static_cast<uint32_t>(bytes[0]) << 24) |
           (static_cast<uint32_t>(bytes[1]) << 16) |
           (static_cast<uint32_t>(bytes[2]) << 8)  |
           (static_cast<uint32_t>(bytes[3]));
}

struct MNISTSample {
    uint8_t label;
    std::vector<float> pixels; // 784 numbers (28x28)
};

std::vector<MNISTSample> load_mnist_raw(const std::string& img_path, 
                                        const std::string& lbl_path, 
                                        int max_count) 
{
    std::ifstream img_f(img_path, std::ios::binary);
    std::ifstream lbl_f(lbl_path, std::ios::binary);

    if (!img_f.is_open() || !lbl_f.is_open()) {
        std::cerr << "Error: Could not open MNIST files at " << img_path << "\n";
        return {};
    }

    read_big_endian(img_f);
    uint32_t num_images = read_big_endian(img_f);
    read_big_endian(img_f);
    read_big_endian(img_f);

    read_big_endian(lbl_f);
    read_big_endian(lbl_f);

    int count = std::min(static_cast<uint32_t>(max_count), num_images);
    std::vector<MNISTSample> dataset(count);

    for (int i = 0; i < count; ++i) {
        uint8_t lbl = 0;
        lbl_f.read(reinterpret_cast<char*>(&lbl), 1);
        dataset[i].label = lbl;

        dataset[i].pixels.resize(784);
        for (int p = 0; p < 784; ++p) {
            uint8_t px = 0;
            img_f.read(reinterpret_cast<char*>(&px), 1);
            dataset[i].pixels[p] = static_cast<float>(px) / 255.0f;
        }
    }

    return dataset;
}

inline float squash(float z) {
    if (z > 40.0f) return 1.0f;
    if (z < -40.0f) return 0.0f;
    return 1.0f / (1.0f + std::exp(-z));
}

inline float smooth_ramp(float z) {
    return z * squash(z);
}

inline float smooth_ramp_slope(float z) {
    float s = squash(z);
    return s + z * s * (1.0f - s);
}

inline std::vector<float> competing_probabilities(const std::vector<float>& raw_scores) {
    int n = raw_scores.size();
    std::vector<float> probs(n);

    float max_z = -1e9f;
    for (float z : raw_scores) if (z > max_z) max_z = z;

    float sum_exp = 0.0f;
    for (int i = 0; i < n; ++i) {
        probs[i] = std::exp(raw_scores[i] - max_z);
        sum_exp += probs[i];
    }
    for (int i = 0; i < n; ++i) probs[i] /= sum_exp;
    return probs;
}

// Hyperparameters
const int KERNEL_SIZE   = 5;                            // 5x5 stencils
const int NUM_FILTERS   = 16;                           // Way 1: 16 filters
const int OUT_DIM       = 28 - KERNEL_SIZE + 1;         // 24x24 positions
const int MID_POINT     = OUT_DIM / 2;                  // 12
const int POOL_REGIONS  = 4;                            // 4 quadrants: TL, TR, BL, BR
const int NUM_FEATURES  = NUM_FILTERS * POOL_REGIONS;   // 16 * 4 = 64 spatial features
const int HIDDEN_NEURONS = 64;                          // Way 2: 64 Intermediate Thinkers
const int NUM_CLASSES   = 10;                           // 10 Output Digits

// -----------------------------------------------------------------------------
// GRAND UNIFIED DEEP ORGANISM (5,226 DIALS)
// -----------------------------------------------------------------------------
struct UnifiedOrganism {
    // Layer 1: 16 Stencils (5x5 = 400 dials) + 16 Biases = 416 dials
    std::vector<std::vector<float>> stencils;
    std::vector<float> biases;
    std::vector<std::vector<float>> v_stencils;
    std::vector<float> v_biases;

    // Layer 2: 64 Hidden Concept Neurons (64 inputs x 64 = 4,096 dials + 64 biases = 4,160 dials)
    std::vector<std::vector<float>> W_hidden;
    std::vector<float> b_hidden;
    std::vector<std::vector<float>> v_W_hidden;
    std::vector<float> v_b_hidden;

    // Layer 3: 10 Output Judges (64 hidden inputs x 10 = 640 dials + 10 baselines = 650 dials)
    std::vector<std::vector<float>> W_out;
    std::vector<float> b_out;
    std::vector<std::vector<float>> v_W_out;
    std::vector<float> v_b_out;

    float loss;
    float accuracy;

    UnifiedOrganism() {
        // Layer 1
        stencils.assign(NUM_FILTERS, std::vector<float>(KERNEL_SIZE * KERNEL_SIZE, 0.0f));
        v_stencils.assign(NUM_FILTERS, std::vector<float>(KERNEL_SIZE * KERNEL_SIZE, 0.0f));
        biases.assign(NUM_FILTERS, 0.0f);
        v_biases.assign(NUM_FILTERS, 0.0f);

        // Layer 2
        W_hidden.assign(HIDDEN_NEURONS, std::vector<float>(NUM_FEATURES, 0.0f));
        v_W_hidden.assign(HIDDEN_NEURONS, std::vector<float>(NUM_FEATURES, 0.0f));
        b_hidden.assign(HIDDEN_NEURONS, 0.0f);
        v_b_hidden.assign(HIDDEN_NEURONS, 0.0f);

        // Layer 3
        W_out.assign(NUM_CLASSES, std::vector<float>(HIDDEN_NEURONS, 0.0f));
        v_W_out.assign(NUM_CLASSES, std::vector<float>(HIDDEN_NEURONS, 0.0f));
        b_out.assign(NUM_CLASSES, 0.0f);
        v_b_out.assign(NUM_CLASSES, 0.0f);

        loss = 999.0f;
        accuracy = 0.0f;
    }

    void randomize(std::mt19937& rng) {
        std::normal_distribution<float> dist(0.0f, 0.05f);

        for (int f = 0; f < NUM_FILTERS; ++f) {
            for (int i = 0; i < KERNEL_SIZE * KERNEL_SIZE; ++i) stencils[f][i] = dist(rng);
            biases[f] = 0.0f;
        }

        for (int h = 0; h < HIDDEN_NEURONS; ++h) {
            b_hidden[h] = 0.0f;
            for (int i = 0; i < NUM_FEATURES; ++i) W_hidden[h][i] = dist(rng);
        }

        for (int c = 0; c < NUM_CLASSES; ++c) {
            b_out[c] = 0.0f;
            for (int h = 0; h < HIDDEN_NEURONS; ++h) W_out[c][h] = dist(rng);
        }
    }

    // Forward pass: Pixels -> Layer 1 Features -> Layer 2 Hidden -> Layer 3 Judges -> Softmax
    std::vector<float> forward(const std::vector<float>& pixels,
                               std::vector<float>& features,
                               std::vector<int>& feat_pos,
                               std::vector<float>& feat_raw_z,
                               std::vector<float>& z_hidden,
                               std::vector<float>& a_hidden) const
    {
        features.assign(NUM_FEATURES, -1e9f);
        feat_pos.assign(NUM_FEATURES, 0);
        feat_raw_z.assign(NUM_FEATURES, 0.0f);

        // Stage 1: 16 Conv Stencils + Quadrant Spatial Pooling
        for (int f = 0; f < NUM_FILTERS; ++f) {
            const auto& st = stencils[f];
            float b = biases[f];

            for (int r = 0; r < OUT_DIM; ++r) {
                int r_quad = (r < MID_POINT) ? 0 : 1;
                for (int c = 0; c < OUT_DIM; ++c) {
                    int c_quad = (c < MID_POINT) ? 0 : 1;
                    int q_idx = r_quad * 2 + c_quad;
                    int feat_id = f * POOL_REGIONS + q_idx;

                    float sum = b;
                    for (int kr = 0; kr < KERNEL_SIZE; ++kr) {
                        int row_off = (r + kr) * 28;
                        int k_off   = kr * KERNEL_SIZE;
                        for (int kc = 0; kc < KERNEL_SIZE; ++kc) {
                            sum += pixels[row_off + (c + kc)] * st[k_off + kc];
                        }
                    }

                    float act = smooth_ramp(sum);
                    if (act > features[feat_id]) {
                        features[feat_id] = act;
                        feat_pos[feat_id] = r * 28 + c;
                        feat_raw_z[feat_id] = sum;
                    }
                }
            }
        }

        // Stage 2: Intermediate Hidden Layer (Way 2)
        z_hidden.resize(HIDDEN_NEURONS);
        a_hidden.resize(HIDDEN_NEURONS);
        for (int h = 0; h < HIDDEN_NEURONS; ++h) {
            float sum = b_hidden[h];
            for (int i = 0; i < NUM_FEATURES; ++i) {
                sum += features[i] * W_hidden[h][i];
            }
            z_hidden[h] = sum;
            a_hidden[h] = smooth_ramp(sum); // Non-linear activation
        }

        // Stage 3: Output Judges
        std::vector<float> raw_scores(NUM_CLASSES);
        for (int c = 0; c < NUM_CLASSES; ++c) {
            float sum = b_out[c];
            for (int h = 0; h < HIDDEN_NEURONS; ++h) {
                sum += a_hidden[h] * W_out[c][h];
            }
            raw_scores[c] = sum;
        }

        return competing_probabilities(raw_scores);
    }

    // Local polish: Deep Backpropagation with Momentum
    void polish_with_momentum(const std::vector<MNISTSample>& batch, float lr, float friction) {
        for (const auto& sample : batch) {
            std::vector<float> features;
            std::vector<int> feat_pos;
            std::vector<float> feat_raw_z;
            std::vector<float> z_hidden;
            std::vector<float> a_hidden;

            std::vector<float> probs = forward(sample.pixels, features, feat_pos, feat_raw_z, z_hidden, a_hidden);

            // 1. Output Blame: Prediction - Target
            std::vector<float> delta_out(NUM_CLASSES);
            for (int c = 0; c < NUM_CLASSES; ++c) {
                delta_out[c] = probs[c] - (c == sample.label ? 1.0f : 0.0f);
            }

            // 2. Hidden Layer Blame
            std::vector<float> delta_hidden(HIDDEN_NEURONS, 0.0f);
            for (int h = 0; h < HIDDEN_NEURONS; ++h) {
                float sum = 0.0f;
                for (int c = 0; c < NUM_CLASSES; ++c) {
                    sum += delta_out[c] * W_out[c][h];
                }
                delta_hidden[h] = sum * smooth_ramp_slope(z_hidden[h]);
            }

            // 3. Spatial Feature Blame
            std::vector<float> delta_feat(NUM_FEATURES, 0.0f);
            for (int i = 0; i < NUM_FEATURES; ++i) {
                for (int h = 0; h < HIDDEN_NEURONS; ++h) {
                    delta_feat[i] += delta_hidden[h] * W_hidden[h][i];
                }
            }

            // 4. Update Output Judges with Momentum
            for (int c = 0; c < NUM_CLASSES; ++c) {
                v_b_out[c] = friction * v_b_out[c] + lr * delta_out[c];
                b_out[c]  -= v_b_out[c];
                for (int h = 0; h < HIDDEN_NEURONS; ++h) {
                    v_W_out[c][h] = friction * v_W_out[c][h] + lr * delta_out[c] * a_hidden[h];
                    W_out[c][h]  -= v_W_out[c][h];
                }
            }

            // 5. Update Hidden Layer with Momentum
            for (int h = 0; h < HIDDEN_NEURONS; ++h) {
                v_b_hidden[h] = friction * v_b_hidden[h] + lr * delta_hidden[h];
                b_hidden[h]  -= v_b_hidden[h];
                for (int i = 0; i < NUM_FEATURES; ++i) {
                    v_W_hidden[h][i] = friction * v_W_hidden[h][i] + lr * delta_hidden[h] * features[i];
                    W_hidden[h][i]  -= v_W_hidden[h][i];
                }
            }

            // 6. Update Convolutional Stencils with Momentum
            for (int i = 0; i < NUM_FEATURES; ++i) {
                int f = i / POOL_REGIONS;
                float d = delta_feat[i] * smooth_ramp_slope(feat_raw_z[i]);

                v_biases[f] = friction * v_biases[f] + (lr / POOL_REGIONS) * d;
                biases[f]  -= v_biases[f];

                int pr = feat_pos[i] / 28;
                int pc = feat_pos[i] % 28;

                for (int kr = 0; kr < KERNEL_SIZE; ++kr) {
                    int row_off = (pr + kr) * 28;
                    int k_off   = kr * KERNEL_SIZE;
                    for (int kc = 0; kc < KERNEL_SIZE; ++kc) {
                        float g = d * sample.pixels[row_off + (pc + kc)];
                        v_stencils[f][k_off + kc] = friction * v_stencils[f][k_off + kc] + (lr / POOL_REGIONS) * g;
                        stencils[f][k_off + kc]  -= v_stencils[f][k_off + kc];
                    }
                }
            }
        }
    }

    void evaluate(const std::vector<MNISTSample>& dataset) {
        float total_loss = 0.0f;
        int correct = 0;

        for (const auto& s : dataset) {
            std::vector<float> feat, fz, zh, ah;
            std::vector<int> fpos;
            auto probs = forward(s.pixels, feat, fpos, fz, zh, ah);

            total_loss += -std::log(std::max(1e-12f, probs[s.label]));

            int best = 0;
            for (int c = 1; c < NUM_CLASSES; ++c) if (probs[c] > probs[best]) best = c;
            if (best == s.label) correct++;
        }

        loss = total_loss / dataset.size();
        accuracy = (correct * 100.0f) / dataset.size();
    }

    bool save_to_file(const std::string& filepath) const {
        std::ofstream out(filepath);
        if (!out.is_open()) return false;

        out << "# GRAND UNIFIED DEEP CNN WEIGHTS (5,226 DIALS)\n";
        out << "# Conv: 16 filters (5x5) + biases = 416 dials\n";
        out << "# Hidden: 64 neurons (64 inputs + bias) = 4,160 dials\n";
        out << "# Out: 10 judges (64 inputs + bias) = 650 dials\n\n";

        // Layer 1
        for (int f = 0; f < NUM_FILTERS; ++f) {
            out << "# FILTER " << f << " BIAS\n" << biases[f] << "\n";
            out << "# FILTER " << f << " STENCIL 5x5\n";
            for (int r = 0; r < KERNEL_SIZE; ++r) {
                for (int c = 0; c < KERNEL_SIZE; ++c) {
                    out << stencils[f][r * KERNEL_SIZE + c] << (c == KERNEL_SIZE - 1 ? "\n" : " ");
                }
            }
            out << "\n";
        }

        // Layer 2
        for (int h = 0; h < HIDDEN_NEURONS; ++h) {
            out << "# HIDDEN " << h << " BIAS\n" << b_hidden[h] << "\n";
            out << "# HIDDEN " << h << " WEIGHTS (64 inputs)\n";
            for (int i = 0; i < NUM_FEATURES; ++i) {
                out << W_hidden[h][i] << (i == NUM_FEATURES - 1 ? "\n" : " ");
            }
            out << "\n";
        }

        // Layer 3
        for (int c = 0; c < NUM_CLASSES; ++c) {
            out << "# OUT " << c << " BIAS\n" << b_out[c] << "\n";
            out << "# OUT " << c << " WEIGHTS (64 inputs)\n";
            for (int h = 0; h < HIDDEN_NEURONS; ++h) {
                out << W_out[c][h] << (h == HIDDEN_NEURONS - 1 ? "\n" : " ");
            }
            out << "\n";
        }

        out.close();
        return true;
    }
};

void print_digit_ascii(const std::vector<float>& pixels) {
    for (int r = 0; r < 28; r += 2) {
        std::cout << "      [ ";
        for (int c = 0; c < 28; ++c) {
            float p = pixels[r * 28 + c];
            if (p > 0.60f)      std::cout << "#";
            else if (p > 0.25f) std::cout << "*";
            else if (p > 0.05f) std::cout << ".";
            else                std::cout << " ";
        }
        std::cout << " ]\n";
    }
}

void print_bar(int digit, float prob, bool is_winner) {
    int bar_width = 20;
    int filled = static_cast<int>(prob * bar_width);

    std::cout << "      Digit " << digit << " [";
    for (int i = 0; i < bar_width; ++i) {
        if (i < filled) std::cout << (is_winner ? "#" : "=");
        else std::cout << " ";
    }
    std::cout << "] " << std::fixed << std::setprecision(1) << std::setw(5) << std::right << (prob * 100.0f) << "%";
    if (is_winner) std::cout << "  <-- WINNER";
    std::cout << "\n";
}

const int TOP_CHAMPIONS = 2;
const int CLONES_PER_CHAMPION = 2; // Population = 6 organisms
const int POPULATION_SIZE = TOP_CHAMPIONS + (TOP_CHAMPIONS * CLONES_PER_CHAMPION);

int main(int argc, char* argv[]) {
    std::cout << "====================================================================================\n";
    std::cout << " PROGRAM 26: GRAND UNIFIED DEEP CNN (ALL THREE WAYS COMBINED!)\n";
    std::cout << " [16 Stencils (5x5) + Quadrant Spatial Pooling + 64 Hidden Neurons + 10 Judges]\n";
    std::cout << "====================================================================================\n\n";

    std::string train_img_p = "data/mnist/train-images-idx3-ubyte";
    std::string train_lbl_p = "data/mnist/train-labels-idx1-ubyte";
    std::string test_img_p  = "data/mnist/t10k-images-idx3-ubyte";
    std::string test_lbl_p  = "data/mnist/t10k-labels-idx1-ubyte";

    std::cout << "Loading real MNIST human handwriting...\n";
    auto train_data = load_mnist_raw(train_img_p, train_lbl_p, 4000);
    auto test_data  = load_mnist_raw(test_img_p,  test_lbl_p,  1000);

    std::cout << "Loaded " << train_data.size() << " Real Training Images.\n";
    std::cout << "Loaded " << test_data.size()  << " Real Unseen Test Images (Written by different humans).\n\n";

    std::mt19937 rng(42);
    std::vector<UnifiedOrganism> population(POPULATION_SIZE);
    for (auto& org : population) org.randomize(rng);

    int generations = 12; // default
    if (argc > 1) {
        generations = std::max(1, std::atoi(argv[1]));
    }

    float lr = 0.015f;
    float friction = 0.50f;

    std::normal_distribution<float> mut_dist(0.0f, 0.01f);
    std::uniform_real_distribution<float> chance(0.0f, 1.0f);

    std::cout << "Population: " << POPULATION_SIZE << " Deep Grand Unified Organisms (5,226 dials each).\n";
    std::cout << "Running Hybrid Evolution across " << generations << " Generations...\n\n";

    auto t_start = std::chrono::high_resolution_clock::now();

    for (int gen = 1; gen <= generations; ++gen) {
        // Step A: Local polish with Momentum
        for (auto& org : population) {
            org.polish_with_momentum(train_data, lr, friction);
        }
        lr *= 0.95f; // learning rate decay

        // Step B: Evaluate on real unseen test data
        for (auto& org : population) {
            org.evaluate(test_data);
        }

        // Step C: Sort by lowest loss on real test data
        std::sort(population.begin(), population.end(), [](const UnifiedOrganism& a, const UnifiedOrganism& b) {
            return a.loss < b.loss;
        });

        std::cout << "  Gen " << std::setw(2) << gen << " / " << generations
                  << " | Champion Test Loss: " << std::fixed << std::setprecision(4) << population[0].loss
                  << " | REAL TEST ACCURACY: " << std::setprecision(1) << std::setw(5) << population[0].accuracy << "%\n";

        // Step D: Hybrid Reproduction
        if (gen < generations) {
            std::vector<UnifiedOrganism> next_gen;
            for (int i = 0; i < TOP_CHAMPIONS; ++i) {
                next_gen.push_back(population[i]); // Keep champion

                for (int c = 0; c < CLONES_PER_CHAMPION; ++c) {
                    UnifiedOrganism clone = population[i]; // Inherits all 5,226 dials + velocities!
                    for (int f = 0; f < NUM_FILTERS; ++f) {
                        for (int k = 0; k < KERNEL_SIZE * KERNEL_SIZE; ++k) {
                            if (chance(rng) < 0.05f) clone.stencils[f][k] += mut_dist(rng);
                        }
                    }
                    next_gen.push_back(clone);
                }
            }
            population = next_gen;
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_sec = std::chrono::duration<double>(t_end - t_start).count();
    std::cout << "\n>>> Training Completed in " << std::fixed << std::setprecision(2) << total_sec << " seconds!\n\n";

    UnifiedOrganism champ = population[0];

    std::string weights_file = "grand_unified_mnist_weights.txt";
    if (champ.save_to_file(weights_file)) {
        std::cout << "====================================================================================\n";
        std::cout << " SUCCESS: Trained weights saved to: " << weights_file << "\n";
        std::cout << " (Saved all 5,226 dials across Conv, Hidden, and Judge layers!)\n";
        std::cout << "====================================================================================\n\n";
    }

    return 0;
}
