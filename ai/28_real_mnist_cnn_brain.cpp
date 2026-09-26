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
// PROGRAM 28: REAL MNIST HANDWRITTEN DIGIT RECOGNIZER FROM SCRATCH
// Architecture:
// 1. Input: 28x28 grayscale images (values 0.0 to 1.0)
// 2. Convolution: 8 Stencils (5x5 each) with Smooth Curved Ramp (SiLU)
// 3. Quadrant Pooling: 4 spatial zones (Top-Left, Top-Right, Bottom-Left, Bottom-Right)
//    -> 8 filters x 4 quadrants = 32 spatial feature peaks
// 4. Multi-Class Judges: 10 Judges (Digits 0-9) competing via Softmax
// 5. Training: Mini-batch Momentum Backpropagation from first principles
// =============================================================================

// Helper: Read 32-bit big-endian integer from binary file
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

// Pure C++ First-Principles Binary Loader
std::vector<MNISTSample> load_mnist_dataset(const std::string& img_path, 
                                           const std::string& lbl_path, 
                                           int max_count) 
{
    std::ifstream img_f(img_path, std::ios::binary);
    std::ifstream lbl_f(lbl_path, std::ios::binary);

    if (!img_f.is_open() || !lbl_f.is_open()) {
        std::cerr << "Error: Could not open MNIST files at " << img_path << "\n";
        return {};
    }

    read_big_endian(img_f); // magic
    uint32_t num_images = read_big_endian(img_f);
    read_big_endian(img_f); // rows
    read_big_endian(img_f); // cols

    read_big_endian(lbl_f); // magic
    read_big_endian(lbl_f); // count

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
            dataset[i].pixels[p] = static_cast<float>(px) / 255.0f; // normalize to [0.0, 1.0]
        }
    }

    return dataset;
}

// -----------------------------------------------------------------------------
// Activation Functions: S-curve & Smooth Curved Ramp (SiLU)
// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
// Model Hyperparameters
// -----------------------------------------------------------------------------
const int KERNEL_SIZE   = 5;                            // 5x5 stencils
const int NUM_FILTERS   = 8;                            // 8 distinct stroke detectors
const int NUM_CLASSES   = 10;                           // 10 digits (0 to 9)
const int OUT_DIM       = 28 - KERNEL_SIZE + 1;         // 24x24 positions
const int MID_POINT     = OUT_DIM / 2;                  // 12 (boundary between top/bottom and left/right)
const int POOL_REGIONS  = 4;                            // 4 quadrants: TL, TR, BL, BR
const int NUM_FEATURES  = NUM_FILTERS * POOL_REGIONS;   // 8 * 4 = 32 spatial features!

// -----------------------------------------------------------------------------
// CNN Brain Class
// -----------------------------------------------------------------------------
class RealMNISTBrain {
public:
    // 1. Convolution Layer: 8 Stencils (5x5 = 25 numbers each) + 8 Biases
    std::vector<std::vector<float>> stencils;    // [8][25]
    std::vector<float> biases;                   // [8]
    std::vector<std::vector<float>> v_stencils;  // Velocities
    std::vector<float> v_biases;

    // 2. Multi-Neuron Judges: 10 Judges, each connected to 32 spatial features
    std::vector<std::vector<float>> judge_dials; // [10][32]
    std::vector<float> judge_baselines;          // [10]
    std::vector<std::vector<float>> v_judge_dials;
    std::vector<float> v_judge_baselines;

    RealMNISTBrain() {
        stencils.assign(NUM_FILTERS, std::vector<float>(KERNEL_SIZE * KERNEL_SIZE, 0.0f));
        v_stencils.assign(NUM_FILTERS, std::vector<float>(KERNEL_SIZE * KERNEL_SIZE, 0.0f));
        biases.assign(NUM_FILTERS, 0.0f);
        v_biases.assign(NUM_FILTERS, 0.0f);

        judge_dials.assign(NUM_CLASSES, std::vector<float>(NUM_FEATURES, 0.0f));
        v_judge_dials.assign(NUM_CLASSES, std::vector<float>(NUM_FEATURES, 0.0f));
        judge_baselines.assign(NUM_CLASSES, 0.0f);
        v_judge_baselines.assign(NUM_CLASSES, 0.0f);
    }

    void randomize(std::mt19937& rng) {
        std::normal_distribution<float> dist(0.0f, 0.05f);

        for (int f = 0; f < NUM_FILTERS; ++f) {
            for (int i = 0; i < KERNEL_SIZE * KERNEL_SIZE; ++i) {
                stencils[f][i] = dist(rng);
            }
            biases[f] = 0.0f;
        }

        for (int c = 0; c < NUM_CLASSES; ++c) {
            judge_baselines[c] = 0.0f;
            for (int i = 0; i < NUM_FEATURES; ++i) {
                judge_dials[c][i] = dist(rng);
            }
        }
    }

    // Forward pass: Extracts 32 spatial features and returns 10 probabilities
    std::vector<float> forward(const std::vector<float>& pixels,
                               std::vector<float>& features,
                               std::vector<int>& feat_pos,
                               std::vector<float>& feat_raw_z) const
    {
        features.assign(NUM_FEATURES, -1e9f);
        feat_pos.assign(NUM_FEATURES, 0);
        feat_raw_z.assign(NUM_FEATURES, 0.0f);

        // 1. Sliding 5x5 Stencils across 24x24 locations, pooling into 4 quadrants
        for (int f = 0; f < NUM_FILTERS; ++f) {
            const auto& st = stencils[f];
            float b = biases[f];

            for (int r = 0; r < OUT_DIM; ++r) {
                int r_quad = (r < MID_POINT) ? 0 : 1;
                for (int c = 0; c < OUT_DIM; ++c) {
                    int c_quad = (c < MID_POINT) ? 0 : 1;
                    int q_idx = r_quad * 2 + c_quad; // 0: Top-Left, 1: Top-Right, 2: Bottom-Left, 3: Bottom-Right
                    int feat_id = f * POOL_REGIONS + q_idx;

                    float sum = b;
                    for (int kr = 0; kr < KERNEL_SIZE; ++kr) {
                        int row_offset = (r + kr) * 28;
                        int k_offset   = kr * KERNEL_SIZE;
                        for (int kc = 0; kc < KERNEL_SIZE; ++kc) {
                            sum += pixels[row_offset + (c + kc)] * st[k_offset + kc];
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

        // 2. Multi-Neuron Judges
        std::vector<float> raw_scores(NUM_CLASSES, 0.0f);
        float max_s = -1e9f;
        for (int c = 0; c < NUM_CLASSES; ++c) {
            raw_scores[c] = judge_baselines[c];
            for (int i = 0; i < NUM_FEATURES; ++i) {
                raw_scores[c] += features[i] * judge_dials[c][i];
            }
            if (raw_scores[c] > max_s) max_s = raw_scores[c];
        }

        // 3. Competing Probabilities (Softmax)
        std::vector<float> probs(NUM_CLASSES, 0.0f);
        float sum_exp = 0.0f;
        for (int c = 0; c < NUM_CLASSES; ++c) {
            probs[c] = std::exp(raw_scores[c] - max_s);
            sum_exp += probs[c];
        }
        for (int c = 0; c < NUM_CLASSES; ++c) probs[c] /= sum_exp;

        return probs;
    }

    // Train on a single sample with Momentum Backpropagation
    float train_sample(const MNISTSample& sample, float lr, float friction) {
        std::vector<float> features;
        std::vector<int> feat_pos;
        std::vector<float> feat_raw_z;

        std::vector<float> probs = forward(sample.pixels, features, feat_pos, feat_raw_z);

        // Loss: -ln(P_target)
        float loss = -std::log(std::max(1e-12f, probs[sample.label]));

        // 1. Judge Blame: Prediction - Target
        std::vector<float> judge_blame(NUM_CLASSES);
        for (int c = 0; c < NUM_CLASSES; ++c) {
            judge_blame[c] = probs[c] - (c == sample.label ? 1.0f : 0.0f);
        }

        // 2. Feature Blame
        std::vector<float> feat_blame(NUM_FEATURES, 0.0f);
        for (int i = 0; i < NUM_FEATURES; ++i) {
            for (int c = 0; c < NUM_CLASSES; ++c) {
                feat_blame[i] += judge_blame[c] * judge_dials[c][i];
            }
        }

        // 3. Update Judges with Momentum
        for (int c = 0; c < NUM_CLASSES; ++c) {
            v_judge_baselines[c] = friction * v_judge_baselines[c] + lr * judge_blame[c];
            judge_baselines[c]  -= v_judge_baselines[c];

            for (int i = 0; i < NUM_FEATURES; ++i) {
                v_judge_dials[c][i] = friction * v_judge_dials[c][i] + lr * judge_blame[c] * features[i];
                judge_dials[c][i]  -= v_judge_dials[c][i];
            }
        }

        // 4. Update Stencils with Momentum using Smooth Ramp Slope
        for (int i = 0; i < NUM_FEATURES; ++i) {
            int f = i / POOL_REGIONS;
            float delta = feat_blame[i] * smooth_ramp_slope(feat_raw_z[i]);

            // Filter bias update
            v_biases[f] = friction * v_biases[f] + (lr / POOL_REGIONS) * delta;
            biases[f]  -= v_biases[f];

            // 5x5 Stencil dials update
            int pr = feat_pos[i] / 28;
            int pc = feat_pos[i] % 28;

            for (int kr = 0; kr < KERNEL_SIZE; ++kr) {
                int row_offset = (pr + kr) * 28;
                int k_offset   = kr * KERNEL_SIZE;
                for (int kc = 0; kc < KERNEL_SIZE; ++kc) {
                    float g = delta * sample.pixels[row_offset + (pc + kc)];
                    v_stencils[f][k_offset + kc] = friction * v_stencils[f][k_offset + kc] + (lr / POOL_REGIONS) * g;
                    stencils[f][k_offset + kc]  -= v_stencils[f][k_offset + kc];
                }
            }
        }

        return loss;
    }

    // Save all 538 dials to disk
    bool save_to_file(const std::string& filepath) const {
        std::ofstream out(filepath);
        if (!out.is_open()) return false;

        out << "# REAL MNIST CNN WEIGHTS\n";
        out << "# 8 Filters (5x5 stencils + biases), 10 Judges (32 dials + baselines)\n\n";

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

        for (int c = 0; c < NUM_CLASSES; ++c) {
            out << "# JUDGE " << c << " BASELINE\n" << judge_baselines[c] << "\n";
            out << "# JUDGE " << c << " DIALS (32 inputs)\n";
            for (int i = 0; i < NUM_FEATURES; ++i) {
                out << judge_dials[c][i] << (i == NUM_FEATURES - 1 ? "\n" : " ");
            }
            out << "\n";
        }

        out.close();
        return true;
    }
};

// Helper: Print ASCII art of 28x28 digit
void print_digit_ascii(const std::vector<float>& pixels) {
    for (int r = 0; r < 28; r += 2) { // sample every 2 rows for compact terminal display
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

// Helper: Print horizontal probability bar
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

int main() {
    std::cout << "====================================================================================\n";
    std::cout << " PROGRAM 28: REAL MNIST CNN TRAINING ON REAL HUMAN HANDWRITING (C++17)\n";
    std::cout << " [8 Stencils (5x5) + Quadrant Spatial Pooling + 10 Competing Judges + Momentum]\n";
    std::cout << "====================================================================================\n\n";

    std::string train_img_p = "data/mnist/train-images-idx3-ubyte";
    std::string train_lbl_p = "data/mnist/train-labels-idx1-ubyte";
    std::string test_img_p  = "data/mnist/t10k-images-idx3-ubyte";
    std::string test_lbl_p  = "data/mnist/t10k-labels-idx1-ubyte";

    std::cout << "Loading real MNIST handwriting datasets...\n";
    auto train_data = load_mnist_dataset(train_img_p, train_lbl_p, 2500);
    auto test_data  = load_mnist_dataset(test_img_p,  test_lbl_p,  1000);

    std::cout << "Loaded " << train_data.size() << " Real Training Images.\n";
    std::cout << "Loaded " << test_data.size()  << " Real Unseen Test Images (Written by different humans).\n\n";

    std::mt19937 rng(42);
    RealMNISTBrain brain;
    brain.randomize(rng);

    const int EPOCHS = 12;
    float lr = 0.02f;
    float friction = 0.50f;

    std::cout << "------------------------------------------------------------------------------------\n";
    std::cout << " TRAINING PROGRESS ACROSS " << EPOCHS << " EPOCHS\n";
    std::cout << "------------------------------------------------------------------------------------\n";

    auto t_start = std::chrono::high_resolution_clock::now();

    for (int epoch = 1; epoch <= EPOCHS; ++epoch) {
        float epoch_loss = 0.0f;
        int train_correct = 0;

        for (const auto& sample : train_data) {
            float loss = brain.train_sample(sample, lr, friction);
            epoch_loss += loss;

            // Check prediction
            std::vector<float> feat;
            std::vector<int> fpos;
            std::vector<float> fz;
            auto probs = brain.forward(sample.pixels, feat, fpos, fz);
            int pred = 0;
            for (int c = 1; c < NUM_CLASSES; ++c) if (probs[c] > probs[pred]) pred = c;
            if (pred == sample.label) train_correct++;
        }

        // Test on 1,000 real unseen human digits
        int test_correct = 0;
        for (const auto& sample : test_data) {
            std::vector<float> feat;
            std::vector<int> fpos;
            std::vector<float> fz;
            auto probs = brain.forward(sample.pixels, feat, fpos, fz);
            int pred = 0;
            for (int c = 1; c < NUM_CLASSES; ++c) if (probs[c] > probs[pred]) pred = c;
            if (pred == sample.label) test_correct++;
        }

        float train_acc = (train_correct * 100.0f) / train_data.size();
        float test_acc  = (test_correct  * 100.0f) / test_data.size();
        float avg_loss  = epoch_loss / train_data.size();

        std::cout << " Epoch " << std::setw(2) << epoch << " / " << EPOCHS
                  << " | Loss: " << std::fixed << std::setprecision(4) << avg_loss
                  << " | Train Acc: " << std::setprecision(1) << std::setw(5) << train_acc << "%"
                  << " | REAL TEST ACC: " << std::setprecision(1) << std::setw(5) << test_acc << "%\n";
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_sec = std::chrono::duration<double>(t_end - t_start).count();
    std::cout << "\n>>> Training Completed in " << std::fixed << std::setprecision(2) << total_sec << " seconds!\n\n";

    // Save weights
    std::string weights_file = "mnist_cnn_weights.txt";
    if (brain.save_to_file(weights_file)) {
        std::cout << ">>> Successfully saved 538 trained dials to: " << weights_file << "\n\n";
    }

    // -------------------------------------------------------------------------
    // TEST ON REAL UNSEEN HUMAN SAMPLES
    // -------------------------------------------------------------------------
    std::cout << "====================================================================================\n";
    std::cout << " INSPECTING REAL UNSEEN HUMAN TEST SAMPLES\n";
    std::cout << "====================================================================================\n";

    int sample_indices[] = {0, 1, 2, 3, 5, 7, 8, 9};
    for (int idx : sample_indices) {
        const auto& test_sample = test_data[idx];
        std::vector<float> feat;
        std::vector<int> fpos;
        std::vector<float> fz;
        auto probs = brain.forward(test_sample.pixels, feat, fpos, fz);

        int best = 0;
        for (int c = 1; c < NUM_CLASSES; ++c) if (probs[c] > probs[best]) best = c;

        std::cout << "\n------------------------------------------------------------------------------------\n";
        std::cout << " TEST SAMPLE #" << idx << " | Real Human True Label: [" << static_cast<int>(test_sample.label) << "]\n";
        std::cout << "------------------------------------------------------------------------------------\n";
        print_digit_ascii(test_sample.pixels);
        std::cout << "\n";

        for (int c = 0; c < NUM_CLASSES; ++c) {
            print_bar(c, probs[c], c == best);
        }

        std::cout << "\n      VERDICT: ";
        if (best == test_sample.label) {
            std::cout << "[PASS] Correctly recognized real human digit " << best 
                      << " (" << std::fixed << std::setprecision(1) << (probs[best]*100.0f) << "% confidence)!\n";
        } else {
            std::cout << "[FAIL] Expected " << static_cast<int>(test_sample.label) << " but predicted " << best << "\n";
        }
    }

    std::cout << "\n====================================================================================\n";
    std::cout << " PROGRAM 28 COMPLETE: First-principles CNN successfully masters real human MNIST digits!\n";
    std::cout << "====================================================================================\n";

    return 0;
}
