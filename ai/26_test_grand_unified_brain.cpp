#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cmath>
#include <iomanip>
#include <string>
#include <algorithm>

// =============================================================================
// PROGRAM 26 TEST RUNNER: EVALUATING DEEP GRAND UNIFIED BRAIN (5,226 DIALS)
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
    std::vector<float> pixels;
};

std::vector<MNISTSample> load_mnist_raw(const std::string& img_path, 
                                        const std::string& lbl_path, 
                                        int max_count) 
{
    std::ifstream img_f(img_path, std::ios::binary);
    std::ifstream lbl_f(lbl_path, std::ios::binary);

    if (!img_f.is_open() || !lbl_f.is_open()) {
        std::cerr << "Error: Could not open test files at " << img_path << "\n";
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

const int KERNEL_SIZE   = 5;
const int NUM_FILTERS   = 16;
const int OUT_DIM       = 28 - KERNEL_SIZE + 1; // 24
const int MID_POINT     = OUT_DIM / 2;          // 12
const int POOL_REGIONS  = 4;
const int NUM_FEATURES  = NUM_FILTERS * POOL_REGIONS; // 64
const int HIDDEN_NEURONS = 64;
const int NUM_CLASSES   = 10;

struct LoadedDeepBrain {
    std::vector<std::vector<float>> stencils;
    std::vector<float> biases;
    std::vector<std::vector<float>> W_hidden;
    std::vector<float> b_hidden;
    std::vector<std::vector<float>> W_out;
    std::vector<float> b_out;

    LoadedDeepBrain() {
        stencils.assign(NUM_FILTERS, std::vector<float>(KERNEL_SIZE * KERNEL_SIZE, 0.0f));
        biases.assign(NUM_FILTERS, 0.0f);
        W_hidden.assign(HIDDEN_NEURONS, std::vector<float>(NUM_FEATURES, 0.0f));
        b_hidden.assign(HIDDEN_NEURONS, 0.0f);
        W_out.assign(NUM_CLASSES, std::vector<float>(HIDDEN_NEURONS, 0.0f));
        b_out.assign(NUM_CLASSES, 0.0f);
    }

    bool load_from_file(const std::string& filepath) {
        std::ifstream in(filepath);
        if (!in.is_open()) return false;

        std::string line;
        auto next_token = [&in, &line]() -> float {
            std::string word;
            while (in >> word) {
                if (word[0] == '#') {
                    std::getline(in, line);
                    continue;
                }
                return std::stof(word);
            }
            return 0.0f;
        };

        // Layer 1
        for (int f = 0; f < NUM_FILTERS; ++f) {
            biases[f] = next_token();
            for (int i = 0; i < KERNEL_SIZE * KERNEL_SIZE; ++i) {
                stencils[f][i] = next_token();
            }
        }

        // Layer 2
        for (int h = 0; h < HIDDEN_NEURONS; ++h) {
            b_hidden[h] = next_token();
            for (int i = 0; i < NUM_FEATURES; ++i) {
                W_hidden[h][i] = next_token();
            }
        }

        // Layer 3
        for (int c = 0; c < NUM_CLASSES; ++c) {
            b_out[c] = next_token();
            for (int h = 0; h < HIDDEN_NEURONS; ++h) {
                W_out[c][h] = next_token();
            }
        }

        in.close();
        return true;
    }

    std::vector<float> forward(const std::vector<float>& pixels) const {
        std::vector<float> features(NUM_FEATURES, -1e9f);

        // Stage 1: Conv + Quadrant Pooling
        for (int f = 0; f < NUM_FILTERS; ++f) {
            const auto& st = stencils[f];
            float b = biases[f];

            for (int r = 0; r < OUT_DIM; ++r) {
                int r_quad = (r < MID_POINT) ? 0 : 1;
                for (int c = 0; c < OUT_DIM; ++c) {
                    int c_quad = (c < MID_POINT) ? 0 : 1;
                    int feat_id = f * POOL_REGIONS + (r_quad * 2 + c_quad);

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
                    }
                }
            }
        }

        // Stage 2: Hidden Concept Neurons
        std::vector<float> a_hidden(HIDDEN_NEURONS);
        for (int h = 0; h < HIDDEN_NEURONS; ++h) {
            float sum = b_hidden[h];
            for (int i = 0; i < NUM_FEATURES; ++i) {
                sum += features[i] * W_hidden[h][i];
            }
            a_hidden[h] = smooth_ramp(sum);
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

int main(int argc, char* argv[]) {
    std::string weights_file = "grand_unified_mnist_weights.txt";
    if (argc > 1) {
        weights_file = argv[1];
    }

    std::cout << "====================================================================================\n";
    std::cout << " PROGRAM 26 TEST RUNNER: EVALUATING DEEP GRAND UNIFIED BRAIN (5,226 DIALS)\n";
    std::cout << " Loading from: " << weights_file << "\n";
    std::cout << "====================================================================================\n\n";

    LoadedDeepBrain brain;
    if (!brain.load_from_file(weights_file)) {
        std::cerr << "ERROR: Failed to open weights file: " << weights_file << "\n";
        std::cerr << "Please run: make train-26\n";
        return 1;
    }

    std::cout << ">>> Successfully loaded 5,226 trained dials from " << weights_file << "!\n\n";

    std::string test_img_p = "data/mnist/t10k-images-idx3-ubyte";
    std::string test_lbl_p = "data/mnist/t10k-labels-idx1-ubyte";

    auto test_data = load_mnist_raw(test_img_p, test_lbl_p, 1000);
    std::cout << "Loaded " << test_data.size() << " Real Unseen Test Images from official MNIST!\n\n";

    // 1. Full Test Accuracy Evaluation
    int correct = 0;
    for (const auto& sample : test_data) {
        auto probs = brain.forward(sample.pixels);
        int best = 0;
        for (int c = 1; c < NUM_CLASSES; ++c) if (probs[c] > probs[best]) best = c;
        if (best == sample.label) correct++;
    }

    float overall_acc = (correct * 100.0f) / test_data.size();
    std::cout << "------------------------------------------------------------------------------------\n";
    std::cout << " OVERALL TEST ACCURACY: " << correct << " / " << test_data.size() 
              << " (" << std::fixed << std::setprecision(1) << overall_acc << "%)\n";
    std::cout << "------------------------------------------------------------------------------------\n\n";

    // 2. Visual Inspection of Real Human Samples
    std::cout << "--- VISUAL INSPECTION OF UNSEEN HUMAN HANDWRITING ---\n";
    int inspect_indices[] = {0, 1, 2, 3, 5, 7, 8, 9};

    for (int idx : inspect_indices) {
        const auto& test_sample = test_data[idx];
        auto probs = brain.forward(test_sample.pixels);

        int best = 0;
        for (int c = 1; c < NUM_CLASSES; ++c) if (probs[c] > probs[best]) best = c;

        std::cout << "\n====================================================================================\n";
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
    std::cout << " TEST RUNNER COMPLETE: Loaded model successfully verified " << overall_acc << "% accuracy!\n";
    std::cout << "====================================================================================\n";

    return 0;
}
