/**
 * ====================================================================================
 * PROGRAM 33: DYNAMIC-CONTEXT RECURRENT NEURAL NETWORK (RNN) & ADAM OPTIMIZER
 * ====================================================================================
 * Kushagra's AI Journey: From Fixed Windows to Dynamic Long Memory
 *
 * Core Breakthroughs Over Program 32:
 * 1. DYNAMIC INPUT LENGTH:
 *    - In Program 32, we were rigidly trapped in a fixed 2-word window.
 *    - In Program 33, the network possesses an INTERNAL MEMORY LOOP (Hidden State h_t).
 *    - It can accept prompts of ANY length (1 word, 3 words, 8 words, a full paragraph!).
 * 2. DYNAMIC MOMENTUM (THE FIRST-PRINCIPLES ADAM OPTIMIZER):
 *    - Eliminates the guesswork of fixed momentum (0.85).
 *    - Every single dial tracks speed (m_t) and terrain bumpiness (v_t) independently.
 * 3. SMOOTH DYNAMIC LEAKY ACTIVATION:
 *    - Replaces the hard, sharp corner of standard LeakyReLU at zero with a smooth,
 *      silky, continuously differentiable curve:
 *         f(z) = alpha * z + (1 - alpha) * z * sigmoid(z)
 *    - The leak rate 'alpha' is itself a DYNAMIC TRAINABLE DIAL tuned by Adam!
 * 4. BACKPROPAGATION THROUGH TIME (BPTT):
 *    - Error flows backwards through time steps across entire sentences.
 *
 * 100% Modern First-Principles C++17 — Zero External Machine Learning Libraries!
 * ====================================================================================
 */

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <unordered_map>
#include <cmath>
#include <random>
#include <iomanip>
#include <algorithm>
#include <chrono>

// ====================================================================================
// STEP 1: SMOOTH DYNAMIC LEAKY ACTIVATION (FIRST PRINCIPLES)
// ====================================================================================

inline double sigmoid(double z) {
    if (z > 20.0) return 1.0;
    if (z < -20.0) return 0.0;
    return 1.0 / (1.0 + std::exp(-z));
}

// Smooth Dynamic Leaky Function:
// Combines a linear leak (alpha * z) with a silky sigmoid ramp ((1 - alpha) * z * sigmoid(z))
// Completely smooth everywhere, no sharp kinks at zero, and never dies on negative inputs!
inline double smooth_dynamic_leaky(double z, double alpha) {
    double s = sigmoid(z);
    return alpha * z + (1.0 - alpha) * z * s;
}

// Derivative of Smooth Dynamic Leaky with respect to z (Chain Rule Slope)
inline double smooth_dynamic_leaky_derivative(double z, double alpha) {
    double s = sigmoid(z);
    return alpha + (1.0 - alpha) * (s + z * s * (1.0 - s));
}

// Derivative with respect to the trainable leak parameter alpha
inline double smooth_dynamic_leaky_d_alpha(double z) {
    double s = sigmoid(z);
    return z * (1.0 - s);
}

std::vector<double> softmax(const std::vector<double>& logits) {
    double max_val = *std::max_element(logits.begin(), logits.end());
    std::vector<double> probs(logits.size());
    double sum = 0.0;
    for (size_t i = 0; i < logits.size(); ++i) {
        probs[i] = std::exp(logits[i] - max_val);
        sum += probs[i];
    }
    for (size_t i = 0; i < logits.size(); ++i) {
        probs[i] /= sum;
    }
    return probs;
}

// ====================================================================================
// STEP 2: FIRST-PRINCIPLES ADAM OPTIMIZER (DYNAMIC MOMENTUM)
// ====================================================================================

struct AdamOptimizer {
    double beta1 = 0.9;
    double beta2 = 0.999;
    double eps = 1e-8;
    int t = 0; // Global time step counter

    inline void step(double& weight, double grad, double& m, double& v, double lr) {
        // 1. Update running speed (1st moment)
        m = beta1 * m + (1.0 - beta1) * grad;

        // 2. Update running terrain bumpiness / variance (2nd moment)
        v = beta2 * v + (1.0 - beta2) * (grad * grad);

        // 3. Bias corrections (fixes zero-start bias)
        double m_hat = m / (1.0 - std::pow(beta1, t));
        double v_hat = v / (1.0 - std::pow(beta2, t));

        // 4. Dynamic adaptive step!
        weight -= lr * (m_hat / (std::sqrt(v_hat) + eps));
    }
};

// ====================================================================================
// STEP 3: RECURRENT NEURAL NETWORK ARCHITECTURE
// ====================================================================================

class RecurrentLanguageModel {
public:
    int V;  // Vocabulary size
    int D;  // Word coordinate dimension (12 dials per word)
    int H;  // Hidden memory state dimension (32 memory cells)

    // Dynamic Leaky Dial: Learned by Adam during training!
    double alpha = 0.03;
    double m_alpha = 0.0, v_alpha = 0.0;

    // 1. Word Coordinate Table: C[V][D]
    std::vector<std::vector<double>> C;
    std::vector<std::vector<double>> mC, vC;

    // 2. Input-to-Hidden Weights: Wxh[D][H]
    std::vector<std::vector<double>> Wxh;
    std::vector<std::vector<double>> mWxh, vWxh;

    // 3. Hidden-to-Hidden (RECURRENT MEMORY LOOP): Whh[H][H]
    std::vector<std::vector<double>> Whh;
    std::vector<std::vector<double>> mWhh, vWhh;

    // 4. Hidden Bias: Bh[H]
    std::vector<double> Bh;
    std::vector<double> mBh, vBh;

    // 5. Hidden-to-Output (Vocabulary Judges): Why[H][V]
    std::vector<std::vector<double>> Why;
    std::vector<std::vector<double>> mWhy, vWhy;

    // 6. Output Bias: By[V]
    std::vector<double> By;
    std::vector<double> mBy, vBy;

    AdamOptimizer adam;

    RecurrentLanguageModel(int vocab_size, int coord_dim = 12, int hidden_dim = 32)
        : V(vocab_size), D(coord_dim), H(hidden_dim) {

        std::mt19937 rng(42);
        double scale_wxh = std::sqrt(2.0 / D);
        double scale_whh = std::sqrt(2.0 / H);
        double scale_why = std::sqrt(2.0 / H);
        double scale_emb = 0.1;

        std::normal_distribution<double> dist_wxh(0.0, scale_wxh);
        std::normal_distribution<double> dist_whh(0.0, scale_whh);
        std::normal_distribution<double> dist_why(0.0, scale_why);
        std::normal_distribution<double> dist_emb(0.0, scale_emb);

        C.assign(V, std::vector<double>(D));
        mC.assign(V, std::vector<double>(D, 0.0));
        vC.assign(V, std::vector<double>(D, 0.0));
        for (int i = 0; i < V; ++i) {
            for (int d = 0; d < D; ++d) C[i][d] = dist_emb(rng);
        }

        Wxh.assign(D, std::vector<double>(H));
        mWxh.assign(D, std::vector<double>(H, 0.0));
        vWxh.assign(D, std::vector<double>(H, 0.0));
        for (int i = 0; i < D; ++i) {
            for (int j = 0; j < H; ++j) Wxh[i][j] = dist_wxh(rng);
        }

        Whh.assign(H, std::vector<double>(H));
        mWhh.assign(H, std::vector<double>(H, 0.0));
        vWhh.assign(H, std::vector<double>(H, 0.0));
        for (int i = 0; i < H; ++i) {
            for (int j = 0; j < H; ++j) Whh[i][j] = dist_whh(rng);
        }

        Bh.assign(H, 0.0);
        mBh.assign(H, 0.0);
        vBh.assign(H, 0.0);

        Why.assign(H, std::vector<double>(V));
        mWhy.assign(H, std::vector<double>(V, 0.0));
        vWhy.assign(H, std::vector<double>(V, 0.0));
        for (int i = 0; i < H; ++i) {
            for (int k = 0; k < V; ++k) Why[i][k] = dist_why(rng);
        }

        By.assign(V, 0.0);
        mBy.assign(V, 0.0);
        vBy.assign(V, 0.0);
    }

    // ------------------------------------------------------------------------
    // FORWARD PASS & TRAINING STEP (BACKPROPAGATION THROUGH TIME)
    // ------------------------------------------------------------------------
    double train_sentence(const std::vector<int>& tokens, double lr, int& correct_words, int& total_words) {
        int T = static_cast<int>(tokens.size()) - 1;
        if (T <= 0) return 0.0;

        adam.t++;

        std::vector<std::vector<double>> h_pre(T, std::vector<double>(H, 0.0));
        std::vector<std::vector<double>> h(T, std::vector<double>(H, 0.0));
        std::vector<std::vector<double>> logits(T, std::vector<double>(V, 0.0));
        std::vector<std::vector<double>> probs(T, std::vector<double>(V, 0.0));

        double sentence_loss = 0.0;

        // ====================================================================
        // 1. FORWARD PASS ACROSS TIME (t = 0 ... T-1)
        // ====================================================================
        for (int t = 0; t < T; ++t) {
            int input_word = tokens[t];
            int target_word = tokens[t + 1];

            // A. Calculate h_pre[t] = x[t] * Wxh + h[t-1] * Whh + Bh
            for (int j = 0; j < H; ++j) {
                double sum = Bh[j];

                for (int d = 0; d < D; ++d) {
                    sum += C[input_word][d] * Wxh[d][j];
                }

                if (t > 0) {
                    for (int prev_j = 0; prev_j < H; ++prev_j) {
                        sum += h[t - 1][prev_j] * Whh[prev_j][j];
                    }
                }

                h_pre[t][j] = sum;
                // Use Smooth Dynamic Leaky activation!
                h[t][j] = smooth_dynamic_leaky(sum, alpha);
            }

            // B. Calculate logits[t] = h[t] * Why + By
            for (int k = 0; k < V; ++k) {
                double sum = By[k];
                for (int j = 0; j < H; ++j) {
                    sum += h[t][j] * Why[j][k];
                }
                logits[t][k] = sum;
            }

            // C. Softmax Probabilities
            probs[t] = softmax(logits[t]);

            // D. Measure Error: Cross-Entropy Loss
            double p_target = std::max(probs[t][target_word], 1e-12);
            sentence_loss += -std::log(p_target);

            int best_k = static_cast<int>(std::distance(probs[t].begin(), 
                         std::max_element(probs[t].begin(), probs[t].end())));
            if (best_k == target_word) correct_words++;
            total_words++;
        }

        // ====================================================================
        // 2. BACKPROPAGATION THROUGH TIME (BPTT: Flows backwards from T-1 to 0)
        // ====================================================================
        std::vector<std::vector<double>> dWxh(D, std::vector<double>(H, 0.0));
        std::vector<std::vector<double>> dWhh(H, std::vector<double>(H, 0.0));
        std::vector<double> dBh(H, 0.0);
        std::vector<std::vector<double>> dWhy(H, std::vector<double>(V, 0.0));
        std::vector<double> dBy(V, 0.0);
        double d_alpha = 0.0;

        std::vector<double> dh_next(H, 0.0);

        for (int t = T - 1; t >= 0; --t) {
            int input_word = tokens[t];
            int target_word = tokens[t + 1];

            // A. Output Judge error: dLogits = probs - target
            std::vector<double> dLogits = probs[t];
            dLogits[target_word] -= 1.0;

            for (int k = 0; k < V; ++k) {
                dBy[k] += dLogits[k];
                for (int j = 0; j < H; ++j) {
                    dWhy[j][k] += dLogits[k] * h[t][j];
                }
            }

            // B. Error arriving at hidden state h[t]
            std::vector<double> dh(H, 0.0);
            for (int j = 0; j < H; ++j) {
                for (int k = 0; k < V; ++k) {
                    dh[j] += dLogits[k] * Why[j][k];
                }
                dh[j] += dh_next[j];
            }

            // C. Push error backwards through Smooth Dynamic Leaky gate
            std::vector<double> dh_pre_t(H, 0.0);
            for (int j = 0; j < H; ++j) {
                dh_pre_t[j] = dh[j] * smooth_dynamic_leaky_derivative(h_pre[t][j], alpha);
                dBh[j] += dh_pre_t[j];
                d_alpha += dh[j] * smooth_dynamic_leaky_d_alpha(h_pre[t][j]);
            }

            // D. Gradients for Wxh and input word coordinates C
            for (int d = 0; d < D; ++d) {
                double d_coord = 0.0;
                for (int j = 0; j < H; ++j) {
                    dWxh[d][j] += dh_pre_t[j] * C[input_word][d];
                    d_coord += dh_pre_t[j] * Wxh[d][j];
                }
                adam.step(C[input_word][d], d_coord, mC[input_word][d], vC[input_word][d], lr);
            }

            // E. Gradients for Whh (Recurrent Memory Loop)
            dh_next.assign(H, 0.0);
            if (t > 0) {
                for (int prev_j = 0; prev_j < H; ++prev_j) {
                    for (int j = 0; j < H; ++j) {
                        dWhh[prev_j][j] += dh_pre_t[j] * h[t - 1][prev_j];
                        dh_next[prev_j] += dh_pre_t[j] * Whh[prev_j][j];
                    }
                }
            }
        }

        // ====================================================================
        // 3. APPLY DYNAMIC MOMENTUM (ADAM) TO ALL DIALS
        // ====================================================================
        for (int d = 0; d < D; ++d) {
            for (int j = 0; j < H; ++j) {
                adam.step(Wxh[d][j], dWxh[d][j], mWxh[d][j], vWxh[d][j], lr);
            }
        }

        for (int i = 0; i < H; ++i) {
            for (int j = 0; j < H; ++j) {
                adam.step(Whh[i][j], dWhh[i][j], mWhh[i][j], vWhh[i][j], lr);
            }
        }

        for (int j = 0; j < H; ++j) {
            adam.step(Bh[j], dBh[j], mBh[j], vBh[j], lr);
        }

        for (int j = 0; j < H; ++j) {
            for (int k = 0; k < V; ++k) {
                adam.step(Why[j][k], dWhy[j][k], mWhy[j][k], vWhy[j][k], lr);
            }
        }

        for (int k = 0; k < V; ++k) {
            adam.step(By[k], dBy[k], mBy[k], vBy[k], lr);
        }

        // Dynamically update the leak rate alpha using Adam!
        adam.step(alpha, d_alpha, m_alpha, v_alpha, lr * 0.05);
        if (alpha < 0.005) alpha = 0.005; // Keep healthy bounds
        if (alpha > 0.250) alpha = 0.250;

        return sentence_loss / T;
    }

    // ------------------------------------------------------------------------
    // PREDICT NEXT WORD GIVEN ANY PROMPT OF ANY LENGTH!
    // ------------------------------------------------------------------------
    std::vector<double> predict_next(const std::vector<int>& prompt_tokens) {
        if (prompt_tokens.empty()) return std::vector<double>(V, 1.0 / V);

        std::vector<double> h_curr(H, 0.0);

        for (int word_id : prompt_tokens) {
            std::vector<double> h_new(H, 0.0);
            for (int j = 0; j < H; ++j) {
                double sum = Bh[j];
                for (int d = 0; d < D; ++d) {
                    sum += C[word_id][d] * Wxh[d][j];
                }
                for (int prev_j = 0; prev_j < H; ++prev_j) {
                    sum += h_curr[prev_j] * Whh[prev_j][j];
                }
                h_new[j] = smooth_dynamic_leaky(sum, alpha);
            }
            h_curr = h_new;
        }

        std::vector<double> logits(V, 0.0);
        for (int k = 0; k < V; ++k) {
            double sum = By[k];
            for (int j = 0; j < H; ++j) {
                sum += h_curr[j] * Why[j][k];
            }
            logits[k] = sum;
        }

        return softmax(logits);
    }
};

// ====================================================================================
// STEP 4: MAIN EXPERIMENTATION & VERIFICATION
// ====================================================================================

int main() {
    std::cout << "\033[1;36m====================================================================================\n";
    std::cout << " PROGRAM 33: DYNAMIC-CONTEXT RECURRENT NEURAL NETWORK (RNN) & ADAM OPTIMIZER\n";
    std::cout << " Arbitrary-Length Prompt Memory + Smooth Dynamic Leaky Brain + Adam (C++17)\n";
    std::cout << "====================================================================================\033[0m\n\n";

    std::vector<std::string> sentences = {
        "the king sits on the golden throne .",
        "the queen sits on the golden throne .",
        "the king rules the royal castle .",
        "the queen rules the royal castle .",
        "the prince lives in the royal palace .",
        "the princess lives in the royal palace .",
        "the monkey eats a sweet apple .",
        "the monkey eats a sweet banana .",
        "the monkey eats a juicy orange .",
        "the monkey eats a juicy grape .",
        "apple is a sweet delicious fruit .",
        "banana is a sweet delicious fruit .",
        "orange is a juicy delicious fruit .",
        "grape is a juicy delicious fruit .",
        "the wild wolf hunts in the deep forest .",
        "the wild lion hunts in the deep forest .",
        "the friendly dog runs in the green park .",
        "the friendly cat runs in the green park .",
        "the little girl was happy in the castle .",
        "the brave knight rode through the deep forest ."
    };

    std::vector<std::string> vocab;
    std::unordered_map<std::string, int> word_to_id;
    std::unordered_map<int, std::string> id_to_word;

    for (const auto& s : sentences) {
        std::stringstream ss(s);
        std::string w;
        while (ss >> w) {
            if (word_to_id.find(w) == word_to_id.end()) {
                int id = static_cast<int>(vocab.size());
                word_to_id[w] = id;
                id_to_word[id] = w;
                vocab.push_back(w);
            }
        }
    }

    int V = static_cast<int>(vocab.size());
    std::cout << " [*] Loaded " << sentences.size() << " sentences. Vocabulary: " << V << " unique words.\n\n";

    std::vector<std::vector<int>> tokenized_corpus;
    for (const auto& s : sentences) {
        std::stringstream ss(s);
        std::string w;
        std::vector<int> tokens;
        while (ss >> w) tokens.push_back(word_to_id[w]);
        tokenized_corpus.push_back(tokens);
    }

    int D = 12;
    int H = 32;
    RecurrentLanguageModel rnn(V, D, H);

    std::cout << "------------------------------------------------------------------------------------\n";
    std::cout << " PART 1: THE DYNAMIC MEMORY LOOP ARCHITECTURE\n";
    std::cout << "------------------------------------------------------------------------------------\n";
    std::cout << " 1. Word Coordinates:      " << D << " dials per word (" << (V * D) << " parameters)\n";
    std::cout << " 2. Input-to-Hidden Wxh:   " << D << "x" << H << " (" << (D * H) << " parameters)\n";
    std::cout << " 3. Recurrent Loop Whh:    " << H << "x" << H << " (" << (H * H) << " parameters - MEMORY LOOP)\n";
    std::cout << " 4. Hidden Bias Bh:        " << H << " parameters\n";
    std::cout << " 5. Output Judges Why:     " << H << "x" << V << " (" << (H * V) << " parameters)\n";
    std::cout << " 6. Output Bias By:        " << V << " parameters\n";
    std::cout << " 7. Activation:            Smooth Dynamic Leaky (Trainable alpha starting at " 
              << std::fixed << std::setprecision(3) << rnn.alpha << ")\n";
    int total_params = (V * D) + (D * H) + (H * H) + H + (H * V) + V + 1;
    std::cout << " >>> Total Trainable Dials: " << total_params << " parameters\n";
    std::cout << " >>> Optimizer:            Dynamic Momentum (Adam: beta1=0.9, beta2=0.999)\n\n";

    std::cout << "------------------------------------------------------------------------------------\n";
    std::cout << " PART 2: TRAINING WITH DYNAMIC MOMENTUM & SMOOTH DYNAMIC LEAKY BRAIN\n";
    std::cout << "------------------------------------------------------------------------------------\n";

    int epochs = 350;
    double learning_rate = 0.008;

    auto t_start = std::chrono::high_resolution_clock::now();

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        double total_loss = 0.0;
        int correct_words = 0;
        int total_words = 0;

        for (const auto& tokens : tokenized_corpus) {
            total_loss += rnn.train_sentence(tokens, learning_rate, correct_words, total_words);
        }

        if (epoch == 1 || epoch % 50 == 0 || epoch == epochs) {
            double avg_loss = total_loss / tokenized_corpus.size();
            double acc = (100.0 * correct_words) / total_words;
            std::cout << "   Epoch " << std::setw(3) << epoch << " / " << epochs
                      << " | Loss: \033[1;33m" << std::fixed << std::setprecision(4) << avg_loss << "\033[0m"
                      << " | Next-Word Accuracy: \033[1;32m" << std::setprecision(1) << acc << "%\033[0m"
                      << " | Dynamic Leak (alpha): \033[36m" << std::setprecision(4) << rnn.alpha << "\033[0m\n";
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    std::cout << "\n[+] Training completed in " << std::fixed << std::setprecision(1) << elapsed_ms << " ms.\n\n";

    std::cout << "====================================================================================\n";
    std::cout << " PART 3: PROVING DYNAMIC CONTEXT (PROMPTS OF ARBITRARY LENGTH)\n";
    std::cout << "====================================================================================\n";

    auto test_prompt = [&](const std::vector<std::string>& words) {
        std::cout << " Prompt (length " << words.size() << "): \"";
        for (size_t i = 0; i < words.size(); ++i) {
            std::cout << words[i] << (i + 1 < words.size() ? " " : "");
        }
        std::cout << "\" ---> What comes next?\n";

        std::vector<int> p_tokens;
        for (const auto& w : words) {
            if (word_to_id.count(w)) p_tokens.push_back(word_to_id[w]);
        }

        auto probs = rnn.predict_next(p_tokens);

        std::vector<std::pair<double, int>> ranked;
        for (int i = 0; i < V; ++i) ranked.push_back({probs[i], i});
        std::sort(ranked.rbegin(), ranked.rend());

        for (int rank = 0; rank < 3; ++rank) {
            std::cout << "    #" << (rank + 1) << " " << std::setw(12) << std::left 
                      << ("\"" + id_to_word[ranked[rank].second] + "\"")
                      << " Probability: " << std::fixed << std::setprecision(1) 
                      << (ranked[rank].first * 100.0) << "%\n";
        }
        std::cout << "\n";
    };

    test_prompt({"apple"});
    test_prompt({"the", "king"});
    test_prompt({"the", "wild", "wolf"});
    test_prompt({"the", "brave", "knight", "rode"});
    test_prompt({"the", "friendly", "dog", "runs", "in", "the"});

    std::cout << "====================================================================================\n";
    std::cout << " PART 4: AUTONOMOUS STORY GENERATION WITH DEEP MEMORY\n";
    std::cout << "====================================================================================\n";

    auto generate_story = [&](const std::vector<std::string>& seed) {
        std::vector<int> tokens;
        for (const auto& w : seed) tokens.push_back(word_to_id[w]);

        std::cout << " Starting Prompt: \"";
        for (size_t i = 0; i < seed.size(); ++i) std::cout << seed[i] << (i + 1 < seed.size() ? " " : "");
        std::cout << "\"\n Full Generated Story: \033[1;32m";
        for (const auto& w : seed) std::cout << w << " ";

        for (int step = 0; step < 15; ++step) {
            auto probs = rnn.predict_next(tokens);
            int next_id = static_cast<int>(std::distance(probs.begin(), 
                          std::max_element(probs.begin(), probs.end())));

            std::string next_word = id_to_word[next_id];
            std::cout << next_word << " ";
            tokens.push_back(next_id);

            if (next_word == ".") break;
        }
        std::cout << "\033[0m\n\n";
    };

    generate_story({"the", "queen"});
    generate_story({"the", "monkey", "eats"});
    generate_story({"the", "brave", "knight"});
    generate_story({"banana"});

    return 0;
}
