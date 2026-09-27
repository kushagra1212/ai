/**
 * ====================================================================================
 * PROGRAM 34: GATED MEMORY (LSTM) & FIRST-PRINCIPLES ADAM OPTIMIZER
 * ====================================================================================
 * Kushagra's AI Journey: Conquering the Vanishing Gradient & Long Memory
 *
 * Core Breakthroughs Over Program 33:
 * 1. THE CONVEYOR BELT HIGHWAY (Cell State C_t):
 *    - In simple RNNs, memory was repeatedly MULTIPLIED by Whh every step,
 *      causing signals to decay by 0.85^20 = 0.038 (vanishing to zero).
 *    - In LSTM, long-term memory travels along a dedicated conveyor belt using
 *      ADDITION (+). The derivative of addition is 1.0, preserving memory across
 *      long sentences!
 * 2. THREE INTELLIGENT SECURITY DOORS (GATES):
 *    - Door 1: Forget Gate (f_t)  -> Decides what old obsolete facts to discard.
 *    - Door 2: Input Gate (i_t)   -> Decides what new facts to write onto the belt.
 *    - Door 3: Output Gate (o_t)  -> Decides what to reveal for next-word prediction.
 * 3. FIRST-PRINCIPLES ADAM OPTIMIZER:
 *    - Every gate dial independently adapts its learning speed and variance.
 * 4. BACKPROPAGATION THROUGH TIME (BPTT) FOR LSTM:
 *    - Exact chain-rule gradient flow through all 4 gate channels.
 *
 * 100% Modern First-Principles C++17 — Zero External ML Libraries!
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
// STEP 1: MATHEMATICAL PRIMITIVES (SIGMOID, TANH, SOFTMAX)
// ====================================================================================

inline double sigmoid(double z) {
    if (z > 20.0) return 1.0;
    if (z < -20.0) return 0.0;
    return 1.0 / (1.0 + std::exp(-z));
}

inline double dtanh(double tanh_val) {
    // Derivative of tanh(x) with respect to x is (1 - tanh^2(x))
    return 1.0 - tanh_val * tanh_val;
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
// STEP 2: FIRST-PRINCIPLES ADAM OPTIMIZER
// ====================================================================================

struct AdamOptimizer {
    double beta1 = 0.9;
    double beta2 = 0.999;
    double eps = 1e-8;
    int t = 0;

    inline void step(double& weight, double grad, double& m, double& v, double lr) {
        // 1. Running average of speed (1st moment)
        m = beta1 * m + (1.0 - beta1) * grad;

        // 2. Running average of terrain bumpiness (2nd moment)
        v = beta2 * v + (1.0 - beta2) * (grad * grad);

        // 3. Bias corrections (fixes early-step zero bias)
        double m_hat = m / (1.0 - std::pow(beta1, t));
        double v_hat = v / (1.0 - std::pow(beta2, t));

        // 4. Adaptive dial nudge
        weight -= lr * (m_hat / (std::sqrt(v_hat) + eps));
    }
};

// ====================================================================================
// STEP 3: LONG SHORT-TERM MEMORY (LSTM) ARCHITECTURE
// ====================================================================================

class LSTMLanguageModel {
public:
    int V; // Vocabulary size
    int D; // Word coordinate dimension (e.g. 12 dials per word)
    int H; // Hidden & Cell memory dimension (e.g. 32 memory cells)

    // 1. Word Coordinate Table: C[V][D]
    std::vector<std::vector<double>> C;
    std::vector<std::vector<double>> mC, vC;

    // 2. Forget Gate Weights & Biases: f_t = sigmoid(Wxf * x + Whf * h + Bf)
    std::vector<std::vector<double>> Wxf, Whf;
    std::vector<double> Bf;
    std::vector<std::vector<double>> mWxf, vWxf, mWhf, vWhf;
    std::vector<double> mBf, vBf;

    // 3. Input Gate Weights & Biases: i_t = sigmoid(Wxi * x + Whi * h + Bi)
    std::vector<std::vector<double>> Wxi, Whi;
    std::vector<double> Bi;
    std::vector<std::vector<double>> mWxi, vWxi, mWhi, vWhi;
    std::vector<double> mBi, vBi;

    // 4. Candidate Cell Weights & Biases: c_cand = tanh(Wxc * x + Whc * h + Bc)
    std::vector<std::vector<double>> Wxc, Whc;
    std::vector<double> Bc;
    std::vector<std::vector<double>> mWxc, vWxc, mWhc, vWhc;
    std::vector<double> mBc, vBc;

    // 5. Output Gate Weights & Biases: o_t = sigmoid(Wxo * x + Who * h + Bo)
    std::vector<std::vector<double>> Wxo, Who;
    std::vector<double> Bo;
    std::vector<std::vector<double>> mWxo, vWxo, mWho, vWho;
    std::vector<double> mBo, vBo;

    // 6. Vocabulary Output Judges: Why[H][V], By[V]
    std::vector<std::vector<double>> Why;
    std::vector<double> By;
    std::vector<std::vector<double>> mWhy, vWhy;
    std::vector<double> mBy, vBy;

    AdamOptimizer adam;

    LSTMLanguageModel(int vocab_size, int coord_dim = 12, int hidden_dim = 32)
        : V(vocab_size), D(coord_dim), H(hidden_dim) {

        std::mt19937 rng(42);
        double scale_x = std::sqrt(2.0 / D);
        double scale_h = std::sqrt(2.0 / H);
        double scale_y = std::sqrt(2.0 / H);
        double scale_emb = 0.1;

        std::normal_distribution<double> dist_x(0.0, scale_x);
        std::normal_distribution<double> dist_h(0.0, scale_h);
        std::normal_distribution<double> dist_y(0.0, scale_y);
        std::normal_distribution<double> dist_emb(0.0, scale_emb);

        // Helper to allocate 2D matrices
        auto alloc_2d = [](int r, int c, double val = 0.0) {
            return std::vector<std::vector<double>>(r, std::vector<double>(c, val));
        };

        // Word coordinates
        C = alloc_2d(V, D); mC = alloc_2d(V, D); vC = alloc_2d(V, D);
        for (int i = 0; i < V; ++i) {
            for (int d = 0; d < D; ++d) C[i][d] = dist_emb(rng);
        }

        // Initialize 4 Gates (Forget, Input, Candidate, Output)
        auto init_gate = [&](std::vector<std::vector<double>>& Wx, std::vector<std::vector<double>>& mWx, std::vector<std::vector<double>>& vWx,
                             std::vector<std::vector<double>>& Wh, std::vector<std::vector<double>>& mWh, std::vector<std::vector<double>>& vWh,
                             std::vector<double>& B, std::vector<double>& mB, std::vector<double>& vB, double b_init = 0.0) {
            Wx = alloc_2d(D, H); mWx = alloc_2d(D, H); vWx = alloc_2d(D, H);
            Wh = alloc_2d(H, H); mWh = alloc_2d(H, H); vWh = alloc_2d(H, H);
            B.assign(H, b_init); mB.assign(H, 0.0); vB.assign(H, 0.0);

            for (int d = 0; d < D; ++d) {
                for (int j = 0; j < H; ++j) Wx[d][j] = dist_x(rng);
            }
            for (int i = 0; i < H; ++i) {
                for (int j = 0; j < H; ++j) Wh[i][j] = dist_h(rng);
            }
        };

        // Note: Initialize Forget Gate Bias to +1.0!
        // This ensures the brain defaults to remembering everything at the start.
        init_gate(Wxf, mWxf, vWxf, Whf, mWhf, vWhf, Bf, mBf, vBf, 1.0); // Bf = 1.0
        init_gate(Wxi, mWxi, vWxi, Whi, mWhi, vWhi, Bi, mBi, vBi, 0.0);
        init_gate(Wxc, mWxc, vWxc, Whc, mWhc, vWhc, Bc, mBc, vBc, 0.0);
        init_gate(Wxo, mWxo, vWxo, Who, mWho, vWho, Bo, mBo, vBo, 0.0);

        // Output Judges
        Why = alloc_2d(H, V); mWhy = alloc_2d(H, V); vWhy = alloc_2d(H, V);
        for (int i = 0; i < H; ++i) {
            for (int k = 0; k < V; ++k) Why[i][k] = dist_y(rng);
        }
        By.assign(V, 0.0); mBy.assign(V, 0.0); vBy.assign(V, 0.0);
    }

    // ------------------------------------------------------------------------
    // FORWARD & BACKWARD PASS: BACKPROPAGATION THROUGH TIME FOR LSTM
    // ------------------------------------------------------------------------
    double train_sentence(const std::vector<int>& tokens, double lr, int& correct_words, int& total_words) {
        int T = static_cast<int>(tokens.size()) - 1;
        if (T <= 0) return 0.0;

        adam.t++;

        // Store gate activations across all time steps for backprop
        std::vector<std::vector<double>> f_gate(T, std::vector<double>(H, 0.0));
        std::vector<std::vector<double>> i_gate(T, std::vector<double>(H, 0.0));
        std::vector<std::vector<double>> c_cand(T, std::vector<double>(H, 0.0));
        std::vector<std::vector<double>> o_gate(T, std::vector<double>(H, 0.0));
        std::vector<std::vector<double>> C_cell(T, std::vector<double>(H, 0.0));
        std::vector<std::vector<double>> tanh_C(T, std::vector<double>(H, 0.0));
        std::vector<std::vector<double>> h_state(T, std::vector<double>(H, 0.0));
        std::vector<std::vector<double>> logits(T, std::vector<double>(V, 0.0));
        std::vector<std::vector<double>> probs(T, std::vector<double>(V, 0.0));

        double sentence_loss = 0.0;

        // ====================================================================
        // 1. FORWARD PASS ACROSS TIME STEPS (t = 0 ... T-1)
        // ====================================================================
        for (int t = 0; t < T; ++t) {
            int in_word = tokens[t];
            int targ_word = tokens[t + 1];

            for (int j = 0; j < H; ++j) {
                double f_sum = Bf[j];
                double i_sum = Bi[j];
                double c_sum = Bc[j];
                double o_sum = Bo[j];

                // Add input word projection
                for (int d = 0; d < D; ++d) {
                    double x_d = C[in_word][d];
                    f_sum += x_d * Wxf[d][j];
                    i_sum += x_d * Wxi[d][j];
                    c_sum += x_d * Wxc[d][j];
                    o_sum += x_d * Wxo[d][j];
                }

                // Add previous hidden thought (if t > 0)
                if (t > 0) {
                    for (int pj = 0; pj < H; ++pj) {
                        double h_prev = h_state[t - 1][pj];
                        f_sum += h_prev * Whf[pj][j];
                        i_sum += h_prev * Whi[pj][j];
                        c_sum += h_prev * Whc[pj][j];
                        o_sum += h_prev * Who[pj][j];
                    }
                }

                // Compute the 3 Gates + 1 Candidate Note
                f_gate[t][j] = sigmoid(f_sum);
                i_gate[t][j] = sigmoid(i_sum);
                c_cand[t][j] = std::tanh(c_sum);
                o_gate[t][j] = sigmoid(o_sum);

                // ============================================================
                // 👉 THE CONVEYOR BELT HIGHWAY (ADDITION IN ACTION!)
                // C_t = f_t * C_{t-1} + i_t * c_cand
                // ============================================================
                double prev_cell = (t > 0) ? C_cell[t - 1][j] : 0.0;
                C_cell[t][j] = f_gate[t][j] * prev_cell + i_gate[t][j] * c_cand[t][j];

                // Output Thought h_t = o_t * tanh(C_t)
                tanh_C[t][j] = std::tanh(C_cell[t][j]);
                h_state[t][j] = o_gate[t][j] * tanh_C[t][j];
            }

            // Output Judges
            for (int k = 0; k < V; ++k) {
                double sum = By[k];
                for (int j = 0; j < H; ++j) {
                    sum += h_state[t][j] * Why[j][k];
                }
                logits[t][k] = sum;
            }

            probs[t] = softmax(logits[t]);

            // Loss
            double p_targ = std::max(probs[t][targ_word], 1e-12);
            sentence_loss += -std::log(p_targ);

            int pred_k = static_cast<int>(std::distance(probs[t].begin(),
                         std::max_element(probs[t].begin(), probs[t].end())));
            if (pred_k == targ_word) correct_words++;
            total_words++;
        }

        // ====================================================================
        // 2. BACKPROPAGATION THROUGH TIME (BPTT FOR LSTM)
        // ====================================================================
        // Gate weight gradients
        auto alloc_2d = [](int r, int c) { return std::vector<std::vector<double>>(r, std::vector<double>(c, 0.0)); };
        auto dWxf = alloc_2d(D, H), dWhf = alloc_2d(H, H); std::vector<double> dBf(H, 0.0);
        auto dWxi = alloc_2d(D, H), dWhi = alloc_2d(H, H); std::vector<double> dBi(H, 0.0);
        auto dWxc = alloc_2d(D, H), dWhc = alloc_2d(H, H); std::vector<double> dBc(H, 0.0);
        auto dWxo = alloc_2d(D, H), dWho = alloc_2d(H, H); std::vector<double> dBo(H, 0.0);

        auto dWhy = alloc_2d(H, V); std::vector<double> dBy(V, 0.0);

        std::vector<double> dh_next(H, 0.0);
        std::vector<double> dC_next(H, 0.0);

        for (int t = T - 1; t >= 0; --t) {
            int in_word = tokens[t];
            int targ_word = tokens[t + 1];

            // 1. Error at Output Judges: dLogits = probs - target
            std::vector<double> dLogits = probs[t];
            dLogits[targ_word] -= 1.0;

            for (int k = 0; k < V; ++k) {
                dBy[k] += dLogits[k];
                for (int j = 0; j < H; ++j) {
                    dWhy[j][k] += dLogits[k] * h_state[t][j];
                }
            }

            // 2. Error arriving at hidden state h[t]
            std::vector<double> dh(H, 0.0);
            for (int j = 0; j < H; ++j) {
                for (int k = 0; k < V; ++k) {
                    dh[j] += dLogits[k] * Why[j][k];
                }
                dh[j] += dh_next[j];
            }

            // 3. Error arriving at Cell State Conveyor Belt C[t]
            // h_t = o_t * tanh(C_t)
            std::vector<double> dC(H, 0.0);
            std::vector<double> do_pre(H, 0.0);
            std::vector<double> df_pre(H, 0.0);
            std::vector<double> di_pre(H, 0.0);
            std::vector<double> dc_pre(H, 0.0);

            for (int j = 0; j < H; ++j) {
                double o = o_gate[t][j];
                double tc = tanh_C[t][j];

                // Derivative through output gate: do = dh * tanh(C)
                double do_val = dh[j] * tc;
                do_pre[j] = do_val * o * (1.0 - o); // Sigmoid slope

                // Derivative through cell state: dC = dh * o * dtanh(C) + dC_next
                dC[j] = dh[j] * o * dtanh(tc) + dC_next[j];

                // Derivatives through forget gate, input gate, and candidate note:
                double prev_c = (t > 0) ? C_cell[t - 1][j] : 0.0;
                double df_val = dC[j] * prev_c;
                double di_val = dC[j] * c_cand[t][j];
                double dc_val = dC[j] * i_gate[t][j];

                double f = f_gate[t][j];
                double i = i_gate[t][j];
                double c = c_cand[t][j];

                df_pre[j] = df_val * f * (1.0 - f);     // Sigmoid slope
                di_pre[j] = di_val * i * (1.0 - i);     // Sigmoid slope
                dc_pre[j] = dc_val * (1.0 - c * c);     // Tanh slope

                // Accumulate biases
                dBf[j] += df_pre[j];
                dBi[j] += di_pre[j];
                dBc[j] += dc_pre[j];
                dBo[j] += do_pre[j];
            }

            // 4. Accumulate weights for input word coordinates C[in_word]
            for (int d = 0; d < D; ++d) {
                double x_d = C[in_word][d];
                double d_coord = 0.0;

                for (int j = 0; j < H; ++j) {
                    dWxf[d][j] += df_pre[j] * x_d;
                    dWxi[d][j] += di_pre[j] * x_d;
                    dWxc[d][j] += dc_pre[j] * x_d;
                    dWxo[d][j] += do_pre[j] * x_d;

                    d_coord += df_pre[j] * Wxf[d][j] + di_pre[j] * Wxi[d][j]
                             + dc_pre[j] * Wxc[d][j] + do_pre[j] * Wxo[d][j];
                }
                adam.step(C[in_word][d], d_coord, mC[in_word][d], vC[in_word][d], lr);
            }

            // 5. Flow blame backward across time: dh_next and dC_next
            dh_next.assign(H, 0.0);
            dC_next.assign(H, 0.0);

            for (int j = 0; j < H; ++j) {
                // ============================================================
                // 👉 THE CONVEYOR BELT GRADIENT HIGHWAY!
                // dC_{t-1} = dC_t * f_t (Multiplied by f_t ≈ 1.0, NO DECAY!)
                // ============================================================
                dC_next[j] = dC[j] * f_gate[t][j];
            }

            if (t > 0) {
                for (int pj = 0; pj < H; ++pj) {
                    double prev_h = h_state[t - 1][pj];
                    for (int j = 0; j < H; ++j) {
                        dWhf[pj][j] += df_pre[j] * prev_h;
                        dWhi[pj][j] += di_pre[j] * prev_h;
                        dWhc[pj][j] += dc_pre[j] * prev_h;
                        dWho[pj][j] += do_pre[j] * prev_h;

                        dh_next[pj] += df_pre[j] * Whf[pj][j] + di_pre[j] * Whi[pj][j]
                                     + dc_pre[j] * Whc[pj][j] + do_pre[j] * Who[pj][j];
                    }
                }
            }
        }

        // ====================================================================
        // 3. APPLY ADAM OPTIMIZER TO ALL GATE DIALS
        // ====================================================================
        auto update_gate_adam = [&](std::vector<std::vector<double>>& Wx, const std::vector<std::vector<double>>& dWx,
                                    std::vector<std::vector<double>>& mWx, std::vector<std::vector<double>>& vWx,
                                    std::vector<std::vector<double>>& Wh, const std::vector<std::vector<double>>& dWh,
                                    std::vector<std::vector<double>>& mWh, std::vector<std::vector<double>>& vWh,
                                    std::vector<double>& B, const std::vector<double>& dB,
                                    std::vector<double>& mB, std::vector<double>& vB) {
            for (int d = 0; d < D; ++d) {
                for (int j = 0; j < H; ++j) adam.step(Wx[d][j], dWx[d][j], mWx[d][j], vWx[d][j], lr);
            }
            for (int i = 0; i < H; ++i) {
                for (int j = 0; j < H; ++j) adam.step(Wh[i][j], dWh[i][j], mWh[i][j], vWh[i][j], lr);
            }
            for (int j = 0; j < H; ++j) adam.step(B[j], dB[j], mB[j], vB[j], lr);
        };

        update_gate_adam(Wxf, dWxf, mWxf, vWxf, Whf, dWhf, mWhf, vWhf, Bf, dBf, mBf, vBf);
        update_gate_adam(Wxi, dWxi, mWxi, vWxi, Whi, dWhi, mWhi, vWhi, Bi, dBi, mBi, vBi);
        update_gate_adam(Wxc, dWxc, mWxc, vWxc, Whc, dWhc, mWhc, vWhc, Bc, dBc, mBc, vBc);
        update_gate_adam(Wxo, dWxo, mWxo, vWxo, Who, dWho, mWho, vWho, Bo, dBo, mBo, vBo);

        for (int j = 0; j < H; ++j) {
            for (int k = 0; k < V; ++k) adam.step(Why[j][k], dWhy[j][k], mWhy[j][k], vWhy[j][k], lr);
        }
        for (int k = 0; k < V; ++k) adam.step(By[k], dBy[k], mBy[k], vBy[k], lr);

        return sentence_loss / T;
    }

    // ------------------------------------------------------------------------
    // PREDICT NEXT WORD FOR ANY PROMPT
    // ------------------------------------------------------------------------
    std::vector<double> predict_next(const std::vector<int>& prompt_tokens) {
        if (prompt_tokens.empty()) return std::vector<double>(V, 1.0 / V);

        std::vector<double> h_curr(H, 0.0);
        std::vector<double> C_curr(H, 0.0);

        for (int wid : prompt_tokens) {
            std::vector<double> h_new(H, 0.0);
            std::vector<double> C_new(H, 0.0);

            for (int j = 0; j < H; ++j) {
                double f_sum = Bf[j];
                double i_sum = Bi[j];
                double c_sum = Bc[j];
                double o_sum = Bo[j];

                for (int d = 0; d < D; ++d) {
                    double x_d = C[wid][d];
                    f_sum += x_d * Wxf[d][j];
                    i_sum += x_d * Wxi[d][j];
                    c_sum += x_d * Wxc[d][j];
                    o_sum += x_d * Wxo[d][j];
                }

                for (int pj = 0; pj < H; ++pj) {
                    double ph = h_curr[pj];
                    f_sum += ph * Whf[pj][j];
                    i_sum += ph * Whi[pj][j];
                    c_sum += ph * Whc[pj][j];
                    o_sum += ph * Who[pj][j];
                }

                double f = sigmoid(f_sum);
                double i = sigmoid(i_sum);
                double c = std::tanh(c_sum);
                double o = sigmoid(o_sum);

                C_new[j] = f * C_curr[j] + i * c;
                h_new[j] = o * std::tanh(C_new[j]);
            }

            h_curr = h_new;
            C_curr = C_new;
        }

        std::vector<double> logits(V, 0.0);
        for (int k = 0; k < V; ++k) {
            double sum = By[k];
            for (int j = 0; j < H; ++j) sum += h_curr[j] * Why[j][k];
            logits[k] = sum;
        }

        return softmax(logits);
    }
};

// ====================================================================================
// STEP 4: MAIN EXPERIMENTATION & DEEP MEMORY VERIFICATION
// ====================================================================================

int main() {
    std::cout << "\033[1;36m====================================================================================\n";
    std::cout << " PROGRAM 34: GATED MEMORY (LSTM) & FIRST-PRINCIPLES ADAM OPTIMIZER\n";
    std::cout << " The Highway Conveyor Belt (C_t) + 3 Security Doors (f, i, o) (C++17)\n";
    std::cout << "====================================================================================\033[0m\n\n";

    // Long-distance sentences with intervening relative clauses!
    std::vector<std::string> sentences = {
        "the king who lived in the royal palace sits on the golden throne .",
        "the queen who lived in the royal palace sits on the golden throne .",
        "the king who ruled the ancient kingdom lives in the royal castle .",
        "the queen who ruled the ancient kingdom lives in the royal castle .",
        "the monkey who climbed the tall green tree eats a sweet apple .",
        "the monkey who climbed the tall green tree eats a sweet banana .",
        "the wild wolf that hunted across the deep forest runs in the night .",
        "the wild lion that hunted across the deep forest runs in the night .",
        "the friendly dog that played with the happy child runs in the park .",
        "the friendly cat that played with the happy child runs in the park .",
        "apple is a sweet delicious juicy fruit .",
        "banana is a sweet delicious juicy fruit .",
        "orange is a sweet delicious juicy fruit .",
        "the brave knight who rode through the deep dark forest was fearless .",
        "the little girl who was happy in the castle smiled ."
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
    std::cout << " [*] Loaded " << sentences.size() << " long-distance sentences. Vocabulary: " << V << " unique words.\n\n";

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
    LSTMLanguageModel lstm(V, D, H);

    std::cout << "------------------------------------------------------------------------------------\n";
    std::cout << " PART 1: THE GATED CONVEYOR BELT ARCHITECTURE\n";
    std::cout << "------------------------------------------------------------------------------------\n";
    std::cout << " 1. Word Coordinates:      " << D << " dials per word (" << (V * D) << " parameters)\n";
    std::cout << " 2. Forget Gate (Wxf, Whf, Bf):   " << (D * H + H * H + H) << " dials (Controls what to discard)\n";
    std::cout << " 3. Input Gate (Wxi, Whi, Bi):    " << (D * H + H * H + H) << " dials (Controls what to write)\n";
    std::cout << " 4. Candidate Note (Wxc, Whc, Bc):" << (D * H + H * H + H) << " dials (New candidate content)\n";
    std::cout << " 5. Output Gate (Wxo, Who, Bo):   " << (D * H + H * H + H) << " dials (Controls what to speak)\n";
    std::cout << " 6. Output Judges (Why, By):       " << (H * V + V) << " dials\n";
    int total_params = (V * D) + 4 * (D * H + H * H + H) + (H * V + V);
    std::cout << " >>> Total Trainable Dials: " << total_params << " parameters\n";
    std::cout << " >>> Conveyor Belt State:   C_t (Linear additive highway with dC/dC = 1.0!)\n";
    std::cout << " >>> Optimizer:            Dynamic Momentum (Adam)\n\n";

    std::cout << "------------------------------------------------------------------------------------\n";
    std::cout << " PART 2: TRAINING LSTM WITH LONG-DISTANCE DEPENDENCIES\n";
    std::cout << "------------------------------------------------------------------------------------\n";

    int epochs = 350;
    double learning_rate = 0.012;

    auto t_start = std::chrono::high_resolution_clock::now();

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        double total_loss = 0.0;
        int correct_words = 0;
        int total_words = 0;

        for (const auto& tokens : tokenized_corpus) {
            total_loss += lstm.train_sentence(tokens, learning_rate, correct_words, total_words);
        }

        if (epoch == 1 || epoch % 50 == 0 || epoch == epochs) {
            double avg_loss = total_loss / tokenized_corpus.size();
            double acc = (100.0 * correct_words) / total_words;
            std::cout << "   Epoch " << std::setw(3) << epoch << " / " << epochs
                      << " | Loss: \033[1;33m" << std::fixed << std::setprecision(4) << avg_loss << "\033[0m"
                      << " | Next-Word Accuracy: \033[1;32m" << std::setprecision(1) << acc << "%\033[0m\n";
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    std::cout << "\n[+] LSTM Training completed in " << std::fixed << std::setprecision(1) << elapsed_ms << " ms.\n\n";

    std::cout << "====================================================================================\n";
    std::cout << " PART 3: PROVING LONG-DISTANCE MEMORY OVER 8+ WORDS\n";
    std::cout << "====================================================================================\n";

    auto test_prompt = [&](const std::vector<std::string>& words) {
        std::cout << " Prompt (" << words.size() << " words): \"";
        for (size_t i = 0; i < words.size(); ++i) {
            std::cout << words[i] << (i + 1 < words.size() ? " " : "");
        }
        std::cout << "\" ---> What comes next?\n";

        std::vector<int> p_tokens;
        for (const auto& w : words) {
            if (word_to_id.count(w)) p_tokens.push_back(word_to_id[w]);
        }

        auto probs = lstm.predict_next(p_tokens);

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

    // Test long-range sentences (connecting subject across 6+ words of distraction!)
    test_prompt({"the", "king", "who", "lived", "in", "the", "royal", "palace"});
    test_prompt({"the", "monkey", "who", "climbed", "the", "tall", "green", "tree"});
    test_prompt({"the", "wild", "wolf", "that", "hunted", "across", "the", "deep", "forest"});
    test_prompt({"the", "friendly", "dog", "that", "played", "with", "the", "happy", "child"});

    std::cout << "====================================================================================\n";
    std::cout << " PART 4: AUTONOMOUS GENERATION WITH DEEP CONVEYOR BELT MEMORY\n";
    std::cout << "====================================================================================\n";

    auto generate_story = [&](const std::vector<std::string>& seed) {
        std::vector<int> tokens;
        for (const auto& w : seed) tokens.push_back(word_to_id[w]);

        std::cout << " Starting Prompt: \"";
        for (size_t i = 0; i < seed.size(); ++i) std::cout << seed[i] << (i + 1 < seed.size() ? " " : "");
        std::cout << "\"\n Full Generated Story: \033[1;32m";
        for (const auto& w : seed) std::cout << w << " ";

        for (int step = 0; step < 18; ++step) {
            auto probs = lstm.predict_next(tokens);
            int next_id = static_cast<int>(std::distance(probs.begin(), 
                          std::max_element(probs.begin(), probs.end())));

            std::string next_word = id_to_word[next_id];
            std::cout << next_word << " ";
            tokens.push_back(next_id);

            if (next_word == ".") break;
        }
        std::cout << "\033[0m\n\n";
    };

    generate_story({"the", "king", "who", "lived"});
    generate_story({"the", "monkey", "who", "climbed"});
    generate_story({"the", "wild", "lion"});

    return 0;
}
