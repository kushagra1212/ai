/**
 * ====================================================================================
 * PROGRAM 36: DEEP 4-LAYER TRANSFORMER ON CONVERSATIONAL CHAT DATA (C++17)
 * ====================================================================================
 * Solving the "Chat Problem" and Deep Reasoning from First Principles!
 *
 * Architecture (Exact 4-Layer Stack Requested):
 *   Input:  Tokens x -> Embedding E[x] + Positional P[t] -> H0
 *   Layer 1: Self-Attention (Q1, K1, V1) + Residual Connection -> H1 = H0 + Attn1(H0)
 *   Layer 2: Self-Attention (Q2, K2, V2) + Residual Connection -> H2 = H1 + Attn2(H1)
 *   Layer 3: Feed-Forward Network (V -> V -> V Transformation) + Residual -> H3 = H2 + FFN(H2)
 *   Layer 4: Output Projection Head (H3 -> Vocabulary Logits Z) -> Softmax
 *
 * Training Dataset:
 *   - Conversational dialogue pairs: "User: <prompt>\nAssistant: <response>"
 *   - Everyday social greetings ("Hi", "Hello", "How are you?") + Knowledge Q&A
 *
 * 100% First-Principles C++17 with OpenMP Multi-Threading — Zero External ML Libraries!
 * ====================================================================================
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <unordered_map>
#include <cmath>
#include <random>
#include <iomanip>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <unistd.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ====================================================================================
// STEP 1: TOKENIZER WITH CHAT FORMATTING
// ====================================================================================

struct ChatTokenizer {
    int V; // Vocab size
    std::unordered_map<std::string, int> word_to_id;
    std::vector<std::string> id_to_word;

    const int PAD_ID = 0;
    const int UNK_ID = 1;
    const int BOS_ID = 2;
    const int EOS_ID = 3;

    void build_vocab(const std::string& filepath, int target_vocab_size = 4096, size_t max_bytes = 10000000) {
        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "[-] Error opening dataset: " << filepath << "\n";
            exit(1);
        }

        std::cout << "[*] Scanning chat dialogue dataset: " << filepath << "...\n";
        std::unordered_map<std::string, int> freq;
        std::string word;
        char c;
        size_t bytes_read = 0;

        auto add_word = [&](std::string& w) {
            if (!w.empty()) { freq[w]++; w.clear(); }
        };

        while (file.get(c) && bytes_read < max_bytes) {
            bytes_read++;
            if (std::isalnum(c) || c == '\'') {
                word += std::tolower(c);
            } else if (c == '.' || c == ',' || c == '!' || c == '?' || c == ':') {
                add_word(word);
                std::string p(1, c);
                freq[p]++;
            } else if (c == '\n') {
                add_word(word);
                freq["<newline>"]++;
            } else {
                add_word(word);
            }
        }
        add_word(word);

        std::vector<std::pair<int, std::string>> sorted_words;
        for (const auto& kv : freq) sorted_words.push_back({kv.second, kv.first});
        std::sort(sorted_words.rbegin(), sorted_words.rend());

        id_to_word.clear();
        word_to_id.clear();
        id_to_word.push_back("<PAD>"); word_to_id["<PAD>"] = PAD_ID;
        id_to_word.push_back("<UNK>"); word_to_id["<UNK>"] = UNK_ID;
        id_to_word.push_back("<BOS>"); word_to_id["<BOS>"] = BOS_ID;
        id_to_word.push_back("<EOS>"); word_to_id["<EOS>"] = EOS_ID;

        int num_words = std::min((int)sorted_words.size(), target_vocab_size - 4);
        for (int i = 0; i < num_words; ++i) {
            int id = id_to_word.size();
            id_to_word.push_back(sorted_words[i].second);
            word_to_id[sorted_words[i].second] = id;
        }
        V = id_to_word.size();
        std::cout << "[+] Top-" << V << " vocabulary compiled for Conversational Chat Model.\n";
    }

    std::vector<int> encode_file(const std::string& filepath, size_t max_bytes = 10000000) {
        std::ifstream file(filepath, std::ios::binary);
        std::vector<int> tokens;
        tokens.reserve(max_bytes / 4);

        std::string word;
        char c;
        size_t bytes_read = 0;

        auto add_tok = [&](const std::string& w) {
            if (w.empty()) return;
            auto it = word_to_id.find(w);
            tokens.push_back(it != word_to_id.end() ? it->second : UNK_ID);
        };

        while (file.get(c) && bytes_read < max_bytes) {
            bytes_read++;
            if (std::isalnum(c) || c == '\'') {
                word += std::tolower(c);
            } else if (c == '.' || c == ',' || c == '!' || c == '?' || c == ':') {
                if (!word.empty()) { add_tok(word); word.clear(); }
                std::string p(1, c);
                add_tok(p);
            } else if (c == '\n') {
                if (!word.empty()) { add_tok(word); word.clear(); }
                add_tok("<newline>");
            } else {
                if (!word.empty()) { add_tok(word); word.clear(); }
            }
        }
        if (!word.empty()) add_tok(word);

        std::cout << "[+] Encoded " << tokens.size() << " tokens of chat dialogue.\n";
        return tokens;
    }

    std::vector<int> encode_text(const std::string& text) const {
        std::vector<int> tokens;
        std::string word;

        auto add_tok = [&](const std::string& w) {
            if (w.empty()) return;
            auto it = word_to_id.find(w);
            tokens.push_back(it != word_to_id.end() ? it->second : UNK_ID);
        };

        for (char c : text) {
            if (std::isalnum(c) || c == '\'') {
                word += std::tolower(c);
            } else if (c == '.' || c == ',' || c == '!' || c == '?' || c == ':') {
                if (!word.empty()) { add_tok(word); word.clear(); }
                std::string p(1, c);
                add_tok(p);
            } else if (c == '\n') {
                if (!word.empty()) { add_tok(word); word.clear(); }
                add_tok("<newline>");
            } else {
                if (!word.empty()) { add_tok(word); word.clear(); }
            }
        }
        if (!word.empty()) add_tok(word);
        return tokens;
    }

    std::string decode(int id) const {
        if (id >= 0 && id < (int)id_to_word.size()) {
            if (id_to_word[id] == "<newline>") return "\n";
            return id_to_word[id];
        }
        return "<UNK>";
    }
};

// ====================================================================================
// STEP 2: MATHEMATICAL PRIMITIVES & DYNAMIC LEARNING RATE
// ====================================================================================

std::vector<double> softmax(const std::vector<double>& logits) {
    double max_val = *std::max_element(logits.begin(), logits.end());
    std::vector<double> probs(logits.size());
    double sum = 0.0;
    for (size_t i = 0; i < logits.size(); ++i) {
        probs[i] = std::exp(logits[i] - max_val);
        sum += probs[i];
    }
    for (size_t i = 0; i < logits.size(); ++i) probs[i] /= sum;
    return probs;
}

inline double get_dynamic_lr(int step, int total_steps, double max_lr = 0.006, double min_lr = 0.0002, double warmup_pct = 0.05) {
    int warmup_steps = static_cast<int>(total_steps * warmup_pct);
    if (warmup_steps < 1) warmup_steps = 1;

    if (step <= warmup_steps) {
        double pct = static_cast<double>(step) / warmup_steps;
        return 1e-4 + pct * (max_lr - 1e-4);
    } else {
        double decay_pct = static_cast<double>(step - warmup_steps) / (total_steps - warmup_steps);
        return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + std::cos(decay_pct * M_PI));
    }
}

// ====================================================================================
// STEP 3: FIRST-PRINCIPLES ADAM OPTIMIZER
// ====================================================================================

struct AdamState {
    double beta1 = 0.9, beta2 = 0.999, eps = 1e-8;
    int t = 0;

    inline void step(double& w, double g, double& m, double& v, double lr) {
        m = beta1 * m + (1.0 - beta1) * g;
        v = beta2 * v + (1.0 - beta2) * (g * g);
        double m_hat = m / (1.0 - std::pow(beta1, t));
        double v_hat = v / (1.0 - std::pow(beta2, t));
        w -= lr * (m_hat / (std::sqrt(v_hat) + eps));
    }
};

// ====================================================================================
// STEP 4: DEEP 4-LAYER TRANSFORMER MODEL CLASS
// ====================================================================================

class DeepChatTransformer {
public:
    int V;         // Vocab size (e.g. 4096)
    int D;         // Embedding dimension (e.g. 32)
    int D_ff;      // Feed-Forward expansion dimension (2 * D = 64)
    int max_T;     // Max context length (e.g. 32)
    double scale;

    // Layer 0: Embeddings
    std::vector<std::vector<double>> E; // [V][D]
    std::vector<std::vector<double>> P; // [max_T][D]

    // Layer 1: Self-Attention 1 (Q1, K1, V1)
    std::vector<std::vector<double>> Wq1, Wk1, Wv1; // [D][D]

    // Layer 2: Self-Attention 2 (Q2, K2, V2)
    std::vector<std::vector<double>> Wq2, Wk2, Wv2; // [D][D]

    // Layer 3: Feed-Forward Network ("Value Value Value" FFN)
    std::vector<std::vector<double>> Wf1; // [D][D_ff] (Expansion)
    std::vector<std::vector<double>> Wf2; // [D_ff][D] (Projection)

    // Layer 4: Output Projection Head
    std::vector<std::vector<double>> W_out; // [D][V]

    // Adam buffers
    std::vector<std::vector<double>> m_E, v_E, m_P, v_P;
    std::vector<std::vector<double>> m_Wq1, v_Wq1, m_Wk1, v_Wk1, m_Wv1, v_Wv1;
    std::vector<std::vector<double>> m_Wq2, v_Wq2, m_Wk2, v_Wk2, m_Wv2, v_Wv2;
    std::vector<std::vector<double>> m_Wf1, v_Wf1, m_Wf2, v_Wf2;
    std::vector<std::vector<double>> m_Wout, v_Wout;
    AdamState adam;

    DeepChatTransformer(int vocab_size, int embed_dim = 32, int context_len = 32, unsigned int seed = 42)
        : V(vocab_size), D(embed_dim), D_ff(2 * embed_dim), max_T(context_len) {
        scale = 1.0 / std::sqrt(static_cast<double>(D));
        std::mt19937 gen(seed);

        double std_emb = 1.0 / std::sqrt(D);
        double std_proj = std::sqrt(2.0 / (D + D));
        double std_ff1 = std::sqrt(2.0 / (D + D_ff));
        double std_ff2 = std::sqrt(2.0 / (D_ff + D));
        double std_out = std::sqrt(2.0 / (D + V));

        std::normal_distribution<double> d_emb(0.0, std_emb);
        std::normal_distribution<double> d_proj(0.0, std_proj);
        std::normal_distribution<double> d_ff1(0.0, std_ff1);
        std::normal_distribution<double> d_ff2(0.0, std_ff2);
        std::normal_distribution<double> d_out(0.0, std_out);

        auto alloc_2d = [](int r, int c, auto& dist, auto& rng) {
            std::vector<std::vector<double>> m(r, std::vector<double>(c));
            for (int i = 0; i < r; ++i) for (int j = 0; j < c; ++j) m[i][j] = dist(rng);
            return m;
        };

        E = alloc_2d(V, D, d_emb, gen);
        P = alloc_2d(max_T, D, d_emb, gen);

        Wq1 = alloc_2d(D, D, d_proj, gen);
        Wk1 = alloc_2d(D, D, d_proj, gen);
        Wv1 = alloc_2d(D, D, d_proj, gen);

        Wq2 = alloc_2d(D, D, d_proj, gen);
        Wk2 = alloc_2d(D, D, d_proj, gen);
        Wv2 = alloc_2d(D, D, d_proj, gen);

        Wf1 = alloc_2d(D, D_ff, d_ff1, gen);
        Wf2 = alloc_2d(D_ff, D, d_ff2, gen);

        W_out = alloc_2d(D, V, d_out, gen);

        auto zero_2d = [](int r, int c) { return std::vector<std::vector<double>>(r, std::vector<double>(c, 0.0)); };
        m_E = zero_2d(V, D); v_E = m_E;
        m_P = zero_2d(max_T, D); v_P = m_P;
        m_Wq1 = zero_2d(D, D); v_Wq1 = m_Wq1;
        m_Wk1 = zero_2d(D, D); v_Wk1 = m_Wk1;
        m_Wv1 = zero_2d(D, D); v_Wv1 = m_Wv1;
        m_Wq2 = zero_2d(D, D); v_Wq2 = m_Wq2;
        m_Wk2 = zero_2d(D, D); v_Wk2 = m_Wk2;
        m_Wv2 = zero_2d(D, D); v_Wv2 = m_Wv2;
        m_Wf1 = zero_2d(D, D_ff); v_Wf1 = m_Wf1;
        m_Wf2 = zero_2d(D_ff, D); v_Wf2 = m_Wf2;
        m_Wout = zero_2d(D, V); v_Wout = m_Wout;
    }

    struct Cache {
        int T;
        std::vector<int> tokens;
        std::vector<std::vector<double>> H0;     // [T][D] (Embeddings)

        // Layer 1
        std::vector<std::vector<double>> Q1, K1, V1_mat; // [T][D]
        std::vector<std::vector<double>> S1, A1;         // [T][T]
        std::vector<std::vector<double>> Attn1;          // [T][D]
        std::vector<std::vector<double>> H1;             // [T][D] (Residual 1)

        // Layer 2
        std::vector<std::vector<double>> Q2, K2, V2_mat; // [T][D]
        std::vector<std::vector<double>> S2, A2;         // [T][T]
        std::vector<std::vector<double>> Attn2;          // [T][D]
        std::vector<std::vector<double>> H2;             // [T][D] (Residual 2)

        // Layer 3 (FFN)
        std::vector<std::vector<double>> F1;             // [T][D_ff] (Post-ReLU)
        std::vector<std::vector<double>> F2;             // [T][D]
        std::vector<std::vector<double>> H3;             // [T][D] (Residual 3)

        // Layer 4 (Head)
        std::vector<std::vector<double>> Z;              // [T][V]
        std::vector<std::vector<double>> probs;          // [T][V]
    };

    struct Gradients {
        std::vector<std::vector<double>> dE, dP;
        std::vector<std::vector<double>> dWq1, dWk1, dWv1;
        std::vector<std::vector<double>> dWq2, dWk2, dWv2;
        std::vector<std::vector<double>> dWf1, dWf2;
        std::vector<std::vector<double>> dWout;

        void init(int V, int D, int D_ff, int max_T) {
            auto zero_2d = [](int r, int c) { return std::vector<std::vector<double>>(r, std::vector<double>(c, 0.0)); };
            dE = zero_2d(V, D); dP = zero_2d(max_T, D);
            dWq1 = zero_2d(D, D); dWk1 = zero_2d(D, D); dWv1 = zero_2d(D, D);
            dWq2 = zero_2d(D, D); dWk2 = zero_2d(D, D); dWv2 = zero_2d(D, D);
            dWf1 = zero_2d(D, D_ff); dWf2 = zero_2d(D_ff, D);
            dWout = zero_2d(D, V);
        }

        void accumulate(const Gradients& o) {
            auto acc = [](auto& a, const auto& b) {
                for (size_t i = 0; i < a.size(); ++i)
                    for (size_t j = 0; j < a[i].size(); ++j) a[i][j] += b[i][j];
            };
            acc(dE, o.dE); acc(dP, o.dP);
            acc(dWq1, o.dWq1); acc(dWk1, o.dWk1); acc(dWv1, o.dWv1);
            acc(dWq2, o.dWq2); acc(dWk2, o.dWk2); acc(dWv2, o.dWv2);
            acc(dWf1, o.dWf1); acc(dWf2, o.dWf2);
            acc(dWout, o.dWout);
        }
    };

    void forward(const std::vector<int>& tokens, Cache& c) const {
        int T = std::min((int)tokens.size(), max_T);
        c.T = T;
        c.tokens = tokens;

        auto alloc_2d = [T](int cols, double init_v = 0.0) {
            return std::vector<std::vector<double>>(T, std::vector<double>(cols, init_v));
        };

        c.H0 = alloc_2d(D);
        c.Q1 = alloc_2d(D); c.K1 = alloc_2d(D); c.V1_mat = alloc_2d(D);
        c.S1 = alloc_2d(T, -1e9); c.A1 = alloc_2d(T, 0.0);
        c.Attn1 = alloc_2d(D); c.H1 = alloc_2d(D);

        c.Q2 = alloc_2d(D); c.K2 = alloc_2d(D); c.V2_mat = alloc_2d(D);
        c.S2 = alloc_2d(T, -1e9); c.A2 = alloc_2d(T, 0.0);
        c.Attn2 = alloc_2d(D); c.H2 = alloc_2d(D);

        c.F1 = alloc_2d(D_ff); c.F2 = alloc_2d(D); c.H3 = alloc_2d(D);
        c.Z = alloc_2d(V); c.probs = alloc_2d(V);

        // Step 0: Input Embeddings + Positional
        for (int t = 0; t < T; ++t) {
            int tok = tokens[t];
            for (int d = 0; d < D; ++d) c.H0[t][d] = E[tok][d] + P[t][d];
        }

        // Helper lambda for Self-Attention Layer
        auto compute_attention = [&](const auto& H_in, const auto& Wq, const auto& Wk, const auto& Wv,
                                     auto& Q, auto& K, auto& V_mat, auto& S, auto& A, auto& Attn, auto& H_out) {
            for (int t = 0; t < T; ++t) {
                for (int d = 0; d < D; ++d) {
                    double q = 0.0, k = 0.0, v = 0.0;
                    for (int k_idx = 0; k_idx < D; ++k_idx) {
                        q += H_in[t][k_idx] * Wq[k_idx][d];
                        k += H_in[t][k_idx] * Wk[k_idx][d];
                        v += H_in[t][k_idx] * Wv[k_idx][d];
                    }
                    Q[t][d] = q; K[t][d] = k; V_mat[t][d] = v;
                }
            }

            for (int t = 0; t < T; ++t) {
                double max_s = -1e9;
                for (int i = 0; i <= t; ++i) {
                    double dot = 0.0;
                    for (int d = 0; d < D; ++d) dot += Q[t][d] * K[i][d];
                    S[t][i] = dot * scale;
                    if (S[t][i] > max_s) max_s = S[t][i];
                }

                double sum_exp = 0.0;
                for (int i = 0; i <= t; ++i) {
                    A[t][i] = std::exp(S[t][i] - max_s);
                    sum_exp += A[t][i];
                }
                for (int i = 0; i <= t; ++i) A[t][i] /= sum_exp;

                for (int d = 0; d < D; ++d) {
                    double c_val = 0.0;
                    for (int i = 0; i <= t; ++i) c_val += A[t][i] * V_mat[i][d];
                    Attn[t][d] = c_val;
                    H_out[t][d] = H_in[t][d] + c_val; // RESIDUAL HIGHWAY!
                }
            }
        };

        // Layer 1: Self-Attention 1 (Q1, K1, V1) + Residual
        compute_attention(c.H0, Wq1, Wk1, Wv1, c.Q1, c.K1, c.V1_mat, c.S1, c.A1, c.Attn1, c.H1);

        // Layer 2: Self-Attention 2 (Q2, K2, V2) + Residual
        compute_attention(c.H1, Wq2, Wk2, Wv2, c.Q2, c.K2, c.V2_mat, c.S2, c.A2, c.Attn2, c.H2);

        // Layer 3: Feed-Forward "Value Value Value" Network (FFN with Leaky ReLU) + Residual
        const double leaky_alpha = 0.02; // 2% leakage prevents dying neurons!
        for (int t = 0; t < T; ++t) {
            // Expansion to D_ff with Leaky ReLU
            for (int f = 0; f < D_ff; ++f) {
                double val = 0.0;
                for (int d = 0; d < D; ++d) val += c.H2[t][d] * Wf1[d][f];
                c.F1[t][f] = (val > 0.0 ? val : leaky_alpha * val); // Leaky ReLU
            }

            // Projection back to D
            for (int d = 0; d < D; ++d) {
                double val = 0.0;
                for (int f = 0; f < D_ff; ++f) val += c.F1[t][f] * Wf2[f][d];
                c.F2[t][d] = val;
                c.H3[t][d] = c.H2[t][d] + val; // RESIDUAL HIGHWAY!
            }
        }

        // Layer 4: Output Projection to Vocabulary Logits
        for (int t = 0; t < T; ++t) {
            for (int v = 0; v < V; ++v) {
                double logit = 0.0;
                for (int d = 0; d < D; ++d) logit += c.H3[t][d] * W_out[d][v];
                c.Z[t][v] = logit;
            }
            c.probs[t] = softmax(c.Z[t]);
        }
    }

    double backward(const std::vector<int>& targets, const Cache& c, Gradients& g) const {
        int T = c.T;
        double loss = 0.0;

        auto alloc_2d = [T](int cols, double init_v = 0.0) {
            return std::vector<std::vector<double>>(T, std::vector<double>(cols, init_v));
        };

        auto dZ = alloc_2d(V);
        auto dH3 = alloc_2d(D);
        auto dF2 = alloc_2d(D);
        auto dF1 = alloc_2d(D_ff);
        auto dH2 = alloc_2d(D);
        auto dH1 = alloc_2d(D);
        auto dH0 = alloc_2d(D);

        // 1. Loss & Output Head Gradients
        for (int t = 0; t < T; ++t) {
            int target = targets[t];
            double p = std::max(c.probs[t][target], 1e-12);
            loss += -std::log(p);

            for (int v = 0; v < V; ++v) {
                dZ[t][v] = (c.probs[t][v] - (v == target ? 1.0 : 0.0)) / T;
                for (int d = 0; d < D; ++d) {
                    g.dWout[d][v] += c.H3[t][d] * dZ[t][v];
                    dH3[t][d] += dZ[t][v] * W_out[d][v];
                }
            }
        }

        // 2. Backprop through Layer 3 (Feed-Forward Network)
        for (int t = 0; t < T; ++t) {
            for (int d = 0; d < D; ++d) {
                dH2[t][d] += dH3[t][d]; // Residual skip connection gradient!
                dF2[t][d] = dH3[t][d];
            }

            for (int f = 0; f < D_ff; ++f) {
                double df1 = 0.0;
                for (int d = 0; d < D; ++d) {
                    g.dWf2[f][d] += c.F1[t][f] * dF2[t][d];
                    df1 += dF2[t][d] * Wf2[f][d];
                }
                const double leaky_alpha = 0.02;
                dF1[t][f] = (c.F1[t][f] > 0.0 ? df1 : leaky_alpha * df1); // Leaky ReLU gradient
            }

            for (int d = 0; d < D; ++d) {
                for (int f = 0; f < D_ff; ++f) {
                    g.dWf1[d][f] += c.H2[t][d] * dF1[t][f];
                    dH2[t][d] += dF1[t][f] * Wf1[d][f];
                }
            }
        }

        // Helper lambda for Backpropagating through a Self-Attention Layer
        auto backprop_attention = [&](const auto& dH_out, const auto& H_in,
                                      const auto& Wq, const auto& Wk, const auto& Wv,
                                      const auto& Q, const auto& K, const auto& V_mat,
                                      const auto& A, auto& dWq, auto& dWk, auto& dWv,
                                      auto& dH_in) {
            auto dAttn = alloc_2d(D);
            auto dA = alloc_2d(T);
            auto dS = alloc_2d(T);
            auto dV_mat = alloc_2d(D);
            auto dQ = alloc_2d(D);
            auto dK = alloc_2d(D);

            for (int t = 0; t < T; ++t) {
                for (int d = 0; d < D; ++d) {
                    dH_in[t][d] += dH_out[t][d]; // Residual skip connection!
                    dAttn[t][d] = dH_out[t][d];
                }

                for (int i = 0; i <= t; ++i) {
                    for (int d = 0; d < D; ++d) {
                        dV_mat[i][d] += A[t][i] * dAttn[t][d];
                        dA[t][i] += dAttn[t][d] * V_mat[i][d];
                    }
                }

                double sum_dA_A = 0.0;
                for (int i = 0; i <= t; ++i) sum_dA_A += dA[t][i] * A[t][i];
                for (int i = 0; i <= t; ++i) dS[t][i] = A[t][i] * (dA[t][i] - sum_dA_A);

                for (int i = 0; i <= t; ++i) {
                    double grad_s = dS[t][i] * scale;
                    for (int d = 0; d < D; ++d) {
                        dQ[t][d] += grad_s * K[i][d];
                        dK[i][d] += grad_s * Q[t][d];
                    }
                }
            }

            for (int t = 0; t < T; ++t) {
                for (int k_idx = 0; k_idx < D; ++k_idx) {
                    for (int d = 0; d < D; ++d) {
                        dWq[k_idx][d] += H_in[t][k_idx] * dQ[t][d];
                        dWk[k_idx][d] += H_in[t][k_idx] * dK[t][d];
                        dWv[k_idx][d] += H_in[t][k_idx] * dV_mat[t][d];

                        dH_in[t][k_idx] += dQ[t][d] * Wq[k_idx][d] + 
                                           dK[t][d] * Wk[k_idx][d] + 
                                           dV_mat[t][d] * Wv[k_idx][d];
                    }
                }
            }
        };

        // 3. Backprop through Layer 2 (Self-Attention 2)
        backprop_attention(dH2, c.H1, Wq2, Wk2, Wv2, c.Q2, c.K2, c.V2_mat, c.A2, g.dWq2, g.dWk2, g.dWv2, dH1);

        // 4. Backprop through Layer 1 (Self-Attention 1)
        backprop_attention(dH1, c.H0, Wq1, Wk1, Wv1, c.Q1, c.K1, c.V1_mat, c.A1, g.dWq1, g.dWk1, g.dWv1, dH0);

        // 5. Backprop into Embeddings
        for (int t = 0; t < T; ++t) {
            int tok = c.tokens[t];
            for (int d = 0; d < D; ++d) {
                g.dE[tok][d] += dH0[t][d];
                g.dP[t][d] += dH0[t][d];
            }
        }

        return loss / T;
    }

    void apply_gradients(const Gradients& g, double lr, double clip_norm = 1.0) {
        adam.t++;

        // Gradient clipping
        double total_norm_sq = 0.0;
        auto add_norm = [&total_norm_sq](const auto& mat) {
            for (const auto& row : mat) for (double val : row) total_norm_sq += val * val;
        };
        add_norm(g.dWq1); add_norm(g.dWk1); add_norm(g.dWv1);
        add_norm(g.dWq2); add_norm(g.dWk2); add_norm(g.dWv2);
        add_norm(g.dWf1); add_norm(g.dWf2); add_norm(g.dWout);

        double total_norm = std::sqrt(total_norm_sq);
        double scale_g = (total_norm > clip_norm ? clip_norm / total_norm : 1.0);

        auto update = [&](auto& W, const auto& dW, auto& mW, auto& vW) {
            for (size_t i = 0; i < W.size(); ++i)
                for (size_t j = 0; j < W[i].size(); ++j)
                    adam.step(W[i][j], dW[i][j] * scale_g, mW[i][j], vW[i][j], lr);
        };

        update(E, g.dE, m_E, v_E);
        update(P, g.dP, m_P, v_P);
        update(Wq1, g.dWq1, m_Wq1, v_Wq1);
        update(Wk1, g.dWk1, m_Wk1, v_Wk1);
        update(Wv1, g.dWv1, m_Wv1, v_Wv1);
        update(Wq2, g.dWq2, m_Wq2, v_Wq2);
        update(Wk2, g.dWk2, m_Wk2, v_Wk2);
        update(Wv2, g.dWv2, m_Wv2, v_Wv2);
        update(Wf1, g.dWf1, m_Wf1, v_Wf1);
        update(Wf2, g.dWf2, m_Wf2, v_Wf2);
        update(W_out, g.dWout, m_Wout, v_Wout);
    }

    void save_checkpoint(const std::string& filepath, const ChatTokenizer& tok) const {
        std::ofstream out(filepath, std::ios::binary);
        if (!out.is_open()) return;

        out.write((char*)&V, sizeof(int));
        out.write((char*)&D, sizeof(int));
        out.write((char*)&D_ff, sizeof(int));
        out.write((char*)&max_T, sizeof(int));

        for (int i = 0; i < V; ++i) {
            int len = tok.id_to_word[i].size();
            out.write((char*)&len, sizeof(int));
            out.write(tok.id_to_word[i].data(), len);
        }

        auto write_mat = [&out](const auto& mat) {
            for (const auto& row : mat) out.write((char*)row.data(), row.size() * sizeof(double));
        };
        write_mat(E); write_mat(P);
        write_mat(Wq1); write_mat(Wk1); write_mat(Wv1);
        write_mat(Wq2); write_mat(Wk2); write_mat(Wv2);
        write_mat(Wf1); write_mat(Wf2); write_mat(W_out);
        std::cout << "[+] Deep Chat Transformer Checkpoint saved: " << filepath << " (" 
                  << (out.tellp() / 1024) << " KB)\n";
    }

    bool load_checkpoint(const std::string& filepath, ChatTokenizer& tok) {
        std::ifstream in(filepath, std::ios::binary);
        if (!in.is_open()) return false;

        in.read((char*)&V, sizeof(int));
        in.read((char*)&D, sizeof(int));
        in.read((char*)&D_ff, sizeof(int));
        in.read((char*)&max_T, sizeof(int));

        tok.V = V;
        tok.id_to_word.resize(V);
        tok.word_to_id.clear();

        for (int i = 0; i < V; ++i) {
            int len;
            in.read((char*)&len, sizeof(int));
            std::string w(len, ' ');
            in.read(&w[0], len);
            tok.id_to_word[i] = w;
            tok.word_to_id[w] = i;
        }

        auto read_mat = [&in](auto& mat) {
            for (auto& row : mat) in.read((char*)row.data(), row.size() * sizeof(double));
        };
        read_mat(E); read_mat(P);
        read_mat(Wq1); read_mat(Wk1); read_mat(Wv1);
        read_mat(Wq2); read_mat(Wk2); read_mat(Wv2);
        read_mat(Wf1); read_mat(Wf2); read_mat(W_out);
        return true;
    }
};

// ====================================================================================
// STEP 5: PURE DYNAMIC K (TOP-P) & DYNAMIC TEMPERATURE SAMPLING (NO MASKING)
// ====================================================================================

struct SamplingResult {
    int token;
    int k_used;
    double temp_used;
};

SamplingResult sample_token_dynamic(const std::vector<double>& raw_probs, double nucleus_p = 0.90) {
    std::vector<std::pair<double, int>> ranked;
    ranked.reserve(raw_probs.size());
    for (size_t i = 0; i < raw_probs.size(); ++i) {
        if (raw_probs[i] > 1e-9) ranked.push_back({raw_probs[i], (int)i});
    }
    std::sort(ranked.rbegin(), ranked.rend());

    if (ranked.empty()) return {3, 1, 0.0};

    double top_prob = ranked[0].first;
    double dynamic_temp = 0.20 + 0.60 * (1.0 - std::min(1.0, std::pow(top_prob, 1.2)));

    double cum_prob = 0.0;
    int dynamic_k = 0;
    for (size_t i = 0; i < ranked.size(); ++i) {
        dynamic_k++;
        cum_prob += ranked[i].first;
        if (cum_prob >= nucleus_p || dynamic_k >= 30) break;
    }

    std::vector<double> scaled_probs(dynamic_k);
    double scale_sum = 0.0;
    for (int i = 0; i < dynamic_k; ++i) {
        scaled_probs[i] = std::pow(ranked[i].first, 1.0 / dynamic_temp);
        scale_sum += scaled_probs[i];
    }
    for (int i = 0; i < dynamic_k; ++i) scaled_probs[i] /= scale_sum;

    static std::mt19937 gen(42);
    std::uniform_real_distribution<double> dis(0.0, 1.0);
    double r = dis(gen);
    double accum = 0.0;
    int chosen_token = ranked[0].second;
    for (int i = 0; i < dynamic_k; ++i) {
        accum += scaled_probs[i];
        if (r <= accum) {
            chosen_token = ranked[i].second;
            break;
        }
    }

    return {chosen_token, dynamic_k, dynamic_temp};
}

std::string generate_chat_response(const DeepChatTransformer& model, const ChatTokenizer& tok,
                                   const std::string& chat_prompt, int max_new_tokens = 25, 
                                   double nucleus_p = 0.90) {
    std::vector<int> tokens = tok.encode_text(chat_prompt);
    if (tokens.empty()) tokens.push_back(tok.BOS_ID);

    std::stringstream out;
    DeepChatTransformer::Cache cache;

    for (int step = 0; step < max_new_tokens; ++step) {
        int window_start = std::max(0, (int)tokens.size() - model.max_T);
        std::vector<int> context(tokens.begin() + window_start, tokens.end());

        model.forward(context, cache);

        int last_pos = cache.T - 1;
        SamplingResult decision = sample_token_dynamic(cache.probs[last_pos], nucleus_p);

        tokens.push_back(decision.token);
        std::string word = tok.decode(decision.token);

        if (word == "<EOS>" || word == "<|endoftext|>") break;

        if (word == "\n") {
            out << "\n";
        } else if (word == "." || word == "," || word == "!" || word == "?" || word == ":") {
            out << word;
        } else {
            out << " " << word;
        }
    }

    return out.str();
}

// ====================================================================================
// STEP 6: MAIN TRAINING & INTERACTIVE CHATBOT REPL
// ====================================================================================

int main(int argc, char* argv[]) {
    std::cout << "====================================================================================\n";
    std::cout << " PROGRAM 36: DEEP 4-LAYER TRANSFORMER ON CONVERSATIONAL CHAT DATA\n";
    std::cout << " Layer 1: Attn(Q1, K1, V1)  |  Layer 2: Attn(Q2, K2, V2)  |  Layer 3: FFN(V->V->V)\n";
    std::cout << " Multi-Core OpenMP C++17 Engine with Residual Connections & Dynamic Sampling\n";
    std::cout << "====================================================================================\n\n";

    std::string dataset_path = "data/chat_conversations.txt";
    const int TARGET_VOCAB = 4096;
    const int EMBED_DIM = 32;
    const int CONTEXT_LEN = 32;
    const int BATCH_SIZE = 16;
    const int TOTAL_STEPS = 900;

    ChatTokenizer tok;
    tok.build_vocab(dataset_path, TARGET_VOCAB, 10000000);
    std::vector<int> stream_tokens = tok.encode_file(dataset_path, 10000000);

    DeepChatTransformer model(tok.V, EMBED_DIM, CONTEXT_LEN, 42);

    std::string checkpoint_file = "build/deep_chat_transformer.bin";

    if (model.load_checkpoint(checkpoint_file, tok)) {
        std::cout << "[+] Found existing checkpoint: " << checkpoint_file << " (Vocab: " << tok.V << ")\n";
    }

    bool run_training = true;
    if (argc > 1 && std::string(argv[1]) == "test") run_training = false;

    if (run_training) {
        std::cout << "\n====================================================================================\n";
        std::cout << " TRAINING DEEP CHAT TRANSFORMER (4 Layers, Residual Connections, OpenMP)\n";
        std::cout << " Config: Vocab=" << tok.V << " | Dim=" << EMBED_DIM << " | Context=" << CONTEXT_LEN 
                  << " | Batch=" << BATCH_SIZE << " | Steps=" << TOTAL_STEPS << "\n";
        std::cout << " Dynamic Learning Rate: Warmup (1e-4 -> 0.006) -> Cosine Decay (-> 0.0002)\n";
        std::cout << "====================================================================================\n";

        std::mt19937 rng(42);
        std::uniform_int_distribution<size_t> dist(0, stream_tokens.size() - CONTEXT_LEN - 2);

        auto start_time = std::chrono::high_resolution_clock::now();
        size_t total_tokens_processed = 0;

        for (int step = 1; step <= TOTAL_STEPS; ++step) {
            double lr = get_dynamic_lr(step, TOTAL_STEPS, 0.006, 0.0002, 0.05);

            DeepChatTransformer::Gradients batch_grad;
            batch_grad.init(tok.V, EMBED_DIM, model.D_ff, CONTEXT_LEN);
            double batch_loss = 0.0;

            #pragma omp parallel
            {
                DeepChatTransformer::Cache local_cache;
                DeepChatTransformer::Gradients local_grad;
                local_grad.init(tok.V, EMBED_DIM, model.D_ff, CONTEXT_LEN);
                double local_loss = 0.0;

                #pragma omp for
                for (int b = 0; b < BATCH_SIZE; ++b) {
                    size_t start_idx;
                    #pragma omp critical
                    {
                        start_idx = dist(rng);
                    }

                    std::vector<int> seq_x(stream_tokens.begin() + start_idx, 
                                           stream_tokens.begin() + start_idx + CONTEXT_LEN);
                    std::vector<int> seq_y(stream_tokens.begin() + start_idx + 1, 
                                           stream_tokens.begin() + start_idx + CONTEXT_LEN + 1);

                    model.forward(seq_x, local_cache);
                    local_loss += model.backward(seq_y, local_cache, local_grad);
                }

                #pragma omp critical
                {
                    batch_grad.accumulate(local_grad);
                    batch_loss += local_loss;
                }
            }

            // Average gradients over batch
            double b_inv = 1.0 / BATCH_SIZE;
            auto scale_mat = [b_inv](auto& mat) {
                for (auto& row : mat) for (double& val : row) val *= b_inv;
            };
            scale_mat(batch_grad.dE); scale_mat(batch_grad.dP);
            scale_mat(batch_grad.dWq1); scale_mat(batch_grad.dWk1); scale_mat(batch_grad.dWv1);
            scale_mat(batch_grad.dWq2); scale_mat(batch_grad.dWk2); scale_mat(batch_grad.dWv2);
            scale_mat(batch_grad.dWf1); scale_mat(batch_grad.dWf2);
            scale_mat(batch_grad.dWout);

            model.apply_gradients(batch_grad, lr, 1.0);
            total_tokens_processed += BATCH_SIZE * CONTEXT_LEN;

            if (step % 100 == 0 || step == 1 || step == TOTAL_STEPS) {
                auto now = std::chrono::high_resolution_clock::now();
                double elapsed_s = std::chrono::duration<double>(now - start_time).count();
                double tok_per_sec = total_tokens_processed / (elapsed_s + 1e-9);

                std::cout << ">>> [Step " << std::setw(3) << step << "/" << TOTAL_STEPS << "] "
                          << "Loss: " << std::fixed << std::setprecision(4) << (batch_loss / BATCH_SIZE) << " | "
                          << "LR: " << std::setprecision(5) << lr << " | "
                          << "Speed: " << (int)tok_per_sec << " tok/s | "
                          << "Time: " << std::setprecision(1) << elapsed_s << "s\n";

                // Sample chat response
                std::string sample = generate_chat_response(model, tok, "User: Hi\nAssistant:", 12, 0.90);
                std::cout << "    [Sample Response to 'Hi']: \"" << sample << "\"\n\n";
            }
        }

        model.save_checkpoint(checkpoint_file, tok);
    }

    // ====================================================================================
    // INTERACTIVE CHATBOT REPL
    // ====================================================================================
    std::cout << "\n====================================================================================\n";
    std::cout << " INTERACTIVE CHATBOT REPL (DEEP 4-LAYER TRANSFORMER)\n";
    std::cout << " Type ANY message (e.g. 'Hi', 'Hello', 'What is your name?') to chat with your AI!\n";
    std::cout << " Type 'quit' or 'exit' to finish.\n";
    std::cout << "====================================================================================\n\n";

    std::vector<std::string> demo_chats = {
        "User: Hi\nAssistant:",
        "User: Hello\nAssistant:",
        "User: How are you?\nAssistant:",
        "User: What is your name?\nAssistant:"
    };

    std::cout << ">>> Running Preset Conversational Benchmark:\n";
    for (const auto& p : demo_chats) {
        std::cout << "\n[Input Prompt]:\n" << p;
        std::string response = generate_chat_response(model, tok, p, 18, 0.85);
        std::cout << "\n[AI Response]:" << response << "\n";
    }

    if (isatty(STDIN_FILENO)) {
        std::cout << "\n>>> Entering Live Interactive Chat (type your message and press Enter):\n";
        std::string user_msg;
        while (true) {
            std::cout << "\nYou: ";
            if (!std::getline(std::cin, user_msg) || user_msg == "quit" || user_msg == "exit") break;
            if (user_msg.empty()) continue;

            std::string formatted_prompt = "User: " + user_msg + "\nAssistant:";
            std::string reply = generate_chat_response(model, tok, formatted_prompt, 20, 0.85);
            std::cout << "AI:" << reply << "\n";
        }
        std::cout << "\n[+] Goodbye!\n";
    }

    return 0;
}
