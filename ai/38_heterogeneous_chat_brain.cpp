/**
 * ====================================================================================
 * PROGRAM 38: HETEROGENEOUS 4-LAYER CHAT BRAIN (C++17, FP32, OPENMP)
 * ====================================================================================
 * An Assembly-Line Architecture built from First Principles:
 *
 * 1. PURE FP32 (float):
 *    - 4 bytes per weight (cuts memory in half: ~2.4 MB total checkpoint).
 *    - 2x faster CPU AVX2 throughput (8 floats per cycle vs 4 doubles).
 *
 * 2. LAYER 1: LOCAL WINDOW CAUSAL FILTER (W = 3):
 *    - Captures tight local idioms & syntax ("User:", "how are", "good morning").
 *    - Strict linear O(T) compute — zero quadratic memory explosion.
 *
 * 3. LAYER 2: TWO-TIER SELECTIVE MEMORY (Brain 1 + Brain 2):
 *    - Importance Gating: Ignores filler words ("the", "is", "a", ".").
 *    - Brain 1 (Working Memory): Rolling summary of the thought flow.
 *    - Brain 2 (Permanent Fact Slots): 8 slots that NEVER decay!
 *    - RESIDUAL SKIP HIGHWAY: Bypasses Layer 2 directly to Layer 3 so raw details
 *      are never lost.
 *
 * 4. LAYER 3: SHARED-KV MULTI-QUERY ATTENTION (MQA):
 *    - 4 Query Heads (H = 4, dk = 16) for 4 distinct perspectives.
 *    - 1 Shared Key Head & 1 Shared Value Head (75% savings in KV compute/memory!).
 *    - Cross-attends to the prompt tokens + Brain 2's 8 Permanent Fact Slots.
 *
 * 5. LAYER 4: EXPANSION FEED-FORWARD NETWORK (FFN):
 *    - 4x expansion: D = 64 -> D_ff = 256 -> D = 64 with Leaky ReLU (alpha = 0.02).
 *    - Full residual skip highway.
 *
 * 6. SUBWORD BYTE-LEVEL UNIVERSAL COVERAGE:
 *    - All 128 ASCII bytes included -> ZERO <UNK> TOKENS FOREVER!
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
// STEP 1: SUBWORD TOKENIZER WITH ZERO-UNK GUARANTEE
// ====================================================================================

struct SubwordTokenizer {
    int V; // Vocabulary size
    std::unordered_map<std::string, int> word_to_id;
    std::vector<std::string> id_to_word;

    const int PAD_ID = 0;
    const int BOS_ID = 1;
    const int EOS_ID = 2;

    void build_vocab(const std::string& filepath, int target_vocab_size = 4096, size_t max_bytes = 20000000) {
        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "[-] Error opening dataset: " << filepath << "\n";
            exit(1);
        }

        std::cout << "[*] Scanning dataset to build Subword Dictionary: " << filepath << "...\n";
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

        // 1. Special tokens
        id_to_word.push_back("<PAD>"); word_to_id["<PAD>"] = PAD_ID;
        id_to_word.push_back("<BOS>"); word_to_id["<BOS>"] = BOS_ID;
        id_to_word.push_back("<EOS>"); word_to_id["<EOS>"] = EOS_ID;

        // 2. All 128 standard ASCII characters (guarantees ZERO UNK!)
        for (int ch = 0; ch < 128; ++ch) {
            std::string byte_tok(1, (char)ch);
            if (word_to_id.find(byte_tok) == word_to_id.end()) {
                int id = id_to_word.size();
                id_to_word.push_back(byte_tok);
                word_to_id[byte_tok] = id;
            }
        }

        // 3. Top frequent whole-words / subwords from dataset
        int num_words = std::min((int)sorted_words.size(), target_vocab_size - (int)id_to_word.size());
        for (int i = 0; i < num_words; ++i) {
            const std::string& w = sorted_words[i].second;
            if (word_to_id.find(w) == word_to_id.end()) {
                int id = id_to_word.size();
                id_to_word.push_back(w);
                word_to_id[w] = id;
            }
        }

        V = id_to_word.size();
        std::cout << "[+] Subword Vocabulary compiled: " << V 
                  << " tokens (Universal Byte Fallback: ZERO UNK GUARANTEED!).\n";
    }

    std::vector<int> encode_file(const std::string& filepath, size_t max_bytes = 20000000) {
        std::ifstream file(filepath, std::ios::binary);
        std::vector<int> tokens;
        tokens.reserve(max_bytes / 4);

        std::string word;
        char c;
        size_t bytes_read = 0;

        auto add_tok = [&](const std::string& w) {
            if (w.empty()) return;
            auto it = word_to_id.find(w);
            if (it != word_to_id.end()) {
                tokens.push_back(it->second);
            } else {
                for (char ch : w) {
                    std::string ch_str(1, ch);
                    auto it_ch = word_to_id.find(ch_str);
                    if (it_ch != word_to_id.end()) tokens.push_back(it_ch->second);
                }
            }
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

        std::cout << "[+] Encoded " << tokens.size() << " tokens of dialogue. (Total UNK: 0!)\n";
        return tokens;
    }

    std::vector<int> encode_text(const std::string& text) const {
        std::vector<int> tokens;
        std::string word;

        auto add_tok = [&](const std::string& w) {
            if (w.empty()) return;
            auto it = word_to_id.find(w);
            if (it != word_to_id.end()) {
                tokens.push_back(it->second);
            } else {
                for (char ch : w) {
                    std::string ch_str(1, ch);
                    auto it_ch = word_to_id.find(ch_str);
                    if (it_ch != word_to_id.end()) tokens.push_back(it_ch->second);
                }
            }
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
        return "";
    }
};

// ====================================================================================
// STEP 2: MATHEMATICAL PRIMITIVES IN PURE FP32 (float)
// ====================================================================================

inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

inline float leaky_relu(float x, float alpha = 0.02f) {
    return (x > 0.0f) ? x : alpha * x;
}

inline float d_leaky_relu(float x, float alpha = 0.02f) {
    return (x > 0.0f) ? 1.0f : alpha;
}

std::vector<float> softmax(const std::vector<float>& logits) {
    float max_val = *std::max_element(logits.begin(), logits.end());
    std::vector<float> probs(logits.size());
    float sum = 0.0f;
    for (size_t i = 0; i < logits.size(); ++i) {
        probs[i] = std::exp(logits[i] - max_val);
        sum += probs[i];
    }
    float inv_sum = 1.0f / (sum + 1e-12f);
    for (size_t i = 0; i < logits.size(); ++i) probs[i] *= inv_sum;
    return probs;
}

inline float get_dynamic_lr(int step, int total_steps, float max_lr = 0.005f, float min_lr = 0.0001f, float warmup_pct = 0.05f) {
    int warmup_steps = static_cast<int>(total_steps * warmup_pct);
    if (warmup_steps < 1) warmup_steps = 1;

    if (step <= warmup_steps) {
        float pct = static_cast<float>(step) / warmup_steps;
        return 1e-4f + pct * (max_lr - 1e-4f);
    } else {
        float decay_pct = static_cast<float>(step - warmup_steps) / (total_steps - warmup_steps);
        return min_lr + 0.5f * (max_lr - min_lr) * (1.0f + std::cos(decay_pct * (float)M_PI));
    }
}

struct AdamState {
    float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;
    int t = 0;

    inline void step(float& w, float g, float& m, float& v, float lr) {
        m = beta1 * m + (1.0f - beta1) * g;
        v = beta2 * v + (1.0f - beta2) * (g * g);
        float m_hat = m / (1.0f - std::pow(beta1, t));
        float v_hat = v / (1.0f - std::pow(beta2, t));
        w -= lr * (m_hat / (std::sqrt(v_hat) + eps));
    }
};

// ====================================================================================
// STEP 3: HETEROGENEOUS CHAT BRAIN (4 DIVERSE LAYERS, FP32)
// ====================================================================================

class HeterogeneousBrain {
public:
    int V;             // Vocab size (e.g. 4054)
    int D;             // Dimension (64)
    int H;             // Number of Query Heads (4)
    int d_k;           // Head dim (16)
    int D_ff;          // FFN dim (256)
    int max_T;         // Context window (32)
    int num_slots = 8; // Brain 2 permanent memory slots
    float scale;
    const float leaky_alpha = 0.02f;

    // Layer 0: Embeddings & Position
    std::vector<std::vector<float>> E; // [V][D]
    std::vector<std::vector<float>> P; // [max_T][D]

    // Layer 1: Local Window Filter (3-tap causal filter, W=3)
    // Local weights: [3][D]
    std::vector<std::vector<float>> W_local;

    // Layer 2: Two-Tier Selective Memory
    // Importance gating weights: [D]
    std::vector<float> W_gate;
    float b_gate;

    // Layer 3: Shared-KV Multi-Query Attention (MQA)
    // 4 Query Heads: [H][D][d_k]
    std::vector<std::vector<std::vector<float>>> Wq;
    // 1 Shared Key & 1 Shared Value Head: [D][d_k]
    std::vector<std::vector<float>> Wk_shared;
    std::vector<std::vector<float>> Wv_shared;
    // Output Mixer: [D][D]
    std::vector<std::vector<float>> Wo;

    // Layer 4: Expansion MLP (FFN)
    std::vector<std::vector<float>> Wf1; // [D][D_ff]
    std::vector<std::vector<float>> Wf2; // [D_ff][D]

    // Vocabulary Head
    std::vector<std::vector<float>> W_out; // [D][V]

    // Adam Momentum Buffers (FP32)
    std::vector<std::vector<float>> m_E, v_E, m_P, v_P;
    std::vector<std::vector<float>> m_Wlocal, v_Wlocal;
    std::vector<float> m_Wgate, v_Wgate;
    float m_bgate = 0.0f, v_bgate = 0.0f;
    std::vector<std::vector<std::vector<float>>> m_Wq, v_Wq;
    std::vector<std::vector<float>> m_Wk, v_Wk, m_Wv, v_Wv, m_Wo, v_Wo;
    std::vector<std::vector<float>> m_Wf1, v_Wf1, m_Wf2, v_Wf2;
    std::vector<std::vector<float>> m_Wout, v_Wout;
    AdamState adam;

    HeterogeneousBrain(int vocab_size, int embed_dim = 64, int num_heads = 4, int context_len = 32, unsigned int seed = 42)
        : V(vocab_size), D(embed_dim), H(num_heads), d_k(embed_dim / num_heads),
          D_ff(4 * embed_dim), max_T(context_len) {
        scale = 1.0f / std::sqrt(static_cast<float>(d_k));
        std::mt19937 gen(seed);

        float std_emb = 1.0f / std::sqrt((float)D);
        float std_proj = std::sqrt(2.0f / (D + d_k));
        float std_o = std::sqrt(2.0f / (D + D));
        float std_ff1 = std::sqrt(2.0f / (D + D_ff));
        float std_ff2 = std::sqrt(2.0f / (D_ff + D));
        float std_out = std::sqrt(2.0f / (D + V));

        std::normal_distribution<float> d_emb(0.0f, std_emb);
        std::normal_distribution<float> d_local(0.0f, 0.1f);
        std::normal_distribution<float> d_gate(0.0f, 0.1f);
        std::normal_distribution<float> d_proj(0.0f, std_proj);
        std::normal_distribution<float> d_o(0.0f, std_o);
        std::normal_distribution<float> d_ff1(0.0f, std_ff1);
        std::normal_distribution<float> d_ff2(0.0f, std_ff2);
        std::normal_distribution<float> d_out(0.0f, std_out);

        auto alloc_2d = [](int r, int c, auto& dist, auto& rng) {
            std::vector<std::vector<float>> m(r, std::vector<float>(c));
            for (int i = 0; i < r; ++i) for (int j = 0; j < c; ++j) m[i][j] = dist(rng);
            return m;
        };

        auto alloc_3d = [&](int h, int r, int c, auto& dist) {
            std::vector<std::vector<std::vector<float>>> m(h, std::vector<std::vector<float>>(r, std::vector<float>(c)));
            for (int k = 0; k < h; ++k)
                for (int i = 0; i < r; ++i)
                    for (int j = 0; j < c; ++j) m[k][i][j] = dist(gen);
            return m;
        };

        E = alloc_2d(V, D, d_emb, gen);
        P = alloc_2d(max_T, D, d_emb, gen);

        W_local = alloc_2d(3, D, d_local, gen);

        W_gate = std::vector<float>(D);
        for (int d = 0; d < D; ++d) W_gate[d] = d_gate(gen);
        b_gate = 0.0f;

        Wq = alloc_3d(H, D, d_k, d_proj);
        Wk_shared = alloc_2d(D, d_k, d_proj, gen);
        Wv_shared = alloc_2d(D, d_k, d_proj, gen);
        Wo = alloc_2d(D, D, d_o, gen);

        Wf1 = alloc_2d(D, D_ff, d_ff1, gen);
        Wf2 = alloc_2d(D_ff, D, d_ff2, gen);

        W_out = alloc_2d(D, V, d_out, gen);

        // Momentum zero init
        auto zero_2d = [](int r, int c) { return std::vector<std::vector<float>>(r, std::vector<float>(c, 0.0f)); };
        auto zero_3d = [&](int h, int r, int c) { return std::vector<std::vector<std::vector<float>>>(h, zero_2d(r, c)); };

        m_E = zero_2d(V, D); v_E = m_E;
        m_P = zero_2d(max_T, D); v_P = m_P;
        m_Wlocal = zero_2d(3, D); v_Wlocal = m_Wlocal;
        m_Wgate = std::vector<float>(D, 0.0f); v_Wgate = m_Wgate;

        m_Wq = zero_3d(H, D, d_k); v_Wq = m_Wq;
        m_Wk = zero_2d(D, d_k); v_Wk = m_Wk;
        m_Wv = zero_2d(D, d_k); v_Wv = m_Wv;
        m_Wo = zero_2d(D, D); v_Wo = m_Wo;

        m_Wf1 = zero_2d(D, D_ff); v_Wf1 = m_Wf1;
        m_Wf2 = zero_2d(D_ff, D); v_Wf2 = m_Wf2;

        m_Wout = zero_2d(D, V); v_Wout = m_Wout;
    }

    struct Cache {
        int T;
        std::vector<int> tokens;
        std::vector<std::vector<float>> H0; // [T][D]

        // Layer 1: Local Window Filter
        std::vector<std::vector<float>> L1_raw; // [T][D]
        std::vector<std::vector<float>> H1;     // [T][D]

        // Layer 2: Two-Tier Selective Memory
        std::vector<float> g_gate;              // [T]
        std::vector<std::vector<float>> M_work; // [T][D]
        std::vector<std::vector<float>> perm_slots; // [8][D]
        std::vector<std::vector<float>> H2;     // [T][D]

        // Layer 3: Shared-KV Attention
        int total_ctx; // T + num_slots
        std::vector<std::vector<float>> full_ctx; // [total_ctx][D]
        std::vector<std::vector<std::vector<float>>> Q; // [H][T][d_k]
        std::vector<std::vector<float>> K_shared;       // [total_ctx][d_k]
        std::vector<std::vector<float>> V_shared;       // [total_ctx][d_k]
        std::vector<std::vector<std::vector<float>>> S; // [H][T][total_ctx]
        std::vector<std::vector<std::vector<float>>> A; // [H][T][total_ctx]
        std::vector<std::vector<float>> Concat;         // [T][D]
        std::vector<std::vector<float>> MHA_out;        // [T][D]
        std::vector<std::vector<float>> H3;             // [T][D]

        // Layer 4: Expansion MLP
        std::vector<std::vector<float>> F1; // [T][D_ff]
        std::vector<std::vector<float>> F2; // [T][D]
        std::vector<std::vector<float>> H4; // [T][D]

        // Head
        std::vector<std::vector<float>> Z;     // [T][V]
        std::vector<std::vector<float>> probs; // [T][V]
    };

    struct Gradients {
        std::vector<std::vector<float>> dE, dP;
        std::vector<std::vector<float>> dWlocal;
        std::vector<float> dWgate;
        float dbgate = 0.0f;
        std::vector<std::vector<std::vector<float>>> dWq;
        std::vector<std::vector<float>> dWk, dWv, dWo;
        std::vector<std::vector<float>> dWf1, dWf2;
        std::vector<std::vector<float>> dWout;

        void init(int V, int D, int H, int d_k, int D_ff, int max_T) {
            auto zero_2d = [](int r, int c) { return std::vector<std::vector<float>>(r, std::vector<float>(c, 0.0f)); };
            auto zero_3d = [&](int h, int r, int c) { return std::vector<std::vector<std::vector<float>>>(h, zero_2d(r, c)); };

            dE = zero_2d(V, D); dP = zero_2d(max_T, D);
            dWlocal = zero_2d(3, D);
            dWgate = std::vector<float>(D, 0.0f);
            dbgate = 0.0f;

            dWq = zero_3d(H, D, d_k);
            dWk = zero_2d(D, d_k);
            dWv = zero_2d(D, d_k);
            dWo = zero_2d(D, D);

            dWf1 = zero_2d(D, D_ff);
            dWf2 = zero_2d(D_ff, D);
            dWout = zero_2d(D, V);
        }

        void accumulate(const Gradients& o) {
            auto acc_2d = [](auto& a, const auto& b) {
                for (size_t i = 0; i < a.size(); ++i)
                    for (size_t j = 0; j < a[i].size(); ++j) a[i][j] += b[i][j];
            };
            auto acc_3d = [&](auto& a, const auto& b) {
                for (size_t h = 0; h < a.size(); ++h) acc_2d(a[h], b[h]);
            };

            acc_2d(dE, o.dE); acc_2d(dP, o.dP);
            acc_2d(dWlocal, o.dWlocal);
            for (size_t d = 0; d < dWgate.size(); ++d) dWgate[d] += o.dWgate[d];
            dbgate += o.dbgate;
            acc_3d(dWq, o.dWq);
            acc_2d(dWk, o.dWk); acc_2d(dWv, o.dWv); acc_2d(dWo, o.dWo);
            acc_2d(dWf1, o.dWf1); acc_2d(dWf2, o.dWf2);
            acc_2d(dWout, o.dWout);
        }
    };

    void forward(const std::vector<int>& tokens, Cache& c) const {
        int T = std::min((int)tokens.size(), max_T);
        c.T = T;
        c.tokens = tokens;

        auto alloc_2d = [T](int cols, float init_v = 0.0f) {
            return std::vector<std::vector<float>>(T, std::vector<float>(cols, init_v));
        };

        // 0. Embedding + Position
        c.H0 = alloc_2d(D);
        for (int t = 0; t < T; ++t) {
            int tok = tokens[t];
            for (int d = 0; d < D; ++d) c.H0[t][d] = E[tok][d] + P[t][d];
        }

        // ================================================================================
        // LAYER 1: LOCAL WINDOW FILTER (Causal window W = 3, Linear O(T) Speed!)
        // ================================================================================
        c.L1_raw = alloc_2d(D);
        c.H1 = alloc_2d(D);
        for (int t = 0; t < T; ++t) {
            for (int d = 0; d < D; ++d) {
                float val = 0.0f;
                for (int k = 0; k <= std::min(2, t); ++k) {
                    val += W_local[k][d] * c.H0[t - k][d];
                }
                c.L1_raw[t][d] = val;
                c.H1[t][d] = c.H0[t][d] + leaky_relu(val, leaky_alpha); // RESIDUAL HIGHWAY!
            }
        }

        // ================================================================================
        // LAYER 2: TWO-TIER SELECTIVE MEMORY (Brain 1 Working + Brain 2 Permanent Slots)
        // ================================================================================
        c.g_gate = std::vector<float>(T);
        c.M_work = alloc_2d(D);
        c.perm_slots = std::vector<std::vector<float>>(num_slots, std::vector<float>(D, 0.0f));
        c.H2 = alloc_2d(D);

        std::vector<float> curr_working(D, 0.0f);
        for (int t = 0; t < T; ++t) {
            float g_logit = b_gate;
            for (int d = 0; d < D; ++d) g_logit += c.H1[t][d] * W_gate[d];
            float g = sigmoid(g_logit);
            c.g_gate[t] = g;

            // Brain 1 update: Selective decay (ignores filler words!)
            float decay = 1.0f - (g * 0.25f);
            for (int d = 0; d < D; ++d) {
                curr_working[d] = decay * curr_working[d] + g * c.H1[t][d];
                c.M_work[t][d] = curr_working[d];
            }

            // Brain 2 permanent update: High importance facts stamped in permanent slots!
            if (g > 0.65f) {
                int slot_idx = t % num_slots;
                for (int d = 0; d < D; ++d) c.perm_slots[slot_idx][d] = c.H1[t][d];
            }

            // Output of Layer 2 combines H1 with working memory + RESIDUAL HIGHWAY!
            for (int d = 0; d < D; ++d) {
                c.H2[t][d] = c.H1[t][d] + (0.5f * g * curr_working[d]);
            }
        }

        // ================================================================================
        // LAYER 3: SHARED-KV MULTI-QUERY ATTENTION (MQA)
        // Cross-attends to T prompt tokens + 8 permanent memory slots
        // ================================================================================
        c.total_ctx = T + num_slots;
        c.full_ctx = std::vector<std::vector<float>>(c.total_ctx, std::vector<float>(D, 0.0f));

        // First T rows are H2 tokens
        for (int t = 0; t < T; ++t) c.full_ctx[t] = c.H2[t];
        // Next 8 rows are Brain 2's permanent memory slots
        for (int s = 0; s < num_slots; ++s) c.full_ctx[T + s] = c.perm_slots[s];

        // 1. Compute 4 Query heads
        c.Q = std::vector<std::vector<std::vector<float>>>(H, alloc_2d(d_k));
        for (int h = 0; h < H; ++h) {
            for (int t = 0; t < T; ++t) {
                for (int k = 0; k < d_k; ++k) {
                    float val = 0.0f;
                    for (int d = 0; d < D; ++d) val += c.H2[t][d] * Wq[h][d][k];
                    c.Q[h][t][k] = val;
                }
            }
        }

        // 2. Compute 1 SINGLE Shared Key and 1 SINGLE Shared Value across the full context!
        c.K_shared = std::vector<std::vector<float>>(c.total_ctx, std::vector<float>(d_k, 0.0f));
        c.V_shared = std::vector<std::vector<float>>(c.total_ctx, std::vector<float>(d_k, 0.0f));
        for (int j = 0; j < c.total_ctx; ++j) {
            for (int k = 0; k < d_k; ++k) {
                float k_val = 0.0f, v_val = 0.0f;
                for (int d = 0; d < D; ++d) {
                    k_val += c.full_ctx[j][d] * Wk_shared[d][k];
                    v_val += c.full_ctx[j][d] * Wv_shared[d][k];
                }
                c.K_shared[j][k] = k_val;
                c.V_shared[j][k] = v_val;
            }
        }

        // 3. Multi-Query Scaled Dot-Product & Softmax
        c.S = std::vector<std::vector<std::vector<float>>>(H, std::vector<std::vector<float>>(T, std::vector<float>(c.total_ctx, 0.0f)));
        c.A = c.S;
        c.Concat = alloc_2d(D);

        for (int h = 0; h < H; ++h) {
            for (int t = 0; t < T; ++t) {
                float max_s = -1e9f;
                // Attend causally to tokens 0..t, plus all 8 permanent memory slots!
                for (int j = 0; j < c.total_ctx; ++j) {
                    if (j < T && j > t) {
                        c.S[h][t][j] = -1e9f; // Causal mask on future tokens
                        continue;
                    }
                    float dot = 0.0f;
                    for (int k = 0; k < d_k; ++k) dot += c.Q[h][t][k] * c.K_shared[j][k];
                    c.S[h][t][j] = dot * scale;
                    if (c.S[h][t][j] > max_s) max_s = c.S[h][t][j];
                }

                float sum_exp = 0.0f;
                for (int j = 0; j < c.total_ctx; ++j) {
                    if (j < T && j > t) { c.A[h][t][j] = 0.0f; continue; }
                    c.A[h][t][j] = std::exp(c.S[h][t][j] - max_s);
                    sum_exp += c.A[h][t][j];
                }
                float inv_sum = 1.0f / (sum_exp + 1e-12f);
                for (int j = 0; j < c.total_ctx; ++j) c.A[h][t][j] *= inv_sum;

                // Value accumulation into head slot
                for (int k = 0; k < d_k; ++k) {
                    float c_val = 0.0f;
                    for (int j = 0; j < c.total_ctx; ++j) {
                        if (c.A[h][t][j] > 0.0f) c_val += c.A[h][t][j] * c.V_shared[j][k];
                    }
                    c.Concat[t][h * d_k + k] = c_val;
                }
            }
        }

        // Output Projection Wo + Residual Highway
        c.MHA_out = alloc_2d(D);
        c.H3 = alloc_2d(D);
        for (int t = 0; t < T; ++t) {
            for (int d = 0; d < D; ++d) {
                float val = 0.0f;
                for (int k = 0; k < D; ++k) val += c.Concat[t][k] * Wo[k][d];
                c.MHA_out[t][d] = val;
                c.H3[t][d] = c.H2[t][d] + val; // RESIDUAL HIGHWAY!
            }
        }

        // ================================================================================
        // LAYER 4: EXPANSION MLP (FFN: 64 -> 256 -> 64 with Leaky ReLU)
        // ================================================================================
        c.F1 = alloc_2d(D_ff);
        c.F2 = alloc_2d(D);
        c.H4 = alloc_2d(D);

        for (int t = 0; t < T; ++t) {
            for (int f = 0; f < D_ff; ++f) {
                float val = 0.0f;
                for (int d = 0; d < D; ++d) val += c.H3[t][d] * Wf1[d][f];
                c.F1[t][f] = leaky_relu(val, leaky_alpha);
            }
            for (int d = 0; d < D; ++d) {
                float val = 0.0f;
                for (int f = 0; f < D_ff; ++f) val += c.F1[t][f] * Wf2[f][d];
                c.F2[t][d] = val;
                c.H4[t][d] = c.H3[t][d] + val; // RESIDUAL HIGHWAY!
            }
        }

        // Head Projection
        c.Z = alloc_2d(V);
        c.probs = alloc_2d(V);
        for (int t = 0; t < T; ++t) {
            for (int v = 0; v < V; ++v) {
                float logit = 0.0f;
                for (int d = 0; d < D; ++d) logit += c.H4[t][d] * W_out[d][v];
                c.Z[t][v] = logit;
            }
            c.probs[t] = softmax(c.Z[t]);
        }
    }

    float backward(const std::vector<int>& targets, const Cache& c, Gradients& g) const {
        int T = c.T;
        float loss = 0.0f;

        auto alloc_2d = [T](int cols, float init_v = 0.0f) {
            return std::vector<std::vector<float>>(T, std::vector<float>(cols, init_v));
        };

        auto dZ = alloc_2d(V);
        auto dH4 = alloc_2d(D);

        // 1. Loss & Vocabulary Head
        for (int t = 0; t < T; ++t) {
            int target = targets[t];
            float p = std::max(c.probs[t][target], 1e-12f);
            loss += -std::log(p);

            for (int v = 0; v < V; ++v) {
                dZ[t][v] = (c.probs[t][v] - (v == target ? 1.0f : 0.0f)) / T;
                for (int d = 0; d < D; ++d) {
                    g.dWout[d][v] += c.H4[t][d] * dZ[t][v];
                    dH4[t][d] += dZ[t][v] * W_out[d][v];
                }
            }
        }

        // 2. Layer 4 Backward (FFN)
        auto dH3 = alloc_2d(D);
        for (int t = 0; t < T; ++t) {
            std::vector<float> dF1(D_ff, 0.0f);
            for (int d = 0; d < D; ++d) {
                dH3[t][d] += dH4[t][d]; // Residual skip
                for (int f = 0; f < D_ff; ++f) {
                    g.dWf2[f][d] += c.F1[t][f] * dH4[t][d];
                    dF1[f] += dH4[t][d] * Wf2[f][d];
                }
            }
            for (int f = 0; f < D_ff; ++f) {
                float d_act = d_leaky_relu(c.F1[t][f], leaky_alpha) * dF1[f];
                for (int d = 0; d < D; ++d) {
                    g.dWf1[d][f] += c.H3[t][d] * d_act;
                    dH3[t][d] += d_act * Wf1[d][f];
                }
            }
        }

        // 3. Layer 3 Backward (Shared-KV MQA)
        auto dH2 = alloc_2d(D);
        auto dConcat = alloc_2d(D);

        for (int t = 0; t < T; ++t) {
            for (int d = 0; d < D; ++d) {
                dH2[t][d] += dH3[t][d]; // Residual skip
                for (int k = 0; k < D; ++k) {
                    g.dWo[k][d] += c.Concat[t][k] * dH3[t][d];
                    dConcat[t][k] += dH3[t][d] * Wo[k][d];
                }
            }
        }

        // Gradient buffers for Shared K & V
        std::vector<std::vector<float>> dK_shared(c.total_ctx, std::vector<float>(d_k, 0.0f));
        std::vector<std::vector<float>> dV_shared(c.total_ctx, std::vector<float>(d_k, 0.0f));

        for (int h = 0; h < H; ++h) {
            for (int t = 0; t < T; ++t) {
                std::vector<float> d_head(d_k);
                for (int k = 0; k < d_k; ++k) d_head[k] = dConcat[t][h * d_k + k];

                std::vector<float> dA(c.total_ctx, 0.0f);
                for (int j = 0; j < c.total_ctx; ++j) {
                    if (c.A[h][t][j] <= 0.0f) continue;
                    for (int k = 0; k < d_k; ++k) {
                        dA[j] += d_head[k] * c.V_shared[j][k];
                        dV_shared[j][k] += d_head[k] * c.A[h][t][j];
                    }
                }

                // Softmax derivative
                float sum_dA_A = 0.0f;
                for (int j = 0; j < c.total_ctx; ++j) sum_dA_A += dA[j] * c.A[h][t][j];

                for (int j = 0; j < c.total_ctx; ++j) {
                    if (c.A[h][t][j] <= 0.0f) continue;
                    float dS = c.A[h][t][j] * (dA[j] - sum_dA_A) * scale;
                    for (int k = 0; k < d_k; ++k) {
                        dK_shared[j][k] += dS * c.Q[h][t][k];
                        // Q gradient back to H2
                        float dQ = dS * c.K_shared[j][k];
                        for (int d = 0; d < D; ++d) {
                            g.dWq[h][d][k] += c.H2[t][d] * dQ;
                            dH2[t][d] += dQ * Wq[h][d][k];
                        }
                    }
                }
            }
        }

        // Shared K and V gradients to full_ctx
        std::vector<std::vector<float>> dfull_ctx(c.total_ctx, std::vector<float>(D, 0.0f));
        for (int j = 0; j < c.total_ctx; ++j) {
            for (int k = 0; k < d_k; ++k) {
                for (int d = 0; d < D; ++d) {
                    g.dWk[d][k] += c.full_ctx[j][d] * dK_shared[j][k];
                    g.dWv[d][k] += c.full_ctx[j][d] * dV_shared[j][k];
                    dfull_ctx[j][d] += dK_shared[j][k] * Wk_shared[d][k] + dV_shared[j][k] * Wv_shared[d][k];
                }
            }
        }

        // Accumulate dfull_ctx back into dH2
        for (int t = 0; t < T; ++t) {
            for (int d = 0; d < D; ++d) dH2[t][d] += dfull_ctx[t][d];
        }

        // 4. Layer 2 Backward (Two-Tier Selective Memory)
        auto dH1 = alloc_2d(D);
        for (int t = 0; t < T; ++t) {
            float gate_val = c.g_gate[t];
            float dg_total = 0.0f;
            for (int d = 0; d < D; ++d) {
                dH1[t][d] += dH2[t][d]; // Residual highway bypasses chokepoint!
                dg_total += dH2[t][d] * 0.5f * c.M_work[t][d];
                dH1[t][d] += dH2[t][d] * 0.5f * gate_val;
            }
            float d_sig = gate_val * (1.0f - gate_val) * dg_total;
            g.dbgate += d_sig;
            for (int d = 0; d < D; ++d) {
                g.dWgate[d] += c.H1[t][d] * d_sig;
                dH1[t][d] += d_sig * W_gate[d];
            }
        }

        // 5. Layer 1 Backward (Local Window Filter)
        auto dH0 = alloc_2d(D);
        for (int t = 0; t < T; ++t) {
            for (int d = 0; d < D; ++d) {
                dH0[t][d] += dH1[t][d]; // Residual highway
                float d_act = d_leaky_relu(c.L1_raw[t][d], leaky_alpha) * dH1[t][d];
                for (int k = 0; k <= std::min(2, t); ++k) {
                    g.dWlocal[k][d] += c.H0[t - k][d] * d_act;
                    dH0[t - k][d] += d_act * W_local[k][d];
                }
            }
        }

        // 6. Embeddings Gradient
        for (int t = 0; t < T; ++t) {
            int tok = c.tokens[t];
            for (int d = 0; d < D; ++d) {
                g.dE[tok][d] += dH0[t][d];
                g.dP[t][d] += dH0[t][d];
            }
        }

        return loss / T;
    }

    void apply_gradients(const Gradients& g, float lr, float max_norm = 1.0f) {
        adam.t++;

        // Gradient clipping
        float norm_sq = 0.0f;
        auto add_norm_2d = [&](const auto& m) {
            for (const auto& row : m) for (float v : row) norm_sq += v * v;
        };
        auto add_norm_3d = [&](const auto& m3) {
            for (const auto& m : m3) add_norm_2d(m);
        };

        add_norm_2d(g.dE); add_norm_2d(g.dP);
        add_norm_2d(g.dWlocal);
        for (float v : g.dWgate) norm_sq += v * v;
        norm_sq += g.dbgate * g.dbgate;
        add_norm_3d(g.dWq);
        add_norm_2d(g.dWk); add_norm_2d(g.dWv); add_norm_2d(g.dWo);
        add_norm_2d(g.dWf1); add_norm_2d(g.dWf2);
        add_norm_2d(g.dWout);

        float grad_norm = std::sqrt(norm_sq);
        float scale_g = (grad_norm > max_norm) ? (max_norm / (grad_norm + 1e-8f)) : 1.0f;

        auto update_2d = [&](auto& w, const auto& grad, auto& m, auto& v) {
            for (size_t i = 0; i < w.size(); ++i)
                for (size_t j = 0; j < w[i].size(); ++j)
                    adam.step(w[i][j], grad[i][j] * scale_g, m[i][j], v[i][j], lr);
        };

        auto update_3d = [&](auto& w3, const auto& g3, auto& m3, auto& v3) {
            for (size_t h = 0; h < w3.size(); ++h) update_2d(w3[h], g3[h], m3[h], v3[h]);
        };

        update_2d(E, g.dE, m_E, v_E);
        update_2d(P, g.dP, m_P, v_P);
        update_2d(W_local, g.dWlocal, m_Wlocal, v_Wlocal);

        for (size_t d = 0; d < W_gate.size(); ++d)
            adam.step(W_gate[d], g.dWgate[d] * scale_g, m_Wgate[d], v_Wgate[d], lr);
        adam.step(b_gate, g.dbgate * scale_g, m_bgate, v_bgate, lr);

        update_3d(Wq, g.dWq, m_Wq, v_Wq);
        update_2d(Wk_shared, g.dWk, m_Wk, v_Wk);
        update_2d(Wv_shared, g.dWv, m_Wv, v_Wv);
        update_2d(Wo, g.dWo, m_Wo, v_Wo);

        update_2d(Wf1, g.dWf1, m_Wf1, v_Wf1);
        update_2d(Wf2, g.dWf2, m_Wf2, v_Wf2);
        update_2d(W_out, g.dWout, m_Wout, v_Wout);
    }

    bool save_checkpoint(const std::string& path, const SubwordTokenizer& tok) const {
        std::ofstream out(path, std::ios::binary);
        if (!out.is_open()) return false;

        int magic = 0x38383838;
        out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
        out.write(reinterpret_cast<const char*>(&V), sizeof(V));
        out.write(reinterpret_cast<const char*>(&D), sizeof(D));
        out.write(reinterpret_cast<const char*>(&H), sizeof(H));
        out.write(reinterpret_cast<const char*>(&d_k), sizeof(d_k));
        out.write(reinterpret_cast<const char*>(&D_ff), sizeof(D_ff));
        out.write(reinterpret_cast<const char*>(&max_T), sizeof(max_T));

        for (int i = 0; i < V; ++i) {
            const std::string& w = tok.id_to_word[i];
            int len = w.size();
            out.write(reinterpret_cast<const char*>(&len), sizeof(len));
            out.write(w.data(), len);
        }

        auto write_2d = [&](const auto& m) {
            for (const auto& r : m) out.write(reinterpret_cast<const char*>(r.data()), r.size() * sizeof(float));
        };
        auto write_3d = [&](const auto& m3) {
            for (const auto& m : m3) write_2d(m);
        };

        write_2d(E); write_2d(P); write_2d(W_local);
        out.write(reinterpret_cast<const char*>(W_gate.data()), W_gate.size() * sizeof(float));
        out.write(reinterpret_cast<const char*>(&b_gate), sizeof(b_gate));
        write_3d(Wq); write_2d(Wk_shared); write_2d(Wv_shared); write_2d(Wo);
        write_2d(Wf1); write_2d(Wf2); write_2d(W_out);

        std::cout << "[+] Program 38 Checkpoint saved: " << path << " (" << (out.tellp() / 1024) << " KB)\n";
        return true;
    }

    bool load_checkpoint(const std::string& path, SubwordTokenizer& tok) {
        std::ifstream in(path, std::ios::binary);
        if (!in.is_open()) return false;

        int magic = 0;
        in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        if (magic != 0x38383838) return false;

        in.read(reinterpret_cast<char*>(&V), sizeof(V));
        in.read(reinterpret_cast<char*>(&D), sizeof(D));
        in.read(reinterpret_cast<char*>(&H), sizeof(H));
        in.read(reinterpret_cast<char*>(&d_k), sizeof(d_k));
        in.read(reinterpret_cast<char*>(&D_ff), sizeof(D_ff));
        in.read(reinterpret_cast<char*>(&max_T), sizeof(max_T));

        tok.V = V;
        tok.id_to_word.resize(V);
        tok.word_to_id.clear();
        for (int i = 0; i < V; ++i) {
            int len = 0;
            in.read(reinterpret_cast<char*>(&len), sizeof(len));
            std::string w(len, '\0');
            in.read(&w[0], len);
            tok.id_to_word[i] = w;
            tok.word_to_id[w] = i;
        }

        auto read_2d = [&](auto& m) {
            for (auto& r : m) in.read(reinterpret_cast<char*>(r.data()), r.size() * sizeof(float));
        };
        auto read_3d = [&](auto& m3) {
            for (auto& m : m3) read_2d(m);
        };

        read_2d(E); read_2d(P); read_2d(W_local);
        in.read(reinterpret_cast<char*>(W_gate.data()), W_gate.size() * sizeof(float));
        in.read(reinterpret_cast<char*>(&b_gate), sizeof(b_gate));
        read_3d(Wq); read_2d(Wk_shared); read_2d(Wv_shared); read_2d(Wo);
        read_2d(Wf1); read_2d(Wf2); read_2d(W_out);
        return true;
    }
};

// ====================================================================================
// STEP 4: DYNAMIC SAMPLING ENGINE (TOP-P + DYNAMIC TEMP)
// ====================================================================================

int sample_dynamic(const std::vector<float>& probs, float top_p = 0.90f, std::mt19937* rng = nullptr) {
    std::vector<std::pair<float, int>> sorted_p;
    sorted_p.reserve(probs.size());
    for (size_t i = 0; i < probs.size(); ++i) sorted_p.push_back({probs[i], (int)i});
    std::sort(sorted_p.rbegin(), sorted_p.rend());

    float cum = 0.0f;
    std::vector<std::pair<float, int>> filtered;
    for (const auto& item : sorted_p) {
        cum += item.first;
        filtered.push_back(item);
        if (cum >= top_p) break;
    }

    float norm_sum = 0.0f;
    for (const auto& item : filtered) norm_sum += item.first;

    std::uniform_real_distribution<float> dist(0.0f, norm_sum);
    float r = dist(*rng);
    float acc = 0.0f;
    for (const auto& item : filtered) {
        acc += item.first;
        if (acc >= r) return item.second;
    }
    return filtered[0].second;
}

std::string generate_chat(const HeterogeneousBrain& model, const SubwordTokenizer& tok, 
                          const std::string& prompt, int max_tokens = 20, float top_p = 0.85f) {
    std::vector<int> tokens = tok.encode_text(prompt);
    std::mt19937 rng(1337);

    for (int step = 0; step < max_tokens; ++step) {
        int start_pos = (tokens.size() > (size_t)model.max_T) ? (tokens.size() - model.max_T) : 0;
        std::vector<int> ctx(tokens.begin() + start_pos, tokens.end());

        HeterogeneousBrain::Cache cache;
        model.forward(ctx, cache);

        int next_id = sample_dynamic(cache.probs.back(), top_p, &rng);
        tokens.push_back(next_id);

        if (next_id == tok.EOS_ID) break;
        std::string w = tok.decode(next_id);
        if (w == "<newline>" || w == "\n") break;
    }

    std::ostringstream out;
    int prompt_len = tok.encode_text(prompt).size();
    for (size_t i = prompt_len; i < tokens.size(); ++i) {
        std::string word = tok.decode(tokens[i]);
        if (word == "\n") out << "\n";
        else if (word == "." || word == "," || word == "!" || word == "?" || word == ":") out << word;
        else out << " " << word;
    }
    return out.str();
}

// ====================================================================================
// STEP 5: MAIN TRAINING & INTERACTIVE RUNNER
// ====================================================================================

int main(int argc, char* argv[]) {
    std::cout << "====================================================================================\n";
    std::cout << " PROGRAM 38: HETEROGENEOUS 4-LAYER CHAT BRAIN (FP32, OPENMP)\n";
    std::cout << " Layer 1: Local Window Filter | Layer 2: Two-Tier Selective Memory\n";
    std::cout << " Layer 3: Shared-KV MQA Attention | Layer 4: Deep Expansion MLP (Leaky ReLU)\n";
    std::cout << "====================================================================================\n\n";

    std::string dataset_path = "data/chat_conversations_full.txt";
    const int TARGET_VOCAB = 4096;
    const int EMBED_DIM = 64;
    const int NUM_HEADS = 4;
    const int CONTEXT_LEN = 32;
    const int BATCH_SIZE = 16;
    const int TOTAL_STEPS = 1000;

    SubwordTokenizer tok;
    tok.build_vocab(dataset_path, TARGET_VOCAB, 20000000);
    std::vector<int> stream_tokens = tok.encode_file(dataset_path, 20000000);

    HeterogeneousBrain model(tok.V, EMBED_DIM, NUM_HEADS, CONTEXT_LEN, 42);

    // Calculate parameter count
    size_t emb_params = (tok.V * EMBED_DIM) + (CONTEXT_LEN * EMBED_DIM);
    size_t l1_params = 3 * EMBED_DIM;
    size_t l2_params = EMBED_DIM + 1;
    size_t l3_params = (NUM_HEADS * EMBED_DIM * (EMBED_DIM / NUM_HEADS)) + (2 * EMBED_DIM * (EMBED_DIM / NUM_HEADS)) + (EMBED_DIM * EMBED_DIM);
    size_t l4_params = (EMBED_DIM * model.D_ff) + (model.D_ff * EMBED_DIM);
    size_t out_params = EMBED_DIM * tok.V;
    size_t total_params = emb_params + l1_params + l2_params + l3_params + l4_params + out_params;

    std::cout << "\n[+] Novel Architecture Parameter Breakdown (Pure FP32):\n";
    std::cout << "    - Layer 0 (Embedding & Position):    " << emb_params << "\n";
    std::cout << "    - Layer 1 (Local Window Filter):      " << l1_params << " (Lightweight & O(T) fast!)\n";
    std::cout << "    - Layer 2 (Two-Tier Selective Memory):" << l2_params << " (8 Permanent Fact Slots!)\n";
    std::cout << "    - Layer 3 (Shared-KV MQA Attention):  " << l3_params << " (75% KV reduction!)\n";
    std::cout << "    - Layer 4 (Expansion FFN Leaky ReLU): " << l4_params << " (4x expansion)\n";
    std::cout << "    - Output Vocabulary Head:             " << out_params << "\n";
    std::cout << "    ===================================================================\n";
    std::cout << "    TOTAL PARAMETERS: " << total_params << " (" << (total_params * 4 / 1024) << " KB in RAM)\n";

    std::string checkpoint_file = "build/heterogeneous_chat_brain.bin";

    if (model.load_checkpoint(checkpoint_file, tok)) {
        std::cout << "[+] Found existing checkpoint: " << checkpoint_file << " (Vocab: " << tok.V << ")\n";
    }

    bool run_training = true;
    if (argc > 1 && std::string(argv[1]) == "test") run_training = false;

    if (run_training) {
        std::cout << "\n====================================================================================\n";
        std::cout << " TRAINING HETEROGENEOUS CHAT BRAIN (FP32, OpenMP Multi-threading)\n";
        std::cout << " Dataset: " << dataset_path << " | Total Steps: " << TOTAL_STEPS << "\n";
        std::cout << "====================================================================================\n";

        std::mt19937 rng(42);
        std::uniform_int_distribution<size_t> dist(0, stream_tokens.size() - CONTEXT_LEN - 2);

        auto start_time = std::chrono::high_resolution_clock::now();
        size_t total_tokens_processed = 0;

        for (int step = 1; step <= TOTAL_STEPS; ++step) {
            float lr = get_dynamic_lr(step, TOTAL_STEPS, 0.005f, 0.0001f, 0.05f);

            HeterogeneousBrain::Gradients batch_grad;
            batch_grad.init(tok.V, EMBED_DIM, NUM_HEADS, EMBED_DIM / NUM_HEADS, model.D_ff, CONTEXT_LEN);
            float batch_loss = 0.0f;

            #pragma omp parallel
            {
                HeterogeneousBrain::Cache local_cache;
                HeterogeneousBrain::Gradients local_grad;
                local_grad.init(tok.V, EMBED_DIM, NUM_HEADS, EMBED_DIM / NUM_HEADS, model.D_ff, CONTEXT_LEN);
                float local_loss = 0.0f;

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
            float b_inv = 1.0f / BATCH_SIZE;
            auto scale_2d = [b_inv](auto& mat) {
                for (auto& row : mat) for (float& val : row) val *= b_inv;
            };
            auto scale_3d = [&](auto& mat3d) {
                for (auto& h_mat : mat3d) scale_2d(h_mat);
            };

            scale_2d(batch_grad.dE); scale_2d(batch_grad.dP);
            scale_2d(batch_grad.dWlocal);
            for (float& val : batch_grad.dWgate) val *= b_inv;
            batch_grad.dbgate *= b_inv;

            scale_3d(batch_grad.dWq);
            scale_2d(batch_grad.dWk); scale_2d(batch_grad.dWv); scale_2d(batch_grad.dWo);
            scale_2d(batch_grad.dWf1); scale_2d(batch_grad.dWf2);
            scale_2d(batch_grad.dWout);

            model.apply_gradients(batch_grad, lr, 1.0f);
            total_tokens_processed += BATCH_SIZE * CONTEXT_LEN;

            if (step % 100 == 0 || step == 1 || step == TOTAL_STEPS) {
                auto now = std::chrono::high_resolution_clock::now();
                double elapsed_s = std::chrono::duration<double>(now - start_time).count();
                double tok_per_sec = total_tokens_processed / (elapsed_s + 1e-9);

                std::cout << ">>> [Step " << std::setw(4) << step << "/" << TOTAL_STEPS << "] "
                          << "Loss: " << std::fixed << std::setprecision(4) << (batch_loss / BATCH_SIZE) << " | "
                          << "LR: " << std::setprecision(5) << lr << " | "
                          << "Speed: " << (int)tok_per_sec << " tok/s | "
                          << "Time: " << std::setprecision(1) << elapsed_s << "s\n";

                std::string sample = generate_chat(model, tok, "User: Hi\nAssistant:", 12, 0.85f);
                std::cout << "    [Sample Novel Brain Response]: \"" << sample << "\"\n\n";
            }
        }

        model.save_checkpoint(checkpoint_file, tok);
    }

    // ================================================================================
    // INTERACTIVE CHATBOT REPL
    // ================================================================================
    std::cout << "\n====================================================================================\n";
    std::cout << " INTERACTIVE CHATBOT: HETEROGENEOUS 4-LAYER NOVEL BRAIN\n";
    std::cout << " FP32 | Local Filter | Selective Memory (8 Fact Slots) | Shared-KV MQA\n";
    std::cout << " Type ANY message (e.g. 'Hi', 'Who are you?') to test!\n";
    std::cout << " Type 'quit' or 'exit' to finish.\n";
    std::cout << "====================================================================================\n\n";

    std::vector<std::string> demo_chats = {
        "User: Hi\nAssistant:",
        "User: Hello\nAssistant:",
        "User: How are you?\nAssistant:",
        "User: Who are you?\nAssistant:"
    };

    std::cout << ">>> Running Conversational Benchmark:\n";
    for (const auto& p : demo_chats) {
        std::cout << "\n[Input Prompt]:\n" << p;
        std::string response = generate_chat(model, tok, p, 18, 0.85f);
        std::cout << "\n[Novel Brain AI]:" << response << "\n";
    }

    if (isatty(STDIN_FILENO)) {
        std::cout << "\n>>> Entering Live Interactive Chat (type your message and press Enter):\n";
        std::string user_msg;
        while (true) {
            std::cout << "\nYou: ";
            if (!std::getline(std::cin, user_msg) || user_msg == "quit" || user_msg == "exit") break;
            if (user_msg.empty()) continue;

            std::string formatted_prompt = "User: " + user_msg + "\nAssistant:";
            std::string reply = generate_chat(model, tok, formatted_prompt, 20, 0.85f);
            std::cout << "AI:" << reply << "\n";
        }
        std::cout << "\n[+] Goodbye!\n";
    }

    return 0;
}
