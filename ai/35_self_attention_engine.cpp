/**
 * ====================================================================================
 * PROGRAM 35: THE CAUSAL SELF-ATTENTION ENGINE & FIRST-PRINCIPLES ADAM OPTIMIZER
 * ====================================================================================
 * Kushagra's AI Journey: The Heart of Modern Transformers (ChatGPT / LLaMA)
 *
 * Core Breakthroughs Over Program 34 (LSTM):
 * 1. ZERO GATES, ZERO LOOPS:
 *    - Replaced the 4 complex LSTM gates (Forget, Input, Candidate, Output) and
 *      conveyor belt with a clean, elegant WEIGHTED PREFIX SUM.
 * 2. THE THREE SEARCH ROLES (Q, K, V):
 *    - Query (Q): What the current word is searching for.
 *    - Key   (K): What each past word advertises.
 *    - Value (V): The actual information payload retrieved when Q and K match.
 * 3. CAUSAL MASKING (THE PREFIX RULE):
 *    - Each word can ONLY attend to itself and previous words (i <= t), never the future!
 * 4. DIRECT WORD-TO-WORD CONNECTIONS:
 *    - Word 10 connects directly to Word 1 in a single step (dot product).
 *    - Zero fading memory, zero vanishing gradients!
 * 5. ATTENTION HEATMAP VISUALIZER:
 *    - Renders the internal attention spotlight directly in terminal ASCII.
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
// STEP 1: MATHEMATICAL PRIMITIVES (SOFTMAX)
// ====================================================================================

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
        m = beta1 * m + (1.0 - beta1) * grad;
        v = beta2 * v + (1.0 - beta2) * (grad * grad);
        double m_hat = m / (1.0 - std::pow(beta1, t));
        double v_hat = v / (1.0 - std::pow(beta2, t));
        weight -= lr * (m_hat / (std::sqrt(v_hat) + eps));
    }
};

// ====================================================================================
// STEP 3: CAUSAL SELF-ATTENTION LANGUAGE MODEL
// ====================================================================================

class SelfAttentionLanguageModel {
public:
    int V;       // Vocabulary size
    int D;       // Word coordinate dimension (e.g. 16 dials per word)
    int max_len; // Maximum sentence context window (e.g. 32 words)

    // 1. Word Coordinates: C[V][D]
    std::vector<std::vector<double>> C;
    std::vector<std::vector<double>> mC, vC;

    // 2. Position Coordinates: P[max_len][D] (Tells the model WHERE each word sits!)
    std::vector<std::vector<double>> P;
    std::vector<std::vector<double>> mP, vP;

    // 3. Query Projection: Wq[D][D] (What am I searching for?)
    std::vector<std::vector<double>> Wq, mWq, vWq;

    // 4. Key Projection: Wk[D][D] (What do I advertise to others?)
    std::vector<std::vector<double>> Wk, mWk, vWk;

    // 5. Value Projection: Wv[D][D] (What content do I provide?)
    std::vector<std::vector<double>> Wv, mWv, vWv;

    // 6. Vocabulary Output Judges: Wy[D][V], By[V]
    std::vector<std::vector<double>> Wy, mWy, vWy;
    std::vector<double> By, mBy, vBy;

    AdamOptimizer adam;

    SelfAttentionLanguageModel(int vocab_size, int coord_dim = 16, int max_seq_len = 32)
        : V(vocab_size), D(coord_dim), max_len(max_seq_len) {

        std::mt19937 rng(42);
        double scale_proj = std::sqrt(2.0 / D);
        double scale_out = std::sqrt(2.0 / D);
        double scale_emb = 0.1;

        std::normal_distribution<double> dist_proj(0.0, scale_proj);
        std::normal_distribution<double> dist_out(0.0, scale_out);
        std::normal_distribution<double> dist_emb(0.0, scale_emb);

        auto alloc_2d = [](int r, int c, double val = 0.0) {
            return std::vector<std::vector<double>>(r, std::vector<double>(c, val));
        };

        // Initialize Word Coordinates C
        C = alloc_2d(V, D); mC = alloc_2d(V, D); vC = alloc_2d(V, D);
        for (int i = 0; i < V; ++i) {
            for (int d = 0; d < D; ++d) C[i][d] = dist_emb(rng);
        }

        // Initialize Position Coordinates P
        P = alloc_2d(max_len, D); mP = alloc_2d(max_len, D); vP = alloc_2d(max_len, D);
        for (int pos = 0; pos < max_len; ++pos) {
            for (int d = 0; d < D; ++d) P[pos][d] = dist_emb(rng);
        }

        // Initialize Q, K, V Projection Matrices
        auto init_proj = [&](std::vector<std::vector<double>>& W,
                             std::vector<std::vector<double>>& mW,
                             std::vector<std::vector<double>>& vW) {
            W = alloc_2d(D, D); mW = alloc_2d(D, D); vW = alloc_2d(D, D);
            for (int r = 0; r < D; ++r) {
                for (int c = 0; c < D; ++c) W[r][c] = dist_proj(rng);
            }
        };

        init_proj(Wq, mWq, vWq);
        init_proj(Wk, mWk, vWk);
        init_proj(Wv, mWv, vWv);

        // Initialize Output Judges Wy, By
        Wy = alloc_2d(D, V); mWy = alloc_2d(D, V); vWy = alloc_2d(D, V);
        for (int r = 0; r < D; ++r) {
            for (int c = 0; c < V; ++c) Wy[r][c] = dist_out(rng);
        }
        By.assign(V, 0.0); mBy.assign(V, 0.0); vBy.assign(V, 0.0);
    }

    // ------------------------------------------------------------------------
    // FORWARD & BACKWARD PASS: FIRST-PRINCIPLES CAUSAL SELF-ATTENTION
    // ------------------------------------------------------------------------
    double train_sentence(const std::vector<int>& tokens, double lr, int& correct_words, int& total_words) {
        int T = static_cast<int>(tokens.size()) - 1;
        if (T <= 0 || T >= max_len) return 0.0;

        adam.t++;

        auto alloc_2d = [](int r, int c, double val = 0.0) {
            return std::vector<std::vector<double>>(r, std::vector<double>(c, val));
        };

        // 1. Embeddings + Position: X[t] = C[token] + P[t]
        std::vector<std::vector<double>> X = alloc_2d(T, D);
        for (int t = 0; t < T; ++t) {
            for (int d = 0; d < D; ++d) {
                X[t][d] = C[tokens[t]][d] + P[t][d];
            }
        }

        // 2. Project into Queries (Q), Keys (K), and Values (V)
        std::vector<std::vector<double>> Q = alloc_2d(T, D);
        std::vector<std::vector<double>> K = alloc_2d(T, D);
        std::vector<std::vector<double>> V_mat = alloc_2d(T, D);

        for (int t = 0; t < T; ++t) {
            for (int j = 0; j < D; ++j) {
                double q_sum = 0.0, k_sum = 0.0, v_sum = 0.0;
                for (int d = 0; d < D; ++d) {
                    q_sum += X[t][d] * Wq[d][j];
                    k_sum += X[t][d] * Wk[d][j];
                    v_sum += X[t][d] * Wv[d][j];
                }
                Q[t][j] = q_sum;
                K[t][j] = k_sum;
                V_mat[t][j] = v_sum;
            }
        }

        // 3. Compute Attention Weights Matrix A[T][T] with Causal Mask (Prefix Rule)
        std::vector<std::vector<double>> scores = alloc_2d(T, T, 0.0);
        std::vector<std::vector<double>> A = alloc_2d(T, T, 0.0);
        double scale = std::sqrt(static_cast<double>(D));

        for (int t = 0; t < T; ++t) {
            // Compute dot product of Q[t] with all past keys K[i] (i <= t)
            double max_score = -1e9;
            for (int i = 0; i <= t; ++i) {
                double dot = 0.0;
                for (int d = 0; d < D; ++d) {
                    dot += Q[t][d] * K[i][d];
                }
                scores[t][i] = dot / scale;
                if (scores[t][i] > max_score) max_score = scores[t][i];
            }

            // Softmax over past words (0 ... t)
            double sum_exp = 0.0;
            for (int i = 0; i <= t; ++i) {
                A[t][i] = std::exp(scores[t][i] - max_score);
                sum_exp += A[t][i];
            }
            for (int i = 0; i <= t; ++i) {
                A[t][i] /= sum_exp;
            }
        }

        // 4. Weighted Prefix Sum: Context[t] = sum_{i=0}^t A[t][i] * V[i]
        std::vector<std::vector<double>> Context = alloc_2d(T, D);
        std::vector<std::vector<double>> Rep = alloc_2d(T, D); // Residual Connection: Rep = X + Context

        for (int t = 0; t < T; ++t) {
            for (int i = 0; i <= t; ++i) {
                double weight = A[t][i];
                for (int d = 0; d < D; ++d) {
                    Context[t][d] += weight * V_mat[i][d];
                }
            }
            // Residual connection (helps signal flow straight through)
            for (int d = 0; d < D; ++d) {
                Rep[t][d] = X[t][d] + Context[t][d];
            }
        }

        // 5. Output Judges & Loss
        std::vector<std::vector<double>> logits = alloc_2d(T, V);
        std::vector<std::vector<double>> probs = alloc_2d(T, V);
        double sentence_loss = 0.0;

        for (int t = 0; t < T; ++t) {
            int targ_word = tokens[t + 1];

            for (int k = 0; k < V; ++k) {
                double sum = By[k];
                for (int d = 0; d < D; ++d) {
                    sum += Rep[t][d] * Wy[d][k];
                }
                logits[t][k] = sum;
            }

            probs[t] = softmax(logits[t]);

            double p_targ = std::max(probs[t][targ_word], 1e-12);
            sentence_loss += -std::log(p_targ);

            int pred_k = static_cast<int>(std::distance(probs[t].begin(),
                         std::max_element(probs[t].begin(), probs[t].end())));
            if (pred_k == targ_word) correct_words++;
            total_words++;
        }

        // ====================================================================
        // BACKPROPAGATION THROUGH SELF-ATTENTION
        // ====================================================================
        auto dWy = alloc_2d(D, V);
        std::vector<double> dBy(V, 0.0);
        auto dRep = alloc_2d(T, D);

        for (int t = 0; t < T; ++t) {
            int targ_word = tokens[t + 1];
            std::vector<double> dLogits = probs[t];
            dLogits[targ_word] -= 1.0;

            for (int k = 0; k < V; ++k) {
                dBy[k] += dLogits[k];
                for (int d = 0; d < D; ++d) {
                    dWy[d][k] += dLogits[k] * Rep[t][d];
                    dRep[t][d] += dLogits[k] * Wy[d][k];
                }
            }
        }

        // Rep = X + Context -> dX = dRep, dContext = dRep
        auto dContext = dRep;
        auto dX = dRep; // Direct gradient highway via residual!

        auto dV_mat = alloc_2d(T, D);
        auto dA = alloc_2d(T, T);

        for (int t = 0; t < T; ++t) {
            for (int i = 0; i <= t; ++i) {
                for (int d = 0; d < D; ++d) {
                    dV_mat[i][d] += dContext[t][d] * A[t][i];
                    dA[t][i] += dContext[t][d] * V_mat[i][d];
                }
            }
        }

        // Gradient through Softmax on Attention Weights
        auto dScores = alloc_2d(T, T);
        for (int t = 0; t < T; ++t) {
            double sum_A_dA = 0.0;
            for (int i = 0; i <= t; ++i) sum_A_dA += A[t][i] * dA[t][i];
            for (int i = 0; i <= t; ++i) {
                dScores[t][i] = A[t][i] * (dA[t][i] - sum_A_dA);
            }
        }

        // Gradient into Q and K from dot product scores: score(t, i) = (Q[t] . K[i]) / scale
        auto dQ = alloc_2d(T, D);
        auto dK = alloc_2d(T, D);

        for (int t = 0; t < T; ++t) {
            for (int i = 0; i <= t; ++i) {
                double ds = dScores[t][i] / scale;
                for (int d = 0; d < D; ++d) {
                    dQ[t][d] += ds * K[i][d];
                    dK[i][d] += ds * Q[t][d];
                }
            }
        }

        // Gradients into projection matrices Wq, Wk, Wv and into dX
        auto dWq = alloc_2d(D, D);
        auto dWk = alloc_2d(D, D);
        auto dWv = alloc_2d(D, D);

        for (int t = 0; t < T; ++t) {
            for (int d = 0; d < D; ++d) {
                for (int j = 0; j < D; ++j) {
                    dWq[d][j] += dQ[t][j] * X[t][d];
                    dWk[d][j] += dK[t][j] * X[t][d];
                    dWv[d][j] += dV_mat[t][j] * X[t][d];

                    dX[t][d] += dQ[t][j] * Wq[d][j] + dK[t][j] * Wk[d][j] + dV_mat[t][j] * Wv[d][j];
                }
            }
        }

        // Update Word Coordinates C and Position Coordinates P
        for (int t = 0; t < T; ++t) {
            int wid = tokens[t];
            for (int d = 0; d < D; ++d) {
                adam.step(C[wid][d], dX[t][d], mC[wid][d], vC[wid][d], lr);
                adam.step(P[t][d], dX[t][d], mP[t][d], vP[t][d], lr);
            }
        }

        // Update Attention Projections with Adam
        for (int r = 0; r < D; ++r) {
            for (int c = 0; c < D; ++c) {
                adam.step(Wq[r][c], dWq[r][c], mWq[r][c], vWq[r][c], lr);
                adam.step(Wk[r][c], dWk[r][c], mWk[r][c], vWk[r][c], lr);
                adam.step(Wv[r][c], dWv[r][c], mWv[r][c], vWv[r][c], lr);
            }
        }

        // Update Output Judges
        for (int r = 0; r < D; ++r) {
            for (int c = 0; c < V; ++c) {
                adam.step(Wy[r][c], dWy[r][c], mWy[r][c], vWy[r][c], lr);
            }
        }
        for (int k = 0; k < V; ++k) adam.step(By[k], dBy[k], mBy[k], vBy[k], lr);

        return sentence_loss / T;
    }

    // ------------------------------------------------------------------------
    // PREDICT NEXT WORD & RETURN ATTENTION WEIGHTS FOR VISUALIZATION
    // ------------------------------------------------------------------------
    std::pair<std::vector<double>, std::vector<std::vector<double>>> 
    predict_next_with_attention(const std::vector<int>& prompt_tokens) {
        int T = static_cast<int>(prompt_tokens.size());
        if (T == 0) return {std::vector<double>(V, 1.0 / V), {}};

        auto alloc_2d = [](int r, int c, double val = 0.0) {
            return std::vector<std::vector<double>>(r, std::vector<double>(c, val));
        };

        std::vector<std::vector<double>> X = alloc_2d(T, D);
        for (int t = 0; t < T; ++t) {
            for (int d = 0; d < D; ++d) X[t][d] = C[prompt_tokens[t]][d] + P[t][d];
        }

        std::vector<std::vector<double>> Q = alloc_2d(T, D);
        std::vector<std::vector<double>> K = alloc_2d(T, D);
        std::vector<std::vector<double>> V_mat = alloc_2d(T, D);

        for (int t = 0; t < T; ++t) {
            for (int j = 0; j < D; ++j) {
                double q_sum = 0.0, k_sum = 0.0, v_sum = 0.0;
                for (int d = 0; d < D; ++d) {
                    q_sum += X[t][d] * Wq[d][j];
                    k_sum += X[t][d] * Wk[d][j];
                    v_sum += X[t][d] * Wv[d][j];
                }
                Q[t][j] = q_sum;
                K[t][j] = k_sum;
                V_mat[t][j] = v_sum;
            }
        }

        std::vector<std::vector<double>> scores = alloc_2d(T, T, 0.0);
        std::vector<std::vector<double>> A = alloc_2d(T, T, 0.0);
        double scale = std::sqrt(static_cast<double>(D));

        for (int t = 0; t < T; ++t) {
            double max_score = -1e9;
            for (int i = 0; i <= t; ++i) {
                double dot = 0.0;
                for (int d = 0; d < D; ++d) dot += Q[t][d] * K[i][d];
                scores[t][i] = dot / scale;
                if (scores[t][i] > max_score) max_score = scores[t][i];
            }
            double sum_exp = 0.0;
            for (int i = 0; i <= t; ++i) {
                A[t][i] = std::exp(scores[t][i] - max_score);
                sum_exp += A[t][i];
            }
            for (int i = 0; i <= t; ++i) A[t][i] /= sum_exp;
        }

        // Weighted Prefix Sum
        std::vector<std::vector<double>> Context = alloc_2d(T, D);
        std::vector<std::vector<double>> Rep = alloc_2d(T, D);

        for (int t = 0; t < T; ++t) {
            for (int i = 0; i <= t; ++i) {
                for (int d = 0; d < D; ++d) Context[t][d] += A[t][i] * V_mat[i][d];
            }
            for (int d = 0; d < D; ++d) Rep[t][d] = X[t][d] + Context[t][d];
        }

        // Predict next word from the LAST position (T - 1)
        int last_t = T - 1;
        std::vector<double> logits(V, 0.0);
        for (int k = 0; k < V; ++k) {
            double sum = By[k];
            for (int d = 0; d < D; ++d) sum += Rep[last_t][d] * Wy[d][k];
            logits[k] = sum;
        }

        return {softmax(logits), A};
    }
};

// ====================================================================================
// STEP 4: MAIN EXPERIMENTATION & HEATMAP SPOTLIGHT VERIFICATION
// ====================================================================================

int main() {
    std::cout << "\033[1;36m====================================================================================\n";
    std::cout << " PROGRAM 35: THE CAUSAL SELF-ATTENTION ENGINE & ADAM OPTIMIZER\n";
    std::cout << " The Weighted Prefix Sum (Q, K, V) + Causal Mask (No Gates, No Loops!) (C++17)\n";
    std::cout << "====================================================================================\033[0m\n\n";

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
    std::cout << " [*] Loaded " << sentences.size() << " sentences. Vocabulary: " << V << " unique words.\n\n";

    std::vector<std::vector<int>> tokenized_corpus;
    for (const auto& s : sentences) {
        std::stringstream ss(s);
        std::string w;
        std::vector<int> tokens;
        while (ss >> w) tokens.push_back(word_to_id[w]);
        tokenized_corpus.push_back(tokens);
    }

    int D = 16;
    SelfAttentionLanguageModel model(V, D, 32);

    std::cout << "------------------------------------------------------------------------------------\n";
    std::cout << " PART 1: THE CAUSAL SELF-ATTENTION ARCHITECTURE\n";
    std::cout << "------------------------------------------------------------------------------------\n";
    std::cout << " 1. Word Coordinates:      " << D << " dials per word (" << (V * D) << " parameters)\n";
    std::cout << " 2. Position Coordinates:  " << D << " dials per position (" << (32 * D) << " parameters)\n";
    std::cout << " 3. Query Projection (Wq): " << D << "x" << D << " (" << (D * D) << " parameters)\n";
    std::cout << " 4. Key Projection   (Wk): " << D << "x" << D << " (" << (D * D) << " parameters)\n";
    std::cout << " 5. Value Projection (Wv): " << D << "x" << D << " (" << (D * D) << " parameters)\n";
    std::cout << " 6. Output Judges (Wy, By):" << (D * V + V) << " parameters\n";
    int total_params = (V * D) + (32 * D) + 3 * (D * D) + (D * V + V);
    std::cout << " >>> Total Trainable Dials: " << total_params << " parameters\n";
    std::cout << " >>> Mechanism:            Weighted Prefix Sum (Self-Attention)\n";
    std::cout << " >>> Optimizer:            Dynamic Momentum (Adam)\n\n";

    std::cout << "------------------------------------------------------------------------------------\n";
    std::cout << " PART 2: TRAINING SELF-ATTENTION ON LONG-RANGE SENTENCES\n";
    std::cout << "------------------------------------------------------------------------------------\n";

    int epochs = 300;
    double learning_rate = 0.015;

    auto t_start = std::chrono::high_resolution_clock::now();

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        double total_loss = 0.0;
        int correct_words = 0;
        int total_words = 0;

        for (const auto& tokens : tokenized_corpus) {
            total_loss += model.train_sentence(tokens, learning_rate, correct_words, total_words);
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
    std::cout << "\n[+] Self-Attention Training completed in " << std::fixed << std::setprecision(1) << elapsed_ms << " ms.\n\n";

    std::cout << "====================================================================================\n";
    std::cout << " PART 3: PROVING LONG-DISTANCE ATTENTION SPOTLIGHT\n";
    std::cout << "====================================================================================\n";

    auto test_prompt = [&](const std::vector<std::string>& words) {
        std::vector<int> p_tokens;
        for (const auto& w : words) p_tokens.push_back(word_to_id[w]);

        auto [probs, A] = model.predict_next_with_attention(p_tokens);

        std::cout << " Prompt (" << words.size() << " words): \"";
        for (size_t i = 0; i < words.size(); ++i) std::cout << words[i] << (i + 1 < words.size() ? " " : "");
        std::cout << "\"\n";

        // Show where the LAST word ("palace") was shining its attention spotlight!
        int last_pos = static_cast<int>(words.size()) - 1;
        std::cout << " Attention Spotlight from \"" << words[last_pos] << "\" to past words:\n";
        for (int i = 0; i <= last_pos; ++i) {
            double pct = A[last_pos][i] * 100.0;
            int bars = static_cast<int>(pct / 5.0);
            std::string bar_str(bars, '#');
            std::cout << "   [" << std::setw(8) << std::left << words[i] << "] "
                      << std::setw(5) << std::right << std::fixed << std::setprecision(1) << pct << "% "
                      << "\033[1;36m" << bar_str << "\033[0m\n";
        }

        std::vector<std::pair<double, int>> ranked;
        for (int i = 0; i < V; ++i) ranked.push_back({probs[i], i});
        std::sort(ranked.rbegin(), ranked.rend());

        std::cout << " Top Predicted Next Words:\n";
        for (int rank = 0; rank < 3; ++rank) {
            std::cout << "    #" << (rank + 1) << " " << std::setw(12) << std::left 
                      << ("\"" + id_to_word[ranked[rank].second] + "\"")
                      << " Probability: " << std::fixed << std::setprecision(1) 
                      << (ranked[rank].first * 100.0) << "%\n";
        }
        std::cout << "\n";
    };

    test_prompt({"the", "king", "who", "lived", "in", "the", "royal", "palace"});
    test_prompt({"the", "monkey", "who", "climbed", "the", "tall", "green", "tree"});
    test_prompt({"the", "wild", "wolf", "that", "hunted", "across", "the", "deep", "forest"});

    std::cout << "====================================================================================\n";
    std::cout << " PART 4: AUTONOMOUS GENERATION WITH SELF-ATTENTION\n";
    std::cout << "====================================================================================\n";

    auto generate_story = [&](const std::vector<std::string>& seed) {
        std::vector<int> tokens;
        for (const auto& w : seed) tokens.push_back(word_to_id[w]);

        std::cout << " Starting Prompt: \"";
        for (size_t i = 0; i < seed.size(); ++i) std::cout << seed[i] << (i + 1 < seed.size() ? " " : "");
        std::cout << "\"\n Full Generated Story: \033[1;32m";
        for (const auto& w : seed) std::cout << w << " ";

        for (int step = 0; step < 18; ++step) {
            auto [probs, A] = model.predict_next_with_attention(tokens);
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
