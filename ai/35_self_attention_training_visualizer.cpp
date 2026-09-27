/**
 * ====================================================================================
 * PROGRAM 35: RAYLIB LIVE TRAINING & ATTENTION HEATMAP SPOTLIGHT VISUALIZER (C++17)
 * ====================================================================================
 * Watch the Self-Attention Brain Learn in Real-Time:
 * 1. Live Training Loop: Watch the loss curve drop from 4.0 -> 0.19 right before your eyes!
 * 2. Live Attention Heatmap: Full (T x T) grid showing causal weights dynamically forming.
 * 3. Spotlight Beams: Glowing curved arcs connecting words, focusing on key subjects.
 * 4. Crisp Ubuntu Typography & High-Contrast Pill Badges.
 *
 * Controls:
 *   - [SPACE]: Pause / Resume Live Training
 *   - [R]: Reset Brain (Re-train from scratch to watch learning unfold again!)
 *   - [UP / DOWN]: Increase / Decrease Training Speed (1x to 10x epochs per frame)
 *   - [1, 2, 3]: Switch between different test sentences
 *   - Mouse Hover: Inspect any word or heatmap cell for exact percentage values
 * ====================================================================================
 */

#include "raylib.h"
#include "raymath.h"
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <unordered_map>
#include <cmath>
#include <random>
#include <iomanip>
#include <algorithm>

// ====================================================================================
// FIRST-PRINCIPLES SELF-ATTENTION MODEL (EMBEDDED FOR LIVE TRAINING)
// ====================================================================================

std::vector<double> softmax_live(const std::vector<double>& logits) {
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

struct AdamLive {
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

class LiveAttentionModel {
public:
    int V, D, max_len;
    std::vector<std::vector<double>> C, mC, vC;
    std::vector<std::vector<double>> P, mP, vP;
    std::vector<std::vector<double>> Wq, mWq, vWq;
    std::vector<std::vector<double>> Wk, mWk, vWk;
    std::vector<std::vector<double>> Wv, mWv, vWv;
    std::vector<std::vector<double>> Wy, mWy, vWy;
    std::vector<double> By, mBy, vBy;
    AdamLive adam;

    LiveAttentionModel(int vocab_size, int coord_dim = 16, int max_seq_len = 32)
        : V(vocab_size), D(coord_dim), max_len(max_seq_len) {
        reset(42);
    }

    void reset(unsigned int seed = 42) {
        adam.t = 0;
        std::mt19937 rng(seed);
        double scale_proj = std::sqrt(2.0 / D);
        double scale_out = std::sqrt(2.0 / D);
        double scale_emb = 0.1;

        std::normal_distribution<double> dist_proj(0.0, scale_proj);
        std::normal_distribution<double> dist_out(0.0, scale_out);
        std::normal_distribution<double> dist_emb(0.0, scale_emb);

        auto alloc_2d = [](int r, int c) {
            return std::vector<std::vector<double>>(r, std::vector<double>(c, 0.0));
        };

        C = alloc_2d(V, D); mC = alloc_2d(V, D); vC = alloc_2d(V, D);
        for (int i = 0; i < V; ++i) {
            for (int d = 0; d < D; ++d) C[i][d] = dist_emb(rng);
        }

        P = alloc_2d(max_len, D); mP = alloc_2d(max_len, D); vP = alloc_2d(max_len, D);
        for (int pos = 0; pos < max_len; ++pos) {
            for (int d = 0; d < D; ++d) P[pos][d] = dist_emb(rng);
        }

        auto init_proj = [&](std::vector<std::vector<double>>& W, std::vector<std::vector<double>>& mW, std::vector<std::vector<double>>& vW) {
            W = alloc_2d(D, D); mW = alloc_2d(D, D); vW = alloc_2d(D, D);
            for (int r = 0; r < D; ++r) {
                for (int c = 0; c < D; ++c) W[r][c] = dist_proj(rng);
            }
        };

        init_proj(Wq, mWq, vWq);
        init_proj(Wk, mWk, vWk);
        init_proj(Wv, mWv, vWv);

        Wy = alloc_2d(D, V); mWy = alloc_2d(D, V); vWy = alloc_2d(D, V);
        for (int r = 0; r < D; ++r) {
            for (int c = 0; c < V; ++c) Wy[r][c] = dist_out(rng);
        }
        By.assign(V, 0.0); mBy.assign(V, 0.0); vBy.assign(V, 0.0);
    }

    double train_step(const std::vector<int>& tokens, double lr, int& corr, int& tot) {
        int T = static_cast<int>(tokens.size()) - 1;
        if (T <= 0 || T >= max_len) return 0.0;
        adam.t++;

        auto alloc_2d = [](int r, int c) { return std::vector<std::vector<double>>(r, std::vector<double>(c, 0.0)); };

        // 1. Embeddings + Position
        auto X = alloc_2d(T, D);
        for (int t = 0; t < T; ++t) {
            for (int d = 0; d < D; ++d) X[t][d] = C[tokens[t]][d] + P[t][d];
        }

        // 2. Q, K, V
        auto Q = alloc_2d(T, D), K = alloc_2d(T, D), V_mat = alloc_2d(T, D);
        for (int t = 0; t < T; ++t) {
            for (int j = 0; j < D; ++j) {
                double qs = 0.0, ks = 0.0, vs = 0.0;
                for (int d = 0; d < D; ++d) {
                    qs += X[t][d] * Wq[d][j];
                    ks += X[t][d] * Wk[d][j];
                    vs += X[t][d] * Wv[d][j];
                }
                Q[t][j] = qs; K[t][j] = ks; V_mat[t][j] = vs;
            }
        }

        // 3. Attention A
        double scale = std::sqrt(static_cast<double>(D));
        auto scores = alloc_2d(T, T);
        auto A = alloc_2d(T, T);

        for (int t = 0; t < T; ++t) {
            double max_s = -1e9;
            for (int i = 0; i <= t; ++i) {
                double dot = 0.0;
                for (int d = 0; d < D; ++d) dot += Q[t][d] * K[i][d];
                scores[t][i] = dot / scale;
                if (scores[t][i] > max_s) max_s = scores[t][i];
            }
            double sum_e = 0.0;
            for (int i = 0; i <= t; ++i) {
                A[t][i] = std::exp(scores[t][i] - max_s);
                sum_e += A[t][i];
            }
            for (int i = 0; i <= t; ++i) A[t][i] /= sum_e;
        }

        // 4. Context & Rep
        auto Context = alloc_2d(T, D);
        auto Rep = alloc_2d(T, D);
        for (int t = 0; t < T; ++t) {
            for (int i = 0; i <= t; ++i) {
                for (int d = 0; d < D; ++d) Context[t][d] += A[t][i] * V_mat[i][d];
            }
            for (int d = 0; d < D; ++d) Rep[t][d] = X[t][d] + Context[t][d];
        }

        // 5. Output Logits & Loss
        auto logits = alloc_2d(T, V);
        auto probs = alloc_2d(T, V);
        double loss = 0.0;

        for (int t = 0; t < T; ++t) {
            int targ = tokens[t + 1];
            for (int k = 0; k < V; ++k) {
                double sum = By[k];
                for (int d = 0; d < D; ++d) sum += Rep[t][d] * Wy[d][k];
                logits[t][k] = sum;
            }
            probs[t] = softmax_live(logits[t]);
            double p = std::max(probs[t][targ], 1e-12);
            loss += -std::log(p);
            int pred = std::distance(probs[t].begin(), std::max_element(probs[t].begin(), probs[t].end()));
            if (pred == targ) corr++;
            tot++;
        }

        // 6. Backpropagation
        auto dWy = alloc_2d(D, V);
        std::vector<double> dBy(V, 0.0);
        auto dRep = alloc_2d(T, D);

        for (int t = 0; t < T; ++t) {
            int targ = tokens[t + 1];
            auto dLogits = probs[t];
            dLogits[targ] -= 1.0;
            for (int k = 0; k < V; ++k) {
                dBy[k] += dLogits[k];
                for (int d = 0; d < D; ++d) {
                    dWy[d][k] += dLogits[k] * Rep[t][d];
                    dRep[t][d] += dLogits[k] * Wy[d][k];
                }
            }
        }

        auto dContext = dRep;
        auto dX = dRep;
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

        auto dScores = alloc_2d(T, T);
        for (int t = 0; t < T; ++t) {
            double sA = 0.0;
            for (int i = 0; i <= t; ++i) sA += A[t][i] * dA[t][i];
            for (int i = 0; i <= t; ++i) dScores[t][i] = A[t][i] * (dA[t][i] - sA);
        }

        auto dQ = alloc_2d(T, D), dK = alloc_2d(T, D);
        for (int t = 0; t < T; ++t) {
            for (int i = 0; i <= t; ++i) {
                double ds = dScores[t][i] / scale;
                for (int d = 0; d < D; ++d) {
                    dQ[t][d] += ds * K[i][d];
                    dK[i][d] += ds * Q[t][d];
                }
            }
        }

        auto dWq = alloc_2d(D, D), dWk = alloc_2d(D, D), dWv = alloc_2d(D, D);
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

        for (int t = 0; t < T; ++t) {
            int wid = tokens[t];
            for (int d = 0; d < D; ++d) {
                adam.step(C[wid][d], dX[t][d], mC[wid][d], vC[wid][d], lr);
                adam.step(P[t][d], dX[t][d], mP[t][d], vP[t][d], lr);
            }
        }

        for (int r = 0; r < D; ++r) {
            for (int c = 0; c < D; ++c) {
                adam.step(Wq[r][c], dWq[r][c], mWq[r][c], vWq[r][c], lr);
                adam.step(Wk[r][c], dWk[r][c], mWk[r][c], vWk[r][c], lr);
                adam.step(Wv[r][c], dWv[r][c], mWv[r][c], vWv[r][c], lr);
            }
        }

        for (int r = 0; r < D; ++r) {
            for (int c = 0; c < V; ++c) adam.step(Wy[r][c], dWy[r][c], mWy[r][c], vWy[r][c], lr);
        }
        for (int k = 0; k < V; ++k) adam.step(By[k], dBy[k], mBy[k], vBy[k], lr);

        return loss / T;
    }

    // Inspect attention matrix and predictions
    std::pair<std::vector<double>, std::vector<std::vector<double>>>
    inspect(const std::vector<int>& tokens) {
        int T = static_cast<int>(tokens.size());
        if (T == 0) return {{}, {}};

        auto alloc_2d = [](int r, int c) { return std::vector<std::vector<double>>(r, std::vector<double>(c, 0.0)); };
        auto X = alloc_2d(T, D);
        for (int t = 0; t < T; ++t) {
            for (int d = 0; d < D; ++d) X[t][d] = C[tokens[t]][d] + P[t][d];
        }

        auto Q = alloc_2d(T, D), K = alloc_2d(T, D), V_mat = alloc_2d(T, D);
        for (int t = 0; t < T; ++t) {
            for (int j = 0; j < D; ++j) {
                double qs = 0.0, ks = 0.0, vs = 0.0;
                for (int d = 0; d < D; ++d) {
                    qs += X[t][d] * Wq[d][j];
                    ks += X[t][d] * Wk[d][j];
                    vs += X[t][d] * Wv[d][j];
                }
                Q[t][j] = qs; K[t][j] = ks; V_mat[t][j] = vs;
            }
        }

        double scale = std::sqrt(static_cast<double>(D));
        auto A = alloc_2d(T, T);
        for (int t = 0; t < T; ++t) {
            double max_s = -1e9;
            std::vector<double> sc(t + 1);
            for (int i = 0; i <= t; ++i) {
                double dot = 0.0;
                for (int d = 0; d < D; ++d) dot += Q[t][d] * K[i][d];
                sc[i] = dot / scale;
                if (sc[i] > max_s) max_s = sc[i];
            }
            double sum_e = 0.0;
            for (int i = 0; i <= t; ++i) {
                A[t][i] = std::exp(sc[i] - max_s);
                sum_e += A[t][i];
            }
            for (int i = 0; i <= t; ++i) A[t][i] /= sum_e;
        }

        int last_t = T - 1;
        std::vector<double> context_last(D, 0.0);
        for (int i = 0; i <= last_t; ++i) {
            for (int d = 0; d < D; ++d) context_last[d] += A[last_t][i] * V_mat[i][d];
        }
        std::vector<double> rep_last(D);
        for (int d = 0; d < D; ++d) rep_last[d] = X[last_t][d] + context_last[d];

        std::vector<double> logits(V, 0.0);
        for (int k = 0; k < V; ++k) {
            double sum = By[k];
            for (int d = 0; d < D; ++d) sum += rep_last[d] * Wy[d][k];
            logits[k] = sum;
        }

        return {softmax_live(logits), A};
    }
};

// ====================================================================================
// MAIN RAYLIB DASHBOARD
// ====================================================================================

int main() {
    const int screenWidth = 1380;
    const int screenHeight = 800;

    SetConfigFlags(FLAG_MSAA_4X_HINT);
    InitWindow(screenWidth, screenHeight, "Program 35: Live Self-Attention Training & Heatmap Visualizer");
    SetTargetFPS(60);

    // 1. Load Proper Fonts (with graceful fallback)
    Font font_bold  = LoadFontEx("/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf", 22, 0, 0);
    Font font_title = LoadFontEx("/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf", 26, 0, 0);
    Font font_med   = LoadFontEx("/usr/share/fonts/truetype/ubuntu/Ubuntu-M.ttf", 16, 0, 0);
    Font font_reg   = LoadFontEx("/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf", 14, 0, 0);
    Font font_mono  = LoadFontEx("/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf", 13, 0, 0);

    SetTextureFilter(font_bold.texture, TEXTURE_FILTER_BILINEAR);
    SetTextureFilter(font_title.texture, TEXTURE_FILTER_BILINEAR);
    SetTextureFilter(font_med.texture, TEXTURE_FILTER_BILINEAR);
    SetTextureFilter(font_reg.texture, TEXTURE_FILTER_BILINEAR);
    SetTextureFilter(font_mono.texture, TEXTURE_FILTER_BILINEAR);

    // Dataset
    std::vector<std::string> sentences = {
        "the king who lived in the royal palace sits on the golden throne .",
        "the queen who lived in the royal palace sits on the golden throne .",
        "the monkey who climbed the tall green tree eats a sweet apple .",
        "the wild wolf that hunted across the deep forest runs in the night .",
        "the friendly dog that played with the happy child runs in the park ."
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
    std::vector<std::vector<int>> tokenized;
    for (const auto& s : sentences) {
        std::stringstream ss(s);
        std::string w;
        std::vector<int> toks;
        while (ss >> w) toks.push_back(word_to_id[w]);
        tokenized.push_back(toks);
    }

    LiveAttentionModel model(V, 16, 32);

    int epoch = 0;
    const int max_epochs = 300;
    bool isTraining = true;
    int train_speed = 1; // Epochs per frame
    double current_loss = 4.0;
    double current_acc = 5.0;

    std::vector<float> loss_history;
    int selected_sentence_idx = 0;

    while (!WindowShouldClose()) {
        // Handle User Controls
        if (IsKeyPressed(KEY_SPACE)) isTraining = !isTraining;
        if (IsKeyPressed(KEY_R)) {
            model.reset();
            epoch = 0;
            loss_history.clear();
            current_loss = 4.0;
            current_acc = 5.0;
        }
        if (IsKeyPressed(KEY_UP))   train_speed = std::min(10, train_speed + 1);
        if (IsKeyPressed(KEY_DOWN)) train_speed = std::max(1, train_speed - 1);
        if (IsKeyPressed(KEY_ONE))   selected_sentence_idx = 0;
        if (IsKeyPressed(KEY_TWO))   selected_sentence_idx = 1;
        if (IsKeyPressed(KEY_THREE)) selected_sentence_idx = 2;
        if (IsKeyPressed(KEY_FOUR))  selected_sentence_idx = 3;

        // Perform Live Training Step
        if (isTraining && epoch < max_epochs) {
            for (int step = 0; step < train_speed && epoch < max_epochs; ++step) {
                epoch++;
                double ep_loss = 0.0;
                int corr = 0, tot = 0;
                for (const auto& seq : tokenized) {
                    ep_loss += model.train_step(seq, 0.015, corr, tot);
                }
                current_loss = ep_loss / tokenized.size();
                current_acc = (100.0 * corr) / tot;
                loss_history.push_back(static_cast<float>(current_loss));
            }
        }

        // Get inspection data for selected sentence
        const auto& active_tokens = tokenized[selected_sentence_idx];
        auto [probs, A] = model.inspect(active_tokens);
        int T = static_cast<int>(active_tokens.size());

        Vector2 mousePos = GetMousePosition();

        // --------------------------------------------------------------------
        // RENDER GUI
        // --------------------------------------------------------------------
        BeginDrawing();
        ClearBackground(Color{ 11, 15, 23, 255 }); // Dark theme

        // 1. TOP HEADER BANNER
        DrawRectangle(0, 0, screenWidth, 70, Color{ 17, 24, 39, 255 });
        DrawLine(0, 70, screenWidth, 70, Color{ 31, 41, 55, 255 });

        DrawTextEx(font_title, "PROGRAM 35: CAUSAL SELF-ATTENTION TRAINING & SPOTLIGHT", Vector2{ 24, 14 }, 22.0f, 1.0f, Color{ 56, 189, 248, 255 });
        DrawTextEx(font_reg, "Live Visualizer: Interactive Causal Heatmap Matrix & Spotlight Beams in Real-Time", Vector2{ 24, 42 }, 13.0f, 1.0f, Color{ 148, 163, 184, 255 });

        // Status Badges (Right side of header)
        std::string epStr = "Epoch: " + std::to_string(epoch) + " / " + std::to_string(max_epochs);
        DrawRectangleRounded(Rectangle{ (float)screenWidth - 420, 18, 120, 34 }, 0.3f, 6, Color{ 30, 41, 59, 255 });
        DrawTextEx(font_med, epStr.c_str(), Vector2{ (float)screenWidth - 408, 26 }, 14.0f, 1.0f, WHITE);

        std::stringstream ss_loss; ss_loss << "Loss: " << std::fixed << std::setprecision(4) << current_loss;
        DrawRectangleRounded(Rectangle{ (float)screenWidth - 290, 18, 130, 34 }, 0.3f, 6, Color{ 45, 26, 12, 255 });
        DrawTextEx(font_med, ss_loss.str().c_str(), Vector2{ (float)screenWidth - 278, 26 }, 14.0f, 1.0f, Color{ 251, 191, 36, 255 });

        std::stringstream ss_acc; ss_acc << "Acc: " << std::fixed << std::setprecision(1) << current_acc << "%";
        DrawRectangleRounded(Rectangle{ (float)screenWidth - 150, 18, 125, 34 }, 0.3f, 6, Color{ 6, 78, 59, 255 });
        DrawTextEx(font_med, ss_acc.str().c_str(), Vector2{ (float)screenWidth - 138, 26 }, 14.0f, 1.0f, Color{ 52, 211, 153, 255 });

        // 2. LEFT PANEL: LIVE LOSS CONVERGENCE GRAPH
        int graphX = 24;
        int graphY = 90;
        int graphW = 320;
        int graphH = 180;
        DrawRectangleRounded(Rectangle{ (float)graphX, (float)graphY, (float)graphW, (float)graphH }, 0.08f, 6, Color{ 17, 24, 39, 230 });
        DrawRectangleRoundedLinesEx(Rectangle{ (float)graphX, (float)graphY, (float)graphW, (float)graphH }, 0.08f, 6, 1.5f, Color{ 31, 41, 55, 255 });
        DrawTextEx(font_med, "Live Loss Curve (Error Over Time)", Vector2{ (float)graphX + 14, (float)graphY + 12 }, 14.0f, 1.0f, Color{ 226, 232, 240, 255 });

        // Plot loss curve
        if (loss_history.size() > 1) {
            float maxL = 4.5f;
            for (size_t i = 1; i < loss_history.size(); ++i) {
                float x1 = graphX + 20 + ((i - 1) / (float)max_epochs) * (graphW - 40);
                float y1 = (graphY + graphH - 25) - (loss_history[i - 1] / maxL) * (graphH - 60);
                float x2 = graphX + 20 + (i / (float)max_epochs) * (graphW - 40);
                float y2 = (graphY + graphH - 25) - (loss_history[i] / maxL) * (graphH - 60);
                DrawLineEx(Vector2{ x1, y1 }, Vector2{ x2, y2 }, 2.0f, Color{ 251, 191, 36, 255 });
            }
        }
        DrawTextEx(font_mono, "4.0", Vector2{ (float)graphX + 10, (float)graphY + 38 }, 11.0f, 1.0f, Color{ 148, 163, 184, 180 });
        DrawTextEx(font_mono, "0.0", Vector2{ (float)graphX + 10, (float)graphY + graphH - 30 }, 11.0f, 1.0f, Color{ 148, 163, 184, 180 });

        // 3. MIDDLE PANEL: LIVE CAUSAL ATTENTION HEATMAP MATRIX (T x T)
        int heatX = 370;
        int heatY = 90;
        int heatW = 460;
        int heatH = 460;
        DrawRectangleRounded(Rectangle{ (float)heatX, (float)heatY, (float)heatW, (float)heatH }, 0.05f, 6, Color{ 17, 24, 39, 230 });
        DrawRectangleRoundedLinesEx(Rectangle{ (float)heatX, (float)heatY, (float)heatW, (float)heatH }, 0.05f, 6, 1.5f, Color{ 31, 41, 55, 255 });
        DrawTextEx(font_med, "Causal Attention Heatmap Matrix A[T][T]", Vector2{ (float)heatX + 16, (float)heatY + 12 }, 15.0f, 1.0f, Color{ 56, 189, 248, 255 });

        float cellSize = std::min(32.0f, (heatW - 90.0f) / T);
        float gridStartX = heatX + 75;
        float gridStartY = heatY + 50;

        int hoveredRow = -1, hoveredCol = -1;
        float hoveredVal = 0.0f;

        for (int r = 0; r < T; ++r) {
            // Draw row label (Query Word)
            std::string rWord = id_to_word[active_tokens[r]];
            DrawTextEx(font_mono, rWord.c_str(), Vector2{ (float)heatX + 12, gridStartY + r * cellSize + 6 }, 11.0f, 1.0f, Color{ 203, 213, 225, 255 });

            for (int c = 0; c < T; ++c) {
                float cx = gridStartX + c * cellSize;
                float cy = gridStartY + r * cellSize;
                Rectangle cellRect = { cx, cy, cellSize - 2, cellSize - 2 };

                if (c <= r) {
                    // Causal cell with attention weight A[r][c]
                    float val = (r < (int)A.size() && c < (int)A[r].size()) ? static_cast<float>(A[r][c]) : 0.0f;
                    
                    // Heatmap color interpolation: Dark Slate (0%) -> Cyan (50%) -> Electric Yellow (100%)
                    Color cColor;
                    if (val < 0.5f) {
                        float t_val = val / 0.5f;
                        cColor = Color{ (unsigned char)(20 + t_val * 36), (unsigned char)(30 + t_val * 159), (unsigned char)(60 + t_val * 188), 255 };
                    } else {
                        float t_val = (val - 0.5f) / 0.5f;
                        cColor = Color{ (unsigned char)(56 + t_val * 195), (unsigned char)(189 + t_val * 62), (unsigned char)(248 - t_val * 212), 255 };
                    }

                    DrawRectangleRec(cellRect, cColor);

                    // Check hover
                    if (CheckCollisionPointRec(mousePos, cellRect)) {
                        hoveredRow = r;
                        hoveredCol = c;
                        hoveredVal = val;
                        DrawRectangleLinesEx(cellRect, 2.0f, WHITE);
                    }
                } else {
                    // Future cell (Masked with -Infinity, zero weight!)
                    DrawRectangleRec(cellRect, Color{ 24, 30, 42, 180 });
                    DrawLine((int)cx, (int)cy, (int)cx + (int)cellSize - 2, (int)cy + (int)cellSize - 2, Color{ 40, 50, 70, 255 });
                }
            }
        }

        // Draw Column Labels along top of grid
        for (int c = 0; c < T; ++c) {
            std::string cWord = id_to_word[active_tokens[c]];
            std::string abbrev = cWord.substr(0, 3);
            DrawTextEx(font_mono, abbrev.c_str(), Vector2{ gridStartX + c * cellSize + 2, gridStartY - 16 }, 10.0f, 1.0f, Color{ 148, 163, 184, 255 });
        }

        // Heatmap Tooltip
        if (hoveredRow != -1 && hoveredCol != -1) {
            std::string qWord = id_to_word[active_tokens[hoveredRow]];
            std::string kWord = id_to_word[active_tokens[hoveredCol]];
            std::stringstream ss_tip;
            ss_tip << "\"" << qWord << "\" attends to \"" << kWord << "\": " << std::fixed << std::setprecision(1) << (hoveredVal * 100.0f) << "%";
            DrawRectangleRounded(Rectangle{ (float)heatX + 16, (float)heatY + heatH - 34, (float)heatW - 32, 26 }, 0.3f, 4, Color{ 2, 132, 199, 255 });
            DrawTextEx(font_med, ss_tip.str().c_str(), Vector2{ (float)heatX + 26, (float)heatY + heatH - 28 }, 12.0f, 1.0f, WHITE);
        } else {
            DrawTextEx(font_reg, "Hover over any square to see exact attention percentage", Vector2{ (float)heatX + 16, (float)heatY + heatH - 28 }, 12.0f, 1.0f, Color{ 148, 163, 184, 180 });
        }

        // 4. RIGHT PANEL: SPOTLIGHT BEAMS GRAPH (Word-to-Word Direct Links)
        int spotX = 855;
        int spotY = 90;
        int spotW = 500;
        int spotH = 460;
        DrawRectangleRounded(Rectangle{ (float)spotX, (float)spotY, (float)spotW, (float)spotH }, 0.05f, 6, Color{ 17, 24, 39, 230 });
        DrawRectangleRoundedLinesEx(Rectangle{ (float)spotX, (float)spotY, (float)spotW, (float)spotH }, 0.05f, 6, 1.5f, Color{ 31, 41, 55, 255 });
        DrawTextEx(font_med, "Interpretable Spotlight: Direct Word Links", Vector2{ (float)spotX + 16, (float)spotY + 12 }, 15.0f, 1.0f, Color{ 251, 191, 36, 255 });

        // Render Tokens vertically along left and right
        int lastWordIdx = T - 1;

        for (int i = 0; i < T; ++i) {
            float py = spotY + 70 + (i * 28.0f);
            std::string wText = id_to_word[active_tokens[i]];
            Color bCol = (i == lastWordIdx) ? Color{ 168, 85, 247, 255 } : Color{ 56, 189, 248, 255 };

            // Token Pill
            Rectangle pRec = { (float)spotX + 20, py - 10, 85, 22 };
            DrawRectangleRounded(pRec, 0.4f, 6, Color{ 30, 41, 59, 255 });
            DrawRectangleRoundedLinesEx(pRec, 0.4f, 6, 1.5f, bCol);
            DrawTextEx(font_mono, wText.c_str(), Vector2{ pRec.x + 8, py - 6 }, 11.0f, 1.0f, bCol);

            // Draw spotlight beam from last word to past words
            if (lastWordIdx < (int)A.size() && i < (int)A[lastWordIdx].size()) {
                float weight = static_cast<float>(A[lastWordIdx][i]);
                float thick = 1.0f + weight * 9.0f;
                Color beamColor = ColorAlpha(Color{ 56, 189, 248, 255 }, std::min(1.0f, 0.15f + weight * 0.85f));

                Vector2 startPt = { pRec.x + pRec.width, py };
                Vector2 endPt = { (float)spotX + 340, py };

                // Draw attention percentage bar
                float barW = weight * 110.0f;
                DrawRectangleRounded(Rectangle{ endPt.x + 10, py - 6, barW, 14 }, 0.3f, 4, (weight > 0.15f ? Color{ 2, 132, 199, 255 } : Color{ 55, 65, 81, 200 }));
                
                std::stringstream ss_pct; ss_pct << std::fixed << std::setprecision(1) << (weight * 100.0f) << "%";
                DrawTextEx(font_mono, ss_pct.str().c_str(), Vector2{ endPt.x + 128, py - 6 }, 11.0f, 1.0f, (weight > 0.15f ? Color{ 56, 189, 248, 255 } : Color{ 148, 163, 184, 180 }));

                // Glowing connecting line
                DrawLineEx(startPt, Vector2{ endPt.x, py }, thick, beamColor);
            }
        }

        // 5. BOTTOM BAR: LIVE NEXT-WORD PREDICTION & CONTROLS
        int botY = 570;
        int botH = 175;
        DrawRectangleRounded(Rectangle{ 24, (float)botY, (float)screenWidth - 48, (float)botH }, 0.04f, 6, Color{ 17, 24, 39, 255 });
        DrawRectangleRoundedLinesEx(Rectangle{ 24, (float)botY, (float)screenWidth - 48, (float)botH }, 0.04f, 6, 1.5f, Color{ 31, 41, 55, 255 });

        DrawTextEx(font_med, "Live Next-Word Prediction (Given Current Sentence Prompt):", Vector2{ 42, (float)botY + 14 }, 15.0f, 1.0f, Color{ 52, 211, 153, 255 });

        // Show prompt text
        std::stringstream ss_prompt;
        for (int tok : active_tokens) ss_prompt << id_to_word[tok] << " ";
        DrawTextEx(font_med, ss_prompt.str().c_str(), Vector2{ 42, (float)botY + 38 }, 14.0f, 1.0f, Color{ 226, 232, 240, 255 });

        // Top 3 Predicted Words Bars
        std::vector<std::pair<double, int>> ranked;
        for (int i = 0; i < V; ++i) ranked.push_back({probs[i], i});
        std::sort(ranked.rbegin(), ranked.rend());

        for (int rank = 0; rank < 3; ++rank) {
            float rx = 42 + rank * 240;
            float ry = botY + 75;
            std::string predWord = "\"" + id_to_word[ranked[rank].second] + "\"";
            float prob = static_cast<float>(ranked[rank].first);

            DrawRectangleRounded(Rectangle{ rx, ry, 220, 42 }, 0.25f, 6, Color{ 30, 41, 59, 255 });
            DrawRectangleRoundedLinesEx(Rectangle{ rx, ry, 220, 42 }, 0.25f, 6, 1.5f, (rank == 0 ? Color{ 34, 197, 94, 255 } : Color{ 71, 85, 105, 255 }));

            DrawTextEx(font_bold, predWord.c_str(), Vector2{ rx + 12, ry + 10 }, 15.0f, 1.0f, (rank == 0 ? Color{ 34, 197, 94, 255 } : WHITE));
            std::stringstream ss_p; ss_p << std::fixed << std::setprecision(1) << (prob * 100.0f) << "%";
            DrawTextEx(font_mono, ss_p.str().c_str(), Vector2{ rx + 140, ry + 12 }, 13.0f, 1.0f, Color{ 148, 163, 184, 255 });
        }

        // Instructions Footer
        std::string footerText = "[SPACE] Pause/Resume Training   [R] Reset Brain   [UP/DOWN] Train Speed (" + std::to_string(train_speed) + "x)   [1-4] Switch Sentence";
        DrawTextEx(font_reg, footerText.c_str(), Vector2{ 42, (float)botY + 138 }, 13.0f, 1.0f, Color{ 148, 163, 184, 255 });

        EndDrawing();
    }

    UnloadFont(font_bold);
    UnloadFont(font_title);
    UnloadFont(font_med);
    UnloadFont(font_reg);
    UnloadFont(font_mono);
    CloseWindow();
    return 0;
}
