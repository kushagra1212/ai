#include "raylib.h"
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
//  PROGRAM 31: FIRST-PRINCIPLES SEMANTIC SEARCH ENGINE (DESKTOP GUI)
//  Built from scratch using C++17 and Raylib
//  Demonstrates why Vector / Coordinate Search dominates traditional Keyword Search
// ====================================================================================

inline double sigmoid(double x) {
    if (x > 10.0) return 1.0;
    if (x < -10.0) return 0.0;
    return 1.0 / (1.0 + std::exp(-x));
}

double dot_product_vec(const std::vector<double>& a, const std::vector<double>& b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) sum += a[i] * b[i];
    return sum;
}

double vector_magnitude(const std::vector<double>& v) {
    double sum_sq = 0.0;
    for (double val : v) sum_sq += val * val;
    return std::sqrt(sum_sq);
}

double cosine_alignment(const std::vector<double>& a, const std::vector<double>& b) {
    double len_a = vector_magnitude(a);
    double len_b = vector_magnitude(b);
    if (len_a < 1e-9 || len_b < 1e-9) return 0.0;
    return dot_product_vec(a, b) / (len_a * len_b);
}

// Convert string to lowercase
std::string to_lower(const std::string& str) {
    std::string out = str;
    for (char& c : out) c = std::tolower(static_cast<unsigned char>(c));
    return out;
}

// Tokenize text into words
std::vector<std::string> split_words(const std::string& text) {
    std::vector<std::string> tokens;
    std::string clean;
    for (char ch : text) {
        if (std::isalnum(static_cast<unsigned char>(ch))) {
            clean += std::tolower(static_cast<unsigned char>(ch));
        } else {
            clean += ' ';
        }
    }
    std::stringstream ss(clean);
    std::string w;
    while (ss >> w) tokens.push_back(w);
    return tokens;
}

struct Document {
    int id;
    std::string text;
    std::string category;
    std::vector<double> vector;
};

struct SearchResult {
    int doc_id;
    std::string text;
    std::string category;
    double semantic_score;
    bool keyword_matched;
    int keyword_overlap_count;
};

int main() {
    std::cout << "====================================================================================\n";
    std::cout << " INITIALIZING SEMANTIC SEARCH ENGINE (C++17 + RAYLIB GUI)\n";
    std::cout << " Training Word Coordinates from First Principles...\n";
    std::cout << "====================================================================================\n";

    // --------------------------------------------------------------------------------
    // 1. THE DOCUMENT DATABASE (CORPUS)
    // --------------------------------------------------------------------------------
    std::vector<std::pair<std::string, std::string>> raw_docs = {
        {"the king sits on the golden throne", "Royalty"},
        {"the queen rules the royal castle", "Royalty"},
        {"the prince rules the royal palace", "Royalty"},
        {"the princess lives in the royal palace", "Royalty"},
        {"the castle has tall stone walls and towers", "Royalty"},
        {"the monkey eats a sweet apple in the sun", "Fruit & Nature"},
        {"the monkey eats a sweet banana in the tree", "Fruit & Nature"},
        {"the juicy orange is a delicious sweet fruit", "Fruit & Nature"},
        {"ripe grape clusters are a sweet purple fruit", "Fruit & Nature"},
        {"the wild wolf hunts in the deep forest", "Wild Animals"},
        {"the wild lion hunts in the deep forest", "Wild Animals"},
        {"the friendly dog runs in the green park", "Friendly Pets"},
        {"the friendly cat runs in the green park", "Friendly Pets"},
        {"a puppy dog barks and plays with children", "Friendly Pets"}
    };

    // Extract unique vocabulary
    std::vector<std::string> vocab;
    std::unordered_map<std::string, int> word_to_id;

    for (const auto& doc : raw_docs) {
        auto tokens = split_words(doc.first);
        for (const auto& w : tokens) {
            if (word_to_id.find(w) == word_to_id.end()) {
                word_to_id[w] = static_cast<int>(vocab.size());
                vocab.push_back(w);
            }
        }
    }

    int V = static_cast<int>(vocab.size());
    const int D = 8; // 8-dimensional continuous semantic space

    // Extract training context pairs (sliding window)
    std::vector<std::pair<int, int>> pairs;
    int window_size = 2;
    for (const auto& doc : raw_docs) {
        auto tokens = split_words(doc.first);
        std::vector<int> ids;
        for (const auto& w : tokens) ids.push_back(word_to_id[w]);
        for (int i = 0; i < static_cast<int>(ids.size()); ++i) {
            for (int j = std::max(0, i - window_size); j <= std::min(static_cast<int>(ids.size()) - 1, i + window_size); ++j) {
                if (i != j) pairs.push_back({ids[i], ids[j]});
            }
        }
    }

    // Train word coordinates (First principles: pull neighbors, push strangers)
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> init_dist(-0.1, 0.1);
    std::vector<std::vector<double>> U(V, std::vector<double>(D));
    std::vector<std::vector<double>> W(V, std::vector<double>(D));
    for (int i = 0; i < V; ++i) {
        for (int d = 0; d < D; ++d) {
            U[i][d] = init_dist(rng);
            W[i][d] = init_dist(rng);
        }
    }

    double lr = 0.05;
    int epochs = 1800;
    std::uniform_int_distribution<int> rand_word(0, V - 1);

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        for (const auto& p : pairs) {
            int target = p.first;
            int context = p.second;

            // Positive neighbor
            double s_pos = dot_product_vec(U[target], W[context]);
            double prob_pos = sigmoid(s_pos);
            double err_pos = 1.0 - prob_pos;

            // Negative stranger
            int neg = rand_word(rng);
            while (neg == context) neg = rand_word(rng);
            double s_neg = dot_product_vec(U[target], W[neg]);
            double prob_neg = sigmoid(s_neg);
            double err_neg = 0.0 - prob_neg;

            for (int d = 0; d < D; ++d) {
                double u_val = U[target][d];
                double w_ctx = W[context][d];
                double w_neg = W[neg][d];

                U[target][d] += lr * (err_pos * w_ctx + err_neg * w_neg);
                W[context][d] += lr * (err_pos * u_val);
                W[neg][d]     += lr * (err_neg * u_val);
            }
        }
    }

    // Combined word coordinates
    std::vector<std::vector<double>> word_coords(V, std::vector<double>(D));
    for (int i = 0; i < V; ++i) {
        for (int d = 0; d < D; ++d) {
            word_coords[i][d] = (U[i][d] + W[i][d]) / 2.0;
        }
    }
    std::cout << ">>> Word Coordinates Trained! Vocabulary: " << V << " words across " << D << " dimensions.\n";

    // --------------------------------------------------------------------------------
    // 2. BUILD DOCUMENT VECTORS (AVERAGE OF WORD COORDINATES)
    // --------------------------------------------------------------------------------
    std::vector<Document> documents;
    for (size_t doc_idx = 0; doc_idx < raw_docs.size(); ++doc_idx) {
        Document doc;
        doc.id = static_cast<int>(doc_idx);
        doc.text = raw_docs[doc_idx].first;
        doc.category = raw_docs[doc_idx].second;
        doc.vector.assign(D, 0.0);

        auto tokens = split_words(doc.text);
        int valid_tokens = 0;
        for (const auto& t : tokens) {
            if (word_to_id.find(t) != word_to_id.end()) {
                int wid = word_to_id[t];
                for (int d = 0; d < D; ++d) doc.vector[d] += word_coords[wid][d];
                valid_tokens++;
            }
        }
        if (valid_tokens > 0) {
            for (int d = 0; d < D; ++d) doc.vector[d] /= valid_tokens;
        }
        documents.push_back(doc);
    }
    std::cout << ">>> Indexed " << documents.size() << " documents into vector database.\n\n";

    // Function to compute Query Vector from arbitrary text
    auto compute_query_vector = [&](const std::string& query_str) -> std::vector<double> {
        std::vector<double> q_vec(D, 0.0);
        auto tokens = split_words(query_str);
        int valid_tokens = 0;
        for (const auto& t : tokens) {
            if (word_to_id.find(t) != word_to_id.end()) {
                int wid = word_to_id[t];
                for (int d = 0; d < D; ++d) q_vec[d] += word_coords[wid][d];
                valid_tokens++;
            }
        }
        if (valid_tokens > 0) {
            for (int d = 0; d < D; ++d) q_vec[d] /= valid_tokens;
        }
        return q_vec;
    };

    // --------------------------------------------------------------------------------
    // 3. RAYLIB INTERACTIVE DESKTOP WINDOW
    // --------------------------------------------------------------------------------
    const int screen_width = 1320;
    const int screen_height = 800;
    SetConfigFlags(FLAG_MSAA_4X_HINT | FLAG_WINDOW_RESIZABLE);
    InitWindow(screen_width, screen_height, "Program 31: First-Principles Semantic Search Engine (Vector vs Keyword)");

    // Load crisp TrueType fonts rasterized at EXACT target pixel sizes (No downscaling artifacts!)
    Font font_large = LoadFontEx("/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf", 24, 0, 0);
    Font font_med   = LoadFontEx("/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf", 18, 0, 0);
    Font font_reg   = LoadFontEx("/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf", 15, 0, 0);
    Font font_small = LoadFontEx("/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf", 13, 0, 0);

    SetTextureFilter(font_large.texture, TEXTURE_FILTER_BILINEAR);
    SetTextureFilter(font_med.texture,   TEXTURE_FILTER_BILINEAR);
    SetTextureFilter(font_reg.texture,   TEXTURE_FILTER_BILINEAR);
    SetTextureFilter(font_small.texture, TEXTURE_FILTER_BILINEAR);

    SetTargetFPS(60);

    std::string search_query = "king throne"; // Initial default query
    int cursor_blink_counter = 0;

    // Preset query buttons for quick exploration
    std::vector<std::string> presets = {
        "king throne",
        "sweet fruit",
        "wild wolf",
        "friendly dog",
        "royal palace",
        "purple grape"
    };

    while (!WindowShouldClose()) {
        cursor_blink_counter++;

        // ----------------------------------------------------------------------------
        // KEYBOARD INPUT HANDLING FOR SEARCH BAR
        // ----------------------------------------------------------------------------
        int key = GetCharPressed();
        while (key > 0) {
            if ((key >= 32) && (key <= 125) && (search_query.length() < 40)) {
                search_query += static_cast<char>(key);
            }
            key = GetCharPressed();
        }

        if (IsKeyPressed(KEY_BACKSPACE)) {
            if (!search_query.empty()) {
                search_query.pop_back();
            }
        }

        // ----------------------------------------------------------------------------
        // COMPUTE LIVE SEARCH RESULTS
        // ----------------------------------------------------------------------------
        std::vector<double> query_vec = compute_query_vector(search_query);
        auto query_tokens = split_words(search_query);

        std::vector<SearchResult> results;
        for (const auto& doc : documents) {
            SearchResult res;
            res.doc_id = doc.id;
            res.text = doc.text;
            res.category = doc.category;
            res.semantic_score = cosine_alignment(query_vec, doc.vector);

            // Check traditional keyword substring overlap
            res.keyword_overlap_count = 0;
            auto doc_tokens = split_words(doc.text);
            for (const auto& qt : query_tokens) {
                for (const auto& dt : doc_tokens) {
                    if (qt == dt) {
                        res.keyword_overlap_count++;
                        break;
                    }
                }
            }
            res.keyword_matched = (res.keyword_overlap_count > 0);
            results.push_back(res);
        }

        // Sort by Semantic Cosine Similarity (Highest match first)
        std::sort(results.begin(), results.end(), [](const SearchResult& a, const SearchResult& b) {
            return a.semantic_score > b.semantic_score;
        });

        // ----------------------------------------------------------------------------
        // RENDERING
        // ----------------------------------------------------------------------------
        BeginDrawing();
        ClearBackground(Color{ 13, 17, 26, 255 }); // Dark sleek background

        // 1. Top Header Banner
        DrawRectangle(0, 0, GetScreenWidth(), 74, Color{ 19, 25, 38, 255 });
        DrawLine(0, 74, GetScreenWidth(), 74, Color{ 45, 55, 80, 255 });

        DrawTextEx(font_large, "FIRST-PRINCIPLES SEMANTIC SEARCH ENGINE", Vector2{ 30, 14 }, 24.0f, 1.0f, WHITE);
        DrawTextEx(font_reg, "Comparing Continuous Vector Search (Cosine Similarity) vs Traditional Keyword Matching", Vector2{ 30, 44 }, 15.0f, 1.0f, Color{ 148, 163, 184, 255 });

        // Database status badge
        int badge_x = GetScreenWidth() - 260;
        DrawRectangleRounded(Rectangle{ static_cast<float>(badge_x), 18.0f, 230.0f, 38.0f }, 0.4f, 6, Color{ 28, 36, 54, 255 });
        DrawRectangleRoundedLinesEx(Rectangle{ static_cast<float>(badge_x), 18.0f, 230.0f, 38.0f }, 0.4f, 6, 1.0f, Color{ 52, 211, 153, 200 });
        DrawCircle(badge_x + 18, 37, 5, Color{ 52, 211, 153, 255 });
        DrawTextEx(font_small, "Vector DB Active: 14 Docs", Vector2{ static_cast<float>(badge_x + 32), 29.0f }, 13.0f, 1.0f, WHITE);

        // ----------------------------------------------------------------------------
        // 2. INTERACTIVE SEARCH INPUT BAR
        // ----------------------------------------------------------------------------
        int bar_x = 30;
        int bar_y = 94;
        int bar_w = GetScreenWidth() - 60;
        int bar_h = 56;
        Rectangle search_bar_rect = { static_cast<float>(bar_x), static_cast<float>(bar_y), static_cast<float>(bar_w), static_cast<float>(bar_h) };

        DrawRectangleRounded(search_bar_rect, 0.25f, 6, Color{ 22, 28, 44, 255 });
        DrawRectangleRoundedLinesEx(search_bar_rect, 0.25f, 6, 2.0f, Color{ 96, 165, 250, 255 }); // Glowing blue border

        // Search icon magnifier label
        DrawTextEx(font_med, "SEARCH >", Vector2{ static_cast<float>(bar_x + 20), static_cast<float>(bar_y + 18) }, 18.0f, 1.0f, Color{ 96, 165, 250, 255 });

        // Display current typed text
        std::string display_query = search_query;
        if ((cursor_blink_counter / 30) % 2 == 0) {
            display_query += "|";
        }
        DrawTextEx(font_large, display_query.c_str(), Vector2{ static_cast<float>(bar_x + 130), static_cast<float>(bar_y + 15) }, 22.0f, 1.0f, WHITE);

        // ----------------------------------------------------------------------------
        // 3. PRESET QUICK-CLICK BUTTONS
        // ----------------------------------------------------------------------------
        int preset_y = 164;
        DrawTextEx(font_reg, "Quick Test Queries:", Vector2{ static_cast<float>(bar_x), static_cast<float>(preset_y + 6) }, 15.0f, 1.0f, Color{ 148, 163, 184, 255 });

        int cur_btn_x = bar_x + 160;
        for (const auto& p : presets) {
            Vector2 p_dim = MeasureTextEx(font_small, p.c_str(), 13.0f, 1.0f);
            Rectangle btn_rect = { static_cast<float>(cur_btn_x), static_cast<float>(preset_y), p_dim.x + 26.0f, 32.0f };

            bool is_hover = CheckCollisionPointRec(GetMousePosition(), btn_rect);
            if (is_hover && IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
                search_query = p;
            }

            Color btn_bg = (search_query == p) ? Color{ 37, 99, 235, 255 } : (is_hover ? Color{ 40, 52, 80, 255 } : Color{ 25, 33, 50, 255 });
            DrawRectangleRounded(btn_rect, 0.4f, 6, btn_bg);
            DrawRectangleRoundedLinesEx(btn_rect, 0.4f, 6, 1.0f, Color{ 70, 85, 120, 255 });
            DrawTextEx(font_small, p.c_str(), Vector2{ btn_rect.x + 13.0f, btn_rect.y + 9.0f }, 13.0f, 1.0f, WHITE);

            cur_btn_x += static_cast<int>(btn_rect.width + 12.0f);
        }

        // ----------------------------------------------------------------------------
        // 4. RANKED RESULTS LIST (LEFT 65% OF SCREEN)
        // ----------------------------------------------------------------------------
        int feed_x = 30;
        int feed_y = 214;
        int feed_w = 820;

        DrawTextEx(font_med, "RANKED SEARCH RESULTS (SORTED BY MEANING ALIGNMENT):", Vector2{ static_cast<float>(feed_x), static_cast<float>(feed_y) }, 17.0f, 1.0f, Color{ 255, 215, 0, 255 });

        int item_y = feed_y + 30;
        for (int k = 0; k < 6 && k < static_cast<int>(results.size()); ++k) {
            const auto& r = results[k];

            Rectangle card_rect = { static_cast<float>(feed_x), static_cast<float>(item_y), static_cast<float>(feed_w), 80.0f };

            // Card background
            Color card_bg = (k == 0) ? Color{ 24, 33, 54, 255 } : Color{ 17, 22, 35, 255 };
            DrawRectangleRounded(card_rect, 0.16f, 6, card_bg);

            Color border_col = (k == 0) ? Color{ 255, 215, 0, 220 } : Color{ 42, 53, 78, 220 };
            DrawRectangleRoundedLinesEx(card_rect, 0.16f, 6, (k == 0 ? 2.0f : 1.0f), border_col);

            // Rank badge (#1, #2...)
            Color rank_col = (k == 0) ? Color{ 255, 215, 0, 255 } : ((k == 1) ? Color{ 226, 232, 240, 255 } : Color{ 148, 163, 184, 255 });
            std::string rank_str = "#" + std::to_string(k + 1);
            DrawTextEx(font_large, rank_str.c_str(), Vector2{ card_rect.x + 18.0f, card_rect.y + 26.0f }, 24.0f, 1.0f, rank_col);

            // Document snippet text
            DrawTextEx(font_med, r.text.c_str(), Vector2{ card_rect.x + 74.0f, card_rect.y + 16.0f }, 18.0f, 1.0f, WHITE);

            // Category tag
            DrawTextEx(font_reg, ("Category: " + r.category).c_str(), Vector2{ card_rect.x + 74.0f, card_rect.y + 46.0f }, 14.0f, 1.0f, Color{ 148, 163, 184, 255 });

            // Semantic Cosine Score Badge & Bar
            int score_x = feed_x + feed_w - 235;
            std::stringstream score_ss;
            score_ss << std::fixed << std::setprecision(1) << (r.semantic_score * 100.0) << "% Semantic Match";
            Color score_col = (r.semantic_score > 0.80) ? Color{ 52, 211, 153, 255 } : ((r.semantic_score > 0.50) ? Color{ 250, 204, 21, 255 } : Color{ 248, 113, 113, 255 });
            DrawTextEx(font_med, score_ss.str().c_str(), Vector2{ static_cast<float>(score_x), card_rect.y + 14.0f }, 17.0f, 1.0f, score_col);

            // Mini visual progress bar
            DrawRectangleRounded(Rectangle{ static_cast<float>(score_x), card_rect.y + 38.0f, 215.0f, 7.0f }, 0.5f, 4, Color{ 30, 41, 59, 255 });
            float bar_fill = 215.0f * static_cast<float>(std::max(0.0, r.semantic_score));
            DrawRectangleRounded(Rectangle{ static_cast<float>(score_x), card_rect.y + 38.0f, bar_fill, 7.0f }, 0.5f, 4, score_col);

            // Keyword match pill badge
            Rectangle kw_rect = { static_cast<float>(score_x), card_rect.y + 51.0f, 130.0f, 21.0f };
            if (r.keyword_matched) {
                DrawRectangleRounded(kw_rect, 0.4f, 4, Color{ 16, 45, 35, 255 });
                DrawRectangleRoundedLinesEx(kw_rect, 0.4f, 4, 1.0f, Color{ 52, 211, 153, 200 });
                DrawTextEx(font_small, "KW: MATCHED", Vector2{ kw_rect.x + 12.0f, kw_rect.y + 4.0f }, 13.0f, 1.0f, Color{ 52, 211, 153, 255 });
            } else {
                DrawRectangleRounded(kw_rect, 0.4f, 4, Color{ 48, 20, 26, 255 });
                DrawRectangleRoundedLinesEx(kw_rect, 0.4f, 4, 1.0f, Color{ 248, 113, 113, 200 });
                DrawTextEx(font_small, "KW: 0 MATCHES", Vector2{ kw_rect.x + 10.0f, kw_rect.y + 4.0f }, 13.0f, 1.0f, Color{ 248, 113, 113, 255 });
            }

            item_y += 88;
        }

        // ----------------------------------------------------------------------------
        // 5. VECTOR MATHEMATICS INSPECTOR (RIGHT 30% OF SCREEN)
        // ----------------------------------------------------------------------------
        int info_x = feed_x + feed_w + 25;
        int info_y = feed_y;
        int info_w = GetScreenWidth() - info_x - 30;
        int info_h = 540;
        Rectangle info_rect = { static_cast<float>(info_x), static_cast<float>(info_y), static_cast<float>(info_w), static_cast<float>(info_h) };

        DrawRectangleRounded(info_rect, 0.08f, 6, Color{ 19, 25, 38, 255 });
        DrawRectangleRoundedLinesEx(info_rect, 0.08f, 6, 1.5f, Color{ 55, 68, 96, 255 });

        DrawTextEx(font_med, "VECTOR MATH ENGINE BREAKDOWN", Vector2{ static_cast<float>(info_x + 18), static_cast<float>(info_y + 18) }, 16.0f, 1.0f, Color{ 96, 165, 250, 255 });
        DrawLine(info_x + 18, info_y + 42, info_x + info_w - 18, info_y + 42, Color{ 45, 55, 80, 255 });

        // Query Vector display
        DrawTextEx(font_med, "1. CURRENT QUERY VECTOR:", Vector2{ static_cast<float>(info_x + 18), static_cast<float>(info_y + 56) }, 15.0f, 1.0f, WHITE);
        std::stringstream qv_ss;
        qv_ss << std::fixed << std::setprecision(2) << "[ ";
        for (int d = 0; d < std::min(4, D); ++d) qv_ss << query_vec[d] << " ";
        qv_ss << "... ] (8-D)";
        DrawTextEx(font_reg, qv_ss.str().c_str(), Vector2{ static_cast<float>(info_x + 18), static_cast<float>(info_y + 80) }, 15.0f, 1.0f, Color{ 52, 211, 153, 255 });

        // Top match vector display
        if (!results.empty()) {
            DrawTextEx(font_med, "2. TOP MATCH DOCUMENT VECTOR:", Vector2{ static_cast<float>(info_x + 18), static_cast<float>(info_y + 116) }, 15.0f, 1.0f, WHITE);
            std::stringstream dv_ss;
            dv_ss << std::fixed << std::setprecision(2) << "[ ";
            int top_id = results[0].doc_id;
            for (int d = 0; d < std::min(4, D); ++d) dv_ss << documents[top_id].vector[d] << " ";
            dv_ss << "... ] (8-D)";
            DrawTextEx(font_reg, dv_ss.str().c_str(), Vector2{ static_cast<float>(info_x + 18), static_cast<float>(info_y + 140) }, 15.0f, 1.0f, Color{ 255, 215, 0, 255 });

            // Cosine Similarity explanation
            DrawLine(info_x + 18, info_y + 174, info_x + info_w - 18, info_y + 174, Color{ 45, 55, 80, 255 });
            DrawTextEx(font_med, "3. WHY SEMANTIC SEARCH WINS:", Vector2{ static_cast<float>(info_x + 18), static_cast<float>(info_y + 190) }, 15.0f, 1.0f, Color{ 250, 204, 21, 255 });

            std::string expl1 = "- Keyword search (grep / Ctrl+F) fails";
            std::string expl2 = "  if you use synonyms or different words.";
            std::string expl3 = "- Vector search maps concepts to coordinates.";
            std::string expl4 = "- Dot product measures angle alignment";
            std::string expl5 = "  between Query Vector and Doc Vector!";
            std::string expl6 = "- Result: Matches by MEANING, not spelling!";

            DrawTextEx(font_reg, expl1.c_str(), Vector2{ static_cast<float>(info_x + 18), static_cast<float>(info_y + 220) }, 15.0f, 1.0f, Color{ 203, 213, 225, 255 });
            DrawTextEx(font_reg, expl2.c_str(), Vector2{ static_cast<float>(info_x + 18), static_cast<float>(info_y + 242) }, 15.0f, 1.0f, Color{ 203, 213, 225, 255 });
            DrawTextEx(font_reg, expl3.c_str(), Vector2{ static_cast<float>(info_x + 18), static_cast<float>(info_y + 274) }, 15.0f, 1.0f, Color{ 52, 211, 153, 255 });
            DrawTextEx(font_reg, expl4.c_str(), Vector2{ static_cast<float>(info_x + 18), static_cast<float>(info_y + 296) }, 15.0f, 1.0f, Color{ 52, 211, 153, 255 });
            DrawTextEx(font_reg, expl5.c_str(), Vector2{ static_cast<float>(info_x + 18), static_cast<float>(info_y + 318) }, 15.0f, 1.0f, Color{ 52, 211, 153, 255 });
            DrawTextEx(font_med, expl6.c_str(), Vector2{ static_cast<float>(info_x + 18), static_cast<float>(info_y + 350) }, 15.0f, 1.0f, Color{ 96, 165, 250, 255 });

            // Industry note
            DrawRectangle(info_x + 18, info_y + 390, info_w - 36, 120, Color{ 14, 18, 28, 255 });
            DrawRectangleLines(info_x + 18, info_y + 390, info_w - 36, 120, Color{ 40, 50, 75, 255 });
            DrawTextEx(font_med, "INDUSTRY USAGE:", Vector2{ static_cast<float>(info_x + 28), static_cast<float>(info_y + 402) }, 13.0f, 1.0f, Color{ 255, 215, 0, 255 });
            DrawTextEx(font_reg, "This vector calculation is the exact core of:", Vector2{ static_cast<float>(info_x + 28), static_cast<float>(info_y + 424) }, 13.0f, 1.0f, WHITE);
            DrawTextEx(font_reg, "* Modern Vector Databases (Pinecone, Chroma)", Vector2{ static_cast<float>(info_x + 28), static_cast<float>(info_y + 446) }, 13.0f, 1.0f, Color{ 148, 163, 184, 255 });
            DrawTextEx(font_reg, "* RAG (Retrieval-Augmented Generation)", Vector2{ static_cast<float>(info_x + 28), static_cast<float>(info_y + 468) }, 13.0f, 1.0f, Color{ 148, 163, 184, 255 });
            DrawTextEx(font_reg, "* Google Search Semantic Ranking", Vector2{ static_cast<float>(info_x + 28), static_cast<float>(info_y + 490) }, 13.0f, 1.0f, Color{ 148, 163, 184, 255 });
        }

        // ----------------------------------------------------------------------------
        // 6. BOTTOM STATUS / INSTRUCTIONS BAR
        // ----------------------------------------------------------------------------
        int b_y = GetScreenHeight() - 44;
        DrawRectangle(0, b_y, GetScreenWidth(), 44, Color{ 14, 18, 28, 255 });
        DrawLine(0, b_y, GetScreenWidth(), b_y, Color{ 40, 50, 75, 255 });
        DrawTextEx(font_reg, "INSTRUCTIONS: Type directly with keyboard to search in real time | Use Backspace to edit | Click preset buttons above",
                   Vector2{ 30, static_cast<float>(b_y + 13) }, 14.0f, 1.0f, Color{ 203, 213, 225, 255 });

        EndDrawing();
    }

    UnloadFont(font_large);
    UnloadFont(font_med);
    UnloadFont(font_reg);
    UnloadFont(font_small);
    CloseWindow();
    return 0;
}
