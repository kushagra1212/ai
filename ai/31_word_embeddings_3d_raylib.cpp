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
//  PROGRAM 31: 3D INTERACTIVE WORD EMBEDDINGS VISUALIZER (RAYLIB + C++17)
//  Explore Word Coordinates in 3D (X, Y, Z) with Mouse Drag & High-Res Graphics
// ====================================================================================

inline double sigmoid(double x) {
    if (x > 10.0) return 1.0;
    if (x < -10.0) return 0.0;
    return 1.0 / (1.0 + std::exp(-x));
}

double dot_product_3d(const std::vector<double>& a, const std::vector<double>& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

double vector_length_3d(const std::vector<double>& v) {
    return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

double direction_alignment_3d(const std::vector<double>& a, const std::vector<double>& b) {
    double len_a = vector_length_3d(a);
    double len_b = vector_length_3d(b);
    if (len_a < 1e-9 || len_b < 1e-9) return 0.0;
    return dot_product_3d(a, b) / (len_a * len_b);
}

// Word cluster category for thematic coloring
enum WordCategory {
    CAT_ROYALTY,
    CAT_FRUIT,
    CAT_WILD,
    CAT_PETS,
    CAT_OTHER
};

Color get_category_color(WordCategory cat) {
    switch (cat) {
        case CAT_ROYALTY: return Color{ 255, 215, 0, 255 };    // Lustrous Gold
        case CAT_FRUIT:   return Color{ 255, 82, 82, 255 };     // Coral Crimson
        case CAT_WILD:    return Color{ 168, 85, 247, 255 };    // Vivid Electric Purple
        case CAT_PETS:    return Color{ 56, 189, 248, 255 };    // Bright Cyan Sky
        default:          return Color{ 156, 163, 175, 255 };   // Cool Slate
    }
}

WordCategory classify_word(const std::string& w) {
    if (w == "king" || w == "queen" || w == "prince" || w == "princess" ||
        w == "castle" || w == "palace" || w == "throne" || w == "rules" ||
        w == "royal" || w == "golden" || w == "sits") {
        return CAT_ROYALTY;
    }
    if (w == "apple" || w == "banana" || w == "orange" || w == "grape" ||
        w == "sweet" || w == "juicy" || w == "fruit" || w == "eats" || w == "monkey") {
        return CAT_FRUIT;
    }
    if (w == "wolf" || w == "lion" || w == "wild" || w == "hunts" ||
        w == "forest" || w == "deep") {
        return CAT_WILD;
    }
    if (w == "dog" || w == "cat" || w == "friendly" || w == "park" ||
        w == "green" || w == "runs") {
        return CAT_PETS;
    }
    return CAT_OTHER;
}

int main() {
    std::cout << "====================================================================================\n";
    std::cout << " TRAINING 3D WORD EMBEDDINGS (X, Y, Z) FOR RAYLIB VISUALIZATION...\n";
    std::cout << "====================================================================================\n";

    // 1. Corpus of sentences
    std::vector<std::string> sentences = {
        "the king rules the royal castle",
        "the queen rules the royal castle",
        "the prince rules the royal palace",
        "the princess rules the royal palace",
        "the king sits on the golden throne",
        "the queen sits on the golden throne",
        "the monkey eats a sweet apple",
        "the monkey eats a sweet banana",
        "the monkey eats a juicy orange",
        "the monkey eats a juicy grape",
        "apple is a sweet fruit",
        "banana is a sweet fruit",
        "orange is a juicy fruit",
        "grape is a juicy fruit",
        "the wild wolf hunts in the deep forest",
        "the wild lion hunts in the deep forest",
        "the friendly dog runs in the green park",
        "the friendly cat runs in the green park"
    };

    std::vector<std::string> vocab;
    std::unordered_map<std::string, int> word_to_id;
    for (const auto& s : sentences) {
        std::stringstream ss(s);
        std::string w;
        while (ss >> w) {
            if (word_to_id.find(w) == word_to_id.end()) {
                word_to_id[w] = static_cast<int>(vocab.size());
                vocab.push_back(w);
            }
        }
    }
    int V = static_cast<int>(vocab.size());
    const int D = 3; // 3D Coordinates: (X, Y, Z)

    // 2. Extract neighbor pairs
    int window_size = 2;
    std::vector<std::pair<int, int>> pairs;
    for (const auto& s : sentences) {
        std::stringstream ss(s);
        std::string w;
        std::vector<int> tokens;
        while (ss >> w) tokens.push_back(word_to_id[w]);
        for (int i = 0; i < static_cast<int>(tokens.size()); ++i) {
            for (int j = std::max(0, i - window_size); j <= std::min(static_cast<int>(tokens.size()) - 1, i + window_size); ++j) {
                if (i != j) pairs.push_back({tokens[i], tokens[j]});
            }
        }
    }

    // 3. Initialize 3D coordinates
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

    // 4. Train coordinates using pull/push dynamics
    double learning_rate = 0.05;
    int epochs = 2000;
    std::uniform_int_distribution<int> rand_word(0, V - 1);

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        for (const auto& p : pairs) {
            int target = p.first;
            int context = p.second;

            // Positive pair
            double s_pos = dot_product_3d(U[target], W[context]);
            double prob_pos = sigmoid(s_pos);
            double err_pos = 1.0 - prob_pos;

            // Negative stranger
            int neg = rand_word(rng);
            while (neg == context) neg = rand_word(rng);
            double s_neg = dot_product_3d(U[target], W[neg]);
            double prob_neg = sigmoid(s_neg);
            double err_neg = 0.0 - prob_neg;

            for (int d = 0; d < D; ++d) {
                double u_val = U[target][d];
                double w_ctx = W[context][d];
                double w_neg = W[neg][d];

                U[target][d] += learning_rate * (err_pos * w_ctx + err_neg * w_neg);
                W[context][d] += learning_rate * (err_pos * u_val);
                W[neg][d]     += learning_rate * (err_neg * u_val);
            }
        }
    }
    std::cout << ">>> 3D Coordinates Trained Successfully!\n";

    // 5. Combine and scale coordinates for 3D rendering
    std::vector<std::vector<double>> coords(V, std::vector<double>(D));
    double max_mag = 0.0;
    for (int i = 0; i < V; ++i) {
        for (int d = 0; d < D; ++d) {
            coords[i][d] = (U[i][d] + W[i][d]) / 2.0;
        }
        max_mag = std::max(max_mag, vector_length_3d(coords[i]));
    }

    float world_scale = 15.0f / static_cast<float>(max_mag + 1e-6);
    std::vector<Vector3> pos3d(V);
    for (int i = 0; i < V; ++i) {
        pos3d[i] = Vector3{
            static_cast<float>(coords[i][0]) * world_scale,
            static_cast<float>(coords[i][1]) * world_scale,
            static_cast<float>(coords[i][2]) * world_scale
        };
    }

    // --------------------------------------------------------------------------------
    // RAYLIB 3D WINDOW & INTERACTION SETUP
    // --------------------------------------------------------------------------------
    const int screen_width = 1366;
    const int screen_height = 820;
    SetConfigFlags(FLAG_MSAA_4X_HINT | FLAG_WINDOW_RESIZABLE);
    InitWindow(screen_width, screen_height, "Program 31: 3D Interactive Word Embeddings (First Principles AI)");

    // Load High-Res TrueType font for crisp typography
    Font font_bold = LoadFontEx("/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf", 40, 0, 0);
    Font font_reg  = LoadFontEx("/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf", 36, 0, 0);
    SetTextureFilter(font_bold.texture, TEXTURE_FILTER_BILINEAR);
    SetTextureFilter(font_reg.texture,  TEXTURE_FILTER_BILINEAR);

    // Spherical coordinates for smooth mouse orbiting
    float cam_yaw = 0.85f;        // Horizontal angle
    float cam_pitch = 0.40f;      // Vertical angle
    float cam_distance = 34.0f;   // Zoom distance
    Vector3 cam_target = { 0.0f, 0.0f, 0.0f };

    Camera3D camera;
    camera.up = Vector3{ 0.0f, 1.0f, 0.0f };
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    SetTargetFPS(60);

    int selected_word = word_to_id["king"]; // Default selected
    int hovered_word = -1;
    bool auto_rotate = false;              // RULE: NO auto-move by default!

    Vector2 mouse_drag_start = { 0.0f, 0.0f };
    bool is_dragging = false;

    while (!WindowShouldClose()) {
        // Toggle auto-rotation with SPACE (default is OFF)
        if (IsKeyPressed(KEY_SPACE)) {
            auto_rotate = !auto_rotate;
        }

        // Reset camera with 'R'
        if (IsKeyPressed(KEY_R)) {
            cam_yaw = 0.85f;
            cam_pitch = 0.40f;
            cam_distance = 34.0f;
            cam_target = Vector3{ 0.0f, 0.0f, 0.0f };
            auto_rotate = false;
        }

        // ----------------------------------------------------------------------------
        // SMOOTH MOUSE ORBIT & EXPLORATION
        // ----------------------------------------------------------------------------
        Vector2 mouse_pos = GetMousePosition();
        Vector2 mouse_delta = GetMouseDelta();

        // 1. Mouse Drag Orbit (Left Mouse Button)
        if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
            mouse_drag_start = mouse_pos;
            is_dragging = false;
        }
        if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
            float dist_moved = Vector2Distance(mouse_pos, mouse_drag_start);
            if (dist_moved > 4.0f) {
                is_dragging = true;
                cam_yaw -= mouse_delta.x * 0.0055f;
                cam_pitch += mouse_delta.y * 0.0055f;
                // Clamp pitch so camera never flips upside down
                cam_pitch = std::clamp(cam_pitch, -1.45f, 1.45f);
            }
        }

        // 2. Mouse Drag Pan (Right Mouse Button or Middle Mouse Button)
        if (IsMouseButtonDown(MOUSE_BUTTON_RIGHT) || IsMouseButtonDown(MOUSE_BUTTON_MIDDLE)) {
            Vector3 forward = Vector3Normalize(Vector3Subtract(cam_target, camera.position));
            Vector3 right = Vector3Normalize(Vector3CrossProduct(forward, camera.up));
            Vector3 cam_up_real = Vector3CrossProduct(right, forward);

            float pan_speed = 0.035f * (cam_distance / 30.0f);
            cam_target = Vector3Add(cam_target, Vector3Scale(right, -mouse_delta.x * pan_speed));
            cam_target = Vector3Add(cam_target, Vector3Scale(cam_up_real, mouse_delta.y * pan_speed));
        }

        // 3. Mouse Wheel Zoom
        float wheel = GetMouseWheelMove();
        if (wheel != 0.0f) {
            cam_distance -= wheel * 2.2f;
            cam_distance = std::clamp(cam_distance, 6.0f, 85.0f);
        }

        // Optional auto-rotation only if explicitly toggled on by user
        if (auto_rotate) {
            cam_yaw += 0.0035f;
        }

        // Compute camera position in spherical coordinates
        camera.position.x = cam_target.x + cam_distance * std::cos(cam_pitch) * std::sin(cam_yaw);
        camera.position.y = cam_target.y + cam_distance * std::sin(cam_pitch);
        camera.position.z = cam_target.z + cam_distance * std::cos(cam_pitch) * std::cos(cam_yaw);
        camera.target = cam_target;

        // ----------------------------------------------------------------------------
        // RAYCAST HOVER & CLICK DETECTION
        // ----------------------------------------------------------------------------
        Ray ray = GetScreenToWorldRay(mouse_pos, camera);
        hovered_word = -1;
        float min_hit_dist = 1e9f;

        for (int i = 0; i < V; ++i) {
            RayCollision col = GetRayCollisionSphere(ray, pos3d[i], 1.15f);
            if (col.hit && col.distance < min_hit_dist) {
                min_hit_dist = col.distance;
                hovered_word = i;
            }
        }

        // Left click selection (only if not dragging camera)
        if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT) && !is_dragging && hovered_word != -1) {
            selected_word = hovered_word;
        }

        // ----------------------------------------------------------------------------
        // RENDERING
        // ----------------------------------------------------------------------------
        BeginDrawing();
        ClearBackground(Color{ 11, 14, 22, 255 }); // Dark cosmic canvas

        BeginMode3D(camera);

        // Subtle 3D Coordinate Floor Grid
        DrawGrid(30, 2.0f);

        // Origin reference axes
        DrawLine3D(Vector3{ -16, 0, 0 }, Vector3{ 16, 0, 0 }, Color{ 239, 68, 68, 80 });   // X Axis (Red)
        DrawLine3D(Vector3{ 0, -16, 0 }, Vector3{ 0, 16, 0 }, Color{ 34, 197, 94, 80 });   // Y Axis (Green)
        DrawLine3D(Vector3{ 0, 0, -16 }, Vector3{ 0, 0, 16 }, Color{ 59, 130, 246, 80 });  // Z Axis (Blue)

        // 1. Draw Volumetric Laser Beams to Top Neighbors for Selected Word
        if (selected_word != -1) {
            std::vector<std::pair<double, int>> sims;
            for (int i = 0; i < V; ++i) {
                if (i != selected_word) {
                    double s = direction_alignment_3d(coords[selected_word], coords[i]);
                    sims.push_back({s, i});
                }
            }
            std::sort(sims.rbegin(), sims.rend());

            for (int k = 0; k < 4 && k < static_cast<int>(sims.size()); ++k) {
                int n_id = sims[k].second;
                Color beam_col = Color{ 255, 220, 90, static_cast<unsigned char>(220 - k * 40) };
                float beam_radius = 0.07f - k * 0.01f;
                DrawCylinderEx(pos3d[selected_word], pos3d[n_id], beam_radius, beam_radius, 8, beam_col);
            }
        }

        // 2. Draw 3D High-Res Spheres with Aura and Glowing Specular Cores
        for (int i = 0; i < V; ++i) {
            WordCategory cat = classify_word(vocab[i]);
            Color col = get_category_color(cat);

            float r = 0.52f;
            if (i == selected_word) {
                r = 0.95f;
                // Outer selection wireframe halo
                DrawSphereWires(pos3d[i], r * 1.35f, 16, 16, WHITE);
            } else if (i == hovered_word) {
                r = 0.75f;
                DrawSphereWires(pos3d[i], r * 1.25f, 12, 12, YELLOW);
            }

            // Outer translucent glow aura
            DrawSphere(pos3d[i], r * 1.30f, ColorAlpha(col, 0.22f));

            // Solid high-res sphere
            DrawSphere(pos3d[i], r, col);

            // Inner bright specular core
            DrawSphere(pos3d[i], r * 0.38f, ColorAlpha(WHITE, 0.90f));
        }

        EndMode3D();

        // ----------------------------------------------------------------------------
        // DEPTH-SORTED CRISP 2D PILL BADGES
        // ----------------------------------------------------------------------------
        // Calculate camera distance for all words so near words draw on top of far words!
        std::vector<std::pair<float, int>> depth_order;
        for (int i = 0; i < V; ++i) {
            float dist = Vector3Distance(camera.position, pos3d[i]);
            depth_order.push_back({dist, i});
        }
        // Sort from furthest to nearest
        std::sort(depth_order.rbegin(), depth_order.rend());

        for (const auto& entry : depth_order) {
            int i = entry.second;
            float dist = entry.first;

            Vector2 s_pos = GetWorldToScreen(pos3d[i], camera);
            if (s_pos.x < 0 || s_pos.x > GetScreenWidth() || s_pos.y < 0 || s_pos.y > GetScreenHeight()) {
                continue;
            }

            WordCategory cat = classify_word(vocab[i]);
            Color theme_col = get_category_color(cat);

            // Scale badge and font cleanly with 3D depth
            float scale = std::clamp(30.0f / dist, 0.65f, 1.40f);
            if (i == selected_word) scale = std::max(scale, 1.25f);

            float font_size = 18.0f * scale;
            std::string label = vocab[i];
            Vector2 text_dim = MeasureTextEx(font_bold, label.c_str(), font_size, 1.0f);

            float pad_x = 10.0f * scale;
            float pad_y = 5.0f * scale;
            Rectangle badge_rect = {
                s_pos.x - text_dim.x / 2.0f - pad_x,
                s_pos.y - text_dim.y - pad_y * 2.0f - (15.0f * scale),
                text_dim.x + pad_x * 2.0f,
                text_dim.y + pad_y * 2.0f
            };

            // Badge Background
            Color bg_col = Color{ 14, 18, 28, static_cast<unsigned char>(i == selected_word ? 245 : 215) };
            DrawRectangleRounded(badge_rect, 0.45f, 8, bg_col);

            // Badge Border
            Color border_col = (i == selected_word) ? YELLOW : ((i == hovered_word) ? WHITE : ColorAlpha(theme_col, 0.75f));
            DrawRectangleRoundedLinesEx(badge_rect, 0.45f, 8, (i == selected_word ? 2.5f : 1.5f), border_col);

            // Crisp Typography
            Color text_col = (i == selected_word) ? YELLOW : ((i == hovered_word) ? WHITE : theme_col);
            Vector2 text_pos = { badge_rect.x + pad_x, badge_rect.y + pad_y };
            DrawTextEx(font_bold, label.c_str(), text_pos, font_size, 1.0f, text_col);
        }

        // ----------------------------------------------------------------------------
        // SLEEK HUD / INTERFACE OVERLAY
        // ----------------------------------------------------------------------------
        // 1. Top Header Banner
        DrawRectangle(0, 0, GetScreenWidth(), 68, Color{ 16, 21, 33, 235 });
        DrawLine(0, 68, GetScreenWidth(), 68, Color{ 45, 55, 80, 255 });
        DrawTextEx(font_bold, "PROGRAM 31: 3D WORD EMBEDDINGS EXPLORER", Vector2{ 26, 12 }, 23.0f, 1.0f, WHITE);
        DrawTextEx(font_reg, "Continuous Semantic Coordinates in 3D (Trained from First Principles)", Vector2{ 26, 38 }, 15.0f, 1.0f, Color{ 148, 163, 184, 255 });

        // 2. Category Legend (Top Right)
        int leg_x = GetScreenWidth() - 390;
        DrawRectangleRounded(Rectangle{ static_cast<float>(leg_x - 12), 10.0f, 380.0f, 48.0f }, 0.25f, 6, Color{ 12, 16, 25, 220 });
        DrawRectangleRoundedLinesEx(Rectangle{ static_cast<float>(leg_x - 12), 10.0f, 380.0f, 48.0f }, 0.25f, 6, 1.0f, Color{ 40, 50, 75, 255 });

        DrawCircle(leg_x + 8, 23, 6, get_category_color(CAT_ROYALTY));
        DrawTextEx(font_bold, "Royalty", Vector2{ static_cast<float>(leg_x + 20), 16.0f }, 14.0f, 1.0f, WHITE);

        DrawCircle(leg_x + 110, 23, 6, get_category_color(CAT_FRUIT));
        DrawTextEx(font_bold, "Fruit & Food", Vector2{ static_cast<float>(leg_x + 122), 16.0f }, 14.0f, 1.0f, WHITE);

        DrawCircle(leg_x + 235, 23, 6, get_category_color(CAT_WILD));
        DrawTextEx(font_bold, "Wild Animals", Vector2{ static_cast<float>(leg_x + 247), 16.0f }, 14.0f, 1.0f, WHITE);

        DrawCircle(leg_x + 8, 43, 6, get_category_color(CAT_PETS));
        DrawTextEx(font_bold, "Friendly Pets", Vector2{ static_cast<float>(leg_x + 20), 36.0f }, 14.0f, 1.0f, WHITE);

        DrawCircle(leg_x + 110, 43, 6, get_category_color(CAT_OTHER));
        DrawTextEx(font_bold, "Grammar/Verbs", Vector2{ static_cast<float>(leg_x + 122), 36.0f }, 14.0f, 1.0f, WHITE);

        // 3. Left Sidebar: Selected Word Inspector
        int panel_w = 350;
        int panel_h = 360;
        int panel_x = 24;
        int panel_y = 86;
        Rectangle panel_rect = { static_cast<float>(panel_x), static_cast<float>(panel_y), static_cast<float>(panel_w), static_cast<float>(panel_h) };
        DrawRectangleRounded(panel_rect, 0.08f, 6, Color{ 16, 21, 33, 235 });
        DrawRectangleRoundedLinesEx(panel_rect, 0.08f, 6, 1.5f, Color{ 55, 68, 96, 255 });

        if (selected_word != -1) {
            std::string sel_word = vocab[selected_word];
            WordCategory cat = classify_word(sel_word);

            DrawTextEx(font_reg, "SELECTED WORD INSPECTOR", Vector2{ static_cast<float>(panel_x + 18), static_cast<float>(panel_y + 16) }, 13.0f, 1.0f, Color{ 148, 163, 184, 255 });
            DrawTextEx(font_bold, sel_word.c_str(), Vector2{ static_cast<float>(panel_x + 18), static_cast<float>(panel_y + 34) }, 28.0f, 1.0f, get_category_color(cat));

            std::stringstream coord_ss;
            coord_ss << std::fixed << std::setprecision(2)
                     << "3D Position: (" << coords[selected_word][0] << ", "
                     << coords[selected_word][1] << ", "
                     << coords[selected_word][2] << ")";
            DrawTextEx(font_reg, coord_ss.str().c_str(), Vector2{ static_cast<float>(panel_x + 18), static_cast<float>(panel_y + 70) }, 14.0f, 1.0f, Color{ 203, 213, 225, 255 });

            DrawLine(panel_x + 18, panel_y + 94, panel_x + panel_w - 18, panel_y + 94, Color{ 45, 55, 80, 255 });
            DrawTextEx(font_bold, "CLOSEST SEMANTIC NEIGHBORS (ALIGNMENT)", Vector2{ static_cast<float>(panel_x + 18), static_cast<float>(panel_y + 106) }, 13.0f, 1.0f, Color{ 255, 215, 0, 255 });

            std::vector<std::pair<double, int>> sims;
            for (int i = 0; i < V; ++i) {
                if (i != selected_word) {
                    double s = direction_alignment_3d(coords[selected_word], coords[i]);
                    sims.push_back({s, i});
                }
            }
            std::sort(sims.rbegin(), sims.rend());

            for (int k = 0; k < 5 && k < static_cast<int>(sims.size()); ++k) {
                int n_id = sims[k].second;
                double alignment = sims[k].first;
                int bar_y = panel_y + 138 + k * 38;

                DrawTextEx(font_bold, vocab[n_id].c_str(), Vector2{ static_cast<float>(panel_x + 18), static_cast<float>(bar_y) }, 17.0f, 1.0f, WHITE);

                std::stringstream sim_ss;
                sim_ss << std::fixed << std::setprecision(1) << (alignment * 100.0) << "%";
                DrawTextEx(font_bold, sim_ss.str().c_str(), Vector2{ static_cast<float>(panel_x + panel_w - 82), static_cast<float>(bar_y) }, 16.0f, 1.0f, Color{ 52, 211, 153, 255 });

                // Sleek progress bar
                DrawRectangleRounded(Rectangle{ static_cast<float>(panel_x + 18), static_cast<float>(bar_y + 22), static_cast<float>(panel_w - 36), 5.0f }, 0.5f, 4, Color{ 30, 41, 59, 255 });
                float fill_w = (panel_w - 36) * static_cast<float>(std::max(0.0, alignment));
                DrawRectangleRounded(Rectangle{ static_cast<float>(panel_x + 18), static_cast<float>(bar_y + 22), fill_w, 5.0f }, 0.5f, 4, get_category_color(classify_word(vocab[n_id])));
            }
        }

        // 4. Bottom Controls Help Bar
        int b_y = GetScreenHeight() - 44;
        DrawRectangle(0, b_y, GetScreenWidth(), 44, Color{ 12, 16, 25, 240 });
        DrawLine(0, b_y, GetScreenWidth(), b_y, Color{ 40, 50, 75, 255 });

        std::string status_text = "CONTROLS: [Left-Drag] Orbit Camera  |  [Right/Mid-Drag] Pan  |  [Scroll] Zoom  |  [Click Word] Select  |  [R] Reset View";
        if (auto_rotate) status_text += "  |  [SPACE] Auto-Orbit: ON";
        else status_text += "  |  [SPACE] Auto-Orbit: OFF";

        DrawTextEx(font_reg, status_text.c_str(), Vector2{ 24, static_cast<float>(b_y + 13) }, 15.0f, 1.0f, Color{ 203, 213, 225, 255 });

        EndDrawing();
    }

    UnloadFont(font_bold);
    UnloadFont(font_reg);
    CloseWindow();
    return 0;
}
