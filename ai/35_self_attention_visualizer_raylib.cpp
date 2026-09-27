/**
 * ====================================================================================
 * PROGRAM 35: RAYLIB CAUSAL SELF-ATTENTION VISUALIZER (C++17)
 * ====================================================================================
 * Interactive Animated Exploration of Q, K, V Projections, the Attention Spotlight,
 * and the Backward Pass Blame Propagation using Raylib.
 *
 * Controls:
 *   - SPACE: Play / Pause Animation
 *   - LEFT / RIGHT Arrow: Step Backward / Forward through Scenes
 *   - TAB: Toggle Forward Pass (Signal Flow) vs Backward Pass (Blame Flow)
 *   - 1, 2, 3, 4, 5: Jump directly to Scenes 1 - 5
 *   - Mouse Hover: Inspect any word's internal coordinates & Q, K, V vectors
 * ====================================================================================
 */

#include "raylib.h"
#include "raymath.h"
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip>
#include <sstream>

struct TokenVisual {
    std::string text;
    int pos;
    Vector2 basePos;
    Color color;
    float attnWeight; // Attention received from the active word ("sits")
};

struct Particle {
    Vector2 pos;
    Vector2 target;
    float progress;
    float speed;
    Color color;
};

int main() {
    const int screenWidth = 1200;
    const int screenHeight = 720;

    SetConfigFlags(FLAG_MSAA_4X_HINT);
    InitWindow(screenWidth, screenHeight, "Causal Self-Attention Visualizer (Program 35) - Raylib");
    SetTargetFPS(60);

    std::vector<TokenVisual> tokens = {
        { "the",    0, { 140, 160 }, Color{ 148, 163, 184, 255 }, 0.01f },
        { "king",   1, { 280, 160 }, Color{ 56, 189, 248, 255 },  0.95f },
        { "who",    2, { 420, 160 }, Color{ 148, 163, 184, 255 }, 0.01f },
        { "lived",  3, { 560, 160 }, Color{ 148, 163, 184, 255 }, 0.01f },
        { "in",     4, { 700, 160 }, Color{ 148, 163, 184, 255 }, 0.00f },
        { "palace", 5, { 840, 160 }, Color{ 251, 191, 36, 255 },  0.02f },
        { "sits",   6, { 980, 160 }, Color{ 168, 85, 247, 255 },  0.00f }
    };

    int activeScene = 2; // 0: Inputs, 1: QKV, 2: Spotlight, 3: Prefix Sum, 4: Output / Backward
    bool isPlaying = true;
    bool isBackwardMode = false;
    float sceneTimer = 0.0f;
    float sceneDuration = 4.0f; // 4 seconds per scene

    // Flowing particles along the spotlight beam (from sits to king)
    std::vector<Particle> particles;
    for (int i = 0; i < 30; ++i) {
        particles.push_back({
            tokens[6].basePos,
            tokens[1].basePos,
            (float)i / 30.0f,
            0.015f + (float)(rand() % 10) * 0.001f,
            Color{ 56, 189, 248, 255 }
        });
    }

    while (!WindowShouldClose()) {
        float dt = GetFrameTime();

        // --------------------------------------------------------------------
        // USER INPUTS & CONTROLS
        // --------------------------------------------------------------------
        if (IsKeyPressed(KEY_SPACE)) isPlaying = !isPlaying;
        if (IsKeyPressed(KEY_TAB)) isBackwardMode = !isBackwardMode;
        if (IsKeyPressed(KEY_RIGHT)) { activeScene = (activeScene + 1) % 5; sceneTimer = 0.0f; }
        if (IsKeyPressed(KEY_LEFT))  { activeScene = (activeScene + 4) % 5; sceneTimer = 0.0f; }
        if (IsKeyPressed(KEY_ONE))   { activeScene = 0; sceneTimer = 0.0f; }
        if (IsKeyPressed(KEY_TWO))   { activeScene = 1; sceneTimer = 0.0f; }
        if (IsKeyPressed(KEY_THREE)) { activeScene = 2; sceneTimer = 0.0f; }
        if (IsKeyPressed(KEY_FOUR))  { activeScene = 3; sceneTimer = 0.0f; }
        if (IsKeyPressed(KEY_FIVE))  { activeScene = 4; sceneTimer = 0.0f; }

        if (isPlaying) {
            sceneTimer += dt;
            if (sceneTimer >= sceneDuration) {
                sceneTimer = 0.0f;
                activeScene = (activeScene + 1) % 5;
            }
        }

        // Update particles
        for (auto& p : particles) {
            p.progress += p.speed;
            if (p.progress > 1.0f) p.progress = 0.0f;
        }

        // --------------------------------------------------------------------
        // RENDERING
        // --------------------------------------------------------------------
        BeginDrawing();
        ClearBackground(Color{ 12, 16, 23, 255 }); // Dark theme background

        // Background subtle grid
        for (int x = 0; x < screenWidth; x += 40) {
            for (int y = 0; y < screenHeight; y += 40) {
                DrawPixel(x, y, Color{ 255, 255, 255, 20 });
            }
        }

        // 1. Top Header Bar
        DrawRectangle(0, 0, screenWidth, 65, Color{ 17, 24, 39, 255 });
        DrawLine(0, 65, screenWidth, 65, Color{ 31, 41, 55, 255 });
        DrawText("PROGRAM 35: CAUSAL SELF-ATTENTION VISUALIZER", 25, 18, 20, Color{ 56, 189, 248, 255 });
        DrawText("Weighted Prefix Sum: Q, K, V Projections & Spotlight Beam (Raylib)", 25, 42, 12, Color{ 156, 163, 175, 255 });

        // Mode badge button
        Color modeBg = isBackwardMode ? Color{ 225, 29, 72, 255 } : Color{ 2, 132, 199, 255 };
        const char* modeText = isBackwardMode ? "Mode: BACKWARD PASS (Blame Flow) [TAB]" : "Mode: FORWARD PASS (Signal Flow) [TAB]";
        DrawRectangleRounded(Rectangle{ (float)(screenWidth - 340), 16, 315, 34 }, 0.3f, 6, modeBg);
        DrawText(modeText, screenWidth - 325, 27, 12, WHITE);

        // 2. Scene Progress Indicators (1 to 5)
        const char* sceneNames[5] = {
            "1. Word & Pos (X = C + P)",
            "2. Projections (Q, K, V)",
            "3. Spotlight (Q . K)",
            "4. Prefix Sum (Context)",
            "5. Output Prediction"
        };

        for (int i = 0; i < 5; ++i) {
            float bx = 25.0f + i * 230.0f;
            float by = 80.0f;
            Color scCol = (i == activeScene) ? Color{ 56, 189, 248, 255 } : Color{ 55, 65, 81, 255 };
            DrawRectangleRounded(Rectangle{ bx, by, 220, 28 }, 0.25f, 6, scCol);
            DrawText(sceneNames[i], (int)bx + 12, (int)by + 7, 12, (i == activeScene) ? Color{ 15, 23, 42, 255 } : Color{ 209, 213, 219, 255 });
        }

        // 3. Render Tokens (Word and Position Nodes)
        for (size_t i = 0; i < tokens.size(); ++i) {
            const auto& tok = tokens[i];

            // Glow ring for the active word ("sits") and matched word ("king")
            if (i == 6) { // sits
                DrawCircleGradient(tok.basePos, 45.0f, Color{ 168, 85, 247, 80 }, Color{ 0, 0, 0, 0 });
            } else if (i == 1) { // king
                DrawCircleGradient(tok.basePos, 45.0f, Color{ 56, 189, 248, 80 }, Color{ 0, 0, 0, 0 });
            }

            // Word Pill
            Rectangle pillRect = { tok.basePos.x - 48, tok.basePos.y - 18, 96, 36 };
            DrawRectangleRounded(pillRect, 0.4f, 8, Color{ 30, 41, 59, 255 });
            DrawRectangleRoundedLinesEx(pillRect, 0.4f, 8, 2.0f, tok.color);

            int textW = MeasureText(tok.text.c_str(), 16);
            DrawText(tok.text.c_str(), (int)(tok.basePos.x - textW / 2), (int)(tok.basePos.y - 10), 16, tok.color);

            // Position index below word
            std::string posStr = "pos:" + std::to_string(tok.pos);
            DrawText(posStr.c_str(), (int)(tok.basePos.x - MeasureText(posStr.c_str(), 10) / 2), (int)(tok.basePos.y + 22), 10, Color{ 148, 163, 184, 255 });

            // Scene 1: Show Coordinate Vectors Box
            if (activeScene >= 0) {
                float boxY = tok.basePos.y + 60;
                DrawLine((int)tok.basePos.x, (int)tok.basePos.y + 18, (int)tok.basePos.x, (int)boxY - 14, Color{ 71, 85, 105, 255 });
                DrawRectangleRounded(Rectangle{ tok.basePos.x - 45, boxY - 12, 90, 24 }, 0.2f, 4, Color{ 15, 23, 42, 255 });
                DrawRectangleRoundedLinesEx(Rectangle{ tok.basePos.x - 45, boxY - 12, 90, 24 }, 0.2f, 4, 1.0f, Color{ 56, 189, 248, 150 });
                std::string xStr = "X[" + std::to_string(tok.pos) + "]=C+P";
                DrawText(xStr.c_str(), (int)(tok.basePos.x - MeasureText(xStr.c_str(), 10) / 2), (int)boxY - 6, 10, WHITE);
            }

            // Scene 2: Show Q, K, V Triplet Nodes
            if (activeScene >= 1) {
                float qkvY = tok.basePos.y + 130;
                DrawLine((int)tok.basePos.x, (int)tok.basePos.y + 72, (int)tok.basePos.x, (int)qkvY - 14, Color{ 71, 85, 105, 255 });

                // Q (Cyan), K (Amber), V (Emerald)
                DrawCircle((int)(tok.basePos.x - 22), (int)qkvY, 10, Color{ 2, 132, 199, 255 });
                DrawCircle((int)tok.basePos.x,        (int)qkvY, 10, Color{ 217, 119, 6, 255 });
                DrawCircle((int)(tok.basePos.x + 22), (int)qkvY, 10, Color{ 5, 150, 105, 255 });

                DrawText("Q", (int)(tok.basePos.x - 25), (int)qkvY - 4, 10, WHITE);
                DrawText("K", (int)(tok.basePos.x - 3),  (int)qkvY - 4, 10, WHITE);
                DrawText("V", (int)(tok.basePos.x + 19), (int)qkvY - 4, 10, WHITE);
            }
        }

        // 4. Scene 3 & 4: Render Attention Spotlight Beams (from sits to king)
        if (activeScene >= 2) {
            Vector2 qPos = { tokens[6].basePos.x - 22, tokens[6].basePos.y + 130 }; // sits Q
            Vector2 kKingPos = { tokens[1].basePos.x, tokens[1].basePos.y + 130 };  // king K

            // Draw glowing curved spotlight arc
            Vector2 controlPt = { (qPos.x + kKingPos.x) / 2.0f, qPos.y + 120 };

            // Outer thick glow
            for (int t_step = 0; t_step < 20; ++t_step) {
                float f1 = (float)t_step / 20.0f;
                float f2 = (float)(t_step + 1) / 20.0f;
                Vector2 p1 = Vector2Add(Vector2Scale(qPos, (1 - f1) * (1 - f1)),
                             Vector2Add(Vector2Scale(controlPt, 2 * (1 - f1) * f1), Vector2Scale(kKingPos, f1 * f1)));
                Vector2 p2 = Vector2Add(Vector2Scale(qPos, (1 - f2) * (1 - f2)),
                             Vector2Add(Vector2Scale(controlPt, 2 * (1 - f2) * f2), Vector2Scale(kKingPos, f2 * f2)));
                DrawLineEx(p1, p2, 8.0f, Color{ 56, 189, 248, 60 });
                DrawLineEx(p1, p2, 3.0f, Color{ 56, 189, 248, 255 });
            }

            // Flowing particles along the arc
            for (const auto& p : particles) {
                float f = isBackwardMode ? (1.0f - p.progress) : p.progress; // Reverse direction in backward mode!
                Vector2 pt = Vector2Add(Vector2Scale(qPos, (1 - f) * (1 - f)),
                             Vector2Add(Vector2Scale(controlPt, 2 * (1 - f) * f), Vector2Scale(kKingPos, f * f)));
                Color pCol = isBackwardMode ? Color{ 239, 68, 68, 255 } : Color{ 56, 189, 248, 255 };
                DrawCircle((int)pt.x, (int)pt.y, 4, pCol);
            }

            // Spotlight Match Banner
            DrawRectangleRounded(Rectangle{ controlPt.x - 110, controlPt.y - 16, 220, 32 }, 0.4f, 6, Color{ 2, 132, 199, 255 });
            DrawText("Spotlight: Dot(sits, king) = +4.2 (95%)", (int)controlPt.x - 100, (int)controlPt.y - 6, 11, WHITE);
        }

        // 5. Scene 4 & 5: Weighted Prefix Sum (Context) & Output
        if (activeScene >= 3) {
            float ctxY = 460;
            Rectangle ctxRect = { tokens[6].basePos.x - 70, ctxY, 140, 42 };
            DrawRectangleRounded(ctxRect, 0.3f, 6, Color{ 6, 78, 59, 255 });
            DrawRectangleRoundedLinesEx(ctxRect, 0.3f, 6, 2.0f, Color{ 52, 211, 153, 255 });
            DrawText("Context = 95% * V(king)", (int)ctxRect.x + 8, (int)ctxRect.y + 14, 11, WHITE);

            // Connect Value from king to context
            DrawLineEx(Vector2{ tokens[1].basePos.x + 22, tokens[1].basePos.y + 140 }, Vector2{ ctxRect.x, ctxY + 21 }, 2.0f, Color{ 52, 211, 153, 150 });
        }

        if (activeScene >= 4) {
            float outY = 550;
            DrawLine((int)tokens[6].basePos.x, 502, (int)tokens[6].basePos.x, (int)outY - 14, Color{ 34, 197, 94, 255 });

            Rectangle outRect = { tokens[6].basePos.x - 65, outY, 130, 40 };
            DrawRectangleRounded(outRect, 0.4f, 8, Color{ 22, 101, 52, 255 });
            DrawRectangleRoundedLinesEx(outRect, 0.4f, 8, 2.0f, Color{ 34, 197, 94, 255 });
            DrawText("Predict: \"on\"", (int)outRect.x + 22, (int)outRect.y + 7, 14, WHITE);
            DrawText("Prob: 98.5%", (int)outRect.x + 35, (int)outRect.y + 24, 10, Color{ 187, 247, 208, 255 });

            if (isBackwardMode) {
                // Red glowing pulse banner
                DrawRectangle(0, 600, screenWidth, 40, Color{ 159, 18, 57, 180 });
                DrawText("⚡ BACKWARD PASS: Blame travels straight back across the spotlight beam into 'king' with ZERO decay!", 120, 612, 15, WHITE);
            }
        }

        // 6. Bottom Controls Legend
        DrawRectangle(0, screenHeight - 45, screenWidth, 45, Color{ 15, 23, 42, 255 });
        DrawLine(0, screenHeight - 45, screenWidth, screenHeight - 45, Color{ 30, 41, 59, 255 });
        DrawText("[SPACE] Play/Pause    [LEFT/RIGHT] Change Scene    [TAB] Toggle Forward/Backward    [1-5] Jump to Scene", 25, screenHeight - 28, 12, Color{ 148, 163, 184, 255 });

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
