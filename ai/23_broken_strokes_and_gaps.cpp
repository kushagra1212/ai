#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <string>

// -----------------------------------------------------------------------------
// Activation functions
// -----------------------------------------------------------------------------
double squash(double z) {
    if (z > 50.0) return 1.0;
    if (z < -50.0) return 0.0;
    return 1.0 / (1.0 + std::exp(-z));
}
double smooth_ramp(double z) {
    return z * squash(z);
}

// -----------------------------------------------------------------------------
// FIRST PRINCIPLES: HOW A 3x3 STENCIL HANDLES BROKEN STROKES (GAPS)
// -----------------------------------------------------------------------------
int main() {
    std::cout << "=========================================================================\n";
    std::cout << " FIRST PRINCIPLES: HOW THE SLIDING STENCIL HANDLES GAPS & BROKEN STROKES\n";
    std::cout << "=========================================================================\n\n";

    // 1. Our learned vertical stencil from Program 22
    std::vector<std::vector<double>> vertical_stencil = {
        {-0.2,  0.6, -0.2},
        {-0.2,  0.6, -0.2},
        {-0.2,  0.6, -0.2}
    };

    // Case A: Continuous Solid Line (3 pixels of ink)
    std::vector<std::vector<double>> solid_patch = {
        {0.0, 1.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 1.0, 0.0}
    };

    // Case B: Broken Line with a 1-pixel Gap in the Middle!
    std::vector<std::vector<double>> broken_patch = {
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 0.0}, // PEN LIFTED / SKIPPED PIXEL!
        {0.0, 1.0, 0.0}
    };

    // Case C: Empty Background (No line)
    std::vector<std::vector<double>> empty_patch = {
        {0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0}
    };

    auto score_patch = [&](const std::vector<std::vector<double>>& patch) {
        double sum = 0.0;
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                sum += patch[r][c] * vertical_stencil[r][c];
            }
        }
        return sum;
    };

    double score_solid  = score_patch(solid_patch);
    double score_broken = score_patch(broken_patch);
    double score_empty  = score_patch(empty_patch);

    std::cout << "1. SOLID VERTICAL LINE (3 pixels ink):\n";
    std::cout << "   Raw Score  : " << std::fixed << std::setprecision(2) << score_solid << "\n";
    std::cout << "   Activation : " << smooth_ramp(score_solid) << "  [Strong Peak]\n\n";

    std::cout << "2. BROKEN VERTICAL LINE (Middle pixel missing / Gap):\n";
    std::cout << "   Raw Score  : " << std::fixed << std::setprecision(2) << score_broken << "\n";
    std::cout << "   Activation : " << smooth_ramp(score_broken) << "  [Still Strongly Detected!]\n\n";

    std::cout << "3. EMPTY BACKGROUND (0 pixels ink):\n";
    std::cout << "   Raw Score  : " << std::fixed << std::setprecision(2) << score_empty << "\n";
    std::cout << "   Activation : " << smooth_ramp(score_empty) << "  [No Detection]\n\n";

    std::cout << "-------------------------------------------------------------------------\n";
    std::cout << "WHY THE STENCIL SURVIVES GAPS (Unlike BFS Island Finder):\n";
    std::cout << "- In BFS: A single missing pixel breaks the connection (turns 1 island into 2).\n";
    std::cout << "- In CNN: The stencil integrates the ENTIRE 3x3 patch. Even with a gap,\n";
    std::cout << "  the score is +1.20 (67% of full signal), easily crossing the threshold!\n";
    std::cout << "=========================================================================\n";

    return 0;
}
