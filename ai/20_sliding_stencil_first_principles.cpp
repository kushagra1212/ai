#include <iostream>
#include <vector>
#include <iomanip>

// -----------------------------------------------------------------------------
// FIRST PRINCIPLES: SLIDING STENCIL (CONVOLUTION)
// -----------------------------------------------------------------------------

// Slide a KxK stencil across an HxW image
std::vector<std::vector<double>> convolve_2d(
    const std::vector<std::vector<double>>& image,
    const std::vector<std::vector<double>>& stencil)
{
    int H = image.size();
    int W = image[0].size();
    int K_rows = stencil.size();
    int K_cols = stencil[0].size();

    int out_H = H - K_rows + 1;
    int out_W = W - K_cols + 1;

    std::vector<std::vector<double>> output(out_H, std::vector<double>(out_W, 0.0));

    // Nested loops: direct mathematical definition
    for (int r = 0; r < out_H; ++r) {
        for (int c = 0; c < out_W; ++c) {
            double sum = 0.0;
            for (int kr = 0; kr < K_rows; ++kr) {
                for (int kc = 0; kc < K_cols; ++kc) {
                    sum += image[r + kr][c + kc] * stencil[kr][kc];
                }
            }
            output[r][c] = sum;
        }
    }
    return output;
}

void print_matrix(const std::string& title, const std::vector<std::vector<double>>& mat) {
    std::cout << title << "\n";
    for (const auto& row : mat) {
        std::cout << "  ";
        for (double val : row) {
            std::cout << std::setw(6) << std::fixed << std::setprecision(1) << val << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

int main() {
    std::cout << "=========================================================================\n";
    std::cout << " FIRST PRINCIPLES: THE SLIDING STENCIL (CONVOLUTION)\n";
    std::cout << "=========================================================================\n\n";

    // 1. A 5x5 image with a vertical line in the MIDDLE (Column 2)
    std::vector<std::vector<double>> img_center = {
        {0, 0, 1, 0, 0},
        {0, 0, 1, 0, 0},
        {0, 0, 1, 0, 0},
        {0, 0, 1, 0, 0},
        {0, 0, 1, 0, 0}
    };

    // 2. A 5x5 image with the vertical line SHIFTED to the LEFT (Column 1)
    std::vector<std::vector<double>> img_shifted = {
        {0, 1, 0, 0, 0},
        {0, 1, 0, 0, 0},
        {0, 1, 0, 0, 0},
        {0, 1, 0, 0, 0},
        {0, 1, 0, 0, 0}
    };

    // 3. The 3x3 Vertical Edge Stencil (9 numbers total)
    std::vector<std::vector<double>> vertical_stencil = {
        {-1.0,  2.0, -1.0},
        {-1.0,  2.0, -1.0},
        {-1.0,  2.0, -1.0}
    };

    print_matrix("IMAGE A: 5x5 (Vertical Line at Column 2)", img_center);
    print_matrix("IMAGE B: 5x5 (Vertical Line Shifted to Column 1)", img_shifted);
    print_matrix("3x3 VERTICAL STENCIL (9 Dials)", vertical_stencil);

    auto result_A = convolve_2d(img_center, vertical_stencil);
    auto result_B = convolve_2d(img_shifted, vertical_stencil);

    print_matrix("OUTPUT A: Response Grid for Image A (3x3)", result_A);
    print_matrix("OUTPUT B: Response Grid for Image B (3x3)", result_B);

    std::cout << "-------------------------------------------------------------------------\n";
    std::cout << "FIRST-PRINCIPLES INSIGHT:\n";
    std::cout << "- In Output A, Column 1 lit up with +6.0!\n";
    std::cout << "- In Output B, Column 0 lit up with +6.0!\n";
    std::cout << "- The vertical feature was detected at +6.0 in BOTH images!\n";
    std::cout << "- Only 9 dials were used across all 25 pixels!\n";
    std::cout << "=========================================================================\n";

    return 0;
}
