#include <iostream>
#include <vector>
#include <iomanip>

// -----------------------------------------------------------------------------
// FIRST-PRINCIPLES SLIDING STENCIL (CONVOLUTION)
// -----------------------------------------------------------------------------

// A 3x3 stencil designed to detect vertical lines:
// Left column: negative (-1.0) -> wants black background
// Center column: positive (+2.0) -> wants bright ink
// Right column: negative (-1.0) -> wants black background
const std::vector<std::vector<double>> VERTICAL_STENCIL = {
    {-1.0,  2.0, -1.0},
    {-1.0,  2.0, -1.0},
    {-1.0,  2.0, -1.0}
};

// Slide the 3x3 stencil across an 8x8 image to produce a 6x6 output grid
std::vector<std::vector<double>> apply_sliding_stencil(
    const std::vector<std::vector<double>>& image,
    const std::vector<std::vector<double>>& stencil) 
{
    int img_rows = image.size();
    int img_cols = image[0].size();
    int sten_rows = stencil.size();
    int sten_cols = stencil[0].size();

    int out_rows = img_rows - sten_rows + 1; // 8 - 3 + 1 = 6
    int out_cols = img_cols - sten_cols + 1; // 8 - 3 + 1 = 6

    std::vector<std::vector<double>> output(out_rows, std::vector<double>(out_cols, 0.0));

    // Slide over every valid position
    for (int r = 0; r < out_rows; ++r) {
        for (int c = 0; c < out_cols; ++c) {
            double sum = 0.0;
            // 3x3 element-wise multiply and accumulate
            for (int sr = 0; sr < sten_rows; ++sr) {
                for (int sc = 0; sc < sten_cols; ++sc) {
                    sum += image[r + sr][c + sc] * stencil[sr][sc];
                }
            }
            output[r][c] = sum;
        }
    }
    return output;
}

void print_grid(const std::string& title, const std::vector<std::vector<double>>& grid) {
    std::cout << title << "\n";
    for (const auto& row : grid) {
        std::cout << "  ";
        for (double val : row) {
            if (val > 0.5) {
                std::cout << std::setw(5) << std::fixed << std::setprecision(1) << val << " ";
            } else if (val < -0.5) {
                std::cout << std::setw(5) << std::fixed << std::setprecision(1) << val << " ";
            } else {
                std::cout << "    . ";
            }
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

int main() {
    std::cout << "=========================================================================\n";
    std::cout << " FIRST PRINCIPLES: THE SHIFT PROBLEM & THE SLIDING STENCIL\n";
    std::cout << "=========================================================================\n\n";

    // Image 1: Digit "1" in the CENTER (Columns 3 and 4)
    std::vector<std::vector<double>> center_one = {
        {0, 0, 0, 1, 1, 0, 0, 0},
        {0, 0, 0, 1, 1, 0, 0, 0},
        {0, 0, 0, 1, 1, 0, 0, 0},
        {0, 0, 0, 1, 1, 0, 0, 0},
        {0, 0, 0, 1, 1, 0, 0, 0},
        {0, 0, 0, 1, 1, 0, 0, 0},
        {0, 0, 0, 1, 1, 0, 0, 0},
        {0, 0, 0, 1, 1, 0, 0, 0}
    };

    // Image 2: Digit "1" SHIFTED 2 PIXELS LEFT (Columns 1 and 2)
    std::vector<std::vector<double>> shifted_one = {
        {0, 1, 1, 0, 0, 0, 0, 0},
        {0, 1, 1, 0, 0, 0, 0, 0},
        {0, 1, 1, 0, 0, 0, 0, 0},
        {0, 1, 1, 0, 0, 0, 0, 0},
        {0, 1, 1, 0, 0, 0, 0, 0},
        {0, 1, 1, 0, 0, 0, 0, 0},
        {0, 1, 1, 0, 0, 0, 0, 0},
        {0, 1, 1, 0, 0, 0, 0, 0}
    };

    print_grid("IMAGE 1: Center '1' (8x8)", center_one);
    print_grid("IMAGE 2: Shifted-Left '1' (8x8)", shifted_one);

    auto result_center = apply_sliding_stencil(center_one, VERTICAL_STENCIL);
    auto result_shifted = apply_sliding_stencil(shifted_one, VERTICAL_STENCIL);

    print_grid("OUTPUT 1: Stencil Responses for Center '1' (6x6)", result_center);
    print_grid("OUTPUT 2: Stencil Responses for Shifted-Left '1' (6x6)", result_shifted);

    std::cout << "OBSERVATION:\n";
    std::cout << "Both images produce strong vertical detection spikes (+3.0)!\n";
    std::cout << "The stencil found the vertical line regardless of where it shifted!\n";
    std::cout << "Total dials used: ONLY 9 dials (instead of 64 fixed dials)!\n";

    return 0;
}
