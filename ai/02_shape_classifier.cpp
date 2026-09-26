#include <iostream>
#include <vector>
#include <queue>
#include <cmath>
#include <string>
#include <algorithm>

struct Point {
    int r, c;
};

struct Island {
    int id;
    std::vector<Point> pixels;
    int min_r, max_r;
    int min_c, max_c;
};

// Reuse our island exploration from Project 1
Island explore_island(const std::vector<std::vector<int>>& image, 
                      std::vector<std::vector<bool>>& visited, 
                      int start_r, int start_c, int island_id) {
    int rows = image.size();
    int cols = image[0].size();

    Island island;
    island.id = island_id;
    island.min_r = start_r; island.max_r = start_r;
    island.min_c = start_c; island.max_c = start_c;

    std::queue<Point> q;
    q.push({start_r, start_c});
    visited[start_r][start_c] = true;

    const int dr[4] = {-1, 1, 0, 0};
    const int dc[4] = {0, 0, -1, 1};

    while (!q.empty()) {
        Point curr = q.front();
        q.pop();
        island.pixels.push_back(curr);

        if (curr.r < island.min_r) island.min_r = curr.r;
        if (curr.r > island.max_r) island.max_r = curr.r;
        if (curr.c < island.min_c) island.min_c = curr.c;
        if (curr.c > island.max_c) island.max_c = curr.c;

        for (int i = 0; i < 4; ++i) {
            int nr = curr.r + dr[i];
            int nc = curr.c + dc[i];
            if (nr >= 0 && nr < rows && nc >= 0 && nc < cols) {
                if (image[nr][nc] == 1 && !visited[nr][nc]) {
                    visited[nr][nc] = true;
                    q.push({nr, nc});
                }
            }
        }
    }
    return island;
}

std::vector<Island> find_islands(const std::vector<std::vector<int>>& image) {
    std::vector<Island> islands;
    int rows = image.size();
    int cols = image[0].size();
    std::vector<std::vector<bool>> visited(rows, std::vector<bool>(cols, false));

    int current_id = 1;
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (image[r][c] == 1 && !visited[r][c]) {
                islands.push_back(explore_island(image, visited, r, c, current_id));
                current_id++;
            }
        }
    }
    return islands;
}

// Function to classify the shape of an island using simple geometric numbers
std::string classify_shape(const Island& island, const std::vector<std::vector<int>>& image) {
    int area = island.pixels.size();
    int width = island.max_c - island.min_c + 1;
    int height = island.max_r - island.min_r + 1;
    int box_area = width * height;

    double fill_ratio = (double)area / box_area;
    double aspect_ratio = (double)width / height;

    // Minimum size check to avoid classifying 1 or 2 stray noise pixels
    if (width < 3 || height < 3) {
        return "Not Identifiable (too small / line)";
    }

    // 1. SOLID BOX CHECK (Fill ratio close to 100%)
    if (fill_ratio >= 0.88) {
        // If width and height are nearly equal -> Square
        if (aspect_ratio >= 0.80 && aspect_ratio <= 1.25) {
            return "Square";
        } else {
            return "Rectangle";
        }
    }

    // 2. TRIANGLE CHECK
    // In geometry, Area = 0.5 * base * height.
    // So fill_ratio should be around 0.40 to 0.65.
    // Also, row widths should expand steadily from tip to base.
    if (fill_ratio >= 0.40 && fill_ratio <= 0.65) {
        // Measure pixel count row by row
        std::vector<int> row_counts(height, 0);
        for (const auto& p : island.pixels) {
            row_counts[p.r - island.min_r]++;
        }

        // Check if rows consistently widen from top to bottom (pointing up)
        bool pointing_up = true;
        for (int i = 0; i < height - 1; ++i) {
            if (row_counts[i] > row_counts[i + 1]) {
                pointing_up = false;
                break;
            }
        }
        // Or rows widen from bottom to top (pointing down)
        bool pointing_down = true;
        for (int i = 0; i < height - 1; ++i) {
            if (row_counts[i] < row_counts[i + 1]) {
                pointing_down = false;
                break;
            }
        }

        if (pointing_up || pointing_down) {
            return "Triangle";
        }
    }

    // 3. CIRCLE CHECK
    // Circle inside a square box fills roughly pi / 4 = 78.5% of the box.
    // Width and height should also be roughly equal.
    if (aspect_ratio >= 0.80 && aspect_ratio <= 1.25 &&
        fill_ratio >= 0.65 && fill_ratio <= 0.85) {
        
        // Let's verify roundness:
        // Center of the bounding box:
        double center_r = island.min_r + (height - 1) / 2.0;
        double center_c = island.min_c + (width - 1) / 2.0;
        double expected_radius = (width + height) / 4.0;

        // In a true circle, the 4 corners of its bounding box are white (0)
        bool corners_empty = (image[island.min_r][island.min_c] == 0) &&
                             (image[island.min_r][island.max_c] == 0) &&
                             (image[island.max_r][island.min_c] == 0) &&
                             (image[island.max_r][island.max_c] == 0);

        if (corners_empty) {
            return "Circle";
        }
    }

    // If it doesn't match clean geometric rules:
    return "Not Identifiable";
}

int main() {
    // A canvas with 5 different shapes:
    // 1. Square (4x4 solid)
    // 2. Rectangle (3x6 solid)
    // 3. Triangle (pointing up)
    // 4. Circle (round shape with empty corners)
    // 5. Letter "1" (handwritten style: hook at top, vertical body) -> Unidentifiable by basic geometry!
    std::vector<std::vector<int>> canvas = {
        // Row 0-5: Square and Rectangle
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0, 1,1,1,1, 0,0, 1,1,1,1,1,1, 0,0,0,0,0,0,0},
        {0, 1,1,1,1, 0,0, 1,1,1,1,1,1, 0,0,0,0,0,0,0},
        {0, 1,1,1,1, 0,0, 1,1,1,1,1,1, 0,0,0,0,0,0,0},
        {0, 1,1,1,1, 0,0, 0,0,0,0,0,0, 0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},

        // Row 6-12: Triangle, Circle, and Handwritten "1"
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0, 0,0,1,0,0, 0,0, 0,1,1,1,0, 0,0, 0,1,1,0, 0},
        {0, 0,1,1,1,0, 0,0, 1,1,1,1,1, 0,0, 0,0,1,0, 0},
        {0, 1,1,1,1,1, 0,0, 1,1,1,1,1, 0,0, 0,0,1,0, 0},
        {0, 0,0,0,0,0, 0,0, 1,1,1,1,1, 0,0, 0,0,1,0, 0},
        {0, 0,0,0,0,0, 0,0, 0,1,1,1,0, 0,0, 1,1,1,1, 0},
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}
    };

    std::cout << "================= CANVAS ================\n";
    for (const auto& row : canvas) {
        for (int pixel : row) {
            std::cout << (pixel ? "# " : ". ");
        }
        std::cout << "\n";
    }

    std::vector<Island> islands = find_islands(canvas);
    std::cout << "\nTotal Islands Found: " << islands.size() << "\n\n";

    for (const auto& island : islands) {
        int width = island.max_c - island.min_c + 1;
        int height = island.max_r - island.min_r + 1;
        int area = island.pixels.size();
        double fill = (double)area / (width * height);
        double aspect = (double)width / height;

        std::string shape = classify_shape(island, canvas);

        std::cout << "Island #" << island.id << ":\n";
        std::cout << "  - Size: " << width << "x" << height 
                  << " (Box Area: " << width * height << ", Black Pixels: " << area << ")\n";
        std::cout << "  - Fill Factor: " << (int)(fill * 100) << "%\n";
        std::cout << "  - Width/Height Ratio: " << aspect << "\n";
        std::cout << "  ===> Classified As: [" << shape << "]\n\n";
    }

    return 0;
}
