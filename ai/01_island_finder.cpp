#include <iostream>
#include <vector>
#include <queue>

// A simple structure to hold the coordinates of a single pixel (row, col)
struct Point {
    int r;
    int c;
};

// Represents one connected island of black pixels
struct Island {
    int id;
    std::vector<Point> pixels; // All pixels belonging to this island
    int min_r, max_r;         // Top and bottom boundaries
    int min_c, max_c;         // Left and right boundaries
};

// Breadth-First Search (BFS) to explore all connected black pixels
Island explore_island(const std::vector<std::vector<int>>& image, 
                      std::vector<std::vector<bool>>& visited, 
                      int start_r, int start_c, int island_id) {
    int rows = image.size();
    int cols = image[0].size();

    Island island;
    island.id = island_id;
    island.min_r = start_r;
    island.max_r = start_r;
    island.min_c = start_c;
    island.max_c = start_c;

    std::queue<Point> q;
    q.push({start_r, start_c});
    visited[start_r][start_c] = true;

    // Directions to move: Up, Down, Left, Right
    const int dr[4] = {-1, 1, 0, 0};
    const int dc[4] = {0, 0, -1, 1};

    while (!q.empty()) {
        Point curr = q.front();
        q.pop();
        island.pixels.push_back(curr);

        // Update boundaries
        if (curr.r < island.min_r) island.min_r = curr.r;
        if (curr.r > island.max_r) island.max_r = curr.r;
        if (curr.c < island.min_c) island.min_c = curr.c;
        if (curr.c > island.max_c) island.max_c = curr.c;

        // Check 4 adjacent neighbors
        for (int i = 0; i < 4; ++i) {
            int nr = curr.r + dr[i];
            int nc = curr.c + dc[i];

            // Make sure neighbor is within the image bounds
            if (nr >= 0 && nr < rows && nc >= 0 && nc < cols) {
                // If it is black ink (1) and not visited yet
                if (image[nr][nc] == 1 && !visited[nr][nc]) {
                    visited[nr][nc] = true;
                    q.push({nr, nc});
                }
            }
        }
    }

    return island;
}

// Function to find all black islands in an image
std::vector<Island> find_islands(const std::vector<std::vector<int>>& image) {
    std::vector<Island> islands;
    if (image.empty() || image[0].empty()) return islands;

    int rows = image.size();
    int cols = image[0].size();

    // Matrix to keep track of visited pixels
    std::vector<std::vector<bool>> visited(rows, std::vector<bool>(cols, false));

    int current_id = 1;

    // Scan every pixel from top-left to bottom-right
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            // Found a new unvisited black pixel!
            if (image[r][c] == 1 && !visited[r][c]) {
                Island island = explore_island(image, visited, r, c, current_id);
                islands.push_back(island);
                current_id++;
            }
        }
    }

    return islands;
}

int main() {
    // 0 = White background
    // 1 = Black ink
    // Here we have 3 distinct shapes on a white background:
    // Shape 1: A 2x2 solid block (Square)
    // Shape 2: A 3x1 vertical line (looks like a "1")
    // Shape 3: A 3x3 hollow box
    std::vector<std::vector<int>> screen = {
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 1, 1, 0, 0, 0, 1, 0, 0, 0},
        {0, 1, 1, 0, 0, 0, 1, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 1, 1, 1, 0, 0, 0, 0, 0},
        {0, 0, 1, 0, 1, 0, 0, 0, 0, 0},
        {0, 0, 1, 1, 1, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
    };

    std::cout << "--- Screen Grid (0 = White, 1 = Black) ---\n";
    for (const auto& row : screen) {
        for (int pixel : row) {
            std::cout << (pixel ? "# " : ". ");
        }
        std::cout << "\n";
    }

    std::vector<Island> islands = find_islands(screen);

    std::cout << "\nFound " << islands.size() << " black islands!\n\n";

    for (const auto& island : islands) {
        int width = island.max_c - island.min_c + 1;
        int height = island.max_r - island.min_r + 1;
        std::cout << "Island #" << island.id << ":\n";
        std::cout << "  - Total black pixels (Area): " << island.pixels.size() << "\n";
        std::cout << "  - Bounding Box: from (" << island.min_r << ", " << island.min_c 
                  << ") to (" << island.max_r << ", " << island.max_c << ")\n";
        std::cout << "  - Width: " << width << ", Height: " << height << "\n";
    }

    return 0;
}
