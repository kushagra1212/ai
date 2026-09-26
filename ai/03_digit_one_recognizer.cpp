#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <string>

// Standard size to which every cropped island is scaled
const int GRID_SIZE = 8;
const int NUM_PIXELS = GRID_SIZE * GRID_SIZE; // 64 pixels

// Structure to hold an 8x8 image and its answer (1 for "is a 1", 0 for "not a 1")
struct Sample {
    std::vector<double> pixels; // 64 numbers (0.0 or 1.0)
    double target;             // 1.0 = "1", 0.0 = "not 1"
    std::string name;
};

// Resizes any cropped island into an 8x8 grid
std::vector<double> resize_to_8x8(const std::vector<std::vector<int>>& patch) {
    std::vector<double> result(NUM_PIXELS, 0.0);
    int h = patch.size();
    int w = patch[0].size();

    for (int r = 0; r < GRID_SIZE; ++r) {
        for (int c = 0; c < GRID_SIZE; ++c) {
            // Map 8x8 coordinates back to original patch coordinates
            int orig_r = (r * h) / GRID_SIZE;
            int orig_c = (c * w) / GRID_SIZE;
            if (orig_r >= h) orig_r = h - 1;
            if (orig_c >= w) orig_c = w - 1;

            result[r * GRID_SIZE + c] = (patch[orig_r][orig_c] == 1) ? 1.0 : 0.0;
        }
    }
    return result;
}

// Squashes any raw score into a smooth percentage from 0.0 (0%) to 1.0 (100%)
double squash_to_confidence(double raw_score) {
    return 1.0 / (1.0 + std::exp(-raw_score));
}

// Our First-Principles Learning Model
class DigitOneRecognizer {
public:
    std::vector<double> dials; // One multiplier dial per pixel (64 total)
    double baseline;           // One baseline adjustment dial

    DigitOneRecognizer() {
        // Start all dials at zero
        dials = std::vector<double>(NUM_PIXELS, 0.0);
        baseline = 0.0;
    }

    // Calculates the score and confidence for an 8x8 image
    double predict(const std::vector<double>& pixels) const {
        double raw_score = baseline;
        for (int i = 0; i < NUM_PIXELS; ++i) {
            raw_score += pixels[i] * dials[i];
        }
        return squash_to_confidence(raw_score);
    }

    // Trains the dials using examples
    void train(const std::vector<Sample>& dataset, int passes, double step_size) {
        for (int pass = 0; pass < passes; ++pass) {
            double total_difference = 0.0;

            for (const auto& sample : dataset) {
                double guess = predict(sample.pixels);
                double difference = guess - sample.target;
                total_difference += std::abs(difference);

                // Nudge the baseline dial
                baseline -= step_size * difference;

                // Nudge the pixel multiplier dials
                for (int i = 0; i < NUM_PIXELS; ++i) {
                    if (sample.pixels[i] > 0.0) {
                        dials[i] -= step_size * difference * sample.pixels[i];
                    }
                }
            }

            // Print progress every 100 passes
            if (pass == 0 || pass == 100 || pass == 200 || pass == passes - 1) {
                std::cout << "Pass " << std::setw(3) << pass 
                          << " | Average Error: " << std::fixed << std::setprecision(4) 
                          << (total_difference / dataset.size()) << "\n";
            }
        }
    }

    // Visualizes what the machine learned across the 8x8 grid
    void print_dials() const {
        std::cout << "\n--- Learned Dials (Positive = 'looks like 1', Negative = 'looks like NOT 1') ---\n";
        for (int r = 0; r < GRID_SIZE; ++r) {
            for (int c = 0; c < GRID_SIZE; ++c) {
                double val = dials[r * GRID_SIZE + c];
                if (val > 0.8) {
                    std::cout << " +++ "; // Strong positive
                } else if (val > 0.2) {
                    std::cout << "  +  "; // Mild positive
                } else if (val < -0.8) {
                    std::cout << " --- "; // Strong negative
                } else if (val < -0.2) {
                    std::cout << "  -  "; // Mild negative
                } else {
                    std::cout << "  .  "; // Neutral
                }
            }
            std::cout << "\n";
        }
        std::cout << "Baseline dial value: " << baseline << "\n\n";
    }
};

int main() {
    std::cout << "================ PROJECT 3: LEARNING THE DIGIT '1' ================\n\n";

    // 1. Create a training set of various ways people write "1", and various "non-1" shapes
    std::vector<Sample> dataset;

    // --- Examples of "1" (Target = 1.0) ---
    // Example 1: Plain vertical stick
    std::vector<std::vector<int>> one_stick = {
        {0,1,0},
        {0,1,0},
        {0,1,0},
        {0,1,0},
        {0,1,0}
    };
    dataset.push_back({resize_to_8x8(one_stick), 1.0, "Plain 1 stick"});

    // Example 2: 1 with a top hook
    std::vector<std::vector<int>> one_hook = {
        {1,1,0},
        {0,1,0},
        {0,1,0},
        {0,1,0},
        {0,1,0}
    };
    dataset.push_back({resize_to_8x8(one_hook), 1.0, "1 with top hook"});

    // Example 3: 1 with top hook and bottom bar
    std::vector<std::vector<int>> one_full = {
        {1,1,0},
        {0,1,0},
        {0,1,0},
        {0,1,0},
        {1,1,1}
    };
    dataset.push_back({resize_to_8x8(one_full), 1.0, "1 with hook and base"});

    // Example 4: Thick 1
    std::vector<std::vector<int>> one_thick = {
        {1,1,1,0},
        {0,1,1,0},
        {0,1,1,0},
        {0,1,1,0},
        {1,1,1,1}
    };
    dataset.push_back({resize_to_8x8(one_thick), 1.0, "Thick 1"});

    // --- Examples of "NOT 1" (Target = 0.0) ---
    // Example 5: A "0" (empty middle, ink on borders)
    std::vector<std::vector<int>> zero_box = {
        {1,1,1},
        {1,0,1},
        {1,0,1},
        {1,0,1},
        {1,1,1}
    };
    dataset.push_back({resize_to_8x8(zero_box), 0.0, "Digit 0"});

    // Example 6: A plus sign "+"
    std::vector<std::vector<int>> plus_sign = {
        {0,0,1,0,0},
        {0,0,1,0,0},
        {1,1,1,1,1},
        {0,0,1,0,0},
        {0,0,1,0,0}
    };
    dataset.push_back({resize_to_8x8(plus_sign), 0.0, "Plus sign"});

    // Example 7: An "L" shape
    std::vector<std::vector<int>> letter_L = {
        {1,0,0,0},
        {1,0,0,0},
        {1,0,0,0},
        {1,1,1,1}
    };
    dataset.push_back({resize_to_8x8(letter_L), 0.0, "Letter L"});

    // Example 8: A horizontal line "-"
    std::vector<std::vector<int>> dash = {
        {0,0,0,0,0},
        {0,0,0,0,0},
        {1,1,1,1,1},
        {0,0,0,0,0},
        {0,0,0,0,0}
    };
    dataset.push_back({resize_to_8x8(dash), 0.0, "Dash line"});

    // 2. Train our recognizer
    DigitOneRecognizer model;
    std::cout << "Starting Training (Tuning the dials)...\n";
    model.train(dataset, 1000000, 0.0005);

    // 3. Look at what the dials learned!
    model.print_dials();

    // 4. TEST 1: The 'Unidentifiable' Island #5 from Project 2!
    std::vector<std::vector<int>> test_island_5 = {
        {1, 1, 1, 0},
        {0, 0, 1, 0},
        {0, 0, 1, 0},
        {0, 0, 1, 0},
        {0, 1, 1, 0}
    };
    std::vector<double> input_5 = resize_to_8x8(test_island_5);
    double confidence_5 = model.predict(input_5);

    std::cout << "--- Testing Island #5 (The unidentifiable shape from Project 2) ---\n";
    std::cout << "Machine Confidence: " << std::fixed << std::setprecision(1) 
              << (confidence_5 * 100.0) << "%\n";
    if (confidence_5 >= 0.70) {
        std::cout << "===> RESULT: Confirmed! This island is the digit '1'.\n\n";
    } else {
        std::cout << "===> RESULT: NOT a '1'.\n\n";
    }

    // 5. TEST 2: A mystery non-1 shape (like a "U")
    std::vector<std::vector<int>> test_letter_U = {
        {1, 0, 0, 1},
        {1, 0, 0, 1},
        {1, 0, 0, 1},
        {1, 1, 1, 1}
    };
    std::vector<double> input_U = resize_to_8x8(test_letter_U);
    double confidence_U = model.predict(input_U);

    std::cout << "--- Testing Mystery Shape (A 'U' shape) ---\n";
    std::cout << "Machine Confidence: " << std::fixed << std::setprecision(1) 
              << (confidence_U * 100.0) << "%\n";
    if (confidence_U >= 0.70) {
        std::cout << "===> RESULT: Confirmed! This island is the digit '1'.\n";
    } else {
        std::cout << "===> RESULT: Correctly rejected! NOT a '1'.\n";
    }


    return 0;
}
