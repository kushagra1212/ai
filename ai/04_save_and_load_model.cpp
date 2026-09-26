#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <string>

const int GRID_SIZE = 8;
const int NUM_PIXELS = GRID_SIZE * GRID_SIZE; // 64

// Structure to hold sample training data
struct Sample {
    std::vector<double> pixels;
    double target;
};

std::vector<double> resize_to_8x8(const std::vector<std::vector<int>>& patch) {
    std::vector<double> result(NUM_PIXELS, 0.0);
    int h = patch.size();
    int w = patch[0].size();
    for (int r = 0; r < GRID_SIZE; ++r) {
        for (int c = 0; c < GRID_SIZE; ++c) {
            int orig_r = (r * h) / GRID_SIZE;
            int orig_c = (c * w) / GRID_SIZE;
            if (orig_r >= h) orig_r = h - 1;
            if (orig_c >= w) orig_c = w - 1;
            result[r * GRID_SIZE + c] = (patch[orig_r][orig_c] == 1) ? 1.0 : 0.0;
        }
    }
    return result;
}

double squash_to_confidence(double raw_score) {
    return 1.0 / (1.0 + std::exp(-raw_score));
}

class DigitOneRecognizer {
public:
    std::vector<double> dials; // The 64 weights
    double baseline;           // The bias

    DigitOneRecognizer() {
        dials = std::vector<double>(NUM_PIXELS, 0.0);
        baseline = 0.0;
    }

    double predict(const std::vector<double>& pixels) const {
        double raw_score = baseline;
        for (int i = 0; i < NUM_PIXELS; ++i) {
            raw_score += pixels[i] * dials[i];
        }
        return squash_to_confidence(raw_score);
    }

    void train(const std::vector<Sample>& dataset, int passes, double step_size) {
        for (int pass = 0; pass < passes; ++pass) {
            for (const auto& sample : dataset) {
                double guess = predict(sample.pixels);
                double difference = guess - sample.target;

                baseline -= step_size * difference;
                for (int i = 0; i < NUM_PIXELS; ++i) {
                    if (sample.pixels[i] > 0.0) {
                        dials[i] -= step_size * difference * sample.pixels[i];
                    }
                }
            }
        }
    }

    // 1. SAVE DIALS (WEIGHTS) TO A FILE
    bool save_to_file(const std::string& filepath) const {
        std::ofstream outfile(filepath);
        if (!outfile.is_open()) {
            std::cerr << "Error opening file to write: " << filepath << "\n";
            return false;
        }

        // First line: the baseline dial
        outfile << baseline << "\n";

        // Next lines: all 64 pixel multiplier dials
        for (int i = 0; i < NUM_PIXELS; ++i) {
            outfile << dials[i] << (i % 8 == 7 ? "\n" : " ");
        }

        outfile.close();
        return true;
    }

    // 2. LOAD DIALS (WEIGHTS) FROM A FILE
    bool load_from_file(const std::string& filepath) {
        std::ifstream infile(filepath);
        if (!infile.is_open()) {
            std::cerr << "Error opening file to read: " << filepath << "\n";
            return false;
        }

        // Read baseline
        infile >> baseline;

        // Read all 64 dials
        for (int i = 0; i < NUM_PIXELS; ++i) {
            infile >> dials[i];
        }

        infile.close();
        return true;
    }
};

int main() {
    std::cout << "=== DEMO: TRAINING, SAVING, AND LOADING MODEL DIALS ===\n\n";

    // Create training samples
    std::vector<Sample> dataset = {
        {resize_to_8x8({{0,1,0},{0,1,0},{0,1,0},{0,1,0},{0,1,0}}), 1.0},
        {resize_to_8x8({{1,1,0},{0,1,0},{0,1,0},{0,1,0},{0,1,0}}), 1.0},
        {resize_to_8x8({{1,1,0},{0,1,0},{0,1,0},{0,1,0},{1,1,1}}), 1.0},
        {resize_to_8x8({{1,1,1},{1,0,1},{1,0,1},{1,0,1},{1,1,1}}), 0.0},
        {resize_to_8x8({{1,0,0,0},{1,0,0,0},{1,0,0,0},{1,1,1,1}}), 0.0},
        {resize_to_8x8({{0,0,1,0,0},{0,0,1,0,0},{1,1,1,1,1},{0,0,1,0,0},{0,0,1,0,0}}), 0.0}
    };

    // -------------------------------------------------------------------
    // STEP 1: Train the original model and save to file
    // -------------------------------------------------------------------
    std::cout << "1. Training original model on training data...\n";
    DigitOneRecognizer original_model;
    original_model.train(dataset, 300, 0.15);

    std::string filename = "weights.txt";
    std::cout << "2. Saving learned dials to '" << filename << "'...\n";
    if (original_model.save_to_file(filename)) {
        std::cout << "   Successfully saved dials to " << filename << "!\n\n";
    }

    // -------------------------------------------------------------------
    // STEP 2: Create a fresh new model with all dials = 0 (Untrained)
    // -------------------------------------------------------------------
    std::cout << "3. Creating a brand new, empty model (no training)...\n";
    DigitOneRecognizer loaded_model;

    // Test Island #5 before loading:
    std::vector<std::vector<int>> island_5 = {
        {0, 1, 1, 0},
        {0, 0, 1, 0},
        {0, 0, 1, 0},
        {0, 0, 1, 0},
        {1, 1, 1, 1}
    };
    std::vector<double> input_5 = resize_to_8x8(island_5);

    std::cout << "   Confidence BEFORE loading weights: " 
              << (loaded_model.predict(input_5) * 100.0) << "% (Blank model knows nothing!)\n";

    // -------------------------------------------------------------------
    // STEP 3: Load the saved dials from the file!
    // -------------------------------------------------------------------
    std::cout << "4. Loading dials from '" << filename << "'...\n";
    loaded_model.load_from_file(filename);

    double conf_after = loaded_model.predict(input_5);
    std::cout << "   Confidence AFTER loading weights: " 
              << std::fixed << std::setprecision(1) << (conf_after * 100.0) << "%\n";

    if (conf_after >= 0.70) {
        std::cout << "===> SUCCESS! The loaded model correctly recognized the '1' without needing any training!\n";
    }

    return 0;
}
