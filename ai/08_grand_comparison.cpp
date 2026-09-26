#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <random>
#include <algorithm>
#include <string>

const int GRID_SIZE = 8;
const int NUM_PIXELS = 64;

struct Sample {
    std::vector<double> pixels;
    double target;
    std::string label;
};

double squash(double z) {
    if (z > 50.0) return 1.0;
    if (z < -50.0) return 0.0;
    return 1.0 / (1.0 + std::exp(-z));
}

std::string mini_bar(double error) {
    double acc = std::max(0.0, 1.0 - error * 2.0);
    int filled = static_cast<int>(acc * 10);
    if (filled > 10) filled = 10;
    std::string b = "[";
    for (int i = 0; i < 10; ++i) b += (i < filled) ? "#" : ".";
    b += "]";
    return b;
}

double compute_avg_error(const std::vector<double>& dials, double baseline, const std::vector<Sample>& dataset) {
    double total = 0.0;
    for (const auto& s : dataset) {
        double raw = baseline;
        for (int i = 0; i < NUM_PIXELS; ++i) raw += s.pixels[i] * dials[i];
        total += std::abs(squash(raw) - s.target);
    }
    return total / dataset.size();
}

int main() {
    std::cout << "====================================================================================\n";
    std::cout << " PROGRAM C: THE GRAND ARENA (HEAD-TO-HEAD COMPARISON OF ALL 3 METHODS)\n";
    std::cout << "====================================================================================\n\n";

    // 1. Prepare Standard Dataset
    std::vector<Sample> dataset;

    std::vector<double> one_center(64, 0.0);
    for (int r = 0; r < 8; ++r) one_center[r * 8 + 3] = 1.0;
    dataset.push_back({one_center, 1.0, "1 (center line)"});

    std::vector<double> one_hook = one_center;
    one_hook[0 * 8 + 2] = 1.0;
    dataset.push_back({one_hook, 1.0, "1 (with hook)"});

    std::vector<double> one_base = one_center;
    for (int c = 2; c <= 4; ++c) one_base[7 * 8 + c] = 1.0;
    dataset.push_back({one_base, 1.0, "1 (with base)"});

    std::vector<double> zero_box(64, 0.0);
    for (int i = 0; i < 8; ++i) {
        zero_box[0 * 8 + i] = 1.0; zero_box[7 * 8 + i] = 1.0;
        zero_box[i * 8 + 0] = 1.0; zero_box[i * 8 + 7] = 1.0;
    }
    dataset.push_back({zero_box, 0.0, "0 (box)"});

    std::vector<double> letter_L(64, 0.0);
    for (int r = 0; r < 8; ++r) letter_L[r * 8 + 1] = 1.0;
    for (int c = 1; c < 7; ++c) letter_L[7 * 8 + c] = 1.0;
    dataset.push_back({letter_L, 0.0, "Letter L"});

    std::vector<double> plus_sign(64, 0.0);
    for (int i = 0; i < 8; ++i) {
        plus_sign[3 * 8 + i] = 1.0;
        plus_sign[i * 8 + 3] = 1.0;
    }
    dataset.push_back({plus_sign, 0.0, "Plus sign"});

    // ----------------------------------------------------------------------------------
    // Method 1 Setup: Basic Fixed Step Size (0.15)
    // ----------------------------------------------------------------------------------
    std::vector<double> m1_dials(NUM_PIXELS, 0.0);
    double m1_baseline = 0.0;

    // ----------------------------------------------------------------------------------
    // Method 2 Setup: Momentum + Step Decay
    // ----------------------------------------------------------------------------------
    std::vector<double> m2_dials(NUM_PIXELS, 0.0);
    double m2_baseline = 0.0;
    std::vector<double> m2_v_dials(NUM_PIXELS, 0.0);
    double m2_v_baseline = 0.0;

    // ----------------------------------------------------------------------------------
    // Method 3 Setup: Darwinian Evolution (100 population)
    // ----------------------------------------------------------------------------------
    struct Organism {
        std::vector<double> dials;
        double baseline;
        double error;
    };
    std::vector<Organism> population(100);
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> init_d(-0.5, 0.5);
    for (auto& org : population) {
        org.dials.resize(NUM_PIXELS);
        org.baseline = init_d(rng);
        for (int i = 0; i < NUM_PIXELS; ++i) org.dials[i] = init_d(rng);
        org.error = 999.0;
    }

    std::cout << "Running 50 Rounds across all 3 methods simultaneously...\n\n";
    std::cout << "------------------------------------------------------------------------------------\n";
    std::cout << " Round | Method 1: Basic Fixed   | Method 2: Momentum+Decay | Method 3: Evolution   \n";
    std::cout << "       | Error     Progress      | Error     Progress       | Best Error  Progress  \n";
    std::cout << "------------------------------------------------------------------------------------\n";

    for (int round = 0; round < 50; ++round) {
        // --- 1. Step Method 1 ---
        for (const auto& s : dataset) {
            double raw = m1_baseline;
            for (int i = 0; i < NUM_PIXELS; ++i) raw += s.pixels[i] * m1_dials[i];
            double diff = squash(raw) - s.target;
            m1_baseline -= 0.15 * diff;
            for (int i = 0; i < NUM_PIXELS; ++i) {
                if (s.pixels[i] > 0.0) m1_dials[i] -= 0.15 * diff * s.pixels[i];
            }
        }

        // --- 2. Step Method 2 (Momentum + Decay) ---
        double step2 = 0.40 / (1.0 + 0.05 * round);
        for (const auto& s : dataset) {
            double raw = m2_baseline;
            for (int i = 0; i < NUM_PIXELS; ++i) raw += s.pixels[i] * m2_dials[i];
            double diff = squash(raw) - s.target;
            m2_v_baseline = 0.80 * m2_v_baseline - step2 * diff;
            m2_baseline += m2_v_baseline;
            for (int i = 0; i < NUM_PIXELS; ++i) {
                if (s.pixels[i] > 0.0) {
                    m2_v_dials[i] = 0.80 * m2_v_dials[i] - step2 * diff * s.pixels[i];
                    m2_dials[i] += m2_v_dials[i];
                }
            }
        }

        // --- 3. Step Method 3 (Evolution) ---
        for (auto& org : population) {
            org.error = compute_avg_error(org.dials, org.baseline, dataset);
        }
        std::sort(population.begin(), population.end(), [](const Organism& a, const Organism& b) {
            return a.error < b.error;
        });
        std::normal_distribution<double> mutate(0.0, 0.15);
        std::uniform_int_distribution<int> pick_parent(0, 4);
        std::uniform_real_distribution<double> chance(0.0, 1.0);
        std::vector<Organism> next_pop;
        next_pop.reserve(100);
        for (int i = 0; i < 5; ++i) next_pop.push_back(population[i]); // Elitism
        while (next_pop.size() < 100) {
            Organism child = population[pick_parent(rng)];
            if (chance(rng) < 0.40) child.baseline += mutate(rng);
            for (int i = 0; i < NUM_PIXELS; ++i) {
                if (chance(rng) < 0.30) child.dials[i] += mutate(rng);
            }
            next_pop.push_back(child);
        }
        population = std::move(next_pop);

        // Display Scoreboard at intervals
        if (round == 0 || round == 5 || round == 10 || round == 20 || round == 35 || round == 49) {
            double err1 = compute_avg_error(m1_dials, m1_baseline, dataset);
            double err2 = compute_avg_error(m2_dials, m2_baseline, dataset);
            double err3 = population[0].error;

            std::cout << "  " << std::setw(3) << (round + 1) << "  | "
                      << std::fixed << std::setprecision(4) << err1 << "  " << mini_bar(err1) << "   | "
                      << std::fixed << std::setprecision(4) << err2 << "  " << mini_bar(err2) << "   | "
                      << std::fixed << std::setprecision(4) << err3 << "  " << mini_bar(err3) << "\n";
        }
    }

    std::cout << "------------------------------------------------------------------------------------\n\n";

    // Test Island #5 (The mystery unidentifiable island from Project 2)
    std::vector<double> island_5(64, 0.0);
    // Draw the 1 with hook and base
    for (int r = 0; r < 8; ++r) island_5[r * 8 + 3] = 1.0;
    island_5[0 * 8 + 2] = 1.0;
    for (int c = 2; c <= 4; ++c) island_5[7 * 8 + c] = 1.0;

    auto test_prediction = [&](const std::string& name, const std::vector<double>& dials, double base) {
        double raw = base;
        for (int i = 0; i < NUM_PIXELS; ++i) raw += island_5[i] * dials[i];
        double conf = squash(raw);
        std::cout << "  " << std::setw(28) << std::left << name 
                  << " -> Confidence: " << std::fixed << std::setprecision(1) << (conf * 100.0) << "% "
                  << (conf >= 0.70 ? "[CONFIRMED 1]" : "[NOT 1]") << "\n";
    };

    std::cout << "=== FINAL PREDICTION ON UNIDENTIFIABLE ISLAND #5 ===\n";
    test_prediction("Method 1 (Basic Fixed)", m1_dials, m1_baseline);
    test_prediction("Method 2 (Momentum + Decay)", m2_dials, m2_baseline);
    test_prediction("Method 3 (Darwin Evolution)", population[0].dials, population[0].baseline);

    return 0;
}
