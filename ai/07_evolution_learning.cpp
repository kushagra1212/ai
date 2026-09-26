#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <random>
#include <algorithm>
#include <string>

const int GRID_SIZE = 8;
const int NUM_PIXELS = 64;
const int POPULATION_SIZE = 100; // 100 competing models
const int TOP_SURVIVORS = 5;      // Top 5 winners survive to reproduce

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

std::string render_progress_bar(double error) {
    double accuracy = std::max(0.0, 1.0 - error * 2.0);
    int filled = static_cast<int>(accuracy * 20);
    if (filled > 20) filled = 20;
    std::string bar = "[";
    for (int i = 0; i < 20; ++i) bar += (i < filled) ? "#" : ".";
    bar += "]";
    return bar;
}

// One individual model in the population
struct Individual {
    std::vector<double> dials;
    double baseline;
    double total_error; // Lower is better!

    Individual() {
        dials = std::vector<double>(NUM_PIXELS, 0.0);
        baseline = 0.0;
        total_error = 999.0;
    }

    double predict(const std::vector<double>& pixels) const {
        double raw = baseline;
        for (int i = 0; i < NUM_PIXELS; ++i) raw += pixels[i] * dials[i];
        return squash(raw);
    }

    // Calculates how bad this individual performs across all test images
    void evaluate(const std::vector<Sample>& dataset) {
        total_error = 0.0;
        for (const auto& sample : dataset) {
            double guess = predict(sample.pixels);
            total_error += std::abs(guess - sample.target);
        }
        total_error /= dataset.size();
    }
};

class EvolutionLearner {
public:
    std::vector<Individual> population;
    std::mt19937 rng;

    EvolutionLearner() : rng(42) { // Fixed seed for reproducible learning
        population.resize(POPULATION_SIZE);

        std::uniform_real_distribution<double> init_dist(-0.5, 0.5);

        // Generation 0: Random dials for all 100 individuals
        for (auto& ind : population) {
            ind.baseline = init_dist(rng);
            for (int i = 0; i < NUM_PIXELS; ++i) {
                ind.dials[i] = init_dist(rng);
            }
        }
    }

    Individual train(const std::vector<Sample>& dataset, int generations, double mutation_amount) {
        std::cout << "------------------------------------------------------------------------------------\n";
        std::cout << " Generation | Best Error | Avg Error  | Worst Error | Champion Accuracy Bar\n";
        std::cout << "------------------------------------------------------------------------------------\n";

        std::normal_distribution<double> mutate_dist(0.0, mutation_amount);
        std::uniform_int_distribution<int> parent_dist(0, TOP_SURVIVORS - 1);
        std::uniform_real_distribution<double> chance(0.0, 1.0);

        for (int gen = 0; gen < generations; ++gen) {
            // 1. EVALUATE ALL 100 INDIVIDUALS
            double sum_error = 0.0;
            for (auto& ind : population) {
                ind.evaluate(dataset);
                sum_error += ind.total_error;
            }

            // 2. SORT POPULATION: Best (lowest error) at index 0, Worst at index 99
            std::sort(population.begin(), population.end(), [](const Individual& a, const Individual& b) {
                return a.total_error < b.total_error;
            });

            double best_err = population[0].total_error;
            double avg_err = sum_error / POPULATION_SIZE;
            double worst_err = population[POPULATION_SIZE - 1].total_error;

            if (gen % 5 == 0 || gen == generations - 1 || best_err < 0.01) {
                std::cout << "   Gen " << std::setw(3) << gen << "  |   "
                          << std::fixed << std::setprecision(4) << best_err << "   |   "
                          << std::fixed << std::setprecision(4) << avg_err << "   |   "
                          << std::fixed << std::setprecision(4) << worst_err << "    | "
                          << render_progress_bar(best_err) << "\n";
            }

            if (best_err < 0.005) {
                std::cout << "------------------------------------------------------------------------------------\n";
                std::cout << ">>> Champion found at Generation " << gen << "! Error near zero without calculus! <<<\n\n";
                break;
            }

            // 3. CREATE NEXT GENERATION
            std::vector<Individual> next_gen;
            next_gen.reserve(POPULATION_SIZE);

            // A) ELITISM: The top 5 Champions survive directly into the next generation untouched
            for (int i = 0; i < TOP_SURVIVORS; ++i) {
                next_gen.push_back(population[i]);
            }

            // B) REPRODUCTION & MUTATION: The remaining 95 are children of the Champions
            while (next_gen.size() < POPULATION_SIZE) {
                // Pick one of the top 5 champions as the parent
                int parent_idx = parent_dist(rng);
                Individual child = population[parent_idx];

                // Mutate baseline with 40% probability
                if (chance(rng) < 0.40) {
                    child.baseline += mutate_dist(rng);
                }

                // Mutate some dials with 30% probability per dial
                for (int i = 0; i < NUM_PIXELS; ++i) {
                    if (chance(rng) < 0.30) {
                        child.dials[i] += mutate_dist(rng);
                    }
                }

                next_gen.push_back(child);
            }

            population = std::move(next_gen);
        }

        return population[0]; // Return the grand champion
    }
};

int main() {
    std::cout << "====================================================================================\n";
    std::cout << " PROGRAM B: DARWINIAN EVOLUTION (SURVIVAL OF THE FITTEST & MUTATIONS)\n";
    std::cout << "====================================================================================\n\n";

    // Dataset
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

    EvolutionLearner evolution;
    // 60 generations, mutation step = 0.15
    Individual champion = evolution.train(dataset, 60, 0.15);

    std::cout << "\n--- Visual Map of Champion's Learned Dials (Zero Calculus Used!) ---\n";
    for (int r = 0; r < GRID_SIZE; ++r) {
        for (int c = 0; c < GRID_SIZE; ++c) {
            double val = champion.dials[r * GRID_SIZE + c];
            if (val > 0.8)       std::cout << " +++ ";
            else if (val > 0.2)  std::cout << "  +  ";
            else if (val < -0.8) std::cout << " --- ";
            else if (val < -0.2) std::cout << "  -  ";
            else                 std::cout << "  .  ";
        }
        std::cout << "\n";
    }
    std::cout << "Champion Baseline: " << champion.baseline << "\n\n";

    std::cout << "--- TESTING CHAMPION ON DATASET ---\n";
    for (const auto& sample : dataset) {
        double conf = champion.predict(sample.pixels);
        std::cout << "  Sample: " << std::setw(18) << std::left << sample.label
                  << " | Target: " << sample.target
                  << " | Prediction: " << std::fixed << std::setprecision(1) << (conf * 100.0) << "% "
                  << (conf >= 0.70 ? "[Match: 1]" : "[Match: Not 1]") << "\n";
    }

    return 0;
}
