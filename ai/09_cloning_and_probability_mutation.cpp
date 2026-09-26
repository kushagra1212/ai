#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <random>
#include <algorithm>
#include <string>

const int GRID_SIZE = 8;
const int NUM_PIXELS = 64;
const int POPULATION_SIZE = 100;
const int TOP_PICKS = 5;
const int COPIES_PER_PICK = 19; // 5 * 19 = 95 copies (+ 5 original = 100)

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

struct Individual {
    std::vector<double> dials;
    double baseline;
    double total_error;

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

    void evaluate(const std::vector<Sample>& dataset) {
        total_error = 0.0;
        for (const auto& sample : dataset) {
            double guess = predict(sample.pixels);
            total_error += std::abs(guess - sample.target);
        }
        total_error /= dataset.size();
    }
};

class CloneMutateLearner {
public:
    std::vector<Individual> population;
    std::mt19937 rng;

    CloneMutateLearner() : rng(42) {
        population.resize(POPULATION_SIZE);
        std::uniform_real_distribution<double> init_d(-0.5, 0.5);

        // Generation 0 starts with random dials
        for (auto& ind : population) {
            ind.baseline = init_d(rng);
            for (int i = 0; i < NUM_PIXELS; ++i) ind.dials[i] = init_d(rng);
        }
    }

    Individual train(const std::vector<Sample>& dataset, int generations, 
                     double mutation_prob, double mutation_amount) {
        
        std::cout << "------------------------------------------------------------------------------------\n";
        std::cout << " Gen | Best Error | Avg Error  | Worst Error | Champion Accuracy Bar\n";
        std::cout << "------------------------------------------------------------------------------------\n";

        std::normal_distribution<double> mutate_dist(0.0, mutation_amount);
        std::uniform_real_distribution<double> chance(0.0, 1.0);

        for (int gen = 0; gen < generations; ++gen) {
            // 1. Evaluate all 100 individuals
            double sum_error = 0.0;
            for (auto& ind : population) {
                ind.evaluate(dataset);
                sum_error += ind.total_error;
            }

            // 2. Sort from best (lowest error) to worst
            std::sort(population.begin(), population.end(), [](const Individual& a, const Individual& b) {
                return a.total_error < b.total_error;
            });

            double best_err = population[0].total_error;
            double avg_err = sum_error / POPULATION_SIZE;
            double worst_err = population[POPULATION_SIZE - 1].total_error;

            if (gen % 5 == 0 || gen == generations - 1 || best_err < 0.01) {
                std::cout << " " << std::setw(3) << gen << " |   "
                          << std::fixed << std::setprecision(4) << best_err << "   |   "
                          << std::fixed << std::setprecision(4) << avg_err << "   |   "
                          << std::fixed << std::setprecision(4) << worst_err << "    | "
                          << render_progress_bar(best_err) << "\n";
            }

            if (best_err < 0.005) {
                std::cout << "------------------------------------------------------------------------------------\n";
                std::cout << ">>> Champion found at Generation " << gen << "! Goal achieved successfully! <<<\n\n";
                break;
            }

            // 3. BUILD NEXT GENERATION:
            // - Keep the 5 champions untouched
            // - Make exactly 19 copies of each of the 5 champions (5 * 19 = 95 copies)
            // - Update each clone's dials and baseline using mutation probability
            std::vector<Individual> next_gen;
            next_gen.reserve(POPULATION_SIZE);

            // A) Keep the 5 Champions (Elites)
            for (int i = 0; i < TOP_PICKS; ++i) {
                next_gen.push_back(population[i]);
            }

            // B) For each Champion, create 19 copies with probabilistic mutation
            for (int p = 0; p < TOP_PICKS; ++p) {
                for (int c = 0; c < COPIES_PER_PICK; ++c) {
                    Individual child = population[p]; // Exact clone of champion #p

                    // Mutate baseline with given probability
                    if (chance(rng) < mutation_prob) {
                        child.baseline += mutate_dist(rng);
                    }

                    // Walk through all 64 dials: mutate each dial with given probability
                    for (int i = 0; i < NUM_PIXELS; ++i) {
                        if (chance(rng) < mutation_prob) {
                            child.dials[i] += mutate_dist(rng);
                        }
                    }

                    next_gen.push_back(child);
                }
            }

            population = std::move(next_gen);
        }

        return population[0]; // The champion
    }
};

int main() {
    std::cout << "====================================================================================\n";
    std::cout << " PROGRAM: 5 TOP PICKS -> 95 CLONED COPIES (19 PER CHAMPION) + PROBABILITY MUTATIONS\n";
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

    CloneMutateLearner learner;

    // Train with Mutation Probability = 25% (0.25) per dial, Mutation Amount = 0.15
    Individual champion = learner.train(dataset, 60, 0.25, 0.15);

    std::cout << "--- Champion's Learned Dials ---\n";
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

    // Test on the unidentifiable Island #5 from Project 2!
    std::vector<double> island_5(64, 0.0);
    for (int r = 0; r < 8; ++r) island_5[r * 8 + 3] = 1.0;
    island_5[0 * 8 + 2] = 1.0;
    for (int c = 2; c <= 4; ++c) island_5[7 * 8 + c] = 1.0;

    std::cout << "--- TESTING CHAMPION ON DATASET & ISLAND #5 ---\n";
    for (const auto& sample : dataset) {
        double conf = champion.predict(sample.pixels);
        std::cout << "  Sample: " << std::setw(18) << std::left << sample.label
                  << " | Target: " << sample.target
                  << " | Prediction: " << std::fixed << std::setprecision(1) << (conf * 100.0) << "% "
                  << (conf >= 0.70 ? "[Match: 1]" : "[Match: Not 1]") << "\n";
    }

    double conf_5 = champion.predict(island_5);
    std::cout << "\n  Island #5 (Handwritten 1) -> Confidence: " 
              << std::fixed << std::setprecision(1) << (conf_5 * 100.0) << "% "
              << (conf_5 >= 0.70 ? "[CONFIRMED 1!]" : "[NOT 1]") << "\n";

    return 0;
}
