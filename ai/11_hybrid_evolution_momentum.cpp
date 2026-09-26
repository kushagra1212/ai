#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <random>
#include <algorithm>
#include <string>

const int GRID_SIZE = 8;
const int NUM_PIXELS = 64;
const int TOP_PICKS = 5;
const int COPIES_PER_PICK = 19; // 5 * 19 = 95 copies (+ 5 original = 100)
const int POPULATION_SIZE = TOP_PICKS + (TOP_PICKS * COPIES_PER_PICK);

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

// Each individual has DIALS (Position) AND VELOCITIES (Speed from Momentum!)
struct Individual {
    std::vector<double> dials;
    double baseline;

    std::vector<double> v_dials; // Inherited momentum velocity
    double v_baseline;

    double error;

    Individual() {
        dials = std::vector<double>(NUM_PIXELS, 0.0);
        baseline = 0.0;
        v_dials = std::vector<double>(NUM_PIXELS, 0.0);
        v_baseline = 0.0;
        error = 999.0;
    }

    double predict(const std::vector<double>& pixels) const {
        double raw = baseline;
        for (int i = 0; i < NUM_PIXELS; ++i) raw += pixels[i] * dials[i];
        return squash(raw);
    }

    void evaluate(const std::vector<Sample>& dataset) {
        double total = 0.0;
        for (const auto& s : dataset) {
            total += std::abs(predict(s.pixels) - s.target);
        }
        error = total / dataset.size();
    }

    // 06 Feature: Individual practices with Momentum + Decay for 1 or 2 quick passes
    void polish_with_momentum(const std::vector<Sample>& dataset, double step_size, double friction) {
        for (const auto& s : dataset) {
            double raw = baseline;
            for (int i = 0; i < NUM_PIXELS; ++i) raw += s.pixels[i] * dials[i];
            double diff = squash(raw) - s.target;

            v_baseline = friction * v_baseline - step_size * diff;
            baseline += v_baseline;

            for (int i = 0; i < NUM_PIXELS; ++i) {
                if (s.pixels[i] > 0.0) {
                    v_dials[i] = friction * v_dials[i] - step_size * diff * s.pixels[i];
                    dials[i] += v_dials[i];
                }
            }
        }
    }
};

class HybridLearner {
public:
    std::vector<Individual> population;
    std::mt19937 rng;

    HybridLearner() : rng(42) {
        population.resize(POPULATION_SIZE);
        std::uniform_real_distribution<double> init_d(-0.5, 0.5);

        for (auto& ind : population) {
            ind.baseline = init_d(rng);
            for (int i = 0; i < NUM_PIXELS; ++i) ind.dials[i] = init_d(rng);
        }
    }

    Individual train(const std::vector<Sample>& dataset, int max_generations, 
                     double initial_step, double decay, double friction, 
                     double mutation_prob, double mutation_amount) {

        std::cout << "------------------------------------------------------------------------------------\n";
        std::cout << " Gen | Step Size | Best Error | Avg Error  | Worst Error | Champion Accuracy Bar\n";
        std::cout << "------------------------------------------------------------------------------------\n";

        std::normal_distribution<double> mutate_dist(0.0, mutation_amount);
        std::uniform_real_distribution<double> chance(0.0, 1.0);

        for (int gen = 0; gen < max_generations; ++gen) {
            // Decaying step size for this generation
            double current_step = initial_step / (1.0 + decay * gen);

            // 1. ALL 100 INDIVIDUALS DO 1 PASS OF MOMENTUM POLISHING (Program 06)
            for (auto& ind : population) {
                ind.polish_with_momentum(dataset, current_step, friction);
                ind.evaluate(dataset);
            }

            // 2. SORT POPULATION FROM BEST TO WORST (Program 09)
            std::sort(population.begin(), population.end(), [](const Individual& a, const Individual& b) {
                return a.error < b.error;
            });

            double best_err = population[0].error;
            double worst_err = population[POPULATION_SIZE - 1].error;
            double sum_err = 0.0;
            for (const auto& ind : population) sum_err += ind.error;
            double avg_err = sum_err / POPULATION_SIZE;

            std::cout << " " << std::setw(3) << gen << " |  "
                      << std::fixed << std::setprecision(4) << current_step << "   |   "
                      << std::fixed << std::setprecision(4) << best_err << "   |   "
                      << std::fixed << std::setprecision(4) << avg_err << "   |   "
                      << std::fixed << std::setprecision(4) << worst_err << "    | "
                      << render_progress_bar(best_err) << "\n";

            if (best_err < 0.002) {
                std::cout << "------------------------------------------------------------------------------------\n";
                std::cout << ">>> Hybrid Champion converged at Generation " << gen << "! Near-perfect accuracy! <<<\n\n";
                break;
            }

            // 3. REPRODUCTION: 5 CHAMPIONS + 19 CLONES EACH (Program 09)
            std::vector<Individual> next_gen;
            next_gen.reserve(POPULATION_SIZE);

            // Keep top 5 champions untouched
            for (int i = 0; i < TOP_PICKS; ++i) {
                next_gen.push_back(population[i]);
            }

            // 19 Clones per champion
            for (int p = 0; p < TOP_PICKS; ++p) {
                for (int c = 0; c < COPIES_PER_PICK; ++c) {
                    Individual child = population[p]; // Clones dials AND momentum velocities!

                    // Mutate baseline with probability
                    if (chance(rng) < mutation_prob) {
                        child.baseline += mutate_dist(rng);
                    }

                    // Mutate dials with probability
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

        return population[0];
    }
};

int main() {
    std::cout << "====================================================================================\n";
    std::cout << " PROGRAM 11: HYBRID (EVOLUTION 09 + MOMENTUM/DECAY 06 COMBINED)\n";
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

    HybridLearner hybrid;

    // Train with all features active:
    // Initial Step = 0.40, Decay = 0.05, Friction = 0.80, Mutate Prob = 0.20, Mutate Amount = 0.10
    Individual champion = hybrid.train(dataset, 30, 0.40, 0.05, 0.80, 0.20, 0.10);

    std::cout << "--- Hybrid Champion's Learned Dials ---\n";
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

    std::cout << "--- TESTING HYBRID CHAMPION ON ALL DATASET & ISLAND #5 ---\n";
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
