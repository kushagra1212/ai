#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <random>
#include <algorithm>
#include <string>

const int NUM_PIXELS = 64;

struct Sample {
    std::vector<double> pixels;
    double target;
};

double squash(double z) {
    if (z > 50.0) return 1.0;
    if (z < -50.0) return 0.0;
    return 1.0 / (1.0 + std::exp(-z));
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

double test_island_5(const std::vector<double>& dials, double baseline) {
    std::vector<double> island_5(64, 0.0);
    for (int r = 0; r < 8; ++r) island_5[r * 8 + 3] = 1.0;
    island_5[0 * 8 + 2] = 1.0;
    for (int c = 2; c <= 4; ++c) island_5[7 * 8 + c] = 1.0;

    double raw = baseline;
    for (int i = 0; i < NUM_PIXELS; ++i) raw += island_5[i] * dials[i];
    return squash(raw);
}

int main() {
    // Dataset
    std::vector<Sample> dataset;
    std::vector<double> one_center(64, 0.0);
    for (int r = 0; r < 8; ++r) one_center[r * 8 + 3] = 1.0;
    dataset.push_back({one_center, 1.0});

    std::vector<double> one_hook = one_center;
    one_hook[0 * 8 + 2] = 1.0;
    dataset.push_back({one_hook, 1.0});

    std::vector<double> one_base = one_center;
    for (int c = 2; c <= 4; ++c) one_base[7 * 8 + c] = 1.0;
    dataset.push_back({one_base, 1.0});

    std::vector<double> zero_box(64, 0.0);
    for (int i = 0; i < 8; ++i) {
        zero_box[0 * 8 + i] = 1.0; zero_box[7 * 8 + i] = 1.0;
        zero_box[i * 8 + 0] = 1.0; zero_box[i * 8 + 7] = 1.0;
    }
    dataset.push_back({zero_box, 0.0});

    std::vector<double> letter_L(64, 0.0);
    for (int r = 0; r < 8; ++r) letter_L[r * 8 + 1] = 1.0;
    for (int c = 1; c < 7; ++c) letter_L[7 * 8 + c] = 1.0;
    dataset.push_back({letter_L, 0.0});

    std::vector<double> plus_sign(64, 0.0);
    for (int i = 0; i < 8; ++i) {
        plus_sign[3 * 8 + i] = 1.0;
        plus_sign[i * 8 + 3] = 1.0;
    }
    dataset.push_back({plus_sign, 0.0});

    std::cout << "====================================================================================\n";
    std::cout << " BENCHMARK TEST: PROGRAM 05/06 vs PROGRAM 07 vs PROGRAM 09\n";
    std::cout << "====================================================================================\n\n";

    // -------------------------------------------------------------------------
    // 1. BENCHMARK PROGRAM 05/06 (Momentum + Decay)
    // -------------------------------------------------------------------------
    auto start_05 = std::chrono::high_resolution_clock::now();
    std::vector<double> dials_05(NUM_PIXELS, 0.0);
    double baseline_05 = 0.0;
    std::vector<double> v_dials_05(NUM_PIXELS, 0.0);
    double v_baseline_05 = 0.0;
    int passes_05 = 0;

    for (int p = 0; p < 60; ++p) {
        passes_05++;
        double step = 0.40 / (1.0 + 0.05 * p);
        for (const auto& s : dataset) {
            double raw = baseline_05;
            for (int i = 0; i < NUM_PIXELS; ++i) raw += s.pixels[i] * dials_05[i];
            double diff = squash(raw) - s.target;
            v_baseline_05 = 0.80 * v_baseline_05 - step * diff;
            baseline_05 += v_baseline_05;
            for (int i = 0; i < NUM_PIXELS; ++i) {
                if (s.pixels[i] > 0.0) {
                    v_dials_05[i] = 0.80 * v_dials_05[i] - step * diff * s.pixels[i];
                    dials_05[i] += v_dials_05[i];
                }
            }
        }
        if (compute_avg_error(dials_05, baseline_05, dataset) < 0.005) break;
    }
    auto end_05 = std::chrono::high_resolution_clock::now();
    auto time_05 = std::chrono::duration_cast<std::chrono::microseconds>(end_05 - start_05).count();

    // -------------------------------------------------------------------------
    // 2. BENCHMARK PROGRAM 07 (Evolution - Random Hat Draw)
    // -------------------------------------------------------------------------
    auto start_07 = std::chrono::high_resolution_clock::now();
    struct Org07 {
        std::vector<double> dials;
        double baseline;
        double error;
    };
    std::vector<Org07> pop_07(100);
    std::mt19937 rng_07(42);
    std::uniform_real_distribution<double> init_d(-0.5, 0.5);
    for (auto& o : pop_07) {
        o.dials.resize(NUM_PIXELS);
        o.baseline = init_d(rng_07);
        for (int i = 0; i < NUM_PIXELS; ++i) o.dials[i] = init_d(rng_07);
        o.error = 999.0;
    }
    int gens_07 = 0;
    for (int g = 0; g < 60; ++g) {
        gens_07++;
        for (auto& o : pop_07) o.error = compute_avg_error(o.dials, o.baseline, dataset);
        std::sort(pop_07.begin(), pop_07.end(), [](const Org07& a, const Org07& b) { return a.error < b.error; });
        if (pop_07[0].error < 0.005) break;

        std::normal_distribution<double> mutate(0.0, 0.15);
        std::uniform_int_distribution<int> pick_parent(0, 4);
        std::uniform_real_distribution<double> chance(0.0, 1.0);
        std::vector<Org07> next_pop;
        next_pop.reserve(100);
        for (int i = 0; i < 5; ++i) next_pop.push_back(pop_07[i]);
        while (next_pop.size() < 100) {
            Org07 child = pop_07[pick_parent(rng_07)];
            if (chance(rng_07) < 0.40) child.baseline += mutate(rng_07);
            for (int i = 0; i < NUM_PIXELS; ++i) {
                if (chance(rng_07) < 0.30) child.dials[i] += mutate(rng_07);
            }
            next_pop.push_back(child);
        }
        pop_07 = std::move(next_pop);
    }
    auto end_07 = std::chrono::high_resolution_clock::now();
    auto time_07 = std::chrono::duration_cast<std::chrono::microseconds>(end_07 - start_07).count();

    // -------------------------------------------------------------------------
    // 3. BENCHMARK PROGRAM 09 (Cloning - 19 per champion)
    // -------------------------------------------------------------------------
    auto start_09 = std::chrono::high_resolution_clock::now();
    struct Org09 {
        std::vector<double> dials;
        double baseline;
        double error;
    };
    std::vector<Org09> pop_09(100);
    std::mt19937 rng_09(42);
    for (auto& o : pop_09) {
        o.dials.resize(NUM_PIXELS);
        o.baseline = init_d(rng_09);
        for (int i = 0; i < NUM_PIXELS; ++i) o.dials[i] = init_d(rng_09);
        o.error = 999.0;
    }
    int gens_09 = 0;
    for (int g = 0; g < 60; ++g) {
        gens_09++;
        for (auto& o : pop_09) o.error = compute_avg_error(o.dials, o.baseline, dataset);
        std::sort(pop_09.begin(), pop_09.end(), [](const Org09& a, const Org09& b) { return a.error < b.error; });
        if (pop_09[0].error < 0.005) break;

        std::normal_distribution<double> mutate(0.0, 0.15);
        std::uniform_real_distribution<double> chance(0.0, 1.0);
        std::vector<Org09> next_pop;
        next_pop.reserve(100);
        for (int i = 0; i < 5; ++i) next_pop.push_back(pop_09[i]);
        for (int p = 0; p < 5; ++p) {
            for (int c = 0; c < 19; ++c) {
                Org09 child = pop_09[p];
                if (chance(rng_09) < 0.25) child.baseline += mutate(rng_09);
                for (int i = 0; i < NUM_PIXELS; ++i) {
                    if (chance(rng_09) < 0.25) child.dials[i] += mutate(rng_09);
                }
                next_pop.push_back(child);
            }
        }
        pop_09 = std::move(next_pop);
    }
    auto end_09 = std::chrono::high_resolution_clock::now();
    auto time_09 = std::chrono::duration_cast<std::chrono::microseconds>(end_09 - start_09).count();

    // -------------------------------------------------------------------------
    // RESULTS TABLE
    // -------------------------------------------------------------------------
    std::cout << "------------------------------------------------------------------------------------\n";
    std::cout << " Method                  | Rounds/Gens | Time (Microseconds) | Final Error | Island #5\n";
    std::cout << "------------------------------------------------------------------------------------\n";
    std::cout << " 05/06 (Momentum+Decay)  |  " << std::setw(3) << passes_05 
              << " passes  | " << std::setw(11) << time_05 << " us       | " 
              << std::fixed << std::setprecision(5) << compute_avg_error(dials_05, baseline_05, dataset) 
              << "     | " << std::fixed << std::setprecision(1) << (test_island_5(dials_05, baseline_05) * 100.0) << "%\n";

    std::cout << " 07 (Evolution Random)   |  " << std::setw(3) << gens_07 
              << " gens    | " << std::setw(11) << time_07 << " us       | " 
              << std::fixed << std::setprecision(5) << pop_07[0].error 
              << "     | " << std::fixed << std::setprecision(1) << (test_island_5(pop_07[0].dials, pop_07[0].baseline) * 100.0) << "%\n";

    std::cout << " 09 (Cloning Quota 19)   |  " << std::setw(3) << gens_09 
              << " gens    | " << std::setw(11) << time_09 << " us       | " 
              << std::fixed << std::setprecision(5) << pop_09[0].error 
              << "     | " << std::fixed << std::setprecision(1) << (test_island_5(pop_09[0].dials, pop_09[0].baseline) * 100.0) << "%\n";
    std::cout << "------------------------------------------------------------------------------------\n";

    return 0;
}
