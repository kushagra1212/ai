#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

// S-curve squasher and its slope
double squash(double z) {
    return 1.0 / (1.0 + std::exp(-z));
}
double squash_slope(double a) {
    return a * (1.0 - a);
}

// The Ramp squasher and its slope
double ramp(double z) {
    return (z > 0.0) ? z : 0.0;
}
double ramp_slope(double z) {
    return (z > 0.0) ? 1.0 : 0.0;
}

int main() {
    std::cout << "====================================================================================\n";
    std::cout << " FIRST PRINCIPLES PROOF: S-CURVE (VANISHING) vs THE RAMP (PRESERVED)\n";
    std::cout << " We trace a blame signal of 1.0 backward through 5 layers of neurons!\n";
    std::cout << "====================================================================================\n\n";

    double initial_error = 1.0;

    // ---------------------------------------------------------------------------------
    // 1. TRACING BLAME BACKWARD USING S-CURVE (SIGMOID)
    // ---------------------------------------------------------------------------------
    std::cout << "--- METHOD 1: Using S-CURVE in all layers ---\n";
    double blame_sigmoid = initial_error;
    // Assume all neurons are in their BEST-CASE sensitivity (activation = 0.5)
    double max_sigmoid_slope = squash_slope(0.5); // 0.5 * (1 - 0.5) = 0.25

    for (int layer = 5; layer >= 1; --layer) {
        blame_sigmoid *= max_sigmoid_slope; // Multiplied by 0.25 at each layer!
        std::cout << "  Layer " << layer << " receives blame signal: " 
                  << std::fixed << std::setprecision(6) << blame_sigmoid;
        if (layer == 1) std::cout << "  <--- DIES TO NEAR ZERO! (Vanished!)";
        std::cout << "\n";
    }

    std::cout << "\n---------------------------------------------------------------------------------\n";

    // ---------------------------------------------------------------------------------
    // 2. TRACING BLAME BACKWARD USING THE RAMP (ReLU)
    // ---------------------------------------------------------------------------------
    std::cout << "--- METHOD 2: Using THE RAMP in middle layers ---\n";
    // Layer 5 (Output Judge) uses S-curve slope to start (0.25)
    double blame_ramp = initial_error * squash_slope(0.5);
    std::cout << "  Layer 5 receives blame signal: " 
              << std::fixed << std::setprecision(6) << blame_ramp << " (From finish line)\n";

    // Layers 4, 3, 2, 1 use THE RAMP (slope is exactly 1.0!)
    for (int layer = 4; layer >= 1; --layer) {
        double r_slope = ramp_slope(1.0); // 1.0 for active neurons
        blame_ramp *= r_slope;            // Multiplied by 1.0 at each layer!
        std::cout << "  Layer " << layer << " receives blame signal: " 
                  << std::fixed << std::setprecision(6) << blame_ramp;
        if (layer == 1) std::cout << "  <--- 100% FULL STRENGTH! (Alive!)";
        std::cout << "\n";
    }

    std::cout << "---------------------------------------------------------------------------------\n\n";

    std::cout << "SUMMARY AT LAYER 1 (The front inputs):\n";
    std::cout << "  - S-Curve signal strength:  " << std::fixed << std::setprecision(6) << blame_sigmoid << "\n";
    std::cout << "  - The Ramp signal strength: " << std::fixed << std::setprecision(6) << blame_ramp << "\n";
    std::cout << "  ===> The Ramp is " << (blame_ramp / blame_sigmoid) 
              << " TIMES STRONGER than the S-curve at the front layer!\n";

    return 0;
}
