#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>

// -----------------------------------------------------------------------------
// Activation slopes
// -----------------------------------------------------------------------------
double sigmoid_slope(double z) {
    double a = 1.0 / (1.0 + std::exp(-z));
    return a * (1.0 - a); // Maximum is 0.25
}

double relu_slope(double z) {
    return (z > 0.0) ? 1.0 : 0.0;
}

double leaky_relu_slope(double z, double alpha = 0.01) {
    return (z > 0.0) ? 1.0 : alpha; // Never 0.0!
}

// Simulate blame traveling backward through 4 layers
double propagate_blame_sigmoid(double initial_blame, const std::vector<double>& raw_scores) {
    double blame = initial_blame;
    for (double z : raw_scores) {
        blame *= sigmoid_slope(z); // Multiplies by <= 0.25 at each layer
    }
    return blame;
}

double propagate_blame_relu(double initial_blame, const std::vector<double>& raw_scores) {
    double blame = initial_blame;
    for (double z : raw_scores) {
        blame *= relu_slope(z); // Multiplies by 1.0 or 0.0
    }
    return blame;
}

double propagate_blame_leaky(double initial_blame, const std::vector<double>& raw_scores, double alpha = 0.01) {
    double blame = initial_blame;
    for (double z : raw_scores) {
        blame *= leaky_relu_slope(z, alpha); // Multiplies by 1.0 or 0.01
    }
    return blame;
}

int main() {
    std::cout << "=========================================================================\n";
    std::cout << " FIRST-PRINCIPLES PROOF: SOLVING THE SLOPE MULTIPLICATION PROBLEM\n";
    std::cout << "=========================================================================\n\n";

    double initial_blame = 1.0; // The output made an error of 1.0

    // Scenario 1: All neurons are active (positive raw scores)
    std::vector<double> active_scores = {1.5, 0.8, 2.1, 0.5};

    std::cout << "SCENARIO 1: All 4 layers are active (Raw scores > 0)\n";
    std::cout << "---------------------------------------------------------\n";
    double sig1 = propagate_blame_sigmoid(initial_blame, active_scores);
    double relu1 = propagate_blame_relu(initial_blame, active_scores);
    double leaky1 = propagate_blame_leaky(initial_blame, active_scores);

    std::cout << "  Initial Blame at Output : " << initial_blame << "\n";
    std::cout << "  1. S-Curve (Sigmoid)     : " << std::fixed << std::setprecision(6) << sig1 
              << "  (Shrank by " << (1.0 - sig1) * 100.0 << "%!)\n";
    std::cout << "  2. Standard Ramp (ReLU)  : " << std::fixed << std::setprecision(6) << relu1 
              << "  (100% of signal preserved! 1.0 * 1.0 * 1.0 * 1.0)\n";
    std::cout << "  3. Leaky Ramp            : " << std::fixed << std::setprecision(6) << leaky1 
              << "  (100% of signal preserved! 1.0 * 1.0 * 1.0 * 1.0)\n\n";

    // Scenario 2: Layer 2 happens to be negative (z = -1.2)
    std::vector<double> mixed_scores = {1.5, 0.8, -1.2, 0.5};

    std::cout << "SCENARIO 2: Layer 2 goes negative (Raw scores: [1.5, 0.8, -1.2, 0.5])\n";
    std::cout << "---------------------------------------------------------\n";
    double sig2 = propagate_blame_sigmoid(initial_blame, mixed_scores);
    double relu2 = propagate_blame_relu(initial_blame, mixed_scores);
    double leaky2 = propagate_blame_leaky(initial_blame, mixed_scores);

    std::cout << "  Initial Blame at Output : " << initial_blame << "\n";
    std::cout << "  1. S-Curve (Sigmoid)     : " << std::fixed << std::setprecision(6) << sig2 
              << "  (Still severely shrunk)\n";
    std::cout << "  2. Standard Ramp (ReLU)  : " << std::fixed << std::setprecision(6) << relu2 
              << "  <-- KILLED! Multiplied by 0.0! Permanent death!\n";
    std::cout << "  3. Leaky Ramp            : " << std::fixed << std::setprecision(6) << leaky2 
              << "  <-- ALIVE! Multiplied by 0.01! Can recover!\n\n";

    std::cout << "=========================================================================\n";
    std::cout << " CONCLUSION:\n";
    std::cout << " 1. For positive neurons: Slope is 1.0 -> No shrinking (multiplies by 1.0)\n";
    std::cout << " 2. For negative neurons: Slope is 0.01 -> Never 0.0 -> No severed wires!\n";
    std::cout << "=========================================================================\n";

    return 0;
}
