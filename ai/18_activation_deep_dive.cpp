#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

// -----------------------------------------------------------------------------
// Activation Functions and Their Slopes
// -----------------------------------------------------------------------------

// 1. Sigmoid (S-Curve)
double sigmoid(double z) {
    return 1.0 / (1.0 + std::exp(-z));
}
double sigmoid_slope(double z) {
    double a = sigmoid(z);
    return a * (1.0 - a); // Max = 0.25 at z=0
}

// 2. Tanh (Centered S-Curve, -1 to +1)
double tanh_act(double z) {
    return std::tanh(z);
}
double tanh_slope(double z) {
    double a = std::tanh(z);
    return 1.0 - a * a; // Max = 1.0 at z=0, but drops fast!
}

// 3. ReLU (Standard Ramp)
double relu(double z) {
    return (z > 0.0) ? z : 0.0;
}
double relu_slope(double z) {
    return (z > 0.0) ? 1.0 : 0.0;
}

// 4. Leaky ReLU (Leaky Ramp)
double leaky_relu(double z, double alpha = 0.01) {
    return (z > 0.0) ? z : alpha * z;
}
double leaky_relu_slope(double z, double alpha = 0.01) {
    return (z > 0.0) ? 1.0 : alpha;
}

// 5. Modern Smooth Ramp (SiLU / Swish: z * sigmoid(z))
double silu(double z) {
    return z * sigmoid(z);
}
double silu_slope(double z) {
    double s = sigmoid(z);
    return s + z * s * (1.0 - s);
}

int main() {
    std::cout << "========================================================================================\n";
    std::cout << " DEEP-DIVE: COMPARING ACTIVATIONS & SLOPES ACROSS DIFFERENT RAW SCORES\n";
    std::cout << "========================================================================================\n\n";

    std::vector<double> test_z = {-3.0, -1.0, 0.0, 1.0, 3.0};

    std::cout << std::setw(8) << "Raw (z)" 
              << std::setw(15) << "Sigmoid Slope" 
              << std::setw(15) << "Tanh Slope" 
              << std::setw(15) << "ReLU Slope" 
              << std::setw(18) << "Leaky ReLU Slope" 
              << std::setw(15) << "SiLU Slope\n";
    std::cout << "----------------------------------------------------------------------------------------\n";

    for (double z : test_z) {
        std::cout << std::fixed << std::setprecision(4)
                  << std::setw(8) << z 
                  << std::setw(15) << sigmoid_slope(z)
                  << std::setw(15) << tanh_slope(z)
                  << std::setw(15) << relu_slope(z)
                  << std::setw(18) << leaky_relu_slope(z)
                  << std::setw(15) << silu_slope(z)
                  << "\n";
    }

    std::cout << "\n========================================================================================\n";
    std::cout << "KEY FIRST-PRINCIPLES INSIGHTS:\n";
    std::cout << "1. Tanh has slope 1.0 at z=0, BUT at z=3.0 its slope is 0.0099 (collapses!)\n";
    std::cout << "2. ReLU slope is strictly 0.0 for all negative z (permanent death!)\n";
    std::cout << "3. Leaky ReLU stays at 0.01 for negative z (never completely dead!)\n";
    std::cout << "4. SiLU / Modern AI functions are smooth everywhere with a gentle non-zero dip!\n";
    std::cout << "========================================================================================\n";

    return 0;
}
