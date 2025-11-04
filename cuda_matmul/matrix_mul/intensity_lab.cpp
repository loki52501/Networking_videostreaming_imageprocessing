#include <iomanip>
#include <iostream>
#include <limits>
#include <string>

// Computes arithmetic intensity for GEMM-like workloads and classifies them as math- vs memory-bound.
int main() {
    std::cout << "Arithmetic Intensity Explorer\n";
    std::cout << "--------------------------------\n";

    double M = 0.0, N = 0.0, K = 0.0;
    double element_bytes = 2.0;  // default for FP16
    double target_ratio = 138.9; // FLOPS per byte (V100 example)

    std::cout << "Enter M (rows of A/C): ";
    std::cin >> M;
    std::cout << "Enter N (cols of B/C): ";
    std::cin >> N;
    std::cout << "Enter K (shared dim): ";
    std::cin >> K;

    std::cout << "Enter element size in bytes (default 2.0): ";
    if (!(std::cin >> element_bytes)) {
        std::cin.clear();
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        element_bytes = 2.0;
    }

    std::cout << "Enter GPU FLOPS:byte target (default 138.9): ";
    if (!(std::cin >> target_ratio)) {
        std::cin.clear();
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        target_ratio = 138.9;
    }

    const double flops = 2.0 * M * N * K;
    const double bytes = element_bytes * (M * K + K * N + M * N);
    const double intensity = flops / bytes;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Total FLOPS            : " << flops << '\n';
    std::cout << "Bytes accessed (model) : " << bytes << '\n';
    std::cout << "Arithmetic intensity   : " << intensity << " FLOPS/B\n";

    if (intensity >= target_ratio) {
        std::cout << "Likely math-bound on target GPU." << '\n';
    } else {
        std::cout << "Likely memory-bound on target GPU." << '\n';
    }

    std::cout << "\nTry varying dimensions to see when intensity crosses the threshold.\n";
    return 0;
}