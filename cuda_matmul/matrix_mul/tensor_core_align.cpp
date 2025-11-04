#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <string>
#include <tuple>

struct AlignmentInfo {
    int element_bytes;
    int preferred_multiple;
};

int round_up(int value, int multiple) {
    if (multiple == 0) return value;
    const int remainder = value % multiple;
    if (remainder == 0) return value;
    return value + (multiple - remainder);
}

int main() {
    std::map<std::string, AlignmentInfo> table = {
        {"fp16", {2, 8}}, {"bf16", {2, 8}}, {"int8", {1, 16}},
        {"tf32", {4, 4}}, {"fp32", {4, 4}}, {"fp64", {8, 2}}
    };

    std::string dtype;
    int M = 0, N = 0, K = 0;

    std::cout << "Tensor Core Alignment Checker\n";
    std::cout << "--------------------------------\n";
    std::cout << "Data type (fp16/bf16/int8/tf32/fp32/fp64): ";
    std::cin >> dtype;

    auto it = table.find(dtype);
    if (it == table.end()) {
        std::cerr << "Unknown dtype. Using fp16 defaults.\n";
        it = table.find("fp16");
    }

    std::cout << "Enter M, N, K: ";
    std::cin >> M >> N >> K;

    const int mult = it->second.preferred_multiple;
    const int M_aligned = round_up(M, mult);
    const int N_aligned = round_up(N, mult);
    const int K_aligned = round_up(K, mult);

    const int elem_bytes = it->second.element_bytes;
    const long long elements_orig = static_cast<long long>(M) * N * K;
    const long long elements_aligned = static_cast<long long>(M_aligned) * N_aligned * K_aligned;
    const long long extra = elements_aligned - elements_orig;

    std::cout << "\nPreferred multiple (elements): " << mult << '\n';
    std::cout << "Aligned dims (M, N, K): " << M_aligned << ", " << N_aligned << ", " << K_aligned << '\n';
    std::cout << "Extra elements required: " << extra << '\n';
    std::cout << "Extra storage (bytes)  : " << extra * elem_bytes << '\n';
    std::cout << "Overhead (%)           : "
              << (elements_orig == 0 ? 0.0 : (100.0 * extra) / static_cast<double>(elements_orig)) << '\n';

    std::cout << "\nConsider padding tensors to these sizes to maximize Tensor Core efficiency.\n";
    return 0;
}