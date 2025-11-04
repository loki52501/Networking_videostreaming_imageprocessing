#include <cuda_runtime.h>

#include <chrono>
#include <iostream>
#include <random>
#include <vector>

constexpr int TILE = 16;

// TODO: implement a simple CPU matmul (triple loop)
std::vector<float> cpu_matmul(const std::vector<float>& A,
                              const std::vector<float>& B,
                              int M, int N, int K) {
    (void)A; (void)B; (void)M; (void)N; (void)K;
    return {}; // replace with real implementation
}

// TODO: implement elementwise matrix addition on CPU
std::vector<float> cpu_add(const std::vector<float>& X,
                           const std::vector<float>& Y,
                           int rows, int cols) {
    (void)X; (void)Y; (void)rows; (void)cols;
    return {};
}

// TODO: write a tiled CUDA kernel for matmul (shared memory optional, but recommended)
__global__ void matmul_kernel(const float* A, const float* B, float* C,
                              int M, int N, int K) {
    // implement kernel body
}

// TODO: write a simple CUDA kernel for elementwise addition
__global__ void add_kernel(const float* X, const float* Y, float* Z, int total) {
    // implement kernel body
}

int main() {
    const int M = 512;
    const int N = 512;
    const int K = 512;

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> A(M * K), B(K * N);
    for (auto& v : A) v = dist(gen);
    for (auto& v : B) v = dist(gen);

    // TODO: run cpu_matmul and time it
    // TODO: run cpu_add on two matrices

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    cudaMalloc(&dA, sizeof(float) * M * K);
    cudaMalloc(&dB, sizeof(float) * K * N);
    cudaMalloc(&dC, sizeof(float) * M * N);
    cudaMemcpy(dA, A.data(), sizeof(float) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B.data(), sizeof(float) * K * N, cudaMemcpyHostToDevice);

    // TODO: launch matmul kernel and time with cudaEvent

    std::vector<float> C_cpu;      // fill from CPU path
    std::vector<float> C_gpu(M * N);
    cudaMemcpy(C_gpu.data(), dC, sizeof(float) * M * N, cudaMemcpyDeviceToHost);

    // TODO: compare CPU vs GPU matmul results (max abs diff)

    // TODO: set up addition buffers on device and run add_kernel

    // TODO: compare CPU vs GPU addition results

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    return 0;
}
