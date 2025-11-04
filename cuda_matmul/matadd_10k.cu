#include <cuda_runtime.h>

#include <iostream>
#include <vector>
#include <numeric>

constexpr int ROWS = 100;
constexpr int COLS = 100;
constexpr int TOTAL = ROWS * COLS; // 10,000 elements

__global__ void MatAdd(const float* A, const float* B, float* C, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < ROWS && col < COLS) {
        const int idx = row * cols + col;
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    std::vector<float> hA(TOTAL);
    std::vector<float> hB(TOTAL);
    std::vector<float> hC(TOTAL, 0.0f);

    std::iota(hA.begin(), hA.end(), 0.0f);
    std::iota(hB.begin(), hB.end(), 10000.0f);

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    const size_t bytes = TOTAL * sizeof(float);
    cudaMalloc(&dA, bytes);
    cudaMalloc(&dB, bytes);
    cudaMalloc(&dC, bytes);

    cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB.data(), bytes, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((COLS + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (ROWS + threadsPerBlock.y - 1) / threadsPerBlock.y);

    MatAdd<<<numBlocks, threadsPerBlock>>>(dA, dB, dC, COLS);
    cudaDeviceSynchronize();

    cudaMemcpy(hC.data(), dC, bytes, cudaMemcpyDeviceToHost);

    std::cout << "First 5 results:\n";
    for (int i = 0; i < 5; ++i) ;
    std::cout << "\nElement 9999: " << hC[TOTAL - 1] << '\n';

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    return 0;
}
