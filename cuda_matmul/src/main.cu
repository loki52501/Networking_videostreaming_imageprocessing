#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

struct Args {
    int M = 1024;
    int N = 1024;
    int K = 1024;
    std::string mode = "kernel"; // cpu | kernel | cublas
    std::string dtype = "fp32";   // fp32 only in starter
    unsigned seed = 42;
};

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string token(argv[i]);
        if (token == "--mode" && i + 1 < argc) {
            args.mode = argv[++i];
        } else if (token == "--dtype" && i + 1 < argc) {
            args.dtype = argv[++i];
        } else if (token == "--seed" && i + 1 < argc) {
            args.seed = static_cast<unsigned>(std::stoul(argv[++i]));
        } else if (args.M == 1024) {
            args.M = std::stoi(token);
        } else if (args.N == 1024) {
            args.N = std::stoi(token);
        } else if (args.K == 1024) {
            args.K = std::stoi(token);
        } else {
            std::cerr << "Unrecognized argument: " << token << '\n';
        }
    }
    return args;
}

std::vector<float> random_matrix(int rows, int cols, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> mat(rows * cols);
    for (auto& v : mat) v = dist(gen);
    return mat;
}

std::vector<float> cpu_matmul(const std::vector<float>& A,
                              const std::vector<float>& B,
                              int M, int N, int K) {
    std::vector<float> C(M * N, 0.0f);
    auto start = std::chrono::high_resolution_clock::now();
    for (int m = 0; m < M; ++m) {
        for (int k = 0; k < K; ++k) {
            const float a = A[m * K + k];
            for (int n = 0; n < N; ++n) {
                C[m * N + n] += a * B[k * N + n];
            }
        }
    }
    auto stop = std::chrono::high_resolution_clock::now();
    const double elapsed = std::chrono::duration<double>(stop - start).count();
    const double gflops = (2.0 * M * N * K) / (elapsed * 1e9);
    std::cout << "CPU matmul: " << elapsed * 1000.0 << " ms | " << gflops << " GFLOPS\n";
    return C;
}

constexpr int TILE = 16;

__global__ void matmul_kernel(const float* A, const float* B, float* C,
                              int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;

    for (int tile = 0; tile < (K + TILE - 1) / TILE; ++tile) {
        int a_col = tile * TILE + threadIdx.x;
        int b_row = tile * TILE + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
        __syncthreads();
        for (int k = 0; k < TILE; ++k) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

float gpu_kernel_matmul(const float* dA, const float* dB, float* dC,
                        int M, int N, int K) {
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    matmul_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    double gflops = (2.0 * M * N * K) / (ms * 1e6);
    std::cout << "CUDA kernel: " << ms << " ms | " << gflops << " GFLOPS\n";
    return ms;
}

float cublas_matmul(cublasHandle_t handle,
                    const float* dA, const float* dB, float* dC,
                    int M, int N, int K) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // cuBLAS uses column-major by default; we treat input row-major by swapping roles
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,
                &alpha,
                dB, N,
                dA, K,
                &beta,
                dC, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    double gflops = (2.0 * M * N * K) / (ms * 1e6);
    std::cout << "cuBLAS SGEMM: " << ms << " ms | " << gflops << " GFLOPS\n";
    return ms;
}

double max_abs_diff(const std::vector<float>& A, const std::vector<float>& B) {
    double max_diff = 0.0;
    for (std::size_t i = 0; i < A.size(); ++i) {
        max_diff = std::max(max_diff, std::abs(static_cast<double>(A[i] - B[i])));
    }
    return max_diff;
}

int main(int argc, char** argv) {
    try {
        Args args = parse_args(argc, argv);
        std::cout << "Running matmul with M=" << args.M << " N=" << args.N << " K=" << args.K
                  << " mode=" << args.mode << '\n';

        auto A = random_matrix(args.M, args.K, args.seed);
        auto B = random_matrix(args.K, args.N, args.seed + 1);

        std::vector<float> cpu_result;
        if (args.mode == "cpu" || args.mode == "compare") {
            cpu_result = cpu_matmul(A, B, args.M, args.N, args.K);
        }

        float *dA = nullptr, *dB = nullptr, *dC = nullptr;
        cudaMalloc(&dA, sizeof(float) * args.M * args.K);
        cudaMalloc(&dB, sizeof(float) * args.K * args.N);
        cudaMalloc(&dC, sizeof(float) * args.M * args.N);

        cudaMemcpy(dA, A.data(), sizeof(float) * args.M * args.K, cudaMemcpyHostToDevice);
        cudaMemcpy(dB, B.data(), sizeof(float) * args.K * args.N, cudaMemcpyHostToDevice);

        if (args.mode == "kernel" || args.mode == "compare") {
            gpu_kernel_matmul(dA, dB, dC, args.M, args.N, args.K);
        }

        if (args.mode == "cublas" || args.mode == "compare") {
            cublasHandle_t handle;
            cublasCreate(&handle);
            cublas_matmul(handle, dA, dB, dC, args.M, args.N, args.K);
            cublasDestroy(handle);
        }

        if (args.mode != "cpu") {
            std::vector<float> gpu_result(args.M * args.N);
            cudaMemcpy(gpu_result.data(), dC, sizeof(float) * args.M * args.N, cudaMemcpyDeviceToHost);
            if (!cpu_result.empty()) {
                const double err = max_abs_diff(cpu_result, gpu_result);
                std::cout << "Max |CPU-GPU| = " << err << '\n';
            }
        }

        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << '\n';
        return 1;
    }
}
