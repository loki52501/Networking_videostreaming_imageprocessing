#include <cuda_runtime.h>
#include <iostream>
#include<chrono>

__global__ void add(float* a, float* b, float* c, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    constexpr int n = 10'000'000;
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);

    float *hA = nullptr, *hB = nullptr, *hC = nullptr;
    cudaHostAlloc(&hA, bytes, cudaHostAllocDefault);
    cudaHostAlloc(&hB, bytes, cudaHostAllocDefault);
    cudaHostAlloc(&hC, bytes, cudaHostAllocDefault);

    for (int i = 0; i < n; ++i) {
        hA[i] = static_cast<float>(i);
        hB[i] = static_cast<float>(2 * i);
        hC[i] = 0.0f;
    }

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    cudaMalloc(&dA, bytes);
    cudaMalloc(&dB, bytes);
    cudaMalloc(&dC, bytes);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaEvent_t startEvt, stopEvt, kernelStart, kernelStop;
    cudaEventCreate(&startEvt);
    cudaEventCreate(&stopEvt);
    cudaEventCreate(&kernelStart);
    cudaEventCreate(&kernelStop);

    cudaEventRecord(startEvt, stream);
    auto sys_start = std::chrono::high_resolution_clock::now();
    cudaMemcpyAsync(dA, hA, bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dB, hB, bytes, cudaMemcpyHostToDevice, stream);

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    cudaEventRecord(kernelStart, stream);
    add<<<blocks, threads, 0, stream>>>(dA, dB, dC, n);
    cudaEventRecord(kernelStop, stream);

    cudaMemcpyAsync(hC, dC, bytes, cudaMemcpyDeviceToHost, stream);
    cudaEventRecord(stopEvt, stream);
    cudaEventSynchronize(stopEvt);
    auto sys_stop = std::chrono::high_resolution_clock::now();

    float host_ms = 0.0f;
    float kernel_ms = 0.0f;
    cudaEventElapsedTime(&host_ms, startEvt, stopEvt);
    cudaEventElapsedTime(&kernel_ms, kernelStart, kernelStop);

    std::cout << "first few results: ";
    for (int i = 0; i < 5; ++i) ;
    std::cout << "\nend-to-end (copy+kernel+copy): " << host_ms << " ms\n";
    std::cout << "kernel only: " << kernel_ms << " ms\n";
    std::cout << "system elapsed: "
              << std::chrono::duration<double, std::milli>(sys_stop - sys_start).count()
              << " ms\n";

    cudaEventDestroy(startEvt);
    cudaEventDestroy(stopEvt);
    cudaEventDestroy(kernelStart);
    cudaEventDestroy(kernelStop);
    cudaStreamDestroy(stream);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cudaFreeHost(hA);
    cudaFreeHost(hB);
    cudaFreeHost(hC);
    return 0;
}
