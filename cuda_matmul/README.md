# CUDA Matrix Multiply Lab

This project compares CPU and GPU implementations of dense matrix multiplication on your RTX 2050.

## Layout
- `src/main.cu` – entry point with CPU baseline, CUDA kernel, and optional cuBLAS path.
- `build/` – create via CMake (`cmake -S . -B build`).

## Goals
1. Time a triple-loop CPU matmul and report GFLOPS.
2. Launch a tiled CUDA kernel and measure runtime via CUDA events.
3. (Optional) Call cuBLAS `sgemm`/`hgemm` for a high-performance baseline.
4. Validate results (`max|C_cpu - C_gpu|`) to ensure numerical correctness.

## Build & Run
```bash
cmake -S . -B build
cmake --build build
build/matmul 8192 8192 8192 --dtype fp32 --mode kernel
```

Adjust CLI flags to sweep problem sizes and observe when workloads shift between memory- and math-bound regimes.