# Mini Matmul Lab

Goal: fill in the TODOs in `mini_matmul.cu` to time CPU vs CUDA matrix multiplication **and addition** on a single 2D problem size.

Steps:
1. Implement the CPU triple-loop GEMM.
2. Implement the CPU elementwise matrix addition.
3. Write the tiled CUDA matmul kernel + launcher.
4. Write a simple CUDA matrix-addition kernel.
5. Add timing and result verification for both ops.
6. Build with `nvcc mini_matmul.cu -o mini_matmul.exe` (or make a CMake target).
