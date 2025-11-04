# Day 5 — Math & Memory Bounds Lab

## Objectives
- Build intuition for arithmetic intensity vs GPU FLOPS:byte limits.
- Explore tile and wave quantization effects via small C++ experiments.
- Practice reasoning about Tensor Core alignment requirements.

## Exercises
1. **Arithmetic Intensity Explorer (`intensity_lab.cpp`)**
   - Prompt for M, N, K, element size (bytes).
   - Compute FLOPS, bytes touched (assume naive GEMM), and arithmetic intensity.
   - Compare against a GPU target ratio (default 138.9 FLOPS/B for V100; allow custom input).
   - Classify the operation as "math-bound" or "memory-bound".

2. **Tile Quantization Simulator (`tiling_sim.cpp`)**
   - Ask for matrix dims M, N and tile size Mtile, Ntile.
   - Compute number of tiles, useful vs wasted work.
   - Report efficiency loss (% of tile work that multiplies zero padding).
   - Include a sweep helper to iterate N over a range and print when new tiles spawn.

3. **Wave Quantization Playground (`wave_quant.cpp`)**
   - Parameters: M, N, K, tile dims, SM count, tiles per SM.
   - Derive tiles per wave, tail wave occupancy, and predicted slowdown.
   - Print a small table showing waves, active SM %, and expected relative throughput.

4. **Tensor Core Alignment Checker (`tensor_core_align.cpp`)** *(stretch goal)*
   - Input data type (`fp16`, `tf32`, etc.).
   - Suggest the nearest multiples that meet alignment guidance.
   - Estimate padding overhead (extra elements & bytes).

## Suggested Workflow
- Start with `intensity_lab.cpp` to review the math from the NVIDIA guide.
- Move to tiling + wave quantization to visualize scheduling and occupancy.
- Use the alignment checker to plan GEMM shapes for Tensor Core efficiency.
- Run each program with multiple scenarios; capture notes in your journal.

## Build Hints
- Provide a minimal `CMakeLists.txt` or simple `g++` compile command.
- Keep everything CPU-only; we’re modeling GPU math, not running CUDA.

## Reflection Prompts
- Which GEMM shapes on your RTX 2050 would be memory-bound?
- How do tile dimensions influence both bandwidth and parallelism?
- What trade-offs did you notice when sweeping matrix sizes?

Capture insights in your Day 4/5 journal after completing the exercises.