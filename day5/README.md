# Day 5 – Threading Mini Labs

Use these bite-sized examples to explore each concept before you write the full Day 5 solution. Build or run one file at a time, experiment, then take notes for your journal.

## Files
- `1_condition_variable_basics.cpp` – Observe how `condition_variable` hands control between threads.
- `2_safe_queue_demo.cpp` – Minimal safe task queue with timed waits and graceful shutdown.
- `3_thread_pool_stub.cpp` – Thread-pool skeleton with TODOs for latency logging and shutdown tweaks.

## How to Build
```bash
cmake -S day5 -B build/day5
cmake --build build/day5 --target day5_condition_variable
cmake --build build/day5 --target day5_safe_queue
cmake --build build/day5 --target day5_thread_pool
```

or compile a single file quickly:
```bash
g++ -std=c++17 -pthread day5/1_condition_variable_basics.cpp -o cond_demo
```

## Study Prompts
1. Where does each example block waiting threads, and what wakes them up?
2. How do the queue and thread pool track work that is outstanding?
3. Which measurements would let you compare dispatch vs execution latency?

Curious? Add timers, logging, or extra tasks and observe the impact.
