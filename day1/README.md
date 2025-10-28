# Day 1 - Frame Memory Baseline

Welcome to the Day 1 Netflix-aligned security-and-streaming curriculum lab. Use this folder to complete every artifact for "How a Video Frame Lives in Memory."

## Objectives
- Trace how a raw camera frame touches heap, stack, and GPU memory.
- Implement a grayscale capture loop in C++ with FPS plus buffer logging.
- Record findings, create a visual lifecycle diagram, and reflect on trade-offs.

## Required Assets
| File | Purpose |
| --- | --- |
| `gray_video.cpp` | Core lab with TODOs you must fill in. |
| `reflection_day1.md` | 150-word reflection responses. |
| `docs/architecture_day1.drawio` | Diagram of the frame lifecycle; export PNG if preferred. |

## Build and Run (example)
```bash
cmake -S .. -B ../build
cmake --build ../build
../build/gray_video.exe
```
Adapt paths to your environment. Install OpenCV with C++ bindings before running the sample.

## Git Flow Checklist
1. `git checkout -b feature/day1-video-basics`
2. Implement the lab and document build instructions in this folder.
3. `git add day1/`
4. `git commit -m "feat(day1): complete frame memory baseline"`
5. After the reflection: `git add day1/reflection_day1.md`
6. `git commit -m "docs(day1): reflection on frame memory model"`
7. `git tag -a v0.1 -m "Day 1 complete - frame lifecycle"`

## Diagram Guidance
- Show stack vs heap vs GPU along the pipeline.
- Label copy versus reference transitions and note approximate byte sizes (for example, 1080p RGB is about 6 MB).
- Call out latency hot spots and how buffer reuse mitigates leaks and performance issues.

## Active Recall Flashcards
Create at least five question-and-answer cards covering:
- Location of the `cv::Mat` header versus pixel buffers.
- First heap allocation in the lab.
- Definition of a deep copy within OpenCV.
- Why the stack cannot hold full-resolution frames.
- FPS versus resolution trade-offs.

## Reflection Prompt
See `reflection_day1.md` and capture outcomes after measuring FPS and memory behavior.
