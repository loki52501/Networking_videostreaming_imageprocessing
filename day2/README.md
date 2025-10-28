# Day 2 -- Multithreading, Ring Buffers, and Secure Streaming

You already understand the mechanics of mutexes, condition variables, and ring
buffers. Today you will rehearse those skills with a security-oriented spin:
the capture thread must never stall while crypto work happens off to the side.

## Learning Objectives
- Wire up a producer-consumer pipeline that tolerates encryption latency.
- Capture throughput/latency metrics you can justify in a code review.
- Practice disciplined Git workflow and daily documentation.

## Required Lab Steps
1. **Read** `threaded_stream_exercise.cpp` and finish every `TODO`.
   - Implement bounded push/pop predicates.
   - Call `processFrame()` before display and extend it with any extra work you need.
   - Emit FPS logs so you can reason about performance under simulated crypto load.
2. **Sketch** the pipeline diagram.
   - Use PlantUML or a drawing tool of your choice.
   - Highlight mutex scope, condition variables, and the simulated crypto delay.
   - Save as `architecture_day2.png`.
3. **Reflect** in `reflection_day2.md`.
4. **Active recall**: fill answers inside `anki_day2.txt`.
5. **Git routine**:
   - `git checkout -b feature/threaded-buffer` (or similar).
   - Small commits (100 or fewer changed lines).
   - Messages describe what + why + how you tested.
   - Finish the day with documentation and tag `v0.2`.

## Optional Stretch Goals
- Add a watchdog thread that detects stalled consumers and toggles `stop_flag`.
- Log queue depth over time to visualise backpressure.
- Run under ThreadSanitizer or a synthetic frame-drop scenario.

## Deliverables Checklist
- [ ] `threaded_stream.cpp` (copy the finished exercise if you prefer a clean file).
- [ ] `architecture_day2.png`
- [ ] `reflection_day2.md`
- [ ] Anki cards updated with your own answers.
- [ ] Branch merged and tagged `v0.2`.

Take notes as you work; tomorrow's networking lesson will build directly on the
metrics and design choices you document today.
