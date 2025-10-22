# Day 1 — Reflection

Prompt:

"What surprised me about how video frames move through memory?

Which part of the process can cause bottlenecks — capture, processing, or display — and why?"

Suggested notes template (≈150 words):

- Today I learned that each frame is a 3-D matrix (H × W × C) in heap memory (cv::Mat). The most surprising part was …
- I noticed FPS changed when … and I think the reason is …
- Bottleneck analysis:
  - Capture: e.g., camera/driver limits (USB bandwidth, exposure)
  - Processing: e.g., cvtColor cost, CPU cache misses, resizing
  - Display: e.g., vsync, GUI refresh interval, waitKey delay
- Next time I’ll test … (e.g., resolution changes, thread split, queueing)

