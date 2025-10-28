# Day 1 Concept Overview - Frame Touchpoints

| Stage | What Happens | Memory Area | Why It Matters |
| --- | --- | --- | --- |
| Capture | `cv::VideoCapture` driver pulls frame bytes | Heap (driver buffer) | Entry point for latency; DMA into RAM |
| Store | `cv::Mat` header captures shape and stride pointer | Stack (header) | Fast lifecycle management, avoids heap churn |
| Process | `cv::cvtColor` writes grayscale pixels | Heap | Dominant CPU plus cache cost; potential copy |
| Display | `cv::imshow` hands buffer to window or GPU | GPU / VRAM | Final hop into display queue |

Rule of thumb: a 1080p RGB frame is about 6 MB, so heap allocation is unavoidable. Track when new buffers appear versus when pointers are reused to understand performance and memory hygiene.
