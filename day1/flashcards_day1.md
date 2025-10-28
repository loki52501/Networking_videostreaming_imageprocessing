# Day 1 Active Recall Cards

| Where does the `cv::Mat` header live in memory for a live capture frame? | On the stack, holding metadata and the pointer to heap pixels. |
| Which line in `gray_video.cpp` triggers the first heap allocation for pixel data? | The first `cap.read(frame)` call (or `cap >> frame`) allocates/locks the driver buffer on the heap. |
| Why is heap allocation unavoidable for a 1280x720 RGB frame? | The frame is roughly 1280×720×3 ≈ 2.6 MB, far larger than typical stack limits, so pixels must live on the heap. |
| What indicates buffer reuse when you log frame addresses? | Seeing `frame.data` and `gray.data` print the same pointer across iterations shows OpenCV reuses the heap blocks. |
| How do you compute instantaneous FPS with OpenCV timing APIs? | Measure ticks via `now = getTickCount()`, convert `dt = (now - prev)/getTickFrequency()`, then `fps = 1.0 / dt`. |
| What is a practical trade-off between raising resolution and maintaining FPS? | Higher resolution increases per-frame bytes and processing time, risking drops below 30 FPS unless the pipeline stays optimized. |


Suggested topics:
- Location of the `cv::Mat` header vs pixel buffer.
- First point where heap allocation occurs in `gray_video.cpp`.
- Definition of a deep copy in OpenCV.
- Why the stack cannot hold full-resolution frames.
- FPS versus resolution trade-offs observed during testing.
