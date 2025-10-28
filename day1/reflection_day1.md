# Day 1 Reflection - Frame Memory Model

Respond in ~150 words after completing `gray_video.cpp` and measuring FPS.

### Prompt
1. Which operations in your lab allocate heap memory, and why are they unavoidable for 1080p frames? capturing video frames and converting them into grey would need heap memory. It's unavoidable for 1080p frames, since they are huge in size, and cpu can't hold it. stack has only limited amount of space, but this needs around 6mb for eg: 1280x 720x 3= 2.6 mb bgr, 0.9 mb gray.

2. How does reusing the same buffer improve both security (fewer leaks) and performance (cache and locality)? since we're using the same heap address space, it's easy to identify if there was any leakage and it improves performance since the location is cached and it become local such that it is easy to fetch the buffer.

3. What trade-off did you observe between FPS and image size or color depth? FPS held steady at 30 even after bumping to 1980 x 1080. I did notice a longer startup time taken to run because higher resolution/3-channel capture allocates more heap, but once the loop warmed up the throughput didn't drop. 

### Suggested Notes Outline
- Biggest surprise about stack versus heap in OpenCV. that mat object's header is located at stack and it's data is in heap obv.
- FPS measurements (instantaneous plus any fluctuations). i used system time , to calculate how many frames are processed for a second.
- Where you suspect bottlenecks (capture versus process versus display) and supporting evidence. maybe when multiple captures.
- One experiment you plan for Day 2 (for example, threaded capture, ring buffer, or resolution tweak). ring buffer.. 

Commit this file separately with:
```bash
git add day1/reflection_day1.md
git commit -m "docs(day1): reflection on frame memory model"
```
