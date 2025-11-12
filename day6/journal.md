# Day 6 Journal – Reflection Prompts

Respond in complete sentences. Keep notes concise but thoughtful (≈150–250 words total).

## Real-Time Streaming and Concurrency
How did today’s concurrency patterns make a real-time streaming workflow possible? Mention specific mechanisms (queues, pools, non-blocking I/O) and how they reduce latency spikes.
i used queues to enter tasks before hand.. and was waiting using condition variable, which kept out of entering the queue to pop tasks, until it recieves any tasks at all by submit which takes function and arguments as parameters. 
## Parallelism vs Asynchrony
Explain the difference in your own words. Include one example from today’s exercises that highlights each idea.

## Integrating FFmpeg or OpenCV
Sketch the steps you would take to slot FFmpeg frame decoding (or OpenCV capture) into this architecture. Note which thread(s) would own decode, network send, and any transformation stages.

## Open Questions
List anything that still feels fuzzy (e.g., backpressure handling, flow control, TLS layering). These guide tomorrow’s focus.
i'm still figuring out how to start multiple servers and classess without any trouble...