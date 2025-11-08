# Day 6 - Multi-Threaded Frame Streamer

Welcome to **Day 6 (Tue, Nov 5)**. Today you connect your threading fundamentals to networking by building a multi-client frame streaming prototype.

## Checklist
- [ ] Concept notes for three concurrency patterns
- [ ] `server_basic.cpp` runs with thread-per-connection handling
- [ ] `client.cpp` sends and receives five frames successfully
- [ ] `server_pool.cpp` integrates Day 5 thread pool
- [ ] Benchmark notes captured in `measurements.md`
- [ ] Reflection answered in `journal.md`

---

## 1. Concept Overview (~1 h)
1. Read about these patterns and add ~2 paragraph notes to **Concept Notes** section below:
   - Thread-per-connection (naive TCP server model)
   - Thread pool plus queue (QUIC/WebRTC/HTTP3 style scheduling)
   - Producer-consumer with bounded buffer (media pipelines / FFmpeg)
2. Note pros, cons, and real-world systems using each.

## 2. Coding Tasks (~3 h)
### Step 1 - Baseline server (`server_basic.cpp`)
- Implement the provided starter in `server_basic.cpp`.
- Goals:
  - Bind to port 8080 and accept clients.
  - Spawn one `std::thread` per accepted client.
  - Inside `handle_client`, log incoming text frames and echo a confirmation.
  - Ensure sockets close on exit; handle `read` returning 0 or -1 gracefully.
- Validation: run the server, start two terminal clients (see **Quick Test** section) and confirm interleaved logs.

### Step 2 - Simple client (`client.cpp`)
- Use the provided starter to connect to localhost:8080.
- Loop five times sending `Frame_i` strings, wait for response, sleep one second.
- Ensure the socket closes cleanly and the process exits without hanging.

### Step 3 - Thread pool upgrade (`server_pool.cpp`)
- Copy your Day 5 thread pool implementation (header/source) or create a minimal pool as described in `server_pool.cpp`.
- Replace direct thread spawning with `pool.submit([client_socket, id]{ handle_client(...); });`.
- Make sure the pool lifetime outlives active clients; call `join()` or `shutdown()` before `main` exits.
- Add logging to observe worker thread IDs handling clients.

## 3. Measurement Study (~1 h)
1. In `measurements.md`, record total wall time for all clients to finish in these scenarios:
   - Thread-per-client server: 1, 4, 8 simultaneous clients.
   - Thread pool server (4 workers): 1, 4, 8 simultaneous clients.
2. Capture notes about CPU usage, responsiveness, and when saturation occurs.
3. Plot or tabulate latency versus client count (hand-drawn or digital) and summarize findings.

## 4. Networking Tie-in (~30 min)
- Read Cloudflare blog "Inside a QUIC Connection".
- Optional: watch Computerphile "How HTTPS Uses TLS".
- Add three bullets to **Networking Reflection** below tying QUIC stream multiplexing to your thread pool design.

## 5. Active Recall (~10 min)
Answer the prompts in **Active Recall** section without peeking at notes; then verify and correct.

## 6. Reflection (~15 min)
Fill `journal.md` using the prompts provided in that file.

## Concept Notes
_(Summarize the three patterns here.)_

## Networking Reflection
_(Add three bullets connecting QUIC concepts to your implementation.)_

## Active Recall
1. Why does thread-per-client scale poorly?
2. What does a thread pool save in terms of context switching and resource usage?
3. How does latency depend on job size versus number of workers?

## Quick Test Commands
Run from repo root in separate terminals:
```
# Terminal 1 - basic server
./build/server_basic

# Terminal 2 - client
./build/client
```
Adjust commands if you use another build system; ensure binaries go in `build/`.

## Tomorrow's Prep
- Skim OpenCV socket streaming examples.
- Ensure FFmpeg CLI is installed so you can experiment with GPU-accelerated encoding on Day 7.
