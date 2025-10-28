# Day 2 Secure Threading Worksheet

Use this worksheet whenever you explore a new concurrency idea. The prompts
link every experiment back to secure transport and adaptive streaming.

---

## 1. Learning Goal
- **Concept focus:** ___________________________________________
- **Why it matters in secure streaming:** _______________________
- **Reference pages / links:** __________________________________

## 2. Pre-flight Checks
- [ ] I can restate the concept in my own words.
- [ ] I know how to validate the behaviour (tests, logging, benchmark).
- [ ] I created a feature branch: `git checkout -b feature/__________________`.

## 3. Experiment Sketch
1. **Shared state / buffers:** __________________________________
2. **Threads (names + responsibilities):**
   - Thread A: _________________________________________________
   - Thread B: _________________________________________________
   - Optional helpers: _________________________________________
3. **Success signal (what "good" looks like):** __________________

## 4. Coding Sandbox
Drop this into a new `.cpp` file or function. Fill the TODOs before compiling.

```cpp
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

// TODO: add shared state (mutex, condition_variable, atomic flag, etc.)

void worker_producer(/* TODO: args */) {
    // TODO: simulate capture or ingest load
}

void worker_consumer(/* TODO: args */) {
    // TODO: simulate crypto or encoding work
}

int main() {
    std::cout << "[setup] starting secure-thread experiment\n";

    // TODO: initialize shared state and metrics

    std::thread producer(worker_producer /* TODO: args */);
    std::thread consumer(worker_consumer /* TODO: args */);

    // TODO: start optional watchdog or logger thread

    producer.join();
    consumer.join();

    std::cout << "[result] throughput vs. latency notes: __________\n";
    return 0;
}
```

## 5. Instrumentation Plan
- **Logs / metrics to capture:** ________________________________
- **Failure cues (deadlock, drops, stalls):** ___________________
- **Validation step (unit test, manual check, profiler):** ______

## 6. Post-run Reflection
- What happened vs. expectation? ________________________________
- How did synchronization affect attack surface? ________________
- What breaks if a guard or predicate is removed? _______________
- Single sentence takeaway: ____________________________________

## 7. Git Discipline
- Run `git status` and review staged vs. unstaged changes.
- `git add ____________________`
- Commit template:
  ```
  feat(threading): __________________________________

  - change 1
  - change 2
  - test evidence
  ```
- Update `reflection_day2.md` with findings and open questions.

## 8. Next Iterations
- [ ] Stress test (more threads, smaller buffers, injected delay).
- [ ] Swap primitives (for example `std::scoped_lock` or `std::jthread`).
- [ ] Port to another compiler or sanitizer build.
- [ ] Capture screenshots or diagrams for the knowledge base.

---

Archive the filled worksheet next to your experiment code before moving on.
