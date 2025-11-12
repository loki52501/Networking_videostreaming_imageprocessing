# C++ Concurrency Study Plan (Chapter 4 Focus)

You will work through the Chapter 4 material in *C++ Concurrency in Action* (sections 4.1–4.4 in `exerc_2.pdf`). Each study block mixes reading, experiments, and mini-deliverables that plug into the companion problem template stored beside this file.

## Week 1 — Futures as One-Off Synchronization

- **Day 1 – Futures vs. Condition Variables (pp. 81–85)**  
  - Read §4.1 and §4.2 introduction; note why futures model one-shot events.  
  - Re-implement the condition-variable notification example and then refactor it to use `std::promise`/`std::future`.  
  - Fill in the “Context & Constraints” portion of the problem template.
- **Day 2 – Returning Values (pp. 85–90)**  
  - Study §4.2.1 on returning data from background tasks.  
  - Implement the “Background report” practice in the template; compare using `std::async` vs. manual `std::promise`.  
  - Capture observations under “Future Retrieval Checklist”.
- **Day 3 – Associating Tasks (pp. 90–95)**  
  - Work through §4.2.2 (`std::packaged_task`).  
  - Extend the ongoing problem by swapping between packaged tasks and promises; measure ease of cancellation/error handling.  
  - Update template’s “Implementation Log”.

## Week 2 — Controlling Lifetime, Timing, and Fan-Out

- **Day 4 – Timing APIs (pp. 95–99)**  
  - Read §4.2.3 about `wait_for`/`wait_until` and `future_status`.  
  - Add timeout handling to your scenario; document timeout branches in the template.  
  - Summarize pros/cons in “Risk Notes”.
- **Day 5 – Detaching and Launch Policies (pp. 99–105)**  
  - Cover §4.2.4–§4.2.5 (`std::async`, launch policies, detached worker).  
  - Experiment with `std::launch::async` vs. `std::launch::deferred`. Record measurements (latency, threads created) under “Experiment Table”.
- **Day 6 – Sharing Results (pp. 105–110)**  
  - Study §4.3: waiting for multiple consumers with `std::shared_future`.  
  - Implement the “Fan-out processing” task from the template so that multiple threads consume the same future value safely.

## Week 3 — Higher-Level Synchronization Primitives

- **Day 7 – Barriers and Latches (pp. 110–118)**  
  - Read §4.4 on barriers/latches (focus on Listing 4.26).  
  - Add a phase barrier to the problem so worker threads collectively publish progress.  
  - Document barrier setup steps in the template.
- **Day 8 – Integrating Condition Variables and Futures (pp. 118–123)**  
  - Explore hybrid patterns where a condition variable feeds a `std::promise`.  
  - Stress-test with spurious wakeups and double-set protection.
- **Day 9 – Error Propagation and Cancellation (pp. 123–129)**  
  - Review exception propagation through futures and the effect of `std::future_error`.  
  - Extend template scenario with an injected fault; ensure the future carries exceptions correctly.
- **Day 10 – Capstone Build & Reflection**  
  - Assemble the complete problem solution document: final architecture, benchmarks, lessons learned.  
  - Complete the prompts in `reflection.md`.

### Ongoing Habits

- **Daily Warm-up:** Re-implement one small snippet (e.g., a `std::packaged_task` helper) from memory.  
- **Peer Review Simulation:** After each major change, write a short review comment for yourself focusing on synchronization correctness.  
- **Reference Log:** Maintain a short glossary in the problem template (e.g., shared state, readiness, launch policy) as you encounter new terminology.

