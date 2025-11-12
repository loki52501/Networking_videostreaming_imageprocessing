# One-Off Event Synchronization Scenario

Use this template to explore Chapter 4 synchronization tools. Duplicate the “Practice Run” section for each exercise (promises, packaged tasks, `std::async`, shared futures, barriers).

## 1. Context & Constraints
- **Scenario title:** (e.g., “Flight-board monitor”)
- **One-off event being modeled:**  
- **Threads involved and their roles:**  
- **Shared state owner (promise/packaged task/future):**  
- **Lifetime boundaries (start/stop triggers):**  
- **Non-functional constraints (latency, data freshness, CPU budget):**

## 2. Future Retrieval Checklist
- [ ] `std::future` or `std::shared_future`? Why?  
- [ ] Expected return type (value/void/exception).  
- [ ] Readiness validation (`valid()`, logging).  
- [ ] Timeout behavior (`wait_for` vs. `wait_until`).  
- [ ] Clean-up after `get()` (is the future consumed?).  
- [ ] Exception propagation plan.

## 3. Implementation Log
Document each synchronization variant in the table.

| Variant | API Focus | Thread Roles | Notable Code Snippet/Link | Outcome Summary |
|---------|-----------|--------------|---------------------------|-----------------|
| Promise | `std::promise` + thread | | | |
| Packaged Task | `std::packaged_task` | | | |
| Async | `std::async` launch policy | | | |
| Shared Future | `std::shared_future` fan-out | | | |
| Barrier/Latch | `std::barrier` or `std::latch` | | | |

## 4. Experiment Table
Capture timing/behavioral data for each variant.

| Variant | Launch Policy | Threads Spawned | Mean Latency | Timeout Handling Notes |
|---------|---------------|-----------------|--------------|------------------------|

## 5. Risk Notes
- **Deadlock risks spotted:**  
- **Exception handling edge cases:**  
- **Race detection strategy/tests:**  
- **Fallback plan if the event never happens:**  
- **Opportunities to replace futures with other primitives (call_once, semaphores, etc.):**

## 6. Practice Run (duplicate per experiment)
- **Goal:**  
- **Steps:**  
  1.  
  2.  
  3.  
- **Verification:** (e.g., `std::future_status::ready` within X ms)  
- **Observations:**  
- **Follow-up questions for mentor:**

