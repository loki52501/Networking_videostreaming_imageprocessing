# Reflection Prompts (Chapter 4 Threading Study)

Use these after each study day and again during the capstone review.

1. **Concept Resonance**  
   - Which synchronization primitive felt most natural today and why?  
   - Where did futures feel awkward compared with condition variables?
2. **Readiness Semantics**  
   - How did you confirm the future’s shared state was set exactly once?  
   - What signals or logs helped you trust the readiness transition?
3. **Error Paths**  
   - Describe an exception path you intentionally triggered. Did it cross the thread boundary as expected?  
   - What guardrails (RAII, scope guards, cancellation flags) will you add next time?
4. **Performance Awareness**  
   - Which launch policy was used, and how did it affect CPU utilization and latency?  
   - What would you measure if production SLAs were tighter?
5. **Team Communication**  
   - How would you explain today’s design decision to a code reviewer unfamiliar with futures?  
   - Which invariants should be documented alongside the code?
6. **Next Experiment**  
   - What variation (shared futures, barriers, combining with coroutines) are you curious to trial next?  
   - What questions remain unanswered after today’s work?

