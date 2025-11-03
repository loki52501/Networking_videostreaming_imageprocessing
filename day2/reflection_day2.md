# Reflection – Day 2 Secure Pipeline

Write ~150 words addressing the prompts below. Capture raw observations as well
as the metrics you measured. Commit this file after you finish the lab.

1. Where did concurrency improve performance, and where did it enlarge the attack surface? 
simpley question. Where would you add monitoring, retries, or circuit breakers in a production version?
in the beginning, middle and at the end
Notes / outline before drafting:
- concurrency improved performance for larger values either during insertion or printing from 1 to 500000,  the attack surface enlarges when you don't use mutex or locks at all since the shared memory might reveal data that are about some other variable. it improved performance in cases where it requires more data processing, and attack surfaces are mostly prevalent only on race conditions, shared resources. 


Final reflection paragraph:

I learnt how threads are important in concurrency programming.
