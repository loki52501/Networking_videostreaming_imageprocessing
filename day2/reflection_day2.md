# Reflection – Day 2 Secure Pipeline

Write ~150 words addressing the prompts below. Capture raw observations as well
as the metrics you measured. Commit this file after you finish the lab.

1. Where did concurrency improve performance, and where did it enlarge the attack surface?
2. What happens to latency and frame drops when the simulated crypto cost spikes?
3. If the processing (encryption) thread crashes or livelocks, how is the system detected and recovered?
4. Where would you add monitoring, retries, or circuit breakers in a production version?

Notes / outline before drafting:
- concurrency improved performance for larger values either during insertion or printing from 1 to 500000,  the attack surface enlarges when you don't use mutex or locks at all since the shared memory might reveal data that are about some other variable. ___________________________________________________________________________
- ___________________________________________________________________________
- ___________________________________________________________________________

Final reflection paragraph:

______________________________________________________________________________
______________________________________________________________________________
______________________________________________________________________________
______________________________________________________________________________
