# Day 3 Checklist

## Pre-flight
- [ ] Wireshark captures on the intended interface without errors.
- [ ] Python 3.10+ imports `matplotlib` successfully.
- [ ] C++ toolchain builds a trivial socket sample.
- [ ] Network impairment tool ready (tc/netem, Clumsy, VM, etc.).
- [ ] OSI L3–L5, TCP vs UDP, latency and jitter definitions refreshed.

## Execution
- [ ] Baseline TCP echo server and latency client implemented and committed.
- [ ] ≥50 RTT samples saved to `logs/rtt_baseline.txt`.
- [ ] Two impairment profiles executed with dedicated log files.
- [ ] Latency plots rendered and archived.
- [ ] UDP probe created with drop/out-of-order accounting.
- [ ] Reflection (150+ words) drafted in `reflection_day3_template.md`.
- [ ] Flashcards answered and synced to your personal deck.

## Wrap-up
- [ ] Summarise findings in your daily log or journal.
- [ ] Push branch and open a draft PR or learning log entry.
- [ ] Tag `v0.3` once review-ready.
