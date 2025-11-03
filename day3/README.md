# Day 3 — Networking & Latency Modeling

You own the execution; these notes keep you aimed at real-time streaming outcomes.

## Learning Objectives
- Map how packets travel across the stack and where latency hides.
- Measure RTT, jitter, and throughput with tooling you can defend in a review.
- Build intuition for when TCP hurts and why UDP/QUIC exist.

## Preparation (lock this in before the timer starts)
- ✅ Tools: see `checklist_day3.md` for install/verify steps (Wireshark, Python, compiler, delay simulator).
- ✅ Concept refresh: OSI layers 3–5, TCP vs UDP trade-offs, latency and jitter definitions.
- ✅ Mental model: keep the producer → NIC → receiver pipeline sketch visible while you work.

## Execution Blueprint
1. **Instrument baseline RTT**
   - Implement `templates/tcp_echo_server.cpp` and `templates/tcp_latency_client.cpp`.
   - Run 50+ iterations, persist to `logs/rtt_baseline.txt`.
   - Capture an initial Wireshark trace for the loopback run.
2. **Stress with delay and loss**
   - Prepare at least two impairment scenarios in `templates/netem_playbook.md`.
   - Rerun the client, store logs per scenario (e.g. `logs/rtt_delay_100ms.txt`), annotate observations.
3. **Visualise and quantify**
   - Feed each log into `templates/rtt_plot.py`.
   - Record mean, p95, and jitter readings in `latency_experiment_log.md`.
4. **UDP contrast**
   - Adapt `templates/udp_probe.cpp` to fire datagrams with sequence IDs.
   - Track drops and out-of-order packets; compare against TCP notes.
5. **Reflection and retention**
   - Finish `reflection_day3_template.md` (≥150 words).
   - Answer prompts in `flashcards_day3.md` and sync to your deck.

## Deliverables (for your end-of-day review)
- Baseline and impaired RTT logs in `day3/logs/`.
- Screenshots or exports of key Wireshark traces (timestamped).
- Plots generated from the Python template.
- Completed reflection and flashcards.
- Git branch and commits documenting the experiments (tag suggestion: `v0.3`).

Keep iterating until the metrics narrative feels defendable in a systems design interview.
