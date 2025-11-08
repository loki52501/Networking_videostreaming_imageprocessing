# Day 6 Measurements

Track how the server versions respond to increasing client load. Use one section per configuration and capture raw data plus observations.

## Environment Notes
- OS / Compiler / Build flags:
- CPU / core count:
- Any other background load:

## Thread-Per-Client Server
| Clients | Total Time (s) | CPU Usage Notes | Observations |
|---------|----------------|-----------------|--------------|
| 1       |                |                 |              |
| 4       |                |                 |              |
| 8       |                |                 |              |

### Summary
- Bottleneck(s):
- Symptoms (e.g., context switching overhead, jitter):

## Thread Pool Server (4 workers)
| Clients | Total Time (s) | CPU Usage Notes | Observations |
|---------|----------------|-----------------|--------------|
| 1       |                |                 |              |
| 4       |                |                 |              |
| 8       |                |                 |              |

### Summary
- Improvements vs baseline:
- When does queueing start to dominate?

## Analysis
- Plot or describe latency trend (attach external chart if you create one).
- Reflect on how worker count impacts throughput vs latency.
- Note any surprises or debugging pain points encountered during measurement.
