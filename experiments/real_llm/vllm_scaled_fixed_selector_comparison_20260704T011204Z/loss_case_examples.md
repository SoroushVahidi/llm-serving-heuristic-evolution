# Loss-Case Examples — Selector vs. Fixed Baselines

20 of 149 representative loss cases (sorted by most-negative delta weighted-goodput contribution, i.e. worst losses first).

| Request | Baseline | Regime | Bucket | Target tok | Concurrency | Reason | Selector SLO met | Baseline SLO met | Delta WG contrib | Delta latency (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| 246 | fifo | bursty_tight | medium | 64 | 2 | long-output underestimation | False | True | -2.000 | -0.001 |
| 246 | shortest_output_first | bursty_tight | medium | 64 | 2 | long-output underestimation | False | True | -2.000 | -0.008 |
| 246 | vllm_direct | bursty_tight | medium | 64 | 2 | long-output underestimation | False | True | -2.000 | 0.012 |
| 194 | least_laxity_first | bursty_tight | short | 64 | 4 | long-output underestimation | False | True | -2.000 | -0.062 |
| 194 | edf | bursty_tight | short | 64 | 4 | long-output underestimation | False | True | -2.000 | -0.069 |
| 207 | edf | bursty_tight | short | 128 | 2 | long-output underestimation | False | True | -2.000 | 0.093 |
| 194 | estimated_service_time_first | bursty_tight | short | 64 | 4 | long-output underestimation | False | True | -2.000 | -0.057 |
| 435 | vllm_direct | overloaded_mixed_priority | medium | 64 | 8 | long-output underestimation | False | True | -2.000 | -0.001 |
| 427 | estimated_service_time_first | overloaded_mixed_priority | medium | 64 | 2 | long-output underestimation | False | True | -2.000 | -0.023 |
| 435 | fifo | overloaded_mixed_priority | medium | 64 | 8 | long-output underestimation | False | True | -2.000 | -0.002 |
| 435 | edf | overloaded_mixed_priority | medium | 64 | 8 | long-output underestimation | False | True | -2.000 | -0.027 |
| 427 | edf | overloaded_mixed_priority | medium | 64 | 2 | long-output underestimation | False | True | -2.000 | -0.020 |
| 427 | least_laxity_first | overloaded_mixed_priority | medium | 64 | 2 | long-output underestimation | False | True | -2.000 | -0.013 |
| 499 | vllm_direct | overloaded_mixed_priority | long | 64 | 8 | long-output underestimation | False | True | -2.000 | 0.004 |
| 435 | shortest_output_first | overloaded_mixed_priority | medium | 64 | 8 | long-output underestimation | False | True | -2.000 | 0.002 |
| 435 | estimated_service_time_first | overloaded_mixed_priority | medium | 64 | 8 | long-output underestimation | False | True | -2.000 | -0.014 |
| 499 | estimated_service_time_first | overloaded_mixed_priority | long | 64 | 8 | long-output underestimation | False | True | -2.000 | -0.014 |
| 499 | shortest_output_first | overloaded_mixed_priority | long | 64 | 8 | long-output underestimation | False | True | -2.000 | -0.001 |
| 499 | least_laxity_first | overloaded_mixed_priority | long | 64 | 8 | long-output underestimation | False | True | -2.000 | -0.020 |
| 499 | edf | overloaded_mixed_priority | long | 64 | 8 | long-output underestimation | False | True | -2.000 | -0.024 |
