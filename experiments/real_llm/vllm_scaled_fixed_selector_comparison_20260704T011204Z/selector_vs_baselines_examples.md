# Selector vs. Fixed-Baseline Decision Divergence — Examples

Every row below is a request where the selector's actual admission run and a baseline's actual admission run (both real, over the identical request plan) produced a DIFFERENT SLO outcome for the same request_id. This is measured from real execution, not shadow evaluation: each policy including the selector ran independently against the real vLLM server over the identical plan.

| Regime | Bucket | Target tok | Concurrency | Baseline | Request | Selector SLO violated | Baseline SLO violated | Selector rank | Baseline rank |
|---|---|---|---|---|---|---|---|---|---|
| bursty_tight | long | 128 | 1 | fifo | 320 | True | False | 3 | 0 |
| bursty_tight | long | 128 | 1 | shortest_output_first | 320 | True | False | 3 | 0 |
| bursty_tight | long | 128 | 1 | vllm_direct | 320 | True | False | 3 | 0 |
| bursty_tight | long | 256 | 2 | fifo | 345 | True | False | 4 | 0 |
| bursty_tight | long | 256 | 2 | shortest_output_first | 345 | True | False | 4 | 0 |
| bursty_tight | long | 256 | 2 | vllm_direct | 345 | True | False | 4 | 0 |
| bursty_tight | medium | 64 | 1 | edf | 240 | False | True | 0 | 1 |
| bursty_tight | medium | 64 | 1 | least_laxity_first | 240 | False | True | 0 | 1 |
| bursty_tight | medium | 64 | 1 | estimated_service_time_first | 240 | False | True | 0 | 1 |
| bursty_tight | medium | 64 | 2 | fifo | 246 | True | False | 2 | 1 |

... and 166 more (see decision_divergence.csv).
