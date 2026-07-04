# Selector vs. Fixed-Baseline Decision Divergence — Examples

Every row below is a request where the selector's actual admission run and a baseline's actual admission run (both real, over the identical request plan) produced a DIFFERENT SLO outcome for the same request_id. This is measured from real execution, not shadow evaluation: each policy including the selector ran independently against the real vLLM server over the identical plan.

| Regime | Bucket | Target tok | Concurrency | Baseline | Request | Selector SLO violated | Baseline SLO violated | Selector rank | Baseline rank |
|---|---|---|---|---|---|---|---|---|---|
| steady_moderate | medium | 128 | 1 | fifo | 19 | False | True | 0 | 1 |
| steady_moderate | medium | 128 | 1 | shortest_output_first | 19 | False | True | 0 | 1 |
| steady_moderate | medium | 128 | 1 | vllm_direct | 19 | False | True | 0 | 1 |
