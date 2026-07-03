# vLLM External-Admission Baseline Comparison — Summary

**Model:** `Qwen/Qwen2.5-0.5B-Instruct`
**Run status:** completed
**Generated:** 2026-07-03 19:18:00 UTC

## Policies compared

| Policy | n_total | n_completed | n_failed | Arrival-norm. WG | SLO violation (completed) | Mean TTFT (s) | Mean server latency (s) | Req/s |
|---|---|---|---|---|---|---|---|---|
| edf | 24 | 24 | 0 | 0.7955 | 0.1250 | 0.0158 | 0.9083 | 1.639 |
| estimated_service_time_first | 24 | 24 | 0 | 0.7955 | 0.1250 | 0.0154 | 0.9123 | 1.675 |
| fifo | 24 | 24 | 0 | 0.7500 | 0.1667 | 0.0163 | 0.9930 | 1.568 |
| least_laxity_first | 24 | 24 | 0 | 0.7955 | 0.1250 | 0.0158 | 0.9153 | 1.667 |
| selector | 24 | 24 | 0 | 0.7955 | 0.1250 | 0.0157 | 0.9174 | 1.486 |
| shortest_output_first | 24 | 24 | 0 | 0.7500 | 0.1667 | 0.0164 | 0.8850 | 1.693 |
| vllm_direct | 24 | 24 | 0 | 0.7500 | 0.1667 | 0.0163 | 0.8817 | 1.701 |

Policies not compared: generated_heuristic, best_generated — see `docs/vllm_real_serving_external_baseline_pilot.md` for why.

See `aggregate_by_policy.csv`, `aggregate_by_concurrency.csv`, `aggregate_by_target_output_tokens.csv`, `aggregate_by_prompt_bucket.csv` for breakdowns, and `errors.jsonl` for failure detail.
