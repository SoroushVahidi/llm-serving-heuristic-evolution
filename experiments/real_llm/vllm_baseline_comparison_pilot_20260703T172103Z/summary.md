# vLLM External-Admission Baseline Comparison — Summary

**Model:** `Qwen/Qwen2.5-0.5B-Instruct`
**Run status:** completed
**Generated:** 2026-07-03 17:23:06 UTC

## Policies compared

| Policy | n_total | n_completed | n_failed | Arrival-norm. WG | SLO violation (completed) | Mean TTFT (s) | Mean server latency (s) | Req/s |
|---|---|---|---|---|---|---|---|---|
| edf | 24 | 24 | 0 | 0.7955 | 0.1250 | 0.0157 | 0.8816 | 1.704 |
| estimated_service_time_first | 24 | 24 | 0 | 0.7955 | 0.1250 | 0.0162 | 0.9298 | 1.643 |
| fifo | 24 | 24 | 0 | 0.7500 | 0.1667 | 0.0159 | 0.9371 | 1.584 |
| least_laxity_first | 24 | 24 | 0 | 0.7955 | 0.1250 | 0.0161 | 0.8791 | 1.705 |
| shortest_output_first | 24 | 24 | 0 | 0.7500 | 0.1667 | 0.0160 | 0.9294 | 1.641 |
| vllm_direct | 24 | 24 | 0 | 0.7500 | 0.1667 | 0.0160 | 1.0153 | 1.509 |

Policies not compared: generated_heuristic, best_generated, selector — see `docs/vllm_real_serving_external_baseline_pilot.md` for why.

See `aggregate_by_policy.csv`, `aggregate_by_concurrency.csv`, `aggregate_by_target_output_tokens.csv`, `aggregate_by_prompt_bucket.csv` for breakdowns, and `errors.jsonl` for failure detail.
