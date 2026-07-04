# vLLM External-Admission Baseline Comparison — Summary

**Model:** `Qwen/Qwen2.5-0.5B-Instruct`
**Run status:** completed
**Generated:** 2026-07-04 00:24:18 UTC

## Policies compared

| Policy | n_total | n_completed | n_failed | Declined (load-shed) | Never-admitted (adapter bug) | Arrival-norm. WG | SLO violation (completed) | Mean TTFT (s) | Mean server latency (s) | Req/s |
|---|---|---|---|---|---|---|---|---|---|---|
| edf | 24 | 24 | 0 | 0 | 0 | 0.7955 | 0.1250 | 0.0160 | 0.8698 | 1.722 |
| estimated_service_time_first | 24 | 24 | 0 | 0 | 0 | 0.7955 | 0.1250 | 0.0159 | 0.8699 | 1.722 |
| fifo | 24 | 24 | 0 | 0 | 0 | 0.7500 | 0.1667 | 0.0157 | 0.8972 | 1.662 |
| least_laxity_first | 24 | 24 | 0 | 0 | 0 | 0.7955 | 0.1250 | 0.0159 | 0.9241 | 1.655 |
| selector | 24 | 24 | 0 | 0 | 0 | 0.7955 | 0.1250 | 0.0171 | 0.9802 | 1.450 |
| shortest_output_first | 24 | 24 | 0 | 0 | 0 | 0.7500 | 0.1667 | 0.0159 | 0.9216 | 1.658 |
| vllm_direct | 24 | 24 | 0 | 0 | 0 | 0.7500 | 0.1667 | 0.0156 | 0.8693 | 1.723 |

## Bootstrap 95% confidence intervals (arrival-norm. WG)

| Policy | Point estimate | CI low | CI high |
|---|---|---|---|
| edf | 0.7955 | 0.6000 | 1.0000 |
| estimated_service_time_first | 0.7955 | 0.5909 | 1.0000 |
| fifo | 0.7500 | 0.5471 | 0.9500 |
| least_laxity_first | 0.7955 | 0.5999 | 1.0000 |
| selector | 0.7955 | 0.5908 | 1.0000 |
| shortest_output_first | 0.7500 | 0.5400 | 0.9319 |
| vllm_direct | 0.7500 | 0.5510 | 0.9474 |
| selector_minus_edf | -0.0021 | -0.2782 | 0.2695 |
| selector_minus_estimated_service_time_first | -0.0040 | -0.2791 | 0.2609 |
| selector_minus_fifo | 0.0378 | -0.2498 | 0.3200 |
| selector_minus_least_laxity_first | -0.0019 | -0.2755 | 0.2727 |
| selector_minus_shortest_output_first | 0.0456 | -0.2411 | 0.3274 |
| selector_minus_vllm_direct | 0.0384 | -0.2355 | 0.3098 |

## Decision divergence

- Cells compared (selector vs. each baseline): 72
- Requests where selector's real SLO outcome differed from a baseline's real SLO outcome: 3
- See `decision_divergence.csv` (per-cell Kendall tau / rank mismatches) and `selector_vs_baselines_examples.md` (worked examples).

Policies not compared: generated_heuristic, best_generated — see `docs/vllm_real_serving_external_baseline_pilot.md` for why.

See `aggregate_by_policy.csv`, `aggregate_by_policy_and_regime.csv`, `aggregate_by_concurrency.csv`, `aggregate_by_target_output_tokens.csv`, `aggregate_by_prompt_bucket.csv` for breakdowns, and `errors.jsonl` for failure detail.
