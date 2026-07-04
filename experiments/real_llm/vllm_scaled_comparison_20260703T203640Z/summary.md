# vLLM External-Admission Baseline Comparison — Summary

**Model:** `Qwen/Qwen2.5-0.5B-Instruct`
**Run status:** completed
**Generated:** 2026-07-03 21:30:53 UTC

## Policies compared

| Policy | n_total | n_completed | n_failed | Arrival-norm. WG | SLO violation (completed) | Mean TTFT (s) | Mean server latency (s) | Req/s |
|---|---|---|---|---|---|---|---|---|
| edf | 540 | 540 | 0 | 0.2303 | 0.6481 | 0.0187 | 1.4500 | 1.169 |
| estimated_service_time_first | 540 | 540 | 0 | 0.2235 | 0.6556 | 0.0194 | 1.4849 | 1.148 |
| fifo | 540 | 540 | 0 | 0.2440 | 0.6148 | 0.0190 | 1.4672 | 1.157 |
| least_laxity_first | 540 | 540 | 0 | 0.2277 | 0.6500 | 0.0191 | 1.4681 | 1.157 |
| selector | 540 | 481 | 59 | 0.2274 | 0.6154 | 0.0202 | 1.4227 | 1.149 |
| shortest_output_first | 540 | 540 | 0 | 0.2432 | 0.6167 | 0.0193 | 1.4858 | 1.142 |
| vllm_direct | 540 | 540 | 0 | 0.2414 | 0.6185 | 0.0198 | 1.4956 | 1.139 |

## Weighted goodput by policy and regime

| Policy | Regime | n_total | n_completed | Arrival-norm. WG | SLO violation |
|---|---|---|---|---|---|
| edf | bursty_tight | 180 | 180 | 0.1774 | 0.7500 |
| edf | overloaded_mixed_priority | 180 | 180 | 0.0983 | 0.7944 |
| edf | steady_moderate | 180 | 180 | 0.4787 | 0.4000 |
| estimated_service_time_first | bursty_tight | 180 | 180 | 0.1640 | 0.7667 |
| estimated_service_time_first | overloaded_mixed_priority | 180 | 180 | 0.1004 | 0.7889 |
| estimated_service_time_first | steady_moderate | 180 | 180 | 0.4665 | 0.4111 |
| fifo | bursty_tight | 180 | 180 | 0.1962 | 0.7000 |
| fifo | overloaded_mixed_priority | 180 | 180 | 0.1368 | 0.7222 |
| fifo | steady_moderate | 180 | 180 | 0.4512 | 0.4222 |
| least_laxity_first | bursty_tight | 180 | 180 | 0.1720 | 0.7556 |
| least_laxity_first | overloaded_mixed_priority | 180 | 180 | 0.1047 | 0.7833 |
| least_laxity_first | steady_moderate | 180 | 180 | 0.4665 | 0.4111 |
| selector | bursty_tight | 180 | 164 | 0.1893 | 0.7073 |
| selector | overloaded_mixed_priority | 180 | 142 | 0.0843 | 0.7535 |
| selector | steady_moderate | 180 | 175 | 0.4535 | 0.4171 |
| shortest_output_first | bursty_tight | 180 | 180 | 0.1962 | 0.7000 |
| shortest_output_first | overloaded_mixed_priority | 180 | 180 | 0.1346 | 0.7278 |
| shortest_output_first | steady_moderate | 180 | 180 | 0.4512 | 0.4222 |
| vllm_direct | bursty_tight | 180 | 180 | 0.1935 | 0.7056 |
| vllm_direct | overloaded_mixed_priority | 180 | 180 | 0.1325 | 0.7278 |
| vllm_direct | steady_moderate | 180 | 180 | 0.4512 | 0.4222 |

## Bootstrap 95% confidence intervals (arrival-norm. WG)

| Policy | Point estimate | CI low | CI high |
|---|---|---|---|
| edf | 0.2303 | 0.1984 | 0.2659 |
| estimated_service_time_first | 0.2235 | 0.1919 | 0.2580 |
| fifo | 0.2440 | 0.2103 | 0.2782 |
| least_laxity_first | 0.2277 | 0.1957 | 0.2634 |
| selector | 0.2158 | 0.1848 | 0.2489 |
| shortest_output_first | 0.2432 | 0.2107 | 0.2785 |
| vllm_direct | 0.2414 | 0.2082 | 0.2756 |
| selector_minus_edf | -0.0147 | -0.0590 | 0.0294 |
| selector_minus_estimated_service_time_first | -0.0078 | -0.0551 | 0.0388 |
| selector_minus_fifo | -0.0279 | -0.0739 | 0.0198 |
| selector_minus_least_laxity_first | -0.0123 | -0.0605 | 0.0341 |
| selector_minus_shortest_output_first | -0.0279 | -0.0758 | 0.0193 |
| selector_minus_vllm_direct | -0.0253 | -0.0717 | 0.0211 |

## Decision divergence

- Cells compared (selector vs. each baseline): 648
- Requests where selector's real SLO outcome differed from a baseline's real SLO outcome: 186
- See `decision_divergence.csv` (per-cell Kendall tau / rank mismatches) and `selector_vs_baselines_examples.md` (worked examples).

Policies not compared: generated_heuristic, best_generated — see `docs/vllm_real_serving_external_baseline_pilot.md` for why.

See `aggregate_by_policy.csv`, `aggregate_by_policy_and_regime.csv`, `aggregate_by_concurrency.csv`, `aggregate_by_target_output_tokens.csv`, `aggregate_by_prompt_bucket.csv` for breakdowns, and `errors.jsonl` for failure detail.
