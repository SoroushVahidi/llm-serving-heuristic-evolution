# vLLM External-Admission Baseline Comparison — Summary

**Model:** `Qwen/Qwen2.5-0.5B-Instruct`
**Run status:** completed
**Generated:** 2026-07-04 02:06:15 UTC

## Policies compared

| Policy | n_total | n_completed | n_failed | Declined (load-shed) | Never-admitted (adapter bug) | Arrival-norm. WG | SLO violation (completed) | Mean TTFT (s) | Mean server latency (s) | Req/s |
|---|---|---|---|---|---|---|---|---|---|---|
| edf | 540 | 540 | 0 | 0 | 0 | 0.2277 | 0.6500 | 0.0194 | 1.4826 | 1.149 |
| estimated_service_time_first | 540 | 540 | 0 | 0 | 0 | 0.2277 | 0.6500 | 0.0193 | 1.4860 | 1.147 |
| fifo | 540 | 540 | 0 | 0 | 0 | 0.2414 | 0.6185 | 0.0193 | 1.4768 | 1.148 |
| least_laxity_first | 540 | 540 | 0 | 0 | 0 | 0.2235 | 0.6556 | 0.0194 | 1.4968 | 1.137 |
| selector | 540 | 483 | 57 | 57 | 0 | 0.2241 | 0.6211 | 0.0194 | 1.3951 | 1.167 |
| shortest_output_first | 540 | 540 | 0 | 0 | 0 | 0.2432 | 0.6167 | 0.0193 | 1.4906 | 1.142 |
| vllm_direct | 540 | 540 | 0 | 0 | 0 | 0.2397 | 0.6204 | 0.0193 | 1.4674 | 1.156 |

## Weighted goodput by policy and regime

| Policy | Regime | n_total | n_completed | Arrival-norm. WG | SLO violation |
|---|---|---|---|---|---|
| edf | bursty_tight | 180 | 180 | 0.1720 | 0.7556 |
| edf | overloaded_mixed_priority | 180 | 180 | 0.1047 | 0.7833 |
| edf | steady_moderate | 180 | 180 | 0.4665 | 0.4111 |
| estimated_service_time_first | bursty_tight | 180 | 180 | 0.1720 | 0.7556 |
| estimated_service_time_first | overloaded_mixed_priority | 180 | 180 | 0.1047 | 0.7833 |
| estimated_service_time_first | steady_moderate | 180 | 180 | 0.4665 | 0.4111 |
| fifo | bursty_tight | 180 | 180 | 0.1935 | 0.7056 |
| fifo | overloaded_mixed_priority | 180 | 180 | 0.1325 | 0.7278 |
| fifo | steady_moderate | 180 | 180 | 0.4512 | 0.4222 |
| least_laxity_first | bursty_tight | 180 | 180 | 0.1720 | 0.7556 |
| least_laxity_first | overloaded_mixed_priority | 180 | 180 | 0.0983 | 0.7944 |
| least_laxity_first | steady_moderate | 180 | 180 | 0.4604 | 0.4167 |
| selector | bursty_tight | 180 | 165 | 0.1790 | 0.7212 |
| selector | overloaded_mixed_priority | 180 | 143 | 0.0937 | 0.7413 |
| selector | steady_moderate | 180 | 175 | 0.4411 | 0.4286 |
| shortest_output_first | bursty_tight | 180 | 180 | 0.1935 | 0.7056 |
| shortest_output_first | overloaded_mixed_priority | 180 | 180 | 0.1325 | 0.7278 |
| shortest_output_first | steady_moderate | 180 | 180 | 0.4573 | 0.4167 |
| vllm_direct | bursty_tight | 180 | 180 | 0.1935 | 0.7056 |
| vllm_direct | overloaded_mixed_priority | 180 | 180 | 0.1282 | 0.7333 |
| vllm_direct | steady_moderate | 180 | 180 | 0.4512 | 0.4222 |

## Bootstrap 95% confidence intervals (arrival-norm. WG)

| Policy | Point estimate | CI low | CI high |
|---|---|---|---|
| edf | 0.2277 | 0.1967 | 0.2624 |
| estimated_service_time_first | 0.2277 | 0.1964 | 0.2614 |
| fifo | 0.2414 | 0.2081 | 0.2753 |
| least_laxity_first | 0.2235 | 0.1910 | 0.2594 |
| selector | 0.2123 | 0.1821 | 0.2463 |
| shortest_output_first | 0.2432 | 0.2100 | 0.2781 |
| vllm_direct | 0.2397 | 0.2068 | 0.2740 |
| selector_minus_edf | -0.0154 | -0.0592 | 0.0302 |
| selector_minus_estimated_service_time_first | -0.0154 | -0.0625 | 0.0317 |
| selector_minus_fifo | -0.0287 | -0.0746 | 0.0185 |
| selector_minus_least_laxity_first | -0.0113 | -0.0584 | 0.0348 |
| selector_minus_shortest_output_first | -0.0313 | -0.0775 | 0.0151 |
| selector_minus_vllm_direct | -0.0269 | -0.0733 | 0.0187 |

## Decision divergence

- Cells compared (selector vs. each baseline): 648
- Requests where selector's real SLO outcome differed from a baseline's real SLO outcome: 176
- See `decision_divergence.csv` (per-cell Kendall tau / rank mismatches) and `selector_vs_baselines_examples.md` (worked examples).

## Loss cases (selector vs. each baseline)

- Total loss cases: 149
- By baseline: {'fifo': 31, 'shortest_output_first': 31, 'vllm_direct': 30, 'estimated_service_time_first': 20, 'edf': 19, 'least_laxity_first': 18}
- See `loss_cases.csv`/`loss_cases.jsonl` (full detail), `loss_case_summary.md` (aggregates), and `loss_case_examples.md` (representative examples).

Policies not compared: generated_heuristic, best_generated — see `docs/vllm_real_serving_external_baseline_pilot.md` for why.

See `aggregate_by_policy.csv`, `aggregate_by_policy_and_regime.csv`, `aggregate_by_concurrency.csv`, `aggregate_by_target_output_tokens.csv`, `aggregate_by_prompt_bucket.csv` for breakdowns, and `errors.jsonl` for failure detail.
