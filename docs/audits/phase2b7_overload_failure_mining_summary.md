# Phase 2B.7 Overloaded Sweep Summary

**Phase:** 2B.7  
**Date:** 2026-06-25  
**Config:** `configs/phase2b7_overload_failure_mining.yaml`  
**Policies:** 19 deployable (no oracle)  
**Workloads:** 4 (see below)  
**Seeds:** 3 per workload  

---

## Did overloaded workloads produce policy differentiation?

**Yes — substantial differentiation.** WG ranges:

| Workload | Best WG | Worst WG | Range |
|---|---|---|---|
| overloaded_mixed_slo | 0.905 | 0.344 | **0.561** |
| high_prediction_noise | 0.988 | 0.478 | **0.510** |
| overloaded_prefill_heavy | 1.000 | 1.000 | 0.000 (underloaded) |
| kv_pressure_decode_heavy | 0.477 | 0.051 | **0.426** |

3 of 4 workloads showed meaningful differentiation. `overloaded_prefill_heavy` was accidentally underloaded (loose SLO + not enough arrival pressure).

---

## Per-workload policy rankings (top 5 each)

### overloaded_mixed_slo
| Rank | Policy | WG | SLO viol rate |
|---|---|---|---|
| 1 | `slo_slack_score` | 0.905 | 0.106 |
| 1 | `orca_style` | 0.905 | 0.106 |
| 3 | `weighted_shortest_processing` | 0.893 | 0.098 |
| 4 | `shortest_output_first` | 0.851 | 0.127 |
| 4 | `vllm_style_token_budget` | 0.851 | 0.127 |
| ... | `admission_control` | 0.813 | 0.170 |
| ... | `least_laxity_first` | 0.474 | 0.462 |
| last | `sarathi_style` | 0.344 | 0.534 |

### high_prediction_noise
| Rank | Policy | WG | SLO viol rate |
|---|---|---|---|
| 1 | **`admission_control`** | **0.988** | 0.009 |
| 2 | `edf` | 0.986 | 0.011 |
| 3 | `orca_style` | 0.984 | 0.014 |
| 3 | `slo_slack_score` | 0.984 | 0.014 |
| 5 | `weighted_shortest_processing` | 0.937 | 0.052 |

### overloaded_prefill_heavy
All 19 policies: WG = 1.0000, SLO violation = 0.0. Underloaded — no differentiation.

### kv_pressure_decode_heavy
| Rank | Policy | WG | SLO viol rate |
|---|---|---|---|
| 1 | **`weighted_shortest_processing`** | **0.477** | 0.554 |
| 2 | `estimated_service_time_first` | 0.446 | 0.516 |
| 2 | `shortest_output_first` | 0.446 | 0.514 |
| 2 | `vllm_style_token_budget` | 0.446 | 0.514 |
| 5 | `multi_bin_batching` | 0.395 | 0.575 |
| ... | `admission_control` | 0.051 | 0.962 |
| last | `admission_control` | 0.051 | 0.962 |

---

## Overall (mean across 4 workloads)

| Rank | Policy | Mean WG |
|---|---|---|
| 1 | `weighted_shortest_processing` | **0.827** |
| 2 | `vllm_style_token_budget` | 0.805 |
| 2 | `shortest_output_first` | 0.805 |
| 4 | `estimated_service_time_first` | 0.799 |
| ... | `admission_control` | 0.713 |
| ... | `least_laxity_first` | 0.540 |
| last | `sarathi_style` | 0.484 |

**Best fixed baseline overall:** `weighted_shortest_processing` (WG=0.827)

---

## Which selector is best on average?

The rule-based selector always predicts `least_laxity_first` (Rule 1 fires for all 4 workloads).
- Rule-based WG ≈ 0.540 (mean across 3 non-trivial workloads)
- Best fixed WG = 0.827
- **Selector loss: −0.287**

The trained RF/DT selectors from Phase 2A.4 were not re-evaluated on these new workloads.

---

## Does admission control help after unit correction?

`admission_control` with `laxity_threshold=inf, step_size=0.001`:
- Win in `high_prediction_noise` (WG=0.988, rank 1)
- Average in `overloaded_mixed_slo` (WG=0.813, rank 9)
- Loss in `kv_pressure_decode_heavy` (WG=0.051, rank last)

The unit fix is correct. The policy now computes laxity in seconds. However, since the default
threshold is `inf` (no filtering), behavior with the default is unchanged from before the fix.
The fix matters only for non-inf thresholds. A threshold of `0.0s` would now correctly mean
"admit only requests whose estimated service time fits within remaining deadline."

---

## Suspicious high WG / low completion fraction?

All 19 policies have `num_completed ≈ 728` (mean across 4 workloads, 3 seeds) — completion fractions are all ≈ 1.0. No suspicious low-completion high-WG outliers found. `admission_control` with `threshold=inf` admits all requests.

---

## Are results sufficient for publication claims?

**No.** Current limitations:
- Only 3 workloads produced differentiation (1 was underloaded)
- No real trace workloads (BurstGPT/ShareGPT) in this sweep
- Selector evaluation is estimated (approximated features), not from actual window dataset
- Phase 2A.4 trained selectors not evaluated on these new regimes

These results are sufficient for failure-case identification and rule improvement, but not publication-quality comparisons.

---

## Main failure patterns

1. **Rule 1 over-fires**: `min_slack < 1.0s` matches nearly all overloaded workloads with any tight-SLO class, routing to `least_laxity_first` which is poor under high overload.
2. **LLF + KV pressure = catastrophic**: LLF promotes large-output requests that hold KV slots; WG=0.101 in kv_pressure regime.
3. **Prediction noise degrades LLF more than EDF/AC**: With 70% noise, laxity estimates are unreliable.

See `docs/audits/phase2b7_failure_cases_summary.md` for registry and suggested actions.

---

## Recommended next step

3 unresolved failure cases were identified with clear patterns.

**Use CloudRift/LLM to synthesize improved rule conditions** targeting:
1. KV-pressure regime: add rule `kv_utilization > 0.8 and output_mean > 200 → weighted_shortest_processing`
2. High-noise regime: add rule `pred_output_cv > 0.8 → edf`
3. Mixed-SLO overload: replace LLF in Rule 1 with `slo_slack_score` for high-load scenarios

**API experiment constraints:**
- Use a small, capped CloudRift/Cohere call (< 1000 tokens in, < 500 tokens out)
- Log in `results/api_usage/api_usage_ledger.csv`
- Limit to 1-2 calls per failure pattern
- Focus on rule synthesis, not end-to-end generation
