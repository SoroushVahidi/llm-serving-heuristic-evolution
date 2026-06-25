# Phase 2B.7 Failure Case Summary

**Phase:** 2B.7  
**Date:** 2026-06-25  
**Experiment:** `phase2b7_overload_failure_mining`  
**Config:** `configs/phase2b7_overload_failure_mining.yaml`  
**Seeds:** 3 per workload  

---

## Summary

The Phase 2B.7 overloaded sweep revealed 3 significant failure cases for the rule-based selector.
In all 3 cases, **Rule 1 fires too broadly**: it checks `fraction_tight_slo > 0.4 or min_slack < 1.0s`
and routes to `least_laxity_first`, which performs poorly in high-overload regimes.

| failure_id | workload | selector_policy | selector_wg | best_fixed | best_wg | delta |
|---|---|---|---|---|---|---|
| fail_001 | overloaded_mixed_slo | `least_laxity_first` | 0.474 | `slo_slack_score` | 0.905 | **−0.431** |
| fail_002 | high_prediction_noise | `least_laxity_first` | 0.584 | `admission_control` | 0.988 | **−0.404** |
| fail_003 | kv_pressure_decode_heavy | `least_laxity_first` | 0.101 | `weighted_shortest_processing` | 0.477 | **−0.376** |

**Phase 2B.7 Status:** All 3 unresolved when filed. No LLM escalation in Phase 2B.7.  
**Phase 2B.8 Status:** ✓ All 3 resolved by rule selector repair (`phase2b8-rule-selector-repair`).  
See `docs/audits/phase2b8_rule_selector_repair_summary.md` for details.

---

## Root Cause Analysis

### Common pattern: Rule 1 fires for all workloads

Rule 1 in `RuleBasedSelector.predict_one()`:
```python
if fts > 0.4 or min_slack < 1.0:
    return "least_laxity_first"
```

In all 4 overloaded workloads, at least one condition is met:
- `fraction_tight_slo > 0.4`: workloads with ≥40% tight-SLO requests (mixed_slo, prefill_heavy)
- `min_slack < 1.0s`: workloads with any SLO class ≤ 0.8s (kv_pressure tight SLO = 0.8s)

Once Rule 1 fires, `least_laxity_first` is chosen — but LLF performs poorly when:
1. The workload is genuinely overloaded (queuing builds, urgency sorting causes cascade)
2. Prediction noise is high (LLF service estimates unreliable)
3. KV cache is saturated (urgency sorting promotes large-output requests)

### Per-failure breakdown

**fail_001: overloaded_mixed_slo (delta=−0.431)**
- Rule fired: Rule 1 (`fraction_tight_slo=0.50 > 0.4`)
- Optimal policy: `slo_slack_score` (composite urgency + throughput)
- LLF loss: sorts aggressively by laxity → high-priority requests get early admission → cascades into FIFO-like behavior under queue build-up
- Pattern: **wrong_rule_fired / overload**

**fail_002: high_prediction_noise (delta=−0.404)**
- Rule fired: Rule 1 (`min_slack < 1.0s`)
- Optimal policy: `admission_control` (inf threshold, urgency-sorted)
- LLF loss: with 70% prediction noise, service estimates are unreliable → laxity rank is unreliable → LLF ordering becomes random
- Pattern: **wrong_rule_fired / output-prediction noise**

**fail_003: kv_pressure_decode_heavy (delta=−0.376)**
- Rule fired: Rule 1 (`min_slack = 0.8s < 1.0s`)
- Optimal policy: `weighted_shortest_processing`
- LLF loss: urgency sorting promotes long-output (large KV) requests → KV cache saturates → GPU thrashes → catastrophic WG=0.101
- Pattern: **wrong_rule_fired / high KV pressure**

---

## Missing selector features

The rule_based selector lacks features to distinguish:

| Bottleneck | Feature needed | Available? |
|---|---|---|
| KV saturation vs SLO tightness | `kv_utilization` (yes, but Rule 3 threshold=0.7 doesn't fire first) | Partial |
| Prediction noise magnitude | `pred_output_cv` (yes, but Rule 5 only checks low-CV condition) | Partial |
| System overload level | `arrival_rate_est`, `queue_length` (available in features) | Yes — unused |

---

## Admission control post unit-fix

`admission_control` (default `laxity_threshold=inf`, `step_size=0.001`) after the Phase 2B.7 unit fix:

| workload | WG | rank |
|---|---|---|
| overloaded_mixed_slo | 0.813 | 9th/19 |
| high_prediction_noise | **0.988** | **1st/19** |
| overloaded_prefill_heavy | 1.000 | (all tie) |
| kv_pressure_decode_heavy | 0.051 | **last/19** |

**Finding:** `admission_control` is not universally good. It wins with high prediction noise but
loses catastrophically in KV-saturated decode-heavy workloads — same root cause as LLF.
The unit-fix (laxity in seconds) is correct but threshold=inf means no filtering, so the policy
is still just an urgency sorter. Meaningful filtering requires `threshold=0.0s` calibration.

---

## Suggested next actions

| Failure | Suggested action | Priority |
|---|---|---|
| Rule 1 too broad | Add `queue_length` and `kv_utilization` as early rules; raise `min_slack` threshold from 1.0s to 0.5s or remove it | High |
| LLF poor under overload | Replace LLF in Rule 1 with `slo_slack_score` for high-SLO + high-load scenarios | High |
| admission_control bad in KV pressure | Add Rule: `kv_utilization > 0.8 and mean_pred_output_tokens > 200 → weighted_shortest_processing` | High |
| Missing noise feature | Add Rule: `pred_output_cv > 0.8 → edf or admission_control` | Medium |

These suggest using CloudRift/LLM to synthesise new rule conditions for the 3 unresolved patterns.
