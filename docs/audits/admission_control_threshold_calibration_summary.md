# AdmissionControlPolicy Threshold Calibration Summary

**Phase:** 2B.7 (updated; original in 2B.6)  
**Date:** 2026-06-25  
**Script:** `scripts/calibrate_admission_threshold.py`  
**Workload:** Mixed-SLO, Poisson 25 req/s, 10s, 3 seeds (42, 0, 1)  
**GPU:** 1× proxy (max_active_seq=4, max_kv=32768)

---

## Phase 2B.7 Unit Fix

Prior to Phase 2B.7, `AdmissionControlPolicy._laxity()` mixed units:
```python
# BEFORE (broken): seconds - steps
laxity = slo_deadline - now - predicted_service_proxy(req)  # proxy in steps!
```

After the fix, all terms are in **seconds**:
```python
# AFTER (correct): all in seconds
est_seconds = predicted_service_proxy(req) * step_size     # steps → seconds
laxity = slo_deadline - now - est_seconds
```

With `step_size=0.001 s/step`:
- A request with `prompt=128, output=64` has `est_steps = 128` → `est_s = 0.128s`
- A tight SLO slack of `0.4s` gives `laxity = 0.4 - 0 - 0.128 = 0.272s > 0` (feasible!)
- `threshold=0.0s` now correctly admits feasible requests and filters infeasible ones

---

## Calibration Results (post unit fix)

| threshold | mean_wg | slo_violation_rate | completion_fraction | note |
|---|---|---|---|---|
| inf | 0.9949 | 0.0039 | 1.0000 | default — no filtering, pure urgency sort |
| 200.0 s | 0.9949 | 0.0039 | 1.0000 | very loose — no requests have laxity < −200s |
| 100.0 s | 0.9949 | 0.0039 | 1.0000 | loose — no requests filtered |
| 50.0 s | 0.9949 | 0.0039 | 1.0000 | moderate — no requests filtered |
| **0.0 s** | **0.9983** | **0.0013** | **0.9882** | **correct: filters infeasible requests** |
| −50.0 s | NaN | NaN | 0.0000 | too strict — filters all (needs laxity ≥ 50s) |
| −100.0 s | NaN | NaN | 0.0000 | too strict — filters all |

**Key finding:** `threshold=0.0s` correctly filters ~1.2% infeasible requests while
*improving* WG (0.9983 > 0.9949). This is the semantically correct behavior.

---

## Comparison with Phase 2B.6 results (before unit fix)

Before the fix, `threshold=200.0` dropped 21% of requests and `threshold=0.0` dropped 100%.
After the fix:
- `threshold=0.0s` drops only ~1.2% (truly infeasible requests)
- `threshold=200.0s` drops nothing (200 seconds is extremely loose)

The pre-fix behavior was broken. All previous calibration results in Phase 2B.6 are superseded.

---

## Recommendations (updated)

1. **`laxity_threshold=float("inf")` (default):** All requests admitted; urgency-sorted.
   Appropriate when admission filtering is not the research goal.

2. **`laxity_threshold=0.0` + `step_size=0.001`:** Correct semantic: filters requests
   whose estimated service time exceeds remaining deadline. WG slightly improves (+0.003).

3. **To use as genuine admission control:** Set `laxity_threshold=T_s` where `T_s` is
   a tolerance in seconds for prediction error (e.g., `T_s=0.1` allows up to 100ms of
   laxity deficit before dropping).

4. **For Phase 2B.7+ experiments:** Always pass the correct `step_size` matching your
   config's `simulator.step_size`. Default `0.001` matches all current experiment configs.

---

## Phase 2B.7 sweep results

Under overloaded conditions, `admission_control` (threshold=inf, step_size=0.001):

| Workload | WG | Rank |
|---|---|---|
| overloaded_mixed_slo | 0.813 | 9/19 |
| high_prediction_noise | **0.988** | **1/19** |
| overloaded_prefill_heavy | 1.000 | (all tie) |
| kv_pressure_decode_heavy | 0.051 | **19/19** |

**Finding:** `admission_control` with `threshold=inf` is essentially an urgency-sorter.
It wins when SLO management matters (high_prediction_noise) but loses catastrophically
when the bottleneck is KV saturation (kv_pressure_decode_heavy), same as LLF.
Setting `threshold=0.0s` would actively drop the infeasible requests in that regime,
potentially recovering WG — this is a recommended next experiment.
