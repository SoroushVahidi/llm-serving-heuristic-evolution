# AdmissionControlPolicy Threshold Calibration Summary

**Phase:** 2B.6  
**Date:** 2026-06-25  
**Script:** `scripts/calibrate_admission_threshold.py`  
**Workload:** Mixed-SLO, Poisson 25 req/s, 10s, 3 seeds (42, 0, 1)  
**GPU:** 1× proxy (max_active_seq=4, max_kv=32768)

---

## Key Finding: Unit Mismatch in Raw Laxity

The `AdmissionControlPolicy._laxity()` formula mixes incompatible units:

```python
laxity = slo_deadline(s) - now(s) - est(steps)
```

where `est = α × prompt_tokens + β × predicted_output_tokens` is in **decode steps**
(dimensionless count), while `(slo_deadline − now)` is in **seconds**.

With `step_size = 0.001 s/step`:
- A medium request (128 prompt + 96 output): `est ≈ 0.5×128 + 1.0×96 = 160 steps`
- A tight SLO of `slo_slack = 0.4s` gives `(deadline − now) ≈ 0.4`
- Raw laxity ≈ `0.4 − 160 = −159.6` — deeply negative even though the request **is feasible**

This means:
- `laxity_threshold = 0.0` admits **nothing** (completion=0%)
- `laxity_threshold = inf` (default) admits **everything** (no filtering)
- Only thresholds in the range [−200, +200] produce partial filtering

---

## Calibration Results

| threshold | mean_wg | slo_violation_rate | completion_fraction | note |
|---|---|---|---|---|
| inf | 0.9949 | 0.0039 | 1.0000 | default — no filtering, pure urgency sort |
| 200.0 | 1.0000 | 0.0000 | 0.7871 | drops ~21% requests; achieves 0 SLO violations |
| 100.0 | 1.0000 | 0.0000 | 0.3149 | drops ~69% requests |
| 50.0 | 1.0000 | 0.0000 | 0.0493 | drops ~95% requests (near-empty admission) |
| 0.0 | NaN | NaN | 0.0000 | admits nothing |
| −50.0 | NaN | NaN | 0.0000 | admits nothing |
| −100.0 | NaN | NaN | 0.0000 | admits nothing |

---

## Interpretation

1. **`threshold = inf` (default, recommended):** All requests admitted. Acts as urgency-sorted
   admission with no drop policy. WG=0.9949 with ~0.4% SLO violation rate. Safe for all workloads.

2. **`threshold = 200.0`:** Drops requests with raw laxity < −200. Achieves WG=1.0 (zero SLO
   violations) at the cost of dropping ~21% of arrivals. This threshold works because requests
   with `est > (deadline − now) + 200` are almost certainly infeasible, even after accounting
   for the unit mismatch.

3. **Thresholds ≤ 0:** Admit nothing in typical workloads. Avoid.

---

## Fix for Consistent Units

To use `admission_control` as a genuine admission-control filter, convert service proxy to seconds:

```python
# Option A: Convert est to seconds (multiply by step_size)
def _laxity(self, req, now, step_size=0.001):
    est_seconds = predicted_service_proxy(req, self.alpha, self.beta) * step_size
    return req.slo_deadline - now - est_seconds
    # Now threshold is in seconds; threshold=0.0 means "drop if est > remaining time"
```

With this fix, `threshold = 0.0` would correctly drop infeasible requests. Not implemented in
Phase 2B.6 (would change existing test behavior). Tracked as a future improvement.

---

## Recommendations

1. **Use `laxity_threshold = float("inf")` (default)** in all current experiments.
   This gives the policy a fair comparison as a pure urgency-sorter.

2. **Do not set `laxity_threshold ≤ 100`** without the unit-consistent fix — it will
   drop too many requests and produce NaN metrics.

3. **For a genuine admission-control experiment:** Apply Option A above, then sweep
   `threshold ∈ {0.0, 0.5, 1.0, 2.0, 5.0}` (all in seconds).

4. **Document safe claims:** The current baseline is "laxity-based urgency sorter with
   optional admission filtering" — not a reproduction of any published system.
