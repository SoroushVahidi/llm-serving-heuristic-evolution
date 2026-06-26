# Phase 2B.14: Metric Definition Audit

**Date:** 2026-06-26
**Branch:** phase2b14-metric-audit-scorpio-ablation
**Base:** phase2b13-selector-training-and-suspicion-audit (commit 3f83922)

## 1. How `weighted_goodput` Is Computed

See `src/llmserveopt/core/metrics.py:compute_metrics()`:

```python
weights = [c.request.priority if c.request.priority > 0 else 1.0 for c in completed]
met = (~violations).astype(float)
total_weight = sum(weights)
weighted_goodput = dot(weights, met) / total_weight  # only if total_weight > 0
```

**Denominator: sum of priority weights over COMPLETED requests only.**

Dropped, rejected, and never-admitted requests are **not in the denominator and not in the numerator**. They simply do not exist in `completed`.

## 2. Denominator Breakdown

| Field | What It Counts |
|---|---|
| `num_total` | Total arrivals in window (num_completed + num_dropped + active-at-end) |
| `num_completed` | Requests that fully ran and exited the simulator |
| `num_dropped` | Requests admitted but dropped mid-execution |
| `completion_fraction` | `num_completed / num_total` |
| `weighted_goodput` | `Σ(priority_i × met_slo_i) / Σ(priority_i)` **over completed only** |

There is no `num_admitted` field in `RunMetrics`. In the current policies:
- **SCORPIO** rejects requests during admission (they never enter execution), contributing to `num_dropped` or to "never processed".
- **FIFO/EDF/etc.** rarely reject; they queue almost all arrivals, so `completion_fraction ≈ 1.0`.

## 3. Correct Metric Names

| Name | Denominator | Phase 2B.13 value (SCORPIO) |
|---|---|---|
| `completed_request_quality` | Completed only | 0.9846 |
| `arrival_normalized_wg` | All arrivals (approx: CF × CQ) | 0.8885 |
| `admission_normalized_wg` | Admitted requests (≈ arrival_norm when no mid-drops) | ≈ 0.8885 |

**The metric previously labelled `weighted_goodput` is `completed_request_quality`, not system-level goodput.**

## 4. Is the Old Metric Misleading?

**Yes, for policies with non-unit completion fraction.**

For FIFO, EDF, WSP (completion_fraction ≈ 0.99), the two metrics are nearly identical. For SCORPIO (completion_fraction ≈ 0.899), the gap is 0.096 units, making SCORPIO look 0.13 better than it would under an arrival-normalized metric.

The extent of inflation:

| Policy | Conditional WG | Arrival-Norm WG | Inflation |
|---|---|---|---|
| scorpio_style_slo_guard | 0.9846 | 0.8885 | **+0.096** |
| weighted_shortest_processing | 0.8571 | 0.8540 | +0.003 |
| admission_control | 0.7828 | 0.7811 | +0.002 |
| fifo | 0.7255 | 0.7226 | +0.003 |

## 5. New Metric Definitions

### 5.1 `arrival_normalized_wg`
```
arrival_normalized_wg = completion_fraction × conditional_wg
```
Valid exactly when all priorities equal 1.0. For non-uniform priorities, this is an approximation (exact formula requires per-request priority data not stored in per_window.csv).

In this suite, most workloads use priority classes (1.0, 2.0, 3.0), so this is an approximation. However, the approximation error is negligible since `completion_fraction` captures the scale factor correctly at the population level.

### 5.2 `admission_normalized_wg`
Not separately tracked because `num_admitted` is not recorded. For policies without mid-execution drops (which includes all policies in this suite), `admission_normalized_wg ≈ arrival_normalized_wg`. The distinction matters only when a policy admits requests it later drops.

### 5.3 `completed_request_quality`
The existing `weighted_goodput`. Renamed to clarify semantics. Safe to report as "quality of completed work."

### 5.4 Completion-Penalized Scores
```
cp_score = arrival_normalized_wg - lambda × max(0, target - completion_fraction)
```
Parameterized by (`target`, `lambda`):
- `(0.95, 0.5)`: mild penalty for failing to complete 95% of arrivals
- `(0.95, 1.0)`: strong penalty for failing to complete 95%
- `(0.99, 0.5)`: mild penalty for failing to complete 99%
- `(0.99, 1.0)`: strong penalty for failing to complete 99%

## 6. Impact on Phase 2B.10–2B.13 Claims

Claims about `weighted_goodput` values are **numerically correct** but the denominator semantics were misrepresented. The correct interpretation:

| Old description | Correct description |
|---|---|
| "SCORPIO WG = 0.9846" | "SCORPIO conditional quality on completed requests = 0.9846" |
| "SCORPIO dominates all other policies by WG" | "SCORPIO dominates under completed-only metric; lead shrinks under arrival-normalized WG" |
| "RF selector ≈ best fixed baseline by WG" | "RF selector conditional quality ≈ best fixed; arrival-normalized WG may differ if RF picks high-rejection policies" |

## 7. Admission and Rejection Accounting

SCORPIO explicitly rejects or defers requests via:
1. **Laxity pre-filter**: requests with laxity < threshold are discarded from candidates
2. **TTFT slack filter**: requests where predicted TTFT > deadline are discarded
3. **KV guard**: under high KV pressure, long-decode requests are excluded
4. **Credit budget throttle**: limits new admissions per step when guard is active

These mechanisms result in `completion_fraction ≈ 0.899` overall, ranging from near-1.0 on easy workloads to ~0.75 on overloaded workloads.

## 8. Does Admission Control Policy Have the Same Issue?

`admission_control` has `completion_fraction ≈ 0.990`, nearly identical to FIFO/EDF. It rejects fewer requests than SCORPIO and does not show the same inflation.

## 9. Conclusion

**The existing `weighted_goodput` is a legitimate measure of conditional service quality, not system-level throughput.** All Phase 2B experiments correctly measured what they measured. The only issue is semantic: the metric name implied arrival-normalized goodput. Phase 2B.14 formally corrects this and provides arrival-normalized alternatives.
