# Phase 2B.13 Failure Cases Summary

**Experiment:** `phase2b13_selector_training_after_diversity`  
**Branch:** `phase2b13-selector-training-after-diversity`  
**Date:** 2026-06-26

---

## Carry-Forward From Phase 2B.12

| ID | Description | Status in 2B.13 |
|---|---|---|
| fail_007 | Rule selector under-dispatches SCORPIO | Carry-forward (partially deferred; offline artifact) |
| fail_008 | Missing rule targets: best_fit, mbb, SOF, estST | Carry-forward (RF may cover; needs result check) |
| fail_009 | sarathi_style rule target wrong; AC wins prefill-heavy | **REPAIRED** (Rule 5: sarathi→AC) |
| fail_010 | 172 < 200 window threshold; RF/DT blocked | Addressed by extending seeds to [6..11] + 2 new workloads |
| fail_011 | All-complete diversity windows have tie-breaking labels | Partially addressed (2 new workloads target genuine WG gaps) |

---

## Phase 2B.13 New Failure Cases

### fail_012: All-complete regime limits RF/DT signal quality

**Observed:** 238/256 windows (93%) are "trivial all-complete" (best_policy WG < 0.99 for
only 18 windows). RF train accuracy = 100% (overfit to tie-breaking labels). RF test WG =
0.9975 ≈ best fixed SCORPIO WG = 0.9975. Gap vs best_fixed = +0.0000.

**Expected:** RF/DT trained on ≥200 windows should generalize to held-out regimes with
WG improvements over fixed baselines.

**Actual:** RF achieves no WG improvement over simply dispatching SCORPIO for all windows.
The RF effectively learns a weighted combination of SCORPIO-first features.

**Root cause:** 93% of windows have WG≈1.0 for virtually all policies. Labels primarily
reflect tie-breaking on secondary metrics (slo_violation_rate → p95_ttft → p95_latency),
not genuine WG gaps. RF accuracy on secondary metrics (60.6% test) does not translate to
WG improvement.

**Impact:** RF selector is WG-equivalent to best fixed baseline; no selector outperforms
always-SCORPIO in WG terms on this dataset. RF is valuable for secondary metric accuracy
(p95_ttft, SLO violation rate) but that is not captured in the WG objective.

**Resolution:** To get genuine WG signal: (1) use workloads with tighter SLO thresholds
where many requests fail to complete, (2) create high-overload scenarios where GPU queue
depth regularly exceeds max_active, (3) use SLO violation rate as a supplementary label.

**Status:** Open (known limitation; documented)

---

### fail_013: DT underperforms on heldout test set

**Observed:** DT test accuracy = 45.5%, test WG = 0.9752, gap vs best_fixed = −0.0224.
DT fails to generalize to heldout workload conditions (seeds 3–5).

**Expected:** DT test WG ≥ rule selector WG (0.9803 on heldout) and ≥ best_fixed (0.9975).

**Root cause:** DT max_depth=8 is insufficient for an 11-class label space when the
decision boundary requires nuanced feature interactions. The DT overfits to majority-class
SCORPIO labels and mispredicts the distribution under heldout workloads.

**Impact:** DT should not be used as primary selector. RF (test WG = 0.9975, gap = 0.0000)
is the correct ML choice if an ML selector is needed.

**Resolution:** Increase DT max_depth (to 12+) or drop DT in favor of RF. However, given
the all-complete regime (fail_012), higher DT accuracy would still not yield WG gains.

**Status:** Known limitation; DT discarded in favor of RF.

---

### fail_014: New high-differentiation workloads still mostly all-complete

**Observed:** Overall trivial fraction remained 238/256 (93%) after adding 2 new workloads.
The new workloads contributed meaningful WG-gap windows (out of 18 total meaningful windows)
but most of their windows were still all-complete.

**Expected:** `div_overloaded_all_loose_slo` and `div_kv_saturated_medium_slo` would
produce genuine WG gaps where SCORPIO's admission throttling hurts completion.

**Actual:** At 75 req/s with max_active_sequences=4 and max_kv_tokens=32768, the GPU
simulator can still complete most requests on time with loose SLO (slo_slack=3.0, 12.0).
The KV saturation workload (300-token outputs) does produce some WG gaps, but the SLO
slack (4.0× medium) is still loose enough for most policies to complete all requests.

**Impact:** Limited: only 18/256 windows have genuine WG gaps overall. New workloads
helped at the margin but did not create the decisive policy differentiation needed for
RF to learn truly discriminative features.

**Resolution:** To create meaningful WG gaps, workloads need either:
  (a) Much tighter SLOs (slo_slack ≤ 1.5× for majority of requests)
  (b) Arrival rates so high that even FIFO can't complete all requests (70%+ utilization)
  (c) Max_active_sequences=4 is a strong binding constraint — workloads hitting this limit
      systematically differentiate admission/scheduling policies.

**Status:** Known limitation; informative for Phase 2B.14 workload design.

---

## Summary Table (After Results Available)

| ID | Phase | Description | Resolution | Status |
|---|---|---|---|---|
| fail_007 | 2B.12 | SCORPIO under-dispatched by rule selector | Deferred (offline artifact) | Carry-forward |
| fail_008 | 2B.12 | Missing rule targets: best_fit, mbb, SOF, estST | RF may cover via learned features | Carry-forward |
| fail_009 | 2B.12 | sarathi_style wrong for prefill-heavy | **RESOLVED** (Rule 5 → AC) | Closed |
| fail_010 | 2B.12 | 172 < 200 window threshold | Extended seeds + 2 new workloads → 256 windows | **Resolved** |
| fail_011 | 2B.12 | All-complete tie-breaking labels | 2 new targeted workloads; RF training | Partially addressed |
| fail_012 | 2B.13 | All-complete regime limits RF signal | [TBD] | Open |
| fail_013 | 2B.13 | RF/DT may overfit tie-breaking patterns | [TBD] | Open |
| fail_014 | 2B.13 | New workloads may still be all-complete | [TBD] | Open |
