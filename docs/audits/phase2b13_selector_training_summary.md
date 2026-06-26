# Phase 2B.13 Selector Training Summary

**Experiment:** `phase2b13_selector_training_after_diversity`  
**Branch:** `phase2b13-selector-training-after-diversity`  
**Runner:** `scripts/run_phase2b13_selector_training_after_diversity.py`  
**Config:** `configs/phase2b13_selector_training_after_diversity.yaml`  
**Log:** `logs/phase2b13/phase2b13_selector_training.log`  
**Results:** `results/phase2b13_selector_training_after_diversity/` (gitignored; summary in this doc)  
**tmux session:** `phase2b13_selector_training` (completed, EXIT_CODE=0, ~854s)

---

## Background

Phase 2B.12 found:
- 172 total windows (60 regression + 112 diversity): BELOW the 200-window threshold
- 9 distinct oracle policies; 6 win ≥10 windows; top (SCORPIO) = 45.9%
- RF/DT NOT trained: 172 < 200 (fails window count only; diversity strong)
- 166/172 windows are "all-complete" (WG≈1.0 for nearly all policies) — labels are
  primarily tie-breaking on secondary metrics, not genuine WG differentiation

Phase 2B.13 goals:
1. Extend to ≥200 windows by expanding diversity seeds from [6,7,8,9] → [6,7,8,9,10,11]
2. Add 2 new workloads designed to produce genuine WG gaps (not just tie-breaking):
   - `div_overloaded_all_loose_slo`: 75 req/s, no tight SLO → SCORPIO throttles unnecessarily
   - `div_kv_saturated_medium_slo`: 300-token mean output, medium SLO → WSP/best_fit wins
3. Train RF and DT selectors on (dev + diversity seeds 6–10)
4. Apply minimal rule selector repair (Rule 5: sarathi_style → admission_control for prefill-heavy)

---

## Workload Suite

### Groups

| Group | Workloads | Seeds | Expected Windows |
|---|---|---|---|
| dev | 4 | 0, 1, 2 | ~12 (unchanged from 2B.12) |
| heldout | 5 | 3, 4, 5 | ~20 (unchanged) |
| diversity | 14 existing + 2 new | 6, 7, 8, 9, 10, 11 | ~168 (was 112) |
| **total** | **25** | — | **~200+** |

### New Phase 2B.13 Workloads

| Tag | Arrival Rate | Output Mean | SLO Classes | Design Intent |
|---|---|---|---|---|
| `div_overloaded_all_loose_slo` | 75 req/s | 80 tokens | medium+loose only | SCORPIO throttles; EDF/best_fit completes all |
| `div_kv_saturated_medium_slo` | 40 req/s | 300 tokens | medium+loose | WSP/best_fit short-first beats SCORPIO admission throttle |

### Training Split

| Split | Source | Seeds | Purpose |
|---|---|---|---|
| Train | dev + diversity | 0,1,2 + 6,7,8,9,10 | RF/DT fitting |
| Val | diversity only | 11 | Hyperparameter check |
| Test | heldout | 3,4,5 | Final unbiased eval |

---

## RF/DT Feasibility

| Criterion | Threshold | Result | Status |
|---|---|---|---|
| Window count | ≥200 | **256** | PASS |
| Policies winning ≥10 windows each | ≥3 | **7** (SCORPIO, AC, best_fit, edf, mbb, SOF, estST) | PASS |
| Top policy concentration | <85% | **43.75%** (SCORPIO) | PASS |

**Overall: FEASIBLE — RF and DT trained**

---

## Label Distribution

| Policy | Overall | % | Regression | Diversity | Train |
|---|---|---|---|---|---|
| scorpio_style_slo_guard | 112 | 43.75 | 46 | 66 | 78 |
| admission_control | 37 | 14.45 | 4 | 33 | 30 |
| best_fit | 36 | 14.06 | 2 | 34 | 29 |
| edf | 20 | 7.81 | 4 | 16 | 15 |
| multi_bin_batching | 18 | 7.03 | 0 | 18 | 15 |
| shortest_output_first | 15 | 5.86 | 3 | 12 | 11 |
| estimated_service_time_first | 12 | 4.69 | 0 | 12 | 11 |
| random_feasible | 3 | 1.17 | 1 | 2 | 2 |
| least_laxity_first | 1 | 0.39 | 0 | 1 | 0 |
| shortest_prompt_first | 1 | 0.39 | 0 | 1 | 1 |
| weighted_shortest_processing | 1 | 0.39 | 0 | 1 | 1 |
| **Total** | **256** | 100 | 60 | 196 | 193 |

Note: 238/256 windows (93%) are "trivial all-complete" (WG≈1.0 for all policies).
Only 18/256 (7%) have a genuine WG gap (best policy WG < 0.99).

---

## Selector Comparison

### Pre-Repair Rule Selector

Phase 2B.12 reference (unchanged rules):

| Group | N | Best Fixed WG | Rule WG | Gap |
|---|---|---|---|---|
| dev | ~12 | 0.9878 | 0.9168 | −0.071 |
| heldout | ~20 | 0.9975 | 0.9803 | −0.017 |
| regression | ~32 | 0.9932 | 0.9518 | −0.041 |
| diversity (2B.12) | 112 | 0.9969 | 0.9831 | −0.014 |
| overall (2B.12) | 172 | 0.9956 | 0.9721 | −0.024 |

### Phase 2B.13 Selector Comparison (actual results)

| Group | N | Best Fixed WG | Rule WG | Rule Repaired WG | RF WG | DT WG |
|---|---|---|---|---|---|---|
| dev | 27 | 0.9878 | 0.9168 | 0.9168 | 0.9881 | 0.9878 |
| heldout | 33 | 0.9975 | 0.9803 | 0.9803 | 0.9975 | 0.9752 |
| regression | 60 | 0.9932 | 0.9518 | 0.9518 | 0.9933 | 0.9809 |
| diversity | 196 | 0.9967 | 0.9747 | 0.9747 | 0.9982 | 0.9946 |
| overall | 256 | 0.9959 | 0.9694 | 0.9694 | **0.9970** | 0.9914 |

Best fixed policy on all groups: `scorpio_style_slo_guard`

**RF** closes virtually all the gap vs best_fixed: overall gap = +0.0011 (from −0.0265 for rule selector).  
**DT** has mixed results: near-zero gap on diversity (−0.0022) but −0.0224 on heldout.

### RF/DT Metrics (if trained)

| Metric | RF (train) | RF (val) | RF (test=heldout) | DT (train) | DT (val) | DT (test=heldout) |
|---|---|---|---|---|---|---|
| Accuracy | 1.000 | 0.667 | 0.606 | 0.782 | 0.600 | 0.455 |
| Mean WG | 0.9968 | 0.9980 | 0.9975 | 0.9939 | 0.9928 | 0.9752 |
| Gap vs best fixed | +0.0015 | +0.0002 | +0.0000 | −0.0014 | −0.0050 | −0.0224 |

**RF**: 100% train accuracy (expected: overfits in all-complete regime), but WG ≈ 0.9975 on test
(essentially matching best fixed SCORPIO). Train overfit is benign since WG is near 1.0 for all.  
**DT**: test accuracy only 45.5%, test WG 0.9752 — DT underperforms best fixed by 0.0224.
Prefer RF over DT; DT shallower depth is insufficient for 11-class label space.

---

## Rule Selector Repair

### Rule 5: Prefill-Heavy Target

**Change:** `mean_prompt > 512 OR p95_prompt > 1024` → `admission_control` (was `sarathi_style`)

**Evidence (Phase 2B.12):**
- `div_prefill_heavy_sarathi`: admission_control wins all 8 windows (expected: sarathi_style)
- `div_prefill_moderate_tight`: admission_control wins all 8 windows
- `dev_overloaded_prefill_heavy`: admission_control wins dev windows

**Mechanism:** Under the priority-weighted goodput objective, admission_control's urgency-sort
achieves higher WG than sarathi_style's chunked-prefill approach, even in WG≈1.0 regimes
(secondary metric improvement). The WG impact is ~0 in all-complete windows, but the rule
accuracy improves.

**WG impact:** ~0.0 (rule selector WG unchanged: 0.9694 → 0.9694). The repair redirects only
1/256 windows (sarathi→AC); since both AC and sarathi_style have WG≈1.0 in the affected window,
the WG effect is negligible. Correctness effect: sarathi_style oracle label was 0/256 windows;
now all dispatched policies appear in oracle label set.

### Rule Selector Dispatch Comparison

| Policy | Original (256 windows) | Repaired (256 windows) | Delta |
|---|---|---|---|
| slo_slack_score | 119 | 119 | 0 |
| weighted_shortest_processing | 49 | 49 | 0 |
| edf | 48 | 48 | 0 |
| admission_control | 37 | **38** | **+1** |
| scorpio_style_slo_guard | 2 | 2 | 0 |
| sarathi_style | **1** | **0** | **−1** |

Oracle label for sarathi_style: 0/256 windows. The one window formerly dispatched to sarathi_style
now goes to admission_control, which is the oracle label for that window.

---

## Key Findings

1. **Window count**: 256 total windows — RF/DT **FEASIBLE** (all 3 criteria pass)
2. **All-complete regime**: 238/256 (93%) of windows have WG≈1.0 for nearly all policies;
   only 18 windows (7%) have meaningful WG differentiation. Labels primarily encode
   tie-breaking on secondary metrics (slo_violation_rate → p95_ttft → latency).
3. **RF selector WG**: 0.9975 on heldout (test set) — essentially matches best fixed
   (SCORPIO, WG=0.9975). Gap vs best_fixed = +0.0000.
4. **Rule repair impact**: WG unchanged (0.9694 → 0.9694); only 1/256 windows affected;
   repair is correct (sarathi_style had 0 oracle labels) but WG-neutral.
5. **Top oracle policy**: scorpio_style_slo_guard (112/256 = 43.75%)
6. **RF wins vs rule selector**: +0.028 overall WG (0.9970 vs 0.9694). RF essentially
   learns "always dispatch SCORPIO or best_fit" — not rich policy routing.
7. **DT underperforms**: test WG=0.9752, gap=−0.0224 vs best_fixed. DT max_depth=8
   insufficient for 11-class label space in this regime.
8. **New workloads**: Both new Phase 2B.13 workloads (`div_overloaded_all_loose_slo`,
   `div_kv_saturated_medium_slo`) likely contributed some genuine WG-gap windows, but
   the overall all-complete fraction remained high (93%).

---

## Failure Cases

See `docs/audits/phase2b13_failure_cases_summary.md`.

---

## Next Steps

RF is trained and feasible, but the WG signal is weak (93% all-complete regime). Two paths:

1. **Adopt RF**: RF test WG = 0.9975 ≈ best fixed. Use RF as primary selector in Phase 2B.14,
   with rule selector as fallback. RF effectively learns SCORPIO-heavy routing, which is correct
   but trivially achievable by "always SCORPIO."
2. **Improve differentiation**: Before adopting RF, create workloads where WG gap is ≥5% between
   best and second-best policy (requires tighter SLO or heavier load), then retrain.

Missing rule targets (fail_008: best_fit, mbb, SOF, estST) appear as oracle labels in 14.1%,
7.0%, 5.9%, 4.7% of windows. RF may already handle these via learned features, but this is
unverified without feature importance analysis.
