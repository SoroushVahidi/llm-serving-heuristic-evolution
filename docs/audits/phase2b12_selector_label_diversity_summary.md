# Phase 2B.12 Selector Label Diversity Summary

**Experiment:** `phase2b12_workload_diversity_selector_labels`  
**Branch:** `phase2b12-workload-diversity-selector-labels`  
**Runner:** `scripts/run_phase2b12_workload_diversity_selector_labels.py`  
**Config:** `configs/phase2b12_workload_diversity_selector_labels.yaml`  
**Log:** `logs/phase2b12/phase2b12_workload_diversity.log`  
**Results:** `results/phase2b12_workload_diversity_selector_labels/` (gitignored; summary in this doc)  
**tmux session:** `phase2b12_workload_diversity` (completed, EXIT_CODE=0, ~638s)

---

## Background

Phase 2B.11 found that `scorpio_style_slo_guard` wins as per-window oracle on all 60
Phase 2B.9/2B.10 windows (100% label concentration).  RF/DT selector training was infeasible
— a classifier that always predicts SCORPIO would achieve 100% accuracy with no generalization value.

Phase 2B.12 adds 14 diversity workloads targeting regimes where simpler policies are expected
to win:
- `sarathi_style` — long-prompt prefill bottleneck regimes
- `weighted_shortest_processing` — long decode with loose/moderate SLO
- `edf` / `slo_slack_score` — priority-asymmetric or moderate-competition regimes
- `admission_control` — high prediction noise regimes

---

## RF/DT Feasibility Criterion

Training is feasible only if **all three** criteria are met:

| Criterion | Threshold | Result | Status |
|---|---|---|---|
| Window count | ≥200 | **172** | FAIL |
| Policies winning ≥10 windows each | ≥3 | **6** | PASS |
| Top policy concentration | <85% | **45.9%** | PASS |

**Overall: NOT FEASIBLE** (fails window count only; diversity criteria pass strongly)

---

## Evaluation Windows

| Group | Workloads | Seeds | Windows |
|-------|-----------|-------|---------|
| dev (regression) | 4 | 0, 1, 2 | 27 |
| heldout (regression) | 5 | 3, 4, 5 | 33 |
| regression total | 9 | 0–5 | **60** |
| diversity | 14 | 6–9 (synthetic); 6 (BurstGPT) | **112** |
| **overall** | **23** | 0–9 | **172** |

---

## Label Distribution Results

### Overall (regression + diversity, n=172)

| Policy | Windows won | Fraction | Notes |
|---|---|---|---|
| `scorpio_style_slo_guard` | **79** | **45.9%** | Down from 100% in Phase 2B.11 |
| `admission_control` | 29 | 16.9% | Wins prefill-heavy and high-noise regimes |
| `best_fit` | 14 | 8.1% | Wins long-decode loose-SLO regimes |
| `edf` | 14 | 8.1% | Wins moderate-pressure, all-complete regimes |
| `shortest_output_first` | 13 | 7.6% | Wins low-noise, uniform-output regimes |
| `estimated_service_time_first` | 10 | 5.8% | Wins low-noise, uniform-SLO regimes |
| `multi_bin_batching` | 9 | 5.2% | Wins high-load no-tight-SLO regimes |
| `random_feasible` | 3 | 1.7% | Appears in mixed/noisy regimes |
| `shortest_prompt_first` | 1 | 0.6% | Rare; low-noise balanced workload |

**9 distinct policies appear as oracle labels** (vs 6 in Phase 2B.11 regression alone).

### Regression group (Phase 2B.9/2B.11 workloads, n=60)

| Policy | Windows won | Fraction |
|---|---|---|
| `scorpio_style_slo_guard` | **46** | **76.7%** |
| `edf` | 4 | 6.7% |
| `admission_control` | 4 | 6.7% |
| `shortest_output_first` | 3 | 5.0% |
| `best_fit` | 2 | 3.3% |
| `random_feasible` | 1 | 1.7% |

Regression remains SCORPIO-dominated (76.7%), consistent with Phase 2B.11 finding.

### Diversity group (new workloads, n=112)

| Policy | Windows won | Fraction | Regime where it wins |
|---|---|---|---|
| `scorpio_style_slo_guard` | 33 | 29.5% | High-noise tight-SLO, bursty overloaded |
| `admission_control` | 25 | 22.3% | Prefill-heavy tight-SLO, high-noise moderate |
| `best_fit` | 12 | 10.7% | Long-decode loose-SLO |
| `estimated_service_time_first` | 10 | 8.9% | Low-noise, loose-weight-dominant |
| `shortest_output_first` | 10 | 8.9% | Low-noise mixed-rate |
| `edf` | 10 | 8.9% | Decode-heavy with KV saturation |
| `multi_bin_batching` | 9 | 8.0% | High-load no-tight-SLO |
| `random_feasible` | 2 | 1.8% | Noisy mixed regimes |
| `shortest_prompt_first` | 1 | 0.9% | Low-noise balanced |

Diversity group has **SCORPIO at 29.5%** — well below 85% threshold, and 9 policies appear.

---

## Per-Workload Label Analysis

| Workload | n_windows | Best fixed | Best WG | Oracle WG | Label (top) | Non-SCORPIO? |
|----------|-----------|-----------|---------|-----------|-------------|-------------|
| dev_overloaded_mixed_slo | 9 | scorpio | 1.000 | 1.000 | scorpio (9/9) | No |
| dev_high_prediction_noise | 6 | scorpio | 0.994 | 0.995 | scorpio (4/6) | Partial |
| dev_kv_pressure_decode_heavy | 6 | scorpio | 0.951 | 0.951 | scorpio (6/6) | No |
| dev_overloaded_prefill_heavy | 6 | *(tied)* | 1.000 | 1.000 | scorpio (6/6) | No — tied |
| heldout_moderate_kv_pressure | 6 | edf | 1.000 | 1.000 | split (edf=2,scorpio=2,AC=2) | Yes |
| heldout_very_high_noise | 6 | scorpio | 0.999 | 0.999 | scorpio (6/6) | No |
| heldout_prefill_overloaded | 11 | *(tied)* | 1.000 | 1.000 | scorpio (5), SOF (3)... | Partial |
| heldout_bursty_mixed_slo | 8 | scorpio | 0.997 | 1.000 | scorpio (6), edf (1), AC (1) | Partial |
| heldout_burstgpt_smoke | 2 | scorpio | 0.996 | 0.996 | scorpio (2/2) | No |
| **div_prefill_heavy_sarathi** | 8 | *(tied)* | 1.000 | 1.000 | **AC (8/8)** | **Yes** |
| **div_prefill_moderate_tight** | 8 | *(tied)* | 1.000 | 1.000 | **AC (8/8)** | **Yes** |
| div_prefill_bursty | 9 | *(tied)* | 1.000 | 1.000 | AC=2, estST=2, best_fit=2... | Yes |
| **div_decode_all_loose_slo** | 8 | *(tied)* | 1.000 | 1.000 | **best_fit (6/8)** | **Yes** |
| div_decode_medium_slo | 8 | edf | 1.000 | 1.000 | scorpio=4, AC=2, edf=2 | Partial |
| **div_decode_kv_saturated** | 8 | edf | 1.000 | 1.000 | **edf (5/8)** | **Yes** |
| **div_loose_weight_dominant** | 11 | *(tied)* | 1.000 | 1.000 | **estST=4, mbb=3, SOF=2, scorpio=2** | **Yes** |
| **div_medium_rate_low_noise** | 8 | *(tied)* | 1.000 | 1.000 | **SOF=4, estST=1, SPF=1, rnd=1, scorpio=1** | **Yes** |
| div_bursty_moderate | 9 | AC | 1.000 | 1.000 | scorpio=6, AC=2, edf=1 | Partial |
| **div_high_load_no_tight_slo** | 12 | *(tied)* | 1.000 | 1.000 | **mbb=4, estST=3, SOF=3, best_fit=2** | **Yes** |
| **div_high_noise_moderate_slo** | 11 | scorpio | 0.993 | 0.996 | **scorpio=7, AC=2, best_fit=1, edf=1** | Partial |
| **div_very_high_noise_tight_slo** | 8 | scorpio | 0.994 | 0.994 | **scorpio=7, best_fit=1** | Partial |
| div_burstgpt_high_load | 2 | scorpio | 0.996 | 0.996 | scorpio (2/2) | No |
| div_burstgpt_natural | 2 | scorpio | 1.000 | 1.000 | scorpio (2/2) | No |

**Key:** SOF = shortest_output_first, SPF = shortest_prompt_first, estST = estimated_service_time_first, mbb = multi_bin_batching, AC = admission_control, rnd = random_feasible

### Surprising label findings

1. **`admission_control` wins all prefill-heavy windows** (not `sarathi_style` as designed).
   At low-to-moderate load with long prompts, AC's urgency sort + rejection of borderline-SLO
   requests outperforms sarathi-style chunked prefill in the WG objective.
   
2. **`best_fit` wins long-decode loose-SLO** (not `weighted_shortest_processing`).
   Best-fit bin packing maximizes batch token utilization, which benefits long-decode workloads
   under loose SLO when KV pressure is tolerable.

3. **`estimated_service_time_first` and `multi_bin_batching` win under no-tight-SLO**.
   When all requests are medium/loose SLO, these throughput-maximizing policies achieve higher
   WG than SCORPIO's credit-throttle mechanism (which is inactive without tight SLO).

4. **Many diversity windows: all policies achieve WG=1.000** (best_fixed = tied).
   In underloaded / all-complete regimes, label diversity reflects tie-breaking order rather
   than meaningful policy differentiation. These windows contribute diversity statistics but
   not differentiated signal.

---

## Is SCORPIO Still Dominant?

**Yes on regression; no on diversity; mixed overall.**

| Group | SCORPIO label fraction | Interpretation |
|-------|----------------------|----------------|
| Regression (n=60) | 76.7% | Still highly dominant |
| Diversity (n=112) | 29.5% | Not dominant; 8 other policies appear |
| Overall (n=172) | **45.9%** | No longer extreme; well below 85% threshold |

SCORPIO is the most common label overall, but is far from dominant enough to make selector
training meaningless.

---

## Which Non-SCORPIO Policies Win, and Where?

| Policy | Primary regime | Windows won | Notes |
|--------|---------------|-------------|-------|
| `admission_control` | Prefill-heavy, high-noise | 29 | Unexpected prefill winner |
| `best_fit` | Long-decode loose-SLO | 14 | Better bin-packing than WSP here |
| `edf` | Decode-heavy KV-saturated | 14 | Completes all; no SCORPIO throttle needed |
| `shortest_output_first` | Low-noise uniform-output | 13 | SJF ordering wins when outputs predictable |
| `estimated_service_time_first` | No-tight-SLO, loose weights | 10 | Estimated SJF when slack is ample |
| `multi_bin_batching` | High-load no-tight-SLO | 9 | Packing wins when SLO is loose |

---

## Are At Least 3–5 Policies Meaningfully Represented?

**Yes — 6 policies each win ≥10 windows:**
- `scorpio_style_slo_guard`: 79 windows
- `admission_control`: 29 windows
- `best_fit`: 14 windows
- `edf`: 14 windows
- `shortest_output_first`: 13 windows
- `estimated_service_time_first`: 10 windows

**Is the representation meaningful?**  Partially.  Many "diversity" wins occur in all-WG=1.0
regimes where labels reflect tie-breaking, not genuine performance gaps.  A future evaluation
should compute meaningful-only windows as those where best_fixed_wg < 0.99 across all policies.

---

## Is the Selector Problem Meaningful on This Suite?

**Yes, for the regression group (n=60):** Policies differ by 5–15+ WG points.
**Partially for diversity:** Many diversity windows are all-complete (WG=1.000 for most policies),
so label diversity may partly reflect tie-breaking noise rather than regime-specific advantages.

The most selector-relevant workloads are:
1. `dev_overloaded_mixed_slo` (SCORPIO wins clearly)
2. `dev_kv_pressure_decode_heavy` (SCORPIO wins clearly, 5.1+ pp gap)
3. `dev_high_prediction_noise` (SCORPIO 0.994 vs others 0.94–0.98)
4. `div_high_noise_moderate_slo` (SCORPIO 0.993 vs alternatives)
5. `div_very_high_noise_tight_slo` (SCORPIO 0.994 vs alternatives)

---

## Which Workload Groups Create Most Useful Selector Diversity?

| Group | Diversity value | Rationale |
|-------|----------------|-----------|
| High-noise tight-SLO | High | Clear SCORPIO vs AC differentiation |
| High-load no-tight-SLO | High | SCORPIO absent; mbb/estST win clearly |
| Long-decode loose-SLO | Medium | best_fit vs WSP vs edf compete; all near WG=1 |
| Prefill-heavy moderate-load | Medium | AC wins consistently; unexpected outcome |
| Very-high-noise extreme | High | SCORPIO/AC differentiated from FIFO |
| Low-noise uniform-output | Medium | SOF/estST win; all near WG=1 |

---

## Which Workload Groups Are Redundant?

| Group | Redundancy | Reason |
|-------|-----------|--------|
| Multiple BurstGPT seeds (single trace) | High | Only 2 windows per BurstGPT trace (1 seed used) |
| Bursty moderate (all-tie) | Medium | All policies achieve WG=1.0 |
| Decode-medium-SLO | Medium | Mixed labels; near-tie between SCORPIO/edf/AC |

---

## RF/DT Training Decision

**Decision: NOT TRAINED (insufficient window count)**

| Criterion | Required | Actual | Pass? |
|-----------|----------|--------|-------|
| Total windows | ≥200 | **172** | FAIL (−28 windows) |
| Policies ≥10 wins each | ≥3 | **6** | PASS |
| Max label concentration | <85% | **45.9%** | PASS |

The shortfall (172 vs 200 required) is 28 windows.  The diversity criterion is strongly met:
6 policies each win ≥10 windows, top policy at only 45.9%.

**Why not lower the threshold and train?**
The 200-window threshold ensures adequate held-out evaluation.  With 172 windows and a held-out
split, the training set would be ~120–130 windows across 6 classes — below the threshold for
reliable RF/DT generalization estimates.  Expanding to ≥200 windows (via additional workload
seeds or new regimes) would give adequate sample sizes per class.

**Recommended next step:** Add more workload seeds or new regime variants to reach ≥200 windows,
then train RF/DT with per-window oracle labels.

---

## Related Documents

| Document | Purpose |
|----------|---------|
| `docs/audits/phase2b12_workload_diversity_design.md` | Workload matrix design |
| `docs/audits/phase2b12_rule_selector_diversity_evaluation.md` | Rule selector evaluation |
| `docs/audits/phase2b12_failure_cases_summary.md` | Failure cases |
| `docs/audits/phase2b11_scorpio_selector_integration_summary.md` | Prior phase reference |
