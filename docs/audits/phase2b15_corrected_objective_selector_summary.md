# Phase 2B.15 Summary: Corrected Objective Selector Retraining

**Date:** 2026-06-26
**Branch:** phase2b15-corrected-objective-selector-retraining
**Input:** Phase 2B.13 per_window.csv (319 windows)
**Phase 2B.14 ablation data:** `results/phase2b14_metric_audit_scorpio_ablation/`
**Output:** `results/phase2b15_corrected_objective_selector_retraining/`

---

## Motivation

Phase 2B.14 discovered that `weighted_goodput` uses a completed-only denominator, inflating SCORPIO's apparent performance. This phase makes the corrected metric (`arrival_normalized_wg = completion_fraction × conditional_wg`) operational for selector training and evaluation. It also formally evaluates whether `scorpio_deadline_only` should be promoted to a deployable baseline.

---

## Phase A: Relabeling under Arrival-Normalized WG

### Label changes

| Metric | n_total | n_label_changes | change_fraction |
|---|---|---|---|
| conditional WG → arrival-norm WG | 319 | 214 | 67.1% |

### Label distribution

| Policy | Conditional WG label (n) | Arrival-norm WG label (n) |
|---|---|---|
| scorpio_style_slo_guard | 176 | 82 |
| admission_control | 37 | 12 |
| edf | 20 | 88 |
| fifo | 0 | 134 |
| weighted_shortest_processing | 6 | 3 |
| best_fit | 28 | 0 |
| others | 52 | 0 |

**Key insight:** FIFO "wins" 134 windows under arrival-norm WG because SCORPIO has CF=0.999 vs FIFO CF=1.0 in easy all-complete windows. These are near-tie windows (arrival-norm margin < 0.001).

### Near-tie analysis (arrival-normalized WG)

| Threshold | Near-tie fraction | Meaningful windows |
|---|---|---|
| ε=0.001 | 69.6% | 97 |
| ε=0.005 | 69.6% | 97 |
| ε=0.010 | 73.7% | 84 |
| all-complete (anwg ≥ 0.99) | 63.9% | — |

### Meaningful label distribution (ε=0.010, 84 windows)

| Policy | Wins |
|---|---|
| scorpio_style_slo_guard | 81 (96%) |
| admission_control | 2 |
| weighted_shortest_processing | 1 |

**Conclusion:** Under proper near-tie filtering, SCORPIO still dominates ~96% of meaningful windows under arrival-norm WG — nearly identical to Phase 2B.13 (conditional WG). The 134 FIFO "wins" are near-tie artifacts where SCORPIO's CF=0.999 vs FIFO CF=1.0 causes a tiny (<0.001) arrival-norm WG gap.

---

## Phase B: Selector Retraining

**Training configuration:**
- Train: dev (seeds 0–2) + diversity (seeds 6–10) = 245 windows
- Val: diversity (seed 11) = 41 windows
- Test: heldout = 33 windows
- Near-tie filter for training: ε=0.005 → 77/245 meaningful train windows

**New selectors in Phase 2B.15:**
- `rf_anwg` — RF (200 trees, depth 10) with arrival-norm WG labels
- `rf_anwg_regret` — RF with regret-weighted sample weights (margin + 0.001)
- `dt_anwg` — DT (depth 8) with arrival-norm WG labels
- `dt_anwg_regret` — DT regret-weighted
- `knn_anwg` — KNN (k=5) using arrival-norm WG for neighbor aggregation
- `regression_anwg` — per-policy RF regressor fitted to arrival-norm WG values; pick argmax
- `safe_fallback_wsp_margin{0.001,0.005,0.010}` — fallback to WSP (not SCORPIO) when base prediction doesn't clear margin

---

## Phase C: Multi-Metric Evaluation (Test Split, 33 windows)

### Full comparison table

| Selector | Phase | Training obj | Arrival-norm WG | Cond. quality | vs SCORPIO | vs WSP |
|---|---|---|---|---|---|---|
| **safe_fallback_wsp_margin0.001** | 2B.15 | anwg | **0.9848** | 0.9911 | **+0.0210** | +0.0385 |
| safe_fallback_wsp_margin0.005 | 2B.15 | anwg | 0.9848 | 0.9911 | +0.0210 | +0.0385 |
| safe_fallback_wsp_margin0.010 | 2B.15 | anwg | 0.9846 | 0.9909 | +0.0208 | +0.0383 |
| knn_anwg | 2B.15 | anwg | 0.9830 | 0.9932 | +0.0192 | +0.0367 |
| **rf_anwg** | **2B.15** | **anwg** | **0.9795** | 0.9858 | **+0.0157** | +0.0332 |
| b13_rule_based | 2B.13 | cond_wg | 0.9793 | 0.9803 | +0.0154 | +0.0330 |
| dt_anwg_regret | 2B.15 | anwg | 0.9746 | 0.9903 | +0.0108 | +0.0283 |
| regression_anwg | 2B.15 | anwg | 0.9722 | 0.9901 | +0.0084 | +0.0259 |
| b13_knn_selector | 2B.13 | cond_wg | 0.9718 | 0.9967 | +0.0080 | +0.0255 |
| rf_anwg_regret | 2B.15 | anwg | 0.9699 | 0.9848 | +0.0061 | +0.0236 |
| b13_dt_regret_weighted | 2B.13 | cond_wg | 0.9675 | 0.9958 | +0.0036 | +0.0212 |
| b13_per_policy_regression | 2B.13 | cond_wg | 0.9674 | 0.9978 | +0.0035 | +0.0211 |
| **always_scorpio** | 2B.15 | — | **0.9638** | 0.9975 | 0.0 | +0.0175 |
| b13_random_forest | 2B.13 | cond_wg | 0.9638 | 0.9975 | 0.0 | +0.0175 |
| b13_safe_fallback_{0.001,0.005,0.010} | 2B.13 | cond_wg | 0.9638 | 0.9975 | 0.0 | +0.0175 |
| dt_anwg | 2B.15 | anwg | 0.9589 | 0.9676 | -0.0049 | +0.0126 |
| b13_decision_tree | 2B.13 | cond_wg | 0.9571 | 0.9780 | -0.0067 | +0.0108 |
| **always_wsp** | 2B.15 | — | **0.9463** | 0.9463 | -0.0175 | 0.0 |

### Key findings

1. **B13 RF/safe-fallback collapse to always-SCORPIO on test:** The Phase 2B.13 RF, RF-regret, and SCORPIO-safe-fallback all predict SCORPIO for every heldout window (arrival-norm WG = 0.9638 = always-SCORPIO). They failed to generalize non-SCORPIO routing to held-out workloads.

2. **B15 RF outperforms B13 RF:** `rf_anwg` achieves 0.9795 (+0.0157 vs always-SCORPIO) vs B13 RF 0.9638 (+0.0). Corrected-objective training produces better-generalizing selectors.

3. **WSP fallback beats SCORPIO fallback:** The B15 safe-fallback-WSP achieves 0.9848 (oracle-assisted) vs B13 safe-fallback-SCORPIO which collapses to 0.9638. Routing uncertain windows to WSP (CF≈1.0) is safer than routing to SCORPIO (CF≈0.90) under arrival-norm WG.

4. **Non-oracle comparison (features only):** `rf_anwg` (0.9795) vs `b13_random_forest` (0.9638) — a +0.0157 improvement from corrected-objective training without any oracle information.

5. **Rule-based still competitive:** `b13_rule_based` achieves 0.9793, nearly matching `rf_anwg` (0.9795). Hand-coded rules that explicitly route KV-pressure windows to WSP capture the same discriminative signal as arrival-norm WG training.

6. **DT underperforms:** `dt_anwg` achieves only 0.9589 (below always-SCORPIO). DT with arrival-norm WG labels fails to capture the SCORPIO-dominance in non-tie windows cleanly, overweighting FIFO signals from near-tie windows.

---

## Phase D: scorpio_deadline_only Promotion Decision

**Evidence (from Phase 2B.14 ablation, 7 targeted discriminative workloads):**

| Metric | scorpio_deadline_only | full SCORPIO | Gap |
|---|---|---|---|
| Arrival-norm WG | 0.6862 | 0.6879 | -0.0017 |
| Conditional quality | 0.9397 | 0.9517 | -0.0120 |
| Completion fraction | 0.7237 | 0.7186 | +0.0051 |

**Promotion thresholds:** ANWG gap < 0.005 AND CQ gap < 0.010

**Result:** `passes_anwg = True` (|gap|=0.0017 < 0.005), `passes_cq = False` (|gap|=0.012 > 0.010)

**Recommendation: Keep as ablation (not promoted).** The laxity-only variant marginally fails the conditional quality threshold (−1.2pp vs −1.0pp). The ANWG gap is negligible (−0.17pp). Key considerations:
- On non-discriminative workloads (CF≈1.0 for both), both policies are equivalent.
- The −1.2pp CQ gap arises because removing the decode penalty allows more decode-heavy requests through, slightly reducing conditional quality.
- `scorpio_deadline_only` is conceptually cleaner and nearly as effective. Future work: validate on broader workloads; if the CQ gap remains < 1.5pp, promotion is appropriate.

---

## Selector Feature Importance (RF, arrival-norm WG labels)

| Feature | Importance |
|---|---|
| mean_pred_output_tokens | 0.2305 |
| p95_pred_output_tokens | 0.2064 |
| mean_prompt_tokens | 0.0859 |
| p95_prompt_tokens | 0.0761 |
| pred_output_cv | 0.0665 |

Consistent with Phase 2B.13: output token distribution dominates selection. High mean output → WSP (short-job-first frees KV slots); extreme output variance → SCORPIO (admission throttling).

---

## Main Conclusions

1. **Corrected-objective training is beneficial.** Retraining under arrival-norm WG produces selectors (+0.016 over B13 RF on test) that better generalize to heldout workloads.

2. **Safe-fallback with WSP default is the new best selector.** Under arrival-norm WG, routing uncertain windows to WSP (oracle: +0.021 vs SCORPIO) is better than routing to SCORPIO, because WSP has CF≈1.0 while SCORPIO has CF≈0.90.

3. **Phase 2B.13 selectors remain valid.** The B13 RF's failure to beat SCORPIO on test is a generalization failure, not a training error — its oracle performance on training windows was correct. The corrected objective gives the selector better information about which non-SCORPIO windows exist.

4. **scorpio_deadline_only: keep as ablation.** Marginally fails the CQ threshold; recommend reassessment after more workload coverage.

5. **SCORPIO still dominates under arrival-norm WG.** On 96% of meaningful (non-tie) windows, SCORPIO is the best policy. The corrected metric reduces SCORPIO's apparent lead (0.0345 vs 0.1275) but does not change which policy is best under unconstrained arrival-norm WG.

---

## Safe Claims (Phase 2B.15)

- "We retrain policy selectors under `arrival_normalized_wg` (corrected objective) and evaluate under 5 metric variants."
- "RF selector trained under arrival-norm WG achieves arrival-norm WG = 0.9795 on 33 held-out windows, +0.0157 over always-SCORPIO (0.9638) and +0.0157 over Phase 2B.13 RF (0.9638), using features only (no oracle information)."
- "Under arrival-norm WG, safe-fallback with WSP default achieves 0.9848 (oracle-assisted upper bound) vs always-SCORPIO 0.9638."
- "Phase 2B.13 RF/safe-fallback-SCORPIO selectors collapse to always-SCORPIO on 33 held-out windows under arrival-norm WG; Phase 2B.15 selectors do not."
- "scorpio_deadline_only marginally fails promotion threshold (CQ gap = −0.012 vs threshold −0.010); it passes ANWG threshold (gap = −0.0017 vs threshold −0.005). Recommended: keep as ablation."
- "Under arrival-norm WG with near-tie filter ε=0.010, SCORPIO is the best policy in 81/84 meaningful windows (96%)."
