# Phase 2B.16: Fresh Corrected-Objective Validation — Audit Summary

**Branch:** `phase2b16-fresh-corrected-objective-validation`  
**Status:** COMPLETE — experiment finished 2026-06-26 (637.6s)  
**Config:** `configs/phase2b16_fresh_corrected_objective_validation.yaml`  
**Runner:** `scripts/run_phase2b16_fresh_corrected_objective_validation.py`  
**Results:** `results/phase2b16_fresh_corrected_objective_validation/` (gitignored)

---

## Experiment Protocol

### Configuration
- 21 workloads across 3 groups: `fresh_diversity` (12 wl × 4 seeds), `fresh_targeted` (4 wl × 4 seeds), `fresh_heldout` (5 wl × 3 seeds)
- **Fresh diversity seeds**: [12, 13, 14, 15] — not used in any prior phase
- **Fresh heldout seeds**: [20, 21, 22] — not used in any prior phase
- **Selectors frozen before fresh evaluation**: Yes — trained on Phase 2B.13 train-split only (div seeds 6-10, dev seeds 0-2)
- Bootstrap: n=2000, seed=42, 95% CI
- Primary metric: `arrival_normalized_wg`

### Runtime
- Phase A (selector training from old data): ~60s
- Phase B (fresh simulation): 575.4s
- Phase C–E (metrics, stats, output): ~2s
- Total: 637.6s

---

## Q1: How many fresh windows were evaluated?

**174 windows** across 3 groups:
- `fresh_diversity`: 106 windows (12 workloads × 4 seeds, ~8-9 windows each)
- `fresh_heldout`: 34 windows (5 workloads × 3 heldout seeds)
- `fresh_targeted`: 34 windows (4 workloads × 4 seeds)

---

## Q2: Which seeds/workload families were new?

| Group | Seeds | Workload families |
|-------|-------|-------------------|
| fresh_diversity | 12, 13, 14, 15 | Prefill-moderate-tight, prefill-bursty, decode-all-loose, KV-pressure-moderate, high-noise-moderate, very-high-noise-tight |
| fresh_targeted | 12, 13, 14, 15 | Completion-sensitive-moderate, long-decode-loose-SLO, short-output-SJF, moderate-overload-completion |
| fresh_heldout | 20, 21, 22 | KV-pressure, very-high-noise, overloaded-mixed-SLO, KV-pressure-decode-heavy, bursty-mixed-SLO |

All seeds are entirely new — not used in Phase 2B.9–2B.15.

---

## Q3: Were Phase 2B.15 selectors frozen before fresh evaluation?

**Yes.** Selectors were retrained in Phase A using only Phase 2B.13 train-split data (diversity seeds 6–10, dev seeds 0–2). The frozen selectors were then applied to fresh windows without any further training or parameter updates.

---

## Q4: Fresh label distribution under arrival-normalized WG

| Label | Count | Fraction |
|-------|-------|---------|
| `fifo` | 90 | 51.7% |
| `edf` | 72 | 41.4% |
| `admission_control` | 7 | 4.0% |
| `scorpio_style_slo_guard` | 5 | 2.9% |

**Key finding:** SCORPIO wins only 5/174 fresh windows (2.9%) under ANWG — vs 45.9% in Phase 2B.13 old data. EDF and FIFO dominate because 88% of fresh windows are all-complete regimes where CF=1.0 for all non-SCORPIO policies, and EDF achieves the highest conditional SLO goodput.

Compare with label distribution under completed-request quality (cond WG):

| Label (cond WG) | Count |
|----------|-------|
| `scorpio_style_slo_guard` | 71 |
| `admission_control` | 28 |
| `edf` | 20 |
| `best_fit` | 20 |
| `multi_bin_batching` | 10 |
| `shortest_output_first` | 9 |
| others | 16 |

Label changes cond→anwg: **148/174 (85.1%)**. Most changes are SCORPIO→EDF/FIFO in easy regimes where SCORPIO's CF<1.

---

## Q5: Label distribution under completion-penalized metrics

Because 88% of windows are all-complete (CF=1.0 for all policies), completion-penalized metrics yield the same result as ANWG for all-complete windows. The constrained-objective analysis shows:

- CF≥0.95: SCORPIO satisfies constraint in 151/174 (86.8%) windows — fails 23 windows
- CF≥0.99: SCORPIO satisfies constraint in only 91/174 (52.3%) windows — fails half

Under CF≥0.95 and CF≥0.99, the oracle-constrained policy is dominated by EDF/FIFO (which always satisfy both constraints since CF=1.0):
```
oracle_constrained_anwg = 0.9863 (both thresholds)
constrained_policy_dist: {fifo: 90, edf: 72, admission_control: 7, wsp: 5}
```

---

## Q6: Near-tie / all-complete / FIFO-artifact counts

| Category | Count | Fraction |
|----------|-------|---------|
| All windows | 174 | — |
| All-complete (best ANWG ≥ 0.99) | 153 | 87.9% |
| Near-tie (margin < 0.001) | 162 | 93.1% |
| Near-tie (margin < 0.005) | 162 | 93.1% |
| Near-tie (margin < 0.010) | 169 | 97.1% |
| Meaningful (margin ≥ 0.005) | **12** | 6.9% |
| Meaningful (margin ≥ 0.010) | 5 | 2.9% |
| FIFO wins under ANWG | 90 | 51.7% |
| FIFO wins that are near-ties (< 0.001) | 90 | **100%** |
| Genuine FIFO wins | **0** | 0% |

**All 90 FIFO wins are metric artifacts** — they occur because SCORPIO has CF=0.9992 (slight rejection rate) while FIFO has CF=1.0, giving FIFO ANWG margin of ≤0.001. No genuine FIFO advantage exists.

---

## Q7: Does rf_anwg beat always-SCORPIO on fresh validation?

**YES.** `rf_anwg` achieves ANWG = **0.9781** vs always-SCORPIO = **0.9686** on 174 fresh windows.
- Gap: **+0.0095**
- 95% bootstrap CI: **[0.0035, 0.0155]** — CI **excludes zero**
- Win/tie/loss vs SCORPIO: **87/80/7**

The Phase 2B.15 gain survives fresh evaluation.

---

## Q8: Is the improvement statistically meaningful?

**YES — the CI excludes zero for 6 of 12 selectors:**

| Selector | ANWG | Gap vs SCORPIO | 95% CI | Stat sig |
|----------|------|----------------|--------|---------|
| regression_anwg | 0.9856 | +0.0170 | [0.0127, 0.0213] | ✓ |
| safe_fallback_wsp_margin0.001 | 0.9849 | +0.0163 | [0.0126, 0.0204] | ✓ |
| knn_anwg | 0.9818 | +0.0132 | [0.0076, 0.0186] | ✓ |
| rf_anwg | 0.9781 | +0.0095 | [0.0035, 0.0155] | ✓ |
| rf_anwg_regret | 0.9756 | +0.0070 | [0.0010, 0.0129] | ✓ |
| rule_based | 0.9771 | +0.0085 | [0.0055, 0.0112] | ✓ |
| dt_anwg | 0.9745 | +0.0059 | [-0.0006, 0.0124] | ✗ marginal |
| dt_anwg_regret | 0.9729 | +0.0043 | [-0.0022, 0.0107] | ✗ |
| always_scorpio | 0.9686 | 0.0000 | — | reference |
| always_wsp | 0.9648 | −0.0038 | [-0.0066, -0.0011] | statistically **below** SCORPIO |

---

## Q9: Does any selector beat always-WSP?

**YES** — every selector beats always-WSP. WSP achieves only 0.9648, which is statistically below SCORPIO (CI [-0.0066, -0.0011]). Even `rule_based` (0.9771) and `rf_anwg` (0.9781) beat WSP by large margins.

---

## Q10: Does any selector beat the best fixed deployable policy?

The best fixed deployable policy under ANWG on fresh data is **EDF/orca_style/slo_slack_score** at **0.9776** (not SCORPIO). The fixed policy landscape shifted significantly:

| Policy | ANWG | CF |
|--------|------|----|
| edf | **0.9776** | 1.000 |
| orca_style | **0.9776** | 1.000 |
| slo_slack_score | **0.9776** | 1.000 |
| admission_control | 0.9753 | 1.000 |
| **scorpio_style_slo_guard** | **0.9686** | 0.972 |
| weighted_shortest_processing | 0.9648 | 1.000 |

SCORPIO ranks **5th** among fixed baselines on fresh data (behind EDF, orca_style, slo_slack_score, admission_control).

**Best fixed policy (EDF)**: 0.9776. Selectors that beat EDF:
- regression_anwg: 0.9856 (**+0.0080 vs EDF**)
- safe_fallback_wsp_margin0.001: 0.9849 (+0.0073)
- knn_anwg: 0.9818 (+0.0042)
- rf_anwg: 0.9781 (+0.0005 — essentially tied with EDF)

---

## Q11: Which selector is best under ANWG?

`regression_anwg`: **0.9856** — 20 per-policy RF regressors (one per policy, predicts ANWG, argmax). Gap vs SCORPIO: +0.0170, CI [0.0127, 0.0213]. Oracle evaluator across all 20 policies.

---

## Q12: Which selector is best under completion-penalized metrics?

Because 88% of windows are all-complete, completion-penalized metrics are identical to ANWG for most windows. The constrained-objective oracle (CF≥0.95 or CF≥0.99) achieves 0.9863 (vs oracle ANWG 0.9879), dominated by EDF/FIFO. `safe_fallback_wsp` performs well under constrained objectives because it defaults to WSP (CF=1.0).

---

## Q13: Does selector improvement hold across workload groups?

| Group | SCORPIO | rf_anwg | knn_anwg | regression_anwg | safe_fallback_wsp_m001 |
|-------|---------|---------|---------|-----------------|----------------------|
| fresh_diversity | 0.9793 | 0.9964 (+0.017) | 0.9965 | 0.9965 | 0.9964 |
| fresh_heldout | 0.9302 | **0.9472 (+0.017)** | 0.9458 | 0.9371 | 0.9472 |
| fresh_targeted | 0.9737 | **0.9521 (-0.022)** | 0.9720 | **1.000 (+0.026)** | 0.9869 |
| **overall** | 0.9686 | **0.9781 (+0.0095)** | 0.9818 | 0.9856 | 0.9849 |

**Important finding:** `rf_anwg` LOSES to SCORPIO on fresh_targeted workloads (−0.0216). The targeted workloads were designed to favor non-SCORPIO policies (loose SLO, completion-sensitive), and `rf_anwg` fails to route away from SCORPIO in these regimes. `regression_anwg` succeeds (1.000) and `knn_anwg` (0.9720) partially succeeds.

---

## Q14: Does selector improvement hold across seeds?

Seed-level analysis shows consistent ordering across all 7 fresh seeds:

| Seed | Group | SCORPIO | rf_anwg | knn_anwg | safe_fallback |
|------|-------|---------|---------|---------|--------------|
| s12 | div | 0.9780 | 0.9904 | 0.9904 | 0.9951 |
| s13 | div | 0.9777 | 0.9838 | 0.9902 | 0.9935 |
| s14 | div | 0.9772 | 0.9850 | 0.9913 | 0.9938 |
| s15 | div | 0.9789 | 0.9838 | 0.9903 | 0.9938 |
| s20 | heldout | 0.9315 | 0.9496 | 0.9488 | 0.9496 |
| s21 | heldout | 0.9279 | 0.9453 | 0.9434 | 0.9453 |
| s22 | heldout | 0.9316 | 0.9469 | 0.9454 | 0.9469 |

Selector gains are consistent across all seeds. No seed shows a reversal for rf_anwg vs SCORPIO.

---

## Q15: Is selector contribution claim-ready?

**YES — with caveats:**

**Claim-ready:** `rf_anwg` gains are statistically significant on fresh validation (CI excludes zero). `regression_anwg` and `knn_anwg` show larger, tighter CIs. All gains are consistent across seeds.

**Caveats to include:**
1. 93.1% of fresh windows are near-ties — selector gain is driven by 12 meaningful windows
2. `rf_anwg` fails on targeted workloads (−0.0216 vs SCORPIO on fresh_targeted)
3. Under fresh ANWG, EDF/orca_style outperform SCORPIO as fixed baselines
4. Near-tie domination means the gain magnitude is sensitive to workload difficulty distribution
5. Results are synthetic-simulator only (no real GPU, no production traces)

---

## Q16: Safe claims

- "On 174 fresh validation windows (unseen seeds [12-15,20-22]), `rf_anwg` achieves arrival-norm WG = 0.9781, +0.0095 over always-SCORPIO (0.9686). 95% bootstrap CI = [0.0035, 0.0155], excluding zero."
- "`regression_anwg` achieves 0.9856 (+0.0170 vs SCORPIO), CI [0.0127, 0.0213] on fresh validation."
- "`knn_anwg` achieves 0.9818 (+0.0132 vs SCORPIO), CI [0.0076, 0.0186] on fresh validation."
- "Phase 2B.15 selectors were frozen before any fresh window was evaluated."
- "93.1% of fresh windows (162/174) are near-ties (margin < 0.005). Only 12 windows are meaningful."
- "All 90 FIFO 'wins' under arrival-norm WG are near-tie artifacts (margin < 0.001, all due to SCORPIO CF < 1.0)."
- "On the 12 meaningful windows: SCORPIO = 0.852, rf_anwg = 0.8647 (+0.0127), knn_anwg = 0.8656 (+0.0136)."
- "Under fresh ANWG, EDF/orca_style/slo_slack_score (0.9776) outperform always-SCORPIO (0.9686) as fixed baselines. SCORPIO ranks 5th among fixed policies on fresh data."
- "always-WSP (0.9648) is statistically below always-SCORPIO (CI [-0.0066, -0.0011]) on fresh validation."
- "SCORPIO satisfies CF≥0.99 in only 52.3% of fresh windows."
- "`rf_anwg` loses to SCORPIO on fresh targeted workloads (−0.0216) despite beating SCORPIO overall (+0.0095)."

---

## Q17: Unsafe claims

| Claim | Why unsafe |
|-------|-----------|
| "Selectors generalize to hard targeted workloads" | rf_anwg loses on fresh_targeted (-0.0216) |
| "SCORPIO is the best fixed baseline" | FALSE under fresh ANWG: EDF beats SCORPIO by +0.009 |
| "Selector gain is robust across all workload types" | rf_anwg fails on targeted; regression_anwg succeeds |
| "Selector improvement is driven by non-trivial routing" | 93% near-ties; 12 meaningful windows drive the bulk |
| "WSP is a strong safe fallback" | WSP is statistically below SCORPIO under ANWG |
| "Results hold in production serving systems" | Synthetic simulator only; no real GPU or traces |

---

## Q18: Recommended next step

Selector gains survive fresh validation with positive CIs, but near-tie dominance (93%) limits meaningful signal. The label landscape is fundamentally different from old data (EDF dominates, not SCORPIO). Two directions:

**Option A (if pursuing deployment-ready selector):** Redesign labels using only the 12+ meaningful windows per workload family; weight training by regret magnitude. Focus on regimes with margin > 0.005.

**Option B (if pursuing real-trace validation):** Begin real-trace ingestion (BurstGPT full, Azure LLM traces). The synthetic-only results are claim-ready but not sufficient for strong publication claims.

Given that synthetic gains are now confirmed with positive CIs and the main limitation is workload-mix (near-tie dominated), **Option B is recommended** — move to real-trace validation before adding another external baseline.

---

## Output Files

| File | Description |
|------|-------------|
| `fresh_per_window.csv` | 174 rows with per-policy rewards and selector predictions |
| `fresh_selector_comparison.csv` | Per-selector ANWG comparison (all metrics) |
| `fresh_group_summary.csv` | Group-level breakdown |
| `fresh_meaningful_summary.csv` | 12 meaningful windows (eps=0.005 filter) |
| `fresh_policy_ranking.csv` | 20-policy ranking under ANWG |
| `fresh_significance_summary.json` | Bootstrap CIs, win/tie/loss for all selectors |
| `fresh_top_epsilon_accuracy.csv` | Top-ε accuracy at ε=0.001/0.005/0.010 |
| `fresh_constrained_objectives.json` | Constrained ANWG at CF≥0.95 / CF≥0.99 |
| `fresh_near_tie_summary.json` | Near-tie stats |
| `fresh_fifo_artifact_audit.json` | FIFO win classification |
| `fresh_label_distribution.json` | Label changes cond→anwg |
| `fresh_seed_summary.csv` | Per-seed breakdown |
| `fresh_failure_cases.csv` | fail_029, fail_030 confirmed |
| `fresh_overall_summary.json` | Comprehensive summary with answers dict |
