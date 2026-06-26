# Phase 2B.16: Fresh Corrected-Objective Validation — Audit Summary

**Branch:** `phase2b16-fresh-corrected-objective-validation`  
**Status:** Experiment queued (run via tmux `phase2b16_fresh_validation`)  
**Config:** `configs/phase2b16_fresh_corrected_objective_validation.yaml`  
**Runner:** `scripts/run_phase2b16_fresh_corrected_objective_validation.py`  
**Results:** `results/phase2b16_fresh_corrected_objective_validation/` (gitignored)

---

## Goal

Validate whether the Phase 2B.15 selector gains (+0.0157 vs always-SCORPIO for `rf_anwg` on 33
heldout windows) hold on entirely fresh, unseen simulation windows using:
- New seeds: diversity=[12,13,14,15], heldout=[20,21,22] (not used in any prior phase)
- New workloads: 21 workloads across `fresh_diversity`, `fresh_targeted`, `fresh_heldout` groups
- Frozen selectors: retrained from Phase 2B.13 train-split only (dev s0-2, div s6-10)

---

## Experiment Protocol

### Phase A: Selector training (from old Phase 2B.13 data only)
- Load `results/phase2b13_selector_training_and_suspicion_audit/per_window.csv`
- Apply Phase 2B.13 train split (dev seeds 0-2, diversity seeds 6-10)
- Retrain Phase 2B.15 selectors under `arrival_normalized_wg`:
  `rf_anwg`, `rf_anwg_regret`, `dt_anwg`, `dt_anwg_regret`, `knn_anwg`,
  `regression_anwg`, `safe_fallback_wsp_{0.001,0.005,0.010}`
- **Selectors are frozen before any fresh window is seen**

### Phase B: Fresh simulation (~50-90 min)
- 21 workloads × 4 diversity seeds + 5 heldout workloads × 3 heldout seeds
- All 20 deployable policies evaluated per window
- Results saved to `fresh_per_window.csv`

### Phase C: Corrected metric evaluation
- Relabel under `arrival_normalized_wg`
- Compute 5 metric variants per window
- FIFO-artifact audit

### Phase D: Statistical analysis
- Bootstrap 95% CI (n=2000, seed=42) for key gaps
- Win/tie/loss counts
- Top-epsilon accuracy at ε=0.001, 0.005, 0.010
- Constrained objective (CF ≥ 0.95 / 0.99)
- Group-level and seed-level analysis

---

## Validation Questions

These questions will be answered after the experiment runs. Placeholders:

| # | Question | Status |
|---|----------|--------|
| 1 | Does `rf_anwg` beat always-SCORPIO on fresh windows? | TBD |
| 2 | Is the CI for `rf_anwg` vs SCORPIO above zero? | TBD |
| 3 | Does any selector beat always-WSP under arrival-norm WG? | TBD |
| 4 | Does WSP still beat SCORPIO under arrival-norm WG on fresh data? | TBD |
| 5 | What fraction of label changes (cond→anwg) are near-tie FIFO artifacts? | TBD |
| 6 | What fraction of FIFO wins have margin < 0.005? | TBD |
| 7 | Does `safe_fallback_wsp` outperform `rf_anwg` as an oracle? | TBD |
| 8 | Under constrained obj (CF≥0.95), does oracle-constrained > always-WSP? | TBD |
| 9 | Does `knn_anwg` generalize better than `rf_anwg` on fresh targeted workloads? | TBD |
| 10 | Is the SCORPIO label fraction <80% on fresh targeted workloads? | TBD |
| 11 | Do seed-level results show consistent selector ordering? | TBD |
| 12 | Does `dt_anwg` beat always-SCORPIO on fresh data? | TBD |
| 13 | What is the fresh oracle gap vs always-SCORPIO? | TBD |
| 14 | Is the label distribution shift (cond→anwg) similar to Phase 2B.15? | TBD |
| 15 | Does `regression_anwg` overfit vs `rf_anwg` on fresh windows? | TBD |
| 16 | Do fresh targeted workloads expose new failure patterns? | TBD |
| 17 | Is `safe_fallback_wsp_margin0.001` better than `margin0.005`? | TBD |
| 18 | Does the meaningful-windows subset (eps=0.005) tell a different story? | TBD |
| 19 | Fresh n_windows vs expected (~84): is simulation complete? | TBD |
| 20 | Are all 20 deployable policies represented in fresh_policy_ranking.csv? | TBD |
| 21 | Does constrained obj (CF≥0.99) favor WSP over rf_anwg on fresh data? | TBD |
| 22 | Does the gain generalize: is B16 rf_anwg gap larger/smaller than B15? | TBD |

---

## Selector Hierarchy (Phase 2B.15 test split baseline)

| Selector | ANWG | Gap vs SCORPIO | CI (vs SCORPIO) |
|----------|------|----------------|-----------------|
| safe_fallback_wsp_margin0.001 | 0.9848 | +0.0210 | oracle — not deployable standalone |
| rf_anwg | 0.9795 | +0.0157 | TBD (fresh) |
| always_scorpio | 0.9638 | 0.0000 | reference |
| rule_based | TBD | TBD | TBD |
| always_wsp | 0.9463 | −0.0175 | below scorpio |
| b13_random_forest | 0.9638 | 0.0000 | collapsed to scorpio |

---

## Key Hypotheses to Test

1. **rf_anwg generalizes**: Phase 2B.15 +0.0157 gain is real, not a 33-window fluke
2. **WSP fallback safety**: `safe_fallback_wsp` systematically improves over `rf_anwg` only when it has oracle access to rewards
3. **FIFO artifacts persist**: Under arrival-norm WG, FIFO "wins" on fresh windows are also near-tie artifacts
4. **Constrained objective**: WSP dominates under CF≥0.99 on fresh data as it does on B15 data

---

## Output Files

All outputs in `results/phase2b16_fresh_corrected_objective_validation/`:

| File | Description |
|------|-------------|
| `fresh_per_window.csv` | Raw simulation + selector predictions per window |
| `fresh_selector_comparison.csv` | Per-selector ANWG comparison (all metrics) |
| `fresh_group_summary.csv` | Group-level breakdown (diversity, targeted, heldout) |
| `fresh_meaningful_summary.csv` | Meaningful-windows only (eps=0.005 filter) |
| `fresh_policy_ranking.csv` | Fixed-policy ranking under ANWG |
| `fresh_significance_summary.json` | Bootstrap CIs, win/tie/loss |
| `fresh_top_epsilon_accuracy.csv` | Top-ε accuracy at ε=0.001/0.005/0.010 |
| `fresh_constrained_objectives.json` | Constrained ANWG at CF≥0.95 / CF≥0.99 |
| `fresh_near_tie_summary.json` | Near-tie stats at ε=0.001/0.005/0.010 |
| `fresh_fifo_artifact_audit.json` | FIFO win classification |
| `fresh_label_distribution.json` | Label changes cond→anwg |
| `fresh_seed_summary.csv` | Per-seed breakdown |
| `fresh_failure_cases.csv` | Detected failure patterns |
| `fresh_overall_summary.json` | Comprehensive summary with answers dict |

---

## Notes

- Selector training uses ONLY Phase 2B.13 old data (seeds 0-11) — no leakage to fresh windows
- `SafeFallbackWspSelector` uses oracle rewards at predict time — valid for upper-bound analysis only
- Simulation expected runtime: ~50-90 min (21 workloads × 4-7 seeds × 20 policies)
- Run via: `tmux new-session -d -s phase2b16_fresh_validation` then
  `python scripts/run_phase2b16_fresh_corrected_objective_validation.py --config configs/phase2b16_fresh_corrected_objective_validation.yaml 2>&1 | tee logs/phase2b16/phase2b16_fresh_validation.log`
