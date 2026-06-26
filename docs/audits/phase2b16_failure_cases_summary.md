# Phase 2B.16: Failure Cases Summary

**Branch:** `phase2b16-fresh-corrected-objective-validation`  
**Status:** Pre-experiment placeholders; will be updated after experiment runs

---

## Inherited failure patterns (from Phase 2B.14 and 2B.15)

| ID | Pattern | Phase | Status |
|----|---------|-------|--------|
| fail_021 | B13 selector collapse on heldout (all windows predict SCORPIO) | 2B.15 | Resolved: B15 corrected-objective training |
| fail_022 | WSP beats SCORPIO as safe fallback default | 2B.15 | Resolved: B15 uses WSP as default |
| fail_023 | Near-tie FIFO labels create noise in training data | 2B.15 | Resolved: near-tie filter at ε=0.005 |
| fail_024 | dt_anwg underperforms SCORPIO on test split | 2B.15 | Known: DT overfits without regret weighting |
| fail_025 | scorpio_deadline_only marginally fails promotion threshold | 2B.15 | Known: CQ gap −1.2pp vs threshold −1.0pp |
| fail_026 | Test split dominated by easy windows (82% all-complete) | 2B.15 | Known: Need fresh targeted workloads |

---

## New Phase 2B.16 failure cases

These patterns are anticipated based on Phase 2B.15 findings. Status will be updated
after the experiment runs.

| ID | Pattern | Anticipated Risk | Resolution Strategy |
|----|---------|-----------------|---------------------|
| fail_027 | rf_anwg does not beat always-SCORPIO on fresh validation | Medium | Verify CI; check if targeted workloads expose SCORPIO dominance |
| fail_028 | rf_anwg vs SCORPIO CI includes zero — not statistically reliable | High | Expected: 33-window gain may not replicate with wide CI |
| fail_029 | All FIFO label wins under arrival-norm WG are near-tie artifacts | Low (expected) | Confirm near-tie classification; not a failure if expected |
| fail_030 | Near-tie labels dominate fresh validation (>60% at eps=0.005) | Medium | Means most windows are too easy to discriminate selectors |
| fail_031 | No selector beats always-WSP under arrival-norm WG on fresh validation | Low | WSP should still be below SCORPIO under ANWG |
| fail_032 | Fresh targeted workloads still dominated by SCORPIO | Medium | Targeted workloads designed to reduce SCORPIO dominance |
| fail_033 | knn_anwg collapses on fresh heldout (new regime coverage gap) | Low | KNN depends on training set coverage |
| fail_034 | safe_fallback_wsp shows no gain over rf_anwg (oracle threshold never fires) | Medium | Occurs if rf_anwg always predicts policies worse than WSP |
| fail_035 | Selector ordering inconsistent across fresh seeds | Medium | Seed variance is the key question of Phase 2B.16 |
| fail_036 | dt_anwg worse than random forest on fresh data | Low (expected) | DT known to be weaker; use as ablation |
| fail_037 | Constrained objective (CF≥0.99) dominated by WSP alone | Low | Expected behavior; WSP dominates high-completion constraint |
| fail_038 | Phase B simulation fails for some workloads | Low | Config tested analytically; simulator robustness assumed |

---

## Resolution policy

1. **If fail_027 confirmed** (rf_anwg doesn't beat SCORPIO on fresh data):
   - Check whether Phase 2B.15 gain was an artifact of the small 33-window test split
   - Examine group-level results: does rf_anwg win on targeted workloads but lose overall?
   - Consider: the gain may be real but require more evaluation windows for statistical reliability

2. **If fail_028 confirmed** (CI includes zero):
   - Phase 2B.15 claims must be hedged: "possible gain, not statistically confirmed"
   - Do NOT promote rf_anwg as production selector without further validation
   - Document the limitation clearly

3. **If fail_030 confirmed** (near-tie dominates):
   - Most windows are "all-complete" regimes where ANWG ≈ cond WG for all policies
   - This is an inherent property of the workload mix, not a bug
   - Focus evaluation on meaningful-windows subset

4. **If fail_032 confirmed** (targeted workloads still SCORPIO-dominated):
   - Fresh targeted workloads may need redesign (even lower load, even longer outputs)
   - Consider: SCORPIO dominance may be fundamental under this metric formulation

---

## Notes on FIFO near-tie artifacts

Under `arrival_normalized_wg`, FIFO (and other all-complete policies) appear to "win" when:
1. All policies achieve `conditional_WG = 1.0` (no SLO violations for any policy)
2. SCORPIO has `completion_fraction < 1.0` due to its admission controller
3. FIFO has `completion_fraction = 1.0` (never rejects)

Therefore: `anwg(FIFO) = 1.0 * 1.0 = 1.0 > anwg(SCORPIO) = 0.999... * 1.0 ≈ 0.999`

This is a near-tie: the numerical "win" for FIFO is ≤0.001 in easy regimes. These windows
should be excluded from meaningful analysis via the near-tie filter (ε=0.005). This behavior
was first documented in Phase 2B.15 (214/319 label changes, all near-tie) and is expected
to appear in fresh data at similar rates.
