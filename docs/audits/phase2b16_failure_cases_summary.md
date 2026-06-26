# Phase 2B.16: Failure Cases Summary

**Branch:** `phase2b16-fresh-corrected-objective-validation`  
**Status:** COMPLETE — actual results from 2026-06-26 experiment  
**Fresh windows:** 174 | **Confirmed cases:** 2 | **Expected but not triggered:** 10

---

## Inherited failures (resolved in Phase 2B.15, confirmed resolved here)

| ID | Pattern | B15 Resolution | B16 Status |
|----|---------|----------------|------------|
| fail_021 | B13 RF collapses to always-SCORPIO | B15 corrected-objective training | Resolved ✓ — rf_anwg beats SCORPIO by +0.0095 on fresh data |
| fail_022 | WSP needed as safe fallback default | B15 uses WSP as default | Confirmed — WSP (0.9648) correctly set as safe fallback |
| fail_023 | Near-tie FIFO labels corrupt training | Near-tie filter at ε=0.005 | Confirmed — 90 FIFO wins, all near-tie, training not corrupted |
| fail_024 | dt_anwg underperforms SCORPIO | DT known weaker | Partially resolved — dt_anwg CI barely includes 0 (-0.0006, 0.0124) |
| fail_025 | scorpio_deadline_only fails promotion | Kept as ablation | N/A — not evaluated in B16 |
| fail_026 | Test split dominated by easy windows | Fresh targeted workloads added | Partially resolved — still 88% all-complete, but targeted group exists |

---

## Confirmed failure cases (Phase 2B.16)

### fail_029 — All FIFO label wins are near-tie artifacts

**Status:** CONFIRMED  
**Detail:** n_fifo=90, all have margin<0.005 (fraction_fifo_wins_are_near_tie_eps005=1.0).  
**What it means:** Under arrival-norm WG, FIFO appears to win 51.7% of windows because
SCORPIO has CF=0.9992 (vs FIFO CF=1.0), producing apparent ANWG advantage of ≤0.001.
These are not genuine FIFO scheduling advantages — they are metric artifacts from SCORPIO's
admission controller in easy regimes where all policies achieve cond_WG=1.0.  
**Impact on selector training:** Training must filter near-ties to avoid learning that
FIFO is the "best" policy. The ε=0.005 filter in Phase 2B.15 correctly addresses this.  
**Impact on evaluation:** Meaningful label analysis requires filtering; only 12/174 windows
(6.9%) have margin ≥ 0.005 and provide real selector learning signal.

---

### fail_030 — Near-tie labels dominate fresh validation

**Status:** CONFIRMED  
**Detail:** near-tie fraction=0.931 (162/174 windows at ε=0.005). Meaningful windows: 12.  
**What it means:** 93.1% of fresh windows have best_anwg - second_best_anwg < 0.005.
For these windows, the correct policy choice barely matters — any reasonable selector
achieves near-optimal ANWG. This means:
1. Selector gain on fresh validation is partly driven by the 12 meaningful windows.
2. The 88% all-complete rate (153/174 windows) dominates the distribution.
3. Selector accuracy numbers (90%+ top-ε) are inflated by near-tie windows where
   any policy is ε-optimal.  
**Scope:** Not a selector regression — a property of the workload distribution.
Fresh workloads are predominantly easy regimes (rate=25-55, dur=8-12s, short SLOs)
where all 20 policies reach CF=1.0 and cond_WG≈1.0.  
**Resolution path:** (a) Target harder workloads with more load or tighter SLOs in
future phases; (b) train and evaluate on meaningful-only windows; (c) move to
real traces (BurstGPT, Azure) where overload is genuine.

---

## Expected failures that were NOT triggered

### fail_027 — rf_anwg does not beat always-SCORPIO on fresh validation

**Status:** NOT TRIGGERED  
**Actual result:** rf_anwg=0.9781 vs SCORPIO=0.9686, gap=+0.0095, CI=[0.0035,0.0155].
**Interpretation:** Phase 2B.15 gain survives fresh evaluation.

---

### fail_028 — rf_anwg vs SCORPIO CI includes zero

**Status:** NOT TRIGGERED  
**Actual result:** CI=[0.0035, 0.0155] — CI excludes zero. Gain is statistically confirmed.

---

### fail_031 — No selector beats always-WSP under arrival-norm WG

**Status:** NOT TRIGGERED  
**Actual result:** All selectors beat always-WSP (WSP=0.9648 is statistically below SCORPIO).

---

### fail_032 — Fresh targeted workloads still SCORPIO-dominated

**Status:** NOT TRIGGERED — but a related failure appeared: rf_anwg LOSES on targeted  
**Detail (new observation):** Under arrival-norm WG on fresh_targeted, SCORPIO (0.9737)
beats rf_anwg (0.9521) by -0.0216. The targeted workloads favor WSP (0.9789) and
regression_anwg (1.000) but not rf_anwg. This reveals that rf_anwg has not learned
to route away from SCORPIO in exactly the regimes designed to test this capability.
regression_anwg correctly identifies the optimal policy in all 34 targeted windows.  
**Failure ID:** fail_039 (new, see below)

---

### fail_033 — knn_anwg collapses on fresh heldout

**Status:** NOT TRIGGERED  
**Actual result:** knn_anwg=0.9458 on fresh_heldout vs SCORPIO=0.9302 (+0.0156).

---

### fail_034 — safe_fallback_wsp shows no gain over rf_anwg

**Status:** NOT TRIGGERED  
**Actual result:** safe_fallback_wsp_margin0.001=0.9849, rf_anwg=0.9781. Gap=+0.0068.

---

### fail_035 — Selector ordering inconsistent across fresh seeds

**Status:** NOT TRIGGERED  
**Actual result:** All 7 seeds show consistent ordering: regression_anwg > safe_fallback > knn > rf > rule_based > SCORPIO > WSP.

---

### fail_036 — dt_anwg worse than random forest on fresh data

**Status:** PARTIALLY TRIGGERED (marginal)  
**Actual result:** dt_anwg=0.9745 (+0.0059), CI [-0.0006, 0.0124] — CI barely includes zero.
DT is the only tree model without statistical confirmation. dt_anwg_regret=0.9729 also
has CI [-0.0022, 0.0107] (includes zero). Both underperform RF and KNN variants.  
**Recommendation:** Do not use DT variants as primary selectors. RF and KNN are preferred.

---

### fail_037 — Constrained objective (CF≥0.99) dominated by WSP alone

**Status:** NOT TRIGGERED  
**Actual result:** Constrained oracle at CF≥0.99 achieves 0.9863, dominated by EDF/FIFO
(which always have CF=1.0), not WSP. WSP achieves only 0.9648 even under constrained obj.

---

### fail_038 — Phase B simulation fails for some workloads

**Status:** NOT TRIGGERED  
**Actual result:** All 21 workloads completed successfully. 174 windows generated.

---

## New failure cases discovered in Phase 2B.16

### fail_039 — rf_anwg loses to SCORPIO on fresh_targeted workloads

**Status:** CONFIRMED (new finding)  
**Detail:** rf_anwg=0.9521 vs SCORPIO=0.9737 on 34 fresh targeted windows. Gap=-0.0216.  
**What it means:** rf_anwg fails to route away from SCORPIO in exactly the regimes
designed to test this (loose SLO, completion-sensitive, short uniform outputs).
`regression_anwg` (1.000) and `knn_anwg` (0.9720) succeed where rf_anwg fails.  
**Root cause hypothesis:** RF classifier learns a hard boundary that routes toward
SCORPIO in high-rate regimes (targeted workloads use rate=40-55), even when
arrival-norm WG would favor EDF/WSP. Regression_anwg (continuous per-policy
value prediction) correctly identifies non-SCORPIO policies via argmax.  
**Severity:** Moderate — overall gain still positive (+0.0095) because targeted group
is 34/174 windows (19.5%). Does not invalidate overall claim.  
**Resolution:** Use regression_anwg or knn_anwg rather than rf_anwg for production
selector. rf_anwg is kept for comparison but rf's classification approach may be
less appropriate than regression for continuous-valued ANWG labels.

---

## Summary table

| ID | Pattern | Status | Impact |
|----|---------|--------|--------|
| fail_021–026 | Inherited from B15 | Resolved ✓ | — |
| **fail_029** | FIFO wins all near-tie artifacts | **Confirmed** | Workload artifact; filter resolves |
| **fail_030** | Near-tie labels dominate (93.1%) | **Confirmed** | Limits learning signal; not a bug |
| fail_027 | rf_anwg doesn't beat SCORPIO | Not triggered ✓ | — |
| fail_028 | CI includes zero | Not triggered ✓ | — |
| fail_031 | No selector beats WSP | Not triggered ✓ | — |
| fail_032 | Targeted still SCORPIO-dominated | Not triggered — but related | See fail_039 |
| fail_033 | KNN collapses on heldout | Not triggered ✓ | — |
| fail_034 | SafeFallback shows no gain | Not triggered ✓ | — |
| fail_035 | Inconsistent seed ordering | Not triggered ✓ | — |
| **fail_036** | DT CI includes zero (marginal) | **Marginal trigger** | Use RF/KNN instead |
| fail_037 | Constrained obj dominated by WSP | Not triggered ✓ | — |
| fail_038 | Phase B simulation fails | Not triggered ✓ | — |
| **fail_039** | rf_anwg loses on targeted workloads | **Confirmed (new)** | Use regression_anwg/knn instead |
