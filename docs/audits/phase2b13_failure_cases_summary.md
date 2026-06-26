# Phase 2B.13 Failure Cases Summary

**Experiment:** `phase2b13_selector_training_and_suspicion_audit`  
**Branch:** `phase2b13-selector-training-and-suspicion-audit`  
**Results:** `results/phase2b13_selector_training_and_suspicion_audit/` (gitignored)

---

## Inherited Open Cases from Phase 2B.12

| ID | Status | Notes |
|----|--------|-------|
| fail_007 | Partially deferred | Offline eval understates SCORPIO rule dispatch |
| fail_008 | Open | Rule selector missing best_fit/MBB/SOF/ESTF targets |
| fail_009 | Open | AC beats sarathi on prefill-heavy (design assumption wrong) |
| fail_010 | Open | 172 < 200 windows blocked RF/DT → **resolved in 2B.13** |
| fail_011 | Open | All-complete diversity windows create tie-breaking labels |

---

## Phase 2B.13 Failure Cases

### fail_012: Rule selector loses to always-SCORPIO on held-out

| Field | Value |
|-------|-------|
| failure_case_id | fail_012 |
| selector | rule_based |
| held-out WG | 0.9803 |
| always-SCORPIO WG | 0.9975 |
| gap | −0.0172 |
| status | **Unresolved** |
| next action | Do not claim rule selector beats SCORPIO; consider targeted rule repairs only with held-out evidence |

---

### fail_013: RF does not beat always-SCORPIO on held-out

| Field | Value |
|-------|-------|
| failure_case_id | fail_013 |
| selector | random_forest |
| held-out WG | 0.9975 |
| always-SCORPIO WG | 0.9975 |
| held-out accuracy | 0.606 |
| chosen policy dist | SCORPIO 32/33 windows |
| status | **Unresolved** — RF ties WG but collapses to SCORPIO |
| next action | Report always-SCORPIO as baseline; selector adds no held-out value |

---

### fail_014: Near-tie labels dominate (eps=0.005)

| Field | Value |
|-------|-------|
| failure_case_id | fail_014 |
| pattern | policy_margin < 0.005 on majority of windows |
| all-complete fraction | ~93% |
| status | **Unresolved** |
| next action | Regret-weight training (done); exclude near-tie windows from future training; add more KV-differentiated workloads |

---

### fail_015: All-complete windows create tie-breaking labels

| Field | Value |
|-------|-------|
| failure_case_id | fail_015 |
| pattern | best_wg ≥ 0.99 for most diversity windows |
| meaningful WG-gap windows | ~18/256 |
| status | **Unresolved** |
| next action | Do not treat all-complete label wins as strong supervision signal |

---

### fail_016: always-SCORPIO within 1pp of per-window oracle on held-out

| Field | Value |
|-------|-------|
| failure_case_id | fail_016 |
| always-SCORPIO WG | 0.9975 |
| oracle mean WG | 0.9995 |
| gap | −0.0020 |
| status | **Unresolved** — little headroom for any selector |
| next action | SCORPIO ablation; real-trace held-out before more selector tuning |

---

### fail_017: RF collapses to always-SCORPIO on held-out

| Field | Value |
|-------|-------|
| failure_case_id | fail_017 |
| selector | random_forest |
| SCORPIO dispatch fraction | 32/33 held-out windows |
| status | **Unresolved** |
| next action | Safe-fallback and KNN also tie SCORPIO; fixed SCORPIO is sufficient |

---

### fail_018: SCORPIO admission/completion trade-off ambiguous

| Field | Value |
|-------|-------|
| failure_case_id | fail_018 |
| pattern | WG computed on completed requests only; SCORPIO may throttle more |
| metrics | completion_fraction per policy in `completion_admission_summary.csv` |
| status | **Open — monitor** |
| next action | Report completion-penalized objectives; verify no metric gaming before publication |

---

## Rule Selector Repair Decision (Phase 2B.13)

**No rule repair applied.** RF/DT and alternative selectors were evaluated first.
RF ties always-SCORPIO but does not beat it; ad hoc rule additions risk overfitting to
diversity windows with all-complete tie-breaking. Trained selectors are the better path,
but current RF also fails to add held-out value.

---

## Recommended Next Step

Perform SCORPIO ablation and real-trace dataset ingestion before additional selector work.
If near-tie labels remain dominant, redesign workload suite to target sub-0.99 WG regimes.
