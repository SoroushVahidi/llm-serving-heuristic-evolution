# Phase 2B.10 Failure Case Summary

**Phase:** 2B.10  
**Date:** 2026-06-25  
**Experiment:** `phase2b10_scorpio_slo_guard`  
**Config:** `configs/phase2b10_scorpio_slo_guard.yaml`

---

## Summary

Adding `scorpio_style_slo_guard` changes the competitive landscape. The repaired rule
selector **remains unchanged** (same WG as Phase 2B.9) but **no longer beats best fixed**
because SCORPIO-style becomes the dominant fixed baseline.

| failure_id | pattern | status |
|---|---|---|
| fail_005 | Selector loses to SCORPIO-style best fixed (overall gap −0.042) | **new — unresolved** |
| fail_006 | Selector never chooses SCORPIO-style despite rank #1 on all groups | **new — unresolved** |
| fail_004 | `heldout_very_high_noise_s4` rule selector vs best | **partially resolved for SCORPIO**; rule selector still fails |

---

## fail_005: Selector vs SCORPIO-style best fixed

| Group | Rule selector WG | Best fixed (SCORPIO) WG | Gap |
|-------|------------------|-------------------------|-----|
| Dev | 0.917 | 0.988 | **−0.071** |
| Held-out | 0.979 | 0.998 | **−0.019** |
| Overall | 0.951 | 0.993 | **−0.042** |

**Pattern:** Rule selector was tuned without SCORPIO-style in the candidate set. After Phase 2B.10,
per-window best deployable reference rises (oracle WG overall 0.994 vs 0.954 in Phase 2B.9).

**Status:** Unresolved — requires selector rule/training update to route to `scorpio_style_slo_guard`.

---

## fail_006: Selector does not dispatch to SCORPIO-style

The rule selector has no feature rule targeting `scorpio_style_slo_guard`. Policy distribution
unchanged from Phase 2B.9 (`slo_slack_score`, `admission_control`, `weighted_shortest_processing` only).

**Pattern:** selector_does_not_choose_best_new_baseline

**Status:** Unresolved — add SCORPIO routing rule or retrain RF/DT with 20-class labels.

---

## fail_004 update: heldout_very_high_noise_s4

| Policy | WG (Phase 2B.10) |
|--------|------------------|
| Rule selector (`admission_control`) | 0.970 |
| SCORPIO-style | **1.000** |
| Per-window best | 1.000 (`scorpio_style_slo_guard`) |

**SCORPIO-style resolves the high-noise failure** that EDF/admission_control could not fully fix.
The rule selector failure **persists** because it still routes to `admission_control`.

**Status:** Unresolved for rule selector; resolved for SCORPIO-style fixed baseline.

---

## SCORPIO-style admission/completion watch item

SCORPIO-style achieves lower SLO violation rate but **lower completion fraction** than EDF/AC:

| Group | SCORPIO completion | EDF completion | SCORPIO SLO viol. | EDF SLO viol. |
|-------|---------------------|----------------|-------------------|---------------|
| Dev | 0.928 | 1.000 | 0.009 | 0.132 |
| Held-out | 0.966 | 1.000 | 0.002 | 0.026 |

**Pattern:** trades completion for SLO compliance (expected guard behavior).  
**Status:** Monitor — not catastrophic; document in manuscript as admission/completion trade-off.
Not blocking selector candidacy, but completion accounting should be reported honestly.

---

## Phase 2B.9 failures (unchanged)

`heldout_very_high_noise_s4` rule-selector gap remains for the rule-based selector.
SCORPIO-style does not introduce catastrophic regime failures.

---

## Recommended actions

1. Add selector rule or training label for `scorpio_style_slo_guard` under overload / tight-SLO / high-noise.
2. Re-run Phase 2B.9/2B.10 suite after selector update.
3. Report completion_fraction alongside WG when citing SCORPIO-style results.
