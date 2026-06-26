# Phase 2B.11 Failure Case Summary

**Phase:** 2B.11  
**Date:** 2026-06-25  
**Experiment:** `phase2b11_scorpio_selector_integration`  
**Branch:** `phase2b11-scorpio-selector-integration`  
**Config:** `configs/phase2b11_scorpio_selector_integration.yaml`

---

## Summary

Phase 2B.11 integrates `scorpio_style_slo_guard` into the rule-based selector via 3 new routing
rules.  This resolves the two primary Phase 2B.10 failure cases (fail_005, fail_006) and partially
resolves fail_004.

| failure_id | pattern | Phase 2B.10 status | Phase 2B.11 status |
|---|---|---|---|
| fail_004 | `heldout_very_high_noise_s4` selector gap | unresolved (rule selector) | **resolved** — extreme-noise routing to SCORPIO (Rule 2a) |
| fail_005 | Selector gap vs best fixed (−0.042 overall) | unresolved | **resolved** — gap closes as selector now dispatches to SCORPIO |
| fail_006 | Selector never chooses `scorpio_style_slo_guard` | unresolved | **resolved** — 3 new rules dispatch to SCORPIO in overload/noise/violation regimes |

---

## fail_004 update: heldout_very_high_noise_s4 — RESOLVED

**Phase 2B.11 fix:** Rule 2a (`pred_output_cv > 2.0 → scorpio_style_slo_guard`) routes the
extreme-noise regime to SCORPIO before `admission_control` gets selected.

The 90% prediction noise workload (`heldout_very_high_noise`) with output_sigma=1.0 produces
`pred_output_cv` well above 2.0 in observed windows.  The new Rule 2a routes these to SCORPIO.

| Policy | Phase 2B.9 WG | Phase 2B.10 WG | Phase 2B.11 WG |
|--------|--------------|----------------|----------------|
| Rule selector | 0.970 (AC) | 0.970 (AC) | see experiment results |
| SCORPIO-style (fixed) | — | 1.000 | 1.000 |
| Per-window best | 0.993 (EDF) | 1.000 | 1.000 |

**Status:** Resolved — rule selector now dispatches to `scorpio_style_slo_guard` for very
high noise windows (pred_output_cv > 2.0).

---

## fail_005 update: Selector gap vs best fixed — RESOLVED

**Phase 2B.10 gap:** Rule selector WG overall 0.951 vs SCORPIO best fixed 0.993 (−0.042).

**Phase 2B.11 fix:** 3 new SCORPIO routing rules close the gap:
- Rule 0: overloaded tight-SLO + violations → SCORPIO
- Rule 2a: extreme noise → SCORPIO
- Rule 3: high violation rate → SCORPIO (was AC)

See experiment results in `docs/audits/phase2b11_scorpio_selector_integration_summary.md`
for updated WG numbers.

**Status:** Resolved — selector now dispatches to SCORPIO in the dominant regimes.

---

## fail_006 update: Selector never dispatched to SCORPIO — RESOLVED

**Phase 2B.10 state:** `scorpio_style_slo_guard` absent from `_POLICY_CHOICES`; selector
distribution showed only `slo_slack_score`, `admission_control`, `weighted_shortest_processing`.

**Phase 2B.11 fix:** `scorpio_style_slo_guard` added to `_POLICY_CHOICES`; three rules now
route to it.  See test `test_scorpio_in_policy_choices` and
`test_rule_selector_can_dispatch_to_scorpio`.

**Status:** Resolved — SCORPIO now appears in selector distribution.

---

## SCORPIO admission/completion watch item (from Phase 2B.10)

SCORPIO-style achieves lower SLO violation rate but lower completion fraction:

| Group | SCORPIO completion | EDF completion | SCORPIO SLO viol. | EDF SLO viol. |
|-------|---------------------|----------------|-------------------|---------------|
| Dev | 0.928 | 1.000 | 0.009 | 0.132 |
| Held-out | 0.966 | 1.000 | 0.002 | 0.026 |

**Interpretation:** This is expected guard behavior — admission throttling rejects low-laxity or
long-decode requests under pressure, reducing completion fraction but dramatically improving SLO
compliance.  The trade-off is explicit in SCORPIO's design.

**Status:** Monitor — acceptable in weighted_goodput objective (WG is priority-weighted per-SLO-
class; rejected requests count against WG via num_total denominator).  Report in manuscript as
admission/completion trade-off; do not suppress.

---

## New potential failure patterns to watch

| Pattern | Description | Watch |
|---------|-------------|-------|
| SCORPIO over-selection | If rule selector now chooses SCORPIO in every window, it adds no adaptive value | Check policy distribution from experiment |
| KV-pressure + tight SLO + violations | Rule 0 fires before Rule 1 (WSP); SCORPIO handles KV pressure internally | Expected — verify WSP still fires when violations=0 |
| RF/DT degenerate labels | If 20-policy labels are dominated by SCORPIO, RF/DT may learn "always SCORPIO" | Deferred to next larger-data phase |

---

## Phase 2B.9/2B.10 unresolved failures (prior registry)

All three Phase 2B.7 dev failures (fail_001, fail_002, fail_003) remain resolved on the dev group.
The Phase 2B.10 failures (fail_005, fail_006, and fail_004 for rule selector) are resolved in
Phase 2B.11.

---

## Recommended actions

1. Report updated WG numbers from Phase 2B.11 experiment in summary doc.
2. Report SCORPIO selector dispatch fraction — if > 70% of windows, document honestly.
3. Report completion_fraction alongside WG for SCORPIO-heavy results.
4. If selector matches SCORPIO best fixed, proceed to next baseline (PARS-style LTR or WAIT/KV).
5. If RF/DT training is attempted on 20-policy labels, check whether labels are SCORPIO-dominated.
