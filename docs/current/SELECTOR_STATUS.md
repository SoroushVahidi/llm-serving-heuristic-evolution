# Selector Status

Current selector evidence as of 2026-07-21.

## Completed Selector Milestones

- Leakage-safe Selector Dataset v2 pipeline was generated and audited.
- Selector v2 OOD diagnosis identified distribution shift and WSP-vs-SCORPIO routing failures.
- Selector v3 added broader domains and richer causal-state features.
- WSP-vs-SCORPIO delta learnability improved with v3 dynamic features.

## Main Results

| Experiment | Status | Key result |
| --- | --- | --- |
| Selector v2 Overnight Scale | COMPLETE | 1600 leakage-safe windows, 775 meaningful windows, audit PASS. RF per-policy regressor beat WSP on ID (`0.559481` vs `0.527113`) but lost on OOD (`0.247707` vs `0.256383`). |
| Selector v2 OOD Investigation | COMPLETE | `SELECTOR_STATUS = IMPROVE_DATA_OR_FEATURES`; OOD shift detectable, generic uncertainty fallback did not fix robustness. |
| Selector v3 Multi-Domain Causal-State | COMPLETE | `SELECTOR_STATUS = DATA_LIMITED`; richer dynamic causal features helped boundary learnability, but selectors still did not consistently beat fixed WSP on held-out ID/OOD/final splits. |

## Current Interpretation

The selector is not fundamentally invalid: it can exploit in-distribution policy differences. The current bottleneck is robust generalization across source/time/domain shifts. Evidence points most strongly to insufficient workload/domain coverage and missing or lossy causal serving-state representation, with model swapping alone unlikely to be the next high-value lever.

## Current Stop/Go Position

- Freeze broad generic selector-model sweeps.
- Continue only targeted selector work tied to new domain coverage, richer causal state, or new policy/frontier evidence.
- Do not tune against final OOD labels.
- Use grouped leakage-safe splits and development-only selection for any future selector/composition/synthesis experiment.
