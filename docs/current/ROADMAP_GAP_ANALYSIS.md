# Roadmap Gap Analysis

This document records the current gap diagnosis after Selector v2/v3, composition readiness, native composition pilot, and structural synthesis readiness.

## Evidence So Far

- Selector v2 proved adaptive policy selection can help in-distribution.
- Selector v2/v3 OOD evaluations repeatedly showed fixed WSP remains hard to beat on held-out shifts.
- Dynamic causal features improve WSP-vs-SCORPIO delta learnability.
- Generic uncertainty fallback and source balancing were insufficient.
- Naive rank mixtures and component-wise composition did not clear the native pilot decision bar.
- Structural synthesis machinery is ready for small, typed child generation.

## Likely Bottlenecks

| Bottleneck | Current likelihood | Evidence |
| --- | --- | --- |
| Workload/domain coverage | high | Selector v3 status is `DATA_LIMITED`; OOD shift remains detectable. |
| Causal feature representation | medium-high | Dynamic features helped WSP-vs-SCORPIO boundary learning but did not fully solve robustness. |
| Selector modeling | medium-low | Stronger generic model swapping has not been the decisive lever. |
| Policy library incompleteness | unknown pending | Policy Library v2 frontier workflow is still running. |
| Naive composition | low | Native pilot returned `NO_GO` for rank/component-wise composition. |
| Structural synthesis/evolution | promising but unvalidated | Harness is ready, but no scaled scientific claim yet. |
| Simulator action-space limits | medium | Several literature families require unsupported cache/routing/splitting/chunking capabilities. |

## Current Research Posture

Do not spend the next major effort on broad selector model sweeps or dense weighted policy averaging. Wait for the frontier and Policy Library v2 reports, then either:

1. launch a narrowly justified full composition experiment;
2. generate structural symbolic children from high-value parent policies;
3. expand simulator capabilities if missing action/state mechanisms dominate the frontier gaps.
