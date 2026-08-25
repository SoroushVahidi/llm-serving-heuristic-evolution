# Family-A Mechanism Composite Rule Static Feasibility v1

Date: 2026-08-24

Classification: `MECHANISM_COMPOSITE_STATIC_NO_GO`

## Preflight

- branch: `contextual-compositional-heuristics-20260731`
- HEAD: `2987b7181efa2bc550d8a894c537eca8f6393eb6`
- upstream: `origin/contextual-compositional-heuristics-20260731`
- ahead/behind raw (`@{u}...HEAD`): `0	2`
- worktrees: 1
- git locks: 0
- tmux: `no server running on /tmp/tmux-1000/default`
- CPU count: 20

## Frozen Mechanism Contract

- WFS is the robust default and protects fairness, SLO/timeliness, high-priority/class-deficit states.
- ESTF is a completion-protection mechanism for service/size-sensitive states where WFS can incur completion regret.
- Runtime inputs are causal feature-contract fields: laxity, queue age, predicted service/output, class deficit, priority contrast, queue/load/KV pressure, and bounded history slopes.

## Prior Analytic-Index Exclusion

Excluded equivalent forms: priority/service, priority/laxity, c-mu-style priority/laxity/service, age-adjusted priority/service, Whittle-inspired priority*service/laxity, priority-only, and favored-size-only regime identity. The previous failures were collapse to static/regime identity or inverted quadrant semantics.

## Candidate Family

Parameterizations evaluated: 48 across templates `HARD_GUARD, HISTORY_STATIC_APPROX, LOW_PRESSURE_COMPLETION, PRIORITY_SAFE_RELEASE, TWO_STAGE_GUARD`. All candidates default to WFS and release ESTF only through guarded predicates; no learned model, coefficient fit, or high-dimensional scalar blend was introduced.

## Baselines

| rule | mean regret | p90 regret | ESTF rate | improvement vs WFS | overlap favored-rule |
|---|---:|---:|---:|---:|---:|
| `always_WFS` | 3.258517 | 14.000000 | 0.000 | 0.00% | 0.725 |
| `always_ESTF` | 24.551102 | 69.000000 | 1.000 | -653.44% | 0.275 |
| `pi0_frozen` | 0.872745 | 0.000000 | 0.356 | 73.22% | 0.917 |
| `idx_B_deadline_urgency` | 0.470942 | 0.000000 | 0.266 | 85.55% | 0.975 |
| `regime_favored_size_only` | 0.154309 | 0.000000 | 0.275 | 95.26% | 1.000 |
| `priority_only_rel` | 0.164329 | 0.000000 | 0.273 | 94.96% | 0.998 |

## Best Candidate

Best candidate: `PRIORITY_SAFE_RELEASE_09` (`PRIORITY_SAFE_RELEASE`)
- mean regret: 3.157315
- p90 regret: 14.000000
- total regret: 3151.000000
- improvement vs WFS: 3.11%
- ESTF decision rate: 0.006
- strict NO-GO reasons: ['mean regret reduction versus WFS <10%', 'mechanism quadrant ordering failed']

## Favored Side

| side | best mean regret | WFS mean regret | relative worse than WFS |
|---|---:|---:|---:|
| `long` | 0.212707 | 0.212707 | 0.00% |
| `short` | 10.937956 | 11.306569 | -3.26% |

## Quadrant Semantics

Ordering pass for best candidate: `False`

| quadrant | n | ESTF release rate | WFS retention rate |
|---|---:|---:|---:|
| `BOTH_RISKS` | 22 | 0.000 | 1.000 |
| `COMPLETION_ONLY` | 18 | 0.000 | 1.000 |
| `NEITHER_RISK` | 37 | 0.027 | 0.973 |
| `SLO_RISK_ONLY` | 14 | 0.000 | 1.000 |

## Non-Collapse

- overlap with favored-size rule: 0.731
- overlap with priority-only rule: 0.733
- overlap with best prior analytic rule `idx_B_deadline_urgency`: 0.740

## Robustness

- threshold sensitivity cases: 5
- leave-one-configuration checks: 13
- gain concentration: {'top_1pct_gain_share': 1.0, 'top_5pct_gain_share': 1.0, 'top_10pct_gain_share': 1.0, 'top_20pct_gain_share': 1.0}

## Best Rule Pseudocode

```text
action = WFS by default; for the selected template, compute causal guard predicates from service ratio, laxity, priority/deficit/pressure/history fields; return ESTF only if all release predicates pass, else WFS.
```

## Decision

Exact classification: `MECHANISM_COMPOSITE_STATIC_NO_GO`

Native closed-loop implementation allowed next: NO

Next scientific action: stop this guarded-rule line if treating the strict gate literally; return to broader portfolio-policy evolution rather than promoting this candidate.

## Confirmation

- no new selector training
- no DEV-driven redesign
- no FINAL
- no TEST
- no new oracle labels
- no new simulations
- no Wulver jobs
- no GPUs
- no git mutation
