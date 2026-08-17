# Family C v2 — KV-Pressure Reserve Refinement — Audit

**Date:** 2026-08-17
**Verdict:** `KV_FAMILY_COMPOSITION_READY`
**Run:** [`experiments/kv_pressure_pilot_v2_20260817T165053Z/`](../../experiments/kv_pressure_pilot_v2_20260817T165053Z/)
**Design:** [`docs/design/POLICY_SEPARATION_FAMILY_KV_PRESSURE_V2.md`](../design/POLICY_SEPARATION_FAMILY_KV_PRESSURE_V2.md)
**v1 (frozen, untouched):** `experiments/kv_pressure_pilot_v1_20260817T162650Z/`, audit [`family_c_kv_pressure_pairwise_separation_v1_20260817.md`](family_c_kv_pressure_pairwise_separation_v1_20260817.md)
**Parents (unmodified):** `kv_constrained_online` vs `least_laxity_first`

This is still a **pairwise-separation pilot**, not a composition experiment.
No selector was fit, no child policy was built or run. Per explicit task
scope, composition work does not start in this task even though the verdict
below is `KV_FAMILY_COMPOSITION_READY` — see §M for exactly what that
verdict does and does not authorize.

## A. v1 tie-rate diagnosis (summary; full detail in design doc v2 §1)

- Ties concentrated jointly in `bulk_pressure=low × urgent_tightness=loose`
  (100% — the intended placebo control), not uniformly.
- ANWG resolution was the dominant cause: only 9 distinct achievable values
  across all 64 v1 rows (population ~20 requests/scenario); every "tied"
  cell was an *exact* zero delta, none were small-but-nonzero.
- Reserve activated broadly (every v1 cell), not narrowly.
- A previously-undiagnosed confound: at v1's `BULK_SLACK_S=1.5`, the median
  bulk tenant's own laxity (0.176s) was below
  `KVConstrainedOnlinePolicy.urgent_laxity_seconds=0.25`, so bulk tenants
  frequently bypassed the reserve gate as "urgent" themselves.
- Seed variance was material (tie rate 25%-88% across v1's 4 seeds).
- The late>early advantage was stable in sign across all 4 v1 seeds but
  variable in magnitude (5× range).

## B. Scientific rationale for v2 changes

Population roughly doubled (ANWG resolution fix); `BULK_SLACK_S` raised
1.5→1.65 (accidental-urgency partial fix, verified not to be the dominant
driver of the resulting kv-favoring asymmetry — a further sweep to 2.1 barely
changed bidirectionality); third `urgent_arrival_phase` level (`middle`)
added for finer trajectory resolution; seeds increased 4→6 with 2 explicitly
held out. Parent algorithms and reserve semantics: **unchanged**. Full
justification table in design doc v2 §2.

## C. Frozen v2 factors and levels

`bulk_pressure` (low/high) × `urgent_arrival_phase` (early/middle/late) ×
`urgent_tightness` (loose/tight) × 6 seeds = 72 scenarios. In-sample
(gate-computation) seeds: `20260910-13`. Held-out (G6-only) seeds:
`20260914-15`. `max_kv_tokens=6000` unchanged from v1.

## D. Measured KV-pressure regimes

Peak-KV-utilization is not a usable low/high differentiator (both regimes
saturate ~0.97-1.10 once any backlog exists, a structural property of
greedy-to-capacity admission — same finding as v1). Reserve-deferral count
and sustained-backlog duration (`n_steps`) are: `bulk_pressure=high` shows
mean 2535 deferrals / 3871 steps vs `low`'s 1767 deferrals / 1825 steps
(smoke, §L) — a real, measurable low/high regime distinction.

## E. Real / derived / intervention fields

Unchanged from v1 (design doc v2 §5) except population size and
`BULK_SLACK_S`. Provenance tagging (`burstgpt_staged`/`burstgpt_anchored`/
`synthetic_lognormal`) recorded per-scenario as before.

## F. Observable / hidden fields

Unchanged from v1; leakage guard (`assert_policy_visible_fields_clean_kv_v2`)
additionally forbids `"phasemiddle"`/`"kvp2."` tokens.

## G. Preregistered hypotheses (H1-H5)

See design doc v2 §8. H3 (timing) was revised *during calibration, before
any v2 cell was scored* from a strict `early<late` ordering to "materially
larger at middle and/or late than early" after calibration showed the peak
effect at `middle`, not `late` (§L).

## H. Preregistered gates (G1-G10) and results

| Gate | Test | Result |
|---|---|---|
| G1 | Bidirectional (ε=0.01) | **PASS** — kv wins 29/48, llf wins 4/48 |
| G2 | Aggregate tie rate <50% | **PASS** — 31.2% (15/48); signal quadrant (`high,tight`) tie rate 16.7% (2/12), better than v1's 25% |
| G3 | Seed stability (`high,tight`: middle-or-late > early, ≥3/4 seeds) | **PASS** — 3/4 seeds (`20260913` shows a small negative delta-of-delta but still a substantial positive absolute advantage throughout, see §Q) |
| G4 | Reserve activation nontrivial | **PASS** — deferrals logged in 48/48 in-sample cells (100%), not saturated-with-zero-outcome (33/48 non-tied) |
| G5 | Timing/pressure interaction | **PASS** — middle/late > early in `urgent_tightness=tight` for **both** `bulk_pressure` levels (low: 0.10→0.24/0.25; high: 0.118→0.169/0.176) |
| G6 | **Held-out replication** | **PASS** — same direction on 2 held-out seeds never used for any threshold decision; effect is if anything *stronger* on held-out (`high,tight,late` mean=0.309 held-out vs 0.176 in-sample) |
| G7 | Action disagreement in decisive states | **PASS** — every cell with a nonzero ANWG delta also shows nonzero `n_reserve_deferrals` |
| G8 | No universal dominant parent | **PASS** — `least_laxity_first` wins 4/48 (not zero) |
| G9 | Integrity/safety | **PASS** — 0/144 failed, 0 duplicate `(scenario_id,policy)` pairs, 0 NaN/Inf, leakage guard passed on all 72 scenarios |
| G10 | **Within-scenario (not merely cross-scenario) opportunity** | **PASS** — 6/16 `(bulk_pressure, urgent_tightness, seed)` in-sample combinations show a *different practical winner* (`kv`/`llf`/`tie`) at different `urgent_arrival_phase`, holding everything else in the scenario fixed |

**10/10 gates pass.**

## I. Held-out design

Confirmed as specified (design doc v2 §7): 4 in-sample seeds used for every
gate/threshold decision; 2 held-out seeds (`20260914-15`) used exclusively
for G6, never for calibration or threshold-setting. Both partitions run in
the same frozen pilot invocation; the partition was decided in the design
doc before the pilot was launched.

## J. Implementation changes

New: `src/llmserveopt/policy_separation/templates_kv_pressure_v2.py`
(imports shared BurstGPT/`req`/`kv_scarce_gpu` helpers from v1's module and
`templates_prefill_decode.py`, per instruction item 8 — no parallel
infrastructure). Extended (not duplicated):
`scripts/run_policy_separation_kv_pressure_pilot_v1.py` gained
`--template-version {v1,v2}` and `held_out_seeds` config support; v1
invocations are unaffected (verified, §K). New configs:
`configs/kv_pressure_{smoke,pilot}_v2.yaml`. `kv_constrained_online.py` /
`least_laxity_first.py`: **zero changes** (verified by contract tests, §K).

## K. Tests

- `tests/test_policy_separation_kv_pressure_v2.py`: 18 new tests (parent-policy
  contract unchanged; population/phase-level correctness; bulk-median-laxity
  regression guard; leakage guard; full-grid uniqueness; held-out split
  wiring via the runner; v1-invalid-for-held-out-seeds guard; reserve
  activation under calibrated pressure).
- `tests/test_policy_separation_kv_pressure_v1.py`: all 20 still pass
  unmodified (v1 runner backward-compatibility verified directly, §L).
- 38/38 total, plus `scripts/check_project_handoff_consistency.py` passed.

## L. Smoke/calibration history

Four rounds, all pre-registered-before-scoring (full detail: design doc v2
§4):

1. `BULK_SLACK_S=2.0` (theoretically safe bulk laxity) → bidirectionality
   collapsed (32-vs-1). Rejected.
2-3. Swept `BULK_SLACK_S ∈ {1.6, 1.65}` → **1.65 selected**: bidirectional
   (36-vs-6 on calibration seeds), tie rate 12%, ANWG resolution 9→20
   distinct values.
4. Swept `max_kv_tokens ∈ {8000..11000}` at `BULK_SLACK_S=1.65` → worse on
   every axis (tie rate rose to 46-81%, bidirectionality did not improve).
   `max_kv_tokens=6000` (unchanged from v1) confirmed as the better choice.

A fifth check (per-seed, not per-nominal-median, measurement) found the
accidental-bulk-urgency confound was only *partially* fixed by
`BULK_SLACK_S=1.65` under real heavy-tailed BurstGPT sampling (10-80% of
bulk tenants "urgent" depending on seed/cell); a further sweep to 2.1 barely
moved bidirectionality (0-1 llf wins regardless), so this was accepted as an
honest, documented residual limitation rather than chased further (design
doc v2 §4 follow-up finding).

**Smoke run** (`configs/kv_pressure_smoke_v2.yaml`, 4 calibration seeds, 48
cells, via the frozen runner): 96/96 success; wins_kv=36/wins_llf=6/ties=6
(tie rate 12.5%, matching the ad hoc calibration exactly — cross-checked);
100% of cells show reserve deferrals; low/high regime differentiation
confirmed via deferral count and duration (§D); no failures.

## M. Final run integrity

72 scenarios, 144 evaluations (72×2 policies), 0 failed, 0 duplicate
`(scenario_id, policy)` pairs, 0 NaN/Inf. Leakage guard passed on all 72
scenarios (generation would raise `AssertionError` otherwise). 96 in-sample
rows + 48 held-out rows, correctly tagged and separable via the `held_out`
CSV column.

## N. Pairwise separation / ties

In-sample: 29 kv-wins / 4 llf-wins / 15 ties (48 cells), aggregate tie rate
31.2% (down from v1's 59.4%). Signal quadrant (`bulk_pressure=high,
urgent_tightness=tight`) tie rate 16.7% (down from v1's 25%).

## O. Reserve activation and action disagreement

100% of in-sample `kv_constrained_online` cells log ≥1 reserve-deferral
event (48/48); every cell with a nonzero ANWG delta also shows nonzero
deferrals (G7) — the mechanism that differs between the policies is
demonstrably the one associated with the outcome difference, not a
coincidental correlation.

## P. Within-scenario occupancy/timing evidence

The central new result. Holding `(bulk_pressure, urgent_tightness, seed)`
fixed and varying only `urgent_arrival_phase`:

- 6 of 16 such matched triples show a **different practical winner**
  (`kv`/`llf`/`tie`) purely as a function of arrival timing (G10).
- Mean advantage at `urgent_tightness=tight`: `low` bulk_pressure
  0.10(early)→0.24(middle)→0.25(late); `high` bulk_pressure
  0.118(early)→0.169(middle)→0.176(late) — both bulk-pressure levels show
  the same qualitative pattern (G5).
- Placebo (`urgent_tightness=loose`) stays near-zero at every phase in both
  bulk-pressure levels (−0.025 to 0.096), confirming H2/H4.

## Q. Held-out / seed robustness

G6: the middle/late > early pattern **replicates on the 2 held-out seeds**,
which were never used for any calibration or threshold decision, at
comparable or larger magnitude (`high,tight,late` mean=0.309 held-out vs
0.176 in-sample — if anything the held-out seeds show a *stronger* effect).
G3: 3 of 4 in-sample seeds individually show middle-or-late exceeding early
in the signal cell; the one exception (`20260913`) still shows a
substantial positive kv-advantage at every phase (0.206/0.176/0.176), it
simply does not grow further from an already-high early value for that
specific seed — not a sign reversal, a magnitude plateau.

## R. Final KV-family verdict

**`KV_FAMILY_COMPOSITION_READY`** — all 10 preregistered gates pass,
computed mechanically from frozen thresholds set before any v2 cell was
scored (design doc v2 §9), with the key novel criteria (G6 held-out
replication, G10 within-scenario winner-flip) passing on genuinely
out-of-calibration data.

## S. Is composition scientifically justified?

**Yes, more than for either prior family, but on a specific and bounded
claim — not yet proof that a state-dependent child beats both parents on
one trajectory.** Per instruction item 13, these are explicitly NOT
equated:

- **What is now shown:** the *scenario-level optimal parent choice* is not
  fixed for a given `(bulk_pressure, urgent_tightness, seed)` — it depends
  on *when within the trajectory* urgent tenants arrive (G10), and this
  timing-dependent pattern replicates on held-out seeds (G6). This is
  stronger motivating evidence for composition than either ESTF/WFS or
  PrefillControl v2 produced, because in both of those a scenario-level
  fitted selector had already been shown sufficient to reach the two-parent
  oracle — no equivalent within-scenario-timing dependency was ever
  demonstrated for either.
- **What is NOT yet shown:** that a genuinely state-dependent KV-admission
  child, making an online per-step or per-arrival decision, would actually
  *outperform both fixed parents on the same single trajectory* (rather
  than a scenario-level selector merely picking whichever parent that
  trajectory favors). Parent action disagreement (G7) is not performance
  improvement; reserve activation (G4) is not composition readiness; the
  cross-*timing* complementarity shown here (G10) is not yet direct
  evidence of within-*trajectory* complementarity in the stronger sense a
  composition falsification would test. Only an actual falsification
  experiment (not run in this task) can determine that.

## T. Exact next action

Per instruction item 14, since the verdict is `KV_FAMILY_COMPOSITION_READY`:
the smallest scientifically justified next step would be a **two-parent
composition falsification** for `kv_constrained_online` vs
`least_laxity_first`, structured exactly like the Family B v2 PrefillControl
falsification that preceded it — TRAIN/VAL-fitted scenario-level top-1
selector as the baseline to beat, held-out TEST/OOD, and a genuinely
state-dependent child (here: an admission policy that reads live
`current_kv_tokens` and adjusts its reserve threshold or urgency bypass
per-arrival, analogous to how `PrefillControlChildPolicy` read live
step-level features) as the composition target, using this pilot's `middle`
phase (its own held-out-confirmed strongest cell) as the natural design
anchor. **This is explicitly not implemented or run in this task.**

## U. Tests / checkers

38/38 new+existing focused tests pass; `check_project_handoff_consistency.py`
passes; v1's frozen artifacts (design doc, audit, `experiments/kv_pressure_pilot_v1_.../`)
are untouched (verified via `git status` showing zero modifications to any
v1 file).
