# Policy Separation Family C v2 — KV-Pressure Reserve Refinement

**Date:** 2026-08-17
**Status:** PREREGISTERED — pilot not yet executed
**Predecessor:** [`POLICY_SEPARATION_FAMILY_KV_PRESSURE_V1.md`](POLICY_SEPARATION_FAMILY_KV_PRESSURE_V1.md), frozen and unchanged
**v1 verdict:** `KV_FAMILY_USEFUL_NEEDS_REFINEMENT` (5/6 gates; tie-rate gate 59.4%, target <50%)
**v1 run (frozen, untouched):** `experiments/kv_pressure_pilot_v1_20260817T162650Z/`

This document diagnoses v1's tie-rate problem quantitatively (§1), justifies
every v2 change against that diagnosis (§2), and preregisters v2's frozen
factors, hypotheses, and gates (§3-6) **before** the v2 pilot is scored. Per
explicit task scope, this is still a pairwise-separation refinement, not a
composition experiment.

## 1. v1 tie-rate diagnosis (from the frozen v1 CSV, no new runs)

**A. Are ties concentrated in low-pressure cells where the reserve never
binds?** Partially, but the concentration is by `bulk_pressure` **and**
`urgent_tightness` jointly, not `bulk_pressure` alone:

| Split | Tie rate |
|---|---:|
| `bulk_pressure=low` | 81% (13/16) |
| `bulk_pressure=high` | 38% (6/16) |
| `urgent_tightness=loose` | 75% (12/16) |
| `urgent_tightness=tight` | 44% (7/16) |
| `bulk_pressure=low, urgent_tightness=loose` (the intended placebo control) | **100% (8/8)** |
| `bulk_pressure=high, urgent_tightness=tight` (the intended signal cell) | **25% (2/8)** |

The aggregate 59.4% is dragged up almost entirely by the `low×loose`
placebo-control quadrant, which is *supposed* to tie (H4) — the signal
quadrant (`high×tight`) already has a 25% tie rate, closer to (though still
above) Family B v2's 3.1%. **v1's own preregistered gate (a single flat
<50% aggregate bound) did not distinguish "ties where the mechanism
shouldn't differentiate" from "ties where it should" — this is a gate-design
lesson v2 corrects (§6, G2 is now reported both aggregate and signal-cell-only).**

**B/E. Are the two policies making identical decisions, or is ANWG too
coarse?** The latter, decisively. Across the entire 64-row v1 CSV, ANWG
takes only **9 distinct discrete values** (`[0.65, 0.75, 0.8, 0.8333, 0.85,
0.9, 0.9167, 0.95, 1.0]`), a direct consequence of the ~20-request-per-scenario
population (achievable ANWG values are multiples of ~1/12 to 1/20). Every
single one of the 19 "tied" cells is an **exact** zero delta — there are
**zero** cells with a small-but-nonzero gap (`0 < |Δ| ≤ 0.01`). The ε=0.01
threshold is not doing discriminating work at this resolution: a "tie" is
simply "both policies produced the identical discretized success count."
This — not policy-decision identity — is the dominant cause of the elevated
tie rate.

**C/D. Is pressure too weak, too short-lived, or too binary? Does the
reserve activate only in a narrow region?** No — `kv_constrained_online`
logs reserve-deferral events in **every one of its 32 v1 cells**, including
nominally `bulk_pressure=low`, because the 6 fixed urgent tenants alone
already create meaningful concurrent KV load at `max_kv_tokens=6000`. The
mechanism activates broadly, not narrowly.

**A new finding, not in the original v1 diagnosis checklist but surfaced by
it:** the v1-calibrated `BULK_SLACK_S=1.5` put the *median* bulk tenant's
own laxity (`1.5 − (0.5×2048 + 1.0×300)×0.001 = 0.176s`) **below**
`KVConstrainedOnlinePolicy.urgent_laxity_seconds=0.25` — bulk tenants were
themselves frequently classified "urgent" by the policy's own definition and
bypassing the reserve gate via the urgent override. This confounds the
intended "purely deferrable background load, never itself urgent" role of
the bulk tenant class (design doc v1 §5) and is visible in v1's deferral
counts: `bulk_pressure=low, urgent_tightness=tight` cells show near-zero
deferrals (0-40) despite real KV pressure, because bulk tenants in those
cells were bypassing the gate as "urgent" themselves. This is a workload
calibration defect (analogous to v1's Round-3 infeasible-prompt bug), not a
finding about the mechanism.

**F. Is seed variance material?** Yes, substantially: tie rate by seed
ranges from 25% (`20260904`) to 88% (`20260902`) — a real source of
noise at n=4 seeds/cell, motivating more seeds in v2 (§3) and an explicit
held-out replication check (§7).

**G. Is the late>early advantage stable across seeds, or driven by a
subset?** Stable in **sign** (all 4 seeds positive: `[0.1, 0.05, 0.1,
0.25]` in the `bulk_pressure=high, urgent_tightness=tight` cell) but
variable in **magnitude** (one seed shows 5× the effect of another) — a
real effect, not a single-seed artifact, but with enough seed-to-seed
variance that v2 needs more seeds and a genuine held-out check before
treating the pattern's exact shape as established.

## 2. Scientific rationale for v2 changes (every change traces to §1)

| Change | Diagnosis it addresses | NOT changed |
|---|---|---|
| Population roughly doubled (`N_BULK`: 6/14→10/24, `N_URGENT`: 6→10) | B/E: ANWG resolution (9→20 distinct values in calibration, §4) | Prompt/output *distributions*, medians, provenance tagging |
| `BULK_SLACK_S`: 1.5→1.65 | The new accidental-urgency finding (§1): keeps median bulk laxity safely non-urgent (0.326s > 0.25s) while preserving genuine bulk-tenant risk under deferral (bidirectionality, verified empirically §4, not assumed) | Urgent-tenant slack levels (`loose`/`tight` unchanged) |
| Third `urgent_arrival_phase` level (`middle`, fraction=0.35) | F/G: finer trajectory resolution to characterize the timing effect's actual shape, not just its endpoints | `early`(0.0)/`late`(0.7) fractions unchanged |
| Seeds: 4 calibration (reused v1 seeds) → 6 fresh final-pilot seeds, 2 held out | F: seed variance; item 7's held-out requirement | Nothing about the mechanism |
| `max_kv_tokens=6000` | — | **Unchanged** — re-verified in calibration (§4), not itself a diagnosed problem |
| Parent policies (`kv_constrained_online`, `least_laxity_first`) | — | **Unchanged**, per instruction — no algorithm or reserve-semantics edits |

No new mechanism, no prefix-reuse (still unsupported by the simulator), no
hidden generator label added to any observable feature (leakage guard
`assert_policy_visible_fields_clean_kv_v2` re-verified, §8).

## 3. Frozen v2 factors and levels

| Factor | Levels | Change from v1 |
|---|---|---|
| `bulk_pressure` | `low` (n_bulk=10) / `high` (n_bulk=24) | counts doubled |
| `urgent_arrival_phase` | `early` (0.0) / `middle` (0.35) / `late` (0.7) | **new middle level** |
| `urgent_tightness` | `loose` (slack=3.0s) / `tight` (slack=0.55s) | unchanged |
| `seed` | 6 fresh seeds (§7) | 4→6, none reused from v1 or v2 calibration |

`2 × 3 × 2 × 6 = 72` scenarios (in-sample 4 seeds × 12 cells = 48 scored for
gates; 2 held-out seeds × 12 cells = 24 scored only for the G6 replication
check, §7).

GPU: `kv_scarce_gpu(max_kv_tokens=6000, max_active_sequences=64,
max_batch_tokens=64)` — unchanged from v1, re-verified in calibration (§4).

## 4. Measured KV-pressure regimes (calibration, before the frozen grid)

Calibration used the 4 v1 seeds (already "spent" on v1 tuning — appropriate
re-use for calibration, not final-pilot scoring) across the full v2 factor
grid at trial parameter values, checking against v1's diagnosis:

| Trial | `BULK_SLACK_S` | `max_kv_tokens` | Bulk median laxity | Bidirectional? | Tie rate (48 cells) | ANWG distinct values |
|---|---:|---:|---:|---|---:|---:|
| 1 | 2.0 | 6000 | 0.676s (safely non-urgent) | **No** — 32-vs-1 | 31% | 20 |
| 2 | 1.6 | 6000 | 0.276s (non-urgent) | Yes — 35-vs-6 | 15% | — |
| 3 | 1.65 | 6000 | 0.326s (non-urgent) | Yes — 36-vs-6 | **12%** | 20 |
| 4 | 1.65 | 8000-11000 | 0.326s | Worse (2-3 llf wins, rising tie rate to 81% at 11000) | 46-81% | — |

Trial 3 (`BULK_SLACK_S=1.65`, `max_kv_tokens=6000` unchanged) selected: at
the *nominal median* prompt/output values, the bulk tenant is safely
non-urgent (laxity=0.326s > 0.25s); bidirectional (LLF wins 6/48, satisfying
gate G8); tie rate 12% (well under both v1's 59.4% and the <50% target); and
ANWG resolution more than doubled (9→20 distinct values). **This is an
asymmetric result (kv_constrained wins far more often than LLF) — accepted
as-is, not further tuned toward "balance."** Per instruction, the objective
was to expose the natural boundary, not force bidirectional parity.

**Follow-up finding (honest caveat, not a further round of win-chasing):**
per-scenario/per-seed measurement (not just the nominal-median check above)
showed real BurstGPT-anchored prompt sampling is heavy-tailed enough that
the *fraction* of individual bulk tenants classified "urgent" by the
policy's own threshold varies 10-80% across seeds/cells at
`BULK_SLACK_S=1.65` — the accidental-urgency confound (§1) is *reduced*
relative to v1, not eliminated. A further sweep (`BULK_SLACK_S ∈ {1.8, 1.9,
2.0, 2.1}`, calibration seeds only) lowered the mean urgent-bulk fraction
(32%→10%) but did **not** materially change bidirectionality (LLF wins
stayed at 0-1/48 regardless) — i.e., the residual confound is not the
dominant driver of the observed kv-favoring asymmetry once the population is
scaled up, so further chasing it does not change the qualitative picture and
was stopped. `BULK_SLACK_S=1.65` is kept (best bidirectionality found, 6/48)
rather than a higher value that only reduces statistical power without
restoring balance. The test suite (`tests/test_policy_separation_kv_pressure_v2.py::
TestV2ScenarioGeneration::test_bulk_tenants_are_mostly_not_urgent`) checks
the aggregate rate stays a minority (<50%), not that any single scenario is
exactly zero.

Regime coverage confirmed at final parameters: `bulk_pressure=low` (reserve
binds, but on a smaller population — "transitional" pressure),
`bulk_pressure=high` (reserve binds heavily — "sustained high" pressure);
no cell showed zero deferral events (ruling out a "reserve never binds"
regime) and no cell showed 100%-of-steps-over-threshold saturation with zero
distinguishing outcome (ruling out "always maximally saturated, no
resolution" — the failure mode that made peak-KV-utilization unusable as a
low/high differentiator in v1, still true here, so v2 continues to use
reserve-deferral-event counts as the pressure-regime diagnostic, not peak
utilization).

## 5. Real / derived / intervention fields

Unchanged from v1 (design doc v1 §6) except population size and
`BULK_SLACK_S`. `bulk_prompt_source`/`urgent_prompt_source` provenance
tagging (`burstgpt_staged`/`burstgpt_anchored`/`synthetic_lognormal`) is
recorded identically per-scenario.

## 6. Observable / hidden fields

Unchanged from v1 (design doc v1 §7). Leakage guard extended
(`assert_policy_visible_fields_clean_kv_v2`) to also forbid `"phasemiddle"`
and `"kvp2."` tokens in `class_id`.

## 7. Held-out evaluation structure

- **Calibration seeds:** the 4 original v1 seeds (`20260901-04`) — reused
  for v2 calibration only (§4), never scored in the frozen v2 pilot.
- **Final pilot, in-sample (gate-computation) seeds:** `20260910, 20260911,
  20260912, 20260913` — fresh, never used in any calibration decision.
  Gates G1-G5, G7-G10 (§9) are computed from these 4×12=48 cells only.
- **Held-out seeds:** `20260914, 20260915` — fresh, never used in any
  calibration decision, and **excluded from every gate threshold/parameter
  decision in this document**. Used exclusively for gate G6 (does the
  timing/occupancy pattern from the in-sample seeds replicate on these two
  seeds not involved in setting any threshold).

All 6 final-pilot seeds are evaluated in a single frozen run
(`configs/kv_pressure_pilot_v2.yaml`) for efficiency; the in-sample/held-out
partition is applied at analysis time, decided here before any v2 cell is
scored.

## 8. Preregistered hypotheses (H)

- **H1 (mechanism, retained from v1):** `kv_constrained_online`'s
  urgent-tenant SLO advantage over `least_laxity_first` is not constant
  across trajectory timing.
- **H2 (placebo, retained):** the H1 effect is small under
  `urgent_tightness=loose` (both policies trivially meet SLO).
- **H3 (timing, revised from v1's strict `early<late` after calibration,
  §4):** the H1 effect is materially larger at `urgent_arrival_phase`
  `middle` and/or `late` than at `early`; the exact `middle` vs `late`
  ordering is not preregistered as monotonic (calibration showed the peak at
  `middle`, not `late`, plausibly because sustained mid-convoy pressure can
  exceed end-of-convoy pressure once earlier admissions start completing —
  see §4).
- **H4 (no universal dominance, new in v2):** `least_laxity_first` wins a
  nonzero number of cells even after the accidental bulk-urgency confound
  is corrected — the boundary is asymmetric, not one-sided.
- **H5 (replication, new in v2):** H1/H3 replicate in sign and rough
  magnitude on the 2 held-out seeds not used in any calibration decision.

## 9. Preregistered gates (G), frozen thresholds

| Gate | Test | Threshold |
|---|---|---|
| G1 | Bidirectional practical separation (ε=0.01) | each policy wins ≥1/48 in-sample cells |
| G2 | Near-tie rate | aggregate <50% (v1's original bound) **and** reported separately for the signal quadrant (`bulk_pressure=high, urgent_tightness=tight`) for comparability to v1's 25% |
| G3 | Seed stability | sign of the `(high,tight,middle-or-late) − (high,tight,early)` mean delta is the same across ≥3 of 4 in-sample seeds |
| G4 | Reserve activation nontrivial | ≥1 deferral event on ≥75% of `kv_constrained_online` in-sample cells; not 100% of cells at saturation with zero distinguishing outcome |
| G5 | Timing/pressure interaction (H3) | mean `(kv_constrained−llf)` delta at `middle` or `late` exceeds the mean delta at `early`, within `urgent_tightness=tight`, for both `bulk_pressure` levels |
| G6 | **Held-out replication (H5)** | the G5 direction (middle/late > early) holds on the 2 held-out seeds too, evaluated identically but never used to set any threshold |
| G7 | Action disagreement in decisive states | ≥1 cell shows both a nonzero `n_reserve_deferrals` count and a nonzero ANWG delta (i.e., the mechanism that differs is the one associated with the outcome difference, not a coincidence) |
| G8 | No universal dominant parent (H4) | `least_laxity_first` wins ≥1/48 in-sample cells (not literally 0) |
| G9 | Canonical ANWG / safety | 0 failed evals, 0 duplicate `(scenario_id, policy)` pairs, 0 NaN/Inf, leakage guard passes on all 72 scenarios |
| G10 | Evidence for within-scenario (not merely cross-scenario) opportunity | G5 holds **and** the same scenario-factor cell (same `bulk_pressure`/`urgent_tightness`) shows a different winner at different `urgent_arrival_phase` for at least one seed (i.e., trajectory timing alone, not just scenario identity, flips the practical winner) |

`KV_FAMILY_COMPOSITION_READY` requires **all** of G1-G10.
`KV_FAMILY_USEFUL_NEEDS_REFINEMENT` if G1/G4/G7/G8/G9 pass (mechanism is
real and non-degenerate) but G2, G3, G5, G6, or G10 fails (statistical
cleanliness or within-scenario robustness not yet fully established).
`KV_FAMILY_NOT_USEFUL` if G1, G8, or G9 fails (no real or safe separation).

## 10. Explicitly out of scope for this task

No selector fitting, no child policy, no composition falsification. No
MAP-Elites/GP/symbolic distillation/LLM synthesis. No changes to
`kv_constrained_online`/`least_laxity_first` algorithms or reserve
semantics. No prefix-reuse mechanism (still unsupported by the simulator).
v1's frozen run/design/audit are untouched.
