# Family C v1 — KV-Pressure Reserve Pairwise-Separation Pilot — Audit

**Date:** 2026-08-17
**Verdict:** `KV_FAMILY_USEFUL_NEEDS_REFINEMENT`
**Run:** [`experiments/kv_pressure_pilot_v1_20260817T162650Z/`](../../experiments/kv_pressure_pilot_v1_20260817T162650Z/)
**Design:** [`docs/design/POLICY_SEPARATION_FAMILY_KV_PRESSURE_V1.md`](../design/POLICY_SEPARATION_FAMILY_KV_PRESSURE_V1.md)
**Parents:** `kv_constrained_online` (soft KV-reserve admission gate) vs `least_laxity_first` (KV-blind laxity-greedy admission)
**Primary metric:** canonical `arrival_normalized_weighted_goodput` (ANWG)

This is a **pairwise-separation pilot**, not a composition experiment. No
selector was fit, no child policy was built or run. Per explicit task scope,
composition work on this pair was not started in this task.

## 1. Why this family, not a repeat of ESTF/WFS or PrefillControl v2

Both prior composition falsifications (ESTF↔WFS, Family B v2 PrefillControl)
landed `SELECTION_SUFFICIENT_FOR_THIS_PAIR`: their parent mechanisms differed
by a value fixed for the whole scenario, so a scenario-level fitted top-1
selector could already match the two-parent oracle exactly. This family was
deliberately selected (see design doc §2-3, from a repository capability
audit — `kv_constrained_online`/`least_laxity_first` were the only
already-implemented, single-mechanism-difference, zero-new-simulator-work
candidate) because its mechanism difference (a KV-occupancy admission
reserve) is a **live, time-varying gate** driven by `current_kv_tokens`,
which genuinely rises and falls within one scenario's trajectory — giving
this family, uniquely among the three studied so far, a testable
within-scenario timing hypothesis (H3/gate G4).

## 2. Run integrity

| Check | Result |
|---|---|
| Scenarios | 32 (2×2×2×4 grid: `bulk_pressure` × `urgent_arrival_phase` × `urgent_tightness` × 4 seeds) |
| Evaluations | 64 (32 × 2 policies), 0 failed |
| Duplicate `(scenario_id, policy)` pairs | 0 |
| Non-finite (NaN/Inf) primary-metric values | 0 |
| Leakage guard | `assert_policy_visible_fields_clean_kv_v1` passed on all 32 scenarios (raises during generation otherwise — 0 exceptions) |
| BurstGPT | not staged in this environment; `allow_synthetic_tokens=True`, prompt-length provenance recorded per-scenario (`burstgpt_anchored`/`synthetic_lognormal`, see `scenario.params`) |
| Elapsed | <1s (local, 8 workers) |

## 3. Calibration (pre-registered, before any full-pilot cell was scored)

Three rounds, all workload-*generation*/*scale* adjustments — no policy
code, ranking logic, or gate definition touched (full detail: design doc §5):

1. **Urgency binding:** original targets (`max_kv_tokens=24000`, tight
   slack=0.9s) produced large KV-occupancy differences but zero outcome
   difference (slack fully absorbed all observed admission delays).
2. **Bidirectionality:** at the Round-1 fix, `kv_constrained_online` never
   lost a single cell across a full 32-cell check — an artificial
   one-sided result because deferring a cost-free (30s-slack) bulk tenant
   is free. Diagnosed via the design's own bidirectionality gate (G1) /
   smoke-gate criterion, not by preference for either policy's win rate.
3. **Infeasible-request bug:** the original bulk prompt window `[2048,
   8192]` could sample a single request whose `prompt_tokens` alone
   exceeded `max_kv_tokens`, making it permanently unadmittable and
   inflating scenario step counts by ~25×. A workload-generation
   correctness fix, not a calibration choice — capped to `[1024, 3072]`.

Final calibrated parameters: `max_kv_tokens=6000`, `BULK_SLACK_S=1.5s`,
`urgent_tightness=tight` slack=0.55s, bulk prompt window `[1024, 3072]`.
Smoke gate (`configs/kv_pressure_smoke_v1.yaml`, 8 cells) passed at these
values before the full pilot was run (design doc §9.1).

## 4. Gate results (full 32-scenario pilot)

| Gate | Test | Result |
|---|---|---|
| G1 | Bidirectional wins (ε=0.01) | **PASS** — `kv_constrained_online` wins 9/32, `least_laxity_first` wins 4/32, 19/32 ties |
| G2 | Near-tie rate < 50% | **FAIL** — 59.4% (19/32) |
| G3 | Mechanism activates (≥1 logged reserve-deferral event) | **PASS** — 28,695 total deferral events logged across `kv_constrained_online`'s 32 cells |
| G4 | **Within-scenario/trajectory evidence** — KV-constrained's `late`-phase advantage over `early`-phase, matched on `(bulk_pressure=high, urgent_tightness=tight)` | **PASS** — mean ANWG delta `late`=+0.125 vs `early`=+0.0625 (2×) |
| G5 | No twin | **PASS** — 13/32 cells show a real (non-tied) ANWG difference; not byte-identical everywhere |
| G6 | Integrity | **PASS** — see §2 |

5/6 gates pass. `KV_FAMILY_COMPOSITION_READY` requires **all** gates
(design doc §8), so the literal verdict is `KV_FAMILY_USEFUL_NEEDS_REFINEMENT`
— real, mechanistically-caused, bidirectional, within-scenario-varying
separation exists, but the pilot's own preregistered statistical-cleanliness
bound (G2) was not met at this scenario count/calibration.

## 5. Full breakdown by factor cell (mean `kv_constrained − llf` ANWG, n=4 seeds/cell)

| `bulk_pressure` | `urgent_tightness` | `urgent_arrival_phase` | Mean Δ |
|---|---|---|---:|
| low | loose | early | 0.0000 |
| low | loose | late | 0.0000 |
| low | tight | early | 0.0208 |
| low | tight | late | 0.0625 |
| high | loose | early | −0.0250 |
| high | loose | late | −0.0250 |
| high | tight | early | 0.0625 |
| high | tight | late | **0.1250** |

Reading this table directly against the preregistered hypotheses:

- **H4 (placebo control) confirmed:** `urgent_tightness=loose` cells show a
  small, phase-*independent* effect (−0.025 in both `early` and `late`) —
  when urgent tenants aren't actually deadline-critical, admission timing
  doesn't matter, exactly as designed. (The small negative sign here is a
  genuine minor bulk-tenant cost of the reserve under loose urgent slack —
  discussed in §7.)
- **H3 (within-scenario timing) confirmed, strongest where designed:** the
  `late > early` gap widens specifically in `urgent_tightness=tight` cells
  (0.0208→0.0625 at `bulk_pressure=low`; 0.0625→0.1250 at `high`) and the
  single largest effect in the entire pilot is exactly the cell the
  hypothesis targeted: `bulk_pressure=high, urgent_tightness=tight,
  urgent_arrival_phase=late`.

## 6. Mechanism check (not just scalar outcomes)

`kv_constrained_online` logged 28,695 admission-deferral-due-to-reserve
events across its 32 cells (mean ≈897/cell) — the gate genuinely activates,
frequently, exactly as its code path predicts (`_admit_filter`: defer
non-urgent admission once projected utilization exceeds 0.82).
`least_laxity_first` has no equivalent code path at all (`hasattr` check in
`tests/test_policy_separation_kv_pressure_v1.py::TestInstrumentedPolicy::
test_least_laxity_first_has_no_deferral_concept`) — the mechanism
difference is structural, not incidental. `kv_constrained_online` never
exceeds hard KV capacity at admission time (the reserve is an *additional*
soft gate on top of the shared hard `_feasible_on_gpu` check both policies
respect) — confirmed by the absence of any admission-rejection warning
across all 32 `kv_constrained_online` runs.

## 7. Honest caveats

- **G2 (tie rate) did not pass.** 59.4% near-ties is meaningfully worse than
  Family B v2's 3.1% (though far better than Family B v1's 96%). This is
  most likely because admission-timing effects are more binary (a request
  either crosses its deadline or doesn't) than the continuous chunk-size
  effects Family B v2 studied — a genuine property of this mechanism family
  worth noting, not necessarily fixable by more calibration alone.
- **`urgent_tightness=loose` shows a small (−0.025) cost, not neutrality,**
  for `kv_constrained_online` under `bulk_pressure=high`. The reserve
  slightly delays some bulk-tenant admissions relative to LLF's greedy
  packing, at the (small, `BULK_SLACK_S=1.5s`-bounded) cost of occasional
  bulk-tenant SLO misses that LLF avoids by never deferring anything. This
  is the reserve's genuine trade-off cost, not noise — see design doc §3's
  "bidirectional mechanism plausibility" claim, now empirically confirmed
  rather than asserted.
- **This is a 4-seed-per-cell pilot at a small (20-request) scenario
  scale** — matching Family B v2's precedent scale, but not a
  large-n statistical study. G4's 2× effect is clear directionally but not
  bootstrap-CI-quantified in this pilot (that level of rigor belongs to a
  future composition falsification, not this pairwise-separation pilot).

## 8. Tests / checkers

- `python3 -m pytest tests/test_policy_separation_kv_pressure_v1.py -q`: 20 passed.
- `python3 -m pytest tests/ -k "kv_constrained or least_laxity or policy_separation_kv or builders" -q`: 46 passed.
- `python3 scripts/check_project_handoff_consistency.py`: passed.
- No existing simulator-core, policy-library, or other family's frozen
  artifacts were modified (all new files; `git status` confirms zero
  modifications to tracked files this task).

## 9. Final preregistered verdict

**`KV_FAMILY_USEFUL_NEEDS_REFINEMENT`.**

This is the first of the three studied families to demonstrate genuine
**within-scenario** mechanism opportunity (gate G4) rather than a purely
scenario-level contrast — the structural precondition the prior two
families' `SELECTION_SUFFICIENT_FOR_THIS_PAIR` results lacked. It is not yet
`KV_FAMILY_COMPOSITION_READY` because gate G2 (tie-rate) did not clear its
preregistered bound.

## 10. Exact next action

Per instruction, **no composition work starts from this result.** The
family itself needs refinement before a composition falsification would be
well-motivated:

1. **Refine, don't abandon:** the tie-rate gap (G2) is the only failing
   gate; a larger pilot (more seeds, or a wider factor range around the
   `bulk_pressure=high, urgent_tightness=tight` cell that already shows the
   cleanest signal) is the natural next step to test whether G2 clears with
   more statistical power, before concluding this family needs a different
   mechanism pair entirely.
2. Do not start MAP-Elites, GP, symbolic distillation, or LLM-guided
   synthesis — none of that is justified by a pairwise-separation pilot
   regardless of verdict.
3. Do not start a composition falsification for this pair yet — that is
   only warranted after a `KV_FAMILY_COMPOSITION_READY` verdict, per this
   family's own design doc (§1) and the same discipline applied to Family B
   v1→v2 (refine, then re-gate, before falsifying).
