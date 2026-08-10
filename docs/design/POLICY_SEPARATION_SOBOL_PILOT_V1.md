# Policy Separation Sobol Pilot v1 -- Design

**Date:** 2026-08-10
**Status:** Design, implementation, tests, and a local dry-run smoke are
complete. **The scientific sweep has NOT been executed or submitted to
Slurm.** This document exists so that decision can be made explicitly,
separately from this design work.

Relationship to prior art: `docs/design/POLICY_SEPARATION_DATASET_V1.md`
(untracked, pre-existing, 2026-08-09) sketched a 7-stage roadmap
(theory-grounded manual cases -> Sobol -> MAP-Elites -> boundary
refinement -> module interventions -> selector retraining -> symbolic
composition) and a 25-template/5-family Phase 1 corpus. That corpus was
**never actually built** -- no `templates_fcfs.py`/`templates_sjf.py`/
`templates_edf.py`/`templates_fairness.py`/`templates_cache.py`/
`perturbations.py`/`metrics.py` exist in this repository, and the document
itself was never committed to git history. What was actually implemented
and run instead is a narrower three-mechanism diagnostic
(`templates_three_case.py`, job 1170116) followed by a boundary refinement
of exactly those three mechanisms (`templates_boundary_refinement.py`, job
1171116). This design salvages that document's staged-roadmap *concept*
(manual cases -> Sobol -> QD) and its module-intervention schema hooks
(already present, unused, in `schema.py`), but treats its "Phase 1
implemented" and 5-family scope claims as aspirational/superseded, not
current fact -- see `docs/audits/policy_separation_boundary_refinement_v1_20260810.md`
for what is actually validated.

## 1. Purpose (not to maximize separation)

Per the task that requested this design, the purpose of this pilot is
**landscape characterization**, not policy-separation maximization:

- A. Do policy winners vary smoothly/nontrivially across the validated
  parameter space, or is one policy dominant everywhere?
- B. Does oracle headroom persist away from job 1171116's hand-picked
  stress points, or was it an artifact of cherry-picked cells?
- C. Does the space contain multiple policy footprints (several policies
  each win somewhere) rather than a single dominant policy?
- D. Are there regions plausibly worth a later QD/MAP-Elites search?
- E. Does space-filling sampling expose interactions the one-factor-at-a-time
  grids in jobs 1170116/1171116 could have missed?

## 2. Two independent Sobol subspaces, not one hybrid space

`target_utilization`/`inversion_fraction` (Study B's mechanism: heterogeneous
job sizes + a controllable size-prediction error) and `overload_factor`/
`fraction_impossible` (Study C's mechanism: deadline pressure + a fraction of
genuinely unsalvageable jobs) come from **structurally different request
generators** (`case2_prediction_inversion_boundary` vs `case3_edf_overload`).
There is no way to vary all four in one scenario without either (a) building
a third, unvalidated hybrid generator, or (b) silently ignoring two of the
four dimensions per scenario. Per the task's explicit instruction against
"nonsense hybrid workloads," this design uses two separate Sobol subspaces
instead:

- **Family B -- prediction-sensitive scheduling**: Sobol-sampled
  `(target_utilization, inversion_fraction)` in `[0.50,1.10] x [0.00,1.00]`,
  crossed with a categorical `heterogeneity in {moderate, strong}` (job
  1171116 showed this categorically gates *whether* a crossing exists at
  all, not just its magnitude -- not a candidate for continuous sampling).
  Reuses `case2_prediction_inversion_boundary` unmodified.
- **Family C -- deadline/admission scheduling**: Sobol-sampled
  `(overload_factor, fraction_impossible)` in `[0.85,1.40] x [0.00,0.80]`.
  Reuses `case3_edf_overload` unmodified, `role="stress"` only (the
  loosened-deadline control mechanism is already validated -- job 1171116
  found *exactly* 0.0 margin in all 240 control cells -- so re-sampling
  controls here would not add landscape information). No third "slack"
  dimension: `overload_factor` already parameterizes deadline slack
  directly inside `case3_edf_overload` (`window_s` is inversely
  proportional to it), so an independent slack axis would double-count the
  same mechanism -- the same design decision job 1171116 already made and
  documented for its own grid.
- **FCFS categorical add-on (NOT Sobol)**: job 1171116 proved
  `arrival_offset` is a discontinuous mechanism switch under
  `max_active_sequences=1` (exact +0.348 separation at offset=0.0, exact
  0.0 at any offset>0, std=0.0 in both cases across every seed) and a
  persistently significant but continuous-looking effect under
  `max_active_sequences=4`. A space-filling sampler would waste nearly all
  its budget rediscovering the offset=0 cliff. Instead: **Template A1**
  (`max_active_sequences=1`, `offset=0.0`, the proven genuine-choice
  regime) over a small `(ratio, n_short)` grid and both roles; **Template
  A2** (`max_active_sequences=4`, two representative positive offsets, the
  proven general-convoy regime) at one representative `(ratio, n_short)`
  cell and both roles. Reuses `case1_fcfs_convoy` unmodified.

Both continuous subspaces use `scipy.stats.qmc.Sobol` with a fixed,
documented integer scramble seed (Family B: 20260810; Family C: 20260812 --
deliberately different so the two subspaces' sequences are not correlated
by sharing a seed), verified deterministic (same seed -> byte-identical
points across repeated calls, 128/128 unique points at `m=7`).

## 3. Policy roster

| Family | Roster | Rationale |
|---|---|---|
| B (prediction-sensitive) | `fifo`, `estimated_service_time_first`, `shortest_output_first`, `aging_priority` | Matches job 1171116's Study B roster minus `weighted_shortest_processing`. |
| C (deadline/admission) | `fifo`, `edf`, `least_laxity_first`, `scorpio_style_slo_guard`, `admission_control` | Unchanged from job 1171116's Study C roster. |
| FCFS add-on | same as Family B | Same size-aware-vs-fifo mechanism as Family B. |

**Dropped: `weighted_shortest_processing`.** Job 1171116 found it produces
**exactly** identical ANWG to `estimated_service_time_first` in 2,820/2,820
(100%) of scenarios across both of that job's studies, causally explained:
`weighted_shortest_processing_score = predicted_service_proxy(req) / priority`,
and every template used sets `priority=1.0` for every request (the
`builders.req()` default, never overridden anywhere in this codebase's
Policy Separation templates) -- WSP's sort key collapses to exactly ESTF's
whenever priority is uniform. Running both would spend roughly 20% of this
pilot's Family B budget on a policy proven redundant here. If a future
corpus introduces heterogeneous request priority, WSP should be
reintroduced, since the equivalence is parameter-induced, not structural.

**Kept: `admission_control`**, despite being 94.0% (451/480) exactly
identical to `edf` in job 1171116. The residual 6% divergence is a real,
causally-explained finding (laxity-order vs deadline-order divergence when
predicted service time varies across requests -- see
`docs/audits/policy_separation_edf_admission_mechanism_20260810.md`), not
noise, and worth confirming outside the handcrafted grid.

## 4. Scale

| Family | Sobol points | Categorical | Seeds | Scenarios | Policies | Evaluations |
|---|---|---|---|---|---|---|
| B | 128 (`m=7`) | 2 (heterogeneity) | 4 | 1,024 | 4 | 4,096 |
| C | 128 (`m=7`) | 1 (role=stress only) | 4 | 512 | 5 | 2,560 |
| FCFS A1 | -- | 2 ratios x 3 n_short x 2 roles | 5 | 60 | 4 | 240 |
| FCFS A2 | -- | 2 offsets x 2 roles | 5 | 20 | 4 | 80 |
| **Total** | | | | **1,616** | | **6,976** |

Within the requested pilot envelope (1,000-1,800 scenarios; ~7,000-12,000
evaluations; far below the 50,000+ that would indicate scope creep or the
eventual 8K-12K "real" dataset).

**Expected runtime**: jobs 1170116 (4,040 evals / 68s execution) and
1171116 (15,660 evals / 297s execution) both ran at ~50-60 (scenario,
policy) tasks/s with 8 CPU workers on pure Python/numpy simulator code (no
GPU/vLLM/torch dependency). At the same throughput, 6,976 evaluations
should take roughly 6,976/55 ~= 127s (~2 minutes) of execution phase.

## 5. Seeding / replication metadata

4 seeds per Sobol point for both continuous families (5 for the cheaper
FCFS add-on) -- enough to estimate sign stability without over-spending
compute on replication depth at the expense of coordinate coverage, per
the task's explicit "4-6 seeds" guidance. Every scenario's `params` records
`sobol_index`, `sobol_scramble_seed`, `generator_family`, `generator_version`,
and `seed` (the per-replicate RNG seed) -- sufficient metadata to later
construct held-out splits by any of: Sobol index block, scramble seed,
generator family, or categorical combination (see section 7).

## 6. Validity guards

Both Sobol coordinate ranges were chosen (from job 1171116's own
calibration) to lie entirely within the domain where the reused,
already-tested generator functions (`case2_prediction_inversion_boundary`,
`case3_edf_overload`) produce valid output at *every* point in the unit
hypercube -- e.g. `target_utilization` is bounded away from 0 (`>= 0.50`,
never a division-by-zero risk), `fraction_impossible in [0, 0.8]` maps to
`n_impossible = round(fraction_impossible * 30) in [0, 24]` via
`case3_edf_overload`'s own existing deterministic rounding (not a new
repair mechanism -- the existing generator already handles fractional
inputs this way, reused as-is). `sobol_pilot.validate_scenario()` is
implemented as the required explicit safety net (checks: non-negative
arrivals, positive prompt/output token counts, `slo_deadline >= arrival_time`,
positive GPU capacity fields, no generator-only field name colliding with
a `Request` attribute) and was run over the full dry-run corpus with zero
violations. Because validity is guaranteed by the deliberate range choice
rather than by a repair/rejection branch, no Sobol points have been
discarded or need discarding -- coverage is exactly what the raw scrambled
sequence produced.

## 7. Anti-leakage field classification (section 13)

| Class | Fields | Notes |
|---|---|---|
| **Generator-only / oracle** (`sobol_pilot.GENERATOR_ONLY_FIELDS`) | `sobol_index`, `sobol_scramble_seed`, `generator_family`, `generator_version`, `template_name`, `pair_id`, `seed`, `heterogeneity`, `target_utilization`, `overload_factor`, `fraction_impossible`, `rank_agreement_kendall_tau`, `rank_agreement_spearman`, `window_s`, `n_normal`, `n_impossible`, `total_required_service_s`, `role`, `inversion_fraction`, `ratio`, `n_short`, `offset`, `max_active_sequences` | Live only in `scenario.params` / `scenarios.jsonl` / `scenario_features.csv`. A future selector must never read these as input features -- several (`rank_agreement_*`) are computed from `actual_output_tokens`, which no online policy observes; the rest encode generator identity or design labels, not measurable system state. |
| **Policy-visible** (`sobol_pilot.POLICY_VISIBLE_FIELDS`) | `request_id`, `arrival_time`, `prompt_tokens`, `predicted_output_tokens`, `slo_deadline`, `priority`, `class_id`, `gpu_id`, `max_active_sequences` (as a `GPUConfig` field, not the scenario-param `max_active_sequences` used by the FCFS categorical add-on's mas>1 condition -- same name, two different objects, never merged into one struct), `max_batch_tokens`, `max_kv_tokens` | Only ever populated on `Request`/`ObservableRequest`/`GPUConfig`. Enforced structurally by reusing the existing template functions unmodified -- this module never constructs a `Request` itself. |
| **Documented future selector-safe proxies** (not currently emitted; `sobol_pilot.DEPLOYMENT_ESTIMABLE_PROXIES`) | `queue_length`, `realized_arrival_rate`, `current_active_sequences`, `kv_occupancy_fraction`, `predicted_service_time_distribution_summary`, `recent_prediction_residual_stats`, `slo_slack_distribution_summary`, `recent_drop_or_violation_rate` | A future selector should be trained on proxies like these (computable online from simulator/deployment state), never on the generator-only column of the table above. |

`tests/test_policy_separation_sobol_pilot.py::test_no_generator_field_leaks_into_policy_visible_state`
asserts programmatically that no generator-only field name collides with a
`Request` attribute on any generated scenario.

## 8. Held-out split design (documented now, not built)

No selector is trained by this pilot. For a future selector to be evaluated
without leakage, splits should be constructable along at least:

- **Held-out Sobol blocks**: partition each family's 128 points by index
  range (e.g. first 96 for training, last 32 held out) so held-out points
  are genuinely un-sampled-from during training, not just unseen seeds of
  an already-trained-on coordinate.
- **Held-out scramble seeds**: re-run generation with a different
  `sobol_scramble_seed` entirely and hold out the whole resulting sequence,
  to test generalization to a differently-distributed (but same-range)
  sample.
- **Held-out generator families**: train on Family B, evaluate
  zero-shot on Family C (and vice versa) to test whether learned features
  transfer across mechanisms, not just across coordinates within one
  mechanism.
- **Held-out categorical combinations**: e.g. train on `heterogeneity=moderate`
  only, evaluate on `heterogeneity=strong` -- tests whether the selector
  learned the *load-dependent crossing* concept or merely memorized
  moderate-heterogeneity behavior.
- **Real-trace OOD evaluation**: the SwissAI reanalysis
  (`docs/audits/swissai_v2_policy_sweep_reanalysis_20260809.md`, 512
  real-trace-derived windows, oracle headroom ~=0) remains available as a
  genuinely out-of-family evaluation set once a selector exists.

## 9. Analysis plan (to run after the scientific job -- not run by this task)

Global diagnostics, computed per `generator_family` by
`scripts/run_policy_separation_sobol_pilot_v1.py`'s aggregation functions:
near-tie rate, inter-policy variance (via `family_summary.csv`'s std),
unique winners and winner entropy (`policy_winner_summary.csv`), oracle
headroom / best-fixed-vs-oracle gap (`oracle_headroom.csv`), pairwise
separation coverage (`pairwise_separation.csv`), and Sobol coordinate
coverage (`coverage_summary.csv`, a descriptive n_bins x n_bins occupancy
count -- not a statistical test). Family-B-specific: rank-agreement
(Kendall tau / Spearman, already recorded per scenario) vs. the
ESTF-vs-FIFO advantage surface, plus a crossing/boundary re-estimate the
same way job 1171116 did (linear interpolation, no complex fit). Family-C-
specific: SCORPIO-vs-EDF and admission_control-vs-EDF margin surfaces,
SLO-violation/completion trade-off. No arbitrary pass/fail thresholds are
pre-specified as scientific truth; QD-readiness is a descriptive judgment
call based on whether winner entropy and oracle headroom are non-flat
across the sampled space, matching this document's section 1 purpose.

## 10. Readiness

- `SOBOL_DESIGN_READY_FOR_REVIEW`: YES.
- `READY_TO_SBATCH_AFTER_REVIEW`: YES, mechanically (script, config, and
  sbatch wrapper are implemented and smoke-tested) -- but **submission is
  a distinct future action requiring explicit authorization**, not implied
  by this document.
- `MAP_ELITES_STARTED`: NO.
- `SELECTOR_RETRAINING_STARTED`: NO.
