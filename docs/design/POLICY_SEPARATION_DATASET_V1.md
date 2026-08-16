# Policy Separation Dataset v1 — Design Document

**Original draft date:** 2026-08-09  
**Document role:** Design / roadmap reference (retained for provenance)  
**Status (reconciled 2026-08-16):** **PARTIAL — do not treat the 2026-08-09
“Phase 1 implemented” wording below as current fact.**

## Current status (authoritative; supersedes claims later in this file)

| Stage | Status | Evidence |
|---|---|---|
| Design intent (5 classical families, stress/control, anti-leakage schema) | **DESIGNED** | This document |
| Schema + builders (`schema.py`, `builders.py`) | **IMPLEMENTED** | `src/llmserveopt/policy_separation/` |
| Manual theory-grounded diagnostics (3 cases) | **COMPLETE** | Jobs 1170116; audits under `docs/audits/policy_separation_three_case_*` |
| Boundary refinement | **COMPLETE** | Job 1171116; audits under `docs/audits/policy_separation_boundary_refinement_*` |
| Sobol / space-filling pilot (prediction + deadline families) | **COMPLETE + ANALYZED** | Job 1182183; `docs/audits/policy_separation_sobol_pilot_v1_20260816.md` |
| Family A fairness / weight-skew / aging generator + 120×4 pilot | **EXECUTED; ANALYSIS PENDING** | Job 1182306; provenance `experiments/policy_separation_fairness_starvation_pilot_20260816T211029Z_1182306/` |
| Full 5-family × 5-template (25) manual corpus named in §5 | **NOT BUILT** as specified | Only selected templates/families exist; no `templates_fcfs.py` / `templates_cache.py` pack matching §5 |
| MAP-Elites / QD, selector retraining, module interventions on PSD | **NOT STARTED / NOT JUSTIFIED YET** | Waiting on Family A scientific analysis and remaining mechanism coverage |
| Typed DSL / module composition elsewhere in the repo | **EXISTS (separate thread)** | Does **not** by itself solve policy-separation decision-boundary characterization |

**Job 1182306 caveats (frozen historical pilot):** synthetic token-length
fallback (BurstGPT filename miss); CSV column `anwg` is unweighted SLO-success,
not canonical `RunMetrics.arrival_normalized_weighted_goodput`. See the
experiment README under `experiments/…_1182306/`.

**Next WS-P action:** scientific analysis of Job 1182306 (no MAP-Elites,
selector retrain, or new composition experiments until that analysis lands).

---

## Original design text (2026-08-09; historical)

The sections below preserve the original design narrative. Where they say
“implemented” for the full Phase-1 25-template corpus, treat that as
**aspirational at time of writing** and superseded by the status table above.

# Policy Separation Dataset v1 — Design & Phase 1 Scope

**Date:** 2026-08-09
**Historical status line (superseded):** Phase 1 (manual theory-grounded cases +
small deterministic perturbation smoke) *planned as* implemented. Sobol /
MAP-Elites / module-intervention stages were explicitly not started by this
document's original implementation claim.

## 1. Purpose

Every existing scenario-generation subsystem in this repository
(`workloads/synthetic.py`, `selector/dataset_v2/scenario_families.py`,
`selector/dataset_v2/frontier_workload_families.py`,
`scripts/stress_tests/generators.py`) is built to either (a) match a
plausible real-world distribution, or (b) hunt adversarially for *any*
scenario that makes policies disagree, without regard to *why*. Neither
one is designed to answer a narrower question this project now needs
answered before further Selector/CC-line investment: **for each of the
five classical scheduling-policy families this repo implements, does the
simulator reproduce the specific, textbook-predicted qualitative
behavior — including its strengths, its known weaknesses, and the
control cases where that behavior should *disappear*?**

The Policy Separation Dataset (PSD) is intentionally **discriminative by
construction, not by search**. Every scenario is hand-derived from a
specific scheduling-theory claim ("FCFS suffers convoy effect when one
huge job blocks many short ones", "EDF degrades under deadline
homogeneity", "cache-aware policies gain nothing without reuse", ...),
paired with a control that removes the one mechanism under test, and
tagged with an explicit falsifiable hypothesis before it is ever run.
This is what makes it useful for the next roadmap stages: a MAP-Elites
or Sobol search over an unvalidated simulator would happily report
"policy X wins here" without anyone knowing whether that reflects a real
scheduling phenomenon or a simulator quirk. PSD v1 is the gate that
must pass first.

## 2. Staged roadmap

```
 1. theory-grounded manual cases              <- PARTIAL (diagnostics done; full 25-template pack NOT built)
        |  (+ small deterministic perturbation grid)
        v
 2. Sobol / low-discrepancy exploration        <- COMPLETE (Job 1182183; analyzed)
        v
 3. Family A fairness / starvation pilot       <- EXECUTED (Job 1182306); ANALYSIS PENDING
        v
 4. MAP-Elites / QD search                     <- NOT STARTED (not yet justified)
        v
 5. winner-boundary refinement                 <- PARTIAL (earlier grid jobs; not post-Family-A)
        v
 6. module interventions                       <- schema hooks only; no PSD experiment run
        v
 7. selector retraining                        <- NOT STARTED
        v
 8. symbolic policy composition                <- NOT STARTED on PSD path
```

Only stage 1 (manual templates) plus a **small** deterministic
perturbation smoke around each manual template is implemented and
validated by this change. See §11 ("Scale-up gate") for the explicit
go/no-go decision on stages 2+.

## 3. Relationship to existing infrastructure (do not duplicate)

Reused as-is, unmodified:

| Need | Reused from |
|---|---|
| Request / observable-request split | `core.types.Request`, `core.types.ObservableRequest` (no new fields added to either — see §6 for why) |
| GPU / cache config | `core.types.GPUConfig`, including its existing Apt-Serve dual-tier fields |
| Simulation execution | `evaluation.run_policy.run_policy` (thin wrapper already used by every other evaluation pathway in this repo) |
| Policy instantiation | `policies.registry.make_policy` |
| Primary metric | `core.metrics.RunMetrics.arrival_normalized_weighted_goodput` (ANWG) |
| Discriminativeness thresholds | `selector.dataset_v2.discriminativeness` practical/strong margin constants (imported, not re-derived, so PSD's "tie" language matches Selector Dataset v2's) |

**New** (this change), under `src/llmserveopt/policy_separation/`:

| Module | Why it doesn't already exist |
|---|---|
| `schema.py` (`PolicySeparationScenario`) | Existing scenario schemas (`ScenarioIdentifiers`/`WindowRecordV2` in Selector Dataset v2, catalog entries in `algorithm_stress_test_catalog.yaml`) have no fields for a falsifiable qualitative hypothesis, a stress/control pair id, or module-intervention hooks — all required by this task. Extends the *pattern* (dataclass + `to_manifest_dict`) rather than inventing a new one. |
| `templates_*.py` | The stress-test catalog already has per-*algorithm* TARGET/COUNTER pairs (§8 of the infra survey below), but none of its entries are built around a *falsifiable hypothesis with an expected qualitative direction* the way PSD requires, and several PSD templates (prediction inversion, cyclic cache thrashing, tenant aging inversion) have no equivalent anywhere in the repo. |
| `metrics.py` | `discriminativeness.py` computes winner/margin/tie-set for *one* objective at a time and does not compute niche/counterexample scores, top-two margin, MAD, multi-epsilon tie counts, or stress-control effect size — all explicitly required outputs of this task. It reuses `discriminativeness.py`'s threshold constants rather than re-deriving them. |
| `perturbations.py` | No Sobol/perturbation-grid utility exists anywhere in the repository (confirmed by grep before writing this). |

## 4. The five policy families

| Family | Registry name(s) used | Fidelity (per `docs/current/BASELINES.md`) | Notes |
|---|---|---|---|
| A. FCFS / FIFO | `fifo` | historical | Pure arrival-order baseline. |
| B. SJF / size-aware | `shortest_output_first` (primary), `estimated_service_time_first` (prediction-based variant) | historical; `estimated_service_time_first` also tagged "style/proxy" (PARS-inspired, not a PARS reproduction) | `shortest_output_first` uses the hidden-from-policy notion of "predicted" length via `predicted_output_tokens`, which is itself the online-visible field — no oracle leakage. |
| C. EDF / SLO-aware | `edf` (primary), `least_laxity_first`, `scorpio_style_slo_guard` (admission+guard style, "SCORPIO-inspired", not official) | historical / style-proxy | `scorpio_style_slo_guard` used specifically for the overload template, since plain EDF has no admission control. |
| D. fairness / aging | `aging_priority`, `weighted_fair_share` | Policy Library v2 additions | **Known limitation, disclosed up front:** the simulator has no `tenant_id` field on `Request` (confirmed by inspection of `core/types.py`; also documented in `policies/weighted_fair_share.py` and `baselines/vtc/fairness_workloads.py`). PSD v1 follows the repo's existing convention of carrying tenant identity via `Request.class_id` (e.g. `"tenant_a"`, `"tenant_b"`), exactly as `baselines/vtc/fairness_workloads.py` already does. This is disclosed, not hidden, in every fairness scenario's `params`. |
| E. cache / KV-aware | `kv_constrained_online` (KV-pressure admission), `apt_serve_faithful` (dual-tier hidden-cache transition cost) | Policy Library v2 / faithful external baseline | **Known limitation, disclosed up front:** the simulator has *no* prefix-sharing / prompt-reuse cost-reduction mechanism at all — `Request` carries no prefix/reuse-group field, and no policy reads one. `kv_constrained_online` and `apt_serve_faithful` are the only two policies with any cache-capacity-aware mechanism, and it is capacity/eviction/transition-cost-based, not reuse-based. PSD v1's "reuse" templates (§5E) are therefore built and evaluated, but their reuse-specific hypothesis is expected to classify `UNSUPPORTED_BY_SIMULATOR` (§9) — they are retained because (a) they still exercise genuine KV-capacity and working-set-size effects, and (b) a future module-intervention stage may add a reuse mechanism, at which point these templates become immediately usable without re-authoring. |

## 5. Manual templates (Phase 1 corpus)

Each family has 5 templates: one primary stress case, one or more
matched controls, and supplementary variants. Every template is
implemented as a pure function in `src/llmserveopt/policy_separation/templates_<family>.py`
returning a `PolicySeparationScenario`. Full per-template hypotheses and
observed classifications are in §9 and in the smoke run's
`hypothesis_validation.csv` — this section lists names and the
mechanism under test only.

- **A. FCFS/FIFO** (`templates_fcfs.py`): `convoy_long_first`,
  `convoy_short_first_control`, `homogeneous_length_control`,
  `long_prefill_first`, `kv_heavy_first`.
- **B. SJF/estimated-service** (`templates_sjf.py`):
  `accurate_length_heterogeneous_positive`, `long_job_starvation`,
  `prediction_inversion`, `fairness_conflict`, `homogeneous_control`.
- **C. EDF/LLF/SLO-aware** (`templates_edf.py`):
  `feasible_deadline_heterogeneity`, `impossible_urgent_overload`,
  `equal_deadline_control`, `mixed_tight_loose_slo`,
  `deadline_prediction_error`.
- **D. fairness/aging/VTC-like** (`templates_fairness.py`):
  `tenant_flooding`, `starvation_rescue`,
  `single_tenant_light_load_control`, `aging_inversion`,
  `weighted_tenant_skew`.
- **E. cache/KV-aware** (`templates_cache.py`):
  `high_reuse_working_set_fit`, `cyclic_working_set_over_capacity`,
  `low_reuse_control`, `high_kv_pressure_low_reuse`,
  `high_transition_cost_counterexample`.

25 manual templates total (5 families x 5).

## 6. Scenario schema (no oracle/future leakage)

`PolicySeparationScenario` (frozen dataclass, `schema.py`) fields:

- Identity/provenance: `scenario_id` (deterministic, e.g.
  `"fcfs.convoy_long_first.v1.s0"`), `family`, `template_name`,
  `generator_version`, `seed`, `params` (the exact generation kwargs —
  sufficient to regenerate the scenario byte-for-byte, so raw `Request`
  lists never need to be serialized to disk).
- Payload: `requests: Tuple[Request, ...]`, `gpu_configs: Tuple[GPUConfig, ...]`,
  `service_model_kwargs: Dict[str, Any]` (kwargs, not an object, so the
  scenario stays trivially comparable/hashable for dedup checks).
- Hypothesis metadata: `target_policy_family`, `target_mechanism`,
  `expected_qualitative_hypothesis` (free text, falsifiable), `stress_control_relationship`
  (`None | "stress" | "control"`).
- Pairing: `pair_id` (shared between a stress case and its control(s)),
  `changed_parameters` (tuple of param names that differ from the paired
  counterpart — required whenever `pair_id` is set).
- Module-intervention hooks (§7 of the task; unused this phase, always
  `None` in Phase 1 output — present so Phase 5 doesn't require a schema
  migration): `target_module`, `module_enabled`, `intervention_parent_policy`,
  `intervention_pair_id`, `expected_module_effect`.

No field derives from `Request.actual_output_tokens` when constructing
anything a policy can see (`slo_deadline`, `predicted_output_tokens`,
arrival order). `prediction_inversion` and `deadline_prediction_error`
deliberately set `predicted_output_tokens` far from `actual_output_tokens`
— this is the intended *mechanism under test*, not leakage, since
policies still only ever read `predicted_output_tokens` via
`ObservableRequest`. `tests/test_policy_separation_v1_schema.py`
enforces this with a structural check (no template constructs
`slo_deadline` or `predicted_output_tokens` as a function of
`actual_output_tokens`).

## 7. Module-intervention hooks (schema only, not exercised)

The five fields listed in §6 exist so that a future module-intervention
experiment (swapping one `GenomeModule` — admission/priority/prefill/
KV-guard/fairness rule, per `docs/current/scheduler_genome_v1.schema.json`
— on a fixed base policy) can attach directly to a PSD scenario without a
schema change. Candidate modules for that future stage: admission
control, aging, SLO guard, chunked prefill, preemption, KV tiering,
cache-aware admission (matching `composition_operators.json`'s module
list). **No module-intervention experiment is run by this change.**

## 8. Stress/control pairing

Every stress template lists a `pair_id` shared with its control(s) and a
`changed_parameters` tuple naming exactly which generation parameters
differ. Examples: `convoy_long_first` / `convoy_short_first_control`
share `pair_id="fcfs_convoy"` and differ only in `arrival_order`;
`feasible_deadline_heterogeneity` / `equal_deadline_control` share
`pair_id="edf_deadline_heterogeneity"` and differ only in
`deadline_spread`. `stress_control_pairs.csv` in the smoke output
records, per pair, the ANWG margin under stress vs. under control for
every policy (§10 "stress-control effect size").

## 9. Hypothesis validation protocol

Every template's `expected_qualitative_hypothesis` is checked against
the smoke run's actual per-policy ANWG values and classified into one of:
`CONFIRMED`, `PARTIALLY_CONFIRMED`, `AMBIGUOUS`, `CONTRADICTED`,
`UNSUPPORTED_BY_SIMULATOR`. When a hypothesis is not `CONFIRMED`, the
diagnosis (never a simulator-semantics change to force the expected
result) is recorded against one of: (A) template parameterization too
weak, (B) actual implemented policy differs from the textbook version,
(C) ANWG masks the effect (visible in a secondary metric instead), (D)
secondary metric shows it, (E) simulator lacks the needed mechanism, (F)
LLM batching changes the classical expectation, (G) cache/memory
constraints dominate, (H) bug. Full per-template results are in
`docs/audits/policy_separation_v1_smoke_<timestamp>.md` (written by the
smoke runner) and `hypothesis_validation.csv`.

## 10. Separation metrics (`metrics.py`)

Computed per scenario over the set of evaluated policies' ANWG values
`R_i`: signed pairwise advantage `R_i - R_j`, absolute pairwise
separation `|R_i - R_j|`, positive niche `R_i - max_{j!=i} R_j`,
counterexample score `max_{j!=i} R_j - R_i`, top-two margin (best vs.
second-best overall), inter-policy variance, median absolute deviation,
tie counts at epsilon in `{0, 0.001, 0.005, 0.01, 0.05}` (count of
policies within epsilon of the max), unique-winner indicator, full
ranking, per-family winner, and (for paired scenarios) stress-control
effect size per policy. Behavioral-equivalence classification
(`CLEARLY_SEPARABLE` / `WEAKLY_SEPARABLE` / `EFFECTIVELY_EQUIVALENT_ON_V1`
/ `INSUFFICIENT_EVIDENCE`) reuses `discriminativeness.py`'s margin
constants so PSD's notion of "tie" agrees with Selector Dataset v2's.
Raw variance is never used alone to define dataset quality, per the
task's explicit prohibition — it is one of many diagnostic fields.

## 11. Small perturbation grid (not Sobol)

For each template, `perturbations.py` sweeps a handful (2-4 values each,
one or two parameters at a time — never a full Cartesian product across
all of a template's parameters) of the specific parameters named in the
task spec (e.g. FCFS: long/short ratio, short-job count, arrival offset;
cache: reuse ratio, working-set/cache ratio, transition cost, KV
pressure). This produces on the order of 5-15 extra scenarios per
template — enough to check whether a manually-observed separation is
robust to nearby parameter choices, nowhere close to a space-filling
Sobol design. Total Phase 1 smoke corpus size is capped in the low
hundreds (`scripts/run_policy_separation_v1_smoke.py`).

## 12. Success criteria (relative, not absolute thresholds)

Per the task's explicit instruction, Phase 1 does not use arbitrary
universal thresholds (e.g. "tie rate <5%"). Success is instead assessed
by comparison against the discriminativeness distribution already
observed in this project's existing Selector Dataset v2 /
stress-test-catalog corpora (see `docs/selector_dataset_v2.md` §6,
`docs/research/algorithm_stress_tests/STRESS_TEST_CATALOG.md`) and by
the nine qualitative checks in the smoke run's `final_summary.json`
(deterministic and valid; several families separate clearly; controls
reduce margins; honest equivalence detection; cache/KV cases expose
LLM-specific behavior; no leakage; perturbations show some robust
effects; failures documented; infra ready for stage-2 search).
