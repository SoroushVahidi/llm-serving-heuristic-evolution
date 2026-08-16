# Work Status

Detailed current status table. The roadmap authority is
[`../PROJECT_MAP.md`](../PROJECT_MAP.md); this file is the operational status
companion. Historical audit reports remain authoritative only for the time they
were written.

Status vocabulary: `COMPLETE`, `COMPLETE_REGIME_SPECIFIC`,
`FOUNDATIONAL_CANDIDATE`, `EVALUATION_ONLY`, `IN_PROGRESS`,
`IMPLEMENTED_NEEDS_VALIDATION`, `NOT_STARTED`, `BLOCKED`, `DEFERRED`,
`SUPERSEDED`.

| Workstream | Status | Evidence | Current Gap | Next Action |
|---|---|---|---|---|
| Contextual composition CC0-CC5 | `COMPLETE_REGIME_SPECIFIC` | CC5 operating-envelope evaluation and audits under `docs/audits/contextual_composition_cc5_*_20260803.md` | Regime-specific, not universal | Use as the current composition result until a scoped CC6 decision is made |
| CC6 dynamic adaptation | `NOT_STARTED` | Restricted scope exists in historical CC docs | Requires explicit authorization and updated design | Do not start by default |
| Internal scheduler/policy library | `COMPLETE` | Registry and Policy Library V2 tests | None immediate | Maintain as baseline/envelope input |
| Typed DSL / AST / verifier | `COMPLETE` | CC2/CC3 docs and tests | External faithful policies are not yet decomposed into reusable modules | Use Phase G as input to module-decomposition planning |
| Contextual performance learning | `IMPLEMENTED_NEEDS_VALIDATION` | Selector/CC5 lineages; ANWG objective correction | Generalization and discriminative-power limits remain | Feed module-envelope work, not broad retraining by default |
| Module decomposition / library-envelope tooling | `IN_PROGRESS` | Prior module-credit and composition-opportunity artifacts | No standing reusable `MC_i`/`MG_c` tool for arbitrary candidates | Canonical next task: post-Phase-G module-envelope interpretation/design |
| Sarathi-Serve | `COMPLETE` | Faithful reimplementation plus Wulver repeated-trial validation | Known simulator structural limit documented | None queued |
| VTC | `FOUNDATIONAL_CANDIDATE` scientific / `EVALUATION_ONLY` deployable | Adapter wraps official `VTCReqQueue`; fairness sweep complete | Native non-wrapped implementation needed before deployable registration | Deferred |
| Llumnix | `FOUNDATIONAL_CANDIDATE` | Faithful reimplementation, stress catalog, comparative evaluation, independent verification | Not registered as deployable foundational library element | None queued |
| DistServe | `FOUNDATIONAL_CANDIDATE_FOR_DISAGGREGATION_PRIMITIVES_ONLY` | Faithful reimplementation, stress entries, comparative evaluation | Useful mainly for disaggregated primitives | None queued |
| PARS-Serve-2026 / vLLM-LTR | `EVALUATION_ONLY` | Official-code/checkpoint validation and comparative evaluations | Dominated in tested regimes | None queued |
| Apt-Serve A-F | `COMPLETE` | Dual-tier cache, adapter, rollback, stress generators, SS15 fix provenance | Phase F alone was inconclusive | Superseded by completed Phase G |
| Apt-Serve Phase G collection | `COMPLETE` | `results/apt_serve_phase_g_resume_20260807_174028/` | None for collection | Preserve artifacts |
| Apt-Serve Phase G analysis | `COMPLETE` | `results/apt_serve_phase_g_analysis_20260809_190000/`, wrapper `exit_code=0`, audit `docs/audits/apt_serve_phase_g_analysis_20260809.md` | Interpretation is bounded: marginal contribution yes, global superiority no | Use as input to module-envelope interpretation |
| Stress-test library | `COMPLETE` for several baselines, `IN_PROGRESS` as an evolving library | `configs/stress_tests/algorithm_stress_test_catalog.yaml` and stress-test docs | Apt-Serve findings need integration as module/mechanism evidence, not another broad sweep | Document future module-level tests after post-Phase-G review |
| Real-system validation | `IMPLEMENTED_NEEDS_VALIDATION` | Local vLLM, hosted API, Wulver validation artifacts | Not unified into a final transfer protocol | Defer until next candidate/system exists |
| Repository hygiene | `IN_PROGRESS` | This reconciliation pass | Avoid erasing history while reducing live-doc drift | Keep historical audits immutable; maintain one current status path |
| Policy Separation Dataset v1 (three-case + boundary refinement) | `COMPLETE` for the two diagnostic jobs | Jobs 1170116, 1171116; `docs/audits/policy_separation_three_case_v1_20260810.md`, `docs/audits/policy_separation_boundary_refinement_v1_20260810.md`, `docs/audits/policy_separation_edf_admission_mechanism_20260810.md` | Full 5-family/25-template corpus in `docs/design/POLICY_SEPARATION_DATASET_V1.md` was never fully built; treat that doc's original Phase-1 claim as superseded | Use completed diagnostics + Sobol + Family A pilot as the live PSD path |
| Policy Separation Sobol pilot v1 | `COMPLETE` | Job 1182183; `docs/audits/policy_separation_sobol_pilot_v1_20260816.md`; local `experiments/policy_separation_sobol_pilot_20260816T183600Z_1182183/` | Coverage gaps for fairness/KV remain | Keep as analyzed predecessor to Family A |
| Policy Separation Family A (fairness / starvation) v1 | `COMPLETE` analysis; `REDESIGN_REQUIRED` for corpus use | Job 1182306; audit `docs/audits/policy_separation_fairness_starvation_pilot_v1_20260816.md`; provenance + `analysis/` under `experiments/…_1182306/` | ESTF↔WFS bidirectional niche absent; Aging saturated; size–priority collinearity; historical `anwg` = unweighted SLO-success; synthetic lengths | Retain frozen; do not train selectors from v1 |
| Policy Separation Family A v2 (fairness vs size) | `COMPLETE` analysis; `USEFUL_BUT_NEEDS_REFINEMENT` | Job 1182377; audit `docs/audits/policy_separation_fairness_starvation_pilot_v2_20260816.md`; provenance + `analysis/` under `experiments/…_v2_…_1182377/` | Seed agree 72% (near 75% bar); WFS ANWG non-monotone in skew under conflict; aligned short cells ESTF-dominated | Include in PSD; next mechanism family |
| ESTF↔WFS composition falsification v1 | `COMPLETE`; `SELECTION_SUFFICIENT_FOR_THIS_PAIR` | `docs/audits/estf_wfs_composition_falsification_v1_20260816.md`; run `experiments/estf_wfs_composition_falsification_v1_20260816T222108Z/` | Contextual α does not beat top-1; envelope gain 0 | Do not distill/MAP-Elites/LLM-synth from this pair; next mechanism family |

## Current Blocker

There is no active failed job to diagnose. On WS-P, Family A v2 is analyzed and
usable; ESTF/WFS composition did not unlock envelope gain beyond selection. The
remaining gap is coverage of the **next mechanism family**. Independently, the
Apt-Serve/CC blocker remains organizational: translate Phase G's bounded
evidence into module-decomposition / library-envelope work without overclaiming
global Apt-Serve superiority.

## Current Next Action

Read:

1. [`RESUME_HERE.md`](RESUME_HERE.md)
2. [`../PROJECT_MAP.md`](../PROJECT_MAP.md)
3. [`../audits/estf_wfs_composition_falsification_v1_20260816.md`](../audits/estf_wfs_composition_falsification_v1_20260816.md)
4. [`NEXT_ACTIONS.md`](NEXT_ACTIONS.md)
5. Family A v2 provenance:
   [`../../experiments/policy_separation_fairness_starvation_pilot_v2_20260816T220113Z_1182377/README.md`](../../experiments/policy_separation_fairness_starvation_pilot_v2_20260816T220113Z_1182377/README.md)

Then: (WS-P) design/execute the next mechanism family; and/or (Apt-Serve thread)
post-Phase-G module-envelope interpretation.
