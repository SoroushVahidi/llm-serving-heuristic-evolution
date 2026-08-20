# Work Status

Detailed current status table. The roadmap authority is
[`../PROJECT_MAP.md`](../PROJECT_MAP.md); this file is the operational status
companion. Historical audit reports remain authoritative only for the time they
were written.

Last reconciled: 2026-08-19, current HEAD `4dac220` (preregister
decision-criticality train-val study). No active jobs; no concurrent writer.

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
| ESTF↔WFS composition falsification v1 | `COMPLETE`; `SELECTION_SUFFICIENT_FOR_THIS_PAIR` | `docs/audits/estf_wfs_composition_falsification_v1_20260816.md`; run `experiments/estf_wfs_composition_falsification_v1_20260816T222108Z/` | Contextual α does not beat top-1; envelope gain 0 | Do not distill/MAP-Elites/LLM-synth from this pair |
| Policy Separation Family B v1 (prefill/decode chunk-control) | `COMPLETE` analysis; `USEFUL_BUT_NEEDS_REFINEMENT`; `PREFILL_COMPOSITION_NOT_YET_JUSTIFIED` | Design `docs/design/POLICY_SEPARATION_FAMILY_PREFILL_DECODE_V1.md`; audit `docs/audits/policy_separation_prefill_decode_pilot_v1_20260817.md`; run `experiments/policy_separation_prefill_decode_pilot_v1_20260817T020803Z/` (720/720) | Unique-winner diversity failed; `decode_priority` ≡ small-chunk; 96% near-ties; adaptive envelope gain 0 | Frozen; superseded scientifically by v2 |
| Policy Separation Family B v2 (TTFT-contention anchors) | `COMPLETE` analysis; `FAMILY_B_COMPOSITION_READY` | Design `docs/design/POLICY_SEPARATION_FAMILY_PREFILL_DECODE_V2.md`; audit `docs/audits/policy_separation_prefill_decode_pilot_v2_20260817.md`; run `experiments/policy_separation_prefill_decode_pilot_v2_20260817T024204Z/` (64/64) | Short-output intervention; decode_first still un-activatable on clean traces | Smallest two-parent PrefillControl composition falsification |
| PrefillControl composition falsification (`full_prefill` vs `chunked_prefill_small`) | `COMPLETE`; `SELECTION_SUFFICIENT_FOR_THIS_PAIR` | Audit `docs/audits/family_b_v2_prefill_control_composition_falsification_20260817.md`; run `experiments/prefill_control_composition_v2_20260817T154633Z/` (32 scenarios, 120/120) | Real fitted top-1 selector = oracle (0 regret) TEST+OOD; genuinely dynamic `prefill_control_child` never beats selector or expands envelope; rule tested only used 3/6 chunk options | Do not distill/MAP-Elites/LLM-synth from this pair; next mechanism family per roadmap |
| Family C v1 KV-pressure reserve pairwise-separation pilot (`kv_constrained_online` vs `least_laxity_first`) | `COMPLETE`; `KV_FAMILY_USEFUL_NEEDS_REFINEMENT`; frozen, superseded by v2 | Design `docs/design/POLICY_SEPARATION_FAMILY_KV_PRESSURE_V1.md`; audit `docs/audits/family_c_kv_pressure_pairwise_separation_v1_20260817.md`; run `experiments/kv_pressure_pilot_v1_20260817T162650Z/` (32 scenarios, 64/64) | 5/6 gates pass; tie-rate gate 59.4% did not clear <50% bound | Refined into v2 |
| Family C v2 KV-pressure reserve refinement (`kv_constrained_online` vs `least_laxity_first`) | `COMPLETE`; `KV_FAMILY_COMPOSITION_READY` | Design `docs/design/POLICY_SEPARATION_FAMILY_KV_PRESSURE_V2.md`; audit `docs/audits/family_c_kv_pressure_pairwise_separation_v2_20260817.md`; run `experiments/kv_pressure_pilot_v2_20260817T165053Z/` (72 scenarios, 144/144) | **All 10 gates pass** — bidirectional (29-vs-4/48), tie rate 31.2% (v1: 59.4%), held-out-seed replication (G6), and within-scenario winner-flip evidence (G10, 6/16 matched cells show a different practical winner depending only on urgent-arrival timing) — the first family of three studied to reach a `_READY` verdict | Composition falsification run (below) |
| KV-aware composition falsification v1 (`kv_constrained_online` vs `least_laxity_first`) | `COMPLETE`; `KV_COMPOSITION_INCONCLUSIVE` | Design `docs/design/KV_COMPOSITION_FALSIFICATION_V1.md`; audit `docs/audits/kv_composition_falsification_v1_20260817.md`; run `experiments/kv_composition_falsification_v1_20260817T172446Z/` (72 scenarios, 576/576) | 6/8 gates pass with real signal — positive TEST envelope gain (G2), 5/12 TEST scenarios beat both parents by >ε (G4), 24/36 held-out scenarios non-degenerate within-trajectory mode-switching (G1), directionally-consistent OOD replication (G6) — but **G7 (safety) fails**: child's peak KV utilization exceeds `max(parent peaks)` on 6/36 held-out scenarios, a composition-specific risk no pairwise pilot can surface. Frozen decision rule: G7 failing forces `INCONCLUSIVE` regardless of G1-G6. Also surfaced (not gated): the current environment cannot reproduce the historical KV v2 CSV bit-for-bit even via the original unmodified runner (99/144 mismatch) — forensic follow-up below | Do not escalate to a more complex child/MAP-Elites/synthesis; smallest defensible next step is a transition-aware admission-cap variant, not started |
| KV v2 reproducibility forensic audit | `COMPLETE`; `REPRODUCIBILITY_GAP_BOUNDED` | Audit `docs/audits/kv_v2_reproducibility_forensic_20260817.md`; provenance guard `scripts/run_policy_separation_kv_pressure_pilot_v1.py` (`_collect_provenance`); tests `tests/test_kv_pressure_provenance.py` (14 new) | Root cause not demonstrated. Ruled out/narrowed: code drift (0 diff on the entire KV v2 execution path since launch commit `6be526e`), runtime/multiprocessing nondeterminism (current runs are byte-identical-SHA-256-reproducible), and both locally available BurstGPT files (neither reproduces history; their sampling pools are nearly but not exactly identical — 7335 vs 7337 filtered rows). Historical mismatch is material: 99/144 cells differ, 17/72 scenarios flip practical winner. Composition falsification's internal validity unaffected (all its methods share one current-environment run); only cross-run comparison to historical v2 numbers requires caution. No historical CSV/verdict rewritten | Future KV runs now record git SHA/dirty, config+dataset SHA-256, library versions, result-CSV SHA-256 automatically; no further action required unless root cause is later pursued |
| Higher-level structural reassessment of the composition hypothesis | `COMPLETE`; `COMPOSITION_DEMOTED` | Audit `docs/audits/reassessment_composition_hypothesis_20260817.md` | Composition/synthesis demoted from central hypothesis to exploratory future work | Revised roadmap: policy-separating workloads -> complementary policy library -> contextual selection (multi-family) -> mechanism attribution -> bounded envelope; Step 1 = MF-PSD (below) |
| MF-PSD v1 (Multi-Family Policy Separation Dataset, revised-roadmap Step 1) | `COMPLETE`; `MF_PSD_READY` | Audit `docs/audits/multi_family_policy_separation_dataset_v1_20260817.md`; artifacts `experiments/mf_psd_v1/`; builder `src/llmserveopt/policy_separation/mf_psd.py`; CLI `scripts/build_mf_psd_v1.py`; tests `tests/test_mf_psd_v1.py` (31/31) | Unifies Family A v2 (288 rows/72 scenarios), Family B v2 (64 rows/32 scenarios), Family C/KV v2 (144 rows/72 scenarios) into one canonical long-form table (496 rows) + scenario-context table (176 scenarios); explicit learnable-feature allowlist (34 family-prefixed columns) vs forbidden/audit-only denylist; exact row/scenario conservation; 0 duplicates; deterministic byte-for-byte rebuild; 0 mutation of frozen sources (checksum-verified). Six-anchor policy matrix is sparse (each family only ran its own 2 anchors), not dense — documented, not fabricated | Data unification only; no selector trained, no composition run. Step 2 (unified six-policy utility-matrix evaluation, ~704 new policy-scenario evaluations) is next but NOT started |
| Unified Utility Matrix v2 (revised-roadmap Step 2) | `COMPLETE`; `UNIFIED_UTILITY_MATRIX_READY` | Artifacts `experiments/unified_utility_matrix_v2/`; audit `docs/audits/family_c_reconstruction_v1_and_unified_matrix_completion_20260817.md` | 176×6 = 1,056/1,056 cells populated; 54.0% unique-winner rate; positive oracle gain over best-fixed in every family. Missing a build manifest (v1 has one, v2 does not) — additive-only debt, not a content defect | Fed Step 3 (flat/pooled selector) |
| Family C Reconstruction v1 | `COMPLETE`; `FAMILY_C_RECONSTRUCTION_BOUNDED` | Artifacts `experiments/family_c_reconstruction_v1/`; audit `docs/audits/family_c_step2_reconstruction_audit_20260817.md` | Historical KV v2 replay was structurally impossible (runner never serialized request data); resolved via a new versioned generate-once/serialize/replay layer, 432/432 cells | Fed Unified Utility Matrix v2 |
| Flat/pooled multi-family selector (revised-roadmap Step 3) | `COMPLETE`; `MULTIFAMILY_SELECTOR_NO_GO` | Artifacts `experiments/multifamily_contextual_selector_v1/`; audit `docs/audits/multifamily_contextual_selector_v1_20260817.md` | All 5 preregistered gates failed on pooled Regime-B holdout; within-family (Regime A) selection strong. Root cause: `mechanism_family` is 100% classifiable from the feature schema alone (family-identifying leakage, not mechanism understanding) | Motivated the shared-feature-schema redesign |
| Shared Cross-Family Feature Schema v1 | `COMPLETE`; `SHARED_FEATURE_SCHEMA_NO_GO` | Artifacts `experiments/shared_cross_family_features_v1/`; audit `docs/audits/shared_cross_family_feature_schema_feasibility_v1_20260817.md` | 17-feature zero-missingness schema built/replay-verified for 176/176 scenarios; family still 100% classifiable (disjoint feature-space regions, not missingness); cross-family nearest neighbors not more utility-consistent than random | Motivated the mechanism-choice-target redesign |
| Mechanism-Choice Target Feasibility v1 | `COMPLETE`; `MECHANISM_TARGET_NO_GO` | Artifacts `experiments/mechanism_choice_target_feasibility_v1/`; audit `docs/audits/mechanism_choice_target_feasibility_v1_20260817.md` | `kv` mechanism contrast confounded (largest on Family A, which has no KV pressure); two-stage pipeline retains zero net advantage over a fixed global policy; only `ranking` shows genuine cross-family activation | Motivated the cross-family transfer well-posedness reassessment |
| Cross-Family Transfer Well-Posedness Reassessment v1 | `COMPLETE`; `CROSS_FAMILY_TRANSFER_DEMOTED_HIERARCHICAL_ROUTING_READY` | Artifacts `experiments/cross_family_transfer_wellposedness_reassessment_v1/`; audit `docs/audits/cross_family_transfer_wellposedness_reassessment_20260817.md` | Three independent NO_GOs converge on genuinely different root causes; within-family evidence strong; cross-family structure exists but is insufficient for a universal per-scenario selector | Motivated hierarchical regime-router design |
| Online Regime-Signal Feasibility v1 | `COMPLETE`; `ONLINE_REGIME_SIGNALS_READY` | Artifacts `experiments/online_regime_signal_feasibility_v1/` (127,319-row per-step telemetry CSV, ~29.9MB); audit `docs/audits/online_regime_signal_feasibility_v1_20260817.md` | Family-B contention detectable after a documented correction; all three activity signals achieve perfect precision; zero regime overlap observed | Fed hierarchical router Stage-1 |
| Hierarchical Regime Router v1 (TEST, offline majority-vote) | `COMPLETE`; `HIERARCHICAL_ROUTER_NO_GO` | Artifacts `experiments/hierarchical_regime_router_v1_test_evaluation/`; impl commit `2923087`; audit `docs/audits/hierarchical_regime_router_v1_20260818.md` | G4 (Stage-2 preservation) and G5 (beat global fixed) fail on TEST; Stage-1/Stage-2 individually excellent but offline majority-vote integration washes out minority-of-steps activity; Family B got 0 TEST scenarios | Motivated the live per-step harness |
| Hierarchical Router Live Harness v1 (smoke) | `COMPLETE`; `LIVE_HIERARCHICAL_HARNESS_READY` | Artifacts `experiments/hierarchical_router_live_harness_v1_smoke/`; commit `723a39c`; audit `docs/audits/hierarchical_router_live_harness_validation_v1_20260818.md` | 6/6 forced-parent equivalence checks bit-exact; causal-switch microcase confirms genuine per-step causal dispatch | Enabled the live re-evaluation |
| Hierarchical Router Live Re-evaluation v1 | `COMPLETE` (formally gate-rescored); `HIERARCHICAL_ROUTER_NO_GO` | Artifacts `experiments/hierarchical_regime_router_live_reeval_v1/live_reeval_results.json`; run commit `9fde981`, fix `ed74276`; audit `docs/audits/hierarchical_regime_router_live_reeval_v1_20260818.md` | Live ANWG 0.8136 vs best-fixed 0.8075 (delta 0.00616, below the 0.01 bar); oracle-gap closure 0.143 (below the 0.75 bar); Family B again got 0 scenarios. Two current tracked-provenance-only diffs exist in `gate_rescoring_v1.json` (timestamp/HEAD-SHA, no metric change) | Motivated Family-B-specific live evaluation |
| Family-B Balanced Replication v1 | `IMPLEMENTATION_READY`, scientific run `NOT_STARTED` | Design `docs/design/FAMILY_B_BALANCED_REPLICATION_V1.md`; artifacts `experiments/family_b_balanced_replication_v1/`; added in commit `9d8f997`; runner `scripts/run_family_b_balanced_replication_v1.py` | Only `--source smoke_train`/`smoke_synthetic` have been run. `--source replication` (the frozen 36-scenario scientific set) requires `--i-am-authorized` in addition and has **not** been invoked — no held-out scientific run exists | Requires separate, explicit scientific authorization before launching `--source replication` |
| Public Trace Corpus v1 (workload-input layer, Layers 0-1) | `COMPLETE`; committed and pushed | Design `docs/design/PUBLIC_TRACE_CORPUS_V1.md`; artifacts `data/public_trace_corpus_v1/`; adapter `src/llmserveopt/workloads/public_trace_corpus.py`; builder `scripts/build_public_trace_corpus_v1.py`; tests `tests/test_public_trace_corpus_v1.py`; commits `84fa31b` + `179a6fe` | Ingests BurstGPT + Azure 2023 conv/code; classifies AgentPerfBench as `REAL_SYSTEM_VALIDATION_SOURCE` (not ingested); no policy outcomes, no oracle labels, no paid API use | Layer 2+ (policy replay across the completed policy library) is NOT started; requires authorization |
| Decision-Criticality / Regime-Timescale TRAIN/VAL analysis | `IMPLEMENTED`; committed and pushed; scientific run `NOT_STARTED` | Design `docs/design/DECISION_CRITICALITY_TIMESCALE_TRAINVAL_V1.md`; module `src/llmserveopt/analysis/decision_criticality_timescale_trainval_v1.py`; runner `scripts/run_decision_criticality_timescale_trainval_v1.py`; tests `tests/test_decision_criticality_timescale_trainval_v1.py` (40/40 pass); committed in `4dac220` | Design/code/tests are complete and tracked; no `experiments/decision_criticality_timescale_trainval_v1/` output directory and no run log exist — the full 144-scenario TRAIN/VAL run has not been executed and produced no scientific conclusion. Must not import or use the Family-B held-out replication as evidence (guarded) | Requires separate, explicit authorization before launching |
| New-policy synthesis/evolution | `NOT_STARTED` (long-term goal) | n/a | Depends on Public Trace Corpus Layer 2+ (policy replay) and the decision-criticality/mechanism-attribution layer above | Not actionable until both upstream dependencies land |

## Current Blocker

There is no active failed job to diagnose and no active process/tmux session
running anywhere for this repository as of 2026-08-19. The revised roadmap's
Steps 1-3 (MF-PSD → Unified Utility Matrix → flat/pooled selector) are all
complete; the flat/pooled selector, shared-feature-schema, and
mechanism-choice-target redesigns each independently returned `NO_GO`,
converging on `CROSS_FAMILY_TRANSFER_DEMOTED_HIERARCHICAL_ROUTING_READY`.
The hierarchical regime router that followed is also `HIERARCHICAL_ROUTER_NO_GO`
at both TEST and live re-evaluation (Family B got 0 scenarios in both). The
live blocker for a Family-B-specific verdict is the Family-B Balanced
Replication scientific run, which is implementation-ready but **not yet
authorized to launch**. A known, pre-existing environment limitation: 6
`tests/test_unified_utility_matrix_v1.py` tests fail off-cluster because
Family-A v2 "production mode" refuses a silent synthetic fallback when
staged BurstGPT isn't present under the expected `/mmfs1/...` cluster path
(or `LLM_SERVEOPT_BURSTGPT_CSV`) — this is expected locally, not a
regression (see `logs/overnight_full_repo_validation_20260819.log`).

Independently, the Public Trace Corpus v1 workload-input layer (Layers 0-1)
is complete and committed (`84fa31b`/`179a6fe`); Layer 2+ (policy replay) is
not started. The decision-criticality/regime-timescale analysis's design,
implementation, and tests are complete, committed, and pushed (`4dac220`),
but the actual TRAIN/VAL run has not been executed and produced no run
output or scientific conclusion — it must not be launched without separate
explicit authorization.

Historical context below (WS-P family-by-family pairwise-separation and
composition-falsification work) remains valid evidence and is unchanged; it
fed directly into the reassessment above. On WS-P, Family A v2 is analyzed and
usable; ESTF/WFS composition did not unlock envelope gain beyond selection.
Family B v1 is frozen (`USEFUL_BUT_NEEDS_REFINEMENT` /
`PREFILL_COMPOSITION_NOT_YET_JUSTIFIED`). Family B v2 (the next mechanism family
after ESTF/WFS, refined) is `FAMILY_B_COMPOSITION_READY`, and its PrefillControl
composition falsification is now COMPLETE (`SELECTION_SUFFICIENT_FOR_THIS_PAIR`
— second independent pair to land this verdict, after ESTF/WFS). Family C
v2 KV-pressure reserve (`kv_constrained_online` vs `least_laxity_first`)
reached `KV_FAMILY_COMPOSITION_READY` — the first of the three studied
pairs to reach this verdict — and its composition falsification has since
run to completion: `KV_COMPOSITION_INCONCLUSIVE` (real envelope-gain signal,
blocked by a composition-specific KV-safety gate failure, not by absence of
signal; see `docs/audits/kv_composition_falsification_v1_20260817.md`).
Do not escalate to a more complex child or synthesis without explicit
authorization. Independently, the Apt-Serve/CC
blocker remains organizational: translate Phase G's bounded evidence into
module-decomposition / library-envelope work without overclaiming global
Apt-Serve superiority.

## Current Next Action

Read:

1. [`RESUME_HERE.md`](RESUME_HERE.md)
2. [`../PROJECT_MAP.md`](../PROJECT_MAP.md)
3. [`../audits/cross_family_transfer_wellposedness_reassessment_20260817.md`](../audits/cross_family_transfer_wellposedness_reassessment_20260817.md)
   (`CROSS_FAMILY_TRANSFER_DEMOTED_HIERARCHICAL_ROUTING_READY`)
4. [`../audits/hierarchical_regime_router_live_reeval_v1_20260818.md`](../audits/hierarchical_regime_router_live_reeval_v1_20260818.md)
   (`HIERARCHICAL_ROUTER_NO_GO`, formal)
5. [`NEXT_ACTIONS.md`](NEXT_ACTIONS.md)

Three independent threads are queued, each requiring separate explicit
authorization before launch: (1) the Family-B Balanced Replication
scientific run (`--source replication --i-am-authorized`); (2) Public Trace
Corpus v1 Layer 2+ policy replay; (3) the decision-criticality/timescale
analysis's actual TRAIN/VAL run (design/code/tests committed in `4dac220`;
the run itself is not yet launched). Do not start selector retraining,
MAP-Elites, or new-policy synthesis before the decision-critical-state
evidence these threads are meant to produce exists.

Historical WS-P entrypoint (superseded as the "current next action" by the
above, but still valid background):
[`../design/POLICY_SEPARATION_FAMILY_PREFILL_DECODE_V2.md`](../design/POLICY_SEPARATION_FAMILY_PREFILL_DECODE_V2.md),
Family B v2 audit:
[`../audits/policy_separation_prefill_decode_pilot_v2_20260817.md`](../audits/policy_separation_prefill_decode_pilot_v2_20260817.md),
PrefillControl composition falsification audit (COMPLETE, `SELECTION_SUFFICIENT_FOR_THIS_PAIR`):
[`../audits/family_b_v2_prefill_control_composition_falsification_20260817.md`](../audits/family_b_v2_prefill_control_composition_falsification_20260817.md).
(Apt-Serve thread, independent) post-Phase-G module-envelope interpretation
remains available as a parallel task.
