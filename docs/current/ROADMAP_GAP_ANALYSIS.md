# Roadmap Gap Analysis

Status: reconciled for `wulver-selector-v2-and-composition-integrated`,
combining the composition/structural-synthesis readiness diagnosis from
`origin/wulver-final-integration-20260721` with the Phase 2C
selector-improvement gap analysis from `phase2c-final-selector-improvement`.

This document distinguishes **implemented infrastructure** (code exists and
is tested) from **experimentally validated results** (a completed evaluation
supports a specific claim) from **ongoing Wolverine experiments** (running or
recently run, results not yet reconciled here) from **future planned
research** (not started). Do not read an infrastructure item as a validated
result.

## Evidence So Far (validated results)

- Selector v2 proved adaptive policy selection can help in-distribution
  (Phase 2B.16: `regression_anwg` ANWG 0.9856 vs. best fixed 0.9776, oracle
  0.9879, on synthetic fresh validation).
- On the Phase 2C real-trace eval split (325 windows, all-non-oracle pool,
  8-policy Option B action space), the prior Phase 2C.3 `native_non_oracle_dt`
  selector remains strict-best at ANWG 0.8063 vs. best fixed SCORPIO 0.7963
  and oracle/envelope 0.8298 (29.8% gap closed); the new causal
  advanced-selector formulations in `selector/advanced.py` did not beat it
  under strict validation-based model selection --
  `docs/audits/phase2c_final_selector_improvement_audit.md`,
  `SELECTOR_STATUS = IMPROVABLE`.
- Selector v2/v3 OOD evaluations repeatedly showed fixed WSP remains hard to
  beat on held-out shifts.
- Dynamic causal features improve WSP-vs-SCORPIO delta learnability, but the
  dominant remaining real-eval failure mode (Azure-conv-like long-prompt,
  mixed-tight-SLO windows) has zero original train/val examples -- the gap is
  training-distribution/formulation, not proven causal-feature insufficiency.
- Generic uncertainty fallback, regime gating, and pairwise ranking (all in
  `selector/advanced.py`) did not close that gap either; pairwise ranking
  collapsed to SCORPIO on final eval.
- Naive rank mixtures and component-wise composition did not clear the
  native composition pilot's decision bar (`NO_GO`).
- **Structural synthesis and composition machinery is ready for small, typed
  child generation. This is an infrastructure-readiness statement, not a
  performance claim: no completed experiment has yet shown a synthesized or
  composed child beating the 27-policy library's envelope/frontier.**
- A real-trace split-grouping leakage bug (transform-specific `group_key`
  letting sibling windows of the same raw row range cross splits) was found
  and independently fixed on both lineages; the two fixes have now been
  reconciled into one implementation -- see [LOCAL_BRANCH_STATUS.md](LOCAL_BRANCH_STATUS.md).

## Ongoing Wolverine Experiments (not yet reconciled into this document)

Per `docs/current/WULVER_UNPUSHED_WORK_AUDIT.md`, results for these live only
as local artifacts on the Wulver filesystem (`/mmfs1/project/ikoutis/sv96/llmserveopt-data/`)
and have not been pulled into this repo's committed docs/claims yet:

- Policy Frontier Cartography (`policy_frontier_cartography_20260721T154408Z`).
- Policy Library V2 Expanded Frontier (`policy_library_v2_expanded_20260721T171933Z`).
- V2 real-OOD library evaluation (`selector_v2_ood_conclusive_20260721T133408Z`,
  `selector_v3_multidomain_causal_20260721T151341Z`).
- Native composition pilot (`native_composition_pilot_20260721T194929Z`).

Do not assume a specific outcome for any of these until their results are
pulled into a committed doc with a specific artifact path and commit hash.

## Likely Bottlenecks

| Bottleneck | Current likelihood | Evidence |
| --- | --- | --- |
| Workload/domain coverage | high | Selector v3 status is `DATA_LIMITED`; OOD shift remains detectable; Phase 2C's Azure-conv-like failure regime has zero train/val examples. |
| Causal feature representation | medium-high | Dynamic features helped WSP-vs-SCORPIO boundary learning but did not fully solve robustness. |
| Selector modeling | medium-low | Stronger generic model swapping (RF/Extra Trees/HGB/pairwise/uncertainty-fallback/regime-gating, all now implemented in `selector/advanced.py`) has not been the decisive lever on either the synthetic or real-trace eval splits. |
| Policy library incompleteness | unknown pending | Policy Library v2 frontier workflow is still running (see Ongoing Wolverine Experiments above). |
| Naive composition | low | Native pilot returned `NO_GO` for rank/component-wise composition. |
| Structural synthesis/evolution | promising but unvalidated | Harness is ready (genome, structural_synthesis.py, parent_selection.py), but no scaled scientific claim yet. |
| Simulator action-space limits | medium | Several literature families require unsupported cache/routing/splitting/chunking capabilities. |
| Selector v2 split-leakage | resolved (reconciled) | See [LOCAL_BRANCH_STATUS.md](LOCAL_BRANCH_STATUS.md); the stale pre-fix calibrated pilot artifact is still not safe to use for held-out claims until regenerated. |

## DONE (implemented infrastructure, this branch)

| Roadmap item | Verified implementation |
|---|---|
| BurstGPT loader | `workloads/burstgpt.py`, conversion configs, tests, local raw and processed data present. |
| Azure LLM 2023 loader/materialization | Raw code/conv CSVs and processed JSONL are present; Phase 2C scripts use them. |
| Canonical workload schema | `core.types.Request` plus JSONL/extended JSONL serializers. |
| Workload window construction | `selector/windows.py`; Dataset v2 builder also uses non-overlapping windows. |
| Causal feature extraction | `selector/features.py`, `selector/dataset_v2/features.py`, `selector/advanced.py::validate_feature_columns`; tests verify no actual-output/future-window feature leakage. |
| Candidate scheduling-policy registry | `policies/registry.py`: 27 deployable policies (20 historical + 7 Policy Library v2); Dataset v2 Option B has 8 trainable policies within that set. |
| Typed policy genome / canonical representation | `policies/genome.py` (`SchedulerGenomeV1`), canonical JSON + SHA256 hash, module taxonomy (`SUPPORTED_MODULE_TYPES`). |
| Composition infrastructure | `policies/composition.py`, `selector/composition_experiment.py`, `selector/parent_selection.py`. |
| Structural synthesis infrastructure | `policies/structural_synthesis.py`. |
| External-baseline registry | 6 faithful/paper-style external baselines are registered separately as evaluation-only. |
| Simulator metrics | TTFT, TPOT/TBT, latency p50/p95/p99 where available, throughput, SLO violation, completion, WG, ANWG. |
| Oracle/utility matrix generation | Selector v1 and v2 builders run each policy per window and preserve policy vectors. |
| Selector training | v1 DT/RF/rule/regression selectors; v2 RF prototype; `selector/advanced.py` (RF/Extra Trees/HGB reward regression, margin/regret-weighted classification, pairwise ranking, uncertainty fallback, regime gating). |
| Leakage-safe real-trace splits | `selector/dataset_v2/splits.py`, reconciled architecture -- see [LOCAL_BRANCH_STATUS.md](LOCAL_BRANCH_STATUS.md). |
| Reproducibility basics | Seeded configs/scripts, manifests, JSON/CSV artifacts, local test suite. |
| SLURM/Wulver starting point | Existing `scripts/slurm/*.sbatch` and `tools/*.sbatch` templates and GPU validation docs. |
| Small local E2E prototype | `scripts/run_local_e2e_smoke.py`; extended for this integration to exercise the full 27-policy registry. |

## PARTIAL

| Roadmap item | Current state |
|---|---|
| Azure 2024/2025 traces | Download script exists for newer Azure traces, but they are not locally acquired or integrated into the smoke path. |
| ShareGPT | Loader and tests exist; raw data is not acquired locally. |
| ServeGen | Manifest/interface-level documentation only; no pinned artifact or adapter. |
| TraceLab | Documented acquisition candidate only; no loader or schema adapter. |
| Unified canonical workload provenance | `Request` is unified, but source-specific real vs synthetic fields are tracked unevenly outside Dataset v2 manifests. |
| Selector Dataset v2 clean pilot | Infrastructure exists and leakage fix is reconciled, but a clean regenerated pilot has not been run against the reconciled architecture yet. |
| Phase 2C selector-improvement result against the 27-policy library | `run_phase2c_final_selector_improvement.py` and `selector/advanced.py` were evaluated only against the 8-policy Option B pool; not yet re-run against all 27 policies. |
| Result aggregation/reporting | Many scripts emit JSON/CSV/Markdown, but there is no single unified reporting CLI across all experiment families. |
| Real-serving validation hooks | vLLM/GPU scripts and artifacts exist; current selector claims still rely on simulator evaluation unless explicitly stated otherwise. |

## MISSING

| Roadmap item | Missing piece |
|---|---|
| TraceLab loader | Need schema inspection, canonical Request mapping, provenance fields, tests, and acquisition instructions. |
| ServeGen integration | Need pinned repo/version, generator wrapper, canonical trace emitter, and resource/runtime profile. |
| Contextual bandit / causal policy learning | No valid logged-bandit/off-policy estimator is implemented. |
| Counterfactual frontier from logged serving data | Current counterfactuals come from simulator replays, not logged production overlap/support. |
| State x policy-representation suitability model | `f(x, policy_representation) -> predicted_reward, uncertainty` is not implemented; this is the next planned research stage (do not begin without explicit direction). |
| Module-level structural credit | `C(x, policy, module)` is not implemented. |
| Production-grade selector deployment wrapper | Current selectors are research artifacts, not an integrated scheduler control plane. |

## BLOCKED

| Blocker | Why it matters |
|---|---|
| Stale pre-reconciliation Selector v2 calibrated pilot | `experiments/selector_v2_calibrated_pilot_20260720T163235Z/` was generated before the leakage fix; regenerate against the reconciled `splits.py` before trusting VALIDATION/ID_TEST claims. |
| Causal claims | No logged propensities, randomized overlap, or overlap diagnostics exist for real serving. Do not claim causal/off-policy validity yet. |
| Real-vLLM selector superiority | Existing real-vLLM selector/action-space artifacts are not decisive production comparisons. |
| Dataset acquisition/licensing | ShareGPT, TraceLab, ServeGen, Azure 2024/2025, and Mooncake/Kimi require explicit acquisition/licensing decisions. |
| Structural-synthesis/composition performance claims | Harness is implemented and testable, but no completed experiment has shown a synthesized or composed child beating the current policy-library envelope -- do not claim otherwise until one has. |

## DEFER TO WULVER

| Work | Reason to defer |
|---|---|
| Large clean Selector v2 pilot regeneration | CPU-parallel but many policy-window simulations; should use the reconciled split architecture. |
| Full external-baseline comparisons | Faithful vLLM/Sarathi/DistServe/TetriInfer/Llumnix simulations are heavier and should be decomposed by source/policy/window. |
| Real vLLM A100 validation | Requires GPU hardware, model serving, warmup, memory checks, and repeated trials. |
| Large calibration sweeps | GPU service-curve fitting and validation should use Wulver GPU queues once local scripts are stable. |
| Full composition/structural-synthesis experiments | Requires broader parallel candidate evaluation; see Ongoing Wolverine Experiments above. |

## Current Research Posture

Do not spend the next major effort on broad selector model sweeps or dense
weighted policy averaging -- `selector/advanced.py` already covers this
space (RF/Extra Trees/HGB regression, margin-weighted classification,
pairwise ranking, uncertainty fallback, regime gating) and it was not the
decisive lever on either the synthetic or real-trace eval splits. Wait for
the frontier and Policy Library v2 reports (Ongoing Wolverine Experiments
above), then either:

1. launch a narrowly justified full composition experiment;
2. generate structural symbolic children from high-value parent policies;
3. expand simulator capabilities if missing action/state mechanisms dominate
   the frontier gaps;
4. build the state x policy-representation suitability model
   (`f(x, policy_representation) -> predicted_reward, uncertainty`) as the
   next deployable-selector baseline, per the project's stated research
   architecture -- not yet started.

## Methodology Notes

- The primary current selector objective is ANWG, not completed-request-only
  `weighted_goodput`.
- Features must remain `feat_*` columns and must not include rewards, labels,
  oracle identities, actual outputs, completion outcomes, or future windows
  (`selector/advanced.py::validate_feature_columns` enforces this).
- The split invariant for transformed real traces is: overlapping raw row
  ranges under the same raw-trace ancestor and temporal pool must not cross
  TRAIN/VALIDATION/ID_TEST. See [LOCAL_BRANCH_STATUS.md](LOCAL_BRANCH_STATUS.md)
  for the reconciled key-format/verification details.
- Simulator counterfactual policy matrices are valid for simulator selector
  research. They are not real-serving causal estimates.
