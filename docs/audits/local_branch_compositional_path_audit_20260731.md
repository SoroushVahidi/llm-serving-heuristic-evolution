# Local Branch Compositional Path Audit - 2026-07-31

Repository: `/home/soroush/llm-serving-heuristic-evolution`
Audit scope: current local branch at `21023c149b089ff8d53af603b03ce094735e4b56`
Audit rule: no implementation of the new compositional method; only this report was intentionally added.

## Synchronization Addendum - 2026-07-31

This report was originally written against local commit
`21023c149b089ff8d53af603b03ce094735e4b56` on
`reality-grounded-dataset-expansion-20260724`. Query 1 of the contextual
composition sequence subsequently fast-forwarded that branch to
`775147beec997b14039bbaa088d17630a32156cf`, matching
`origin/reality-grounded-dataset-expansion-20260724`.

The 18 incoming commits are expected project work. They add Tier 1 real-dataset
staging and validation tooling, streaming real-window construction, repaired
load-discrimination pilot tooling/tests, pause/handoff documentation under
`docs/current/pause_2026_07_25/`, and `.gitignore` hygiene for generated
cluster and experiment artifacts. They do not add a file at this audit report
path, and the pre-synchronization report checksum was preserved through the
fast-forward.

The central audit verdict does not change: the repository has useful selector,
DSL, generated-heuristic, and native composition prototypes, but still lacks a
verified contextual weighted-composition path. The incoming commits add
important real-trace and load-discrimination context that should be considered
before Query 2 freezes the roadmap. Findings tied to "current branch" or
"current GitHub synchronization" below refer to the original audited local state
unless this addendum explicitly supersedes them.

## 1. Executive Verdict

The local checkout is scientifically rich but not a clean authoritative starting point for the next phase without synchronization and artifact decisions.

Central findings:

- The worktree has no tracked or staged source changes, but the local tracking ref is stale. The local branch equals the local `origin/reality-grounded-dataset-expansion-20260724` ref, while live GitHub has the same branch at `775147beec997b14039bbaa088d17630a32156cf`, 18 commits ahead of the local checkout.
- `main` is an ancestor of the current local branch. Locally, the current branch is 130 commits ahead of `main` and 0 behind. Live GitHub `main` is still `277e53533d063db4e609c48d64c7463ec8566bfb`.
- The repository contains multiple generations of selector work. The corrected-objective selector artifact exists locally and is manifest-verifiable, but it is gitignored and depends on gitignored training/validation CSVs. It is usable locally, not reproducible from a fresh clone alone.
- The DSL can express verified numeric request-scoring heuristics with regimes and admission conditions. It cannot currently express named reusable policy primitives, policy-score calls, rank aggregation, learned contextual weights, or uncertainty-aware fallback as DSL constructs.
- A native Python composition harness already exists and is tested: rank aggregation, reciprocal rank, normalized score aggregation, contextual hand-weighting, top-k sparsity, hysteresis/min-commitment, logging, and fallback. This is not yet the intended verified DSL composition system.
- Existing artifacts can compute best fixed policy, hard selector performance, and oracle per-window fixed-policy performance. They cannot compute the true composition opportunity gap for weighted mixtures without executing composed policies in the simulator, because per-window policy reward vectors do not contain counterfactual outcomes for mixtures.
- The smallest missing layer is a canonical primitive interface plus a compiler path from contextual primitive weights into verified DSL/genome programs. Existing native composition code is the right implementation seed, but it is not yet a verified, parameterized composition pipeline.

## 2. Exact Git And Branch State

Commands used included:

- `git status --short --branch --ignored=matching`
- `git status --porcelain=v2 --branch`
- `git branch --show-current`
- `git rev-parse HEAD`
- `git rev-parse --abbrev-ref --symbolic-full-name @{u}`
- `git log --oneline --decorate --graph --all -n 30`
- `git branch -vv --all`
- `git rev-list --left-right --count HEAD...main`
- `git rev-list --left-right --count HEAD...origin/main`
- `git ls-remote --heads origin reality-grounded-dataset-expansion-20260724 main`
- a temporary clone under `/tmp/llmserveopt-remote-audit.*` to compare the local SHA to the live GitHub branch without updating local refs.

Exact state:

- Current branch: `reality-grounded-dataset-expansion-20260724`
- Current local commit: `21023c149b089ff8d53af603b03ce094735e4b56`
- Local upstream branch: `origin/reality-grounded-dataset-expansion-20260724`
- Local ahead/behind against the local upstream ref: `+0 -0`
- Staged changes: none
- Tracked unstaged changes: none
- Untracked non-ignored changes: none before this report
- Ignored local artifacts: many, including `.claude/`, caches, `logs/`, gitignored `results/*`, raw/processed data, local selector artifacts, local experiment logs, and untracked experiment CSVs under `experiments/selector_v2_calibrated_pilot_20260720T163235Z/`.

Important remote finding:

- Local tracking ref `origin/reality-grounded-dataset-expansion-20260724` points to `21023c1`.
- Live GitHub branch `refs/heads/reality-grounded-dataset-expansion-20260724` points to `775147beec997b14039bbaa088d17630a32156cf`.
- Temporary remote comparison: local `21023c1` is an ancestor of live `775147b`; live is 18 commits ahead and 0 behind local.
- Therefore, the local branch is clean against stale local refs but not synchronized with live GitHub.

Recent local history:

```text
21023c1 fix: correct dataset licenses, Mooncake ms timestamps, and tiers
2068ab0 test: cover canonical schema and real-dataset converters with fixtures
e87fec0 feat: add verified Bailian/Mooncake/Azure adapters and BurstGPT metadata
e160cc5 feat: add canonical ingestion schema with field provenance
f4d7d4f docs: inventory reality-grounded dataset expansion status and provenance
b0768f2 fix: make BurstGPT smoke path resolution portable
e1eef2b test: pin heuristic integration gaps and fix BurstGPT smoke fallback
994ff49 docs: reconcile post-pause remote tip and verified gaps
7345015 chore: declare optional selector ML dependencies
```

Live GitHub commits beyond local begin:

```text
775147b docs: normalize legacy classification TSV whitespace
e02e1a0 docs: record legacy composition worktree resolution at pause
6aea97d docs: stabilize Part 2 completion tip references
9310381 docs: record authoritative Part 2 HEAD in completion files
d8fd116 docs: sync Part 2 completion artifacts with HEAD
```

Relationship to `main`:

- Local `git rev-list --left-right --count HEAD...main`: `130 0`
- Local `git rev-list --left-right --count HEAD...origin/main`: `130 0`
- `main` is an ancestor of current `HEAD`; current `HEAD` is not an ancestor of `main`.
- Local `main` and `origin/main` both point to `277e53533d063db4e609c48d64c7463ec8566bfb`.

Relevant local branches:

- Most phase branches through Phase 2C.1, repo polish, baseline branches, selector-dataset branches, Wulver integration branches, and `main` are ancestors of the current branch.
- `phase2c-final-selector-improvement` is divergent from current: current has 43 commits not in that branch; that branch has 3 commits not in current.
- `phase2b13-selector-training-after-diversity` is also divergent by one branch-side commit.
- Only the current local branch contains `HEAD`.
- `wulver-selector-v2-and-composition-integrated` is 5 commits behind current local HEAD and is an ancestor.

Ignored-but-important artifacts:

- `results/corrected_selector_artifact_regression_anwg/`: corrected selector `.joblib` and manifest.
- `results/phase2b13_selector_training_and_suspicion_audit/`: local training table for corrected selector.
- `results/phase2b16_fresh_corrected_objective_validation/`: local fresh validation table and summaries.
- `results/phase2c_final_selector_improvement/`: local Phase 2C selector evaluation summaries.
- `results/composition_score_rank_smoke/` and `results/composition_smart_pilot/`: local composition smoke/proxy outputs.
- `results/wulver_imports/module_intervention_credit_20260721T224322Z/`: imported Wulver module-credit artifacts.
- `data/raw/*` and `data/processed/*`: local raw/processed datasets.
- `experiments/selector_v2_calibrated_pilot_20260720T163235Z/full_policy_vectors.csv` and `window_features.csv`: ignored but important for the local clean-pilot/composition proxy analyses.

## 3. Current Architecture

Implemented architecture, from source inspection:

- `src/llmserveopt/core/`: request/action/types and metrics, including corrected `arrival_normalized_weighted_goodput`.
- `src/llmserveopt/simulator/`: deterministic discrete-event simulator, GPU/KV state, service model, calibrated service model, prefill/decode contention, decode hold.
- `src/llmserveopt/policies/registry.py`: 20 historical deployable simulator policies in `BASELINE_NAMES`.
- `src/llmserveopt/policies/external_baselines_registry.py`: 7 external/literature baselines kept separate from `BASELINE_NAMES`: `vllm_faithful`, `vllm_chunked_prefill_faithful`, `sarathi_faithful`, `distserve_faithful`, `tetriinfer_paper_reimplementation`, `llumnix_faithful`, `slai_faithful`.
- `src/llmserveopt/policies/registry.py`: 7 Policy Library v2 policies in `POLICY_LIBRARY_V2_NEW_NAMES`, constructible via `make_policy_library_v2` but not part of `BASELINE_NAMES`.
- `src/llmserveopt/selector/`: legacy selector windows/features/labels/models plus advanced corrected-objective selector wrappers.
- `src/llmserveopt/selector/dataset_v2/`: full per-policy outcome-vector dataset construction, causal features, objective discriminativeness, regret, and leakage-safe split helpers.
- `src/llmserveopt/heuristics/`: verified DSL schema, evaluator, compiler, verifier, and simulator policy wrapper.
- `src/llmserveopt/llm_generation/`: offline/mock-capable LLM heuristic generation, repair, archive, deduplication, and synthetic evaluation.
- `src/llmserveopt/policies/composition.py`, `score_aggregation.py`, `capabilities.py`: native Python composition prototypes.
- `src/llmserveopt/policies/genome.py`, `structural_synthesis.py`: typed scheduler genome wrapper around the verified DSL, parent mappings, module swaps, conditional composition, mutation, and structural synthesis helpers.
- `scripts/run_vllm_external_baseline_comparison.py` and `scripts/run_hosted_policy_comparison.py`: real-serving external admission-control harnesses.

Stale documentation note:

- `README.md` says to start at `docs/current/RESUME_HERE.md` and also contains older Selector v2 split-leakage warnings.
- `docs/current/PROJECT_STATUS.md` and composition docs record later Wulver/Policy Library v2/composition work and supersede parts of the README.
- Some result manifests in `results/composition_smart_pilot/*` record dirty worktrees and branches from earlier integration states. Treat them as provenance, not current branch truth.

## 4. Selector Audit

### Pipeline

Workload-window construction:

- Legacy: `selector/windows.py` partitions sorted requests into non-overlapping request-count windows; default `window_size=200`, partial tail kept if at least 50 requests.
- Selector Dataset v2: `selector/dataset_v2/builder.py` uses the same window helper but builds `WindowRecordV2` records preserving full per-policy outcomes.

Causal feature extraction:

- Legacy: `selector/features.py` has 18 features: queue length, active sequence count, KV utilization, free sequence ratio, prompt/output statistics, SLO slack/tightness, waiting time, arrival rate/burstiness, recent SLO violation rate.
- Deployable mode is `FeatureMode.CAUSAL`; `OFFLINE_WINDOW_LOOKAHEAD` and `TRACE_WINDOW_DESCRIPTIVE` are explicitly non-deployable.
- Dataset v2: `selector/dataset_v2/features.py` adds topology/resource-aware features and still only emits model-eligible `feat_*` columns.

Labels/objectives:

- Legacy `selector/labels.py` still labels `best_policy` by completed-request `weighted_goodput`. This is an old objective path.
- Corrected-objective scripts compute `arrival_normalized_wg = completion_fraction * completed_request_quality`.
- Dataset v2 objective layer defines `PRIMARY_SELECTOR_OBJECTIVE = "arrival_normalized_weighted_goodput"` while retaining historical `weighted_goodput`.

Candidate policy pools:

- Legacy selector candidates: all 20 `BASELINE_NAMES`, oracle excluded.
- Selector Dataset v2 diagnostic monolithic pool: external monolithic baselines plus 11 historical policies.
- Current Selector v2 Option B action space: exactly 8 policies: `fifo`, `edf`, `scorpio_style_slo_guard`, `admission_control`, `weighted_shortest_processing`, `estimated_service_time_first`, `best_fit`, `multi_bin_batching`.
- Phase 2C/Policy Library v2 work separately uses 27 internal policies in local Wulver-derived artifacts.

Splits:

- Phase 2B.15 corrected retraining uses train dev seeds plus diversity seeds 6-10, validation diversity seed 11, heldout split for test.
- Dataset v2 has group-aware hash splits and explicit real-trace row-range overlap checks in `selector/dataset_v2/splits.py`.
- The calibrated pilot artifact has documented leakage caveats for non-OOD splits in current docs; OOD split is cleaner.

Persisted artifacts:

- Pre-correction selector artifacts exist historically in ignored result trees but lack the corrected manifest contract.
- Corrected artifact exists at `results/corrected_selector_artifact_regression_anwg/regression_anwg_selector.joblib`.
- Manifest exists at `results/corrected_selector_artifact_regression_anwg/manifest.json`.

Inference path:

- Offline simulator selectors predict from `feat_*` rows and select one policy name.
- Real vLLM/hosted harness can load only `PerPolicyRegressionAnwgSelector` with a sibling manifest declaring `objective_definition.name == "arrival_normalized_wg"`.
- Real vLLM selector feature adapter reconstructs 17/18 features from client-side state and uses a documented `kv_utilization=0.0` placeholder because it does not scrape `/metrics`.

Use in simulator:

- Legacy dataset/simulation wrappers build rows by running policy candidates on isolated windows.
- Phase 2B/2C scripts evaluate hard selector predictions against per-window policy outcome vectors.
- Native composition policy execution is separate from hard selector execution.

Use in real-vLLM/hosted comparisons:

- `run_vllm_external_baseline_comparison.py` supports `selector` only with the corrected artifact.
- `run_hosted_policy_comparison.py` reuses the vLLM harness logic and similarly supports `selector` with the corrected artifact.
- These are client-side admission-control comparisons, not vLLM internal scheduling replacements.

### Implemented Selectors

`RuleBasedSelector`

- Type: hand-coded rules.
- Features: legacy 18 causal features.
- Target/loss: none.
- Output: one policy name.
- Deployability: deployable in simulator; policy choices must be in `SELECTOR_CANDIDATES`.
- Evaluation: Phase 2B/2C local results.
- Strongest verified result: Phase 2B.16 fresh ANWG `0.9771`, gap vs SCORPIO `+0.0085`, CI `[0.0055, 0.0112]`.
- Failure modes: brittle thresholds, stale rule comments, routes only among hand-coded subset, older comments refer to policies not in narrower Option B contexts.

`DecisionTreeSelector` and `RandomForestSelector`

- Type: sklearn classifier on `best_policy`.
- Features: legacy 18 features.
- Target/loss: exact best-policy classification under whatever label field the training rows provide.
- Output: one policy name.
- Deployability: deployable if trained on causal features and valid action set.
- Evaluation: legacy weighted-goodput phases and corrected-objective relabeling variants.
- Failure modes: objective-dependent labels, near-tie instability, weak OOD robustness.

`PerPolicyRegressionAnwgSelector`

- Type: one `RandomForestRegressor` per candidate policy.
- Features: legacy 18 `feat_*` columns.
- Target/loss: per-policy `arrival_normalized_wg`, computed as `completion_policy * reward_policy`.
- Output: argmax policy across predicted policy values.
- Deployability: deployable in simulator and externally loadable by vLLM/hosted harness with manifest.
- Trained: local ignored `results/phase2b13_selector_training_and_suspicion_audit/per_window.csv`, train split 245 windows.
- Evaluated: local Phase 2B.13 heldout 33 windows and Phase 2B.16 fresh 174 windows.
- Strongest verified result: fresh 174-window mean ANWG `0.9856`, gap vs always-SCORPIO `+0.0170`, CI `[0.0127, 0.0213]`; manifest reproducibility check matches published claim.
- Failure modes: local-only artifact, created at dirty commit `6938b8b...`; 93.1% of fresh windows are near-ties at margin `<0.005`; `kv_utilization` not honestly client-observable in real-vLLM external harness without metrics scrape.

`PolicyRewardRegressorSelector`

- Type: advanced one-regressor-per-policy wrapper, estimator selectable among RF/ExtraTrees/HGB.
- Features: validated causal `feat_*` columns.
- Target/loss: per-policy ANWG.
- Output: argmax policy, also exposes `predict_scores`.
- Deployability: deployable if trained/frozen and action set is dispatchable.
- Evaluation: Phase 2C scripts and local result summaries.
- Strongest verified result observed in local final no-augmentation Phase 2C real eval: prior RF ANWG variants slightly beat best fixed SCORPIO in point estimate, but results are caveated by docs around split issues and simulator discriminativeness.
- Failure modes: can learn weak labels under saturated objectives; no canonical persisted artifact in current branch for Phase 2C.

`PolicyClassifierSelector`

- Type: advanced classifier with optional regret/margin sample weights.
- Features: causal `feat_*`.
- Target/loss: best-policy label column.
- Output: one policy, exposes class probability scores.
- Deployability: deployable if action set is valid.
- Failure modes: label near-ties and OOD distribution shift.

`PairwisePolicyRanker`

- Type: one classifier per policy pair with margin filtering and votes.
- Features: causal `feat_*`.
- Target/loss: pairwise winner when ANWG margin exceeds threshold.
- Output: one policy by vote argmax, exposes vote scores.
- Deployability: deployable if trained and pair set frozen.
- Failure modes: pairwise coverage gaps, no calibrated utilities, can collapse under near-ties.

`UncertaintyFallbackSelector`

- Type: wrapper around a selector exposing `predict_scores`.
- Features: whatever base selector uses.
- Target/loss: none.
- Output: base top policy unless top-two predicted margin below threshold, then fixed fallback.
- Deployability: deployable if threshold frozen on development data.
- Failure modes: prediction-margin uncertainty was documented as insufficient for robustness; margin is not calibrated epistemic uncertainty.

`RegimeGatedSelector`

- Type: hand gate between specialist and default selector.
- Features: feature-only gate, e.g. Azure-conv-like.
- Output: one policy from specialist/default selector.
- Deployability: deployable if gate is causal.
- Failure modes: gate can be stale or too narrow; not a mixture.

Oracle-assisted selectors:

- `SafeFallbackWspSelector` in Phase 2B.15 uses actual per-window rewards at prediction time to decide whether to switch from WSP. This is explicitly oracle-assisted and not deployable.

### Objective Distinctions

- Completed-request weighted goodput: `weighted_goodput`, denominator is completed requests only. This is still used in legacy label code and old generated-heuristic evaluation.
- Arrival-normalized weighted goodput: corrected primary selector objective. Dropped/rejected/unfinished arrivals count as zero through completion fraction.
- Old objective results: Phase 2A/2B.4 and early LLM-generated heuristic claims are completed-request WG and include selective-dropping caveats.
- Corrected-objective results: Phase 2B.15/2B.16 local ignored artifacts and corrected selector manifest.
- Synthetic fresh-validation results: Phase 2B.16 fresh 174-window results, strong but near-tie dominated.
- Phase 2C real-trace-derived results: local `results/phase2c_*`, docs report useful but insufficient selector/suitability signals and simulator discriminativeness blockers.

## 5. DSL And Generated-Heuristic Audit

DSL schema:

- Top-level required fields: `name`, `default`, `tie_breaker`.
- Required rule field: `request_score`.
- Optional rule fields: `batch_score`, `admission_condition`.
- Optional `regimes`: each has `condition` and `request_score`.

Expression primitives:

- Variables: request, system, and batch variables from `heuristics/dsl_schema.py`.
- Operators: `const`, `var`, arithmetic, `clip`, `sqrt_safe`, `log1p_safe`, `weighted_sum`, `if_then_else`.
- Forbidden: actual output/future/oracle/ground-truth/completion-time variables, side-effecting operations, randomness, imports/eval/exec.

Compiler/runtime:

- `compile_heuristic` verifies first, then returns `CompiledHeuristic`.
- `HeuristicPolicy` binds request/system/batch variables, scores queued requests, sorts by descending score and tie-breaker, greedily admits feasible requests.
- Regime selection falls back to `default`.
- Request scoring returns `0.0` on evaluation error.
- Admission condition fail-open returns `True` on evaluation error.

Verifier:

- Error codes include schema, forbidden variable/op, unknown variable/op, depth/node/term limits, constant limits, tie-breaker validation, non-finite expression, regime errors.

LLM generation subsystem:

- `llm_generation/generation_loop.py`: generate, verify, repair, deduplicate, archive.
- Providers include mock and configured provider wrappers; dry run uses mock.
- Candidate archive writes prompt, raw response, candidate JSON, verifier result, repair attempts, metadata, and index.
- Repair loop re-prompts with verifier errors.
- Deduplication is exact canonical JSON hash.
- `multi_regime_evaluation.py` runs synthetic train/validation/test regimes and aggregates old WG-style metrics.
- Historical shortlist freezing and held-out evaluation exist in Phase 2B.4 artifacts, but generated-heuristic shortlist predates corrected ANWG and is explicitly not wired in real-vLLM harness.

Current usability of generated heuristics under corrected objective:

- Previously generated heuristics are not currently validated as corrected-objective artifacts.
- `run_vllm_external_baseline_comparison.py` explicitly refuses `generated_heuristic` and `best_generated` because the shortlist predates objective correction and was never re-ranked under ANWG.

### DSL Capability Matrix

- Weighted sums of features: yes.
- Reusable named heuristic primitives: no, except through code-side genome helpers, not the DSL document schema.
- References to complete existing policies: no.
- Soft combinations of policy scores: no in DSL; yes in native `score_aggregation.py`.
- Rank aggregation: no in DSL; yes in native `composition.py`.
- Conditional mixtures: limited numeric `if_then_else`/regimes in DSL, not named policy mixtures.
- Scenario-dependent coefficients: only hardcoded constants in expressions/regimes; no external parameter binding.
- Runtime coefficient updates: no.
- Program templates with externally supplied parameters: not in executable DSL; code can generate JSON with constants.
- Safe fallback behavior: limited default regime and score/admission error fallback; no robust fallback policy/abstention in DSL.
- Uncertainty-aware execution: no.

## 6. Real-Serving Validation Audit

Real vLLM:

- Harness: `scripts/run_vllm_external_baseline_comparison.py`.
- Scope: client-side admission control into a fixed concurrency budget, issuing real HTTP requests to vLLM.
- Not scope: vLLM internal continuous batching/KV scheduling.
- Corrected selector is wired only when the local manifest-validated `PerPolicyRegressionAnwgSelector` artifact is supplied.
- Corrected scaled vLLM run in `docs/vllm_real_serving_scaled_comparison_corrected.md`: 3,780 real requests, Qwen/Qwen2.5-0.5B-Instruct, 540 requests per policy, selector completed 483/540 and had lowest ANWG point estimate but best conditional WG; all pairwise CIs included zero; server package version not recorded in the run manifest.

Hosted APIs:

- Harness: `scripts/run_hosted_policy_comparison.py`.
- Scope: Cohere/Gemini client-side admission comparisons with explicit `--allow-live-api` and cost/request/token caps.
- Mock and dry-run modes exist and are tested.
- This audit did not run hosted or real-vLLM live requests.

Faithful external baselines:

- Seven external/literature baselines are implemented as simulator policies and kept evaluation-only.
- Several have real-GPU/Wulver validation docs and benchmark packs, but this audit did not rerun GPU validation.

## 7. Mapping Intended Compositional Idea To Existing Code

Intended method:

1. Represent scheduling knowledge as reusable, verifiable heuristic components.
2. Given a workload scenario, estimate each component's usefulness.
3. Produce context-dependent weights or parameters.
4. Combine components into a new heuristic for that scenario.
5. Compile the composition into the verified DSL.
6. Execute it with safety checks and robust fallback.

Existing approximations:

- Reusable components: `CompositionModuleSpec`, `ScorpioAdmissionComponent`, `KVReservePlacementComponent`, `AdaptivePrefillGuardComponent`, `SchedulerGenomeV1` modules. Directly related, partially reusable.
- Usefulness estimation: `PolicyRewardRegressorSelector`, `PerPolicyRegressionAnwgSelector`, suitability/module-credit models. Directly related, reusable after refactoring from policy-level to component-level.
- Context-dependent weights: `ContextualRankEnsemblePolicy` hand rules, composition smart pilot proxy softmax/top-k weights, native pilot reward-model weights. Conceptually related, not canonical or DSL-compiled.
- Combine components: `StaticRankEnsemblePolicy`, `StaticScoreEnsemblePolicy`, `ComponentWiseCompositionPolicy`, `ConditionalRegimeCompositionPolicy`. Directly reusable as native prototypes.
- Compile into DSL: `SchedulerGenomeV1` compiles module expressions and conditional regimes into DSL. Reusable after extension; it does not compile rank/score ensembles over named policies.
- Safety/fallback: native composition logs fallback and uses deterministic fallback policy. DSL fallback is weaker.

Search-term classification:

- Policy-score prediction: implemented (`PolicyRewardRegressorSelector`, `PerPolicyRegressionAnwgSelector`), directly reusable.
- Per-policy regression: implemented, directly reusable.
- Weighted voting: pairwise ranker votes and native rank aggregation, reusable.
- Softmax weighting: implemented in native contextual rank ensemble and composition proxy, reusable after refactoring/training.
- Rank aggregation: implemented natively, directly reusable.
- Score normalization: implemented natively (`none`, `min_max`, `zscore`, `robust_mad`), directly reusable.
- Policy ensembles: implemented natively, reusable.
- Mixtures of experts: conceptually implemented as native ensembles, not full trained MoE.
- Gating networks: hand gates only; learned gate absent.
- Hypernetworks: absent.
- Primitive libraries: partial via capabilities/genome/module specs, reusable after canonical interface.
- Heuristic templates: partial via genome helpers and LLM prompts, insufficient for runtime parameterization.
- Contextual coefficients: hand-coded/proxy only, reusable after training interface.
- Policy switching: implemented in hard selectors and conditional composition.
- Regime gating: implemented.
- Hysteresis/minimum commitment: implemented in native contextual/conditional composition.
- Uncertainty estimates: selector prediction-margin fallback and suitability tree uncertainty exist, but not integrated into composition execution.
- OOD detection: conceptually discussed; no canonical runtime detector.
- Conservative fallback: native composition has deterministic fallback; selector wrappers have fixed fallback; DSL lacks robust policy fallback.
- Oracle mixture search: spec exists; no local decisive sweep artifacts.
- Composition opportunity analysis: proxy/local smoke and Wulver handoff exist; true opportunity gap not measured locally.

## 8. Directly Reusable Components

Best files to reuse:

- `src/llmserveopt/policies/composition.py`: rank experts, contextual weights, Borda/RRF aggregation, component-wise prototype, fallback/logging/hysteresis.
- `src/llmserveopt/policies/score_aggregation.py`: scalar score extraction, normalization, weighted score aggregation.
- `src/llmserveopt/policies/capabilities.py`: current capability metadata for rank/score/admission/DSL mapping.
- `src/llmserveopt/policies/genome.py`: typed genome, validation, canonicalization, compile-to-DSL.
- `src/llmserveopt/policies/structural_synthesis.py`: policy-to-genome mappings, module swap, conditional composition, mutation, prompt rendering.
- `src/llmserveopt/selector/advanced.py`: policy reward regressors, pairwise ranker, uncertainty fallback, regime gate.
- `src/llmserveopt/selector/dataset_v2/*`: full outcome vectors, discriminativeness, regrets, split hygiene.
- `src/llmserveopt/selector/composition_experiment.py`: development-only selection and split leakage guards.
- `tools/native_composition_pilot.py`: experiment pattern for true execution, but hardcoded to Wulver path and not locally portable.
- `tools/composition_score_rank_smoke.py`: small correctness smoke for score/rank composition.
- `scripts/persist_corrected_selector_artifact.py`: manifest pattern for objective/data/commit verification.

## 9. Missing Components

Dependency-ordered inventory:

1. Canonical primitive interface: absent. Extend `policies/capabilities.py` and `policies/composition.py` rather than creating a parallel registry.
2. Refactor existing policies to expose `rank_requests`, `score_requests`, `admission_filter`, `choose_gpu`, and component metadata: partially present as adapters, not native methods. Extend `BasePolicy` conservatively or add adapter protocol.
3. Score/rank normalization contract: score normalization exists; rank normalization exists. Need a formal manifest/schema and tests for scale/sign/monotonicity.
4. Separate admission, ranking, batching, GPU-assignment primitives: partially present in native components/genome. Need canonical module objects and state snapshot safety.
5. Compositional DSL extension: absent for named primitives, weights, and policy references. Extend `heuristics/dsl_schema.py`, `expressions.py`, `verifier.py`, `compiler.py`, and `policies/genome.py`.
6. Program templates with externally supplied parameters: absent. Add parameterized genome/DSL template schema, not ad hoc JSON string replacement.
7. Contextual weight predictor: policy-level predictors exist; component-level weight predictor absent. Reuse `selector/advanced.py` and `selector/module_credit/*`.
8. Offline oracle composition optimizer: spec exists, not implemented locally as a completed runner. Extend `tools/native_composition_pilot.py` or a new tool using `selector/composition_experiment.py` guards.
9. Composition training dataset: policy vectors exist; primitive/component counterfactual dataset absent. Extend Dataset v2 rows with primitive outputs and true composed-policy outcomes.
10. Regret-aware or ranking-aware training: present for policy selection; adapt to component/mixture weights.
11. Sparse/top-k gating: native implemented; need trained and manifest-frozen path.
12. Uncertainty estimation: partial in suitability/module-credit and prediction-margin fallback; not composition-integrated.
13. Fallback and abstention rules: native fallback exists; need verified DSL fallback policy/abstention semantics.
14. Switching hysteresis: native implemented; absent in DSL.
15. Adversarial workload generation: partial scenario search exists; needs composition-specific counterexamples.
16. Counterexample-guided repair: LLM DSL repair exists; not tied to simulator counterexamples or composition failures.
17. New simulator experiments: required for true mixture performance.
18. New real-trace experiments: required after simulator smoke.
19. Clean real-vLLM validation: required only after simulator evidence; current vLLM harness validates client-side admission, not full scheduler composition.

## 10. Scientific And Engineering Risks

- Branch risk: local branch is 18 commits behind live GitHub branch with the same name.
- Artifact risk: key trained model and raw CSVs are local-only under ignored `results/`.
- Objective risk: older code still defaults to completed-request WG in several paths.
- Split risk: docs identify stale/leaky calibrated-pilot non-OOD splits; do not treat older VALIDATION/ID_TEST claims as clean.
- Near-tie risk: corrected selector fresh validation is dominated by near-ties.
- Simulator discriminativeness risk: current docs report weak KV/cache/prefill/decode coupling and ANWG saturation for some datasets.
- Composition evidence risk: native composition has correctness tests and small/proxy pilots, but no local decisive oracle-mixture result.
- DSL expressiveness risk: verified DSL is intentionally narrow and cannot yet express complete policy primitive composition.
- Real-serving risk: current real-serving harnesses measure external admission, not internal scheduler behavior.

## 11. Reproducibility Problems

Critical results depending on gitignored local artifacts:

- Corrected selector artifact: committed script and tests exist; input CSVs, `.joblib`, and manifest are local-only. Manifest records objective, features, split, commit SHA, dirty flag, and SHA256 of input CSV. Fresh clone cannot reproduce without ignored `results/phase2b13...` and `results/phase2b16...`.
- Phase 2B.15/2B.16 corrected summaries: local-only under `results/`, not fresh-clone reproducible.
- Phase 2C final selector improvement: local-only result roots under `results/phase2c_final_selector_improvement`.
- Composition smart pilot: local-only and records dirty worktree from an older branch.
- Composition score/rank smoke: local-only output, but runner is committed and cheap to rerun.
- Native Wulver composition pilot: docs point to `/mmfs1/project/ikoutis/sv96/llmserveopt-data/native_composition_pilot_20260721T194929Z/`; not present locally.
- Module-credit Wulver import: local-only under `results/wulver_imports/`; runner/docs exist, but imported raw artifacts are not committed.
- Real trace raw/processed data: local-only under `data/raw` and `data/processed`.

Fresh clone reproducibility:

- Source/tests/configs are cloneable.
- Most high-value experimental raw results are not cloneable.
- `experiments/` contains curated committed results, but several important current CSVs are ignored inside `experiments/selector_v2_calibrated_pilot_20260720T163235Z/`.

## 12. Test And Validation Results

Commands run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest --collect-only -q -p no:cacheprovider
```

Result:

- Python 3.12.3, pytest 9.0.3
- Collected 2,901 tests in 0.57 seconds.

Focused test command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider \
  tests/test_selector_windows.py \
  tests/test_selector_features.py \
  tests/test_selector_labels.py \
  tests/test_selector_models.py \
  tests/test_selector_dataset_v2.py \
  tests/test_advanced_selector_models.py \
  tests/test_phase2b15_corrected_selector.py \
  tests/test_phase2b16_fresh_validation.py \
  tests/test_persist_corrected_selector_artifact.py \
  tests/test_heuristic_dsl_expressions.py \
  tests/test_heuristic_dsl_verifier.py \
  tests/test_heuristic_dsl_no_leakage.py \
  tests/test_heuristic_policy_wrapper.py \
  tests/test_heuristic_policy_feasibility.py \
  tests/test_llm_generation_dry_run.py \
  tests/test_candidate_deduplication.py \
  tests/test_llm_prompt_templates.py \
  tests/test_policy_composition.py \
  tests/test_score_and_reciprocal_rank_composition.py \
  tests/test_structural_synthesis.py \
  tests/test_policy_genome_coverage.py \
  tests/test_module_credit.py \
  tests/test_run_vllm_external_baseline_comparison.py \
  tests/test_run_hosted_policy_comparison.py
```

Result:

- `758 passed in 100.05s`
- No failures, no skips in this focused set.

Configuration/import checks:

```bash
python3 - <<'PY'
from pathlib import Path
import yaml
...
PY
```

- Parsed 65 YAML files under `configs/`.
- Bad YAML count: 0.

Import check:

- Imported `llmserveopt`, selector models/advanced/dataset_v2 builder, heuristic compiler/verifier, LLM generation loop, policy composition, score aggregation, structural synthesis, and external baseline registry successfully.

Compile check:

- `python3 -m compileall -q -x '(^|/)(results|experiments|data|logs|\.git|\.mypy_cache|\.pytest_cache|\.ruff_cache)(/|$)' src scripts tools tests`
- Exit code 0.

Not run:

- Full 2,901-test suite was not run because the focused 758-test set covered the requested selector, DSL, verifier, generation, composition, artifact, and real-serving harness surfaces while avoiding long/GPU-heavy coverage.
- No GPU tests.
- No live vLLM server calls.
- No hosted API calls.
- No new simulator experiments beyond unit tests.

## 13. Smallest Decisive Composition-Opportunity Experiment

Target comparison:

1. Best fixed heuristic.
2. Hard learned selector.
3. Oracle best fixed heuristic per window.
4. Best global weighted mixture.
5. Oracle best weighted mixture per window.

Key quantity:

```text
composition opportunity gap =
  oracle per-window mixture performance
  - oracle per-window fixed-policy performance
```

Existing artifacts are enough for:

- Best fixed policy under ANWG on a given vector table.
- Hard learned selector performance when predictions are recorded or model can be loaded.
- Oracle best fixed policy per window.

Existing artifacts are not enough for:

- True best global weighted mixture performance.
- True oracle best weighted mixture per window.

Why not enough:

- Policy reward-vector CSVs contain final metrics for already-run complete policies, not counterfactual outcomes under arbitrary score/rank mixtures.
- Most policies expose final actions through `select_action`, not a canonical primitive score/rank/admission API.
- Native adapters expose ranks for 12 experts and scores for 8 experts, but full policy behavior is not uniformly decomposed.
- Proxy composition smart-pilot rows combine policy-level vectors and predicted utilities; they are not true simulator execution of weighted mixtures.

Smallest correct experiment:

- Use a tiny but representative fixed set of existing request-window definitions from a committed or local Dataset v2 design.
- Freeze objective to `arrival_normalized_weighted_goodput`.
- Completion constraints: report both unconstrained ANWG and constrained variants with completion fraction thresholds 0.95 and 0.99, because selective admission/drop behavior confounds completed-request WG.
- Candidate experts: start with score-capable policies only: `fifo`, `edf`, `shortest_output_first`, `shortest_prompt_first`, `weighted_shortest_processing`, `estimated_service_time_first`, `least_laxity_first`, `slo_slack_score`.
- Methods: best fixed, corrected hard selector if artifact available, per-window fixed oracle, global grid/random weighted score mixture, per-window oracle weighted score mixture.
- Execute mixtures in the simulator, not by vector arithmetic.
- Use development split only for global mixture search; use held-out split only once for final reporting.

Expected computational cost:

- A small grid over 8 experts, top-k 2/3, and tens of windows is on the order of hundreds to low thousands of simulator runs.
- Existing smoke tests run quickly; native pilot estimated Wulver-scale few-thousand rows as single-node SLURM array sized, not a large training job.
- Do not run this during audit; it is the first next implementation/experiment layer.

## 14. Dependency-Ordered Implementation Roadmap

1. Synchronize to the authoritative live GitHub branch before coding.
2. Add a canonical `PolicyPrimitive` or adapter protocol for causal `score`, `rank`, `admission`, and `placement`.
3. Move existing rank/score adapters into that protocol and add tests for all currently declared capabilities.
4. Add a primitive-output trace schema and manifest: per request, per state, per primitive score/rank/admission decision.
5. Extend `SchedulerGenomeV1` and DSL schema to support named primitive references and externally supplied numeric parameters, or explicitly create a separate `CompositionGenomeV1` that compiles to plain DSL expressions when primitives are expression-backed.
6. Implement static weighted score/rank mixture compilation for the expression-backed subset.
7. Implement fallback and abstention semantics in the verified path.
8. Implement an offline oracle/global mixture optimizer that executes true composed policies in simulator.
9. Build the smallest composition-opportunity experiment above.
10. Only if the opportunity gap is positive, add contextual weight prediction trained on development-only composition outcomes.
11. Add uncertainty/fallback/OOD gates and hysteresis to the verified path.
12. Rerun real-trace simulator evaluation.
13. Only after simulator evidence is clean, perform real-vLLM client-side validation of the selected deployable composition.

## 15. Recommended Files To Modify Next Phase

Primary:

- `src/llmserveopt/policies/capabilities.py`
- `src/llmserveopt/policies/composition.py`
- `src/llmserveopt/policies/score_aggregation.py`
- `src/llmserveopt/policies/genome.py`
- `src/llmserveopt/policies/structural_synthesis.py`
- `src/llmserveopt/heuristics/dsl_schema.py`
- `src/llmserveopt/heuristics/expressions.py`
- `src/llmserveopt/heuristics/verifier.py`
- `src/llmserveopt/heuristics/compiler.py`
- `src/llmserveopt/selector/composition_experiment.py`
- `src/llmserveopt/selector/advanced.py`

Experiment/tooling:

- `tools/composition_score_rank_smoke.py`
- `tools/native_composition_pilot.py` or a new local-portable runner based on it.
- `docs/current/wolverine_oracle_mixture_spec.json`
- `tests/test_policy_composition.py`
- `tests/test_score_and_reciprocal_rank_composition.py`
- `tests/test_structural_synthesis.py`
- new tests for primitive protocol and DSL compilation.

Avoid duplicating:

- Dataset split guards in `selector/dataset_v2/splits.py`.
- Manifest/objective checks from `scripts/persist_corrected_selector_artifact.py`.
- Existing native aggregation logic.

## 16. Questions That Remain Unresolved

- Should the authoritative next branch be live `reality-grounded-dataset-expansion-20260724` at `775147b`, or a different integration branch after fetching?
- Are the 18 live GitHub commits documentation-only, or do they alter composition/source state?
- Should local-only corrected selector artifacts be promoted into a reproducible artifact bundle, or should the selector be regenerated from committed inputs?
- Which workload slice should be the first decisive composition-opportunity benchmark: Phase 2B.16 fresh, Phase 2C real-trace-derived, or a new tiny controlled Dataset v2 subset?
- Should the verified DSL be extended to call named primitives, or should all composition be compiled down to pure numeric expressions only for expression-backed primitives?
- What completion threshold should gate deployable composition claims: none, 0.95, 0.99, or workload-dependent?
- Should true mixture search focus first on score-capable experts only, or include rank-only experts via RRF/Borda from the beginning?

## 17. Next Codex Query Should Do

First, synchronize safely:

```text
Fetch the live GitHub refs, inspect the 18 commits currently ahead of local
reality-grounded-dataset-expansion-20260724, and decide the authoritative
starting commit without losing local ignored artifacts.
```

Then implement only the smallest missing layer:

```text
Add a canonical primitive adapter protocol for score/rank/admission/placement,
wire the existing score_aggregation and composition adapters through it,
and add tests proving the existing 8 score-capable and 12 rank-capable
experts expose stable causal primitive outputs without changing policy behavior.
```

Do not start the oracle-mixture experiment until that protocol exists.

## Direct Answers

Is the current local branch clean and synchronized?

- Clean for tracked files and staged changes before this report. Synchronized with the stale local upstream ref. Not synchronized with live GitHub: live branch is 18 commits ahead.

What branch should be treated as the authoritative starting point?

- The live GitHub `reality-grounded-dataset-expansion-20260724` branch should be treated as authoritative after inspecting/fetching its 18 newer commits. The audited local state is `21023c1`, but it is stale relative to GitHub.

Is the trained selector complete and usable?

- The corrected `regression_anwg` selector artifact is complete and usable locally. Scientifically, it is still caveated by local-only inputs, dirty creation commit, near-tie-dominated validation, and real-serving mismatch.

Is the corrected-objective selector artifact available and verifiable?

- Yes locally: `results/corrected_selector_artifact_regression_anwg/regression_anwg_selector.joblib` with `manifest.json`. The manifest verifies objective, features, split, input CSV SHA, commit, and fresh-validation reproduction. It is not committed.

Can existing heuristics currently be represented as composable primitives?

- Partially. Native adapters expose ranks for 12 experts, scores for 8, and admission for SCORPIO-style logic. Complete policies are not uniformly decomposed into canonical primitives.

Can the DSL currently express a contextual weighted composition?

- No. It can express weighted sums of feature expressions and conditional regimes, but not named policy primitives, policy-score calls, rank aggregation, learned contextual weights, or runtime parameter updates.

Is any true policy-composition mechanism already implemented?

- Yes as native Python prototypes (`composition.py`, `score_aggregation.py`, component-wise and conditional composition). No as a complete verified DSL-compiled contextual composition method.

What is the smallest missing layer between the current repository and the intended method?

- A canonical primitive adapter/protocol that exposes causal score/rank/admission/placement outputs and can be compiled or wrapped into a verified DSL/genome representation.

Can the composition-opportunity gap be measured without major refactoring?

- Not exactly from existing artifacts alone. A small true-execution simulator experiment can measure it for the currently score/rank-capable subset with modest new code, but policy-vector CSV arithmetic is insufficient.

What exact next implementation step should be performed first?

- Add and test the canonical primitive interface by refactoring existing `rank_with_named_expert`, `score_with_named_expert`, and component admission/placement helpers into a single audited adapter layer, with no behavior change.
