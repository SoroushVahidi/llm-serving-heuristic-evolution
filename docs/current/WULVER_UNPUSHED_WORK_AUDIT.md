# Wulver Unpushed Work Audit

Generated during Query 1 cleanup audit on 2026-07-21.

## Current Git State

- Current branch: `wulver-policy-composition-readiness`
- Current commit: `c8aee129f553f8dc3ede99eac60d5b14484beb41`
- Upstream configured for current branch: none
- Closest remote source lineage: `origin/repo-polish-query5-final-verification` at `8087ee1`
- Local commit `c8aee12` is one commit ahead of `origin/repo-polish-query5-final-verification`.
- Worktree was dirty before cleanup and remains dirty because important Wulver-only research files are intentionally uncommitted.

Query 2 integration uses a separate worktree:

- Final integration branch: `wulver-final-integration-20260721`
- Final integration worktree: `/mmfs1/project/ikoutis/sv96/github/llm-serving-heuristic-evolution-final-integration`
- Starting commit: `c8aee129f553f8dc3ede99eac60d5b14484beb41`

This protects active SLURM workflows that import code from the original checkout path.

## Important Safety Finding

Before Query 2 topic commits, yes: important Wulver-only work would be lost if the current worktree were reset. Do not run `git reset --hard`, `git clean`, or checkout another branch in the original active-job checkout.

After Query 2 topic commits, the important source work should be preserved on `wulver-final-integration-20260721`; the original active-job checkout may remain dirty as a protected copy until running workflows finish.

## KEEP_AND_COMMIT

These are source, tests, docs, and scripts that should be preserved in Git during Query 2 integration.

### Policy Library v2

- `src/llmserveopt/policies/__init__.py`
- `src/llmserveopt/policies/registry.py`
- `src/llmserveopt/policies/sola_style_state_aware.py`
- `src/llmserveopt/policies/slai_style_phase_aware.py`
- `src/llmserveopt/policies/flow_control_stability.py`
- `src/llmserveopt/policies/kv_constrained_online.py`
- `src/llmserveopt/policies/adaptive_chunked_prefill.py`
- `src/llmserveopt/policies/aging_priority.py`
- `src/llmserveopt/policies/weighted_fair_share.py`
- `src/llmserveopt/policies/policy_library_v2_helpers.py`
- `tests/test_policy_library_v2.py`
- `tools/policy_library_v2_experiment.py`
- `docs/policy_library_v2.md`

### Policy Composition Readiness and Harness

- `src/llmserveopt/policies/composition.py`
- `src/llmserveopt/selector/composition_experiment.py`
- `tests/test_policy_composition.py`
- `tools/composition_smoke_experiment.py`
- `tools/composition_harness_tests.sbatch`
- `tools/composition_experiment_when_ready.sbatch`
- `tools/native_composition_pilot.py`
- `tools/native_composition_pilot.sbatch`
- `docs/current/POLICY_COMPOSITION_READINESS.md`
- `docs/current/COMPOSITION_EXPERIMENT_DESIGN.md`
- `docs/current/COMPOSITION_IMPLEMENTATION_STATUS.md`
- `docs/current/composition_experiment_schema.json`
- `docs/current/composition_hypotheses.json`
- `docs/current/policy_component_matrix.json`
- `docs/current/composable_primitives.json`
- `docs/current/composition_operators.json`
- `docs/current/policy_complementarity.json`

### Structural Synthesis Readiness

- `src/llmserveopt/policies/genome.py`
- `src/llmserveopt/policies/structural_synthesis.py`
- `src/llmserveopt/selector/parent_selection.py`
- `tests/test_structural_synthesis.py`
- `tools/structural_synthesis_tests.sbatch`
- `docs/current/STRUCTURAL_SYNTHESIS_READINESS.md`
- `docs/current/scheduler_genome_v1.schema.json`
- `docs/current/structural_synthesis_experiment_design.md`

### Query 1 Audit Deliverables

- `docs/current/ACTIVE_EXPERIMENT_PROTECTED_PATHS.md`
- `docs/current/WULVER_UNPUSHED_WORK_AUDIT.md`
- `docs/current/WULVER_BRANCH_LINEAGE_AUDIT.md`
- `docs/current/EXPERIMENT_INDEX.md`
- `docs/current/README.md`

## KEEP_LOCAL_ARTIFACT

These should remain outside Git and should not be moved or deleted during cleanup.

- `/mmfs1/project/ikoutis/sv96/llmserveopt-data/selector_v2_overnight_20260720T235405`
- `/mmfs1/project/ikoutis/sv96/llmserveopt-data/selector_v2_ood_conclusive_20260721T133408Z`
- `/mmfs1/project/ikoutis/sv96/llmserveopt-data/selector_v3_multidomain_causal_20260721T151341Z`
- `/mmfs1/project/ikoutis/sv96/llmserveopt-data/policy_frontier_cartography_20260721T154408Z`
- `/mmfs1/project/ikoutis/sv96/llmserveopt-data/policy_library_v2_expanded_20260721T171933Z`
- `/mmfs1/project/ikoutis/sv96/llmserveopt-data/native_composition_pilot_20260721T194929Z`
- `/mmfs1/project/ikoutis/sv96/llmserveopt-data/composition_harness_readiness`
- `/mmfs1/project/ikoutis/sv96/llmserveopt-data/structural_synthesis_readiness`
- repository-local `logs/` historical SLURM/runtime logs
- repository-local ignored `data/raw/` and `data/processed/` trees

## ARCHIVE

No repository-root files were moved to an archive in Query 1. The current evidence does not justify moving historical scientific logs or docs before the Query 2 documentation rewrite.

## DELETE_SAFE

Deleted during Query 1:

- `.pytest_cache/`
- Python `__pycache__/` directories under `scripts/`, `src/`, `tests/`, and `tools/`

No source files, final reports, manifests, logs, or experiment outputs were deleted.

## UNCERTAIN

No currently visible untracked source files were classified as uncertain. The risk is not ambiguity; the risk is that the untracked files are important and must be committed deliberately.

## Ignored Artifact Policy

`.gitignore` already covers Python caches, pytest caches, repository-local logs, large raw/processed data, binary/model artifacts, and parquet files. No `.gitignore` change is required in Query 1.
