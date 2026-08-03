# Resume Contextual Composition

Use this file to resume the contextual-compositional heuristic branch.

## Checkout

```bash
git checkout contextual-compositional-heuristics-20260731
```

- Authoritative branch: `contextual-compositional-heuristics-20260731`
- Query 6 checkpoint SHA: `f6b4be9dc15fc4f13286f23b5aae39f48fbd01fb`
- CC2 (Query 8) checkpoint SHA: verify against the CC2 primitive interface
  report's `New SHA` with `git rev-parse HEAD`
- Starting SHA before the CC2 checkpoint commit:
  `4d806c8b1be0c4c9e202bbc7a20b3455c9c510b8`
- CC3 (Query 9) checkpoint SHA: verify against the CC3 DSL/verifier
  report's `New SHA` with `git rev-parse HEAD`
- Starting SHA before the CC3 checkpoint commit:
  `ed85e585bb42a37f47530939b1d2d11bb1ea0b3e`
- CC4 (Query 10) checkpoint SHA: verify against the CC4 oracle dataset
  report's `New SHA` with `git rev-parse HEAD`
- Starting SHA before the CC4 checkpoint commit:
  `19708f741d0bfb944b4a11ff34572a811df94d66`

## Read In Order

1. [START_HERE_CONTEXTUAL_COMPOSITION.md](START_HERE_CONTEXTUAL_COMPOSITION.md)
2. [contextual_composition_roadmap.md](contextual_composition_roadmap.md)
3. [contextual_composition_decisions.md](contextual_composition_decisions.md)
4. [audits/contextual_composition_pause_checkpoint_20260731.md](audits/contextual_composition_pause_checkpoint_20260731.md)
5. [audits/contextual_composition_query5_discriminativeness_review_20260731.md](audits/contextual_composition_query5_discriminativeness_review_20260731.md)
6. [audits/contextual_composition_query7_final_pause_readiness_20260731.md](audits/contextual_composition_query7_final_pause_readiness_20260731.md)
7. [architecture/contextual_composition_primitives.md](architecture/contextual_composition_primitives.md)
8. [audits/contextual_composition_cc2_primitive_interface_report_20260802.md](audits/contextual_composition_cc2_primitive_interface_report_20260802.md)
9. [architecture/contextual_composition_dsl.md](architecture/contextual_composition_dsl.md)
10. [audits/contextual_composition_cc3_dsl_verifier_report_20260803.md](audits/contextual_composition_cc3_dsl_verifier_report_20260803.md)
11. [audits/contextual_composition_cc4_oracle_dataset_report_20260803.md](audits/contextual_composition_cc4_oracle_dataset_report_20260803.md)
12. GitHub issue #5

## Verify State

```bash
git status --short --branch
git rev-parse HEAD
git rev-parse --abbrev-ref --symbolic-full-name @{u}
git rev-list --left-right --count @{u}...HEAD
python scripts/check_contextual_composition_status.py
python scripts/check_contextual_composition_status.py --resume-readiness
python -m pytest tests/test_contextual_composition_status_checker.py tests/test_cc1_composition_opportunity.py tests/test_policy_composition.py tests/test_score_and_reciprocal_rank_composition.py tests/test_primitive_interface.py tests/test_primitive_reconstructed_policies.py tests/test_contextual_composition_cc3_dsl.py tests/test_cc4_oracle_composition_dataset.py -q
```

The expected state is a clean working tree, upstream
`origin/contextual-compositional-heuristics-20260731`, `0` ahead and `0`
behind, and a passing contextual-composition status checker.

## Current Phase

- Current phase: `CC5 - Contextual composition predictor`
- Current status: `NEXT` (queued, not started)
- Decision gate: CC4's exit gate passed (12 windows/34 candidates/408
  simulator executions, 0 rejected, reproducible+resumable, 66.7%
  evaluation-window composition-oracle gain, completion constraints hold on
  all windows); CC5 may start, but only in a separate, explicitly
  authorized query.

## Exact Next Implementation Task

CC5 has **not** been started. A future, explicitly authorized query should
begin it by reading
[audits/contextual_composition_cc4_oracle_dataset_report_20260803.md](audits/contextual_composition_cc4_oracle_dataset_report_20260803.md)
(its "Exact CC5 Entry Condition" section) first, then train against
`oracle_labels.parquet`/`regret_matrix.parquet`/`causal_features.parquet`
(joined on `window_id`), fitting only on `development_splits` windows and
reserving `evaluation_splits` windows exclusively for the reported
validation claim.

## Do Not Start Prematurely

Do not begin CC5 predictor training in this same query even though CC4's
gate passed. Do not begin CC6 adaptation, CC7 hardening, CC8 real-serving
validation, hosted API jobs, GPU jobs, real-vLLM jobs, or new experiments
before the roadmap gates allow them.

## CC1b Evidence

Primary local result directory:

```text
results/cc1b_composition_discriminative/query5_cc1b_full_20260731/
```

Key files:

- `manifest.json`
- `verdict.json`
- `method_comparison.csv`
- `per_window_summary.csv`
- `composition_weights.csv`
- `cc1_report.md`

Regenerate only if needed:

```bash
python scripts/run_cc1_composition_opportunity.py \
  --config configs/cc1b_composition_discriminative.yaml \
  --dry-run

python scripts/run_cc1_composition_opportunity.py \
  --config configs/cc1b_composition_discriminative.yaml \
  --full-run
```

Do not use live APIs, GPU jobs, or real-vLLM jobs for this evidence.

## CC4 Evidence

Primary local result directory (untracked, per repository convention --
regenerate via `replay_commands.sh` inside it):

```text
results/cc4_oracle_composition_dataset/20260803T170735Z/
```

Key files: `manifest.json`, `dataset_card.md`, `oracle_labels.parquet`,
`regret_matrix.parquet`, `causal_features.parquet`, `search_summary.csv`.

Regenerate only if needed:

```bash
python scripts/run_cc4_oracle_composition_dataset.py \
  --config configs/cc4_oracle_composition_dataset.yaml \
  --dry-run

python scripts/run_cc4_oracle_composition_dataset.py \
  --config configs/cc4_oracle_composition_dataset.yaml \
  --full-run --allow-dirty --timestamp <new_timestamp>
```

Do not use live APIs, GPU jobs, or real-vLLM jobs for this evidence.

## GitHub

Continue with GitHub issue #5 (only in a separate, explicitly authorized
query):
[#5](https://github.com/SoroushVahidi/llm-serving-heuristic-evolution/issues/5).
Issue #1 is the completed CC1/CC1b evidence gate; issue #2 is the completed
CC2 primitive interface gate; issue #3 is the completed CC3 DSL/verifier
gate; issue #4 is the completed CC4 oracle dataset gate.
