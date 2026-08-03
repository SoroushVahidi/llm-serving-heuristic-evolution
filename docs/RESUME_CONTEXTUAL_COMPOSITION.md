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
- CC5 (Query 11) checkpoint SHA: verify against the CC5 predictor
  report's `New SHA` with `git rev-parse HEAD` (CC5's exit gate did NOT
  pass -- this checkpoint is an attempted-and-INCONCLUSIVE checkpoint, not
  a completion checkpoint)
- Starting SHA before the CC5 checkpoint commit:
  `db143fc7aef5cb604ed56b778b948b5d4f271891`

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
12. [audits/contextual_composition_cc5_predictor_report_20260803.md](audits/contextual_composition_cc5_predictor_report_20260803.md)
13. GitHub issue #5

## Verify State

```bash
git status --short --branch
git rev-parse HEAD
git rev-parse --abbrev-ref --symbolic-full-name @{u}
git rev-list --left-right --count @{u}...HEAD
python scripts/check_contextual_composition_status.py
python scripts/check_contextual_composition_status.py --resume-readiness
python -m pytest tests/test_contextual_composition_status_checker.py tests/test_cc1_composition_opportunity.py tests/test_policy_composition.py tests/test_score_and_reciprocal_rank_composition.py tests/test_primitive_interface.py tests/test_primitive_reconstructed_policies.py tests/test_contextual_composition_cc3_dsl.py tests/test_cc4_oracle_composition_dataset.py tests/test_cc5_contextual_predictor.py -q
```

The expected state is a clean working tree, upstream
`origin/contextual-compositional-heuristics-20260731`, `0` ahead and `0`
behind, and a passing contextual-composition status checker.

## Current Phase

- Current phase: `CC5 - Contextual composition predictor`
- Current status: `NEXT` (attempted, exit gate NOT passed -- verdict `INCONCLUSIVE`)
- Decision gate: CC4's exit gate passed and CC5 was attempted against it;
  the trained predictor ties the best fixed policy on CC4's 6 evaluation
  windows (mean ANWG 0.2306 vs 0.2310) and is beaten by
  `best_global_composition` (0.2633). Judged a data-scarcity finding (n=6
  held-out windows cannot statistically distinguish these methods), not a
  methodology failure -- see the CC5 predictor report.

## Exact Next Implementation Task

CC5's exit gate did not pass, so it must be **retried**, not begun fresh. A
future, explicitly authorized query should first expand the CC4 dataset
(more windows, more per regime -- read
[audits/contextual_composition_cc5_predictor_report_20260803.md](audits/contextual_composition_cc5_predictor_report_20260803.md)
section 10 for the exact rationale), then retrain against the larger
dataset using the existing, tested CC5 pipeline
(`src/llmserveopt/experiments/cc5_contextual_predictor.py`,
`scripts/run_cc5_contextual_predictor.py --dataset-dir <new CC4 dir>
--full-run`) -- no code changes are anticipated to be required.

## Do Not Start Prematurely

Do not retry CC5 predictor training in this same query. Do not begin CC6
adaptation until CC5 actually passes its exit gate. Do not begin CC7
hardening, CC8 real-serving validation, hosted API jobs, GPU jobs,
real-vLLM jobs, or new experiments before the roadmap gates allow them.

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

## CC5 Evidence

Primary local result directory (untracked, per repository convention --
regenerate via `replay_commands.sh` inside it):

```text
results/cc5_contextual_composition_predictor/20260803T175456Z/
```

Key files: `manifest.json`, `verdict.json`, `model_card.md`,
`cv_model_selection.csv`, `per_window_predictions.csv`,
`uncertainty_ood_diagnostics.csv`, `fallback_analysis.csv`.

Regenerate only if needed (against the same CC4 dataset -- a real retry
needs a larger CC4 dataset first, per the exact next task above):

```bash
python scripts/run_cc5_contextual_predictor.py \
  --dataset-dir results/cc4_oracle_composition_dataset/20260803T170735Z \
  --dry-run

python scripts/run_cc5_contextual_predictor.py \
  --dataset-dir results/cc4_oracle_composition_dataset/20260803T170735Z \
  --full-run --timestamp <new_timestamp>
```

Do not use live APIs, GPU jobs, or real-vLLM jobs for this evidence.

## GitHub

Continue with GitHub issue #5 (only in a separate, explicitly authorized
query -- it remains OPEN, not closed, since CC5's exit gate did not pass):
[#5](https://github.com/SoroushVahidi/llm-serving-heuristic-evolution/issues/5).
Issue #1 is the completed CC1/CC1b evidence gate; issue #2 is the completed
CC2 primitive interface gate; issue #3 is the completed CC3 DSL/verifier
gate; issue #4 is the completed CC4 oracle dataset gate.
