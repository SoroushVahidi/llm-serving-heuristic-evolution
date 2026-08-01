# Resume Contextual Composition

Use this file to resume the contextual-compositional heuristic branch.

## Checkout

```bash
git checkout contextual-compositional-heuristics-20260731
```

- Authoritative branch: `contextual-compositional-heuristics-20260731`
- Expected checkpoint SHA: verify against the Query 6 final result's `New SHA`
  with `git rev-parse HEAD`
- Starting SHA before the checkpoint commit:
  `db4dcaa40abe1312ea71c40c440445172cd1c509`

## Read In Order

1. [START_HERE_CONTEXTUAL_COMPOSITION.md](START_HERE_CONTEXTUAL_COMPOSITION.md)
2. [contextual_composition_roadmap.md](contextual_composition_roadmap.md)
3. [contextual_composition_decisions.md](contextual_composition_decisions.md)
4. [audits/contextual_composition_pause_checkpoint_20260731.md](audits/contextual_composition_pause_checkpoint_20260731.md)
5. [audits/contextual_composition_query5_discriminativeness_review_20260731.md](audits/contextual_composition_query5_discriminativeness_review_20260731.md)
6. GitHub issue #2

## Verify State

```bash
git status --short --branch
git rev-parse HEAD
git rev-parse --abbrev-ref --symbolic-full-name @{u}
git rev-list --left-right --count @{u}...HEAD
python scripts/check_contextual_composition_status.py
```

The expected state is a clean working tree, upstream
`origin/contextual-compositional-heuristics-20260731`, `0` ahead and `0`
behind, and a passing contextual-composition status checker.

## Current Phase

- Current phase: `CC2 - Canonical primitive interface`
- Current status: `NEXT`
- Decision gate: CC1b passed with verdict `PROCEED`; CC2 may start after the
  pause is lifted.

## Exact Next Implementation Task

Define the canonical primitive interface for ranking, admission, placement, batching, and resource guards, then add representative-policy equivalence tests. Do not extend the DSL yet.

## Do Not Start Prematurely

Do not begin CC3 DSL changes, CC4 dataset generation, CC5 predictor training,
CC6 adaptation, CC7 hardening, CC8 real-serving validation, hosted API jobs,
GPU jobs, real-vLLM jobs, or new experiments before the roadmap gates allow
them.

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

## GitHub

Continue with GitHub issue
[#2](https://github.com/SoroushVahidi/llm-serving-heuristic-evolution/issues/2).
Issue #1 is the completed CC1/CC1b evidence gate.
