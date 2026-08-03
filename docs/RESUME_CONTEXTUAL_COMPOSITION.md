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
11. GitHub issue #4

## Verify State

```bash
git status --short --branch
git rev-parse HEAD
git rev-parse --abbrev-ref --symbolic-full-name @{u}
git rev-list --left-right --count @{u}...HEAD
python scripts/check_contextual_composition_status.py
python scripts/check_contextual_composition_status.py --resume-readiness
python -m pytest tests/test_contextual_composition_status_checker.py tests/test_cc1_composition_opportunity.py tests/test_policy_composition.py tests/test_score_and_reciprocal_rank_composition.py tests/test_primitive_interface.py tests/test_primitive_reconstructed_policies.py -q
```

The expected state is a clean working tree, upstream
`origin/contextual-compositional-heuristics-20260731`, `0` ahead and `0`
behind, and a passing contextual-composition status checker.

## Current Phase

- Current phase: `CC4 - Offline oracle composition dataset`
- Current status: `NEXT` (queued, not started)
- Decision gate: CC3's exit gate passed (8/8 required constructs
  implemented, 447 focused+regression tests, legacy compatibility
  preserved); CC4 may start, but only in a separate, explicitly authorized
  query.

## Exact Next Implementation Task

CC4 has **not** been started. A future, explicitly authorized query should
begin it by reading
[architecture/contextual_composition_dsl.md](architecture/contextual_composition_dsl.md)
and
[audits/contextual_composition_cc3_dsl_verifier_report_20260803.md](audits/contextual_composition_cc3_dsl_verifier_report_20260803.md)
first, then search for high-quality composition parameters through true
simulator execution for each training window, sampling/mutating over
`CompiledHeuristic.primitive_refs`/`placement_keys`/`admission_budget_spec`/
`param_declarations` (the causal-input surface CC3 exposes) per the
roadmap's CC4 required comparisons.

## Do Not Start Prematurely

Do not begin CC4 dataset generation in this same query even though CC3's
gate passed. Do not begin CC5 predictor training, CC6 adaptation, CC7
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

## GitHub

Continue with GitHub issue #4 (only in a separate, explicitly authorized
query):
[#4](https://github.com/SoroushVahidi/llm-serving-heuristic-evolution/issues/4).
Issue #1 is the completed CC1/CC1b evidence gate; issue #2 is the completed
CC2 primitive interface gate; issue #3 is the completed CC3 DSL/verifier gate.
