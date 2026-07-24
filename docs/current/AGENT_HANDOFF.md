# Agent Handoff

Short operational handoff for the current Wulver integration branch. For the
full story, read `PROJECT_STATUS.md` first.

## Start Here

1. [`README.md`](README.md) - current-doc navigation.
2. [`PROJECT_STATUS.md`](PROJECT_STATUS.md) - canonical scientific state.
3. [`ROADMAP_GAP_ANALYSIS.md`](ROADMAP_GAP_ANALYSIS.md) - evidence-ranked bottlenecks.
4. [`RESEARCH_ROADMAP.md`](RESEARCH_ROADMAP.md) - staged next plan.

## Current Scientific State

- The deployable V2 library has 27 policies.
- V2 real-OOD policy vectors show strong oracle-envelope expansion:
  `+0.008904` ANWG, about `3.54%` relative, CI `[0.008191, 0.009646]`.
- The 27-policy selector benchmark is complete and useful, but top-1 selection
  still does not meaningfully capture the V2 OOD oracle gain.
- Native composition failed its decision bar.
- Module intervention has sparse positive transfer, but module-credit learning
  remains weak and pairwise interventions did not expand the V2 envelope.
- SwissAI and TraceLab add raw workload novelty but saturated ANWG and produced
  zero strict V2 marginal gain.
- SLO/deadline augmentation adds useful synthetic pressure and class-balance
  support, but it is not natural real-OOD evidence.

## Current Bottleneck

The primary bottleneck is simulator/objective discriminative power. Important
workload differences, especially KV/cache reuse, long context,
prefill/decode structure, and missing/neutral SLOs, are not consistently
translated into resource pressure and policy reward separation.

Current audit verdicts:

- `KV_CACHE_COUPLING_VERDICT = WEAK_DIRECT_COUPLING`
- `PREFILL_DECODE_COUPLING_VERDICT = PARTIAL_AND_WEAK_UNDER_CURRENT_WORKLOADS`
- `COMBINER_TRAINING_SIGNAL = WEAK`
- `COMBINER_EVALUATION_READINESS = NEEDS_SIMULATOR_FIX`

## Exact Next Recommended Task

Run a bounded simulator calibration and discriminative-power validation task.

Do this before:

- collecting more generic datasets;
- retraining the 27-policy selector;
- refreshing module-credit/intervention data;
- launching composition or structural synthesis.

The calibration task should explicitly test and, in a separate development
phase, fix coupling from:

- prefix/cache reuse to prefill/service cost;
- KV occupancy and capacity pressure;
- prefill/decode contention;
- overload/capacity constraints;
- SLO feasibility and deadline pressure;
- ANWG ceiling behavior.

## Source-Of-Truth Constants

```text
POLICY_LIBRARY_V1_COUNT = 20
POLICY_LIBRARY_V2_NEW_COUNT = 7
POLICY_LIBRARY_V2_COUNT = 27
```

The current deployable policy registry lives in
`src/llmserveopt/policies/registry.py`.

## Protected Evidence

Completed Wulver roots under `/mmfs1/project/ikoutis/sv96/llmserveopt-data/`
are scientific provenance. Do not rewrite them. Create new derived roots for
new audits or summaries.

See [`ACTIVE_EXPERIMENT_PROTECTED_PATHS.md`](ACTIVE_EXPERIMENT_PROTECTED_PATHS.md)
and [`EXPERIMENT_INDEX.md`](EXPERIMENT_INDEX.md).

## Do-Not-Do List

- Do not push from cleanup queries unless explicitly requested.
- Do not merge into `main` without a separate review step.
- Do not treat SwissAI/TraceLab zero-gain results as proof that FIFO is
  naturally optimal for those workloads.
- Do not treat synthetic SLO augmentation as natural real-OOD evidence.
- Do not launch broad synthesis while combiner training/evaluation signal is
  weak.
- Do not use `weighted_goodput` as the primary objective; use
  arrival-normalized weighted goodput/ANWG where applicable.
- Do not trust a bare `pytest` invocation if the environment differs; use the
  repository Python environment and record the exact command.
