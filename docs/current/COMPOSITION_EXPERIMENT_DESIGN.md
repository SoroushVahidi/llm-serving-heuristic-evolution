# Composition Experiment Design

> **SUPERSEDED FOR CURRENT STATUS.**
> See [`docs/current/RESUME_HERE.md`](RESUME_HERE.md) for authoritative current state.
> Composition was subsequently formally `COMPOSITION_DEMOTED` (`docs/audits/reassessment_composition_hypothesis_20260817.md`, commit `dc5757b`).

This document defines the decisive composition experiment to run after both upstream workflows finish:

- `/mmfs1/project/ikoutis/sv96/llmserveopt-data/policy_frontier_cartography_20260721T154408Z`
- `/mmfs1/project/ikoutis/sv96/llmserveopt-data/policy_library_v2_expanded_20260721T171933Z`

The full experiment must not start until both final reports exist. The readiness harness is implemented now so expert selection and evaluation can be launched immediately afterward.

## Treatments

| ID | Treatment | Implementation path | Selection rule |
| --- | --- | --- | --- |
| A | Best fixed policy | Existing policy registry | Select by TRAIN/VALIDATION/ROBUST_DEV only |
| B | Discrete selector | Existing selector model path | Frozen selector chosen by development evidence only |
| C | Static normalized-rank ensemble | `StaticRankEnsemblePolicy` | Expert weights selected from development evidence only |
| D | Contextual normalized-rank ensemble | `ContextualRankEnsemblePolicy` | Causal state-weight model selected from development folds only |
| E | Component-wise composition | `ComponentWiseCompositionPolicy` | SCORPIO admission + WSP priority + supported guards |

Treatment A must be determined from development data. WSP is the expected first candidate but is not hard-coded as the scientific answer.

Treatment B chooses one complete policy per window. Treatments C and D compose compatible priority outputs after rank normalization. Treatment E composes typed modules and must not claim support for unavailable action semantics.

## Implemented Harness

The experimental composition interface is in `src/llmserveopt/policies/composition.py`.

Implemented pieces:

- typed module specs for admission, priority/ranking, batching/prefill, placement, preemption, KV/cache, and fairness/aging;
- normalized rank adapters for compatible causal priority experts;
- static weighted normalized-rank ensemble;
- contextual rank ensemble with deterministic causal feature rules and a placeholder for trained weights;
- component-wise SCORPIO-admission + WSP-priority prototype;
- deterministic fallback policy;
- feasibility projection through standard policy feasibility checks;
- sparse `top_k` expert selection;
- weight entropy logging;
- switching-frequency logging;
- optional minimum commitment/hysteresis for contextual weights;
- invalid composition detection.

The current implementation deliberately does not register composed policies as production baselines.

## Normalization

Raw score averaging is not a valid default because policies emit heterogeneous objects: ranks, private scores, binary admission filters, placement decisions, and direct actions. The safe representation is:

1. Generate a deterministic ranking over the current waiting set for each compatible expert.
2. Convert each ranking to normalized ranks in `[0, 1]`.
3. Weight and aggregate normalized ranks.
4. Project the resulting request order through admission, feasibility, and placement constraints.

Requests that an expert cannot score are logged as missing for that expert. Requests missing from all active experts are not admitted by the ensemble and trigger fallback if no valid action remains.

## Contextual Weights

The contextual ensemble supports weights based only on causal state features:

- system pressure;
- queue pressure;
- KV pressure;
- decode pressure;
- prefill pressure;
- mean prompt tokens in the current waiting queue;
- mean predicted output tokens in the current waiting queue;
- urgent deadline fraction in the current waiting queue.

The hand-coded rule provided now is a correctness placeholder. The final trained contextual weighting model must be selected only from TRAIN/VALIDATION/ROBUST_DEV or comparable group-aware development folds produced by the completed upstream workflows.

## Component-Wise Prototype

The first component-wise prototype is valid under current simulator semantics because it composes:

- SCORPIO-style admission filtering and budget throttling;
- WSP-style request priority over surviving candidates;
- KV reserve-aware placement;
- adaptive prefill pressure guard;
- aging tiebreak.

It does not implement:

- prefix/cache reuse;
- cache loading;
- true action-level chunked prefill;
- request splitting;
- disaggregated prefill/decode routing;
- heterogeneous GPU routing.

## Downstream Selection Pipeline

After upstream reports exist, the full workflow should:

1. Read Policy Frontier final artifacts to identify high-value WSP/SCORPIO and other policy-boundary regions.
2. Read Policy Library v2 final artifacts to identify new policies with meaningful unique wins and non-redundant reward profiles.
3. Select expert policies using development-only data.
4. Pre-register expert sets, rank operators, contextual features, and component modules.
5. Run leakage audits before training/evaluation.
6. Train/tune only on TRAIN/VALIDATION/ROBUST_DEV.
7. Freeze candidates.
8. Evaluate on ID_TEST, TEMPORAL_OOD, CROSS_SOURCE_OOD, and FINAL_OOD.

## Final Evaluation Protocol

Use group-aware, leakage-safe splits with atomic `split_group_key` or equivalent source/time groups. Test/OOD labels must never be used to select:

- best fixed policy;
- discrete selector;
- expert policies;
- expert weights;
- contextual model hyperparameters;
- fallback thresholds;
- component-wise guard thresholds.

Required metrics:

- ANWG/current primary objective;
- regret versus best fixed policy;
- regret versus hindsight oracle;
- SLO violation rate;
- throughput/utilization where supported;
- fairness/starvation metrics where supported;
- feasibility violations;
- switching frequency;
- composition weight entropy;
- meaningful-window-only metrics;
- bootstrap confidence intervals.

## Launch Readiness

Prepared but intentionally unsubmitted:

```bash
sbatch tools/composition_experiment_when_ready.sbatch
```

That script exits without launching the full experiment unless both upstream final reports exist.

## Main Scientific Risk

The main risk is that rank composition may smooth over sharp policy-frontier boundaries instead of improving them. The experiment is only successful if held-out and OOD gains persist against the fixed WSP baseline and the discrete selector under leakage-safe evaluation.
