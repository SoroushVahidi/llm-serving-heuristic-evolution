# Composition and Synthesis Architecture

This document is the concise source-of-truth overview for the path:

```text
Policy Library -> Discrete Selection -> Contextual Composition -> Structural Synthesis
```

It summarizes what is implemented, what is prototype-only, and what is
currently blocked by simulator/objective discriminative power.

## Implemented Components

### Policy Library

`src/llmserveopt/policies/registry.py` exposes two deployable libraries:

- `BASELINE_NAMES`: historical 20-policy library.
- `POLICY_LIBRARY_V2_NAMES`: historical library plus 7 new deployable approximation policies.

Policy Library v2 is implemented and unit-tested. Real-OOD evidence shows the
V2 library materially improves the oracle envelope, but SwissAI and TraceLab
also show that raw workload novelty can collapse to near-tie policy rewards
under the current simulator/objective.

### Discrete Selection

The existing selector path chooses one complete deployable policy per window/context. Prior Selector v2/v3 evidence shows:

- adaptive selection can beat WSP in-distribution on the 1600-window Selector v2 dataset;
- robust held-out/OOD performance still tends to favor fixed WSP;
- richer causal features make WSP-vs-SCORPIO behavior more learnable, but have not yet produced a selector that consistently beats WSP across held-out domains.

### Rank Composition

`src/llmserveopt/policies/composition.py` implements typed composition scaffolding:

- `RankExpertSpec`
- `RankExpertOutput`
- `StaticRankEnsemblePolicy`
- `ContextualRankEnsemblePolicy`
- normalized rank/percentile aggregation
- deterministic tie handling
- sparse top-k expert support
- fallback policy handling
- contribution, entropy, fallback, and switching logs

Raw score averaging is not the default and is not considered generally valid because policies expose heterogeneous quantities. Normalized rank aggregation is the safest common representation.

### Component-Wise Composition

`ComponentWiseCompositionPolicy` prototypes a conservative combination of:

- SCORPIO-style admission filtering;
- WSP-style priority;
- KV, prefill, and aging safeguards when expressible in the current action space.

This is a prototype, not proof that the component combination is scientifically superior.

### Experiment Harness

`src/llmserveopt/selector/composition_experiment.py` provides:

- upstream final-report readiness checks;
- policy-vector CSV loading;
- split-group leakage validation;
- development-only best-fixed selection;
- guards that prevent held-out split use during treatment selection.

`tools/composition_smoke_experiment.py`, `tools/native_composition_pilot.py`, and sbatch wrappers provide smoke and pilot execution paths.

### Structural Synthesis

`src/llmserveopt/policies/genome.py` defines `SchedulerGenomeV1`, a typed genome wrapper around the verified heuristic DSL. It supports:

- canonical JSON serialization;
- deterministic SHA256 hashing;
- reproducible parsing;
- semantic validation;
- causal feature whitelist enforcement;
- conversion to verified heuristic policies.

`src/llmserveopt/policies/structural_synthesis.py` implements:

- representative parent-policy mappings;
- module swap;
- conditional regime composition;
- typed subtree crossover;
- bounded constant mutation;
- whitelisted feature/operator mutation;
- development-only frontier-value scoring;
- structured prompt-template rendering for future LLM-guided synthesis without calling an LLM.

`src/llmserveopt/selector/parent_selection.py` implements deterministic parent-pair scoring and a composition gate.

## Validation Status

- Composition harness: 37/37 tests previously passed under SLURM job `1119434`.
- Structural synthesis harness: 14/14 tests previously passed under SLURM job `1120181`.
- Native composition pilot: job `1120123` completed with `NATIVE_COMPOSITION_PILOT_DECISION = NO_GO`.

The native pilot found that static/contextual rank mixtures and component-wise composition did not clear the decisive held-out meaningful-window bar against the discrete selector. That argues against launching a full naive composition sweep immediately.

## Current Scientific Status

The full composition experiment should not be launched from the current
evidence.

Completed evidence now indicates:

- naive/native rank composition did not beat discrete selection or expand the
  frontier;
- single-module interventions contain sparse positive transfer, but module
  credit is weakly learnable;
- pairwise module combinations did not expand the 27-policy envelope;
- the selector suitability vector is useful but not yet reliable enough for
  unrestricted donor/module selection;
- `COMBINER_TRAINING_SIGNAL = WEAK`;
- `COMBINER_EVALUATION_READINESS = NEEDS_SIMULATOR_FIX`.

## What Is Not Implemented

The current simulator cannot faithfully execute:

- prefix-cache reuse policies;
- cache-loading-aware policies;
- disaggregated prefill/decode routing;
- request splitting;
- heterogeneous GPU routing;
- exact chunk-size prefill actions;
- durable multi-tenant credit state.

Those capabilities require simulator/action-space extensions before they can be used in policy composition or structural synthesis.

## Current Recommendation

Do not launch broad composition, unrestricted structural synthesis, or
large-scale module-credit expansion as the next step. First run bounded
simulator calibration and discriminative-power validation so KV/cache reuse,
prefill/decode contention, overload, and SLO pressure produce scientifically
meaningful policy-reward differences. After that, rerun controlled subsets,
retrain suitability models, and only then resume restricted evidence-guided
combination/synthesis.
