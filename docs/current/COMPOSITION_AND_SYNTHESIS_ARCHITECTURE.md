# Composition and Synthesis Architecture

This document is the concise source-of-truth overview for the path:

```text
Policy Library -> Discrete Selection -> Contextual Composition -> Structural Synthesis
```

It summarizes what is implemented, what is prototype-only, and what still requires completed experiment outputs.

## Implemented Components

### Policy Library

`src/llmserveopt/policies/registry.py` exposes two deployable libraries:

- `BASELINE_NAMES`: historical 20-policy library.
- `POLICY_LIBRARY_V2_NAMES`: historical library plus 7 new deployable approximation policies.

Policy Library v2 is implemented and unit-tested, but its expanded-frontier scientific value is still pending the running SLURM workflow.

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

## Dependencies on Running Workflows

The full composition experiment should not be launched until both final reports exist:

- Policy Frontier Cartography: `/mmfs1/project/ikoutis/sv96/llmserveopt-data/policy_frontier_cartography_20260721T154408Z/reports/FINAL_REPORT.md`
- Policy Library v2 Expanded Frontier: `/mmfs1/project/ikoutis/sv96/llmserveopt-data/policy_library_v2_expanded_20260721T171933Z/reports/FINAL_POLICY_LIBRARY_REPORT.md`

These outputs are needed to select expert policies, parent pairs, frontier regions, and top-k candidates using development evidence only.

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

Do not launch the full composition sweep solely from the native pilot. The stronger next path is structural symbolic synthesis or targeted evolutionary crossover from high-value parent policies, informed by the still-running frontier and Policy Library v2 results.
