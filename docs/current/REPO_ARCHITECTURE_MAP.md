# Repository Architecture Map

Current map for the integrated Wulver branch.

## Core Runtime

- `src/llmserveopt/core/` - request/state/action/metric types.
- `src/llmserveopt/simulator/` - simulator, constraints, service models, calibrated service-model factory.
- `src/llmserveopt/evaluation/` - policy execution and metric aggregation.

## Policy Layer

- `src/llmserveopt/policies/base.py` - policy interface and feasibility helpers.
- `src/llmserveopt/policies/registry.py` - historical and Policy Library v2 registries.
- `src/llmserveopt/policies/policy_library_v2_helpers.py` - shared causal helpers for new monolithic policies.
- `src/llmserveopt/policies/*` - deployable policy implementations.
- `src/llmserveopt/policies/composition.py` - rank/contextual/component-wise composition harness.
- `src/llmserveopt/policies/genome.py` - typed `SchedulerGenomeV1` representation.
- `src/llmserveopt/policies/structural_synthesis.py` - structural child-generation operators.

## Selector Layer

- `src/llmserveopt/selector/` - selector features/models/datasets plus composition experiment helpers.
- `src/llmserveopt/selector/dataset_v2/` - leakage-safe Selector Dataset v2 infrastructure.
- `src/llmserveopt/selector/parent_selection.py` - parent-pair scoring and composition gate for structural synthesis.

## Heuristic DSL

- `src/llmserveopt/heuristics/` - verified DSL compiler, schema, expressions, and policy wrapper.
- `SchedulerGenomeV1` compiles only into the subset of this DSL that remains causally and semantically valid.

## Tools

- `tools/policy_library_v2_experiment.py` - expanded-library frontier workflow driver.
- `tools/composition_smoke_experiment.py` - correctness-only composition smoke.
- `tools/native_composition_pilot.py` - small Wulver-native composition falsification pilot.
- `tools/*sbatch` - SLURM launchers for focused tests and deferred composition/synthesis work.

## Tests

- `tests/test_policy_library_v2.py`
- `tests/test_policy_composition.py`
- `tests/test_structural_synthesis.py`

These focused tests are the medium-validation target for Query 2 and should be part of Query 3 pre-push validation.
