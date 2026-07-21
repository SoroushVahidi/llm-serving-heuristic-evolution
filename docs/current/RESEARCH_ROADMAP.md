# Research Roadmap

Current roadmap as of 2026-07-21.

## Completed

- Clean leakage-safe Selector Dataset v2 pipeline.
- Selector v2 OOD diagnosis and fresh OOD evaluation.
- Selector v3 multi-domain causal-feature workflow.
- Policy Library v2 implementation with 7 new deployable approximation policies.
- Composition readiness harness with normalized-rank aggregation and typed module scaffolding.
- Native Wulver composition falsification pilot.
- Structural synthesis readiness harness with typed genomes, parent selection, module swaps, conditional composition, crossover, mutation, and frontier scoring.

## Running

- Policy Frontier Cartography:
  `/mmfs1/project/ikoutis/sv96/llmserveopt-data/policy_frontier_cartography_20260721T154408Z`
- Policy Library v2 Expanded Frontier:
  `/mmfs1/project/ikoutis/sv96/llmserveopt-data/policy_library_v2_expanded_20260721T171933Z`

Both are protected active experiment roots.

## Pending

- Final interpretation of frontier cartography.
- Final interpretation of Policy Library v2 expanded frontier.
- Full composition experiment only if upstream final reports and development-only gates justify it.
- Scaled structural synthesis or evolutionary crossover after frontier/Policy Library v2 evidence identifies high-value parent policies and boundary regions.

## Decision Gates

### Launch Full Composition Only If

- Policy Frontier Cartography and Policy Library v2 final reports both exist.
- Expert selection can use training/development evidence only.
- Candidate sparse/contextual or component-wise compositions show a clear reason to beat discrete top-1 selection.
- Native pilot `NO_GO` is superseded by stronger evidence from completed frontier/library outputs.

### Stop Naive Composition If

- Top-1/discrete selection remains as strong as or stronger than top-k/dense mixtures on held-out meaningful windows.
- Component-wise composition does not produce unique frontier wins.
- Improvements are confined to training/validation or near-tie windows.

### Move to Structural Synthesis If

- Frontier maps identify complementary parent policies and boundary niches.
- Native or expanded-library evidence suggests rank averaging is too blunt.
- High-value parent modules can be represented in `SchedulerGenomeV1` exactly or with clear, bounded approximations.

### Expand Simulator Capabilities If

- Important missing policy families require unsupported actions/state, such as cache reuse, cache loading, disaggregated routing, request splitting, exact chunked prefill, or heterogeneous GPU routing.
- Policy/frontier reports show those missing capabilities plausibly dominate remaining regret.

### Freeze Selector Work If

- Additional domain coverage, expanded causal features, and expanded policy library do not improve robust held-out performance.
- Local ambiguity or partial observability remains high across strong models.
- Fixed WSP or another fixed policy remains superior on meaningful fresh OOD evaluations.

## Recommended Next Action

Wait for the two active frontier workflows to complete. Then perform Query 3 validation/push if the integration branch is clean, and use the completed frontier reports to decide between full composition, structural symbolic synthesis, or simulator-capability expansion.
