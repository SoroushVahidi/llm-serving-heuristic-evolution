# portfolio_guided_typed_gp_screen_v1 Design - 2026-08-24

## Status

- DESIGNED: yes
- IMPLEMENTED: no, except this design package
- TESTED: JSON/design validation only
- RUN: no

## Current Local Scientific State

The selector hypothesis is closed as `SELECTOR_HYPOTHESIS_FALSIFIED`; the guarded ESTF/WFS composite feasibility line is closed as `MECHANISM_COMPOSITE_STATIC_NO_GO`. The active direction is `PORTFOLIO_POLICY_SYNTHESIS` under H1/H0: test whether structural recombination of scheduler mechanisms can produce compact state-adaptive symbolic schedulers that expand the current six-policy envelope more effectively than random grammar search or parent-seeded mutation-only search.

The current six-policy portfolio is `full_prefill, chunked_prefill_small, estimated_service_time_first, weighted_fair_share, least_laxity_first, kv_constrained_online`. The screen must not reopen statewise ESTF/WFS selection, DAgger, support expansion, static scalar analytic indices, or oracle-label acquisition.

## Reused Infrastructure

The design reuses `src/llmserveopt/heuristics/` for leakage-checked expression evaluation, `src/llmserveopt/policies/genome.py` for canonical genome JSON and stable hashing, `src/llmserveopt/policies/structural_synthesis.py` for existing parent mappings/operators, `src/llmserveopt/policies/primitives.py` for causal mechanism primitives, `experiments/mf_psd_v1/` for candidate TRAIN-screen scenarios, and `experiments/unified_utility_matrix_v2/` for the frozen six-policy envelope.

## Parent Mechanism Decomposition

- `estimated_service_time_first`: exact primary completion/service ranking from `estimated_service_time_first.py`.
- `weighted_fair_share`: instantaneous class deficit times priority over service from `weighted_fair_share.py`; exact admitted-count/class-deficit behavior is not currently representable by the stateless DSL.
- `least_laxity_first`: exact primary laxity/SLO urgency ranking from `least_laxity_first.py`.
- `kv_constrained_online`: KV reserve, urgent-laxity bypass, and KV-aware placement from `kv_constrained_online.py`; current genome mapping is approximate.
- `full_prefill`: service-model full prefill control from `prefill_control_variants.py`; not a request-ranking DSL module.
- `chunked_prefill_small`: service-model small-chunk prefill control from `prefill_control_variants.py`; not a request-ranking DSL module.

This means exact parent wrappers or extra typed slots are required before the screen can run. Approximate decompositions may be used as design hints, but parent reproduction is a hard validity gate.

## Typed DSL / Genome Design

The minimum grammar is a constrained profile over existing `SchedulerGenomeV1` rather than a second DSL. It uses typed terminals for Time, Tokens, Ratio, Count, Score, Boolean, RankingRule, AdmissionCondition, PrefillRule, KVGuard, and Policy. The expression subset is deliberately small: `add`, `sub`, `mul`, `div_safe`, `min`, `max`, `clip`, Boolean comparisons/combinators, and typed `IF`. Complexity is capped at depth <= 4, nodes <= 31, constants <= 4, and if-nodes <= 3. No unrestricted loops, Python execution, future information, actual output length, TEST, FINAL, or hidden labels are allowed.

## Canonical Representation

The genome identity is canonical JSON with sorted keys and normalized finite constants, plus a SHA256 structural hash. Required invariant: `genome -> AST -> genome` must be byte-identical. A behavioral fingerprint is separate and hashes deterministic decisions on the fixed probe/screen scenarios.

## Search Treatments

1. `A_RANDOM_GRAMMAR_GP`: random valid typed programs, standard mutation/crossover, no parent seeding.
2. `B_PARENT_SEEDED_MUTATION_ONLY`: initialized from the six exact parent wrappers/genomes; mutation only.
3. `C_PORTFOLIO_STRUCTURAL_CROSSOVER`: same parent initialization as B, plus type-compatible subtree/module crossover.

All treatments receive 60 evaluated candidates on the same 24-scenario TRAIN-only screen unless the pre-launch timing calibration requires a smaller budget to keep the first run local and short.

## Fitness And Gates

Primary fitness is mean marginal gain over the frozen six-policy envelope: `MG_c(x;P6)=max(R_c(x),E6(x))-E6(x)`, using ANWG. GO requires mean MG >= 0.005, at least 3 unique wins at epsilon 0.005, wins across at least 2 regions, max parent decision overlap <= 95%, max reward correlation <= 0.985, no group regression above 0.03 ANWG, no concentration in one scenario/family, mechanism behavior matching the hypothesis, and all parent reproduction tests passing.

NO_GO is triggered by mean MG < 0.001, no unique wins, isolated wins only, parent-overlap collapse with weak MG, regression above 0.05 ANWG, mechanism inversion, parent behavioral recovery, or failed parent reproduction.

## TRAIN-Only Screening Subset

The design manifest contains 24 deterministic scenarios: {'FAMILY_A_FAIRNESS_STARVATION_V2': 8, 'FAMILY_B_PREFILL_DECODE_V2': 8, 'FAMILY_C_KV_PRESSURE_V2': 8}. They are drawn from `experiments/mf_psd_v1/mf_psd_scenarios_v1.csv` with all six anchor values populated in `experiments/unified_utility_matrix_v2/unified_utility_matrix_wide_v2.csv`. They are marked `TRAIN_SCREEN_ONLY_NON_DEV_NON_TEST_NON_FINAL`.

## Compute Budget

Planned first screen: 3 treatments x 60 candidate evaluations x 24 scenarios = 4320 candidate-scenario runs, local CPU only. A tiny timing calibration must be run before the actual screen command; no calibration or screen was launched in this task.

## Test Plan

Required tests before any run: genome round-trip, canonical hashing, type safety, invalid crossover rejection, valid crossover execution, mutation validity, no future-information access, deterministic candidate evaluation, exact parent behavior reproduction for all six parents, fingerprint determinism, decision-overlap correctness, envelope MG correctness, equal-budget accounting, and complexity limit enforcement.

## Implementation Readiness

The design is not yet executable. The largest blocker is exact parent reproduction: ESTF/LLF are close to exact in the current genome layer, but WFS, KV, full_prefill, and chunked_prefill_small require exact wrappers or typed module extensions before the screen can be scientifically valid.

## Next Task

Implement the GP profile, exact parent wrappers/module slots, fingerprint harness, equal-budget evaluator, and tests. Only after the parent reproduction and no-leakage tests pass should the first screen command be run.
