# portfolio_guided_typed_gp_screen_v1 Implementation - 2026-08-24

## Status

- DESIGNED: yes
- IMPLEMENTED: exact parent representation, serialization, typed crossover/mutation primitives, behavioral fingerprinting, envelope MG helpers, and equal-budget accounting
- TESTED: targeted parent/GP readiness tests plus adjacent heuristic tests
- TIMING_CALIBRATED: no
- NOT_RUN: evolutionary screen not launched

## Implementation Summary

The implementation adds `src/llmserveopt/policies/portfolio_gp.py`, a narrow typed representation for the `portfolio_guided_typed_gp_screen_v1` gate. It does not replace the existing `SchedulerGenomeV1` heuristic DSL. It wraps the exact mechanisms that the scalar DSL cannot faithfully encode: WFS class-deficit ranking, KV reserve/admission/placement, and fixed prefill chunk execution.

The six frozen parent genomes are exposed through `PARENT_GENOMES_V1`:

- `full_prefill`
- `chunked_prefill_small`
- `estimated_service_time_first`
- `weighted_fair_share`
- `least_laxity_first`
- `kv_constrained_online`

Each parent has a canonical JSON genome string and stable SHA256 structural hash. Exact prefill control uses the existing `Action.prefill_chunk_override` path, which is the simulator-supported per-step equivalent of fixed `ServiceModel.max_prefill_chunk_tokens` execution.

## Parent Reproduction

All six parents pass the deterministic action-level reproduction harness on the compact mechanism probe suite:

- `full_prefill`: `PARENT_REPRODUCTION_PASS`
- `chunked_prefill_small`: `PARENT_REPRODUCTION_PASS`
- `estimated_service_time_first`: `PARENT_REPRODUCTION_PASS`
- `weighted_fair_share`: `PARENT_REPRODUCTION_PASS`
- `least_laxity_first`: `PARENT_REPRODUCTION_PASS`
- `kv_constrained_online`: `PARENT_REPRODUCTION_PASS`

The harness compares original parent execution against decoded parent genome execution on copied observable states and reports the first action mismatch if any. The prefill tests also check that decoded per-step chunk overrides reproduce fixed service-model chunk behavior on a tiny simulator trace.

## Implemented Components

- Canonical genome serialization and parsing
- Stable SHA256 structural hashes
- Typed parent module registry
- Exact executable parent interpreter
- Action-level reproduction harness
- Strongly typed subtree/module crossover for supported module types
- Bounded parameter mutation over declared free numeric parameters
- Behavioral fingerprinting over fixed probe states
- Decision-overlap calculation
- Envelope and marginal-gain helpers for `E6(x)` and `MG_c(x;P6)`
- Equal-budget treatment accounting

## Tests

Passed:

- `pytest -q tests/test_portfolio_guided_typed_gp_screen_v1.py`
  - 14 passed
- `pytest -q tests/test_heuristic_dsl_no_leakage.py tests/test_heuristic_policy_determinism.py`
  - 68 passed

Blocked by local dependency, not by this implementation:

- `tests/test_policy_separation_prefill_decode_v2.py`: collection requires `pandas`
- `tests/test_policy_genome_coverage.py`: collection reaches `workloads/burstgpt.py`, which requires `pandas`

## Readiness

Machine-readable readiness is recorded in `experiments/portfolio_guided_typed_gp_screen_v1/implementation_readiness.json`.

Current value:

`screen_ready = false`

Reasons:

- Tiny timing calibration was not run in this task.
- The full evolutionary screen driver/smoke launch still needs a separate review before launch.

The primary scientific blocker from the design audit, exact six-parent reproduction, is resolved for the implemented probe suite. The next task should review this implementation and run only the smallest timing/smoke calibration before any evolutionary treatment comparison.

## Safety

No TEST or FINAL data was used. No evolutionary screen was launched. No Wulver, GPU, API, or heavy simulation job was launched. No git push or destructive git operation was performed.
