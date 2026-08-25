# portfolio_guided_typed_gp_smoke_v1 Analysis - 2026-08-24

`NON_SCIENTIFIC_SMOKE_CALIBRATION`

## Scope

This was a tiny operational smoke/timing calibration for `portfolio_guided_typed_gp_screen_v1`. It was not the scientific three-treatment evolutionary screen and must not be interpreted as evidence that any treatment is better.

## Inputs

- Source design: `experiments/portfolio_guided_typed_gp_screen_v1/`
- Smoke output: `experiments/portfolio_guided_typed_gp_smoke_v1/`
- Scenario source: frozen 24-row TRAIN-only screen manifest
- Smoke scenarios: 3 total, one each from Family A/B/C
- Candidate budget: 4 evaluated candidates per treatment
- Total candidate-scenario evaluations: 36

Smoke scenario IDs:

- `FAMILY_A_FAIRNESS_STARVATION_V2::fs2.util1.1000.skew1.0000.favlong.noise0.00.s20260816`
- `FAMILY_B_PREFILL_DECODE_V2::pd2.hog12.late12.slohog_ttft.s20260820`
- `FAMILY_C_KV_PRESSURE_V2::kvp2.bulk10.phaseearly.tightloose.s20260910`

## Pre-Smoke Tests

Command:

```bash
python -m pytest -q tests/test_portfolio_guided_typed_gp_screen_v1.py tests/test_heuristic_dsl_no_leakage.py tests/test_heuristic_policy_determinism.py tests/test_policy_genome_coverage.py tests/test_policy_separation_prefill_decode_v2.py::test_v2_policy_set_is_exactly_two_anchors tests/test_policy_separation_prefill_decode_v2.py::test_v2_policies_use_only_online_observables
```

Result: 202 passed.

The prior pandas issue was an interpreter issue. `pandas>=2.0` is declared in `pyproject.toml` and `requirements.txt`; using `python -m pytest` with the active interpreter resolved collection.

## Treatment Accounting

All three treatments consumed the same evaluated-candidate budget:

- `A_RANDOM_GRAMMAR_GP`: 4 evaluated, 5 proposed, 1 rejected, 0 duplicate
- `B_PARENT_SEEDED_MUTATION_ONLY`: 4 evaluated, 4 proposed, 0 rejected, 0 duplicate
- `C_PORTFOLIO_STRUCTURAL_CROSSOVER`: 4 evaluated, 5 proposed, 1 rejected, 0 duplicate

The rejection in crossover was the intended invalid `KVGuard` crossover probe. The random-grammar rejection was a complexity-limit rejection.

## Plumbing Checks

- Random grammar generation worked and was not parent-seeded.
- Mutation-only provenance used exactly one parent and no crossover.
- Structural crossover produced 4 valid children with parent IDs and crossover paths recorded.
- Behavioral fingerprints were produced.
- Parent-overlap measurements were produced.
- MG values were computed against frozen E6 anchors for pipeline validation only.
- Equal evaluated-candidate accounting passed.

Observed smoke pathology:

- Structural parent canonicalization: 0 / 12 evaluated candidates
- Behavioral parent overlap of 1.0 on the tiny probe set: 6 / 12 evaluated candidates
- Duplicate rate: 0.0
- Invalid-candidate rate: 2 / 14 proposals = 0.142857

This is a measurement/readiness signal only, not a treatment-quality conclusion.

## Timing

- Total smoke runtime: 7.904640 s
- Candidate-scenario evaluations: 36
- Wall-clock per candidate-scenario evaluation: 0.218693 s
- Candidate generation overhead: 0.001591 s
- Candidate-scenario evaluation time: 7.872962 s
- Fingerprint overhead: 0.027964 s
- MG aggregation overhead: 0.000350 s

Projected serial cost for the frozen full screen:

- 60 evaluated candidates/treatment
- 3 treatments
- 24 scenarios
- 4,320 candidate-scenario evaluations
- Naive projected wall time: 944.755480 s = 15.745925 min

The original <=10 minute ambition is not realistic for the current serial runner. The projected runtime is still acceptable for local CPU, so no reward-dependent or timing-dependent budget change was made.

## Readiness

`experiments/portfolio_guided_typed_gp_screen_v1/implementation_readiness.json` now records:

`screen_ready: true`

The full evolutionary screen remains:

`NOT_RUN`

## Safety

No TEST or FINAL data was used. No Wulver, SLURM, GPU, API, or heavy job was launched. No git push or destructive git operation was performed. Smoke outputs are not scientific evidence.
