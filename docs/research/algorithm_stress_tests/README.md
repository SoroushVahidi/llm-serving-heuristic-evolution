# Algorithm Stress-Test Library

A literature-grounded catalog of TARGET (the regime an algorithm was
designed to solve) and COUNTER (documented limitations, formally-proven
worst cases, or reasoned adversarial regimes) workload generators for
this project's scheduling algorithms.

## Documents, in reading order

1. [`CONCURRENCY_SAFETY_20260805.md`](CONCURRENCY_SAFETY_20260805.md) —
   repository/resource safety check and exclusion list this task worked
   under (VTC's files, CC5/CC6, the canonical suite — never touched).
2. [`ALGORITHM_INVENTORY_20260805.md`](ALGORITHM_INVENTORY_20260805.md) —
   full inventory of 18 algorithms across internal/foundational,
   integrated-external, and planned-external categories.
3. [`LITERATURE_RESEARCH_20260805.md`](LITERATURE_RESEARCH_20260805.md) —
   primary-source research backing every catalog entry's evidence
   classification (PROVEN_WORST_CASE / DOCUMENTED_LIMITATION /
   PAPER_MOTIVATING_STRESS_CASE / HYPOTHESIZED_ADVERSARIAL_REGIME /
   INTERNAL_EMPIRICAL_FINDING).
4. [`STRESS_TEST_VALIDATION_20260805.md`](STRESS_TEST_VALIDATION_20260805.md) —
   the calibration record: what the first-draft workloads/gates got
   wrong, how each was diagnosed and fixed, and the final validated
   result (16/16 auto-evaluable gates pass at both smoke and full scale).

## Code

- `configs/stress_tests/algorithm_stress_test_catalog.yaml` — the
  machine-readable catalog itself, 22 entries (11 algorithms x
  TARGET/COUNTER).
- `scripts/stress_tests/generators.py` — one workload-generator function
  per catalog entry (18 runnable, 4 explicit `NotImplementedError` stubs
  for the offline-scored vLLM-LTR/PARS domain-shift entries).
- `scripts/stress_tests/run_stress_test_smoke.py` — runs every catalog
  entry's workload against its algorithm-under-test + comparison
  algorithms and evaluates the catalog's own `acceptance_gates`
  expression. `--full` switches from smoke to full scale.
- `tests/stress_tests/test_stress_test_generators.py` — 56 tests (52
  executed + 4 structurally skipped for the offline-scored stubs).
- `results/stress_test_catalog/` — smoke and full-scale run outputs.

## Quick start

```bash
python scripts/stress_tests/run_stress_test_smoke.py            # smoke scale
python scripts/stress_tests/run_stress_test_smoke.py --full     # full scale
python -m pytest tests/stress_tests/ -q
```

## Scope notes

- 6 of 22 catalog entries are explicitly out of automated-execution scope
  this pass: `regression_anwg`'s 2 entries need a persisted, trained
  `PerPolicyRegressionAnwgSelector` artifact this task does not load or
  retrain; `vllm_ltr`/`pars`'s 4 entries need a new offline scoring pass
  against the real checkpoints on a reasoning-prompt corpus that does not
  yet exist. Both are disclosed via `real_system_followup_required`/
  `requires_new_scoring_pass` in the catalog, not silently skipped.
- Does not touch `baselines/vtc/**`, any `docs/audits/vtc_*.md`, CC5/CC6
  config/core files, or `benchmarks/canonical_suite/` — see
  `CONCURRENCY_SAFETY_20260805.md`'s exclusion list.
- VTC itself already has its own dedicated, separately-validated
  fairness-extension workload library (`baselines/vtc/fairness_workloads.py`)
  and is intentionally out of scope for this stress-test library.
