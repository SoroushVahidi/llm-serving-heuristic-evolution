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
5. [`STRESS_TEST_CATALOG.md`](STRESS_TEST_CATALOG.md) — human-readable
   rendering of the full 29-entry catalog (added 2026-08-05, alongside
   the Sarathi-Serve section).
6. [`COVERAGE_MATRIX.md`](COVERAGE_MATRIX.md) — per-algorithm coverage
   rollup plus Sarathi-Serve-specific real-hardware/simulator/local-GPU/
   Wulver-requirement/commit-fidelity detail (added 2026-08-05).
7. [`SARATHI_MECHANISM_CALIBRATION_20260805.md`](SARATHI_MECHANISM_CALIBRATION_20260805.md) /
   [`SARATHI_COMMIT_DRIFT_20260805.md`](SARATHI_COMMIT_DRIFT_20260805.md) —
   Sarathi-Serve-specific diagnostic and commit-provenance records; see
   `docs/audits/sarathi_stress_test_catalog_completion_20260805.md` for
   the full task record.

## Code

- `configs/stress_tests/algorithm_stress_test_catalog.yaml` — the
  machine-readable catalog itself, 29 entries (12 algorithms x
  TARGET/COUNTER, Sarathi-Serve contributing 7).
- `scripts/stress_tests/generators.py` — one workload-generator function
  per catalog entry (25 runnable, 5 explicit `NotImplementedError` stubs:
  4 offline-scored vLLM-LTR/PARS domain-shift entries + 1 Sarathi
  long-context entry this simulator's timing model cannot represent).
- `scripts/stress_tests/run_stress_test_smoke.py` — runs every catalog
  entry's workload against its algorithm-under-test + comparison
  algorithms and evaluates the catalog's own `acceptance_gates`
  expression. `--full` switches from smoke to full scale. Also supports
  Phase-1.5 prefill/decode-contention `ServiceModel` construction
  per-entry (`simulator_requirements.enable_prefill_modeling`/
  `enable_decode_prefill_contention`) for the Sarathi entries.
- `scripts/stress_tests/run_sarathi_headroom_check.py` — Sarathi-specific
  companion: dumps deterministic per-seed workload files to
  `configs/stress_tests/generated/sarathi/` and runs the 4-way headroom
  comparison, writing `results/stress_test_catalog/sarathi_smoke/report.{json,md}`.
- `tests/stress_tests/test_stress_test_generators.py` — tests for the
  pre-Sarathi 22-entry catalog; `tests/stress_tests/test_sarathi_stress_test_catalog.py` —
  tests for the 7 Sarathi entries specifically (schema, provenance,
  target/counter pairing, determinism, no future-info leakage,
  commit-drift disclosure, canonical-suite non-interference).
- `results/stress_test_catalog/` — smoke and full-scale run outputs.

## Quick start

```bash
python scripts/stress_tests/run_stress_test_smoke.py            # smoke scale
python scripts/stress_tests/run_stress_test_smoke.py --full     # full scale
python scripts/stress_tests/run_sarathi_headroom_check.py       # Sarathi-specific: dumps workloads + headroom report
python -m pytest tests/stress_tests/ -q
```

## Scope notes

- 7 of 29 catalog entries are explicitly out of automated-execution scope
  this pass: `regression_anwg`'s 2 entries need a persisted, trained
  `PerPolicyRegressionAnwgSelector` artifact this task does not load or
  retrain; `vllm_ltr`/`pars`'s 4 entries need a new offline scoring pass
  against the real checkpoints on a reasoning-prompt corpus that does not
  yet exist; `sarathi_counter_long_context_attention_recompute` needs an
  attention-cost scaling term this simulator's timing model does not
  have. All disclosed via `real_system_followup_required`/
  `requires_new_scoring_pass`/generator `NotImplementedError` in the
  catalog, not silently skipped.
- The 5 real-hardware Sarathi entries' `acceptance_gates` test a coarser
  mechanism (chunked vs. non-chunked admission) than the one real Wulver
  hardware validated (decode-protection vs. shared contention) — see
  `SARATHI_MECHANISM_CALIBRATION_20260805.md` for why, found and disclosed
  rather than papered over with a misleading gate.
- Does not touch `baselines/vtc/**`, any `docs/audits/vtc_*.md`, CC5/CC6
  config/core files, or `benchmarks/canonical_suite/` — see
  `CONCURRENCY_SAFETY_20260805.md`'s exclusion list.
- VTC itself already has its own dedicated, separately-validated
  fairness-extension workload library (`baselines/vtc/fairness_workloads.py`)
  and is intentionally out of scope for this stress-test library.
