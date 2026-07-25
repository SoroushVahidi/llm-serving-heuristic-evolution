# Part 2 Validation

## Environment
- conda env: `repo-env`
- date (UTC): 2026-07-25
- branch: `reality-grounded-dataset-expansion-20260724`

## compileall
```
python3 -m compileall -q src scripts tools
```
Result: success (ELAPSED ~6.0s)

## Focused tests
```
python3 -m pytest -q \
  tests/test_repaired_discrimination_pilot.py \
  tests/test_real_window_construction.py \
  tests/test_canonical_ingestion_schema.py \
  tests/test_real_dataset_converters.py \
  tests/test_burstgpt_streaming.py \
  tests/test_selector_no_leakage.py \
  tests/test_final_evaluation_no_test_leakage.py
```
Result: **59 passed** in 38.41s (wall ~47.8s). Failures: 0. Skips: 0.

## Full non-hardware suite
```
python3 -m pytest -q \
  --deselect tests/test_compare_simulator_to_real_llm_latency.py \
  --deselect tests/test_calibration_gpu.py \
  --deselect tests/test_gpu_external_validity_audit.py
```
Result: **2809 passed**, **90 skipped**, **26 deselected**, 1 warning in 1267.69s (~21m; wall ELAPSED 1274.20s).
Failures: 0.

Deselections: three GPU/hardware modules as specified (26 collected tests deselected total).

Warning: `UserWarning` in `test_disaggregated_prefill_decode` (mid-transfer skip) — preexisting, not introduced by Part 2.
