# Phase 1.7A: Real Trace Ingestion Infrastructure

**Date:** 2026-06-10
**Status:** COMPLETE

---

## Summary

Phase 1.7A adds infrastructure for loading, converting, augmenting, and replaying
real LLM serving workload traces. No raw datasets are downloaded in this phase;
only the ingestion code and test fixtures are committed.

---

## Components Added

### Loaders
- `src/llmserveopt/workloads/burstgpt.py` — BurstGPT CSV loader with multi-column-name
  detection, zero-token filtering, timestamp normalization, time scaling
- `src/llmserveopt/workloads/sharegpt.py` — ShareGPT JSON loader with tokenizer
  integration or whitespace fallback, Poisson/bursty arrival synthesis

### Augmentation
- `src/llmserveopt/workloads/augmentation.py` — deterministic synthetic field
  generation: prediction noise (exact/lognormal/biased_under/biased_over/bucket),
  SLO class assignment (configurable weights, slacks, priorities)

### Extended Trace I/O
- `src/llmserveopt/workloads/trace_io_extended.py` — JSONL format with `source`
  provenance and `metadata.synthetic_fields` list

### Conversion Scripts
- `scripts/convert_burstgpt.py`
- `scripts/convert_sharegpt.py`
- `scripts/download_burstgpt.py`
- `scripts/summarize_trace.py`
- `scripts/run_real_trace_comparison.py`

### Configs
- `configs/traces/burstgpt_conversion.yaml`
- `configs/traces/sharegpt_conversion.yaml`
- `configs/traces/prediction_noise_exact.yaml`
- `configs/traces/prediction_noise_moderate.yaml`
- `configs/traces/prediction_noise_high.yaml`
- `configs/burstgpt_replay_comparison.yaml`
- `configs/burstgpt_replay_scaled_load.yaml`
- `configs/sharegpt_poisson_comparison.yaml`
- `configs/sharegpt_bursty_comparison.yaml`

### Test Fixtures
- `tests/fixtures/burstgpt_tiny.csv` — 10-row BurstGPT fixture for unit tests
- `tests/fixtures/sharegpt_tiny.json` — 5-conversation ShareGPT fixture

---

## Test Status at Completion

- **49 new tests** added (burstgpt_loader: 17, sharegpt_loader: 18, augmentation: 13,
  trace round-trip: 1)
- Total tests: 165 (up from ~102 at Phase 1.5 freeze)
- 1 known warning: pandas boolean-index reindexing in `test_burstgpt_loader.py:59`
  (cosmetic; will be fixed in Phase 1.7C)

---

## Field Provenance Policy

Enforced via `metadata.synthetic_fields`:

| Dataset | Real fields | Synthetic fields |
|---|---|---|
| BurstGPT | arrival_time, prompt_tokens, actual_output_tokens | predicted_output_tokens, class_id, priority, slo_deadline |
| ShareGPT | prompt_tokens, actual_output_tokens (via tokenizer) | arrival_time, predicted_output_tokens, class_id, priority, slo_deadline |

See `docs/data_field_provenance.md` for full specification.

---

## Known Limitations

- No raw datasets downloaded (network/license constraints handled at user level)
- Replay configs blocked until `data/processed/*/` are populated
- ShareGPT tokenization falls back to whitespace splitting when no tokenizer is specified
