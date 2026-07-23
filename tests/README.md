# tests/

~100 test files organized by filename prefix, not by directory. This file is
the navigation aid; it does not replace reading the actual test code.

## Correct invocation

```bash
python3 -m pytest --collect-only -q   # count/list without running
python3 -m pytest -m 'not gpu' -q     # full non-GPU-safe suite
python3 -m pytest -m gpu              # GPU-only tests (requires CUDA)
python3 -m pytest tests/test_foo.py   # a single file
```

**Use `python3 -m pytest`, not bare `pytest`.** The bare `pytest` resolved
from `PATH` may be a different interpreter missing `pandas`/`pyarrow`, and
will silently drop test files with collection errors rather than failing
loudly. If you must use bare `pytest`, first confirm
`python3 -c "import pandas"` succeeds in that same interpreter.

Approximate scale: on the order of 2,500 tests across ~100 files as of this
writing. Don't hard-code that number anywhere permanent -- it drifts with
every commit; re-run `--collect-only` for the current count.

## Test categories (by filename pattern)

| Pattern | Covers |
|---|---|
| `test_phase2a*`, `test_phase2b*`, `test_phase2c*` | Historical, phase-numbered regression tests. Still run, still pass, document a specific completed research phase -- see `docs/audits/` for the corresponding write-up. |
| `test_*_faithful_scheduler.py`, `test_external_baseline_integration.py`, `test_run_vllm_external_baseline_comparison.py` | The 7 faithful external baselines: scheduler correctness, integration harness, real-serving comparison. Largest category by test count. |
| `test_selector_*.py`, `test_rule_based_selector.py` | Selector v1: features, labels, models, no-leakage checks. |
| `test_selector_dataset_v2.py`, `test_phase2c_labeled_selector_dataset.py`, `test_selector_v2_candidate_source_of_truth.py` | Selector v2: Dataset v2 builder/schema/candidates, and the source-of-truth invariant tests for the Option B action space. |
| `test_calibration_gpu.py`, `test_gpu_external_validity_audit.py` | GPU-marked (`@pytest.mark.gpu`), skip cleanly without CUDA. |
| `test_compare_simulator_to_real_llm_latency.py`, `test_fit_real_llm_latency_model*.py`, `test_run_hosted_policy_comparison.py` | Runtime/benchmark-pack and real-LLM-vs-simulator comparison. |
| `test_cohere_api_calibration.py`, `test_gemini_*_calibration.py`, `test_real_llm_*.py` | Real-LLM API-calibration pilots (mocked/dry-run in CI; live calls require explicit opt-in). |
| `test_script_cli_safety.py` | Script/CLI safety: `--help` doesn't mutate the working tree, dry-run scripts report intent without writing, and the two manual paid-API shell scripts refuse to run without `LLMSERVEOPT_ALLOW_PAID_API_CALLS=1`. |
| `test_heuristic_dsl_*.py`, `test_llm_generation_*.py`, `test_generated_heuristic_*.py` | The (currently dormant relative to selector work) LLM-heuristic-DSL track. |
| Everything else (`test_simulator_*.py`, `test_*_policy.py`, `test_metrics.py`, `test_kv_block_manager.py`, ...) | Core simulator / policy / metrics unit tests. |

## `tests/fixtures/`

Small, purpose-named fixture files (`sharegpt_tiny.json`, `burstgpt_tiny.csv`,
`service_curves_fixture.json`) -- not full datasets, just enough to exercise
loaders without needing `data/raw/` populated.

## `tests/conftest.py`

Minimal -- only sets `sys.path` to include `src/`. No shared fixtures or
custom markers registered beyond `gpu` (declared in `pyproject.toml`).
