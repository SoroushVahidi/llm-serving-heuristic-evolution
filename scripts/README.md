# Scripts

All scripts are in `scripts/`. Run from the repository root.

## Current entry points (Selector v2 / external-baseline program)

The sections below this one document Phase-1.7C-and-earlier scripts only.
For the project's actual current active work, use these instead:

| Script | Purpose |
|---|---|
| `build_selector_dataset_v2_calibrated_targeted_pilot.py` | **Current** Dataset v2 generator (Option B scope, 8-policy action space). Superseded `build_selector_dataset_v2_pilot.py` / `build_selector_dataset_v2_redesigned_pilot.py` / `build_selector_dataset.py` -- each of those now carries an in-file HISTORICAL banner. |
| `train_selector_v2_calibrated_prototype.py` | Trains/evaluates the prototype selector on a calibrated pilot's output. |
| `audit_selector_v2_calibrated_pilot_leakage.py` | Independent leakage audit for a calibrated pilot's splits -- run this before trusting any pilot's VALIDATION/ID_TEST numbers; see `docs/current/SELECTOR_V2.md` §10. |
| `selector_v2_slo_calibrated_frontier_search.py` | SLO-calibration / scenario-discriminativeness search. |
| `selector_v2_contention_frontier_search.py`, `selector_v2_contention_frontier_slo_sensitivity.py` | Contention-model frontier search (historical step, superseded by the SLO-calibrated version above -- kept for reproducibility). |
| `selector_v2_faithful_baseline_audit.py` | The faithful-baseline scope audit (Option B decision). |
| `run_gpu_external_validity_audit.py` | Central Wulver/local-GPU external-validity harness (real vLLM server vs. simulator). |
| `run_sarathi_gpu_smoke_and_validation.py`, `compare_sarathi_vllm_matched_runtime.py`, `analyze_repeated_trials.py` | Sarathi real-runtime validation (including the N=5 repeated-trial comparison). |
| `build_runtime_validation_benchmark_pack.py` | Builds the committed, checksummed runtime-validation benchmark pack. |
| `scripts/slurm/wulver_*.sbatch` | Wulver A100 cluster job scripts, one per experiment -- submit via `sbatch`, see `docs/wulver_gpu_validation_handoff.md`. |

Manual, paid-API, one-off scripts -- **require an explicit opt-in
environment variable, refuse to run otherwise** (see each script's own
header comment):

- `_run_cohere_v2_live_pilot.sh`, `_run_gemini_v2_live_pilot.sh` --
  `LLMSERVEOPT_ALLOW_PAID_API_CALLS=1` required.

Do-not-rerun (historical postprocessing, self-documented in-file):

- `phase17c_postprocess.sh`.

Full current-program context: **[docs/current/SELECTOR_V2.md](../docs/current/SELECTOR_V2.md)**
and **[docs/current/ARCHITECTURE.md](../docs/current/ARCHITECTURE.md)**.

---

**Coverage note (below this point):** the rest of this file documents the
Phase-1.7C-and-earlier data/GPU-calibration scripts. Phase 2A/2B/2C selector,
LLM-heuristic-generation, and real-LLM/vLLM scripts beyond the current-entry-points
table above are not yet individually documented here — see
[docs/current/PROJECT_STATUS.md](../docs/current/PROJECT_STATUS.md) and the
per-phase docs under [docs/audits/](../docs/audits/) for which script
produced which historical result.

All scripts support `--help` safely (prints usage, no file writes). Scripts that
write tracked docs or reports (`inspect_gpu_environment.py`, `update_phase17c_docs.py`)
also support `--dry-run` to preview output without writing, and accept explicit
output-path flags to override the defaults.

---

## Data download and conversion

### `download_burstgpt.py`
Downloads the BurstGPT dataset from HuggingFace Hub.  
Requires `HF_TOKEN` in the environment (see `.env.example`).

```bash
python scripts/download_burstgpt.py
```

Output: `data/raw/burstgpt/BurstGPT_1.csv`  
Long-running: No (network-bound, typically <5 min).

---

### `convert_burstgpt.py`
Converts raw BurstGPT CSV to the simulator's JSONL schema.
Augments synthetic SLOs, priorities, and predicted output lengths.

```bash
python scripts/convert_burstgpt.py \
  --input data/raw/burstgpt/BurstGPT_1.csv \
  --out-dir data/processed/burstgpt/
```

Output: `data/processed/burstgpt/*.jsonl`  
Long-running: No (seconds).

---

### `convert_sharegpt.py`
Converts raw ShareGPT JSON to JSONL.

```bash
python scripts/convert_sharegpt.py \
  --input data/raw/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json \
  --out-dir data/processed/sharegpt/
```

Output: `data/processed/sharegpt/*.jsonl`  
Long-running: No.

---

## Trace utilities

### `generate_synthetic_traces.py`
Generates and saves synthetic trace JSONL files from simulator config.

```bash
python scripts/generate_synthetic_traces.py --out-dir traces/ --seeds 0 1 2
```

Long-running: No.

---

### `summarize_trace.py`
Prints statistics (request count, arrival rate, token distribution) for a JSONL trace.

```bash
python scripts/summarize_trace.py --input data/processed/burstgpt/burstgpt_natural_10k.jsonl
```

---

## Synthetic experiments

### `run_baseline_comparison.py`
Runs all registered policies against a synthetic workload config.

```bash
python scripts/run_baseline_comparison.py --config configs/baseline_comparison.yaml
python scripts/run_baseline_comparison.py --config configs/small_debug.yaml  # fast
```

Output: `results/<experiment_name>/<timestamp>/`  
Long-running: Yes for large configs (run in tmux).

---

### `build_baseline_tables.py`
Aggregates result CSVs from multiple runs into a comparison table.

```bash
python scripts/build_baseline_tables.py --results-dir results/baseline_comparison/
```

---

### `smoke_test.py`
Quick end-to-end sanity check (seconds). Does not write results.

```bash
python scripts/smoke_test.py
```

---

## Real-trace experiments (Phase 1.7C)

### `run_real_trace_comparison.py`
Runs all registered policies against a real-trace replay config with calibrated
or synthetic service model.

```bash
python scripts/run_real_trace_comparison.py \
  --config configs/real_trace/burstgpt_scaled_moderate_calibrated.yaml
```

Output: `results/<experiment_name>/<timestamp>/`  
Long-running: **Yes — typically 90–150 min per config. Run in tmux.**

```bash
tmux new -s real_trace_run
python scripts/run_real_trace_comparison.py --config configs/real_trace/burstgpt_scaled_moderate_calibrated.yaml
```

---

## GPU calibration (Phase 1.7B)

### `run_gpu_calibration.py`
Measures prefill and decode timing curves on the local GPU.
Requires a CUDA-capable GPU and loaded model weights.

```bash
python scripts/run_gpu_calibration.py --config configs/gpu_calibration/calibration_grid.yaml
```

Output: `results/gpu_calibration/service_curves.json`  
Long-running: **Yes (~30–60 min). Run in tmux.**

---

### `fit_service_curves.py`
Fits power-law curves to raw calibration measurements.

```bash
python scripts/fit_service_curves.py \
  --raw results/gpu_calibration/raw_measurements.json \
  --out results/gpu_calibration/service_curves.json
```

---

### `validate_simulator_calibration.py`
Validates that the calibrated service model reproduces held-out measurements
within the target MAPE.

```bash
python scripts/validate_simulator_calibration.py
```

---

### `inspect_gpu_environment.py`
Prints CUDA, driver, GPU name, VRAM, and PyTorch version. Writes
`results/gpu_calibration/environment.json` and `docs/gpu_environment.md`
by default (override with `--json-output`/`--md-output`, or preview with
`--dry-run`).

```bash
python scripts/inspect_gpu_environment.py
python scripts/inspect_gpu_environment.py --dry-run
```

---

## Phase 1.7C post-processing

### `phase17c_postprocess.sh`
Orchestrates the full post-processing pipeline after all 7 Phase 1.7C experiments
complete: generates summaries, runs pytest, updates the milestone doc, and commits.

```bash
bash scripts/phase17c_postprocess.sh 2>&1 | tee results/phase17c/postprocess.log
```

Long-running: **Yes (~15 min including tests). Run in tmux.**  
**Do not re-run after the Phase 1.7C commit exists** — it would create a duplicate commit.

---

### `generate_phase17c_summary.py`
Reads all `summary.csv` files from Phase 1.7C experiment directories and generates:
- `results/phase17c/phase17c_experiment_summary.md`
- `results/phase17c/prediction_noise_sensitivity.md`
- `results/phase17c/prediction_noise_sensitivity.csv`
- `results/phase17c/calibrated_vs_synthetic_comparison.md`
- `results/phase17c/calibrated_vs_synthetic_rank_correlations.csv`
- `results/phase17c/plots/*.png`

```bash
python scripts/generate_phase17c_summary.py
```

---

### `update_phase17c_docs.py`
Updates `docs/milestones/phase1_7c_calibrated_real_trace.md` with experiment results
(override with `--milestone-output`/`--claims-output`, or preview with `--dry-run`).

```bash
python scripts/update_phase17c_docs.py
python scripts/update_phase17c_docs.py --dry-run
```

---

## Audit / planning utilities

These scripts produce reports but do not modify source, configs, or experiment data.

- `scripts/inspect_gpu_environment.py` — GPU hardware inspection
- `scripts/validate_simulator_calibration.py` — calibration validation against ground truth
- `scripts/summarize_trace.py` — trace statistics
