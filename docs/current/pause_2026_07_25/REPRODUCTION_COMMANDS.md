# Reproduction Commands

Paths with `/mmfs1/...` are historical; substitute local durable roots after migration.

## Environment
```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate repo-env
cd /path/to/llm-serving-heuristic-evolution
git checkout reality-grounded-dataset-expansion-20260724
```

## Reconstruct Tier 1 datasets
Use `scripts/data/download_*.py` and `convert_*.py` / `validate_*.py` per dataset.
Mooncake requires an explicitly supplied local path and license acknowledgment; do not redistribute.

See also `scripts/data/reconstruct_tier1_datasets.sh` (wrapper) and
`scripts/data/verify_tier1_dataset_checksums.py`.

## Reconstruct real windows
```bash
python3 scripts/data/run_real_window_dataset_pipeline.py \
  --dataset <name> --help
```
Or use `scripts/workloads/reconstruct_real_windows.py` / cluster templates under `scripts/cluster/`.

## Re-run repaired pilot (after windows exist)
```bash
export REPO_ROOT=$PWD
export RUN_ROOT=<validated_window_root>
export PILOT_ROOT=<new_pilot_root>
# edit partition/account in template or pass sbatch -A/-p
sbatch scripts/cluster/submit_repaired_pilot.sbatch.template
# or:
python3 scripts/data/run_repaired_load_discrimination_pilot.py \
  --run-root "$RUN_ROOT" --pilot-root "$PILOT_ROOT" \
  --git-sha "$(git rev-parse HEAD)" --seed 20260725 --workers 8
```

## Tests
```bash
python3 -m compileall -q src scripts tools
python3 -m pytest -q tests/test_repaired_discrimination_pilot.py
python3 -m pytest -q \
  --deselect tests/test_compare_simulator_to_real_llm_latency.py \
  --deselect tests/test_calibration_gpu.py \
  --deselect tests/test_gpu_external_validity_audit.py
```
