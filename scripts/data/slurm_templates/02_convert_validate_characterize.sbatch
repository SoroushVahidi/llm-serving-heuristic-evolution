#!/bin/bash
#SBATCH --job-name=tier1_cv_burstgpt
#SBATCH --partition=general
#SBATCH --qos=standard
#SBATCH --account=ikoutis
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/mmfs1/project/ikoutis/sv96/llmserveopt-data/datasets/burstgpt_v2/logs/%x_%j.out
#SBATCH --error=/mmfs1/project/ikoutis/sv96/llmserveopt-data/datasets/burstgpt_v2/logs/%x_%j.err

set -euo pipefail

REPO=/mmfs1/project/ikoutis/sv96/github/llm-serving-heuristic-evolution-dataset-expansion
DATA=/mmfs1/project/ikoutis/sv96/llmserveopt-data/datasets/burstgpt_v2
PYTHON=/home/sv96/.conda/envs/repo-env/bin/python
cd "$REPO"
export PYTHONPATH=src

{
  echo "job_id=${SLURM_JOB_ID:-local}"
  echo "start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host=$(hostname)"
  echo "git_sha=$(git rev-parse HEAD)"
  echo "python=$($PYTHON --version 2>&1)"
} | tee "$DATA/manifests/job_${SLURM_JOB_ID:-local}_convert_meta.txt"

# Natural replay only (time_scale=1.0). No silent SLO claim beyond disclosed augmentation.
for split in 1 2 3; do
  in="$DATA/raw/BurstGPT_without_fails_${split}.csv"
  out="$DATA/processed/burstgpt_without_fails_${split}.jsonl"
  $PYTHON scripts/convert_burstgpt.py \
    --input "$in" \
    --output "$out" \
    --time-scale 1.0 \
    --seed 17 \
    --chunked \
    --chunksize 100000
  $PYTHON scripts/data/validate_processed_trace.py \
    --input "$out" \
    --out "$DATA/manifests/processed_validation_without_fails_${split}.json"
  $PYTHON scripts/data/characterize_processed_trace.py \
    --input "$out" \
    --out "$DATA/characterization/without_fails_${split}.json"
done

# Determinism check on bounded sample of file 1
$PYTHON - <<'PY'
from pathlib import Path
import tempfile
from llmserveopt.workloads.burstgpt import BurstGPTConversionConfig, load_burstgpt_trace
from llmserveopt.workloads.trace_io_extended import save_extended_jsonl

src = Path("/mmfs1/project/ikoutis/sv96/llmserveopt-data/datasets/burstgpt_v2/raw/BurstGPT_without_fails_1.csv")
cfg = BurstGPTConversionConfig(max_requests=5000, time_scale=1.0)
r1, _ = load_burstgpt_trace(src, config=cfg, seed=17, use_chunked=True, chunksize=1000)
r2, _ = load_burstgpt_trace(src, config=cfg, seed=17, use_chunked=True, chunksize=2500)
assert [(x.request_id, x.arrival_time, x.prompt_tokens, x.actual_output_tokens, x.predicted_output_tokens) for x in r1] == \
       [(x.request_id, x.arrival_time, x.prompt_tokens, x.actual_output_tokens, x.predicted_output_tokens) for x in r2]
print("determinism_ok n=5000")
PY

echo "finish_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$DATA/manifests/job_${SLURM_JOB_ID:-local}_convert_meta.txt"
echo "exit_status=0" | tee -a "$DATA/manifests/job_${SLURM_JOB_ID:-local}_convert_meta.txt"
