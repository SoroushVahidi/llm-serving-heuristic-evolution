#!/bin/bash
#SBATCH --job-name=tier1_cv_bailian
#SBATCH --partition=general
#SBATCH --qos=standard
#SBATCH --account=ikoutis
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/mmfs1/project/ikoutis/sv96/llmserveopt-data/datasets/bailian_qwen/logs/%x_%j.out
#SBATCH --error=/mmfs1/project/ikoutis/sv96/llmserveopt-data/datasets/bailian_qwen/logs/%x_%j.err

set -euo pipefail

REPO=/mmfs1/project/ikoutis/sv96/github/llm-serving-heuristic-evolution-dataset-expansion
DATA=/mmfs1/project/ikoutis/sv96/llmserveopt-data/datasets/bailian_qwen
PYTHON=/home/sv96/.conda/envs/repo-env/bin/python
cd "$REPO"
export PYTHONPATH=src

{
  echo "job_id=${SLURM_JOB_ID:-local}"
  echo "start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host=$(hostname)"
  echo "git_sha=$(git rev-parse HEAD)"
} | tee "$DATA/manifests/job_${SLURM_JOB_ID:-local}_convert_meta.txt"

declare -A SPLITS=(
  [qwen_traceA_blksz_16.jsonl]=to_c_traceA
  [qwen_traceB_blksz_16.jsonl]=to_b_traceB
  [qwen_thinking_blksz_16.jsonl]=thinking
  [qwen_coder_blksz_16.jsonl]=coder
)

for fname in "${!SPLITS[@]}"; do
  split="${SPLITS[$fname]}"
  in="$DATA/raw/$fname"
  out="$DATA/processed/bailian_${split}.jsonl"
  $PYTHON scripts/data/convert_bailian_trace.py \
    --input "$in" \
    --output "$out" \
    --source-split "$split" \
    --time-scale 1.0 \
    --seed 17
  $PYTHON scripts/data/validate_processed_trace.py --input "$out" \
    --out "$DATA/manifests/processed_validation_${split}.json"
  $PYTHON scripts/data/characterize_processed_trace.py --input "$out" \
    --out "$DATA/characterization/${split}.json"
done

echo "finish_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$DATA/manifests/job_${SLURM_JOB_ID:-local}_convert_meta.txt"
echo "exit_status=0" | tee -a "$DATA/manifests/job_${SLURM_JOB_ID:-local}_convert_meta.txt"
