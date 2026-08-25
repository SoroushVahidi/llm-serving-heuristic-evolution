#!/bin/bash
#SBATCH --job-name=tier1_cv_azure23
#SBATCH --partition=general
#SBATCH --qos=standard
#SBATCH --account=ikoutis
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/mmfs1/project/ikoutis/sv96/llmserveopt-data/datasets/azure_llm_2023/logs/%x_%j.out
#SBATCH --error=/mmfs1/project/ikoutis/sv96/llmserveopt-data/datasets/azure_llm_2023/logs/%x_%j.err

set -euo pipefail

REPO=/mmfs1/project/ikoutis/sv96/github/llm-serving-heuristic-evolution-dataset-expansion
DATA=/mmfs1/project/ikoutis/sv96/llmserveopt-data/datasets/azure_llm_2023
PYTHON=/home/sv96/.conda/envs/repo-env/bin/python
cd "$REPO"
export PYTHONPATH=src

{
  echo "job_id=${SLURM_JOB_ID:-local}"
  echo "start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host=$(hostname)"
  echo "git_sha=$(git rev-parse HEAD)"
} | tee "$DATA/manifests/job_${SLURM_JOB_ID:-local}_convert_meta.txt"

for split in code conv; do
  in="$DATA/raw/AzureLLMInferenceTrace_${split}_2023.csv"
  out="$DATA/processed/azure_llm_2023_${split}.jsonl"
  $PYTHON scripts/data/convert_azure_llm_trace.py \
    --input "$in" \
    --output "$out" \
    --source "azure_2023_${split}" \
    --time-scale 1.0 \
    --seed 17
  $PYTHON scripts/data/validate_processed_trace.py --input "$out" \
    --out "$DATA/manifests/processed_validation_${split}.json"
  $PYTHON scripts/data/characterize_processed_trace.py --input "$out" \
    --out "$DATA/characterization/${split}.json"
done

echo "finish_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$DATA/manifests/job_${SLURM_JOB_ID:-local}_convert_meta.txt"
echo "exit_status=0" | tee -a "$DATA/manifests/job_${SLURM_JOB_ID:-local}_convert_meta.txt"
