#!/bin/bash
#SBATCH --job-name=tier1_dl_azure23
#SBATCH --partition=general
#SBATCH --qos=standard
#SBATCH --account=ikoutis
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=/mmfs1/project/ikoutis/sv96/llmserveopt-data/datasets/azure_llm_2023/logs/%x_%j.out
#SBATCH --error=/mmfs1/project/ikoutis/sv96/llmserveopt-data/datasets/azure_llm_2023/logs/%x_%j.err

set -euo pipefail

REPO=/mmfs1/project/ikoutis/sv96/github/llm-serving-heuristic-evolution-dataset-expansion
DATA=/mmfs1/project/ikoutis/sv96/llmserveopt-data/datasets/azure_llm_2023
PYTHON=/home/sv96/.conda/envs/repo-env/bin/python
START_TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)

mkdir -p "$DATA"/{raw,processed,manifests,characterization,jobs,logs}
cd "$REPO"
export PYTHONPATH=src

{
  echo "job_id=${SLURM_JOB_ID:-local}"
  echo "start_utc=$START_TS"
  echo "host=$(hostname)"
  echo "git_sha=$(git rev-parse HEAD)"
  echo "python=$($PYTHON --version 2>&1)"
} | tee "$DATA/manifests/job_${SLURM_JOB_ID:-local}_download_meta.txt"

$PYTHON scripts/data/download_azure_llm_2023.py \
  --output-dir "$DATA/raw" \
  --manifest-out "$DATA/manifests/download_manifest.json"

for f in "$DATA"/raw/*.csv; do
  base=$(basename "$f")
  $PYTHON scripts/data/validate_raw_integrity.py \
    --path "$f" --format csv \
    --source-revision "AzureLLMInferenceDataset2023.md" \
    --out "$DATA/manifests/integrity_${base}.json"
done

echo "finish_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$DATA/manifests/job_${SLURM_JOB_ID:-local}_download_meta.txt"
echo "exit_status=0" | tee -a "$DATA/manifests/job_${SLURM_JOB_ID:-local}_download_meta.txt"
