#!/bin/bash
#SBATCH --job-name=tier1_dl_bailian
#SBATCH --partition=general
#SBATCH --qos=standard
#SBATCH --account=ikoutis
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/mmfs1/project/ikoutis/sv96/llmserveopt-data/datasets/bailian_qwen/logs/%x_%j.out
#SBATCH --error=/mmfs1/project/ikoutis/sv96/llmserveopt-data/datasets/bailian_qwen/logs/%x_%j.err

set -euo pipefail

REPO=/mmfs1/project/ikoutis/sv96/github/llm-serving-heuristic-evolution-dataset-expansion
DATA=/mmfs1/project/ikoutis/sv96/llmserveopt-data/datasets/bailian_qwen
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

$PYTHON scripts/data/download_bailian_traces.py \
  --output-dir "$DATA/raw" \
  --manifest-out "$DATA/manifests/download_manifest.json"

for f in "$DATA"/raw/*.jsonl; do
  base=$(basename "$f")
  $PYTHON scripts/data/validate_raw_integrity.py \
    --path "$f" --format jsonl \
    --source-revision "commit:5f7439c51ec248a0c585f7d90a41a6f57773b912" \
    --out "$DATA/manifests/integrity_${base}.json"
done

echo "finish_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$DATA/manifests/job_${SLURM_JOB_ID:-local}_download_meta.txt"
echo "exit_status=0" | tee -a "$DATA/manifests/job_${SLURM_JOB_ID:-local}_download_meta.txt"
