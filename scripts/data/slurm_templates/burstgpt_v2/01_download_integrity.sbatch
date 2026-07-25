#!/bin/bash
#SBATCH --job-name=tier1_dl_burstgpt
#SBATCH --partition=general
#SBATCH --qos=standard
#SBATCH --account=ikoutis
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/mmfs1/project/ikoutis/sv96/llmserveopt-data/datasets/burstgpt_v2/logs/%x_%j.out
#SBATCH --error=/mmfs1/project/ikoutis/sv96/llmserveopt-data/datasets/burstgpt_v2/logs/%x_%j.err

set -euo pipefail

REPO=/mmfs1/project/ikoutis/sv96/github/llm-serving-heuristic-evolution-dataset-expansion
DATA=/mmfs1/project/ikoutis/sv96/llmserveopt-data/datasets/burstgpt_v2
PYTHON=/home/sv96/.conda/envs/repo-env/bin/python
START_TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
HOST=$(hostname)
GIT_SHA=$(cd "$REPO" && git rev-parse HEAD)

mkdir -p "$DATA"/{raw,processed,manifests,characterization,jobs,logs}
cd "$REPO"
export PYTHONPATH=src

{
  echo "job_id=${SLURM_JOB_ID:-local}"
  echo "start_utc=$START_TS"
  echo "host=$HOST"
  echo "git_sha=$GIT_SHA"
  echo "python=$($PYTHON --version 2>&1)"
  echo "conda_env=repo-env"
  echo "cmdline=download_burstgpt_v2.py --files without_fails"
} | tee "$DATA/manifests/job_${SLURM_JOB_ID:-local}_download_meta.txt"

$PYTHON scripts/data/download_burstgpt_v2.py \
  --output-dir "$DATA/raw" \
  --files without_fails \
  --manifest-out "$DATA/manifests/download_manifest.json"

for f in "$DATA"/raw/BurstGPT_without_fails_*.csv; do
  base=$(basename "$f")
  $PYTHON scripts/data/validate_raw_integrity.py \
    --path "$f" --format csv \
    --source-url "https://github.com/HPMLL/BurstGPT/releases/download/v2.0/${base}" \
    --source-revision "release_tag:v2.0" \
    --out "$DATA/manifests/integrity_${base}.json"
done

# Also download full (with fails) for provenance comparison / 10k lineage, optional small cost
$PYTHON scripts/data/download_burstgpt_v2.py \
  --output-dir "$DATA/raw" \
  --files raw \
  --manifest-out "$DATA/manifests/download_manifest_raw_with_fails.json"

for f in "$DATA"/raw/BurstGPT_{1,2,3}.csv; do
  base=$(basename "$f")
  $PYTHON scripts/data/validate_raw_integrity.py \
    --path "$f" --format csv \
    --source-url "https://github.com/HPMLL/BurstGPT/releases/download/v2.0/${base}" \
    --source-revision "release_tag:v2.0" \
    --out "$DATA/manifests/integrity_${base}.json"
done

END_TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
echo "finish_utc=$END_TS" | tee -a "$DATA/manifests/job_${SLURM_JOB_ID:-local}_download_meta.txt"
echo "exit_status=0" | tee -a "$DATA/manifests/job_${SLURM_JOB_ID:-local}_download_meta.txt"
