#!/bin/bash
#SBATCH --job-name=tier1_cv_mooncake
#SBATCH --partition=general
#SBATCH --qos=standard
#SBATCH --account=ikoutis
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/mmfs1/project/ikoutis/sv96/llmserveopt-data/datasets/mooncake/logs/%x_%j.out
#SBATCH --error=/mmfs1/project/ikoutis/sv96/llmserveopt-data/datasets/mooncake/logs/%x_%j.err

set -euo pipefail

REPO=/mmfs1/project/ikoutis/sv96/github/llm-serving-heuristic-evolution-dataset-expansion
DATA=/mmfs1/project/ikoutis/sv96/llmserveopt-data/datasets/mooncake
PYTHON=/home/sv96/.conda/envs/repo-env/bin/python
cd "$REPO"
export PYTHONPATH=src

{
  echo "job_id=${SLURM_JOB_ID:-local}"
  echo "start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host=$(hostname)"
  echo "git_sha=$(git rev-parse HEAD)"
  echo "DATA_LICENSE=NOT_EXPLICITLY_SPECIFIED"
  echo "convert_real_only=true"
} | tee "$DATA/manifests/job_${SLURM_JOB_ID:-local}_convert_meta.txt"

# Never convert synthetic as real
for split in conversation_trace toolagent_trace; do
  in="$DATA/raw_real/${split}.jsonl"
  out="$DATA/processed_real/mooncake_${split}.jsonl"
  $PYTHON scripts/data/convert_mooncake_trace.py \
    --input "$in" \
    --output "$out" \
    --source-split "$split" \
    --time-scale 1.0 \
    --seed 17 \
    --require-real-only
  $PYTHON scripts/data/validate_processed_trace.py --input "$out" \
    --out "$DATA/manifests/processed_validation_${split}.json"
  $PYTHON scripts/data/characterize_processed_trace.py --input "$out" \
    --out "$DATA/characterization/${split}.json"
done

# Prove synthetic path is refused under require_real_only
set +e
$PYTHON scripts/data/convert_mooncake_trace.py \
  --input "$DATA/raw_synthetic_quarantine/synthetic_trace.jsonl" \
  --output "$DATA/processed_real/SHOULD_NOT_EXIST.jsonl" \
  --source-split synthetic_trace \
  --require-real-only
rc=$?
set -e
if [[ $rc -eq 0 ]]; then
  echo "ERROR: synthetic conversion should have failed under --require-real-only" >&2
  exit 2
fi
echo "synthetic_refused_ok rc=$rc"
rm -f "$DATA/processed_real/SHOULD_NOT_EXIST.jsonl" "$DATA/processed_real/SHOULD_NOT_EXIST.report.json"

echo "finish_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$DATA/manifests/job_${SLURM_JOB_ID:-local}_convert_meta.txt"
echo "exit_status=0" | tee -a "$DATA/manifests/job_${SLURM_JOB_ID:-local}_convert_meta.txt"
