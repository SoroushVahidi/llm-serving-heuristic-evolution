#!/bin/bash
#SBATCH --job-name=tier1_dl_mooncake
#SBATCH --partition=general
#SBATCH --qos=standard
#SBATCH --account=ikoutis
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=/mmfs1/project/ikoutis/sv96/llmserveopt-data/datasets/mooncake/logs/%x_%j.out
#SBATCH --error=/mmfs1/project/ikoutis/sv96/llmserveopt-data/datasets/mooncake/logs/%x_%j.err

set -euo pipefail

REPO=/mmfs1/project/ikoutis/sv96/github/llm-serving-heuristic-evolution-dataset-expansion
DATA=/mmfs1/project/ikoutis/sv96/llmserveopt-data/datasets/mooncake
PYTHON=/home/sv96/.conda/envs/repo-env/bin/python
START_TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)

mkdir -p "$DATA"/{raw_real,raw_synthetic_quarantine,processed_real,manifests,characterization,jobs,logs}
cd "$REPO"
export PYTHONPATH=src

{
  echo "job_id=${SLURM_JOB_ID:-local}"
  echo "start_utc=$START_TS"
  echo "host=$(hostname)"
  echo "git_sha=$(git rev-parse HEAD)"
  echo "python=$($PYTHON --version 2>&1)"
  echo "DATA_LICENSE=NOT_EXPLICITLY_SPECIFIED"
  echo "REDISTRIBUTION=PROHIBITED_UNTIL_CLARIFIED"
} | tee "$DATA/manifests/job_${SLURM_JOB_ID:-local}_download_meta.txt"

$PYTHON scripts/data/download_mooncake_traces.py \
  --real-dir "$DATA/raw_real" \
  --synthetic-dir "$DATA/raw_synthetic_quarantine" \
  --manifest-out "$DATA/manifests/download_manifest.json"

for f in "$DATA"/raw_real/*.jsonl; do
  base=$(basename "$f")
  $PYTHON scripts/data/validate_raw_integrity.py \
    --path "$f" --format jsonl \
    --source-revision "FAST25-release/traces" \
    --out "$DATA/manifests/integrity_real_${base}.json"
done
for f in "$DATA"/raw_synthetic_quarantine/*.jsonl; do
  base=$(basename "$f")
  $PYTHON scripts/data/validate_raw_integrity.py \
    --path "$f" --format jsonl \
    --source-revision "FAST25-release/traces" \
    --out "$DATA/manifests/integrity_synthetic_${base}.json"
done

# Refuse any ambiguous unclassified files in raw_real
$PYTHON - <<'PY'
from pathlib import Path
allowed = {"conversation_trace.jsonl", "toolagent_trace.jsonl"}
real = Path("/mmfs1/project/ikoutis/sv96/llmserveopt-data/datasets/mooncake/raw_real")
extra = [p.name for p in real.glob("*") if p.is_file() and p.name not in allowed and not p.name.endswith(".partial")]
if extra:
    raise SystemExit(f"Ambiguous/extra files in raw_real: {extra}")
print("real_only_enforcement_ok")
PY

echo "finish_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$DATA/manifests/job_${SLURM_JOB_ID:-local}_download_meta.txt"
echo "exit_status=0" | tee -a "$DATA/manifests/job_${SLURM_JOB_ID:-local}_download_meta.txt"
