#!/bin/bash
#SBATCH --job-name=tier1_cv_azure24
#SBATCH --partition=general
#SBATCH --qos=standard
#SBATCH --account=ikoutis
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/mmfs1/project/ikoutis/sv96/llmserveopt-data/datasets/azure_llm_2024/logs/%x_%j.out
#SBATCH --error=/mmfs1/project/ikoutis/sv96/llmserveopt-data/datasets/azure_llm_2024/logs/%x_%j.err

set -euo pipefail

REPO=/mmfs1/project/ikoutis/sv96/github/llm-serving-heuristic-evolution-dataset-expansion
DATA=/mmfs1/project/ikoutis/sv96/llmserveopt-data/datasets/azure_llm_2024
PYTHON=/home/sv96/.conda/envs/repo-env/bin/python
cd "$REPO"
export PYTHONPATH=src

{
  echo "job_id=${SLURM_JOB_ID:-local}"
  echo "start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host=$(hostname)"
  echo "git_sha=$(git rev-parse HEAD)"
  echo "note=natural_replay_time_scale_1.0_full_files"
} | tee "$DATA/manifests/job_${SLURM_JOB_ID:-local}_convert_meta.txt"

for split in code conv; do
  in="$DATA/raw/AzureLLMInferenceTrace_${split}_2024.csv"
  out="$DATA/processed/azure_llm_2024_${split}.jsonl"
  $PYTHON scripts/data/convert_azure_llm_trace.py \
    --input "$in" \
    --output "$out" \
    --source "azure_2024_${split}" \
    --time-scale 1.0 \
    --seed 17
  $PYTHON scripts/data/validate_processed_trace.py --input "$out" \
    --out "$DATA/manifests/processed_validation_${split}.json"
  $PYTHON scripts/data/characterize_processed_trace.py --input "$out" \
    --out "$DATA/characterization/${split}.json"
done

# Document schema compatibility vs 2023
$PYTHON - <<'PY'
import csv, json
from pathlib import Path
p23 = Path("/mmfs1/project/ikoutis/sv96/llmserveopt-data/datasets/azure_llm_2023/raw/AzureLLMInferenceTrace_code_2023.csv")
p24 = Path("/mmfs1/project/ikoutis/sv96/llmserveopt-data/datasets/azure_llm_2024/raw/AzureLLMInferenceTrace_code_2024.csv")
h23 = next(csv.reader(open(p23, encoding="utf-8-sig")))
h24 = next(csv.reader(open(p24, encoding="utf-8-sig")))
out = {
  "headers_2023": h23,
  "headers_2024": h24,
  "identical_headers": h23 == h24,
  "differences": [],
}
if h23 != h24:
    out["differences"] = {"only_2023": sorted(set(h23)-set(h24)), "only_2024": sorted(set(h24)-set(h23))}
Path("/mmfs1/project/ikoutis/sv96/llmserveopt-data/datasets/azure_llm_2024/manifests/schema_vs_2023.json").write_text(json.dumps(out, indent=2)+"\n")
print(out)
PY

echo "finish_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$DATA/manifests/job_${SLURM_JOB_ID:-local}_convert_meta.txt"
echo "exit_status=0" | tee -a "$DATA/manifests/job_${SLURM_JOB_ID:-local}_convert_meta.txt"
