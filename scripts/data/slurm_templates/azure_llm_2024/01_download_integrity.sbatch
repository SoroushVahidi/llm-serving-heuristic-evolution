#!/bin/bash
#SBATCH --job-name=tier1_dl_azure24
#SBATCH --partition=general
#SBATCH --qos=standard
#SBATCH --account=ikoutis
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=/mmfs1/project/ikoutis/sv96/llmserveopt-data/datasets/azure_llm_2024/logs/%x_%j.out
#SBATCH --error=/mmfs1/project/ikoutis/sv96/llmserveopt-data/datasets/azure_llm_2024/logs/%x_%j.err

set -euo pipefail

REPO=/mmfs1/project/ikoutis/sv96/github/llm-serving-heuristic-evolution-dataset-expansion
DATA=/mmfs1/project/ikoutis/sv96/llmserveopt-data/datasets/azure_llm_2024
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
  echo "cmdline=download_azure_llm_2024.py"
} | tee "$DATA/manifests/job_${SLURM_JOB_ID:-local}_download_meta.txt"

$PYTHON scripts/data/download_azure_llm_2024.py --output-dir "$DATA/raw"

# Write download manifest with checksums
$PYTHON - <<'PY'
import hashlib, json, time
from pathlib import Path
DATA = Path("/mmfs1/project/ikoutis/sv96/llmserveopt-data/datasets/azure_llm_2024")
expected = {
  "AzureLLMInferenceTrace_code_2024.csv": 691989454,
  "AzureLLMInferenceTrace_conv_2024.csv": 1135195393,
}
files = []
for name, exp in expected.items():
    p = DATA / "raw" / name
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    files.append({
        "file": name,
        "bytes": p.stat().st_size,
        "expected_bytes_estimate": exp,
        "size_matches_estimate": p.stat().st_size == exp,
        "sha256": h.hexdigest(),
        "url": f"https://github.com/Azure/AzurePublicDataset/releases/download/dataset-llm-2024/{name.replace('_2024','').replace('AzureLLMInferenceTrace_code.csv','AzureLLMInferenceTrace_code_1week.csv').replace('AzureLLMInferenceTrace_conv.csv','AzureLLMInferenceTrace_conv_1week.csv')}",
    })
# Fix URLs properly
files[0]["url"] = "https://github.com/Azure/AzurePublicDataset/releases/download/dataset-llm-2024/AzureLLMInferenceTrace_code_1week.csv"
files[1]["url"] = "https://github.com/Azure/AzurePublicDataset/releases/download/dataset-llm-2024/AzureLLMInferenceTrace_conv_1week.csv"
man = {
  "dataset": "Azure_LLM_Inference_2024",
  "official_pin": "release:dataset-llm-2024",
  "access_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
  "code_repo_license": "CC-BY-4.0",
  "data_license": "CC-BY-4.0",
  "files": files,
}
(DATA/"manifests"/"download_manifest.json").write_text(json.dumps(man, indent=2)+"\n")
print(json.dumps(files, indent=2))
PY

for f in "$DATA"/raw/*.csv; do
  base=$(basename "$f")
  $PYTHON scripts/data/validate_raw_integrity.py \
    --path "$f" --format csv \
    --source-revision "release:dataset-llm-2024" \
    --out "$DATA/manifests/integrity_${base}.json"
done

echo "finish_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$DATA/manifests/job_${SLURM_JOB_ID:-local}_download_meta.txt"
echo "exit_status=0" | tee -a "$DATA/manifests/job_${SLURM_JOB_ID:-local}_download_meta.txt"
