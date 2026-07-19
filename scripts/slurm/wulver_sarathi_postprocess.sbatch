#!/bin/bash -l
#
# CPU-only postprocessing for the completed Sarathi float16 GPU job
# (1111723): validate its outputs, run the matched Sarathi-vs-vLLM
# comparison against job 1111706, and copy small canonical artifacts into
# the repo. No GPU requested. Intended to be submitted with
# --dependency=afterok:1111723 so it runs automatically once that job
# finishes, whether or not anyone is watching.

#SBATCH --job-name=sarathi-postprocess
#SBATCH --partition=general
#SBATCH --account=ikoutis
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err

set -euo pipefail

REPO_DIR="/mmfs1/project/ikoutis/sv96/github/llm-serving-heuristic-evolution"
SCRATCH_ROOT="/mmfs1/scratch/ikoutis/sv96"
VLLM_DIR="${SCRATCH_ROOT}/vllm_mistral_match_1111706"
SARATHI_DIR="${SCRATCH_ROOT}/sarathi_mistral_fp16_final_1111723"
COMPARISON_OUT_DIR="${REPO_DIR}/experiments/gpu_external_validity/sarathi_vs_vllm_mistral_comparison"
CANONICAL_DIR="${REPO_DIR}/experiments/gpu_external_validity/sarathi_mistral_fp16_final_1111723"

source /apps/easybuild/software/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate repo-env

mkdir -p "${REPO_DIR}/logs" "${CANONICAL_DIR}"
cd "${REPO_DIR}"

REPORT="${SCRATCH_ROOT}/sarathi_postprocess_${SLURM_JOB_ID}_report.txt"
{
  echo "=== Postprocess job ${SLURM_JOB_ID} for GPU job 1111723 ==="
  echo "timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host=$(hostname)"
  echo "branch=$(git rev-parse --abbrev-ref HEAD)"
  echo "commit=$(git rev-parse HEAD)"
  echo

  echo "--- 1. Final state of GPU job 1111723 ---"
  sacct -j 1111723 --format=JobID,JobName,State,Elapsed,ExitCode -P || true
  echo

  echo "--- 2. Output validation ---"
  for f in smoke_result.json scenario_results.json summary.json; do
    path="${SARATHI_DIR}/${f}"
    if [ -f "${path}" ]; then
      if python3 -c "import json; json.load(open('${path}'))" 2>/dev/null; then
        echo "OK: ${f} exists and is valid JSON"
      else
        echo "ERROR: ${f} exists but is not valid JSON"
      fi
    else
      echo "MISSING: ${f} not found at ${path}"
    fi
  done
  echo

  echo "--- 3. Matched Sarathi-vs-vLLM comparison ---"
  python3 scripts/compare_sarathi_vllm_matched_runtime.py \
    --vllm-dir "${VLLM_DIR}" \
    --sarathi-dir "${SARATHI_DIR}" \
    --output-dir "${COMPARISON_OUT_DIR}"
  echo

  echo "--- 4. Canonical artifact copy (small files only) ---"
  for f in smoke_result.json scenario_results.json summary.json job_manifest.txt; do
    if [ -f "${SARATHI_DIR}/${f}" ]; then
      cp "${SARATHI_DIR}/${f}" "${CANONICAL_DIR}/${f}"
      echo "copied ${f} -> ${CANONICAL_DIR}/${f}"
    fi
  done
  echo

  echo "--- 5. Summary ---"
  cat "${COMPARISON_OUT_DIR}/matched_comparison.md" 2>/dev/null || echo "comparison markdown not found"
} 2>&1 | tee "${REPORT}"

cp "${REPORT}" "${CANONICAL_DIR}/postprocess_report.txt"

echo "report=${REPORT}"
echo "canonical_dir=${CANONICAL_DIR}"
echo "comparison_dir=${COMPARISON_OUT_DIR}"
