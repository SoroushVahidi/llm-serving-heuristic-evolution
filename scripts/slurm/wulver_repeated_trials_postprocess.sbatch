#!/bin/bash -l
#
# CPU-only statistical postprocessing for the repeated-trial Sarathi-vs-vLLM
# arrays. No GPU requested. Submitted with
# --dependency=afterok:<sarathi_array_job_id>,afterok:<vllm_array_job_id>
# so it only runs once every trial in both arrays has completed
# successfully.
#
# Required environment variables (set by the submitting sbatch/shell, not
# hardcoded here, since the array job IDs are only known at submission
# time):
#   SARATHI_ARRAY_JOB_ID
#   VLLM_ARRAY_JOB_ID

#SBATCH --job-name=repeated-trials-postprocess
#SBATCH --partition=general
#SBATCH --account=ikoutis
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err

set -euo pipefail

: "${SARATHI_ARRAY_JOB_ID:?must be set}"
: "${VLLM_ARRAY_JOB_ID:?must be set}"

REPO_DIR="/mmfs1/project/ikoutis/sv96/github/llm-serving-heuristic-evolution"
SCRATCH_ROOT="/mmfs1/scratch/ikoutis/sv96"
COMPARISON_OUT_DIR="${REPO_DIR}/experiments/gpu_external_validity/sarathi_vllm_repeated_trials"

source /apps/easybuild/software/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate repo-env

mkdir -p "${REPO_DIR}/logs" "${COMPARISON_OUT_DIR}"
cd "${REPO_DIR}"

REPORT="${SCRATCH_ROOT}/repeated_trials_postprocess_${SLURM_JOB_ID}_report.txt"
{
  echo "=== Repeated-trials postprocess job ${SLURM_JOB_ID} ==="
  echo "timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "sarathi_array_job_id=${SARATHI_ARRAY_JOB_ID}"
  echo "vllm_array_job_id=${VLLM_ARRAY_JOB_ID}"
  echo

  echo "--- 1. Final states of both arrays ---"
  echo "Sarathi array:"
  sacct -j "${SARATHI_ARRAY_JOB_ID}" --format=JobID,JobName,State,Elapsed,ExitCode -P || true
  echo "vLLM array:"
  sacct -j "${VLLM_ARRAY_JOB_ID}" --format=JobID,JobName,State,Elapsed,ExitCode -P || true
  echo

  echo "--- 2. Statistical analysis ---"
  python3 scripts/analyze_repeated_trials.py \
    --sarathi-glob "${SCRATCH_ROOT}/sarathi_repeated_trial_${SARATHI_ARRAY_JOB_ID}_*" \
    --vllm-glob "${SCRATCH_ROOT}/vllm_repeated_trial_${VLLM_ARRAY_JOB_ID}_*" \
    --output-dir "${COMPARISON_OUT_DIR}"
  echo

  echo "--- 3. Summary ---"
  cat "${COMPARISON_OUT_DIR}/repeated_trials_report.md" 2>/dev/null || echo "report not found"
} 2>&1 | tee "${REPORT}"

cp "${REPORT}" "${COMPARISON_OUT_DIR}/postprocess_report.txt"
echo "report=${REPORT}"
echo "comparison_dir=${COMPARISON_OUT_DIR}"
