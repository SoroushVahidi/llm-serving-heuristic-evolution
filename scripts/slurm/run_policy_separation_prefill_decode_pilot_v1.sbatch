#!/bin/bash -l
#
# Family B v1 Prefill-Decode / Chunk-Control Pilot
# 144 scenarios x 5 mechanism variants = 720 evaluations.
# Requires staged BurstGPT. Primary metric: canonical ANWG.
#
#SBATCH --job-name=policy-separation-prefill-decode-pilot-v1
#SBATCH --partition=general
#SBATCH --account=ikoutis
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/mmfs1/project/ikoutis/sv96/github/llm-serving-heuristic-evolution-policy-separation-v1/logs/%x.%j.out
#SBATCH --error=/mmfs1/project/ikoutis/sv96/github/llm-serving-heuristic-evolution-policy-separation-v1/logs/%x.%j.err

set -uo pipefail

REPO_DIR="/mmfs1/project/ikoutis/sv96/github/llm-serving-heuristic-evolution-policy-separation-v1"
PYTHON="/home/sv96/.conda/envs/repo-env/bin/python"
SCRATCH_ROOT="/mmfs1/scratch/ikoutis/sv96"
TS=$(date -u +%Y%m%dT%H%M%SZ)
RUN_DIR="${SCRATCH_ROOT}/policy_separation_prefill_decode_pilot_v1_${TS}_${SLURM_JOB_ID}"
CONFIG="${REPO_DIR}/configs/policy_separation_prefill_decode_pilot_v1.yaml"

mkdir -p "${REPO_DIR}/logs" "${RUN_DIR}"

{
  echo "job_id=${SLURM_JOB_ID}"
  echo "job_name=${SLURM_JOB_NAME:-}"
  echo "host=$(hostname)"
  echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "repo_dir=${REPO_DIR}"
  echo "run_dir=${RUN_DIR}"
  echo "config=${CONFIG}"
} | tee "${RUN_DIR}/slurm_manifest.txt"

cd "${REPO_DIR}"
git rev-parse HEAD | tee -a "${RUN_DIR}/slurm_manifest.txt"
git rev-parse --abbrev-ref HEAD | tee -a "${RUN_DIR}/slurm_manifest.txt"
echo "git_status_short:" | tee -a "${RUN_DIR}/slurm_manifest.txt"
git status --short | tee -a "${RUN_DIR}/slurm_manifest.txt"

"${PYTHON}" --version 2>&1 | tee -a "${RUN_DIR}/slurm_manifest.txt"
echo "python_bin=${PYTHON}" | tee -a "${RUN_DIR}/slurm_manifest.txt"
echo "cpus_per_task=${SLURM_CPUS_PER_TASK:-unset}" | tee -a "${RUN_DIR}/slurm_manifest.txt"

export PYTHONPATH="${REPO_DIR}/src"
export PYTHONUNBUFFERED=1

WORKERS="${SLURM_CPUS_PER_TASK:-8}"

"${PYTHON}" "${REPO_DIR}/scripts/run_policy_separation_prefill_decode_pilot_v1.py" \
  --config "${CONFIG}" \
  --run-dir "${RUN_DIR}" \
  --workers "${WORKERS}" \
  --require-burstgpt
STATUS=$?

echo "wrapper_exit_code=${STATUS}" | tee -a "${RUN_DIR}/slurm_manifest.txt"
echo "finished_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${RUN_DIR}/slurm_manifest.txt"
echo "run_dir=${RUN_DIR}" | tee -a "${RUN_DIR}/slurm_manifest.txt"

exit "${STATUS}"