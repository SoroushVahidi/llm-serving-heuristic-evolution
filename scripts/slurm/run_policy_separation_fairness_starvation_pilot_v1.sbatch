#!/bin/bash -l
#
# ############################################################################
# # HISTORICAL PILOT COMPLETED: Slurm Job 1182306 (2026-08-16).
# #
# # Provenance (do not overwrite):
# #   scratch: /mmfs1/scratch/ikoutis/sv96/policy_separation_fairness_starvation_pilot_20260816T211029Z_1182306
# #   repo copy: experiments/policy_separation_fairness_starvation_pilot_20260816T211029Z_1182306/
# #
# # Job 1182306 used synthetic token-length fallback (BurstGPT path miss) and
# # wrote unweighted SLO-success under column name `anwg`. Any future resubmit
# # is a NEW corrected run and must use a new run directory / job id.
# ############################################################################
#
# Policy Separation Fairness and Starvation Pilot v1 (Family A)
# Runs scripts/run_policy_separation_fairness_starvation_pilot_v1.py over
# the 120 scenario grid (480 policy evaluations).
#
# Simulator-only, CPU-only workload. Pure Python/numpy, no vLLM/torch/GPU dependency.

#SBATCH --job-name=policy-separation-fairness-starvation-pilot-v1
#SBATCH --partition=general
#SBATCH --account=ikoutis
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/mmfs1/project/ikoutis/sv96/github/llm-serving-heuristic-evolution-policy-separation-v1/logs/%x.%j.out
#SBATCH --error=/mmfs1/project/ikoutis/sv96/github/llm-serving-heuristic-evolution-policy-separation-v1/logs/%x.%j.err

set -uo pipefail

REPO_DIR="/mmfs1/project/ikoutis/sv96/github/llm-serving-heuristic-evolution-policy-separation-v1"
PYTHON="/home/sv96/.conda/envs/repo-env/bin/python"
SCRATCH_ROOT="/mmfs1/scratch/ikoutis/sv96"
TS=$(date -u +%Y%m%dT%H%M%SZ)
RUN_DIR="${SCRATCH_ROOT}/policy_separation_fairness_starvation_pilot_${TS}_${SLURM_JOB_ID}"
CONFIG="${REPO_DIR}/configs/policy_separation_fairness_starvation_pilot_v1.yaml"

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

"${PYTHON}" "${REPO_DIR}/scripts/run_policy_separation_fairness_starvation_pilot_v1.py" \
  --config "${CONFIG}" \
  --run-dir "${RUN_DIR}" \
  --workers "${WORKERS}"
STATUS=$?

echo "wrapper_exit_code=${STATUS}" | tee -a "${RUN_DIR}/slurm_manifest.txt"
echo "finished_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${RUN_DIR}/slurm_manifest.txt"

exit "${STATUS}"
