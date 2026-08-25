#!/bin/bash -l
#
# Policy Separation Dataset v1 -- three-case diagnostic (FIRST compute
# experiment). Runs scripts/run_policy_separation_three_case.py, which
# evaluates exactly three theory-grounded scenario families (FCFS convoy /
# head-of-line blocking, SJF size-prediction inversion, EDF unsalvageable
# overload) against the actual registered policies to check whether the
# current simulator reproduces reproducible, interpretable policy
# separation for each mechanism. NOT the full 5-family/25-template Policy
# Separation Dataset v1 corpus, no Sobol/MAP-Elites/selector training.
#
# Simulator-only, CPU-only workload (no vLLM/torch/GPU dependency) --
# confirmed by inspection: run_policy.py/simulator.py are pure Python/numpy.
# Timed on the login node before submission: ~3.3 (scenario,policy) tasks/s
# with 8 worker processes (2-seed partial timing run, 808 tasks in ~4.5min),
# so the full 10-seed / 4040-task run is expected to take roughly 20-25
# minutes -- 02:00:00 leaves generous margin, not a tight fit.

#SBATCH --job-name=policy-separation-three-case-v1
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
# (Not -e: this job's own diagnostics below should still record hostname/
# date/git-state even if an earlier diagnostic step has a hiccup; the
# actual experiment invocation's exit code is captured explicitly and is
# what determines the wrapper's own exit status -- see the end of this
# script.)

REPO_DIR="/mmfs1/project/ikoutis/sv96/github/llm-serving-heuristic-evolution-policy-separation-v1"
PYTHON="/home/sv96/.conda/envs/repo-env/bin/python"
SCRATCH_ROOT="/mmfs1/scratch/ikoutis/sv96"
TS=$(date -u +%Y%m%dT%H%M%SZ)
RUN_DIR="${SCRATCH_ROOT}/policy_separation_three_case_${TS}_${SLURM_JOB_ID}"
CONFIG="${REPO_DIR}/configs/policy_separation_three_case_v1.yaml"

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

# --resume is always passed: harmless on a fresh RUN_DIR (nothing to
# resume), and means a manual re-submission pointed at the same RUN_DIR
# after a preemption/requeue continues rather than silently overwriting.
"${PYTHON}" "${REPO_DIR}/scripts/run_policy_separation_three_case.py" \
  --config "${CONFIG}" \
  --run-dir "${RUN_DIR}" \
  --workers "${WORKERS}" \
  --resume
STATUS=$?

echo "wrapper_exit_code=${STATUS}" | tee -a "${RUN_DIR}/slurm_manifest.txt"
echo "finished_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${RUN_DIR}/slurm_manifest.txt"

exit "${STATUS}"
