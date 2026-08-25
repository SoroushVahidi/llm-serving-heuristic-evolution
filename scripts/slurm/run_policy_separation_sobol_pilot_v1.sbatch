#!/bin/bash -l
#
# ############################################################################
# # REVIEWED DESIGN REQUIRED BEFORE SBATCH.
# #
# # This script is PREPARED, NOT AUTHORIZED FOR SUBMISSION. Do not `sbatch`
# # this file until the Sobol pilot design has been explicitly reviewed --
# # see docs/design/POLICY_SEPARATION_SOBOL_PILOT_V1.md's section 10
# # (SOBOL_DESIGN_READY_FOR_REVIEW: YES; READY_TO_SBATCH_AFTER_REVIEW is a
# # separate, later authorization, not implied by this file existing).
# ############################################################################
#
# Policy Separation Sobol Pilot v1 -- first space-filling exploration stage,
# following jobs 1170116 (three-case diagnostic) and 1171116 (boundary
# refinement). Runs scripts/run_policy_separation_sobol_pilot_v1.py: two
# independent Sobol subspaces (prediction-sensitive scheduling; deadline/
# admission scheduling) plus one small non-Sobol categorical FCFS add-on,
# built strictly from dimensions validated by those two prior jobs. This is
# a landscape-characterization pilot (~1,616 scenarios, ~6,976 evaluations),
# NOT the eventual 8K-12K MAP-Elites-ready dataset, and NOT MAP-Elites/
# CMA-ES/selector training itself.
#
# Simulator-only, CPU-only workload (same run_policy.py/simulator.py path
# as jobs 1170116/1171116, pure Python/numpy, no vLLM/torch/GPU dependency).
# Both prior jobs ran ~50-60 (scenario,policy) tasks/s with 8 CPU workers;
# at that throughput 6,976 evals should take roughly 6976/55 =~ 127s
# (~2min) of execution phase. 00:30:00 leaves generous margin.

#SBATCH --job-name=policy-separation-sobol-pilot-v1
#SBATCH --partition=general
#SBATCH --account=ikoutis
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=00:30:00
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
RUN_DIR="${SCRATCH_ROOT}/policy_separation_sobol_pilot_${TS}_${SLURM_JOB_ID}"
CONFIG="${REPO_DIR}/configs/policy_separation_sobol_pilot_v1.yaml"

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
# NOTE: no --dry-run flag here -- this invocation runs the FULL scale
# config (1,616 scenarios / 6,976 evaluations), which is the scientific
# sweep this file's header comment says is not yet authorized to submit.
"${PYTHON}" "${REPO_DIR}/scripts/run_policy_separation_sobol_pilot_v1.py" \
  --config "${CONFIG}" \
  --run-dir "${RUN_DIR}" \
  --workers "${WORKERS}" \
  --resume
STATUS=$?

echo "wrapper_exit_code=${STATUS}" | tee -a "${RUN_DIR}/slurm_manifest.txt"
echo "finished_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${RUN_DIR}/slurm_manifest.txt"

exit "${STATUS}"
