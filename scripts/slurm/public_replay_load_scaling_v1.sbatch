#!/bin/bash -l
#
# Public Replay Load Scaling v1 -- SLURM array (CPU)
# One array task = one canonical public-trace window (60 tasks total),
# each producing 8 lambda x 8 policy = 64 cells.
#
#SBATCH --job-name=public_replay_load_scaling_v1
#SBATCH --partition=general
#SBATCH --account=ikoutis
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:20:00
#SBATCH --array=0-59%32
#SBATCH --output=/mmfs1/scratch/ikoutis/sv96/llm-serving-heuristic-evolution/public_replay_load_scaling_v1/logs/%x.%A_%a.out
#SBATCH --error=/mmfs1/scratch/ikoutis/sv96/llm-serving-heuristic-evolution/public_replay_load_scaling_v1/logs/%x.%A_%a.err

set -uo pipefail

BUNDLE="/mmfs1/scratch/ikoutis/sv96/llm-serving-heuristic-evolution/public_replay_load_scaling_v1"
PY="/home/sv96/.conda/envs/feedback-weighted-maximization/bin/python"
OUT="${BUNDLE}/experiments/public_replay_load_scaling_v1"

mkdir -p "${OUT}" "${BUNDLE}/logs"

cd "${BUNDLE}"
export PYTHONPATH="${BUNDLE}/src"
export VTC_OFFICIAL_CLONE_PATH="${BUNDLE}/vtc_official_clone"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export GIT_DISCOVERY_ACROSS_FILESYSTEM=1

echo "job_id=${SLURM_JOB_ID} array_job_id=${SLURM_ARRAY_JOB_ID} array_task_id=${SLURM_ARRAY_TASK_ID}"
echo "host=$(hostname)"
echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
"${PY}" --version

"${PY}" scripts/run_public_replay_load_scaling_v1.py \
  --task-index "${SLURM_ARRAY_TASK_ID}" \
  --out-dir "${OUT}"
STATUS=$?

echo "task_exit_code=${STATUS}"
echo "finished_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit "${STATUS}"
