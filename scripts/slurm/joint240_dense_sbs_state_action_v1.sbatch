#!/bin/bash
#SBATCH --job-name=joint240-dense-sbs-v1
#SBATCH --partition=general
#SBATCH --qos=standard
#SBATCH --account=ikoutis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --array=0-39
#SBATCH --output=/mmfs1/scratch/ikoutis/sv96/joint240_dense_sbs_state_action_v1/logs/%x_%A_%a.out
#SBATCH --error=/mmfs1/scratch/ikoutis/sv96/joint240_dense_sbs_state_action_v1/logs/%x_%A_%a.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/mmfs1/project/ikoutis/sv96/github/llm-serving-heuristic-evolution}"
PYTHON="${PYTHON:-/home/sv96/.conda/envs/repo-env/bin/python}"
RUN_ROOT="${RUN_ROOT:-/mmfs1/scratch/ikoutis/sv96/joint240_dense_sbs_state_action_v1}"
STATE_UNIVERSE="${STATE_UNIVERSE:-${RUN_ROOT}/state_universe.csv}"
NUM_SHARDS="${NUM_SHARDS:-40}"

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/manifests"
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/src"

{
  echo "job_id=${SLURM_JOB_ID:-}"
  echo "array_task_id=${SLURM_ARRAY_TASK_ID:-0}"
  echo "hostname=$(hostname)"
  echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "repo_dir=${REPO_DIR}"
  echo "run_root=${RUN_ROOT}"
  echo "state_universe=${STATE_UNIVERSE}"
  echo "num_shards=${NUM_SHARDS}"
  git rev-parse HEAD || true
  git status --short --branch || true
} > "${RUN_ROOT}/manifests/provenance.${SLURM_ARRAY_JOB_ID:-manual}.${SLURM_ARRAY_TASK_ID:-0}.txt"

"${PYTHON}" scripts/run_joint240_dense_sbs_state_action_v1.py \
  --output-dir "${RUN_ROOT}" \
  run-shard \
  --state-universe "${STATE_UNIVERSE}" \
  --shard-index "${SLURM_ARRAY_TASK_ID:-0}" \
  --num-shards "${NUM_SHARDS}"
