#!/bin/bash
#SBATCH --job-name=j240-sbs-tlabel-full
#SBATCH --partition=general
#SBATCH --qos=standard
#SBATCH --account=ikoutis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --array=0-79
#SBATCH --output=/mmfs1/scratch/ikoutis/sv96/joint240_sbs_targeted_terminal_label_full_v1/logs/%x_%A_%a.out
#SBATCH --error=/mmfs1/scratch/ikoutis/sv96/joint240_sbs_targeted_terminal_label_full_v1/logs/%x_%A_%a.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/mmfs1/project/ikoutis/sv96/llm-serving-heuristic-evolution-joint240-sbs-targeted-terminal-label-full-v1}"
PYTHON="${PYTHON:-/home/sv96/.conda/envs/repo-env/bin/python}"
RUN_ROOT="${RUN_ROOT:-/mmfs1/scratch/ikoutis/sv96/joint240_sbs_targeted_terminal_label_full_v1/full_v1}"
MANIFEST="${MANIFEST:-$RUN_ROOT/full_manifest_all_disagreements.csv}"
NUM_SHARDS="${NUM_SHARDS:-80}"

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/manifests"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR/src:${PYTHONPATH:-}"

{
  echo "job_id=${SLURM_JOB_ID:-}"
  echo "array_task_id=${SLURM_ARRAY_TASK_ID:-0}"
  echo "hostname=$(hostname)"
  echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "repo_dir=${REPO_DIR}"
  echo "run_root=${RUN_ROOT}"
  echo "manifest=${MANIFEST}"
  echo "num_shards=${NUM_SHARDS}"
  "${PYTHON}" --version
  git rev-parse HEAD || true
  git status --short --branch || true
  sha256sum "${MANIFEST}" || true
} > "${RUN_ROOT}/manifests/provenance.${SLURM_ARRAY_JOB_ID:-manual}.${SLURM_ARRAY_TASK_ID:-0}.txt"

"${PYTHON}" scripts/run_joint240_sbs_targeted_terminal_label_full_v1.py \
  --output-dir "$RUN_ROOT" \
  run-shard \
  --pilot-manifest "$MANIFEST" \
  --shard-index "${SLURM_ARRAY_TASK_ID:-0}" \
  --num-shards "$NUM_SHARDS"
