#!/bin/bash
#SBATCH --job-name=sbsfresh-id-tlabel-v1
#SBATCH --partition=general
#SBATCH --qos=standard
#SBATCH --account=ikoutis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --array=0-79
#SBATCH --output=/mmfs1/scratch/ikoutis/sv96/sbs_override_fresh_id_confirmatory_terminal_label_v1/logs/%x_%A_%a.out
#SBATCH --error=/mmfs1/scratch/ikoutis/sv96/sbs_override_fresh_id_confirmatory_terminal_label_v1/logs/%x_%A_%a.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/mmfs1/project/ikoutis/sv96/llm-serving-heuristic-evolution-joint240-sbs-targeted-terminal-label-full-v1-src}"
PYTHON="${PYTHON:-/mmfs1/project/ikoutis/sv96/envs/llm-serving-module-intervention-benchmark-actionable-v1/bin/python}"
RUN_ROOT="${RUN_ROOT:-/mmfs1/scratch/ikoutis/sv96/sbs_override_fresh_id_confirmatory_terminal_label_v1/run_v1}"
NUM_SHARDS="${NUM_SHARDS:-80}"

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/manifests"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR/src:${PYTHONPATH:-}"
export SBS_FRESH_LSSP_WINDOWS_CACHE="${SBS_FRESH_LSSP_WINDOWS_CACHE:-/mmfs1/project/ikoutis/sv96/llmserveopt-data/staging/sbs_override_fresh_confirmatory_corpus_v1/pilot_v2_windows_full_cache.json}"

{
  echo "job_id=${SLURM_JOB_ID:-}"
  echo "array_task_id=${SLURM_ARRAY_TASK_ID:-0}"
  echo "hostname=$(hostname)"
  echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "repo_dir=${REPO_DIR}"
  echo "run_root=${RUN_ROOT}"
  echo "num_shards=${NUM_SHARDS}"
  "${PYTHON}" --version
  git rev-parse HEAD || true
  git status --short --branch || true
  sha256sum experiments/sbs_override_fresh_confirmatory_corpus_v1/fresh_state_manifest.full_support_only.csv
  sha256sum experiments/sbs_override_fresh_confirmatory_corpus_v1/fresh_state_policy_action_map.full_support_only.csv
  sha256sum experiments/sbs_override_fresh_confirmatory_corpus_v1/fresh_candidate_scenario_manifest.csv
} > "${RUN_ROOT}/manifests/provenance.${SLURM_ARRAY_JOB_ID:-manual}.${SLURM_ARRAY_TASK_ID:-0}.txt"

"${PYTHON}" scripts/sbs_override_fresh_id_confirmatory_terminal_label_v1.py \
  --output-dir "$RUN_ROOT" \
  run-shard \
  --shard-index "${SLURM_ARRAY_TASK_ID:-0}" \
  --num-shards "$NUM_SHARDS"
