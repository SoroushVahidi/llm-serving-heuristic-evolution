#!/bin/bash
#SBATCH --job-name=sbs-ood-support-v1
#SBATCH --account=ikoutis
#SBATCH --partition=general
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --array=0-79
#SBATCH --output=experiments/sbs_override_fresh_ood_support_expansion_v1/logs/support_%A_%a.out
#SBATCH --error=experiments/sbs_override_fresh_ood_support_expansion_v1/logs/support_%A_%a.err

set -euo pipefail

ROOT="${SBS_FRESH_REPO_ROOT:-/mmfs1/project/ikoutis/sv96/llm-serving-heuristic-evolution-joint240-sbs-targeted-terminal-label-full-v1-src}"
PYTHON="${SBS_FRESH_PYTHON:-/mmfs1/project/ikoutis/sv96/envs/llm-serving-module-intervention-benchmark-actionable-v1/bin/python}"
cd "$ROOT"

export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export SBS_FRESH_LSSP_WINDOWS_CACHE="${SBS_FRESH_LSSP_WINDOWS_CACHE:-/mmfs1/project/ikoutis/sv96/llmserveopt-data/staging/sbs_override_fresh_confirmatory_corpus_v1/pilot_v2_windows_full_cache.json}"

mkdir -p experiments/sbs_override_fresh_ood_support_expansion_v1/logs

"$PYTHON" scripts/sbs_override_fresh_ood_support_expansion_v1.py \
  scan-shard \
  --shard-index "${SLURM_ARRAY_TASK_ID}" \
  --num-shards "${SLURM_ARRAY_TASK_COUNT}"
