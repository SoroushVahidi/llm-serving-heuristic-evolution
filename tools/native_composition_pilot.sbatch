#!/bin/bash
#SBATCH --job-name=native_composition_pilot
#SBATCH --partition=general
#SBATCH --qos=standard
#SBATCH --account=ikoutis
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:30:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

REPO=/mmfs1/project/ikoutis/sv96/github/llm-serving-heuristic-evolution
PYTHON=/home/sv96/.conda/envs/repo-env/bin/python

if [[ -z "${RUN_ROOT:-}" ]]; then
  echo "RUN_ROOT must be set" >&2
  exit 2
fi

mkdir -p "$RUN_ROOT/logs"
cd "$REPO"
export PYTHONPATH=src

"$PYTHON" tools/native_composition_pilot.py \
  --run-root "$RUN_ROOT" \
  --dev-count "${DEV_COUNT:-35}" \
  --eval-count "${EVAL_COUNT:-65}" \
  --workers "${WORKERS:-4}"
