#!/bin/bash
#SBATCH --partition=general
#SBATCH --account=ikoutis
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --job-name=reserve_denominator
#SBATCH --output=/mmfs1/scratch/ikoutis/sv96/reference_reserve_sensitivity_v1/logs/denominator_%j.out
#SBATCH --error=/mmfs1/scratch/ikoutis/sv96/reference_reserve_sensitivity_v1/logs/denominator_%j.err

set -euo pipefail

FREEZE_SHA="ec1996354de0304dd6ea521c5d584a838401b06a"
REPO_DIR="/mmfs1/project/ikoutis/sv96/github/llm-serving-heuristic-evolution"
PYTHON="/mmfs1/project/ikoutis/sv96/envs/llm-serving-module-intervention-benchmark-actionable-v1/bin/python"
RESULTS="/mmfs1/scratch/ikoutis/sv96/reference_reserve_sensitivity_v1/results"
OUT="/mmfs1/scratch/ikoutis/sv96/reference_reserve_sensitivity_v1/denominator_completion_v1"

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

echo "=== STARTING DENOMINATOR-COMPLETION JOB ==="
date
hostname
pwd
git rev-parse HEAD
git status --short --untracked-files=no
"${PYTHON}" --version
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
if [ "$(git rev-parse HEAD)" != "${FREEZE_SHA}" ]; then
  echo "HEAD is not the frozen implementation commit ${FREEZE_SHA}; refusing to run" >&2
  exit 4
fi
echo "Command: ${PYTHON} scripts/reference_reserve_denominator_completion_v1.py count --existing-results-dir ${RESULTS} --output-dir ${OUT}"

"${PYTHON}" scripts/reference_reserve_denominator_completion_v1.py count \
  --existing-results-dir "${RESULTS}" --output-dir "${OUT}"
