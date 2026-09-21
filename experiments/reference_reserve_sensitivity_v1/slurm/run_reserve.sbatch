#!/bin/bash
#SBATCH --partition=general
#SBATCH --account=ikoutis
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/mmfs1/scratch/ikoutis/sv96/reference_reserve_sensitivity_v1/logs/reserve_%j.out
#SBATCH --error=/mmfs1/scratch/ikoutis/sv96/reference_reserve_sensitivity_v1/logs/reserve_%j.err

set -euo pipefail

REPO_DIR="/mmfs1/project/ikoutis/sv96/github/llm-serving-heuristic-evolution"
PYTHON="/mmfs1/project/ikoutis/sv96/envs/llm-serving-module-intervention-benchmark-actionable-v1/bin/python"

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

echo "=== STARTING JOB ==="
date
hostname
pwd
git rev-parse HEAD
git status --short
"${PYTHON}" --version
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "Selected Reserve: ${RESERVE}"
echo "Output Directory: /mmfs1/scratch/ikoutis/sv96/reference_reserve_sensitivity_v1/results/reserve_${RESERVE_STR}"
echo "Command: ${PYTHON} scripts/reference_reserve_sensitivity_v1.py --reserve ${RESERVE} --output-dir /mmfs1/scratch/ikoutis/sv96/reference_reserve_sensitivity_v1/results/reserve_${RESERVE_STR}"

"${PYTHON}" scripts/reference_reserve_sensitivity_v1.py --reserve "${RESERVE}" --output-dir "/mmfs1/scratch/ikoutis/sv96/reference_reserve_sensitivity_v1/results/reserve_${RESERVE_STR}"
