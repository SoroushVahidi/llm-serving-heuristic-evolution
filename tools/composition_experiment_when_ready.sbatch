#!/bin/bash
#SBATCH --job-name=composition_when_ready
#SBATCH --partition=general
#SBATCH --qos=standard
#SBATCH --account=ikoutis
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:30:00
#SBATCH --output=/mmfs1/project/ikoutis/sv96/llmserveopt-data/composition_experiment/logs/%x_%j.out
#SBATCH --error=/mmfs1/project/ikoutis/sv96/llmserveopt-data/composition_experiment/logs/%x_%j.err

set -euo pipefail

REPO=/mmfs1/project/ikoutis/sv96/github/llm-serving-heuristic-evolution
PYTHON=/home/sv96/.conda/envs/repo-env/bin/python
FRONTIER_ROOT=/mmfs1/project/ikoutis/sv96/llmserveopt-data/policy_frontier_cartography_20260721T154408Z
POLICY_LIBRARY_ROOT=/mmfs1/project/ikoutis/sv96/llmserveopt-data/policy_library_v2_expanded_20260721T171933Z
RESULT_ROOT=/mmfs1/project/ikoutis/sv96/llmserveopt-data/composition_experiment

mkdir -p "$RESULT_ROOT/logs" "$RESULT_ROOT/reports" "$RESULT_ROOT/manifests"
cd "$REPO"
export PYTHONPATH=src

if [[ ! -f "$FRONTIER_ROOT/reports/FINAL_REPORT.md" ]]; then
  echo "Frontier final report is not available yet: $FRONTIER_ROOT/reports/FINAL_REPORT.md"
  exit 2
fi

if [[ ! -f "$POLICY_LIBRARY_ROOT/reports/FINAL_POLICY_LIBRARY_REPORT.md" && ! -f "$POLICY_LIBRARY_ROOT/reports/FINAL_REPORT.md" ]]; then
  echo "Policy Library v2 final report is not available yet under $POLICY_LIBRARY_ROOT/reports"
  exit 2
fi

echo "Composition experiment readiness checks passed."
echo "This sbatch template is intentionally not the full expensive experiment runner yet."
echo "Next implementation should invoke the final composition workflow described in docs/current/COMPOSITION_EXPERIMENT_DESIGN.md."
