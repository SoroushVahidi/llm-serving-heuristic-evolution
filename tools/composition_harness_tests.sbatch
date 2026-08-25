#!/bin/bash
#SBATCH --job-name=composition_harness_tests
#SBATCH --partition=general
#SBATCH --qos=standard
#SBATCH --account=ikoutis
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/mmfs1/project/ikoutis/sv96/llmserveopt-data/composition_harness_readiness/logs/%x_%j.out
#SBATCH --error=/mmfs1/project/ikoutis/sv96/llmserveopt-data/composition_harness_readiness/logs/%x_%j.err

set -euo pipefail

REPO=/mmfs1/project/ikoutis/sv96/github/llm-serving-heuristic-evolution
PYTHON=/home/sv96/.conda/envs/repo-env/bin/python
RESULT_ROOT=/mmfs1/project/ikoutis/sv96/llmserveopt-data/composition_harness_readiness

mkdir -p "$RESULT_ROOT/logs" "$RESULT_ROOT/reports" "$RESULT_ROOT/manifests"
cd "$REPO"
export PYTHONPATH=src

git rev-parse HEAD > "$RESULT_ROOT/manifests/git_commit.txt"
git status --short --branch > "$RESULT_ROOT/manifests/git_status.txt"

"$PYTHON" -m pytest tests/test_policy_composition.py tests/test_policy_library_v2.py -q
"$PYTHON" tools/composition_smoke_experiment.py --output "$RESULT_ROOT/reports/composition_smoke_result.json"
