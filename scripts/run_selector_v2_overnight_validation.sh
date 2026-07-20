#!/usr/bin/env bash
# tmux entrypoint for the overnight Selector v2 contention-validation pilot.
# See scripts/run_selector_v2_overnight_validation.py for the actual
# orchestrator; this wrapper just pins the working directory, timestamps
# the checkpoint output dir, and makes sure a crash doesn't leave the
# tmux pane silently dead with no explanation.
set -uo pipefail

cd "$(dirname "$0")/.."

TS="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="experiments/selector_v2_overnight_${TS}"
mkdir -p "${OUT_DIR}/logs"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] launching overnight orchestrator, output_dir=${OUT_DIR}" \
  | tee -a logs/selector_v2_overnight_validation.log

python3 -u scripts/run_selector_v2_overnight_validation.py \
  --output-dir "${OUT_DIR}" \
  --deadline-seconds 29700 \
  --search-candidates 300 \
  --pilot-target-windows 300 \
  >> logs/selector_v2_overnight_validation.log 2>&1
RC=$?

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] orchestrator exited rc=${RC}" \
  | tee -a logs/selector_v2_overnight_validation.log

exit ${RC}
