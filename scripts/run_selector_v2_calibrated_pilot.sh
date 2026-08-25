#!/usr/bin/env bash
# tmux entrypoint for the Selector v2 calibrated targeted pilot (Option B
# scope, docs/selector_v2_faithful_baseline_scope_audit.md). Generates a
# 250-500 retained-window pilot over the 8 approved historical-monolithic
# policies, then trains the prototype selector ONLY if the pilot's own
# quality gates all pass -- this wrapper just pins the working directory,
# timestamps the output dir, and makes sure a crash doesn't leave the tmux
# pane silently dead with no explanation.
set -uo pipefail

cd "$(dirname "$0")/.."

TS="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="experiments/selector_v2_calibrated_pilot_${TS}"
mkdir -p "${OUT_DIR}/logs"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] launching calibrated targeted pilot, output_dir=${OUT_DIR}" \
  | tee -a logs/selector_v2_calibrated_pilot.log

python3 -u scripts/build_selector_dataset_v2_calibrated_targeted_pilot.py \
  --output-dir "${OUT_DIR}" \
  --target-min-retained 250 \
  --target-max-retained 500 \
  --min-real-trace-retained 60 \
  --min-ood-reserved-retained 20 \
  --multiplier 2.0 \
  --search-seed 20260720 \
  --max-attempts 3000 \
  --batch-size 25 \
  --drain-steps 5000 \
  >> "logs/selector_v2_calibrated_pilot.log" 2>&1
GEN_RC=$?

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] generation exited rc=${GEN_RC}" \
  | tee -a logs/selector_v2_calibrated_pilot.log

if [ ${GEN_RC} -ne 0 ]; then
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] generation failed -- skipping prototype training" \
    | tee -a logs/selector_v2_calibrated_pilot.log
  exit ${GEN_RC}
fi

python3 -u scripts/train_selector_v2_calibrated_prototype.py \
  --pilot-dir "${OUT_DIR}" \
  >> "logs/selector_v2_calibrated_pilot.log" 2>&1
TRAIN_RC=$?

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] prototype training step exited rc=${TRAIN_RC}, output_dir=${OUT_DIR}" \
  | tee -a logs/selector_v2_calibrated_pilot.log

exit ${TRAIN_RC}
