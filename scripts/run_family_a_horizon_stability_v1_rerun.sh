#!/usr/bin/env bash
# run_family_a_horizon_stability_v1_rerun.sh
#
# Corrected horizon-stability validation rerun.
# Uses the fixed horizon_stability_v1.py that imports case_fairness_vs_size_v2
# from the canonical templates_fairness_starvation_v2 module.
#
# Reuses the exact existing 128-row sample manifest from the old failed run.
# Does NOT resample.  Does NOT regenerate D0.  Does NOT relabel D0.
#
# This script is designed to run inside a detached tmux session and survive
# SSH disconnect, agent exit, terminal closure.

set -euo pipefail

cd "$(dirname "$0")/.."

REPO_ROOT="$(pwd)"
D0_OUTPUT_DIR="${REPO_ROOT}/datasets/family_a_oracle_policy_v1"
D0_MERGED="${D0_OUTPUT_DIR}/oracle_rows.csv"
OVERNIGHT_LOG="${REPO_ROOT}/logs/family_a_horizon_stability_v1_rerun.log"
EXPERIMENT_DIR="${REPO_ROOT}/experiments/family_a_horizon_stability_v1_rerun"
OLD_SAMPLE_MANIFEST="${REPO_ROOT}/experiments/family_a_horizon_stability_v1/sample_manifest.csv"
PYTHON="$(which python3)"

exec > >(tee -a "${OVERNIGHT_LOG}") 2>&1

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" ; }

# ======================================================================
# PHASE 0 — VERIFY D0 IS COMPLETE
# ======================================================================
log "=== PHASE 0: VERIFY D0 IS COMPLETE ==="

D0_STATUS="${D0_OUTPUT_DIR}/run_status.json"
if [ ! -f "${D0_STATUS}" ]; then
    log "FATAL: D0 run_status.json missing"
    exit 1
fi

STATUS=$(python -c "import json; print(json.load(open('${D0_STATUS}')).get('status','unknown'))" 2>/dev/null || echo "error")
log "D0 status=${STATUS}"

if [ "${STATUS}" != "complete" ]; then
    log "FATAL: D0 is not complete (status=${STATUS})"
    exit 1
fi

if [ ! -f "${D0_MERGED}" ]; then
    log "FATAL: D0 merged oracle_rows.csv missing"
    exit 1
fi

NROWS=$(python -c "import pandas as pd; print(len(pd.read_csv('${D0_MERGED}')))" 2>/dev/null || echo "0")
log "D0 merged rows: ${NROWS}"

# ======================================================================
# PHASE 1 — VERIFY OLD SAMPLE MANIFEST EXISTS
# ======================================================================
log "=== PHASE 1: VERIFY OLD SAMPLE MANIFEST ==="

if [ ! -f "${OLD_SAMPLE_MANIFEST}" ]; then
    log "FATAL: Old sample manifest missing: ${OLD_SAMPLE_MANIFEST}"
    exit 1
fi

MANIFEST_ROWS=$(python -c "import pandas as pd; print(len(pd.read_csv('${OLD_SAMPLE_MANIFEST}')))" 2>/dev/null || echo "0")
log "Old sample manifest rows: ${MANIFEST_ROWS}"

if [ "${MANIFEST_ROWS}" -lt 100 ]; then
    log "FATAL: Old sample manifest has too few rows (${MANIFEST_ROWS})"
    exit 1
fi

# ======================================================================
# PHASE 1.5 — PRESERVE OLD FAILED EVIDENCE
# ======================================================================
log "=== PHASE 1.5: PRESERVE OLD FAILED EVIDENCE ==="

OLD_DIR="${REPO_ROOT}/experiments/family_a_horizon_stability_v1"
if [ -d "${OLD_DIR}" ]; then
    log "Old failed run directory preserved: ${OLD_DIR}"
else
    log "WARNING: Old failed run directory not found: ${OLD_DIR}"
fi

# ======================================================================
# PHASE 2 — LAUNCH CORRECTED HORIZON STABILITY VALIDATION
# ======================================================================
log "=== PHASE 2: LAUNCHING CORRECTED HORIZON STABILITY VALIDATION ==="

mkdir -p "${EXPERIMENT_DIR}"

log "Starting corrected horizon_stability_v1.py ..."
log "  D0 merged: ${D0_MERGED}"
log "  Output dir: ${EXPERIMENT_DIR}"
log "  Sample manifest (reused): ${OLD_SAMPLE_MANIFEST}"
log "  Workers: 4"

"${PYTHON}" "${REPO_ROOT}/scripts/horizon_stability_v1.py" \
    --d0-merged "${D0_MERGED}" \
    --output-dir "${EXPERIMENT_DIR}" \
    --max-workers 4 \
    --h1500 \
    --h3000 \
    --hnatural \
    --sample-size 128 \
    --safety-cap 10000 \
    --sample-manifest "${OLD_SAMPLE_MANIFEST}" \
    2>&1 | tee -a "${OVERNIGHT_LOG}"

EXIT_CODE=${PIPESTATUS[0]}

log "Horizon stability validation exited with code ${EXIT_CODE}"

if [ ${EXIT_CODE} -eq 0 ]; then
    log "=== HORIZON STABILITY VALIDATION RERUN COMPLETE ==="
else
    log "=== HORIZON STABILITY VALIDATION RERUN FAILED (code ${EXIT_CODE}) ==="
fi

log "=== RERUN WORKFLOW COMPLETE ==="
exit ${EXIT_CODE}
