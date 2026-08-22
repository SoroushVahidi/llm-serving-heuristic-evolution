#!/usr/bin/env bash
# run_family_a_horizon_stability_v1_overnight.sh
#
# Overnight detached workflow:
#   1. Wait for D0 dataset generation to finish successfully
#   2. Verify D0 completion / integrity
#   3. Run whole-branch oracle horizon-stability validation (H1500 vs H3000 vs HNATURAL)
#   4. Write all results / reports
#
# This script is designed to run inside a detached tmux session and survive
# SSH disconnect, agent exit, terminal closure.

set -euo pipefail

cd "$(dirname "$0")/.."

REPO_ROOT="$(pwd)"
D0_OUTPUT_DIR="${REPO_ROOT}/datasets/family_a_oracle_policy_v1"
D0_MERGED="${D0_OUTPUT_DIR}/oracle_rows.csv"
D0_LOG="${REPO_ROOT}/logs/family_a_oracle_dataset_v1_1k.log"
D0_STATUS="${D0_OUTPUT_DIR}/run_status.json"
OVERNIGHT_LOG="${REPO_ROOT}/logs/family_a_horizon_stability_v1_overnight.log"
EXPERIMENT_DIR="${REPO_ROOT}/experiments/family_a_horizon_stability_v1"
PYTHON="$(which python3)"

exec > >(tee -a "${OVERNIGHT_LOG}") 2>&1

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" ; }

# ======================================================================
# PHASE 0 — WAIT FOR D0
# ======================================================================
log "=== PHASE 0: WAITING FOR D0 DATASET GENERATION ==="

TIMEOUT=43200   # 12 hours max wait
ELAPSED=0
INTERVAL=60

while [ ${ELAPSED} -lt ${TIMEOUT} ]; do
    # Check if run_status.json exists and is complete
    if [ -f "${D0_STATUS}" ]; then
        STATUS=$(python -c "import json; print(json.load(open('${D0_STATUS}')).get('status','unknown'))" 2>/dev/null || echo "error")
        log "D0 status=${STATUS} elapsed=${ELAPSED}s"
        if [ "${STATUS}" = "complete" ]; then
            log "D0 reported complete in run_status.json"
            break
        elif [ "${STATUS}" = "failed" ]; then
            log "HORIZON_VALIDATION_BLOCKED_BY_D0_FAILURE"
            echo "BLOCKED: D0 failed" | tee -a "${OVERNIGHT_LOG}"
            exit 1
        fi
    fi

    # Also check if tmux session no longer exists — combined with artifact check
    if ! tmux has-session -t family_a_oracle_dataset_v1_1k 2>/dev/null; then
        # Session gone — check artifacts
        if [ -f "${D0_MERGED}" ]; then
            STATUS=$(python -c "import json; print(json.load(open('${D0_STATUS}')).get('status','unknown'))" 2>/dev/null || echo "unknown")
            if [ "${STATUS}" = "complete" ]; then
                log "D0 session exited, artifacts present and status=complete"
                break
            else
                log "D0 session exited but status=${STATUS} — checking merge completeness"
                # Even if status not recorded, check shard completions
                ALL_SHARDS_OK=true
                for i in 0 1 2 3; do
                    if [ ! -f "${D0_OUTPUT_DIR}/shards/shard_${i:03d}.done.json" ]; then
                        ALL_SHARDS_OK=false
                    fi
                done
                if ${ALL_SHARDS_OK}; then
                    log "All 4 shards have done markers — treating as complete"
                    break
                else
                    log "Some shards incomplete, will wait for run_status.json"
                fi
            fi
        else
            log "D0 session exited but no oracle_rows.csv yet — still waiting"
        fi
    fi

    sleep ${INTERVAL}
    ELAPSED=$((ELAPSED + INTERVAL))

    # Every 6 hours log a progress line
    if [ $((ELAPSED % 21600)) -eq 0 ]; then
        log "Still waiting for D0 after ${ELAPSED}s ($(echo "scale=1; ${ELAPSED}/3600" | bc) hours)"
    fi
done

if [ ${ELAPSED} -ge ${TIMEOUT} ]; then
    log "HORIZON_VALIDATION_BLOCKED_BY_D0_TIMEOUT after ${TIMEOUT}s"
    exit 1
fi

# ======================================================================
# PHASE 1 — D0 INTEGRITY VERIFICATION
# ======================================================================
log "=== PHASE 1: D0 INTEGRITY VERIFICATION ==="

# 1.1 Verify shard done markers
log "Checking shard completion markers..."
for i in 0 1 2 3; do
    DONE="${D0_OUTPUT_DIR}/shards/shard_${i:03d}.done.json"
    ROWS="${D0_OUTPUT_DIR}/shards/shard_${i:03d}.rows.csv"
    if [ ! -f "${DONE}" ]; then
        log "HORIZON_VALIDATION_BLOCKED_BY_D0_FAILURE: missing shard_${i:03d}.done.json"
        exit 1
    fi
    if [ ! -f "${ROWS}" ]; then
        log "HORIZON_VALIDATION_BLOCKED_BY_D0_FAILURE: missing shard_${i:03d}.rows.csv"
        exit 1
    fi
    NROWS=$(python -c "import json; d=json.load(open('${DONE}')); print(d.get('n_rows',0))" 2>/dev/null || echo "0")
    log "  shard_${i:03d}: ${NROWS} rows, sha256=$(python -c "import json; print(json.load(open('${DONE}')).get('rows_sha256','?'))" 2>/dev/null || echo '?')"
done

# 1.2 Verify final merge
if [ ! -f "${D0_MERGED}" ]; then
    log "HORIZON_VALIDATION_BLOCKED_BY_D0_FAILURE: merged oracle_rows.csv missing"
    exit 1
fi

# 1.3 Run Python integrity checks
log "Running Python integrity checks..."
INTEGRITY_PASS=true

INTEGRITY_RESULT=$(python -c "
import json, sys, hashlib
import pandas as pd
import numpy as np

d0_dir = '${D0_OUTPUT_DIR}'
merged = '${D0_MERGED}'

df = pd.read_csv(merged)
errors = []

# Check row count
if len(df) == 0:
    errors.append('merged csv is empty')

# Check no duplicate sample_id
if df['sample_id'].duplicated().any():
    errors.append('duplicate sample_id found')

# Check no duplicate state_fingerprint
if df['state_fingerprint'].duplicated().any():
    errors.append('duplicate state_fingerprint found')

# Check no TEST leakage
if 'TEST' in df['split'].values:
    errors.append('TEST split leakage detected')

# Check label consistency
expected_estf = 'ESTF'
expected_wfs = 'WFS'
expected_tie = 'TIE_OR_UNCERTAIN'
expected = df['delta_J_whole'].apply(lambda x: expected_estf if x > 0 else (expected_wfs if x < 0 else expected_tie))
if not (df['oracle_label'] == expected).all():
    errors.append('label inconsistency: oracle_label != sign(delta_J_whole)')

# Check label distribution
label_counts = df['oracle_label'].value_counts().to_dict()
log_msg = json.dumps({
    'n_rows': int(len(df)),
    'n_scenarios': int(df['scenario_id'].nunique()),
    'n_config_groups': int(df['configuration_group_id'].nunique()),
    'n_splits': int(df['split'].nunique()),
    'label_counts': {k: int(v) for k, v in label_counts.items()},
    'contested_label_counts': {k: int(v) for k, v in df['oracle_label_contested'].value_counts().to_dict().items()},
})
print(log_msg)

if errors:
    for e in errors:
        print(f'ERROR: {e}', file=sys.stderr)
    sys.exit(1)
else:
    print('INTEGRITY_OK')
    sys.exit(0)
" 2>&1)

if [ $? -ne 0 ]; then
    log "D0 INTEGRITY CHECK FAILED:"
    echo "${INTEGRITY_RESULT}" | while read -r line; do log "  ${line}"; done
    INTEGRITY_PASS=false
else
    log "D0 INTEGRITY CHECK PASSED"
    echo "${INTEGRITY_RESULT}" | head -1 | while read -r line; do log "  ${line}"; done
fi

if ! ${INTEGRITY_PASS}; then
    log "HORIZON_VALIDATION_BLOCKED_BY_D0_FAILURE"
    exit 1
fi

# ======================================================================
# PHASE 1.5 — D0 SNAPSHOT
# ======================================================================
log "=== PHASE 1.5: D0 SNAPSHOT ==="

D0_STATS=$(python -c "
import pandas as pd, numpy as np, json
df = pd.read_csv('${D0_MERGED}')
delta = df['delta_J_whole'].abs()
stats = {
    'n_rows': int(len(df)),
    'n_scenarios': int(df['scenario_id'].nunique()),
    'n_config_groups': int(df['configuration_group_id'].nunique()),
    'label_counts': {k: int(v) for k, v in df['oracle_label'].value_counts().to_dict().items()},
    'delta_J_mean': float(df['delta_J_whole'].mean()),
    'delta_J_median': float(df['delta_J_whole'].median()),
    'delta_J_std': float(df['delta_J_whole'].std()),
    'delta_J_abs_mean': float(delta.mean()),
    'delta_J_abs_median': float(delta.median()),
    'delta_J_abs_max': float(delta.max()),
    'delta_J_exact_zero': int((df['delta_J_whole'] == 0).sum()),
    'delta_J_small_1e6': int((delta < 1e-6).sum()),
    'contested_agreement': int((df['oracle_label'] == df['oracle_label_contested']).sum()),
    'contested_total': int(len(df)),
}
print(json.dumps(stats, indent=2))
")
log "D0 snapshot:"
echo "${D0_STATS}" | while read -r line; do log "  ${line}"; done

# ======================================================================
# PHASE 2 — LAUNCH HORIZON STABILITY VALIDATION
# ======================================================================
log "=== PHASE 2: LAUNCHING HORIZON STABILITY VALIDATION ==="

# Launch the Python validation script
# It handles: sampling, reconstruction, horizon evaluation, parallelization,
# residual certificates, summary report.
log "Starting horizon_stability_v1.py ..."

mkdir -p "${EXPERIMENT_DIR}"

"${PYTHON}" "${REPO_ROOT}/scripts/horizon_stability_v1.py" \
    --d0-merged "${D0_MERGED}" \
    --output-dir "${EXPERIMENT_DIR}" \
    --max-workers 4 \
    --h1500 \
    --h3000 \
    --hnatural \
    --sample-size 128 \
    --safety-cap 10000 \
    2>&1 | tee -a "${OVERNIGHT_LOG}"

EXIT_CODE=${PIPESTATUS[0]}

log "Horizon stability validation exited with code ${EXIT_CODE}"

if [ ${EXIT_CODE} -eq 0 ]; then
    log "=== HORIZON STABILITY VALIDATION COMPLETE ==="
else
    log "=== HORIZON STABILITY VALIDATION FAILED (code ${EXIT_CODE}) ==="
fi

log "=== OVERNIGHT WORKFLOW COMPLETE ==="
exit ${EXIT_CODE}
