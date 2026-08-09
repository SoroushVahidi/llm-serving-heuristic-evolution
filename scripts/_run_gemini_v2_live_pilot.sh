#!/usr/bin/env bash
# HISTORICAL / MANUAL LIVE-API SCRIPT. MAY INCUR COST. REQUIRES EXPLICIT OPT-IN.
# This is a one-off launcher for the Gemini v2 length-targeted real-LLM pilot
# (see docs/real_llm_cohere_gemini_comparison.md, docs/real_llm_latency_model_v2.md).
# It fires real, paid Gemini API requests the moment it runs. It will refuse to
# run unless LLMSERVEOPT_ALLOW_PAID_API_CALLS=1 is set explicitly, e.g.:
#   LLMSERVEOPT_ALLOW_PAID_API_CALLS=1 bash scripts/_run_gemini_v2_live_pilot.sh
set -euo pipefail

if [[ "${LLMSERVEOPT_ALLOW_PAID_API_CALLS:-}" != "1" ]]; then
    echo "Refusing to run paid live API pilot: set LLMSERVEOPT_ALLOW_PAID_API_CALLS=1 to opt in." >&2
    echo "This script fires real, billed Gemini API requests (see --max-estimated-cost-usd below)." >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
EXP_DIR="experiments/real_llm/gemini_v2_length_targeted_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$EXP_DIR"
python scripts/run_gemini_real_llm_calibration.py \
  --allow-live-api \
  --stream \
  --model gemini-3.1-flash-lite \
  --workload-version v2 \
  --prompt-buckets short,medium,long \
  --target-output-tokens-list 64,128,256 \
  --concurrency-list 1,2,4,8 \
  --requests-per-cell 3 \
  --timeout-seconds 120 \
  --rpm-limit 20 \
  --max-total-requests 108 \
  --max-total-input-tokens 250000 \
  --max-total-output-tokens 50000 \
  --max-estimated-cost-usd 5 \
  --seed 20260703 \
  --fail-fast \
  --output-dir "$EXP_DIR" \
  2>&1 | tee "$EXP_DIR/run.log"
# Marker file co-located with the experiment output (repo-relative; the
# previous version wrote this to a since-deleted /tmp path from an old,
# unrelated agent session).
echo "$EXP_DIR" > "$EXP_DIR/exp_dir.txt"
