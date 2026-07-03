#!/usr/bin/env bash
set -euo pipefail
cd /home/soroush/llm-serving-heuristic-evolution
EXP_DIR="experiments/real_llm/cohere_v2_length_targeted_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$EXP_DIR"
python scripts/run_cohere_api_calibration.py \
  --allow-live-api \
  --stream \
  --model command-r7b-12-2024 \
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
echo "$EXP_DIR" > /tmp/claude-1000/-home-soroush-llm-serving-heuristic-evolution/fbee337d-f675-45ab-aad0-fbed1cba30c2/scratchpad/cohere_v2_exp_dir.txt
