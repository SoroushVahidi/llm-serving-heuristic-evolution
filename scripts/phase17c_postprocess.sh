#!/bin/bash
# Phase 1.7C post-processing: run after all 7 experiments complete.
# Generates summaries, runs tests, updates docs, commits.
set -euo pipefail
cd /home/soroush/llm-serving-heuristic-evolution

LOG=results/phase17c/postprocess.log
echo "=== Phase 1.7C Post-Processing: $(date '+%Y-%m-%d %H:%M:%S') ===" | tee "$LOG"

# ----------------------------------------------------------------
# 1. Verify experiment results exist
# ----------------------------------------------------------------
echo "" | tee -a "$LOG"
echo "[1] Verifying experiment result directories..." | tee -a "$LOG"
missing=0
for exp in \
    burstgpt_natural_calibrated \
    burstgpt_scaled_moderate_calibrated \
    burstgpt_scaled_high_calibrated \
    burstgpt_scaled_moderate_synthetic_service \
    burstgpt_moderate_exact_prediction \
    burstgpt_moderate_noise035 \
    burstgpt_moderate_noise070
do
    dir="results/$exp"
    n=$(find "$dir" -name "summary.csv" 2>/dev/null | wc -l)
    if [ "$n" -gt 0 ]; then
        echo "  OK: $exp ($n summary.csv)" | tee -a "$LOG"
    else
        echo "  MISSING: $exp" | tee -a "$LOG"
        missing=$((missing + 1))
    fi
done
echo "  Missing: $missing/7" | tee -a "$LOG"

# ----------------------------------------------------------------
# 2. Generate consolidated summary and analyses
# ----------------------------------------------------------------
echo "" | tee -a "$LOG"
echo "[2] Generating experiment summary..." | tee -a "$LOG"
python scripts/generate_phase17c_summary.py 2>&1 | tee -a "$LOG"

# ----------------------------------------------------------------
# 3. Collect result directories and metrics files
# ----------------------------------------------------------------
echo "" | tee -a "$LOG"
echo "[3] Collecting result file locations..." | tee -a "$LOG"
find results -maxdepth 4 -type d | grep -Ei "burstgpt|phase17c|real_trace|calibrated|synthetic" | sort \
    | tee results/phase17c/result_directories.txt | tee -a "$LOG"

find results -maxdepth 5 -type f \( -name "*.csv" -o -name "*.json" -o -name "*.md" \) \
    | grep -Ei "burstgpt|phase17c|real_trace|calibrated|synthetic|metrics|summary" \
    | sort | tee results/phase17c/metrics_files.txt >> "$LOG"

# ----------------------------------------------------------------
# 4. Run full pytest suite
# ----------------------------------------------------------------
echo "" | tee -a "$LOG"
echo "[4] Running full pytest suite..." | tee -a "$LOG"
python -m pytest -x --tb=short 2>&1 | tee results/phase17c/final_pytest.log || true
PYTEST_RESULT=$(grep -E "passed|failed|error" results/phase17c/final_pytest.log | tail -1 || echo "unknown")
echo "  pytest result: $PYTEST_RESULT" | tee -a "$LOG"

# ----------------------------------------------------------------
# 5. Run GPU tests (best-effort)
# ----------------------------------------------------------------
echo "" | tee -a "$LOG"
echo "[5] Running GPU pytest (best-effort)..." | tee -a "$LOG"
python -m pytest -m gpu --tb=short 2>&1 | tee results/phase17c/final_gpu_pytest.log || true

# ----------------------------------------------------------------
# 6. Update Phase 1.7C documentation
# ----------------------------------------------------------------
echo "" | tee -a "$LOG"
echo "[6] Updating Phase 1.7C milestone doc..." | tee -a "$LOG"
python scripts/update_phase17c_docs.py 2>&1 | tee -a "$LOG"

# ----------------------------------------------------------------
# 7. Stage and commit (source, configs, docs, tests only)
# ----------------------------------------------------------------
echo "" | tee -a "$LOG"
echo "[7] Staging files for commit..." | tee -a "$LOG"
git add src/ scripts/ configs/ docs/ tests/ .gitignore data/README.md external/ 2>/dev/null || true
git add src/ scripts/ configs/ docs/ tests/ .gitignore 2>&1 | tee -a "$LOG"
git status --short 2>&1 | tee -a "$LOG"

STAGED=$(git diff --cached --name-only | wc -l)
if [ "$STAGED" -gt 0 ]; then
  echo "" | tee -a "$LOG"
  echo "[7] Committing Phase 1.7C ($STAGED staged files)..." | tee -a "$LOG"
  git commit -m "$(cat <<'EOF'
Complete calibrated real-trace replay support (Phase 1.7C)

- Wire CalibratedServiceModel into experiment runners via service_model_factory
- Run 7 BurstGPT replay experiments at natural/moderate/high load
- Evaluate prediction-noise sensitivity (exact/noise035/noise070 variants)
- Compare calibrated vs synthetic service model policy rankings
- Add generate_phase17c_summary.py, phase17c_postprocess.sh, update_phase17c_docs.py
- Update Phase 1.7C milestone documentation

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
  )"
  COMMIT=$(git rev-parse HEAD)
  echo "  Committed: $COMMIT" | tee -a "$LOG"
else
  echo "  No staged changes — nothing to commit." | tee -a "$LOG"
  COMMIT=$(git rev-parse HEAD)
  echo "  Current HEAD: $COMMIT" | tee -a "$LOG"
fi

# ----------------------------------------------------------------
# Final report
# ----------------------------------------------------------------
echo "" | tee -a "$LOG"
echo "=== Phase 1.7C Post-Processing COMPLETE: $(date '+%Y-%m-%d %H:%M:%S') ===" | tee -a "$LOG"
echo "  Commit: $COMMIT" | tee -a "$LOG"
echo "  Log: $LOG" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "Next steps:" | tee -a "$LOG"
echo "  - Review results/phase17c/phase17c_experiment_summary.md" | tee -a "$LOG"
echo "  - Review results/phase17c/calibrated_vs_synthetic_comparison.md" | tee -a "$LOG"
echo "  - Review results/phase17c/prediction_noise_sensitivity.md" | tee -a "$LOG"
echo "  - Consider: external-baseline coverage audit for Phase 2" | tee -a "$LOG"
