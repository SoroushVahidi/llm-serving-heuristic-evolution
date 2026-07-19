#!/bin/bash -l
#
# CPU-only regression test pass, run after the postprocessing job
# completes. No GPU requested. Intended to be submitted with
# --dependency=afterok:<postprocess job id>.
#
# Known pre-existing baseline (established across three prior sessions on
# this branch lineage, all with the exact same 80/1783/79 split): 80
# failed, 1783 passed, 79 skipped in the full non-GPU suite -- pre-existing
# environment gaps (missing optional SDKs, a Python-3.10-vs-3.12 f-string
# syntax error in an unrelated script, GPU-only tests unavailable in
# repo-env, a numpy.int64 JSON-serialization quirk in one unrelated test
# file). This job compares against that baseline rather than treating any
# failure as automatically new.

#SBATCH --job-name=sarathi-regression-tests
#SBATCH --partition=general
#SBATCH --account=ikoutis
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err

set -euo pipefail

REPO_DIR="/mmfs1/project/ikoutis/sv96/github/llm-serving-heuristic-evolution"
SCRATCH_ROOT="/mmfs1/scratch/ikoutis/sv96"
REPORT="${SCRATCH_ROOT}/sarathi_regression_tests_${SLURM_JOB_ID}_report.txt"
CANONICAL_DIR="${REPO_DIR}/experiments/gpu_external_validity/sarathi_mistral_fp16_final_1111723"

KNOWN_BASELINE_FAILED=80
KNOWN_BASELINE_PASSED=1783
KNOWN_BASELINE_SKIPPED=79

source /apps/easybuild/software/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate repo-env

mkdir -p "${REPO_DIR}/logs" "${CANONICAL_DIR}"
cd "${REPO_DIR}"

set +e
{
  echo "=== Regression test job ${SLURM_JOB_ID} ==="
  echo "timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "branch=$(git rev-parse --abbrev-ref HEAD)"
  echo "commit=$(git rev-parse HEAD)"
  echo "known_baseline=${KNOWN_BASELINE_FAILED} failed / ${KNOWN_BASELINE_PASSED} passed / ${KNOWN_BASELINE_SKIPPED} skipped"
  echo

  echo "--- 1. Targeted tests ---"
  python -m pytest \
    tests/test_gpu_external_validity_audit.py \
    tests/test_sarathi_faithful_scheduler.py \
    tests/test_vllm_faithful_scheduler.py \
    tests/test_sarathi_style_performance.py \
    tests/test_external_baseline_integration.py \
    tests/test_selector_objective_analysis.py \
    -q
  TARGETED_STATUS=$?
  echo "targeted_status=${TARGETED_STATUS}"
  echo

  echo "--- 2. Full non-GPU suite ---"
  python -m pytest tests/ -q
  FULL_STATUS=$?
  echo "full_status=${FULL_STATUS}"
  echo

  echo "--- 3. git diff --check ---"
  git diff --check
  DIFFCHECK_STATUS=$?
  echo "diffcheck_status=${DIFFCHECK_STATUS}"
} > "${REPORT}" 2>&1
set -e

cat "${REPORT}"

# Extract the full-suite summary line and compare counts against the known
# baseline rather than assuming any failure is new.
SUMMARY_LINE=$(grep -E "^=+ .*(passed|failed|error).*=+$" "${REPORT}" | tail -1 || true)
echo "full_suite_summary=${SUMMARY_LINE}"

FAILED_COUNT=$(echo "${SUMMARY_LINE}" | grep -oE "[0-9]+ failed" | grep -oE "[0-9]+" || echo 0)
PASSED_COUNT=$(echo "${SUMMARY_LINE}" | grep -oE "[0-9]+ passed" | grep -oE "[0-9]+" || echo 0)

{
  echo
  echo "=== Compact verdict ==="
  echo "full_suite: ${FAILED_COUNT} failed / ${PASSED_COUNT} passed (known baseline: ${KNOWN_BASELINE_FAILED} failed / ${KNOWN_BASELINE_PASSED} passed)"
  if [ "${FAILED_COUNT}" -le "${KNOWN_BASELINE_FAILED}" ]; then
    echo "verdict=NO_NEW_REGRESSIONS (failed count at or below known baseline)"
  else
    echo "verdict=POSSIBLE_NEW_REGRESSION (failed count exceeds known baseline by $((FAILED_COUNT - KNOWN_BASELINE_FAILED)) -- inspect ${REPORT} for which tests)"
  fi
} | tee -a "${REPORT}"

cp "${REPORT}" "${CANONICAL_DIR}/regression_test_report.txt"
echo "report=${REPORT}"
