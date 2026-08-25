#!/bin/bash -l
#
# Joint-240 same-distribution adaptive exploitability v1 (CPU)
#
#SBATCH --job-name=joint240_same_distribution_adaptive_v1
#SBATCH --partition=general
#SBATCH --account=ikoutis
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/mmfs1/scratch/ikoutis/sv96/joint240_same_distribution_adaptive_v1_bundle/logs/%x.%j.out
#SBATCH --error=/mmfs1/scratch/ikoutis/sv96/joint240_same_distribution_adaptive_v1_bundle/logs/%x.%j.err

set -uo pipefail

BUNDLE="/mmfs1/scratch/ikoutis/sv96/joint240_same_distribution_adaptive_v1_bundle"
PYTHON="/home/sv96/.conda/envs/repo-env/bin/python"
OUT="${BUNDLE}/experiments/joint240_same_distribution_adaptive_exploitability_v1"
TS=$(date -u +%Y%m%dT%H%M%SZ)

mkdir -p "${BUNDLE}/logs" "${OUT}"

{
  echo "job_id=${SLURM_JOB_ID}"
  echo "job_name=${SLURM_JOB_NAME:-}"
  echo "host=$(hostname)"
  echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "bundle=${BUNDLE}"
  echo "out=${OUT}"
  echo "ts=${TS}"
} | tee "${OUT}/slurm_manifest.txt"

cd "${BUNDLE}"
export PYTHONPATH="${BUNDLE}/src"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

"${PYTHON}" --version 2>&1 | tee -a "${OUT}/slurm_manifest.txt"
echo "python_bin=${PYTHON}" | tee -a "${OUT}/slurm_manifest.txt"

# Focused tests first
"${PYTHON}" -m pytest tests/test_joint240_same_distribution_adaptive_v1.py -q --tb=line \
  2>&1 | tee "${OUT}/logs/pytest.log"
TEST_STATUS=${PIPESTATUS[0]}
echo "pytest_exit=${TEST_STATUS}" | tee -a "${OUT}/slurm_manifest.txt"
if [[ "${TEST_STATUS}" -ne 0 ]]; then
  echo "ABORT: unit tests failed" | tee -a "${OUT}/slurm_manifest.txt"
  exit "${TEST_STATUS}"
fi

# Full same-distribution adaptive run (scenario-level + live online router)
"${PYTHON}" scripts/run_joint240_same_distribution_adaptive_v1.py \
  --out-dir "${OUT}" \
  2>&1 | tee "${OUT}/logs/full_run.log"
STATUS=${PIPESTATUS[0]}

echo "wrapper_exit_code=${STATUS}" | tee -a "${OUT}/slurm_manifest.txt"
echo "finished_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${OUT}/slurm_manifest.txt"
if [[ -f "${OUT}/DONE" ]]; then
  echo "DONE_sentinel=present" | tee -a "${OUT}/slurm_manifest.txt"
else
  echo "DONE_sentinel=missing" | tee -a "${OUT}/slurm_manifest.txt"
fi
exit "${STATUS}"
