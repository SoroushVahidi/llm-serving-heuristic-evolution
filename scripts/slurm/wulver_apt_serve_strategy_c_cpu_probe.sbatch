#!/bin/bash -l
#
# Apt-Serve Strategy C feasibility probe -- CPU-ONLY allocation, requested
# first per this task's explicit preference ("prefer a CPU-only allocation
# first if the dependency environment can be constructed without GPU").
#
# Confirmed BEFORE this job was written (docs/audits/
# apt_serve_official_artifact_audit_20260805.md section 9's local pip
# probe, re-checked via PyPI file listing): vllm==0.5.0.post1 has
# PREBUILT wheels for cp38/cp39/cp310/cp311 -- NOT cp312, which is why
# the earlier LOCAL probe (Python 3.12) triggered a from-source build.
# This job therefore creates a Python 3.11 env specifically, where
# `pip install vllm==0.5.0.post1` is expected to resolve a prebuilt
# wheel with no build step at all -- a materially different, and
# untested until now, situation from the local workstation's.
#
# Does NOT build the custom mixed_cache_ops CUDA kernel (not needed for
# scheduler-only import/construction testing -- that kernel is only
# invoked from the worker/cache-engine layer, not the scheduler class).
# Does NOT apply the FULL 13-file insert_designs.sh replacement --
# applies only the 5 files the scheduler itself depends on
# (aptserve_scheduler.py, aptserve_block_manager.py,
# aptserve_interfaces.py, aptserve_sequence.py, aptserve_block.py), to
# keep the import-chain test isolated from the worker/model-executor/
# attention layers' own (likely harder) CUDA dependencies. This is a
# deliberate, disclosed deviation from the official install procedure --
# see docs/audits/apt_serve_strategy_c_wulver_probe_20260806.md's
# "minimum official source subset" discussion.
#
# License note: eddiegaoo/Apt-Serve has NO LICENSE file (confirmed,
# docs/audits/apt_serve_official_artifact_audit_20260805.md section 1).
# This job clones it to scratch/vendor space OUTSIDE this project's git
# tree, never commits any of its source, and this script contains no
# Apt-Serve source of its own -- only generic file-copy/import-test
# commands referencing paths.

#SBATCH --job-name=apt-serve-strategy-c-cpu-probe
#SBATCH --partition=general
#SBATCH --account=ikoutis
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err

set -uo pipefail

REPO_DIR="/mmfs1/project/ikoutis/sv96/github/llm-serving-heuristic-evolution"
VENDOR_DIR="/mmfs1/project/ikoutis/sv96/vendor/apt-serve"
ENV_PREFIX="/mmfs1/project/ikoutis/sv96/conda_env/apt-serve-strategy-c-probe"
SCRATCH_ROOT="/mmfs1/scratch/ikoutis/sv96"
OUT_DIR="${SCRATCH_ROOT}/apt_serve_strategy_c_probe_${SLURM_JOB_ID}"
PINNED_COMMIT="c953217988274a761da35cf06c01033b18dadf68"

mkdir -p "${REPO_DIR}/logs" "${OUT_DIR}" "$(dirname "${VENDOR_DIR}")"

MANIFEST="${OUT_DIR}/manifest.txt"
{
  echo "job_id=${SLURM_JOB_ID}"
  echo "host=$(hostname)"
  echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "pinned_commit=${PINNED_COMMIT}"
  echo "vendor_dir=${VENDOR_DIR}"
  echo "env_prefix=${ENV_PREFIX}"
  echo "out_dir=${OUT_DIR}"
} | tee "${MANIFEST}"

module purge

# --- 1. Isolated Python 3.11 conda env (never touches the project's own env) ---
source /apps/easybuild/software/Anaconda3/2023.09-0/etc/profile.d/conda.sh
if [ ! -d "${ENV_PREFIX}" ]; then
  mamba create -y -p "${ENV_PREFIX}" python=3.11 2>&1 | tee "${OUT_DIR}/env_create.log"
fi
conda activate "${ENV_PREFIX}"
unset PYTHONPATH
export PYTHONNOUSERSITE=1

# --- 2. Official dependency install (expect PREBUILT wheels at cp311, see header) ---
{
  echo "=== pip install torch==2.3.0 ==="
  pip install "torch==2.3.0" 2>&1
  echo "=== pip install vllm==0.5.0.post1 ==="
  pip install "vllm==0.5.0.post1" 2>&1
  echo "=== pip install xformers==0.0.26.post1 vllm-flash-attn==2.5.9 ==="
  pip install "xformers==0.0.26.post1" "vllm-flash-attn==2.5.9" 2>&1
} | tee "${OUT_DIR}/pip_install.log"
INSTALL_STATUS=${PIPESTATUS[0]:-1}
echo "pip_install_status=${INSTALL_STATUS}" | tee -a "${MANIFEST}"

pip freeze > "${OUT_DIR}/pip_freeze.txt" 2>&1
python -c "import sys; print('python=' + sys.version)" | tee -a "${MANIFEST}"

# --- 3. Fetch the pinned Apt-Serve commit to VENDOR space (outside project tree) ---
if [ ! -d "${VENDOR_DIR}" ]; then
  git clone https://github.com/eddiegaoo/Apt-Serve.git "${VENDOR_DIR}" 2>&1 | tee "${OUT_DIR}/git_clone.log"
fi
git -C "${VENDOR_DIR}" fetch origin 2>&1 | tee -a "${OUT_DIR}/git_clone.log"
git -C "${VENDOR_DIR}" checkout "${PINNED_COMMIT}" 2>&1 | tee -a "${OUT_DIR}/git_clone.log"
ACTUAL_COMMIT=$(git -C "${VENDOR_DIR}" rev-parse HEAD)
echo "actual_checked_out_commit=${ACTUAL_COMMIT}" | tee -a "${MANIFEST}"
if [ "${ACTUAL_COMMIT}" != "${PINNED_COMMIT}" ]; then
  echo "FATAL: checked-out commit does not match pin, aborting" | tee -a "${MANIFEST}"
  exit 1
fi

# --- 4. Phase 1: vanilla vLLM import probe (BEFORE any Apt-Serve file copy) ---
python "${REPO_DIR}/scripts/wulver_probes/apt_serve_import_probe.py" \
  --output "${OUT_DIR}/import_probe_vanilla.json" --phase vanilla \
  2>&1 | tee "${OUT_DIR}/import_probe_vanilla.log"

# --- 5. Apply ONLY the scheduler-critical minimal file subset (see header) ---
VLLM_DIR=$(python3 -c "import importlib.util; spec = importlib.util.find_spec('vllm'); print(spec.origin.rsplit('/', 1)[0])")
echo "vllm_dir=${VLLM_DIR}" | tee -a "${MANIFEST}"

cp -v "${VENDOR_DIR}/additional_designs/aptserve_block.py" "${VLLM_DIR}/block.py" 2>&1 | tee -a "${OUT_DIR}/file_copy.log"
cp -v "${VENDOR_DIR}/additional_designs/aptserve_sequence.py" "${VLLM_DIR}/sequence.py" 2>&1 | tee -a "${OUT_DIR}/file_copy.log"
cp -v "${VENDOR_DIR}/additional_designs/core/aptserve_block_manager.py" "${VLLM_DIR}/core/block_manager_v1.py" 2>&1 | tee -a "${OUT_DIR}/file_copy.log"
cp -v "${VENDOR_DIR}/additional_designs/core/aptserve_interfaces.py" "${VLLM_DIR}/core/interfaces.py" 2>&1 | tee -a "${OUT_DIR}/file_copy.log"
cp -v "${VENDOR_DIR}/additional_designs/core/aptserve_scheduler.py" "${VLLM_DIR}/core/scheduler.py" 2>&1 | tee -a "${OUT_DIR}/file_copy.log"

# Hashes of what was actually copied, for provenance (matches the hashes
# recorded in docs/audits/apt_serve_strategy_c_wulver_probe_20260806.md).
sha256sum "${VLLM_DIR}/block.py" "${VLLM_DIR}/sequence.py" \
  "${VLLM_DIR}/core/block_manager_v1.py" "${VLLM_DIR}/core/interfaces.py" \
  "${VLLM_DIR}/core/scheduler.py" | tee "${OUT_DIR}/copied_file_hashes.txt"

# --- 6. Phase 2: patched import + synthetic-construction probe ---
python "${REPO_DIR}/scripts/wulver_probes/apt_serve_import_probe.py" \
  --output "${OUT_DIR}/import_probe_patched.json" --phase patched \
  2>&1 | tee "${OUT_DIR}/import_probe_patched.log"

# --- 7. Micro-trace differential, only if patched construction succeeded ---
if python3 -c "
import json
d = json.load(open('${OUT_DIR}/import_probe_patched.json'))
ok = any(r['label'] == 'patched_scheduler_construct_synthetic_config' and r['status'] == 'OK' for r in d['results'])
raise SystemExit(0 if ok else 1)
"; then
  echo "patched scheduler construction succeeded -- running micro-trace" | tee -a "${MANIFEST}"
  python "${REPO_DIR}/scripts/wulver_probes/apt_serve_micro_trace.py" \
    --output "${OUT_DIR}/micro_trace.json" --commit "${PINNED_COMMIT}" \
    2>&1 | tee "${OUT_DIR}/micro_trace.log"
else
  echo "patched scheduler construction FAILED -- skipping micro-trace (nothing to trace against)" | tee -a "${MANIFEST}"
fi

echo "out_dir=${OUT_DIR}" | tee -a "${MANIFEST}"
echo "APT_SERVE_STRATEGY_C_CPU_PROBE_DONE"
