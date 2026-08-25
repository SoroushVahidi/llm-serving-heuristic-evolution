#!/bin/bash -l
#
# Sarathi-Serve source build — retry of a failed login-node build.
#
# ROOT CAUSE of the prior failure (see
# /mmfs1/scratch/ikoutis/sv96/pip_install_sarathi_build.log): the login node
# has no visible CUDA device. sarathi-serve's setup.py selects target GPU
# architectures via `torch.cuda.device_count()` / `torch.cuda.get_device_
# capability()`; when no GPU is visible it falls back to building for ALL
# supported archs (`{70, 75, 80, 86, 89, 90}` -> 6 `-gencode` targets per
# extension across 4 CUDA extensions). That 6x-wider multi-arch compile is
# what triggered "c++: fatal error: Killed signal terminated program cc1plus"
# on the memory-constrained, cgroup-limited login node -- an OOM kill, not a
# code or dependency problem.
#
# FIX: run the build on a node with an A100 actually visible to torch. With
# device_count()>0, setup.py restricts codegen to just the detected
# capability (80 for A100-SXM4-80GB), a ~6x smaller compile than the
# login-node fallback. This is why this build job requests a GPU even though
# nvcc itself does not need one to execute -- it needs one to be *visible* at
# setup.py's import-time GPU detection. This is the only change from the
# failed attempt; no dependency/version redesign.
#
# Additional conservative memory bound: MAX_JOBS=2 caps the outer ninja
# process parallelism (torch's cpp_extension honors this env var) so at most
# 2 compile units run at once regardless of --cpus-per-task, on top of the
# now much-smaller single-arch per-unit compile.
#
# Pure-Python dependencies (254 packages, including torch==2.9.0+cu128
# matching this build's CUDA/12.8.0 toolchain) were already installed
# successfully in this env in a prior session; this job only (re)attempts
# the editable/source build of the `sarathi` C++/CUDA extensions via
# `pip install -e . --no-build-isolation` (--no-build-isolation reused
# unchanged from the failed attempt: it avoids pip creating a second
# isolated build venv that would duplicate the already-installed torch
# build-system dependency).

#SBATCH --job-name=sarathi-build
#SBATCH --partition=gpu
#SBATCH --account=ikoutis
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:45:00
#SBATCH --mem=64G
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err

set -euo pipefail

module purge

REPO_DIR="/mmfs1/project/ikoutis/sv96/github/llm-serving-heuristic-evolution"
SARATHI_SRC_DIR="/mmfs1/project/ikoutis/sv96/vendor/sarathi-serve"
SARATHI_ENV_PREFIX="/mmfs1/project/ikoutis/sv96/conda_env/sarathi-serve"
SCRATCH_ROOT="/mmfs1/scratch/ikoutis/sv96"
BUILD_LOG="${SCRATCH_ROOT}/sarathi_build_${SLURM_JOB_ID}.log"
MANIFEST="${SCRATCH_ROOT}/sarathi_build_${SLURM_JOB_ID}_manifest.txt"

source /apps/easybuild/software/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate "${SARATHI_ENV_PREFIX}"
unset PYTHONPATH
export PYTHONNOUSERSITE=1

# Matches this env's installed torch==2.9.0+cu128 exactly (do not use the
# vLLM env's bundled cu13 toolchain here -- that pairing is specific to the
# separate vllm-kv-pressure env's cu13 torch build).
CUDA_TOOLCHAIN_DIR="/apps/easybuild/el9_5.x86_64/software/CUDA/12.8.0"
export CUDA_HOME="${CUDA_TOOLCHAIN_DIR}"
export PATH="${CUDA_TOOLCHAIN_DIR}/bin:${PATH}"

# Conservative, bounded outer build parallelism (see header comment).
export MAX_JOBS=2

mkdir -p "${REPO_DIR}/logs"

{
  echo "job_id=${SLURM_JOB_ID}"
  echo "host=$(hostname)"
  echo "sarathi_src_dir=${SARATHI_SRC_DIR}"
  echo "sarathi_src_commit=$(git -C "${SARATHI_SRC_DIR}" rev-parse HEAD)"
  echo "sarathi_env_prefix=${SARATHI_ENV_PREFIX}"
  echo "cuda_toolchain=${CUDA_TOOLCHAIN_DIR}"
  "${CUDA_TOOLCHAIN_DIR}/bin/nvcc" --version | tail -1
  python -c "import torch; print('torch=' + torch.__version__, 'torch_cuda=' + str(torch.version.cuda))"
  python -c "import torch; print('cuda_visible=' + str(torch.cuda.is_available()), 'device_count=' + str(torch.cuda.device_count()))"
  python -c "import torch; maj,minr=torch.cuda.get_device_capability(0); print('detected_compute_capability=' + str(maj*10+minr))" || echo "detected_compute_capability=UNAVAILABLE"
  echo "max_jobs=${MAX_JOBS}"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
} | tee "${MANIFEST}"

cd "${SARATHI_SRC_DIR}"

set +e
pip install -e . --no-build-isolation 2>&1 | tee "${BUILD_LOG}"
BUILD_STATUS=${PIPESTATUS[0]}
set -e

echo "build_status=${BUILD_STATUS}" | tee -a "${MANIFEST}"

if [ "${BUILD_STATUS}" -eq 0 ]; then
  python -c "import sarathi; print('sarathi_import_ok version=' + getattr(sarathi, '__version__', 'unknown'))" | tee -a "${MANIFEST}"
  python -c "import sarathi.pos_encoding_ops, sarathi.layernorm_ops, sarathi.activation_ops, sarathi.moe_ops; print('sarathi_cuda_extensions_import_ok')" | tee -a "${MANIFEST}"
fi

echo "build_log=${BUILD_LOG}"
echo "manifest=${MANIFEST}"
exit "${BUILD_STATUS}"
