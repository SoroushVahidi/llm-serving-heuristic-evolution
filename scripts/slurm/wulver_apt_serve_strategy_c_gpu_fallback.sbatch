#!/bin/bash -l
#
# Apt-Serve Strategy C feasibility probe -- SINGLE-GPU fallback.
#
# Do NOT submit this unless wulver_apt_serve_strategy_c_cpu_probe.sbatch's
# import_probe_vanilla.json or import_probe_patched.json shows a FAILED
# entry whose exception indicates CUDA initialization was required merely
# to IMPORT (not construct/schedule) vllm.core.scheduler or the
# Apt-Serve-patched modules -- e.g. a CUDA driver/runtime error raised
# from module-level code, not from an explicit .cuda() call this probe
# never makes. If the CPU probe's failure is anything else (missing
# package, wrong constructor signature, etc.), fix that and rerun the
# CPU probe instead of jumping to this GPU job -- per this task's own
# "prefer CPU-only first" instruction, this file exists ONLY as the
# documented fallback path, one A100, never more.
#
# Otherwise identical in structure to the CPU probe (same env-creation
# and file-copy steps) -- see that script's header for the license/
# minimal-file-subset/no-CUDA-kernel-build rationale, unchanged here.

#SBATCH --job-name=apt-serve-strategy-c-gpu-fallback
#SBATCH --partition=gpu
#SBATCH --account=ikoutis
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err

set -uo pipefail

REPO_DIR="/mmfs1/project/ikoutis/sv96/github/llm-serving-heuristic-evolution"
VENDOR_DIR="/mmfs1/project/ikoutis/sv96/vendor/apt-serve"
ENV_PREFIX="/mmfs1/project/ikoutis/sv96/conda_env/apt-serve-strategy-c-probe"
SCRATCH_ROOT="/mmfs1/scratch/ikoutis/sv96"
OUT_DIR="${SCRATCH_ROOT}/apt_serve_strategy_c_gpu_probe_${SLURM_JOB_ID}"
PINNED_COMMIT="c953217988274a761da35cf06c01033b18dadf68"

mkdir -p "${REPO_DIR}/logs" "${OUT_DIR}"

{
  echo "job_id=${SLURM_JOB_ID}"
  echo "host=$(hostname)"
  echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "pinned_commit=${PINNED_COMMIT}"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
} | tee "${OUT_DIR}/manifest.txt"

module purge
source /apps/easybuild/software/Anaconda3/2023.09-0/etc/profile.d/conda.sh

# Reuse the CPU probe's env if it already exists (same pins) -- only
# create it here if this job is somehow run without the CPU probe ever
# having run, which should not normally happen given the "CPU first"
# policy above.
if [ ! -d "${ENV_PREFIX}" ]; then
  mamba create -y -p "${ENV_PREFIX}" python=3.11 2>&1 | tee "${OUT_DIR}/env_create.log"
  conda activate "${ENV_PREFIX}"
  unset PYTHONPATH
  export PYTHONNOUSERSITE=1
  pip install "torch==2.3.0" "vllm==0.5.0.post1" "xformers==0.0.26.post1" "vllm-flash-attn==2.5.9" \
    2>&1 | tee "${OUT_DIR}/pip_install.log"
else
  conda activate "${ENV_PREFIX}"
  unset PYTHONPATH
  export PYTHONNOUSERSITE=1
fi

python -c "
import torch
print('cuda_available=', torch.cuda.is_available())
print('device_count=', torch.cuda.device_count())
"

if [ ! -d "${VENDOR_DIR}" ]; then
  git clone https://github.com/eddiegaoo/Apt-Serve.git "${VENDOR_DIR}"
fi
git -C "${VENDOR_DIR}" fetch origin
git -C "${VENDOR_DIR}" checkout "${PINNED_COMMIT}"

VLLM_DIR=$(python3 -c "import importlib.util; spec = importlib.util.find_spec('vllm'); print(spec.origin.rsplit('/', 1)[0])")
cp -v "${VENDOR_DIR}/additional_designs/aptserve_block.py" "${VLLM_DIR}/block.py"
cp -v "${VENDOR_DIR}/additional_designs/aptserve_sequence.py" "${VLLM_DIR}/sequence.py"
cp -v "${VENDOR_DIR}/additional_designs/core/aptserve_block_manager.py" "${VLLM_DIR}/core/block_manager_v1.py"
cp -v "${VENDOR_DIR}/additional_designs/core/aptserve_interfaces.py" "${VLLM_DIR}/core/interfaces.py"
cp -v "${VENDOR_DIR}/additional_designs/core/aptserve_scheduler.py" "${VLLM_DIR}/core/scheduler.py"

python "${REPO_DIR}/scripts/wulver_probes/apt_serve_import_probe.py" \
  --output "${OUT_DIR}/import_probe_gpu_patched.json" --phase patched \
  2>&1 | tee "${OUT_DIR}/import_probe_gpu_patched.log"

echo "out_dir=${OUT_DIR}"
echo "APT_SERVE_STRATEGY_C_GPU_FALLBACK_DONE"
