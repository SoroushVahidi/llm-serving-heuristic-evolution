#!/bin/bash -l
#
# Verified Wulver vLLM KV-pressure/preemption validation job — run 1.
#
# All Slurm values below were confirmed via authenticated read-only commands
# on Wulver on 2026-07-18 (sacctmgr, scontrol, sinfo, module, df, git ls-remote).
# See the audit report for the exact commands and raw output.
#
# Environment: conda env at $VLLM_ENV_PREFIX below, created 2026-07-18/19
# with `conda create -p ... python=3.11` then `pip install vllm==0.24.0`,
# which pinned torch==2.11.0+cu13 (matches the local audit's vLLM 0.24.0
# for comparability). Do not submit this script until a smoke test on an
# allocated A100 (scripts/slurm/wulver_vllm_env_smoke.sbatch) has confirmed
# torch.cuda.is_available() and a real vllm engine load succeed on this
# node/driver combination — pip installed CUDA 13.x wheel runtimes, which
# have not yet been confirmed compatible with the actual GPU node driver.

#SBATCH --job-name=vllm-kv-pressure-run1
#SBATCH --partition=gpu
#SBATCH --account=ikoutis
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err

# Notes on the values above:
# - partition=gpu is the only real GPU partition open to this account
#   (debug_gpu requires qos=debug, which is capped at cpu=4,mem=16G,
#   1 job — unsuitable for a real model-serving run).
# - account=ikoutis is the only account in `sacctmgr show assoc -u sv96`.
# - qos=standard: MaxWall 3-00:00:00; the only per-user cap on this QOS is
#   gres/gpu:l40=0 (blocks L40 GPUs under standard), which does not affect
#   this a100 request. qos=low is an equally valid fallback (same 3-day
#   MaxWall, no GPU-type cap) if standard's higher priority isn't wanted.
# - gres=gpu:a100:1 requests one full (non-MIG) A100 from the 19 gpu-partition
#   nodes carrying `Gres=gpu:a100:4,mps:a100:400` (as opposed to the 4 MIG
#   nodes carrying `gpu:a100_40g/20g/10g` slices), per the handoff's
#   preference for a full high-memory GPU over MIG.
# - cpus-per-task=16, mem=64G: conservative fraction of a node's 128
#   cores / ~502G RAM, since gpu-partition nodes are shared (not exclusive).
# - time=02:00:00: bounded first run; well under the 3-day QOS cap. Covers
#   model download (~15GB, first run only) + server startup + 8-scenario
#   stress phase (local run of a 0.5B model took a few minutes for 140
#   requests; 7B on a full A100 should still comfortably fit in 2h).

set -euo pipefail

module purge

REPO_DIR="/mmfs1/project/ikoutis/sv96/github/llm-serving-heuristic-evolution"
SCRATCH_ROOT="/mmfs1/scratch/ikoutis/sv96"
HF_CACHE_DIR="${SCRATCH_ROOT}/hf_cache"
OUTPUT_DIR="${SCRATCH_ROOT}/vllm_kv_pressure_${SLURM_JOB_ID}"

VLLM_ENV_PREFIX="/mmfs1/project/ikoutis/sv96/conda_env/vllm-kv-pressure"
source /apps/easybuild/software/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate "${VLLM_ENV_PREFIX}"
# Prevent ~/.local/lib/python3.11/site-packages (and any .pth-injected paths
# from unrelated projects) from shadowing this env's pinned package versions.
unset PYTHONPATH
export PYTHONNOUSERSITE=1

# torch==2.11.0 in this env was built against CUDA 13.0 (torch.version.cuda);
# Wulver's module system only offers CUDA/12.6.0 and CUDA/12.8.0 (no 13.x),
# so do NOT `module load` a system CUDA toolkit here (mismatched nvcc/runtime
# ABI risk). FlashInfer JIT-compiles some sampler kernels at runtime and needs
# nvcc; use the exact-matching CUDA 13.2 toolchain pip already installed into
# this env (nvidia-cuda-nvcc / cuda-toolkit packages) instead. Confirmed
# working in the env smoke test (scripts/slurm/wulver_vllm_env_smoke.sbatch).
CUDA_TOOLCHAIN_DIR="${VLLM_ENV_PREFIX}/lib/python3.11/site-packages/nvidia/cu13"
export CUDA_HOME="${CUDA_TOOLCHAIN_DIR}"
export PATH="${CUDA_TOOLCHAIN_DIR}/bin:${PATH}"

# pip resolved cuda-toolkit==13.0.2 (headers) alongside nvidia-cuda-nvcc==13.2.86
# (compiler) — a minor-version skew that trips FlashInfer's bundled cccl
# compatibility assert when it JIT-compiles its custom top-k/top-p sampling
# kernel. Skip that kernel and use vLLM's native sampler instead; confirmed
# working in the env smoke test. Does not affect KV-pressure/preemption metrics.
export VLLM_USE_FLASHINFER_SAMPLER=0

export HF_HOME="${HF_CACHE_DIR}"

mkdir -p "${REPO_DIR}/logs" "${OUTPUT_DIR}" "${HF_CACHE_DIR}"
cd "${REPO_DIR}"

{
  echo "job_id=${SLURM_JOB_ID}"
  echo "host=$(hostname)"
  echo "branch=$(git rev-parse --abbrev-ref HEAD)"
  echo "commit=$(git rev-parse HEAD)"
  echo "cuda_toolchain=${CUDA_TOOLCHAIN_DIR}"
  "${CUDA_TOOLCHAIN_DIR}/bin/nvcc" --version | tail -1
  python -c "import vllm; print('vllm_version=' + vllm.__version__)" || echo "vllm_version=UNKNOWN"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
} | tee "${OUTPUT_DIR}/job_manifest.txt"

nvidia-smi || true

python scripts/run_gpu_external_validity_audit.py \
  --phase stress \
  --start-vllm-server \
  --vllm-executable "${VLLM_ENV_PREFIX}/bin/vllm" \
  --port 8003 \
  --server-url http://127.0.0.1:8003 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --model-revision main \
  --gpu-memory-utilization 0.82 \
  --max-model-len 16384 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 2048 \
  --block-size 16 \
  --enable-chunked-prefill \
  --disable-prefix-caching \
  --timeout-seconds 600 \
  --output-dir "${OUTPUT_DIR}" \
  --server-log "${OUTPUT_DIR}/server.log" \
  --write-calibration-profile \
  --resume

echo "output_dir=${OUTPUT_DIR}"
echo "Copy small canonical artifacts (summary.json, summary.md, scenario_summary.csv,"
echo "scenario_results.json, calibration_profile.json, environment.json) back into"
echo "${REPO_DIR}/experiments/gpu_external_validity/ only after inspection."
