#!/bin/bash -l
#
# Smoke test for the vllm-kv-pressure conda environment on one Wulver A100.
# Verifies GPU visibility, torch CUDA, vllm import, and a tiny model load
# BEFORE the 2-hour main KV-pressure job is submitted.
#
# Verified values (same basis as wulver_vllm_kv_pressure_verified_run1.sbatch):
# account=ikoutis, partition=gpu, qos=standard, one full a100.

#SBATCH --job-name=vllm-env-smoke
#SBATCH --partition=gpu
#SBATCH --account=ikoutis
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err

set -uo pipefail

REPO_DIR="/mmfs1/project/ikoutis/sv96/github/llm-serving-heuristic-evolution"
SCRATCH_ROOT="/mmfs1/scratch/ikoutis/sv96"
HF_CACHE_DIR="${SCRATCH_ROOT}/hf_cache"
VLLM_ENV_PREFIX="/mmfs1/project/ikoutis/sv96/conda_env/vllm-kv-pressure"

mkdir -p "${REPO_DIR}/logs" "${HF_CACHE_DIR}"
cd "${REPO_DIR}"

echo "job_id=${SLURM_JOB_ID}"
echo "host=$(hostname)"
echo "commit=$(git rev-parse HEAD)"

source /apps/easybuild/software/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate "${VLLM_ENV_PREFIX}"
# Prevent ~/.local/lib/python3.11/site-packages (and any .pth-injected paths
# from unrelated projects) from shadowing this env's pinned package versions.
unset PYTHONPATH
export PYTHONNOUSERSITE=1
export HF_HOME="${HF_CACHE_DIR}"

# torch==2.11.0 in this env was built against CUDA 13.0 (torch.version.cuda);
# Wulver's module system only offers CUDA/12.6.0 and CUDA/12.8.0 (no 13.x),
# so do NOT `module load` a system CUDA toolkit here (mismatched nvcc/runtime
# ABI risk). FlashInfer JIT-compiles some sampler kernels at runtime and needs
# nvcc; use the exact-matching CUDA 13.2 toolchain pip already installed into
# this env (nvidia-cuda-nvcc / cuda-toolkit packages) instead.
CUDA_TOOLCHAIN_DIR="${VLLM_ENV_PREFIX}/lib/python3.11/site-packages/nvidia/cu13"
export CUDA_HOME="${CUDA_TOOLCHAIN_DIR}"
export PATH="${CUDA_TOOLCHAIN_DIR}/bin:${PATH}"

# pip resolved cuda-toolkit==13.0.2 (headers) alongside nvidia-cuda-nvcc==13.2.86
# (compiler) — a minor-version skew that trips FlashInfer's bundled cccl
# compatibility assert when it JIT-compiles its custom top-k/top-p sampling
# kernel ("CUDA compiler and CUDA toolkit headers are incompatible"). Skip
# that kernel and use vLLM's native sampler instead; this doesn't affect the
# KV-pressure/preemption/queueing metrics this experiment measures.
export VLLM_USE_FLASHINFER_SAMPLER=0

echo "=== nvidia-smi ==="
nvidia-smi
echo "=== nvidia-smi query ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

echo "=== torch CUDA check ==="
python -c "
import torch
print('torch_version=', torch.__version__)
print('torch_cuda_version=', torch.version.cuda)
print('cuda_available=', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device_name=', torch.cuda.get_device_name(0))
    print('device_capability=', torch.cuda.get_device_capability(0))
    x = torch.randn(1024, 1024, device='cuda')
    y = x @ x
    torch.cuda.synchronize()
    print('matmul_ok=', bool(torch.isfinite(y).all()))
"

echo "=== vllm import / version ==="
python -c "import vllm; print('vllm_version=', vllm.__version__)"
vllm --version || true

echo "=== internet / HF reachability from compute node ==="
python -c "
import urllib.request
for url in ['https://huggingface.co', 'https://github.com']:
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            print(url, r.status)
    except Exception as e:
        print(url, 'FAILED', e)
"

echo "=== tiny HF download/cache check (config only, not full weights) ==="
python -c "
from huggingface_hub import hf_hub_download
p = hf_hub_download(repo_id='Qwen/Qwen2.5-7B-Instruct', filename='config.json')
print('downloaded_config=', p)
"

echo "=== vllm engine smoke load (tiny public model, not the 7B target) ==="
timeout 600 python -c "
from vllm import LLM, SamplingParams
llm = LLM(model='Qwen/Qwen2.5-0.5B-Instruct', gpu_memory_utilization=0.5, max_model_len=2048, enforce_eager=True)
out = llm.generate(['Hello, world!'], SamplingParams(max_tokens=8))
print('generate_ok=', out[0].outputs[0].text)
"

echo "SMOKE_TEST_DONE"
