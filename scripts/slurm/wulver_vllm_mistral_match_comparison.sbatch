#!/bin/bash -l
#
# Matched real-vLLM comparison run for the Sarathi-vs-vLLM runtime
# validation, using the SAME model (mistralai/Mistral-7B-Instruct-v0.1) and
# byte-for-byte identical request shapes as
# scripts/slurm/wulver_sarathi_gpu_mistral_validation.sbatch's 6 scenarios
# (--phase mistral_match_stress in scripts/run_gpu_external_validity_audit.py,
# verified request-for-request identical to
# run_sarathi_gpu_smoke_and_validation.py's make_scenarios() before
# submission).
#
# Server config chosen to be as comparable as practical to the Sarathi run's
# SarathiSchedulerConfig(chunk_size=512, max_num_seqs=16) and
# WorkerConfig(gpu_memory_utilization=0.35): max-num-batched-tokens=512
# mirrors chunk_size as vLLM's per-step new-token budget under chunked
# prefill, max-num-seqs=16 and gpu-memory-utilization=0.35 are set to the
# same values. This is NOT a claim that the two schedulers become identical
# under matched flags -- only that the comparison is not confounded by an
# arbitrary difference in these particular knobs.

#SBATCH --job-name=vllm-mistral-match
#SBATCH --partition=gpu
#SBATCH --account=ikoutis
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err

set -euo pipefail

module purge

REPO_DIR="/mmfs1/project/ikoutis/sv96/github/llm-serving-heuristic-evolution"
SCRATCH_ROOT="/mmfs1/scratch/ikoutis/sv96"
HF_CACHE_DIR="${SCRATCH_ROOT}/hf_cache"
OUTPUT_DIR="${SCRATCH_ROOT}/vllm_mistral_match_${SLURM_JOB_ID}"

VLLM_ENV_PREFIX="/mmfs1/project/ikoutis/sv96/conda_env/vllm-kv-pressure"
source /apps/easybuild/software/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate "${VLLM_ENV_PREFIX}"
unset PYTHONPATH
export PYTHONNOUSERSITE=1

CUDA_TOOLCHAIN_DIR="${VLLM_ENV_PREFIX}/lib/python3.11/site-packages/nvidia/cu13"
export CUDA_HOME="${CUDA_TOOLCHAIN_DIR}"
export PATH="${CUDA_TOOLCHAIN_DIR}/bin:${PATH}"
export VLLM_USE_FLASHINFER_SAMPLER=0

export HF_HOME="${HF_CACHE_DIR}"

mkdir -p "${REPO_DIR}/logs" "${OUTPUT_DIR}" "${HF_CACHE_DIR}"
cd "${REPO_DIR}"

{
  echo "job_id=${SLURM_JOB_ID}"
  echo "host=$(hostname)"
  echo "branch=$(git rev-parse --abbrev-ref HEAD)"
  echo "commit=$(git rev-parse HEAD)"
  echo "matched_to=sarathi_mistral_validation (mistralai/Mistral-7B-Instruct-v0.1)"
  echo "cuda_toolchain=${CUDA_TOOLCHAIN_DIR}"
  "${CUDA_TOOLCHAIN_DIR}/bin/nvcc" --version | tail -1
  python -c "import vllm; print('vllm_version=' + vllm.__version__)" || echo "vllm_version=UNKNOWN"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
} | tee "${OUTPUT_DIR}/job_manifest.txt"

nvidia-smi || true

python scripts/run_gpu_external_validity_audit.py \
  --phase mistral_match_stress \
  --start-vllm-server \
  --vllm-executable "${VLLM_ENV_PREFIX}/bin/vllm" \
  --port 8030 \
  --server-url http://127.0.0.1:8030 \
  --model mistralai/Mistral-7B-Instruct-v0.1 \
  --gpu-memory-utilization 0.35 \
  --max-model-len 16384 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 512 \
  --block-size 16 \
  --enable-chunked-prefill \
  --disable-prefix-caching \
  --timeout-seconds 900 \
  --output-dir "${OUTPUT_DIR}" \
  --server-log "${OUTPUT_DIR}/server.log" \
  --write-calibration-profile \
  --resume

echo "output_dir=${OUTPUT_DIR}"
echo "Copy small canonical artifacts back into"
echo "${REPO_DIR}/experiments/gpu_external_validity/ only after inspection."
