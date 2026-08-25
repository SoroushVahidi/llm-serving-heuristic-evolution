#!/bin/bash -l
#
# Repeated-trial real-vLLM runtime validation: N independent Slurm array
# tasks, each a fresh vLLM server + the 5 primary scenarios matching job
# 1111706's validated matched configuration (short_context_control
# dropped, matching the Sarathi repeated-trial array). Reuses the exact
# validated config from job 1111706 unchanged (mistralai/
# Mistral-7B-Instruct-v0.1, gpu-memory-utilization=0.35, max-num-seqs=16,
# max-num-batched-tokens=512, block-size=16, max-model-len=16384,
# chunked prefill enabled) -- the only new parameter is --trial-index
# (labeling only; prompts/seeds stay fixed and matched across trials, same
# reasoning as the Sarathi array).
#
# Each array task starts and stops its own vLLM server instance (fresh
# process per trial), matching the Sarathi array's "fresh engine per trial"
# independence property.

#SBATCH --job-name=vllm-repeated-trial
#SBATCH --partition=gpu
#SBATCH --account=ikoutis
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:45:00
#SBATCH --mem=64G
#SBATCH --array=0-4
#SBATCH --output=logs/%x.%A_%a.out
#SBATCH --error=logs/%x.%A_%a.err

set -euo pipefail

module purge

REPO_DIR="/mmfs1/project/ikoutis/sv96/github/llm-serving-heuristic-evolution"
SCRATCH_ROOT="/mmfs1/scratch/ikoutis/sv96"
HF_CACHE_DIR="${SCRATCH_ROOT}/hf_cache"
TRIAL_INDEX="${SLURM_ARRAY_TASK_ID}"
OUTPUT_DIR="${SCRATCH_ROOT}/vllm_repeated_trial_${SLURM_ARRAY_JOB_ID}_${TRIAL_INDEX}"
# Distinct port per array task so concurrently-running tasks (possibly on
# different nodes, but ports only need to be locally unique) never collide
# if Slurm ever colocates two tasks -- harmless otherwise.
PORT=$((8040 + TRIAL_INDEX))

SCENARIO_NAMES="mistral_match_long_prompt_moderate_output,mistral_match_active_decode_plus_arriving_prefill,mistral_match_prefill_heavy_burst,mistral_match_mixed_prompt_lengths,mistral_match_kv_pressure"

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
  echo "array_job_id=${SLURM_ARRAY_JOB_ID}"
  echo "array_task_id=${SLURM_ARRAY_TASK_ID}"
  echo "trial_index=${TRIAL_INDEX}"
  echo "host=$(hostname)"
  echo "branch=$(git rev-parse --abbrev-ref HEAD)"
  echo "commit=$(git rev-parse HEAD)"
  echo "matched_to=sarathi_repeated_trial (mistralai/Mistral-7B-Instruct-v0.1)"
  echo "scenario_names=${SCENARIO_NAMES}"
  python -c "import vllm; print('vllm_version=' + vllm.__version__)" || echo "vllm_version=UNKNOWN"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
} | tee "${OUTPUT_DIR}/job_manifest.txt"

python scripts/run_gpu_external_validity_audit.py \
  --phase mistral_match_stress \
  --start-vllm-server \
  --vllm-executable "${VLLM_ENV_PREFIX}/bin/vllm" \
  --port "${PORT}" \
  --server-url "http://127.0.0.1:${PORT}" \
  --model mistralai/Mistral-7B-Instruct-v0.1 \
  --gpu-memory-utilization 0.35 \
  --max-model-len 16384 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 512 \
  --block-size 16 \
  --enable-chunked-prefill \
  --disable-prefix-caching \
  --timeout-seconds 900 \
  --scenario-names "${SCENARIO_NAMES}" \
  --trial-index "${TRIAL_INDEX}" \
  --no-real-traces \
  --output-dir "${OUTPUT_DIR}" \
  --server-log "${OUTPUT_DIR}/server.log" \
  --resume

echo "output_dir=${OUTPUT_DIR}"
