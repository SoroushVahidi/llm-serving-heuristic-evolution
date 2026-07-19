#!/bin/bash -l
#
# Sarathi-Serve GPU smoke test + runtime-validation matrix, using
# mistralai/Mistral-7B-Instruct-v0.1 -- a model this vendored fork's model
# registry actually supports (MistralForCausalLM), unlike Qwen2.5-7B-
# Instruct (Qwen2ForCausalLM, unsupported -- see jobs 1111576/1111663 and
# docs/wulver_vllm_kv_pressure_results.md). The Sarathi build itself
# (job 1111574) already succeeded and is not rebuilt here.
#
# Verified before submission (read-only, via transformers.AutoConfig on the
# login node): architectures=['MistralForCausalLM'] (in the pinned fork's
# supported list: Falcon/Llama/LLaMA/InternLM/Mistral/Mixtral/
# QWenLMHeadModel/Yi), max_position_embeddings=32768, rope_scaling=
# {'rope_theta': 10000.0, 'rope_type': 'default'} (same benign no-op pattern
# as Qwen2.5 -- already handled by this script's
# _patch_get_and_verify_max_len_for_default_rope_type patch), public/
# ungated (Apache 2.0, no HF token required), comparable size to
# Qwen2.5-7B-Instruct (hidden_size=4096, 32 layers).

#SBATCH --job-name=sarathi-mistral-validation
#SBATCH --partition=gpu
#SBATCH --account=ikoutis
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:30:00
#SBATCH --mem=64G
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err

set -euo pipefail

module purge

REPO_DIR="/mmfs1/project/ikoutis/sv96/github/llm-serving-heuristic-evolution"
SARATHI_ENV_PREFIX="/mmfs1/project/ikoutis/sv96/conda_env/sarathi-serve"
SCRATCH_ROOT="/mmfs1/scratch/ikoutis/sv96"
HF_CACHE_DIR="${SCRATCH_ROOT}/hf_cache"
OUTPUT_DIR="${SCRATCH_ROOT}/sarathi_mistral_validation_${SLURM_JOB_ID}"

source /apps/easybuild/software/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate "${SARATHI_ENV_PREFIX}"
unset PYTHONPATH
export PYTHONNOUSERSITE=1

CUDA_TOOLCHAIN_DIR="/apps/easybuild/el9_5.x86_64/software/CUDA/12.8.0"
export CUDA_HOME="${CUDA_TOOLCHAIN_DIR}"
export PATH="${CUDA_TOOLCHAIN_DIR}/bin:${PATH}"
export HF_HOME="${HF_CACHE_DIR}"

mkdir -p "${REPO_DIR}/logs" "${OUTPUT_DIR}"
cd "${REPO_DIR}"

{
  echo "job_id=${SLURM_JOB_ID}"
  echo "host=$(hostname)"
  echo "branch=$(git rev-parse --abbrev-ref HEAD)"
  echo "commit=$(git rev-parse HEAD)"
  echo "model=mistralai/Mistral-7B-Instruct-v0.1"
  echo "model_architecture=MistralForCausalLM"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
} | tee "${OUTPUT_DIR}/job_manifest.txt"

nvidia-smi || true

set +e
python scripts/run_sarathi_gpu_smoke_and_validation.py \
  --model mistralai/Mistral-7B-Instruct-v0.1 \
  --gpu-memory-utilization 0.35 \
  --max-num-seqs 16 \
  --chunk-size 512 \
  --block-size 16 \
  --max-model-len 16384 \
  --output-dir "${OUTPUT_DIR}" \
  2>&1 | tee "${OUTPUT_DIR}/run.log"
STATUS=${PIPESTATUS[0]}
set -e

echo "status=${STATUS}" | tee -a "${OUTPUT_DIR}/job_manifest.txt"
echo "output_dir=${OUTPUT_DIR}"
exit "${STATUS}"
