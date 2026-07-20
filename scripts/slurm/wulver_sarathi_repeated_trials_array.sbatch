#!/bin/bash -l
#
# Repeated-trial real-Sarathi runtime validation: N independent Slurm array
# tasks, each a fresh engine build + the 5 primary scenarios from job
# 1111723's validated configuration (short_context_control dropped -- not
# one of the 5 scientifically prioritized scenarios for this repeated-trial
# pass). Reuses the exact validated config from job 1111723 unchanged
# (mistralai/Mistral-7B-Instruct-v0.1, float16 fork default, gpu-memory-
# utilization=0.35, max-num-seqs=16, chunk-size=512, block-size=16,
# max-model-len=16384) -- the only new parameter is --trial-index (labeling
# only, does not affect request generation: prompts/seeds stay fixed and
# matched across trials so any observed variance is attributable to
# system/execution noise, not workload content differences).
#
# Each array task is a fully independent engine build + scenario matrix
# (not N repetitions sharing one engine instance), so trials are genuinely
# independent runs, not just repeated measurements within one warm process.

#SBATCH --job-name=sarathi-repeated-trial
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
SARATHI_ENV_PREFIX="/mmfs1/project/ikoutis/sv96/conda_env/sarathi-serve"
SCRATCH_ROOT="/mmfs1/scratch/ikoutis/sv96"
HF_CACHE_DIR="${SCRATCH_ROOT}/hf_cache"
TRIAL_INDEX="${SLURM_ARRAY_TASK_ID}"
OUTPUT_DIR="${SCRATCH_ROOT}/sarathi_repeated_trial_${SLURM_ARRAY_JOB_ID}_${TRIAL_INDEX}"

SCENARIO_NAMES="sarathi_long_prompt_moderate_output,sarathi_active_decode_plus_arriving_prefill,sarathi_prefill_heavy_burst,sarathi_mixed_prompt_lengths,sarathi_matched_vllm_kv_pressure"

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
  echo "array_job_id=${SLURM_ARRAY_JOB_ID}"
  echo "array_task_id=${SLURM_ARRAY_TASK_ID}"
  echo "trial_index=${TRIAL_INDEX}"
  echo "host=$(hostname)"
  echo "branch=$(git rev-parse --abbrev-ref HEAD)"
  echo "commit=$(git rev-parse HEAD)"
  echo "model=mistralai/Mistral-7B-Instruct-v0.1"
  echo "dtype_policy=fork_default (float16) -- matches validated job 1111723"
  echo "scenario_names=${SCENARIO_NAMES}"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
} | tee "${OUTPUT_DIR}/job_manifest.txt"

set +e
python scripts/run_sarathi_gpu_smoke_and_validation.py \
  --model mistralai/Mistral-7B-Instruct-v0.1 \
  --gpu-memory-utilization 0.35 \
  --max-num-seqs 16 \
  --chunk-size 512 \
  --block-size 16 \
  --max-model-len 16384 \
  --scenario-names "${SCENARIO_NAMES}" \
  --trial-index "${TRIAL_INDEX}" \
  --output-dir "${OUTPUT_DIR}" \
  2>&1 | tee "${OUTPUT_DIR}/run.log"
STATUS=${PIPESTATUS[0]}
set -e

echo "status=${STATUS}" | tee -a "${OUTPUT_DIR}/job_manifest.txt"
echo "output_dir=${OUTPUT_DIR}"
exit "${STATUS}"
