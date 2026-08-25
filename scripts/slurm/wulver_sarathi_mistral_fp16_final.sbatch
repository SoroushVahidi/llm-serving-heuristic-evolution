#!/bin/bash -l
#
# Final bounded Sarathi-Serve runtime-validation attempt: mistralai/
# Mistral-7B-Instruct-v0.1, using this vendored fork's OWN DEFAULT dtype
# (float16) instead of forcing bfloat16.
#
# Prior attempts on this exact model:
#   1111705: FAILED, rope_scaling KeyError (fixed at the source in
#     _patch_get_and_verify_max_len_for_default_rope_type -- still active,
#     unchanged, in this run).
#   1111711: FAILED, past rope_scaling and model construction, into the
#     first forward pass: "ValueError: The dtype of q torch.bfloat16 does
#     not match the q_data_type torch.float16 specified in plan function."
#     inside sarathi/model_executor/attention/flashinfer_attention_wrapper.py
#     -- this fork's FlashInfer prefill-wrapper plan step does not track
#     the model's configured dtype in at least that code path. That job was
#     forcing dtype="bfloat16" (matching every real-vLLM run in this whole
#     investigation). sarathi/config/config.py's own ModelConfig.dtype
#     defaults to "float16", not "bfloat16".
#
# FIX (this run): scripts/run_sarathi_gpu_smoke_and_validation.py now takes
# an optional --dtype flag (default None). This job intentionally does NOT
# pass --dtype, so ModelConfig is constructed without a dtype kwarg at all
# and this fork's own float16 default applies -- the scientifically
# justified next step per the task instructions, not a speculative fix.
# No vendored sarathi-serve source was modified; no rebuild needed (job
# 1111574's build and its CUDA extensions are unchanged and still import
# cleanly, confirmed before this job was written).
#
# This is the final bounded retry for this validation path. If this run
# still fails, per instruction it will NOT be retried again automatically;
# the failure will be classified precisely and left for a human decision.
#
# Smoke (defined in run_sarathi_gpu_smoke_and_validation.py's run_smoke())
# gates the runtime matrix automatically in a single process/job: main()
# only proceeds past the smoke test if smoke["passed"] is true, and every
# scenario's results are flushed to scenario_results.json immediately after
# that scenario completes (incremental checkpointing), so partial progress
# survives a walltime timeout or any later failure.

#SBATCH --job-name=sarathi-mistral-fp16-final
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

set -euo pipefail

module purge

REPO_DIR="/mmfs1/project/ikoutis/sv96/github/llm-serving-heuristic-evolution"
SARATHI_ENV_PREFIX="/mmfs1/project/ikoutis/sv96/conda_env/sarathi-serve"
SCRATCH_ROOT="/mmfs1/scratch/ikoutis/sv96"
HF_CACHE_DIR="${SCRATCH_ROOT}/hf_cache"
OUTPUT_DIR="${SCRATCH_ROOT}/sarathi_mistral_fp16_final_${SLURM_JOB_ID}"

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
  echo "dtype_policy=fork_default (float16) -- --dtype intentionally NOT passed"
  echo "prior_attempts=1111705(rope_scaling KeyError, fixed),1111711(bfloat16/FlashInfer dtype mismatch)"
  echo "retry_policy=final_bounded_attempt_no_further_automatic_retry"
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
