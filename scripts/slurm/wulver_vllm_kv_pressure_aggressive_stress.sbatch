#!/bin/bash -l
#
# Aggressive KV-pressure/preemption stress job — complementary to
# wulver_vllm_kv_pressure_verified_run1.sbatch (job 1111541, COMPLETED).
#
# 1111541 (defaults: --gpu-memory-utilization 0.82, --max-num-seqs 8) found:
#   Available KV cache memory: 49.45 GiB (925,824 tokens) — vs. peak realistic
#   per-scenario concurrent demand of only ~7k-31.5k tokens (computed below).
#   Result: max_observed_kv_cache_usage topped out at 2.2%, 0 preemptions
#   across all 6 stress scenarios. The default config has ~30x more KV
#   headroom than these scenarios can ever demand, AND --max-num-seqs 8
#   artificially throttles concurrency well below each scenario's actual
#   burst size (up to 24), suppressing simultaneous KV demand further.
#
# Demand estimate (mean_prompt_tokens + target_output_tokens) x concurrency,
# from 1111541's own scenario_summary.csv, at Qwen2.5-7B's measured
# ~57,000 bytes/KV-token (49.45 GiB / 925,824 tokens, confirmed from
# 1111541's server.log):
#   stress_high_concurrency_queue:   296 tok x 24  =  7,104 tok (~0.38 GiB)
#   stress_long_decode_kv:           681 tok x 16  = 10,896 tok (~0.58 GiB)
#   stress_mixed_prefill_decode:   1,210 tok x 16  = 19,360 tok (~1.03 GiB)
#   stress_burst_overload_recovery: 1,268 tok x 24 = 30,432 tok (~1.62 GiB)
#   stress_long_prefill:           1,967 tok x 16  = 31,472 tok (~1.67 GiB)
#   stress_kv_pressure:            2,628 tok x 12  = 31,536 tok (~1.68 GiB)
#
# Two isolated changes from 1111541 (everything else identical, for a clean
# A/B comparison on the SAME 6 built-in stress scenarios):
#
#   1. --max-num-seqs 32 (was 8): removes the artificial concurrency cap.
#      32 is above the largest scenario burst (24), so memory — not the
#      scheduler's seq-count limit — becomes the binding constraint.
#   2. --gpu-memory-utilization 0.35 (was 0.82): from 1111541's own
#      measurements, weights took 14.29 GiB and non-KV overhead was
#      ~1.86 GiB (49.45 GiB KV pool at 0.82*80=65.6 GiB reserved implies
#      65.6-14.29-49.45=1.86 GiB fixed overhead). At 0.35, reserved =
#      0.35*80=28 GiB, so KV pool ~= 28-14.29-1.86 ~= 11.85 GiB (~208k
#      tokens) — ~6.3x smaller than 1111541's pool, while keeping a large
#      safety margin (~7.6x) above the 1.86 GiB floor so the engine can
#      still boot; pushing utilization low enough to sit *below* the
#      ~31.5k-token peak-scenario demand was rejected as unsafe (it would
#      sit within ~1x of the bare weights+overhead floor, i.e. real risk of
#      a startup OOM crash rather than a graceful preemption finding).
#
# Expected outcome: substantially higher observed KV utilization % and
# running-sequence counts than 1111541 in every scenario (the seq-cap
# alone was suppressing true burst concurrency). Actual preemption is NOT
# guaranteed at this pool size per the estimate above — if none is
# observed, that is itself a valid, reportable finding ("this workload's
# volumes don't demand enough concurrent KV memory to force preemption on
# an 80GB A100 even at ~6x tighter budget; would need larger scenario
# volumes, not just a smaller budget, to force it").

#SBATCH --job-name=vllm-kv-pressure-aggressive
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
SCRATCH_ROOT="/mmfs1/scratch/ikoutis/sv96"
HF_CACHE_DIR="${SCRATCH_ROOT}/hf_cache"
OUTPUT_DIR="${SCRATCH_ROOT}/vllm_kv_pressure_aggressive_${SLURM_JOB_ID}"

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
  echo "complementary_to_job=1111541"
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
  --port 8010 \
  --server-url http://127.0.0.1:8010 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --model-revision main \
  --gpu-memory-utilization 0.35 \
  --max-model-len 16384 \
  --max-num-seqs 32 \
  --max-num-batched-tokens 4096 \
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
