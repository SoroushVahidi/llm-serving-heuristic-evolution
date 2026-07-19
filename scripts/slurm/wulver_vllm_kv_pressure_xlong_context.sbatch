#!/bin/bash -l
#
# Final long-context KV-pressure/preemption validation job — follow-up to
# 1111541 (baseline) and 1111545 (aggressive stress), both COMPLETED.
#
# Prior findings (docs/wulver_vllm_kv_pressure_results.md):
#   1111541 (gpu-memory-utilization=0.82, max-num-seqs=8):  max KV usage 2.22%,
#     0 preemptions, ~45x headroom (pool 925,824 tokens).
#   1111545 (gpu-memory-utilization=0.35, max-num-seqs=32): max KV usage 13.96%,
#     0 preemptions, ~7.16x headroom (pool 226,960 tokens).
#   Both used prompts averaging <=1,873 tokens against a 16,384-token window
#   (~11% of it) — the binding constraint was never actually memory; it was
#   sequence count and request volume, not context length.
#
# vLLM's own diagnostic from 1111545's server.log: "Maximum concurrency for
# 16,384 tokens per request: 13.85x" at this same gpu-memory-utilization=0.35
# config — i.e. only ~14 concurrent full-16k-context requests would exhaust
# that 226,960-token pool. This job reuses 1111545's exact server config
# (only the workload changes) and drives ~12k-token-prompt requests at
# concurrency 16-24, which is the more efficient lever than further request-
# volume scaling for actually reaching KV saturation.
#
# ESTIMATED KV DEMAND (computed before submission, reproduced in
# job_manifest.txt below):
#   KV pool (same gpu-memory-utilization=0.35 as 1111545): 226,960 tokens
#     (measured, not re-derived, from 1111545's own gpu_worker.py log line).
#   stress_xlong_context_burst16: 16 requests x ~12,000 prompt + up to 512
#     output =~ 12,500 tokens/request at peak => ~200,000 tokens if all 16
#     are simultaneously near-peak length => ~88% of the pool.
#   stress_xlong_context_saturate: 16 requests x ~12,000 prompt + up to 1,024
#     output (~13,000 tokens/request at peak) admitted at t=0, then 8 more
#     ~12,000-token requests arriving staggered at t=15..50s while the first
#     16 are still mid-decode (still growing towards their own 13,000-token
#     peak). ~18-19 concurrent full-length requests would exhaust the pool;
#     this scenario can have up to 24 requests in flight (16 initial + 8
#     late), i.e. it is deliberately sized to exceed capacity if more than
#     ~18-19 of them are concurrently active with a large fraction of their
#     target length resident.
#   Expected headroom at full admission: roughly 0x-0.3x (i.e. plausibly
#   negative/oversubscribed) if vLLM admits requests up to max-num-seqs=32
#   without additional gating — this is the intended stress condition.
#   max-num-seqs=32 (same as 1111545) stays above the scenario's own
#   concurrency (<=24), so the sequence-count cap is not expected to be the
#   first bottleneck; KV capacity is expected to bind first, which is the
#   condition this job is designed to test.
#
# Primary success criteria (any one is a valid, reportable outcome):
#   - peak KV utilization >= 70%
#   - an actual vLLM preemption/recompute event (vllm:num_preemptions_total)
#   - clear KV-capacity-limited waiting behavior (waiting queue that does not
#     shrink even though max-num-seqs=32 is not yet reached)
# If none of these occur, that is still a valid finding given a completed
# run: preemption remains unvalidated at this hardware/model/context-length
# combination, and the report will say so rather than overstate the result.
#
# Same validated environment/toolchain fixes as 1111541/1111545, unchanged:
#   - PYTHONNOUSERSITE=1, unset PYTHONPATH
#   - env-bundled CUDA 13 toolchain (not a Wulver CUDA module)
#   - VLLM_USE_FLASHINFER_SAMPLER=0
#   - explicit --vllm-executable

#SBATCH --job-name=vllm-kv-pressure-xlong
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
SCRATCH_ROOT="/mmfs1/scratch/ikoutis/sv96"
HF_CACHE_DIR="${SCRATCH_ROOT}/hf_cache"
OUTPUT_DIR="${SCRATCH_ROOT}/vllm_kv_pressure_xlong_${SLURM_JOB_ID}"

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
  echo "follow_up_to_jobs=1111541,1111545"
  echo "cuda_toolchain=${CUDA_TOOLCHAIN_DIR}"
  "${CUDA_TOOLCHAIN_DIR}/bin/nvcc" --version | tail -1
  python -c "import vllm; print('vllm_version=' + vllm.__version__)" || echo "vllm_version=UNKNOWN"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
  echo "--- estimated KV demand (see header comment for derivation) ---"
  echo "estimated_kv_pool_tokens=226960 (same gpu-memory-utilization=0.35 as job 1111545; server.log below has the authoritative measured value)"
  echo "estimated_max_concurrency_at_16384_tokens=13.85x (measured by vLLM itself in job 1111545 at this same config)"
  echo "scenario_burst16_expected_peak_tokens=~200000 (~88 pct of pool)"
  echo "scenario_saturate_expected_peak_tokens=~230000-310000 if all 24 admitted near target length (~101-137 pct of pool -- oversubscription is intentional)"
  echo "expected_headroom_ratio=~0.7x-1.1x (burst16) and plausibly <1x (saturate) -- this job intentionally targets low/negative headroom, unlike 1111541/1111545"
} | tee "${OUTPUT_DIR}/job_manifest.txt"

nvidia-smi || true

python scripts/run_gpu_external_validity_audit.py \
  --phase xlong_stress \
  --start-vllm-server \
  --vllm-executable "${VLLM_ENV_PREFIX}/bin/vllm" \
  --port 8020 \
  --server-url http://127.0.0.1:8020 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --model-revision main \
  --gpu-memory-utilization 0.35 \
  --max-model-len 16384 \
  --max-num-seqs 32 \
  --max-num-batched-tokens 4096 \
  --block-size 16 \
  --enable-chunked-prefill \
  --disable-prefix-caching \
  --timeout-seconds 1200 \
  --output-dir "${OUTPUT_DIR}" \
  --server-log "${OUTPUT_DIR}/server.log" \
  --write-calibration-profile \
  --resume

echo "output_dir=${OUTPUT_DIR}"
echo "Copy small canonical artifacts back into"
echo "${REPO_DIR}/experiments/gpu_external_validity/ only after inspection."
