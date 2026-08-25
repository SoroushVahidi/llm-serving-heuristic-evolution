#!/bin/bash -l
#
# Generic Wulver vLLM KV-pressure/preemption validation template.
#
# Do not submit this file unchanged. Replace every VERIFY_ON_WULVER_* value
# after an authenticated read-only Wulver audit confirms account, QOS,
# partition, storage paths, and module/container options.

#SBATCH --job-name=vllm-kv-pressure
#SBATCH --partition=VERIFY_ON_WULVER_GPU_PARTITION
#SBATCH --account=VERIFY_ON_WULVER_ACCOUNT
#SBATCH --qos=VERIFY_ON_WULVER_QOS
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=VERIFY_ON_WULVER_CPUS
#SBATCH --gres=VERIFY_ON_WULVER_GPU_GRES
#SBATCH --time=VERIFY_ON_WULVER_WALLTIME
#SBATCH --mem=VERIFY_ON_WULVER_MEMORY
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err

set -euo pipefail

module purge

# Load only modules verified on Wulver, for example:
# module load VERIFY_ON_WULVER_CUDA_MODULE
# module load VERIFY_ON_WULVER_PYTHON_OR_CONDA_MODULE
# module load VERIFY_ON_WULVER_APPTAINER_MODULE

REPO_DIR="VERIFY_ON_WULVER_PROJECT_OR_HOME_PATH/llm-serving-heuristic-evolution"
SCRATCH_ROOT="VERIFY_ON_WULVER_SCRATCH_PATH"
OUTPUT_DIR="${SCRATCH_ROOT}/vllm_kv_pressure_${SLURM_JOB_ID}"

mkdir -p "${REPO_DIR}/logs" "${OUTPUT_DIR}"
cd "${REPO_DIR}"

echo "job_id=${SLURM_JOB_ID}"
echo "host=$(hostname)"
echo "branch=$(git rev-parse --abbrev-ref HEAD)"
echo "commit=$(git rev-parse HEAD)"
nvidia-smi || true

# If using a container, replace the direct python call below with:
# apptainer exec --nv VERIFY_ON_WULVER_CONTAINER.sif python scripts/run_gpu_external_validity_audit.py ...

python scripts/run_gpu_external_validity_audit.py \
  --phase stress \
  --start-vllm-server \
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
echo "Copy small canonical artifacts back to GitHub only after inspection."
