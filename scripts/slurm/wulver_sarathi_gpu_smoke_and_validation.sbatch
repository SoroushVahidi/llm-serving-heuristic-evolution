#!/bin/bash -l
#
# Sarathi-Serve GPU smoke test + small runtime-validation matrix.
#
# Submitted with --dependency=afterok:<sarathi_build_job_id> so it only
# starts if the compute-node build (wulver_sarathi_build_gpu.sbatch)
# actually succeeded. Runs scripts/run_sarathi_gpu_smoke_and_validation.py,
# which first does a bounded smoke test (import, GPU visible, CUDA
# extensions load, model loads, one tiny generation, chunked-prefill
# structurally exercised via a >chunk_size prompt) and only proceeds to the
# 5-scenario runtime-validation matrix if that smoke test passes. If smoke
# fails, the script exits non-zero and this job stops there -- it does not
# attempt the runtime matrix on a broken build.
#
# Same conda env used by the build job (not the vLLM env): the editable
# install from wulver_sarathi_build_gpu.sbatch lives there.

#SBATCH --job-name=sarathi-gpu-validation
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
OUTPUT_DIR="${SCRATCH_ROOT}/sarathi_gpu_validation_${SLURM_JOB_ID}"

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
  echo "depends_on_build_job=${SARATHI_BUILD_JOB_ID:-unknown}"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
} | tee "${OUTPUT_DIR}/job_manifest.txt"

nvidia-smi || true

set +e
python scripts/run_sarathi_gpu_smoke_and_validation.py \
  --model Qwen/Qwen2.5-7B-Instruct \
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
