#!/usr/bin/env bash
# Print (and optionally run) Tier 1 dataset reconstruction commands.
# Absolute paths are parameters. Mooncake is never auto-redistributed.
set -euo pipefail

DATASETS_ROOT="${DATASETS_ROOT:?set DATASETS_ROOT}"
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
DRY_RUN="${DRY_RUN:-1}"
EXECUTE="${EXECUTE:-0}"

cd "${REPO_ROOT}"
echo "# Tier 1 reconstruction plan"
echo "# REPO_ROOT=${REPO_ROOT}"
echo "# DATASETS_ROOT=${DATASETS_ROOT}"
echo "# Default DRY_RUN=1; set EXECUTE=1 to run public downloaders."
echo

cmds=(
  "python3 scripts/data/download_burstgpt_v2.py --output-dir ${DATASETS_ROOT}/burstgpt_v2/raw"
  "python3 scripts/data/download_azure_llm_2023.py --help  # use script's documented flags"
  "python3 scripts/data/download_azure_llm_2024.py --help"
  "python3 scripts/data/download_bailian_traces.py --help"
)

for c in "${cmds[@]}"; do
  echo "$c"
done

echo
echo "# Mooncake (INTERNAL OOD ONLY; redistribution prohibited until license clarified)"
echo "# DATA_LICENSE = NOT_EXPLICITLY_SPECIFIED"
echo "# Require explicit local acknowledgment; do not push traces to GitHub."
echo "python3 scripts/data/download_mooncake_traces.py \\"
echo "  --real-dir ${DATASETS_ROOT}/mooncake/raw/real \\"
echo "  --synthetic-dir ${DATASETS_ROOT}/mooncake/raw/synthetic \\"
echo "  --skip-synthetic"
echo
echo "# After downloads: convert + validate with convert_*.py / validate_*.py"
echo "python3 scripts/data/verify_tier1_dataset_checksums.py --datasets-root ${DATASETS_ROOT}"

if [[ "${EXECUTE}" == "1" ]]; then
  echo "EXECUTE=1 is reserved for operators who have reviewed each downloader's --help;" >&2
  echo "this wrapper intentionally does not auto-run downloads (license/path safety)." >&2
  exit 3
fi
