"""Importable provenance constants for the vLLM-LTR baseline.

Mirrors ``baselines/vllm_ltr/PROVENANCE.md``. Kept as plain constants (not
parsed out of the markdown) so tests can assert code and doc agree without
a markdown parser; if you update one, update the other.
"""
from __future__ import annotations

OFFICIAL_REPOSITORY = "https://github.com/hao-ai-lab/vllm-ltr"
PINNED_COMMIT = "13bbf6ff3dab661791d41362551b089e5f77c91c"
PINNED_COMMIT_DATE = "2024-10-31"
LICENSE = "Apache-2.0"

PAPER_TITLE = "Efficient LLM Scheduling by Learning to Rank"
PAPER_ARXIV_ID = "2408.15792"
PAPER_AUTHORS = ("Fu, Yichao", "Zhu, Siqi", "Su, Runlong", "Qiao, Aurick", "Stoica, Ion", "Zhang, Hao")
PAPER_YEAR = 2024

CHECKPOINT_HF_REPO = "LLM-ltr/OPT-Predictors"
TRAINING_DATASET_HF_REPO = "LLM-ltr/Llama3-Trace"

# Environment the official repo's README pins for its own conda env. This
# adapter does not require an exact match (transformers-only inference is
# far more version-tolerant than the full vLLM fork build), but a checkpoint
# sidecar recording versions far outside this range is a signal worth
# surfacing -- see checkpoint_loader.VersionMismatchError.
OFFICIAL_PYTHON_VERSION = "3.10"
OFFICIAL_TORCH_VERSION = "2.2.1"
OFFICIAL_CUDA_VERSION = "12.1"

#: Backbone architecture verified by reading the pinned commit's
#: vllm/model_executor/models/opt.py (see
#: official_reference/opt_predictor_head_excerpt.md). Plain HF OPT backbone
#: + one Linear(word_embed_proj_dim, num_labels, bias=False) head, no
#: pooling/MLP/dropout at inference.
BACKBONE_MODEL_TYPE = "opt"
SCORE_HEAD_BIAS = False

FIDELITY_LABEL = "official predictor reused with a simulator adapter (offline-only)"
SELECTOR_CANDIDATE = False
HISTORICAL = False
