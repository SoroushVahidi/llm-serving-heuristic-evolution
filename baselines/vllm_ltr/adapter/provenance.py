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
#: Main-conference venue -- confirmed 2026-08-04 via proceedings.neurips.cc
#: and papers.nips.cc (paper id 6c8985579293e0209bdaa4f21bb1d237), and the
#: official repository's own title, "[NeurIPS 2024] Efficient LLM
#: Scheduling by Learning to Rank". The arXiv id is retained below as
#: supplementary (preprint) identification only, not as the venue.
PAPER_VENUE = "NeurIPS 2024 (main conference)"
PAPER_NEURIPS_PROCEEDINGS_URL = (
    "https://papers.nips.cc/paper_files/paper/2024/hash/"
    "6c8985579293e0209bdaa4f21bb1d237-Abstract-Conference.html"
)
PAPER_ARXIV_ID = "2408.15792"  # supplementary preprint identifier, not the venue
PAPER_AUTHORS = ("Fu, Yichao", "Zhu, Siqi", "Su, Runlong", "Qiao, Aurick", "Stoica, Ion", "Zhang, Hao")
PAPER_YEAR = 2024

CHECKPOINT_HF_REPO = "LLM-ltr/OPT-Predictors"
#: HF repo commit (`sha` field of the Hub API's model-info response),
#: recorded 2026-08-04. Distinct from PINNED_COMMIT, which is the vllm-ltr
#: *code* repository's commit.
CHECKPOINT_HF_REVISION = "39df2b41ffe88d5ed967c6035d3838b5b5960379"

#: Both checkpoint variants downloaded, hash-recorded, and verified for this
#: audit (smallest backbone, ShareGPT-trained -- matches this project's own
#: ShareGPT ingestion). See CHECKPOINT_PROVENANCE.md for sha256 hashes and
#: the fidelity audit doc for the full verification record.
#:
#: CHECKPOINT_VARIANT_CLASSIFICATION: the pinned source's classification
#: config (config_prefill_opt_classify.txt). num_labels=10 (ordinal
#: output-length-decile bins), argmax-reduced per
#: official_reference/opt_predictor_head_excerpt.md. FINDING: on a small
#: (n=9) sample of short real ShareGPT prompts, this variant's argmax
#: collapses to the same top bin for every prompt -- no ranking
#: discrimination observed on that sample (see audit doc). The underlying
#: pre-argmax logits DO vary by prompt; only the argmax-reduced bin index
#: saturates.
CHECKPOINT_VARIANT_CLASSIFICATION = "opt-125m-llama3-8b-sharegpt-class-trainbucket820-b32"
#: CHECKPOINT_VARIANT_REGRESSION: the pinned source's regression config
#: (config_prefill_opt.txt). num_labels=1, raw logit used directly (no
#: argmax reduction). FINDING: produced 9/9 distinct scores on the same
#: n=9 sample -- retains ranking-relevant signal that the classification
#: variant's argmax reduction discards on short prompts. Recommended
#: variant for an actual ranking comparison.
CHECKPOINT_VARIANT_REGRESSION = "opt-125m-llama3-8b-sharegpt-score-trainbucket10-b32"
#: Kept for backward compatibility with earlier code in this scaffold.
CHECKPOINT_VARIANT = CHECKPOINT_VARIANT_CLASSIFICATION

#: From each variant's own config.json (architectures, word_embed_proj_dim,
#: and the transformers_version it was exported with are identical across
#: both; only num_labels differs).
CHECKPOINT_ARCHITECTURE = "OPTForSequenceClassification"
CHECKPOINT_NUM_LABELS_CLASSIFICATION = 10
CHECKPOINT_NUM_LABELS_REGRESSION = 1
CHECKPOINT_WORD_EMBED_PROJ_DIM = 768
CHECKPOINT_EXPORT_TRANSFORMERS_VERSION = "4.45.2"
CHECKPOINT_TORCH_DTYPE = "float16"
#: CONFIRMED 2026-08-04: the checkpoint repo ships only config.json /
#: model.safetensors / usage_config.json per variant -- no tokenizer files.
#: The correct tokenizer is the base pretrained model, read from each
#: checkpoint's own (pre-load) config.json._name_or_path.
CHECKPOINT_BASE_TOKENIZER = "facebook/opt-125m"
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

#: Updated 2026-08-04 after real-checkpoint verification (see
#: docs/audits/vllm_ltr_baseline_audit_20260804.md and
#: CHECKPOINT_PROVENANCE.md): architecture, weights, and score-extraction
#: formula are fully verified against the real official checkpoint
#: (exact state-dict key/shape match; bit-exact independent recomputation
#: of compute_logits/argmax). The offline scoring pipeline
#: (adapter/offline_scoring.py) is complete and leakage-free. Still NOT a
#: live per-step simulator policy (ObservableRequest carries no prompt
#: text, and modifying it is explicitly out of scope) -- ranking-ready
#: only once a real prompt-text-carrying dataset supplies scores.
FIDELITY_LABEL = "evaluation-ready external baseline (offline-scored; official checkpoint verified)"
SELECTOR_CANDIDATE = False
HISTORICAL = False
