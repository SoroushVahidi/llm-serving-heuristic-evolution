"""Importable provenance constants for the PARS baseline.

Mirrors ``baselines/pars/PROVENANCE.md``. Kept as plain constants (not
parsed out of the markdown) so tests can assert code and doc agree without
a markdown parser; if you update one, update the other. Follows
``baselines/vllm_ltr/adapter/provenance.py``'s established convention.
"""
from __future__ import annotations

OFFICIAL_REPOSITORY = "https://github.com/SPEAR-UIC/PARS"
PINNED_COMMIT = "fd4e125b65bb73aef5eccafa79c2509434be61ec"
PINNED_COMMIT_DATE = "2026-07-24"
#: No LICENSE file exists anywhere in the official repository (verified via
#: the full recursive git tree, GitHub API license field is null). See
#: PROVENANCE.md's License section for the full explanation and the
#: explicit, user-directed decision to proceed with local, non-commercial
#: research use while never committing/redistributing the official source.
LICENSE = "NONE (unlicensed -- all rights reserved by default; see PROVENANCE.md)"

PAPER_TITLE = "Ranking Before Serving: Low-Latency LLM Serving via Pairwise Learning-to-Rank"
#: v1 (2025-09-25) title, matching common shorthand references to this work.
PAPER_TITLE_V1 = "Prompt-Aware Scheduling for Low-Latency LLM Serving"
PAPER_VENUE = "ISC High Performance 2026 (June 22-26, 2026, Hamburg, Germany)"
PAPER_ARXIV_ID = "2510.03243"
PAPER_AUTHORS = (
    "Tao, Yiheng", "Zhang, Yihe", "Dearing, Matthew", "Wang, Xin",
    "Fan, Yuping", "Papka, Michael E.", "Lan, Zhiling",
)
PAPER_YEAR = 2026

#: Local, non-committed clone path (see PROVENANCE.md -- the official repo
#: is never vendored into this project's git history). Overridable via the
#: PARS_OFFICIAL_CLONE_PATH environment variable. Portable (expands to the
#: current user's home directory, not a hardcoded machine-specific path) --
#: on this development machine it resolves to the exact same path the
#: repo was originally cloned to (/home/soroush/.cache/external_baselines/PARS),
#: so this is a non-functional portability fix, not a behavior change.
import os as _os

DEFAULT_OFFICIAL_CLONE_PATH = _os.path.expanduser("~/.cache/external_baselines/PARS")

#: Training dataset chosen from the 4 official options (see PROVENANCE.md
#: for the full license comparison and domain-realism rationale).
TRAINING_DATASET_HF_REPO = "vicgalle/alpaca-gpt4"
TRAINING_DATASET_LICENSE = "CC BY-NC 4.0 (non-commercial)"
TRAINING_PREPROCESSING_SCRIPT = "data_preprocess/scripts/preprocess_alpaca_gpt4.py"
TRAINING_SIMILARITY_THRESHOLD = 0.2
TRAINING_TRAIN_NUM_PAIRS = 10
TRAINING_VAL_NUM_PAIRS = 10
TRAINING_TEST_SIZE = 0.1
TRAINING_PREPROCESS_SEED = 42

#: Official, unmodified training hyperparameters (train_pairwise_bert.py
#: argparse defaults -- none overridden).
MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128
BATCH_SIZE = 128
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
MARGIN = 1.0
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
TRAINING_SEED = 42

#: Architecture, verified directly by reading
#: predictor_train/scripts/train_pairwise_bert.py and
#: predictor_serving/scripts/serve_predictor_score.py (identical
#: PairwiseRanker class definition in both, confirmed byte-for-byte via
#: diff): bert-base-uncased encoder + Linear(hidden_size, 1) on the pooler
#: output (falls back to last_hidden_state[:, 0] / [CLS] if no pooler).
BACKBONE_MODEL_TYPE = "bert"
SCORE_HEAD_OUTPUT_DIM = 1

#: Verified directly from the MarginRankingLoss construction in
#: train_pairwise_bert.py: target = +1 if prompt_A's real response is
#: LONGER than prompt_B's, else -1; loss encourages score_A > score_B when
#: target=+1. A HIGHER PARS score therefore means the model predicts a
#: LONGER response. For SJF-style scheduling (shortest job first), a
#: deployable policy must rank by ASCENDING score, not descending -- the
#: opposite convention from vLLM-LTR (see
#: baselines/vllm_ltr/adapter/ranking_adapter.py, which sorts by
#: descending score because that checkpoint's score is a direct
#: LTR priority signal, not a length prediction).
HIGHER_SCORE_MEANS_LONGER_PREDICTED_RESPONSE = True

FIDELITY_LABEL = "evaluation-ready external baseline (offline-scored; official code + self-trained checkpoint, license-encumbered)"
SELECTOR_CANDIDATE = False
HISTORICAL = False
