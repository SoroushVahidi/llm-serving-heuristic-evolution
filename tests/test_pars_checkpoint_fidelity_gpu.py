"""Real-checkpoint fidelity, determinism, and scoring-behavior tests for
the PARS baseline. Requires torch/transformers, the pinned official clone
(``PARS_OFFICIAL_CLONE_PATH``, default ``/home/soroush/.cache/external_baselines/PARS``),
and a locally-trained checkpoint (see
``docs/audits/pars_baseline_implementation_20260804.md`` -- no official
pretrained checkpoint exists; one was trained here with the official,
unmodified training script) -- gated exactly like
tests/test_vllm_ltr_checkpoint_fidelity_gpu.py.

Run with:
    LLMSERVEOPT_RUN_GPU_TESTS=1 pytest tests/test_pars_checkpoint_fidelity_gpu.py -v
"""
from __future__ import annotations

import importlib.util
import math
import os

import pytest

from baselines.pars.adapter import provenance

DEFAULT_CHECKPOINT_PATH = "results/pars_official/predictor_train/alpaca_gpt4_bert/best_model.pt"
CHECKPOINT_PATH = os.environ.get("PARS_TEST_CHECKPOINT_PATH", DEFAULT_CHECKPOINT_PATH)

pytestmark = pytest.mark.skipif(
    os.environ.get("LLMSERVEOPT_RUN_GPU_TESTS") != "1"
    or importlib.util.find_spec("torch") is None
    or importlib.util.find_spec("transformers") is None
    or not os.path.exists(CHECKPOINT_PATH)
    or not os.path.isdir(os.path.join(provenance.DEFAULT_OFFICIAL_CLONE_PATH, ".git")),
    reason="PARS fidelity tests require LLMSERVEOPT_RUN_GPU_TESTS=1, torch, transformers, "
           "the pinned official clone, and a trained checkpoint.",
)


@pytest.fixture(scope="module")
def handle():
    from baselines.pars.adapter.checkpoint_loader import load_pars_predictor

    return load_pars_predictor(CHECKPOINT_PATH)


class TestArchitectureFidelity:
    def test_encoder_is_bert_base_uncased(self, handle):
        assert handle.model.encoder.config.model_type == "bert"
        assert handle.model.encoder.config.hidden_size == 768

    def test_score_head_is_single_linear_unit(self, handle):
        assert handle.model.fc.out_features == 1
        assert handle.model.fc.in_features == handle.model.encoder.config.hidden_size

    def test_max_length_matches_official_default(self, handle):
        assert handle.max_length == provenance.MAX_LENGTH == 128


class TestDeterministicScoring:
    def test_repeated_calls_are_bit_identical(self, handle):
        prompt = "Explain the difference between TCP and UDP."
        first = handle.score(prompt)
        second = handle.score(prompt)
        assert first == second

    def test_batched_matches_singleton_scoring(self, handle):
        prompts = ["What is 2+2?", "Write a short story about a dragon.", "List three colors."]
        singleton_scores = [handle.score(p) for p in prompts]
        batch_scores = handle.score_batch(prompts, batch_size=8)
        for s, b in zip(singleton_scores, batch_scores):
            assert abs(s - b) < 1e-4


class TestRealScoringBehavior:
    def test_scores_are_finite(self, handle):
        prompts = ["Hi", "Tell me a detailed, multi-paragraph history of the Roman Empire, "
                          "covering its rise, key emperors, and eventual fall."]
        for p in prompts:
            s = handle.score(p)
            assert math.isfinite(s)

    def test_diverse_prompts_produce_diverse_scores(self, handle):
        """Structural discrimination check (mirrors the vLLM-LTR
        checkpoint's TestRealScoringBehavior): the model must not collapse
        every input to the same output score."""
        prompts = [
            "Hi",
            "What time is it?",
            "Explain quantum entanglement in detail, including its implications for quantum computing.",
            "Write a comprehensive 500-word essay on climate change mitigation strategies.",
            "List the planets in order from the sun.",
        ]
        scores = [handle.score(p) for p in prompts]
        assert len(set(round(s, 6) for s in scores)) > 1


class TestLongPromptTruncation:
    """Regression coverage mirroring vLLM-LTR's long-prompt truncation
    tests: bert-base-uncased's max_position_embeddings is 512, and the
    official code already explicitly bounds tokenization to
    max_length=128 (verified by reading the official scripts directly --
    see the implementation doc's Step 4) -- confirm this holds through
    this adapter too."""

    def test_prompt_over_max_length_does_not_crash(self, handle):
        long_prompt = "word " * 2000
        score = handle.score(long_prompt)
        assert math.isfinite(score)

    def test_truncation_never_exceeds_max_length(self, handle):
        long_prompt = "word " * 2000
        inputs = handle.tokenizer([long_prompt], return_tensors="pt", truncation=True,
                                   padding=True, max_length=handle.max_length)
        assert inputs["input_ids"].shape[1] <= handle.max_length

    def test_long_prompt_scoring_is_deterministic(self, handle):
        long_prompt = "word " * 2000
        assert handle.score(long_prompt) == handle.score(long_prompt)
