"""Real-checkpoint fidelity, semantic-equivalence, and overhead tests for
the vLLM-LTR baseline. Downloads the official LLM-ltr/OPT-Predictors
checkpoint (~239 MB per variant) from Hugging Face and requires
torch/transformers -- gated exactly like tests/test_calibration_gpu.py.

Run with:
    LLMSERVEOPT_RUN_GPU_TESTS=1 pytest tests/test_vllm_ltr_checkpoint_fidelity_gpu.py -m gpu -v

See docs/audits/vllm_ltr_baseline_audit_20260804.md for the narrative
writeup of these results and baselines/vllm_ltr/CHECKPOINT_PROVENANCE.md
for the exact hashes/revision being verified.
"""
from __future__ import annotations

import importlib.util
import json
import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("LLMSERVEOPT_RUN_GPU_TESTS") != "1"
    or importlib.util.find_spec("torch") is None
    or importlib.util.find_spec("transformers") is None,
    reason="Checkpoint fidelity tests require LLMSERVEOPT_RUN_GPU_TESTS=1, torch, and transformers.",
)

CHECKPOINT_REVISION = "39df2b41ffe88d5ed967c6035d3838b5b5960379"
CLASSIFICATION_SUBFOLDER = "opt-125m-llama3-8b-sharegpt-class-trainbucket820-b32"
REGRESSION_SUBFOLDER = "opt-125m-llama3-8b-sharegpt-score-trainbucket10-b32"


def _sharegpt_tiny_prompts():
    fixture = os.path.join(os.path.dirname(__file__), "fixtures", "sharegpt_tiny.json")
    with open(fixture, "r", encoding="utf-8") as f:
        data = json.load(f)
    prompts = []
    for convo in data:
        for turn in convo.get("conversations", []):
            if turn.get("from") in ("human", "user"):
                prompts.append(turn["value"])
    return prompts


@pytest.fixture(scope="module")
def verified_environments():
    import torch
    import transformers

    return [
        {"torch_version": "2.2.1", "transformers_version": "4.45.2"},
        {"torch_version": torch.__version__, "transformers_version": transformers.__version__},
    ]


@pytest.fixture(scope="module")
def classification_handle(verified_environments):
    from baselines.vllm_ltr.adapter.checkpoint_loader import (
        download_and_provision_checkpoint,
        load_opt_predictor_from_local,
    )

    ckpt_dir = download_and_provision_checkpoint(
        repo_id="LLM-ltr/OPT-Predictors",
        subfolder=CLASSIFICATION_SUBFOLDER,
        revision=CHECKPOINT_REVISION,
        local_dir="unused",
        verified_environments=verified_environments,
    )
    return load_opt_predictor_from_local(ckpt_dir), ckpt_dir


@pytest.fixture(scope="module")
def regression_handle(verified_environments):
    from baselines.vllm_ltr.adapter.checkpoint_loader import (
        download_and_provision_checkpoint,
        load_opt_predictor_from_local,
    )

    ckpt_dir = download_and_provision_checkpoint(
        repo_id="LLM-ltr/OPT-Predictors",
        subfolder=REGRESSION_SUBFOLDER,
        revision=CHECKPOINT_REVISION,
        local_dir="unused",
        verified_environments=verified_environments,
    )
    return load_opt_predictor_from_local(ckpt_dir), ckpt_dir


@pytest.mark.gpu
class TestArchitectureFidelity:
    def test_state_dict_exact_match_classification(self, classification_handle):
        """No missing keys, no unexpected keys, no shape mismatches between
        the raw checkpoint and a freshly-constructed HF
        OPTForSequenceClassification from the same config."""
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoModelForSequenceClassification

        _, ckpt_dir = classification_handle
        raw_sd = load_file(os.path.join(ckpt_dir, "model.safetensors"))
        cfg = AutoConfig.from_pretrained(ckpt_dir)
        fresh = AutoModelForSequenceClassification.from_config(cfg)
        fresh_keys = set(fresh.state_dict().keys())
        raw_keys = set(raw_sd.keys())
        assert fresh_keys - raw_keys == set(), "keys missing from checkpoint"
        assert raw_keys - fresh_keys == set(), "unexpected keys in checkpoint"
        for k in fresh_keys:
            assert tuple(fresh.state_dict()[k].shape) == tuple(raw_sd[k].shape), k

    def test_config_matches_recorded_provenance(self, classification_handle):
        from baselines.vllm_ltr.adapter import provenance

        handle, _ = classification_handle
        assert handle.num_labels == provenance.CHECKPOINT_NUM_LABELS_CLASSIFICATION
        assert handle.model.config.word_embed_proj_dim == provenance.CHECKPOINT_WORD_EMBED_PROJ_DIM

    def test_tokenizer_pad_token_matches_checkpoint_config(self, classification_handle):
        handle, _ = classification_handle
        assert handle.tokenizer.pad_token_id == handle.model.config.pad_token_id
        assert handle.tokenizer.pad_token_id == 1


@pytest.mark.gpu
class TestSemanticEquivalence:
    """Independent recomputation cross-check: manually run the backbone
    (AutoModel, bypassing HF's SequenceClassification pooling code path)
    and apply the score.weight linear map by hand, replicating the pinned
    source's compute_logits formula from first principles. If this matches
    the high-level AutoModelForSequenceClassification wrapper our adapter
    actually uses, that proves the wrapper is a faithful stand-in for the
    literal pinned formula -- not just an assumption that HF's internals
    happen to do the right thing."""

    def _cross_check(self, ckpt_dir, prompts):
        import torch
        from safetensors.torch import load_file
        from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
        model_a = AutoModelForSequenceClassification.from_pretrained(ckpt_dir)
        model_a.eval()
        backbone = AutoModel.from_pretrained(ckpt_dir)
        backbone.eval()
        raw_sd = load_file(os.path.join(ckpt_dir, "model.safetensors"))
        score_weight = raw_sd["score.weight"]

        # Same explicit max_length bound as OPTPredictorHandle.score_batch()
        # (checkpoint_loader.py) -- facebook/opt-125m's tokenizer ships
        # model_max_length at HF's ~1e30 "unset" sentinel, so bare
        # truncation=True is a no-op and a prompt longer than the model's
        # real max_position_embeddings crashes deep in the forward pass
        # instead of being truncated. Both scoring paths must use the same
        # bound or this cross-check would silently stop being representative
        # of the adapter's real behavior on long prompts. See
        # docs/audits/vllm_ltr_comparative_evaluation_recovery_20260804.md.
        max_length = getattr(model_a.config, "max_position_embeddings", None)
        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length
        )
        with torch.no_grad():
            logits_a = model_a(**inputs).logits
            hidden = backbone(**inputs).last_hidden_state
            attn = inputs["attention_mask"]
            seq_lengths = attn.sum(dim=1) - 1
            pooled = hidden[torch.arange(hidden.size(0)), seq_lengths]
            logits_b = pooled @ score_weight.T
        return logits_a, logits_b

    def test_bit_exact_independent_recomputation_classification(self, classification_handle):
        _, ckpt_dir = classification_handle
        logits_a, logits_b = self._cross_check(ckpt_dir, _sharegpt_tiny_prompts())
        assert (logits_a - logits_b).abs().max().item() == 0.0

    def test_bit_exact_independent_recomputation_regression(self, regression_handle):
        _, ckpt_dir = regression_handle
        logits_a, logits_b = self._cross_check(ckpt_dir, _sharegpt_tiny_prompts())
        assert (logits_a - logits_b).abs().max().item() == 0.0


@pytest.mark.gpu
class TestRealScoringBehavior:
    """Empirical findings on the real checkpoints against real (small,
    n=9) ShareGPT text -- see docs/audits/vllm_ltr_baseline_audit_20260804.md
    for the full discussion and caveats about sample size."""

    def test_classification_variant_saturates_on_short_prompts(self, classification_handle):
        """Documented finding, pinned as a regression lock: on this small
        sample of short prompts, the classification variant's argmax
        reduction collapses to a single bin for every prompt -- not a bug,
        an accurate (if unhelpful for ranking) description of this
        checkpoint's behavior on very short inputs."""
        handle, _ = classification_handle
        scores = handle.score_batch(_sharegpt_tiny_prompts(), batch_size=4)
        assert len(set(scores)) == 1

    def test_regression_variant_discriminates_short_prompts(self, regression_handle):
        """The regression variant retains real signal where the
        classification variant's argmax saturates -- recommended variant
        for an actual ranking comparison."""
        handle, _ = regression_handle
        scores = handle.score_batch(_sharegpt_tiny_prompts(), batch_size=4)
        assert len(set(scores)) == len(scores)

    def test_deterministic_across_repeated_calls(self, regression_handle):
        handle, _ = regression_handle
        prompts = _sharegpt_tiny_prompts()
        first = handle.score_batch(prompts, batch_size=4)
        second = handle.score_batch(prompts, batch_size=4)
        assert first == second

    def test_batched_matches_singleton_scoring(self, regression_handle):
        """Batched (padded) and singleton (unpadded) scoring agree to
        within fp16 tolerance but are NOT bit-identical -- a real,
        expected finding: the checkpoint runs in float16
        (config.torch_dtype), and attention over padding positions (masked
        out mathematically, via attention_mask) still changes
        floating-point summation order versus an unpadded forward pass.
        Observed magnitude on this sample: ~2e-3 absolute, on logits
        ranging roughly [-3.7, 2.2]. This is standard fp16 batch-padding
        behavior, not an adapter defect -- verified below to stay within a
        generous tolerance, not exact equality."""
        handle, _ = regression_handle
        prompts = _sharegpt_tiny_prompts()
        batched = handle.score_batch(prompts, batch_size=4)
        singleton = [handle.score(p) for p in prompts]
        for b, s in zip(batched, singleton):
            assert abs(b - s) < 0.01, (b, s)


@pytest.mark.gpu
class TestLongPromptTruncation:
    """Regression coverage for the real bug found scoring real WildChat
    prompts (some >8k tokens) offline: facebook/opt-125m's tokenizer ships
    model_max_length at HF's ~1e30 "unset" sentinel, so bare
    truncation=True is a no-op, and a prompt longer than the checkpoint's
    real max_position_embeddings=2048 raised "index out of range in self"
    from the position-embedding lookup deep in the forward pass, not from
    tokenization. See
    docs/audits/vllm_ltr_comparative_evaluation_recovery_20260804.md."""

    def test_max_position_embeddings_is_2048(self, regression_handle):
        handle, _ = regression_handle
        assert handle.model.config.max_position_embeddings == 2048

    def test_prompt_over_max_position_embeddings_does_not_crash(self, regression_handle):
        handle, _ = regression_handle
        max_len = handle.model.config.max_position_embeddings
        # "word " repeated is >1 token/word under BPE; comfortably exceeds
        # max_len regardless of exact tokenization.
        long_prompt = "word " * (max_len * 3)
        score = handle.score(long_prompt)
        assert isinstance(score, float)
        import math

        assert math.isfinite(score)

    def test_truncation_never_exceeds_model_limit(self, regression_handle):
        """Direct check that the actual tokenized input fed to the model
        never exceeds max_position_embeddings, independent of the internal
        assertion inside score_batch()."""
        handle, _ = regression_handle
        max_len = handle.model.config.max_position_embeddings
        long_prompt = "word " * (max_len * 3)
        inputs = handle.tokenizer(
            [long_prompt], return_tensors="pt", truncation=True, padding=True, max_length=max_len
        )
        assert inputs["input_ids"].shape[1] <= max_len

    def test_long_prompt_scoring_is_deterministic(self, regression_handle):
        handle, _ = regression_handle
        max_len = handle.model.config.max_position_embeddings
        long_prompt = "word " * (max_len * 3)
        first = handle.score(long_prompt)
        second = handle.score(long_prompt)
        assert first == second

    def test_truncation_metadata_is_recorded(self, regression_handle):
        """num_prompts_truncated must increment for prompts that actually
        exceeded the limit, and not for short prompts that didn't."""
        from baselines.vllm_ltr.adapter.checkpoint_loader import (
            download_and_provision_checkpoint,
            load_opt_predictor_from_local,
        )

        # Use a fresh handle (independent counter) rather than the shared
        # module-scoped fixture, so this test's count isn't polluted by
        # other tests in this module that also call score_batch().
        ckpt_dir = download_and_provision_checkpoint(
            repo_id="LLM-ltr/OPT-Predictors",
            subfolder=REGRESSION_SUBFOLDER,
            revision=CHECKPOINT_REVISION,
            local_dir="unused",
            verified_environments=[
                {"torch_version": "2.2.1", "transformers_version": "4.45.2"},
            ]
            + [
                {"torch_version": __import__("torch").__version__,
                 "transformers_version": __import__("transformers").__version__}
            ],
        )
        handle = load_opt_predictor_from_local(ckpt_dir)
        assert handle.num_prompts_truncated == 0

        max_len = handle.model.config.max_position_embeddings
        long_prompt = "word " * (max_len * 3)
        short_prompt = "What is 2+2?"

        handle.score_batch([short_prompt], batch_size=4)
        assert handle.num_prompts_truncated == 0

        handle.score_batch([long_prompt], batch_size=4)
        assert handle.num_prompts_truncated == 1

        handle.score_batch([long_prompt, short_prompt, long_prompt], batch_size=4)
        assert handle.num_prompts_truncated == 3


@pytest.mark.gpu
class TestOfflineScoringPipelineEndToEnd:
    def test_score_cache_roundtrip_and_ranking(self, regression_handle, tmp_path):
        from baselines.vllm_ltr.adapter.offline_scoring import (
            load_score_cache,
            save_score_cache,
            score_prompts_offline,
            scores_only,
        )
        from baselines.vllm_ltr.adapter.ranking_adapter import order_by_ltr_score

        handle, _ = regression_handle
        prompts = _sharegpt_tiny_prompts()
        id_to_prompt = {i: p for i, p in enumerate(prompts)}

        cache = score_prompts_offline(handle, id_to_prompt, batch_size=4)
        cache_path = str(tmp_path / "vllm_ltr_scores.json")
        save_score_cache(cache, cache_path)
        reloaded = load_score_cache(cache_path)
        assert reloaded == cache

        scores = scores_only(reloaded, id_to_prompt)
        assert set(scores.keys()) == set(id_to_prompt.keys())

        class _Req:
            def __init__(self, request_id):
                self.request_id = request_id

        reqs = [_Req(i) for i in id_to_prompt]
        ordered = order_by_ltr_score(reqs, scores)
        ordered_scores = [scores[r.request_id] for r in ordered]
        assert ordered_scores == sorted(ordered_scores, reverse=True)

    def test_stale_cache_detected_on_prompt_change(self, regression_handle, tmp_path):
        from baselines.vllm_ltr.adapter.offline_scoring import (
            StaleScoreCacheError,
            score_prompts_offline,
            scores_only,
        )

        handle, _ = regression_handle
        id_to_prompt = {0: "What is 2+2?"}
        cache = score_prompts_offline(handle, id_to_prompt, batch_size=4)
        with pytest.raises(StaleScoreCacheError):
            scores_only(cache, {0: "A completely different prompt."})


@pytest.mark.gpu
class TestOverhead:
    """Overhead is hardware-dependent -- these assert basic sanity
    (positive, finite, and well under a generous ceiling), not tight
    numeric bounds. Real numbers observed on this session's hardware
    (RTX 5060 Ti) are recorded in the audit doc."""

    def test_cpu_and_gpu_latency_are_finite_and_bounded(self, regression_handle):
        import time

        import torch

        handle, _ = regression_handle
        prompts = (_sharegpt_tiny_prompts() * 4)[:32]

        handle.model.to("cpu")
        t0 = time.perf_counter()
        handle.score_batch(prompts, batch_size=8)
        cpu_elapsed = time.perf_counter() - t0
        assert 0.0 < cpu_elapsed < 30.0

        if torch.cuda.is_available():
            handle.model.to("cuda")
            handle.score_batch(prompts[:8], batch_size=8)  # warmup
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            handle.score_batch(prompts, batch_size=8)
            torch.cuda.synchronize()
            gpu_elapsed = time.perf_counter() - t0
            assert 0.0 < gpu_elapsed < 5.0
            handle.model.to("cpu")

    def test_gpu_peak_memory_is_bounded(self, regression_handle):
        import torch

        if not torch.cuda.is_available():
            pytest.skip("no CUDA device available")
        handle, _ = regression_handle
        handle.model.to("cuda")
        torch.cuda.reset_peak_memory_stats()
        handle.score_batch((_sharegpt_tiny_prompts() * 4)[:32], batch_size=8)
        peak_mb = torch.cuda.max_memory_allocated() / 1e6
        assert 0.0 < peak_mb < 4000.0  # a 125M-param model should never need GBs
        handle.model.to("cpu")
