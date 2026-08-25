"""Tests for scripts/score_vllm_ltr_eval_dataset.py's provenance/resumability
logic. Checkpoint download + model forward passes are exercised only by the
manual pipeline run (see
docs/audits/vllm_ltr_first_comparative_evaluation_20260804.md) and by the
existing GPU-gated tests in tests/test_vllm_ltr_checkpoint_fidelity_gpu.py
(``LLMSERVEOPT_RUN_GPU_TESTS=1``) -- this file covers only the
network-free, checkpoint-free provenance/cache-management functions.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "score_vllm_ltr_eval_dataset",
    Path(__file__).parent.parent / "scripts" / "score_vllm_ltr_eval_dataset.py",
)
scorer = importlib.util.module_from_spec(_SPEC)
sys.modules["score_vllm_ltr_eval_dataset"] = scorer
_SPEC.loader.exec_module(scorer)

from baselines.vllm_ltr.adapter.offline_scoring import (
    StaleScoreCacheError,
    load_score_cache,
    save_score_cache,
    scores_only,
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class TestProvenanceCompatibilityCheck:
    def _prov(self, **overrides):
        base = {
            "checkpoint_repo_id": "LLM-ltr/OPT-Predictors",
            "checkpoint_revision": "39df2b41ffe88d5ed967c6035d3838b5b5960379",
            "checkpoint_subfolder": "opt-125m-llama3-8b-sharegpt-score-trainbucket10-b32",
            "checkpoint_file_hashes": {"model.safetensors": "abc123"},
        }
        base.update(overrides)
        return base

    def test_identical_provenance_passes(self):
        recorded = self._prov()
        current = self._prov()
        scorer._check_provenance_compatible(recorded, current)  # no raise

    def test_different_revision_rejected(self):
        recorded = self._prov()
        current = self._prov(checkpoint_revision="deadbeef")
        with pytest.raises(scorer.ScoreCacheProvenanceMismatchError):
            scorer._check_provenance_compatible(recorded, current)

    def test_different_subfolder_variant_rejected(self):
        recorded = self._prov()
        current = self._prov(checkpoint_subfolder="opt-125m-llama3-8b-sharegpt-class-trainbucket820-b32")
        with pytest.raises(scorer.ScoreCacheProvenanceMismatchError):
            scorer._check_provenance_compatible(recorded, current)

    def test_different_file_hash_rejected(self):
        recorded = self._prov()
        current = self._prov(checkpoint_file_hashes={"model.safetensors": "different"})
        with pytest.raises(scorer.ScoreCacheProvenanceMismatchError):
            scorer._check_provenance_compatible(recorded, current)

    def test_extra_unrelated_fields_do_not_affect_comparison(self):
        recorded = self._prov()
        current = self._prov()
        current["device"] = "cuda:0"  # not one of the compared keys
        recorded["device"] = "cpu"
        scorer._check_provenance_compatible(recorded, current)  # no raise


class TestStaleScoreCacheRejection:
    def test_hash_mismatch_between_cache_and_current_prompt_raises(self, tmp_path):
        cache = {1: {"score": 0.5, "prompt_sha256": _sha("original prompt")}}
        with pytest.raises(StaleScoreCacheError):
            scores_only(cache, id_to_prompt={1: "a DIFFERENT prompt now"})

    def test_matching_hash_does_not_raise(self, tmp_path):
        cache = {1: {"score": 0.5, "prompt_sha256": _sha("same prompt")}}
        result = scores_only(cache, id_to_prompt={1: "same prompt"})
        assert result == {1: 0.5}


class TestResumabilityMergeSemantics:
    def test_missing_ids_computed_only_existing_ids_kept(self, tmp_path):
        """Mirrors the resume logic in score_vllm_ltr_eval_dataset.py::main:
        merged = {**existing_cache, **new_scores} should preserve existing
        entries untouched and only add newly scored ones."""
        existing_cache = {1: {"score": 0.9, "prompt_sha256": _sha("p1")}}
        id_to_prompt = {1: "p1", 2: "p2", 3: "p3"}
        missing_ids = {rid: p for rid, p in id_to_prompt.items() if rid not in existing_cache}
        assert set(missing_ids.keys()) == {2, 3}

        new_scores = {
            2: {"score": 0.1, "prompt_sha256": _sha("p2")},
            3: {"score": 0.2, "prompt_sha256": _sha("p3")},
        }
        merged = {**existing_cache, **new_scores}
        assert merged[1] == existing_cache[1]
        assert merged[2] == new_scores[2]
        assert merged[3] == new_scores[3]
        assert len(merged) == 3

    def test_cache_round_trips_through_disk(self, tmp_path):
        cache = {5: {"score": 1.23, "prompt_sha256": _sha("hello")}, 7: {"score": -0.5, "prompt_sha256": _sha("world")}}
        path = str(tmp_path / "cache.json")
        save_score_cache(cache, path)
        reloaded = load_score_cache(path)
        assert reloaded == cache

    def test_resuming_without_provenance_sidecar_is_rejected_by_main_contract(self, tmp_path):
        """The provenance sidecar path convention used by main(): a cache
        file with no sibling `<path>.provenance.json` must not be silently
        resumed -- verified here at the path-naming level (the full
        rejection is exercised by main(), which requires a live checkpoint
        download and is therefore covered by the manual pipeline run, not
        this fast unit test)."""
        cache_path = tmp_path / "cache.json"
        cache_path.write_text("{}")
        prov_path = Path(scorer._provenance_path(str(cache_path)))
        assert not prov_path.exists()
        assert prov_path.name == "cache.json.provenance.json"
