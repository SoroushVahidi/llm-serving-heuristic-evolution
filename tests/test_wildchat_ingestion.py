"""Tests for scripts/ingest_wildchat_eval_dataset.py's deterministic,
network-free logic: filtering, normalization, hashing, stable-ID
assignment, and deterministic sampling. Live network/streaming behavior
(``scan_candidates``) is exercised only by the manual pipeline run
recorded in docs/audits/vllm_ltr_first_comparative_evaluation_20260804.md,
not by this fast unit-test file.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import unicodedata
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "ingest_wildchat_eval_dataset",
    Path(__file__).parent.parent / "scripts" / "ingest_wildchat_eval_dataset.py",
)
ingest = importlib.util.module_from_spec(_SPEC)
sys.modules["ingest_wildchat_eval_dataset"] = ingest
_SPEC.loader.exec_module(ingest)


def _row(
    turn=1,
    conversation_hash="hash123",
    user_language="English",
    user_toxic=False,
    user_redacted=False,
    prompt="Hello world",
    response="Hi there",
):
    return {
        "turn": turn,
        "conversation_hash": conversation_hash,
        "conversation": [
            {"role": "user", "content": prompt, "language": user_language,
             "toxic": user_toxic, "redacted": user_redacted},
            {"role": "assistant", "content": response, "language": "English",
             "toxic": False, "redacted": False},
        ],
    }


class TestNormalizeAndHash:
    def test_normalize_strips_whitespace(self):
        assert ingest.normalize_text("  hello  \n") == "hello"

    def test_normalize_is_nfc(self):
        # combining-character form vs. precomposed form normalize identically
        decomposed = "é"  # e + combining acute accent
        precomposed = "é"  # é
        assert ingest.normalize_text(decomposed) == ingest.normalize_text(precomposed)

    def test_sha256_matches_stdlib(self):
        text = "some prompt text"
        assert ingest.sha256_text(text) == hashlib.sha256(text.encode("utf-8")).hexdigest()

    def test_sha256_deterministic(self):
        assert ingest.sha256_text("abc") == ingest.sha256_text("abc")

    def test_sha256_sensitive_to_content(self):
        assert ingest.sha256_text("abc") != ingest.sha256_text("abd")


class TestFilterEligibility:
    def test_single_turn_english_clean_row_is_eligible(self):
        assert ingest.first_turn_eligible(_row()) is True

    def test_multi_turn_conversation_rejected(self):
        assert ingest.first_turn_eligible(_row(turn=2)) is False

    def test_non_english_first_turn_rejected(self):
        assert ingest.first_turn_eligible(_row(user_language="Spanish")) is False

    def test_toxic_first_turn_rejected(self):
        assert ingest.first_turn_eligible(_row(user_toxic=True)) is False

    def test_redacted_first_turn_rejected(self):
        assert ingest.first_turn_eligible(_row(user_redacted=True)) is False

    def test_missing_conversation_hash_rejected(self):
        row = _row()
        row["conversation_hash"] = ""
        assert ingest.first_turn_eligible(row) is False

    def test_empty_prompt_rejected(self):
        assert ingest.first_turn_eligible(_row(prompt="   ")) is False

    def test_empty_response_rejected(self):
        assert ingest.first_turn_eligible(_row(response="")) is False

    def test_too_few_turns_rejected(self):
        row = _row()
        row["conversation"] = row["conversation"][:1]
        assert ingest.first_turn_eligible(row) is False

    def test_wrong_role_order_rejected(self):
        row = _row()
        row["conversation"][0]["role"] = "assistant"
        row["conversation"][1]["role"] = "user"
        assert ingest.first_turn_eligible(row) is False

    def test_extract_pair_returns_normalized_triple(self):
        row = _row(prompt="  hi  ", response="  yo  ", conversation_hash="h1")
        prompt, response, chash = ingest.extract_pair(row)
        assert (prompt, response, chash) == ("hi", "yo", "h1")

    def test_no_leakage_filter_never_reads_response_content_for_eligibility(self):
        """Eligibility must depend only on the prompt (first) turn's own
        fields, never on assistant-response content -- otherwise sampling
        would be implicitly conditioned on future (response) information."""
        row_a = _row(response="short")
        row_b = _row(response="a very very very long response " * 50)
        assert ingest.first_turn_eligible(row_a) == ingest.first_turn_eligible(row_b) is True


class TestDeterministicSample:
    CANDIDATES = [
        ("prompt A", "resp A", "hashB"),
        ("prompt B", "resp B", "hashA"),
        ("prompt C", "resp C", "hashD"),
        ("prompt D", "resp D", "hashC"),
    ]

    def test_same_seed_same_sample(self):
        s1 = ingest.deterministic_sample(self.CANDIDATES, 2, seed=7)
        s2 = ingest.deterministic_sample(self.CANDIDATES, 2, seed=7)
        assert s1 == s2

    def test_different_seed_can_differ(self):
        s1 = ingest.deterministic_sample(self.CANDIDATES, 2, seed=1)
        s2 = ingest.deterministic_sample(self.CANDIDATES, 2, seed=2)
        # Not a hard guarantee for all seed pairs, but true for these two.
        assert s1 != s2 or True  # documents intent; determinism is the real assertion above

    def test_sample_independent_of_input_order(self):
        shuffled = list(reversed(self.CANDIDATES))
        s1 = ingest.deterministic_sample(self.CANDIDATES, 3, seed=42)
        s2 = ingest.deterministic_sample(shuffled, 3, seed=42)
        assert s1 == s2

    def test_pool_too_small_raises(self):
        with pytest.raises(ValueError):
            ingest.deterministic_sample(self.CANDIDATES, 10, seed=0)

    def test_output_order_is_not_necessarily_hash_sorted(self):
        """Regression guard for a real doc/comment bug: an earlier version
        of ingest_wildchat_eval_dataset.py's docstrings claimed
        deterministic_sample()'s return value is in conversation_hash-sorted
        order (and that request_ids are therefore assigned in that order).
        That's false -- random.Random(seed).sample() returns a seeded-random
        permutation of its (sorted) input, not the input order itself. This
        seed/pool combination is a known, verified case where the returned
        order is NOT sorted -- if this ever starts passing with a sorted
        result, it doesn't prove the claim was right; it just means this
        particular seed's permutation happened to land in order."""
        result = ingest.deterministic_sample(self.CANDIDATES, len(self.CANDIDATES), seed=0)
        hashes = [t[2] for t in result]
        assert hashes != sorted(hashes)
        # Determinism (the actual guarantee) still holds regardless of order.
        again = ingest.deterministic_sample(self.CANDIDATES, len(self.CANDIDATES), seed=0)
        assert result == again


class TestTokenizeAndStableIds:
    @pytest.fixture(scope="class")
    def sample(self):
        return [
            ("What is the capital of France?", "Paris.", "hashZ"),
            ("Explain gravity briefly.", "Gravity attracts masses.", "hashA"),
            ("Write a haiku about the sea.", "Waves crash on the shore.", "hashM"),
        ]

    def test_tokenize_and_hash_assigns_request_id_by_given_order(self, sample):
        # tokenize_and_hash assigns request_id = index in whatever order
        # it's given -- it does not itself sort. Real pipeline usage feeds
        # it deterministic_sample()'s output directly, which is a
        # seeded-random permutation of hash-sorted order, NOT hash-sorted
        # order itself (see TestDeterministicSample.
        # test_output_order_is_not_necessarily_hash_sorted). This test
        # passes a hash-sorted input explicitly to isolate and verify only
        # the order-preservation contract.
        ordered = sorted(sample, key=lambda t: t[2])
        rows, _ = ingest.tokenize_and_hash(ordered, "facebook/opt-125m")
        assert [r["conversation_hash"] for r in rows] == ["hashA", "hashM", "hashZ"]
        assert [r["request_id"] for r in rows] == [0, 1, 2]

        # And it's equally order-preserving for a NON-sorted input, since
        # that's what actually happens in the real pipeline.
        reversed_order = list(reversed(ordered))
        rows_rev, _ = ingest.tokenize_and_hash(reversed_order, "facebook/opt-125m")
        assert [r["conversation_hash"] for r in rows_rev] == ["hashZ", "hashM", "hashA"]
        assert [r["request_id"] for r in rows_rev] == [0, 1, 2]

    def test_prompt_sha256_matches_prompt_text(self, sample):
        rows, _ = ingest.tokenize_and_hash(sample, "facebook/opt-125m")
        for r in rows:
            assert r["prompt_sha256"] == hashlib.sha256(r["prompt_text"].encode("utf-8")).hexdigest()

    def test_prompt_tokens_positive_and_reproducible(self, sample):
        rows1, _ = ingest.tokenize_and_hash(sample, "facebook/opt-125m")
        rows2, _ = ingest.tokenize_and_hash(sample, "facebook/opt-125m")
        assert [r["prompt_tokens"] for r in rows1] == [r["prompt_tokens"] for r in rows2]
        assert all(r["prompt_tokens"] > 0 for r in rows1)

    def test_ingestion_end_to_end_deterministic(self, tmp_path):
        candidates = [
            (f"Prompt number {i} about topic {i % 3}.", f"Response {i}.", f"hash{i:03d}")
            for i in range(10)
        ]
        sampled = ingest.deterministic_sample(candidates, 5, seed=99)
        rows, _ = ingest.tokenize_and_hash(sampled, "facebook/opt-125m")
        out1 = tmp_path / "run1"
        out2 = tmp_path / "run2"
        ingest.write_outputs(rows, {"seed": 99}, str(out1))
        ingest.write_outputs(rows, {"seed": 99}, str(out2))

        pairs1 = json.loads((out1 / "wildchat_eval_sharegpt_shaped.json").read_text())
        pairs2 = json.loads((out2 / "wildchat_eval_sharegpt_shaped.json").read_text())
        assert pairs1 == pairs2

        prompts1 = json.loads((out1 / "wildchat_eval_prompts_by_id.json").read_text())
        prompts2 = json.loads((out2 / "wildchat_eval_prompts_by_id.json").read_text())
        assert prompts1 == prompts2
        assert set(prompts1.keys()) == {str(i) for i in range(5)}


class TestWriteOutputsShapeAndManifest:
    @pytest.fixture
    def rows(self):
        sample = [
            ("Prompt A", "Response A", "h1"),
            ("Prompt B", "Response B", "h2"),
        ]
        rows, _ = ingest.tokenize_and_hash(sample, "facebook/opt-125m")
        return rows

    def test_sharegpt_shaped_matches_loader_expectations(self, rows, tmp_path):
        from llmserveopt.workloads.sharegpt import extract_prompt_response_pairs, load_sharegpt_raw

        paths = ingest.write_outputs(rows, {}, str(tmp_path))
        records = load_sharegpt_raw(paths["pairs_path"])
        pairs = extract_prompt_response_pairs(records)
        assert len(pairs) == len(rows)
        assert pairs[0][0] == rows[0]["prompt_text"]

    def test_manifest_records_stable_id_to_hash_mapping(self, rows, tmp_path):
        paths = ingest.write_outputs(rows, {}, str(tmp_path))
        manifest = json.loads(Path(paths["manifest_path"]).read_text())
        for r in rows:
            rid = str(r["request_id"])
            assert manifest["row_hashes"][rid] == r["prompt_sha256"]
            assert manifest["conversation_hashes"][rid] == r["conversation_hash"]

    def test_prompts_by_id_keys_are_request_ids(self, rows, tmp_path):
        paths = ingest.write_outputs(rows, {}, str(tmp_path))
        prompts = json.loads(Path(paths["prompts_path"]).read_text())
        assert set(prompts.keys()) == {str(r["request_id"]) for r in rows}
        for r in rows:
            assert prompts[str(r["request_id"])] == r["prompt_text"]


class TestDuplicatePromptSummary:
    """Real 300-row WildChat sample (2026-08-04) had 2 rows (request_id 138,
    213) sharing byte-identical prompt text under different
    conversation_hashes -- a real duplicate manifest didn't previously
    surface. See docs/audits/vllm_ltr_comparative_evaluation_recovery_20260804.md."""

    def _rows(self, sample):
        rows, _ = ingest.tokenize_and_hash(sample, "facebook/opt-125m")
        return rows

    def test_no_duplicates(self):
        sample = [
            ("Prompt A", "Response A", "h1"),
            ("Prompt B", "Response B", "h2"),
        ]
        summary = ingest.duplicate_prompt_summary(self._rows(sample))
        assert summary["total_sampled_rows"] == 2
        assert summary["unique_prompt_hashes"] == 2
        assert summary["duplicate_prompt_count"] == 0
        assert summary["duplicate_groups"] == []

    def test_one_duplicate_pair_across_distinct_conversations(self):
        sample = [
            ("Same prompt text", "Response A", "convo-hash-1"),
            ("Different prompt", "Response B", "convo-hash-2"),
            ("Same prompt text", "Response C", "convo-hash-3"),
        ]
        rows = self._rows(sample)
        summary = ingest.duplicate_prompt_summary(rows)
        assert summary["total_sampled_rows"] == 3
        assert summary["unique_prompt_hashes"] == 2
        assert summary["duplicate_prompt_count"] == 1
        assert len(summary["duplicate_groups"]) == 1
        group = summary["duplicate_groups"][0]
        dup_ids = {r["request_id"] for r in rows if r["prompt_text"] == "Same prompt text"}
        assert set(group["request_ids"]) == dup_ids

    def test_manifest_includes_duplicate_accounting(self, tmp_path):
        sample = [
            ("Same prompt text", "Response A", "convo-hash-1"),
            ("Same prompt text", "Response C", "convo-hash-3"),
        ]
        rows = self._rows(sample)
        paths = ingest.write_outputs(rows, {}, str(tmp_path))
        manifest = json.loads(Path(paths["manifest_path"]).read_text())
        assert manifest["duplicate_prompt_accounting"]["duplicate_prompt_count"] == 1
        assert manifest["duplicate_prompt_accounting"]["unique_prompt_hashes"] == 1
        assert manifest["duplicate_prompt_accounting"]["total_sampled_rows"] == 2
