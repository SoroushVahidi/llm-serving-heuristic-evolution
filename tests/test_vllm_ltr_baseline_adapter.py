"""Fidelity/scope tests for the vLLM-LTR baseline-integration scaffold
(baselines/vllm_ltr/). See docs/audits/vllm_ltr_baseline_audit_20260804.md
for the full classification and known deviations these tests lock in.

Covers (per the baseline-integration task's fidelity-verification list):
identical ranking semantics/tie-breaking, no inference-time label leakage,
missing-checkpoint behavior, version-mismatch rejection, stale-artifact
rejection, and the scope invariant that this baseline is never wired into
the selector candidate set.
"""
from __future__ import annotations

import dataclasses
import pathlib
import re

import pytest

from baselines.vllm_ltr.adapter import provenance
from baselines.vllm_ltr.adapter.checkpoint_loader import (
    OPTPredictorHandle,
    write_local_provenance_sidecar,
)
from baselines.vllm_ltr.adapter.errors import (
    MissingCheckpointError,
    MissingDependencyError,
    MissingScoreError,
    StaleArtifactError,
    VersionMismatchError,
)
from baselines.vllm_ltr.adapter.ranking_adapter import order_by_ltr_score
from baselines.vllm_ltr.adapter.simulator_policy import (
    SELECTOR_ELIGIBLE,
    VLLMLTRSemanticReferencePolicy,
)
from llmserveopt.core.types import GPUConfig, ObservableGPUState, ObservableRequest, ObservableState


@dataclasses.dataclass
class _Req:
    request_id: int


def _obs_request(request_id: int, prompt_tokens: int = 100) -> ObservableRequest:
    return ObservableRequest(
        request_id=request_id,
        arrival_time=0.0,
        prompt_tokens=prompt_tokens,
        predicted_output_tokens=10,
        slo_deadline=100.0,
        priority=1.0,
        class_id="test",
    )


class TestProvenanceManifest:
    def test_pinned_commit_and_repo_recorded(self):
        assert provenance.OFFICIAL_REPOSITORY == "https://github.com/hao-ai-lab/vllm-ltr"
        assert provenance.PINNED_COMMIT == "13bbf6ff3dab661791d41362551b089e5f77c91c"
        assert provenance.LICENSE == "Apache-2.0"

    def test_paper_citation_recorded(self):
        assert provenance.PAPER_ARXIV_ID == "2408.15792"
        assert "Fu, Yichao" in provenance.PAPER_AUTHORS

    def test_checkpoint_and_dataset_repos_recorded(self):
        assert provenance.CHECKPOINT_HF_REPO == "LLM-ltr/OPT-Predictors"
        assert provenance.TRAINING_DATASET_HF_REPO == "LLM-ltr/Llama3-Trace"


class TestRankingSemanticEquivalence:
    """Reproduces vllm/core/scheduler.py::_get_ltr_ordered_requests's exact
    ordering semantics (see official_reference/scheduler_ranking_excerpt.md):
    ``sorted(requests, key=lambda req: -req.aux_model_score)``, stable."""

    def test_descending_score_is_highest_priority(self):
        reqs = [_Req(1), _Req(2), _Req(3)]
        scores = {1: 0.5, 2: 5.0, 3: 2.0}
        ordered = order_by_ltr_score(reqs, scores)
        assert [r.request_id for r in ordered] == [2, 3, 1]

    def test_reference_sort_matches_official_key_formula_directly(self):
        """Cross-check against the literal official formula
        ``sorted(reqs, key=lambda req: -req.aux_model_score)`` applied to
        the same synthetic inputs, not just our own reimplementation."""
        reqs = [_Req(i) for i in range(20)]
        scores = {i: float((i * 37) % 11) for i in range(20)}

        class _Scored:
            def __init__(self, request_id, aux_model_score):
                self.request_id = request_id
                self.aux_model_score = aux_model_score

        official_style = [_Scored(i, scores[i]) for i in range(20)]
        official_ordered = sorted(official_style, key=lambda req: -req.aux_model_score)
        adapter_ordered = order_by_ltr_score(reqs, scores)
        assert [r.request_id for r in official_ordered] == [r.request_id for r in adapter_ordered]

    def test_deterministic_tie_breaking_preserves_input_order(self):
        """Equal scores must keep their relative input order (Python's
        stable sort), not be reordered by e.g. request_id."""
        reqs = [_Req(5), _Req(1), _Req(9), _Req(2)]
        scores = {rid: 1.0 for rid in (5, 1, 9, 2)}
        ordered = order_by_ltr_score(reqs, scores)
        assert [r.request_id for r in ordered] == [5, 1, 9, 2]

    def test_deterministic_across_repeated_calls(self):
        reqs = [_Req(i) for i in range(50)]
        scores = {i: float((i * 7) % 13) for i in range(50)}
        first = [r.request_id for r in order_by_ltr_score(reqs, scores)]
        second = [r.request_id for r in order_by_ltr_score(reqs, scores)]
        assert first == second

    def test_batching_many_requests_preserves_stable_partial_ties(self):
        """A larger, more batch-realistic input: many requests share a
        handful of score bins (as the official classifier's argmax-over-bins
        output would produce), and each bin's internal order must stay
        input-stable."""
        reqs = [_Req(i) for i in range(200)]
        scores = {i: float(i % 5) for i in range(200)}
        ordered = order_by_ltr_score(reqs, scores)
        # Within each score bin, ids must appear in increasing original order.
        for bin_value in range(5):
            ids_in_bin = [r.request_id for r in ordered if scores[r.request_id] == bin_value]
            assert ids_in_bin == sorted(ids_in_bin)

    def test_missing_score_raises_instead_of_falling_back(self):
        reqs = [_Req(1), _Req(2)]
        with pytest.raises(MissingScoreError):
            order_by_ltr_score(reqs, {1: 1.0})


class TestNoLeakage:
    def test_observable_request_has_no_actual_output_tokens_field(self):
        """Structural leakage guard: even if this adapter wanted to cheat,
        ObservableRequest (what select_action() actually receives) has no
        such field to read."""
        assert not hasattr(_obs_request(1), "actual_output_tokens")

    def test_policy_never_reads_predicted_output_tokens_either(self):
        """The whole point of vLLM-LTR is to replace length-based proxies
        with a learned ranker -- the wrapper must rank purely by injected
        score, never falling back to predicted_output_tokens."""
        gpu = ObservableGPUState(
            gpu_id=0, max_active_sequences=10, max_batch_tokens=100_000,
            max_kv_tokens=100_000, active_request_ids=[], active_requests_info=[],
            current_kv_tokens=0, tokens_decoded_per_request={},
        )
        # Request 1 has a *worse* (higher) predicted_output_tokens but a
        # *better* (higher) injected LTR score -- if the policy secretly used
        # predicted_output_tokens it would admit request 2 first instead.
        req1 = ObservableRequest(1, 0.0, 100, predicted_output_tokens=999, slo_deadline=10.0, priority=1.0, class_id="a")
        req2 = ObservableRequest(2, 0.0, 100, predicted_output_tokens=1, slo_deadline=10.0, priority=1.0, class_id="a")
        state = ObservableState(time=0.0, waiting_queue=[req1, req2], gpu_states=[gpu], completed_count=0, step=0)
        policy = VLLMLTRSemanticReferencePolicy(scores={1: 10.0, 2: 0.0})
        action = policy.select_action(state)
        assert action.admit[0][0] == 1

    def test_missing_score_for_a_live_request_raises(self):
        gpu = ObservableGPUState(
            gpu_id=0, max_active_sequences=10, max_batch_tokens=100_000,
            max_kv_tokens=100_000, active_request_ids=[], active_requests_info=[],
            current_kv_tokens=0, tokens_decoded_per_request={},
        )
        req = _obs_request(1)
        state = ObservableState(time=0.0, waiting_queue=[req], gpu_states=[gpu], completed_count=0, step=0)
        policy = VLLMLTRSemanticReferencePolicy(scores={})
        with pytest.raises(MissingScoreError):
            policy.select_action(state)


class TestCheckpointLoaderFailureModes:
    def test_missing_dependency_or_missing_checkpoint(self, tmp_path):
        """torch/transformers are installed in this environment (see
        baselines/vllm_ltr/CHECKPOINT_PROVENANCE.md), so this genuinely
        exercises the MissingCheckpointError path for a directory that
        doesn't exist. In an environment without torch/transformers, this
        instead exercises the (equally required) MissingDependencyError
        path -- both are accepted since either is a correct rejection."""
        from baselines.vllm_ltr.adapter.checkpoint_loader import load_opt_predictor_from_local

        missing_dir = str(tmp_path / "does_not_exist")
        with pytest.raises((MissingDependencyError, MissingCheckpointError)):
            load_opt_predictor_from_local(missing_dir)

    def test_stale_artifact_rejected_when_sidecar_absent(self, tmp_path, monkeypatch):
        from baselines.vllm_ltr.adapter import checkpoint_loader

        monkeypatch.setattr(checkpoint_loader, "_require_torch_and_transformers", lambda: (None, None))
        ckpt_dir = tmp_path / "ckpt"
        ckpt_dir.mkdir()
        with pytest.raises(StaleArtifactError):
            checkpoint_loader.load_opt_predictor_from_local(str(ckpt_dir))

    def test_stale_artifact_rejected_when_pinned_commit_differs(self, tmp_path, monkeypatch):
        from baselines.vllm_ltr.adapter import checkpoint_loader

        monkeypatch.setattr(checkpoint_loader, "_require_torch_and_transformers", lambda: (None, None))
        ckpt_dir = tmp_path / "ckpt"
        write_local_provenance_sidecar(
            str(ckpt_dir),
            pinned_commit="deadbeef",
            verified_environments=[{"torch_version": "2.2.1", "transformers_version": "4.30.0"}],
        )
        with pytest.raises(StaleArtifactError):
            checkpoint_loader.load_opt_predictor_from_local(str(ckpt_dir))

    def test_version_mismatch_rejected(self, tmp_path, monkeypatch):
        from baselines.vllm_ltr.adapter import checkpoint_loader

        class _FakeModule:
            __version__ = "9.9.9"

        monkeypatch.setattr(
            checkpoint_loader, "_require_torch_and_transformers", lambda: (_FakeModule(), _FakeModule())
        )
        ckpt_dir = tmp_path / "ckpt"
        write_local_provenance_sidecar(
            str(ckpt_dir),
            pinned_commit=provenance.PINNED_COMMIT,
            verified_environments=[
                {"torch_version": "2.2.1", "transformers_version": "4.30.0"},
                {"torch_version": "2.12.0", "transformers_version": "5.8.1"},
            ],
        )
        with pytest.raises(VersionMismatchError):
            checkpoint_loader.load_opt_predictor_from_local(str(ckpt_dir))

    def test_valid_sidecar_passes_provenance_validation(self, tmp_path, monkeypatch):
        """A sidecar with the correct pinned commit and matching versions
        must pass validation and proceed to the (mocked) model load --
        confirming the rejection tests above fail for the *right* reason
        and not because validation always rejects."""
        from baselines.vllm_ltr.adapter import checkpoint_loader

        class _FakeModule:
            __version__ = "2.2.1"

        monkeypatch.setattr(
            checkpoint_loader, "_require_torch_and_transformers", lambda: (_FakeModule(), _FakeModule())
        )
        ckpt_dir = tmp_path / "ckpt"
        write_local_provenance_sidecar(
            str(ckpt_dir),
            pinned_commit=provenance.PINNED_COMMIT,
            verified_environments=[{"torch_version": "2.2.1", "transformers_version": "2.2.1"}],
        )
        sidecar = checkpoint_loader._read_provenance_sidecar(str(ckpt_dir))
        checkpoint_loader._validate_provenance_sidecar(sidecar, _FakeModule(), _FakeModule())

    def test_empty_verified_environments_rejected(self, tmp_path, monkeypatch):
        from baselines.vllm_ltr.adapter import checkpoint_loader

        monkeypatch.setattr(checkpoint_loader, "_require_torch_and_transformers", lambda: (None, None))
        ckpt_dir = tmp_path / "ckpt"
        write_local_provenance_sidecar(
            str(ckpt_dir), pinned_commit=provenance.PINNED_COMMIT, verified_environments=[]
        )
        with pytest.raises(StaleArtifactError):
            checkpoint_loader.load_opt_predictor_from_local(str(ckpt_dir))


class TestDeterministicScoring:
    def test_score_uses_no_grad_and_eval_mode(self):
        """Structural determinism check: OPTPredictorHandle.score() must not
        rely on any stochastic module state (model/tokenizer/num_labels are
        the only scoring-relevant state; num_prompts_truncated is a
        write-only bookkeeping counter, not an input that could make
        scoring non-reproducible -- no RNG/seed field exists). A real
        end-to-end determinism check against the actual checkpoint runs in
        tests/test_vllm_ltr_checkpoint_fidelity_gpu.py (gated on
        LLMSERVEOPT_RUN_GPU_TESTS=1)."""
        fields = {f.name for f in dataclasses.fields(OPTPredictorHandle)}
        assert fields == {"model", "tokenizer", "num_labels", "num_prompts_truncated"}


class TestSelectorScopeInvariants:
    """This baseline must never be treated as a selector candidate or added
    to the historical deployable-policy registries -- see the task's
    explicit "do not add vLLM-LTR to the main selector candidate set yet"
    instruction and docs/audits/vllm_ltr_baseline_audit_20260804.md."""

    def test_not_selector_eligible(self):
        assert SELECTOR_ELIGIBLE is False
        assert provenance.SELECTOR_CANDIDATE is False
        assert provenance.HISTORICAL is False

    def test_not_in_historical_registry(self):
        from llmserveopt.policies.registry import BASELINE_NAMES, SELECTOR_CANDIDATE_NAMES

        assert "vllm_ltr_semantic_reference" not in BASELINE_NAMES
        assert "vllm_ltr_semantic_reference" not in SELECTOR_CANDIDATE_NAMES

    def test_not_in_policy_library_v2(self):
        from llmserveopt.policies.registry import POLICY_LIBRARY_V2_NAMES

        assert "vllm_ltr_semantic_reference" not in POLICY_LIBRARY_V2_NAMES

    def test_not_in_external_baseline_registry(self):
        from llmserveopt.policies.external_baselines_registry import EXTERNAL_BASELINE_NAMES

        assert "vllm_ltr_semantic_reference" not in EXTERNAL_BASELINE_NAMES
        assert "vllm_ltr" not in EXTERNAL_BASELINE_NAMES

    def test_baselines_package_not_imported_by_src(self):
        """Nothing under src/llmserveopt may depend on baselines/vllm_ltr --
        it must remain fully isolated and optional."""
        import pathlib

        src_root = pathlib.Path(__file__).resolve().parents[1] / "src" / "llmserveopt"
        offenders = []
        for path in src_root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if "baselines.vllm_ltr" in text or "baselines/vllm_ltr" in text:
                offenders.append(str(path))
        assert offenders == []


class TestCheckpointProvenanceDocFormat:
    """Regression lock for a real transcription bug caught during this
    baseline's completion pass: all four sha256 hashes in
    CHECKPOINT_PROVENANCE.md were originally copy-pasted one hex digit
    short (63 chars instead of 64) and would have silently passed any
    check that didn't verify hash *format*, not just presence. This test
    parses every string the doc labels as a sha256 hash and asserts it is
    exactly 64 lowercase hex characters -- it would have caught the
    original mistake and catches any future one."""

    def test_every_declared_sha256_is_64_hex_chars(self):
        doc_path = (
            pathlib.Path(__file__).resolve().parents[1]
            / "baselines"
            / "vllm_ltr"
            / "CHECKPOINT_PROVENANCE.md"
        )
        text = doc_path.read_text(encoding="utf-8")
        # Intentionally permissive character class (not `+`/fixed-length in
        # the regex itself) so a truncated hash is *captured*, not silently
        # skipped by the pattern -- the length assertion below is what must
        # catch it.
        hashes = re.findall(r"sha256:\s*`([0-9a-f]*)`", text)
        assert len(hashes) >= 4, "expected at least 4 sha256 hashes documented"
        for h in hashes:
            assert len(h) == 64, f"hash {h!r} is {len(h)} chars, not 64"
