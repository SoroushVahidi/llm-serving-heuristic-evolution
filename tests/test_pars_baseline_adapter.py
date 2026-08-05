"""Fidelity/scope tests for the PARS baseline-integration
(baselines/pars/). See baselines/pars/PROVENANCE.md and
docs/audits/pars_baseline_implementation_20260804.md for the full
provenance record and known deviations these tests lock in. Mirrors
tests/test_vllm_ltr_baseline_adapter.py's structure and coverage.
"""
from __future__ import annotations

import dataclasses
import os

import pytest

from baselines.pars.adapter import provenance
from baselines.pars.adapter.errors import (
    MissingCheckpointError,
    MissingOfficialCloneError,
    MissingScoreError,
    StaleCloneCommitError,
)
from baselines.pars.adapter.ranking_adapter import order_by_pars_score
from baselines.pars.adapter.simulator_policy import (
    SELECTOR_ELIGIBLE,
    PARSSemanticReferencePolicy,
)


@dataclasses.dataclass
class _Req:
    request_id: int


class TestProvenanceManifest:
    def test_pinned_commit_and_repo_recorded(self):
        assert provenance.OFFICIAL_REPOSITORY == "https://github.com/SPEAR-UIC/PARS"
        assert provenance.PINNED_COMMIT == "fd4e125b65bb73aef5eccafa79c2509434be61ec"

    def test_license_gap_explicitly_recorded_not_hidden(self):
        """The official repo has NO license file -- this must be recorded
        explicitly as a known gap, never silently omitted or claimed as a
        permissive license it doesn't have."""
        assert "NONE" in provenance.LICENSE.upper()
        assert "unlicensed" in provenance.LICENSE.lower()

    def test_paper_citation_recorded(self):
        assert provenance.PAPER_ARXIV_ID == "2510.03243"
        assert "Tao, Yiheng" in provenance.PAPER_AUTHORS
        assert "Lan, Zhiling" in provenance.PAPER_AUTHORS

    def test_training_dataset_and_license_recorded(self):
        assert provenance.TRAINING_DATASET_HF_REPO == "vicgalle/alpaca-gpt4"
        assert "NC" in provenance.TRAINING_DATASET_LICENSE

    def test_official_training_hyperparameters_are_unmodified_defaults(self):
        """Regression lock: these must match train_pairwise_bert.py's own
        argparse defaults exactly -- this baseline never overrides the
        official hyperparameters."""
        assert provenance.MODEL_NAME == "bert-base-uncased"
        assert provenance.MAX_LENGTH == 128
        assert provenance.BATCH_SIZE == 128
        assert provenance.NUM_EPOCHS == 3
        assert provenance.LEARNING_RATE == 2e-5
        assert provenance.MARGIN == 1.0


class TestRankingSemanticEquivalence:
    """PARS predicts LONGER response = HIGHER score (verified directly from
    the official MarginRankingLoss construction -- see
    provenance.HIGHER_SCORE_MEANS_LONGER_PREDICTED_RESPONSE's docstring).
    For SJF-style scheduling, ascending score = highest priority -- the
    mirror image of vLLM-LTR's descending-score rule."""

    def test_ascending_score_is_highest_priority(self):
        reqs = [_Req(1), _Req(2), _Req(3)]
        scores = {1: 0.5, 2: 5.0, 3: 2.0}  # lower score = shorter predicted response = admit first
        ordered = order_by_pars_score(reqs, scores)
        assert [r.request_id for r in ordered] == [1, 3, 2]

    def test_opposite_direction_from_vllm_ltr(self):
        """Direct, explicit cross-check against vLLM-LTR's descending rule
        on the identical input -- the two adapters must disagree on order
        whenever scores are non-uniform, since one ranks ascending and the
        other descending."""
        from baselines.vllm_ltr.adapter.ranking_adapter import order_by_ltr_score

        reqs = [_Req(1), _Req(2), _Req(3)]
        scores = {1: 0.5, 2: 5.0, 3: 2.0}
        pars_order = [r.request_id for r in order_by_pars_score(reqs, scores)]
        ltr_order = [r.request_id for r in order_by_ltr_score(reqs, scores)]
        assert pars_order == list(reversed(ltr_order))

    def test_deterministic_tie_breaking_preserves_input_order(self):
        reqs = [_Req(5), _Req(1), _Req(9), _Req(2)]
        scores = {rid: 1.0 for rid in (5, 1, 9, 2)}
        ordered = order_by_pars_score(reqs, scores)
        assert [r.request_id for r in ordered] == [5, 1, 9, 2]

    def test_deterministic_across_repeated_calls(self):
        reqs = [_Req(i) for i in range(50)]
        scores = {i: float((i * 7) % 13) for i in range(50)}
        first = [r.request_id for r in order_by_pars_score(reqs, scores)]
        second = [r.request_id for r in order_by_pars_score(reqs, scores)]
        assert first == second

    def test_missing_score_raises_instead_of_falling_back(self):
        reqs = [_Req(1), _Req(2)]
        with pytest.raises(MissingScoreError):
            order_by_pars_score(reqs, {1: 1.0})


class TestNoLeakage:
    def test_observable_request_has_no_actual_output_tokens_field(self):
        from llmserveopt.core.types import ObservableRequest

        fields = {f.name for f in dataclasses.fields(ObservableRequest)}
        assert "actual_output_tokens" not in fields

    def test_policy_select_action_never_receives_actual_output_tokens(self):
        """PARSSemanticReferencePolicy's select_action only ever reads
        state.waiting_queue (ObservableRequest, no actual_output_tokens)
        and the precomputed score map -- structural guarantee, not just an
        assertion about behavior."""
        import inspect

        source = inspect.getsource(PARSSemanticReferencePolicy.select_action)
        assert "actual_output_tokens" not in source


class TestCheckpointLoaderFailureModes:
    def test_missing_official_clone_raises(self, tmp_path):
        from baselines.pars.adapter.checkpoint_loader import verify_official_clone

        with pytest.raises(MissingOfficialCloneError):
            verify_official_clone(str(tmp_path / "nonexistent_clone"))

    def test_stale_clone_commit_raises(self, tmp_path):
        import subprocess

        from baselines.pars.adapter.checkpoint_loader import verify_official_clone

        clone_dir = tmp_path / "clone"
        clone_dir.mkdir(parents=True)
        env = {
            **os.environ,
            "GIT_AUTHOR_NAME": "test", "GIT_AUTHOR_EMAIL": "test@example.com",
            "GIT_COMMITTER_NAME": "test", "GIT_COMMITTER_EMAIL": "test@example.com",
        }
        subprocess.run(["git", "init", "-q"], cwd=clone_dir, env=env, check=True)
        subprocess.run(["git", "commit", "--allow-empty", "-q", "-m", "init"], cwd=clone_dir, env=env, check=True)
        with pytest.raises(StaleCloneCommitError):
            verify_official_clone(str(clone_dir))

    def test_missing_checkpoint_raises(self, tmp_path):
        pytest.importorskip("torch")
        from baselines.pars.adapter.checkpoint_loader import load_pars_predictor

        with pytest.raises(MissingCheckpointError):
            load_pars_predictor(str(tmp_path / "nonexistent.pt"))


class TestSelectorScopeInvariants:
    """This baseline must never be treated as a selector candidate or added
    to the historical deployable-policy registries."""

    def test_not_selector_eligible(self):
        assert SELECTOR_ELIGIBLE is False
        assert provenance.SELECTOR_CANDIDATE is False
        assert provenance.HISTORICAL is False

    def test_not_in_historical_registry(self):
        from llmserveopt.policies.registry import BASELINE_NAMES, SELECTOR_CANDIDATE_NAMES

        assert "pars_semantic_reference" not in BASELINE_NAMES
        assert "pars_semantic_reference" not in SELECTOR_CANDIDATE_NAMES

    def test_not_in_policy_library_v2(self):
        from llmserveopt.policies.registry import POLICY_LIBRARY_V2_NAMES

        assert "pars_semantic_reference" not in POLICY_LIBRARY_V2_NAMES

    def test_not_in_external_baseline_registry(self):
        from llmserveopt.policies.external_baselines_registry import EXTERNAL_BASELINE_NAMES

        assert "pars_semantic_reference" not in EXTERNAL_BASELINE_NAMES
        assert "pars" not in EXTERNAL_BASELINE_NAMES

    def test_baselines_package_not_imported_by_src(self):
        """Nothing under src/llmserveopt may depend on baselines/pars -- it
        must remain fully isolated and optional."""
        import pathlib

        src_root = pathlib.Path(__file__).resolve().parents[1] / "src" / "llmserveopt"
        offenders = []
        for path in src_root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if "baselines.pars" in text or "baselines/pars" in text:
                offenders.append(str(path))
        assert offenders == []


class TestOfficialCodeReuseNotVendored:
    """The official repository's source must never be committed into this
    project -- see PROVENANCE.md's License section."""

    def test_no_pairwise_ranker_class_definition_duplicated_in_this_repo(self):
        """This project must dynamically import PairwiseRanker from the
        external clone (see checkpoint_loader._import_pairwise_ranker_class),
        never paste its own copy of the class body."""
        import pathlib

        baselines_pars = pathlib.Path(__file__).resolve().parents[1] / "baselines" / "pars"
        offenders = []
        for path in baselines_pars.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if "class PairwiseRanker" in text:
                offenders.append(str(path))
        assert offenders == []

    def test_official_clone_path_is_outside_this_repo(self):
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        assert not provenance.DEFAULT_OFFICIAL_CLONE_PATH.startswith(repo_root)
