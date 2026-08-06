"""Tests for the Algorithm Stress-Test Library's workload generators and
catalog. See docs/research/algorithm_stress_tests/ for the full design
and validation record.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "scripts" / "stress_tests"))

import generators  # noqa: E402

_CATALOG_PATH = _ROOT / "configs" / "stress_tests" / "algorithm_stress_test_catalog.yaml"


@pytest.fixture(scope="module")
def catalog():
    with open(_CATALOG_PATH) as f:
        return yaml.safe_load(f)


class TestCatalogStructure:
    def test_catalog_parses(self, catalog):
        assert catalog["schema_version"] == 1
        # 22 pre-Sarathi entries + 7 Sarathi-Serve entries added 2026-08-05
        # (docs/audits/sarathi_stress_test_catalog_completion_20260805.md).
        assert len(catalog["stress_tests"]) == 29

    def test_every_entry_has_unique_id(self, catalog):
        ids = [t["stress_test_id"] for t in catalog["stress_tests"]]
        assert len(ids) == len(set(ids))

    def test_every_algorithm_has_target_and_counter(self, catalog):
        from collections import defaultdict
        by_algo = defaultdict(set)
        for t in catalog["stress_tests"]:
            by_algo[t["algorithm_id"]].add(t["test_role"])
        for algo, roles in by_algo.items():
            assert roles == {"TARGET", "COUNTER"}, f"{algo} missing a role: {roles}"

    def test_evidence_class_is_one_of_the_six_allowed(self, catalog):
        allowed = {
            "PROVEN_WORST_CASE", "DOCUMENTED_LIMITATION",
            "PAPER_MOTIVATING_STRESS_CASE", "HYPOTHESIZED_ADVERSARIAL_REGIME",
            "INTERNAL_EMPIRICAL_FINDING",
            # Added 2026-08-05 for the Sarathi-Serve section -- a strictly
            # stronger evidentiary tier than INTERNAL_EMPIRICAL_FINDING,
            # reserved for claims backed by real GPU hardware execution.
            "EXPERIMENTALLY_VALIDATED_ON_REAL_HARDWARE",
        }
        for t in catalog["stress_tests"]:
            assert t["evidence_class"] in allowed, t["stress_test_id"]

    def test_every_entry_forbids_actual_output_tokens(self, catalog):
        for t in catalog["stress_tests"]:
            assert "actual_output_tokens" in t["forbidden_oracle_inputs"], t["stress_test_id"]

    def test_every_entry_has_source_citations(self, catalog):
        for t in catalog["stress_tests"]:
            assert len(t["source_citations"]) >= 1, t["stress_test_id"]


class TestGeneratorsProduceValidRequests:
    @pytest.mark.parametrize("name", [
        n for n in generators.GENERATORS
        if n not in {
            "vllm_ltr_target_predictive_prompt_semantics",
            "vllm_ltr_counter_reasoning_domain_shift",
            "pars_target_alpaca_style_instruction_prompts",
            "pars_counter_reasoning_domain_shift",
            "sarathi_counter_long_context_attention_recompute",
        }
    ])
    def test_generator_produces_sorted_nonempty_requests(self, name):
        reqs = generators.GENERATORS[name](smoke=True)
        assert len(reqs) > 0
        assert reqs == sorted(reqs, key=lambda r: r.arrival_time)
        for r in reqs:
            assert r.arrival_time >= 0
            assert r.prompt_tokens > 0
            assert r.predicted_output_tokens > 0
            assert r.actual_output_tokens > 0
            assert r.slo_deadline >= r.arrival_time

    @pytest.mark.parametrize("name", [
        "vllm_ltr_target_predictive_prompt_semantics",
        "vllm_ltr_counter_reasoning_domain_shift",
        "pars_target_alpaca_style_instruction_prompts",
        "pars_counter_reasoning_domain_shift",
        "sarathi_counter_long_context_attention_recompute",
    ])
    def test_offline_scored_generators_are_explicit_stubs(self, name):
        """These MUST raise NotImplementedError with a clear reason --
        never silently return a fabricated Request list. vLLM-LTR/PARS
        require real offline-scoring infrastructure this task does not
        build; the Sarathi long-context entry requires an attention-cost
        scaling term this simulator's timing model does not have (see
        docs/research/algorithm_stress_tests/SARATHI_COMMIT_DRIFT_20260805.md
        and the audit doc's simulator-compatibility section)."""
        with pytest.raises(NotImplementedError):
            generators.GENERATORS[name](smoke=True)

    def test_all_catalog_entries_have_a_generator_entry(self, catalog):
        catalog_ids = {t["stress_test_id"] for t in catalog["stress_tests"]}
        generator_ids = set(generators.GENERATORS)
        assert catalog_ids == generator_ids

    def test_smoke_is_smaller_than_full_for_scaling_generators(self):
        """Spot-check a handful of generators that take an n_requests-style
        smoke/full split -- full scale should never be SMALLER than smoke."""
        for name in ["fifo_target_homogeneous_low_contention",
                     "sof_counter_long_job_starvation",
                     "edf_counter_domino_effect_transient_overload"]:
            n_smoke = len(generators.GENERATORS[name](smoke=True))
            n_full = len(generators.GENERATORS[name](smoke=False))
            assert n_full >= n_smoke, name


class TestDeterminism:
    @pytest.mark.parametrize("name", list(generators.GENERATORS))
    def test_same_seed_reproducible(self, name):
        if name in {
            "vllm_ltr_target_predictive_prompt_semantics",
            "vllm_ltr_counter_reasoning_domain_shift",
            "pars_target_alpaca_style_instruction_prompts",
            "pars_counter_reasoning_domain_shift",
            "sarathi_counter_long_context_attention_recompute",
        }:
            pytest.skip("offline-scored / not-representable stub, not applicable")
        a = generators.GENERATORS[name](smoke=True)
        b = generators.GENERATORS[name](smoke=True)
        assert [(r.request_id, r.arrival_time, r.prompt_tokens, r.predicted_output_tokens,
                  r.actual_output_tokens, r.class_id) for r in a] == \
               [(r.request_id, r.arrival_time, r.prompt_tokens, r.predicted_output_tokens,
                  r.actual_output_tokens, r.class_id) for r in b]


class TestMispredictionGeneratorsDoNotLeak:
    """Structural check specific to the misprediction-based counter cases:
    predicted_output_tokens must differ from actual_output_tokens for at
    least the mispredicted subset (otherwise the "misprediction" workload
    isn't actually testing misprediction), while every Request still
    carries real ground truth in actual_output_tokens (used only by the
    simulator's own decode-length modeling and hidden metrics, never
    readable by a policy -- structurally enforced by ObservableRequest,
    not re-tested here)."""

    def test_estf_counter_has_a_mispredicted_subset(self):
        reqs = generators.estf_counter_reasoning_prompt_length_misprediction(smoke=True)
        mismatched = [r for r in reqs if r.predicted_output_tokens != r.actual_output_tokens]
        assert len(mismatched) > 0
        for r in mismatched:
            assert r.predicted_output_tokens < r.actual_output_tokens  # understated, not overstated

    def test_llf_counter_has_a_mispredicted_subset(self):
        reqs = generators.llf_counter_laxity_instability_under_prediction_error(smoke=True)
        mismatched = [r for r in reqs if r.predicted_output_tokens != r.actual_output_tokens]
        assert len(mismatched) > 0

    def test_scorpio_counter_predictions_are_pessimistic_not_optimistic(self):
        reqs = generators.scorpio_counter_false_rejection_near_threshold(smoke=True)
        for r in reqs:
            assert r.predicted_output_tokens >= r.actual_output_tokens


class TestSmokeRunnerRegressionLock:
    """Locks in the validated result from
    docs/research/algorithm_stress_tests/STRESS_TEST_VALIDATION_20260805.md:
    every auto-evaluable gate passes at smoke scale. If this regresses,
    either a generator/gate was changed without re-validating, or a
    dependency (e.g. a policy's scoring formula) changed underneath it."""

    def test_all_executable_gates_pass_at_smoke_scale(self):
        import importlib.util

        runner_path = _ROOT / "scripts" / "stress_tests" / "run_stress_test_smoke.py"
        spec = importlib.util.spec_from_file_location("run_stress_test_smoke", runner_path)
        runner = importlib.util.module_from_spec(spec)
        sys.modules["run_stress_test_smoke"] = runner
        spec.loader.exec_module(runner)

        argv_backup = sys.argv
        try:
            sys.argv = ["run_stress_test_smoke.py"]
            code = runner.main()
        finally:
            sys.argv = argv_backup
        assert code == 0
