"""Tests for the Sarathi-Serve stress-test catalog section (7 entries,
added 2026-08-05 -- docs/audits/sarathi_stress_test_catalog_completion_20260805.md).

Covers: catalog schema for the new EXPERIMENTALLY_VALIDATED_ON_REAL_HARDWARE
entries, the 5 Wulver provenance records, target/counter pairing,
deterministic generation, no future-information leakage, commit-drift
disclosure, the Sarathi headroom checker, and non-interference with the
canonical benchmark suite / VTC / CC5/CC6.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "scripts" / "stress_tests"))
sys.path.insert(0, str(_ROOT / "src"))

import generators  # noqa: E402

_CATALOG_PATH = _ROOT / "configs" / "stress_tests" / "algorithm_stress_test_catalog.yaml"

_SARATHI_IDS = [
    "sarathi_counter_long_prompt_moderate_output",
    "sarathi_target_active_decode_plus_arriving_prefill",
    "sarathi_counter_prefill_heavy_burst",
    "sarathi_counter_mixed_prompt_lengths",
    "sarathi_target_kv_pressure",
    "sarathi_counter_short_prompt_decode_dominated_regime",
    "sarathi_counter_long_context_attention_recompute",
]
_REAL_HARDWARE_IDS = [
    "sarathi_counter_long_prompt_moderate_output",
    "sarathi_target_active_decode_plus_arriving_prefill",
    "sarathi_counter_prefill_heavy_burst",
    "sarathi_counter_mixed_prompt_lengths",
    "sarathi_target_kv_pressure",
]
# All 5 real-hardware entries reference this exact pair of Wulver array
# job IDs (the repeated-trial N=5 comparison, jobs 1111988/1111989,
# postprocessed by job 1111990) -- see
# docs/wulver_sarathi_vllm_repeated_validation.md.
_EXPECTED_WULVER_JOB_IDS = ["1111988", "1111989", "1111990"]


@pytest.fixture(scope="module")
def catalog():
    with open(_CATALOG_PATH) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def sarathi_entries(catalog):
    return {e["stress_test_id"]: e for e in catalog["stress_tests"] if e["algorithm_id"] == "sarathi_faithful"}


class TestCatalogSchema:
    def test_exactly_7_sarathi_entries(self, sarathi_entries):
        assert set(sarathi_entries) == set(_SARATHI_IDS)

    def test_algorithm_id_is_the_runnable_policy_not_the_paper_name(self, sarathi_entries):
        # Deliberate deviation from "algorithm_id: sarathi_serve" (the
        # literal instruction text) -- see docs/audits/
        # sarathi_stress_test_catalog_completion_20260805.md for why:
        # every other catalog row's algorithm_id is a runnable policy key,
        # and "sarathi_serve" is not registered anywhere in POLICY_FACTORIES.
        for e in sarathi_entries.values():
            assert e["algorithm_id"] == "sarathi_faithful"

    def test_2_target_5_counter_role_split(self, sarathi_entries):
        roles = [e["test_role"] for e in sarathi_entries.values()]
        assert roles.count("TARGET") == 2
        assert roles.count("COUNTER") == 5

    def test_5_real_hardware_entries_use_the_new_evidence_class(self, sarathi_entries):
        for eid in _REAL_HARDWARE_IDS:
            assert sarathi_entries[eid]["evidence_class"] == "EXPERIMENTALLY_VALIDATED_ON_REAL_HARDWARE"

    def test_2_literature_entries_use_conservative_evidence_classes(self, sarathi_entries):
        assert sarathi_entries["sarathi_counter_short_prompt_decode_dominated_regime"]["evidence_class"] \
            == "HYPOTHESIZED_ADVERSARIAL_REGIME"
        assert sarathi_entries["sarathi_counter_long_context_attention_recompute"]["evidence_class"] \
            == "PAPER_MOTIVATING_STRESS_CASE"
        # Neither literature entry is allowed to claim real-hardware
        # validation or a proven worst case -- the task explicitly
        # requires conservative labeling here.
        for eid in ["sarathi_counter_short_prompt_decode_dominated_regime",
                    "sarathi_counter_long_context_attention_recompute"]:
            assert sarathi_entries[eid]["evidence_class"] not in {
                "EXPERIMENTALLY_VALIDATED_ON_REAL_HARDWARE", "PROVEN_WORST_CASE",
            }

    def test_every_entry_forbids_actual_output_tokens(self, sarathi_entries):
        for eid, e in sarathi_entries.items():
            assert "actual_output_tokens" in e["forbidden_oracle_inputs"], eid

    def test_every_entry_declares_simulator_requirements_for_prefill_modeling(self, sarathi_entries):
        for eid, e in sarathi_entries.items():
            assert e["simulator_requirements"].get("enable_prefill_modeling") is True, eid


class TestWulverProvenance:
    """The 5 real-hardware entries must each cite the exact Wulver job IDs
    and reproducible result-artifact provenance, not a vague reference."""

    @pytest.mark.parametrize("eid", _REAL_HARDWARE_IDS)
    def test_cites_all_three_wulver_job_ids(self, sarathi_entries, eid):
        citation_text = " ".join(sarathi_entries[eid]["source_citations"])
        for job_id in _EXPECTED_WULVER_JOB_IDS:
            assert job_id in citation_text, f"{eid} missing Wulver job {job_id}"

    @pytest.mark.parametrize("eid", _REAL_HARDWARE_IDS)
    def test_cites_the_validation_doc(self, sarathi_entries, eid):
        citation_text = " ".join(sarathi_entries[eid]["source_citations"])
        assert "wulver_sarathi_vllm_repeated_validation.md" in citation_text

    def test_result_artifacts_referenced_in_catalog_header_exist_on_disk(self):
        # The catalog's own section-12 header comment cites two result
        # artifacts by path and sha256 -- confirm they still exist (a
        # provenance record pointing at a deleted/moved file is worse
        # than no record).
        summary = _ROOT / "experiments/gpu_external_validity/sarathi_vllm_repeated_trials/repeated_trials_summary.json"
        bootstrap = _ROOT / "experiments/gpu_external_validity/sarathi_vllm_repeated_trials/bootstrap_comparison.json"
        assert summary.exists()
        assert bootstrap.exists()

    def test_result_artifact_hashes_match_catalog_header(self):
        import hashlib
        catalog_text = _CATALOG_PATH.read_text()
        summary = _ROOT / "experiments/gpu_external_validity/sarathi_vllm_repeated_trials/repeated_trials_summary.json"
        bootstrap = _ROOT / "experiments/gpu_external_validity/sarathi_vllm_repeated_trials/bootstrap_comparison.json"
        summary_hash = hashlib.sha256(summary.read_bytes()).hexdigest()
        bootstrap_hash = hashlib.sha256(bootstrap.read_bytes()).hexdigest()
        assert summary_hash in catalog_text, "repeated_trials_summary.json sha256 in catalog header is stale"
        assert bootstrap_hash in catalog_text, "bootstrap_comparison.json sha256 in catalog header is stale"


class TestTargetCounterPairing:
    def test_target_scenarios_match_robust_sarathi_wins(self, sarathi_entries):
        # docs/wulver_sarathi_vllm_repeated_validation.md: Sarathi robustly
        # won active_decode_plus_arriving_prefill and kv_pressure (5/5,
        # CI excludes zero in Sarathi's favor).
        assert sarathi_entries["sarathi_target_active_decode_plus_arriving_prefill"]["test_role"] == "TARGET"
        assert sarathi_entries["sarathi_target_kv_pressure"]["test_role"] == "TARGET"

    def test_counter_scenarios_match_robust_vllm_wins(self, sarathi_entries):
        # The other 3 real-hardware scenarios were robust vLLM wins --
        # correctly labeled COUNTER, not TARGET, matching the real
        # direction rather than a hoped-for one.
        for eid in ["sarathi_counter_long_prompt_moderate_output",
                    "sarathi_counter_prefill_heavy_burst",
                    "sarathi_counter_mixed_prompt_lengths"]:
            assert sarathi_entries[eid]["test_role"] == "COUNTER"


class TestDeterministicGeneration:
    @pytest.mark.parametrize("eid", [i for i in _SARATHI_IDS if i != "sarathi_counter_long_context_attention_recompute"])
    def test_smoke_generation_is_reproducible(self, eid):
        a = generators.GENERATORS[eid](smoke=True)
        b = generators.GENERATORS[eid](smoke=True)
        assert [(r.request_id, r.arrival_time, r.prompt_tokens, r.predicted_output_tokens, r.class_id) for r in a] == \
               [(r.request_id, r.arrival_time, r.prompt_tokens, r.predicted_output_tokens, r.class_id) for r in b]

    @pytest.mark.parametrize("eid", [i for i in _SARATHI_IDS if i != "sarathi_counter_long_context_attention_recompute"])
    def test_smoke_smaller_or_equal_to_full(self, eid):
        n_smoke = len(generators.GENERATORS[eid](smoke=True))
        n_full = len(generators.GENERATORS[eid](smoke=False))
        assert n_full >= n_smoke, eid

    def test_5_real_hardware_smoke_counts_exactly_match_wulver_request_counts(self):
        # Smoke scale is designed to be an EXACT reproduction of the real
        # trial's request count (not just "similar") -- verify precisely.
        expected = {
            "sarathi_counter_long_prompt_moderate_output": 4,
            "sarathi_target_active_decode_plus_arriving_prefill": 8,
            "sarathi_counter_prefill_heavy_burst": 6,
            "sarathi_counter_mixed_prompt_lengths": 6,
            "sarathi_target_kv_pressure": 12,
        }
        for eid, n in expected.items():
            assert len(generators.GENERATORS[eid](smoke=True)) == n, eid

    def test_long_context_entry_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            generators.GENERATORS["sarathi_counter_long_context_attention_recompute"](smoke=True)


class TestNoFutureInformationLeakage:
    """No Sarathi generator may construct requests whose predicted_output_tokens
    is derived FROM actual_output_tokens in a way a real online policy
    could not observe -- mirrors the project-wide ObservableRequest
    discipline (BasePolicy never sees actual_output_tokens)."""

    @pytest.mark.parametrize("eid", [i for i in _SARATHI_IDS if i != "sarathi_counter_long_context_attention_recompute"])
    def test_predicted_equals_actual_no_hidden_bias(self, eid):
        # Every Sarathi generator sets predicted_output_tokens ==
        # actual_output_tokens (no misprediction modeling in this
        # section, unlike e.g. estf_counter_reasoning_prompt_length_misprediction)
        # -- confirm that invariant holds, since a silent divergence would
        # be an undisclosed oracle leak in the other direction (predicted
        # under/overstating actual without a documented reason).
        reqs = generators.GENERATORS[eid](smoke=True)
        for r in reqs:
            assert r.predicted_output_tokens == r.actual_output_tokens, (eid, r.request_id)

    def test_generators_module_never_imports_actual_output_tokens_into_policy_code(self):
        # Structural check: the generator module itself is the only place
        # actual_output_tokens is legitimately set (ground truth for the
        # simulator); policies read only ObservableRequest, which already
        # excludes it project-wide. Confirm sarathi_faithful.py (the
        # policy under test) never references it.
        policy_src = (_ROOT / "src/llmserveopt/policies/sarathi_faithful.py").read_text()
        assert "actual_output_tokens" not in policy_src


class TestCommitDriftDisclosure:
    def test_commit_drift_doc_exists(self):
        doc = _ROOT / "docs/research/algorithm_stress_tests/SARATHI_COMMIT_DRIFT_20260805.md"
        assert doc.exists()

    def test_commit_drift_doc_cites_both_pins(self):
        text = (_ROOT / "docs/research/algorithm_stress_tests/SARATHI_COMMIT_DRIFT_20260805.md").read_text()
        assert "ceaa0660ea2487976101a8167aad5c8046e85b27" in text
        assert "96f9911790ecc00af12ee9fae47cb8fa9ba0d199" in text

    def test_commit_drift_doc_states_a_classification(self):
        text = (_ROOT / "docs/research/algorithm_stress_tests/SARATHI_COMMIT_DRIFT_20260805.md").read_text()
        assert "MECHANISM-LEVEL VALIDATION" in text

    def test_catalog_header_references_commit_drift_doc(self):
        assert "SARATHI_COMMIT_DRIFT_20260805.md" in _CATALOG_PATH.read_text()

    def test_mechanism_calibration_doc_exists_and_is_referenced(self, sarathi_entries):
        doc = _ROOT / "docs/research/algorithm_stress_tests/SARATHI_MECHANISM_CALIBRATION_20260805.md"
        assert doc.exists()
        for eid in _REAL_HARDWARE_IDS:
            assert "calibration_note" in sarathi_entries[eid], eid


class TestHeadroomChecker:
    def test_headroom_check_script_exists(self):
        assert (_ROOT / "scripts/stress_tests/run_sarathi_headroom_check.py").exists()

    def test_headroom_check_runs_and_exits_zero(self):
        result = subprocess.run(
            [sys.executable, "scripts/stress_tests/run_sarathi_headroom_check.py"],
            cwd=_ROOT, capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    def test_headroom_check_writes_generated_workloads(self):
        gen_dir = _ROOT / "configs/stress_tests/generated/sarathi"
        assert gen_dir.exists()
        files = list(gen_dir.glob("*.json"))
        assert len(files) >= 3 * 6  # >= 3 seeds x 6 executable entries

    def test_headroom_check_writes_report(self):
        report = _ROOT / "results/stress_test_catalog/sarathi_smoke/report.json"
        assert report.exists()
        import json
        data = json.loads(report.read_text())
        assert set(data["accepted"]) | set(data.get("rejected", [])) or data["headroom"]


class TestNoModificationOfProtectedFiles:
    """This task's own exclusion list: never touch the canonical suite,
    VTC's files, or CC5/CC6 core/config files."""

    def test_canonical_suite_directory_untouched(self):
        import subprocess as sp
        result = sp.run(["git", "status", "--porcelain", "--", "benchmarks/canonical_suite/"],
                         cwd=_ROOT, capture_output=True, text=True)
        assert result.stdout.strip() == "", "canonical_suite has uncommitted changes: " + result.stdout

    def test_vtc_directory_untouched(self):
        import subprocess as sp
        result = sp.run(["git", "status", "--porcelain", "--", "baselines/vtc/"],
                         cwd=_ROOT, capture_output=True, text=True)
        assert result.stdout.strip() == "", "baselines/vtc has uncommitted changes: " + result.stdout

    def test_sarathi_faithful_policy_file_itself_untouched_by_this_task(self):
        import subprocess as sp
        result = sp.run(["git", "status", "--porcelain", "--",
                          "src/llmserveopt/policies/sarathi_faithful.py"],
                         cwd=_ROOT, capture_output=True, text=True)
        assert result.stdout.strip() == "", "sarathi_faithful.py has uncommitted changes: " + result.stdout
