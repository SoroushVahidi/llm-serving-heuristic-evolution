"""Tests for the Public Trace Replay Scenarios v1 Layer-2 builder and the
Layer-3/4 runner engine (design doc
`docs/design/PUBLIC_TRACE_REPLAY_SCENARIOS_V1.md`). These tests exercise
individual cells and synthetic checkpoint data only -- none of them runs the
full 480-cell corpus."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from llmserveopt.policy_separation import public_trace_replay_v1 as ptr


@pytest.fixture(scope="module")
def all_records():
    return ptr.build_all_scenarios()


# ---------------------------------------------------------------------------
# Deterministic counts (design doc SS4/SS5)
# ---------------------------------------------------------------------------

def test_expected_record_count(all_records):
    assert len(all_records) == 120


def test_expected_source_balance(all_records):
    counts = {}
    for r in all_records:
        counts[r["source_dataset"]] = counts.get(r["source_dataset"], 0) + 1
    assert counts == {"burstgpt": 40, "azure_2023_conv": 40, "azure_2023_code": 40}


def test_expected_evidence_class_split(all_records):
    counts = {}
    for r in all_records:
        counts[r["scenario_evidence_class"]] = counts.get(r["scenario_evidence_class"], 0) + 1
    assert counts == {ptr.FAITHFUL: 60, ptr.AUGMENTED: 60}


def test_expected_layer3_cell_count(all_records):
    total = sum(len(r["applicable_policies"]) for r in all_records)
    assert total == 480


# ---------------------------------------------------------------------------
# No duplicate scenario IDs
# ---------------------------------------------------------------------------

def test_no_duplicate_scenario_ids(all_records):
    ids = [r["canonical_scenario_id"] for r in all_records]
    assert len(ids) == len(set(ids))


def test_scenario_id_reconstructible(all_records):
    for r in all_records:
        expected = ptr.canonical_scenario_id(
            r["source_dataset"], r["window_index"], r["scenario_evidence_class"]
        )
        assert r["canonical_scenario_id"] == expected


# ---------------------------------------------------------------------------
# Window extraction / source lineage
# ---------------------------------------------------------------------------

def test_monotonic_arrival_time_per_source():
    for source in ptr.SOURCES:
        df = ptr.load_source_records(source)
        assert (df["relative_arrival_time"].diff().dropna() >= 0).all()


def test_window_selection_deterministic():
    a = ptr.select_window_indices(1_404_294, ptr.WINDOW_SIZE, ptr.WINDOWS_PER_SOURCE)
    b = ptr.select_window_indices(1_404_294, ptr.WINDOW_SIZE, ptr.WINDOWS_PER_SOURCE)
    assert a == b
    assert len(a) == ptr.WINDOWS_PER_SOURCE
    assert len(set(a)) == len(a)


def test_window_selection_no_overlap_within_source():
    for source in ptr.SOURCES:
        df = ptr.load_source_records(source)
        indices = ptr.select_window_indices(len(df), ptr.WINDOW_SIZE, ptr.WINDOWS_PER_SOURCE)
        starts = sorted(i * ptr.WINDOW_SIZE for i in indices)
        for a, b in zip(starts, starts[1:]):
            assert b >= a + ptr.WINDOW_SIZE, "overlapping windows detected"


def test_window_size_is_exact(all_records):
    for r in all_records:
        assert len(r["scenario"].requests) == ptr.WINDOW_SIZE


def test_arrival_time_rebased_to_zero(all_records):
    for r in all_records:
        arrivals = [req.arrival_time for req in r["scenario"].requests]
        assert arrivals[0] == 0.0
        assert arrivals == sorted(arrivals)


# ---------------------------------------------------------------------------
# Observed fields unchanged (faithful view == real trace values)
# ---------------------------------------------------------------------------

def test_faithful_view_actual_output_equals_source(all_records):
    faithful = [r for r in all_records if r["scenario_evidence_class"] == ptr.FAITHFUL]
    for r in faithful:
        df = ptr.load_source_records(r["source_dataset"])
        window = ptr.extract_window(df, r["window_index"], ptr.WINDOW_SIZE)
        expected = window["output_tokens"].fillna(1).clip(lower=1).to_numpy(dtype=int)
        actual = np.array([req.actual_output_tokens for req in r["scenario"].requests])
        assert (expected == actual).all()


def test_faithful_view_prompt_tokens_equals_source(all_records):
    faithful = [r for r in all_records if r["scenario_evidence_class"] == ptr.FAITHFUL]
    for r in faithful:
        df = ptr.load_source_records(r["source_dataset"])
        window = ptr.extract_window(df, r["window_index"], ptr.WINDOW_SIZE)
        expected = window["prompt_tokens"].fillna(1).clip(lower=1).to_numpy(dtype=int)
        actual = np.array([req.prompt_tokens for req in r["scenario"].requests])
        assert (expected == actual).all()


def test_faithful_view_predicted_equals_actual_output(all_records):
    """No clairvoyance test (inverted): the faithful view sets predicted ==
    actual purely because no faithful-view policy reads predicted_output_tokens
    (verified below) -- this is documented as an inert default, not a claim
    of real prediction accuracy."""
    faithful = [r for r in all_records if r["scenario_evidence_class"] == ptr.FAITHFUL]
    for r in faithful:
        for req in r["scenario"].requests:
            assert req.predicted_output_tokens == req.actual_output_tokens


def test_faithful_policies_never_read_synthesized_fields():
    """Structural leakage/clairvoyance guard: confirms by source inspection
    (not by assumption) that full_prefill/chunked_prefill_small's admission
    logic never reads priority/slo_deadline/class_id/predicted_output_tokens."""
    import inspect

    from llmserveopt.policies.prefill_control_variants import GreedyArrivalPrefillControlPolicy

    src = inspect.getsource(GreedyArrivalPrefillControlPolicy.select_action)
    src += inspect.getsource(GreedyArrivalPrefillControlPolicy)
    for forbidden in ("priority", "slo_deadline", "class_id", "predicted_output_tokens"):
        assert forbidden not in src, f"GreedyArrivalPrefillControlPolicy unexpectedly references {forbidden!r}"


# ---------------------------------------------------------------------------
# Controlled-annotation rules are frozen and correctly applied (design doc SS3)
# ---------------------------------------------------------------------------

def test_augmented_view_class_id_is_source_dataset(all_records):
    augmented = [r for r in all_records if r["scenario_evidence_class"] == ptr.AUGMENTED]
    for r in augmented:
        for req in r["scenario"].requests:
            assert req.class_id == r["source_dataset"]


def test_augmented_view_priority_uniform_one(all_records):
    augmented = [r for r in all_records if r["scenario_evidence_class"] == ptr.AUGMENTED]
    for r in augmented:
        assert all(req.priority == 1.0 for req in r["scenario"].requests)


def test_augmented_view_slo_deadline_matches_frozen_formula(all_records):
    from llmserveopt.policies.scoring import DEFAULT_ALPHA, DEFAULT_BETA

    augmented = [r for r in all_records if r["scenario_evidence_class"] == ptr.AUGMENTED]
    for r in augmented[:5]:  # spot-check a subset; formula is identical for all
        for req in r["scenario"].requests:
            service_est = DEFAULT_ALPHA * req.prompt_tokens + DEFAULT_BETA * req.predicted_output_tokens
            expected = req.arrival_time + service_est * (1.0 + ptr.SLACK_MULTIPLIER)
            assert req.slo_deadline == pytest.approx(expected, rel=1e-9)


def test_augmented_view_predicted_output_tokens_bounded(all_records):
    augmented = [r for r in all_records if r["scenario_evidence_class"] == ptr.AUGMENTED]
    for r in augmented:
        for req in r["scenario"].requests:
            assert 1 <= req.predicted_output_tokens <= 4096


def test_field_provenance_labeling_correct(all_records):
    for r in all_records:
        prov = r["field_provenance"]
        if r["scenario_evidence_class"] == ptr.FAITHFUL:
            assert ptr.EXPERIMENTAL_CONTROLLED_ANNOTATION not in prov.values()
        else:
            for f in ("predicted_output_tokens", "slo_deadline", "priority", "class_id"):
                assert prov[f] == ptr.EXPERIMENTAL_CONTROLLED_ANNOTATION
            assert prov["arrival_time"] == ptr.NATIVE
            assert prov["prompt_tokens"] == ptr.NATIVE
            assert prov["actual_output_tokens"] == ptr.NATIVE


# ---------------------------------------------------------------------------
# Policy applicability / no outcome-dependent filtering
# ---------------------------------------------------------------------------

def test_faithful_view_only_two_policies(all_records):
    faithful = [r for r in all_records if r["scenario_evidence_class"] == ptr.FAITHFUL]
    for r in faithful:
        assert set(r["applicable_policies"]) == {"full_prefill", "chunked_prefill_small"}


def test_augmented_view_all_six_policies(all_records):
    augmented = [r for r in all_records if r["scenario_evidence_class"] == ptr.AUGMENTED]
    for r in augmented:
        assert set(r["applicable_policies"]) == set(ptr.CANONICAL_ANCHOR_IDS)
        assert len(r["applicable_policies"]) == 6


def test_window_selection_independent_of_any_replay_result():
    """Regression guard: build_all_scenarios must never import anything from
    the evaluation/replay path (unified_utility_matrix's Simulator-dependent
    run_cell), proving window/scenario selection cannot depend on an
    outcome."""
    import inspect

    src = inspect.getsource(ptr.build_all_scenarios) + inspect.getsource(ptr.select_window_indices)
    for forbidden in ("run_cell", "primary_utility_anwg", "Simulator", "disagreement"):
        assert forbidden not in src


# ---------------------------------------------------------------------------
# No NaN/Inf, no invalid resource configuration
# ---------------------------------------------------------------------------

def test_no_nan_inf_in_requests(all_records):
    for r in all_records:
        for req in r["scenario"].requests:
            for field in (req.arrival_time, req.prompt_tokens, req.predicted_output_tokens,
                          req.actual_output_tokens, req.slo_deadline, req.priority):
                assert np.isfinite(field)


def test_gpu_config_valid(all_records):
    for r in all_records:
        for gpu in r["scenario"].gpu_configs:
            assert gpu.max_active_sequences > 0
            assert gpu.max_batch_tokens > 0
            assert gpu.max_kv_tokens > 0


# ---------------------------------------------------------------------------
# Deterministic reconstruction / serialization roundtrip
# ---------------------------------------------------------------------------

def test_deterministic_reconstruction():
    df = ptr.load_source_records("azure_2023_code")
    window = ptr.extract_window(df, 0, ptr.WINDOW_SIZE)
    s1, p1 = ptr.build_scenario_from_window(
        window, source="azure_2023_code", window_index=0, evidence_class=ptr.AUGMENTED,
    )
    s2, p2 = ptr.build_scenario_from_window(
        window, source="azure_2023_code", window_index=0, evidence_class=ptr.AUGMENTED,
    )
    assert s1.requests == s2.requests
    assert p1 == p2


def test_scenario_serialization_roundtrip(all_records):
    from dataclasses import asdict

    r = all_records[0]
    d = asdict(r["scenario"])
    assert d["scenario_id"] == r["scenario"].scenario_id
    assert len(d["requests"]) == ptr.WINDOW_SIZE


# ---------------------------------------------------------------------------
# No frozen artifact mutated by import/build
# ---------------------------------------------------------------------------

def test_frozen_layer1_artifacts_not_mutated_by_build():
    import subprocess

    before = subprocess.check_output(
        ["git", "status", "--short", "--", "data/public_trace_corpus_v1/",
         "experiments/mf_psd_v1/", "experiments/unified_utility_matrix_v2/"],
        cwd=ptr.ROOT, text=True,
    )
    ptr.build_all_scenarios()
    after = subprocess.check_output(
        ["git", "status", "--short", "--", "data/public_trace_corpus_v1/",
         "experiments/mf_psd_v1/", "experiments/unified_utility_matrix_v2/"],
        cwd=ptr.ROOT, text=True,
    )
    assert before == after


# ---------------------------------------------------------------------------
# Layer 3/4 runner: canonical key set (design doc SS5, task SS1/SS7)
# ---------------------------------------------------------------------------

def test_expected_cell_key_count_and_uniqueness(all_records):
    keys = ptr.expected_cell_keys(all_records)
    assert len(keys) == 480
    assert len(set(keys)) == 480


def test_expected_cell_key_faithful_augmented_split(all_records):
    keys = ptr.expected_cell_keys(all_records)
    faithful = [k for k in keys if "::faithful::" in k]
    augmented = [k for k in keys if "::augmented::" in k]
    assert len(faithful) == 120
    assert len(augmented) == 360
    assert len(faithful) + len(augmented) == len(keys)


def test_cell_key_processing_order_is_deterministic(all_records):
    a = ptr.expected_cell_keys(all_records)
    b = ptr.expected_cell_keys(ptr.build_all_scenarios())
    assert a == b


def test_canonical_cell_key_format():
    assert ptr.canonical_cell_key("SID", "PID") == "SID::PID"


# ---------------------------------------------------------------------------
# Checkpoint I/O: duplicate/corruption/resume semantics (task SS4)
# ---------------------------------------------------------------------------

def _fake_row(scenario_id="S1", policy_id="P1", status="success", anwg=0.9):
    return {
        "canonical_scenario_id": scenario_id, "canonical_policy_id": policy_id,
        "source_dataset": "burstgpt", "scenario_evidence_class": ptr.FAITHFUL,
        "status": status, "primary_utility_anwg": anwg,
        "secondary_completion_fraction": 1.0, "error": "",
    }


def test_append_and_load_checkpoint_roundtrip(tmp_path):
    ckpt = tmp_path / "checkpoint.jsonl"
    ptr.append_checkpoint_row(ckpt, _fake_row("S1", "P1"))
    ptr.append_checkpoint_row(ckpt, _fake_row("S2", "P1", status="failed", anwg=float("nan")))
    loaded = ptr.load_checkpoint(ckpt)
    assert set(loaded.keys()) == {"S1::P1", "S2::P1"}
    assert loaded["S1::P1"]["status"] == "success"
    assert loaded["S2::P1"]["status"] == "failed"


def test_checkpoint_duplicate_key_keeps_last_entry(tmp_path):
    """A later line for the same (scenario_id, policy_id) is treated as a
    resumed retry overwriting an earlier attempt -- never silently averaged
    or ambiguously merged."""
    ckpt = tmp_path / "checkpoint.jsonl"
    ptr.append_checkpoint_row(ckpt, _fake_row("S1", "P1", status="failed", anwg=float("nan")))
    ptr.append_checkpoint_row(ckpt, _fake_row("S1", "P1", status="success", anwg=0.8))
    loaded = ptr.load_checkpoint(ckpt)
    assert len(loaded) == 1
    assert loaded["S1::P1"]["status"] == "success"
    assert loaded["S1::P1"]["primary_utility_anwg"] == 0.8


def test_checkpoint_corrupt_trailing_line_detected_not_silently_dropped(tmp_path):
    ckpt = tmp_path / "checkpoint.jsonl"
    ptr.append_checkpoint_row(ckpt, _fake_row("S1", "P1"))
    with open(ckpt, "a") as f:
        f.write("{not valid json\n")
    assert ptr.scan_checkpoint_corruption(ckpt) == [2]
    # load_checkpoint must still return the well-formed row, not raise
    loaded = ptr.load_checkpoint(ckpt)
    assert set(loaded.keys()) == {"S1::P1"}


def test_is_valid_success_row_accepts_well_formed_success():
    assert ptr.is_valid_success_row(_fake_row(status="success", anwg=0.9))


def test_is_valid_success_row_rejects_failed_status():
    assert not ptr.is_valid_success_row(_fake_row(status="failed", anwg=float("nan")))


def test_is_valid_success_row_rejects_missing_required_field():
    row = _fake_row()
    del row["secondary_completion_fraction"]
    assert not ptr.is_valid_success_row(row)


def test_is_valid_success_row_rejects_nan_anwg():
    row = _fake_row(status="success", anwg=float("nan"))
    assert not ptr.is_valid_success_row(row)


def test_resume_skips_valid_success_but_recomputes_failed_and_malformed(tmp_path):
    """Directly exercises the runner's resume-decision logic (mirrors
    scripts/run_public_trace_replay_v1.py's per-cell loop) without running
    any simulation: a valid success is skippable, a failed or malformed
    entry is not."""
    ckpt = tmp_path / "checkpoint.jsonl"
    ptr.append_checkpoint_row(ckpt, _fake_row("S1", "P1", status="success", anwg=0.9))
    ptr.append_checkpoint_row(ckpt, _fake_row("S2", "P1", status="failed", anwg=float("nan")))
    loaded = ptr.load_checkpoint(ckpt)

    def should_skip(key):
        existing = loaded.get(key)
        return existing is not None and ptr.is_valid_success_row(existing)

    assert should_skip("S1::P1") is True
    assert should_skip("S2::P1") is False  # failed -> must be recomputed, not skipped
    assert should_skip("S3::P1") is False  # missing entirely -> must be computed


def test_interrupted_run_recovery_partial_checkpoint(tmp_path):
    """Simulates an interrupted run: only some expected cells are present.
    Recovery must recompute exactly the missing ones, and the integrity
    report must correctly flag the run as incomplete until they are added."""
    ckpt = tmp_path / "checkpoint.jsonl"
    expected = ["S1::P1", "S2::P1", "S3::P1"]
    ptr.append_checkpoint_row(ckpt, _fake_row("S1", "P1", status="success"))
    loaded = ptr.load_checkpoint(ckpt)
    report = ptr.validate_full_result_set(expected, loaded, ckpt)
    assert report["ok"] is False
    assert report["n_missing"] == 2
    assert set(report["missing_keys"]) == {"S2::P1", "S3::P1"}

    # "recovery": append the two missing cells
    ptr.append_checkpoint_row(ckpt, _fake_row("S2", "P1", status="success"))
    ptr.append_checkpoint_row(ckpt, _fake_row("S3", "P1", status="failed", anwg=float("nan")))
    loaded2 = ptr.load_checkpoint(ckpt)
    report2 = ptr.validate_full_result_set(expected, loaded2, ckpt)
    assert report2["ok"] is True  # complete: every expected cell present (success or failed)
    assert report2["n_success"] == 2
    assert report2["n_failed"] == 1


def test_finalizer_refuses_incomplete_result_set(tmp_path):
    ckpt = tmp_path / "checkpoint.jsonl"
    ptr.append_checkpoint_row(ckpt, _fake_row("S1", "P1"))
    loaded = ptr.load_checkpoint(ckpt)
    report = ptr.validate_full_result_set(["S1::P1", "S2::P1"], loaded, ckpt)
    assert report["ok"] is False


def test_finalizer_flags_unexpected_extra_cells(tmp_path):
    ckpt = tmp_path / "checkpoint.jsonl"
    ptr.append_checkpoint_row(ckpt, _fake_row("S1", "P1"))
    ptr.append_checkpoint_row(ckpt, _fake_row("S_UNEXPECTED", "P1"))
    loaded = ptr.load_checkpoint(ckpt)
    report = ptr.validate_full_result_set(["S1::P1"], loaded, ckpt)
    assert report["ok"] is False
    assert "S_UNEXPECTED::P1" in report["unexpected_keys"]


# ---------------------------------------------------------------------------
# Layer 3/4 execution: tiny real end-to-end cell (not the 480-cell corpus)
# ---------------------------------------------------------------------------

def test_evaluate_scenario_policy_returns_row_and_trajectory(all_records):
    r = next(x for x in all_records if x["scenario_evidence_class"] == ptr.FAITHFUL)
    row, traj = ptr.evaluate_scenario_policy(
        r["scenario"], "full_prefill", capture_trajectory=True,
        canonical_scenario_id=r["canonical_scenario_id"],
        source_dataset=r["source_dataset"],
        scenario_evidence_class=r["scenario_evidence_class"],
    )
    assert row["status"] == "success"
    assert row["canonical_scenario_id"] == r["canonical_scenario_id"]
    assert row["canonical_policy_id"] == "full_prefill"
    assert len(traj) > 0


def test_trajectory_rows_map_to_the_requested_cell(all_records):
    r = next(x for x in all_records if x["scenario_evidence_class"] == ptr.FAITHFUL)
    _, traj = ptr.evaluate_scenario_policy(
        r["scenario"], "chunked_prefill_small", capture_trajectory=True,
        canonical_scenario_id=r["canonical_scenario_id"],
        source_dataset=r["source_dataset"],
        scenario_evidence_class=r["scenario_evidence_class"],
    )
    assert all(row["canonical_scenario_id"] == r["canonical_scenario_id"] for row in traj)
    assert all(row["policy_id"] == "chunked_prefill_small" for row in traj)
    assert all(row["scenario_evidence_class"] == ptr.FAITHFUL for row in traj)
    # step numbers strictly increasing (each row is one select_action call)
    steps = [row["step"] for row in traj]
    assert steps == sorted(steps)


def test_trajectory_capture_does_not_expose_actual_output_tokens():
    """Leakage guard: the Layer-4 trajectory row schema itself must never
    carry a clairvoyant actual-output-length field -- only the fields
    computable from ObservableState (queue/active/KV/admission), matching
    design doc SS6."""
    r = ptr.build_all_scenarios()[0]
    _, traj = ptr.evaluate_scenario_policy(
        r["scenario"], "full_prefill", capture_trajectory=True,
        canonical_scenario_id=r["canonical_scenario_id"],
        source_dataset=r["source_dataset"],
        scenario_evidence_class=r["scenario_evidence_class"],
    )
    for row in traj:
        assert "actual_output_tokens" not in row
        assert "predicted_output_tokens" not in row


def test_no_trajectory_capture_by_default():
    r = ptr.build_all_scenarios()[0]
    row, traj = ptr.evaluate_scenario_policy(
        r["scenario"], "full_prefill",
        canonical_scenario_id=r["canonical_scenario_id"],
        source_dataset=r["source_dataset"],
        scenario_evidence_class=r["scenario_evidence_class"],
    )
    assert traj == []
    assert row["status"] == "success"


def test_write_trajectory_parquet_roundtrip(tmp_path):
    rows = [{"step": 0, "queue_length": 1}, {"step": 1, "queue_length": 0}]
    out = ptr.write_trajectory_parquet(tmp_path, "PUBLIC_TRACE::x::w0::faithful", "full_prefill", rows)
    assert out is not None and out.exists()
    df = pd.read_parquet(out)
    assert len(df) == 2


def test_write_trajectory_parquet_empty_rows_writes_nothing(tmp_path):
    out = ptr.write_trajectory_parquet(tmp_path, "PUBLIC_TRACE::x::w0::faithful", "full_prefill", [])
    assert out is None
    assert list(tmp_path.iterdir()) == []


def test_evaluation_does_not_mutate_scenario_object(all_records):
    import copy

    r = next(x for x in all_records if x["scenario_evidence_class"] == ptr.AUGMENTED)
    before = copy.deepcopy(r["scenario"].requests)
    ptr.evaluate_scenario_policy(
        r["scenario"], "weighted_fair_share",
        canonical_scenario_id=r["canonical_scenario_id"],
        source_dataset=r["source_dataset"],
        scenario_evidence_class=r["scenario_evidence_class"],
    )
    assert r["scenario"].requests == before


# ---------------------------------------------------------------------------
# Provenance (task SS6)
# ---------------------------------------------------------------------------

def test_runner_provenance_has_required_fields():
    import subprocess
    import sys as _sys

    runner = ptr.ROOT / "scripts" / "run_public_trace_replay_v1.py"
    result = subprocess.run(
        [_sys.executable, "-c",
         "import sys; sys.path.insert(0, 'scripts'); sys.path.insert(0, 'src'); "
         "import run_public_trace_replay_v1 as m; "
         "p = m.build_provenance('test-command'); "
         "import json; print(json.dumps(sorted(p.keys())))"],
        cwd=ptr.ROOT, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    import json
    keys = set(json.loads(result.stdout))
    required = {
        "git_head_sha", "git_tree_dirty", "design_doc_sha256", "builder_module_sha256",
        "layer1_manifest_sha256", "runner_sha256", "python_executable", "python_version",
        "exact_command", "window_size", "windows_per_source", "seed",
        "prediction_noise_sigma", "slack_multiplier",
    }
    assert required <= keys
    assert runner.exists()


# ---------------------------------------------------------------------------
# No scientific-design CLI flags exposed (task SS2 explicit prohibition)
# ---------------------------------------------------------------------------

def test_runner_exposes_no_scientific_parameter_flags():
    import ast

    runner_src = (ptr.ROOT / "scripts" / "run_public_trace_replay_v1.py").read_text()
    tree = ast.parse(runner_src)
    add_argument_calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
    ]
    flag_names = set()
    for call in add_argument_calls:
        for arg in call.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                flag_names.add(arg.value)
    forbidden_substrings = (
        "window", "gpu", "slo", "deadline", "priority", "class", "noise", "sigma",
        "seed", "policy", "evidence",
    )
    for flag in flag_names:
        lowered = flag.lower()
        for forbidden in forbidden_substrings:
            assert forbidden not in lowered, (
                f"CLI flag {flag!r} looks like it could silently change frozen scientific design "
                f"(matched {forbidden!r})"
            )
    assert flag_names == {"--smoke", "--resume", "--out-dir"}
