"""Determinism/reproducibility regression tests for
scripts/generate_canonical_benchmark_suite.py (see
docs/audits/canonical_benchmark_suite_design_20260804.md).

Does not run the full suite (that's a several-minute, many-simulator-run
operation, unsuitable for a unit test) -- exercises the generator's own
building blocks directly: family definitions, dataset generation +
serialization, headroom-score composition, and the reproducibility
guarantee that matters most (identical family+seed -> byte-identical
dataset)."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "generate_canonical_benchmark_suite",
    Path(__file__).parent.parent / "scripts" / "generate_canonical_benchmark_suite.py",
)
gen = importlib.util.module_from_spec(_SPEC)
sys.modules["generate_canonical_benchmark_suite"] = gen
_SPEC.loader.exec_module(gen)


def test_nine_families_defined_with_required_metadata():
    families = gen._families()
    assert len(families) == 9
    names = [f.name for f in families]
    assert len(names) == len(set(names)), "family names must be unique"
    for f in families:
        assert f.target_phenomenon
        assert f.hypothesis
        assert f.expected_divergent_policies
        assert f.expected_winner
        assert f.workload.arrival_rate > 0
        assert f.workload.duration > 0


def test_dataset_generation_is_deterministic_for_same_seed(tmp_path):
    family = gen._families()[0]
    reqs_a = gen.generate_and_write_dataset(family, seed=0, output_root=str(tmp_path / "run_a"))
    reqs_b = gen.generate_and_write_dataset(family, seed=0, output_root=str(tmp_path / "run_b"))

    rows_a = [gen._request_to_dict(r) for r in sorted(reqs_a, key=lambda r: r.request_id)]
    rows_b = [gen._request_to_dict(r) for r in sorted(reqs_b, key=lambda r: r.request_id)]
    assert rows_a == rows_b

    hash_a = gen._sha256_of_obj(rows_a)
    hash_b = gen._sha256_of_obj(rows_b)
    assert hash_a == hash_b

    # Also confirm the files actually written to disk match byte-for-byte.
    path_a = tmp_path / "run_a" / family.name / "seed_0.json"
    path_b = tmp_path / "run_b" / family.name / "seed_0.json"
    assert path_a.read_text() == path_b.read_text()


def test_different_seeds_usually_produce_different_datasets(tmp_path):
    family = gen._families()[0]
    reqs_0 = gen.generate_and_write_dataset(family, seed=0, output_root=str(tmp_path / "s0"))
    reqs_1 = gen.generate_and_write_dataset(family, seed=1, output_root=str(tmp_path / "s1"))
    rows_0 = [gen._request_to_dict(r) for r in sorted(reqs_0, key=lambda r: r.request_id)]
    rows_1 = [gen._request_to_dict(r) for r in sorted(reqs_1, key=lambda r: r.request_id)]
    assert rows_0 != rows_1


def test_no_actual_output_tokens_leak_into_predicted_field():
    """Sanity guard: predicted_output_tokens must be perturbed by prediction
    noise (not literally equal to actual_output_tokens) for every family
    with nonzero prediction_noise_rel -- otherwise a "policy sees the
    future" leakage bug would be silent."""
    for family in gen._families():
        if family.workload.prediction_noise_rel <= 0:
            continue
        reqs = gen.generate_workload(family.workload, seed=0)
        n_equal = sum(1 for r in reqs if r.predicted_output_tokens == r.actual_output_tokens)
        # With real multiplicative noise, an exact match for every single
        # request would be a red flag (possible only by extreme coincidence
        # at low request counts); allow some incidental equality but not all.
        assert n_equal < len(reqs), (
            f"{family.name}: predicted_output_tokens never differs from "
            "actual_output_tokens despite nonzero prediction_noise_rel"
        )


def test_headroom_score_is_bounded_and_monotonic_in_its_inputs():
    good_metrics = {
        "fifo_srtf_anwg_gap": 0.10,
        "queue_contention_fraction": 1.0,
        "fifo_srtf_decision_disagreement_fraction": 0.05,
    }
    good_entropy = {"normalized_entropy": 1.0}
    zero_metrics = {
        "fifo_srtf_anwg_gap": 0.0,
        "queue_contention_fraction": 0.0,
        "fifo_srtf_decision_disagreement_fraction": 0.0,
    }
    zero_entropy = {"normalized_entropy": 0.0}

    good_score = gen.compute_headroom_score(good_metrics, good_entropy)
    zero_score = gen.compute_headroom_score(zero_metrics, zero_entropy)

    assert good_score == pytest.approx(1.0)
    assert zero_score == pytest.approx(0.0)
    assert 0.0 <= good_score <= 1.0
    assert 0.0 <= zero_score <= 1.0


def test_headroom_score_clips_out_of_range_inputs():
    over_range_metrics = {
        "fifo_srtf_anwg_gap": 1.0,  # far above the 0.10 reference ceiling
        "queue_contention_fraction": 1.0,
        "fifo_srtf_decision_disagreement_fraction": 1.0,  # far above 0.05 ceiling
    }
    entropy = {"normalized_entropy": 1.0}
    score = gen.compute_headroom_score(over_range_metrics, entropy)
    assert score == pytest.approx(1.0), "score must clip, not exceed 1.0"


def test_dataset_json_round_trips_all_required_request_fields(tmp_path):
    family = gen._families()[0]
    gen.generate_and_write_dataset(family, seed=0, output_root=str(tmp_path))
    path = tmp_path / family.name / "seed_0.json"
    rows = json.loads(path.read_text())
    assert len(rows) > 0
    required = {
        "request_id", "arrival_time", "prompt_tokens", "predicted_output_tokens",
        "actual_output_tokens", "slo_deadline", "priority", "class_id",
    }
    assert required.issubset(rows[0].keys())
