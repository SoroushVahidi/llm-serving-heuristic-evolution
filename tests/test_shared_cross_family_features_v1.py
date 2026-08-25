"""Focused tests for SHARED_CORE_V1 (feature-schema redesign investigation).

See docs/audits/shared_cross_family_feature_schema_feasibility_v1_20260817.md
and src/llmserveopt/policy_separation/shared_context_features_v1.py. These
tests validate the pure feature-computation function and (where the frozen
build artifacts are present) the built table's anti-leakage/conservation
properties -- they do NOT train or evaluate any selector.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from llmserveopt.core.types import GPUConfig, Request  # noqa: E402
from llmserveopt.policy_separation.shared_context_features_v1 import (  # noqa: E402
    SHARED_CORE_V1_FEATURES,
    compute_shared_context_features,
)

SHARED_DIR = REPO_ROOT / "experiments" / "shared_cross_family_features_v1"
MF_PSD_LONG = REPO_ROOT / "experiments" / "mf_psd_v1" / "mf_psd_long_v1.csv"


def _mk_request(rid, arrival, prompt, pred_out, actual_out=None, deadline=None, priority=1.0, class_id="c") -> Request:
    return Request(
        request_id=rid,
        arrival_time=arrival,
        prompt_tokens=prompt,
        predicted_output_tokens=pred_out,
        actual_output_tokens=actual_out if actual_out is not None else pred_out,
        slo_deadline=deadline if deadline is not None else arrival + 5.0,
        priority=priority,
        class_id=class_id,
    )


def _mk_gpu(max_active=64, max_batch=64, max_kv=6000) -> GPUConfig:
    return GPUConfig(gpu_id=0, max_active_sequences=max_active, max_batch_tokens=max_batch, max_kv_tokens=max_kv)


# ---------------------------------------------------------------------------
# Pure-function formula tests
# ---------------------------------------------------------------------------


def test_shared_core_v1_feature_set_has_no_family_prefixed_or_identity_names():
    for name in SHARED_CORE_V1_FEATURES:
        assert not name.startswith("feat_A__")
        assert not name.startswith("feat_B__")
        assert not name.startswith("feat_C__")
        assert "family" not in name.lower()
        assert "scenario" not in name.lower()
        assert "utility" not in name.lower() and "anwg" not in name.lower()
        assert "policy" not in name.lower()


def test_compute_shared_context_features_returns_exactly_the_allowlist():
    reqs = [_mk_request(0, 0.0, 100, 20), _mk_request(1, 1.0, 200, 30)]
    gpus = [_mk_gpu()]
    feats = compute_shared_context_features(reqs, gpus)
    assert set(feats.keys()) == set(SHARED_CORE_V1_FEATURES)
    for v in feats.values():
        assert math.isfinite(v)


def test_hand_computed_formulas_on_a_tiny_deterministic_scenario():
    # 3 requests, arrival at t=0,2,4s; prompt tokens 100/200/300; predicted
    # output tokens all 50; slack = deadline - arrival = 10s for all but one.
    reqs = [
        _mk_request(0, 0.0, 100, 50, deadline=10.0, priority=1.0, class_id="x"),
        _mk_request(1, 2.0, 200, 50, deadline=12.0, priority=1.0, class_id="x"),
        _mk_request(2, 4.0, 300, 50, deadline=6.0, priority=2.0, class_id="y"),
    ]
    gpus = [_mk_gpu(max_active=10, max_batch=10, max_kv=1000)]
    feats = compute_shared_context_features(reqs, gpus)

    assert feats["n_requests"] == 3.0
    assert feats["window_span_s"] == pytest.approx(4.0)
    assert feats["offered_rate_rps"] == pytest.approx(2 / 4.0)
    assert feats["mean_prompt_tokens"] == pytest.approx((100 + 200 + 300) / 3)
    assert feats["mean_predicted_output_tokens"] == pytest.approx(50.0)
    assert feats["cv_predicted_output_tokens"] == pytest.approx(0.0)
    assert feats["mean_predicted_total_tokens"] == pytest.approx((150 + 250 + 350) / 3)
    # slacks: 10, 10, 2 -> mean=22/3, min=2
    assert feats["mean_slack_s"] == pytest.approx(22 / 3)
    assert feats["min_slack_s"] == pytest.approx(2.0)
    assert feats["n_distinct_request_classes"] == 2.0
    assert feats["max_active_sequences"] == pytest.approx(10.0)
    assert feats["max_kv_tokens"] == pytest.approx(1000.0)
    assert feats["concurrency_pressure"] == pytest.approx(3 / 10)
    expected_footprint = (feats["mean_predicted_total_tokens"] * 3) / 1000.0
    assert feats["token_footprint_per_kv"] == pytest.approx(expected_footprint)


def test_frac_tight_slack_and_priority_cv_ranges():
    reqs = [_mk_request(i, float(i), 100, 20, deadline=float(i) + 0.01 * i, priority=1.0 + i) for i in range(10)]
    gpus = [_mk_gpu()]
    feats = compute_shared_context_features(reqs, gpus)
    assert 0.0 <= feats["frac_tight_slack"] <= 1.0
    assert feats["priority_cv"] >= 0.0


def test_actual_output_tokens_never_affects_computed_features():
    """Anti-leakage: actual_output_tokens is policy-hidden ground truth --
    the shared feature function must be invariant to it."""
    base = [_mk_request(0, 0.0, 100, 50, actual_out=50), _mk_request(1, 1.0, 200, 60, actual_out=60)]
    perturbed = [_mk_request(0, 0.0, 100, 50, actual_out=9999), _mk_request(1, 1.0, 200, 60, actual_out=1)]
    gpus = [_mk_gpu()]
    assert compute_shared_context_features(base, gpus) == compute_shared_context_features(perturbed, gpus)


def test_empty_requests_or_gpu_configs_rejected():
    with pytest.raises(ValueError):
        compute_shared_context_features([], [_mk_gpu()])
    with pytest.raises(ValueError):
        compute_shared_context_features([_mk_request(0, 0.0, 10, 10)], [])


# ---------------------------------------------------------------------------
# Built-table tests (skipped if the artifact hasn't been built locally --
# building requires LLM_SERVEOPT_BURSTGPT_CSV, a local dataset path, so CI
# without staged BurstGPT data should skip rather than fail).
# ---------------------------------------------------------------------------

_TABLE_PATH = SHARED_DIR / "shared_core_v1_scenarios.csv"
_SCHEMA_PATH = SHARED_DIR / "shared_core_v1_schema.json"

pytestmark_skip = pytest.mark.skipif(
    not _TABLE_PATH.exists() or not MF_PSD_LONG.exists(),
    reason="shared_core_v1 build artifact or MF-PSD source not present locally",
)


def _read_table():
    with open(_TABLE_PATH, newline="") as f:
        return list(csv.DictReader(f))


@pytestmark_skip
def test_built_table_has_exactly_176_rows_matching_mf_psd_scenarios():
    rows = _read_table()
    assert len(rows) == 176
    built_ids = {r["canonical_scenario_id"] for r in rows}
    assert len(built_ids) == 176

    with open(MF_PSD_LONG, newline="") as f:
        mf_psd_ids = {r["canonical_scenario_id"] for r in csv.DictReader(f)}
    assert built_ids == mf_psd_ids


@pytestmark_skip
def test_built_table_has_no_missing_values_for_any_learnable_feature():
    rows = _read_table()
    for r in rows:
        for feat in SHARED_CORE_V1_FEATURES:
            assert r[feat] != "", f"{r['canonical_scenario_id']} missing {feat}"
            float(r[feat])  # must parse


@pytestmark_skip
def test_built_table_replay_verified_for_every_row():
    rows = _read_table()
    for r in rows:
        assert r["replay_verified"] == "True"


@pytestmark_skip
def test_schema_denies_identity_and_family_fields_from_learnable_set():
    with open(_SCHEMA_PATH) as f:
        schema = json.load(f)
    allowlist = set(schema["learnable_feature_allowlist"])
    assert "mechanism_family" not in allowlist
    assert "canonical_scenario_id" not in allowlist
    assert "source_scenario_id" not in allowlist
    assert allowlist == set(SHARED_CORE_V1_FEATURES)
    for forbidden in schema["forbidden_audit_only_fields"]:
        assert forbidden not in allowlist


@pytestmark_skip
def test_built_table_family_counts_match_mf_psd():
    rows = _read_table()
    counts = {}
    for r in rows:
        counts[r["mechanism_family"]] = counts.get(r["mechanism_family"], 0) + 1
    assert counts == {
        "FAMILY_A_FAIRNESS_STARVATION_V2": 72,
        "FAMILY_B_PREFILL_DECODE_V2": 32,
        "FAMILY_C_KV_PRESSURE_V2": 72,
    }


@pytestmark_skip
def test_mf_psd_source_not_mutated_by_shared_feature_build():
    """This build only reads mf_psd_long_v1.csv and the frozen Family C
    reconstruction artifact -- it must never rewrite them."""
    sha_before = hashlib.sha256(MF_PSD_LONG.read_bytes()).hexdigest()
    with open(REPO_ROOT / "experiments" / "mf_psd_v1" / "mf_psd_provenance_v1.json") as f:
        prov = json.load(f)
    assert prov["output_files"]["mf_psd_long_v1.csv"] == sha_before
