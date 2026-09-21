"""Validation of the corrected derivative of the fresh-causal state-level artifact.

Repo-only tests use the committed SBS-reference extraction and the frozen action-level file as independent
sources.  Shard-dependent tests skip when the local provenance archive is not present.
"""
from __future__ import annotations

import csv
import json
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

from scripts import fresh_causal_correct_state_level_v1 as cor
from scripts import fresh_causal_robustness_v1 as rob

FROZEN, OUT = cor.FROZEN, cor.OUT
STATE = FROZEN / "FRESH_LATENCY_STATE_LEVEL_V1.csv"
CORRECTED = OUT / cor.CORRECTED_NAME
PROV = OUT / cor.PROV_NAME
pytestmark = pytest.mark.skipif(not (CORRECTED.exists() and STATE.exists()), reason="corrected derivative not generated")


def _prov():
    return json.loads(PROV.read_text())


def _str_frame(p):
    return pd.read_csv(p, dtype=str)


# ------------------------------------------------------------------ 1. frozen original untouched
def test_frozen_compact_artifacts_match_their_frozen_hash_file():
    h = json.loads((FROZEN / "FRESH_LATENCY_ARTIFACT_HASHES_V1.json").read_text())["compact_artifacts"]
    bad = [f for f, x in h.items() if cor.sha256_file(FROZEN / f) != x]
    assert not bad, bad


def test_frozen_inputs_match_hashes_recorded_at_correction_time():
    rec = _prov()["frozen_inputs_sha256"]
    assert rec and all(cor.sha256_file(FROZEN / f) == h for f, h in rec.items())
    assert _prov()["frozen_directory_unchanged_by_run"] is True


def test_frozen_directory_has_no_git_modifications():
    try:
        out = subprocess.check_output(["git", "status", "--porcelain", "--", str(FROZEN.relative_to(cor.ROOT))], cwd=cor.ROOT, text=True, stderr=subprocess.DEVNULL)
    except Exception:
        pytest.skip("git unavailable")
    assert out.strip() == ""


# ------------------------------------------------------------------ 2. corrected values match the SBS reference sources
def _sbs():
    fields, rows = cor.read_rows(OUT / cor.SBS_NAME)
    return {r["state_id"]: r for r in rows}


def test_corrected_columns_equal_committed_sbs_reference_rows():
    sbs = _sbs()
    _, rows = cor.read_rows(CORRECTED)
    assert len(rows) == 720 and set(sbs) == {r["state_id"] for r in rows}
    for r in rows:
        assert float(r["mean_ref_latency"]) == float(sbs[r["state_id"]]["mean_latency"])
        assert float(r["p95_ref_latency"]) == float(sbs[r["state_id"]]["p95_latency"])
        assert sbs[r["state_id"]]["branch_type"] == "SBS_REFERENCE"


def test_sbs_mean_matches_action_level_source_independently():
    """Action-level artifact stores counterfactual mean and a_lat = ref - cf, so ref = cf + a_lat (independent path)."""
    a = pd.read_csv(FROZEN / "FRESH_LATENCY_ACTION_LEVEL_V1.csv")
    rec = (a.mean_latency + a.a_lat).groupby(a.state_id).agg(["min", "max"])
    assert (rec["max"] - rec["min"]).max() < 1e-12
    sbs = _sbs()
    diff = max(abs(rec.loc[s, "min"] - float(sbs[s]["mean_latency"])) for s in rec.index)
    assert diff < 1e-15


def test_sbs_p95_matches_frozen_regime_p95_effect_independently():
    """p95_effect_mean (frozen regime CSV, computed from the executor's correct `ref` frame) must equal the mean of
    (SBS p95 - counterfactual p95) recomputed from the corrected reference values and the action-level file."""
    sbs = _sbs()
    a = pd.read_csv(FROZEN / "FRESH_LATENCY_ACTION_LEVEL_V1.csv")
    a["ref_p95"] = a.state_id.map(lambda s: float(sbs[s]["p95_latency"]))
    a["d"] = a.ref_p95 - a.p95_latency
    reg = pd.read_csv(FROZEN / "FRESH_LATENCY_WORKLOAD_REGIME_V1.csv")
    for _, r in reg.iterrows():
        g = a[(a.source_dataset == r.source_dataset) & (a.axis == r.axis) & (a.condition_id == r.condition_id)]
        assert abs(g.d.mean() - r.p95_effect_mean) < 1e-12


def test_defect_source_is_first_counterfactual_branch_of_each_state():
    a = pd.read_csv(FROZEN / "FRESH_LATENCY_ACTION_LEVEL_V1.csv", dtype=str)
    first = a.groupby("state_id", sort=True).first()
    _, rows = cor.read_rows(STATE)
    for r in rows:
        f = first.loc[r["state_id"]]
        assert float(r["mean_ref_latency"]) == float(f["mean_latency"])
        assert float(r["p95_ref_latency"]) == float(f["p95_latency"])


# ------------------------------------------------------------------ 3. primary endpoint unchanged
def test_primary_endpoint_identical_in_original_and_corrected():
    _, o = cor.read_rows(STATE)
    _, c = cor.read_rows(CORRECTED)
    ho = np.array([float(r["oracle_headroom"]) for r in o])
    hc = np.array([float(r["oracle_headroom"]) for r in c])
    assert (ho == hc).all()
    frozen = json.loads((FROZEN / "FRESH_LATENCY_CAUSAL_RESULT_V1.json").read_text())["primary"]
    assert float(hc.mean()) == frozen["mean_oracle_headroom"]  # exact when parsed with float()
    assert int((hc > 0).sum()) == frozen["beneficial_states"] == 590
    assert [r["beneficial_opportunity"] for r in o] == [r["beneficial_opportunity"] for r in c]


def test_headroom_recomputed_from_action_level_equals_corrected_file():
    # correctly-rounded float parsing (pandas' default parser is off by up to ~1e-12 relative)
    a = pd.read_csv(FROZEN / "FRESH_LATENCY_ACTION_LEVEL_V1.csv", float_precision="round_trip")
    h = a.groupby("state_id").a_lat.max().clip(lower=0)
    _, c = cor.read_rows(CORRECTED)
    assert max(abs(h[r["state_id"]] - float(r["oracle_headroom"])) for r in c) == 0.0


def test_bootstrap_ci_from_corrected_file_equals_frozen_bootstrap():
    fb = json.loads((FROZEN / "FRESH_LATENCY_BOOTSTRAP_V1.json").read_text())
    ci = rob.canonical_bootstrap_loop(pd.read_csv(CORRECTED))
    assert ci["clusters"] == fb["clusters"] == 36
    assert ci["lo"] == fb["mean_oracle_headroom_ci95_low"] and ci["hi"] == fb["mean_oracle_headroom_ci95_high"]


# ------------------------------------------------------------------ 4. structure and no unrelated changes
def test_rows_identifiers_columns_unchanged_and_only_two_columns_differ():
    o, c = _str_frame(STATE), _str_frame(CORRECTED)
    assert len(o) == len(c) == 720
    assert list(o.columns) == list(c.columns)
    assert o.state_id.tolist() == c.state_id.tolist()
    differing = [k for k in o.columns if not (o[k] == c[k]).all()]
    assert set(differing) <= set(cor.CORRECTED_COLUMNS) and differing  # only the proven-incorrect columns, and they do differ
    for k in o.columns:
        if k not in cor.CORRECTED_COLUMNS:
            assert (o[k] == c[k]).all(), k


def test_change_counts_are_recorded_and_consistent():
    audit = pd.read_csv(OUT / cor.AUDIT_NAME)
    v = _prov()["validation"]
    assert int(audit.mean_ref_latency_changed.sum()) == v["n_rows_mean_ref_latency_changed"]
    assert int(audit.p95_ref_latency_changed.sum()) == v["n_rows_p95_ref_latency_changed"]
    assert v["n_original_values_equal_first_cf_branch_value"] == 720
    assert len(audit) == 720


# ------------------------------------------------------------------ 5. hashes / provenance reproducible
def test_output_hashes_match_provenance():
    for name, h in _prov()["outputs_sha256"].items():
        assert cor.sha256_file(OUT / name) == h, name


def test_corrected_csv_is_reproduced_deterministically_from_frozen_original_and_committed_extraction():
    fields, orig = cor.read_rows(STATE)
    rebuilt = cor.csv_bytes(fields, cor.build_corrected_rows(orig, _sbs()))
    assert cor.sha256_bytes(rebuilt) == _prov()["outputs_sha256"][cor.CORRECTED_NAME]
    assert rebuilt == CORRECTED.read_bytes()


def test_provenance_has_no_timestamps_or_absolute_paths():
    s = PROV.read_text()
    assert "/home/" not in s and "/tmp/" not in s
    assert _prov()["sources"]["shards_verified_against_frozen_manifest"]["all_match"] is True


def test_build_corrected_rows_replaces_only_two_columns():
    orig = [{"state_id": "a", "x": "1", "mean_ref_latency": "9.0", "p95_ref_latency": "9.0", "y": "z"}]
    sbs = {"a": {"mean_latency": "0.25", "p95_latency": "0.5"}}
    out = cor.build_corrected_rows(orig, sbs)
    assert out == [{"state_id": "a", "x": "1", "mean_ref_latency": "0.25", "p95_ref_latency": "0.5", "y": "z"}]
    assert orig[0]["mean_ref_latency"] == "9.0"  # input not mutated


# ------------------------------------------------------------------ 6. 590 vs 589 (repo-only)
def test_590_strict_vs_589_guarded_and_residue_is_one_ulp():
    _, rows = cor.read_rows(STATE)
    h = np.array([float(r["oracle_headroom"]) for r in rows])
    assert int((h > 0).sum()) == 590 and int((h > 1e-12).sum()) == 589
    resid = h[(h > 0) & (h <= 1e-12)]
    assert len(resid) == 1 and resid[0] == 5.551115123125783e-17
    sid = [r["state_id"] for r in rows if 0 < float(r["oracle_headroom"]) <= 1e-12][0]
    ref = float(_sbs()[sid]["mean_latency"])
    assert resid[0] == float(np.spacing(ref))  # exactly one unit in the last place of the latency
    smallest_genuine = h[h > 1e-12].min()
    assert smallest_genuine > 5e-6 and smallest_genuine / resid[0] > 1e10


# ------------------------------------------------------------------ 7. shard-dependent (skipped without the archive)
RUN_ROOT = cor.resolve_run_root(None)
needs_shards = pytest.mark.skipif(RUN_ROOT is None, reason="local provenance archive with continuation shards not present")


@needs_shards
def test_shards_match_frozen_manifest():
    assert cor.verify_shards(RUN_ROOT)["all_match"]


@needs_shards
def test_replay_reproduces_frozen_artifacts_byte_for_byte_and_downstream_is_independent_of_ref_columns():
    proof = cor.replay_proof(cor.load_shard_rows(RUN_ROOT))
    assert all(v for k, v in proof.items() if not k.startswith("_")), proof


@needs_shards
def test_committed_sbs_extraction_is_reproduced_from_shards():
    fields, rows = cor.sbs_reference_table(cor.load_shard_rows(RUN_ROOT))
    assert cor.csv_bytes(fields, rows) == (OUT / cor.SBS_NAME).read_bytes()
