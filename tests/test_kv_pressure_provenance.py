"""Focused tests for the KV pilot runner's additive run-provenance metadata.

See docs/audits/kv_v2_reproducibility_forensic_20260817.md section 9. These
fields are collected purely for FUTURE runs; they must never affect
scenario generation, RNG order, policy execution, or scientific metrics.
"""
from __future__ import annotations

import hashlib
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
RUNNER_PATH = ROOT / "scripts" / "run_policy_separation_kv_pressure_pilot_v1.py"

spec = importlib.util.spec_from_file_location("kv_pilot_runner", RUNNER_PATH)
runner = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(ROOT / "src"))
spec.loader.exec_module(runner)


def test_sha256_file_matches_known_digest(tmp_path):
    p = tmp_path / "known.txt"
    p.write_bytes(b"hello world")
    expected = hashlib.sha256(b"hello world").hexdigest()
    assert runner._sha256_file(p) == expected


def test_sha256_file_none_for_missing_file(tmp_path):
    assert runner._sha256_file(tmp_path / "does_not_exist.txt") is None


def test_sha256_file_none_for_none_path():
    assert runner._sha256_file(None) is None


def test_config_sha256_matches_actual_bytes(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("sweep_grid:\n  seeds: [1, 2]\n")
    digest = runner._sha256_file(cfg)
    assert digest == hashlib.sha256(cfg.read_bytes()).hexdigest()


def test_dataset_sha256_matches_actual_bytes(tmp_path):
    ds = tmp_path / "dataset.csv"
    ds.write_bytes(b"Timestamp,Request tokens,Response tokens\n1,10,20\n")
    digest = runner._sha256_file(ds)
    assert digest == hashlib.sha256(ds.read_bytes()).hexdigest()


def test_git_sha_populated_in_repo_checkout():
    sha = runner._git_sha()
    assert sha is not None
    expected = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    assert sha == expected
    assert len(sha) == 40  # full git SHA-1 hex length


def test_git_dirty_is_boolean():
    dirty = runner._git_dirty()
    assert isinstance(dirty, bool)


def test_pkg_version_present_for_installed_packages():
    assert runner._pkg_version("numpy") is not None
    assert runner._pkg_version("pandas") is not None


def test_pkg_version_null_safe_for_missing_package():
    assert runner._pkg_version("this_package_does_not_exist_xyz") is None


def test_collect_provenance_null_safe_and_complete(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("sweep_grid: {}\n")
    prov = runner._collect_provenance(
        config_path=cfg,
        dataset_path=None,  # simulate an unavailable/unresolved dataset
        seeds=[3, 1, 2],
        template_version="v2",
        policy_names=["kv_constrained_online", "least_laxity_first"],
    )
    # required fields all present
    for key in (
        "git_sha", "git_dirty", "command", "config_path", "config_sha256",
        "dataset_path", "dataset_sha256", "python_version", "numpy_version",
        "pandas_version", "scipy_version", "sklearn_version", "seeds",
        "template_version", "policy_names", "utc_timestamp",
    ):
        assert key in prov
    # unresolved dataset -> null-safe, not a crash
    assert prov["dataset_path"] is None
    assert prov["dataset_sha256"] is None
    # config hash matches actual bytes
    assert prov["config_sha256"] == hashlib.sha256(cfg.read_bytes()).hexdigest()
    # seeds recorded sorted
    assert prov["seeds"] == [1, 2, 3]
    assert prov["template_version"] == "v2"
    assert prov["policy_names"] == ["kv_constrained_online", "least_laxity_first"]


def test_collect_provenance_scipy_sklearn_null_safe_if_absent(tmp_path, monkeypatch):
    """scipy/sklearn are optional per the design; simulate absence and
    confirm no crash, not that they are literally uninstalled here."""
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("sweep_grid: {}\n")

    original_pkg_version = runner._pkg_version

    def fake_pkg_version(name):
        if name in ("scipy", "sklearn"):
            return None
        return original_pkg_version(name)

    monkeypatch.setattr(runner, "_pkg_version", fake_pkg_version)
    prov = runner._collect_provenance(
        config_path=cfg, dataset_path=None, seeds=[1],
        template_version="v1", policy_names=["least_laxity_first"],
    )
    assert prov["scipy_version"] is None
    assert prov["sklearn_version"] is None


def test_result_csv_sha256_matches_final_written_file(tmp_path):
    csv_path = tmp_path / "per_policy_results.csv"
    csv_path.write_text("scenario_id,policy_name,status\nabc,kv,success\n")
    digest = runner._sha256_file(csv_path)
    assert digest == hashlib.sha256(csv_path.read_bytes()).hexdigest()


def test_default_runner_invocation_still_defines_result_fieldnames():
    """Backward compatibility: the provenance patch must not change the
    scientific CSV schema."""
    assert "scenario_id" in runner.RESULT_FIELDNAMES
    assert "arrival_normalized_weighted_goodput" in runner.RESULT_FIELDNAMES
    assert "policy_name" in runner.RESULT_FIELDNAMES


@pytest.mark.skipif(
    not (ROOT / ".local_data" / "burstgpt_v2" / "raw").is_dir(),
    reason="staged BurstGPT not available in this environment",
)
def test_provenance_does_not_alter_scientific_rows(tmp_path):
    """End-to-end: run a tiny real grid and confirm the scientific CSV rows
    are exactly what the (unmodified) simulation path would produce --
    the provenance patch only adds a `provenance` key to final_summary.json,
    it does not touch per_policy_results.csv row generation."""
    import yaml

    cfg_path = tmp_path / "tiny.yaml"
    cfg = {
        "sweep_grid": {
            "bulk_pressure": ["high"],
            "urgent_arrival_phase": ["middle"],
            "urgent_tightness": ["tight"],
            "seeds": [20260910],
        },
        "max_kv_tokens": 6000,
        "max_active_sequences": 64,
        "max_batch_tokens": 64,
    }
    cfg_path.write_text(yaml.safe_dump(cfg))
    run_dir = tmp_path / "run"

    scenarios = runner.build_scenarios(
        cfg, template_version="v2", allow_synthetic_tokens=False,
        datasets_root=ROOT / ".local_data",
    )
    assert len(scenarios) == 1

    import subprocess as sp
    result = sp.run(
        [sys.executable, str(RUNNER_PATH),
         "--config", str(cfg_path), "--run-dir", str(run_dir),
         "--template-version", "v2", "--workers", "1",
         "--datasets-root", str(ROOT / ".local_data")],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr

    import csv
    with open(run_dir / "per_policy_results.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2  # 1 scenario x 2 policies
    assert all(r["status"] == "success" for r in rows)
    for r in rows:
        assert 0.0 <= float(r["arrival_normalized_weighted_goodput"]) <= 1.0 + 1e-9

    import json
    summary = json.loads((run_dir / "final_summary.json").read_text())
    assert "provenance" in summary
    assert summary["provenance"]["result_csv_sha256"] == hashlib.sha256(
        (run_dir / "per_policy_results.csv").read_bytes()
    ).hexdigest()
    assert summary["provenance"]["dataset_path"] is not None
