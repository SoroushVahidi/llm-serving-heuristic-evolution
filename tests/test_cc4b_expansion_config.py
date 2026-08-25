"""CC4b: expansion-config generator and quality-gate script tests."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from generate_cc4b_expansion_config import build_config  # noqa: E402


def test_build_config_is_deterministic():
    c1 = build_config()
    c2 = build_config()
    assert c1 == c2


def test_build_config_workload_tags_and_seeds_unique():
    config = build_config()
    workloads = config["workloads"]
    tags = [w["tag"] for w in workloads]
    seeds = [w["seed"] for w in workloads]
    assert len(tags) == len(set(tags))
    assert len(seeds) == len(set(seeds))


def test_build_config_reuses_cc4_candidate_search_unchanged():
    import yaml

    cc4_config = yaml.safe_load((ROOT / "configs" / "cc4_oracle_composition_dataset.yaml").read_text())
    cc4b_config = build_config()
    assert cc4b_config["candidate_search"] == cc4_config["candidate_search"]
    assert cc4b_config["policy_subset"] == cc4_config["policy_subset"]
    assert cc4b_config["cc1b_borda_baseline"] == cc4_config["cc1b_borda_baseline"]


def test_build_config_split_counts_meet_expansion_target():
    config = build_config()
    by_split: dict[str, int] = {}
    for w in config["workloads"]:
        by_split[w["split"]] = by_split.get(w["split"], 0) + 1
    held_out = by_split.get("ID_TEST", 0) + by_split.get("OOD_TEST", 0)
    assert held_out >= 50, f"held-out windows {held_out} below the 50-100+ target"


def test_build_config_no_synthetic_window_produces_zero_requests():
    """Every synthetic workload entry must actually generate a non-empty
    trace (the self-healing seed-retry logic in build_workloads() must have
    resolved any zero-arrival draws before this config is ever used)."""
    from llmserveopt.experiments.cc1_composition_opportunity import _build_synthetic_requests

    config = build_config()
    for w in config["workloads"]:
        if w["kind"] != "synthetic":
            continue
        reqs = _build_synthetic_requests(w, seed=w["seed"], max_requests=w.get("max_requests"))
        assert len(reqs) > 0, w["tag"]


def test_committed_cc4b_config_matches_freshly_generated_config():
    """The committed configs/cc4b_oracle_composition_expansion.yaml must be
    exactly what the generator produces -- not hand-edited out of sync."""
    import yaml

    committed_path = ROOT / "configs" / "cc4b_oracle_composition_expansion.yaml"
    if not committed_path.exists():
        pytest.skip("configs/cc4b_oracle_composition_expansion.yaml not present")
    committed = yaml.safe_load(committed_path.read_text())
    fresh = build_config()
    assert committed == fresh


def test_quality_gates_script_reports_required_fields_on_real_dataset():
    dataset_dir = ROOT / "results" / "cc4b_oracle_composition_expansion" / "20260803T182426Z"
    if not (dataset_dir / "manifest.json").exists():
        pytest.skip("CC4b reference dataset not present locally")
    result = subprocess.run(
        [sys.executable, "scripts/check_cc4b_quality_gates.py", str(dataset_dir)],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    out = result.stdout
    for required in (
        "total_windows=", "held_out_windows=", "non_near_tie_held_out_windows=",
        "fixed_policy_spread(held-out)", "oracle_family_distribution", "completion accounting",
        "replay_commands.sh", "Verdict:",
    ):
        assert required in out, f"missing {required!r} in quality-gate output:\n{out}"


def test_quality_gates_script_rejects_missing_dataset(tmp_path):
    result = subprocess.run(
        [sys.executable, "scripts/check_cc4b_quality_gates.py", str(tmp_path / "does_not_exist")],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    assert result.returncode == 1
    assert "FAIL" in result.stdout
