"""
Tests for Phase 2C.2 causal selector retraining workflow.
"""
from __future__ import annotations

import csv
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent
RUNNER_PATH = ROOT / "scripts" / "run_phase2c2_causal_selector_retraining.py"
CONFIG_PATH = ROOT / "configs" / "phase2c2_causal_selector_retraining.yaml"


def _load_runner():
    spec = importlib.util.spec_from_file_location("run_phase2c2", RUNNER_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_phase2c2_config_loads_and_validates():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    assert cfg["experiment"] == "phase2c2_causal_selector_retraining"
    assert cfg["feature_mode"] == "causal"
    mod = _load_runner()
    issues, plan = mod.validate_phase2c2_config(cfg)
    assert issues == [], issues
    assert plan["feature_mode"] == "causal"
    assert plan["feature_mode_deployable"] is True
    assert plan["primary_rank_metric"] == "mean_arrival_normalized_wg"
    assert plan["n_evaluation_workloads"] == 6
    assert len(plan["external_style_baselines"]) == 7


def test_training_eval_workload_tags_do_not_overlap():
    mod = _load_runner()
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    eval_tags = mod._eval_tags(cfg)
    train_cfg = mod._load_nested_config(cfg["training_config"])
    train_tags = {w["tag"] for w in train_cfg.get("workloads", [])}
    assert train_tags.isdisjoint(eval_tags)


def test_external_style_baselines_identified():
    from llmserveopt.selector.roles import (
        EXTERNAL_STYLE_BASELINES,
        is_external_style_baseline,
    )

    assert is_external_style_baseline("orca_style")
    assert is_external_style_baseline("scorpio_style_slo_guard")
    assert not is_external_style_baseline("fifo")
    assert len(EXTERNAL_STYLE_BASELINES) == 7


def test_deployable_headline_excludes_oracle_assisted():
    from llmserveopt.selector.roles import (
        is_deployable_headline_selector,
        is_oracle_assisted_selector,
    )

    assert is_oracle_assisted_selector("safe_fallback_wsp_margin0.005")
    assert not is_deployable_headline_selector("safe_fallback_wsp_margin0.005")
    assert is_deployable_headline_selector("regression_anwg")


def test_assert_no_eval_leakage_raises_on_overlap():
    mod = _load_runner()
    eval_tags = {"burstgpt_scaled_high"}
    rows = [{"trace_id": "burstgpt_scaled_high_s0"}]
    try:
        mod.assert_no_eval_leakage(rows, eval_tags)
    except RuntimeError as exc:
        assert "leakage" in str(exc).lower()
    else:
        raise AssertionError("expected RuntimeError for eval tag in training rows")


def test_causal_training_rows_use_causal_feature_mode():
    mod = _load_runner()
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    train_cfg = mod._load_nested_config(cfg["training_config"])
    train_rows, _val, all_rows = mod.build_causal_training_rows(
        cfg, train_cfg, smoke=True, verbose=False,
    )
    assert train_rows
    assert all(r.get("feature_mode") == "causal" for r in all_rows)
    mod.assert_no_eval_leakage(train_rows, mod._eval_tags(cfg))


def test_phase2c2_runner_dry_run():
    result = subprocess.run(
        [sys.executable, str(RUNNER_PATH), "--dry-run"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "causal" in result.stdout.lower()
    assert "dry-run" in result.stdout.lower()


def test_phase2c2_runner_refuses_without_mode_flag():
    result = subprocess.run(
        [sys.executable, str(RUNNER_PATH)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "--allow-full-run" in result.stderr or "dry-run" in result.stderr


def test_phase2c2_smoke_produces_deployable_summary_and_analysis():
    mod = _load_runner()
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "smoke_out"
        metadata = mod.run_phase2c2(cfg, out_dir, smoke=True)
        eval_dir = out_dir / "evaluation"
        assert (eval_dir / "deployable_selector_summary.csv").exists()
        assert (eval_dir / "external_baseline_failure_analysis.json").exists()
        assert metadata["primary_rank_metric"] == "mean_arrival_normalized_wg"
        assert "safe_fallback" in "".join(metadata["oracle_assisted_selectors"])
        with open(eval_dir / "deployable_selector_summary.csv") as f:
            rows = list(csv.DictReader(f))
        assert rows
        assert all(r.get("deployable_headline") == "True" for r in rows)
        assert all("safe_fallback" not in r["selector"] for r in rows)
        assert all(r.get("primary_rank_metric") == "mean_arrival_normalized_wg" for r in rows)
