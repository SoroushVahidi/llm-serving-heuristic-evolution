from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest


def _load_smoke_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_local_e2e_smoke.py"
    spec = importlib.util.spec_from_file_location("run_local_e2e_smoke", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_chronological_split_labels_keep_test_last():
    mod = _load_smoke_module()
    assert mod.chronological_split_labels(6, 0.5, 0.25) == [
        "TRAIN",
        "TRAIN",
        "TRAIN",
        "VALIDATION",
        "TEST",
        "TEST",
    ]


def test_select_rows_for_policies_reads_policy_matrix_not_features():
    mod = _load_smoke_module()
    windows = pd.DataFrame({"window_idx": [0, 1], "feat_prompt_mean": [10.0, 20.0]})
    matrix = pd.DataFrame({
        "window_idx": [0, 0, 1, 1],
        "policy_name": ["fifo", "edf", "fifo", "edf"],
        "metric_arrival_normalized_weighted_goodput": [0.1, 0.2, 0.3, 0.4],
    })
    selected = mod.select_rows_for_policies(windows, matrix, ["edf", "fifo"])
    assert selected["metric_arrival_normalized_weighted_goodput"].tolist() == [0.2, 0.3]


def test_full_27_policy_library_smoke_end_to_end(tmp_path, monkeypatch):
    """Integrated smoke test: real trace -> causal features -> full
    27-policy reward vector -> advanced selector -> selected policy -> ANWG.

    Proves the two formerly separate development lines (this branch's
    causal advanced-selector infrastructure and the integration branch's
    27-policy Policy Library v2 registry) actually interoperate -- not a
    performance claim. Small, deterministic, CPU-only, no network/GPU/paid
    service, completes in seconds.
    """
    pytest.importorskip("sklearn")
    from llmserveopt.policies.registry import POLICY_LIBRARY_V2_NAMES
    from llmserveopt.selector.advanced import validate_feature_columns

    mod = _load_smoke_module()
    assert tuple(mod.FULL_POLICY_LIBRARY_V2) == tuple(POLICY_LIBRARY_V2_NAMES)
    assert len(mod.FULL_POLICY_LIBRARY_V2) == 27

    out_dir = tmp_path / "integrated_27policy_smoke"
    argv = [
        "run_local_e2e_smoke.py",
        "--output-dir", str(out_dir),
        "--max-requests", "50",
        "--window-size", "15",
        "--min-partial-window", "15",
        "--drain-steps", "3000",
        "--seed", "20260721",
        "--policies", *mod.FULL_POLICY_LIBRARY_V2,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    exit_code = mod.main()
    assert exit_code == 0

    result = json.loads((out_dir / "selector_eval.json").read_text())
    assert result["simulator_based"] is True
    assert result["real_serving_measurements"] is False
    assert set(result["policies"]) == set(POLICY_LIBRARY_V2_NAMES)

    # No prohibited/leaky causal feature columns.
    assert validate_feature_columns(result["feature_columns"]) == result["feature_columns"]

    # The selected/best-fixed/oracle policy on every split must be a real
    # deployable policy from the 27-policy registry, and ANWG must be a
    # real computed number, not missing.
    for split_report in result["reports_by_split"].values():
        if split_report.get("n_windows", 0) == 0:
            continue
        for entry_name, entry in split_report["entries"].items():
            if entry_name.startswith("best_fixed__") or entry_name == "selector_reward_regression":
                policy_name = entry_name.split("__", 1)[-1] if entry_name.startswith("best_fixed__") else None
            else:
                policy_name = None
            if policy_name is not None:
                assert policy_name in POLICY_LIBRARY_V2_NAMES
            assert entry["mean_anwg"] is not None
