from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


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
