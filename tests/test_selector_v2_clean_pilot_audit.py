from __future__ import annotations

import csv
import importlib.util
from pathlib import Path


def _load_audit_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "audit_selector_v2_calibrated_pilot_leakage.py"
    spec = importlib.util.spec_from_file_location("audit_selector_v2_calibrated_pilot_leakage", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_independent_audit_rejects_cross_split_raw_row_overlap(tmp_path):
    mod = _load_audit_module()
    rows = [
        {
            "window_idx": 0,
            "group_key": "real_trace__burst__representative__historical",
            "split_group_key": "real_trace__burst__historical",
            "dataset_family": "real_trace",
            "source_trace": "burstgpt",
            "time_slice_pool": "historical",
            "request_plan_ancestor_id": "real_trace__burst",
            "time_slice_row_start": 10,
            "time_slice_row_end": 154,
            "split": "TRAIN",
        },
        {
            "window_idx": 1,
            "group_key": "real_trace__burst__compressed__historical",
            "split_group_key": "real_trace__burst__historical",
            "dataset_family": "real_trace",
            "source_trace": "burstgpt",
            "time_slice_pool": "historical",
            "request_plan_ancestor_id": "real_trace__burst",
            "time_slice_row_start": 10,
            "time_slice_row_end": 154,
            "split": "VALIDATION",
        },
    ]
    _write_csv(tmp_path / "retained_windows.csv", rows)
    _write_csv(tmp_path / "window_features.csv", [
        {"window_idx": 0, "feat_prompt_mean": 1.0},
        {"window_idx": 1, "feat_prompt_mean": 2.0},
    ])
    _write_csv(tmp_path / "full_policy_vectors.csv", [
        {"window_idx": 0, "policy_name": "fifo", "metric_arrival_normalized_weighted_goodput": 1.0},
        {"window_idx": 1, "policy_name": "fifo", "metric_arrival_normalized_weighted_goodput": 1.0},
    ])

    result = mod.audit(tmp_path)
    assert result["passed"] is False
    assert result["cross_split_row_overlap_pairs"] == 1
    assert "cross_split_row_overlap" in result["hard_failures"]
