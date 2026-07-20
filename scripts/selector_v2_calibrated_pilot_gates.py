"""Quality-gate evaluation for the Selector v2 calibrated targeted pilot.

Implements every required gate from the task's "QUALITY GATES" section,
operating purely on the pilot's own retained-window rows and per-window ANWG
vectors -- never on any faithful-baseline comparison (that gate no longer
applies under the Option B scope decision).
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from llmserveopt.selector.dataset_v2.splits import (  # noqa: E402
    verify_group_atomicity, verify_ood_holdout,
)

DISCRIMINATIVE_CLASSES = ("STRONGLY_DISCRIMINATIVE", "MODERATELY_DISCRIMINATIVE")


def evaluate_quality_gates(
    *,
    retained_rows: List[Dict],
    per_window_anwg: List[Dict[str, float]],
    n_retained_real_trace: int,
    n_retained_ood_reserved: int,
    split_counts: Dict[str, int],
) -> Dict:
    n = len(retained_rows)
    gates: Dict[str, Dict] = {}

    if n == 0:
        return {
            "all_gates_passed": False,
            "n_retained": 0,
            "gates": {"nonzero_retained_windows": {"passed": False, "value": 0}},
            "reason": "Zero windows retained -- no gate can be meaningfully evaluated.",
        }

    class_counts = Counter(r["primary_objective_classification"] for r in retained_rows)
    all_equiv = class_counts.get("ALL_COMPLETE_OR_EFFECTIVELY_TIED", 0)
    all_equiv_fraction = all_equiv / n
    gates["all_equivalent_fraction_below_40pct"] = {
        "passed": all_equiv_fraction < 0.40, "value": round(all_equiv_fraction, 4), "threshold": 0.40,
    }

    # Best-fixed policy (dataset-wide mean ANWG) and oracle headroom.
    values_by_policy: Dict[str, List[float]] = {}
    for w in per_window_anwg:
        for pname, v in w.items():
            values_by_policy.setdefault(pname, []).append(v)
    means = {name: sum(vals) / len(vals) for name, vals in values_by_policy.items() if vals}
    best_fixed_name = max(means, key=means.get) if means else None
    best_fixed_mean = means.get(best_fixed_name) if best_fixed_name else None

    oracle_vals, best_fixed_vals = [], []
    disc_oracle_vals, disc_best_fixed_vals = [], []
    for w, row in zip(per_window_anwg, retained_rows):
        if not w or best_fixed_name not in w:
            continue
        oracle_vals.append(max(w.values()))
        best_fixed_vals.append(w[best_fixed_name])
        if row["primary_objective_classification"] in DISCRIMINATIVE_CLASSES:
            disc_oracle_vals.append(max(w.values()))
            disc_best_fixed_vals.append(w[best_fixed_name])

    oracle_headroom = (
        (sum(oracle_vals) / len(oracle_vals)) - (sum(best_fixed_vals) / len(best_fixed_vals))
        if oracle_vals else None
    )
    discriminative_oracle_headroom = (
        (sum(disc_oracle_vals) / len(disc_oracle_vals)) - (sum(disc_best_fixed_vals) / len(disc_best_fixed_vals))
        if disc_oracle_vals else None
    )
    gates["oracle_headroom_at_least_0.01"] = {
        "passed": oracle_headroom is not None and oracle_headroom >= 0.01,
        "value": round(oracle_headroom, 4) if oracle_headroom is not None else None, "threshold": 0.01,
    }
    gates["discriminative_oracle_headroom_at_least_0.03"] = {
        "passed": discriminative_oracle_headroom is not None and discriminative_oracle_headroom >= 0.03,
        "value": round(discriminative_oracle_headroom, 4) if discriminative_oracle_headroom is not None else None,
        "threshold": 0.03,
    }

    # Win diversity.
    meaningful_win_rows = [r for r in retained_rows if r["primary_objective_classification"] in DISCRIMINATIVE_CLASSES]
    meaningful_win_counts = Counter(r["primary_objective_best_policy"] for r in meaningful_win_rows)
    n_policies_meaningful_wins = len(meaningful_win_counts)
    gates["at_least_3_policies_with_meaningful_wins"] = {
        "passed": n_policies_meaningful_wins >= 3, "value": n_policies_meaningful_wins,
        "distribution": dict(meaningful_win_counts), "threshold": 3,
    }

    strong_win_rows = [r for r in retained_rows if r["primary_objective_classification"] == "STRONGLY_DISCRIMINATIVE"]
    strong_win_counts = Counter(r["primary_objective_best_policy"] for r in strong_win_rows)
    top_strong_share = (max(strong_win_counts.values()) / sum(strong_win_counts.values())) if strong_win_counts else 0.0
    gates["no_policy_above_85pct_of_strong_wins"] = {
        "passed": (not strong_win_counts) or top_strong_share <= 0.85,
        "value": round(top_strong_share, 4), "threshold": 0.85, "distribution": dict(strong_win_counts),
    }

    # Saturation check.
    n_all_success = sum(1 for w in per_window_anwg if w and min(w.values()) >= 0.999)
    n_all_fail = sum(1 for w in per_window_anwg if w and max(w.values()) <= 0.001)
    all_success_fraction = n_all_success / n
    all_fail_fraction = n_all_fail / n
    gates["no_universal_success_or_failure_saturation"] = {
        "passed": all_success_fraction < 0.95 and all_fail_fraction < 0.5,
        "all_success_fraction": round(all_success_fraction, 4),
        "all_fail_fraction": round(all_fail_fraction, 4),
        "thresholds": {"all_success_fraction_below": 0.95, "all_fail_fraction_below": 0.5},
    }

    gates["real_trace_representation_exists"] = {
        "passed": n_retained_real_trace > 0, "value": n_retained_real_trace,
    }
    gates["newer_time_slice_representation_exists"] = {
        "passed": n_retained_ood_reserved > 0, "value": n_retained_ood_reserved,
    }
    gates["ood_split_exists"] = {
        "passed": split_counts.get("OOD_TEST", 0) > 0, "value": split_counts.get("OOD_TEST", 0),
    }

    leakage_ok = True
    leakage_detail = "verified"
    try:
        verify_group_atomicity(retained_rows, group_key_field="group_key", split_field="split")
        ood_groups = {r["group_key"] for r in retained_rows if r["time_slice_pool"] == "ood_reserved"}
        verify_ood_holdout(retained_rows, group_key_field="group_key", ood_group_keys=ood_groups, split_field="split")
    except ValueError as e:
        leakage_ok = False
        leakage_detail = str(e)
    gates["no_leakage"] = {"passed": leakage_ok, "detail": leakage_detail}

    all_passed = all(g["passed"] for g in gates.values())
    return {
        "all_gates_passed": all_passed,
        "n_retained": n,
        "best_fixed_policy": best_fixed_name,
        "best_fixed_policy_mean_anwg": round(best_fixed_mean, 4) if best_fixed_mean is not None else None,
        "primary_objective_classification_counts": dict(class_counts),
        "gates": gates,
    }
