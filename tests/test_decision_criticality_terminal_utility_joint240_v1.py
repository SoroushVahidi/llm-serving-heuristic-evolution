"""Tests for joint-240 terminal utility replay / continuous metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd

from llmserveopt.analysis.decision_criticality_terminal_utility_joint240_v1 import (
    MEANINGFUL_EPS,
    bootstrap_scenario_stats,
    concentration_curve,
    metrics_from_request_rows,
    scenario_top_k_share_mult,
)


def _rows(completed_times, deadlines, weights=None, arrivals=None):
    n = len(completed_times)
    weights = weights or [1.0] * n
    arrivals = arrivals or [0.0] * n
    out = []
    for i in range(n):
        C = completed_times[i]
        D = deadlines[i]
        A = arrivals[i]
        T = max(0.0, C - D)
        S = max(D - A, 1e-12)
        out.append(
            {
                "weight": weights[i],
                "arrival_time": A,
                "deadline": D,
                "completion_time": C,
                "completed": True,
                "dropped": False,
                "tardiness": T,
                "slo_window": S,
            }
        )
    return out


def test_anwg_wcg_wmt_soft_basic():
    # two requests: one on time, one late by 2; equal weights
    rows = _rows([1.0, 5.0], [2.0, 3.0], weights=[1.0, 1.0], arrivals=[0.0, 0.0])
    m = metrics_from_request_rows(rows)
    assert abs(m["wcg"] - 1.0) < 1e-12
    assert abs(m["anwg"] - 0.5) < 1e-12  # only first meets deadline
    assert abs(m["wmt"] - 1.0) < 1e-12  # (0 + 2) / 2
    assert m["wnt"] > 0
    assert 0 < m["soft"] < 1


def test_unfinished_uses_sim_duration_as_C():
    rows = [
        {
            "weight": 1.0,
            "arrival_time": 0.0,
            "deadline": 1.0,
            "completion_time": 10.0,  # unfinished convention
            "completed": False,
            "dropped": True,
            "tardiness": 9.0,
            "slo_window": 1.0,
        }
    ]
    m = metrics_from_request_rows(rows)
    assert m["wcg"] == 0.0
    assert m["anwg"] == 0.0
    assert m["soft"] == 0.0
    assert m["wmt"] == 9.0


def test_bootstrap_retains_scenario_multiplicity():
    """Regression: with-replacement bootstrap must not collapse duplicate scenarios."""
    rows = []
    # one scenario dominates mass
    for i in range(20):
        rows.append(
            {
                "scenario_id": "A",
                "delta_anwg_live": 1.0 if i == 0 else 0.0,
                "acquisition_type": "DISAGREEMENT",
            }
        )
    for i in range(20):
        rows.append(
            {
                "scenario_id": "B",
                "delta_anwg_live": 0.01 if i == 0 else 0.0,
                "acquisition_type": "AGREEMENT_CONTROL",
            }
        )
    df = pd.DataFrame(rows)
    # point top-1 scenario share ~ 1/1.01
    mass = df.groupby("scenario_id")["delta_anwg_live"].apply(lambda s: float(np.abs(s).sum())).to_numpy()
    point = scenario_top_k_share_mult(mass, 1)
    boot = bootstrap_scenario_stats(df, effect_col="delta_anwg_live", n_boot=200, seed=20260825)
    lo = boot["top5_scenario_mass"]["ci95_low"]
    hi = boot["top5_scenario_mass"]["ci95_high"]
    # With only 2 scenarios, top5 == all mass share = 1.0 always if multiplicity ok / total>0
    assert boot["multiplicity_retained"] is True
    assert lo is not None and hi is not None
    assert point > 0.9


def test_concentration_zero_mass_rule():
    c = concentration_curve(np.zeros(10))
    assert c["0.01"]["share"] == 0.0
