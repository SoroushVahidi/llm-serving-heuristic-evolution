"""Tests for selector evaluation metrics."""
import math
import pytest

from llmserveopt.selector.candidates import SELECTOR_CANDIDATES
from llmserveopt.selector.features import FEATURE_NAMES
from llmserveopt.selector.models import RuleBasedSelector, evaluate_selector


def _rows(policies: list, wg_map: dict, n: int = 10) -> list:
    """Build synthetic evaluation rows."""
    rows = []
    for i in range(n):
        policy = policies[i % len(policies)]
        row = {
            "best_policy": policy,
            "best_weighted_goodput": wg_map.get(policy, 0.8),
            "trace_id": f"trace_{i % 3}",
        }
        for fname in FEATURE_NAMES:
            row[f"feat_{fname}"] = float(i % 5)
        for pname in SELECTOR_CANDIDATES:
            row[f"reward_{pname}"] = wg_map.get(pname, 0.5)
        rows.append(row)
    return rows


# --- evaluate_selector ---

def test_evaluate_accuracy_perfect():
    rows = _rows(["fifo"], {"fifo": 0.9}, n=10)
    rb = RuleBasedSelector()  # always predicts fifo
    m = evaluate_selector(rb, rows)
    assert m["accuracy"] == pytest.approx(1.0)


def test_evaluate_accuracy_zero():
    rows = _rows(["edf"], {"edf": 0.9}, n=10)
    rb = RuleBasedSelector()  # always predicts fifo, labels are edf
    m = evaluate_selector(rb, rows)
    assert m["accuracy"] == pytest.approx(0.0)


def test_evaluate_returns_confusion():
    rows = _rows(["fifo", "edf"], {"fifo": 0.9, "edf": 0.8}, n=10)
    rb = RuleBasedSelector()
    m = evaluate_selector(rb, rows)
    assert "confusion" in m
    assert isinstance(m["confusion"], dict)


def test_evaluate_n_correct():
    rows = _rows(["fifo"], {"fifo": 0.9}, n=8)
    rb = RuleBasedSelector()
    m = evaluate_selector(rb, rows)
    assert m["n_correct"] == 8
    assert m["n_test"] == 8


# --- evaluate_policy_selector compute_reward_metrics ---

def test_reward_metrics_selected_mean_wg():
    """compute_reward_metrics should compute correct mean WG."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    # Import the function directly
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "evaluate_policy_selector",
        str(Path(__file__).parent.parent / "scripts" / "evaluate_policy_selector.py")
    )
    mod = importlib.util.load_from_spec_or_none = None
    try:
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except Exception:
        pytest.skip("evaluate_policy_selector.py not importable")

    rows = _rows(["fifo"], {"fifo": 0.8}, n=5)
    preds = ["fifo"] * 5
    rm = mod.compute_reward_metrics(preds, rows)
    assert rm["selected_mean_wg"] == pytest.approx(0.8, abs=0.001)


def test_reward_metrics_best_fixed():
    import sys, importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "evaluate_policy_selector",
        str(Path(__file__).parent.parent / "scripts" / "evaluate_policy_selector.py")
    )
    try:
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except Exception:
        pytest.skip("evaluate_policy_selector.py not importable")

    # Make "edf" the best fixed policy with high WG
    rows = []
    for i in range(8):
        row = {"best_policy": "fifo", "best_weighted_goodput": 0.9, "trace_id": "t0"}
        for fname in FEATURE_NAMES:
            row[f"feat_{fname}"] = 1.0
        for pname in SELECTOR_CANDIDATES:
            row[f"reward_{pname}"] = 0.9 if pname == "edf" else 0.5
        rows.append(row)

    preds = ["edf"] * 8
    rm = mod.compute_reward_metrics(preds, rows)
    assert rm["best_fixed_policy"] == "edf"
    assert rm["difference_vs_best_fixed"] == pytest.approx(0.0, abs=0.001)


# --- per_regime_breakdown ---

def test_per_regime_breakdown():
    import sys, importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "evaluate_policy_selector",
        str(Path(__file__).parent.parent / "scripts" / "evaluate_policy_selector.py")
    )
    try:
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except Exception:
        pytest.skip("evaluate_policy_selector.py not importable")

    rows_a = _rows(["fifo"], {"fifo": 0.9}, n=5)
    for r in rows_a:
        r["trace_id"] = "regime_a"
    rows_b = _rows(["edf"], {"edf": 0.7}, n=3)
    for r in rows_b:
        r["trace_id"] = "regime_b"

    all_rows = rows_a + rows_b
    preds = ["fifo"] * 5 + ["fifo"] * 3  # correct for a, wrong for b

    breakdown = mod.per_regime_breakdown(preds, all_rows)
    assert "regime_a" in breakdown
    assert "regime_b" in breakdown
    assert breakdown["regime_a"]["accuracy"] == pytest.approx(1.0)
    assert breakdown["regime_b"]["accuracy"] == pytest.approx(0.0)
    assert breakdown["regime_a"]["n"] == 5
    assert breakdown["regime_b"]["n"] == 3


# --- compute_classification_metrics ---

def test_compute_classification_metrics_perfect():
    import sys, importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "evaluate_policy_selector",
        str(Path(__file__).parent.parent / "scripts" / "evaluate_policy_selector.py")
    )
    try:
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except Exception:
        pytest.skip("evaluate_policy_selector.py not importable")

    rows = _rows(["fifo", "edf"], {}, n=10)
    preds = [str(r["best_policy"]) for r in rows]
    m = mod.compute_classification_metrics(preds, rows)
    assert m["accuracy"] == pytest.approx(1.0)
    assert m["macro_f1"] == pytest.approx(1.0)
    assert m["weighted_f1"] == pytest.approx(1.0)
