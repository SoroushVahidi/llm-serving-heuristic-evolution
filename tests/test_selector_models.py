"""Tests for selector model wrappers (using tiny synthetic CSV data)."""
import pytest
import math

from llmserveopt.selector.candidates import SELECTOR_CANDIDATES
from llmserveopt.selector.features import FEATURE_NAMES
from llmserveopt.selector.models import (
    RuleBasedSelector,
    evaluate_selector,
    save_metrics,
)


def _make_rows(n: int = 20, policy: str = "fifo") -> list:
    """Synthetic dataset rows with all columns."""
    rows = []
    for i in range(n):
        row = {f"feat_{name}": float(i % 5) for name in FEATURE_NAMES}
        row["best_policy"] = SELECTOR_CANDIDATES[i % len(SELECTOR_CANDIDATES)]
        row["best_weighted_goodput"] = 0.8
        for pname in SELECTOR_CANDIDATES:
            row[f"reward_{pname}"] = 0.5
        rows.append(row)
    return rows


# --- RuleBasedSelector ---

def test_rule_based_predict_length():
    rows = _make_rows(10)
    sel = RuleBasedSelector()
    preds = sel.predict(rows)
    assert len(preds) == 10


def test_rule_based_predict_one():
    row = _make_rows(1)[0]
    sel = RuleBasedSelector()
    pred = sel.predict_one(row)
    assert pred in SELECTOR_CANDIDATES


def test_rule_based_evaluate():
    rows = _make_rows(20)
    # Force label to "fifo" so RuleBasedSelector gets all correct
    for r in rows:
        r["best_policy"] = "fifo"
    sel = RuleBasedSelector()
    metrics = evaluate_selector(sel, rows)
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["n_test"] == 20


def test_evaluate_partial_accuracy():
    rows = _make_rows(20)
    for i, r in enumerate(rows):
        r["best_policy"] = "fifo" if i < 10 else "edf"
    sel = RuleBasedSelector()  # always predicts "fifo"
    metrics = evaluate_selector(sel, rows)
    assert metrics["accuracy"] == pytest.approx(0.5)


# --- save_metrics ---

def test_save_metrics(tmp_path):
    metrics = {"accuracy": 0.85, "n_test": 100}
    path = str(tmp_path / "metrics.json")
    save_metrics(metrics, path)
    import json
    with open(path) as f:
        loaded = json.load(f)
    assert loaded["accuracy"] == pytest.approx(0.85)


# --- DecisionTree (only if sklearn available) ---

def test_decision_tree_if_sklearn():
    pytest.importorskip("sklearn")
    from llmserveopt.selector.models import DecisionTreeSelector

    rows = _make_rows(50)
    dt = DecisionTreeSelector(max_depth=4, min_samples_leaf=5, random_state=0)
    dt.fit(rows)
    preds = dt.predict(rows)
    assert len(preds) == 50
    assert all(p in SELECTOR_CANDIDATES for p in preds)


def test_decision_tree_feature_importances():
    pytest.importorskip("sklearn")
    from llmserveopt.selector.models import DecisionTreeSelector

    rows = _make_rows(50)
    dt = DecisionTreeSelector(max_depth=4, min_samples_leaf=5, random_state=0)
    dt.fit(rows)
    fi = dt.feature_importances()
    assert set(fi.keys()) == set(FEATURE_NAMES)
    assert abs(sum(fi.values()) - 1.0) < 1e-6


def test_decision_tree_save_load(tmp_path):
    pytest.importorskip("sklearn")
    pytest.importorskip("joblib")
    from llmserveopt.selector.models import DecisionTreeSelector

    rows = _make_rows(50)
    dt = DecisionTreeSelector(max_depth=4, min_samples_leaf=5, random_state=0)
    dt.fit(rows)
    path = str(tmp_path / "dt.joblib")
    dt.save(path)
    dt2 = DecisionTreeSelector.load(path)
    preds1 = dt.predict(rows)
    preds2 = dt2.predict(rows)
    assert preds1 == preds2


# --- RandomForest (only if sklearn available) ---

def test_random_forest_if_sklearn():
    pytest.importorskip("sklearn")
    from llmserveopt.selector.models import RandomForestSelector

    rows = _make_rows(60)
    rf = RandomForestSelector(n_estimators=10, max_depth=4, random_state=0, n_jobs=1)
    rf.fit(rows)
    preds = rf.predict(rows)
    assert len(preds) == 60
    assert all(p in SELECTOR_CANDIDATES for p in preds)


def test_random_forest_feature_importances():
    pytest.importorskip("sklearn")
    from llmserveopt.selector.models import RandomForestSelector

    rows = _make_rows(60)
    rf = RandomForestSelector(n_estimators=10, max_depth=4, random_state=0, n_jobs=1)
    rf.fit(rows)
    fi = rf.feature_importances()
    assert set(fi.keys()) == set(FEATURE_NAMES)
    assert abs(sum(fi.values()) - 1.0) < 1e-6
