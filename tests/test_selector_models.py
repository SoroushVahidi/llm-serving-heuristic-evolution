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
    """RuleBasedSelector predictions are all valid policy names; check n_test."""
    rows = _make_rows(20)
    sel = RuleBasedSelector()
    metrics = evaluate_selector(sel, rows)
    assert metrics["n_test"] == 20
    # accuracy is 0-1, valid regardless of label distribution
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_rule_based_all_predictions_valid():
    """Every prediction is a valid selector candidate."""
    rows = _make_rows(20)
    sel = RuleBasedSelector()
    preds = sel.predict(rows)
    for p in preds:
        assert p in SELECTOR_CANDIDATES, f"Invalid prediction: {p}"


def test_evaluate_partial_accuracy():
    """evaluate_selector computes correct accuracy for any selector."""
    rows = _make_rows(20)
    sel = RuleBasedSelector()
    preds = sel.predict(rows)
    # Count how many labels match predictions
    labels = [r["best_policy"] for r in rows]
    expected_acc = sum(p == l for p, l in zip(preds, labels)) / 20
    metrics = evaluate_selector(sel, rows)
    assert metrics["accuracy"] == pytest.approx(expected_acc)


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


# --- PerPolicyRegressionAnwgSelector (only if sklearn available) ---

def _make_anwg_rows(n: int = 60) -> list:
    """Synthetic rows with completion_/reward_ columns for every candidate,
    matching the schema PerPolicyRegressionAnwgSelector trains on."""
    rows = []
    for i in range(n):
        row = {f"feat_{name}": float((i * 7 + idx) % 11) for idx, name in enumerate(FEATURE_NAMES)}
        for pname in SELECTOR_CANDIDATES:
            row[f"completion_{pname}"] = 1.0
            row[f"reward_{pname}"] = 0.5 + 0.01 * (hash((i, pname)) % 10)
        rows.append(row)
    return rows


def test_regression_anwg_predict_length():
    pytest.importorskip("sklearn")
    from llmserveopt.selector.models import PerPolicyRegressionAnwgSelector

    rows = _make_anwg_rows(60)
    sel = PerPolicyRegressionAnwgSelector(n_estimators=10, max_depth=4, random_state=0)
    sel.fit(rows)
    preds = sel.predict(rows)
    assert len(preds) == 60
    assert all(p in SELECTOR_CANDIDATES for p in preds)


def test_regression_anwg_predict_one():
    pytest.importorskip("sklearn")
    from llmserveopt.selector.models import PerPolicyRegressionAnwgSelector

    rows = _make_anwg_rows(60)
    sel = PerPolicyRegressionAnwgSelector(n_estimators=10, max_depth=4, random_state=0)
    sel.fit(rows)
    pred = sel.predict_one(rows[0])
    assert pred in SELECTOR_CANDIDATES


def test_regression_anwg_is_feature_only_at_predict_time():
    """predict() must not require completion_/reward_ (hindsight) columns --
    only feat_* columns. This is the online-observable-field guarantee."""
    pytest.importorskip("sklearn")
    from llmserveopt.selector.models import PerPolicyRegressionAnwgSelector

    rows = _make_anwg_rows(60)
    sel = PerPolicyRegressionAnwgSelector(n_estimators=10, max_depth=4, random_state=0)
    sel.fit(rows)

    feature_only_rows = [
        {k: v for k, v in r.items() if k.startswith("feat_")} for r in rows
    ]
    preds_full = sel.predict(rows)
    preds_feature_only = sel.predict(feature_only_rows)
    assert preds_full == preds_feature_only


def test_regression_anwg_save_load(tmp_path):
    pytest.importorskip("sklearn")
    pytest.importorskip("joblib")
    from llmserveopt.selector.models import PerPolicyRegressionAnwgSelector

    rows = _make_anwg_rows(60)
    sel = PerPolicyRegressionAnwgSelector(n_estimators=10, max_depth=4, random_state=0)
    sel.fit(rows)
    path = str(tmp_path / "regression_anwg.joblib")
    sel.save(path)
    sel2 = PerPolicyRegressionAnwgSelector.load(path)
    assert sel.predict(rows) == sel2.predict(rows)


def test_regression_anwg_load_rejects_wrong_type(tmp_path):
    pytest.importorskip("sklearn")
    pytest.importorskip("joblib")
    import joblib
    from llmserveopt.selector.models import PerPolicyRegressionAnwgSelector, RandomForestSelector

    rows = _make_rows(50)
    rf = RandomForestSelector(n_estimators=5, max_depth=3, random_state=0, n_jobs=1)
    rf.fit(rows)
    path = str(tmp_path / "wrong_type.joblib")
    joblib.dump(rf._clf, path)  # not a PerPolicyRegressionAnwgSelector instance

    with pytest.raises(TypeError):
        PerPolicyRegressionAnwgSelector.load(path)
