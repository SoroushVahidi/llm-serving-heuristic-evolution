"""Tests for training script outputs (smoke check on pre-built artifacts)."""
import csv
import json
import pytest
from pathlib import Path

MODELS_DIR = Path("results/phase2a3_selector_eval/models")
EVAL_DIR = Path("results/phase2a3_selector_eval/evaluation")


def _skip_if_missing(path: Path):
    if not path.exists():
        pytest.skip(f"Not found (build datasets first): {path}")


# --- training outputs ---

def test_rule_based_metrics_exist():
    p = MODELS_DIR / "rule_based" / "metrics.json"
    _skip_if_missing(p)
    with open(p) as f:
        m = json.load(f)
    assert "train" in m or "validation" in m


def test_decision_tree_model_file():
    pytest.importorskip("sklearn")
    pytest.importorskip("joblib")
    p = MODELS_DIR / "decision_tree" / "model.joblib"
    _skip_if_missing(p)
    import joblib
    model = joblib.load(p)
    assert hasattr(model, "predict")


def test_decision_tree_feature_importance_csv():
    pytest.importorskip("sklearn")
    p = MODELS_DIR / "decision_tree" / "feature_importance.csv"
    _skip_if_missing(p)
    rows = list(csv.DictReader(open(p)))
    assert len(rows) == 18, f"Expected 18 features, got {len(rows)}"
    importances = [float(r["importance"]) for r in rows]
    assert abs(sum(importances) - 1.0) < 1e-4


def test_random_forest_model_file():
    pytest.importorskip("sklearn")
    pytest.importorskip("joblib")
    p = MODELS_DIR / "random_forest" / "model.joblib"
    _skip_if_missing(p)
    import joblib
    model = joblib.load(p)
    assert hasattr(model, "predict")


def test_random_forest_feature_importance_csv():
    pytest.importorskip("sklearn")
    p = MODELS_DIR / "random_forest" / "feature_importance.csv"
    _skip_if_missing(p)
    rows = list(csv.DictReader(open(p)))
    assert len(rows) == 18
    importances = [float(r["importance"]) for r in rows]
    assert abs(sum(importances) - 1.0) < 1e-4


def test_training_summary_json():
    p = MODELS_DIR / "training_summary.json"
    _skip_if_missing(p)
    with open(p) as f:
        data = json.load(f)
    assert "models" in data
    assert data["train_n"] > 0


# --- evaluation outputs ---

def test_evaluation_summary_csv():
    p = EVAL_DIR / "summary.csv"
    _skip_if_missing(p)
    rows = list(csv.DictReader(open(p)))
    assert len(rows) > 0
    for row in rows:
        assert "model" in row
        assert "split" in row
        assert "selected_mean_wg" in row


def test_evaluation_full_json():
    p = EVAL_DIR / "evaluation_full.json"
    _skip_if_missing(p)
    with open(p) as f:
        data = json.load(f)
    for model_name in ["rule_based", "decision_tree", "random_forest"]:
        if model_name in data:
            for split_name in ["validation", "test"]:
                if split_name in data[model_name]:
                    s = data[model_name][split_name]
                    assert "accuracy" in s
                    assert "selected_mean_wg" in s


def test_random_forest_beats_rule_based_on_test():
    """RF should have higher selected WG than rule_based on test set."""
    p = EVAL_DIR / "evaluation_full.json"
    _skip_if_missing(p)
    with open(p) as f:
        data = json.load(f)
    rf_wg = data.get("random_forest", {}).get("test", {}).get("selected_mean_wg", 0)
    rb_wg = data.get("rule_based", {}).get("test", {}).get("selected_mean_wg", 1)
    if isinstance(rf_wg, str) or isinstance(rb_wg, str):
        pytest.skip("WG values not available")
    assert float(rf_wg) > float(rb_wg), (
        f"RF WG ({rf_wg}) should beat rule_based WG ({rb_wg})"
    )
