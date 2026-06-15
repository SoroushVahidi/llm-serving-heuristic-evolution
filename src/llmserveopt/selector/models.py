"""
Baseline selector model wrappers.

v1 selector models
------------------
* RuleBasedSelector   — always picks the policy with the highest predicted throughput
  from features; simple placeholder for comparison.
* DecisionTreeSelector
* RandomForestSelector

scikit-learn is optional.  When not installed, tree/forest classes raise ImportError
with a clear message; the rule-based selector always works.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .candidates import SELECTOR_CANDIDATES
from .features import FEATURE_NAMES


class RuleBasedSelector:
    """Placeholder: always recommends 'fifo' (fixed, deterministic baseline)."""
    name = "rule_based"

    def predict(self, features: List[Dict[str, float]]) -> List[str]:
        return ["fifo"] * len(features)

    def predict_one(self, features: Dict[str, float]) -> str:
        return "fifo"


def _check_sklearn():
    try:
        import sklearn  # noqa: F401
    except ImportError:
        raise ImportError(
            "scikit-learn is required for tree/forest selectors. "
            "Install with: pip install scikit-learn"
        )


def _feature_matrix(rows: List[Dict[str, float]]) -> np.ndarray:
    return np.array([[r.get(f"feat_{n}", 0.0) for n in FEATURE_NAMES] for r in rows], dtype=float)


def _labels(rows: List[Dict]) -> List[str]:
    return [r["best_policy"] for r in rows]


class DecisionTreeSelector:
    """DecisionTreeClassifier(max_depth=8, min_samples_leaf=20)."""
    name = "decision_tree"

    def __init__(self, max_depth: int = 8, min_samples_leaf: int = 20, random_state: int = 42):
        _check_sklearn()
        from sklearn.tree import DecisionTreeClassifier
        self._clf = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )

    def fit(self, rows: List[Dict]) -> "DecisionTreeSelector":
        X = _feature_matrix(rows)
        y = _labels(rows)
        self._clf.fit(X, y)
        return self

    def predict(self, rows: List[Dict]) -> List[str]:
        X = _feature_matrix(rows)
        return list(self._clf.predict(X))

    def feature_importances(self) -> Dict[str, float]:
        return dict(zip(FEATURE_NAMES, self._clf.feature_importances_))

    def tree_text(self) -> str:
        from sklearn.tree import export_text
        return export_text(self._clf, feature_names=FEATURE_NAMES)

    def save(self, path: str) -> None:
        import joblib
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._clf, path)

    @classmethod
    def load(cls, path: str) -> "DecisionTreeSelector":
        import joblib
        obj = cls.__new__(cls)
        obj._clf = joblib.load(path)
        return obj


class RandomForestSelector:
    """RandomForestClassifier(n_estimators=200, max_depth=10)."""
    name = "random_forest"

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 10,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        _check_sklearn()
        from sklearn.ensemble import RandomForestClassifier
        self._clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    def fit(self, rows: List[Dict]) -> "RandomForestSelector":
        X = _feature_matrix(rows)
        y = _labels(rows)
        self._clf.fit(X, y)
        return self

    def predict(self, rows: List[Dict]) -> List[str]:
        X = _feature_matrix(rows)
        return list(self._clf.predict(X))

    def feature_importances(self) -> Dict[str, float]:
        return dict(zip(FEATURE_NAMES, self._clf.feature_importances_))

    def save(self, path: str) -> None:
        import joblib
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._clf, path)

    @classmethod
    def load(cls, path: str) -> "RandomForestSelector":
        import joblib
        obj = cls.__new__(cls)
        obj._clf = joblib.load(path)
        return obj


def evaluate_selector(
    selector,
    test_rows: List[Dict],
) -> Dict:
    """Return accuracy and per-class counts."""
    preds = selector.predict(test_rows)
    labels = _labels(test_rows)
    n = len(labels)
    correct = sum(p == l for p, l in zip(preds, labels))
    acc = correct / n if n > 0 else 0.0

    # Confusion: predicted → actual counts
    from collections import Counter
    confusion: Dict[str, Counter] = {}
    for p, l in zip(preds, labels):
        confusion.setdefault(p, Counter())[l] += 1

    return {
        "accuracy": acc,
        "n_test": n,
        "n_correct": correct,
        "confusion": {k: dict(v) for k, v in confusion.items()},
    }


def save_metrics(metrics: Dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
