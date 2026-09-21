"""Simple contextual models for ESTF/WFS composition falsification."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from .estf_wfs_features import FEATURE_NAMES, assert_no_hidden_leakage, feature_vector

ALPHA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
PARENT_ESTF = "estimated_service_time_first"
PARENT_WFS = "weighted_fair_share"


@dataclass
class FittedSelector:
    model_type: str
    feature_names: Tuple[str, ...]
    classes_: List[str]  # ["wfs", "estf"]
    # sklearn objects stored separately
    raw: Any

    def predict_parent(self, features: Mapping[str, float]) -> str:
        assert_no_hidden_leakage(features)
        x = feature_vector(features, self.feature_names).reshape(1, -1)
        pred = int(self.raw.predict(x)[0])
        return self.classes_[pred]

    def predict_proba_estf(self, features: Mapping[str, float]) -> float:
        assert_no_hidden_leakage(features)
        x = feature_vector(features, self.feature_names).reshape(1, -1)
        if hasattr(self.raw, "predict_proba"):
            proba = self.raw.predict_proba(x)[0]
            # class index of estf
            estf_idx = self.classes_.index("estf")
            return float(proba[estf_idx])
        return 1.0 if self.predict_parent(features) == "estf" else 0.0


@dataclass
class FittedAlphaModel:
    model_type: str
    feature_names: Tuple[str, ...]
    alpha_grid: Tuple[float, ...]
    raw: Any

    def predict_alpha(self, features: Mapping[str, float]) -> float:
        assert_no_hidden_leakage(features)
        x = feature_vector(features, self.feature_names).reshape(1, -1)
        idx = int(self.raw.predict(x)[0])
        return float(self.alpha_grid[idx])


def _parent_label(estf: float, wfs: float, eps: float = 1e-12) -> int:
    """1 = ESTF preferred, 0 = WFS preferred (ties → WFS to break symmetry)."""
    return 1 if estf > wfs + eps else 0


def _alpha_label(estf: float, wfs: float, grid: Sequence[float] = ALPHA_GRID) -> int:
    """Map parent ANWG margin to a discrete alpha index.

    Larger ESTF advantage → higher alpha (more ESTF weight).
    """
    delta = float(estf) - float(wfs)
    if abs(delta) <= 0.01:
        target = 0.5
    elif delta > 0.05:
        target = 1.0
    elif delta > 0.0:
        target = 0.75
    elif delta < -0.05:
        target = 0.0
    else:
        target = 0.25
    dists = [abs(target - a) for a in grid]
    return int(np.argmin(dists))


class _ConstantClassifier:
    """Fallback when training labels contain a single class."""

    def __init__(self, label: int) -> None:
        self.label = int(label)

    def fit(self, X, y):  # noqa: ANN001
        return self

    def predict(self, X):  # noqa: ANN001
        import numpy as np

        return np.full(shape=(len(X),), fill_value=self.label, dtype=int)

    def predict_proba(self, X):  # noqa: ANN001
        import numpy as np

        n = len(X)
        # binary-looking; callers using predict_proba on selector only
        proba = np.zeros((n, 2), dtype=float)
        idx = 0 if self.label == 0 else 1
        proba[:, idx] = 1.0
        return proba


def fit_top1_selector(
    feature_rows: Sequence[Mapping[str, float]],
    estf_scores: Sequence[float],
    wfs_scores: Sequence[float],
    *,
    model_type: str = "logreg",
    seed: int = 20260816,
) -> FittedSelector:
    X = np.vstack([feature_vector(f) for f in feature_rows])
    y = np.asarray(
        [_parent_label(e, w) for e, w in zip(estf_scores, wfs_scores)], dtype=int
    )
    if len(set(int(v) for v in y)) < 2:
        clf: Any = _ConstantClassifier(int(y[0]))
        clf.fit(X, y)
    elif model_type == "tree":
        clf = DecisionTreeClassifier(
            max_depth=3, min_samples_leaf=max(2, len(y) // 10), random_state=seed
        )
        clf.fit(X, y)
    else:
        clf = LogisticRegression(max_iter=2000, random_state=seed)
        clf.fit(X, y)
    return FittedSelector(
        model_type=model_type,
        feature_names=FEATURE_NAMES,
        classes_=["wfs", "estf"],
        raw=clf,
    )


def fit_alpha_model(
    feature_rows: Sequence[Mapping[str, float]],
    estf_scores: Sequence[float],
    wfs_scores: Sequence[float],
    *,
    model_type: str = "tree",
    seed: int = 20260816,
    alpha_grid: Sequence[float] = ALPHA_GRID,
) -> FittedAlphaModel:
    X = np.vstack([feature_vector(f) for f in feature_rows])
    y = np.asarray(
        [_alpha_label(e, w, alpha_grid) for e, w in zip(estf_scores, wfs_scores)],
        dtype=int,
    )
    if len(set(int(v) for v in y)) < 2:
        clf: Any = _ConstantClassifier(int(y[0]))
        clf.fit(X, y)
    elif model_type == "logreg":
        clf = LogisticRegression(max_iter=2000, random_state=seed)
        clf.fit(X, y)
    else:
        clf = DecisionTreeClassifier(
            max_depth=3, min_samples_leaf=max(2, len(y) // 12), random_state=seed
        )
        clf.fit(X, y)
    return FittedAlphaModel(
        model_type=model_type,
        feature_names=FEATURE_NAMES,
        alpha_grid=tuple(float(a) for a in alpha_grid),
        raw=clf,
    )


def select_model_on_val(
    feature_train: Sequence[Mapping[str, float]],
    estf_train: Sequence[float],
    wfs_train: Sequence[float],
    feature_val: Sequence[Mapping[str, float]],
    estf_val: Sequence[float],
    wfs_val: Sequence[float],
) -> Tuple[FittedSelector, FittedAlphaModel, Dict[str, Any]]:
    """Fit candidate models; pick by validation accuracy / discrete-alpha regret."""
    candidates = []
    for mt in ("logreg", "tree"):
        sel = fit_top1_selector(feature_train, estf_train, wfs_train, model_type=mt)
        correct = 0
        for f, e, w in zip(feature_val, estf_val, wfs_val):
            pred = sel.predict_parent(f)
            truth = "estf" if e > w + 1e-12 else "wfs"
            correct += int(pred == truth)
        acc = correct / max(1, len(feature_val))
        candidates.append((acc, mt, sel))
    candidates.sort(key=lambda t: (-t[0], t[1]))
    best_sel = candidates[0][2]

    alpha_cands = []
    for mt in ("tree", "logreg"):
        am = fit_alpha_model(feature_train, estf_train, wfs_train, model_type=mt)
        # validation: predicted alpha vs oracle parent (regret of static blend approx)
        # Use discrete accuracy to nearest parent: alpha>=0.5 → ESTF else WFS
        correct = 0
        for f, e, w in zip(feature_val, estf_val, wfs_val):
            a = am.predict_alpha(f)
            pred = "estf" if a >= 0.5 else "wfs"
            truth = "estf" if e > w + 1e-12 else "wfs"
            correct += int(pred == truth)
        acc = correct / max(1, len(feature_val))
        alpha_cands.append((acc, mt, am))
    alpha_cands.sort(key=lambda t: (-t[0], t[1]))
    best_alpha = alpha_cands[0][2]
    meta = {
        "selector_val_accuracy": candidates[0][0],
        "selector_model_type": candidates[0][1],
        "alpha_val_proxy_accuracy": alpha_cands[0][0],
        "alpha_model_type": alpha_cands[0][1],
        "selector_candidates": [
            {"model_type": mt, "val_accuracy": acc} for acc, mt, _ in candidates
        ],
        "alpha_candidates": [
            {"model_type": mt, "val_proxy_accuracy": acc} for acc, mt, _ in alpha_cands
        ],
    }
    return best_sel, best_alpha, meta


def hard_conditional_rule(features: Mapping[str, float]) -> str:
    """Simple interpretable if/else: size pressure vs priority pressure.

    - If short-job fraction high and priority skew low → ESTF
    - If priority skew high and class imbalance high → WFS
    - Else prefer ESTF when short_job_fraction >= 0.5
    """
    assert_no_hidden_leakage(features)
    short = float(features.get("short_job_fraction", 0.5))
    skew = float(features.get("priority_skew", 1.0))
    imb = float(features.get("class_imbalance", 0.5))
    if skew >= 4.0 and imb >= 0.45:
        return "wfs"
    if short >= 0.55 and skew <= 2.0:
        return "estf"
    return "estf" if short >= 0.5 else "wfs"


def save_model_meta(path: Path, meta: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(meta), indent=2, sort_keys=True) + "\n", encoding="utf-8")
