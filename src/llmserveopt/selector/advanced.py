"""Advanced causal selector formulations for corrected-objective ANWG.

These selectors are intentionally small scikit-learn wrappers around
feature-only prediction. Training may use simulator-derived reward columns,
but prediction consumes only causal ``feat_*`` columns supplied by the caller.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


LEAKY_FEATURE_TOKENS: Tuple[str, ...] = (
    "reward_",
    "completion_",
    "anwg_",
    "best_",
    "label",
    "oracle",
    "sel_",
    "selected_",
    "external_",
)


def validate_feature_columns(feature_cols: Sequence[str]) -> List[str]:
    """Return validated causal feature columns.

    The Phase 2C labeled dataset uses ``feat_*`` columns for causal features.
    This guard rejects post-hoc rewards, labels, selectors, and oracle fields.
    """
    cols = list(feature_cols)
    bad = [
        c for c in cols
        if not c.startswith("feat_")
        or any(token in c.lower() for token in LEAKY_FEATURE_TOKENS)
    ]
    if bad:
        raise ValueError(f"Leaky or non-causal feature columns detected: {sorted(bad)}")
    if not cols:
        raise ValueError("At least one causal feat_* column is required")
    return cols


def anwg_column(policy: str) -> str:
    return f"anwg_{policy}"


def anwg_value(row: Mapping[str, object], policy: str) -> float:
    col = anwg_column(policy)
    if col in row:
        return float(row[col] or 0.0)
    reward = float(row.get(f"reward_{policy}", 0.0) or 0.0)
    completion = float(row.get(f"completion_{policy}", 0.0) or 0.0)
    return reward * completion


def policy_margin_weights(
    rows: Sequence[Mapping[str, object]],
    allowed_policies: Sequence[str],
    *,
    scheme: str = "margin_plus_epsilon",
    epsilon: float = 0.001,
    power: float = 1.0,
) -> np.ndarray:
    """Compute regret-aware sample weights from realized policy margins.

    The weights use only training-label outcome vectors and are never needed at
    prediction time. Near-tie windows receive low weight.
    """
    margins: List[float] = []
    for row in rows:
        scores = sorted((anwg_value(row, p) for p in allowed_policies), reverse=True)
        if len(scores) < 2:
            margin = 0.0
        else:
            margin = max(0.0, scores[0] - scores[1])
        margins.append(margin)

    arr = np.asarray(margins, dtype=float)
    if scheme == "uniform":
        weights = np.ones_like(arr)
    elif scheme == "margin":
        weights = np.maximum(arr, 0.0)
    elif scheme == "margin_plus_epsilon":
        weights = arr + epsilon
    elif scheme == "sqrt_margin_plus_epsilon":
        weights = np.sqrt(arr + epsilon)
    elif scheme == "power_margin_plus_epsilon":
        weights = np.power(arr + epsilon, power)
    else:
        raise ValueError(f"Unknown weighting scheme: {scheme}")
    weights = np.clip(weights, epsilon, None)
    return weights


def _frame_to_matrix(rows, feature_cols: Sequence[str]) -> np.ndarray:
    # Works for pandas DataFrame and for list[dict].
    if _is_dataframe(rows):
        return rows.loc[:, list(feature_cols)].to_numpy(dtype=float)
    return np.asarray(
        [[float(row.get(col, 0.0) or 0.0) for col in feature_cols] for row in rows],
        dtype=float,
    )


def _is_dataframe(rows) -> bool:
    return hasattr(rows, "loc") and hasattr(rows, "columns")


def _make_regressor(estimator: str, *, random_state: int, n_estimators: int, max_depth: Optional[int]):
    from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor

    if estimator == "random_forest":
        return RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=1,
        )
    if estimator == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=1,
        )
    if estimator == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            max_iter=n_estimators,
            max_leaf_nodes=max_depth,
            learning_rate=0.05,
            random_state=random_state,
        )
    raise ValueError(f"Unknown regressor estimator: {estimator}")


def _make_classifier(estimator: str, *, random_state: int, n_estimators: int, max_depth: Optional[int]):
    from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier

    if estimator == "random_forest":
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=1,
        )
    if estimator == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=1,
        )
    if estimator == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            max_iter=n_estimators,
            max_leaf_nodes=max_depth,
            learning_rate=0.05,
            random_state=random_state,
        )
    raise ValueError(f"Unknown classifier estimator: {estimator}")


class FixedPolicySelector:
    """Always select one fixed policy."""

    def __init__(self, policy: str, *, name: Optional[str] = None):
        self.policy = policy
        self.name = name or f"always_{policy}"

    def fit(self, rows) -> "FixedPolicySelector":
        return self

    def predict(self, rows) -> List[str]:
        return [self.policy] * len(rows)


class PolicyRewardRegressorSelector:
    """One reward model per policy, then argmax predicted ANWG."""

    def __init__(
        self,
        *,
        name: str,
        allowed_policies: Sequence[str],
        feature_cols: Sequence[str],
        estimator: str = "random_forest",
        n_estimators: int = 200,
        max_depth: Optional[int] = 10,
        random_state: int = 42,
        weight_scheme: str = "uniform",
        weight_epsilon: float = 0.001,
    ):
        self.name = name
        self.allowed_policies = list(allowed_policies)
        self.feature_cols = validate_feature_columns(feature_cols)
        self.estimator = estimator
        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.random_state = int(random_state)
        self.weight_scheme = weight_scheme
        self.weight_epsilon = float(weight_epsilon)
        self.models: Dict[str, object] = {}

    def fit(self, rows) -> "PolicyRewardRegressorSelector":
        x = _frame_to_matrix(rows, self.feature_cols)
        sample_weight = None
        if self.weight_scheme != "uniform":
            records = rows.to_dict("records") if hasattr(rows, "to_dict") else list(rows)
            sample_weight = policy_margin_weights(
                records,
                self.allowed_policies,
                scheme=self.weight_scheme,
                epsilon=self.weight_epsilon,
            )

        for idx, policy in enumerate(self.allowed_policies):
            model = _make_regressor(
                self.estimator,
                random_state=self.random_state + idx,
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
            )
            y = np.asarray(
                rows[anwg_column(policy)] if _is_dataframe(rows)
                else [anwg_value(r, policy) for r in rows],
                dtype=float,
            )
            if sample_weight is not None:
                model.fit(x, y, sample_weight=sample_weight)
            else:
                model.fit(x, y)
            self.models[policy] = model
        return self

    def predict_scores(self, rows) -> np.ndarray:
        x = _frame_to_matrix(rows, self.feature_cols)
        return np.column_stack([self.models[p].predict(x) for p in self.allowed_policies])

    def predict(self, rows) -> List[str]:
        scores = self.predict_scores(rows)
        return [self.allowed_policies[int(i)] for i in np.argmax(scores, axis=1)]


class PolicyClassifierSelector:
    """Multiclass policy classifier with optional regret-margin weights."""

    def __init__(
        self,
        *,
        name: str,
        allowed_policies: Sequence[str],
        feature_cols: Sequence[str],
        label_col: str,
        estimator: str = "random_forest",
        n_estimators: int = 200,
        max_depth: Optional[int] = 10,
        random_state: int = 42,
        weight_scheme: str = "uniform",
        weight_epsilon: float = 0.001,
    ):
        self.name = name
        self.allowed_policies = list(allowed_policies)
        self.feature_cols = validate_feature_columns(feature_cols)
        self.label_col = label_col
        self.estimator_name = estimator
        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.random_state = int(random_state)
        self.weight_scheme = weight_scheme
        self.weight_epsilon = float(weight_epsilon)
        self.estimator = _make_classifier(
            estimator,
            random_state=random_state,
            n_estimators=n_estimators,
            max_depth=max_depth,
        )

    def fit(self, rows) -> "PolicyClassifierSelector":
        x = _frame_to_matrix(rows, self.feature_cols)
        y = (
            rows[self.label_col].astype(str).to_numpy()
            if _is_dataframe(rows)
            else np.asarray([r[self.label_col] for r in rows], dtype=str)
        )
        sample_weight = None
        if self.weight_scheme != "uniform":
            records = rows.to_dict("records") if hasattr(rows, "to_dict") else list(rows)
            sample_weight = policy_margin_weights(
                records,
                self.allowed_policies,
                scheme=self.weight_scheme,
                epsilon=self.weight_epsilon,
            )
        self.estimator.fit(x, y, sample_weight=sample_weight)
        return self

    def predict_scores(self, rows) -> np.ndarray:
        x = _frame_to_matrix(rows, self.feature_cols)
        scores = np.zeros((len(rows), len(self.allowed_policies)), dtype=float)
        if hasattr(self.estimator, "predict_proba"):
            probs = self.estimator.predict_proba(x)
            classes = list(self.estimator.classes_)
            for j, cls in enumerate(classes):
                if cls in self.allowed_policies:
                    scores[:, self.allowed_policies.index(cls)] = probs[:, j]
        else:
            preds = self.estimator.predict(x)
            for i, pred in enumerate(preds):
                if pred in self.allowed_policies:
                    scores[i, self.allowed_policies.index(pred)] = 1.0
        return scores

    def predict(self, rows) -> List[str]:
        scores = self.predict_scores(rows)
        return [self.allowed_policies[int(i)] for i in np.argmax(scores, axis=1)]


@dataclass(frozen=True)
class PairwiseModel:
    policy_a: str
    policy_b: str
    estimator: object
    constant_winner: Optional[str] = None


class PairwisePolicyRanker:
    """Pairwise policy ranking by voting over meaningful policy margins."""

    def __init__(
        self,
        *,
        name: str,
        allowed_policies: Sequence[str],
        feature_cols: Sequence[str],
        pairs: Sequence[Tuple[str, str]],
        estimator: str = "random_forest",
        n_estimators: int = 200,
        max_depth: Optional[int] = 8,
        random_state: int = 42,
        min_pair_margin: float = 0.001,
    ):
        self.name = name
        self.allowed_policies = list(allowed_policies)
        self.feature_cols = validate_feature_columns(feature_cols)
        self.pairs = [(a, b) for a, b in pairs if a in self.allowed_policies and b in self.allowed_policies and a != b]
        self.estimator_name = estimator
        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.random_state = int(random_state)
        self.min_pair_margin = float(min_pair_margin)
        self.models: List[PairwiseModel] = []

    def fit(self, rows) -> "PairwisePolicyRanker":
        for idx, (a, b) in enumerate(self.pairs):
            diff = np.asarray(rows[anwg_column(a)], dtype=float) - np.asarray(rows[anwg_column(b)], dtype=float)
            mask = np.abs(diff) >= self.min_pair_margin
            if not np.any(mask):
                continue
            y = np.where(diff[mask] > 0.0, a, b)
            if len(set(y.tolist())) == 1:
                self.models.append(PairwiseModel(a, b, estimator=None, constant_winner=y[0]))
                continue
            est = _make_classifier(
                self.estimator_name,
                random_state=self.random_state + idx,
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
            )
            x = _frame_to_matrix(rows.loc[mask], self.feature_cols)
            sample_weight = np.maximum(np.abs(diff[mask]), self.min_pair_margin)
            est.fit(x, y, sample_weight=sample_weight)
            self.models.append(PairwiseModel(a, b, estimator=est))
        return self

    def predict_scores(self, rows) -> np.ndarray:
        x = _frame_to_matrix(rows, self.feature_cols)
        scores = np.zeros((len(rows), len(self.allowed_policies)), dtype=float)
        for model in self.models:
            a_idx = self.allowed_policies.index(model.policy_a)
            b_idx = self.allowed_policies.index(model.policy_b)
            if model.constant_winner is not None:
                scores[:, self.allowed_policies.index(model.constant_winner)] += 1.0
                continue
            est = model.estimator
            probs = est.predict_proba(x)
            classes = list(est.classes_)
            p_a = probs[:, classes.index(model.policy_a)] if model.policy_a in classes else np.zeros(len(rows))
            p_b = probs[:, classes.index(model.policy_b)] if model.policy_b in classes else np.zeros(len(rows))
            scores[:, a_idx] += p_a
            scores[:, b_idx] += p_b
        return scores

    def predict(self, rows) -> List[str]:
        scores = self.predict_scores(rows)
        return [self.allowed_policies[int(i)] for i in np.argmax(scores, axis=1)]


class UncertaintyFallbackSelector:
    """Fallback to a fixed policy when predicted top-two scores are close."""

    def __init__(
        self,
        *,
        name: str,
        base_selector,
        fallback_policy: str,
        margin_threshold: float,
    ):
        if not hasattr(base_selector, "predict_scores"):
            raise TypeError("base_selector must expose predict_scores()")
        self.name = name
        self.base_selector = base_selector
        self.fallback_policy = fallback_policy
        self.margin_threshold = float(margin_threshold)
        self.allowed_policies = list(base_selector.allowed_policies)

    def fit(self, rows) -> "UncertaintyFallbackSelector":
        return self

    def predict_scores(self, rows) -> np.ndarray:
        return self.base_selector.predict_scores(rows)

    def predict(self, rows) -> List[str]:
        scores = self.predict_scores(rows)
        preds: List[str] = []
        for row_scores in scores:
            order = np.argsort(row_scores)
            best_idx = int(order[-1])
            second_idx = int(order[-2]) if len(order) > 1 else best_idx
            margin = float(row_scores[best_idx] - row_scores[second_idx])
            if margin < self.margin_threshold:
                preds.append(self.fallback_policy)
            else:
                preds.append(self.allowed_policies[best_idx])
        return preds


class RegimeGatedSelector:
    """Route matching rows to a specialist selector, others to a global selector."""

    def __init__(
        self,
        *,
        name: str,
        gate: Callable[[Mapping[str, object]], bool],
        specialist_selector,
        default_selector,
    ):
        self.name = name
        self.gate = gate
        self.specialist_selector = specialist_selector
        self.default_selector = default_selector

    def fit(self, rows) -> "RegimeGatedSelector":
        return self

    def predict(self, rows) -> List[str]:
        records = rows.to_dict("records") if hasattr(rows, "to_dict") else list(rows)
        spec_preds = self.specialist_selector.predict(rows)
        default_preds = self.default_selector.predict(rows)
        return [
            spec if self.gate(row) else default
            for row, spec, default in zip(records, spec_preds, default_preds)
        ]


def azure_conv_like_gate(row: Mapping[str, object]) -> bool:
    """Feature-only long-prompt + mixed-tight-SLO gate used in Phase 2C."""
    if "is_azure_conv_like" in row:
        return bool(row["is_azure_conv_like"])
    mean_prompt = float(row.get("feat_mean_prompt_tokens", 0.0) or 0.0)
    tight = float(row.get("feat_fraction_tight_slo", 0.0) or 0.0)
    return mean_prompt > 1000.0 and 0.4 <= tight <= 0.7


def all_pair_combinations(policies: Sequence[str]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for i, a in enumerate(policies):
        for b in policies[i + 1:]:
            out.append((a, b))
    return out
