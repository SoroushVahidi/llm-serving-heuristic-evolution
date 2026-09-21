"""Joint state-policy reward models: f(x, pi) -> (predicted_reward, uncertainty).

Three CPU-friendly baselines sharing one interface (fit / predict_mean /
predict_uncertainty / predict_suitability):

  Model 1: state + policy_id            -> predicted ANWG   (encoding="identity")
  Model 2: state + structural_features  -> predicted ANWG   (encoding="structural")
  Model 3: state + policy_id + struct.  -> predicted ANWG   (encoding="hybrid")

Uncertainty is per-tree prediction variance within a single
RandomForestRegressor (a bootstrap-ensemble-equivalent method: each tree is
already fit on a bootstrap resample of the training rows, so the spread of
per-tree predictions is a legitimate, practical, deterministic-under-seed
uncertainty estimate -- and it costs one fit, not N refits, keeping this
CPU-friendly). This is the "per-tree prediction variance" method named as
acceptable in the task brief.
"""
from __future__ import annotations

from typing import Any, List, Mapping, Sequence

import numpy as np

from .encoders import PolicyEncoder


class JointRewardModel:
    """f(x, pi) -> (predicted_reward, uncertainty) for one encoding."""

    def __init__(
        self,
        *,
        name: str,
        encoding: str,
        all_policies: Sequence[str],
        n_estimators: int = 200,
        max_depth: int | None = 8,
        random_state: int = 42,
        min_samples_leaf: int = 2,
    ):
        self.name = name
        self.encoding = encoding
        self.all_policies = list(all_policies)
        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.random_state = int(random_state)
        self.min_samples_leaf = int(min_samples_leaf)
        self.encoder = PolicyEncoder(encoding, self.all_policies)
        self.model = None

    def fit(self, rows: Sequence[Mapping[str, Any]]) -> "JointRewardModel":
        from sklearn.ensemble import RandomForestRegressor

        if not rows:
            raise ValueError("Cannot fit a JointRewardModel on zero rows")
        x = self.encoder.fit_transform(rows)
        y = np.asarray([float(r["reward_anwg"]) for r in rows], dtype=float)
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=1,
        )
        self.model.fit(x, y)
        return self

    def _check_fitted(self) -> None:
        if self.model is None:
            raise RuntimeError(f"JointRewardModel {self.name!r} must be fit() before predicting")

    def predict_mean(self, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        self._check_fitted()
        x = self.encoder.transform(rows)
        return self.model.predict(x)

    def predict_uncertainty(self, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        """Nonnegative, policy-specific, deterministic-under-seed uncertainty:
        std of per-tree predictions within the fitted forest."""
        self._check_fitted()
        x = self.encoder.transform(rows)
        per_tree = np.stack([tree.predict(x) for tree in self.model.estimators_], axis=0)
        uncertainty = per_tree.std(axis=0)
        return np.maximum(uncertainty, 0.0)

    def predict_suitability(self, rows: Sequence[Mapping[str, Any]], *, lam: float = 0.5) -> np.ndarray:
        """S(x, pi) = mu(x, pi) - lambda * u(x, pi)."""
        mu = self.predict_mean(rows)
        u = self.predict_uncertainty(rows)
        return mu - float(lam) * u


class IndependentPerPolicyRewardModel:
    """One regressor per policy over state features only -- the existing
    (pre-joint) baseline formulation from selector/advanced.py, reimplemented
    here only to share this module's evaluation harness; the canonical
    implementation for production selector use remains
    selector.advanced.PolicyRewardRegressorSelector. Used as a comparison
    baseline to answer "does joint modeling beat independent per-policy
    regression?" (see the scientific report)."""

    def __init__(self, *, name: str, all_policies: Sequence[str], n_estimators: int = 200, max_depth: int | None = 8, random_state: int = 42):
        self.name = name
        self.all_policies = list(all_policies)
        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.random_state = int(random_state)
        self.models: dict = {}
        self._state_cols: List[str] = []

    def fit(self, rows: Sequence[Mapping[str, Any]]) -> "IndependentPerPolicyRewardModel":
        from sklearn.ensemble import RandomForestRegressor

        self._state_cols = sorted({k for row in rows for k in row["state_features"].keys()})
        for idx, policy in enumerate(self.all_policies):
            policy_rows = [r for r in rows if r["policy_name"] == policy]
            if not policy_rows:
                continue
            x = self._state_matrix(policy_rows)
            y = np.asarray([float(r["reward_anwg"]) for r in policy_rows], dtype=float)
            model = RandomForestRegressor(
                n_estimators=self.n_estimators, max_depth=self.max_depth,
                random_state=self.random_state + idx, n_jobs=1,
            )
            model.fit(x, y)
            self.models[policy] = model
        return self

    def _state_matrix(self, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        out = np.zeros((len(rows), len(self._state_cols)), dtype=float)
        col_index = {c: i for i, c in enumerate(self._state_cols)}
        for r, row in enumerate(rows):
            for k, v in row["state_features"].items():
                idx = col_index.get(k)
                if idx is not None:
                    out[r, idx] = float(v)
        return out

    def predict_mean(self, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        out = np.zeros(len(rows), dtype=float)
        by_policy: dict = {}
        for i, row in enumerate(rows):
            by_policy.setdefault(row["policy_name"], []).append(i)
        for policy, indices in by_policy.items():
            model = self.models.get(policy)
            if model is None:
                continue
            sub_rows = [rows[i] for i in indices]
            preds = model.predict(self._state_matrix(sub_rows))
            for local_i, global_i in enumerate(indices):
                out[global_i] = preds[local_i]
        return out

    def predict_uncertainty(self, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        out = np.zeros(len(rows), dtype=float)
        by_policy: dict = {}
        for i, row in enumerate(rows):
            by_policy.setdefault(row["policy_name"], []).append(i)
        for policy, indices in by_policy.items():
            model = self.models.get(policy)
            if model is None:
                continue
            sub_rows = [rows[i] for i in indices]
            x = self._state_matrix(sub_rows)
            per_tree = np.stack([t.predict(x) for t in model.estimators_], axis=0)
            u = np.maximum(per_tree.std(axis=0), 0.0)
            for local_i, global_i in enumerate(indices):
                out[global_i] = u[local_i]
        return out

    def predict_suitability(self, rows: Sequence[Mapping[str, Any]], *, lam: float = 0.5) -> np.ndarray:
        return self.predict_mean(rows) - float(lam) * self.predict_uncertainty(rows)
