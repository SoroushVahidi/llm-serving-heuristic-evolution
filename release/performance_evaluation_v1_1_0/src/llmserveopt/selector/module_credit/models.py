"""CPU-friendly module credit models with uncertainty."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from ...policies.registry import POLICY_LIBRARY_V2_NAMES
from .encoders import ModuleCreditEncoder


class ModuleCreditModel:
    """RandomForest model for C_base/C_parent/C_env targets."""

    def __init__(
        self,
        *,
        name: str,
        encoding: str,
        target: str = "C_base",
        all_policies: Sequence[str] = POLICY_LIBRARY_V2_NAMES,
        n_estimators: int = 120,
        max_depth: int | None = 7,
        min_samples_leaf: int = 1,
        random_state: int = 42,
    ) -> None:
        self.name = name
        self.encoding = encoding
        self.target = target
        self.all_policies = list(all_policies)
        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.min_samples_leaf = int(min_samples_leaf)
        self.random_state = int(random_state)
        self.encoder = ModuleCreditEncoder(encoding, all_policies=all_policies)
        self.model = None

    def fit(self, rows: Sequence[Mapping[str, Any]]) -> "ModuleCreditModel":
        from sklearn.ensemble import RandomForestRegressor

        if not rows:
            raise ValueError("Cannot fit ModuleCreditModel on zero rows")
        x = self.encoder.fit_transform(rows)
        y = np.asarray([float(r[self.target]) for r in rows], dtype=float)
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=1,
        )
        self.model.fit(x, y)
        return self

    def predict_mean(self, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        self._check_fitted()
        return self.model.predict(self.encoder.transform(rows))

    def predict_uncertainty(self, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        self._check_fitted()
        x = self.encoder.transform(rows)
        per_tree = np.stack([tree.predict(x) for tree in self.model.estimators_], axis=0)
        return np.maximum(per_tree.std(axis=0), 0.0)

    def predict_score(self, rows: Sequence[Mapping[str, Any]], *, lambda_m: float = 0.5) -> np.ndarray:
        return self.predict_mean(rows) - float(lambda_m) * self.predict_uncertainty(rows)

    def _check_fitted(self) -> None:
        if self.model is None:
            raise RuntimeError(f"ModuleCreditModel {self.name!r} must be fit before predict")
