"""Hierarchical Regime Router v1 -- Stage-2 native-pair selectors.

Implements docs/design/HIERARCHICAL_REGIME_ROUTER_V1.md SS G/H. One
independent logistic-regression binary classifier per regime (native pair
-> 2-class problem), reusing the exact model class/hyperparameters and
feature-preprocessing recipe already established by
`multifamily_contextual_selector_v1` (Step-3) -- not reimplemented from
scratch, imported directly.

IMPLEMENTATION + VALIDATION ONLY. No final scientific TEST evaluation is
performed here.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ..policy_separation.hierarchical_regime_router_v1 import (
    REGIME_A,
    REGIME_B,
    REGIME_C,
    STAGE2_CANDIDATES,
)
from .multifamily_contextual_selector_v1 import (
    FEATURE_COLUMNS,
    build_preprocessor,
    build_X,
    infer_column_kinds,
)

SEED = 20260817
EPS = 0.01


def _assert_native_pair(regime: str, predictions: np.ndarray) -> None:
    candidates = set(STAGE2_CANDIDATES[regime])
    unique = set(predictions.tolist())
    foreign = unique - candidates
    if foreign:
        raise AssertionError(
            f"Stage-2 selector for regime {regime!r} emitted foreign policy id(s) "
            f"{foreign} outside its native pair {candidates} -- this must never happen"
        )


def compute_native_pair_winner(df: pd.DataFrame, regime: str) -> pd.Series:
    """Binary target: argmax ANWG restricted to the regime's own 2 native
    candidates only (never the full 6-policy set), bit-exact ties (<1e-9)
    broken by fixed alphabetical order of the pair."""
    p0, p1 = sorted(STAGE2_CANDIDATES[regime])
    v0 = df[p0].to_numpy(dtype=float)
    v1 = df[p1].to_numpy(dtype=float)
    winner = np.where(v1 > v0 + 1e-9, p1, p0)
    return pd.Series(winner, index=df.index, name="native_pair_winner")


class Stage2Selector:
    """One regime's native-pair logistic-regression selector. Hard
    assertion: `predict` can never return a policy id outside the regime's
    frozen native pair (design doc SS G: 'Add hard assertions that each
    Stage-2 selector can only emit its native pair')."""

    def __init__(self, regime: str, seed: int = SEED) -> None:
        if regime not in (REGIME_A, REGIME_B, REGIME_C):
            raise ValueError(f"regime must be one of A/B/C, got {regime!r}")
        self.regime = regime
        self.candidates: Tuple[str, str] = STAGE2_CANDIDATES[regime]
        self.seed = seed
        self.pipe: Pipeline | None = None
        self._numeric_cols: List[str] | None = None
        self._categorical_cols: List[str] | None = None

    def fit(self, df_train: pd.DataFrame) -> "Stage2Selector":
        if len(df_train) == 0:
            raise ValueError(f"Stage-2 fit for regime {self.regime} called with 0 training rows")
        numeric_cols, categorical_cols = infer_column_kinds(df_train, FEATURE_COLUMNS)
        self._numeric_cols, self._categorical_cols = numeric_cols, categorical_cols
        X = build_X(df_train, numeric_cols, categorical_cols)
        y = compute_native_pair_winner(df_train, self.regime)
        preprocessor: ColumnTransformer = build_preprocessor(numeric_cols, categorical_cols)
        if y.nunique() < 2:
            # Degenerate training slice (one candidate always wins) -- a
            # constant-ish logistic regression is still fit (sklearn
            # handles single-class y by raising, so fall back to a
            # trivial always-predict-that-class pipeline via a 2-row
            # duplicate trick is NOT used here; instead we record the
            # constant explicitly and never call sklearn on 1 class).
            self.pipe = None
            self._constant_prediction = y.iloc[0]
        else:
            self._constant_prediction = None
            self.pipe = Pipeline([
                ("prep", preprocessor),
                ("clf", LogisticRegression(C=1.0, max_iter=2000, random_state=self.seed)),
            ])
            self.pipe.fit(X, y)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self._numeric_cols is None:
            raise RuntimeError(f"Stage2Selector({self.regime}) predict called before fit")
        if self.pipe is None:
            preds = np.array([self._constant_prediction] * len(df))
        else:
            X = build_X(df, self._numeric_cols, self._categorical_cols)
            preds = np.asarray(self.pipe.predict(X))
        _assert_native_pair(self.regime, preds)
        return preds


def fit_all_stage2_selectors(train_by_regime: Dict[str, pd.DataFrame]) -> Dict[str, Stage2Selector]:
    """`train_by_regime`: regime -> TRAIN-split scenario dataframe already
    restricted to that regime's own family (design doc SS J: Stage-1/2
    share split boundaries; a regime's Stage-2 selector is trained only on
    its own regime's scenarios, matching the within-family pattern
    reused from Step-3)."""
    out = {}
    for regime in (REGIME_A, REGIME_B, REGIME_C):
        df = train_by_regime.get(regime)
        if df is None or len(df) == 0:
            continue
        out[regime] = Stage2Selector(regime).fit(df)
    return out
