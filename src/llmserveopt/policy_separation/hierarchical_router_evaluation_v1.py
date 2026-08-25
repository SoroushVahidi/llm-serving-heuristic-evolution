"""Hierarchical Regime Router v1 -- offline scenario-level baselines and
metric formulas (design doc SS I/Q, gates json `baselines`).

Operates on the already-frozen scenario-level ANWG matrix
(`experiments/unified_utility_matrix_v2/unified_utility_matrix_wide_v2.csv`)
joined against MF-PSD's learnable features/group_key
(`experiments/mf_psd_v1/mf_psd_scenarios_v1.csv`) -- the same pattern
`multifamily_contextual_selector_v1` (Step-3) already established, rather
than re-running the simulator per baseline. A scenario's end-to-end
"hierarchy" outcome is approximated by the MAJORITY effective regime over
its per-step online telemetry (after dwell/fallback), dispatched to that
regime's Stage-2 prediction -- an offline scenario-level approximation
explicitly scoped to this task's TRAIN/VAL/smoke-only mandate; true
per-step live-simulation routing evaluation is deferred to the actual,
separately-authorized scientific run (design doc S13/T).

IMPLEMENTATION + VALIDATION ONLY. No TEST-split conclusion is drawn by
calling this module -- it is capable of evaluating on TEST (needed for the
future scientific run) but nothing here computes or reports a scientific
verdict; that is `hierarchical_router_gates_v1.py`, and running it against
TEST is a separate authorization.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ..selector.hierarchical_stage2_selectors_v1 import Stage2Selector
from ..selector.multifamily_contextual_selector_v1 import (
    FEATURE_COLUMNS,
    POLICY_COLUMNS,
    build_preprocessor,
    build_X,
    infer_column_kinds,
)
from .hierarchical_regime_router_v1 import (
    ACTIVE_REGIMES,
    FALLBACK_POLICY,
    REGIME_A,
    REGIME_B,
    REGIME_C,
    REGIME_OF_FAMILY,
    STAGE2_CANDIDATES,
    build_splits,
)

ROOT = Path(__file__).resolve().parents[3]
UUM_WIDE = ROOT / "experiments/unified_utility_matrix_v2/unified_utility_matrix_wide_v2.csv"
MF_PSD_SCENARIOS = ROOT / "experiments/mf_psd_v1/mf_psd_scenarios_v1.csv"


# ---------------------------------------------------------------------------
# Dataset assembly
# ---------------------------------------------------------------------------

def load_scenario_level_dataset() -> pd.DataFrame:
    """One row per MF-PSD scenario: identity + group_key + seed + 33
    learnable features + 6 policy ANWG columns (renamed to bare policy id)
    + `regime_ground_truth` (mechanism_family, mapped 1:1 -- design doc
    SS I baseline C note: partitions identically to the true activity
    label in the feasibility telemetry) + `split` (design doc SS J)."""
    wide = pd.read_csv(UUM_WIDE)
    scen = pd.read_csv(MF_PSD_SCENARIOS)
    df = wide.merge(
        scen[["canonical_scenario_id", "group_key", "seed"] + FEATURE_COLUMNS],
        on="canonical_scenario_id", how="inner",
    )
    if len(df) != 176:
        raise ValueError(f"expected 176 scenarios after join, got {len(df)}")
    df = df.rename(columns={f"anwg__{p}": p for p in POLICY_COLUMNS})
    df["regime_ground_truth"] = df["mechanism_family"].map(REGIME_OF_FAMILY)
    split_map = build_splits(scen)
    df["split"] = df["canonical_scenario_id"].map(split_map)
    return df.reset_index(drop=True)


def scenario_regime_from_telemetry(
    telemetry_by_scenario: Dict[str, List[str]], stage1_router, dwell_fn
) -> Dict[str, str]:
    """For each scenario id, run Stage-1 raw predictions over its ordered
    per-step telemetry rows, apply the dwell/fallback FSM, and return the
    MAJORITY effective regime as that scenario's single dispatched regime
    (the offline scenario-level approximation documented in the module
    docstring)."""
    out: Dict[str, str] = {}
    for scenario_id, raw_regimes in telemetry_by_scenario.items():
        effective, _diag = dwell_fn(raw_regimes)
        vals, counts = np.unique(effective, return_counts=True)
        out[scenario_id] = str(vals[np.argmax(counts)])
    return out


# ---------------------------------------------------------------------------
# Baseline A: best global fixed policy (frozen: weighted_fair_share)
# ---------------------------------------------------------------------------

def baseline_a_anwg(df: pd.DataFrame) -> pd.Series:
    return df[FALLBACK_POLICY]


# ---------------------------------------------------------------------------
# Baseline C: oracle regime router + oracle native-pair selector (audit only)
# ---------------------------------------------------------------------------

def baseline_c_anwg(df: pd.DataFrame) -> pd.Series:
    out = np.empty(len(df), dtype=float)
    for i, (regime, row) in enumerate(zip(df["regime_ground_truth"], df.itertuples())):
        if regime not in ACTIVE_REGIMES:
            out[i] = getattr(row, FALLBACK_POLICY)
            continue
        p0, p1 = STAGE2_CANDIDATES[regime]
        out[i] = max(getattr(row, p0), getattr(row, p1))
    return pd.Series(out, index=df.index)


# ---------------------------------------------------------------------------
# Baseline G: global six-policy oracle (audit only)
# ---------------------------------------------------------------------------

def baseline_g_anwg(df: pd.DataFrame) -> pd.Series:
    return df[POLICY_COLUMNS].max(axis=1)


# ---------------------------------------------------------------------------
# Baseline E: learned Stage-1 + regime-specific fixed-best (ablation)
# ---------------------------------------------------------------------------

def regime_fixed_best_from_train(train_df: pd.DataFrame) -> Dict[str, str]:
    """Per regime: which of its 2 native candidates has the higher mean
    ANWG on that regime's own TRAIN scenarios (design doc SS I baseline E:
    'that regime's own single best-fixed native policy (from TRAIN)')."""
    out = {}
    for regime in (REGIME_A, REGIME_B, REGIME_C):
        sub = train_df[train_df["regime_ground_truth"] == regime]
        if len(sub) == 0:
            continue
        p0, p1 = STAGE2_CANDIDATES[regime]
        out[regime] = p0 if sub[p0].mean() >= sub[p1].mean() else p1
    return out


def baseline_e_anwg(df: pd.DataFrame, predicted_regime: pd.Series, regime_fixed_best: Dict[str, str]) -> pd.Series:
    out = np.empty(len(df), dtype=float)
    for i, (regime, row) in enumerate(zip(predicted_regime, df.itertuples())):
        if regime in ACTIVE_REGIMES and regime in regime_fixed_best:
            out[i] = getattr(row, regime_fixed_best[regime])
        else:
            out[i] = getattr(row, FALLBACK_POLICY)
    return pd.Series(out, index=df.index)


# ---------------------------------------------------------------------------
# Baseline D: learned Stage-1 + learned Stage-2 (system under test)
# ---------------------------------------------------------------------------

def baseline_d_anwg(
    df: pd.DataFrame, predicted_regime: pd.Series, stage2_selectors: Dict[str, Stage2Selector]
) -> pd.Series:
    out = np.empty(len(df), dtype=float)
    for regime in ACTIVE_REGIMES:
        mask = (predicted_regime == regime).to_numpy()
        if not mask.any():
            continue
        sub = df.loc[mask]
        if regime in stage2_selectors:
            picks = stage2_selectors[regime].predict(sub)
        else:
            # No trained selector for this regime (e.g. 0 TRAIN rows) --
            # fall back to reading the regime's own native-pair oracle
            # column-wise max is NOT used here (that would be baseline C);
            # instead fall back to the global fixed policy, the same safe
            # default as an unroutable case.
            picks = np.array([FALLBACK_POLICY] * len(sub))
        out[mask] = [sub[p].iloc[j] for j, p in enumerate(picks)]
    fallback_mask = ~np.isin(predicted_regime.to_numpy(), list(ACTIVE_REGIMES))
    if fallback_mask.any():
        out[fallback_mask] = df.loc[fallback_mask, FALLBACK_POLICY].to_numpy()
    return pd.Series(out, index=df.index)


# ---------------------------------------------------------------------------
# Baseline B: prior flat 6-policy selector (Step-3 multifamily_contextual_selector_v1)
# ---------------------------------------------------------------------------

def fit_baseline_b(train_df: pd.DataFrame) -> Tuple[Pipeline, List[str], List[str]]:
    """Refits Step-3's exact frozen recipe (pooled logreg over all 6
    policies) on THIS design's own TRAIN split -- no persisted model
    artifact exists to reload verbatim (checked: only a results.json was
    ever written), so refitting the identical, unmodified recipe on the
    new split boundaries is the closest faithful reuse available (design
    doc SS I baseline B: 're-evaluated on this design's own splits for a
    fair comparison, not retrained' -- read here as 'not retuned', since
    literal weight reuse is unavailable)."""
    numeric_cols, categorical_cols = infer_column_kinds(train_df, FEATURE_COLUMNS)
    X_train = build_X(train_df, numeric_cols, categorical_cols)
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    vals = train_df[POLICY_COLUMNS].to_numpy(dtype=float)
    best = vals.max(axis=1)
    winners = [POLICY_COLUMNS[int(np.argmax(row >= b - 1e-9))] for row, b in zip(vals, best)]
    y_train = pd.Series(winners, index=train_df.index)
    pipe = Pipeline([("prep", preprocessor), ("clf", LogisticRegression(C=1.0, max_iter=2000, random_state=20260817))])
    pipe.fit(X_train, y_train)
    return pipe, numeric_cols, categorical_cols


def baseline_b_anwg(df: pd.DataFrame, pipe: Pipeline, numeric_cols: List[str], categorical_cols: List[str]) -> pd.Series:
    X = build_X(df, numeric_cols, categorical_cols)
    picks = pipe.predict(X)
    return pd.Series([df[p].iloc[i] for i, p in enumerate(picks)], index=df.index)


# ---------------------------------------------------------------------------
# Baseline F: hidden-family-aware selector (audit only; deliberate leakage)
# ---------------------------------------------------------------------------

def fit_baseline_f(train_df: pd.DataFrame) -> Tuple[Pipeline, List[str], List[str]]:
    """Same recipe as baseline B, but `mechanism_family` is explicitly
    included as an extra categorical input -- the deliberate-leakage
    upper-bound reference (design doc SS I baseline F). Never a candidate
    for deployment, never used by G1/G8 leakage checks (which apply only
    to Stage-1/Stage-2, baselines D/E)."""
    numeric_cols, categorical_cols = infer_column_kinds(train_df, FEATURE_COLUMNS)
    categorical_cols = categorical_cols + ["mechanism_family"]
    X_train = build_X(train_df, numeric_cols, categorical_cols)
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    vals = train_df[POLICY_COLUMNS].to_numpy(dtype=float)
    best = vals.max(axis=1)
    winners = [POLICY_COLUMNS[int(np.argmax(row >= b - 1e-9))] for row, b in zip(vals, best)]
    y_train = pd.Series(winners, index=train_df.index)
    pipe = Pipeline([("prep", preprocessor), ("clf", LogisticRegression(C=1.0, max_iter=2000, random_state=20260817))])
    pipe.fit(X_train, y_train)
    return pipe, numeric_cols, categorical_cols


def baseline_f_anwg(df: pd.DataFrame, pipe: Pipeline, numeric_cols: List[str], categorical_cols: List[str]) -> pd.Series:
    X = build_X(df, numeric_cols, categorical_cols)
    picks = pipe.predict(X)
    return pd.Series([df[p].iloc[i] for i, p in enumerate(picks)], index=df.index)


# ---------------------------------------------------------------------------
# SS Q -- metric formulas
# ---------------------------------------------------------------------------

def mean_regret(achieved: pd.Series, oracle: pd.Series) -> float:
    return float((oracle.to_numpy() - achieved.to_numpy()).mean())


def delta_anwg(hierarchy: pd.Series, comparator: pd.Series) -> float:
    return float((hierarchy.to_numpy() - comparator.to_numpy()).mean())


def oracle_gap_closure(hierarchy_mean: float, fixed_mean: float, oracle_mean: float) -> Optional[float]:
    denom = oracle_mean - fixed_mean
    if abs(denom) < 1e-12:
        return None
    return (hierarchy_mean - fixed_mean) / denom


def multi_regime_benefit_count(
    df: pd.DataFrame, hierarchy: pd.Series, fixed: pd.Series
) -> Tuple[int, Dict[str, float]]:
    per_regime_delta: Dict[str, float] = {}
    n_benefit = 0
    for regime in ACTIVE_REGIMES:
        mask = (df["regime_ground_truth"] == regime).to_numpy()
        if not mask.any():
            continue
        d = float(hierarchy[mask].mean() - fixed[mask].mean())
        per_regime_delta[regime] = d
        if d > 0:
            n_benefit += 1
    return n_benefit, per_regime_delta


def group_resampled_bootstrap_ci(
    df: pd.DataFrame, hierarchy: pd.Series, fixed: pd.Series, n_boot: int = 2000, ci: float = 0.90, seed: int = 20260817
) -> Tuple[float, float]:
    """Group-resampled (by `group_key`) bootstrap CI on mean(hierarchy -
    fixed), matching Step-3's own CI convention (design doc G5)."""
    rng = np.random.default_rng(seed)
    delta = (hierarchy.to_numpy() - fixed.to_numpy())
    groups = df["group_key"].to_numpy()
    unique_groups = np.unique(groups)
    if len(unique_groups) == 0:
        return (float("nan"), float("nan"))
    group_to_idx = {g: np.where(groups == g)[0] for g in unique_groups}
    means = np.empty(n_boot)
    for b in range(n_boot):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([group_to_idx[g] for g in sampled_groups])
        means[b] = delta[idx].mean()
    alpha = (1.0 - ci) / 2.0
    lo = float(np.quantile(means, alpha))
    hi = float(np.quantile(means, 1.0 - alpha))
    return lo, hi


def catastrophic_misroute_rate(predicted_regime: pd.Series, true_regime: pd.Series) -> float:
    """Design doc G3: rate of A<->B<->C wrong-active-regime routing,
    EXCLUDING any row where predicted or true regime is NONE/OVERLAP
    (those are safe-by-construction fallback outcomes, never catastrophic)."""
    both_active = predicted_regime.isin(ACTIVE_REGIMES) & true_regime.isin(ACTIVE_REGIMES)
    if not both_active.any():
        return 0.0
    sub_pred = predicted_regime[both_active]
    sub_true = true_regime[both_active]
    wrong = (sub_pred != sub_true).to_numpy()
    return float(wrong.mean())
