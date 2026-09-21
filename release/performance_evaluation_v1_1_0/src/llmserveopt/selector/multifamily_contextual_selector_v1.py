"""Multi-Family Contextual Selector v1 -- Step 3.

Implements docs/design/MULTIFAMILY_CONTEXTUAL_SELECTOR_V1.md. Uses the
frozen, complete unified utility matrix (experiments/unified_utility_matrix_v2/,
UNIFIED_UTILITY_MATRIX_READY) joined against MF-PSD v1's learnable-feature
allowlist to fit small, fixed-hyperparameter selector models and evaluate
them under three preregistered split regimes (within-family, pooled,
leave-one-family-out). No mechanism attribution, no composition/synthesis,
no neural networks, no AutoML.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[3]
MF_PSD_SCENARIOS = ROOT / "experiments/mf_psd_v1/mf_psd_scenarios_v1.csv"
MF_PSD_SCHEMA = ROOT / "experiments/mf_psd_v1/mf_psd_schema_v1.json"
UUM_WIDE = ROOT / "experiments/unified_utility_matrix_v2/unified_utility_matrix_wide_v2.csv"

SEED = 20260817
EPS = 0.01

POLICY_COLUMNS: List[str] = sorted([
    "chunked_prefill_small", "estimated_service_time_first", "full_prefill",
    "kv_constrained_online", "least_laxity_first", "weighted_fair_share",
])

FAMILIES = (
    "FAMILY_A_FAIRNESS_STARVATION_V2",
    "FAMILY_B_PREFILL_DECODE_V2",
    "FAMILY_C_KV_PRESSURE_V2",
)

FORBIDDEN_COLUMNS = {
    "mechanism_family", "canonical_scenario_id", "source_scenario_id",
    "group_key", "seed",
} | {f"anwg__{p}" for p in POLICY_COLUMNS} | set(POLICY_COLUMNS)


def load_feature_allowlist() -> List[str]:
    schema = json.loads(MF_PSD_SCHEMA.read_text())
    return list(schema["learnable_feature_allowlist"])


FEATURE_COLUMNS: List[str] = load_feature_allowlist()  # 33 columns


def load_dataset() -> pd.DataFrame:
    """Join the unified utility matrix (wide) with MF-PSD v1's learnable
    feature allowlist, on canonical_scenario_id. Returns one row per
    scenario: identity columns, 33 feature columns, 6 policy ANWG columns
    (renamed to bare policy_id, e.g. 'weighted_fair_share')."""
    wide = pd.read_csv(UUM_WIDE)
    scen = pd.read_csv(MF_PSD_SCENARIOS)
    df = wide.merge(
        scen[["canonical_scenario_id", "group_key"] + FEATURE_COLUMNS],
        on="canonical_scenario_id", how="inner",
    )
    if len(df) != 176:
        raise ValueError(f"expected 176 scenarios after join, got {len(df)}")
    df = df.rename(columns={f"anwg__{p}": p for p in POLICY_COLUMNS})
    return df.reset_index(drop=True)


def infer_column_kinds(df: pd.DataFrame, columns: Sequence[str] = FEATURE_COLUMNS) -> Tuple[List[str], List[str]]:
    """Numeric vs categorical, inferred from whether non-missing values are
    parseable as numbers -- avoids hardcoding a fragile column-type list."""
    numeric, categorical = [], []
    for c in columns:
        vals = df[c].replace("", np.nan).dropna()
        try:
            pd.to_numeric(vals)
            numeric.append(c)
        except (ValueError, TypeError):
            categorical.append(c)
    return numeric, categorical


def build_X(df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str]) -> pd.DataFrame:
    """Explicit, frozen missing-value handling (design doc S2): numeric
    missing -> 0.0 + a same-named '<col>__missing' indicator; categorical
    missing -> literal '__NONE__' category."""
    X = df[numeric_cols + categorical_cols].copy()
    for c in numeric_cols:
        col = pd.to_numeric(X[c].replace("", np.nan))
        X[f"{c}__missing"] = col.isna().astype(int)
        X[c] = col.fillna(0.0)
    for c in categorical_cols:
        X[c] = X[c].replace("", "__NONE__").fillna("__NONE__").astype(str)
    return X


def build_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    missing_cols = [f"{c}__missing" for c in numeric_cols]
    return ColumnTransformer([
        ("num", "passthrough", numeric_cols + missing_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ])


# ---------------------------------------------------------------------------
# Target construction (design doc S3)
# ---------------------------------------------------------------------------

def compute_exact_winner(df: pd.DataFrame) -> pd.Series:
    """Primary target: argmax ANWG, bit-exact ties (<1e-9) broken by fixed
    alphabetical canonical_policy_id order (POLICY_COLUMNS is already
    sorted, so the first index at/above max-1e-9 is the tie-break winner)."""
    vals = df[POLICY_COLUMNS].to_numpy(dtype=float)
    best = vals.max(axis=1)
    winners = []
    for row, b in zip(vals, best):
        idx = np.argmax(row >= b - 1e-9)
        winners.append(POLICY_COLUMNS[idx])
    return pd.Series(winners, index=df.index, name="winner")


def achieved_anwg_of(df: pd.DataFrame, predicted: pd.Series) -> np.ndarray:
    return np.array([df.loc[i, p] for i, p in zip(df.index, predicted)])


def regret_of(df: pd.DataFrame, predicted: pd.Series) -> np.ndarray:
    best = df[POLICY_COLUMNS].max(axis=1).to_numpy()
    return best - achieved_anwg_of(df, predicted)


def evaluate_predictions(df: pd.DataFrame, predicted: pd.Series) -> Dict[str, float]:
    """Pure evaluation of one prediction series against the frozen matrix --
    never derives any baseline from df itself (df may be a TEST/held-out
    set; deriving 'best fixed' from it would leak test information into a
    metric meant to be a fair external comparison). Callers combine this
    with a separately-computed baseline_best_fixed(TRAIN, TEST) result to
    get gap-to-best-fixed (see run_multifamily_contextual_selector_v1.py)."""
    achieved = achieved_anwg_of(df, predicted)
    best = df[POLICY_COLUMNS].max(axis=1).to_numpy()
    regret = best - achieved
    exact_winner = compute_exact_winner(df)
    exact_acc = float((predicted.to_numpy() == exact_winner.to_numpy()).mean())
    eps_acc = float((regret <= EPS).mean())
    return {
        "n": int(len(df)),
        "mean_regret": float(np.mean(regret)),
        "median_regret": float(np.median(regret)),
        "p95_regret": float(np.percentile(regret, 95)),
        "frac_regret_le_eps": float((regret <= EPS).mean()),
        "exact_winner_accuracy": exact_acc,
        "epsilon_optimal_accuracy": eps_acc,
        "mean_achieved_anwg": float(np.mean(achieved)),
        "gap_to_oracle_mean_regret": float(np.mean(regret)),
    }


# ---------------------------------------------------------------------------
# Group-aware split builders (design doc S4)
# ---------------------------------------------------------------------------

def _allocate_counts(n: int, fracs: Sequence[float]) -> List[int]:
    counts = [max(1, round(n * f)) for f in fracs]
    while sum(counts) > n:
        counts[int(np.argmax(counts))] -= 1
    while sum(counts) < n:
        counts[int(np.argmin(counts))] += 1
    return counts


def split_groups_n_way(groups: Sequence[str], fracs: Sequence[float], seed: int = SEED) -> List[List[str]]:
    groups = np.array(sorted(set(groups)))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(groups))
    groups = groups[perm]
    counts = _allocate_counts(len(groups), fracs)
    idx = np.cumsum(counts)
    out, start = [], 0
    for end in idx:
        out.append(list(groups[start:end]))
        start = end
    return out


def regime_a_within_family_splits(df: pd.DataFrame, seed: int = SEED) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Per family: independent grouped 60/20/20 train/val/test."""
    out = {}
    for fam in FAMILIES:
        sub = df[df["mechanism_family"] == fam]
        train_g, val_g, test_g = split_groups_n_way(sub["group_key"], (0.6, 0.2, 0.2), seed=seed)
        out[fam] = {
            "train": sub[sub["group_key"].isin(train_g)],
            "val": sub[sub["group_key"].isin(val_g)],
            "test": sub[sub["group_key"].isin(test_g)],
        }
    return out


def regime_b_pooled_split(df: pd.DataFrame, seed: int = SEED) -> Dict[str, pd.DataFrame]:
    """All families pooled: grouped 60/20/20 (group_key is already
    family-prefixed, so no cross-family group collision)."""
    train_g, val_g, test_g = split_groups_n_way(df["group_key"], (0.6, 0.2, 0.2), seed=seed)
    return {
        "train": df[df["group_key"].isin(train_g)],
        "val": df[df["group_key"].isin(val_g)],
        "test": df[df["group_key"].isin(test_g)],
    }


def regime_c_lofo_splits(df: pd.DataFrame, seed: int = SEED) -> Dict[str, Dict[str, pd.DataFrame]]:
    """For each held-out family: train+val = grouped 80/20 of the OTHER
    two families pooled; test = ALL of the held-out family, untouched by
    any split/selection decision."""
    out = {}
    for held_out in FAMILIES:
        train_val = df[df["mechanism_family"] != held_out]
        train_g, val_g = split_groups_n_way(train_val["group_key"], (0.8, 0.2), seed=seed)
        out[held_out] = {
            "train": train_val[train_val["group_key"].isin(train_g)],
            "val": train_val[train_val["group_key"].isin(val_g)],
            "test": df[df["mechanism_family"] == held_out],
        }
    return out


# ---------------------------------------------------------------------------
# Baselines and models (design doc S5)
# ---------------------------------------------------------------------------

def baseline_best_fixed(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.Series:
    best_policy = train_df[POLICY_COLUMNS].mean().idxmax()
    return pd.Series([best_policy] * len(test_df), index=test_df.index)


def baseline_majority(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.Series:
    winner = compute_exact_winner(train_df)
    majority_class = winner.value_counts().idxmax()
    return pd.Series([majority_class] * len(test_df), index=test_df.index)


def _model_factories() -> Dict[str, object]:
    return {
        "logreg": lambda: LogisticRegression(C=1.0, max_iter=2000),
        "tree": lambda: DecisionTreeClassifier(max_depth=4, min_samples_leaf=3, random_state=SEED),
        "forest": lambda: RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=2, random_state=SEED),
    }


def fit_classifier(name: str, X_train: pd.DataFrame, y_train: pd.Series, preprocessor: ColumnTransformer) -> Pipeline:
    pipe = Pipeline([("prep", preprocessor), ("clf", _model_factories()[name]())])
    pipe.fit(X_train, y_train)
    return pipe


def predict_classifier(pipe: Pipeline, X_test: pd.DataFrame, index) -> pd.Series:
    return pd.Series(pipe.predict(X_test), index=index)


def fit_utility_regressors(X_train: pd.DataFrame, y_train_df: pd.DataFrame, preprocessor: ColumnTransformer) -> Dict[str, Pipeline]:
    models = {}
    for p in POLICY_COLUMNS:
        pipe = Pipeline([("prep", preprocessor), ("reg", Ridge(alpha=1.0, random_state=SEED))])
        pipe.fit(X_train, y_train_df[p])
        models[p] = pipe
    return models


def predict_utility_argmax(models: Dict[str, Pipeline], X_test: pd.DataFrame, index) -> pd.Series:
    preds = np.column_stack([models[p].predict(X_test) for p in POLICY_COLUMNS])
    idx = preds.argmax(axis=1)
    return pd.Series([POLICY_COLUMNS[i] for i in idx], index=index)


PAIRS: List[Tuple[str, str]] = list(itertools.combinations(POLICY_COLUMNS, 2))


def fit_pairwise(X_train: pd.DataFrame, y_train_df: pd.DataFrame, preprocessor: ColumnTransformer) -> Dict[Tuple[str, str], Pipeline]:
    models = {}
    for (pi, pj) in PAIRS:
        label = (y_train_df[pi] - y_train_df[pj] > EPS).astype(int)
        if label.nunique() < 2:
            pipe = Pipeline([("prep", preprocessor), ("clf", DummyClassifier(strategy="constant", constant=int(label.iloc[0])))])
        else:
            pipe = Pipeline([("prep", preprocessor), ("clf", LogisticRegression(C=1.0, max_iter=2000))])
        pipe.fit(X_train, label)
        models[(pi, pj)] = pipe
    return models


def predict_pairwise(models: Dict[Tuple[str, str], Pipeline], X_test: pd.DataFrame, index) -> pd.Series:
    n = len(X_test)
    beats = {p: np.zeros(n) for p in POLICY_COLUMNS}
    for (pi, pj), pipe in models.items():
        pred = pipe.predict(X_test)
        beats[pi] += pred
        beats[pj] += (1 - pred)
    mat = np.column_stack([beats[p] for p in POLICY_COLUMNS])
    idx = mat.argmax(axis=1)
    return pd.Series([POLICY_COLUMNS[i] for i in idx], index=index)


# ---------------------------------------------------------------------------
# Family-predictability diagnostic and shared-feature robustness (S2)
# ---------------------------------------------------------------------------

def family_predictability_diagnostic(df: pd.DataFrame, seed: int = SEED) -> Dict[str, float]:
    """5-fold grouped CV: predict mechanism_family from X alone."""
    numeric_cols, categorical_cols = infer_column_kinds(df)
    X = build_X(df, numeric_cols, categorical_cols)
    y = df["mechanism_family"]
    groups = df["group_key"]
    unique_groups = np.array(sorted(groups.unique()))
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(unique_groups), 5)
    accs = []
    for k in range(5):
        test_g = set(folds[k])
        train_mask = ~groups.isin(test_g)
        test_mask = groups.isin(test_g)
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue
        preprocessor = build_preprocessor(numeric_cols, categorical_cols)
        pipe = Pipeline([("prep", preprocessor), ("clf", RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=2, random_state=seed))])
        pipe.fit(X[train_mask], y[train_mask])
        pred = pipe.predict(X[test_mask])
        accs.append(float((pred == y[test_mask].to_numpy()).mean()))
    return {"mean_accuracy": float(np.mean(accs)), "n_folds": len(accs), "fold_accuracies": accs}


SHARED_AB_FEATURES = ["max_active_sequences", "stress_control_relationship"]


def shared_feature_robustness_check(df: pd.DataFrame, seed: int = SEED) -> Dict[str, object]:
    """A<->B only (design doc S2): the two features with a matching
    semantic role in Family A and Family B, unified into shared columns.
    Family C has no analog and is explicitly excluded from this slice."""
    ab = df[df["mechanism_family"].isin(["FAMILY_A_FAIRNESS_STARVATION_V2", "FAMILY_B_PREFILL_DECODE_V2"])].copy()
    for shared in SHARED_AB_FEATURES:
        col_a, col_b = f"feat_A__{shared}", f"feat_B__{shared}"
        ab[f"shared__{shared}"] = ab[col_a].where(ab[col_a] != "", ab[col_b])
    shared_cols = [f"shared__{s}" for s in SHARED_AB_FEATURES]
    numeric_cols, categorical_cols = infer_column_kinds(ab, shared_cols)
    X = build_X(ab, numeric_cols, categorical_cols)
    y = compute_exact_winner(ab)

    train_g, test_g = split_groups_n_way(ab["group_key"], (0.7, 0.3), seed=seed)
    train_mask, test_mask = ab["group_key"].isin(train_g), ab["group_key"].isin(test_g)
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    pipe = Pipeline([("prep", preprocessor), ("clf", RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=2, random_state=seed))])
    pipe.fit(X[train_mask], y[train_mask])
    predicted = pd.Series(pipe.predict(X[test_mask]), index=ab[test_mask].index)
    metrics = evaluate_predictions(ab[test_mask], predicted)
    fixed = baseline_best_fixed(ab[train_mask], ab[test_mask])
    fixed_metrics = evaluate_predictions(ab[test_mask], fixed)
    return {
        "n_scenarios": int(len(ab)),
        "shared_features_used": shared_cols,
        "family_c_excluded": True,
        "selector_mean_regret": metrics["mean_regret"],
        "best_fixed_mean_regret": fixed_metrics["mean_regret"],
        "improvement_over_fixed": fixed_metrics["mean_regret"] - metrics["mean_regret"],
    }
