#!/usr/bin/env python3
"""Build the Family-A oracle-labeled ESTF/WFS pilot dataset v1.

Pilot-only, deterministic, TRAIN/VAL-only. This script does not run a new
simulation. It converts the existing repaired 91-event Family-A contested
request artifacts into a small quality-study dataset using a priority-weighted
SLO utility over the two directly contested requests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier, XGBRegressor
except Exception:  # noqa: BLE001 - optional dependency, fallback below
    XGBClassifier = None
    XGBRegressor = None


REPO_ROOT = Path(__file__).resolve().parents[1]
CONTESTED_REQUESTS_CSV = (
    REPO_ROOT / "experiments/family_a_contested_request_value_diagnosis/contested_requests.csv"
)
CONSTRAINED_EVENTS_CSV = (
    REPO_ROOT / "experiments/family_a_contested_request_value_diagnosis/constrained_formulation_event_table.csv"
)
OBS_EVENTS_CSV = (
    REPO_ROOT / "experiments/family_a_observability_continuation_v1/family_a_observability_continuation_events.csv"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "datasets/family_a_oracle_policy_pilot_v1"

SCHEMA_VERSION = "family_a_oracle_policy_pilot_v1.0.0"
DATASET_DATE = "2026-08-21"
ESTF = "ESTF"
WFS = "WFS"
TIE = "TIE_OR_UNCERTAIN"
LABELS = [ESTF, WFS, TIE]
BINARY_LABELS = [ESTF, WFS]
EPS = 1e-9

GLOBAL_FEATURES = [
    "queue_length",
    "active_count",
    "n_gpus",
    "queue_age_p10",
    "queue_age_p50",
    "queue_age_p90",
    "queue_age_mean",
    "predicted_output_tokens_p10",
    "predicted_output_tokens_p50",
    "predicted_output_tokens_p90",
    "predicted_output_tokens_mean",
    "prompt_tokens_p10",
    "prompt_tokens_p50",
    "prompt_tokens_p90",
    "prompt_tokens_mean",
    "est_service_time_p10",
    "est_service_time_p50",
    "est_service_time_p90",
    "est_service_time_mean",
    "max_class_deficit_ratio",
    "longest_waiting_age",
    "n_distinct_classes_in_queue",
    "laxity_p10",
    "laxity_p50",
    "laxity_p90",
    "laxity_mean",
    "fraction_laxity_negative",
    "fraction_laxity_near_deadline",
    "mean_kv_utilization",
    "max_kv_utilization",
    "free_kv_capacity",
    "prefilling_count",
    "decoding_count",
    "agg_n_admit_estf",
    "agg_n_admit_wfs",
    "admit_symmetric_diff_size",
    "history_queue_len_slope",
    "history_kv_util_slope",
    "history_admitted_count_slope",
]

SIDE_FEATURES = [
    "priority",
    "prompt_tokens",
    "predicted_output_tokens",
    "predicted_service_proxy",
    "remaining_predicted_service_proxy",
    "queue_age",
    "laxity_own",
]

PAIR_FEATURES = [
    "priority_diff_estf_minus_wfs",
    "prompt_tokens_diff_estf_minus_wfs",
    "predicted_output_tokens_diff_estf_minus_wfs",
    "predicted_service_proxy_diff_estf_minus_wfs",
    "queue_age_diff_estf_minus_wfs",
    "laxity_own_diff_estf_minus_wfs",
    "priority_ratio_estf_over_wfs",
    "predicted_service_proxy_ratio_estf_over_wfs",
    "queue_age_ratio_estf_over_wfs",
    "laxity_own_ratio_estf_over_wfs",
]


class XGBStringClassifier:
    """Small wrapper so the pilot can use installed XGBoost with string labels."""

    def __init__(self) -> None:
        self.imputer = SimpleImputer(strategy="median")
        self.model = XGBClassifier(
            n_estimators=50,
            max_depth=2,
            learning_rate=0.1,
            subsample=1.0,
            colsample_bytree=1.0,
            eval_metric="logloss",
            random_state=0,
        )
        self.classes_ = np.asarray(BINARY_LABELS)

    def fit(self, x: pd.DataFrame, y: np.ndarray) -> "XGBStringClassifier":
        xt = self.imputer.fit_transform(x)
        yt = (np.asarray(y) == WFS).astype(int)
        self.model.fit(xt, yt)
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        xt = self.imputer.transform(x)
        pred = self.model.predict(xt)
        return np.where(pred == 1, WFS, ESTF)

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        xt = self.imputer.transform(x)
        proba = self.model.predict_proba(xt)
        if proba.shape[1] == 1:
            if int(self.model.classes_[0]) == 1:
                p_wfs = proba[:, 0]
            else:
                p_wfs = np.zeros(len(x), dtype=float)
            return np.column_stack([1.0 - p_wfs, p_wfs])
        return proba


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_head() -> str:
    return subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True).strip()


def _git_dirty() -> bool:
    return bool(subprocess.check_output(["git", "-C", str(REPO_ROOT), "status", "--short"], text=True).strip())


def _parse_scenario(sid: str) -> dict[str, Any]:
    m = re.search(
        r"util(?P<util>[0-9.]+)\.skew(?P<skew>[0-9.]+)\.(?P<fav>favlong|favshort)\.noise(?P<noise>[0-9.]+)\.s(?P<seed>[0-9]+)",
        sid,
    )
    if not m:
        return {"utilization": math.nan, "skew": math.nan, "fav": "", "noise": math.nan, "seed": ""}
    return {
        "utilization": float(m.group("util")),
        "skew": float(m.group("skew")),
        "fav": m.group("fav"),
        "noise": float(m.group("noise")),
        "seed": int(m.group("seed")),
    }


def _label(delta_j: float) -> str:
    if delta_j > 0.0:
        return ESTF
    if delta_j < 0.0:
        return WFS
    return TIE


def _feature_prefixed(name: str) -> str:
    return f"feat_{name}"


def build_rows() -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    req = pd.read_csv(CONTESTED_REQUESTS_CSV)
    events = pd.read_csv(CONSTRAINED_EVENTS_CSV)
    obs = pd.read_csv(OBS_EVENTS_CSV)

    assert len(events) == 91
    assert len(req) == 182
    assert set(events["split"].unique()) <= {"train", "val"}
    assert set(req["split"].unique()) <= {"train", "val"}
    assert set(obs["split"].unique()) <= {"train", "val"}
    assert (events["n_estf_only"] == 1).all()
    assert (events["n_wfs_only"] == 1).all()
    assert (events["n_common"] == 0).all()

    event_features = events.set_index("event_id")
    rows: list[dict[str, Any]] = []
    for event_id, group in req.groupby("event_id", sort=True):
        if set(group["contested_side"]) != {"estf_only", "wfs_only"} or len(group) != 2:
            raise ValueError(f"expected exactly one ESTF and one WFS contested row for {event_id}")
        ev = event_features.loc[event_id]
        estf_req = group[group["contested_side"] == "estf_only"].iloc[0]
        wfs_req = group[group["contested_side"] == "wfs_only"].iloc[0]
        j_estf = float(group["br_estf_estf_weighted_contribution"].sum())
        j_wfs = float(group["br_wfs_wfs_weighted_contribution"].sum())
        delta_j = j_estf - j_wfs
        scenario = str(ev["canonical_scenario_id"])
        parsed = _parse_scenario(scenario)

        row: dict[str, Any] = {
            "sample_id": f"{SCHEMA_VERSION}::{event_id}",
            "event_id": event_id,
            "scenario_id": scenario,
            "canonical_scenario_id": scenario,
            "step": int(ev["step"]),
            "time": float(ev["step"]),
            "split": ev["split"],
            "group_key": scenario,
            "provenance_version": SCHEMA_VERSION,
            "label_utility_name": "contested_pair_priority_weighted_slo_native_continuation",
            "continuation_semantics": "native_estf_vs_native_wfs_bounded_1500_step_counterfactual",
            "horizon_steps": 1500,
            "future_arrivals_included": True,
            "tie_threshold": 0.0,
            "J_ESTF": j_estf,
            "J_WFS": j_wfs,
            "delta_J": delta_j,
            "oracle_label": _label(delta_j),
            "completion_benefit_label": int(ev["completion_benefit_label"]),
            "slo_risk_label": int(ev["slo_risk_label"]),
            "estf_contested_request_id": int(estf_req["request_id"]),
            "wfs_contested_request_id": int(wfs_req["request_id"]),
            "estf_contested_class_id": estf_req["class_id"],
            "wfs_contested_class_id": wfs_req["class_id"],
            "analysis_fav": parsed["fav"],
            "analysis_utilization": parsed["utilization"],
            "analysis_skew": parsed["skew"],
            "analysis_noise": parsed["noise"],
            "analysis_seed": parsed["seed"],
            "raw_completion_gt_label_biased_reference": ev["gt_label"],
            "delta_native_whole_branch_raw_biased_reference": float(ev["delta_native_whole_branch_raw"]),
        }

        for col in GLOBAL_FEATURES:
            row[_feature_prefixed(col)] = ev[col]

        for side_name, side_row in [("estf", estf_req), ("wfs", wfs_req)]:
            for col in SIDE_FEATURES:
                if col == "remaining_predicted_service_proxy":
                    value = side_row["predicted_service_proxy"]
                elif col == "laxity_own":
                    value = ev[f"{side_name}_laxity_own"]
                else:
                    value = side_row[col]
                row[_feature_prefixed(f"{side_name}_{col}")] = value

        pairs = {
            "priority": (estf_req["priority"], wfs_req["priority"]),
            "prompt_tokens": (estf_req["prompt_tokens"], wfs_req["prompt_tokens"]),
            "predicted_output_tokens": (estf_req["predicted_output_tokens"], wfs_req["predicted_output_tokens"]),
            "predicted_service_proxy": (estf_req["predicted_service_proxy"], wfs_req["predicted_service_proxy"]),
            "queue_age": (estf_req["queue_age"], wfs_req["queue_age"]),
            "laxity_own": (ev["estf_laxity_own"], ev["wfs_laxity_own"]),
        }
        for name, (a, b) in pairs.items():
            row[_feature_prefixed(f"{name}_diff_estf_minus_wfs")] = float(a) - float(b)
        for name, (a, b) in {
            "priority": pairs["priority"],
            "predicted_service_proxy": pairs["predicted_service_proxy"],
            "queue_age": pairs["queue_age"],
            "laxity_own": pairs["laxity_own"],
        }.items():
            row[_feature_prefixed(f"{name}_ratio_estf_over_wfs")] = float(a) / max(float(b), EPS)

        rows.append(row)

    df = pd.DataFrame(rows).sort_values(["scenario_id", "step"]).reset_index(drop=True)
    feature_cols = [c for c in df.columns if c.startswith("feat_")]
    concat_cols = feature_cols
    pair_cols = [c for c in feature_cols if "_diff_estf_minus_wfs" in c or "_ratio_estf_over_wfs" in c]
    return df, feature_cols, concat_cols, pair_cols


def _json_ready(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_ready(v) for v in obj]
    return obj


def _cv_splits(df: pd.DataFrame, *, binary_only: bool) -> list[tuple[np.ndarray, np.ndarray]]:
    data = df[df["oracle_label"].isin(BINARY_LABELS)].reset_index(drop=True) if binary_only else df.reset_index(drop=True)
    n_groups = data["group_key"].nunique()
    n_splits = min(5, n_groups)
    return list(GroupKFold(n_splits=n_splits).split(data, data["oracle_label"], data["group_key"]))


def _preprocessor(cols: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), cols)],
        remainder="drop",
    )


def _classifier_ladder(cols: list[str]) -> dict[str, Any]:
    pre = _preprocessor(cols)
    ladder: dict[str, Any] = {
        "majority": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": Pipeline([
            ("pre", pre),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=0)),
        ]),
        "shallow_tree_depth3": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=0)),
        ]),
        "random_forest_modest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(n_estimators=100, max_depth=4, class_weight="balanced", random_state=0)),
        ]),
    }
    if XGBClassifier is not None:
        ladder["xgboost_modest"] = XGBStringClassifier()
    else:
        ladder["hist_gradient_boosting"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", HistGradientBoostingClassifier(max_iter=50, max_leaf_nodes=8, random_state=0)),
        ])
    return ladder


def _binary_proba(model: Any, x: pd.DataFrame) -> np.ndarray | None:
    if not hasattr(model, "predict_proba"):
        return None
    proba = model.predict_proba(x)
    classes = list(model.classes_) if hasattr(model, "classes_") else list(model.named_steps["clf"].classes_)
    if WFS not in classes:
        return np.zeros(len(x), dtype=float)
    return proba[:, classes.index(WFS)]


def evaluate_classifiers(df: pd.DataFrame, cols: list[str]) -> dict[str, Any]:
    data = df[df["oracle_label"].isin(BINARY_LABELS)].reset_index(drop=True)
    y = data["oracle_label"].to_numpy()
    groups = data["group_key"].to_numpy()
    splits = list(GroupKFold(n_splits=min(5, data["group_key"].nunique())).split(data, y, groups))
    out: dict[str, Any] = {
        "binary_rule": "TIE_OR_UNCERTAIN rows preserved in dataset and excluded from binary sanity-check training",
        "n_binary_rows": int(len(data)),
        "n_excluded_ties": int((df["oracle_label"] == TIE).sum()),
        "models": {},
    }
    y_binary = (y == WFS).astype(int)
    for name, model in _classifier_ladder(cols).items():
        fold_metrics = []
        cm_total = np.zeros((2, 2), dtype=int)
        probs_all: list[float] = []
        true_all: list[int] = []
        pred_oof = np.empty(len(data), dtype=object)
        for fold, (train_idx, test_idx) in enumerate(splits):
            x_train, x_test = data.iloc[train_idx][cols], data.iloc[test_idx][cols]
            y_train, y_test = y[train_idx], y[test_idx]
            model.fit(x_train, y_train)
            pred = model.predict(x_test)
            pred_oof[test_idx] = pred
            cm_total += confusion_matrix(y_test, pred, labels=BINARY_LABELS)
            m: dict[str, Any] = {
                "fold": fold,
                "n_test": int(len(test_idx)),
                "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
                "macro_f1": float(f1_score(y_test, pred, labels=BINARY_LABELS, average="macro", zero_division=0)),
            }
            proba = _binary_proba(model, x_test)
            if proba is not None:
                y_test_bin = (y_test == WFS).astype(int)
                probs_all.extend(proba.tolist())
                true_all.extend(y_test_bin.tolist())
                if len(np.unique(y_test_bin)) == 2:
                    m["roc_auc"] = float(roc_auc_score(y_test_bin, proba))
                    m["pr_auc_wfs"] = float(average_precision_score(y_test_bin, proba))
                else:
                    m["roc_auc"] = float("nan")
                    m["pr_auc_wfs"] = float("nan")
                m["brier"] = float(brier_score_loss(y_test_bin, proba))
            fold_metrics.append(m)
        precision, recall, _, support = precision_recall_fscore_support(
            y, pred_oof, labels=BINARY_LABELS, zero_division=0
        )
        metric_names = ["balanced_accuracy", "macro_f1", "roc_auc", "pr_auc_wfs", "brier"]
        summary = {}
        for metric in metric_names:
            vals = [m[metric] for m in fold_metrics if metric in m and not math.isnan(m[metric])]
            summary[f"{metric}_mean"] = float(np.mean(vals)) if vals else float("nan")
            summary[f"{metric}_std"] = float(np.std(vals, ddof=0)) if vals else float("nan")
        cal = {}
        if probs_all:
            probs = np.asarray(probs_all)
            true = np.asarray(true_all)
            cal = {
                "mean_p_wfs": float(probs.mean()),
                "mean_p_wfs_true_estf": float(probs[true == 0].mean()) if (true == 0).any() else float("nan"),
                "mean_p_wfs_true_wfs": float(probs[true == 1].mean()) if (true == 1).any() else float("nan"),
            }
        out["models"][name] = {
            **summary,
            "per_class_precision": {BINARY_LABELS[i]: float(precision[i]) for i in range(2)},
            "per_class_recall": {BINARY_LABELS[i]: float(recall[i]) for i in range(2)},
            "per_class_support": {BINARY_LABELS[i]: int(support[i]) for i in range(2)},
            "confusion_matrix_labels": BINARY_LABELS,
            "confusion_matrix": cm_total.tolist(),
            "folds": fold_metrics,
            "calibration_summary": cal,
        }
    return out


def _regressor_ladder(cols: list[str]) -> dict[str, Any]:
    pre = _preprocessor(cols)
    ladder: dict[str, Any] = {
        "ridge_regression": Pipeline([("pre", pre), ("reg", Ridge(alpha=1.0))]),
    }
    if XGBRegressor is not None:
        ladder["xgboost_regressor_modest"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("reg", XGBRegressor(n_estimators=50, max_depth=2, learning_rate=0.1, random_state=0)),
        ])
    else:
        ladder["hist_gradient_boosting_regressor"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("reg", HistGradientBoostingRegressor(max_iter=50, max_leaf_nodes=8, random_state=0)),
        ])
    return ladder


def _sign_label(values: np.ndarray) -> np.ndarray:
    return np.where(values > 0, ESTF, np.where(values < 0, WFS, TIE))


def evaluate_regressors(df: pd.DataFrame, cols: list[str]) -> dict[str, Any]:
    data = df.reset_index(drop=True)
    y = data["delta_J"].to_numpy(dtype=float)
    splits = list(GroupKFold(n_splits=min(5, data["group_key"].nunique())).split(data, y, data["group_key"]))
    out = {"models": {}}
    for name, model in _regressor_ladder(cols).items():
        folds = []
        preds = np.full(len(data), np.nan)
        for fold, (train_idx, test_idx) in enumerate(splits):
            model.fit(data.iloc[train_idx][cols], y[train_idx])
            pred = model.predict(data.iloc[test_idx][cols])
            preds[test_idx] = pred
            rho = spearmanr(y[test_idx], pred).correlation if len(np.unique(y[test_idx])) > 1 else float("nan")
            folds.append({
                "fold": fold,
                "n_test": int(len(test_idx)),
                "mae": float(mean_absolute_error(y[test_idx], pred)),
                "spearman": float(rho) if rho == rho else float("nan"),
                "non_tie_sign_accuracy": float((_sign_label(pred)[y[test_idx] != 0] == _sign_label(y[test_idx])[y[test_idx] != 0]).mean())
                if (y[test_idx] != 0).any() else float("nan"),
            })
        rho_all = spearmanr(y, preds).correlation if len(np.unique(y)) > 1 else float("nan")
        out["models"][name] = {
            "mae_mean": float(np.mean([f["mae"] for f in folds])),
            "mae_std": float(np.std([f["mae"] for f in folds], ddof=0)),
            "spearman_mean": float(np.nanmean([f["spearman"] for f in folds])),
            "spearman_std": float(np.nanstd([f["spearman"] for f in folds], ddof=0)),
            "spearman_oof": float(rho_all) if rho_all == rho_all else float("nan"),
            "non_tie_sign_accuracy_mean": float(np.nanmean([f["non_tie_sign_accuracy"] for f in folds])),
            "non_tie_sign_accuracy_oof": float((_sign_label(preds)[y != 0] == _sign_label(y)[y != 0]).mean()),
            "folds": folds,
        }
    return out


def representation_check(df: pd.DataFrame, concat_cols: list[str], pair_cols: list[str]) -> dict[str, Any]:
    return {
        "classification_concat": evaluate_classifiers(df, concat_cols)["models"]["logistic_regression"],
        "classification_pairwise_diff_only": evaluate_classifiers(df, pair_cols)["models"]["logistic_regression"],
        "regression_concat": evaluate_regressors(df, concat_cols)["models"]["ridge_regression"],
        "regression_pairwise_diff_only": evaluate_regressors(df, pair_cols)["models"]["ridge_regression"],
    }


def dataset_quality(df: pd.DataFrame, feature_cols: list[str]) -> dict[str, Any]:
    label_counts = df["oracle_label"].value_counts().reindex(LABELS, fill_value=0)
    margins = df["delta_J"].abs()
    exact_hashes = pd.util.hash_pandas_object(df[feature_cols].round(12), index=False)
    near_hashes = pd.util.hash_pandas_object(df[feature_cols].round(3), index=False)
    by_scenario = df.groupby("scenario_id").agg(
        samples=("sample_id", "count"),
        first_step=("step", "min"),
        last_step=("step", "max"),
        labels=("oracle_label", lambda s: ",".join(sorted(set(s)))),
    ).reset_index()
    temporal_diffs = []
    for _, g in df.sort_values(["scenario_id", "step"]).groupby("scenario_id"):
        diffs = g["step"].diff().dropna().tolist()
        temporal_diffs.extend(diffs)
    return {
        "n_rows": int(len(df)),
        "n_scenarios": int(df["scenario_id"].nunique()),
        "label_counts": {k: int(v) for k, v in label_counts.to_dict().items()},
        "label_fractions": {k: float(v / len(df)) for k, v in label_counts.to_dict().items()},
        "label_by_split": df.groupby(["split", "oracle_label"]).size().unstack(fill_value=0).to_dict(orient="index"),
        "label_by_fav_analysis": df.groupby(["analysis_fav", "oracle_label"]).size().unstack(fill_value=0).to_dict(orient="index"),
        "scenarios_per_class": {
            label: int(df[df["oracle_label"] == label]["scenario_id"].nunique()) for label in LABELS
        },
        "samples_per_scenario": {
            "mean": float(by_scenario["samples"].mean()),
            "median": float(by_scenario["samples"].median()),
            "min": int(by_scenario["samples"].min()),
            "max": int(by_scenario["samples"].max()),
        },
        "temporal_distance_between_samples_steps": {
            "count": int(len(temporal_diffs)),
            "median": float(np.median(temporal_diffs)) if temporal_diffs else float("nan"),
            "min": float(np.min(temporal_diffs)) if temporal_diffs else float("nan"),
            "p25": float(np.percentile(temporal_diffs, 25)) if temporal_diffs else float("nan"),
            "p75": float(np.percentile(temporal_diffs, 75)) if temporal_diffs else float("nan"),
        },
        "duplicate_feature_rows_exact_round12": int(exact_hashes.duplicated().sum()),
        "near_duplicate_feature_rows_round3": int(near_hashes.duplicated().sum()),
        "delta_J_distribution": {
            "mean": float(df["delta_J"].mean()),
            "median": float(df["delta_J"].median()),
            "p25": float(df["delta_J"].quantile(0.25)),
            "p75": float(df["delta_J"].quantile(0.75)),
            "p90_abs": float(margins.quantile(0.90)),
            "p95_abs": float(margins.quantile(0.95)),
            "min": float(df["delta_J"].min()),
            "max": float(df["delta_J"].max()),
            "fraction_exact_tie": float((df["delta_J"] == 0.0).mean()),
        },
        "parameter_coverage": {
            "utilization": sorted(float(x) for x in df["analysis_utilization"].dropna().unique()),
            "skew": sorted(float(x) for x in df["analysis_skew"].dropna().unique()),
            "fav": sorted(str(x) for x in df["analysis_fav"].dropna().unique()),
            "noise": sorted(float(x) for x in df["analysis_noise"].dropna().unique()),
        },
    }


def feature_classification(feature_cols: list[str], df_cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in df_cols:
        if col in feature_cols:
            kind = "ONLINE_CAUSAL_MODEL_FEATURE"
        elif col in {
            "sample_id", "event_id", "scenario_id", "canonical_scenario_id", "step", "time",
            "split", "group_key", "provenance_version", "label_utility_name",
            "continuation_semantics", "horizon_steps", "future_arrivals_included",
            "tie_threshold", "estf_contested_request_id", "wfs_contested_request_id",
            "estf_contested_class_id", "wfs_contested_class_id",
        }:
            kind = "METADATA_ONLY"
        elif col in {"analysis_fav", "analysis_utilization", "analysis_skew", "analysis_noise", "analysis_seed"}:
            kind = "EXPERIMENT_METADATA"
        elif col in {
            "J_ESTF", "J_WFS", "delta_J", "oracle_label", "completion_benefit_label",
            "slo_risk_label", "raw_completion_gt_label_biased_reference",
            "delta_native_whole_branch_raw_biased_reference",
        }:
            kind = "LABEL_OR_FUTURE_OUTCOME"
        else:
            kind = "METADATA_ONLY"
        rows.append({"column": col, "classification": kind})
    return pd.DataFrame(rows)


def write_readme(out_dir: Path, quality: dict[str, Any]) -> None:
    readme = f"""# Family-A Oracle Policy Pilot Dataset v1

Date: {DATASET_DATE}

Pilot-only dataset for learning when ESTF vs WFS should be used in Family-A
scheduler disagreement states. Rows are the 91 repaired symmetric ESTF/WFS
disagreement events already extracted from TRAIN/VAL scenarios.

Label utility:

`J_ESTF = sum(priority * 1[completed and SLO-safe])` over the two directly
contested requests under native ESTF continuation.

`J_WFS = sum(priority * 1[completed and SLO-safe])` over the same two requests
under native WFS continuation.

`delta_J = J_ESTF - J_WFS`; labels are `ESTF`, `WFS`, or `TIE_OR_UNCERTAIN`
with exact ties only (`tie_threshold=0.0`).

This avoids the prior raw-completed-count bias by using priority-weighted SLO
credit, but it is still a contested-pair bounded-window target, not full
scenario ANWG.

Rows: {quality["n_rows"]}
Scenarios: {quality["n_scenarios"]}
Labels: {quality["label_counts"]}

Files:

- `pilot_rows.csv`
- `schema.json`
- `feature_classification.csv`
- `provenance.json`
- `quality_summary.json`
- `model_sanity_summary.json`
- `delta_j_regression_summary.json`
- `representation_check_summary.json`
"""
    (out_dir / "README.md").write_text(readme)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df, feature_cols, concat_cols, pair_cols = build_rows()
    quality = dataset_quality(df, feature_cols)
    cls = evaluate_classifiers(df, concat_cols)
    reg = evaluate_regressors(df, concat_cols)
    rep = representation_check(df, concat_cols, pair_cols)
    feature_classes = feature_classification(feature_cols, list(df.columns))

    schema = {
        "schema_version": SCHEMA_VERSION,
        "row_unit": "one eligible online scheduler decision state where ESTF and WFS disagree symmetrically",
        "label": {
            "utility": "contested_pair_priority_weighted_slo_native_continuation",
            "J_ESTF": "sum over two contested requests of br_estf_estf_weighted_contribution",
            "J_WFS": "sum over two contested requests of br_wfs_wfs_weighted_contribution",
            "delta_J": "J_ESTF - J_WFS",
            "classes": LABELS,
            "tie_threshold": 0.0,
        },
        "feature_columns": feature_cols,
        "concat_feature_columns": concat_cols,
        "pairwise_difference_feature_columns": pair_cols,
        "group_key": "scenario_id",
        "forbidden_model_feature_patterns": [
            "scenario_id", "canonical_scenario_id", "split", "seed", "analysis_fav",
            "J_", "delta_J", "oracle_label", "br_", "raw_completion",
        ],
    }
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "date": DATASET_DATE,
        "git_head": _git_head(),
        "git_dirty": _git_dirty(),
        "inputs": {
            str(CONTESTED_REQUESTS_CSV.relative_to(REPO_ROOT)): _sha256_file(CONTESTED_REQUESTS_CSV),
            str(CONSTRAINED_EVENTS_CSV.relative_to(REPO_ROOT)): _sha256_file(CONSTRAINED_EVENTS_CSV),
            str(OBS_EVENTS_CSV.relative_to(REPO_ROOT)): _sha256_file(OBS_EVENTS_CSV),
        },
        "command": "python3 scripts/build_family_a_oracle_policy_pilot_v1.py",
        "offline_only": True,
        "test_rows_used": 0,
    }

    df.to_csv(out_dir / "pilot_rows.csv", index=False)
    feature_classes.to_csv(out_dir / "feature_classification.csv", index=False)
    (out_dir / "schema.json").write_text(json.dumps(_json_ready(schema), indent=2, sort_keys=True) + "\n")
    (out_dir / "provenance.json").write_text(json.dumps(_json_ready(provenance), indent=2, sort_keys=True) + "\n")
    (out_dir / "quality_summary.json").write_text(json.dumps(_json_ready(quality), indent=2, sort_keys=True) + "\n")
    (out_dir / "model_sanity_summary.json").write_text(json.dumps(_json_ready(cls), indent=2, sort_keys=True) + "\n")
    (out_dir / "delta_j_regression_summary.json").write_text(json.dumps(_json_ready(reg), indent=2, sort_keys=True) + "\n")
    (out_dir / "representation_check_summary.json").write_text(json.dumps(_json_ready(rep), indent=2, sort_keys=True) + "\n")
    write_readme(out_dir, quality)

    manifest = {
        "files": sorted(p.name for p in out_dir.iterdir() if p.is_file()),
        "pilot_rows_sha256": _sha256_file(out_dir / "pilot_rows.csv"),
        "quality_summary_sha256": _sha256_file(out_dir / "quality_summary.json"),
    }
    (out_dir / "manifest.json").write_text(json.dumps(_json_ready(manifest), indent=2, sort_keys=True) + "\n")

    print(json.dumps({
        "output_dir": str(out_dir),
        "n_rows": quality["n_rows"],
        "n_scenarios": quality["n_scenarios"],
        "label_counts": quality["label_counts"],
        "best_binary_balanced_accuracy": max(
            m["balanced_accuracy_mean"] for m in cls["models"].values()
            if not math.isnan(m["balanced_accuracy_mean"])
        ),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
