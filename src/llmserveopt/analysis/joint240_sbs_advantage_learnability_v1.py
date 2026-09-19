"""Held-out learnability for SBS-relative one-step advantage labels.

This module trains only offline regressors on frozen counterfactual labels.  It
does not run the simulator, a closed-loop scheduler, or real serving.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SCHEMA_VERSION = "joint240_sbs_advantage_learnability_v1.0.0"
SEED = 20260919
PRIMARY_TARGET = "a_sbs_anwg"
SBS_POLICY = "kv_constrained_online"
EXPECTED_ROWS_SHA256 = "a96af891b70bcd0895440207df6febb5d1c1545af4eaaf455367e56152950dca"
EXPECTED_MAP_SHA256 = "7a4118b65e4222e189aa1139ba548ade99951ae285a50b7b193fec6ef55d1324"
EXPECTED_NON_SBS_ROWS = 21858
EXPECTED_STATES = 8888
EXPECTED_SCENARIOS = 236
EXPECTED_FOLDS = 5
THRESHOLDS = [0.0, 0.001, 0.0025, 0.005, 0.01, 0.02]
BOOTSTRAP_REPLICATES = 10_000
DISAGREEMENT_RATE_D_SBS = 8888 / 453016


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_ROOT = Path(
    "/home/soroush/llm-serving-heuristic-evolution-joint240-sbs-targeted-terminal-label-full-v1"
    "/experiments/joint240_sbs_targeted_terminal_label_full_v1/full_v1/analysis_v1"
)
DEFAULT_ROWS = DEFAULT_INPUT_ROOT / "state_action_rows_full.csv"
DEFAULT_MAP = DEFAULT_INPUT_ROOT / "state_policy_action_map_full.csv"
DEFAULT_OUT = ROOT / "experiments" / "joint240_sbs_advantage_learnability_v1" / "run_v1"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git(args: Sequence[str], cwd: Path = ROOT) -> Optional[str]:
    try:
        return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()
    except Exception:
        return None


def write_json(path: Path, obj: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def stable_json(obj: Mapping[str, Any]) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def numeric_summary(values: Iterable[float]) -> Dict[str, Optional[float]]:
    arr = np.asarray([float(x) for x in values if pd.notna(x)], dtype=float)
    if arr.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p5": None,
            "p10": None,
            "p25": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "min": None,
            "max": None,
        }
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p5": float(np.percentile(arr, 5)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def validate_input_checksums(rows_path: Path, map_path: Path) -> Dict[str, Any]:
    rows_sha = sha256_file(rows_path)
    map_sha = sha256_file(map_path)
    return {
        "rows_path": str(rows_path),
        "map_path": str(map_path),
        "rows_sha256": rows_sha,
        "map_sha256": map_sha,
        "rows_sha256_matches": rows_sha == EXPECTED_ROWS_SHA256,
        "map_sha256_matches": map_sha == EXPECTED_MAP_SHA256,
    }


def learnable_columns(rows: pd.DataFrame, feature_set: str) -> List[str]:
    state_cols = [c for c in rows.columns if c.startswith("state__")]
    action_cols = [c for c in rows.columns if c.startswith("action__")]
    if feature_set == "STATE_CORE_V1":
        cols = state_cols
    elif feature_set == "STATE_ACTION_V1":
        cols = state_cols + action_cols
    else:
        raise ValueError(f"unknown feature set: {feature_set}")
    forbidden = leakage_columns(cols)
    if forbidden:
        raise ValueError({"feature_leakage_columns": forbidden})
    return cols


def leakage_columns(cols: Sequence[str]) -> List[str]:
    forbidden_markers = [
        "state_id",
        "scenario_id",
        "fold",
        "policy",
        "q_sbs",
        "a_sbs",
        "target",
        "terminal",
        "future",
        "actual_output",
        "post_decision",
        "vbs",
        "traj_",
        "num_completed",
        "num_dropped",
        "sim_duration",
    ]
    return [c for c in cols if any(marker in c.lower() for marker in forbidden_markers)]


def load_learning_table(rows_path: Path, map_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    checks = validate_input_checksums(rows_path, map_path)
    if not checks["rows_sha256_matches"] or not checks["map_sha256_matches"]:
        raise ValueError({"input_checksum_mismatch": checks})
    rows = pd.read_csv(rows_path)
    maps = pd.read_csv(map_path)
    non_sbs = rows[rows["is_sbs_action"].astype(int) == 0].copy()
    non_sbs[PRIMARY_TARGET] = pd.to_numeric(non_sbs[PRIMARY_TARGET], errors="raise")
    feature_counts = {
        "state_features": len([c for c in rows.columns if c.startswith("state__")]),
        "action_features": len([c for c in rows.columns if c.startswith("action__")]),
    }
    integrity = {
        **checks,
        "rows": int(len(rows)),
        "non_sbs_rows": int(len(non_sbs)),
        "states": int(rows["state_id"].nunique()),
        "scenarios": int(rows["scenario_id"].nunique()),
        "folds": int(rows["fold"].nunique()),
        "feature_counts": feature_counts,
    }
    if integrity["non_sbs_rows"] != EXPECTED_NON_SBS_ROWS:
        raise ValueError({"unexpected_non_sbs_rows": integrity})
    if integrity["states"] != EXPECTED_STATES or integrity["scenarios"] != EXPECTED_SCENARIOS:
        raise ValueError({"unexpected_dataset_shape": integrity})
    if feature_counts != {"state_features": 95, "action_features": 70}:
        raise ValueError({"unexpected_feature_counts": feature_counts})
    return non_sbs.reset_index(drop=True), maps, integrity


def state_equal_weights(df: pd.DataFrame) -> np.ndarray:
    counts = df.groupby("state_id")["canonical_action_id"].transform("count").astype(float)
    return (1.0 / counts).to_numpy()


def outer_split_audit(df: pd.DataFrame) -> Dict[str, Any]:
    scenario_folds = df[["scenario_id", "fold"]].drop_duplicates()
    per_scenario = scenario_folds.groupby("scenario_id")["fold"].nunique()
    fold_counts = df.drop_duplicates("state_id").groupby("fold")["state_id"].nunique()
    audits = []
    for fold in sorted(df["fold"].unique()):
        test = df[df["fold"] == fold]
        train = df[df["fold"] != fold]
        audits.append(
            {
                "outer_fold": int(fold),
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                "train_states": int(train["state_id"].nunique()),
                "test_states": int(test["state_id"].nunique()),
                "train_scenarios": int(train["scenario_id"].nunique()),
                "test_scenarios": int(test["scenario_id"].nunique()),
                "scenario_overlap": int(
                    len(set(train["scenario_id"].astype(str)) & set(test["scenario_id"].astype(str)))
                ),
                "state_overlap": int(
                    len(set(train["state_id"].astype(str)) & set(test["state_id"].astype(str)))
                ),
            }
        )
    return {
        "scenario_assigned_to_multiple_folds": int((per_scenario > 1).sum()),
        "fold_state_counts": {str(int(k)): int(v) for k, v in fold_counts.items()},
        "outer_folds": audits,
    }


def model_grid(model_name: str, *, n_jobs: int = 1) -> List[Dict[str, Any]]:
    if model_name == "DUMMY_ZERO":
        return [{"model_name": model_name}]
    if model_name == "RIDGE":
        return [{"model_name": model_name, "alpha": a} for a in [0.01, 0.1, 1.0, 10.0, 100.0]]
    if model_name == "HIST_GRADIENT_BOOSTING":
        return [
            {
                "model_name": model_name,
                "max_iter": 100,
                "learning_rate": lr,
                "max_leaf_nodes": leaves,
                "l2_regularization": l2,
            }
            for lr, leaves, l2 in itertools.product([0.05, 0.1], [15, 31], [0.0, 0.1])
        ]
    if model_name == "EXTRA_TREES":
        return [
            {
                "model_name": model_name,
                "n_estimators": 100,
                "max_depth": depth,
                "min_samples_leaf": leaf,
                "max_features": mf,
                "n_jobs": n_jobs,
            }
            for depth, leaf, mf in itertools.product([8, None], [5, 20], [0.5, 1.0])
        ]
    raise ValueError(f"unknown model: {model_name}")


def build_model(config: Mapping[str, Any]):
    name = config["model_name"]
    if name == "DUMMY_ZERO":
        return DummyRegressor(strategy="constant", constant=0.0)
    if name == "RIDGE":
        return make_pipeline(StandardScaler(), Ridge(alpha=float(config["alpha"])))
    if name == "HIST_GRADIENT_BOOSTING":
        return HistGradientBoostingRegressor(
            max_iter=int(config["max_iter"]),
            learning_rate=float(config["learning_rate"]),
            max_leaf_nodes=int(config["max_leaf_nodes"]),
            l2_regularization=float(config["l2_regularization"]),
            random_state=SEED,
        )
    if name == "EXTRA_TREES":
        return ExtraTreesRegressor(
            n_estimators=int(config["n_estimators"]),
            max_depth=config["max_depth"],
            min_samples_leaf=int(config["min_samples_leaf"]),
            max_features=float(config["max_features"]),
            random_state=SEED,
            n_jobs=int(config.get("n_jobs", 1)),
        )
    raise ValueError(name)


def fit_predict(
    config: Mapping[str, Any],
    train: pd.DataFrame,
    predict_df: pd.DataFrame,
    cols: Sequence[str],
    *,
    weighted: bool,
) -> np.ndarray:
    model = build_model(config)
    sample_weight = state_equal_weights(train) if weighted else None
    x_train = train[list(cols)].to_numpy(dtype=float)
    y_train = train[PRIMARY_TARGET].to_numpy(dtype=float)
    x_pred = predict_df[list(cols)].to_numpy(dtype=float)
    if sample_weight is not None:
        model.fit(x_train, y_train, **_sample_weight_kwargs(model, sample_weight))
    else:
        model.fit(x_train, y_train)
    return np.asarray(model.predict(x_pred), dtype=float)


def _sample_weight_kwargs(model: Any, weights: np.ndarray) -> Dict[str, Any]:
    if hasattr(model, "steps"):
        # Pipeline with final Ridge step.
        return {"ridge__sample_weight": weights}
    return {"sample_weight": weights}


def selector_decisions(df: pd.DataFrame, pred_col: str, tau: float) -> pd.DataFrame:
    decisions = []
    for sid, g in df.groupby("state_id", sort=False):
        idx = g[pred_col].astype(float).idxmax()
        best = g.loc[idx]
        override = bool(float(best[pred_col]) > float(tau))
        realized = float(best[PRIMARY_TARGET]) if override else 0.0
        oracle = max(0.0, float(g[PRIMARY_TARGET].max()))
        decisions.append(
            {
                "state_id": sid,
                "scenario_id": best["scenario_id"],
                "fold": int(best["fold"]),
                "selected_canonical_action_id": best["canonical_action_id"] if override else "SBS_ABSTAIN",
                "selected_predicted_advantage": float(best[pred_col]),
                "threshold": float(tau),
                "override": int(override),
                "realized_gain": realized,
                "oracle_gain": oracle,
                "regret_to_oracle": float(oracle - realized),
                "selected_true_advantage": float(best[PRIMARY_TARGET]) if override else 0.0,
                "beneficial_override": int(override and float(best[PRIMARY_TARGET]) > 0),
                "harmful_override": int(override and float(best[PRIMARY_TARGET]) < 0),
                "zero_effect_override": int(override and float(best[PRIMARY_TARGET]) == 0),
            }
        )
    return pd.DataFrame(decisions)


def selector_metrics(decisions: pd.DataFrame) -> Dict[str, Any]:
    realized = pd.to_numeric(decisions["realized_gain"])
    overrides = decisions[decisions["override"].astype(int) == 1]
    harmful = decisions[decisions["harmful_override"].astype(int) == 1]
    oracle_sum = float(decisions["oracle_gain"].sum())
    override_count = int(len(overrides))
    return {
        "states": int(len(decisions)),
        "mean_realized_gain": float(realized.mean()),
        "median_realized_gain": float(realized.median()),
        "total_realized_gain": float(realized.sum()),
        "override_count": override_count,
        "override_rate": float(override_count / len(decisions)) if len(decisions) else 0.0,
        "beneficial_override_count": int(decisions["beneficial_override"].sum()),
        "beneficial_override_rate": float(decisions["beneficial_override"].mean()),
        "harmful_override_count": int(decisions["harmful_override"].sum()),
        "harmful_override_rate": float(decisions["harmful_override"].mean()),
        "zero_effect_override_count": int(decisions["zero_effect_override"].sum()),
        "zero_effect_override_rate": float(decisions["zero_effect_override"].mean()),
        "precision_among_overrides": float(decisions["beneficial_override"].sum() / override_count)
        if override_count
        else None,
        "mean_gain_conditional_on_overriding": float(overrides["realized_gain"].mean())
        if override_count
        else None,
        "mean_loss_conditional_on_harmful_override": float(harmful["realized_gain"].mean())
        if len(harmful)
        else None,
        "p5_state_realized_gain": float(realized.quantile(0.05)),
        "p10_state_realized_gain": float(realized.quantile(0.10)),
        "maximum_harmful_override": float(realized.min()),
        "oracle_total_gain": oracle_sum,
        "oracle_mean_gain": float(decisions["oracle_gain"].mean()),
        "gap_closure": float(realized.sum() / oracle_sum) if oracle_sum > 0 else None,
        "mean_regret_to_oracle": float(decisions["regret_to_oracle"].mean()),
    }


def tune_threshold(df: pd.DataFrame, pred_col: str) -> tuple[float, Dict[str, Any]]:
    results = []
    for tau in THRESHOLDS:
        decisions = selector_decisions(df, pred_col, tau)
        metrics = selector_metrics(decisions)
        results.append({"threshold": tau, **metrics})
    best = sorted(
        results,
        key=lambda r: (float(r["mean_realized_gain"]), float(r["threshold"])),
        reverse=True,
    )[0]
    return float(best["threshold"]), {"threshold_grid_results": results, "selected": best}


def prediction_diagnostics(df: pd.DataFrame, pred_col: str) -> Dict[str, Any]:
    y = df[PRIMARY_TARGET].to_numpy(dtype=float)
    pred = df[pred_col].to_numpy(dtype=float)
    nonzero = y != 0
    pred_positive = pred > 0
    true_positive = y > 0
    spearman = scipy_stats.spearmanr(y, pred).correlation if len(np.unique(pred)) > 1 else 0.0
    pearson = scipy_stats.pearsonr(y, pred).statistic if len(np.unique(pred)) > 1 else 0.0
    bins = pd.qcut(pd.Series(pred), q=10, duplicates="drop")
    cal = (
        pd.DataFrame({"bin": bins.astype(str), "pred": pred, "y": y})
        .groupby("bin", sort=False)
        .agg(n=("y", "size"), mean_pred=("pred", "mean"), mean_true=("y", "mean"), positive_rate=("y", lambda s: float((s > 0).mean())))
        .reset_index()
        .to_dict(orient="records")
    )
    return {
        "rows": int(len(df)),
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(math.sqrt(mean_squared_error(y, pred))),
        "spearman": float(spearman) if not math.isnan(float(spearman)) else 0.0,
        "pearson": float(pearson) if not math.isnan(float(pearson)) else 0.0,
        "sign_accuracy_nonzero": float((np.sign(pred[nonzero]) == np.sign(y[nonzero])).mean())
        if nonzero.any()
        else None,
        "predicted_positive_precision": float((true_positive & pred_positive).sum() / pred_positive.sum())
        if pred_positive.sum()
        else None,
        "predicted_positive_recall": float((true_positive & pred_positive).sum() / true_positive.sum())
        if true_positive.sum()
        else None,
        "calibration_bins": cal,
    }


def scenario_metrics(decisions: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, Any]]:
    scen = decisions.groupby("scenario_id").agg(
        fold=("fold", "first"),
        states=("state_id", "nunique"),
        scenario_gain=("realized_gain", "sum"),
        mean_state_gain=("realized_gain", "mean"),
        overrides=("override", "sum"),
        beneficial_overrides=("beneficial_override", "sum"),
        harmful_overrides=("harmful_override", "sum"),
        oracle_gain=("oracle_gain", "sum"),
    ).reset_index()
    summary = {
        "scenarios": int(len(scen)),
        "mean_scenario_gain": float(scen["scenario_gain"].mean()),
        "median_scenario_gain": float(scen["scenario_gain"].median()),
        "scenarios_positive_gain": int((scen["scenario_gain"] > 0).sum()),
        "scenarios_zero_gain": int((scen["scenario_gain"] == 0).sum()),
        "scenarios_negative_gain": int((scen["scenario_gain"] < 0).sum()),
        "worst_scenario_gain": float(scen["scenario_gain"].min()),
        "p10_scenario_gain": float(scen["scenario_gain"].quantile(0.10)),
        "max_harmful_overrides_one_scenario": int(scen["harmful_overrides"].max()),
    }
    return scen, summary


def scenario_bootstrap(decisions: pd.DataFrame, *, replicates: int = BOOTSTRAP_REPLICATES, seed: int = SEED) -> tuple[pd.DataFrame, Dict[str, Any]]:
    scenarios = np.asarray(sorted(decisions["scenario_id"].unique()))
    by_scenario = {sid: g.copy() for sid, g in decisions.groupby("scenario_id")}
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(int(replicates)):
        sample = rng.choice(scenarios, size=len(scenarios), replace=True)
        boot = pd.concat([by_scenario[sid] for sid in sample], ignore_index=True)
        metrics = selector_metrics(boot)
        rows.append(
            {
                "bootstrap_index": i,
                "mean_realized_gain": metrics["mean_realized_gain"],
                "mean_scenario_gain": float(
                    boot.groupby("scenario_id", sort=False)["realized_gain"].sum().mean()
                ),
                "gap_closure": metrics["gap_closure"],
                "precision_among_overrides": metrics["precision_among_overrides"],
            }
        )
    dist = pd.DataFrame(rows)
    summary = {}
    for col in ["mean_realized_gain", "mean_scenario_gain", "gap_closure", "precision_among_overrides"]:
        s = dist[col].dropna()
        summary[col] = {
            "mean": float(s.mean()) if len(s) else None,
            "ci95_low": float(s.quantile(0.025)) if len(s) else None,
            "ci95_high": float(s.quantile(0.975)) if len(s) else None,
        }
    return dist, summary


@dataclass(frozen=True)
class FitResult:
    feature_set: str
    model_name: str
    outer_fold: int
    selected_config: Dict[str, Any]
    selected_threshold: float
    train_selection_metric: float
    test_predictions: pd.DataFrame
    train_threshold_summary: Dict[str, Any]


def crossfit_config(
    train_pool: pd.DataFrame,
    cols: Sequence[str],
    config: Mapping[str, Any],
    *,
    weighted: bool,
) -> pd.DataFrame:
    preds = []
    train_folds = sorted(train_pool["fold"].unique())
    for inner_fold in train_folds:
        inner_train = train_pool[train_pool["fold"] != inner_fold]
        inner_val = train_pool[train_pool["fold"] == inner_fold]
        p = fit_predict(config, inner_train, inner_val, cols, weighted=weighted)
        part = inner_val.copy()
        part["predicted_advantage"] = p
        preds.append(part)
    return pd.concat(preds, ignore_index=True)


def select_config_for_family(
    train_pool: pd.DataFrame,
    cols: Sequence[str],
    model_name: str,
    *,
    weighted: bool,
    n_jobs: int,
) -> tuple[Dict[str, Any], float, Dict[str, Any], pd.DataFrame]:
    candidates = []
    for config in model_grid(model_name, n_jobs=n_jobs):
        cf = crossfit_config(train_pool, cols, config, weighted=weighted)
        tau, threshold_summary = tune_threshold(cf, "predicted_advantage")
        metric = float(threshold_summary["selected"]["mean_realized_gain"])
        candidates.append((metric, tau, stable_json(config), dict(config), threshold_summary, cf))
    best = sorted(candidates, key=lambda x: (x[0], x[1], x[2]), reverse=True)[0]
    return best[3], float(best[1]), best[4], best[5]


def fit_outer_fold(
    df: pd.DataFrame,
    feature_set: str,
    model_name: str,
    outer_fold: int,
    *,
    weighted: bool,
    n_jobs: int,
) -> FitResult:
    cols = learnable_columns(df, feature_set)
    train_pool = df[df["fold"] != outer_fold].copy()
    test = df[df["fold"] == outer_fold].copy()
    selected_config, selected_tau, threshold_summary, _ = select_config_for_family(
        train_pool,
        cols,
        model_name,
        weighted=weighted,
        n_jobs=n_jobs,
    )
    test = test.copy()
    test["predicted_advantage"] = fit_predict(selected_config, train_pool, test, cols, weighted=weighted)
    return FitResult(
        feature_set=feature_set,
        model_name=model_name,
        outer_fold=int(outer_fold),
        selected_config=selected_config,
        selected_threshold=float(selected_tau),
        train_selection_metric=float(threshold_summary["selected"]["mean_realized_gain"]),
        test_predictions=test,
        train_threshold_summary=threshold_summary,
    )


def evaluate_oof(
    pred: pd.DataFrame,
    *,
    selected_thresholds: Mapping[int, float],
    label: str,
    out_dir: Path,
) -> Dict[str, Any]:
    parts = []
    fixed_summaries = {}
    for tau in THRESHOLDS:
        decisions = selector_decisions(pred, "predicted_advantage", tau)
        fixed_summaries[str(tau)] = selector_metrics(decisions)
    for fold, g in pred.groupby("fold", sort=True):
        tau = float(selected_thresholds[int(fold)])
        parts.append(selector_decisions(g, "predicted_advantage", tau))
    decisions = pd.concat(parts, ignore_index=True)
    scen, scen_summary = scenario_metrics(decisions)
    pred_diag = prediction_diagnostics(pred, "predicted_advantage")
    boot_dist, boot_summary = scenario_bootstrap(decisions)
    prefix = label.replace("/", "_")
    pred.to_csv(out_dir / f"oof_action_predictions.{prefix}.csv", index=False)
    decisions.to_csv(out_dir / f"oof_state_decisions.{prefix}.csv", index=False)
    scen.to_csv(out_dir / f"scenario_metrics.{prefix}.csv", index=False)
    boot_dist.to_csv(out_dir / f"scenario_bootstrap.{prefix}.csv", index=False)
    summary = {
        "label": label,
        "selector_metrics": selector_metrics(decisions),
        "fixed_threshold_metrics": fixed_summaries,
        "prediction_diagnostics": pred_diag,
        "scenario_metrics": scen_summary,
        "bootstrap": boot_summary,
        "open_loop_d_sbs_descriptive_mean_gain_per_decision": float(
            DISAGREEMENT_RATE_D_SBS * selector_metrics(decisions)["mean_realized_gain"]
        ),
    }
    write_json(out_dir / f"summary.{prefix}.json", summary)
    return summary


def choose_primary_by_training(fold_results: List[FitResult]) -> Dict[int, str]:
    chosen: Dict[int, str] = {}
    for fold, results in itertools.groupby(
        sorted(fold_results, key=lambda r: (r.outer_fold, r.feature_set, r.model_name)),
        key=lambda r: r.outer_fold,
    ):
        state_action = [r for r in results if r.feature_set == "STATE_ACTION_V1"]
        best = sorted(
            state_action,
            key=lambda r: (r.train_selection_metric, r.selected_threshold, r.model_name),
            reverse=True,
        )[0]
        chosen[int(fold)] = best.model_name
    return chosen


def final_interpretation(summary: Mapping[str, Any]) -> str:
    metrics = summary["selector_metrics"]
    boot = summary["bootstrap"]["mean_realized_gain"]
    scen = summary["scenario_metrics"]
    if (
        metrics["mean_realized_gain"] > 0
        and boot["ci95_low"] is not None
        and boot["ci95_low"] > 0
        and scen["scenarios_positive_gain"] > 20
        and scen["scenarios_negative_gain"] < scen["scenarios_positive_gain"]
        and metrics["total_realized_gain"] > 0
    ):
        return "HELD_OUT_OVERRIDE_SIGNAL_CONFIRMED"
    if metrics["mean_realized_gain"] > 0:
        return "HELD_OUT_SIGNAL_POSITIVE_BUT_UNCERTAIN"
    pred = summary["prediction_diagnostics"]
    if (pred.get("spearman") or 0.0) > 0 or (pred.get("predicted_positive_precision") or 0.0) > 0.13:
        return "PREDICTION_SIGNAL_WITHOUT_POLICY_GAIN"
    return "NO_USEFUL_HELD_OUT_SIGNAL"


def run_experiment(
    *,
    rows_path: Path,
    map_path: Path,
    out_dir: Path,
    n_jobs: int = 1,
    weighted: bool = True,
    quick: bool = False,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    df, maps, integrity = load_learning_table(rows_path, map_path)
    if quick:
        # Test-only path: keep all invariants but reduce model grid by using dummy and ridge only.
        model_names = ["DUMMY_ZERO", "RIDGE"]
    else:
        model_names = ["DUMMY_ZERO", "RIDGE", "HIST_GRADIENT_BOOSTING", "EXTRA_TREES"]
    feature_sets = ["STATE_ACTION_V1", "STATE_CORE_V1"]
    split_audit = outer_split_audit(df)
    write_json(out_dir / "input_integrity_and_split_audit.json", {"input": integrity, "splits": split_audit})
    rows = []
    fold_results: List[FitResult] = []
    started = time.perf_counter()
    for feature_set in feature_sets:
        for model_name in model_names:
            for outer_fold in sorted(df["fold"].unique()):
                res = fit_outer_fold(
                    df,
                    feature_set,
                    model_name,
                    int(outer_fold),
                    weighted=weighted,
                    n_jobs=n_jobs,
                )
                fold_results.append(res)
                rows.append(
                    {
                        "feature_set": feature_set,
                        "model_name": model_name,
                        "outer_fold": int(outer_fold),
                        "selected_config": stable_json(res.selected_config),
                        "selected_threshold": float(res.selected_threshold),
                        "train_selection_metric": float(res.train_selection_metric),
                    }
                )
    hyper = pd.DataFrame(rows)
    hyper.to_csv(out_dir / "selected_hyperparameters_and_thresholds.csv", index=False)
    summaries = {}
    for feature_set in feature_sets:
        for model_name in model_names:
            relevant = [r for r in fold_results if r.feature_set == feature_set and r.model_name == model_name]
            pred = pd.concat([r.test_predictions for r in relevant], ignore_index=True)
            thresholds = {r.outer_fold: r.selected_threshold for r in relevant}
            label = f"{feature_set}.{model_name}"
            summaries[label] = evaluate_oof(pred, selected_thresholds=thresholds, label=label, out_dir=out_dir)
    primary_by_fold = choose_primary_by_training(fold_results)
    primary_parts = []
    primary_thresholds = {}
    for r in fold_results:
        if r.feature_set == "STATE_ACTION_V1" and primary_by_fold[r.outer_fold] == r.model_name:
            primary_parts.append(r.test_predictions.assign(primary_selected_model=r.model_name))
            primary_thresholds[r.outer_fold] = r.selected_threshold
    primary_pred = pd.concat(primary_parts, ignore_index=True)
    primary_summary = evaluate_oof(
        primary_pred,
        selected_thresholds=primary_thresholds,
        label="STATE_ACTION_V1.PRIMARY_TRAIN_SELECTED",
        out_dir=out_dir,
    )
    verdict = final_interpretation(primary_summary)
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": platform.node(),
        "git_head": git(["rev-parse", "HEAD"]),
        "git_branch": git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "weighted": bool(weighted),
        "quick": bool(quick),
        "n_jobs": int(n_jobs),
        "model_names": model_names,
        "feature_sets": feature_sets,
        "threshold_grid": THRESHOLDS,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "wall_seconds": float(time.perf_counter() - started),
    }
    final = {
        "provenance": provenance,
        "input_integrity": integrity,
        "split_audit": split_audit,
        "selected_hyperparameters_path": str(out_dir / "selected_hyperparameters_and_thresholds.csv"),
        "primary_selected_model_by_fold": primary_by_fold,
        "summaries": summaries,
        "primary_summary": primary_summary,
        "final_interpretation": verdict,
        "artifact_checksums": artifact_checksums(out_dir),
    }
    write_json(out_dir / "learnability_summary.json", final)
    return final


def artifact_checksums(out_dir: Path) -> Dict[str, str]:
    out = {}
    for path in sorted(out_dir.glob("*")):
        if path.is_file() and path.suffix in {".csv", ".json", ".md"}:
            out[path.name] = sha256_file(path)
    return out


def cmd_run(args: argparse.Namespace) -> None:
    summary = run_experiment(
        rows_path=Path(args.rows).resolve(),
        map_path=Path(args.policy_map).resolve(),
        out_dir=Path(args.output_dir).resolve(),
        n_jobs=int(args.n_jobs),
        weighted=not args.unweighted,
        quick=bool(args.quick),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rows", default=str(DEFAULT_ROWS))
    p.add_argument("--policy-map", default=str(DEFAULT_MAP))
    p.add_argument("--output-dir", default=str(DEFAULT_OUT))
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--unweighted", action="store_true")
    p.add_argument("--quick", action="store_true", help="test/smoke mode with only dummy and ridge")
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("run")
    sp.set_defaults(func=cmd_run)
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
