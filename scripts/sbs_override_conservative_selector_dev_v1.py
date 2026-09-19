#!/usr/bin/env python3
"""Development-only conservative SBS override selector freeze.

This script trains and freezes a selector only from the joint240 development
corpus. It refuses fresh confirmatory label paths by construction.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor


SCHEMA_VERSION = "sbs_override_conservative_selector_dev_v1.0.0"
SEED = 20260919
FRESH_FORBIDDEN_SUBSTRING = "sbs_override_fresh_id_confirmatory_terminal_label_v1"
PRIMARY_TARGET = "a_sbs_anwg"
EXPECTED_ROWS_SHA256 = "a96af891b70bcd0895440207df6febb5d1c1545af4eaaf455367e56152950dca"
EXPECTED_MAP_SHA256 = "7a4118b65e4222e189aa1139ba548ade99951ae285a50b7b193fec6ef55d1324"
EXPECTED_NON_SBS_ROWS = 21858
EXPECTED_STATES = 8888
EXPECTED_SCENARIOS = 236
EXPECTED_FOLDS = 5
EXPECTED_STATE_FEATURES = 95
EXPECTED_ACTION_FEATURES = 70
TAU_GRID = [0.0, 0.001, 0.0025, 0.005, 0.01, 0.02]
RESIDUAL_QUANTILES = {"q90": 0.90, "q95": 0.95}
V1_REFERENCE = {
    "mean_realized_gain_per_state": 0.00019594132039085852,
    "bootstrap_ci_low": -0.00020766285012871125,
    "harmful_overrides": 450,
    "negative_scenarios": 57,
    "positive_folds": 3,
}

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = Path(
    "/home/soroush/llm-serving-heuristic-evolution-joint240-sbs-targeted-terminal-label-full-v1"
    "/experiments/joint240_sbs_targeted_terminal_label_full_v1/full_v1/analysis_v1"
)
DEFAULT_ROWS = DEFAULT_INPUT_ROOT / "state_action_rows_full.csv"
DEFAULT_MAP = DEFAULT_INPUT_ROOT / "state_policy_action_map_full.csv"
DEFAULT_OUT = ROOT / "experiments" / "sbs_override_conservative_selector_dev_v1" / "run_v1"


def reject_fresh_confirmatory_path(path: Path | str) -> None:
    text = str(path).lower()
    if FRESH_FORBIDDEN_SUBSTRING in text:
        raise ValueError(
            f"Refusing confirmatory-label path for development training/evaluation: {path}"
        )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def git(args: Sequence[str], cwd: Path = ROOT) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()
    except Exception:
        return None


def stable_json(obj: Mapping[str, Any]) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def check_inputs(rows_path: Path, map_path: Path) -> dict[str, Any]:
    reject_fresh_confirmatory_path(rows_path)
    reject_fresh_confirmatory_path(map_path)
    rows_sha = sha256_file(rows_path)
    map_sha = sha256_file(map_path)
    checks = {
        "rows_path": str(rows_path),
        "map_path": str(map_path),
        "rows_sha256": rows_sha,
        "map_sha256": map_sha,
        "rows_sha256_matches": rows_sha == EXPECTED_ROWS_SHA256,
        "map_sha256_matches": map_sha == EXPECTED_MAP_SHA256,
    }
    if not checks["rows_sha256_matches"] or not checks["map_sha256_matches"]:
        raise ValueError({"development_input_checksum_mismatch": checks})
    return checks


def leakage_columns(cols: Sequence[str]) -> list[str]:
    forbidden = [
        "state_id",
        "scenario_id",
        "fold",
        "policy",
        "q_sbs",
        "a_sbs",
        "target",
        "terminal",
        "future",
        "arrival",
        "actual_output",
        "post_decision",
        "selector",
        "prediction",
    ]
    return [c for c in cols if any(marker in c.lower() for marker in forbidden)]


def feature_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith("state__")]
    cols += [c for c in df.columns if c.startswith("action__")]
    bad = leakage_columns(cols)
    if bad:
        raise ValueError({"feature_leakage_columns": bad})
    if len([c for c in cols if c.startswith("state__")]) != EXPECTED_STATE_FEATURES:
        raise ValueError("unexpected state feature count")
    if len([c for c in cols if c.startswith("action__")]) != EXPECTED_ACTION_FEATURES:
        raise ValueError("unexpected action feature count")
    return cols


def load_development(rows_path: Path, map_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    checks = check_inputs(rows_path, map_path)
    rows = pd.read_csv(rows_path)
    non_sbs = rows[rows["is_sbs_action"].astype(int) == 0].copy().reset_index(drop=True)
    non_sbs[PRIMARY_TARGET] = pd.to_numeric(non_sbs[PRIMARY_TARGET], errors="raise")
    cols = feature_columns(non_sbs)
    maps_head = pd.read_csv(map_path, nrows=5)
    integrity = {
        **checks,
        "all_rows": int(len(rows)),
        "non_sbs_rows": int(len(non_sbs)),
        "states": int(rows["state_id"].nunique()),
        "non_sbs_states": int(non_sbs["state_id"].nunique()),
        "scenarios": int(rows["scenario_id"].nunique()),
        "folds": int(rows["fold"].nunique()),
        "state_features": EXPECTED_STATE_FEATURES,
        "action_features": EXPECTED_ACTION_FEATURES,
        "policy_action_map_columns": list(maps_head.columns),
    }
    expected = {
        "non_sbs_rows": EXPECTED_NON_SBS_ROWS,
        "states": EXPECTED_STATES,
        "non_sbs_states": EXPECTED_STATES,
        "scenarios": EXPECTED_SCENARIOS,
        "folds": EXPECTED_FOLDS,
    }
    for key, value in expected.items():
        if integrity[key] != value:
            raise ValueError({"unexpected_development_shape": integrity, "expected": expected})
    return non_sbs, integrity


def state_equal_weights(df: pd.DataFrame) -> np.ndarray:
    counts = df.groupby("state_id")["canonical_action_id"].transform("count").astype(float)
    return (1.0 / counts).to_numpy(dtype=float)


def scenario_equal_weights(df: pd.DataFrame) -> np.ndarray:
    states_per_scenario = df.drop_duplicates(["scenario_id", "state_id"]).groupby("scenario_id")[
        "state_id"
    ].transform("count")
    state_counts = df.groupby("state_id")["canonical_action_id"].transform("count").astype(float)
    scenario_state_count = df["scenario_id"].map(
        df.drop_duplicates(["scenario_id", "state_id"]).groupby("scenario_id")["state_id"].count()
    ).astype(float)
    del states_per_scenario
    return (1.0 / (scenario_state_count * state_counts)).to_numpy(dtype=float)


def weights_for(df: pd.DataFrame, strategy: str) -> np.ndarray:
    if strategy == "STATE_EQUAL":
        return state_equal_weights(df)
    if strategy == "SCENARIO_EQUAL":
        return scenario_equal_weights(df)
    raise ValueError(strategy)


def model_grid(family: str, n_jobs: int) -> list[dict[str, Any]]:
    if family == "EXTRA_TREES":
        return [
            {
                "family": family,
                "n_estimators": 160,
                "max_depth": depth,
                "min_samples_leaf": leaf,
                "max_features": mf,
                "n_jobs": n_jobs,
            }
            for depth, leaf, mf in itertools.product([8, None], [10, 25], [0.5])
        ]
    if family == "HIST_GRADIENT_BOOSTING":
        return [
            {
                "family": family,
                "max_iter": 120,
                "learning_rate": lr,
                "max_leaf_nodes": leaves,
                "l2_regularization": 0.1,
            }
            for lr, leaves in itertools.product([0.05, 0.1], [15, 31])
        ]
    raise ValueError(family)


def build_model(config: Mapping[str, Any]):
    if config["family"] == "EXTRA_TREES":
        return ExtraTreesRegressor(
            n_estimators=int(config["n_estimators"]),
            max_depth=config["max_depth"],
            min_samples_leaf=int(config["min_samples_leaf"]),
            max_features=float(config["max_features"]),
            random_state=SEED,
            n_jobs=int(config.get("n_jobs", 1)),
        )
    if config["family"] == "HIST_GRADIENT_BOOSTING":
        return HistGradientBoostingRegressor(
            max_iter=int(config["max_iter"]),
            learning_rate=float(config["learning_rate"]),
            max_leaf_nodes=int(config["max_leaf_nodes"]),
            l2_regularization=float(config["l2_regularization"]),
            random_state=SEED,
        )
    raise ValueError(config["family"])


def fit_predict(
    config: Mapping[str, Any],
    train: pd.DataFrame,
    pred_df: pd.DataFrame,
    cols: Sequence[str],
    strategy: str,
):
    model = build_model(config)
    model.fit(
        train[list(cols)].to_numpy(dtype=float),
        train[PRIMARY_TARGET].to_numpy(dtype=float),
        sample_weight=weights_for(train, strategy),
    )
    return model, np.asarray(model.predict(pred_df[list(cols)].to_numpy(dtype=float)), dtype=float)


def state_decisions(
    df: pd.DataFrame,
    pred_col: str,
    tau: float,
    margin: float = 0.0,
    score_col: str = "gate_score",
) -> pd.DataFrame:
    tmp = df.copy()
    tmp[score_col] = tmp[pred_col].astype(float) - float(margin)
    rows: list[dict[str, Any]] = []
    for state_id, g in tmp.groupby("state_id", sort=False):
        idx = g[score_col].idxmax()
        best = g.loc[idx]
        override = bool(float(best[score_col]) > float(tau))
        realized = float(best[PRIMARY_TARGET]) if override else 0.0
        oracle = max(0.0, float(g[PRIMARY_TARGET].max()))
        rows.append(
            {
                "state_id": state_id,
                "scenario_id": best["scenario_id"],
                "fold": int(best["fold"]),
                "selected_canonical_action_id": best["canonical_action_id"] if override else "SBS_ABSTAIN",
                "override": int(override),
                "predicted_advantage": float(best[pred_col]),
                "gate_score": float(best[score_col]),
                "tau": float(tau),
                "margin": float(margin),
                "realized_gain": realized,
                "oracle_gain": oracle,
                "beneficial_override": int(override and realized > 0.0),
                "harmful_override": int(override and realized < 0.0),
                "zero_effect_override": int(override and realized == 0.0),
                "selected_true_advantage": realized,
            }
        )
    return pd.DataFrame(rows)


def selector_metrics(decisions: pd.DataFrame) -> dict[str, Any]:
    n = len(decisions)
    harmful = decisions[decisions["harmful_override"].astype(int) == 1]
    overrides = decisions[decisions["override"].astype(int) == 1]
    scenario = decisions.groupby("scenario_id")["realized_gain"].sum()
    fold = decisions.groupby("fold")["realized_gain"].mean()
    oracle_total = float(decisions["oracle_gain"].sum())
    top_scen = scenario.sort_values(ascending=False)
    positive_total = float(scenario[scenario > 0].sum())
    top5 = float(top_scen.head(5).sum() / positive_total) if positive_total > 0 else None
    return {
        "states": int(n),
        "mean_realized_gain": float(decisions["realized_gain"].mean()) if n else 0.0,
        "total_realized_gain": float(decisions["realized_gain"].sum()),
        "positive_folds": int((fold > 0).sum()),
        "fold_means": {str(int(k)): float(v) for k, v in fold.items()},
        "scenarios": int(len(scenario)),
        "positive_scenarios": int((scenario > 0).sum()),
        "zero_scenarios": int((scenario == 0).sum()),
        "negative_scenarios": int((scenario < 0).sum()),
        "p5_scenario_gain": float(scenario.quantile(0.05)) if len(scenario) else 0.0,
        "p10_scenario_gain": float(scenario.quantile(0.10)) if len(scenario) else 0.0,
        "worst_scenario_gain": float(scenario.min()) if len(scenario) else 0.0,
        "override_count": int(decisions["override"].sum()),
        "override_rate": float(decisions["override"].mean()) if n else 0.0,
        "beneficial_override_count": int(decisions["beneficial_override"].sum()),
        "harmful_override_count": int(decisions["harmful_override"].sum()),
        "harmful_fraction_of_overrides": float(decisions["harmful_override"].sum() / len(overrides))
        if len(overrides)
        else 0.0,
        "mean_harmful_loss": float(harmful["realized_gain"].mean()) if len(harmful) else 0.0,
        "worst_harmful_override": float(harmful["realized_gain"].min()) if len(harmful) else 0.0,
        "oracle_total_gain": oracle_total,
        "oracle_gap_closure": float(decisions["realized_gain"].sum() / oracle_total) if oracle_total > 0 else None,
        "positive_gain_top5_scenario_share": top5,
    }


def scenario_bootstrap(decisions: pd.DataFrame, seed: int, replicates: int) -> dict[str, Any]:
    scenarios = np.asarray(sorted(decisions["scenario_id"].unique()))
    by_scenario = {sid: g for sid, g in decisions.groupby("scenario_id")}
    rng = np.random.default_rng(seed)
    means = np.empty(int(replicates), dtype=float)
    for i in range(int(replicates)):
        sample = rng.choice(scenarios, size=len(scenarios), replace=True)
        boot = pd.concat([by_scenario[sid] for sid in sample], ignore_index=True)
        means[i] = float(boot["realized_gain"].mean())
    return {
        "replicates": int(replicates),
        "mean": float(means.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
    }


def candidate_scores(pred_df: pd.DataFrame, pred_col: str) -> list[dict[str, Any]]:
    residual = pred_df[pred_col].astype(float).to_numpy() - pred_df[PRIMARY_TARGET].astype(float).to_numpy()
    margins = {"MEAN_THRESHOLD": {"none": 0.0}}
    margins["OOF_RESIDUAL_LOWER_BOUND"] = {
        name: float(np.quantile(residual, q)) for name, q in RESIDUAL_QUANTILES.items()
    }
    scored = []
    for gate, values in margins.items():
        for margin_name, margin in values.items():
            for tau in TAU_GRID:
                dec = state_decisions(pred_df, pred_col, tau=tau, margin=margin)
                metrics = selector_metrics(dec)
                fold_means = list(metrics["fold_means"].values())
                scored.append(
                    {
                        "gate": gate,
                        "margin_name": margin_name,
                        "margin": float(margin),
                        "tau": float(tau),
                        "inner_min_fold_mean_gain": float(min(fold_means)) if fold_means else 0.0,
                        "inner_mean_gain": metrics["mean_realized_gain"],
                        "inner_negative_scenarios": metrics["negative_scenarios"],
                        "inner_harmful_overrides": metrics["harmful_override_count"],
                        "inner_override_rate": metrics["override_rate"],
                    }
                )
    return scored


def choose_scored(scored: list[dict[str, Any]]) -> dict[str, Any]:
    return sorted(
        scored,
        key=lambda r: (
            r["inner_min_fold_mean_gain"],
            r["inner_mean_gain"],
            -r["inner_negative_scenarios"],
            -r["inner_harmful_overrides"],
            -r["inner_override_rate"],
            stable_json(r),
        ),
        reverse=True,
    )[0]


@dataclass(frozen=True)
class InnerSelection:
    top_label: str
    family: str
    weight_strategy: str
    config: dict[str, Any]
    gate: dict[str, Any]
    inner_summary: dict[str, Any]


def crossfit_config(
    train_pool: pd.DataFrame,
    cols: Sequence[str],
    config: Mapping[str, Any],
    strategy: str,
) -> pd.DataFrame:
    parts = []
    for fold in sorted(train_pool["fold"].unique()):
        inner_train = train_pool[train_pool["fold"] != fold]
        inner_val = train_pool[train_pool["fold"] == fold].copy()
        _, p = fit_predict(config, inner_train, inner_val, cols, strategy)
        inner_val["predicted_advantage"] = p
        parts.append(inner_val)
    return pd.concat(parts, ignore_index=True)


def select_inner(train_pool: pd.DataFrame, cols: Sequence[str], n_jobs: int) -> tuple[InnerSelection, pd.DataFrame]:
    records = []
    cf_cache: dict[str, pd.DataFrame] = {}
    best_by_family_weight: dict[tuple[str, str], dict[str, Any]] = {}
    for family in ["EXTRA_TREES", "HIST_GRADIENT_BOOSTING"]:
        for strategy in ["STATE_EQUAL", "SCENARIO_EQUAL"]:
            for config in model_grid(family, n_jobs):
                cf = crossfit_config(train_pool, cols, config, strategy)
                scored = candidate_scores(cf, "predicted_advantage")
                best_gate = choose_scored(scored)
                record = {
                    "top_label": f"{family}.{strategy}.{stable_json(config)}",
                    "family": family,
                    "weight_strategy": strategy,
                    "config": config,
                    "gate": best_gate,
                    "all_gate_count": len(scored),
                }
                records.append(record)
                cf_cache[record["top_label"]] = cf
            fw = [r for r in records if r["family"] == family and r["weight_strategy"] == strategy]
            best_by_family_weight[(family, strategy)] = sorted(
                fw,
                key=lambda r: (
                    r["gate"]["inner_min_fold_mean_gain"],
                    r["gate"]["inner_mean_gain"],
                    -r["gate"]["inner_negative_scenarios"],
                    -r["gate"]["inner_harmful_overrides"],
                    -r["gate"]["inner_override_rate"],
                    r["top_label"],
                ),
                reverse=True,
            )[0]
    for strategy in ["STATE_EQUAL", "SCENARIO_EQUAL"]:
        et = best_by_family_weight[("EXTRA_TREES", strategy)]
        hgb = best_by_family_weight[("HIST_GRADIENT_BOOSTING", strategy)]
        cf = cf_cache[et["top_label"]].copy()
        hgb_cf = cf_cache[hgb["top_label"]][["state_id", "canonical_action_id", "predicted_advantage"]].rename(
            columns={"predicted_advantage": "hgb_pred"}
        )
        cf = cf.merge(hgb_cf, on=["state_id", "canonical_action_id"], validate="one_to_one")
        cf["predicted_advantage"] = 0.5 * cf["predicted_advantage"] + 0.5 * cf["hgb_pred"]
        scored = candidate_scores(cf, "predicted_advantage")
        records.append(
            {
                "top_label": f"ENSEMBLE_50_50.{strategy}",
                "family": "ENSEMBLE_50_50",
                "weight_strategy": strategy,
                "config": {"extra_trees": et["config"], "hist_gradient_boosting": hgb["config"]},
                "gate": choose_scored(scored),
                "all_gate_count": len(scored),
            }
        )
        cf_cache[f"ENSEMBLE_50_50.{strategy}"] = cf
    summary = pd.DataFrame(
        [
            {
                "top_label": r["top_label"],
                "family": r["family"],
                "weight_strategy": r["weight_strategy"],
                "config_json": stable_json(r["config"]),
                **{f"gate_{k}": v for k, v in r["gate"].items()},
            }
            for r in records
        ]
    )
    best = sorted(
        records,
        key=lambda r: (
            r["gate"]["inner_min_fold_mean_gain"],
            r["gate"]["inner_mean_gain"],
            -r["gate"]["inner_negative_scenarios"],
            -r["gate"]["inner_harmful_overrides"],
            -r["gate"]["inner_override_rate"],
            r["top_label"],
        ),
        reverse=True,
    )[0]
    return (
        InnerSelection(
            top_label=best["top_label"],
            family=best["family"],
            weight_strategy=best["weight_strategy"],
            config=dict(best["config"]),
            gate=dict(best["gate"]),
            inner_summary=best,
        ),
        summary,
    )


def fit_predict_selection(
    selection: InnerSelection,
    train_pool: pd.DataFrame,
    test: pd.DataFrame,
    cols: Sequence[str],
) -> np.ndarray:
    if selection.family == "ENSEMBLE_50_50":
        _, et = fit_predict(selection.config["extra_trees"], train_pool, test, cols, selection.weight_strategy)
        _, hgb = fit_predict(
            selection.config["hist_gradient_boosting"], train_pool, test, cols, selection.weight_strategy
        )
        return 0.5 * et + 0.5 * hgb
    _, pred = fit_predict(selection.config, train_pool, test, cols, selection.weight_strategy)
    return pred


def nested_oof(df: pd.DataFrame, cols: Sequence[str], n_jobs: int, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred_parts = []
    selection_rows = []
    candidate_tables = []
    for outer_fold in sorted(df["fold"].unique()):
        t0 = time.time()
        train_pool = df[df["fold"] != outer_fold].copy()
        test = df[df["fold"] == outer_fold].copy()
        selection, inner_candidates = select_inner(train_pool, cols, n_jobs)
        test["predicted_advantage"] = fit_predict_selection(selection, train_pool, test, cols)
        gate = selection.gate
        test["selected_gate"] = gate["gate"]
        test["selected_margin_name"] = gate["margin_name"]
        test["selected_margin"] = float(gate["margin"])
        test["selected_tau"] = float(gate["tau"])
        test["selected_family"] = selection.family
        test["selected_weight_strategy"] = selection.weight_strategy
        test["selected_config_json"] = stable_json(selection.config)
        pred_parts.append(test)
        inner_candidates["outer_fold"] = int(outer_fold)
        candidate_tables.append(inner_candidates)
        row = {
            "outer_fold": int(outer_fold),
            "selected_family": selection.family,
            "selected_weight_strategy": selection.weight_strategy,
            "selected_config_json": stable_json(selection.config),
            **{f"gate_{k}": v for k, v in gate.items()},
            "seconds": float(time.time() - t0),
        }
        selection_rows.append(row)
        write_json(out_dir / "outer_fold_progress" / f"outer_{int(outer_fold)}.json", row)
    return pd.concat(pred_parts, ignore_index=True), pd.concat(candidate_tables, ignore_index=True)


def decisions_from_nested_oof(pred: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for fold, g in pred.groupby("fold", sort=True):
        margin = float(g["selected_margin"].iloc[0])
        tau = float(g["selected_tau"].iloc[0])
        parts.append(state_decisions(g, "predicted_advantage", tau=tau, margin=margin))
    return pd.concat(parts, ignore_index=True)


def select_final_config(inner_candidate_table: pd.DataFrame, df: pd.DataFrame, cols: Sequence[str], n_jobs: int) -> dict[str, Any]:
    grouped = (
        inner_candidate_table.groupby(["family", "weight_strategy", "config_json"], dropna=False)
        .agg(
            min_inner_fold_mean=("gate_inner_min_fold_mean_gain", "mean"),
            mean_gain=("gate_inner_mean_gain", "mean"),
            negative_scenarios=("gate_inner_negative_scenarios", "mean"),
            harmful_overrides=("gate_inner_harmful_overrides", "mean"),
            override_rate=("gate_inner_override_rate", "mean"),
        )
        .reset_index()
    )
    best = grouped.sort_values(
        [
            "min_inner_fold_mean",
            "mean_gain",
            "negative_scenarios",
            "harmful_overrides",
            "override_rate",
            "family",
            "weight_strategy",
            "config_json",
        ],
        ascending=[False, False, True, True, True, False, False, False],
    ).iloc[0]
    family = str(best["family"])
    strategy = str(best["weight_strategy"])
    config = json.loads(str(best["config_json"]))
    cf = final_crossfit_predictions(df, cols, family, strategy, config, n_jobs)
    gate = choose_scored(candidate_scores(cf, "predicted_advantage"))
    return {
        "family": family,
        "weight_strategy": strategy,
        "config": config,
        "gate": gate,
        "selection_source": "mean_of_nested_inner_candidate_scores_refit_gate_on_full_development_oof",
    }


def final_crossfit_predictions(
    df: pd.DataFrame,
    cols: Sequence[str],
    family: str,
    strategy: str,
    config: Mapping[str, Any],
    n_jobs: int,
) -> pd.DataFrame:
    if family == "ENSEMBLE_50_50":
        parts = []
        for fold in sorted(df["fold"].unique()):
            train = df[df["fold"] != fold]
            val = df[df["fold"] == fold].copy()
            _, et = fit_predict(config["extra_trees"], train, val, cols, strategy)
            _, hgb = fit_predict(config["hist_gradient_boosting"], train, val, cols, strategy)
            val["predicted_advantage"] = 0.5 * et + 0.5 * hgb
            parts.append(val)
        return pd.concat(parts, ignore_index=True)
    local_config = dict(config)
    if local_config.get("family") == "EXTRA_TREES":
        local_config["n_jobs"] = n_jobs
    return crossfit_config(df, cols, local_config, strategy)


def fit_final_model(
    df: pd.DataFrame,
    cols: Sequence[str],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    family = config["family"]
    strategy = config["weight_strategy"]
    if family == "ENSEMBLE_50_50":
        et = build_model(config["config"]["extra_trees"])
        hgb = build_model(config["config"]["hist_gradient_boosting"])
        x = df[list(cols)].to_numpy(dtype=float)
        y = df[PRIMARY_TARGET].to_numpy(dtype=float)
        w = weights_for(df, strategy)
        et.fit(x, y, sample_weight=w)
        hgb.fit(x, y, sample_weight=w)
        return {"kind": "ensemble_50_50", "models": {"extra_trees": et, "hist_gradient_boosting": hgb}}
    model = build_model(config["config"])
    model.fit(df[list(cols)].to_numpy(dtype=float), df[PRIMARY_TARGET].to_numpy(dtype=float), sample_weight=weights_for(df, strategy))
    return {"kind": "single", "models": {family.lower(): model}}


def write_freeze_docs(out_dir: Path, selector: Mapping[str, Any], integrity: Mapping[str, Any], metrics: Mapping[str, Any]) -> None:
    protocol = {
        "schema_version": SCHEMA_VERSION,
        "status": "CONFIRMATORY_EVALUATION_READY",
        "confirmatory_label_access_status": "NOT_ACCESSED_BY_THIS_TASK",
        "final_selector": selector,
        "development_integrity": integrity,
        "development_oof_metrics": metrics,
        "confirmatory_protocol": {
            "steps": [
                "enumerate unique non-SBS candidate actions per fresh disagreement state",
                "compute STATE_ACTION_V1 in frozen feature order",
                "predict with FINAL_CONFIRMATORY_SELECTOR_V1",
                "apply frozen conservative gate",
                "choose one candidate action or abstain to SBS",
                "only then join fresh A_SBS labels",
                "calculate realized gain",
            ],
            "primary_metric": "mean realized A_SBS per fresh disagreement state",
            "primary_uncertainty": "scenario-level bootstrap over the 78 supported fresh scenarios",
            "baseline": "always SBS = 0",
            "secondary_metrics": [
                "override rate",
                "precision",
                "harmful override rate",
                "negative scenarios",
                "oracle gap closure",
            ],
        },
        "verdict_rule": {
            "FRESH_CONFIRMATION_SUCCESS": [
                "mean realized gain > 0",
                "scenario-bootstrap 95% CI lower bound > 0",
                "benefit not driven by a tiny number of scenarios",
                "downside metrics reported",
            ],
            "FRESH_CONFIRMATION_POSITIVE_BUT_UNCERTAIN": [
                "point estimate > 0",
                "CI includes zero",
            ],
            "FRESH_CONFIRMATION_FAIL": [
                "mean realized gain <= 0 or meaningful generalization is absent",
            ],
        },
    }
    write_json(out_dir / "confirmatory_protocol_and_verdict_freeze.json", protocol)
    md = [
        "# FINAL_CONFIRMATORY_SELECTOR_V1 Freeze",
        "",
        "Status: CONFIRMATORY_EVALUATION_READY",
        "",
        "Fresh confirmatory terminal outcomes were not accessed by this task.",
        "",
        f"Family: `{selector['family']}`",
        f"Weight strategy: `{selector['weight_strategy']}`",
        f"Gate: `{selector['gate']['gate']}` / `{selector['gate']['margin_name']}`",
        f"Margin: `{selector['gate']['margin']}`",
        f"Tau: `{selector['gate']['tau']}`",
        "",
        "The next task may run the one-shot fresh confirmatory evaluation exactly as frozen in `confirmatory_protocol_and_verdict_freeze.json`.",
        "",
    ]
    (out_dir / "FINAL_CONFIRMATORY_SELECTOR_V1_DESIGN.md").write_text("\n".join(md))


def package_versions() -> dict[str, str]:
    import sklearn

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, integrity = load_development(Path(args.rows), Path(args.policy_action_map))
    cols = feature_columns(rows)
    prereg = {
        "schema_version": SCHEMA_VERSION,
        "created_at_unix": time.time(),
        "purpose": "development-only conservative selector search",
        "fresh_path_guard": FRESH_FORBIDDEN_SUBSTRING,
        "feature_set": "STATE_ACTION_V1",
        "feature_count": len(cols),
        "model_families": ["EXTRA_TREES", "HIST_GRADIENT_BOOSTING", "ENSEMBLE_50_50"],
        "weight_strategies": ["STATE_EQUAL", "SCENARIO_EQUAL"],
        "tau_grid": TAU_GRID,
        "residual_quantiles": RESIDUAL_QUANTILES,
        "nested_protocol": "five existing scenario folds; inner grouped by fold within training pool",
        "selection_objective": "maximize minimum inner-fold mean realized gain with safety tie-breakers",
        "git_sha": git(["rev-parse", "HEAD"]),
        "git_branch": git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "package_versions": package_versions(),
        "development_input_integrity": integrity,
    }
    write_json(out_dir / "PREREGISTERED_SEARCH_DESIGN.json", prereg)
    t0 = time.time()
    pred, inner_candidates = nested_oof(rows, cols, int(args.n_jobs), out_dir)
    pred.to_csv(out_dir / "development_outer_oof_action_predictions.csv", index=False)
    inner_candidates.to_csv(out_dir / "development_inner_candidate_results.csv", index=False)
    decisions = decisions_from_nested_oof(pred)
    decisions.to_csv(out_dir / "development_outer_oof_state_decisions.csv", index=False)
    metrics = selector_metrics(decisions)
    metrics["scenario_bootstrap_mean_realized_gain"] = scenario_bootstrap(
        decisions, seed=SEED, replicates=int(args.bootstrap_replicates)
    )
    write_json(out_dir / "development_oof_result.json", metrics)
    final_config = select_final_config(inner_candidates, rows, cols, int(args.n_jobs))
    final_cf = final_crossfit_predictions(
        rows,
        cols,
        final_config["family"],
        final_config["weight_strategy"],
        final_config["config"],
        int(args.n_jobs),
    )
    final_gate = choose_scored(candidate_scores(final_cf, "predicted_advantage"))
    final_config["gate"] = final_gate
    final_decisions = state_decisions(
        final_cf,
        "predicted_advantage",
        tau=float(final_gate["tau"]),
        margin=float(final_gate["margin"]),
    )
    final_oof = selector_metrics(final_decisions)
    final_oof["scenario_bootstrap_mean_realized_gain"] = scenario_bootstrap(
        final_decisions, seed=SEED + 1, replicates=int(args.bootstrap_replicates)
    )
    model_bundle = fit_final_model(rows, cols, final_config)
    selector = {
        "schema_version": SCHEMA_VERSION,
        "name": "FINAL_CONFIRMATORY_SELECTOR_V1",
        "family": final_config["family"],
        "weight_strategy": final_config["weight_strategy"],
        "config": final_config["config"],
        "gate": final_config["gate"],
        "feature_set": "STATE_ACTION_V1",
        "feature_columns": cols,
        "candidate_action_semantics": "unique non-SBS canonical actions; abstain falls back to SBS",
        "random_seed": SEED,
        "package_versions": package_versions(),
        "development_input_checksums": {
            "state_action_rows_full_sha256": integrity["rows_sha256"],
            "state_policy_action_map_full_sha256": integrity["map_sha256"],
        },
        "analysis_git_sha": git(["rev-parse", "HEAD"]),
        "analysis_git_branch": git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "selected_by": final_config["selection_source"],
    }
    joblib.dump({"selector": selector, **model_bundle}, out_dir / "FINAL_CONFIRMATORY_SELECTOR_V1.joblib")
    write_json(out_dir / "FINAL_CONFIRMATORY_SELECTOR_V1.json", selector)
    write_json(out_dir / "final_development_oof_result.json", final_oof)
    reloaded = joblib.load(out_dir / "FINAL_CONFIRMATORY_SELECTOR_V1.joblib")
    sample = rows.iloc[:32]
    if reloaded["selector"]["feature_columns"] != cols:
        raise RuntimeError("feature order reload mismatch")
    x_sample = sample[cols].to_numpy(dtype=float)
    if reloaded["kind"] == "single":
        model = next(iter(reloaded["models"].values()))
        check_pred = np.asarray(model.predict(x_sample), dtype=float)
    else:
        check_pred = 0.5 * np.asarray(reloaded["models"]["extra_trees"].predict(x_sample), dtype=float)
        check_pred += 0.5 * np.asarray(reloaded["models"]["hist_gradient_boosting"].predict(x_sample), dtype=float)
    if check_pred.shape != (len(sample),):
        raise RuntimeError("reload prediction shape mismatch")
    write_json(out_dir / "reload_prediction_test.json", {"passed": True, "rows": int(len(sample))})
    write_freeze_docs(out_dir, selector, integrity, final_oof)
    write_json(
        out_dir / "RUN_SUMMARY.json",
        {
            "status": "CONFIRMATORY_EVALUATION_READY",
            "elapsed_seconds": float(time.time() - t0),
            "nested_development_oof_metrics": metrics,
            "final_refit_selector_oof_metrics": final_oof,
            "v1_reference": V1_REFERENCE,
            "output_paths": {
                "selector_json": str(out_dir / "FINAL_CONFIRMATORY_SELECTOR_V1.json"),
                "selector_joblib": str(out_dir / "FINAL_CONFIRMATORY_SELECTOR_V1.joblib"),
                "protocol": str(out_dir / "confirmatory_protocol_and_verdict_freeze.json"),
            },
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", default=str(DEFAULT_ROWS))
    parser.add_argument("--policy-action-map", default=str(DEFAULT_MAP))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
