#!/usr/bin/env python3
"""V2 development-only conservative SBS override selector freeze.

V2 resolves the V1 final-selector preregistration ambiguity by separating:

1. nested outer OOF evaluation, which remains evaluation-only;
2. full-development grouped-CV final candidate selection;
3. OOF-only residual-margin calibration;
4. all-development final model fitting after the candidate/gate freeze.

Fresh confirmatory terminal outcomes are never accepted as inputs.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd

import sbs_override_conservative_selector_dev_v1 as v1


SCHEMA_VERSION = "sbs_override_conservative_selector_dev_v2.0.0"
SELECTOR_NAME = "FINAL_CONFIRMATORY_SELECTOR_V2"
DEFAULT_OUT = (
    v1.ROOT / "experiments" / "sbs_override_conservative_selector_dev_v2" / "run_v2"
)
OUTER_OOF_FORBIDDEN_MARKERS = (
    "development_outer_oof_action_predictions",
    "development_outer_oof_state_decisions",
    "development_oof_result",
    "outer_fold_progress",
)


@dataclass(frozen=True)
class ModelCandidate:
    family: str
    weight_strategy: str
    config: dict[str, Any]

    @property
    def stable_key(self) -> str:
        return v1.stable_json(
            {
                "family": self.family,
                "weight_strategy": self.weight_strategy,
                "config": self.config,
            }
        )


def reject_forbidden_inputs(paths: Sequence[Path | str]) -> None:
    """Reject fresh confirmatory inputs and V1 outer-OOF final-selection inputs."""
    for path in paths:
        v1.reject_fresh_confirmatory_path(path)
        lowered = str(path).lower()
        if any(marker in lowered for marker in OUTER_OOF_FORBIDDEN_MARKERS):
            raise ValueError({"outer_oof_result_forbidden_for_v2_final_selection": str(path)})


def assert_feature_order(cols: Sequence[str], expected_cols: Sequence[str]) -> None:
    if list(cols) != list(expected_cols):
        raise ValueError(
            {
                "feature_order_mismatch": {
                    "observed_count": len(cols),
                    "expected_count": len(expected_cols),
                    "first_observed": list(cols[:5]),
                    "first_expected": list(expected_cols[:5]),
                }
            }
        )


def assert_group_split_integrity(train: pd.DataFrame, val: pd.DataFrame) -> None:
    train_scenarios = set(train["scenario_id"].unique())
    val_scenarios = set(val["scenario_id"].unique())
    overlap = train_scenarios & val_scenarios
    if overlap:
        raise ValueError({"scenario_group_leakage": sorted(overlap)[:10]})
    if set(train["state_id"].unique()) & set(val["state_id"].unique()):
        raise ValueError("state_id appears in both train and validation split")


def model_candidates(n_jobs: int) -> list[ModelCandidate]:
    """Finite V2 candidate universe from V1 model families and hyperparameter grids."""
    candidates: list[ModelCandidate] = []
    strategies = ["STATE_EQUAL", "SCENARIO_EQUAL"]
    et_grid = v1.model_grid("EXTRA_TREES", n_jobs)
    hgb_grid = v1.model_grid("HIST_GRADIENT_BOOSTING", n_jobs)
    for strategy in strategies:
        for config in et_grid:
            candidates.append(ModelCandidate("EXTRA_TREES", strategy, dict(config)))
        for config in hgb_grid:
            candidates.append(ModelCandidate("HIST_GRADIENT_BOOSTING", strategy, dict(config)))
        for et in et_grid:
            for hgb in hgb_grid:
                candidates.append(
                    ModelCandidate(
                        "ENSEMBLE_50_50",
                        strategy,
                        {"extra_trees": dict(et), "hist_gradient_boosting": dict(hgb)},
                    )
                )
    return candidates


def grouped_crossfit_predictions_for_model(
    df: pd.DataFrame,
    cols: Sequence[str],
    candidate: ModelCandidate,
) -> pd.DataFrame:
    """Generate scenario-grouped OOF predictions for exactly one model candidate."""
    parts: list[pd.DataFrame] = []
    seen_states: set[Any] = set()
    for fold in sorted(df["fold"].unique()):
        train = df[df["fold"] != fold].copy()
        val = df[df["fold"] == fold].copy()
        assert_group_split_integrity(train, val)
        if candidate.family == "ENSEMBLE_50_50":
            _, et = v1.fit_predict(
                candidate.config["extra_trees"], train, val, cols, candidate.weight_strategy
            )
            _, hgb = v1.fit_predict(
                candidate.config["hist_gradient_boosting"],
                train,
                val,
                cols,
                candidate.weight_strategy,
            )
            pred = 0.5 * et + 0.5 * hgb
        else:
            _, pred = v1.fit_predict(candidate.config, train, val, cols, candidate.weight_strategy)
        val["predicted_advantage"] = np.asarray(pred, dtype=float)
        duplicated = seen_states & set(val["state_id"])
        if duplicated:
            raise ValueError({"state_predicted_more_than_once": sorted(duplicated)[:10]})
        seen_states.update(val["state_id"])
        parts.append(val)
    out = pd.concat(parts, ignore_index=True)
    if len(out) != len(df):
        raise ValueError({"crossfit_row_count_mismatch": {"observed": len(out), "expected": len(df)}})
    if out["state_id"].nunique() != df["state_id"].nunique():
        raise ValueError("crossfit state coverage mismatch")
    merged = df[["state_id", "canonical_action_id"]].merge(
        out[["state_id", "canonical_action_id"]],
        on=["state_id", "canonical_action_id"],
        how="left",
        indicator=True,
        validate="one_to_one",
    )
    if not (merged["_merge"] == "both").all():
        raise ValueError("each state-action row must receive exactly one OOF prediction")
    return out


def residual_margins_from_oof(pred_df: pd.DataFrame, pred_col: str) -> dict[str, float]:
    residual = pred_df[pred_col].astype(float).to_numpy() - pred_df[v1.PRIMARY_TARGET].astype(
        float
    ).to_numpy()
    return {name: float(np.quantile(residual, q)) for name, q in v1.RESIDUAL_QUANTILES.items()}


def score_candidate_predictions(
    pred_df: pd.DataFrame,
    pred_col: str,
    candidate: ModelCandidate,
) -> list[dict[str, Any]]:
    margins = {"MEAN_THRESHOLD": {"none": 0.0}}
    margins["OOF_RESIDUAL_LOWER_BOUND"] = residual_margins_from_oof(pred_df, pred_col)
    scored: list[dict[str, Any]] = []
    for gate, values in margins.items():
        for margin_name, margin in values.items():
            for tau in v1.TAU_GRID:
                decisions = v1.state_decisions(pred_df, pred_col, tau=tau, margin=margin)
                metrics = v1.selector_metrics(decisions)
                fold_means = list(metrics["fold_means"].values())
                scored.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "family": candidate.family,
                        "weight_strategy": candidate.weight_strategy,
                        "config": candidate.config,
                        "config_json": v1.stable_json(candidate.config),
                        "gate": gate,
                        "margin_name": margin_name,
                        "margin": float(margin),
                        "tau": float(tau),
                        "min_fold_mean_gain": float(min(fold_means)) if fold_means else 0.0,
                        "mean_gain": metrics["mean_realized_gain"],
                        "negative_scenarios": metrics["negative_scenarios"],
                        "harmful_overrides": metrics["harmful_override_count"],
                        "override_rate": metrics["override_rate"],
                        "stable_key": v1.stable_json(
                            {
                                "family": candidate.family,
                                "weight_strategy": candidate.weight_strategy,
                                "config": candidate.config,
                                "gate": gate,
                                "margin_name": margin_name,
                                "tau": float(tau),
                            }
                        ),
                    }
                )
    return scored


def select_final_candidate(scored: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """V2 lexicographic objective; stable key is ascending and deterministic."""
    if not scored:
        raise ValueError("no scored V2 candidates")
    return dict(
        sorted(
            scored,
            key=lambda r: (
                -float(r["min_fold_mean_gain"]),
                -float(r["mean_gain"]),
                int(r["negative_scenarios"]),
                int(r["harmful_overrides"]),
                float(r["override_rate"]),
                str(r["stable_key"]),
            ),
        )[0]
    )


def run_final_selection_crossfit(
    df: pd.DataFrame,
    cols: Sequence[str],
    n_jobs: int,
    out_dir: Path,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Development-only V2 final-selection stage over all bounded candidates."""
    reject_forbidden_inputs([out_dir])
    rows: list[dict[str, Any]] = []
    selected_predictions: pd.DataFrame | None = None
    selected: dict[str, Any] | None = None
    for candidate in model_candidates(n_jobs):
        pred = grouped_crossfit_predictions_for_model(df, cols, candidate)
        scored = score_candidate_predictions(pred, "predicted_advantage", candidate)
        rows.extend(scored)
        local_best = select_final_candidate(scored)
        if selected is None or select_final_candidate([selected, local_best]) == local_best:
            selected = local_best
            selected_predictions = pred
    if selected is None or selected_predictions is None:
        raise RuntimeError("V2 selection produced no candidate")
    all_scores = pd.DataFrame(rows)
    final_selected = select_final_candidate(rows)
    if final_selected["stable_key"] != selected["stable_key"]:
        selected_candidate = ModelCandidate(
            str(final_selected["family"]),
            str(final_selected["weight_strategy"]),
            dict(final_selected["config"]),
        )
        selected_predictions = grouped_crossfit_predictions_for_model(df, cols, selected_candidate)
    return final_selected, all_scores


def freeze_final_model(
    df: pd.DataFrame,
    cols: Sequence[str],
    selected: Mapping[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    """Fit all-development model only after the V2 candidate/gate has been selected."""
    frozen = {
        "family": selected["family"],
        "weight_strategy": selected["weight_strategy"],
        "config": selected["config"],
        "gate": {
            "gate": selected["gate"],
            "margin_name": selected["margin_name"],
            "margin": float(selected["margin"]),
            "tau": float(selected["tau"]),
        },
    }
    bundle = v1.fit_final_model(df, cols, frozen)
    selector = {
        "schema_version": SCHEMA_VERSION,
        "name": SELECTOR_NAME,
        "family": frozen["family"],
        "weight_strategy": frozen["weight_strategy"],
        "config": frozen["config"],
        "gate": frozen["gate"],
        "feature_set": "STATE_ACTION_V1",
        "feature_columns": list(cols),
        "candidate_action_semantics": "unique non-SBS canonical actions; abstain falls back to SBS",
        "random_seed": v1.SEED,
        "package_versions": v1.package_versions(),
        "analysis_git_sha": v1.git(["rev-parse", "HEAD"]),
        "analysis_git_branch": v1.git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "selected_by": "v2_full_development_grouped_crossfit_all_candidates_oof_gate_calibration",
        "outer_oof_results_used_for_final_selection": False,
        "fresh_confirmatory_outcomes_used": False,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    v1.write_json(out_dir / "FINAL_CONFIRMATORY_SELECTOR_V2.json", selector)
    joblib.dump({"selector": selector, **bundle}, out_dir / "FINAL_CONFIRMATORY_SELECTOR_V2.joblib")
    return selector


def reload_validation(selector_path: Path, joblib_path: Path, df: pd.DataFrame) -> dict[str, Any]:
    selector = json.loads(selector_path.read_text())
    reloaded = joblib.load(joblib_path)
    assert_feature_order(reloaded["selector"]["feature_columns"], selector["feature_columns"])
    sample = df.iloc[: min(32, len(df))]
    x_sample = sample[selector["feature_columns"]].to_numpy(dtype=float)
    if reloaded["kind"] == "single":
        model = next(iter(reloaded["models"].values()))
        pred = np.asarray(model.predict(x_sample), dtype=float)
    else:
        pred = 0.5 * np.asarray(reloaded["models"]["extra_trees"].predict(x_sample), dtype=float)
        pred += 0.5 * np.asarray(
            reloaded["models"]["hist_gradient_boosting"].predict(x_sample), dtype=float
        )
    if pred.shape != (len(sample),):
        raise RuntimeError("V2 reload prediction shape mismatch")
    return {"passed": True, "rows": int(len(sample))}


def write_preregistered_design(out_dir: Path, integrity: Mapping[str, Any], cols: Sequence[str]) -> None:
    prereg = {
        "schema_version": SCHEMA_VERSION,
        "created_at_unix": time.time(),
        "purpose": "V2 development-only final selector selection after V1 preregistration ambiguity",
        "v1_status": "HISTORICAL_DEVELOPMENT_SELECTOR_WITH_AMBIGUOUS_FINAL_SELECTION_PREREGISTRATION",
        "confirmatory_label_access": "NOT_ACCESSED",
        "fresh_confirmatory_outcomes_used": False,
        "outer_oof_results_used_for_final_selection": False,
        "feature_set": "STATE_ACTION_V1",
        "feature_count": len(cols),
        "model_families": ["EXTRA_TREES", "HIST_GRADIENT_BOOSTING", "ENSEMBLE_50_50"],
        "weight_strategies": ["STATE_EQUAL", "SCENARIO_EQUAL"],
        "tau_grid": v1.TAU_GRID,
        "residual_quantiles": v1.RESIDUAL_QUANTILES,
        "grouped_cv_folds": "existing five scenario folds from joint240 development corpus",
        "final_selection_objective": [
            "maximize minimum fold mean realized gain",
            "maximize overall mean realized gain",
            "minimize number of negative aggregate-gain scenarios",
            "minimize number of harmful overrides",
            "minimize override rate",
            "deterministic stable-key tie-break",
        ],
        "gate_calibration": "OOF residual margins only from full-development grouped-crossfit predictions",
        "final_fit": "after candidate/gate freeze, fit selected estimator(s) on all development rows",
        "git_sha": v1.git(["rev-parse", "HEAD"]),
        "git_branch": v1.git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "package_versions": v1.package_versions(),
        "development_input_integrity": dict(integrity),
    }
    v1.write_json(out_dir / "PREREGISTERED_SEARCH_DESIGN_V2.json", prereg)


def run(args: argparse.Namespace) -> None:
    if not args.execute_full_selection:
        raise SystemExit(
            "V2 design/source freeze prepared. Refusing full development selection without "
            "--execute-full-selection."
        )
    out_dir = Path(args.out_dir)
    reject_forbidden_inputs([args.rows, args.policy_action_map, out_dir])
    rows, integrity = v1.load_development(Path(args.rows), Path(args.policy_action_map))
    cols = v1.feature_columns(rows)
    write_preregistered_design(out_dir, integrity, cols)
    selected, scores = run_final_selection_crossfit(rows, cols, int(args.n_jobs), out_dir)
    scores.to_csv(out_dir / "development_full_cv_candidate_results_v2.csv", index=False)
    selector = freeze_final_model(rows, cols, selected, out_dir)
    reload_result = reload_validation(
        out_dir / "FINAL_CONFIRMATORY_SELECTOR_V2.json",
        out_dir / "FINAL_CONFIRMATORY_SELECTOR_V2.joblib",
        rows,
    )
    v1.write_json(out_dir / "reload_prediction_test_v2.json", reload_result)
    v1.write_json(
        out_dir / "RUN_SUMMARY_V2.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": "V2_DEVELOPMENT_SELECTOR_FROZEN",
            "selector": selector,
            "selected_candidate": selected,
            "fresh_confirmatory_outcomes_used": False,
            "outer_oof_results_used_for_final_selection": False,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", default=str(v1.DEFAULT_ROWS))
    parser.add_argument("--policy-action-map", default=str(v1.DEFAULT_MAP))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--execute-full-selection", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
