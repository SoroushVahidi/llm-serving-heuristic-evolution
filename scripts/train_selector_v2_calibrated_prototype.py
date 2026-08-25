#!/usr/bin/env python3
"""Selector v2 prototype training/evaluation -- calibrated targeted pilot.

HISTORICAL ENTRY POINT. Superseded by
scripts/evaluate_selector_v2_clean_pilot.py, which covers everything this
script does (RF regressor + classifier, gated on quality_gates.json) plus:
an independent leakage_audit.json gate (this script only checks
quality_gates.json, whose own no_leakage check is known incomplete -- see
docs/current/PROJECT_STATUS.md), Extra Trees and a decision-tree baseline,
bootstrap confidence intervals, and a meaningful-margin subset breakdown.
Retained only for reproducibility of any prior result computed with this
exact script; use scripts/evaluate_selector_v2_clean_pilot.py for all new
Selector v2 calibrated-pilot evaluation.

Runs ONLY if the pilot's quality_gates.json says `all_gates_passed: true`
(per task instructions: "If any major gate fails: do not train. Write the
exact failure reason."). Fixed, small hyperparameters throughout -- no
sweep.

Preferred approach (per-policy utility regression): one RandomForestRegressor
per of the 8 CANDIDATE_POLICIES, predicting that policy's ANWG from the
window's causal (leakage-free) features; the prototype selector picks
argmax predicted utility. A direct multiclass classifier (predict the best
policy's label) is trained as a secondary baseline for comparison.

Reports, separately for VALIDATION / ID_TEST / OOD_TEST: prototype selector
vs. the single global best-fixed policy (by TRAIN-split mean ANWG) vs. each
of the 8 individual policies vs. the per-window oracle. Primary metric ANWG;
secondary regret-to-oracle, completion, rejection, SLO attainment, TTFT,
TPOT/TBT, E2E latency. Never compares against the 3 faithful external
baselines here -- that is a separate, later evaluation step (Protocol C).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from llmserveopt.selector.dataset_v2.calibrated_targeted_pilot import CANDIDATE_POLICIES  # noqa: E402
from llmserveopt.selector.dataset_v2.splits import ALL_SPLITS, TRAIN  # noqa: E402

RF_PARAMS = dict(n_estimators=150, max_depth=8, random_state=42, n_jobs=-1)

SECONDARY_METRIC_COLUMNS = [
    "metric_completion_fraction", "metric_rejection_fraction", "metric_slo_attainment",
    "metric_mean_ttft", "metric_mean_tpot", "metric_mean_tbt", "metric_mean_latency",
]


def _load_pilot(pilot_dir: Path):
    windows = pd.read_csv(pilot_dir / "retained_windows.csv")
    features = pd.read_csv(pilot_dir / "window_features.csv")
    vectors = pd.read_csv(pilot_dir / "full_policy_vectors.csv")
    return windows, features, vectors


def _feature_columns(features: pd.DataFrame) -> List[str]:
    cols = [c for c in features.columns if c.startswith("feat_") and c != "feat_window_idx"]
    return sorted(cols)


def _build_design_matrix(windows: pd.DataFrame, features: pd.DataFrame, vectors: pd.DataFrame):
    merged = windows.merge(features, on="window_idx", how="left")
    feat_cols = _feature_columns(features)
    X = merged[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    anwg_wide = vectors.pivot_table(
        index="window_idx", columns="policy_name",
        values="metric_arrival_normalized_weighted_goodput", aggfunc="first",
    ).reindex(columns=list(CANDIDATE_POLICIES))
    anwg_wide = anwg_wide.fillna(0.0)

    return merged, X, feat_cols, anwg_wide


def _train_regressors(X_train: pd.DataFrame, anwg_train: pd.DataFrame):
    from sklearn.ensemble import RandomForestRegressor
    models = {}
    for pname in CANDIDATE_POLICIES:
        y = anwg_train[pname].values
        model = RandomForestRegressor(**RF_PARAMS)
        model.fit(X_train.values, y)
        models[pname] = model
    return models


def _train_classifier(X_train: pd.DataFrame, best_policy_train: pd.Series):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_train.values, best_policy_train.values)
    return model


def _predict_regressor_selection(models: Dict, X: pd.DataFrame) -> List[str]:
    preds = {pname: models[pname].predict(X.values) for pname in CANDIDATE_POLICIES}
    pred_df = pd.DataFrame(preds)
    return pred_df.idxmax(axis=1).tolist()


def _metric_lookup(vectors: pd.DataFrame) -> Dict:
    """(window_idx, policy_name) -> row dict of metric_* columns."""
    lut = {}
    for row in vectors.to_dict("records"):
        lut[(row["window_idx"], row["policy_name"])] = row
    return lut


def _evaluate_selection(
    window_idxs: List[int], selection: List[str], anwg_wide: pd.DataFrame,
    metric_lut: Dict, label: str,
) -> Dict:
    anwg_vals, secondary_sums = [], {c: [] for c in SECONDARY_METRIC_COLUMNS}
    for widx, pname in zip(window_idxs, selection):
        anwg_vals.append(anwg_wide.loc[widx, pname])
        row = metric_lut.get((widx, pname), {})
        for c in SECONDARY_METRIC_COLUMNS:
            v = row.get(c)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                secondary_sums[c].append(v)
    result = {
        "label": label, "n_windows": len(window_idxs),
        "mean_anwg": round(float(np.mean(anwg_vals)), 4) if anwg_vals else None,
    }
    for c in SECONDARY_METRIC_COLUMNS:
        vals = secondary_sums[c]
        result[c.replace("metric_", "mean_")] = round(float(np.mean(vals)), 4) if vals else None
    return result


def _oracle_selection(window_idxs: List[int], anwg_wide: pd.DataFrame) -> List[str]:
    return [anwg_wide.loc[w].idxmax() for w in window_idxs]


def _fixed_policy_selection(window_idxs: List[int], policy: str) -> List[str]:
    return [policy] * len(window_idxs)


def run_split_report(
    split_name: str, window_idxs: List[int], anwg_wide: pd.DataFrame, metric_lut: Dict,
    regressor_selection: List[str], classifier_selection: List[str], best_fixed_policy: str,
) -> Dict:
    if not window_idxs:
        return {"split": split_name, "n_windows": 0, "note": "no windows in this split"}

    oracle_sel = _oracle_selection(window_idxs, anwg_wide)
    best_fixed_sel = _fixed_policy_selection(window_idxs, best_fixed_policy)

    entries = {
        "prototype_regressor_argmax": _evaluate_selection(
            window_idxs, regressor_selection, anwg_wide, metric_lut, "prototype_regressor_argmax"),
        "prototype_classifier": _evaluate_selection(
            window_idxs, classifier_selection, anwg_wide, metric_lut, "prototype_classifier"),
        "global_best_fixed": _evaluate_selection(
            window_idxs, best_fixed_sel, anwg_wide, metric_lut, f"global_best_fixed({best_fixed_policy})"),
        "oracle": _evaluate_selection(window_idxs, oracle_sel, anwg_wide, metric_lut, "oracle"),
    }
    for pname in CANDIDATE_POLICIES:
        sel = _fixed_policy_selection(window_idxs, pname)
        entries[f"fixed__{pname}"] = _evaluate_selection(window_idxs, sel, anwg_wide, metric_lut, pname)

    oracle_mean = entries["oracle"]["mean_anwg"]
    best_fixed_mean = entries["global_best_fixed"]["mean_anwg"]
    reg_mean = entries["prototype_regressor_argmax"]["mean_anwg"]
    clf_mean = entries["prototype_classifier"]["mean_anwg"]

    def _headroom_captured(selector_mean):
        if oracle_mean is None or best_fixed_mean is None or selector_mean is None:
            return None
        headroom = oracle_mean - best_fixed_mean
        if abs(headroom) < 1e-9:
            return None
        return round((selector_mean - best_fixed_mean) / headroom, 4)

    return {
        "split": split_name, "n_windows": len(window_idxs),
        "entries": entries,
        "regressor_improvement_over_best_fixed": (
            round(reg_mean - best_fixed_mean, 4) if reg_mean is not None and best_fixed_mean is not None else None
        ),
        "classifier_improvement_over_best_fixed": (
            round(clf_mean - best_fixed_mean, 4) if clf_mean is not None and best_fixed_mean is not None else None
        ),
        "regressor_regret_to_oracle": (
            round(oracle_mean - reg_mean, 4) if oracle_mean is not None and reg_mean is not None else None
        ),
        "classifier_regret_to_oracle": (
            round(oracle_mean - clf_mean, 4) if oracle_mean is not None and clf_mean is not None else None
        ),
        "regressor_oracle_headroom_captured_fraction": _headroom_captured(reg_mean),
        "classifier_oracle_headroom_captured_fraction": _headroom_captured(clf_mean),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-dir", required=True)
    args = parser.parse_args()
    pilot_dir = ROOT / args.pilot_dir

    gates_path = pilot_dir / "quality_gates.json"
    if not gates_path.exists():
        print(json.dumps({"trained": False, "reason": "quality_gates.json not found -- generation incomplete."}))
        return 1
    gates = json.loads(gates_path.read_text())
    if not gates.get("all_gates_passed"):
        failed = {k: v for k, v in gates.get("gates", {}).items() if not v.get("passed")}
        result = {
            "trained": False,
            "reason": "One or more required quality gates failed -- per task instructions, the selector "
                      "prototype must not be trained. See failed_gates for the exact reasons.",
            "failed_gates": failed,
        }
        (pilot_dir / "selector_metrics.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return 0

    windows, features, vectors = _load_pilot(pilot_dir)
    merged, X, feat_cols, anwg_wide = _build_design_matrix(windows, features, vectors)
    metric_lut = _metric_lookup(vectors)

    train_mask = merged["split"] == TRAIN
    X_train = X[train_mask]
    anwg_train = anwg_wide.loc[merged.loc[train_mask, "window_idx"]]
    best_policy_train = merged.loc[train_mask, "window_idx"].map(lambda w: anwg_wide.loc[w].idxmax())

    if X_train.empty:
        result = {"trained": False, "reason": "TRAIN split is empty -- cannot fit any prototype model."}
        (pilot_dir / "selector_metrics.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return 0

    regressors = _train_regressors(X_train, anwg_train)
    classifier = _train_classifier(X_train, best_policy_train)

    best_fixed_policy = anwg_train.mean(axis=0).idxmax()

    reports = {}
    for split_name in ALL_SPLITS:
        split_mask = merged["split"] == split_name
        window_idxs = merged.loc[split_mask, "window_idx"].tolist()
        if not window_idxs:
            reports[split_name] = {"split": split_name, "n_windows": 0, "note": "no windows in this split"}
            continue
        X_split = X.loc[split_mask]
        regressor_selection = _predict_regressor_selection(regressors, X_split)
        classifier_selection = classifier.predict(X_split.values).tolist()
        reports[split_name] = run_split_report(
            split_name, window_idxs, anwg_wide, metric_lut,
            regressor_selection, classifier_selection, best_fixed_policy,
        )

    result = {
        "trained": True,
        "candidate_policies": list(CANDIDATE_POLICIES),
        "feature_columns": feat_cols,
        "n_features": len(feat_cols),
        "n_train_windows": int(train_mask.sum()),
        "best_fixed_policy_on_train": best_fixed_policy,
        "model_hyperparameters": RF_PARAMS,
        "reports_by_split": reports,
        "note": "Faithful external baselines are NOT included in this comparison per the Option B scope "
                "decision -- see docs/selector_v2_faithful_baseline_scope_audit.md and Protocol C "
                "(docs/external_baseline_integration.md) for that later, separate evaluation.",
    }
    (pilot_dir / "selector_metrics.json").write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
