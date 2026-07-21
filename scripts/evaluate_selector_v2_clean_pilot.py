#!/usr/bin/env python3
"""Train and evaluate a clean Selector Dataset v2 pilot.

This script refuses to train unless:
  - quality_gates.json reports all gates passed;
  - leakage_audit.json exists and reports passed=true.

Model choice is validation-only. TEST and OOD_TEST are evaluation-only.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from llmserveopt.selector.advanced import (  # noqa: E402
    PolicyRewardRegressorSelector,
    anwg_column,
    validate_feature_columns,
)


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _feature_columns(features: pd.DataFrame) -> List[str]:
    cols = sorted(c for c in features.columns if c.startswith("feat_"))
    return validate_feature_columns(cols)


def _prepare_tables(pilot_dir: Path):
    windows = pd.read_csv(pilot_dir / "retained_windows.csv")
    features = pd.read_csv(pilot_dir / "window_features.csv")
    vectors = pd.read_csv(pilot_dir / "full_policy_vectors.csv")
    merged = windows.merge(features, on="window_idx", how="left")
    feat_cols = _feature_columns(features)
    policies = sorted(vectors["policy_name"].unique().tolist())

    anwg = vectors.pivot_table(
        index="window_idx",
        columns="policy_name",
        values="metric_arrival_normalized_weighted_goodput",
        aggfunc="first",
    ).reindex(columns=policies).fillna(0.0)

    metric_tables = {
        "anwg": anwg,
        "completion": vectors.pivot_table(
            index="window_idx", columns="policy_name", values="metric_completion_fraction", aggfunc="first"
        ).reindex(columns=policies),
        "quality": vectors.pivot_table(
            index="window_idx", columns="policy_name", values="metric_weighted_goodput", aggfunc="first"
        ).reindex(columns=policies),
    }

    for policy in policies:
        merged[anwg_column(policy)] = merged["window_idx"].map(anwg[policy])
    merged["oracle_policy"] = merged["window_idx"].map(lambda idx: anwg.loc[idx].idxmax())
    merged["oracle_anwg"] = merged["window_idx"].map(lambda idx: float(anwg.loc[idx].max()))
    merged["winner_margin"] = merged["window_idx"].map(lambda idx: _top_two_margin(anwg.loc[idx].to_numpy(dtype=float)))
    merged["best_policy_label"] = merged["oracle_policy"]
    return merged, vectors, policies, feat_cols, metric_tables


def _top_two_margin(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    ordered = np.sort(values)
    return float(ordered[-1] - ordered[-2])


def _fit_train_imputer(train: pd.DataFrame, feat_cols: Sequence[str]) -> Dict[str, float]:
    stats: Dict[str, float] = {}
    for col in feat_cols:
        vals = pd.to_numeric(train[col], errors="coerce")
        med = vals.median()
        stats[col] = 0.0 if pd.isna(med) else float(med)
    return stats


def _apply_imputer(df: pd.DataFrame, feat_cols: Sequence[str], stats: Dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    for col in feat_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(stats[col])
    return out


class DecisionTreePilotSelector:
    name = "decision_tree_classifier"

    def __init__(self, feat_cols: Sequence[str], random_state: int):
        from sklearn.tree import DecisionTreeClassifier

        self.feature_cols = list(feat_cols)
        self.model = DecisionTreeClassifier(max_depth=6, min_samples_leaf=3, random_state=random_state)

    def fit(self, rows: pd.DataFrame):
        self.model.fit(rows[self.feature_cols].to_numpy(dtype=float), rows["best_policy_label"].astype(str))
        return self

    def predict(self, rows: pd.DataFrame) -> List[str]:
        return self.model.predict(rows[self.feature_cols].to_numpy(dtype=float)).tolist()


def _fit_models(
    train: pd.DataFrame,
    policies: Sequence[str],
    feat_cols: Sequence[str],
    random_state: int,
):
    models = [
        DecisionTreePilotSelector(feat_cols, random_state).fit(train),
        PolicyRewardRegressorSelector(
            name="rf_reward_regression",
            allowed_policies=policies,
            feature_cols=feat_cols,
            estimator="random_forest",
            n_estimators=120,
            max_depth=8,
            random_state=random_state,
        ).fit(train),
        PolicyRewardRegressorSelector(
            name="extra_trees_reward_regression",
            allowed_policies=policies,
            feature_cols=feat_cols,
            estimator="extra_trees",
            n_estimators=120,
            max_depth=8,
            random_state=random_state,
        ).fit(train),
    ]
    return models


def _selection_metrics(
    *,
    rows: pd.DataFrame,
    selections: Sequence[str],
    label: str,
    metric_tables: Dict[str, pd.DataFrame],
    best_fixed_policy: str,
    meaningful_margin: float,
    bootstrap: int,
    seed: int,
) -> Dict:
    if rows.empty:
        return {"label": label, "n_windows": 0}
    idxs = rows["window_idx"].tolist()
    selected_anwg = np.asarray([metric_tables["anwg"].loc[idx, pol] for idx, pol in zip(idxs, selections)], dtype=float)
    selected_completion = np.asarray([metric_tables["completion"].loc[idx, pol] for idx, pol in zip(idxs, selections)], dtype=float)
    selected_quality = np.asarray([metric_tables["quality"].loc[idx, pol] for idx, pol in zip(idxs, selections)], dtype=float)
    oracle = np.asarray([metric_tables["anwg"].loc[idx].max() for idx in idxs], dtype=float)
    best_fixed = np.asarray([metric_tables["anwg"].loc[idx, best_fixed_policy] for idx in idxs], dtype=float)
    regrets = oracle - selected_anwg
    mean_selected = float(np.nanmean(selected_anwg))
    mean_fixed = float(np.nanmean(best_fixed))
    mean_oracle = float(np.nanmean(oracle))
    denom = mean_oracle - mean_fixed
    margins = rows["winner_margin"].to_numpy(dtype=float)
    meaningful_mask = margins >= meaningful_margin
    result = {
        "label": label,
        "n_windows": int(len(rows)),
        "anwg": round(mean_selected, 6),
        "completion_fraction": _round_nanmean(selected_completion),
        "completed_request_quality": _round_nanmean(selected_quality),
        "mean_oracle_regret": round(float(np.nanmean(regrets)), 6),
        "p95_oracle_regret": round(float(np.nanpercentile(regrets, 95)), 6),
        "worst_case_oracle_regret": round(float(np.nanmax(regrets)), 6),
        "within_0.001_of_oracle": round(float(np.mean(regrets <= 0.001 + 1e-12)), 6),
        "within_0.005_of_oracle": round(float(np.mean(regrets <= 0.005 + 1e-12)), 6),
        "within_0.010_of_oracle": round(float(np.mean(regrets <= 0.010 + 1e-12)), 6),
        "gap_closed_fraction": (
            round((mean_selected - mean_fixed) / denom, 6) if denom > 1e-12 else None
        ),
        "meaningful_margin_threshold": meaningful_margin,
        "meaningful_window_count": int(meaningful_mask.sum()),
    }
    if meaningful_mask.any():
        m_sel = selected_anwg[meaningful_mask]
        m_oracle = oracle[meaningful_mask]
        m_fixed = best_fixed[meaningful_mask]
        m_regret = m_oracle - m_sel
        m_denom = float(np.nanmean(m_oracle) - np.nanmean(m_fixed))
        result["meaningful"] = {
            "n_windows": int(meaningful_mask.sum()),
            "anwg": round(float(np.nanmean(m_sel)), 6),
            "mean_oracle_regret": round(float(np.nanmean(m_regret)), 6),
            "p95_oracle_regret": round(float(np.nanpercentile(m_regret, 95)), 6),
            "worst_case_oracle_regret": round(float(np.nanmax(m_regret)), 6),
            "gap_closed_fraction": (
                round((float(np.nanmean(m_sel)) - float(np.nanmean(m_fixed))) / m_denom, 6)
                if m_denom > 1e-12 else None
            ),
        }
    else:
        result["meaningful"] = {"n_windows": 0}
    if bootstrap > 0 and len(rows) >= 2:
        result["bootstrap_ci"] = _bootstrap_ci(selected_anwg, best_fixed, oracle, bootstrap, seed)
    return result


def _round_nanmean(values: np.ndarray) -> float | None:
    if np.all(np.isnan(values)):
        return None
    return round(float(np.nanmean(values)), 6)


def _bootstrap_ci(selected: np.ndarray, fixed: np.ndarray, oracle: np.ndarray, n: int, seed: int) -> Dict:
    rng = np.random.default_rng(seed)
    anwg_vals = []
    diff_fixed = []
    regret_vals = []
    size = len(selected)
    for _ in range(n):
        idx = rng.integers(0, size, size=size)
        anwg_vals.append(float(np.nanmean(selected[idx])))
        diff_fixed.append(float(np.nanmean(selected[idx] - fixed[idx])))
        regret_vals.append(float(np.nanmean(oracle[idx] - selected[idx])))
    return {
        "anwg_95ci": [round(float(np.percentile(anwg_vals, 2.5)), 6), round(float(np.percentile(anwg_vals, 97.5)), 6)],
        "diff_vs_best_fixed_95ci": [round(float(np.percentile(diff_fixed, 2.5)), 6), round(float(np.percentile(diff_fixed, 97.5)), 6)],
        "mean_oracle_regret_95ci": [round(float(np.percentile(regret_vals, 2.5)), 6), round(float(np.percentile(regret_vals, 97.5)), 6)],
    }


def _evaluate_all(
    rows: pd.DataFrame,
    models,
    policies: Sequence[str],
    metric_tables: Dict[str, pd.DataFrame],
    best_fixed_policy: str,
    meaningful_margin: float,
    bootstrap: int,
    seed: int,
) -> Dict[str, Dict]:
    reports = {}
    for model in models:
        reports[model.name] = _selection_metrics(
            rows=rows,
            selections=model.predict(rows),
            label=model.name,
            metric_tables=metric_tables,
            best_fixed_policy=best_fixed_policy,
            meaningful_margin=meaningful_margin,
            bootstrap=bootstrap,
            seed=seed,
        )
    for policy in policies:
        reports[f"fixed__{policy}"] = _selection_metrics(
            rows=rows,
            selections=[policy] * len(rows),
            label=f"fixed__{policy}",
            metric_tables=metric_tables,
            best_fixed_policy=best_fixed_policy,
            meaningful_margin=meaningful_margin,
            bootstrap=bootstrap,
            seed=seed,
        )
    oracle_sel = [metric_tables["anwg"].loc[idx].idxmax() for idx in rows["window_idx"].tolist()]
    reports["oracle_per_window"] = _selection_metrics(
        rows=rows,
        selections=oracle_sel,
        label="oracle_per_window",
        metric_tables=metric_tables,
        best_fixed_policy=best_fixed_policy,
        meaningful_margin=meaningful_margin,
        bootstrap=bootstrap,
        seed=seed,
    )
    return reports


def _dataset_stats(rows: pd.DataFrame, policies: Sequence[str], meaningful_margin: float) -> Dict:
    return {
        "n_windows": int(len(rows)),
        "split_counts": rows["split"].value_counts().to_dict(),
        "source_composition": rows["source_trace"].value_counts().to_dict(),
        "dataset_family_composition": rows["dataset_family"].value_counts().to_dict(),
        "time_slice_pool_composition": rows["time_slice_pool"].value_counts().to_dict(),
        "policy_pool": list(policies),
        "near_tie_windows_lt_0.005": int((rows["winner_margin"] < meaningful_margin).sum()),
        "near_tie_fraction_lt_0.005": round(float((rows["winner_margin"] < meaningful_margin).mean()), 6),
        "meaningful_windows_ge_0.005": int((rows["winner_margin"] >= meaningful_margin).sum()),
        "meaningful_fraction_ge_0.005": round(float((rows["winner_margin"] >= meaningful_margin).mean()), 6),
        "oracle_best_policy_distribution": rows["oracle_policy"].value_counts().to_dict(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", required=True, type=Path)
    parser.add_argument("--meaningful-margin", type=float, default=0.005)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260720)
    args = parser.parse_args()

    t0 = time.perf_counter()
    pilot_dir = ROOT / args.pilot_dir if not args.pilot_dir.is_absolute() else args.pilot_dir
    gates = _load_json(pilot_dir / "quality_gates.json")
    audit = _load_json(pilot_dir / "leakage_audit.json")
    if not gates.get("all_gates_passed"):
        result = {
            "trained": False,
            "reason": "quality_gates_failed",
            "failed_gates": {k: v for k, v in gates.get("gates", {}).items() if not v.get("passed")},
        }
        (pilot_dir / "clean_selector_eval.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return 0
    if not audit.get("passed"):
        result = {
            "trained": False,
            "reason": "independent_leakage_audit_failed_or_missing",
            "audit_hard_failures": audit.get("hard_failures"),
        }
        (pilot_dir / "clean_selector_eval.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return 0

    rows, _vectors, policies, feat_cols, metric_tables = _prepare_tables(pilot_dir)
    train_raw = rows[rows["split"] == "TRAIN"].copy()
    validation_raw = rows[rows["split"] == "VALIDATION"].copy()
    test_raw = rows[rows["split"] == "ID_TEST"].copy()
    ood_raw = rows[rows["split"] == "OOD_TEST"].copy()
    if train_raw.empty or validation_raw.empty:
        result = {
            "trained": False,
            "reason": "TRAIN and VALIDATION splits must both be nonempty for validation-only model selection.",
            "split_counts": rows["split"].value_counts().to_dict(),
        }
        (pilot_dir / "clean_selector_eval.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return 0

    imputer_stats = _fit_train_imputer(train_raw, feat_cols)
    transformed = _apply_imputer(rows, feat_cols, imputer_stats)
    train = transformed[transformed["split"] == "TRAIN"].copy()
    validation = transformed[transformed["split"] == "VALIDATION"].copy()
    test = transformed[transformed["split"] == "ID_TEST"].copy()
    ood = transformed[transformed["split"] == "OOD_TEST"].copy()

    best_fixed_policy = metric_tables["anwg"].loc[train["window_idx"]].mean(axis=0).idxmax()
    models = _fit_models(train, policies, feat_cols, args.seed)

    validation_reports = _evaluate_all(
        validation, models, policies, metric_tables, best_fixed_policy,
        args.meaningful_margin, args.bootstrap, args.seed,
    )
    model_validation_scores = {
        model.name: validation_reports[model.name]["anwg"] for model in models
    }
    best_model_name = max(model_validation_scores, key=model_validation_scores.get)
    best_model = next(m for m in models if m.name == best_model_name)

    reports = {
        "TRAIN": _evaluate_all(train, models, policies, metric_tables, best_fixed_policy, args.meaningful_margin, args.bootstrap, args.seed),
        "VALIDATION": validation_reports,
        "ID_TEST": _evaluate_all(test, [best_model], policies, metric_tables, best_fixed_policy, args.meaningful_margin, args.bootstrap, args.seed),
        "OOD_TEST": _evaluate_all(ood, [best_model], policies, metric_tables, best_fixed_policy, args.meaningful_margin, args.bootstrap, args.seed),
    }
    elapsed = time.perf_counter() - t0
    result = {
        "trained": True,
        "pilot_dir": str(pilot_dir),
        "seed": args.seed,
        "primary_objective": "arrival_normalized_weighted_goodput",
        "meaningful_margin": args.meaningful_margin,
        "candidate_models": [m.name for m in models],
        "model_selection_rule": "highest VALIDATION ANWG; fixed hyperparameters; no TEST/OOD tuning",
        "best_selector_name": best_model_name,
        "best_selector_validation_anwg": model_validation_scores[best_model_name],
        "best_fixed_policy_on_train": best_fixed_policy,
        "strongest_previous_valid_causal_selector_where_comparable": (
            "none: previous clean Selector v2 pilot/model is not available; historical leaky pilot is excluded"
        ),
        "dataset_stats_all": _dataset_stats(rows, policies, args.meaningful_margin),
        "dataset_stats_by_split": {
            split: _dataset_stats(rows[rows["split"] == split], policies, args.meaningful_margin)
            for split in ("TRAIN", "VALIDATION", "ID_TEST", "OOD_TEST")
        },
        "preprocessing_audit": {
            "imputer": "per-feature median",
            "fit_split": "TRAIN only",
            "n_imputed_feature_columns": len(imputer_stats),
        },
        "reports_by_split": reports,
        "runtime_s": round(elapsed, 3),
    }
    (pilot_dir / "clean_selector_eval.json").write_text(json.dumps(result, indent=2, default=str))
    _write_summary_csv(result, pilot_dir / "clean_selector_comparison.csv")
    print(json.dumps({
        "trained": True,
        "best_selector_name": best_model_name,
        "best_selector_validation_anwg": model_validation_scores[best_model_name],
        "best_fixed_policy_on_train": best_fixed_policy,
        "runtime_s": round(elapsed, 3),
        "output": str(pilot_dir / "clean_selector_eval.json"),
    }, indent=2))
    return 0


def _write_summary_csv(result: Dict, path: Path) -> None:
    rows = []
    for split, report in result["reports_by_split"].items():
        for name, metrics in report.items():
            row = {"split": split, "name": name}
            for key, value in metrics.items():
                if key in {"meaningful", "bootstrap_ci"}:
                    continue
                row[key] = value
            if isinstance(metrics.get("meaningful"), dict):
                for key, value in metrics["meaningful"].items():
                    row[f"meaningful_{key}"] = value
            rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


if __name__ == "__main__":
    raise SystemExit(main())
