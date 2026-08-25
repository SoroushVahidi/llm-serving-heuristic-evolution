#!/usr/bin/env python3
"""CLI for the Multi-Family Contextual Selector v1 experiment (Step 3).

See docs/design/MULTIFAMILY_CONTEXTUAL_SELECTOR_V1.md. Runs all three
preregistered split regimes (within-family, pooled, leave-one-family-out),
all baselines/models, the family-predictability diagnostic, and the
shared-feature robustness check, then applies the frozen verdict gates.
Writes results to experiments/multifamily_contextual_selector_v1/.

No selector is deployed; no mechanism attribution, composition, or
synthesis work is performed. This is evaluation only.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from llmserveopt.selector import multifamily_contextual_selector_v1 as mcs  # noqa: E402

DEFAULT_OUT_DIR = ROOT / "experiments" / "multifamily_contextual_selector_v1"
MODEL_NAMES = ["logreg", "tree", "forest", "utility_argmax", "pairwise", "best_fixed", "majority"]


def _log(run_dir: Path, msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(run_dir / "run.log", "a") as f:
        f.write(line + "\n")


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _fit_and_predict_all(train: pd.DataFrame, test: pd.DataFrame, smoke: bool = False) -> Dict[str, pd.Series]:
    numeric, categorical = mcs.infer_column_kinds(pd.concat([train, test]))
    X_train = mcs.build_X(train, numeric, categorical)
    X_test = mcs.build_X(test, numeric, categorical)
    y_train = mcs.compute_exact_winner(train)
    preproc = mcs.build_preprocessor(numeric, categorical)

    preds: Dict[str, pd.Series] = {}
    classifier_names = ("forest",) if smoke else ("logreg", "tree", "forest")
    for name in classifier_names:
        pipe = mcs.fit_classifier(name, X_train, y_train, preproc)
        preds[name] = mcs.predict_classifier(pipe, X_test, test.index)
    if not smoke:
        reg_models = mcs.fit_utility_regressors(X_train, train, preproc)
        preds["utility_argmax"] = mcs.predict_utility_argmax(reg_models, X_test, test.index)
        pw_models = mcs.fit_pairwise(X_train, train, preproc)
        preds["pairwise"] = mcs.predict_pairwise(pw_models, X_test, test.index)
    preds["best_fixed"] = mcs.baseline_best_fixed(train, test)
    preds["majority"] = mcs.baseline_majority(train, test)
    return preds


def _eval_all(test: pd.DataFrame, preds: Dict[str, pd.Series]) -> Dict[str, Dict[str, float]]:
    metrics = {name: mcs.evaluate_predictions(test, p) for name, p in preds.items()}
    fixed_regret = metrics["best_fixed"]["mean_regret"]
    for name in metrics:
        metrics[name]["gap_to_best_fixed_mean_regret"] = metrics[name]["mean_regret"] - fixed_regret
    return metrics


def bootstrap_ci_mean_regret(df: pd.DataFrame, predicted: pd.Series, n_boot: int = 1000, seed: int = mcs.SEED) -> Dict[str, float]:
    regret = mcs.regret_of(df, predicted)
    groups = df["group_key"].to_numpy()
    unique_groups = np.array(sorted(set(groups)))
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        mask = np.concatenate([np.where(groups == g)[0] for g in sampled_groups])
        means.append(float(np.mean(regret[mask])))
    lo, hi = np.percentile(means, [5, 95])
    return {"mean": float(np.mean(regret)), "ci90_lo": float(lo), "ci90_hi": float(hi)}


def macro_by_family(df_test: pd.DataFrame, preds: Dict[str, pd.Series]) -> Dict[str, Dict[str, float]]:
    out = {}
    for name, pred in preds.items():
        fam_regrets = []
        per_fam = {}
        for fam in mcs.FAMILIES:
            mask = df_test["mechanism_family"] == fam
            if mask.sum() == 0:
                continue
            m = mcs.evaluate_predictions(df_test[mask], pred[mask])
            per_fam[fam] = m
            fam_regrets.append(m["mean_regret"])
        out[name] = {"macro_mean_regret": float(np.mean(fam_regrets)) if fam_regrets else None, "per_family": per_fam}
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    _log(out_dir, f"Starting multifamily contextual selector v1 (smoke={args.smoke}). git_head={_git_sha()}")

    df = mcs.load_dataset()
    _log(out_dir, f"Loaded dataset: {len(df)} scenarios, {len(mcs.FEATURE_COLUMNS)} features, {len(mcs.POLICY_COLUMNS)} policies.")

    results: Dict[str, object] = {
        "builder_version": "multifamily_contextual_selector_v1.0.0",
        "build_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "build_git_head_sha": _git_sha(),
        "n_scenarios": len(df),
        "n_features": len(mcs.FEATURE_COLUMNS),
        "policy_columns": mcs.POLICY_COLUMNS,
    }

    # ---- Family-predictability diagnostic ----
    _log(out_dir, "Running family-predictability diagnostic...")
    results["family_predictability"] = mcs.family_predictability_diagnostic(df)
    _log(out_dir, f"Family predictability: {results['family_predictability']['mean_accuracy']:.4f}")

    # ---- Shared-feature robustness check ----
    _log(out_dir, "Running shared-feature (A<->B only) robustness check...")
    results["shared_feature_robustness"] = mcs.shared_feature_robustness_check(df)

    # ---- Regime A: within-family ----
    _log(out_dir, "Regime A: within-family grouped holdout...")
    regime_a = mcs.regime_a_within_family_splits(df)
    regime_a_results = {}
    for fam, splits in regime_a.items():
        preds = _fit_and_predict_all(splits["train"], splits["test"], smoke=args.smoke)
        regime_a_results[fam] = _eval_all(splits["test"], preds)
    results["regime_a_within_family"] = regime_a_results

    # ---- Regime B: pooled ----
    _log(out_dir, "Regime B: multi-family pooled holdout...")
    regime_b = mcs.regime_b_pooled_split(df)
    preds_b = _fit_and_predict_all(regime_b["train"], regime_b["test"], smoke=args.smoke)
    results["regime_b_pooled"] = {
        "overall": _eval_all(regime_b["test"], preds_b),
        "macro_by_family": macro_by_family(regime_b["test"], preds_b),
        "bootstrap_ci_forest": bootstrap_ci_mean_regret(regime_b["test"], preds_b["forest"], n_boot=50 if args.smoke else 1000),
        "bootstrap_ci_best_fixed": bootstrap_ci_mean_regret(regime_b["test"], preds_b["best_fixed"], n_boot=50 if args.smoke else 1000),
    }

    # ---- Regime C: LOFO ----
    _log(out_dir, "Regime C: leave-one-family-out...")
    regime_c = mcs.regime_c_lofo_splits(df)
    regime_c_results = {}
    for held_out, splits in regime_c.items():
        preds = _fit_and_predict_all(splits["train"], splits["test"], smoke=args.smoke)
        metrics = _eval_all(splits["test"], preds)
        regime_c_results[held_out] = {
            "overall": metrics,
            "bootstrap_ci_forest": bootstrap_ci_mean_regret(splits["test"], preds["forest"], n_boot=50 if args.smoke else 1000),
            "bootstrap_ci_best_fixed": bootstrap_ci_mean_regret(splits["test"], preds["best_fixed"], n_boot=50 if args.smoke else 1000),
        }
        _log(out_dir, f"LOFO held out {held_out}: forest mean_regret={metrics['forest']['mean_regret']:.4f}, "
                       f"best_fixed mean_regret={metrics['best_fixed']['mean_regret']:.4f}")
    results["regime_c_lofo"] = regime_c_results

    if args.smoke:
        _log(out_dir, "Smoke run complete (verdict gates require the full model set; skipped in --smoke).")
        out_path = out_dir / "multifamily_contextual_selector_v1_results.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        _log(out_dir, f"Smoke results written to {out_path}")
        return

    # ---- Frozen verdict gates ----
    best_model_b = min(("logreg", "tree", "forest"), key=lambda m: results["regime_b_pooled"]["overall"][m]["mean_regret"])
    gate1 = results["regime_b_pooled"]["overall"][best_model_b]["mean_regret"] <= 0.8 * results["regime_b_pooled"]["overall"]["best_fixed"]["mean_regret"]
    gate2 = (results["regime_b_pooled"]["overall"][best_model_b]["epsilon_optimal_accuracy"]
             - results["regime_b_pooled"]["overall"]["majority"]["epsilon_optimal_accuracy"]) >= 0.15
    gate3 = all(
        results["regime_b_pooled"]["macro_by_family"][best_model_b]["per_family"][fam]["mean_regret"]
        < results["regime_b_pooled"]["macro_by_family"]["best_fixed"]["per_family"][fam]["mean_regret"]
        for fam in mcs.FAMILIES
    )
    lofo_wins = sum(
        results["regime_c_lofo"][fam]["overall"][best_model_b]["mean_regret"]
        < results["regime_c_lofo"][fam]["overall"]["best_fixed"]["mean_regret"]
        for fam in mcs.FAMILIES
    )
    gate4 = lofo_wins >= 2
    gate5 = results["shared_feature_robustness"]["improvement_over_fixed"] > 0

    if gate1 and gate2 and gate3 and gate4 and gate5:
        verdict = "MULTIFAMILY_SELECTOR_GO"
    elif gate1 and gate2 and lofo_wins <= 1:
        verdict = "MULTIFAMILY_SELECTOR_ID_ONLY"
    elif not gate1 or not gate2:
        verdict = "MULTIFAMILY_SELECTOR_NO_GO"
    else:
        verdict = "MULTIFAMILY_SELECTOR_INCONCLUSIVE"

    results["verdict_gates"] = {
        "best_model_regime_b": best_model_b,
        "gate1_pooled_regret_beats_fixed_by_20pct": bool(gate1),
        "gate2_eps_optimal_beats_majority_by_15pp": bool(gate2),
        "gate3_macro_family_beats_fixed_in_all_3": bool(gate3),
        "gate4_lofo_wins_at_least_2_of_3": bool(gate4),
        "lofo_wins": int(lofo_wins),
        "gate5_shared_feature_robustness_positive": bool(gate5),
    }
    results["final_verdict"] = verdict
    _log(out_dir, f"FINAL VERDICT: {verdict}")

    out_path = out_dir / "multifamily_contextual_selector_v1_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    _log(out_dir, f"Results written to {out_path}")


if __name__ == "__main__":
    main()
