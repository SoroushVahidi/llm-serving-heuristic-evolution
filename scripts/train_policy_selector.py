#!/usr/bin/env python3
"""
Train baseline policy selector models from a selector dataset CSV.

Usage
-----
python scripts/train_policy_selector.py \
    --dataset results/.../train.csv \
    --validation-dataset results/.../validation.csv \
    --output results/models/

Outputs (organized by model type under --output/)
--------------------------------------------------
decision_tree/model.joblib
decision_tree/metrics.json
decision_tree/tree_text.txt
decision_tree/feature_importance.csv
decision_tree/confusion_matrix.csv
random_forest/model.joblib
random_forest/metrics.json
random_forest/feature_importance.csv
random_forest/confusion_matrix.csv
rule_based/metrics.json
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llmserveopt.selector.models import (
    RuleBasedSelector,
    evaluate_selector,
    save_metrics,
)


def load_dataset(path: str) -> list:
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for k, v in row.items():
                if v == "" or v is None:
                    parsed[k] = float("nan")
                else:
                    try:
                        parsed[k] = float(v)
                    except ValueError:
                        parsed[k] = v
            rows.append(parsed)
    return rows


def split_train_val(rows: list, val_fraction: float = 0.15, seed: int = 42):
    import random
    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    split = max(1, int(len(shuffled) * (1 - val_fraction)))
    return shuffled[:split], shuffled[split:]


def save_confusion_matrix(confusion: dict, path: str) -> None:
    all_classes = sorted(set(
        list(confusion.keys()) +
        [k for v in confusion.values() for k in v.keys()]
    ))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["predicted \\ actual"] + all_classes)
        for pred in all_classes:
            row_data = [pred] + [confusion.get(pred, {}).get(actual, 0) for actual in all_classes]
            writer.writerow(row_data)


def save_feature_importance(fi: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["feature", "importance"])
        for name in sorted(fi, key=lambda k: fi[k], reverse=True):
            writer.writerow([name, fi[name]])


def compute_reward_metrics(selector, rows: list) -> dict:
    """Compute reward-based metrics (not just accuracy)."""
    if not rows:
        return {"selected_mean_wg": float("nan"), "regret_to_window_best": float("nan")}

    preds = selector.predict(rows)
    selected_wg = []
    best_wg = []
    for pred, row in zip(preds, rows):
        wg_col = f"reward_{pred}"
        selected = row.get(wg_col, float("nan"))
        best = row.get("best_weighted_goodput", float("nan"))
        if not (math.isnan(selected) or math.isnan(best)):
            selected_wg.append(float(selected))
            best_wg.append(float(best))

    if not selected_wg:
        return {"selected_mean_wg": float("nan"), "regret_to_window_best": float("nan")}

    import numpy as np
    mean_sel = float(np.mean(selected_wg))
    mean_best = float(np.mean(best_wg))
    return {
        "selected_mean_wg": mean_sel,
        "best_mean_wg": mean_best,
        "regret_to_window_best": mean_best - mean_sel,
    }


def parse_args():
    p = argparse.ArgumentParser(description="Train policy selector models")
    p.add_argument("--dataset", required=True, help="Training selector dataset CSV")
    p.add_argument("--validation-dataset", default=None, help="Validation CSV (if separate)")
    p.add_argument("--test-dataset", default=None, help="Test CSV (evaluation only)")
    p.add_argument("--output", required=True, help="Output base directory")
    p.add_argument("--model-types", default="rule_based,decision_tree,random_forest",
                   help="Comma-separated model types to train")
    p.add_argument("--val-fraction", type=float, default=0.15,
                   help="Fraction of training data to use as validation if no --validation-dataset")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-depth-dt", type=int, default=8)
    p.add_argument("--min-samples-leaf", type=int, default=5)
    p.add_argument("--n-estimators", type=int, default=200)
    p.add_argument("--max-depth-rf", type=int, default=10)
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    model_types = [m.strip() for m in args.model_types.split(",")]

    print(f"Loading training dataset: {args.dataset}")
    train_rows = load_dataset(args.dataset)
    print(f"  {len(train_rows)} windows")

    if not train_rows:
        print("ERROR: empty training dataset.")
        return 1

    # Validation data
    if args.validation_dataset:
        print(f"Loading validation dataset: {args.validation_dataset}")
        val_rows = load_dataset(args.validation_dataset)
        print(f"  {len(val_rows)} windows")
    else:
        train_rows, val_rows = split_train_val(train_rows, val_fraction=args.val_fraction, seed=args.seed)
        print(f"  split: train={len(train_rows)} val={len(val_rows)}")

    # Test data (optional)
    test_rows = []
    if args.test_dataset:
        print(f"Loading test dataset: {args.test_dataset}")
        test_rows = load_dataset(args.test_dataset)
        print(f"  {len(test_rows)} windows")

    print(f"\nModel types: {model_types}")
    all_results = {}

    sklearn_ok = False
    try:
        import sklearn
        sklearn_ok = True
    except ImportError:
        if any(m in model_types for m in ("decision_tree", "random_forest")):
            print("WARNING: scikit-learn not installed. Tree/forest models skipped.")
            print("  pip install scikit-learn")

    # --- Rule-based ---
    if "rule_based" in model_types:
        rb_dir = out / "rule_based"
        rb_dir.mkdir(exist_ok=True)
        print("\n[rule_based] Training...")
        rb = RuleBasedSelector()
        rb_metrics = {
            "train": evaluate_selector(rb, train_rows),
            "validation": evaluate_selector(rb, val_rows) if val_rows else {},
        }
        if test_rows:
            rb_metrics["test"] = evaluate_selector(rb, test_rows)
        rb_metrics["reward"] = {
            "train": compute_reward_metrics(rb, train_rows),
            "validation": compute_reward_metrics(rb, val_rows) if val_rows else {},
        }
        save_metrics(rb_metrics, str(rb_dir / "metrics.json"))
        print(f"  val accuracy: {rb_metrics.get('validation', {}).get('accuracy', float('nan')):.3f}")
        all_results["rule_based"] = rb_metrics

    # --- Decision Tree ---
    if "decision_tree" in model_types and sklearn_ok:
        from llmserveopt.selector.models import DecisionTreeSelector
        dt_dir = out / "decision_tree"
        dt_dir.mkdir(exist_ok=True)
        print("\n[decision_tree] Training...")
        dt = DecisionTreeSelector(
            max_depth=args.max_depth_dt,
            min_samples_leaf=args.min_samples_leaf,
            random_state=args.seed,
        )
        dt.fit(train_rows)
        dt_metrics = {
            "train": evaluate_selector(dt, train_rows),
            "validation": evaluate_selector(dt, val_rows) if val_rows else {},
        }
        if test_rows:
            dt_metrics["test"] = evaluate_selector(dt, test_rows)
        dt_metrics["reward"] = {
            "train": compute_reward_metrics(dt, train_rows),
            "validation": compute_reward_metrics(dt, val_rows) if val_rows else {},
        }
        if test_rows:
            dt_metrics["reward"]["test"] = compute_reward_metrics(dt, test_rows)

        save_metrics(dt_metrics, str(dt_dir / "metrics.json"))
        try:
            dt.save(str(dt_dir / "model.joblib"))
        except ImportError:
            print("  WARNING: joblib not available.")

        fi = dt.feature_importances()
        save_feature_importance(fi, str(dt_dir / "feature_importance.csv"))
        try:
            (dt_dir / "tree_text.txt").write_text(dt.tree_text())
        except Exception as e:
            print(f"  tree_text skipped: {e}")
        save_confusion_matrix(
            dt_metrics.get("validation", {}).get("confusion", {}),
            str(dt_dir / "confusion_matrix.csv")
        )
        print(f"  val accuracy: {dt_metrics.get('validation', {}).get('accuracy', float('nan')):.3f}")
        print(f"  val reward:   {dt_metrics['reward'].get('validation', {}).get('selected_mean_wg', float('nan')):.4f}")
        all_results["decision_tree"] = dt_metrics

    # --- Random Forest ---
    if "random_forest" in model_types and sklearn_ok:
        from llmserveopt.selector.models import RandomForestSelector
        rf_dir = out / "random_forest"
        rf_dir.mkdir(exist_ok=True)
        print("\n[random_forest] Training...")
        rf = RandomForestSelector(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth_rf,
            random_state=args.seed,
            n_jobs=-1,
        )
        rf.fit(train_rows)
        rf_metrics = {
            "train": evaluate_selector(rf, train_rows),
            "validation": evaluate_selector(rf, val_rows) if val_rows else {},
        }
        if test_rows:
            rf_metrics["test"] = evaluate_selector(rf, test_rows)
        rf_metrics["reward"] = {
            "train": compute_reward_metrics(rf, train_rows),
            "validation": compute_reward_metrics(rf, val_rows) if val_rows else {},
        }
        if test_rows:
            rf_metrics["reward"]["test"] = compute_reward_metrics(rf, test_rows)

        save_metrics(rf_metrics, str(rf_dir / "metrics.json"))
        try:
            rf.save(str(rf_dir / "model.joblib"))
        except ImportError:
            print("  WARNING: joblib not available.")

        fi_rf = rf.feature_importances()
        save_feature_importance(fi_rf, str(rf_dir / "feature_importance.csv"))
        save_confusion_matrix(
            rf_metrics.get("validation", {}).get("confusion", {}),
            str(rf_dir / "confusion_matrix.csv")
        )
        print(f"  val accuracy: {rf_metrics.get('validation', {}).get('accuracy', float('nan')):.3f}")
        print(f"  val reward:   {rf_metrics['reward'].get('validation', {}).get('selected_mean_wg', float('nan')):.4f}")
        all_results["random_forest"] = rf_metrics

    # Combined summary
    summary_path = str(out / "training_summary.json")
    save_metrics({"models": all_results, "train_n": len(train_rows), "val_n": len(val_rows)}, summary_path)

    print(f"\n{'='*50}")
    print("Summary (validation):")
    print(f"  {'Model':30s} {'Accuracy':>10} {'Sel WG':>10} {'Regret':>10}")
    for mname, mres in all_results.items():
        acc = mres.get("validation", {}).get("accuracy", float("nan"))
        reward = mres.get("reward", {}).get("validation", {})
        sel_wg = reward.get("selected_mean_wg", float("nan"))
        regret = reward.get("regret_to_window_best", float("nan"))
        print(f"  {mname:30s} {acc:>10.3f} {sel_wg:>10.4f} {regret:>10.4f}")

    print(f"\nAll outputs written to: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
