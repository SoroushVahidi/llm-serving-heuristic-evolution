#!/usr/bin/env python3
"""
Train baseline policy selector models from a selector dataset CSV.

Usage
-----
python scripts/train_policy_selector.py \
    --dataset results/phase2a2_selector_dataset/smoke_selector_dataset.csv \
    --output results/phase2a2_selector_dataset/smoke_selector_model

Outputs (under --output/)
--------------------------
model_decision_tree.joblib
model_random_forest.joblib
metrics.json
confusion_matrix.csv
feature_importance.csv
tree_text.txt
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llmserveopt.selector.candidates import SELECTOR_CANDIDATES
from llmserveopt.selector.features import FEATURE_NAMES
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
            # Cast numeric columns
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


def split_train_test(rows: list, test_fraction: float = 0.2, seed: int = 42):
    import random
    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    split = max(1, int(len(shuffled) * (1 - test_fraction)))
    return shuffled[:split], shuffled[split:]


def save_confusion_matrix(confusion: dict, path: str) -> None:
    all_classes = sorted(set(
        list(confusion.keys()) +
        [k for v in confusion.values() for k in v.keys()]
    ))
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["predicted \\ actual"] + all_classes)
        for pred in all_classes:
            row = [pred] + [confusion.get(pred, {}).get(actual, 0) for actual in all_classes]
            writer.writerow(row)


def save_feature_importance(fi: dict, path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["feature", "importance"])
        for name in sorted(fi, key=lambda k: fi[k], reverse=True):
            writer.writerow([name, fi[name]])


def parse_args():
    p = argparse.ArgumentParser(description="Train policy selector models")
    p.add_argument("--dataset", required=True, help="Selector dataset CSV")
    p.add_argument("--output", required=True, help="Output directory")
    p.add_argument("--test-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-depth-dt", type=int, default=8, help="DecisionTree max_depth")
    p.add_argument("--min-samples-leaf", type=int, default=20)
    p.add_argument("--n-estimators", type=int, default=200, help="RandomForest n_estimators")
    p.add_argument("--max-depth-rf", type=int, default=10, help="RandomForest max_depth")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {args.dataset}")
    rows = load_dataset(args.dataset)
    print(f"  {len(rows)} windows loaded")

    if len(rows) == 0:
        print("ERROR: empty dataset.")
        return 1

    train, test = split_train_test(rows, test_fraction=args.test_fraction, seed=args.seed)
    print(f"  train={len(train)}  test={len(test)}")

    all_metrics = {}

    # --- Rule-based baseline ---
    print("\n[1/3] Rule-based baseline (always 'fifo')...")
    rb = RuleBasedSelector()
    rb_metrics = evaluate_selector(rb, test if test else rows)
    all_metrics["rule_based"] = rb_metrics
    print(f"  accuracy: {rb_metrics['accuracy']:.3f}")

    # --- Decision Tree ---
    sklearn = None
    try:
        import sklearn as _sk
        sklearn = _sk
    except ImportError:
        print("\nWARNING: scikit-learn not installed. Skipping tree/forest models.")
        print("  Install with: pip install scikit-learn")

    if sklearn is not None:
        print("\n[2/3] Decision Tree...")
        from llmserveopt.selector.models import DecisionTreeSelector
        dt = DecisionTreeSelector(
            max_depth=args.max_depth_dt,
            min_samples_leaf=args.min_samples_leaf,
            random_state=args.seed,
        )
        dt.fit(train)
        dt_metrics = evaluate_selector(dt, test if test else rows)
        all_metrics["decision_tree"] = dt_metrics
        print(f"  accuracy: {dt_metrics['accuracy']:.3f}")

        dt_path = str(out / "model_decision_tree.joblib")
        try:
            dt.save(dt_path)
            print(f"  saved → {dt_path}")
        except ImportError:
            print("  WARNING: joblib not available, model not saved.")

        fi = dt.feature_importances()
        save_feature_importance(fi, str(out / "feature_importance_dt.csv"))
        try:
            tree_txt = dt.tree_text()
            (out / "tree_text.txt").write_text(tree_txt)
        except Exception as e:
            print(f"  tree_text skipped: {e}")

        save_confusion_matrix(dt_metrics.get("confusion", {}), str(out / "confusion_matrix_dt.csv"))

        print("\n[3/3] Random Forest...")
        from llmserveopt.selector.models import RandomForestSelector
        rf = RandomForestSelector(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth_rf,
            random_state=args.seed,
            n_jobs=-1,
        )
        rf.fit(train)
        rf_metrics = evaluate_selector(rf, test if test else rows)
        all_metrics["random_forest"] = rf_metrics
        print(f"  accuracy: {rf_metrics['accuracy']:.3f}")

        rf_path = str(out / "model_random_forest.joblib")
        try:
            rf.save(rf_path)
            print(f"  saved → {rf_path}")
        except ImportError:
            print("  WARNING: joblib not available, model not saved.")

        fi_rf = rf.feature_importances()
        save_feature_importance(fi_rf, str(out / "feature_importance_rf.csv"))
        save_confusion_matrix(rf_metrics.get("confusion", {}), str(out / "confusion_matrix_rf.csv"))
    else:
        print("[2/3] Decision Tree — SKIPPED (sklearn missing)")
        print("[3/3] Random Forest — SKIPPED (sklearn missing)")

    # Summary
    metrics_path = str(out / "metrics.json")
    save_metrics(all_metrics, metrics_path)
    print(f"\nMetrics saved → {metrics_path}")

    print("\nSummary:")
    for model_name, m in all_metrics.items():
        print(f"  {model_name:30s}: accuracy={m['accuracy']:.3f}  n_test={m['n_test']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
