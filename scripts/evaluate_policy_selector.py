#!/usr/bin/env python3
"""
Evaluate trained selector models on validation, test, and sanity datasets.

Usage
-----
python scripts/evaluate_policy_selector.py \
    --models-dir results/phase2a3_selector_eval/models \
    --validation-dataset results/.../validation.csv \
    --test-dataset results/.../test.csv \
    --sanity-dataset results/.../sanity.csv \
    --output results/phase2a3_selector_eval/evaluation

Metrics
-------
For each (model, split):
  - classification accuracy, macro F1, weighted F1
  - selected_mean_weighted_goodput
  - regret_to_window_best  (= oracle_per_window - selector)
  - best_fixed_policy on split
  - difference_vs_best_fixed  (= selector - best_fixed)
  - fraction_beating_best_fixed
  - confusion matrix
  - per-regime breakdown if trace_id column present
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llmserveopt.selector.candidates import SELECTOR_CANDIDATES
from llmserveopt.selector.features import FEATURE_NAMES
from llmserveopt.selector.models import (
    DecisionTreeSelector,
    RandomForestSelector,
    RuleBasedSelector,
    evaluate_selector,
)


def load_dataset(path: str) -> list:
    if not Path(path).exists():
        return []
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


def load_model(model_dir: Path):
    """Load a trained selector model from a directory."""
    model_file = model_dir / "model.joblib"
    if not model_file.exists():
        return None
    model_name = model_dir.name
    try:
        import joblib
        clf = joblib.load(model_file)
    except Exception:
        return None

    if "decision_tree" in model_name:
        obj = DecisionTreeSelector.__new__(DecisionTreeSelector)
        obj._clf = clf
        return obj
    elif "random_forest" in model_name:
        obj = RandomForestSelector.__new__(RandomForestSelector)
        obj._clf = clf
        return obj
    return None


def compute_reward_metrics(predictions: List[str], rows: List[dict]) -> Dict:
    """Compute policy-selection reward metrics."""
    selected_wg = []
    best_wg = []
    best_fixed_candidates = {n: [] for n in SELECTOR_CANDIDATES}

    for pred, row in zip(predictions, rows):
        wg_col = f"reward_{pred}"
        sel = row.get(wg_col, float("nan"))
        best = row.get("best_weighted_goodput", float("nan"))
        if isinstance(sel, str):
            sel = float("nan")
        if isinstance(best, str):
            best = float("nan")
        if not math.isnan(sel):
            selected_wg.append(float(sel))
        if not math.isnan(best):
            best_wg.append(float(best))
        # For best-fixed computation
        for pname in SELECTOR_CANDIDATES:
            v = row.get(f"reward_{pname}", float("nan"))
            if isinstance(v, str):
                v = float("nan")
            if not math.isnan(v):
                best_fixed_candidates[pname].append(float(v))

    # Best-fixed policy: single policy with highest mean WG across all windows
    fixed_means = {
        n: float(np.mean(vs)) for n, vs in best_fixed_candidates.items() if vs
    }
    best_fixed_policy = max(fixed_means, key=lambda k: fixed_means[k]) if fixed_means else ""
    best_fixed_wg = fixed_means.get(best_fixed_policy, float("nan"))

    mean_sel = float(np.mean(selected_wg)) if selected_wg else float("nan")
    mean_best = float(np.mean(best_wg)) if best_wg else float("nan")

    diff_vs_fixed = mean_sel - best_fixed_wg if not (math.isnan(mean_sel) or math.isnan(best_fixed_wg)) else float("nan")

    # Fraction of windows where selector beats best-fixed
    beating = 0
    total_comp = 0
    for pred, row in zip(predictions, rows):
        sel = row.get(f"reward_{pred}", float("nan"))
        fixed = row.get(f"reward_{best_fixed_policy}", float("nan"))
        if isinstance(sel, str):
            sel = float("nan")
        if isinstance(fixed, str):
            fixed = float("nan")
        if not (math.isnan(sel) or math.isnan(fixed)):
            total_comp += 1
            if float(sel) > float(fixed):
                beating += 1

    frac_beating = beating / total_comp if total_comp > 0 else float("nan")

    return {
        "selected_mean_wg": mean_sel,
        "best_per_window_mean_wg": mean_best,
        "regret_to_window_best": mean_best - mean_sel if not (math.isnan(mean_best) or math.isnan(mean_sel)) else float("nan"),
        "best_fixed_policy": best_fixed_policy,
        "best_fixed_mean_wg": best_fixed_wg,
        "difference_vs_best_fixed": diff_vs_fixed,
        "fraction_beating_best_fixed": frac_beating,
        "policy_mean_wg": {k: round(v, 4) for k, v in sorted(fixed_means.items(), key=lambda x: -x[1])},
    }


def compute_classification_metrics(predictions: List[str], rows: List[dict]) -> Dict:
    labels = [str(r["best_policy"]) for r in rows]
    n = len(labels)
    correct = sum(p == l for p, l in zip(predictions, labels))
    acc = correct / n if n > 0 else 0.0

    # Macro F1 (simple)
    classes = sorted(set(labels + predictions))
    precisions, recalls, f1s = [], [], []
    for cls in classes:
        tp = sum(1 for p, l in zip(predictions, labels) if p == cls and l == cls)
        fp = sum(1 for p, l in zip(predictions, labels) if p == cls and l != cls)
        fn = sum(1 for p, l in zip(predictions, labels) if p != cls and l == cls)
        p_score = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r_score = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p_score * r_score / (p_score + r_score) if (p_score + r_score) > 0 else 0.0
        precisions.append(p_score)
        recalls.append(r_score)
        f1s.append(f1)

    macro_f1 = float(np.mean(f1s)) if f1s else float("nan")

    # Weighted F1
    class_counts = {cls: labels.count(cls) for cls in classes}
    weighted_f1 = sum(f1 * class_counts[cls] / n for f1, cls in zip(f1s, classes)) if n > 0 else float("nan")

    # Confusion
    confusion = defaultdict(lambda: defaultdict(int))
    for p, l in zip(predictions, labels):
        confusion[p][l] += 1

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "n": n,
        "n_correct": correct,
        "confusion": {k: dict(v) for k, v in confusion.items()},
    }


def per_regime_breakdown(predictions: List[str], rows: List[dict]) -> Dict:
    """Group by trace_id and compute metrics per regime."""
    by_regime: Dict[str, list] = defaultdict(list)
    pred_by_regime: Dict[str, list] = defaultdict(list)
    for pred, row in zip(predictions, rows):
        tid = str(row.get("trace_id", "unknown"))
        by_regime[tid].append(row)
        pred_by_regime[tid].append(pred)

    result = {}
    for tid, regime_rows in by_regime.items():
        regime_preds = pred_by_regime[tid]
        n = len(regime_rows)
        correct = sum(p == str(r["best_policy"]) for p, r in zip(regime_preds, regime_rows))
        sel_wgs = [float(r.get(f"reward_{p}", float("nan"))) for p, r in zip(regime_preds, regime_rows)
                   if not math.isnan(float(r.get(f"reward_{p}", float("nan")) or float("nan")))]
        result[tid] = {
            "n": n,
            "accuracy": correct / n if n > 0 else 0.0,
            "selected_mean_wg": float(np.mean(sel_wgs)) if sel_wgs else float("nan"),
        }
    return result


def save_csv_table(data: list, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not data:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0].keys()))
        writer.writeheader()
        writer.writerows(data)


def evaluate_on_split(
    selector,
    rows: List[dict],
    split_name: str,
    out_dir: Path,
) -> Dict:
    if not rows:
        return {}

    preds = selector.predict(rows)
    cls_metrics = compute_classification_metrics(preds, rows)
    reward_metrics = compute_reward_metrics(preds, rows)
    regime_breakdown = per_regime_breakdown(preds, rows)

    result = {
        "split": split_name,
        **cls_metrics,
        **reward_metrics,
        "per_regime": regime_breakdown,
    }

    # Save confusion matrix
    confusion_path = out_dir / f"confusion_{split_name}.csv"
    all_classes = sorted(set(
        list(cls_metrics["confusion"].keys()) +
        [k for v in cls_metrics["confusion"].values() for k in v.keys()]
    ))
    with open(confusion_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["predicted \\ actual"] + all_classes)
        for pred_cls in all_classes:
            row_data = [pred_cls] + [cls_metrics["confusion"].get(pred_cls, {}).get(ac, 0) for ac in all_classes]
            writer.writerow(row_data)

    return result


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate trained policy selector models")
    p.add_argument("--models-dir", required=True, help="Directory with trained model subdirs")
    p.add_argument("--validation-dataset", default=None)
    p.add_argument("--test-dataset", default=None)
    p.add_argument("--sanity-dataset", default=None)
    p.add_argument("--output", required=True, help="Output directory for evaluation results")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    models_dir = Path(args.models_dir)
    out = Path(args.output)

    splits = {}
    if args.validation_dataset and Path(args.validation_dataset).exists():
        splits["validation"] = load_dataset(args.validation_dataset)
    if args.test_dataset and Path(args.test_dataset).exists():
        splits["test"] = load_dataset(args.test_dataset)
    if args.sanity_dataset and Path(args.sanity_dataset).exists():
        splits["sanity"] = load_dataset(args.sanity_dataset)

    if not splits:
        print("ERROR: No datasets found. Provide at least one of --validation-dataset, --test-dataset, --sanity-dataset")
        return 1

    # Discover models
    selectors = {"rule_based": RuleBasedSelector()}
    for subdir in sorted(models_dir.iterdir()):
        if subdir.is_dir():
            model = load_model(subdir)
            if model is not None:
                selectors[subdir.name] = model
                print(f"Loaded model: {subdir.name}")

    print(f"\nEvaluating {len(selectors)} selectors on {list(splits.keys())} splits")

    all_eval = {}

    for model_name, selector in selectors.items():
        model_out = out / model_name
        model_out.mkdir(parents=True, exist_ok=True)
        model_results = {}

        for split_name, rows in splits.items():
            print(f"  [{model_name}] {split_name}: n={len(rows)}...")
            split_result = evaluate_on_split(selector, rows, split_name, model_out)
            model_results[split_name] = split_result

            print(f"    accuracy={split_result.get('accuracy', float('nan')):.3f}"
                  f"  macro_f1={split_result.get('macro_f1', float('nan')):.3f}"
                  f"  sel_wg={split_result.get('selected_mean_wg', float('nan')):.4f}"
                  f"  vs_fixed={split_result.get('difference_vs_best_fixed', float('nan')):.4f}")

        # Save model results
        with open(model_out / "evaluation.json", "w") as f:
            json.dump(model_results, f, indent=2, default=str)
        all_eval[model_name] = model_results

    # Summary table
    summary_rows = []
    for model_name, model_results in all_eval.items():
        for split_name, split_res in model_results.items():
            summary_rows.append({
                "model": model_name,
                "split": split_name,
                "n": split_res.get("n", 0),
                "accuracy": round(split_res.get("accuracy", float("nan")), 4),
                "macro_f1": round(split_res.get("macro_f1", float("nan")), 4),
                "selected_mean_wg": round(split_res.get("selected_mean_wg", float("nan")), 4),
                "regret_to_window_best": round(split_res.get("regret_to_window_best", float("nan")), 4),
                "best_fixed_policy": split_res.get("best_fixed_policy", ""),
                "best_fixed_mean_wg": round(split_res.get("best_fixed_mean_wg", float("nan")), 4),
                "diff_vs_best_fixed": round(split_res.get("difference_vs_best_fixed", float("nan")), 4),
                "frac_beat_fixed": round(split_res.get("fraction_beating_best_fixed", float("nan")), 4),
            })

    save_csv_table(summary_rows, str(out / "summary.csv"))
    with open(out / "evaluation_full.json", "w") as f:
        json.dump(all_eval, f, indent=2, default=str)

    # Print summary table
    print(f"\n{'='*90}")
    print("EVALUATION SUMMARY")
    print(f"{'='*90}")
    print(f"{'Model':30s} {'Split':12s} {'n':>4} {'Acc':>6} {'F1':>6} {'SelWG':>8} {'VsFixed':>9} {'BeatFixed':>10}")
    print("-" * 90)
    for row in summary_rows:
        print(f"  {row['model']:28s} {row['split']:12s} {row['n']:>4} "
              f"{row['accuracy']:>6.3f} {row['macro_f1']:>6.3f} "
              f"{row['selected_mean_wg']:>8.4f} {row['diff_vs_best_fixed']:>9.4f} "
              f"{row['frac_beat_fixed']:>10.3f}")
    print(f"{'='*90}")

    print(f"\nResults saved to: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
