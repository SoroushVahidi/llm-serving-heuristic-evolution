#!/usr/bin/env python3
"""Overnight module-credit model search on real Wulver intervention data.

This driver is intentionally local and resumable. It never launches Wulver
simulations, never edits imported artifacts, and never uses synthetic fixtures
for model selection.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import pickle
import random
import sys
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sklearn.calibration import calibration_curve
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier, XGBRegressor
except Exception:  # pragma: no cover - optional
    XGBClassifier = None
    XGBRegressor = None

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except Exception:  # pragma: no cover - optional
    LGBMClassifier = None
    LGBMRegressor = None

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover - optional
    torch = None
    nn = None
    F = None


RUN_REAL = ROOT / "scripts" / "run_real_module_credit_evaluation.py"
spec = importlib.util.spec_from_file_location("real_module_credit_adapter", RUN_REAL)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot import adapter from {RUN_REAL}")
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


TARGETS = ("C_base", "C_parent", "C_env")
MEANINGFUL_THRESHOLDS = (0.001, 0.005, 0.010)
PRIMARY_OBJECTIVE_FIELDS = (
    "top1_mean_C_parent",
    "top1_mean_C_env",
    "top1_positive_precision",
    "top1_beats_both_parents_fraction",
    "top1_expands_envelope_fraction",
    "top1_regret_to_oracle",
)


@dataclass
class DatasetBundle:
    rows: list[dict[str, Any]]
    split_rows: dict[str, list[dict[str, Any]]]
    tables: dict[str, pd.DataFrame]
    validation: dict[str, Any]
    split_summary: dict[str, Any]


class JsonlLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def event(self, kind: str, **payload: Any) -> None:
        record = {"ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "kind": kind, **payload}
        with self.path.open("a") as handle:
            handle.write(json.dumps(record, default=json_default, sort_keys=True) + "\n")
        print(json.dumps(record, default=json_default, sort_keys=True), flush=True)


class TrialStore:
    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        self.trial_csv = out_dir / "trial_results.csv"
        self.leaderboard_csv = out_dir / "leaderboard.csv"
        self.trial_jsonl = out_dir / "trial_results.jsonl"
        self.completed_ids = self._read_completed()

    def _read_completed(self) -> set[int]:
        if not self.trial_csv.exists():
            return set()
        try:
            df = pd.read_csv(self.trial_csv)
        except pd.errors.EmptyDataError:
            return set()
        if "trial_id" not in df:
            return set()
        return {int(v) for v in df["trial_id"].dropna().tolist()}

    def append(self, row: Mapping[str, Any]) -> None:
        flat = flatten(row)
        with self.trial_jsonl.open("a") as handle:
            handle.write(json.dumps(flat, default=json_default, sort_keys=True) + "\n")
        records = []
        if self.trial_csv.exists():
            try:
                records = pd.read_csv(self.trial_csv).to_dict(orient="records")
            except pd.errors.EmptyDataError:
                records = []
        records.append(flat)
        pd.DataFrame(records).to_csv(self.trial_csv, index=False)
        self.completed_ids.add(int(row["trial_id"]))
        self.write_leaderboard()

    def write_leaderboard(self) -> None:
        if not self.trial_csv.exists():
            return
        df = pd.read_csv(self.trial_csv)
        if df.empty or "objective_value" not in df:
            return
        df = df.sort_values(["objective_value", "validation.top1_mean_C_base"], ascending=[False, False])
        df.to_csv(self.leaderboard_csv, index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", default="results/wulver_imports/module_intervention_credit_20260721T224322Z")
    parser.add_argument("--output-root", default="results/module_credit_overnight")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--max-hours", type=float, default=8.5)
    parser.add_argument("--final-reserve-min", type=float, default=25.0)
    parser.add_argument("--checkpoint-interval-trials", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--max-trials", type=int, default=10_000)
    parser.add_argument("--resume-dir", default="")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    started = time.time()
    output_root = resolve_path(args.output_root)
    if args.resume_dir:
        out_dir = resolve_path(args.resume_dir)
    else:
        run_name = args.run_name or time.strftime("module-credit-overnight-%Y%m%dT%H%M%S", time.localtime())
        out_dir = output_root / run_name
    for sub in ("logs", "checkpoints", "models", "intermediate"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    logger = JsonlLogger(out_dir / "logs" / "events.jsonl")
    store = TrialStore(out_dir)
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    if torch is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    config = {
        "artifact_root": str(resolve_path(args.artifact_root)),
        "output_dir": str(out_dir),
        "max_hours": args.max_hours,
        "final_reserve_min": args.final_reserve_min,
        "seed": args.seed,
        "max_trials": args.max_trials,
        "branch": run_cmd(["git", "branch", "--show-current"], cwd=ROOT),
        "git_commit": run_cmd(["git", "rev-parse", "HEAD"], cwd=ROOT),
        "cuda_available": bool(torch is not None and torch.cuda.is_available()),
        "cuda_device": torch.cuda.get_device_name(0) if torch is not None and torch.cuda.is_available() else None,
        "optional_models": {
            "xgboost": XGBClassifier is not None,
            "lightgbm": LGBMClassifier is not None,
            "torch": torch is not None,
        },
    }
    write_json(out_dir / "config.json", config)
    logger.event("startup", config=config, pid=os.getpid())

    deadline = started + args.max_hours * 3600.0
    search_deadline = max(started, deadline - args.final_reserve_min * 60.0)
    if args.smoke:
        search_deadline = min(search_deadline, started + 180.0)
        deadline = min(deadline, started + 240.0)

    bundle = load_real_dataset(resolve_path(args.artifact_root), args.seed)
    write_json(out_dir / "intermediate" / "split_summary.json", bundle.split_summary)
    write_json(out_dir / "intermediate" / "artifact_validation.json", bundle.validation)
    diagnosis = diagnose_failures(bundle)
    write_json(out_dir / "intermediate" / "failure_diagnosis.json", diagnosis)
    logger.event(
        "dataset_loaded",
        n_rows=len(bundle.rows),
        split_counts={k: len(v) for k, v in bundle.split_rows.items()},
        core_complete=bundle.validation.get("core_complete"),
    )

    feature_sets = {
        "base": make_feature_matrices(bundle, "base"),
        "interaction": make_feature_matrices(bundle, "interaction"),
        "interaction_regime": make_feature_matrices(bundle, "interaction_regime"),
    }
    y = {
        split: {
            target: np.asarray([float(r[target]) for r in rows], dtype=np.float32)
            for target in TARGETS
        }
        for split, rows in bundle.split_rows.items()
    }
    labels = build_labels(bundle)
    candidates = build_trial_plan(config)
    logger.event("trial_plan_ready", n_trial_templates=len(candidates), templates=[c["kind"] for c in candidates])

    next_trial_id = 0
    while next_trial_id in store.completed_ids:
        next_trial_id += 1

    try:
        trial_count = 0
        while time.time() < search_deadline and trial_count < args.max_trials:
            trial_id = next_trial_id
            next_trial_id += 1
            if trial_id in store.completed_ids:
                continue
            template = candidates[trial_id % len(candidates)]
            trial = sample_trial(template, rng, trial_id)
            heartbeat(out_dir, "running_trial", trial_id=trial_id, kind=trial["kind"], search_deadline=search_deadline)
            logger.event("trial_start", trial=trial)
            try:
                result = run_trial(trial, bundle, feature_sets, y, labels, out_dir, logger)
                result["trial_id"] = trial_id
                result["params"] = trial
                result["elapsed_s"] = time.time() - started
                store.append(result)
                save_rng_state(out_dir / "checkpoints" / "rng_state_latest.pkl", rng)
                heartbeat(out_dir, "trial_complete", trial_id=trial_id, objective=result.get("objective_value"))
                logger.event("trial_complete", trial_id=trial_id, objective=result.get("objective_value"), validation=result.get("validation", {}))
            except Exception as exc:
                err = {
                    "trial_id": trial_id,
                    "params": trial,
                    "status": "ERROR",
                    "objective_value": -1e9,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(limit=12),
                }
                store.append(err)
                heartbeat(out_dir, "trial_error", trial_id=trial_id, error=repr(exc))
                logger.event("trial_error", **err)
            trial_count += 1
            if args.smoke and trial_count >= 2:
                break
    finally:
        logger.event("search_loop_exit", elapsed_s=time.time() - started, remaining_to_deadline_s=deadline - time.time())

    heartbeat(out_dir, "final_evaluation")
    final = final_evaluation(bundle, feature_sets, y, labels, out_dir, logger, max_models=8)
    final.update({
        "runtime_s": time.time() - started,
        "deadline_reached": time.time() >= deadline,
        "MODULE_CREDIT_PREVIOUS_STATUS": "WEAK_GENERALIZATION",
        "STRUCTURAL_SYNTHESIS_PREVIOUS_READINESS": "NOT_READY",
    })
    verdict_status, readiness = verdict(final)
    final["OVERNIGHT_MODULE_CREDIT_STATUS"] = verdict_status
    final["STRUCTURAL_SYNTHESIS_READINESS"] = readiness
    write_json(out_dir / "final_results.json", final)
    (out_dir / "final_report.md").write_text(render_final_report(final, diagnosis))
    heartbeat(out_dir, "complete", status=verdict_status, readiness=readiness)
    logger.event("complete", status=verdict_status, readiness=readiness, output_dir=str(out_dir))
    return 0


def load_real_dataset(artifact_root: Path, seed: int) -> DatasetBundle:
    tables = adapter.load_required_tables(artifact_root)
    validation = adapter.validate_artifacts(artifact_root, tables)
    raw_rows, split_summary = adapter.build_raw_rows(tables, seed=seed)
    suitability_prior = adapter.fit_suitability_prior(seed)
    rows = adapter.build_intervention_dataset(raw_rows, suitability_model=suitability_prior, suitability_lambda=0.5)
    split_rows = {
        "TRAIN": [r for r in rows if r["split"] == "TRAIN"],
        "VALIDATION": [r for r in rows if r["split"] == "VALIDATION"],
        "TEST": [r for r in rows if r["split"] == "TEST"],
    }
    return DatasetBundle(rows=rows, split_rows=split_rows, tables=tables, validation=validation, split_summary=split_summary)


def diagnose_failures(bundle: DatasetBundle) -> dict[str, Any]:
    train = bundle.split_rows["TRAIN"]
    val = bundle.split_rows["VALIDATION"]
    test = bundle.split_rows["TEST"]
    out: dict[str, Any] = {
        "class_balance": {},
        "breakdowns": {},
        "failure_hypotheses": {},
    }
    for split_name, rows in bundle.split_rows.items():
        out["class_balance"][split_name] = class_balance(rows)
    for field in ["module_type", "donor_policy", "base_policy", "trace_family"]:
        out["breakdowns"][field] = {
            "train": breakdown(train, field),
            "validation": breakdown(val, field),
            "test": breakdown(test, field),
        }
    for target in TARGETS:
        out["breakdowns"][f"{target}_magnitude"] = magnitude_breakdown(test, target)
    out["failure_hypotheses"] = infer_failure_hypotheses(bundle, out)
    return out


def class_balance(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    out = {"n": len(rows)}
    for target in TARGETS:
        vals = np.asarray([float(r[target]) for r in rows])
        out[f"{target}_positive_rate"] = float(np.mean(vals > 0.0))
        out[f"{target}_meaningful_001_rate"] = float(np.mean(vals > 0.001))
        out[f"{target}_meaningful_005_rate"] = float(np.mean(vals > 0.005))
        out[f"{target}_meaningful_010_rate"] = float(np.mean(vals > 0.010))
        out[f"{target}_zero_rate"] = float(np.mean(vals == 0.0))
        out[f"{target}_mean"] = float(vals.mean())
        out[f"{target}_std"] = float(vals.std())
    return out


def breakdown(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, Any]:
    out = {}
    for value in sorted({str(r.get(field, "unknown")) for r in rows}):
        sub = [r for r in rows if str(r.get(field, "unknown")) == value]
        out[value] = class_balance(sub)
    return out


def magnitude_breakdown(rows: Sequence[Mapping[str, Any]], target: str) -> dict[str, Any]:
    vals = np.asarray([abs(float(r[target])) for r in rows])
    bins = [0.0, 1e-12, 0.001, 0.005, 0.010, 0.025, 1.0]
    labels = ["zero", "tiny", "lt_001", "001_005", "005_010", "gt_010"]
    out = {}
    for lo, hi, label in zip(bins[:-1], bins[1:], labels):
        mask = (vals >= lo) & (vals < hi)
        out[label] = int(mask.sum())
    return out


def infer_failure_hypotheses(bundle: DatasetBundle, diagnosis: Mapping[str, Any]) -> dict[str, Any]:
    train_balance = diagnosis["class_balance"]["TRAIN"]
    val_balance = diagnosis["class_balance"]["VALIDATION"]
    n_states = len({r["state_id"] for r in bundle.rows})
    return {
        "class_imbalance_likely": train_balance["C_base_positive_rate"] < 0.15 or train_balance["C_base_meaningful_005_rate"] < 0.05,
        "envelope_signal_sparse": train_balance["C_env_positive_rate"] < 0.05,
        "parent_beating_signal_sparse": train_balance["C_parent_positive_rate"] < 0.05,
        "state_sample_size_small": n_states <= 150,
        "validation_distribution_shift_possible": abs(train_balance["C_base_positive_rate"] - val_balance["C_base_positive_rate"]) > 0.03,
        "module_type_imbalance": len({r["module_type"] for r in bundle.rows}) > 1 and max(Counter(r["module_type"] for r in bundle.rows).values()) / len(bundle.rows) > 0.45,
    }


def make_feature_matrices(bundle: DatasetBundle, feature_mode: str) -> dict[str, Any]:
    rows_by_split = bundle.split_rows
    train_dicts = [row_features(r, feature_mode) for r in rows_by_split["TRAIN"]]
    vectorizer = DictVectorizer(sparse=False)
    x_train = vectorizer.fit_transform(train_dicts).astype(np.float32)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    out = {"vectorizer": vectorizer, "scaler": scaler, "feature_mode": feature_mode, "feature_names": vectorizer.feature_names_}
    out["TRAIN"] = x_train
    for split in ("VALIDATION", "TEST"):
        x = vectorizer.transform([row_features(r, feature_mode) for r in rows_by_split[split]]).astype(np.float32)
        out[split] = scaler.transform(x).astype(np.float32)
    return out


def row_features(row: Mapping[str, Any], mode: str) -> dict[str, float]:
    feats: dict[str, float] = {}
    feats.update(prefix_dict("state", row.get("state_features", {})))
    feats.update(prefix_dict("donor_module", row.get("donor_module_representation", {})))
    feats.update(prefix_dict("base_module", row.get("base_module_representation", {})))
    feats.update(prefix_dict("compat", row.get("compatibility_metadata", {})))
    for key in [
        "donor_predicted_reward", "donor_uncertainty", "donor_conservative_suitability",
        "base_predicted_reward", "base_uncertainty", "base_conservative_suitability",
        "predicted_donor_vs_base_advantage",
    ]:
        feats[f"suitability_{key}"] = float(row.get(key, 0.0))
    feats[f"module_type={row['module_type']}"] = 1.0
    feats[f"donor_policy={row['donor_policy']}"] = 1.0
    feats[f"base_policy={row['base_policy']}"] = 1.0
    feats[f"trace_family={row.get('trace_family', 'unknown')}"] = 1.0
    if mode in {"interaction", "interaction_regime"}:
        donor = numeric_map(row.get("donor_module_representation", {}))
        base = numeric_map(row.get("base_module_representation", {}))
        for key in sorted(set(donor) | set(base)):
            d = donor.get(key, 0.0)
            b = base.get(key, 0.0)
            feats[f"module_delta_{key}"] = d - b
            feats[f"module_abs_delta_{key}"] = abs(d - b)
            feats[f"module_product_{key}"] = d * b
        feats["suitability_donor_minus_base_conservative"] = (
            float(row.get("donor_conservative_suitability", 0.0)) - float(row.get("base_conservative_suitability", 0.0))
        )
        feats["suitability_uncertainty_sum"] = float(row.get("donor_uncertainty", 0.0)) + float(row.get("base_uncertainty", 0.0))
        compat = row.get("compatibility_metadata", {})
        feats["compat_distance_x_donor_advantage"] = float(compat.get("structural_distance", 0.0)) * float(row.get("predicted_donor_vs_base_advantage", 0.0)) if isinstance(compat, Mapping) else 0.0
    if mode == "interaction_regime":
        module_type = str(row["module_type"])
        for key, value in row.get("state_features", {}).items():
            try:
                v = float(value)
            except (TypeError, ValueError):
                continue
            feats[f"state_x_module_{module_type}_{key}"] = v
    return feats


def prefix_dict(prefix: str, payload: Any) -> dict[str, float]:
    out = {}
    if not isinstance(payload, Mapping):
        return out
    for key, value in payload.items():
        try:
            f = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            out[f"{prefix}_{key}"] = f
    return out


def numeric_map(payload: Any) -> dict[str, float]:
    return {k.rsplit("_", 1)[-1] if k.startswith("module_name_hash_") else k: v for k, v in prefix_dict("", payload).items()}


def build_labels(bundle: DatasetBundle) -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {}
    for split, rows in bundle.split_rows.items():
        out[split] = {}
        for target in TARGETS:
            vals = np.asarray([float(r[target]) for r in rows], dtype=np.float32)
            out[split][f"{target}>0"] = (vals > 0.0).astype(np.float32)
            for threshold in MEANINGFUL_THRESHOLDS:
                out[split][f"{target}>{threshold:g}"] = (vals > threshold).astype(np.float32)
    return out


def build_trial_plan(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    plan = []
    if torch is not None:
        plan.extend([
            {"kind": "torch_mlp_multitask", "task": "neural", "target": "C_base"},
            {"kind": "torch_siamese_multitask", "task": "neural", "target": "C_base"},
        ])
    plan.extend([
        {"kind": "extra_trees_reg", "task": "regression", "target": "C_base"},
        {"kind": "rf_reg", "task": "regression", "target": "C_base"},
        {"kind": "hgb_reg", "task": "regression", "target": "C_base"},
        {"kind": "extra_trees_clf", "task": "classification", "label": "C_base>0"},
        {"kind": "rf_clf", "task": "classification", "label": "C_base>0"},
        {"kind": "hgb_clf", "task": "classification", "label": "C_base>0"},
        {"kind": "pairwise_logistic", "task": "ranking", "target": "C_base"},
    ])
    for t in MEANINGFUL_THRESHOLDS:
        plan.append({"kind": "extra_trees_clf", "task": "classification", "label": f"C_base>{t:g}"})
        plan.append({"kind": "hgb_clf", "task": "classification", "label": f"C_base>{t:g}"})
    for target in ("C_parent", "C_env"):
        plan.append({"kind": "extra_trees_reg", "task": "regression", "target": target})
        plan.append({"kind": "extra_trees_clf", "task": "classification", "label": f"{target}>0"})
    if XGBClassifier is not None:
        plan.extend([
            {"kind": "xgb_reg", "task": "regression", "target": "C_base"},
            {"kind": "xgb_clf", "task": "classification", "label": "C_base>0"},
        ])
    if LGBMClassifier is not None:
        plan.extend([
            {"kind": "lgbm_reg", "task": "regression", "target": "C_base"},
            {"kind": "lgbm_clf", "task": "classification", "label": "C_base>0"},
        ])
    return plan


def sample_trial(template: Mapping[str, Any], rng: np.random.Generator, trial_id: int) -> dict[str, Any]:
    trial = dict(template)
    trial["trial_id"] = trial_id
    trial["seed"] = int(rng.integers(1, 2**31 - 1))
    trial["feature_mode"] = rng.choice(["base", "interaction", "interaction_regime"], p=[0.2, 0.5, 0.3]).item()
    trial["class_weight"] = rng.choice(["balanced", "none"], p=[0.7, 0.3]).item()
    trial["n_estimators"] = int(rng.choice([120, 200, 320, 500]))
    trial["max_depth"] = None if rng.random() < 0.3 else int(rng.choice([3, 5, 7, 10, 14]))
    trial["min_samples_leaf"] = int(rng.choice([1, 2, 4, 8, 16]))
    trial["learning_rate"] = float(rng.choice([0.01, 0.03, 0.05, 0.08, 0.12]))
    trial["l2"] = float(rng.choice([1e-5, 1e-4, 1e-3, 1e-2]))
    trial["dropout"] = float(rng.choice([0.05, 0.1, 0.2, 0.35]))
    trial["hidden"] = int(rng.choice([64, 96, 128, 192, 256]))
    trial["batch_size"] = int(rng.choice([64, 128, 256]))
    trial["calibration"] = rng.choice(["none", "platt", "isotonic"], p=[0.2, 0.45, 0.35]).item()
    trial["sampling"] = rng.choice(["natural", "positive_oversample"], p=[0.65, 0.35]).item()
    return trial


def run_trial(
    trial: Mapping[str, Any],
    bundle: DatasetBundle,
    feature_sets: Mapping[str, Any],
    y: Mapping[str, Mapping[str, np.ndarray]],
    labels: Mapping[str, Mapping[str, np.ndarray]],
    out_dir: Path,
    logger: JsonlLogger,
) -> dict[str, Any]:
    kind = str(trial["kind"])
    task = str(trial["task"])
    feature_mode = str(trial["feature_mode"])
    features = feature_sets[feature_mode]
    if task == "regression":
        model, pred_val, pred_test, uncertainty_val, uncertainty_test = fit_regression_trial(kind, trial, features, y, out_dir)
        score_val = pred_val
        score_test = pred_test
        metrics_val = evaluate_rank_scores(bundle.split_rows["VALIDATION"], score_val, uncertainty_val)
        metrics_test_preview = evaluate_rank_scores(bundle.split_rows["TEST"], score_test, uncertainty_test)
        target = str(trial["target"])
        metrics_val.update(regression_metrics(y["VALIDATION"][target], pred_val))
        model_path = save_model(out_dir, trial, model, extras={"features": features, "task": "regression"})
        decision = tune_conservative_rule(bundle.split_rows["VALIDATION"], score_val, pred_val, uncertainty_val)
    elif task == "classification":
        model, prob_val, prob_test, uncertainty_val, uncertainty_test, cal = fit_classification_trial(kind, trial, features, labels, out_dir)
        score_val = prob_val - 0.5 * uncertainty_val
        score_test = prob_test - 0.5 * uncertainty_test
        metrics_val = evaluate_rank_scores(bundle.split_rows["VALIDATION"], score_val, uncertainty_val)
        metrics_test_preview = evaluate_rank_scores(bundle.split_rows["TEST"], score_test, uncertainty_test)
        label = str(trial["label"])
        metrics_val.update(classification_metrics(labels["VALIDATION"][label], prob_val))
        metrics_val.update(calibration_metrics(labels["VALIDATION"][label], prob_val))
        model_path = save_model(out_dir, trial, model, extras={"features": features, "calibration": cal, "task": "classification"})
        decision = tune_conservative_rule(bundle.split_rows["VALIDATION"], score_val, prob_val, uncertainty_val, probability_mode=True)
    elif task == "ranking":
        model, score_val, score_test, uncertainty_val, uncertainty_test = fit_pairwise_trial(trial, bundle, features, out_dir)
        metrics_val = evaluate_rank_scores(bundle.split_rows["VALIDATION"], score_val, uncertainty_val)
        metrics_test_preview = evaluate_rank_scores(bundle.split_rows["TEST"], score_test, uncertainty_test)
        model_path = save_model(out_dir, trial, model, extras={"features": features, "task": "ranking"})
        decision = tune_conservative_rule(bundle.split_rows["VALIDATION"], score_val, score_val, uncertainty_val)
    elif task == "neural":
        model, score_val, score_test, uncertainty_val, uncertainty_test = fit_torch_trial(trial, features, y, labels, out_dir, logger)
        metrics_val = evaluate_rank_scores(bundle.split_rows["VALIDATION"], score_val, uncertainty_val)
        metrics_test_preview = evaluate_rank_scores(bundle.split_rows["TEST"], score_test, uncertainty_test)
        metrics_val.update(regression_metrics(y["VALIDATION"]["C_base"], score_val))
        metrics_val.update(calibration_metrics(labels["VALIDATION"]["C_base>0"], sigmoid_np(score_val)))
        model_path = save_torch_model(out_dir, trial, model, extras={"feature_mode": feature_mode})
        decision = tune_conservative_rule(bundle.split_rows["VALIDATION"], score_val, sigmoid_np(score_val), uncertainty_val, probability_mode=True)
    else:
        raise ValueError(f"Unknown task {task}")

    objective = objective_value(metrics_val)
    return {
        "status": "OK",
        "model_path": str(model_path),
        "validation": metrics_val,
        "test_preview_not_for_selection": metrics_test_preview,
        "decision_rule": decision,
        "objective_value": objective,
    }


def fit_regression_trial(kind: str, trial: Mapping[str, Any], features: Mapping[str, Any], y: Mapping[str, Mapping[str, np.ndarray]], out_dir: Path):
    target = str(trial["target"])
    if kind == "rf_reg":
        model = RandomForestRegressor(n_estimators=int(trial["n_estimators"]), max_depth=trial["max_depth"], min_samples_leaf=int(trial["min_samples_leaf"]), random_state=int(trial["seed"]), n_jobs=2)
    elif kind == "extra_trees_reg":
        model = ExtraTreesRegressor(n_estimators=int(trial["n_estimators"]), max_depth=trial["max_depth"], min_samples_leaf=int(trial["min_samples_leaf"]), random_state=int(trial["seed"]), n_jobs=2)
    elif kind == "hgb_reg":
        model = HistGradientBoostingRegressor(max_iter=int(trial["n_estimators"]), max_leaf_nodes=31, learning_rate=float(trial["learning_rate"]), l2_regularization=float(trial["l2"]), random_state=int(trial["seed"]))
    elif kind == "xgb_reg" and XGBRegressor is not None:
        model = XGBRegressor(n_estimators=int(trial["n_estimators"]), max_depth=int(trial["max_depth"] or 5), learning_rate=float(trial["learning_rate"]), subsample=0.85, colsample_bytree=0.85, objective="reg:squarederror", tree_method="hist", random_state=int(trial["seed"]), n_jobs=2)
    elif kind == "lgbm_reg" and LGBMRegressor is not None:
        model = LGBMRegressor(n_estimators=int(trial["n_estimators"]), max_depth=-1 if trial["max_depth"] is None else int(trial["max_depth"]), learning_rate=float(trial["learning_rate"]), random_state=int(trial["seed"]), n_jobs=2, verbose=-1)
    else:
        raise ValueError(f"Unavailable regression kind {kind}")
    model.fit(features["TRAIN"], y["TRAIN"][target])
    pred_val = model.predict(features["VALIDATION"])
    pred_test = model.predict(features["TEST"])
    u_val = model_uncertainty(model, features["VALIDATION"])
    u_test = model_uncertainty(model, features["TEST"])
    return model, pred_val, pred_test, u_val, u_test


def fit_classification_trial(kind: str, trial: Mapping[str, Any], features: Mapping[str, Any], labels: Mapping[str, Mapping[str, np.ndarray]], out_dir: Path):
    label = str(trial["label"])
    y_train = labels["TRAIN"][label].astype(int)
    x_train = features["TRAIN"]
    if str(trial.get("sampling")) == "positive_oversample":
        x_train, y_train = oversample_positive(x_train, y_train, int(trial["seed"]))
    class_weight = "balanced" if trial.get("class_weight") == "balanced" else None
    if len(np.unique(y_train)) < 2:
        raise ValueError(f"Label {label} has one class in training")
    if kind == "rf_clf":
        model = RandomForestClassifier(n_estimators=int(trial["n_estimators"]), max_depth=trial["max_depth"], min_samples_leaf=int(trial["min_samples_leaf"]), class_weight=class_weight, random_state=int(trial["seed"]), n_jobs=2)
    elif kind == "extra_trees_clf":
        model = ExtraTreesClassifier(n_estimators=int(trial["n_estimators"]), max_depth=trial["max_depth"], min_samples_leaf=int(trial["min_samples_leaf"]), class_weight=class_weight, random_state=int(trial["seed"]), n_jobs=2)
    elif kind == "hgb_clf":
        sample_weight = balanced_sample_weight(y_train) if class_weight == "balanced" else None
        model = HistGradientBoostingClassifier(max_iter=int(trial["n_estimators"]), learning_rate=float(trial["learning_rate"]), l2_regularization=float(trial["l2"]), random_state=int(trial["seed"]))
        model.fit(x_train, y_train, sample_weight=sample_weight)
        return finish_classifier(model, trial, features, labels, label)
    elif kind == "xgb_clf" and XGBClassifier is not None:
        pos = max(float(y_train.sum()), 1.0)
        neg = max(float((1 - y_train).sum()), 1.0)
        model = XGBClassifier(n_estimators=int(trial["n_estimators"]), max_depth=int(trial["max_depth"] or 5), learning_rate=float(trial["learning_rate"]), subsample=0.85, colsample_bytree=0.85, eval_metric="logloss", tree_method="hist", scale_pos_weight=(neg / pos if class_weight == "balanced" else 1.0), random_state=int(trial["seed"]), n_jobs=2)
    elif kind == "lgbm_clf" and LGBMClassifier is not None:
        model = LGBMClassifier(n_estimators=int(trial["n_estimators"]), max_depth=-1 if trial["max_depth"] is None else int(trial["max_depth"]), learning_rate=float(trial["learning_rate"]), class_weight=class_weight, random_state=int(trial["seed"]), n_jobs=2, verbose=-1)
    else:
        raise ValueError(f"Unavailable classification kind {kind}")
    model.fit(x_train, y_train)
    return finish_classifier(model, trial, features, labels, label)


def finish_classifier(model: Any, trial: Mapping[str, Any], features: Mapping[str, Any], labels: Mapping[str, Mapping[str, np.ndarray]], label: str):
    raw_val = predict_proba_1(model, features["VALIDATION"])
    raw_test = predict_proba_1(model, features["TEST"])
    cal = fit_calibrator(str(trial.get("calibration", "none")), raw_val, labels["VALIDATION"][label])
    prob_val = apply_calibrator(cal, raw_val)
    prob_test = apply_calibrator(cal, raw_test)
    u_val = classifier_uncertainty(model, features["VALIDATION"])
    u_test = classifier_uncertainty(model, features["TEST"])
    return model, prob_val, prob_test, u_val, u_test, cal


def fit_pairwise_trial(trial: Mapping[str, Any], bundle: DatasetBundle, features: Mapping[str, Any], out_dir: Path):
    x_pair, y_pair = make_pairwise_training(bundle.split_rows["TRAIN"], features["TRAIN"], int(trial["seed"]))
    model = LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0 / max(float(trial["l2"]), 1e-6), random_state=int(trial["seed"]))
    model.fit(x_pair, y_pair)
    score_val = pairwise_scores(model, bundle.split_rows["VALIDATION"], features["VALIDATION"])
    score_test = pairwise_scores(model, bundle.split_rows["TEST"], features["TEST"])
    u_val = np.zeros_like(score_val, dtype=float)
    u_test = np.zeros_like(score_test, dtype=float)
    return model, score_val, score_test, u_val, u_test


class TorchMultiTask(nn.Module):
    def __init__(self, n_in: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.reg = nn.Linear(hidden, 3)
        self.cls = nn.Linear(hidden, 4)

    def forward(self, x):
        h = self.net(x)
        return self.reg(h), self.cls(h)


def fit_torch_trial(trial: Mapping[str, Any], features: Mapping[str, Any], y: Mapping[str, Mapping[str, np.ndarray]], labels: Mapping[str, Mapping[str, np.ndarray]], out_dir: Path, logger: JsonlLogger):
    if torch is None:
        raise ValueError("PyTorch unavailable")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = int(trial["seed"])
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    x_train = torch.tensor(features["TRAIN"], dtype=torch.float32)
    reg_train = torch.tensor(np.vstack([y["TRAIN"][t] for t in TARGETS]).T, dtype=torch.float32)
    cls_train = torch.tensor(np.vstack([
        labels["TRAIN"]["C_base>0"],
        labels["TRAIN"]["C_base>0.001"],
        labels["TRAIN"]["C_parent>0"],
        labels["TRAIN"]["C_env>0"],
    ]).T, dtype=torch.float32)
    dataset = TensorDataset(x_train, reg_train, cls_train)
    loader = DataLoader(dataset, batch_size=int(trial["batch_size"]), shuffle=True, generator=torch.Generator().manual_seed(seed))
    model = TorchMultiTask(features["TRAIN"].shape[1], int(trial["hidden"]), float(trial["dropout"])).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(trial["learning_rate"]), weight_decay=float(trial["l2"]))
    x_val = torch.tensor(features["VALIDATION"], dtype=torch.float32, device=device)
    y_val = y["VALIDATION"]["C_base"]
    best_state = None
    best_loss = float("inf")
    patience = 15
    stale = 0
    max_epochs = 180
    pos_weight = torch.tensor([8.0, 20.0, 20.0, 20.0], device=device)
    for epoch in range(max_epochs):
        model.train()
        total = 0.0
        for xb, yb, cb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            cb = cb.to(device)
            opt.zero_grad(set_to_none=True)
            pred_reg, pred_cls = model(xb)
            reg_loss = F.smooth_l1_loss(pred_reg, yb)
            cls_loss = F.binary_cross_entropy_with_logits(pred_cls, cb, pos_weight=pos_weight)
            loss = reg_loss + 0.35 * cls_loss
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu())
        model.eval()
        with torch.no_grad():
            pred_reg, pred_cls = model(x_val)
            val_score = pred_reg[:, 0].detach().cpu().numpy()
            val_loss = float(mean_absolute_error(y_val, val_score))
        if epoch % 10 == 0:
            logger.event("torch_epoch", trial_id=trial["trial_id"], epoch=epoch, val_mae=val_loss, device=str(device))
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    score_val, u_val = torch_predict(model, features["VALIDATION"], device, mc_dropout=True)
    score_test, u_test = torch_predict(model, features["TEST"], device, mc_dropout=True)
    return model, score_val, score_test, u_val, u_test


def torch_predict(model: Any, x_np: np.ndarray, device: Any, *, mc_dropout: bool = False) -> tuple[np.ndarray, np.ndarray]:
    x = torch.tensor(x_np, dtype=torch.float32, device=device)
    preds = []
    passes = 8 if mc_dropout else 1
    with torch.no_grad():
        for _ in range(passes):
            if mc_dropout:
                model.train()
            else:
                model.eval()
            pred_reg, _pred_cls = model(x)
            preds.append(pred_reg[:, 0].detach().cpu().numpy())
    arr = np.vstack(preds)
    return arr.mean(axis=0), arr.std(axis=0)


def model_uncertainty(model: Any, x: np.ndarray) -> np.ndarray:
    estimators = getattr(model, "estimators_", None)
    if estimators is not None:
        try:
            preds = np.vstack([est.predict(x) for est in estimators])
            return preds.std(axis=0)
        except Exception:
            pass
    return np.zeros(x.shape[0], dtype=float)


def classifier_uncertainty(model: Any, x: np.ndarray) -> np.ndarray:
    estimators = getattr(model, "estimators_", None)
    if estimators is not None:
        try:
            preds = np.vstack([predict_proba_1(est, x) for est in estimators])
            return preds.std(axis=0)
        except Exception:
            pass
    p = predict_proba_1(model, x)
    return np.sqrt(np.maximum(p * (1.0 - p), 0.0))


def predict_proba_1(model: Any, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)
        if proba.shape[1] == 1:
            return np.zeros(x.shape[0], dtype=float) if getattr(model, "classes_", [0])[0] == 0 else np.ones(x.shape[0], dtype=float)
        return proba[:, 1]
    if hasattr(model, "decision_function"):
        return sigmoid_np(model.decision_function(x))
    return sigmoid_np(model.predict(x))


def oversample_positive(x: np.ndarray, y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    if len(pos) == 0 or len(neg) == 0 or len(pos) >= len(neg):
        return x, y
    extra = rng.choice(pos, size=len(neg) - len(pos), replace=True)
    idx = np.concatenate([np.arange(len(y)), extra])
    rng.shuffle(idx)
    return x[idx], y[idx]


def balanced_sample_weight(y: np.ndarray) -> np.ndarray:
    counts = Counter(y.tolist())
    return np.asarray([len(y) / (2.0 * counts[int(v)]) for v in y], dtype=float)


def fit_calibrator(method: str, prob: np.ndarray, y: np.ndarray) -> Any:
    if method == "platt" and len(np.unique(y)) == 2:
        lr = LogisticRegression(max_iter=1000)
        lr.fit(prob.reshape(-1, 1), y.astype(int))
        return {"method": "platt", "model": lr}
    if method == "isotonic" and len(np.unique(y)) == 2:
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(prob, y)
        return {"method": "isotonic", "model": iso}
    return {"method": "none"}


def apply_calibrator(cal: Mapping[str, Any], prob: np.ndarray) -> np.ndarray:
    method = cal.get("method")
    if method == "platt":
        return cal["model"].predict_proba(prob.reshape(-1, 1))[:, 1]
    if method == "isotonic":
        return cal["model"].predict(prob)
    return prob


def make_pairwise_training(rows: Sequence[Mapping[str, Any]], x: np.ndarray, seed: int, max_pairs: int = 50_000) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    groups = group_indices(rows)
    pairs = []
    labels = []
    for indices in groups.values():
        if len(indices) < 2:
            continue
        local_pairs = []
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                a, b = indices[i], indices[j]
                ca = float(rows[a]["C_base"])
                cb = float(rows[b]["C_base"])
                if abs(ca - cb) < 1e-9:
                    continue
                local_pairs.append((a, b, 1 if ca > cb else 0))
        rng.shuffle(local_pairs)
        for a, b, lab in local_pairs[:30]:
            pairs.append(x[a] - x[b])
            labels.append(lab)
            pairs.append(x[b] - x[a])
            labels.append(1 - lab)
    if len(pairs) > max_pairs:
        idx = rng.choice(np.arange(len(pairs)), size=max_pairs, replace=False)
        pairs = [pairs[i] for i in idx]
        labels = [labels[i] for i in idx]
    return np.vstack(pairs), np.asarray(labels, dtype=int)


def pairwise_scores(model: Any, rows: Sequence[Mapping[str, Any]], x: np.ndarray) -> np.ndarray:
    scores = np.zeros(len(rows), dtype=float)
    groups = group_indices(rows)
    for indices in groups.values():
        if len(indices) == 1:
            continue
        wins = np.zeros(len(indices), dtype=float)
        comps = np.zeros(len(indices), dtype=float)
        for i, a in enumerate(indices):
            diffs = []
            others = []
            for j, b in enumerate(indices):
                if a == b:
                    continue
                diffs.append(x[a] - x[b])
                others.append(j)
            p = predict_proba_1(model, np.vstack(diffs))
            wins[i] = float(p.mean())
            comps[i] = len(p)
        for i, a in enumerate(indices):
            scores[a] = wins[i]
    return scores


def group_indices(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], list[int]]:
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        compat = row.get("compatibility_metadata", {})
        if isinstance(compat, Mapping) and float(compat.get("compatible", 1.0)) <= 0:
            continue
        groups[(str(row["state_id"]), str(row["base_policy"]))].append(idx)
    return groups


def evaluate_rank_scores(rows: Sequence[Mapping[str, Any]], scores: np.ndarray, uncertainty: np.ndarray | None = None) -> dict[str, Any]:
    groups = group_indices(rows)
    uncertainty = np.zeros(len(rows), dtype=float) if uncertainty is None else np.asarray(uncertainty)
    out: dict[str, Any] = {"n_groups": len(groups)}
    selected_by_k = {1: [], 3: [], 5: []}
    oracle = []
    regrets = []
    for indices in groups.values():
        ordered = sorted(indices, key=lambda i: float(scores[i]), reverse=True)
        best = max(indices, key=lambda i: float(rows[i]["C_base"]))
        oracle.append(float(rows[best]["C_base"]))
        regrets.append(float(rows[best]["C_base"]) - float(rows[ordered[0]]["C_base"]))
        for k in selected_by_k:
            selected_by_k[k].extend(ordered[: min(k, len(ordered))])
    for k, indices in selected_by_k.items():
        prefix = f"top{k}"
        if not indices:
            continue
        sub = [rows[i] for i in indices]
        c_base = np.asarray([float(r["C_base"]) for r in sub])
        c_parent = np.asarray([float(r["C_parent"]) for r in sub])
        c_env = np.asarray([float(r["C_env"]) for r in sub])
        out[f"{prefix}_n_selected"] = len(indices)
        out[f"{prefix}_mean_C_base"] = float(c_base.mean())
        out[f"{prefix}_mean_C_parent"] = float(c_parent.mean())
        out[f"{prefix}_mean_C_env"] = float(c_env.mean())
        out[f"{prefix}_positive_precision"] = float((c_base > 0).mean())
        out[f"{prefix}_meaningful_001_precision"] = float((c_base > 0.001).mean())
        out[f"{prefix}_meaningful_005_precision"] = float((c_base > 0.005).mean())
        out[f"{prefix}_beats_both_parents_fraction"] = float((c_parent > 0).mean())
        out[f"{prefix}_expands_envelope_fraction"] = float((c_env > 0).mean())
        out[f"{prefix}_negative_transfer_rate"] = float((c_base < 0).mean())
        out[f"{prefix}_mean_uncertainty"] = float(uncertainty[indices].mean()) if len(indices) else 0.0
    out["top1_regret_to_oracle"] = float(np.mean(regrets)) if regrets else None
    out["oracle_mean_C_base"] = float(np.mean(oracle)) if oracle else None
    return out


def regression_metrics(actual: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(actual, pred)),
        "rmse": float(math.sqrt(mean_squared_error(actual, pred))),
        "bias": float(np.mean(pred - actual)),
        "sign_accuracy": float(np.mean(np.sign(pred[actual != 0]) == np.sign(actual[actual != 0]))) if np.any(actual != 0) else 0.0,
    }


def classification_metrics(actual: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    out = {"brier": float(brier_score_loss(actual, prob))}
    try:
        out["roc_auc"] = float(roc_auc_score(actual, prob))
    except Exception:
        out["roc_auc"] = float("nan")
    return out


def calibration_metrics(actual: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> dict[str, float]:
    if len(np.unique(actual)) < 2:
        return {"ece": float("nan")}
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (prob >= lo) & (prob < hi if hi < 1.0 else prob <= hi)
        if not np.any(mask):
            continue
        ece += float(mask.mean()) * abs(float(prob[mask].mean()) - float(actual[mask].mean()))
    out = {"ece": ece}
    for threshold in [0.5, 0.7, 0.8, 0.9]:
        mask = prob >= threshold
        out[f"precision_at_p{int(threshold*100)}"] = float(actual[mask].mean()) if np.any(mask) else None
        out[f"coverage_at_p{int(threshold*100)}"] = float(mask.mean())
    return out


def tune_conservative_rule(rows: Sequence[Mapping[str, Any]], scores: np.ndarray, prob_or_mu: np.ndarray, uncertainty: np.ndarray, *, probability_mode: bool = False) -> dict[str, Any]:
    best = {"objective": -1e9, "threshold": None, "uncertainty_threshold": None, "metrics": {}}
    score_grid = np.quantile(prob_or_mu, np.linspace(0.5, 0.95, 10))
    unc_grid = np.quantile(uncertainty, np.linspace(0.5, 0.95, 6)) if len(uncertainty) else [0.0]
    for threshold in score_grid:
        for unc_threshold in unc_grid:
            accepted = [i for i in range(len(rows)) if prob_or_mu[i] >= threshold and uncertainty[i] <= unc_threshold and is_compatible(rows[i])]
            if not accepted:
                continue
            c_base = np.asarray([float(rows[i]["C_base"]) for i in accepted])
            c_parent = np.asarray([float(rows[i]["C_parent"]) for i in accepted])
            c_env = np.asarray([float(rows[i]["C_env"]) for i in accepted])
            precision = float((c_base > 0).mean())
            mean_parent = float(c_parent.mean())
            mean_env = float(c_env.mean())
            objective = 3.0 * mean_parent + 2.0 * mean_env + precision + 0.1 * float(len(accepted) / len(rows))
            if precision < 0.5:
                objective -= 1.0
            if objective > best["objective"]:
                best = {
                    "objective": objective,
                    "threshold": float(threshold),
                    "uncertainty_threshold": float(unc_threshold),
                    "metrics": {
                        "n_accepted": len(accepted),
                        "coverage": float(len(accepted) / len(rows)),
                        "mean_C_base": float(c_base.mean()),
                        "mean_C_parent": mean_parent,
                        "mean_C_env": mean_env,
                        "positive_precision": precision,
                        "beats_both_parents_fraction": float((c_parent > 0).mean()),
                        "expands_envelope_fraction": float((c_env > 0).mean()),
                        "negative_transfer_rate": float((c_base < 0).mean()),
                    },
                }
    return best


def objective_value(metrics: Mapping[str, Any]) -> float:
    c_parent = float(metrics.get("top1_mean_C_parent", -1.0) or -1.0)
    c_env = float(metrics.get("top1_mean_C_env", -1.0) or -1.0)
    precision = float(metrics.get("top1_positive_precision", 0.0) or 0.0)
    beats = float(metrics.get("top1_beats_both_parents_fraction", 0.0) or 0.0)
    expands = float(metrics.get("top1_expands_envelope_fraction", 0.0) or 0.0)
    regret = float(metrics.get("top1_regret_to_oracle", 1.0) or 1.0)
    return 100.0 * c_parent + 50.0 * c_env + precision + beats + expands - 0.1 * regret


def is_compatible(row: Mapping[str, Any]) -> bool:
    compat = row.get("compatibility_metadata", {})
    return not isinstance(compat, Mapping) or float(compat.get("compatible", 1.0)) > 0.0


def save_model(out_dir: Path, trial: Mapping[str, Any], model: Any, extras: Mapping[str, Any]) -> Path:
    path = out_dir / "models" / f"trial_{int(trial['trial_id']):05d}_{trial['kind']}.joblib"
    joblib.dump({"model": model, "trial": dict(trial), **dict(extras)}, path)
    return path


def save_torch_model(out_dir: Path, trial: Mapping[str, Any], model: Any, extras: Mapping[str, Any]) -> Path:
    path = out_dir / "models" / f"trial_{int(trial['trial_id']):05d}_{trial['kind']}.pt"
    payload = {"state_dict": model.state_dict(), "trial": dict(trial), **dict(extras)}
    torch.save(payload, path)
    return path


def final_evaluation(bundle: DatasetBundle, feature_sets: Mapping[str, Any], y: Mapping[str, Mapping[str, np.ndarray]], labels: Mapping[str, Mapping[str, np.ndarray]], out_dir: Path, logger: JsonlLogger, *, max_models: int = 8) -> dict[str, Any]:
    leaderboard = out_dir / "leaderboard.csv"
    if not leaderboard.exists():
        return {"status": "NO_TRIALS"}
    df = pd.read_csv(leaderboard)
    df = df[df["status"] == "OK"].head(max_models)
    final_rows = []
    for _, row in df.iterrows():
        model_path = Path(str(row["model_path"]))
        if not model_path.exists():
            continue
        trial_id = int(row["trial_id"])
        params = json.loads(row["params"]) if isinstance(row.get("params"), str) and row["params"].startswith("{") else None
        loaded = joblib.load(model_path) if model_path.suffix == ".joblib" else None
        if loaded is None:
            continue
        trial = loaded["trial"]
        features = loaded["features"]
        task = loaded["task"]
        if task == "regression":
            model = loaded["model"]
            score = model.predict(features["TEST"])
            unc = model_uncertainty(model, features["TEST"])
        elif task == "classification":
            model = loaded["model"]
            raw = predict_proba_1(model, features["TEST"])
            score = apply_calibrator(loaded.get("calibration", {"method": "none"}), raw)
            unc = classifier_uncertainty(model, features["TEST"])
        elif task == "ranking":
            model = loaded["model"]
            score = pairwise_scores(model, bundle.split_rows["TEST"], features["TEST"])
            unc = np.zeros_like(score)
        else:
            continue
        metrics = evaluate_rank_scores(bundle.split_rows["TEST"], score, unc)
        decision = apply_saved_decision_rule(bundle.split_rows["TEST"], score, unc, row)
        final_rows.append({"trial_id": trial_id, "kind": trial["kind"], "task": task, "test": metrics, "decision_test": decision, "model_path": str(model_path)})
    final_df = pd.DataFrame([flatten(r) for r in final_rows])
    final_df.to_csv(out_dir / "final_test_evaluation.csv", index=False)
    baselines = decision_baselines(bundle)
    return {"status": "OK", "top_models_test": final_rows, "decision_baselines": baselines}


def apply_saved_decision_rule(rows: Sequence[Mapping[str, Any]], score: np.ndarray, unc: np.ndarray, leaderboard_row: Mapping[str, Any]) -> dict[str, Any]:
    # Conservative fallback for final reporting: choose the best validation-like
    # threshold from score quantiles on the test distribution without using test
    # targets for tuning; this is only applying an already monotone abstention
    # rule shape, not selecting a model.
    threshold = float(np.quantile(score, 0.90))
    unc_threshold = float(np.quantile(unc, 0.75)) if len(unc) else 0.0
    accepted = [i for i in range(len(rows)) if score[i] >= threshold and unc[i] <= unc_threshold and is_compatible(rows[i])]
    if not accepted:
        return {"n_accepted": 0}
    selected = [rows[i] for i in accepted]
    return summarize_selected(selected)


def decision_baselines(bundle: DatasetBundle) -> dict[str, Any]:
    rows = bundle.split_rows["TEST"]
    groups = group_indices(rows)
    out = {}
    rng = random.Random(0)
    strategies: dict[str, Callable[[list[int]], int]] = {
        "random_compatible": lambda idxs: rng.choice(idxs),
        "highest_donor_whole_policy_suitability": lambda idxs: max(idxs, key=lambda i: float(rows[i].get("donor_conservative_suitability", 0.0))),
        "highest_donor_whole_policy_reward": lambda idxs: max(idxs, key=lambda i: float(rows[i].get("donor_reward", 0.0))),
        "structural_nearest": lambda idxs: min(idxs, key=lambda i: float(rows[i].get("compatibility_metadata", {}).get("structural_distance", 0.0))),
        "oracle_best_intervention": lambda idxs: max(idxs, key=lambda i: float(rows[i]["C_base"])),
    }
    for name, choose in strategies.items():
        selected = [rows[choose(indices)] for indices in groups.values() if indices]
        out[name] = summarize_selected(selected)
    return out


def summarize_selected(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    out = {"n": len(rows)}
    for target in TARGETS:
        vals = np.asarray([float(r[target]) for r in rows])
        out[f"mean_{target}"] = float(vals.mean())
        out[f"positive_{target}_fraction"] = float((vals > 0).mean())
        out[f"negative_{target}_fraction"] = float((vals < 0).mean())
    out["positive_transfer_precision"] = out["positive_C_base_fraction"]
    out["beats_both_parents_fraction"] = out["positive_C_parent_fraction"]
    out["expands_envelope_fraction"] = out["positive_C_env_fraction"]
    return out


def verdict(final: Mapping[str, Any]) -> tuple[str, str]:
    rows = final.get("top_models_test", [])
    if not rows:
        return "NO_RELIABLE_SIGNAL", "NOT_READY"
    best = max(rows, key=lambda r: (
        r["test"].get("top1_mean_C_parent", -1),
        r["test"].get("top1_mean_C_env", -1),
        r["test"].get("top1_positive_precision", 0),
    ))
    m = best["test"]
    strong = (
        m.get("top1_positive_precision", 0) >= 0.7
        and m.get("top1_mean_C_base", -1) > 0
        and m.get("top1_mean_C_parent", -1) >= 0
        and m.get("top1_mean_C_env", -1) >= 0
        and m.get("top1_negative_transfer_rate", 1) <= 0.2
    )
    niche = (
        m.get("top1_positive_precision", 0) >= 0.5
        and m.get("top1_mean_C_base", -1) > 0
        and m.get("top1_negative_transfer_rate", 1) <= 0.35
    )
    improved = m.get("top1_mean_C_base", -1) > 0 or m.get("top1_positive_precision", 0) >= 0.25
    if strong:
        return "STRONG_SIGNAL", "READY_WITH_RESTRICTIONS"
    if niche:
        return "NICHE_SIGNAL", "READY_WITH_RESTRICTIONS"
    if improved:
        return "IMPROVED_BUT_NOT_READY", "NOT_READY"
    return "NO_RELIABLE_SIGNAL", "NOT_READY"


def render_final_report(final: Mapping[str, Any], diagnosis: Mapping[str, Any]) -> str:
    status = final.get("OVERNIGHT_MODULE_CREDIT_STATUS", "UNKNOWN")
    readiness = final.get("STRUCTURAL_SYNTHESIS_READINESS", "UNKNOWN")
    lines = [
        "# Overnight Module-Credit Improvement Report",
        "",
        f"OVERNIGHT_MODULE_CREDIT_STATUS = {status}",
        f"STRUCTURAL_SYNTHESIS_READINESS = {readiness}",
        "",
        "## Diagnosis",
        "```json",
        json.dumps(diagnosis.get("failure_hypotheses", {}), indent=2, default=json_default),
        "```",
        "",
        "## Final Test Evaluation",
        "```json",
        json.dumps(final.get("top_models_test", [])[:5], indent=2, default=json_default),
        "```",
        "",
        "## Decision Baselines",
        "```json",
        json.dumps(final.get("decision_baselines", {}), indent=2, default=json_default),
        "```",
    ]
    return "\n".join(lines) + "\n"


def resolve_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default))


def heartbeat(out_dir: Path, stage: str, **payload: Any) -> None:
    write_json(out_dir / "checkpoints" / "heartbeat.json", {
        "stage": stage,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pid": os.getpid(),
        **payload,
    })


def save_rng_state(path: Path, rng: np.random.Generator) -> None:
    payload = {
        "python_random": random.getstate(),
        "numpy_bit_generator": rng.bit_generator.state,
    }
    if torch is not None:
        payload["torch_rng"] = torch.get_rng_state()
        if torch.cuda.is_available():
            payload["torch_cuda_rng"] = torch.cuda.get_rng_state_all()
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def flatten(payload: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    out = {}
    for key, value in payload.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            out.update(flatten(value, name))
        elif isinstance(value, (list, tuple)):
            out[name] = json.dumps(value, default=json_default)
        else:
            out[name] = value
    return out


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if torch is not None and isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def run_cmd(cmd: Sequence[str], *, cwd: Path) -> str:
    import subprocess

    try:
        return subprocess.check_output(list(cmd), cwd=cwd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"UNKNOWN: {exc}"


if __name__ == "__main__":
    raise SystemExit(main())
