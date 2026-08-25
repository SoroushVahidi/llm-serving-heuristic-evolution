#!/usr/bin/env python3
"""
Phase 2C.3: external-aware selector retraining over completed Phase 2C.2 rows.

This is an analytical follow-up to Phase 2C.2. It reuses the completed
train/eval CSVs, reconstructs the corrected ANWG objective, defines several
selector target pools, and evaluates whether external-aware portfolios can
recover the Azure-conv failure regime without leaking evaluation outcomes into
prediction-time features.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from llmserveopt.selector.roles import EXTERNAL_STYLE_BASELINES

from run_phase2b9_selector_robustness import load_config

DEFAULT_CONFIG = "configs/phase2c3_external_aware_orca_recovery.yaml"
DEFAULT_OUTPUT_DIR = "results/phase2c3_external_aware_orca_recovery"
DEFAULT_LOG_FILE = "logs/phase2c3_external_aware_orca_recovery/phase2c3_external_aware_orca_recovery.log"
SMOKE_LOG_FILE = "logs/phase2c3_external_aware_orca_recovery/phase2c3_external_aware_orca_recovery_smoke.log"


@dataclass(frozen=True)
class TargetPool:
    name: str
    allowed_policies: tuple[str, ...]
    description: str


@dataclass
class TrainedSelector:
    key: str
    target_pool: str
    model_family: str
    predictor: Any
    label_col: str
    allowed_policies: tuple[str, ...]
    train_rows_used: int
    near_tie_epsilon: float


class SklearnPolicyClassifier:
    def __init__(self, estimator: Any, feature_cols: Sequence[str]):
        self.estimator = estimator
        self.feature_cols = list(feature_cols)

    def fit(
        self,
        df: pd.DataFrame,
        *,
        label_col: str,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "SklearnPolicyClassifier":
        self.estimator.fit(df[self.feature_cols].to_numpy(dtype=float), df[label_col], sample_weight=sample_weight)
        return self

    def predict(self, df: pd.DataFrame) -> List[str]:
        return self.estimator.predict(df[self.feature_cols].to_numpy(dtype=float)).tolist()


class PoolKNNSelector:
    def __init__(self, feature_cols: Sequence[str], allowed_policies: Sequence[str], *, k: int = 5, metric: str = "euclidean"):
        self.feature_cols = list(feature_cols)
        self.allowed_policies = list(allowed_policies)
        self.k = int(k)
        self.metric = metric
        self._scaler = None
        self._train_x = None
        self._train_scores = None

    def fit(self, df: pd.DataFrame) -> "PoolKNNSelector":
        from sklearn.neighbors import NearestNeighbors
        from sklearn.preprocessing import StandardScaler

        self._nn_cls = NearestNeighbors
        self._scaler = StandardScaler().fit(df[self.feature_cols].to_numpy(dtype=float))
        self._train_x = self._scaler.transform(df[self.feature_cols].to_numpy(dtype=float))
        self._train_scores = df[[f"anwg_{p}" for p in self.allowed_policies]].to_numpy(dtype=float)
        return self

    def predict(self, df: pd.DataFrame) -> List[str]:
        x = self._scaler.transform(df[self.feature_cols].to_numpy(dtype=float))
        k = min(self.k, len(self._train_x))
        nn = self._nn_cls(n_neighbors=k, metric=self.metric)
        nn.fit(self._train_x)
        _, indices = nn.kneighbors(x)
        preds: List[str] = []
        for idxs in indices:
            score_sums = self._train_scores[idxs].sum(axis=0)
            preds.append(self.allowed_policies[int(np.argmax(score_sums))])
        return preds


class PoolRegressionSelector:
    def __init__(
        self,
        feature_cols: Sequence[str],
        allowed_policies: Sequence[str],
        *,
        n_estimators: int = 200,
        max_depth: int = 10,
        random_state: int = 42,
    ):
        self.feature_cols = list(feature_cols)
        self.allowed_policies = list(allowed_policies)
        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "random_state": random_state,
        }
        self.models: Dict[str, Any] = {}

    def fit(self, df: pd.DataFrame) -> "PoolRegressionSelector":
        from sklearn.ensemble import RandomForestRegressor

        x = df[self.feature_cols].to_numpy(dtype=float)
        for policy in self.allowed_policies:
            reg = RandomForestRegressor(**self.params)
            reg.fit(x, df[f"anwg_{policy}"].to_numpy(dtype=float))
            self.models[policy] = reg
        return self

    def predict(self, df: pd.DataFrame) -> List[str]:
        x = df[self.feature_cols].to_numpy(dtype=float)
        by_policy = {policy: model.predict(x) for policy, model in self.models.items()}
        preds: List[str] = []
        for row_idx in range(len(df)):
            preds.append(max(self.allowed_policies, key=lambda policy: by_policy[policy][row_idx]))
        return preds


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2C.3 external-aware orca recovery")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--allow-full-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def _repo_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else ROOT / path


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _setup_logging(log_file: Optional[str], verbose: bool) -> None:
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        log_path = _repo_path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
    )


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=False)


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        pd.DataFrame().to_csv(path, index=False)
        return
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def workload_name(trace_id: str) -> str:
    return trace_id.rsplit("_s", 1)[0] if "_s" in trace_id else trace_id


def discover_candidate_policies(df: pd.DataFrame) -> List[str]:
    reward = {col[len("reward_"):] for col in df.columns if col.startswith("reward_")}
    completion = {col[len("completion_"):] for col in df.columns if col.startswith("completion_")}
    return sorted(reward & completion)


def select_feature_columns(df: pd.DataFrame) -> List[str]:
    features = [col for col in df.columns if col.startswith("feat_")]
    leaks = [col for col in features if any(token in col.lower() for token in ("reward_", "completion_", "sel_", "best_", "label"))]
    if leaks:
        raise ValueError(f"Leaky feature columns detected: {sorted(leaks)}")
    return features


def reconstruct_corrected_anwg(df: pd.DataFrame, policies: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for policy in policies:
        out[f"anwg_{policy}"] = out[f"reward_{policy}"].astype(float) * out[f"completion_{policy}"].astype(float)
    return out


def add_workload_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["workload"] = out["trace_id"].map(workload_name)
    if "workload_group" not in out.columns:
        out["workload_group"] = np.where(out["workload"].str.startswith("azure_"), "azure_2023", "burstgpt")
    out = out.sort_values(["workload", "window_id"]).reset_index(drop=True)
    out["window_rank_in_workload"] = out.groupby("workload").cumcount()
    out["is_exact_prediction"] = out["workload"].eq("burstgpt_moderate_exact_prediction")
    out["is_overlap_sensitive_first_two"] = out["workload"].eq("burstgpt_scaled_high") & (out["window_rank_in_workload"] < 2)
    out["is_realistic_subset"] = ~(out["is_exact_prediction"] | out["is_overlap_sensitive_first_two"])
    out["row_key"] = np.arange(len(out), dtype=int)
    return out


def validate_required_policies(policies: Sequence[str], required: Sequence[str]) -> List[str]:
    missing = [policy for policy in required if policy not in policies]
    if missing:
        raise RuntimeError(f"Missing required policy columns: {missing}")
    return missing


def resolve_target_pools(cfg: dict, candidate_policies: Sequence[str]) -> Dict[str, TargetPool]:
    candidate_set = set(candidate_policies)
    oracle_prefixes = tuple(cfg.get("oracle_assisted_prefixes", ["safe_fallback_wsp_margin"]))
    filtered = [policy for policy in candidate_policies if not policy.startswith(oracle_prefixes)]

    native_excluded = set(cfg.get("native_excluded_policies", []))
    native = [policy for policy in filtered if policy not in native_excluded]
    external_aware = list(filtered)
    gate = [policy for policy in ("orca_style", "scorpio_style_slo_guard") if policy in candidate_set]

    pools = {
        "native_non_oracle": TargetPool(
            name="native_non_oracle",
            allowed_policies=tuple(native),
            description="Phase 2C.2-like native pool without the four purely external approximations.",
        ),
        "external_aware_non_oracle": TargetPool(
            name="external_aware_non_oracle",
            allowed_policies=tuple(external_aware),
            description="All non-oracle candidate policies, including internal external-style approximations.",
        ),
        "orca_vs_scorpio_gate": TargetPool(
            name="orca_vs_scorpio_gate",
            allowed_policies=tuple(gate),
            description="Binary specialist over orca_style vs scorpio_style_slo_guard.",
        ),
        "external_aware_balanced": TargetPool(
            name="external_aware_balanced",
            allowed_policies=tuple(external_aware),
            description="External-aware pool with balanced class weighting for classifier variants.",
        ),
    }
    return {name: pools[name] for name in cfg.get("target_pools", list(pools.keys()))}


def pool_label_columns(pool_name: str) -> Dict[str, str]:
    return {
        "label": f"{pool_name}_label",
        "best_anwg": f"{pool_name}_best_anwg",
        "second_anwg": f"{pool_name}_second_anwg",
        "margin": f"{pool_name}_margin",
    }


def compute_pool_labels(df: pd.DataFrame, pool: TargetPool) -> pd.DataFrame:
    out = df.copy()
    cols = pool_label_columns(pool.name)
    label_vals: List[str] = []
    best_vals: List[float] = []
    second_vals: List[float] = []
    margin_vals: List[float] = []
    allowed = list(pool.allowed_policies)
    for _, row in out.iterrows():
        scored = sorted(((policy, float(row[f"anwg_{policy}"])) for policy in allowed), key=lambda item: item[1], reverse=True)
        label_vals.append(scored[0][0])
        best_vals.append(scored[0][1])
        second_vals.append(scored[1][1] if len(scored) > 1 else scored[0][1])
        margin_vals.append(scored[0][1] - (scored[1][1] if len(scored) > 1 else scored[0][1]))
    out[cols["label"]] = label_vals
    out[cols["best_anwg"]] = best_vals
    out[cols["second_anwg"]] = second_vals
    out[cols["margin"]] = margin_vals
    return out


def apply_smoke_subsampling(train_df: pd.DataFrame, eval_df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    smoke_cfg = cfg.get("smoke", {})
    max_train = int(smoke_cfg.get("max_train_rows", 80))
    max_eval_per_workload = int(smoke_cfg.get("max_eval_rows_per_workload", 8))
    train_smoke = train_df.head(max_train).copy()
    eval_smoke = (
        eval_df.sort_values(["workload", "window_id"])
        .groupby("workload", group_keys=False)
        .head(max_eval_per_workload)
        .reset_index(drop=True)
    )
    return train_smoke, eval_smoke


def compute_selector_anwg_from_existing_eval(eval_df: pd.DataFrame, selector_key: str) -> float:
    policy_col = f"sel_{selector_key}_policy"
    return float(np.mean([float(row[f"anwg_{row[policy_col]}"]) for _, row in eval_df.iterrows()]))


def compute_external_envelope(eval_df: pd.DataFrame, external_policies: Sequence[str]) -> pd.DataFrame:
    out = eval_df.copy()
    best_policies: List[str] = []
    best_anwgs: List[float] = []
    best_rewards: List[float] = []
    best_completions: List[float] = []
    best_slos: List[float] = []
    for _, row in out.iterrows():
        scored = sorted(((policy, float(row[f"anwg_{policy}"])) for policy in external_policies), key=lambda item: item[1], reverse=True)
        policy = scored[0][0]
        best_policies.append(policy)
        best_anwgs.append(scored[0][1])
        best_rewards.append(float(row[f"reward_{policy}"]))
        best_completions.append(float(row[f"completion_{policy}"]))
        best_slos.append(float(row[f"slo_violation_{policy}"]))
    out["external_best_policy"] = best_policies
    out["external_best_anwg"] = best_anwgs
    out["external_best_reward"] = best_rewards
    out["external_best_completion"] = best_completions
    out["external_best_slo_violation"] = best_slos
    return out


def reproduce_phase2c2_metrics(eval_df: pd.DataFrame, cfg: dict, external_policies: Sequence[str]) -> Dict[str, Any]:
    expected = cfg.get("expected_reproduction", {})
    dt_anwg = compute_selector_anwg_from_existing_eval(eval_df, "dt_anwg")
    always_scorpio = float(eval_df["anwg_scorpio_style_slo_guard"].mean())
    ext_env = float(eval_df["external_best_anwg"].mean())
    dt_sel_anwg = [float(row[f"anwg_{row['sel_dt_anwg_policy']}"]) for _, row in eval_df.iterrows()]
    external_loss_windows = int(np.sum(eval_df["external_best_anwg"].to_numpy(dtype=float) > np.array(dt_sel_anwg) + 1e-12))

    checks = {
        "dt_anwg": dt_anwg,
        "always_scorpio": always_scorpio,
        "external_style_envelope": ext_env,
        "external_loss_windows": external_loss_windows,
    }
    for key, expected_value in expected.items():
        actual = checks[key]
        if key == "external_loss_windows":
            if int(actual) != int(expected_value):
                raise RuntimeError(f"Phase 2C.2 reproduction failed for {key}: got {actual}, expected {expected_value}")
        else:
            if abs(float(actual) - float(expected_value)) > 5e-4:
                raise RuntimeError(f"Phase 2C.2 reproduction failed for {key}: got {actual:.6f}, expected {expected_value:.6f}")
    return checks


def compute_sample_weights(
    df: pd.DataFrame,
    *,
    label_col: str,
    margin_col: str,
    regret_epsilon: float,
    balanced: bool,
    regret_weighted: bool,
) -> Optional[np.ndarray]:
    weights = np.ones(len(df), dtype=float)
    if balanced:
        counts = Counter(df[label_col])
        class_weight = {label: len(df) / (len(counts) * count) for label, count in counts.items()}
        weights *= np.array([class_weight[label] for label in df[label_col]], dtype=float)
    if regret_weighted:
        weights *= np.clip(df[margin_col].to_numpy(dtype=float) + regret_epsilon, regret_epsilon, None)
    if np.allclose(weights, 1.0):
        return None
    return weights


def train_pool_selectors(
    train_df: pd.DataFrame,
    feature_cols: Sequence[str],
    pool: TargetPool,
    cfg: dict,
) -> List[TrainedSelector]:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier

    labels = pool_label_columns(pool.name)
    near_tie_epsilon = float(cfg.get("near_tie_filter_epsilon", 0.005))
    filtered = train_df[train_df[labels["margin"]] >= near_tie_epsilon].copy()
    if filtered.empty:
        raise RuntimeError(f"No train rows survive near-tie filtering for pool {pool.name}")

    model_cfg = cfg.get("models", {})
    dt_cfg = model_cfg.get("decision_tree", {})
    rf_cfg = model_cfg.get("random_forest", {})
    knn_cfg = model_cfg.get("knn", {})
    reg_cfg = model_cfg.get("regression", {})
    regret_eps = float(cfg.get("regret_weight_epsilon", 0.001))

    selectors: List[TrainedSelector] = []

    def add_classifier(name_suffix: str, estimator: Any, *, balanced: bool = False, regret_weighted: bool = False) -> None:
        predictor = SklearnPolicyClassifier(estimator, feature_cols)
        sample_weight = compute_sample_weights(
            filtered,
            label_col=labels["label"],
            margin_col=labels["margin"],
            regret_epsilon=regret_eps,
            balanced=balanced,
            regret_weighted=regret_weighted,
        )
        predictor.fit(filtered, label_col=labels["label"], sample_weight=sample_weight)
        selectors.append(
            TrainedSelector(
                key=f"{pool.name}_{name_suffix}",
                target_pool=pool.name,
                model_family=name_suffix,
                predictor=predictor,
                label_col=labels["label"],
                allowed_policies=pool.allowed_policies,
                train_rows_used=len(filtered),
                near_tie_epsilon=near_tie_epsilon,
            )
        )

    add_classifier(
        "dt",
        DecisionTreeClassifier(
            max_depth=int(dt_cfg.get("max_depth", 8)),
            min_samples_leaf=int(dt_cfg.get("min_samples_leaf", 5)),
            random_state=int(dt_cfg.get("random_state", 42)),
        ),
        balanced=False,
        regret_weighted=False,
    )
    add_classifier(
        "rf",
        RandomForestClassifier(
            n_estimators=int(rf_cfg.get("n_estimators", 200)),
            max_depth=int(rf_cfg.get("max_depth", 10)),
            random_state=int(rf_cfg.get("random_state", 42)),
        ),
        balanced=False,
        regret_weighted=False,
    )

    if pool.name == "external_aware_balanced":
        add_classifier(
            "dt_balanced",
            DecisionTreeClassifier(
                max_depth=int(dt_cfg.get("max_depth", 8)),
                min_samples_leaf=int(dt_cfg.get("min_samples_leaf", 5)),
                random_state=int(dt_cfg.get("random_state", 42)),
            ),
            balanced=True,
            regret_weighted=True,
        )
        add_classifier(
            "rf_balanced",
            RandomForestClassifier(
                n_estimators=int(rf_cfg.get("n_estimators", 200)),
                max_depth=int(rf_cfg.get("max_depth", 10)),
                random_state=int(rf_cfg.get("random_state", 42)),
            ),
            balanced=True,
            regret_weighted=True,
        )

    if pool.name != "orca_vs_scorpio_gate":
        knn = PoolKNNSelector(
            feature_cols,
            pool.allowed_policies,
            k=int(knn_cfg.get("k", 5)),
            metric=str(knn_cfg.get("metric", "euclidean")),
        ).fit(filtered)
        selectors.append(
            TrainedSelector(
                key=f"{pool.name}_knn",
                target_pool=pool.name,
                model_family="knn",
                predictor=knn,
                label_col=labels["label"],
                allowed_policies=pool.allowed_policies,
                train_rows_used=len(filtered),
                near_tie_epsilon=near_tie_epsilon,
            )
        )

        reg = PoolRegressionSelector(
            feature_cols,
            pool.allowed_policies,
            n_estimators=int(reg_cfg.get("n_estimators", 200)),
            max_depth=int(reg_cfg.get("max_depth", 10)),
            random_state=int(reg_cfg.get("random_state", 42)),
        ).fit(filtered)
        selectors.append(
            TrainedSelector(
                key=f"{pool.name}_regression",
                target_pool=pool.name,
                model_family="regression",
                predictor=reg,
                label_col=labels["label"],
                allowed_policies=pool.allowed_policies,
                train_rows_used=len(filtered),
                near_tie_epsilon=near_tie_epsilon,
            )
        )

    return selectors


def predict_selector(selector: TrainedSelector, df: pd.DataFrame) -> List[str]:
    return selector.predictor.predict(df)


def compute_best_fixed_policy(df: pd.DataFrame, policies: Sequence[str]) -> tuple[str, float]:
    means = {policy: float(df[f"anwg_{policy}"].mean()) for policy in policies}
    best_policy = max(means, key=means.get)
    return best_policy, means[best_policy]


def chosen_policy_distribution(predictions: Sequence[str]) -> Dict[str, float]:
    counts = Counter(predictions)
    total = float(sum(counts.values()) or 1.0)
    return {policy: count / total for policy, count in sorted(counts.items())}


def flatten_policy_distribution(prefix: str, dist: Mapping[str, float]) -> Dict[str, float]:
    return {f"{prefix}_{policy}": round(float(value), 6) for policy, value in dist.items()}


def evaluate_predictions(
    df: pd.DataFrame,
    selector: TrainedSelector,
    predictions: Sequence[str],
    *,
    external_policies: Sequence[str],
    view_name: str,
) -> Dict[str, Any]:
    label_col = selector.label_col
    selected_anwg = np.array([float(row[f"anwg_{pred}"]) for (_, row), pred in zip(df.iterrows(), predictions)], dtype=float)
    selected_completion = np.array([float(row[f"completion_{pred}"]) for (_, row), pred in zip(df.iterrows(), predictions)], dtype=float)
    selected_reward = np.array([float(row[f"reward_{pred}"]) for (_, row), pred in zip(df.iterrows(), predictions)], dtype=float)
    selected_slo = np.array([float(row[f"slo_violation_{pred}"]) for (_, row), pred in zip(df.iterrows(), predictions)], dtype=float)
    external_best = df["external_best_anwg"].to_numpy(dtype=float)
    label_accuracy = float(np.mean(np.array(predictions) == df[label_col].to_numpy())) if len(df) else 0.0
    best_fixed_external_policy, best_fixed_external_anwg = compute_best_fixed_policy(df, external_policies)
    dist = chosen_policy_distribution(predictions)
    return {
        "view": view_name,
        "selector": selector.key,
        "target_pool": selector.target_pool,
        "model_family": selector.model_family,
        "n_windows": int(len(df)),
        "label_accuracy": round(label_accuracy, 6),
        "mean_arrival_normalized_wg": round(float(np.mean(selected_anwg)), 6),
        "mean_completed_request_quality": round(float(np.mean(selected_reward)), 6),
        "mean_completion_fraction": round(float(np.mean(selected_completion)), 6),
        "mean_slo_violation": round(float(np.mean(selected_slo)), 6),
        "gap_vs_always_scorpio": round(float(np.mean(selected_anwg) - float(df["anwg_scorpio_style_slo_guard"].mean())), 6),
        "gap_vs_external_envelope": round(float(np.mean(selected_anwg) - float(np.mean(external_best))), 6),
        "gap_vs_best_fixed_external": round(float(np.mean(selected_anwg) - best_fixed_external_anwg), 6),
        "best_fixed_external_policy": best_fixed_external_policy,
        "best_fixed_external_anwg": round(best_fixed_external_anwg, 6),
        "orca_choice_count": int(sum(pred == "orca_style" for pred in predictions)),
        "orca_choice_fraction": round(float(np.mean(np.array(predictions) == "orca_style")), 6),
        "scorpio_choice_count": int(sum(pred == "scorpio_style_slo_guard" for pred in predictions)),
        "scorpio_choice_fraction": round(float(np.mean(np.array(predictions) == "scorpio_style_slo_guard")), 6),
        "primary_rank_metric": "mean_arrival_normalized_wg",
        **flatten_policy_distribution("chosen_policy_dist", dist),
    }


def build_eval_views(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    return {
        "all_workloads": df.copy(),
        "burstgpt_only": df[df["workload_group"].eq("burstgpt")].copy(),
        "azure_only": df[df["workload_group"].eq("azure_2023")].copy(),
        "azure_2023_conv": df[df["workload"].eq("azure_2023_conv")].copy(),
        "excluding_exact_prediction": df[~df["is_exact_prediction"]].copy(),
        "excluding_first_two_overlap_windows": df[~df["is_overlap_sensitive_first_two"]].copy(),
        "realistic_subset": df[df["is_realistic_subset"]].copy(),
    }


def build_per_window_predictions(
    eval_df: pd.DataFrame,
    selectors: Sequence[TrainedSelector],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for selector in selectors:
        predictions = predict_selector(selector, eval_df)
        for (_, row), pred in zip(eval_df.iterrows(), predictions):
            rows.append({
                "selector": selector.key,
                "target_pool": selector.target_pool,
                "model_family": selector.model_family,
                "row_key": int(row["row_key"]),
                "workload": row["workload"],
                "workload_group": row["workload_group"],
                "window_id": int(row["window_id"]),
                "predicted_policy": pred,
                "selected_anwg": float(row[f"anwg_{pred}"]),
                "selected_completion_fraction": float(row[f"completion_{pred}"]),
                "selected_completed_request_quality": float(row[f"reward_{pred}"]),
                "selected_slo_violation": float(row[f"slo_violation_{pred}"]),
                "external_best_policy": row["external_best_policy"],
                "external_best_anwg": float(row["external_best_anwg"]),
                "external_best_completion_fraction": float(row["external_best_completion"]),
                "external_best_completed_request_quality": float(row["external_best_reward"]),
                "is_exact_prediction": bool(row["is_exact_prediction"]),
                "is_overlap_sensitive_first_two": bool(row["is_overlap_sensitive_first_two"]),
                "is_realistic_subset": bool(row["is_realistic_subset"]),
                **{feature: row[feature] for feature in eval_df.columns if feature.startswith("feat_")},
            })
    return pd.DataFrame(rows)


def compute_label_count_rows(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    pools: Mapping[str, TargetPool],
    near_tie_epsilon: float,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for split_name, df in (("train", train_df), ("val", val_df)):
        for pool_name, pool in pools.items():
            cols = pool_label_columns(pool_name)
            counts_raw = Counter(df[cols["label"]])
            filtered = df[df[cols["margin"]] >= near_tie_epsilon]
            counts_filtered = Counter(filtered[cols["label"]])
            all_labels = sorted(set(counts_raw) | set(counts_filtered))
            for label in all_labels:
                rows.append({
                    "split": split_name,
                    "target_pool": pool_name,
                    "label": label,
                    "count_raw": int(counts_raw.get(label, 0)),
                    "count_filtered": int(counts_filtered.get(label, 0)),
                    "near_tie_epsilon": near_tie_epsilon,
                })
    return rows


def best_selector_overall(summary_df: pd.DataFrame) -> pd.Series:
    overall = summary_df[summary_df["view"].eq("all_workloads")].copy()
    overall = overall.sort_values(["mean_arrival_normalized_wg", "orca_choice_fraction"], ascending=[False, False])
    return overall.iloc[0]


def build_external_loss_cases(best_selector_row: pd.Series, per_window_df: pd.DataFrame) -> pd.DataFrame:
    selector_key = str(best_selector_row["selector"])
    sub = per_window_df[per_window_df["selector"].eq(selector_key)].copy()
    losses = sub[sub["external_best_anwg"] > sub["selected_anwg"] + 1e-12].copy()
    losses["loss"] = losses["external_best_anwg"] - losses["selected_anwg"]
    losses["completion_fraction_difference"] = losses["external_best_completion_fraction"] - losses["selected_completion_fraction"]
    losses["quality_difference"] = losses["external_best_completed_request_quality"] - losses["selected_completed_request_quality"]
    return losses.sort_values(["loss", "workload", "window_id"], ascending=[False, True, True]).reset_index(drop=True)


def build_azure_conv_orca_summary_from_eval(best_selector_row: pd.Series, per_window_df: pd.DataFrame, eval_df: pd.DataFrame) -> Dict[str, Any]:
    selector_key = str(best_selector_row["selector"])
    sub = per_window_df[
        per_window_df["selector"].eq(selector_key) & per_window_df["workload"].eq("azure_2023_conv")
    ].copy()
    eval_sub = eval_df[eval_df["workload"].eq("azure_2023_conv")].copy()
    if sub.empty:
        return {"selector": selector_key, "n_windows": 0}
    merged = sub.merge(
        eval_sub[["workload", "window_id", "anwg_orca_style", "anwg_scorpio_style_slo_guard"]],
        on=["workload", "window_id"],
        how="left",
    )
    return {
        "selector": selector_key,
        "target_pool": str(best_selector_row["target_pool"]),
        "model_family": str(best_selector_row["model_family"]),
        "n_windows": int(len(merged)),
        "orca_choice_count": int(np.sum(merged["predicted_policy"].eq("orca_style"))),
        "orca_choice_fraction": round(float(np.mean(merged["predicted_policy"].eq("orca_style"))), 6),
        "scorpio_choice_count": int(np.sum(merged["predicted_policy"].eq("scorpio_style_slo_guard"))),
        "scorpio_choice_fraction": round(float(np.mean(merged["predicted_policy"].eq("scorpio_style_slo_guard"))), 6),
        "mean_selected_anwg": round(float(merged["selected_anwg"].mean()), 6),
        "mean_orca_baseline_anwg": round(float(merged["anwg_orca_style"].mean()), 6),
        "mean_scorpio_baseline_anwg": round(float(merged["anwg_scorpio_style_slo_guard"].mean()), 6),
        "gap_to_orca_baseline": round(float(merged["selected_anwg"].mean() - merged["anwg_orca_style"].mean()), 6),
        "gap_to_scorpio_baseline": round(float(merged["selected_anwg"].mean() - merged["anwg_scorpio_style_slo_guard"].mean()), 6),
    }


def safe_claims(best_selector_row: pd.Series, reproduction: Dict[str, Any], realistic_best: float, realistic_external: float) -> Dict[str, List[str]]:
    selected = float(best_selector_row["mean_arrival_normalized_wg"])
    claims_safe = [
        "This is an offline selector retraining analysis over internal simulator outputs and completed Phase 2C.2 rows.",
        "External-aware selectors are selecting among internal approximations, not official external systems.",
        f"The best Phase 2C.3 selector reaches {selected:.4f} ANWG on the held-out Phase 2C.1 evaluation set.",
        f"The realistic-subset best selector reaches {realistic_best:.4f} ANWG versus {realistic_external:.4f} for the external-style envelope.",
    ]
    claims_unsafe = [
        "Claiming to beat official ORCA, vLLM, Sarathi, or SplitFuse implementations.",
        "Claiming deployable online superiority without re-running a causal live/in-simulator deployment path.",
        "Claiming to beat the per-window external-style envelope unless the metric is independently re-audited.",
    ]
    if selected > float(reproduction["external_style_envelope"]) + 1e-9:
        claims_unsafe.append("The best selector appears to beat the per-window external envelope; treat that as a likely leakage or metric bug until disproven.")
    return {"safe": claims_safe, "unsafe": claims_unsafe}


def render_report(
    *,
    cfg: dict,
    reproduction: Dict[str, Any],
    summary_df: pd.DataFrame,
    realistic_df: pd.DataFrame,
    best_selector: pd.Series,
    label_counts_df: pd.DataFrame,
    feature_cols: Sequence[str],
    external_loss_cases: pd.DataFrame,
    azure_summary: Dict[str, Any],
    pool_map: Mapping[str, TargetPool],
    eval_df: pd.DataFrame,
) -> str:
    overall = summary_df[summary_df["view"].eq("all_workloads")].copy()
    best_native = overall[overall["target_pool"].eq("native_non_oracle")].sort_values("mean_arrival_normalized_wg", ascending=False).iloc[0]
    best_external_aware = overall[overall["target_pool"].eq("external_aware_non_oracle")].sort_values("mean_arrival_normalized_wg", ascending=False).iloc[0]
    best_balanced = overall[overall["target_pool"].eq("external_aware_balanced")].sort_values("mean_arrival_normalized_wg", ascending=False).iloc[0]
    gate_rows = overall[overall["target_pool"].eq("orca_vs_scorpio_gate")].sort_values("mean_arrival_normalized_wg", ascending=False)
    best_gate = gate_rows.iloc[0] if not gate_rows.empty else None
    realistic_best = float(realistic_df["mean_arrival_normalized_wg"].max())
    realistic_external = float(
        eval_df[eval_df["is_realistic_subset"]]["external_best_anwg"].mean()
    )
    claim_block = safe_claims(best_selector, reproduction, realistic_best, realistic_external)

    orca_train_count = int(label_counts_df[(label_counts_df["split"].eq("train")) & (label_counts_df["target_pool"].eq("external_aware_non_oracle")) & (label_counts_df["label"].eq("orca_style"))]["count_raw"].sum())
    native_train_orca = int(label_counts_df[(label_counts_df["split"].eq("train")) & (label_counts_df["target_pool"].eq("native_non_oracle")) & (label_counts_df["label"].eq("orca_style"))]["count_raw"].sum())
    best_selector_losses = len(external_loss_cases)
    realistic_phase2c2 = 0.8050
    realistic_external_phase2c2 = 0.8351

    lines = [
        "# Phase 2C.3 External-Aware Orca Recovery",
        "",
        "## Setup",
        f"- Primary metric: `{cfg.get('primary_rank_metric', 'mean_arrival_normalized_wg')}`.",
        f"- Feature count: `{len(feature_cols)}` causal features.",
        f"- Pools evaluated: `{', '.join(pool_map.keys())}`.",
        f"- Phase 2C.2 reproduction: dt `{reproduction['dt_anwg']:.4f}`, always_scorpio `{reproduction['always_scorpio']:.4f}`, external envelope `{reproduction['external_style_envelope']:.4f}`, external losses `{reproduction['external_loss_windows']}`.",
        "",
        "## Answers",
        f"1. External-aware training allows selectors to choose `orca_style` only when the model family can score policies directly or when the binary gate is used. The external-aware classifier label space still has `0` raw `orca_style` labels on the train split, while the `orca_vs_scorpio_gate` has non-zero `orca_style` labels.",
        f"2. Best Phase 2C.3 selector: `{best_selector['selector']}` chooses `orca_style` `{int(best_selector['orca_choice_count'])}` times overall (`{float(best_selector['orca_choice_fraction']):.3f}`) and `{azure_summary.get('orca_choice_count', 0)}` times on `azure_2023_conv` (`{azure_summary.get('orca_choice_fraction', 0.0):.3f}`).",
        f"3. Best Phase 2C.3 ANWG is `{float(best_selector['mean_arrival_normalized_wg']):.4f}` vs Phase 2C.2 best native `{reproduction['dt_anwg']:.4f}` and always_scorpio `{reproduction['always_scorpio']:.4f}`.",
        f"4. Best Phase 2C.3 selector {'does' if float(best_selector['gap_vs_best_fixed_external']) > 0 else 'does not'} beat the best fixed external-style baseline overall; gap `{float(best_selector['gap_vs_best_fixed_external']):+.4f}`.",
        f"5. Azure-conv gap to `orca_style`: `{azure_summary.get('gap_to_orca_baseline', 0.0):+.4f}` for the best selector.",
        f"6. Best Phase 2C.3 selector {'appears to beat' if float(best_selector['gap_vs_external_envelope']) > 0 else 'does not beat'} the per-window external-style envelope; gap `{float(best_selector['gap_vs_external_envelope']):+.4f}`.",
        f"7. Realistic subset best ANWG is `{realistic_best:.4f}` vs Phase 2C.2 learned `{realistic_phase2c2:.4f}` and Phase 2C.2 external envelope `{realistic_external_phase2c2:.4f}`.",
        "8. Leakage checks passed by construction: features are restricted to `feat_*` columns and explicitly reject `reward_*`, `completion_*`, `sel_*`, `best_*`, and label columns.",
        "9. Safe and unsafe claims are listed below.",
        "",
        "## Pool Notes",
        f"- `native_non_oracle` excludes `{', '.join(cfg.get('native_excluded_policies', []))}`.",
        f"- `external_aware_non_oracle` train raw `orca_style` labels: `{orca_train_count}`.",
        f"- `native_non_oracle` train raw `orca_style` labels: `{native_train_orca}`.",
        "",
        "## Best Selectors",
        f"- Best native: `{best_native['selector']}` at `{float(best_native['mean_arrival_normalized_wg']):.4f}`.",
        f"- Best external-aware: `{best_external_aware['selector']}` at `{float(best_external_aware['mean_arrival_normalized_wg']):.4f}`.",
        f"- Best external-aware balanced: `{best_balanced['selector']}` at `{float(best_balanced['mean_arrival_normalized_wg']):.4f}`.",
    ]
    if best_gate is not None:
        lines.append(f"- Best orca/scorpio gate: `{best_gate['selector']}` at `{float(best_gate['mean_arrival_normalized_wg']):.4f}`.")
    lines.extend([
        "",
        "## External Losses",
        f"- Best Phase 2C.3 selector still loses to the external-style envelope on `{best_selector_losses}` windows.",
    ])
    if not external_loss_cases.empty:
        top_loss = external_loss_cases.iloc[0]
        lines.append(
            f"- Largest remaining loss: `{top_loss['workload']}` window `{int(top_loss['window_id'])}` "
            f"`{top_loss['predicted_policy']}` -> `{top_loss['external_best_policy']}` gap `{float(top_loss['loss']):.4f}`."
        )
    lines.extend([
        "",
        "## Safe Claims",
        *[f"- {claim}" for claim in claim_block["safe"]],
        "",
        "## Unsafe Claims",
        *[f"- {claim}" for claim in claim_block["unsafe"]],
    ])
    return "\n".join(lines) + "\n"


def load_inputs(cfg: dict) -> Dict[str, pd.DataFrame]:
    base = _repo_path(cfg["phase2c2_input_dir"])
    return {
        "train": pd.read_csv(base / "training" / "causal_train_split.csv"),
        "val": pd.read_csv(base / "training" / "causal_val_split.csv"),
        "train_all": pd.read_csv(base / "training" / "causal_training_rows.csv"),
        "eval": pd.read_csv(base / "evaluation" / "per_window.csv"),
        "selector_summary": pd.read_csv(base / "evaluation" / "selector_summary.csv"),
        "deployable_selector_summary": pd.read_csv(base / "evaluation" / "deployable_selector_summary.csv"),
    }


def validate_config_and_inputs(cfg: dict) -> Dict[str, Any]:
    inputs = load_inputs(cfg)
    candidate_policies = discover_candidate_policies(inputs["train"])
    validate_required_policies(candidate_policies, cfg.get("required_policies", []))
    feature_cols = select_feature_columns(inputs["train"])
    pools = resolve_target_pools(cfg, candidate_policies)
    for required_pool in ("native_non_oracle", "external_aware_non_oracle"):
        if required_pool not in pools:
            raise RuntimeError(f"Missing required target pool: {required_pool}")
    if "orca_vs_scorpio_gate" in pools and len(pools["orca_vs_scorpio_gate"].allowed_policies) != 2:
        raise RuntimeError("orca_vs_scorpio_gate requires both orca_style and scorpio_style_slo_guard")
    return {
        "candidate_policies": candidate_policies,
        "feature_cols": feature_cols,
        "pools": pools,
        **inputs,
    }


def plan_to_stdout(cfg: dict, validation: Dict[str, Any], smoke: bool) -> None:
    print(f"Phase 2C.3 {'smoke' if smoke else 'dry-run'}")
    print(f"  experiment          : {cfg['experiment']}")
    print(f"  phase2c2_input_dir  : {cfg['phase2c2_input_dir']}")
    print(f"  policies detected   : {len(validation['candidate_policies'])}")
    print(f"  features            : {len(validation['feature_cols'])}")
    print(f"  pools               : {', '.join(validation['pools'].keys())}")
    print(f"  train rows          : {len(validation['train'])}")
    print(f"  val rows            : {len(validation['val'])}")
    print(f"  eval rows           : {len(validation['eval'])}")


def run_phase2c3(cfg: dict, out_dir: Path, *, smoke: bool) -> Dict[str, Any]:
    validation = validate_config_and_inputs(cfg)
    candidate_policies = validation["candidate_policies"]
    feature_cols = validation["feature_cols"]
    pools = validation["pools"]

    train_df = add_workload_flags(reconstruct_corrected_anwg(validation["train"], candidate_policies))
    val_df = add_workload_flags(reconstruct_corrected_anwg(validation["val"], candidate_policies))
    eval_df = add_workload_flags(reconstruct_corrected_anwg(validation["eval"], candidate_policies))
    eval_df = compute_external_envelope(eval_df, cfg.get("external_style_policies", list(EXTERNAL_STYLE_BASELINES)))

    reproduction = reproduce_phase2c2_metrics(eval_df, cfg, cfg.get("external_style_policies", list(EXTERNAL_STYLE_BASELINES)))
    if smoke:
        train_df, eval_df = apply_smoke_subsampling(train_df, eval_df, cfg)
        val_df = val_df.head(min(len(val_df), 20)).copy()

    for pool in pools.values():
        train_df = compute_pool_labels(train_df, pool)
        val_df = compute_pool_labels(val_df, pool)
        eval_df = compute_pool_labels(eval_df, pool)

    label_counts_rows = compute_label_count_rows(train_df, val_df, pools, float(cfg.get("near_tie_filter_epsilon", 0.005)))
    selectors: List[TrainedSelector] = []
    for pool in pools.values():
        selectors.extend(train_pool_selectors(train_df, feature_cols, pool, cfg))

    views = build_eval_views(eval_df)
    summary_rows: List[Dict[str, Any]] = []
    workload_rows: List[Dict[str, Any]] = []
    prediction_rows = build_per_window_predictions(eval_df, selectors)

    for selector in selectors:
        for view_name, view_df in views.items():
            view_pred = prediction_rows[
                prediction_rows["selector"].eq(selector.key)
                & prediction_rows["row_key"].isin(view_df["row_key"])
            ]["predicted_policy"].tolist()
            summary_rows.append(
                evaluate_predictions(
                    view_df,
                    selector,
                    view_pred,
                    external_policies=cfg.get("external_style_policies", list(EXTERNAL_STYLE_BASELINES)),
                    view_name=view_name,
                )
            )
        for workload, workload_df in eval_df.groupby("workload"):
            workload_pred = prediction_rows[
                prediction_rows["selector"].eq(selector.key)
                & prediction_rows["workload"].eq(workload)
            ]["predicted_policy"].tolist()
            workload_rows.append(
                evaluate_predictions(
                    workload_df,
                    selector,
                    workload_pred,
                    external_policies=cfg.get("external_style_policies", list(EXTERNAL_STYLE_BASELINES)),
                    view_name=workload,
                )
            )

    summary_df = pd.DataFrame(summary_rows)
    workload_df = pd.DataFrame(workload_rows)
    realistic_df = summary_df[summary_df["view"].eq("realistic_subset")].copy()
    best_selector = best_selector_overall(summary_df)
    external_loss_cases = build_external_loss_cases(best_selector, prediction_rows)
    azure_summary = build_azure_conv_orca_summary_from_eval(best_selector, prediction_rows, eval_df)
    report = render_report(
        cfg=cfg,
        reproduction=reproduction,
        summary_df=summary_df,
        realistic_df=realistic_df,
        best_selector=best_selector,
        label_counts_df=pd.DataFrame(label_counts_rows),
        feature_cols=feature_cols,
        external_loss_cases=external_loss_cases,
        azure_summary=azure_summary,
        pool_map=pools,
        eval_df=eval_df,
    )

    best_fixed_external_policy, best_fixed_external_anwg = compute_best_fixed_policy(
        eval_df, cfg.get("external_style_policies", list(EXTERNAL_STYLE_BASELINES))
    )
    best_fixed_external_by_workload = {
        workload: {
            "policy": compute_best_fixed_policy(workload_df, cfg.get("external_style_policies", list(EXTERNAL_STYLE_BASELINES)))[0],
            "anwg": round(compute_best_fixed_policy(workload_df, cfg.get("external_style_policies", list(EXTERNAL_STYLE_BASELINES)))[1], 6),
        }
        for workload, workload_df in eval_df.groupby("workload")
    }

    metadata = {
        "experiment": cfg["experiment"],
        "phase2c2_input_dir": cfg["phase2c2_input_dir"],
        "phase2c3_failure_diagnosis_dir": cfg["phase2c3_failure_diagnosis_dir"],
        "smoke": smoke,
        "candidate_policies": candidate_policies,
        "feature_cols": feature_cols,
        "target_pools": {name: list(pool.allowed_policies) for name, pool in pools.items()},
        "selectors": [selector.key for selector in selectors],
        "reproduced_phase2c2_metrics": reproduction,
        "best_selector_overall": best_selector.to_dict(),
        "best_fixed_external_policy_overall": {
            "policy": best_fixed_external_policy,
            "anwg": round(best_fixed_external_anwg, 6),
        },
        "best_fixed_external_policy_by_workload": best_fixed_external_by_workload,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "metadata.json", metadata)
    _write_text(out_dir / "feature_list.txt", "\n".join(feature_cols) + "\n")
    _write_csv(out_dir / "train_label_counts.csv", label_counts_rows)
    summary_df.to_csv(out_dir / "selector_summary.csv", index=False)
    workload_df.to_csv(out_dir / "workload_summary.csv", index=False)
    prediction_rows.to_csv(out_dir / "per_window_predictions.csv", index=False)
    external_loss_cases.to_csv(out_dir / "external_loss_cases.csv", index=False)
    _write_json(out_dir / "azure_conv_orca_recovery_summary.json", azure_summary)
    realistic_df.to_csv(out_dir / "realistic_subset_summary.csv", index=False)
    _write_text(out_dir / "phase2c3_external_aware_orca_recovery_report.md", report)
    return metadata


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    cfg = load_config(_repo_path(args.config))
    validation = validate_config_and_inputs(cfg)

    if args.dry_run:
        eval_df = add_workload_flags(reconstruct_corrected_anwg(validation["eval"], validation["candidate_policies"]))
        eval_df = compute_external_envelope(eval_df, cfg.get("external_style_policies", list(EXTERNAL_STYLE_BASELINES)))
        reproduction = reproduce_phase2c2_metrics(eval_df, cfg, cfg.get("external_style_policies", list(EXTERNAL_STYLE_BASELINES)))
        plan_to_stdout(cfg, validation, smoke=False)
        print(f"  reproduced dt_anwg   : {reproduction['dt_anwg']:.4f}")
        print(f"  reproduced envelope  : {reproduction['external_style_envelope']:.4f}")
        print("  [dry-run] No files written.")
        return 0

    if args.smoke and args.allow_full_run:
        print("ERROR: choose --smoke or --allow-full-run, not both.", file=sys.stderr)
        return 2
    if not args.smoke and not args.allow_full_run:
        print("ERROR: use --dry-run, --smoke, or --allow-full-run.", file=sys.stderr)
        return 2

    base_out = _repo_path(args.out_dir or cfg.get("output_dir", DEFAULT_OUTPUT_DIR))
    out_dir = (base_out / "smoke" / _timestamp()) if args.smoke else (base_out / _timestamp())
    log_file = args.log_file or (SMOKE_LOG_FILE if args.smoke else DEFAULT_LOG_FILE)
    _setup_logging(log_file, args.verbose)
    logging.info("Phase 2C.3 %s run starting", "smoke" if args.smoke else "full")
    metadata = run_phase2c3(cfg, out_dir, smoke=args.smoke)
    logging.info("Phase 2C.3 complete: %s", out_dir)
    logging.info("Best selector overall: %s", metadata["best_selector_overall"]["selector"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
