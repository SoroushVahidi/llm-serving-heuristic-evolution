#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path("/mmfs1/project/ikoutis/sv96/github/llm-serving-heuristic-evolution")
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

PRIMARY = "arrival_normalized_weighted_goodput"
METRIC = f"metric_{PRIMARY}"
WSP = "weighted_shortest_processing"
SCORPIO = "scorpio_style_slo_guard"

from llmserveopt.policies.composition import (  # noqa: E402
    ComponentWiseCompositionPolicy,
    ConditionalRegimeCompositionPolicy,
    RankExpertSpec,
    StaticRankEnsemblePolicy,
)
from llmserveopt.policies.registry import POLICY_LIBRARY_V2_NAMES, POLICY_LIBRARY_V2_NEW_NAMES  # noqa: E402
from llmserveopt.selector.dataset_v2.calibrated_targeted_pilot import _execution_service_model  # noqa: E402
from llmserveopt.simulator.simulator import Simulator, SimulatorConfig  # noqa: E402


def _load_plv2_module():
    path = REPO / "tools" / "policy_library_v2_experiment.py"
    spec = importlib.util.spec_from_file_location("_plv2_experiment", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PLV2 = _load_plv2_module()


def git(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *cmd], cwd=REPO, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"UNKNOWN: {exc}"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _sample_design(design: pd.DataFrame, dev_count: int, eval_count: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dev = design[design["split_group"].isin(["train", "validation"])].copy()
    heldout = design[design["split_group"].isin(["id_test", "synthetic_ood"])].copy()
    if len(dev) < dev_count or len(heldout) < eval_count:
        raise ValueError("Design does not contain enough development or held-out rows")
    dev_sample = dev.sample(n=dev_count, random_state=int(rng.integers(0, 1_000_000)))
    heldout_sample = heldout.groupby("split_group", group_keys=False).apply(
        lambda part: part.sample(
            n=min(len(part), max(1, round(eval_count * len(part) / len(heldout)))),
            random_state=int(rng.integers(0, 1_000_000)),
        )
    )
    if len(heldout_sample) > eval_count:
        heldout_sample = heldout_sample.sample(n=eval_count, random_state=int(rng.integers(0, 1_000_000)))
    elif len(heldout_sample) < eval_count:
        missing = eval_count - len(heldout_sample)
        extra = heldout.drop(index=heldout_sample.index).sample(n=missing, random_state=int(rng.integers(0, 1_000_000)))
        heldout_sample = pd.concat([heldout_sample, extra], ignore_index=False)
    selected = pd.concat([dev_sample, heldout_sample], ignore_index=True)
    selected["pilot_role"] = ["development"] * len(dev_sample) + ["heldout"] * len(heldout_sample)
    return selected


def _run_config_worker(params: dict[str, Any]) -> dict[str, Any]:
    summary, vectors, feature_row = PLV2.run_config(params)
    return {"summary": summary, "vectors": vectors, "features": feature_row}


def _metrics_row_from_metrics(method: str, config_id: str, metrics: Any, logs: list[Any] | None, error: str | None = None) -> dict[str, Any]:
    row = {f"metric_{k}": v for k, v in asdict(metrics).items()}
    entropy_values = [log.weight_entropy for log in logs or []]
    fallback_values = [1.0 if log.fallback_used else 0.0 for log in logs or []]
    active_values = [
        sum(1 for w in log.expert_weights.values() if w > 0.0)
        for log in logs or []
    ]
    switching = max([log.switching_count for log in logs or []], default=0)
    row.update(
        {
            "config_id": config_id,
            "method": method,
            "switching_frequency": switching / max(len(logs or []), 1),
            "avg_active_experts": float(np.mean(active_values)) if active_values else 1.0,
            "weight_entropy": float(np.mean(entropy_values)) if entropy_values else 0.0,
            "fallback_frequency": float(np.mean(fallback_values)) if fallback_values else 0.0,
            "feasibility_violations": 0 if error is None else 1,
            "error": error or "",
        }
    )
    return row


def _run_method_worker(payload: dict[str, Any]) -> dict[str, Any]:
    params = payload["params"]
    method = payload["method"]
    weights = payload.get("weights", {})
    expert_names = payload.get("expert_names", [])
    config_id = str(params["config_id"])
    requests, gpu, _derived = PLV2.make_requests(params)
    budget = int(params["step_token_budget"])
    if method == "contextual_top2":
        policy = StaticRankEnsemblePolicy([RankExpertSpec(name, weights.get(name, 0.0)) for name in expert_names], top_k=2)
    elif method == "contextual_top3":
        policy = StaticRankEnsemblePolicy([RankExpertSpec(name, weights.get(name, 0.0)) for name in expert_names], top_k=3)
    elif method == "static_rank_ensemble":
        policy = StaticRankEnsemblePolicy([RankExpertSpec(name, 1.0) for name in expert_names])
    elif method == "component_wise":
        policy = ComponentWiseCompositionPolicy()
    elif method == "conditional_regime":
        policy = ConditionalRegimeCompositionPolicy(min_commitment_steps=2)
    else:
        raise ValueError(f"Unsupported method {method}")
    sim = Simulator(SimulatorConfig(gpu_configs=[gpu], service_model=_execution_service_model(budget, budget), drain_steps=20_000))
    sim.load_trace(requests)
    try:
        metrics = sim.run(policy, workload_tag=f"{config_id}_{method}", seed=int(params["seed"]))
        return _metrics_row_from_metrics(method, config_id, metrics, getattr(policy, "decision_logs", []))
    except Exception as exc:
        return {
            "config_id": config_id,
            "method": method,
            METRIC: float("nan"),
            "switching_frequency": 0.0,
            "avg_active_experts": 0.0,
            "weight_entropy": 0.0,
            "fallback_frequency": 1.0,
            "feasibility_violations": 1,
            "error": repr(exc),
        }


def _reward_pivot(vectors: pd.DataFrame) -> pd.DataFrame:
    return vectors.pivot_table(index="config_id", columns="policy_name", values=METRIC, aggfunc="first")


def _feature_matrix(features: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    cols = [c for c in features.columns if c.startswith("feat_")]
    matrix = features[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return matrix, cols


def _train_reward_models(features: pd.DataFrame, vectors: pd.DataFrame, dev_ids: list[str]) -> tuple[dict[str, Any], pd.DataFrame]:
    from sklearn.ensemble import RandomForestRegressor

    feature_matrix, cols = _feature_matrix(features)
    feature_matrix.index = features["config_id"].astype(str)
    rewards = _reward_pivot(vectors)
    models: dict[str, Any] = {}
    for policy in POLICY_LIBRARY_V2_NAMES:
        y = rewards.loc[dev_ids, policy].astype(float)
        model = RandomForestRegressor(n_estimators=64, min_samples_leaf=3, random_state=8801, n_jobs=1)
        model.fit(feature_matrix.loc[dev_ids, cols], y)
        models[policy] = model
    pred = pd.DataFrame(index=feature_matrix.index)
    for policy, model in models.items():
        pred[policy] = model.predict(feature_matrix[cols])
    return models, pred


def _advantage_weights(predicted: pd.Series, expert_names: list[str], top_k: int | None = None) -> dict[str, float]:
    values = predicted[expert_names].astype(float)
    if top_k is not None:
        selected = list(values.sort_values(ascending=False).head(top_k).index)
    else:
        selected = list(expert_names)
    selected_values = values[selected]
    shifted = selected_values - selected_values.min()
    shifted = shifted.clip(lower=0.0)
    if float(shifted.sum()) <= 1e-12:
        return {name: 1.0 / len(selected) for name in selected}
    return {name: float(shifted[name] / shifted.sum()) for name in selected}


def _method_metrics_for_vector_method(
    method: str,
    eval_ids: list[str],
    selected_policies: dict[str, str],
    rewards: pd.DataFrame,
    metric_extras: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []
    vectors_by_metric = metric_extras
    for config_id in eval_ids:
        policy = selected_policies[config_id]
        row = {
            "config_id": config_id,
            "method": method,
            METRIC: float(rewards.loc[config_id, policy]),
            "switching_frequency": 0.0,
            "avg_active_experts": 1.0,
            "weight_entropy": 0.0,
            "fallback_frequency": 0.0,
            "feasibility_violations": 0,
            "selected_policy": policy,
        }
        for metric_name, pivot in vectors_by_metric.items():
            if config_id in pivot.index and policy in pivot.columns:
                row[metric_name] = float(pivot.loc[config_id, policy])
        rows.append(row)
    return pd.DataFrame(rows)


def _bootstrap_ci(values: np.ndarray, seed: int = 20260721, reps: int = 500) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = [float(np.mean(rng.choice(values, size=len(values), replace=True))) for _ in range(reps)]
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def _p95(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan")
    return float(np.percentile(values, 95))


def _summarize_methods(all_methods: pd.DataFrame, window_info: pd.DataFrame, best_fixed_rewards: pd.Series, oracle: pd.Series) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    subset_rows: list[dict[str, Any]] = []
    merged = all_methods.merge(window_info, on="config_id", how="left")
    for method, part in merged.groupby("method"):
        rewards = part[METRIC].astype(float).to_numpy()
        regrets = oracle.loc[part["config_id"]].to_numpy() - rewards
        ci_lo, ci_hi = _bootstrap_ci(rewards)
        meaningful = part[part["meaningful"]]
        rows.append({
            "method": method,
            "n": len(part),
            "anwg": float(np.nanmean(rewards)),
            "anwg_ci95_low": ci_lo,
            "anwg_ci95_high": ci_hi,
            "completion_fraction": float(pd.to_numeric(part.get("metric_completion_fraction"), errors="coerce").mean()) if "metric_completion_fraction" in part else float("nan"),
            "completed_request_quality": float(pd.to_numeric(part.get("metric_weighted_goodput"), errors="coerce").mean()) if "metric_weighted_goodput" in part else float("nan"),
            "mean_oracle_regret": float(np.nanmean(regrets)),
            "p95_regret": _p95(regrets),
            "worst_regret": float(np.nanmax(regrets)),
            "meaningful_window_anwg": float(pd.to_numeric(meaningful[METRIC], errors="coerce").mean()) if len(meaningful) else float("nan"),
            "switching_frequency": float(pd.to_numeric(part["switching_frequency"], errors="coerce").mean()),
            "average_active_experts": float(pd.to_numeric(part["avg_active_experts"], errors="coerce").mean()),
            "weight_entropy": float(pd.to_numeric(part["weight_entropy"], errors="coerce").mean()),
            "fallback_frequency": float(pd.to_numeric(part["fallback_frequency"], errors="coerce").mean()),
            "feasibility_violations": int(pd.to_numeric(part["feasibility_violations"], errors="coerce").fillna(1).sum()),
        })
        for subset in ["meaningful", "wsp_dominant", "scorpio_dominant", "other_policy_dominant", "boundary", "near_tie", "high_regret"]:
            sub = part[part[subset]]
            if len(sub) == 0:
                continue
            sub_rewards = pd.to_numeric(sub[METRIC], errors="coerce")
            sub_regrets = oracle.loc[sub["config_id"]].to_numpy() - sub_rewards.to_numpy()
            subset_rows.append({
                "method": method,
                "subset": subset,
                "n": len(sub),
                "anwg": float(sub_rewards.mean()),
                "mean_oracle_regret": float(np.nanmean(sub_regrets)),
            })
    return rows, subset_rows


def _derive_window_info(eval_ids: list[str], rewards: pd.DataFrame, best_fixed_policy: str) -> pd.DataFrame:
    rows = []
    for config_id in eval_ids:
        values = rewards.loc[config_id, POLICY_LIBRARY_V2_NAMES].astype(float).sort_values(ascending=False)
        best_policy = str(values.index[0])
        second = float(values.iloc[1])
        best = float(values.iloc[0])
        margin = best - second
        delta = float(rewards.loc[config_id, SCORPIO] - rewards.loc[config_id, WSP])
        best_fixed_regret = best - float(rewards.loc[config_id, best_fixed_policy])
        rows.append({
            "config_id": config_id,
            "oracle_policy": best_policy,
            "oracle_anwg": best,
            "oracle_margin": margin,
            "delta_scorpio_wsp": delta,
            "meaningful": margin >= 0.002,
            "near_tie": margin < 0.001,
            "wsp_dominant": best_policy == WSP,
            "scorpio_dominant": best_policy == SCORPIO,
            "other_policy_dominant": best_policy not in {WSP, SCORPIO},
            "boundary": 0.001 <= abs(delta) <= 0.02,
            "best_fixed_regret": best_fixed_regret,
        })
    df = pd.DataFrame(rows)
    threshold = float(df["best_fixed_regret"].quantile(0.75)) if len(df) else float("inf")
    df["high_regret"] = df["best_fixed_regret"] >= threshold
    return df


def run_pilot(args: argparse.Namespace) -> int:
    root = args.run_root
    for name in ["logs", "manifests", "reports"]:
        (root / name).mkdir(parents=True, exist_ok=True)
    design = pd.read_csv(args.design_csv)
    selected = _sample_design(design, args.dev_count, args.eval_count, args.seed)
    selected.to_csv(root / "pilot_windows.csv", index=False)

    selected_params = [row.to_dict() for _, row in selected.iterrows()]
    config_results = []
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_run_config_worker, params) for params in selected_params]
        for future in as_completed(futures):
            config_results.append(future.result())

    summaries = pd.DataFrame([r["summary"] for r in config_results])
    vectors = pd.DataFrame([row for r in config_results for row in r["vectors"]])
    features = pd.DataFrame([r["features"] for r in config_results])
    summaries.to_csv(root / "policy_summary.csv", index=False)
    vectors.to_csv(root / "policy_vectors.csv", index=False)
    features.to_csv(root / "features.csv", index=False)

    roles = selected[["config_id", "pilot_role", "split_group"]].copy()
    dev_ids = roles.loc[roles["pilot_role"] == "development", "config_id"].astype(str).tolist()
    eval_ids = roles.loc[roles["pilot_role"] == "heldout", "config_id"].astype(str).tolist()
    rewards = _reward_pivot(vectors)
    metric_extras = {
        "metric_completion_fraction": vectors.pivot_table(index="config_id", columns="policy_name", values="metric_completion_fraction", aggfunc="first"),
        "metric_weighted_goodput": vectors.pivot_table(index="config_id", columns="policy_name", values="metric_weighted_goodput", aggfunc="first"),
        "metric_slo_violation_rate": vectors.pivot_table(index="config_id", columns="policy_name", values="metric_slo_violation_rate", aggfunc="first"),
    }
    dev_means = rewards.loc[dev_ids, POLICY_LIBRARY_V2_NAMES].mean(axis=0).sort_values(ascending=False)
    best_fixed_policy = str(dev_means.index[0])
    new_dev = dev_means[[p for p in POLICY_LIBRARY_V2_NEW_NAMES if p in dev_means.index]].sort_values(ascending=False)
    expert_set = [WSP, SCORPIO, "edf"]
    for policy in list(new_dev.head(2).index):
        if policy not in expert_set:
            expert_set.append(str(policy))
    for fallback in ["kv_constrained_online", "adaptive_chunked_prefill"]:
        if len(expert_set) >= 5:
            break
        if fallback not in expert_set:
            expert_set.append(fallback)

    _models, predicted = _train_reward_models(features, vectors, dev_ids)
    selected_discrete = {config_id: str(predicted.loc[config_id, POLICY_LIBRARY_V2_NAMES].idxmax()) for config_id in eval_ids}
    selected_top1 = {config_id: str(predicted.loc[config_id, expert_set].idxmax()) for config_id in eval_ids}

    method_frames = [
        _method_metrics_for_vector_method("best_fixed", eval_ids, {config_id: best_fixed_policy for config_id in eval_ids}, rewards, metric_extras),
        _method_metrics_for_vector_method("discrete_selector", eval_ids, selected_discrete, rewards, metric_extras),
        _method_metrics_for_vector_method("contextual_top1", eval_ids, selected_top1, rewards, metric_extras),
    ]

    composition_tasks = []
    weights_rows = []
    params_by_id = {str(row["config_id"]): row.to_dict() for _, row in selected.iterrows()}
    for config_id in eval_ids:
        pred_row = predicted.loc[config_id]
        for method, top_k in [("contextual_top2", 2), ("contextual_top3", 3)]:
            weights = _advantage_weights(pred_row, expert_set, top_k=top_k)
            weights_rows.extend([
                {"config_id": config_id, "method": method, "expert": expert, "weight": weight}
                for expert, weight in weights.items()
            ])
            composition_tasks.append({"params": params_by_id[config_id], "method": method, "weights": weights, "expert_names": list(weights)})
        composition_tasks.append({"params": params_by_id[config_id], "method": "static_rank_ensemble", "expert_names": expert_set})
        composition_tasks.append({"params": params_by_id[config_id], "method": "component_wise"})
        composition_tasks.append({"params": params_by_id[config_id], "method": "conditional_regime"})

    composition_rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_run_method_worker, task) for task in composition_tasks]
        for future in as_completed(futures):
            composition_rows.append(future.result())
    comp_df = pd.DataFrame(composition_rows)
    method_frames.append(comp_df)
    all_methods = pd.concat(method_frames, ignore_index=True)

    window_info = _derive_window_info(eval_ids, rewards, best_fixed_policy)
    oracle_with_composition = rewards.loc[eval_ids, POLICY_LIBRARY_V2_NAMES].max(axis=1)
    for method in ["contextual_top2", "contextual_top3", "static_rank_ensemble", "component_wise", "conditional_regime"]:
        vals = all_methods[all_methods["method"] == method].set_index("config_id")[METRIC].astype(float)
        oracle_with_composition = pd.concat([oracle_with_composition, vals], axis=1).max(axis=1)
    best_fixed_rewards = rewards.loc[eval_ids, best_fixed_policy].astype(float)
    comparison, subset_rows = _summarize_methods(all_methods, window_info, best_fixed_rewards, oracle_with_composition)
    comparison_df = pd.DataFrame(comparison).sort_values("anwg", ascending=False)
    comparison_df.to_csv(root / "method_comparison.csv", index=False)
    pd.DataFrame(subset_rows).to_csv(root / "subset_analysis.csv", index=False)
    pd.DataFrame(weights_rows).to_csv(root / "composition_weights.csv", index=False)

    eval_methods_wide = all_methods.pivot_table(index="config_id", columns="method", values=METRIC, aggfunc="first")
    pilot_windows = selected.merge(window_info, on="config_id", how="left")
    pilot_windows = pilot_windows.merge(eval_methods_wide.reset_index(), on="config_id", how="left")
    pilot_windows.to_csv(root / "pilot_windows.csv", index=False)

    composition_methods = ["contextual_top1", "contextual_top2", "contextual_top3", "static_rank_ensemble", "component_wise", "conditional_regime"]
    comp_subset = comparison_df[comparison_df["method"].isin(composition_methods)]
    best_comp = comp_subset.iloc[0].to_dict() if len(comp_subset) else {}
    method_lookup = comparison_df.set_index("method").to_dict(orient="index")
    meaningful_ids = window_info.loc[window_info["meaningful"], "config_id"].tolist()
    meaningful_scores = {}
    for method in comparison_df["method"]:
        part = all_methods[(all_methods["method"] == method) & (all_methods["config_id"].isin(meaningful_ids))]
        meaningful_scores[method] = float(pd.to_numeric(part[METRIC], errors="coerce").mean()) if len(part) else float("nan")
    best_meaningful_comp_method = max(composition_methods, key=lambda m: meaningful_scores.get(m, float("-inf")))
    beats_discrete = (
        meaningful_scores.get(best_meaningful_comp_method, float("-inf"))
        > meaningful_scores.get("discrete_selector", float("inf")) + args.meaningful_margin
    )
    beats_fixed = (
        meaningful_scores.get(best_meaningful_comp_method, float("-inf"))
        > meaningful_scores.get("best_fixed", float("inf")) + args.meaningful_margin
    )
    component_gain = (
        meaningful_scores.get("component_wise", float("-inf"))
        > max(meaningful_scores.get("best_fixed", float("inf")), meaningful_scores.get("discrete_selector", float("inf"))) + args.meaningful_margin
    )
    top1_best_comp = best_meaningful_comp_method == "contextual_top1"
    expands_frontier = bool(
        (all_methods[all_methods["method"].isin(composition_methods)].set_index("config_id")[METRIC] > rewards.loc[eval_ids, POLICY_LIBRARY_V2_NAMES].max(axis=1).reindex(all_methods[all_methods["method"].isin(composition_methods)]["config_id"]).to_numpy() + args.meaningful_margin).any()
    )
    if (component_gain or (beats_fixed and beats_discrete and not top1_best_comp)) or expands_frontier:
        decision = "GO"
    elif top1_best_comp or not beats_discrete or not component_gain:
        decision = "NO_GO"
    else:
        decision = "INCONCLUSIVE"

    fields = {
        "NATIVE_COMPOSITION_PILOT_DECISION": decision,
        "PILOT_WINDOW_COUNT": len(eval_ids),
        "DEVELOPMENT_WINDOW_COUNT": len(dev_ids),
        "MEANINGFUL_WINDOW_COUNT": int(window_info["meaningful"].sum()),
        "EXPERT_SET": expert_set,
        "BEST_FIXED_METHOD": best_fixed_policy,
        "BEST_FIXED_ANWG": method_lookup.get("best_fixed", {}).get("anwg"),
        "DISCRETE_SELECTOR_ANWG": method_lookup.get("discrete_selector", {}).get("anwg"),
        "CONTEXTUAL_TOP1_ANWG": method_lookup.get("contextual_top1", {}).get("anwg"),
        "CONTEXTUAL_TOP2_ANWG": method_lookup.get("contextual_top2", {}).get("anwg"),
        "CONTEXTUAL_TOP3_ANWG": method_lookup.get("contextual_top3", {}).get("anwg"),
        "STATIC_ENSEMBLE_ANWG": method_lookup.get("static_rank_ensemble", {}).get("anwg"),
        "COMPONENT_WISE_COMPOSITION_NAME": "component_wise_scorpio_admission_wsp_priority_kv_prefill_aging",
        "COMPONENT_WISE_COMPOSITION_ANWG": method_lookup.get("component_wise", {}).get("anwg"),
        "CONDITIONAL_COMPOSITION_ANWG": method_lookup.get("conditional_regime", {}).get("anwg"),
        "BEST_COMPOSITION_METHOD": best_comp.get("method"),
        "BEST_COMPOSITION_ANWG": best_comp.get("anwg"),
        "COMPOSITION_BEATS_DISCRETE_SELECTOR": "YES" if beats_discrete else "NO",
        "COMPOSITION_EXPANDS_FRONTIER": "YES" if expands_frontier else "NO",
        "FEASIBILITY_VIOLATIONS": int(all_methods["feasibility_violations"].sum()),
        "FULL_WULVER_COMPOSITION_RUN_JUSTIFIED": "YES" if decision == "GO" else ("UNCLEAR" if decision == "INCONCLUSIVE" else "NO"),
        "MOST_IMPORTANT_FINDING": "",
        "RECOMMENDED_NEXT_ACTION": "",
        "runtime_s": time.time() - started,
    }
    if decision == "GO":
        fields["MOST_IMPORTANT_FINDING"] = "A composition treatment beat both best fixed and discrete selector on held-out meaningful windows or expanded the policy frontier."
        fields["RECOMMENDED_NEXT_ACTION"] = "Submit the full composition workflow after upstream frontier/library reports finish."
    else:
        fields["MOST_IMPORTANT_FINDING"] = "Native component/sparse composition did not clear the held-out meaningful-window bar against the discrete selector."
        fields["RECOMMENDED_NEXT_ACTION"] = "Do not launch the full composition sweep; prioritize structural symbolic synthesis, targeted evolutionary crossover, or conditional DSL generation from high-weight parent policies."

    component_specs = {
        "component_wise": {
            "admission": "SCORPIO-style causal admission and budget",
            "priority": "WSP-style predicted service / priority rank",
            "kv_guard": "KV reserve placement guard",
            "prefill_guard": "adaptive long-prefill pressure guard",
            "aging": "age tiebreak",
        },
        "conditional_regime": {
            "high_pressure": "component_wise",
            "low_pressure": "weighted_shortest_processing",
            "pressure_features": ["queue_pressure", "kv_pressure", "urgent_deadline_fraction"],
        },
    }
    write_json(root / "component_compositions.json", component_specs)
    write_json(root / "pilot_manifest.json", {
        **fields,
        "root": str(root),
        "git_commit": git(["rev-parse", "HEAD"]),
        "git_branch": git(["branch", "--show-current"]),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "policy_library_v2_count": len(POLICY_LIBRARY_V2_NAMES),
        "policy_library_v2_new_names": list(POLICY_LIBRARY_V2_NEW_NAMES),
        "selection_rule": "best fixed, expert policies, and selector models selected from development windows only",
        "heldout_splits": sorted(roles.loc[roles["pilot_role"] == "heldout", "split_group"].unique()),
    })
    report = ["# Native Wulver Composition Falsification Pilot", ""]
    for key, value in fields.items():
        report.append(f"{key} = {value}")
    report += [
        "",
        "## Expert Set",
        ", ".join(expert_set),
        "",
        "## Method Comparison",
        "```",
        comparison_df.to_string(index=False),
        "```",
        "",
        "## Notes",
        "- Full Wulver composition experiment was not submitted.",
        "- Development windows were used only for freezing best fixed policy, expert set, and reward models.",
        "- Held-out labels were used only for final pilot evaluation and subset reporting.",
    ]
    (root / "pilot_report.md").write_text("\n".join(report) + "\n")
    print(json.dumps(fields, indent=2, sort_keys=True, default=_json_default))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--design-csv", type=Path, default=Path("/mmfs1/project/ikoutis/sv96/llmserveopt-data/policy_library_v2_expanded_20260721T171933Z/design/policy_library_v2_design.csv"))
    parser.add_argument("--dev-count", type=int, default=35)
    parser.add_argument("--eval-count", type=int, default=65)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--meaningful-margin", type=float, default=0.002)
    args = parser.parse_args()
    return run_pilot(args)


if __name__ == "__main__":
    raise SystemExit(main())
