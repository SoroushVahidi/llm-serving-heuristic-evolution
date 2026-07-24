#!/usr/bin/env python3
"""Run module-credit evaluation on imported Wulver intervention artifacts.

This adapts the Wulver CSV layout into the local canonical module-credit row
schema, then uses the existing module_credit models and evaluators. Imported
artifacts are read-only inputs; all derived files are written under --out-dir.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from llmserveopt.policies.registry import POLICY_LIBRARY_V2_NAMES  # noqa: E402
from llmserveopt.selector.module_credit import (  # noqa: E402
    ModuleCreditModel,
    ModuleGateConfig,
    build_intervention_dataset,
    evaluate_credit_predictions,
    evaluate_offline_synthesis_decisions,
    evaluate_topk_ranking,
    gate_candidates,
)
from llmserveopt.selector.module_credit import dataset as module_credit_dataset  # noqa: E402
from llmserveopt.selector.module_credit import encoders as module_credit_encoders  # noqa: E402
from llmserveopt.selector.suitability.dataset import rows_with_reward  # noqa: E402
from llmserveopt.selector.suitability.models import JointRewardModel  # noqa: E402

module_credit_dataset.map_policy_to_genome = lru_cache(maxsize=None)(module_credit_dataset.map_policy_to_genome)
module_credit_encoders.map_policy_to_genome = lru_cache(maxsize=None)(module_credit_encoders.map_policy_to_genome)


MODULE_MAP = {
    "admission": "admission_rule",
    "priority": "priority_rule",
    "prefill": "prefill_rule",
    "kv_guard": "kv_guard",
    "fairness_aging": "fairness_rule",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        default="results/wulver_imports/module_intervention_credit_20260721T224322Z",
    )
    parser.add_argument("--out-dir", default="results/module_credit_report/real_wulver_20260721T224322Z")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lambda-m", type=float, default=0.5)
    args = parser.parse_args()

    t0 = time.perf_counter()
    artifact_root = (ROOT / args.artifact_root).resolve() if not Path(args.artifact_root).is_absolute() else Path(args.artifact_root)
    out_dir = (ROOT / args.out_dir).resolve() if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tables = load_required_tables(artifact_root)
    artifact_map = map_artifacts(artifact_root)
    validation = validate_artifacts(artifact_root, tables)

    raw_rows, split_summary = build_raw_rows(tables, seed=args.seed)
    suitability_prior = fit_suitability_prior(args.seed)
    rows = build_intervention_dataset(raw_rows, suitability_model=suitability_prior, suitability_lambda=args.lambda_m)
    split_rows = {
        "TRAIN": [r for r in rows if r["split"] == "TRAIN"],
        "VALIDATION": [r for r in rows if r["split"] == "VALIDATION"],
        "TEST": [r for r in rows if r["split"] == "TEST"],
    }

    model_specs = {
        "identity": "identity",
        "structural": "structural",
        "state_conditioned_structural": "contextual",
        "suitability_augmented": "suitability_augmented",
    }
    models = {
        name: credit_model(name=name, encoding=encoding, target="C_base", seed=args.seed).fit(split_rows["TRAIN"])
        for name, encoding in model_specs.items()
    }
    prediction = {
        name: {
            "validation": evaluate_credit_predictions(model, split_rows["VALIDATION"], target="C_base"),
            "test": evaluate_credit_predictions(model, split_rows["TEST"], target="C_base"),
        }
        for name, model in models.items()
    }
    winner = min(prediction, key=lambda n: prediction[n]["validation"]["mae"])

    target_results = evaluate_targets(model_specs, split_rows, args.seed)
    ranking = {
        name: {
            "test_selection_quality": evaluate_topk_ranking(model, split_rows["TEST"], lambda_m=args.lambda_m),
            "test_ranking_quality": ranking_quality(model, split_rows["TEST"], lambda_m=args.lambda_m),
        }
        for name, model in models.items()
    }
    offline = {
        name: evaluate_offline_synthesis_decisions(split_rows["TEST"], model, lambda_m=args.lambda_m, seed=args.seed)
        for name, model in models.items()
    }
    suitability_delta = {
        "validation_mae_delta_suitability_minus_contextual": (
            prediction["suitability_augmented"]["validation"]["mae"]
            - prediction["state_conditioned_structural"]["validation"]["mae"]
        ),
        "test_mae_delta_suitability_minus_contextual": (
            prediction["suitability_augmented"]["test"]["mae"]
            - prediction["state_conditioned_structural"]["test"]["mae"]
        ),
        "improves_validation": (
            prediction["suitability_augmented"]["validation"]["mae"]
            < prediction["state_conditioned_structural"]["validation"]["mae"]
        ),
        "improves_test": (
            prediction["suitability_augmented"]["test"]["mae"]
            < prediction["state_conditioned_structural"]["test"]["mae"]
        ),
    }

    held_out_donors = held_out_donor_analysis(rows, args.seed, args.lambda_m)
    held_out_modules = held_out_module_type_analysis(rows, args.seed)
    uncertainty = uncertainty_aware_ranking(models[winner], split_rows["TEST"])
    gate = offline_gate_analysis(models[winner], split_rows["TEST"], args.lambda_m)
    pairwise = pairwise_analysis(artifact_root, tables)
    observed_credit = observed_credit_summary(split_rows["TEST"])

    status = status_from_real_results(prediction, ranking, winner)
    readiness = readiness_from_results(status, validation, pairwise, gate)
    wulver_report_compare = compare_wulver_report(artifact_root)

    report = {
        "artifact_root": str(artifact_root),
        "out_dir": str(out_dir),
        "artifact_map": artifact_map,
        "workflow_completion": validation["workflow_completion"],
        "schema_and_row_validation": validation,
        "split_summary": split_summary,
        "canonical_rows": {
            "n_rows": len(rows),
            "n_train_rows": len(split_rows["TRAIN"]),
            "n_validation_rows": len(split_rows["VALIDATION"]),
            "n_test_rows": len(split_rows["TEST"]),
            "module_types": dict(Counter(r["module_type"] for r in rows)),
            "base_policies": dict(Counter(r["base_policy"] for r in rows)),
            "donor_policies": dict(Counter(r["donor_policy"] for r in rows)),
        },
        "prediction": prediction,
        "target_results": target_results,
        "winning_model_by_validation_mae": winner,
        "whole_policy_suitability_effect": suitability_delta,
        "ranking": ranking,
        "offline_synthesis": offline,
        "realized_credit_on_test": observed_credit,
        "uncertainty_aware_ranking": uncertainty,
        "held_out_donors": held_out_donors,
        "held_out_modules": held_out_modules,
        "pairwise_interactions": pairwise,
        "synthesis_gate_offline": gate,
        "wulver_final_report_comparison": wulver_report_compare,
        "MODULE_CREDIT_MODEL_STATUS": status,
        "STRUCTURAL_SYNTHESIS_READINESS": readiness,
        "REAL_MODULE_CREDIT_EVALUATION": "PASS" if validation["core_complete"] and readiness != "NOT_READY" else "NEEDS_ATTENTION",
        "runtime_s": round(time.perf_counter() - t0, 3),
    }

    (out_dir / "real_module_credit_results.json").write_text(json.dumps(report, indent=2, default=json_default))
    (out_dir / "real_module_credit_report.md").write_text(render_aw_report(report))
    print(json.dumps({
        "n_rows": len(rows),
        "split_rows": {k: len(v) for k, v in split_rows.items()},
        "winning_model": winner,
        "MODULE_CREDIT_MODEL_STATUS": status,
        "STRUCTURAL_SYNTHESIS_READINESS": readiness,
        "REAL_MODULE_CREDIT_EVALUATION": report["REAL_MODULE_CREDIT_EVALUATION"],
        "results": str(out_dir / "real_module_credit_results.json"),
        "report": str(out_dir / "real_module_credit_report.md"),
    }, indent=2))
    return 0


def load_required_tables(root: Path) -> dict[str, pd.DataFrame]:
    paths = {
        "single": root / "combined" / "single_module_credit.csv",
        "features": root / "combined" / "features.csv",
        "reward_vectors": root / "combined" / "reward_vectors.csv",
        "workload_design": root / "design" / "workload_design.csv",
        "intervention_definitions": root / "design" / "intervention_definitions.csv",
        "pairwise_definitions": root / "design" / "pairwise_intervention_definitions.csv",
        "module_credit_summary": root / "diagnostics" / "module_credit_summary.csv",
        "module_registry": root / "diagnostics" / "module_registry.csv",
        "reconstruction_fidelity": root / "fidelity" / "reconstruction_fidelity.csv",
        "smoke": root / "smoke" / "intervention_smoke_results.csv",
    }
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required imported artifact(s): {missing}")
    return {name: pd.read_csv(path) for name, path in paths.items()}


def map_artifacts(root: Path) -> dict[str, Any]:
    out = {}
    for dirname in ["combined", "diagnostics", "fidelity", "manifests", "reports", "design", "smoke"]:
        directory = root / dirname
        files = sorted(p.relative_to(root).as_posix() for p in directory.glob("*") if p.is_file()) if directory.exists() else []
        out[dirname] = {"exists": directory.exists(), "files": files}
    return out


def validate_artifacts(root: Path, tables: Mapping[str, pd.DataFrame]) -> dict[str, Any]:
    manifests = {}
    workflow = {}
    non_status_manifests = []
    for path in sorted((root / "manifests").glob("*.json")):
        payload = json.loads(path.read_text())
        manifests[path.name] = payload
        if "status" in payload:
            workflow[path.stem] = payload.get("status")
        else:
            non_status_manifests.append(path.name)

    single_files = sorted((root / "shards").glob("single_task_*/single_module_credit.csv"))
    pairwise_files = sorted((root / "shards").glob("pairwise_task_*/pairwise_credit.csv"))
    pairwise_df = concat_csvs(pairwise_files)
    expected_single = len(tables["features"]) * len(tables["intervention_definitions"])
    expected_reward_vectors = len(tables["features"]) * (
        len(tables["intervention_definitions"]) + tables["reward_vectors"].query("policy_kind == 'native_v2'")["policy_name"].nunique()
    )
    required_single_cols = {
        "config_id", "regime", "split_group", "intervention_id", "base_policy",
        "donor_policy", "module_type", "reward_base", "reward_donor",
        "reward_intervention", "v2_envelope",
    }
    required_feature_cols = {"config_id", "regime"}
    required_pairwise_cols = {
        "config_id", "pairwise_id", "base_policy", "first_module", "second_module",
        "first_reward", "second_reward", "both_reward", "base_reward", "interaction",
    }
    schema_errors = []
    if missing := sorted(required_single_cols - set(tables["single"].columns)):
        schema_errors.append(f"single missing columns: {missing}")
    if missing := sorted(required_feature_cols - set(tables["features"].columns)):
        schema_errors.append(f"features missing columns: {missing}")
    if len(pairwise_df) and (missing := sorted(required_pairwise_cols - set(pairwise_df.columns))):
        schema_errors.append(f"pairwise missing columns: {missing}")

    row_counts = {
        "combined_single_rows": len(tables["single"]),
        "expected_single_rows": expected_single,
        "combined_features_rows": len(tables["features"]),
        "combined_reward_vector_rows": len(tables["reward_vectors"]),
        "expected_reward_vector_rows": expected_reward_vectors,
        "single_shard_files": len(single_files),
        "single_shard_rows": int(sum(len(pd.read_csv(p)) for p in single_files)),
        "pairwise_shard_files": len(pairwise_files),
        "pairwise_shard_rows": len(pairwise_df),
        "pairwise_observed_ids": int(pairwise_df["pairwise_id"].nunique()) if len(pairwise_df) else 0,
        "pairwise_defined_ids": int(tables["pairwise_definitions"]["pairwise_id"].nunique()),
    }
    workflow_completion = {
        "manifest_statuses": workflow,
        "non_status_manifests": non_status_manifests,
        "all_status_bearing_manifests_pass": all(v == "PASS" for v in workflow.values()),
        "single_shards_complete": row_counts["single_shard_rows"] == expected_single,
        "combined_single_complete": len(tables["single"]) == expected_single,
        "pairwise_complete_against_design": (
            row_counts["pairwise_observed_ids"] == row_counts["pairwise_defined_ids"]
            and row_counts["pairwise_shard_rows"] == len(tables["features"]) * row_counts["pairwise_defined_ids"]
        ),
    }
    return {
        "schemas_valid": not schema_errors,
        "schema_errors": schema_errors,
        "row_counts": row_counts,
        "workflow_completion": workflow_completion,
        "core_complete": (
            workflow_completion["all_status_bearing_manifests_pass"]
            and workflow_completion["combined_single_complete"]
            and workflow_completion["single_shards_complete"]
            and not schema_errors
        ),
        "manifests": manifests,
    }


def build_raw_rows(tables: Mapping[str, pd.DataFrame], *, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    features = tables["features"].set_index("config_id")
    design = tables["workload_design"].set_index("config_id")
    registry = tables["module_registry"].set_index("policy_name")
    split_by_config = split_configs(tables["workload_design"], seed=seed)
    numeric_feature_cols = [
        c for c in tables["features"].columns
        if c not in {"config_id", "regime"} and pd.api.types.is_numeric_dtype(tables["features"][c])
    ]
    rows = []
    for _, row in tables["single"].iterrows():
        config_id = str(row["config_id"])
        module_type_raw = str(row["module_type"])
        module_type = MODULE_MAP.get(module_type_raw, module_type_raw)
        state_features = {
            f"feat_{col}": finite_float(features.at[config_id, col], default=0.0)
            for col in numeric_feature_cols
        }
        base_policy = str(row["base_policy"])
        donor_policy = str(row["donor_policy"])
        raw = {
            "state_id": config_id,
            "state_features": state_features,
            "base_policy": base_policy,
            "donor_policy": donor_policy,
            "module_type": module_type,
            "base_reward": finite_float(row["reward_base"]),
            "donor_reward": finite_float(row["reward_donor"]),
            "intervention_reward": finite_float(row["reward_intervention"]),
            "library_best_reward": finite_float(row["v2_envelope"]),
            "source": "wulver_module_intervention_credit_20260721T224322Z",
            "trace_family": str(row["regime"]),
            "temporal_block": config_id,
            "split": split_by_config[config_id],
            "seed": int(design.at[config_id, "seed"]),
            "split_group_key": config_id,
            "dataset_family": str(row["split_group"]),
            "window_idx": int(config_id.split("_")[-1]),
            "request_plan_ancestor_id": config_id,
            "intervention_id": str(row["intervention_id"]),
            "child_name": str(row["child_name"]),
            "base_module_representation": module_repr(registry, base_policy, module_type_raw),
            "donor_module_representation": module_repr(registry, donor_policy, module_type_raw),
            "compatibility_metadata": compatibility(registry, base_policy, donor_policy, module_type_raw),
        }
        rows.append(raw)
    split_summary = {
        "config_split_counts": dict(Counter(split_by_config.values())),
        "row_split_counts": dict(Counter(r["split"] for r in rows)),
        "config_splits_by_regime": config_splits_by_regime(tables["workload_design"], split_by_config),
    }
    return rows, split_summary


def split_configs(design: pd.DataFrame, *, seed: int) -> dict[str, str]:
    del seed
    out = {}
    for regime, group in design.sort_values("config_id").groupby("regime"):
        dev = [str(c) for c in group[group["split_group"] == "development"]["config_id"]]
        val = set(dev[-2:])
        for config_id in dev:
            out[config_id] = "VALIDATION" if config_id in val else "TRAIN"
        held = [str(c) for c in group[group["split_group"] == "heldout"]["config_id"]]
        for config_id in held:
            out[config_id] = "TEST"
    return out


def config_splits_by_regime(design: pd.DataFrame, split_by_config: Mapping[str, str]) -> dict[str, dict[str, int]]:
    out = {}
    for regime, group in design.groupby("regime"):
        out[str(regime)] = dict(Counter(split_by_config[str(c)] for c in group["config_id"]))
    return out


def module_repr(registry: pd.DataFrame, policy: str, module_type_raw: str) -> dict[str, float]:
    if policy not in registry.index:
        return {"module_present": 0.0, "module_substitutable": 0.0}
    record = registry.loc[policy]
    value = record.get(module_type_raw, "")
    substitutable = bool(record.get(f"{module_type_raw}_substitutable", False))
    status = str(record.get("reconstruction_status", "UNSUPPORTED"))
    return {
        "module_present": float(pd.notna(value) and str(value) != ""),
        "module_substitutable": float(substitutable),
        "module_status_exact_candidate": float(status == "EXACT_CANDIDATE"),
        "module_status_partial": float(status == "PARTIAL"),
        "module_status_unsupported": float(status == "UNSUPPORTED"),
        f"module_name_hash_{stable_small_hash(str(value))}": 1.0 if pd.notna(value) and str(value) else 0.0,
    }


def compatibility(registry: pd.DataFrame, base: str, donor: str, module_type_raw: str) -> dict[str, float]:
    base_repr = module_repr(registry, base, module_type_raw)
    donor_repr = module_repr(registry, donor, module_type_raw)
    compatible = donor_repr["module_present"] > 0.0 and donor_repr["module_substitutable"] > 0.0
    distance = 0.0
    keys = set(base_repr) | set(donor_repr)
    if keys:
        distance = float(sum(abs(base_repr.get(k, 0.0) - donor_repr.get(k, 0.0)) for k in keys) / len(keys))
    return {
        "compatible": float(compatible),
        "donor_module_present": donor_repr["module_present"],
        "base_module_present": base_repr["module_present"],
        "donor_substitutable": donor_repr["module_substitutable"],
        "base_substitutable": base_repr["module_substitutable"],
        "structural_distance": distance,
    }


def fit_suitability_prior(seed: int) -> JointRewardModel | None:
    fixture_path = ROOT / "results" / "state_policy_suitability_fixture" / "report_run_v2" / "long_format_rows.json"
    if not fixture_path.exists():
        return None
    rows = rows_with_reward(json.loads(fixture_path.read_text()))
    train = [r for r in rows if r["split"] in ("TRAIN", "VALIDATION")]
    if not train:
        return None
    return JointRewardModel(
        name="real_module_credit_prior",
        encoding="hybrid",
        all_policies=POLICY_LIBRARY_V2_NAMES,
        random_state=seed,
        n_estimators=80,
    ).fit(train)


def credit_model(*, name: str, encoding: str, target: str, seed: int) -> ModuleCreditModel:
    return ModuleCreditModel(
        name=name,
        encoding=encoding,
        target=target,
        random_state=seed,
        n_estimators=16,
        max_depth=6,
        min_samples_leaf=2,
    )


def evaluate_targets(model_specs: Mapping[str, str], split_rows: Mapping[str, Sequence[Mapping[str, Any]]], seed: int) -> dict[str, Any]:
    out = {}
    for target in ["C_base", "C_parent", "C_env"]:
        out[target] = {}
        for name, encoding in model_specs.items():
            model = credit_model(name=f"{name}_{target}", encoding=encoding, target=target, seed=seed).fit(split_rows["TRAIN"])
            out[target][name] = {
                "validation": evaluate_credit_predictions(model, split_rows["VALIDATION"], target=target),
                "test": evaluate_credit_predictions(model, split_rows["TEST"], target=target),
            }
    return out


def ranking_quality(model: ModuleCreditModel, rows: Sequence[Mapping[str, Any]], *, lambda_m: float) -> dict[str, Any]:
    groups = compatible_groups(rows)
    scores = {id(row): float(score) for row, score in zip(rows, model.predict_score(rows, lambda_m=lambda_m))}
    out = {}
    for k in [1, 3, 5]:
        hits = []
        regrets = []
        selected_best_actual = []
        oracle = []
        for candidates in groups.values():
            best = max(candidates, key=lambda r: float(r["C_base"]))
            ordered = sorted(candidates, key=lambda r: scores[id(r)], reverse=True)
            topk = ordered[: min(k, len(ordered))]
            top1 = ordered[0]
            best_in_topk = max(topk, key=lambda r: float(r["C_base"]))
            hits.append(best in topk)
            regrets.append(float(best["C_base"]) - float(top1["C_base"]))
            selected_best_actual.append(float(best_in_topk["C_base"]))
            oracle.append(float(best["C_base"]))
        out[f"top_{k}"] = summarize_rank_lists(hits, regrets, selected_best_actual, oracle)
    return out


def summarize_rank_lists(hits: Sequence[bool], regrets: Sequence[float], selected_best: Sequence[float], oracle: Sequence[float]) -> dict[str, Any]:
    if not hits:
        return {"n_groups": 0}
    return {
        "n_groups": len(hits),
        "contains_true_best_fraction": float(np.mean(hits)),
        "mean_top1_regret_vs_oracle": float(np.mean(regrets)),
        "mean_best_realized_C_base_within_topk": float(np.mean(selected_best)),
        "mean_oracle_C_base": float(np.mean(oracle)),
    }


def compatible_groups(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], list[Mapping[str, Any]]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        compat = row.get("compatibility_metadata", {})
        if isinstance(compat, Mapping) and float(compat.get("compatible", 1.0)) <= 0.0:
            continue
        groups[(str(row["state_id"]), str(row["base_policy"]))].append(row)
    return groups


def uncertainty_aware_ranking(model: ModuleCreditModel, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        f"lambda_{lam:g}": {
            "selection_quality": evaluate_topk_ranking(model, rows, lambda_m=lam),
            "ranking_quality": ranking_quality(model, rows, lambda_m=lam),
        }
        for lam in [0.0, 0.25, 0.5, 1.0]
    }


def held_out_donor_analysis(rows: Sequence[Mapping[str, Any]], seed: int, lambda_m: float) -> dict[str, Any]:
    out = {}
    trainval = [r for r in rows if r["split"] in {"TRAIN", "VALIDATION"}]
    test = [r for r in rows if r["split"] == "TEST"]
    for donor in sorted({r["donor_policy"] for r in rows}):
        train_d = [r for r in trainval if r["donor_policy"] != donor]
        test_d = [r for r in test if r["donor_policy"] == donor]
        if len(train_d) < 20 or len(test_d) < 2:
            out[donor] = {"skipped": True, "n_train_excluding_donor": len(train_d), "n_test_donor": len(test_d)}
            continue
        model = credit_model(name=f"held_out_donor_{donor}", encoding="suitability_augmented", target="C_base", seed=seed).fit(train_d)
        out[donor] = {
            "skipped": False,
            "n_train_excluding_donor": len(train_d),
            "n_test_donor": len(test_d),
            "prediction": evaluate_credit_predictions(model, test_d, target="C_base"),
            "ranking": evaluate_topk_ranking(model, test_d, lambda_m=lambda_m, ks=(1, 3, 5)),
            "ranking_quality": ranking_quality(model, test_d, lambda_m=lambda_m),
        }
    if "edf" not in out:
        out["edf"] = {
            "skipped": True,
            "reason": "EDF is not present as a donor in the real single-module intervention rows; it appears only in the incomplete pairwise design/artifacts.",
            "n_train_excluding_donor": len(trainval),
            "n_test_donor": 0,
        }
    return out


def held_out_module_type_analysis(rows: Sequence[Mapping[str, Any]], seed: int) -> dict[str, Any]:
    out = {}
    trainval = [r for r in rows if r["split"] in {"TRAIN", "VALIDATION"}]
    test = [r for r in rows if r["split"] == "TEST"]
    for module_type in sorted({r["module_type"] for r in rows}):
        train_m = [r for r in trainval if r["module_type"] != module_type]
        test_m = [r for r in test if r["module_type"] == module_type]
        if len(train_m) < 20 or len(test_m) < 2:
            out[module_type] = {"skipped": True, "n_train_excluding_module": len(train_m), "n_test_module": len(test_m)}
            continue
        model = credit_model(name=f"held_out_module_{module_type}", encoding="suitability_augmented", target="C_base", seed=seed).fit(train_m)
        out[module_type] = {
            "skipped": False,
            "n_train_excluding_module": len(train_m),
            "n_test_module": len(test_m),
            "prediction": evaluate_credit_predictions(model, test_m, target="C_base"),
        }
    return out


def offline_gate_analysis(model: ModuleCreditModel, rows: Sequence[Mapping[str, Any]], lambda_m: float) -> dict[str, Any]:
    scored = gate_candidates(model, rows, ModuleGateConfig(lambda_m=lambda_m, max_uncertainty=0.5))
    passed = [s for s in scored if s["passes"]]
    failed_reasons = Counter(reason for s in scored for reason in s["reasons"])
    return {
        "n_candidates": len(scored),
        "n_passed": len(passed),
        "pass_rate": float(len(passed) / len(scored)) if scored else None,
        "failed_reasons": dict(failed_reasons),
        "passed_realized": summarize_selected([s["row"] for s in passed]),
    }


def pairwise_analysis(root: Path, tables: Mapping[str, pd.DataFrame]) -> dict[str, Any]:
    files = sorted((root / "shards").glob("pairwise_task_*/pairwise_credit.csv"))
    df = concat_csvs(files)
    if df.empty:
        return {"available": False}
    defined = int(tables["pairwise_definitions"]["pairwise_id"].nunique())
    observed = int(df["pairwise_id"].nunique())
    by_pair = df.groupby("pairwise_id")["interaction"].agg(["count", "mean", "median", "min", "max"]).reset_index()
    by_module = df.groupby(["first_module", "second_module"])["interaction"].agg(["count", "mean", "median", "min", "max"]).reset_index()
    return {
        "available": True,
        "complete_against_design": observed == defined and len(df) == len(tables["features"]) * defined,
        "observed_pairwise_ids": observed,
        "defined_pairwise_ids": defined,
        "rows": len(df),
        "positive_interaction_rate": float((df["interaction"] > 0).mean()),
        "negative_interaction_rate": float((df["interaction"] < 0).mean()),
        "mean_interaction": float(df["interaction"].mean()),
        "median_interaction": float(df["interaction"].median()),
        "strong_positive_gt_001": int((df["interaction"] > 0.01).sum()),
        "strong_negative_lt_minus_001": int((df["interaction"] < -0.01).sum()),
        "by_pair_top_abs": by_pair.reindex(by_pair["mean"].abs().sort_values(ascending=False).index).head(10).to_dict(orient="records"),
        "by_module": by_module.to_dict(orient="records"),
    }


def observed_credit_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "all_test_candidates": summarize_selected(rows),
        "by_module_type": {
            module_type: summarize_selected([r for r in rows if r["module_type"] == module_type])
            for module_type in sorted({r["module_type"] for r in rows})
        },
        "by_donor_policy": {
            donor: summarize_selected([r for r in rows if r["donor_policy"] == donor])
            for donor in sorted({r["donor_policy"] for r in rows})
        },
    }


def summarize_selected(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    out = {"n": len(rows)}
    for key in ["C_base", "C_parent", "C_env"]:
        vals = np.asarray([float(r[key]) for r in rows], dtype=float)
        out[f"mean_{key}"] = float(vals.mean())
        out[f"median_{key}"] = float(np.median(vals))
        out[f"positive_{key}_fraction"] = float((vals > 0).mean())
        out[f"negative_{key}_fraction"] = float((vals < 0).mean())
        out[f"min_{key}"] = float(vals.min())
        out[f"max_{key}"] = float(vals.max())
    return out


def status_from_real_results(prediction: Mapping[str, Any], ranking: Mapping[str, Any], winner: str) -> str:
    best_test = prediction[winner]["test"]
    top1 = ranking[winner]["test_selection_quality"]["top_1"]
    if (
        best_test["mae"] is not None
        and best_test["mae"] <= 0.01
        and top1.get("positive_transfer_precision", 0.0) >= 0.7
    ):
        return "STRONG_SIGNAL"
    if top1.get("positive_transfer_precision", 0.0) >= 0.6:
        return "NICHE_SIGNAL"
    if prediction["suitability_augmented"]["test"]["mae"] < prediction["identity"]["test"]["mae"]:
        return "WEAK_GENERALIZATION"
    return "NO_SIGNAL"


def readiness_from_results(status: str, validation: Mapping[str, Any], pairwise: Mapping[str, Any], gate: Mapping[str, Any]) -> str:
    if not validation["core_complete"] or status in {"NO_SIGNAL", "WEAK_GENERALIZATION"}:
        return "NOT_READY"
    if not pairwise.get("complete_against_design", False) or gate.get("n_passed", 0) == 0:
        return "READY_WITH_RESTRICTIONS"
    return "READY_WITH_SMALL_EXTENSIONS"


def compare_wulver_report(root: Path) -> dict[str, Any]:
    files = sorted(p.relative_to(root).as_posix() for p in (root / "reports").glob("*") if p.is_file())
    if not files:
        return {"present": False, "files": [], "comparison": "No Wulver final report was present under reports/."}
    return {"present": True, "files": files, "comparison": "Report files present; compare manually with JSON metrics."}


def render_aw_report(report: Mapping[str, Any]) -> str:
    pred = report["prediction"]
    winner = report["winning_model_by_validation_mae"]
    rank = report["ranking"][winner]["test_selection_quality"]
    rq = report["ranking"][winner]["test_ranking_quality"]
    rows = report["schema_and_row_validation"]["row_counts"]
    pair = report["pairwise_interactions"]
    gate = report["synthesis_gate_offline"]
    lines = [
        "# Real Module-Credit Evaluation",
        "",
        f"A. Imported artifact root: `{report['artifact_root']}`.",
        f"B. Artifact map: combined={len(report['artifact_map']['combined']['files'])}, diagnostics={len(report['artifact_map']['diagnostics']['files'])}, fidelity={len(report['artifact_map']['fidelity']['files'])}, manifests={len(report['artifact_map']['manifests']['files'])}, reports={len(report['artifact_map']['reports']['files'])}, design={len(report['artifact_map']['design']['files'])}, smoke={len(report['artifact_map']['smoke']['files'])}.",
        f"C. Workflow completion: all status-bearing manifest statuses PASS={report['workflow_completion']['all_status_bearing_manifests_pass']}; core single-module complete={report['workflow_completion']['combined_single_complete']}; non-status manifests={report['workflow_completion']['non_status_manifests']}.",
        f"D. Schema and rows: schemas_valid={report['schema_and_row_validation']['schemas_valid']}; single rows={rows['combined_single_rows']}/{rows['expected_single_rows']}; features={rows['combined_features_rows']}; reward vectors={rows['combined_reward_vector_rows']}/{rows['expected_reward_vector_rows']}.",
        f"E. Ingestion: canonical rows={report['canonical_rows']['n_rows']}; train={report['canonical_rows']['n_train_rows']}; validation={report['canonical_rows']['n_validation_rows']}; test={report['canonical_rows']['n_test_rows']}.",
        f"F. Pairwise ingestion: available={pair.get('available')}; complete_against_design={pair.get('complete_against_design')}; observed_ids={pair.get('observed_pairwise_ids')}; defined_ids={pair.get('defined_pairwise_ids')}.",
        "G. Leakage-safe splits: state/config-level split_group_key was held atomic; 24 heldout configs were reserved for TEST, with development configs stratified by regime into TRAIN/VALIDATION.",
        f"H. Model baselines: validation winner={winner}; test MAE identity={pred['identity']['test']['mae']:.6f}, structural={pred['structural']['test']['mae']:.6f}, contextual={pred['state_conditioned_structural']['test']['mae']:.6f}, suitability={pred['suitability_augmented']['test']['mae']:.6f}.",
        f"I. Whole-policy suitability effect: validation improvement={report['whole_policy_suitability_effect']['improves_validation']}; test improvement={report['whole_policy_suitability_effect']['improves_test']}; test delta={report['whole_policy_suitability_effect']['test_mae_delta_suitability_minus_contextual']:.6f}.",
        f"J. Top-k donor-module selection quality for {winner}: top1 positive={rank['top_1']['positive_transfer_precision']:.3f}, top3 positive={rank['top_3']['positive_transfer_precision']:.3f}, top5 positive={rank['top_5']['positive_transfer_precision']:.3f}.",
        f"K. Top-k oracle hit quality for {winner}: top1 hit={rq['top_1']['contains_true_best_fraction']:.3f}, top3 hit={rq['top_3']['contains_true_best_fraction']:.3f}, top5 hit={rq['top_5']['contains_true_best_fraction']:.3f}.",
        f"L. Realized test credits: mean C_base={report['realized_credit_on_test']['all_test_candidates']['mean_C_base']:.6f}, mean C_parent={report['realized_credit_on_test']['all_test_candidates']['mean_C_parent']:.6f}, mean C_env={report['realized_credit_on_test']['all_test_candidates']['mean_C_env']:.6f}.",
        f"M. Uncertainty-aware ranking: evaluated lambda_m in {', '.join(report['uncertainty_aware_ranking'].keys())}.",
        f"N. Held-out donor analysis: donors={len(report['held_out_donors'])}; EDF included={'edf' in report['held_out_donors']}.",
        f"O. Held-out module-type analysis: module types={len(report['held_out_modules'])}; skipped={[k for k,v in report['held_out_modules'].items() if v.get('skipped')]}.",
        f"P. Pairwise interactions: mean={pair.get('mean_interaction')}; positive_rate={pair.get('positive_interaction_rate')}; negative_rate={pair.get('negative_interaction_rate')}; interpret as exploratory because complete_against_design={pair.get('complete_against_design')}.",
        f"Q. Offline synthesis gate: candidates={gate['n_candidates']}; passed={gate['n_passed']}; pass_rate={gate['pass_rate']}.",
        f"R. Wulver final report comparison: {report['wulver_final_report_comparison']['comparison']}",
        f"S. C_base/C_parent/C_env target models: see `target_results` in JSON for per-target validation/test MAE, RMSE, bias, and sign accuracy.",
        f"T. Module-credit status rationale: winner={winner}, status={report['MODULE_CREDIT_MODEL_STATUS']}, readiness={report['STRUCTURAL_SYNTHESIS_READINESS']}.",
        "U. Limitations: pairwise data is incomplete against the 24-definition design, and many observed single-module transfers are zero or near-zero.",
        "V. Imported artifact protection: the evaluation reads the Wulver import tree and writes only derived local report files.",
        f"W. Final verdict: MODULE_CREDIT_MODEL_STATUS={report['MODULE_CREDIT_MODEL_STATUS']}; STRUCTURAL_SYNTHESIS_READINESS={report['STRUCTURAL_SYNTHESIS_READINESS']}; REAL_MODULE_CREDIT_EVALUATION={report['REAL_MODULE_CREDIT_EVALUATION']}.",
        "",
        f"REAL_MODULE_CREDIT_EVALUATION = {report['REAL_MODULE_CREDIT_EVALUATION']}",
        "",
    ]
    return "\n".join(lines)


def concat_csvs(files: Sequence[Path]) -> pd.DataFrame:
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(p) for p in files], ignore_index=True)


def finite_float(value: Any, *, default: float | None = None) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        if default is not None:
            return default
        raise
    if math.isfinite(out):
        return out
    if default is not None:
        return default
    raise ValueError(f"Non-finite value: {value!r}")


def stable_small_hash(value: str) -> int:
    return sum((idx + 1) * ord(ch) for idx, ch in enumerate(value)) % 1024


def json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


if __name__ == "__main__":
    raise SystemExit(main())
