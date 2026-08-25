#!/usr/bin/env python3
"""Generate the scaled Family-A oracle-labeled ESTF/WFS dataset v1.

Offline TRAIN/VAL-only dataset generation. This script does not modify
simulator or policy semantics and does not train a model. Long generation is
sharded by scenario/configuration so each worker process writes only its own
files; completed shards are verified by checksum before they are reused.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from llmserveopt.analysis import family_a_observability_continuation_v1 as fac
from llmserveopt.analysis import decision_criticality_timescale_trainval_v1 as dcm
from llmserveopt.core.action import Action
from llmserveopt.core.types import CompletedRequest, ObservableRequest, ObservableState
from llmserveopt.policies.base import BasePolicy
from llmserveopt.policies.scoring import predicted_service_proxy
from llmserveopt.policy_separation.hierarchical_regime_router_v1 import REGIME_A
from llmserveopt.policy_separation.hierarchical_router_live_harness_v1 import (
    LiveHierarchicalRouterPolicy,
    build_feature_rows_by_regime,
    build_native_policy_instances,
)
from llmserveopt.policy_separation.templates_fairness_starvation_v2 import (
    case_fairness_vs_size_v2,
)
from llmserveopt.selector.hierarchical_stage2_selectors_v1 import Stage2Selector
from llmserveopt.simulator.service_model import ServiceModel
from llmserveopt.simulator.simulator import Simulator, SimulatorConfig


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "datasets/family_a_oracle_policy_v1"
DEFAULT_LOG_DIR = REPO_ROOT / "logs"
DATASET_VERSION = "family_a_oracle_policy_v1.0.0"
LABEL_DEFINITION_VERSION = "whole_branch_priority_weighted_slo_v1"
FEATURE_SCHEMA_VERSION = "family_a_oracle_policy_pilot_v1_compatible"
DATASET_DATE = "2026-08-21"

ESTF_ID = fac.ESTF_ID
WFS_ID = fac.WFS_ID
ESTF = "ESTF"
WFS = "WFS"
TIE = "TIE_OR_UNCERTAIN"
EPS = 1e-9

DEFAULT_TARGET_SCENARIOS = 704
DEFAULT_WORKERS = 4
DEFAULT_MAX_EVENTS_PER_SCENARIO = 3
DEFAULT_MIN_EVENT_STEP_GAP = 100
DEFAULT_MAX_EXTRA_STEPS = fac.FAMILY_A_DIAGNOSTIC_MAX_EXTRA_STEPS
SCALE_SEEDS = list(range(20260816, 20260838))

GLOBAL_FEATURES = [
    "queue_length",
    "active_count",
    "n_gpus",
    "queue_age_p10",
    "queue_age_p50",
    "queue_age_p90",
    "queue_age_mean",
    "predicted_output_tokens_p10",
    "predicted_output_tokens_p50",
    "predicted_output_tokens_p90",
    "predicted_output_tokens_mean",
    "prompt_tokens_p10",
    "prompt_tokens_p50",
    "prompt_tokens_p90",
    "prompt_tokens_mean",
    "est_service_time_p10",
    "est_service_time_p50",
    "est_service_time_p90",
    "est_service_time_mean",
    "max_class_deficit_ratio",
    "longest_waiting_age",
    "n_distinct_classes_in_queue",
    "laxity_p10",
    "laxity_p50",
    "laxity_p90",
    "laxity_mean",
    "fraction_laxity_negative",
    "fraction_laxity_near_deadline",
    "mean_kv_utilization",
    "max_kv_utilization",
    "free_kv_capacity",
    "prefilling_count",
    "decoding_count",
    "agg_n_admit_estf",
    "agg_n_admit_wfs",
    "admit_symmetric_diff_size",
    "history_queue_len_slope",
    "history_kv_util_slope",
    "history_admitted_count_slope",
]

SIDE_FEATURES = [
    "priority",
    "prompt_tokens",
    "predicted_output_tokens",
    "predicted_service_proxy",
    "remaining_predicted_service_proxy",
    "queue_age",
    "laxity_own",
]

PAIR_FEATURES = [
    "priority_diff_estf_minus_wfs",
    "prompt_tokens_diff_estf_minus_wfs",
    "predicted_output_tokens_diff_estf_minus_wfs",
    "predicted_service_proxy_diff_estf_minus_wfs",
    "queue_age_diff_estf_minus_wfs",
    "laxity_own_diff_estf_minus_wfs",
    "priority_ratio_estf_over_wfs",
    "predicted_service_proxy_ratio_estf_over_wfs",
    "queue_age_ratio_estf_over_wfs",
    "laxity_own_ratio_estf_over_wfs",
]


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def json_dump(path: Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(json_ready(obj), indent=2, sort_keys=True) + "\n")


def json_ready(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_ready(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_ready(v) for v in obj]
    if isinstance(obj, tuple):
        return [json_ready(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def git_head() -> str:
    return subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True).strip()


def git_dirty() -> bool:
    return bool(subprocess.check_output(["git", "-C", str(REPO_ROOT), "status", "--short"], text=True).strip())


def label(delta: float) -> str:
    if delta > 0.0:
        return ESTF
    if delta < 0.0:
        return WFS
    return TIE


def feature_name(name: str) -> str:
    return f"feat_{name}"


def feature_columns() -> list[str]:
    cols = [feature_name(c) for c in GLOBAL_FEATURES]
    for side in ("estf", "wfs"):
        cols.extend(feature_name(f"{side}_{c}") for c in SIDE_FEATURES)
    cols.extend(feature_name(c) for c in PAIR_FEATURES)
    return cols


def configuration_group_id(row: pd.Series) -> str:
    return (
        f"util{float(row['target_utilization']):.4f}"
        f".skew{float(row['tenant_weight_skew']):.4f}"
        f".fav{row['favored_tenant_size']}"
        f".noise{float(row['prediction_noise_sigma']):.2f}"
        f".n{int(row['n_total_jobs'])}"
        f".maxseq{int(row['max_active_sequences'])}"
    )


def scenario_id_from_row(row: pd.Series) -> str:
    return (
        "FAMILY_A_ORACLE_POLICY_V1::fs2"
        f".util{float(row['target_utilization']):.4f}"
        f".skew{float(row['tenant_weight_skew']):.4f}"
        f".fav{row['favored_tenant_size']}"
        f".noise{float(row['prediction_noise_sigma']):.2f}"
        f".s{int(row['seed'])}"
    )


def build_scenario_manifest(*, output_dir: Path, target_scenarios: int = DEFAULT_TARGET_SCENARIOS) -> pd.DataFrame:
    fam = fac.load_family_a_trainval_scenario_table()
    assert set(fam["split"].unique()) <= {dcm.TRAIN, dcm.VAL}
    configs = (
        fam[
            [
                "feat_A__target_utilization",
                "feat_A__tenant_weight_skew",
                "feat_A__favored_tenant_size",
                "feat_A__prediction_noise_sigma",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "feat_A__target_utilization",
                "feat_A__tenant_weight_skew",
                "feat_A__favored_tenant_size",
                "feat_A__prediction_noise_sigma",
            ]
        )
        .reset_index(drop=True)
    )
    rows: list[dict[str, Any]] = []
    for _, cfg in configs.iterrows():
        for seed in SCALE_SEEDS:
            split = dcm.VAL if seed % 5 == 0 else dcm.TRAIN
            row = {
                "dataset_version": DATASET_VERSION,
                "mechanism_family": dcm.FAMILY_A,
                "split": split,
                "target_utilization": float(cfg["feat_A__target_utilization"]),
                "tenant_weight_skew": float(cfg["feat_A__tenant_weight_skew"]),
                "favored_tenant_size": str(cfg["feat_A__favored_tenant_size"]),
                "prediction_noise_sigma": float(cfg["feat_A__prediction_noise_sigma"]),
                "seed": int(seed),
                "n_total_jobs": 120,
                "max_active_sequences": 1,
            }
            row["configuration_group_id"] = configuration_group_id(pd.Series(row))
            row["scenario_id"] = scenario_id_from_row(pd.Series(row))
            rows.append(row)
    manifest = pd.DataFrame(rows).head(target_scenarios).copy()
    if manifest.empty:
        raise RuntimeError("empty scaled scenario manifest")
    assert set(manifest["split"].unique()) <= {dcm.TRAIN, dcm.VAL}
    assert not (manifest["split"] == dcm.TEST).any()
    assert manifest["scenario_id"].is_unique
    manifest["scenario_index"] = np.arange(len(manifest), dtype=int)

    manifest_path = output_dir / "scenario_manifest.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)
    json_dump(
        output_dir / "scenario_manifest_summary.json",
        {
            "dataset_version": DATASET_VERSION,
            "n_scenarios": int(len(manifest)),
            "n_configuration_groups": int(manifest["configuration_group_id"].nunique()),
            "splits": manifest["split"].value_counts().to_dict(),
            "favored_tenant_size": manifest["favored_tenant_size"].value_counts().to_dict(),
            "target_utilization": manifest["target_utilization"].value_counts().to_dict(),
            "tenant_weight_skew": manifest["tenant_weight_skew"].value_counts().to_dict(),
            "prediction_noise_sigma": manifest["prediction_noise_sigma"].value_counts().to_dict(),
            "manifest_sha256": sha256_file(manifest_path),
        },
    )
    return manifest


def load_or_create_manifest(output_dir: Path, target_scenarios: int) -> pd.DataFrame:
    path = output_dir / "scenario_manifest.csv"
    if path.exists():
        manifest = pd.read_csv(path)
        if len(manifest) != target_scenarios:
            shard_dir = output_dir / "shards"
            done_markers = list(shard_dir.glob("shard_*.done.json")) if shard_dir.exists() else []
            if done_markers:
                raise RuntimeError(
                    f"existing manifest has {len(manifest)} scenarios but target_scenarios={target_scenarios}; "
                    "refusing to change manifest after completed shards exist"
                )
            return build_scenario_manifest(output_dir=output_dir, target_scenarios=target_scenarios)
        assert set(manifest["split"].unique()) <= {dcm.TRAIN, dcm.VAL}
        assert not (manifest["split"] == dcm.TEST).any()
        return manifest
    return build_scenario_manifest(output_dir=output_dir, target_scenarios=target_scenarios)


def shard_assignments(manifest: pd.DataFrame, workers: int) -> dict[int, list[int]]:
    return {i: manifest.index[manifest["scenario_index"] % workers == i].tolist() for i in range(workers)}


def verify_shard_disjointness(manifest: pd.DataFrame, workers: int) -> None:
    assignments = shard_assignments(manifest, workers)
    all_rows = [idx for rows in assignments.values() for idx in rows]
    if len(all_rows) != len(set(all_rows)) or set(all_rows) != set(manifest.index):
        raise AssertionError("shard assignments are not a disjoint cover of the manifest")


def request_weight(req: Any) -> float:
    try:
        p = float(getattr(req, "priority", 1.0))
    except (TypeError, ValueError):
        return 1.0
    return p if p > 0 else 1.0


def success_weight(completed: list[CompletedRequest]) -> float:
    return float(sum(request_weight(c.request) for c in completed if not c.slo_violated))


def completed_by_id(completed: list[CompletedRequest]) -> dict[int, CompletedRequest]:
    return {int(c.request.request_id): c for c in completed}


@dataclass
class BranchResult:
    policy_id: str
    steps_run: int
    bounded_horizon_steps: int
    ran_to_natural_completion: bool
    completed_count: int
    slo_violation_count: int
    success_weight_numerator: float
    completed_request_ids: list[int]
    completed: list[CompletedRequest]


def run_weighted_branch(
    sim: Simulator,
    *,
    policy: BasePolicy,
    policy_id: str,
    first_action: Action,
    max_extra_steps: int,
) -> BranchResult:
    fork = dcm.fork_from_live_simulator(
        sim,
        policy=policy,
        policy_id=policy_id,
        first_action=copy.deepcopy(first_action),
    )
    base_completed = len(sim._completed)
    steps_run = 1
    while not fork.finished and steps_run < max_extra_steps:
        fork.advance_one_step()
        steps_run += 1
    new_completed = list(fork.shell._completed[base_completed:])
    return BranchResult(
        policy_id=policy_id,
        steps_run=steps_run,
        bounded_horizon_steps=max_extra_steps,
        ran_to_natural_completion=bool(fork.finished),
        completed_count=len(new_completed),
        slo_violation_count=int(sum(1 for c in new_completed if c.slo_violated)),
        success_weight_numerator=success_weight(new_completed),
        completed_request_ids=[int(c.request.request_id) for c in new_completed],
        completed=new_completed,
    )


def whole_branch_label(estf_branch: BranchResult, wfs_branch: BranchResult) -> dict[str, Any]:
    j_estf = estf_branch.success_weight_numerator
    j_wfs = wfs_branch.success_weight_numerator
    delta = j_estf - j_wfs
    return {
        "J_ESTF_whole": j_estf,
        "J_WFS_whole": j_wfs,
        "delta_J_whole": delta,
        "oracle_label": label(delta),
    }


def contested_pair_label(
    *,
    estf_req: ObservableRequest,
    wfs_req: ObservableRequest,
    estf_branch: BranchResult,
    wfs_branch: BranchResult,
) -> dict[str, Any]:
    estf_completed = completed_by_id(estf_branch.completed)
    wfs_completed = completed_by_id(wfs_branch.completed)
    estf_ids = completed_by_id(wfs_branch.completed)
    wfs_ids = completed_by_id(estf_branch.completed)

    def contribution(req: ObservableRequest, done: dict[int, CompletedRequest]) -> float:
        comp = done.get(int(req.request_id))
        if comp is None or comp.slo_violated:
            return 0.0
        return request_weight(comp.request)

    j_estf = contribution(estf_req, estf_completed) + contribution(wfs_req, wfs_ids)
    j_wfs = contribution(estf_req, estf_ids) + contribution(wfs_req, wfs_completed)
    delta = j_estf - j_wfs
    return {
        "J_ESTF_contested": j_estf,
        "J_WFS_contested": j_wfs,
        "delta_J_contested": delta,
        "oracle_label_contested": label(delta),
        "completion_benefit_label": int(estf_req.request_id in estf_completed and estf_req.request_id not in estf_ids),
        "slo_risk_label": int(
            (wfs_req.request_id not in wfs_ids)
            or bool(wfs_ids.get(int(wfs_req.request_id)) and wfs_ids[int(wfs_req.request_id)].slo_violated)
        ),
    }


def stable_state_fingerprint(row: dict[str, Any], feature_cols: list[str]) -> str:
    payload = {
        "scenario": row["scenario_id"],
        "step": row["step"],
        "estf_request": row["estf_contested_request_id"],
        "wfs_request": row["wfs_contested_request_id"],
        "features": {c: None if pd.isna(row.get(c)) else round(float(row.get(c)), 12) for c in feature_cols},
    }
    return sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))


class ScaledFamilyAObserver(BasePolicy):
    name = "family_a_oracle_policy_v1_shadow_generator"

    def __init__(
        self,
        *,
        sim_ref: Simulator,
        inner_router: LiveHierarchicalRouterPolicy,
        shadow_policies: dict[str, BasePolicy],
        scenario_meta: dict[str, Any],
        max_events: int,
        min_event_step_gap: int,
        max_extra_steps: int,
        scenario_arrival_weight: float,
    ) -> None:
        self.sim_ref = sim_ref
        self.inner_router = inner_router
        self.shadow_policies = shadow_policies
        self.scenario_meta = scenario_meta
        self.max_events = max_events
        self.min_event_step_gap = min_event_step_gap
        self.max_extra_steps = max_extra_steps
        self.scenario_arrival_weight = scenario_arrival_weight
        self.events: list[dict[str, Any]] = []
        self.last_event_step: int | None = None
        self.invalid_disagreement_count = 0

    def reset(self) -> None:
        self.inner_router.reset()
        for p in self.shadow_policies.values():
            p.reset()
        self.events = []
        self.last_event_step = None
        self.invalid_disagreement_count = 0

    def select_action(self, state: ObservableState) -> Action:
        pre_decision_gpu_state = fac.snapshot_gpu_counters(state)
        real_action = self.inner_router.select_action(state)
        row = self.inner_router.trajectory[-1] if self.inner_router.trajectory else None
        post_real_admission_gpu_state = fac.snapshot_gpu_counters(state)

        if row is not None and row.effective_regime == REGIME_A and len(self.events) < self.max_events:
            if self.last_event_step is None or int(state.step) - self.last_event_step >= self.min_event_step_gap:
                fac.restore_gpu_counters(state, pre_decision_gpu_state)
                action_estf = self.shadow_policies[ESTF_ID].select_action(state)
                fac.restore_gpu_counters(state, pre_decision_gpu_state)
                action_wfs = self.shadow_policies[WFS_ID].select_action(state)
                fac.restore_gpu_counters(state, pre_decision_gpu_state)

                if dcm.actions_disagree(action_estf, action_wfs):
                    admit_estf = fac._admitted_ids(action_estf)
                    admit_wfs = fac._admitted_ids(action_wfs)
                    estf_only = sorted(set(admit_estf) - set(admit_wfs))
                    wfs_only = sorted(set(admit_wfs) - set(admit_estf))
                    common = sorted(set(admit_estf) & set(admit_wfs))
                    if len(estf_only) == 1 and len(wfs_only) == 1 and not common:
                        feature_state = copy.deepcopy(state)
                        by_id = {int(r.request_id): r for r in feature_state.waiting_queue}
                        estf_req = by_id[int(estf_only[0])]
                        wfs_req = by_id[int(wfs_only[0])]
                        history_rows = self.inner_router.trajectory_df()
                        features = fac.extract_causal_features(
                            feature_state,
                            step_size=self.sim_ref.config.service_model.step_size,
                            estf_policy=self.shadow_policies[ESTF_ID],
                            wfs_policy=self.shadow_policies[WFS_ID],
                            admit_ids_estf=admit_estf,
                            admit_ids_wfs=admit_wfs,
                            history_rows=history_rows,
                        )
                        estf_branch = run_weighted_branch(
                            self.sim_ref,
                            policy=self.shadow_policies[ESTF_ID],
                            policy_id=ESTF_ID,
                            first_action=action_estf,
                            max_extra_steps=self.max_extra_steps,
                        )
                        wfs_branch = run_weighted_branch(
                            self.sim_ref,
                            policy=self.shadow_policies[WFS_ID],
                            policy_id=WFS_ID,
                            first_action=action_wfs,
                            max_extra_steps=self.max_extra_steps,
                        )
                        self.events.append(
                            build_row(
                                scenario_meta=self.scenario_meta,
                                state=feature_state,
                                features=features,
                                router_selected_policy=row.selected_policy,
                                estf_req=estf_req,
                                wfs_req=wfs_req,
                                estf_branch=estf_branch,
                                wfs_branch=wfs_branch,
                                scenario_arrival_weight=self.scenario_arrival_weight,
                            )
                        )
                        self.last_event_step = int(state.step)
                    else:
                        self.invalid_disagreement_count += 1

        fac.restore_gpu_counters(state, post_real_admission_gpu_state)
        return real_action


def side_feature_value(req: ObservableRequest, col: str, state: ObservableState) -> float:
    if col == "priority":
        return float(req.priority)
    if col == "prompt_tokens":
        return float(req.prompt_tokens)
    if col == "predicted_output_tokens":
        return float(req.predicted_output_tokens)
    if col == "predicted_service_proxy":
        return float(predicted_service_proxy(req))
    if col == "remaining_predicted_service_proxy":
        return float(predicted_service_proxy(req))
    if col == "queue_age":
        return float(state.time - req.arrival_time)
    if col == "laxity_own":
        return float(req.slo_deadline - state.time)
    raise KeyError(col)


def build_row(
    *,
    scenario_meta: dict[str, Any],
    state: ObservableState,
    features: dict[str, Any],
    router_selected_policy: str,
    estf_req: ObservableRequest,
    wfs_req: ObservableRequest,
    estf_branch: BranchResult,
    wfs_branch: BranchResult,
    scenario_arrival_weight: float,
) -> dict[str, Any]:
    whole = whole_branch_label(estf_branch, wfs_branch)
    contested = contested_pair_label(
        estf_req=estf_req,
        wfs_req=wfs_req,
        estf_branch=estf_branch,
        wfs_branch=wfs_branch,
    )
    row: dict[str, Any] = {
        "sample_id": f"{DATASET_VERSION}::{scenario_meta['scenario_id']}::step{int(state.step)}",
        "dataset_version": DATASET_VERSION,
        "label_definition_version": LABEL_DEFINITION_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "scenario_id": scenario_meta["scenario_id"],
        "canonical_scenario_id": scenario_meta["scenario_id"],
        "configuration_group_id": scenario_meta["configuration_group_id"],
        "split": scenario_meta["split"],
        "step": int(state.step),
        "time": float(state.time),
        "source": "scaled_generation_native_counterfactual",
        "primary_label_source": "whole_branch_priority_weighted_slo",
        "continuation_semantics": "native_estf_vs_native_wfs_bounded_1500_step_counterfactual",
        "horizon_steps": int(estf_branch.bounded_horizon_steps),
        "future_arrivals_included": True,
        "tie_threshold": 0.0,
        "router_selected_policy": router_selected_policy,
        "estf_contested_request_id": int(estf_req.request_id),
        "wfs_contested_request_id": int(wfs_req.request_id),
        "estf_contested_class_id": str(estf_req.class_id),
        "wfs_contested_class_id": str(wfs_req.class_id),
        "analysis_fav": scenario_meta["favored_tenant_size"],
        "analysis_utilization": float(scenario_meta["target_utilization"]),
        "analysis_skew": float(scenario_meta["tenant_weight_skew"]),
        "analysis_noise": float(scenario_meta["prediction_noise_sigma"]),
        "analysis_seed": int(scenario_meta["seed"]),
        "scenario_arrival_weight": float(scenario_arrival_weight),
        "J_ESTF_whole": whole["J_ESTF_whole"],
        "J_WFS_whole": whole["J_WFS_whole"],
        "delta_J_whole": whole["delta_J_whole"],
        "oracle_label": whole["oracle_label"],
        "J_ESTF_whole_anwg": whole["J_ESTF_whole"] / scenario_arrival_weight if scenario_arrival_weight else float("nan"),
        "J_WFS_whole_anwg": whole["J_WFS_whole"] / scenario_arrival_weight if scenario_arrival_weight else float("nan"),
        "delta_J_whole_anwg": whole["delta_J_whole"] / scenario_arrival_weight if scenario_arrival_weight else float("nan"),
        "J_ESTF_contested": contested["J_ESTF_contested"],
        "J_WFS_contested": contested["J_WFS_contested"],
        "delta_J_contested": contested["delta_J_contested"],
        "oracle_label_contested": contested["oracle_label_contested"],
        "completion_benefit_label": contested["completion_benefit_label"],
        "slo_risk_label": contested["slo_risk_label"],
        "br_estf_estf_completed_count": estf_branch.completed_count,
        "br_wfs_wfs_completed_count": wfs_branch.completed_count,
        "br_estf_estf_slo_violation_count": estf_branch.slo_violation_count,
        "br_wfs_wfs_slo_violation_count": wfs_branch.slo_violation_count,
        "br_estf_estf_steps_run": estf_branch.steps_run,
        "br_wfs_wfs_steps_run": wfs_branch.steps_run,
        "br_estf_estf_ran_to_natural_completion": estf_branch.ran_to_natural_completion,
        "br_wfs_wfs_ran_to_natural_completion": wfs_branch.ran_to_natural_completion,
    }
    global_map = dict(features)
    if "n_admit_estf" in global_map:
        global_map["agg_n_admit_estf"] = global_map["n_admit_estf"]
    if "n_admit_wfs" in global_map:
        global_map["agg_n_admit_wfs"] = global_map["n_admit_wfs"]
    for col in GLOBAL_FEATURES:
        row[feature_name(col)] = global_map.get(col, float("nan"))
    for side, req in (("estf", estf_req), ("wfs", wfs_req)):
        for col in SIDE_FEATURES:
            row[feature_name(f"{side}_{col}")] = side_feature_value(req, col, state)

    pairs = {
        "priority": (row[feature_name("estf_priority")], row[feature_name("wfs_priority")]),
        "prompt_tokens": (row[feature_name("estf_prompt_tokens")], row[feature_name("wfs_prompt_tokens")]),
        "predicted_output_tokens": (
            row[feature_name("estf_predicted_output_tokens")],
            row[feature_name("wfs_predicted_output_tokens")],
        ),
        "predicted_service_proxy": (
            row[feature_name("estf_predicted_service_proxy")],
            row[feature_name("wfs_predicted_service_proxy")],
        ),
        "queue_age": (row[feature_name("estf_queue_age")], row[feature_name("wfs_queue_age")]),
        "laxity_own": (row[feature_name("estf_laxity_own")], row[feature_name("wfs_laxity_own")]),
    }
    for name, (a, b) in pairs.items():
        row[feature_name(f"{name}_diff_estf_minus_wfs")] = float(a) - float(b)
    for name in ("priority", "predicted_service_proxy", "queue_age", "laxity_own"):
        a, b = pairs[name]
        row[feature_name(f"{name}_ratio_estf_over_wfs")] = float(a) / max(float(b), EPS)
    row["state_fingerprint"] = stable_state_fingerprint(row, feature_columns())
    return row


def scenario_from_manifest_row(row: pd.Series):
    return case_fairness_vs_size_v2(
        target_utilization=float(row["target_utilization"]),
        tenant_weight_skew=float(row["tenant_weight_skew"]),
        favored_tenant_size=str(row["favored_tenant_size"]),
        prediction_noise_sigma=float(row["prediction_noise_sigma"]),
        seed=int(row["seed"]),
        n_total_jobs=int(row["n_total_jobs"]),
        max_active_sequences=int(row["max_active_sequences"]),
        allow_synthetic_tokens=False,
        datasets_root=dcm.DATASETS_ROOT,
    )


def run_scenario(
    row: pd.Series,
    *,
    stage1: Any,
    stage2_selectors: dict[str, Stage2Selector],
    max_events: int,
    min_event_step_gap: int,
    max_extra_steps: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if row["split"] not in {dcm.TRAIN, dcm.VAL}:
        raise dcm.TestSplitAccessError(f"scaled generator is TRAIN/VAL-only; got split={row['split']!r}")
    if str(row["mechanism_family"]) != dcm.FAMILY_A:
        raise AssertionError("scaled generator only accepts Family-A rows")

    scenario = scenario_from_manifest_row(row)
    scenario_id = str(row["scenario_id"])
    feature_rows = build_feature_rows_by_regime(scenario, scenario_id)
    inner_router = LiveHierarchicalRouterPolicy(
        scenario_id=scenario_id,
        stage1=stage1,
        stage2_selectors=stage2_selectors,
        feature_rows_by_regime=feature_rows,
        record_trajectory=True,
    )
    sim = Simulator(
        SimulatorConfig(
            gpu_configs=list(scenario.gpu_configs),
            service_model=ServiceModel(**scenario.service_model_kwargs),
        )
    )
    sim.load_trace(list(scenario.requests))
    shadow_policies = build_native_policy_instances()
    meta = row.to_dict()
    scenario_arrival_weight = float(sum(request_weight(r) for r in scenario.requests))
    observer = ScaledFamilyAObserver(
        sim_ref=sim,
        inner_router=inner_router,
        shadow_policies=shadow_policies,
        scenario_meta=meta,
        max_events=max_events,
        min_event_step_gap=min_event_step_gap,
        max_extra_steps=max_extra_steps,
        scenario_arrival_weight=scenario_arrival_weight,
    )
    sim.run(observer, workload_tag=scenario_id, seed=int(row["seed"]))
    traj = inner_router.trajectory_df()
    return observer.events, {
        "scenario_id": scenario_id,
        "configuration_group_id": row["configuration_group_id"],
        "split": row["split"],
        "n_steps": int(len(traj)),
        "n_family_a_active_steps": int((traj["effective_regime"] == REGIME_A).sum()) if len(traj) else 0,
        "n_events": int(len(observer.events)),
        "invalid_disagreement_count": int(observer.invalid_disagreement_count),
    }


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = output_columns()
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(json_ready(row))
    os.replace(tmp, path)


def output_columns() -> list[str]:
    metadata = [
        "sample_id",
        "dataset_version",
        "label_definition_version",
        "feature_schema_version",
        "scenario_id",
        "canonical_scenario_id",
        "configuration_group_id",
        "split",
        "step",
        "time",
        "source",
        "primary_label_source",
        "continuation_semantics",
        "horizon_steps",
        "future_arrivals_included",
        "tie_threshold",
        "router_selected_policy",
        "estf_contested_request_id",
        "wfs_contested_request_id",
        "estf_contested_class_id",
        "wfs_contested_class_id",
        "analysis_fav",
        "analysis_utilization",
        "analysis_skew",
        "analysis_noise",
        "analysis_seed",
        "scenario_arrival_weight",
        "J_ESTF_whole",
        "J_WFS_whole",
        "delta_J_whole",
        "oracle_label",
        "J_ESTF_whole_anwg",
        "J_WFS_whole_anwg",
        "delta_J_whole_anwg",
        "J_ESTF_contested",
        "J_WFS_contested",
        "delta_J_contested",
        "oracle_label_contested",
        "completion_benefit_label",
        "slo_risk_label",
        "br_estf_estf_completed_count",
        "br_wfs_wfs_completed_count",
        "br_estf_estf_slo_violation_count",
        "br_wfs_wfs_slo_violation_count",
        "br_estf_estf_steps_run",
        "br_wfs_wfs_steps_run",
        "br_estf_estf_ran_to_natural_completion",
        "br_wfs_wfs_ran_to_natural_completion",
        "state_fingerprint",
    ]
    return metadata + feature_columns()


def done_marker_valid(rows_path: Path, done_path: Path) -> bool:
    if not rows_path.exists() or not done_path.exists():
        return False
    try:
        done = json.loads(done_path.read_text())
    except json.JSONDecodeError:
        return False
    return done.get("rows_sha256") == sha256_file(rows_path)


def run_shard(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    manifest = load_or_create_manifest(output_dir, args.target_scenarios)
    verify_shard_disjointness(manifest, args.workers)
    shard_dir = output_dir / "shards"
    shard_id = int(args.shard_id)
    rows_path = shard_dir / f"shard_{shard_id:03d}.rows.csv"
    done_path = shard_dir / f"shard_{shard_id:03d}.done.json"
    progress_path = shard_dir / f"shard_{shard_id:03d}.progress.json"
    if done_marker_valid(rows_path, done_path):
        print(f"SHARD {shard_id} verified complete; skipping", flush=True)
        return

    shard_rows = manifest.iloc[shard_assignments(manifest, args.workers)[shard_id]].reset_index(drop=True)
    print(f"SHARD {shard_id} start scenarios={len(shard_rows)}", flush=True)
    stage1, stage2_selectors = fac.fit_frozen_models()
    all_rows: list[dict[str, Any]] = []
    scenario_summaries: list[dict[str, Any]] = []
    started = time.time()
    for pos, (_, scenario_row) in enumerate(shard_rows.iterrows(), start=1):
        json_dump(
            progress_path,
            {
                "status": "running",
                "shard_id": shard_id,
                "pid": os.getpid(),
                "scenario_position": pos,
                "scenario_count": len(shard_rows),
                "current_scenario_id": scenario_row["scenario_id"],
                "rows_so_far": len(all_rows),
                "updated_unix": time.time(),
            },
        )
        print(f"SHARD {shard_id} scenario {pos}/{len(shard_rows)} start {scenario_row['scenario_id']}", flush=True)
        rows, summary = run_scenario(
            scenario_row,
            stage1=stage1,
            stage2_selectors=stage2_selectors,
            max_events=args.max_events_per_scenario,
            min_event_step_gap=args.min_event_step_gap,
            max_extra_steps=args.max_extra_steps,
        )
        all_rows.extend(rows)
        scenario_summaries.append(summary)
        write_rows_csv(rows_path, sorted(all_rows, key=lambda r: (r["scenario_id"], r["step"])))
        print(
            f"SHARD {shard_id} scenario {pos}/{len(shard_rows)} done events={len(rows)} rows_total={len(all_rows)}",
            flush=True,
        )
    all_rows = sorted(all_rows, key=lambda r: (r["scenario_id"], r["step"]))
    sample_ids = [r["sample_id"] for r in all_rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError(f"duplicate sample_id within shard {shard_id}")
    fps = [r["state_fingerprint"] for r in all_rows]
    if len(fps) != len(set(fps)):
        raise RuntimeError(f"duplicate state_fingerprint within shard {shard_id}")
    write_rows_csv(rows_path, all_rows)
    rows_sha = sha256_file(rows_path)
    done = {
        "dataset_version": DATASET_VERSION,
        "shard_id": shard_id,
        "pid": os.getpid(),
        "n_rows": len(all_rows),
        "n_scenarios": len(shard_rows),
        "elapsed_s": time.time() - started,
        "rows_sha256": rows_sha,
        "scenario_summaries": scenario_summaries,
        "completed_unix": time.time(),
    }
    json_dump(done_path, done)
    json_dump(progress_path, {**done, "status": "complete"})
    print(f"SHARD {shard_id} complete rows={len(all_rows)} sha256={rows_sha}", flush=True)


def write_dataset_metadata(output_dir: Path, manifest: pd.DataFrame, workers: int, command: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    schema = {
        "dataset_version": DATASET_VERSION,
        "label_definition_version": LABEL_DEFINITION_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "primary_label": "oracle_label",
        "primary_delta": "delta_J_whole",
        "tie_threshold": 0.0,
        "feature_columns": feature_columns(),
        "forbidden_model_feature_tokens": [
            "scenario_id",
            "seed",
            "split",
            "fav",
            "configuration_group_id",
            "actual_output",
            "br_",
            "J_",
            "delta_J",
            "oracle_label",
            "TEST",
        ],
    }
    json_dump(output_dir / "schema.json", schema)
    feature_audit = [
        {"column": c, "classification": "ONLINE_CAUSAL_MODEL_FEATURE"} for c in feature_columns()
    ]
    feature_audit.extend(
        {"column": c, "classification": "METADATA_ONLY"}
        for c in ["sample_id", "scenario_id", "configuration_group_id", "split", "step"]
    )
    feature_audit.extend(
        {"column": c, "classification": "LABEL_OR_FUTURE_OUTCOME"}
        for c in [
            "J_ESTF_whole",
            "J_WFS_whole",
            "delta_J_whole",
            "oracle_label",
            "J_ESTF_contested",
            "J_WFS_contested",
            "delta_J_contested",
            "oracle_label_contested",
        ]
    )
    pd.DataFrame(feature_audit).to_csv(output_dir / "feature_classification.csv", index=False)
    json_dump(
        output_dir / "provenance.json",
        {
            "dataset_version": DATASET_VERSION,
            "date": DATASET_DATE,
            "git_head": git_head(),
            "git_dirty": git_dirty(),
            "python_executable": sys.executable,
            "command": command,
            "worker_count": workers,
            "scenario_manifest_sha256": sha256_file(output_dir / "scenario_manifest.csv"),
            "source_splits": sorted(manifest["split"].unique().tolist()),
            "simulator_policy_semantics": "unchanged; native ESTF/WFS via existing simulator fork machinery",
        },
    )
    readme = f"""# Family-A Oracle Policy Dataset V1

Scaled offline oracle-labeled ESTF/WFS dataset generation.

- Version: `{DATASET_VERSION}`
- Target: approximately 1,000 valid labeled decision states
- Scope: TRAIN/VAL only; TEST rows are rejected at runtime
- Primary label: whole-branch priority-weighted SLO-safe utility difference
- Compatibility label: contested-pair priority-weighted SLO-safe utility difference
- Feature schema: pilot-compatible causal `feat_*` numeric columns
- Shards: `shards/shard_*.rows.csv` plus checksum `*.done.json`

Generation is resumable. Completed shards are reused only when their done
marker checksum matches the immutable shard CSV.
"""
    atomic_write_text(output_dir / "README.md", readme)


def merge_shards(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    manifest = load_or_create_manifest(output_dir, args.target_scenarios)
    verify_shard_disjointness(manifest, args.workers)
    shard_dir = output_dir / "shards"
    rows: list[pd.DataFrame] = []
    done_records = []
    for shard_id in range(args.workers):
        rows_path = shard_dir / f"shard_{shard_id:03d}.rows.csv"
        done_path = shard_dir / f"shard_{shard_id:03d}.done.json"
        if not done_marker_valid(rows_path, done_path):
            raise RuntimeError(f"shard {shard_id} missing or checksum-invalid")
        rows.append(pd.read_csv(rows_path))
        done_records.append(json.loads(done_path.read_text()))
    merged = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=output_columns())
    merged = merged.sort_values(["scenario_id", "step"]).reset_index(drop=True)
    if merged["sample_id"].duplicated().any():
        raise RuntimeError("merge refused duplicate sample_id")
    if merged["state_fingerprint"].duplicated().any():
        raise RuntimeError("merge refused duplicate state_fingerprint")
    if not set(merged["split"].unique()) <= {dcm.TRAIN, dcm.VAL}:
        raise dcm.TestSplitAccessError("TEST or non-TRAIN/VAL split in merged rows")
    for delta_col, label_col in [("delta_J_whole", "oracle_label"), ("delta_J_contested", "oracle_label_contested")]:
        expected = np.where(merged[delta_col] > 0, ESTF, np.where(merged[delta_col] < 0, WFS, TIE))
        if not (merged[label_col].to_numpy() == expected).all():
            raise RuntimeError(f"{label_col} inconsistent with {delta_col}")
    out = output_dir / "oracle_rows.csv"
    tmp = out.with_suffix(".csv.tmp")
    merged.to_csv(tmp, index=False)
    os.replace(tmp, out)
    json_dump(
        output_dir / "quality_summary.json",
        {
            "n_rows": int(len(merged)),
            "n_scenarios": int(merged["scenario_id"].nunique()) if len(merged) else 0,
            "n_configuration_groups": int(merged["configuration_group_id"].nunique()) if len(merged) else 0,
            "label_counts": merged["oracle_label"].value_counts().to_dict() if len(merged) else {},
            "contested_label_counts": merged["oracle_label_contested"].value_counts().to_dict() if len(merged) else {},
            "exact_duplicate_sample_ids": int(merged["sample_id"].duplicated().sum()) if len(merged) else 0,
            "exact_duplicate_state_fingerprints": int(merged["state_fingerprint"].duplicated().sum()) if len(merged) else 0,
            "oracle_rows_sha256": sha256_file(out),
            "shards": done_records,
        },
    )
    print(json.dumps({"merged_rows": len(merged), "output": str(out), "sha256": sha256_file(out)}, indent=2), flush=True)


def prelaunch_validate(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    manifest = load_or_create_manifest(output_dir, args.target_scenarios)
    verify_shard_disjointness(manifest, args.workers)
    if args.workers < 1:
        raise AssertionError("workers must be >=1")
    if args.workers > 8:
        raise AssertionError("scale-up worker count must remain conservative")
    if set(manifest["split"].unique()) - {dcm.TRAIN, dcm.VAL}:
        raise dcm.TestSplitAccessError("manifest contains non-TRAIN/VAL split")
    if (manifest["mechanism_family"] != dcm.FAMILY_A).any():
        raise AssertionError("manifest contains non-Family-A scenario")
    if manifest["scenario_id"].duplicated().any():
        raise AssertionError("duplicate scenario_id in manifest")
    if manifest["configuration_group_id"].isna().any():
        raise AssertionError("missing configuration_group_id")
    cols = feature_columns()
    forbidden_exact = {"scenario_id", "seed", "split", "configuration_group_id", "oracle_label", "delta_J"}
    if any(c in forbidden_exact for c in cols):
        raise AssertionError("forbidden metadata included as model feature")
    if any("deadline_slack_if_admitted_now" in c for c in cols):
        raise AssertionError("invalid mixed-unit slack feature included")
    write_dataset_metadata(output_dir, manifest, args.workers, sys.argv)
    print(
        json.dumps(
            {
                "status": "prelaunch_valid",
                "n_scenarios": len(manifest),
                "n_configuration_groups": manifest["configuration_group_id"].nunique(),
                "workers": args.workers,
                "feature_count": len(cols),
            },
            indent=2,
        ),
        flush=True,
    )


def dry_run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    dry_dir = output_dir / "_dry_run"
    full_manifest = load_or_create_manifest(output_dir, args.target_scenarios)
    manifest = full_manifest.iloc[
        args.dry_run_offset : args.dry_run_offset + args.dry_run_scenarios
    ].copy()
    dry_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(dry_dir / "scenario_manifest.csv", index=False)
    stage1, stage2_selectors = fac.fit_frozen_models()
    rows: list[dict[str, Any]] = []
    summaries = []
    for _, scenario_row in manifest.iterrows():
        r, summary = run_scenario(
            scenario_row,
            stage1=stage1,
            stage2_selectors=stage2_selectors,
            max_events=min(args.max_events_per_scenario, 1),
            min_event_step_gap=args.min_event_step_gap,
            max_extra_steps=args.max_extra_steps,
        )
        rows.extend(r)
        summaries.append(summary)
    write_rows_csv(dry_dir / "dry_run_rows.csv", rows)
    if rows:
        df = pd.read_csv(dry_dir / "dry_run_rows.csv")
        expected = np.where(df["delta_J_whole"] > 0, ESTF, np.where(df["delta_J_whole"] < 0, WFS, TIE))
        assert (df["oracle_label"].to_numpy() == expected).all()
    json_dump(
        dry_dir / "dry_run_summary.json",
        {
            "n_scenarios": len(manifest),
            "n_rows": len(rows),
            "summaries": summaries,
            "sha256": sha256_file(dry_dir / "dry_run_rows.csv"),
        },
    )
    # Deterministic rerun check.
    first_sha = sha256_file(dry_dir / "dry_run_rows.csv")
    rows2: list[dict[str, Any]] = []
    for _, scenario_row in manifest.iterrows():
        r, _ = run_scenario(
            scenario_row,
            stage1=stage1,
            stage2_selectors=stage2_selectors,
            max_events=min(args.max_events_per_scenario, 1),
            min_event_step_gap=args.min_event_step_gap,
            max_extra_steps=args.max_extra_steps,
        )
        rows2.extend(r)
    write_rows_csv(dry_dir / "dry_run_rows_rerun.csv", rows2)
    second_sha = sha256_file(dry_dir / "dry_run_rows_rerun.csv")
    if first_sha != second_sha:
        raise RuntimeError("dry-run deterministic rerun checksum mismatch")
    print(json.dumps({"status": "dry_run_ok", "n_rows": len(rows), "sha256": first_sha}, indent=2), flush=True)


def run_all(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_or_create_manifest(output_dir, args.target_scenarios)
    verify_shard_disjointness(manifest, args.workers)
    write_dataset_metadata(output_dir, manifest, args.workers, sys.argv)
    shard_logs = []
    procs: list[tuple[int, subprocess.Popen[Any]]] = []
    for shard_id in range(args.workers):
        shard_log = log_dir / f"family_a_oracle_dataset_v1_1k.shard_{shard_id:03d}.log"
        shard_logs.append(str(shard_log))
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "run-shard",
            "--output-dir",
            str(output_dir),
            "--target-scenarios",
            str(args.target_scenarios),
            "--workers",
            str(args.workers),
            "--shard-id",
            str(shard_id),
            "--max-events-per-scenario",
            str(args.max_events_per_scenario),
            "--min-event-step-gap",
            str(args.min_event_step_gap),
            "--max-extra-steps",
            str(args.max_extra_steps),
        ]
        f = open(shard_log, "a")
        proc = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=f, stderr=subprocess.STDOUT, text=True)
        procs.append((shard_id, proc))
    master_status = output_dir / "run_status.json"
    json_dump(
        master_status,
        {
            "status": "running",
            "master_pid": os.getpid(),
            "workers": [{"shard_id": sid, "pid": p.pid} for sid, p in procs],
            "worker_count": args.workers,
            "shard_logs": shard_logs,
            "started_unix": time.time(),
            "output_dir": str(output_dir),
        },
    )
    print(f"MASTER started pid={os.getpid()} workers={[p.pid for _, p in procs]}", flush=True)
    while True:
        states = []
        for shard_id, proc in procs:
            states.append({"shard_id": shard_id, "pid": proc.pid, "returncode": proc.poll()})
        json_dump(
            master_status,
            {
                "status": "running",
                "master_pid": os.getpid(),
                "workers": states,
                "worker_count": args.workers,
                "shard_logs": shard_logs,
                "updated_unix": time.time(),
                "output_dir": str(output_dir),
            },
        )
        print(f"MASTER heartbeat {json.dumps(states, sort_keys=True)}", flush=True)
        if all(s["returncode"] is not None for s in states):
            break
        time.sleep(30)
    failed = [s for s in states if s["returncode"] != 0]
    if failed:
        json_dump(master_status, {"status": "failed", "workers": states, "failed": failed, "updated_unix": time.time()})
        raise SystemExit(f"worker failures: {failed}")
    merge_args = argparse.Namespace(**vars(args))
    merge_shards(merge_args)
    json_dump(master_status, {"status": "complete", "workers": states, "updated_unix": time.time(), "output_dir": str(output_dir)})
    print("MASTER complete", flush=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ["prelaunch", "dry-run", "run-shard", "run-all", "merge"]:
        sp = sub.add_parser(name)
        sp.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
        sp.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR))
        sp.add_argument("--target-scenarios", type=int, default=DEFAULT_TARGET_SCENARIOS)
        sp.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
        sp.add_argument("--max-events-per-scenario", type=int, default=DEFAULT_MAX_EVENTS_PER_SCENARIO)
        sp.add_argument("--min-event-step-gap", type=int, default=DEFAULT_MIN_EVENT_STEP_GAP)
        sp.add_argument("--max-extra-steps", type=int, default=DEFAULT_MAX_EXTRA_STEPS)
        sp.add_argument("--dry-run-scenarios", type=int, default=1)
        sp.add_argument("--dry-run-offset", type=int, default=0)
        sp.add_argument("--shard-id", type=int, default=0)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.cmd == "prelaunch":
        prelaunch_validate(args)
    elif args.cmd == "dry-run":
        dry_run(args)
    elif args.cmd == "run-shard":
        run_shard(args)
    elif args.cmd == "run-all":
        run_all(args)
    elif args.cmd == "merge":
        merge_shards(args)
    else:
        raise AssertionError(args.cmd)


if __name__ == "__main__":
    main()
