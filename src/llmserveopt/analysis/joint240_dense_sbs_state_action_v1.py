"""Dense SBS-relative state-action labels for joint240 v1.

This module freezes the one-decision override estimand:

    Q_SBS(s_t, p) = terminal utility after forcing policy p's native action
    at the exact predecision state s_t, followed by fixed SBS
    (kv_constrained_online) continuation.

    A_SBS(s_t, p) = Q_SBS(s_t, p) - Q_SBS(s_t, SBS).

It intentionally does not implement a learned scheduler.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from ..core.action import Action
from ..core.types import ObservableRequest, ObservableState, Request
from ..policies.base import BasePolicy
from ..policy_separation.hierarchical_regime_router_v1 import DWELL_MINIMUM_STEPS
from ..policy_separation.schema import PolicySeparationScenario
from ..policy_separation.unified_utility_matrix import _build_policy
from ..simulator.service_model import ServiceModel
from ..simulator.simulator import Simulator, SimulatorConfig
from . import decision_criticality_timescale_trainval_v1 as dcm
from .decision_criticality_terminal_anwg_joint240_v1 import (
    ANWG_EQ_ATOL,
    P6,
    TracingAliveRouter,
    build_p6_shadow_policies,
    fit_oof_alive_stage1_models,
    load_frozen_joint240_context,
)
from .decision_criticality_terminal_utility_joint240_v1 import (
    collect_terminal_requests,
    extract_request_rows,
    metrics_from_request_rows,
)
from .joint240_same_distribution_adaptive_v1 import LiveP6DwellRouterPolicy

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ARTIFACT_ROOT = Path(
    os.environ.get("LLMSERVEOPT_JOINT240_ARTIFACT_ROOT", str(ROOT))
).resolve()
OUT_DIR = ROOT / "experiments" / "joint240_dense_sbs_state_action_v1"

SCHEMA_VERSION = "joint240_dense_sbs_state_action_v1.0.0"
LIVE_STATE_FEATURES_VERSION = "LIVE_STATE_FEATURES_V1"
ACTION_DIFF_FEATURES_VERSION = "ACTION_DIFF_FEATURES_V1"
SBS_POLICY = "kv_constrained_online"
HISTORY_WINDOWS = (1, 5, 20)
PILOT_SEED = 20260919


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(args: Sequence[str], cwd: Path = ROOT) -> Optional[str]:
    try:
        return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()
    except Exception:
        return None


def provenance(artifact_root: Path = DEFAULT_ARTIFACT_ROOT) -> Dict[str, Any]:
    inputs = parent_artifact_paths(artifact_root)
    return {
        "schema_version": SCHEMA_VERSION,
        "git_head": _git(["rev-parse", "HEAD"]),
        "git_branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "git_status_short": _git(["status", "--short", "--branch"]),
        "hostname": platform.node(),
        "artifact_root": str(artifact_root),
        "input_sha256": {
            k: sha256_file(v) for k, v in inputs.items() if v.exists() and v.is_file()
        },
    }


def parent_artifact_paths(artifact_root: Path) -> Dict[str, Path]:
    return {
        "terminal_anwg_branches": artifact_root
        / "experiments"
        / "decision_criticality_terminal_anwg_joint240_v1"
        / "branches.csv",
        "terminal_utility_branches": artifact_root
        / "experiments"
        / "decision_criticality_terminal_utility_joint240_v1"
        / "branches.csv",
        "sbs_continuation_branches": artifact_root
        / "experiments"
        / "joint240_terminal_criticality_sbs_continuation_v1"
        / "branches_sbs_continuation.csv",
        "split_oof_folds": artifact_root
        / "experiments"
        / "joint240_same_distribution_adaptive_exploitability_v1"
        / "split_oof_folds.csv",
        "utility_matrix_wide": artifact_root
        / "experiments"
        / "joint_multimechanism_generalization_v1"
        / "utility_matrix_wide.csv",
        "scenario_manifest": artifact_root
        / "experiments"
        / "joint_multimechanism_generalization_v1"
        / "scenario_manifest.csv",
    }


def stable_state_id(scenario_id: str, step: int) -> str:
    return f"joint240::{scenario_id}::{int(step)}"


def load_and_reconcile_state_universe(artifact_root: Path = DEFAULT_ARTIFACT_ROOT) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    paths = parent_artifact_paths(artifact_root)
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError({"missing_parent_artifacts": missing})

    anwg = pd.read_csv(paths["terminal_anwg_branches"])
    util = pd.read_csv(paths["terminal_utility_branches"])
    sbs = pd.read_csv(paths["sbs_continuation_branches"])
    folds = pd.read_csv(paths["split_oof_folds"])

    key_cols = ["scenario_id", "step"]
    branch_key = ["scenario_id", "step", "acquisition_type", "alt_policy_id"]
    for name, df in [("anwg", anwg), ("util", util), ("sbs", sbs)]:
        if df[branch_key].duplicated().any():
            raise RuntimeError(f"duplicate branch keys in {name}")

    join_util = anwg[branch_key].merge(util[branch_key], on=branch_key, how="outer", indicator=True)
    join_sbs = anwg[branch_key].merge(sbs[branch_key], on=branch_key, how="outer", indicator=True)
    if not (join_util["_merge"] == "both").all():
        raise RuntimeError("terminal utility branches do not 1:1 match ANWG branches")
    if not (join_sbs["_merge"] == "both").all():
        raise RuntimeError("SBS-continuation branches do not 1:1 match ANWG branches")

    state_dupes = int(anwg.duplicated(key_cols).sum())
    states = (
        anwg.sort_values(["scenario_id", "step", "acquisition_type", "alt_policy_id"])
        .drop_duplicates(key_cols, keep="first")
        .copy()
    )
    states["state_id"] = [stable_state_id(r.scenario_id, r.step) for r in states.itertuples()]
    keep = [
        "state_id",
        "scenario_id",
        "step",
        "fold",
        "seed",
        "acquisition_type",
        "chosen_policy_id",
        "alt_policy_id",
        "canonical_ref_action",
        "canonical_alt_action",
        "n_elevated_mechanisms",
    ]
    states = states[keep]
    merged_folds = states[["scenario_id", "fold"]].drop_duplicates().merge(
        folds[["scenario_id", "fold"]],
        on="scenario_id",
        how="left",
        suffixes=("_branch", "_frozen"),
    )
    fold_mismatch = int((merged_folds["fold_branch"] != merged_folds["fold_frozen"]).sum())
    if fold_mismatch:
        raise RuntimeError(f"fold mismatch for {fold_mismatch} scenarios")

    summary = {
        "n_branch_rows": int(len(anwg)),
        "n_unique_states": int(len(states)),
        "duplicate_state_rows": state_dupes,
        "n_scenarios": int(states["scenario_id"].nunique()),
        "acquisition_type_counts_rows": anwg["acquisition_type"].value_counts().sort_index().to_dict(),
        "acquisition_type_counts_unique_states": states["acquisition_type"].value_counts().sort_index().to_dict(),
        "fold_counts_unique_states": {
            str(k): int(v) for k, v in states["fold"].value_counts().sort_index().items()
        },
        "scenario_counts_by_fold": {
            str(k): int(v)
            for k, v in states[["scenario_id", "fold"]]
            .drop_duplicates()["fold"]
            .value_counts()
            .sort_index()
            .items()
        },
        "parent_rows": {"anwg": int(len(anwg)), "utility": int(len(util)), "sbs": int(len(sbs))},
    }
    return states.sort_values(["fold", "scenario_id", "step"]).reset_index(drop=True), summary


def _q(values: Sequence[float], q: float) -> float:
    vals = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if len(vals) == 0:
        return 0.0
    return float(np.quantile(vals, q))


def _summ(values: Sequence[float], prefix: str) -> Dict[str, float]:
    vals = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not vals:
        return {
            f"{prefix}_min": 0.0,
            f"{prefix}_p25": 0.0,
            f"{prefix}_p50": 0.0,
            f"{prefix}_p75": 0.0,
            f"{prefix}_p90": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_mean": 0.0,
        }
    arr = np.asarray(vals, dtype=float)
    return {
        f"{prefix}_min": float(arr.min()),
        f"{prefix}_p25": _q(arr, 0.25),
        f"{prefix}_p50": _q(arr, 0.50),
        f"{prefix}_p75": _q(arr, 0.75),
        f"{prefix}_p90": _q(arr, 0.90),
        f"{prefix}_max": float(arr.max()),
        f"{prefix}_mean": float(arr.mean()),
    }


def _entropy(labels: Sequence[str]) -> float:
    if not labels:
        return 0.0
    counts = Counter(str(x) for x in labels)
    total = float(sum(counts.values()))
    return float(-sum((c / total) * math.log(c / total) for c in counts.values()))


def _req_id(req: ObservableRequest) -> int:
    return int(req.request_id)


def active_request_rows(state: ObservableState) -> List[Tuple[ObservableRequest, int, str]]:
    rows: List[Tuple[ObservableRequest, int, str]] = []
    for gpu in state.gpu_states:
        for req in gpu.active_requests_info:
            decoded = int(gpu.tokens_decoded_per_request.get(int(req.request_id), 0))
            rows.append((req, decoded, "active"))
        for req in gpu.incoming_migrations:
            rows.append((req, 0, "incoming_migration"))
    return rows


def admissible_request_map(state: ObservableState) -> Dict[int, Tuple[ObservableRequest, str]]:
    out: Dict[int, Tuple[ObservableRequest, str]] = {}
    for req in state.waiting_queue:
        out[_req_id(req)] = (req, "waiting")
    for req in state.migrating_queue:
        out[_req_id(req)] = (req, "migrating_ready")
    for gpu in state.gpu_states:
        for req in gpu.incoming_migrations:
            out[_req_id(req)] = (req, "incoming_migration")
    return out


@dataclass
class FeatureHistory:
    windows: Tuple[int, ...] = HISTORY_WINDOWS
    snapshots: deque = field(default_factory=lambda: deque(maxlen=max(HISTORY_WINDOWS) + 1))
    seen_request_ids: set = field(default_factory=set)
    recent_first_seen: deque = field(default_factory=lambda: deque(maxlen=max(HISTORY_WINDOWS) + 1))

    def features_before_update(self, state: ObservableState, base: Mapping[str, float]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for w in self.windows:
            if len(self.snapshots) >= w:
                prev = self.snapshots[-w]
                out[f"hist_w{w}_waiting_count_delta"] = float(base["waiting_count"] - prev["waiting_count"])
                out[f"hist_w{w}_kv_utilization_mean_delta"] = float(
                    base["kv_utilization_mean"] - prev["kv_utilization_mean"]
                )
                out[f"hist_w{w}_completed_count_delta"] = float(
                    base["completed_count_pre"] - prev["completed_count_pre"]
                )
                out[f"hist_w{w}_queued_token_mass_delta"] = float(
                    base["waiting_predicted_total_tokens_sum"]
                    - prev["waiting_predicted_total_tokens_sum"]
                )
                recent = list(self.recent_first_seen)
                arrivals = sum(recent[-w:]) if len(recent) >= w else 0
                out[f"hist_w{w}_new_request_count"] = float(arrivals)
            else:
                out[f"hist_w{w}_waiting_count_delta"] = 0.0
                out[f"hist_w{w}_kv_utilization_mean_delta"] = 0.0
                out[f"hist_w{w}_completed_count_delta"] = 0.0
                out[f"hist_w{w}_queued_token_mass_delta"] = 0.0
                out[f"hist_w{w}_new_request_count"] = 0.0
        return out

    def update(self, state: ObservableState, base: Mapping[str, float]) -> None:
        ids = set()
        ids.update(_req_id(r) for r in state.waiting_queue)
        ids.update(_req_id(r) for r in state.migrating_queue)
        for req, _, _ in active_request_rows(state):
            ids.add(_req_id(req))
        n_new = len(ids - self.seen_request_ids)
        self.seen_request_ids.update(ids)
        self.recent_first_seen.append(n_new)
        self.snapshots.append(dict(base))


def live_state_features_v1(
    state: ObservableState,
    *,
    history: Optional[FeatureHistory] = None,
    control: Optional[Mapping[str, Any]] = None,
) -> Dict[str, float]:
    waiting = list(state.waiting_queue)
    migrating = list(state.migrating_queue)
    active = active_request_rows(state)
    admissible = list(admissible_request_map(state).values())
    gpu_utils = [
        (float(g.current_kv_tokens) / float(g.max_kv_tokens)) if g.max_kv_tokens else 0.0
        for g in state.gpu_states
    ]
    total_kv = float(sum(g.max_kv_tokens for g in state.gpu_states))
    used_kv = float(sum(g.current_kv_tokens for g in state.gpu_states))
    free_kv = float(sum(g.free_kv_tokens for g in state.gpu_states))
    active_remaining = [
        max(0.0, float(req.predicted_output_tokens) - float(decoded))
        for req, decoded, _ in active
    ]
    active_prompt = [float(req.prompt_tokens) for req, _, _ in active]
    waiting_prompt = [float(req.prompt_tokens) for req in waiting]
    waiting_pred = [float(req.predicted_output_tokens) for req in waiting]
    admiss_prompt = [float(req.prompt_tokens) for req, _ in admissible]
    admiss_pred = [float(req.predicted_output_tokens) for req, _ in admissible]
    slack = [float(req.slo_deadline) - float(state.time) for req, _ in admissible]
    waiting_age = [float(state.time) - float(req.arrival_time) for req in waiting]
    priorities = [float(req.priority) for req, _ in admissible]
    waiting_est_kv = [float(req.prompt_tokens + req.predicted_output_tokens) for req in waiting]
    admiss_total = [float(req.prompt_tokens + req.predicted_output_tokens) for req, _ in admissible]
    urgent = [i for i, s in enumerate(slack) if s <= 5.0]
    base: Dict[str, float] = {
        "state_time_s": float(state.time),
        "state_step": float(state.step),
        "waiting_count": float(len(waiting)),
        "migrating_ready_count": float(len(migrating)),
        "active_count": float(len(active)),
        "admissible_count": float(len(admissible)),
        "total_in_system_count": float(len(waiting) + len(migrating) + len(active)),
        "completed_count_pre": float(state.completed_count),
        "waiting_prompt_tokens_sum": float(sum(waiting_prompt)),
        "waiting_predicted_remaining_tokens_sum": float(sum(waiting_pred)),
        "waiting_predicted_total_tokens_sum": float(sum(w + p for w, p in zip(waiting_prompt, waiting_pred))),
        "active_prompt_tokens_sum": float(sum(active_prompt)),
        "active_predicted_remaining_tokens_sum": float(sum(active_remaining)),
        "active_predicted_total_tokens_sum": float(sum(a + b for a, b in zip(active_prompt, active_remaining))),
        "kv_tokens_used_sum": used_kv,
        "kv_tokens_capacity_sum": total_kv,
        "kv_tokens_free_sum": free_kv,
        "kv_utilization_mean": float(np.mean(gpu_utils)) if gpu_utils else 0.0,
        "kv_utilization_max": float(max(gpu_utils)) if gpu_utils else 0.0,
        "waiting_estimated_kv_demand_sum": float(sum(waiting_est_kv)),
        "projected_kv_pressure_waiting": float((used_kv + sum(waiting_est_kv)) / total_kv) if total_kv else 0.0,
        "waiting_large_req_gt_total_free_kv_count": float(sum(1 for v in waiting_est_kv if v > free_kv)),
        "active_prefilling_count": float(sum(g.prefilling_count for g in state.gpu_states)),
        "active_decoding_count": float(sum(g.decoding_count for g in state.gpu_states)),
        "queued_prefill_token_mass": float(sum(waiting_prompt)),
        "active_decode_remaining_token_mass": float(sum(active_remaining)),
        "prefill_decode_request_ratio": float(len(waiting) / max(1, len(active))),
        "prefill_decode_token_ratio": float(sum(waiting_prompt) / max(1.0, sum(active_remaining))),
        "urgent_request_count_slack_le_5s": float(len(urgent)),
        "urgent_request_fraction_slack_le_5s": float(len(urgent) / max(1, len(admissible))),
        "urgent_token_mass_slack_le_5s": float(sum(admiss_total[i] for i in urgent)),
        "slo_violation_count_slack_le_0": float(sum(1 for s in slack if s <= 0.0)),
        "priority_sum_admissible": float(sum(priorities)),
        "priority_max_admissible": float(max(priorities)) if priorities else 0.0,
        "priority_entropy_admissible": _entropy([str(p) for p in priorities]),
        "class_entropy_admissible": _entropy([req.class_id for req, _ in admissible]),
        "class_count_admissible": float(len(set(req.class_id for req, _ in admissible))),
        "high_priority_waiting_prompt_mass": float(
            sum(float(r.prompt_tokens) for r in waiting if float(r.priority) >= 2.0)
        ),
    }
    base.update(_summ(waiting_age, "waiting_age_s"))
    base.update(_summ(admiss_prompt, "admissible_prompt_tokens"))
    base.update(_summ(admiss_pred, "admissible_predicted_output_tokens"))
    base.update(_summ(admiss_total, "admissible_predicted_total_tokens"))
    base.update(_summ(active_remaining, "active_predicted_remaining_tokens"))
    base.update(_summ(slack, "slack_s"))
    if history is not None:
        base.update(history.features_before_update(state, base))
    if control:
        base["traj_steps_since_policy_switch"] = float(control.get("steps_since_policy_switch", -1))
        base["traj_recent_switch_count_w20"] = float(control.get("recent_switch_count_w20", 0))
        base["traj_effective_policy_index"] = float(P6.index(control["effective_policy"])) if control.get("effective_policy") in P6 else -1.0
    return base


def live_state_feature_metadata() -> List[Dict[str, Any]]:
    dummy_names = list(live_state_features_v1(
        ObservableState(time=0.0, waiting_queue=[], gpu_states=[], completed_count=0, step=0),
        history=FeatureHistory(),
        control={"effective_policy": "", "steps_since_policy_switch": -1, "recent_switch_count_w20": 0},
    ).keys())
    rows = []
    for name in dummy_names:
        rows.append({
            "name": name,
            "definition": "Computed from ObservableState predecision snapshot; see live_state_features_v1.",
            "source_code_fields": "ObservableState, ObservableRequest, ObservableGPUState",
            "online_safe": "YES",
            "available_in_simulator": "YES",
            "expected_real_vllm_availability": "PARTIAL" if name.startswith("traj_") or "predicted" in name else "YES",
            "normalization_unit": "raw_count_or_seconds_or_tokens_or_fraction",
            "missing_value_semantics": "0 for empty sets; -1 only for unavailable trajectory-control index/duration",
        })
    return rows


def _action_ids(action: Action, verb: str) -> set:
    data = getattr(action, verb)
    ids = set()
    if verb == "migrate":
        for pairs in data.values():
            ids.update(int(rid) for rid, _ in pairs)
    else:
        for vals in data.values():
            ids.update(int(v) for v in vals)
    return ids


def canonical_full_action(action: Action) -> Tuple[Any, ...]:
    return (
        tuple(sorted((int(k), tuple(sorted(map(int, v)))) for k, v in action.admit.items() if v)),
        tuple(sorted((int(k), tuple(sorted(map(int, v)))) for k, v in action.preempt.items() if v)),
        tuple(sorted((int(k), tuple(sorted(map(int, v)))) for k, v in action.swap.items() if v)),
        tuple(sorted((int(k), tuple(sorted((int(a), int(b)) for a, b in v))) for k, v in action.migrate.items() if v)),
        tuple(sorted((int(k), tuple(sorted(map(int, v)))) for k, v in action.hold_decode.items() if v)),
        tuple(sorted((int(k), int(v)) for k, v in action.prefill_chunk_override.items())),
    )


def _selected_summary(req_ids: Iterable[int], req_map: Mapping[int, Tuple[ObservableRequest, str]], state_time: float, prefix: str) -> Dict[str, float]:
    reqs = [req_map[rid] for rid in req_ids if rid in req_map]
    prompts = [float(req.prompt_tokens) for req, _ in reqs]
    pred = [float(req.predicted_output_tokens) for req, _ in reqs]
    total = [float(req.prompt_tokens + req.predicted_output_tokens) for req, _ in reqs]
    weights = [float(req.priority) for req, _ in reqs]
    slack = [float(req.slo_deadline) - state_time for req, _ in reqs]
    age = [state_time - float(req.arrival_time) for req, _ in reqs]
    prefill = sum(1 for _, src in reqs if src == "waiting")
    decode = len(reqs) - prefill
    out = {
        f"{prefix}_count": float(len(reqs)),
        f"{prefix}_prompt_tokens_sum": float(sum(prompts)),
        f"{prefix}_predicted_remaining_tokens_sum": float(sum(pred)),
        f"{prefix}_predicted_total_tokens_sum": float(sum(total)),
        f"{prefix}_kv_demand_sum": float(sum(total)),
        f"{prefix}_priority_weight_sum": float(sum(weights)),
        f"{prefix}_urgent_count_slack_le_5s": float(sum(1 for s in slack if s <= 5.0)),
        f"{prefix}_prefill_count": float(prefill),
        f"{prefix}_decode_or_transfer_count": float(decode),
    }
    out.update(_summ(slack, f"{prefix}_slack_s"))
    out.update(_summ(age, f"{prefix}_waiting_age_s"))
    return out


def action_diff_features_v1(state: ObservableState, candidate: Action, sbs: Action) -> Dict[str, Any]:
    req_map = admissible_request_map(state)
    cand_admit = _action_ids(candidate, "admit")
    sbs_admit = _action_ids(sbs, "admit")
    cand_only = cand_admit - sbs_admit
    sbs_only = sbs_admit - cand_admit
    out: Dict[str, Any] = {
        "action_equal_to_sbs_full": canonical_full_action(candidate) == canonical_full_action(sbs),
        "action_equal_to_sbs_admit_only": dcm.canonical_action(candidate) == dcm.canonical_action(sbs),
        "admit_symmetric_difference_count": float(len(cand_admit ^ sbs_admit)),
        "candidate_admit_count": float(len(cand_admit)),
        "sbs_admit_count": float(len(sbs_admit)),
        "candidate_only_admitted_count": float(len(cand_only)),
        "sbs_only_admitted_count": float(len(sbs_only)),
        "candidate_preempt_count": float(len(_action_ids(candidate, "preempt"))),
        "sbs_preempt_count": float(len(_action_ids(sbs, "preempt"))),
        "candidate_swap_count": float(len(_action_ids(candidate, "swap"))),
        "sbs_swap_count": float(len(_action_ids(sbs, "swap"))),
        "candidate_migrate_count": float(len(_action_ids(candidate, "migrate"))),
        "sbs_migrate_count": float(len(_action_ids(sbs, "migrate"))),
        "candidate_hold_decode_count": float(len(_action_ids(candidate, "hold_decode"))),
        "sbs_hold_decode_count": float(len(_action_ids(sbs, "hold_decode"))),
        "prefill_chunk_override_equal": candidate.prefill_chunk_override == sbs.prefill_chunk_override,
    }
    c = _selected_summary(cand_only, req_map, float(state.time), "candidate_only")
    b = _selected_summary(sbs_only, req_map, float(state.time), "sbs_only")
    out.update(c)
    out.update(b)
    for stem in [
        "prompt_tokens_sum",
        "predicted_remaining_tokens_sum",
        "predicted_total_tokens_sum",
        "kv_demand_sum",
        "priority_weight_sum",
        "urgent_count_slack_le_5s",
        "prefill_count",
        "decode_or_transfer_count",
    ]:
        out[f"candidate_minus_sbs_{stem}"] = float(
            c.get(f"candidate_only_{stem}", 0.0) - b.get(f"sbs_only_{stem}", 0.0)
        )
    return out


def action_diff_feature_metadata() -> List[Dict[str, Any]]:
    names = [
        "action_equal_to_sbs_full",
        "action_equal_to_sbs_admit_only",
        "admit_symmetric_difference_count",
        "candidate_admit_count",
        "sbs_admit_count",
        "candidate_only_admitted_count",
        "sbs_only_admitted_count",
        "candidate_preempt_count",
        "sbs_preempt_count",
        "candidate_swap_count",
        "sbs_swap_count",
        "candidate_migrate_count",
        "sbs_migrate_count",
        "candidate_hold_decode_count",
        "sbs_hold_decode_count",
        "prefill_chunk_override_equal",
    ]
    for side in ("candidate_only", "sbs_only"):
        names.extend([
            f"{side}_count",
            f"{side}_prompt_tokens_sum",
            f"{side}_predicted_remaining_tokens_sum",
            f"{side}_predicted_total_tokens_sum",
            f"{side}_kv_demand_sum",
            f"{side}_priority_weight_sum",
            f"{side}_urgent_count_slack_le_5s",
            f"{side}_prefill_count",
            f"{side}_decode_or_transfer_count",
        ])
        for stat in ("min", "p25", "p50", "p75", "p90", "max", "mean"):
            names.append(f"{side}_slack_s_{stat}")
            names.append(f"{side}_waiting_age_s_{stat}")
    names.extend([
        "candidate_minus_sbs_prompt_tokens_sum",
        "candidate_minus_sbs_predicted_remaining_tokens_sum",
        "candidate_minus_sbs_predicted_total_tokens_sum",
        "candidate_minus_sbs_kv_demand_sum",
        "candidate_minus_sbs_priority_weight_sum",
        "candidate_minus_sbs_urgent_count_slack_le_5s",
        "candidate_minus_sbs_prefill_count",
        "candidate_minus_sbs_decode_or_transfer_count",
    ])
    return [
        {
            "name": n,
            "definition": "Predecision comparison between candidate policy native action and SBS native action.",
            "source_code_fields": "Action plus ObservableState admissible requests",
            "online_safe": "YES",
            "available_in_simulator": "YES",
            "expected_real_vllm_availability": "PARTIAL" if "predicted" in n or "kv" in n else "YES",
            "normalization_unit": "raw_count_or_tokens_or_boolean",
            "missing_value_semantics": "0 for empty action differences",
        }
        for n in names
    ]


def run_one_step_then_sbs_terminal(
    sim: Simulator,
    *,
    first_action: Action,
    continuation: BasePolicy,
    all_requests: Sequence[Request],
    workload_tag: str,
    seed: int,
) -> Dict[str, Any]:
    fp_before = dcm._state_fingerprint(sim)
    step_before = int(sim._step)
    cont = copy.deepcopy(continuation)
    cont.name = "sbs_continuation_after_one_step_override"  # type: ignore[attr-defined]
    fork = dcm.fork_from_live_simulator(
        sim,
        policy=cont,
        policy_id=SBS_POLICY,
        first_action=copy.deepcopy(first_action),
    )
    metrics = fork.shell.continue_run(
        cont,
        workload_tag=workload_tag,
        seed=seed,
        num_total=len(all_requests),
        all_requests=all_requests,
    )
    completed, dropped = collect_terminal_requests(fork.shell, all_requests)
    req_rows = extract_request_rows(
        all_requests=all_requests,
        completed=completed,
        dropped=dropped,
        sim_duration=float(metrics.sim_duration),
        branch_meta={},
        branch_role="dense_sbs",
    )
    utility = metrics_from_request_rows(req_rows)
    fp_after = dcm._state_fingerprint(sim)
    return {
        "q_sbs_anwg": float(metrics.arrival_normalized_weighted_goodput),
        "q_sbs_soft": float(utility["soft"]),
        "q_sbs_wcg": float(utility["wcg"]),
        "q_sbs_wmt": float(utility["wmt"]),
        "q_sbs_wnt": float(utility["wnt"]),
        "num_completed": int(metrics.num_completed),
        "num_dropped": int(metrics.num_dropped),
        "sim_duration": float(metrics.sim_duration),
        "extra_steps": int(fork.shell._step - step_before),
        "live_fingerprint_unchanged": fp_before == fp_after,
    }


@dataclass
class DenseSBSObserver(BasePolicy):
    name = "joint240_dense_sbs_state_action_observer_v1"

    sim_ref: Simulator
    tracing_router: TracingAliveRouter
    shadow_policies: Dict[str, BasePolicy]
    sbs_policy: BasePolicy
    all_requests: Sequence[Request]
    scenario_id: str
    fold: int
    seed: int
    target_steps: set

    rows: List[Dict[str, Any]] = field(default_factory=list)
    hit_steps: set = field(default_factory=set)
    feature_history: FeatureHistory = field(default_factory=FeatureHistory)
    switch_steps: deque = field(default_factory=lambda: deque(maxlen=20))
    last_effective_policy: Optional[str] = None
    last_switch_step: Optional[int] = None

    def reset(self) -> None:
        self.tracing_router.reset()
        for p in self.shadow_policies.values():
            if hasattr(p, "reset"):
                p.reset()
        if hasattr(self.sbs_policy, "reset"):
            self.sbs_policy.reset()
        self.rows = []
        self.hit_steps = set()
        self.feature_history = FeatureHistory()
        self.switch_steps = deque(maxlen=20)
        self.last_effective_policy = None
        self.last_switch_step = None

    def select_action(self, state: ObservableState) -> Action:
        real_action = self.tracing_router.select_action(state)
        effective = self.tracing_router.last_effective_policy or "weighted_fair_share"
        step = int(state.step)
        if self.last_effective_policy is None:
            self.last_effective_policy = effective
            self.last_switch_step = step
        elif effective != self.last_effective_policy:
            self.switch_steps.append(step)
            self.last_effective_policy = effective
            self.last_switch_step = step

        control = {
            "effective_policy": effective,
            "steps_since_policy_switch": step - int(self.last_switch_step or step),
            "recent_switch_count_w20": sum(1 for s in self.switch_steps if step - int(s) <= 20),
        }
        state_features = live_state_features_v1(state, history=self.feature_history, control=control)
        self.feature_history.update(state, state_features)

        if step not in self.target_steps or step in self.hit_steps:
            return real_action
        if len(state.waiting_queue) == 0 and len(state.migrating_queue) == 0:
            return real_action

        policy_actions: Dict[str, Action] = {}
        for pid in P6:
            policy = self.shadow_policies[pid]
            policy_actions[pid] = policy.select_action(copy.deepcopy(state))
        sbs_action = policy_actions[SBS_POLICY]

        q_by_policy: Dict[str, Dict[str, Any]] = {}
        for pid in P6:
            q_by_policy[pid] = run_one_step_then_sbs_terminal(
                self.sim_ref,
                first_action=policy_actions[pid],
                continuation=self.sbs_policy,
                all_requests=self.all_requests,
                workload_tag=self.scenario_id,
                seed=self.seed,
            )
        q_ref = q_by_policy[SBS_POLICY]

        for pid in P6:
            diff = action_diff_features_v1(state, policy_actions[pid], sbs_action)
            q = q_by_policy[pid]
            row = {
                "schema_version": SCHEMA_VERSION,
                "state_feature_version": LIVE_STATE_FEATURES_VERSION,
                "action_diff_feature_version": ACTION_DIFF_FEATURES_VERSION,
                "state_id": stable_state_id(self.scenario_id, step),
                "scenario_id": self.scenario_id,
                "step": step,
                "fold": int(self.fold),
                "seed": int(self.seed),
                "candidate_policy_id": pid,
                "candidate_policy_index": int(P6.index(pid)),
                "sbs_policy_id": SBS_POLICY,
                "alive_effective_policy_id": effective,
                "canonical_candidate_action": str(dcm.canonical_action(policy_actions[pid])),
                "canonical_sbs_action": str(dcm.canonical_action(sbs_action)),
                **{f"state__{k}": v for k, v in state_features.items()},
                **{f"action__{k}": v for k, v in diff.items()},
                **q,
                "q_sbs_ref_anwg": float(q_ref["q_sbs_anwg"]),
                "a_sbs_anwg": float(q["q_sbs_anwg"] - q_ref["q_sbs_anwg"]),
                "a_sbs_soft": float(q["q_sbs_soft"] - q_ref["q_sbs_soft"]),
                "a_sbs_wcg": float(q["q_sbs_wcg"] - q_ref["q_sbs_wcg"]),
                "a_sbs_wmt_improvement": float(q_ref["q_sbs_wmt"] - q["q_sbs_wmt"]),
                "a_sbs_wnt_improvement": float(q_ref["q_sbs_wnt"] - q["q_sbs_wnt"]),
            }
            self.rows.append(row)
        self.hit_steps.add(step)
        return real_action


def run_scenario_dense_sbs(
    scenario: PolicySeparationScenario,
    *,
    stage1: Pipeline,
    fold: int,
    seed: int,
    target_steps: Sequence[int],
) -> Dict[str, Any]:
    sid = scenario.scenario_id
    sim = Simulator(
        SimulatorConfig(
            gpu_configs=list(scenario.gpu_configs),
            service_model=ServiceModel(**dict(scenario.service_model_kwargs)),
            max_steps=80_000,
            drain_steps=20_000,
        )
    )
    sim.load_trace(list(scenario.requests))
    observer = DenseSBSObserver(
        sim_ref=sim,
        tracing_router=TracingAliveRouter(
            inner=LiveP6DwellRouterPolicy(stage1, P6, dwell_steps=DWELL_MINIMUM_STEPS)
        ),
        shadow_policies=build_p6_shadow_policies(),
        sbs_policy=_build_policy(SBS_POLICY)[0],
        all_requests=list(scenario.requests),
        scenario_id=sid,
        fold=fold,
        seed=seed,
        target_steps=set(int(s) for s in target_steps),
    )
    observer.sim_ref = sim
    ref_metrics = sim.run(observer, workload_tag=sid, seed=seed)
    missing = set(int(s) for s in target_steps) - observer.hit_steps
    return {
        "scenario_id": sid,
        "fold": int(fold),
        "seed": int(seed),
        "n_target_states": int(len(set(target_steps))),
        "n_hit_states": int(len(observer.hit_steps)),
        "n_missing_states": int(len(missing)),
        "missing_steps": sorted(missing),
        "reference_alive_anwg": float(ref_metrics.arrival_normalized_weighted_goodput),
        "rows": observer.rows,
    }


def deterministic_pilot_states(states: pd.DataFrame, per_fold_acq: int = 1) -> pd.DataFrame:
    df = states.copy()
    df["_h"] = [
        hashlib.sha256(f"{PILOT_SEED}:{r.state_id}:{r.acquisition_type}".encode()).hexdigest()
        for r in df.itertuples()
    ]
    parts = []
    for _, g in df.sort_values("_h").groupby(["fold", "acquisition_type"], sort=True):
        parts.append(g.head(per_fold_acq))
    return pd.concat(parts, ignore_index=True).drop(columns=["_h"]).sort_values(["fold", "scenario_id", "step"])


def run_dense_labels(
    *,
    states: pd.DataFrame,
    out_dir: Path,
    shard_index: int = 0,
    num_shards: int = 1,
    max_states: Optional[int] = None,
    resume: bool = True,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / f"dense_state_action_rows.shard{shard_index:04d}-of-{num_shards:04d}.csv"
    summary_path = out_dir / f"summary.shard{shard_index:04d}-of-{num_shards:04d}.json"
    done_path = out_dir / f"DONE.shard{shard_index:04d}-of-{num_shards:04d}"
    if resume and done_path.exists() and summary_path.exists() and rows_path.exists():
        existing = json.loads(summary_path.read_text())
        existing["resume_skipped"] = True
        print(json.dumps(existing, indent=2, sort_keys=True))
        return existing

    shard_states = states.copy()
    shard_states["_state_ord"] = np.arange(len(shard_states))
    shard_states = shard_states[shard_states["_state_ord"] % int(num_shards) == int(shard_index)]
    if max_states is not None:
        shard_states = shard_states.head(int(max_states))
    shard_states = shard_states.drop(columns=["_state_ord"])

    started = time.perf_counter()
    ctx = load_frozen_joint240_context()
    scenarios: Dict[str, PolicySeparationScenario] = ctx["scenarios"]
    needed_folds = sorted(int(x) for x in shard_states["fold"].unique())
    stage1_by_fold = fit_oof_alive_stage1_models(
        ctx["scenarios"], ctx["matrix"], ctx["folds"], fold_ids=needed_folds
    )

    rows: List[Dict[str, Any]] = []
    scenario_summaries = []
    for sid, g in shard_states.groupby("scenario_id", sort=True):
        fold = int(g["fold"].iloc[0])
        seed = int(g["seed"].iloc[0])
        result = run_scenario_dense_sbs(
            scenarios[sid],
            stage1=stage1_by_fold[fold],
            fold=fold,
            seed=seed,
            target_steps=[int(s) for s in g["step"].tolist()],
        )
        scenario_summaries.append({k: v for k, v in result.items() if k != "rows"})
        rows.extend(result["rows"])

    failed = [s for s in scenario_summaries if int(s["n_missing_states"]) > 0]
    if rows:
        with rows_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    else:
        rows_path.write_text("")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "shard_index": int(shard_index),
        "num_shards": int(num_shards),
        "n_input_states": int(len(shard_states)),
        "n_output_rows": int(len(rows)),
        "expected_rows": int(len(shard_states) * len(P6)),
        "n_scenarios": int(shard_states["scenario_id"].nunique()) if len(shard_states) else 0,
        "n_failed_scenarios": int(len(failed)),
        "failed_scenarios": failed[:20],
        "wall_seconds": float(time.perf_counter() - started),
        "rows_path": str(rows_path),
        "done": len(failed) == 0 and len(rows) == len(shard_states) * len(P6),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    sentinel = out_dir / f"{'DONE' if summary['done'] else 'FAILED'}.shard{shard_index:04d}-of-{num_shards:04d}"
    sentinel.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def aggregate_shards(out_dir: Path, num_shards: int) -> Dict[str, Any]:
    paths = [
        out_dir / f"dense_state_action_rows.shard{i:04d}-of-{num_shards:04d}.csv"
        for i in range(num_shards)
    ]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError({"missing_shards": missing[:20], "n_missing": len(missing)})
    parts = [pd.read_csv(p) for p in paths if p.stat().st_size > 0]
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    final_path = out_dir / "dense_state_action_rows.csv"
    df.to_csv(final_path, index=False)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "n_rows": int(len(df)),
        "n_states": int(df["state_id"].nunique()) if "state_id" in df else 0,
        "n_policies": int(df["candidate_policy_id"].nunique()) if "candidate_policy_id" in df else 0,
        "rows_path": str(final_path),
        "sha256": sha256_file(final_path),
    }
    (out_dir / "aggregation_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def write_design_artifacts(out_dir: Path, artifact_root: Path = DEFAULT_ARTIFACT_ROOT) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "live_state_features_v1.json").write_text(
        json.dumps(live_state_feature_metadata(), indent=2, sort_keys=True) + "\n"
    )
    (out_dir / "action_diff_features_v1.json").write_text(
        json.dumps(action_diff_feature_metadata(), indent=2, sort_keys=True) + "\n"
    )
    (out_dir / "provenance.json").write_text(json.dumps(provenance(artifact_root), indent=2, sort_keys=True) + "\n")


def cmd_prepare(args: argparse.Namespace) -> None:
    artifact_root = Path(args.artifact_root).resolve()
    out_dir = Path(args.output_dir).resolve()
    states, summary = load_and_reconcile_state_universe(artifact_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    states.to_csv(out_dir / "state_universe.csv", index=False)
    (out_dir / "state_universe_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_design_artifacts(out_dir, artifact_root)
    print(json.dumps(summary, indent=2, sort_keys=True))


def cmd_pilot(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.state_universe:
        states = pd.read_csv(args.state_universe)
    else:
        states, _ = load_and_reconcile_state_universe(Path(args.artifact_root).resolve())
    pilot = deterministic_pilot_states(states, per_fold_acq=int(args.per_fold_acq))
    pilot.to_csv(out_dir / "pilot_state_universe.csv", index=False)
    summary = run_dense_labels(states=pilot, out_dir=out_dir, max_states=args.max_states)
    print(json.dumps(summary, indent=2, sort_keys=True))


def cmd_run_shard(args: argparse.Namespace) -> None:
    states = pd.read_csv(args.state_universe)
    summary = run_dense_labels(
        states=states,
        out_dir=Path(args.output_dir).resolve(),
        shard_index=int(args.shard_index),
        num_shards=int(args.num_shards),
        max_states=args.max_states,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def cmd_aggregate(args: argparse.Namespace) -> None:
    summary = aggregate_shards(Path(args.output_dir).resolve(), int(args.num_shards))
    print(json.dumps(summary, indent=2, sort_keys=True))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact-root", default=str(DEFAULT_ARTIFACT_ROOT))
    p.add_argument("--output-dir", default=str(OUT_DIR))
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("prepare")
    sp.set_defaults(func=cmd_prepare)
    sp = sub.add_parser("pilot")
    sp.add_argument("--state-universe")
    sp.add_argument("--per-fold-acq", type=int, default=1)
    sp.add_argument("--max-states", type=int)
    sp.set_defaults(func=cmd_pilot)
    sp = sub.add_parser("run-shard")
    sp.add_argument("--state-universe", required=True)
    sp.add_argument("--shard-index", type=int, required=True)
    sp.add_argument("--num-shards", type=int, required=True)
    sp.add_argument("--max-states", type=int)
    sp.set_defaults(func=cmd_run_shard)
    sp = sub.add_parser("aggregate")
    sp.add_argument("--num-shards", type=int, required=True)
    sp.set_defaults(func=cmd_aggregate)
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
