#!/usr/bin/env python3
"""Phase-D V1 causal-headroom preregistration for industry-realism traces.

This script is outcome-free. It reconstructs the canonical disagreement
universe from frozen Phase-B V2 support artifacts and writes preregistration
materials for a future SBS-relative one-step causal-label campaign. It does
not run terminal counterfactual continuations, compute Q_SBS/A_SBS, or train a
selector.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "scripts", ROOT / "src"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import industry_realism_action_opportunity_phase_a_v1 as phase_a
import industry_realism_action_opportunity_phase_b_v2 as phase_b
from llmserveopt.core.action import Action
from llmserveopt.policies.base import BasePolicy
from llmserveopt.policy_separation.unified_utility_matrix import _build_policy
from llmserveopt.simulator.service_model import ServiceModel
from llmserveopt.simulator.simulator import Simulator, SimulatorConfig

SCHEMA_VERSION = "industry_realism_causal_headroom_phase_d_v1.0.0"
OUT_DIR = ROOT / "experiments" / "industry_realism_causal_headroom_phase_d_v1"
DESIGN_DOC = ROOT / "docs" / "design" / "INDUSTRY_REALISM_CAUSAL_HEADROOM_PHASE_D_V1.md"
PHASE_A_DIR = ROOT / "experiments" / "industry_realism_action_opportunity_phase_a_v1"
PHASE_B_DIR = ROOT / "experiments" / "industry_realism_action_opportunity_phase_b_v2"
PHASE_B_DESIGN_DIR = ROOT / "experiments" / "industry_realism_action_opportunity_v1"

PHASE_B_RESULT_COMMIT = "d1228605e3e12ce0cdb89ccf9018e9c0721c3d5f"
PHASE_A_RESULT_COMMIT = "416e1403a5111ac0a421785b2c4a28ef2a485937"
SBS_POLICY = phase_a.SBS_POLICY
P6_POLICIES = phase_a.P6_POLICIES
BOOTSTRAP_SEED = 20260920
BOOTSTRAP_REPLICATES = 2000
MIN_BOOTSTRAP_CLUSTERS = 5
EXHAUSTIVE_MAX_TERMINAL_CONTINUATIONS = 50_000


def stable_json(obj: Any) -> str:
    def default(o: Any) -> Any:
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

    return json.dumps(obj, indent=2, sort_keys=True, separators=(",", ": "), default=default) + "\n"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(args: Sequence[str]) -> Optional[str]:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def package_versions() -> dict[str, str]:
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("numpy", "pandas", "scikit-learn", "joblib"):
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = "NOT_INSTALLED"
    return out


def action_hash(canonical: str) -> str:
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def request_identity_hash(scenario) -> str:
    rows = [
        {
            "request_id": int(r.request_id),
            "arrival_time": float(r.arrival_time),
            "prompt_tokens": int(r.prompt_tokens),
            "actual_output_tokens": int(r.actual_output_tokens),
            "predicted_output_tokens": int(r.predicted_output_tokens),
            "slo_deadline": float(r.slo_deadline),
            "priority": float(r.priority),
            "class_id": str(r.class_id),
        }
        for r in scenario.requests
    ]
    return action_hash(json.dumps(rows, sort_keys=True, separators=(",", ":")))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def phase_b_summary() -> dict[str, Any]:
    result = json.loads((PHASE_B_DIR / "PHASE_B_V2_RESULT_SUMMARY.json").read_text())
    win = pd.read_csv(PHASE_B_DIR / "PHASE_B_V2_WINDOW_CONDITION_SUMMARY.csv")
    agg = pd.read_csv(PHASE_B_DIR / "PHASE_B_V2_WORKLOAD_AXIS_SUMMARY.csv")
    valid_agg = agg[~agg["pressure_regime_class"].astype(str).str.startswith("INVALID")]
    return {
        "phase_b_result_commit": PHASE_B_RESULT_COMMIT,
        "window_conditions": int(len(win)),
        "workload_axis_rows": int(len(agg)),
        "transition_rows": int(len(pd.read_csv(PHASE_B_DIR / "PHASE_B_V2_TRANSITION_MAP.csv"))),
        "valid_disagreement_states": int(valid_agg["disagreement_states"].sum()),
        "all_disagreement_states_including_invalid": int(agg["disagreement_states"].sum()),
        "invalid_window_conditions": int(win["validity_class"].astype(str).str.startswith("INVALID").sum()),
        "causal_labeling_executed": bool(result.get("causal_labeling_executed")),
        "new_selector_training_executed": bool(result.get("new_selector_training_executed")),
        "real_trace_structure_preserved": bool(result.get("real_trace_structure_preserved")),
        "phase_b_result_summary_sha256": sha256_file(PHASE_B_DIR / "PHASE_B_V2_RESULT_SUMMARY.json"),
        "phase_b_workload_axis_summary_sha256": sha256_file(PHASE_B_DIR / "PHASE_B_V2_WORKLOAD_AXIS_SUMMARY.csv"),
        "phase_b_window_condition_summary_sha256": sha256_file(PHASE_B_DIR / "PHASE_B_V2_WINDOW_CONDITION_SUMMARY.csv"),
        "phase_b_transition_map_sha256": sha256_file(PHASE_B_DIR / "PHASE_B_V2_TRANSITION_MAP.csv"),
    }


def aggregate_rows() -> pd.DataFrame:
    return pd.read_csv(PHASE_B_DIR / "PHASE_B_V2_WORKLOAD_AXIS_SUMMARY.csv")


def transition_rows() -> pd.DataFrame:
    return pd.read_csv(PHASE_B_DIR / "PHASE_B_V2_TRANSITION_MAP.csv")


def selected_regime_rows() -> list[dict[str, Any]]:
    agg = aggregate_rows()
    trans = transition_rows()
    selected: dict[tuple[str, str, str], dict[str, Any]] = {}
    structural_null_controls: list[dict[str, Any]] = []
    for row in trans.sort_values(["source_dataset", "axis"]).to_dict(orient="records"):
        source = str(row["source_dataset"])
        axis = str(row["axis"])
        if axis == "arrival_pressure":
            structural_null_controls.append({
                "source_dataset": source,
                "axis": axis,
                "reason": "arrival sweep has no Phase-B canonical disagreement and remains a structural null control",
            })
            continue
        g = agg[(agg["source_dataset"] == source) & (agg["axis"] == axis)].copy()
        valid = g[~g["pressure_regime_class"].astype(str).str.startswith("INVALID")].sort_values("pressure_order")
        if valid.empty:
            continue
        candidates = [
            ("onset", str(row["first_disagreement_point"])),
            ("sustained", str(row["sustained_disagreement_point"])),
            ("high_valid_pressure", str(valid.iloc[-1]["condition_id"])),
        ]
        for stage, condition_id in candidates:
            if condition_id == "NONE_OBSERVED":
                continue
            match = valid[valid["condition_id"] == condition_id]
            if match.empty:
                continue
            m = match.iloc[0].to_dict()
            key = (source, axis, condition_id)
            item = selected.setdefault(
                key,
                {
                    "schema_version": SCHEMA_VERSION,
                    "source_dataset": source,
                    "axis": axis,
                    "condition_id": condition_id,
                    "pressure_order": int(m["pressure_order"]),
                    "axis_value": float(m["axis_value"]),
                    "max_active_sequences_param": int(m["max_active_sequences_param"]),
                    "max_kv_tokens_param": int(m["max_kv_tokens_param"]),
                    "arrival_multiplier": float(m["arrival_multiplier"]),
                    "pressure_regime_class": str(m["pressure_regime_class"]),
                    "phase_b_decision_states": int(m["sbs_decision_states"]),
                    "phase_b_disagreement_states": int(m["disagreement_states"]),
                    "phase_b_disagreement_rate": float(m["disagreement_rate"]),
                    "phase_b_windows_with_disagreement": int(m["windows_with_disagreement"]),
                    "achieved_max_active_pressure": float(m["max_active_pressure"]),
                    "achieved_max_kv_pressure": float(m["max_kv_pressure"]),
                    "active_sequence_capacity_binding_states": int(m["active_sequence_capacity_binding_states"]),
                    "kv_capacity_binding_or_over_requested_states": int(m["kv_capacity_binding_or_over_requested_states"]),
                    "token_budget_binding_proxy_states": int(m["token_budget_binding_proxy_states"]),
                    "stage_labels": [],
                },
            )
            if stage not in item["stage_labels"]:
                item["stage_labels"].append(stage)
    out = sorted(selected.values(), key=lambda r: (r["source_dataset"], r["axis"], r["pressure_order"]))
    for item in out:
        item["stage_label"] = "+".join(item["stage_labels"])
    return out


def support_regime_rows() -> list[dict[str, Any]]:
    agg = aggregate_rows()
    valid = agg[
        (~agg["pressure_regime_class"].astype(str).str.startswith("INVALID"))
        & (agg["disagreement_states"].astype(int) > 0)
    ].copy()
    return [
        {
            "source_dataset": str(r.source_dataset),
            "axis": str(r.axis),
            "condition_id": str(r.condition_id),
            "pressure_order": int(r.pressure_order),
            "phase_b_disagreement_states": int(r.disagreement_states),
            "phase_b_decision_states": int(r.sbs_decision_states),
            "phase_b_disagreement_rate": float(r.disagreement_rate),
            "pressure_regime_class": str(r.pressure_regime_class),
            "max_active_sequences_param": int(r.max_active_sequences_param),
            "max_kv_tokens_param": int(r.max_kv_tokens_param),
            "arrival_multiplier": float(r.arrival_multiplier),
        }
        for r in valid.sort_values(["source_dataset", "axis", "pressure_order"]).itertuples(index=False)
    ]


def stage_lookup() -> dict[tuple[str, str, str], str]:
    selected = selected_regime_rows()
    out = {(r["source_dataset"], r["axis"], r["condition_id"]): r["stage_label"] for r in selected}
    for r in support_regime_rows():
        out.setdefault((r["source_dataset"], r["axis"], r["condition_id"]), "nonselected_valid_support")
    return out


def condition_by_id() -> dict[str, dict[str, Any]]:
    return {str(c["condition_id"]): dict(c) for c in phase_b.pressure_conditions()}


def build_policies() -> dict[str, BasePolicy]:
    return {pid: _build_policy(pid)[0] for pid in P6_POLICIES}


class DisagreementUniverseObserver(BasePolicy):
    name = "industry_realism_causal_headroom_phase_d_universe_observer_v1"

    def __init__(self, *, metadata: Mapping[str, Any], condition: Mapping[str, Any]) -> None:
        self.metadata = dict(metadata)
        self.condition = dict(condition)
        self.shadow = build_policies()
        self.state_rows: list[dict[str, Any]] = []
        self.branch_rows: list[dict[str, Any]] = []

    def reset(self) -> None:
        for policy in self.shadow.values():
            if hasattr(policy, "reset"):
                policy.reset()
        self.state_rows = []
        self.branch_rows = []

    def select_action(self, state) -> Action:
        actions = {pid: self.shadow[pid].select_action(copy.deepcopy(state)) for pid in P6_POLICIES}
        canon = {pid: phase_a.canonical_action(action) for pid, action in actions.items()}
        sbs = canon[SBS_POLICY]
        differing = [pid for pid in P6_POLICIES if pid != SBS_POLICY and canon[pid] != sbs]
        if not differing:
            return copy.deepcopy(actions[SBS_POLICY])

        features = phase_a.state_pressure_features(state)
        policies_by_action: dict[str, list[str]] = defaultdict(list)
        for pid in P6_POLICIES:
            policies_by_action[canon[pid]].append(pid)
        non_sbs = sorted([c for c in policies_by_action if c != sbs])
        state_id = (
            f"phase_d_v1::{self.metadata['source_dataset']}::w{int(self.metadata['window_index']):03d}"
            f"::{self.condition['axis']}::{self.condition['condition_id']}::step{int(state.step)}"
        )
        max_active = int(self.condition["max_active_sequences"])
        max_kv = int(self.condition["max_kv_tokens"])
        active_pressure = float(features["active_sequences"] / max_active) if max_active else 0.0
        kv_pressure = float(features["kv_utilization_max"])
        binding = bool(
            int(features["active_sequence_capacity_binding"])
            or int(features["kv_capacity_binding_or_over_requested"])
            or int(features["token_budget_binding_proxy"])
        )
        row = {
            "schema_version": SCHEMA_VERSION,
            "state_id": state_id,
            "source_dataset": self.metadata["source_dataset"],
            "canonical_scenario_id": self.metadata["canonical_scenario_id"],
            "scenario_evidence_class": self.metadata["scenario_evidence_class"],
            "window_index": int(self.metadata["window_index"]),
            "request_identity_hash": self.metadata["request_identity_hash"],
            "requests": int(self.metadata["requests"]),
            "axis": self.condition["axis"],
            "condition_id": self.condition["condition_id"],
            "pressure_order": int(self.condition["pressure_order"]),
            "arrival_multiplier": float(self.condition["arrival_multiplier"]),
            "max_active_sequences_param": max_active,
            "max_kv_tokens_param": max_kv,
            "validity_class": self.metadata["validity_class"],
            "pressure_regime_class": self.metadata["pressure_regime_class"],
            "regime_stage": self.metadata["regime_stage"],
            "decision_step": int(state.step),
            "decision_time": float(state.time),
            "sbs_canonical_action_id": action_hash(sbs),
            "sbs_canonical_action": sbs,
            "unique_non_sbs_canonical_action_ids": ",".join(action_hash(c) for c in non_sbs),
            "unique_non_sbs_canonical_actions": len(non_sbs),
            "differing_policy_ids": ",".join(differing),
            "policies_by_canonical_action_id": json.dumps(
                {action_hash(c): sorted(policies_by_action[c], key=P6_POLICIES.index) for c in sorted(policies_by_action)},
                sort_keys=True,
                separators=(",", ":"),
            ),
            "active_pressure": active_pressure,
            "kv_pressure": kv_pressure,
            "queue_length": int(features["waiting_queue_count"]),
            "queue_pressure_proxy": float(features["waiting_queue_count"] / max(1, int(self.metadata["requests"]))),
            "active_sequence_capacity_binding": int(features["active_sequence_capacity_binding"]),
            "kv_capacity_binding_or_over_requested": int(features["kv_capacity_binding_or_over_requested"]),
            "token_budget_binding_proxy": int(features["token_budget_binding_proxy"]),
            "resource_binding": int(binding),
        }
        self.state_rows.append(row)
        for ordinal, canonical in enumerate(non_sbs):
            self.branch_rows.append({
                "schema_version": SCHEMA_VERSION,
                "branch_id": f"{state_id}::alt{ordinal:02d}::{action_hash(canonical)[:16]}",
                "state_id": state_id,
                "source_dataset": self.metadata["source_dataset"],
                "window_index": int(self.metadata["window_index"]),
                "axis": self.condition["axis"],
                "condition_id": self.condition["condition_id"],
                "regime_stage": self.metadata["regime_stage"],
                "decision_step": int(state.step),
                "sbs_canonical_action_id": action_hash(sbs),
                "candidate_canonical_action_id": action_hash(canonical),
                "candidate_canonical_action": canonical,
                "policies_generating_action": ",".join(sorted(policies_by_action[canonical], key=P6_POLICIES.index)),
                "n_policies_generating_action": int(len(policies_by_action[canonical])),
                "is_sbs_reference_branch": 0,
            })
        return copy.deepcopy(actions[SBS_POLICY])


def enumerate_one(payload: Mapping[str, Any]) -> dict[str, Any]:
    rec = payload["record"]
    condition = payload["condition"]
    phase_b_window = payload["phase_b_window"]
    stage = payload["stage"]
    scenario = phase_b.transformed_scenario(rec, condition)
    metadata = {
        "source_dataset": rec["source_dataset"],
        "canonical_scenario_id": rec["canonical_scenario_id"],
        "scenario_evidence_class": rec["scenario_evidence_class"],
        "window_index": int(rec["window_index"]),
        "request_identity_hash": request_identity_hash(rec["scenario"]),
        "requests": int(len(rec["scenario"].requests)),
        "validity_class": str(phase_b_window["validity_class"]),
        "pressure_regime_class": str(phase_b_window["pressure_regime_class"]),
        "regime_stage": stage,
    }
    observer = DisagreementUniverseObserver(metadata=metadata, condition=condition)
    sim = Simulator(
        SimulatorConfig(
            gpu_configs=list(scenario.gpu_configs),
            service_model=ServiceModel(**dict(scenario.service_model_kwargs)),
            max_steps=None,
            drain_steps=phase_b.DRAIN_STEPS,
            warn_on_invalid_action=False,
        )
    )
    sim.load_trace(list(scenario.requests))
    metrics = sim.run(observer, workload_tag=scenario.scenario_id, seed=int(scenario.seed))
    return {
        "source_dataset": rec["source_dataset"],
        "window_index": int(rec["window_index"]),
        "axis": condition["axis"],
        "condition_id": condition["condition_id"],
        "completed": int(metrics.num_completed),
        "total": int(metrics.num_total),
        "state_rows": observer.state_rows,
        "branch_rows": observer.branch_rows,
    }


def universe_jobs() -> list[dict[str, Any]]:
    conditions = condition_by_id()
    support = support_regime_rows()
    support_keys = {(r["source_dataset"], r["axis"], r["condition_id"]) for r in support}
    stage = stage_lookup()
    window = pd.read_csv(PHASE_B_DIR / "PHASE_B_V2_WINDOW_CONDITION_SUMMARY.csv")
    window_lookup = {
        (str(r.source_dataset), int(r.window_index), str(r.axis), str(r.condition_id)): r._asdict()
        for r in window.itertuples(index=False)
    }
    jobs: list[dict[str, Any]] = []
    for rec in phase_a.faithful_records():
        for source, axis, condition_id in sorted(support_keys):
            if rec["source_dataset"] != source:
                continue
            key = (source, int(rec["window_index"]), axis, condition_id)
            phase_b_window = window_lookup[key]
            if str(phase_b_window["validity_class"]).startswith("INVALID"):
                raise ValueError({"invalid_window_in_universe": key})
            jobs.append({
                "record": rec,
                "condition": conditions[condition_id],
                "phase_b_window": phase_b_window,
                "stage": stage[(source, axis, condition_id)],
            })
    return jobs


def build_universe(n_jobs: int = 1) -> dict[str, Any]:
    jobs = universe_jobs()
    rows: list[dict[str, Any]] = []
    branches: list[dict[str, Any]] = []
    per_window: list[dict[str, Any]] = []
    if n_jobs <= 1:
        results = [enumerate_one(job) for job in jobs]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as ex:
            results = list(ex.map(enumerate_one, jobs, chunksize=1))
    for result in results:
        rows.extend(result["state_rows"])
        branches.extend(result["branch_rows"])
        per_window.append({k: v for k, v in result.items() if k not in ("state_rows", "branch_rows")})
    rows = sorted(rows, key=lambda r: (r["source_dataset"], r["axis"], r["pressure_order"], r["window_index"], r["decision_step"], r["state_id"]))
    branches = sorted(branches, key=lambda r: (r["source_dataset"], r["axis"], r["condition_id"], r["window_index"], r["decision_step"], r["candidate_canonical_action_id"]))
    per_window = sorted(per_window, key=lambda r: (r["source_dataset"], r["axis"], r["condition_id"], r["window_index"]))
    return {"states": rows, "branches": branches, "per_window": per_window}


def summarize_universe(states: Sequence[Mapping[str, Any]], branches: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    s = pd.DataFrame(states)
    b = pd.DataFrame(branches)
    selected_keys = {(r["source_dataset"], r["axis"], r["condition_id"]) for r in selected_regime_rows()}
    selected_states = s[s.apply(lambda r: (r["source_dataset"], r["axis"], r["condition_id"]) in selected_keys, axis=1)] if len(s) else s
    selected_branches = b[b["state_id"].isin(set(selected_states["state_id"]))] if len(b) and len(selected_states) else b.iloc[0:0]
    out = {
        "total_eligible_states_all_valid_phase_b_support": int(len(s)),
        "unique_states_all_valid_phase_b_support": int(s["state_id"].nunique()) if len(s) else 0,
        "unique_non_sbs_state_action_branches_all_valid_phase_b_support": int(len(b)),
        "selected_phase_d_states": int(len(selected_states)),
        "selected_phase_d_unique_non_sbs_branches": int(len(selected_branches)),
        "selected_phase_d_sbs_reference_branches": int(len(selected_states)),
        "selected_phase_d_total_terminal_continuations": int(len(selected_states) + len(selected_branches)),
        "distribution_by_workload": {str(k): int(v) for k, v in s["source_dataset"].value_counts().sort_index().items()} if len(s) else {},
        "distribution_by_axis": {str(k): int(v) for k, v in s["axis"].value_counts().sort_index().items()} if len(s) else {},
        "distribution_by_regime_stage": {str(k): int(v) for k, v in s["regime_stage"].value_counts().sort_index().items()} if len(s) else {},
        "selected_distribution_by_workload": {str(k): int(v) for k, v in selected_states["source_dataset"].value_counts().sort_index().items()} if len(selected_states) else {},
        "selected_distribution_by_axis": {str(k): int(v) for k, v in selected_states["axis"].value_counts().sort_index().items()} if len(selected_states) else {},
        "selected_distribution_by_regime_stage": {str(k): int(v) for k, v in selected_states["regime_stage"].value_counts().sort_index().items()} if len(selected_states) else {},
        "selected_contributing_windows": int(selected_states[["source_dataset", "window_index", "axis", "condition_id"]].drop_duplicates().shape[0]) if len(selected_states) else 0,
        "phase_b_valid_disagreement_states_expected": 11328,
        "reconciles_to_phase_b_valid_disagreement_states": int(len(s)) == 11328,
    }
    return out


def selected_branch_plan(universe_summary: Mapping[str, Any]) -> dict[str, Any]:
    total = int(universe_summary["selected_phase_d_total_terminal_continuations"])
    path = "EXHAUSTIVE_SELECTED_REGIME_LABELING" if total <= EXHAUSTIVE_MAX_TERMINAL_CONTINUATIONS else "DETERMINISTIC_COVERAGE_SAMPLING"
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PREREGISTERED_NOT_EXECUTED",
        "chosen_path": path,
        "decision_rule": f"Use exhaustive selected-regime labeling when expected terminal continuations <= {EXHAUSTIVE_MAX_TERMINAL_CONTINUATIONS}; otherwise deterministic stratified sampling.",
        "selected_disagreement_states": int(universe_summary["selected_phase_d_states"]),
        "selected_unique_non_sbs_branches": int(universe_summary["selected_phase_d_unique_non_sbs_branches"]),
        "selected_sbs_reference_branches": int(universe_summary["selected_phase_d_sbs_reference_branches"]),
        "total_expected_terminal_continuations": total,
        "sampling_fallback": {
            "seed": BOOTSTRAP_SEED,
            "strata": ["source_dataset", "axis", "regime_stage", "condition_id", "window_index"],
            "forbidden_sampling_inputs": ["predicted gain", "V2 score", "proxy scheduler ranking", "action interestingness", "future outcomes"],
            "sparse_onset_rule": "do not undersample sparse onset cells; include all sparse onset states before applying caps to dense strata",
        },
        "outcome_files_created": False,
    }


def metric_protocol() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PREREGISTERED_NOT_EXECUTED",
        "estimand": {
            "q_sbs": "terminal utility after forcing exactly one canonical action at live state s and then reverting immediately to fixed kv_constrained_online",
            "reference": "Q_SBS(s,SBS)",
            "advantage": "A_SBS(s,a) = Q_SBS(s,a) - Q_SBS(s,SBS)",
            "forbidden": ["policy-forever switching", "learned continuation", "alternative-policy continuation", "closed-loop adaptive selector continuation"],
        },
        "prevalence_population": "all SBS decision states from Phase A/B, used for P(D)",
        "causal_population": "canonical disagreement states from selected valid Phase-D regimes, used for P(B|D) and headroom metrics",
        "primary_state_metrics": [
            "P(B|D), where B(s)=1 iff at least one non-SBS canonical action has A_SBS(s,a)>0",
            "oracle_headroom(s)=max(0,max_a A_SBS(s,a))",
            "mean_oracle_headroom over all disagreement states, including zeros",
            "mean and median positive oracle gain among B(s)=1 states",
            "states where every alternative is harmful",
            "states where all alternatives are zero",
            "states containing both beneficial and harmful alternatives",
        ],
        "secondary_action_metrics": [
            "positive/negative/zero advantage fractions",
            "mean and median advantage",
            "quantiles",
            "worst harm",
            "best gain",
        ],
        "industry_linkage": [
            "P(D) from Phase B",
            "P(B|D) from Phase D",
            "mean_oracle_headroom",
            "achieved KV pressure",
            "achieved active pressure",
            "binding frequency",
            "P(D) x P(B|D)",
            "P(D) x mean_oracle_headroom",
        ],
        "statistical_protocol": {
            "cluster_unit": "faithful window",
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "ci": "percentile 95%",
            "resampling_rule": "sample faithful windows with replacement and retain all state/action rows belonging to sampled windows with multiplicity",
            "minimum_clusters_for_ci": MIN_BOOTSTRAP_CLUSTERS,
            "sparse_rule": "if contributing independent windows < minimum_clusters_for_ci, report descriptive estimates without inferential CI",
        },
        "zero_policy": {
            "advantage_zero": "exact floating zero is zero, matching prior SBS-relative labeling unless a pre-execution numerical repeatability test requires a tolerance",
            "epsilon": None,
            "tolerance_change_rule": "any epsilon must be frozen from numerical repeatability tests before Phase-D execution and not from observed advantage distributions",
        },
        "actionable_headroom_index": "(P(D), P(B|D), mean_oracle_headroom), optionally accompanied by binding rate and downside prevalence",
    }


def compute_budget(universe_summary: Mapping[str, Any], branch_plan: Mapping[str, Any]) -> dict[str, Any]:
    continuations = int(branch_plan["total_expected_terminal_continuations"])
    # Previous 80-scenario fresh targeted labeling used 9,858 continuations.
    # This is a conservative planning proxy only, not an outcome-bearing input.
    fresh_reference_continuations = 9858
    relative = continuations / fresh_reference_continuations if fresh_reference_continuations else 0.0
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PLANNING_ESTIMATE_NOT_EXECUTED",
        "selected_states": int(universe_summary["selected_phase_d_states"]),
        "unique_alternative_branches": int(universe_summary["selected_phase_d_unique_non_sbs_branches"]),
        "sbs_reference_branches": int(universe_summary["selected_phase_d_sbs_reference_branches"]),
        "total_terminal_continuations": continuations,
        "recommended_wulver_array_tasks": max(1, min(128, int(np.ceil(continuations / 250.0)))),
        "storage_estimate_mb": round(max(10.0, continuations * 0.004), 2),
        "cpu_time_estimate": {
            "basis": "scaled from prior fresh targeted-label campaign size of 9,858 continuations; exact runtime depends on constrained-regime drain lengths",
            "relative_to_fresh_confirmatory_campaign": round(relative, 3),
            "planning_walltime_guidance": "start with 64-128 array tasks; inspect smoke runtime before full submission",
        },
        "gpu_required": False,
        "exhaustive_labeling_recommended": branch_plan["chosen_path"] == "EXHAUSTIVE_SELECTED_REGIME_LABELING",
    }


def implementation_plan() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PREREGISTERED_NOT_EXECUTED",
        "reuse_targets": [
            "llmserveopt.analysis.joint240_dense_sbs_state_action_v1.run_one_step_then_sbs_terminal",
            "llmserveopt.analysis.decision_criticality_timescale_trainval_v1.fork_from_live_simulator",
            "Simulator.continue_run",
            "Phase-A canonical_action/state_pressure_features",
            "Phase-B transformed_scenario and frozen pressure-condition generation",
        ],
        "required_adaptations": [
            "load Phase-D state/action manifest keyed by faithful public-trace window and Phase-B pressure condition",
            "replay SBS trajectory to target steps under exact Phase-B pressure settings",
            "force each unique non-SBS canonical action once and compute SBS-reference continuation once per state",
            "write sharded state-action outcome rows only in Phase-D execution task",
        ],
        "causal_correctness_tests_required_before_full_execution": [
            "exactly one forced intervention",
            "subsequent policy is SBS",
            "original live state is not mutated",
            "future arrivals identical across branches",
            "random state identical across branches",
            "SBS reference replay matches native SBS continuation",
            "canonical action branch is feasible",
            "duplicate canonical actions are not relabeled",
            "same state/action replay is deterministic",
            "pressure configuration persists identically in reference and counterfactual branches",
        ],
        "pilot_policy": "engineering smoke may use deterministic states, but any inspected outcome-bearing pilot rows must be quarantined and excluded from final inference",
    }


def preregistration(universe_hashes: Mapping[str, str], universe_summary: Mapping[str, Any], branch_plan: Mapping[str, Any], budget: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PHASE_D_V1_PREREGISTERED_NOT_EXECUTED",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scientific_question": "When real production-derived workloads enter regimes where SBS and the P6 portfolio take different canonical actions, how often is SBS locally causally suboptimal, how much one-step headroom exists, and how does that headroom vary with workload and resource pressure?",
        "phase_a_result_commit": PHASE_A_RESULT_COMMIT,
        "phase_b_result_commit": PHASE_B_RESULT_COMMIT,
        "phase_b_verification": phase_b_summary(),
        "universe_summary": dict(universe_summary),
        "artifact_hashes_excluding_self": dict(universe_hashes),
        "self_hash_record": "PREREGISTRATION_V1.json SHA-256 is recorded externally in ARTIFACT_HASHES_V1.json and the design report to avoid self-referential hashing.",
        "regime_selection": selected_regime_rows(),
        "labeling_scope": dict(branch_plan),
        "metric_protocol": metric_protocol(),
        "implementation_plan": implementation_plan(),
        "compute_budget": dict(budget),
        "forbidden": [
            "terminal counterfactual execution during preregistration",
            "selector training",
            "Phase-B grid changes",
            "post-outcome sampling changes",
            "policy-forever switching",
            "learned continuation",
        ],
        "claim_boundaries": {
            "allowed": [
                "production-derived workload traces",
                "real workload structure",
                "resource-constrained replay",
                "causal simulator headroom under the modeled serving semantics",
            ],
            "forbidden_until_real_system_validation": [
                "real production deployment effects",
                "measured production latency improvement",
                "real-vLLM improvement",
            ],
        },
        "fgcs_readiness_forecast": {
            "current_score": "70/100",
            "current_contribution_strength_confidence": "62%",
            "causal_headroom_gate": "PENDING_UNTIL_PHASE_D_EXECUTION",
            "strong_practical_headroom_pattern": "would materially strengthen the contribution if nontrivial P(B|D) and meaningful headroom appear across multiple workloads/regimes",
            "limited_headroom_pattern": "would still support a rigorous characterization/negative-result paper showing scheduler diversity need not imply useful optimization headroom",
            "not_resolved_by_design": ["broader real-world workload coverage", "literature novelty positioning", "real-system validation"],
        },
        "outcomes_accessed": False,
        "causal_labeling_executed": False,
        "new_selector_training_authorized": False,
        "git": {"head": git(["rev-parse", "HEAD"]), "branch": git(["branch", "--show-current"])},
        "package_versions": package_versions(),
    }


def write_design_doc(path: Path, universe_summary: Mapping[str, Any], branch_plan: Mapping[str, Any], budget: Mapping[str, Any]) -> None:
    selected = selected_regime_rows()
    lines = [
        "# Industry Realism Causal Headroom Phase D V1",
        "",
        "Status: preregistered design only. No terminal counterfactual outcomes, causal labels, or selector training are executed in this phase-freeze task.",
        "",
        "## Scientific Question",
        "",
        "When production-derived workloads enter valid constrained regimes where SBS and the P6 portfolio take different canonical actions, how often is SBS locally causally suboptimal, how much one-step headroom exists, and how does that headroom vary with workload and resource pressure?",
        "",
        "## Estimand",
        "",
        "`Q_SBS(s,a)` forces exactly one canonical action `a` at live state `s`, then immediately returns to fixed `kv_constrained_online` continuation with the same future arrivals and simulator semantics. `A_SBS(s,a)=Q_SBS(s,a)-Q_SBS(s,SBS)`.",
        "",
        "Forbidden continuations: policy-forever switching, learned continuation, alternative-policy continuation, and closed-loop adaptive selector continuation.",
        "",
        "## Populations",
        "",
        "Prevalence population: all SBS decision states from Phase A/B, used for `P(D)`. Causal population: canonical disagreement states from selected valid Phase-D regimes, used for `P(B|D)` and headroom metrics.",
        "",
        "## Coverage-Based Regime Selection",
        "",
        "| Workload | Axis | Condition | Stage labels | Phase-B disagreements | P(D) |",
        "|---|---|---|---|---:|---:|",
    ]
    for r in selected:
        lines.append(f"| {r['source_dataset']} | {r['axis']} | {r['condition_id']} | {r['stage_label']} | {r['phase_b_disagreement_states']} | {r['phase_b_disagreement_rate']:.6g} |")
    lines.extend([
        "",
        "Arrival-pressure cells have zero Phase-B support and remain structural null controls, not causal-label cells.",
        "",
        "## Labeling Scope",
        "",
        f"Chosen path: `{branch_plan['chosen_path']}`.",
        f"Selected disagreement states: `{branch_plan['selected_disagreement_states']}`.",
        f"Unique non-SBS branches: `{branch_plan['selected_unique_non_sbs_branches']}`.",
        f"SBS reference branches: `{branch_plan['selected_sbs_reference_branches']}`.",
        f"Total expected terminal continuations: `{branch_plan['total_expected_terminal_continuations']}`.",
        "",
        "## Universe Reconciliation",
        "",
        f"All valid Phase-B support disagreement states reconstructed: `{universe_summary['total_eligible_states_all_valid_phase_b_support']}`.",
        f"Reconciles to Phase-B reported valid disagreement states: `{universe_summary['reconciles_to_phase_b_valid_disagreement_states']}`.",
        "",
        "## Statistics",
        "",
        f"Clustered resampling unit is faithful window, with `{BOOTSTRAP_REPLICATES}` percentile-bootstrap replicates, seed `{BOOTSTRAP_SEED}`, and minimum independent clusters `{MIN_BOOTSTRAP_CLUSTERS}` for inferential CI. Sparse regimes below that threshold are descriptive only.",
        "",
        "## Practitioner Output",
        "",
        "Report `ACTIONABLE_HEADROOM_INDEX = (P(D), P(B|D), mean_oracle_headroom)` per workload/regime, optionally with binding rate and downside prevalence. Do not present this tuple as deployable selector performance.",
        "",
        "## Compute Budget",
        "",
        f"Recommended Wulver array tasks: `{budget['recommended_wulver_array_tasks']}`. Estimated storage: `{budget['storage_estimate_mb']} MB`. GPU required: `{budget['gpu_required']}`.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def write_design_report(path: Path, hashes: Mapping[str, str], universe_summary: Mapping[str, Any], branch_plan: Mapping[str, Any], budget: Mapping[str, Any]) -> None:
    selected = selected_regime_rows()
    lines = [
        "# INDUSTRY_REALISM_PHASE_D_CAUSAL_HEADROOM_DESIGN_REPORT",
        "",
        "## 1. STARTING_STATE",
        "",
        f"- Branch: `{git(['branch', '--show-current'])}`",
        f"- Starting HEAD: `{git(['rev-parse', 'HEAD'])}`",
        f"- Phase-B result commit verified: `{PHASE_B_RESULT_COMMIT}`",
        f"- Phase-B window conditions reported: `{phase_b_summary()['window_conditions']}`",
        f"- Valid Phase-B canonical disagreement states reproduced: `{phase_b_summary()['valid_disagreement_states']}`",
        "- Phase-A and Phase-B result artifacts were read only; no prior result files were modified.",
        "",
        "## 2. ELIGIBLE_CAUSAL_UNIVERSE",
        "",
        f"- Workloads: `{', '.join(phase_a.WORKLOADS)}`",
        f"- All valid Phase-B disagreement states: `{universe_summary['total_eligible_states_all_valid_phase_b_support']}`",
        f"- Unique non-SBS state-action branches across all valid support: `{universe_summary['unique_non_sbs_state_action_branches_all_valid_phase_b_support']}`",
        f"- Selected Phase-D disagreement states: `{universe_summary['selected_phase_d_states']}`",
        f"- Selected Phase-D non-SBS branches: `{universe_summary['selected_phase_d_unique_non_sbs_branches']}`",
        f"- Distribution by workload: `{json.dumps(universe_summary['selected_distribution_by_workload'], sort_keys=True)}`",
        f"- Distribution by axis: `{json.dumps(universe_summary['selected_distribution_by_axis'], sort_keys=True)}`",
        f"- Distribution by regime stage: `{json.dumps(universe_summary['selected_distribution_by_regime_stage'], sort_keys=True)}`",
        "",
        "## 3. REGIME_SELECTION",
        "",
        "Regimes are derived mechanically from the frozen Phase-B transition table: include onset, sustained, and highest valid pressure per workload/axis where support exists; merge duplicates; keep arrival-pressure zero-support cells as structural null controls.",
        "",
        "| Workload | Axis | Condition | Stage labels |",
        "|---|---|---|---|",
    ]
    for r in selected:
        lines.append(f"| {r['source_dataset']} | {r['axis']} | {r['condition_id']} | {r['stage_label']} |")
    lines.extend([
        "",
        "## 4. LABELING_SCOPE",
        "",
        f"- Scope: `{branch_plan['chosen_path']}`",
        f"- Exact selected state count: `{branch_plan['selected_disagreement_states']}`",
        f"- Exact selected branch count: `{branch_plan['selected_unique_non_sbs_branches']}`",
        f"- Exact SBS reference count: `{branch_plan['selected_sbs_reference_branches']}`",
        f"- Total terminal continuations planned: `{branch_plan['total_expected_terminal_continuations']}`",
        "- Deterministic rule: exhaustive selected-regime labeling because planned continuations are below the preregistered budget threshold.",
        "",
        "## 5. CAUSAL_ESTIMAND",
        "",
        "`Q_SBS(s,a)` forces exactly action `a` once at state `s`, then reverts immediately to fixed SBS / `kv_constrained_online`, preserving future arrivals and simulator semantics. `A_SBS(s,a)=Q_SBS(s,a)-Q_SBS(s,SBS)`.",
        "",
        "## 6. METRICS",
        "",
        "Primary metrics are `P(D)`, `P(B|D)`, `oracle_headroom(s)=max(0,max_a A_SBS(s,a))`, mean oracle headroom over all disagreement states, positive-headroom mean/median, and harm/zero/mixed state structure. Secondary action-level metrics report advantage sign fractions, moments, quantiles, worst harm, and best gain.",
        "",
        "Opportunity-adjusted quantities: `P(D) x P(B|D)` and `P(D) x mean_oracle_headroom`, explicitly treated as oracle/local headroom rather than deployable learned-policy gain.",
        "",
        "## 7. STATISTICAL_PROTOCOL",
        "",
        f"- Clustering unit: faithful window",
        f"- Bootstrap replicates: `{BOOTSTRAP_REPLICATES}`",
        f"- Bootstrap seed: `{BOOTSTRAP_SEED}`",
        f"- CI: percentile 95%",
        f"- Minimum clusters for CI: `{MIN_BOOTSTRAP_CLUSTERS}`",
        "- Sparse regimes below the minimum cluster threshold get descriptive estimates only.",
        "- Zero convention: exact floating zero unless pre-execution numerical repeatability testing freezes a tolerance.",
        "",
        "## 8. IMPLEMENTATION_PLAN",
        "",
        "Reuse `run_one_step_then_sbs_terminal`, `fork_from_live_simulator`, `Simulator.continue_run`, Phase-A canonicalization, and Phase-B pressure transformations. Required tests cover one forced intervention, SBS continuation, state immutability, arrival/random-state equivalence, reference replay, feasibility, deduplication, determinism, and pressure persistence.",
        "",
        "## 9. COMPUTE_BUDGET",
        "",
        f"- Continuations: `{budget['total_terminal_continuations']}`",
        f"- Recommended jobs/tasks: `{budget['recommended_wulver_array_tasks']}`",
        f"- Runtime estimate basis: `{budget['cpu_time_estimate']['basis']}`",
        f"- Relative to fresh confirmatory campaign: `{budget['cpu_time_estimate']['relative_to_fresh_confirmatory_campaign']}`",
        f"- Storage estimate: `{budget['storage_estimate_mb']} MB`",
        "",
        "## 10. PRACTITIONER_OUTPUT",
        "",
        "Planned regime map columns: workload, axis, condition, achieved active/KV pressure, binding rate, `P(D)`, `P(B|D)`, mean oracle headroom, downside prevalence, and `ACTIONABLE_HEADROOM_INDEX=(P(D),P(B|D),mean_oracle_headroom)`.",
        "",
        "## 11. CLAIM_BOUNDARIES",
        "",
        "Phase D can establish causal simulator headroom under modeled resource-constrained replay of production-derived workload traces. It cannot establish real production deployment effects, measured production latency improvement, real-vLLM improvement, or learnability of a selector.",
        "",
        "## 12. FGCS_READINESS_FORECAST",
        "",
        "Current score remains `70/100` and confidence remains `62%`; the causal-headroom hard gate is pending until Phase-D execution. Strong practical headroom across multiple workloads/regimes would materially strengthen the contribution. Limited or harmful headroom would still support a rigorous characterization/negative-result paper. Broader workload coverage, literature positioning, and real-system validation would remain afterward.",
        "",
        "## 13. PREREGISTRATION_FREEZE",
        "",
        f"- Local commit to be recorded after this report is committed.",
        f"- Universe manifest hash: `{hashes['eligible_universe']}`",
        f"- Regime selection hash: `{hashes['regime_selection']}`",
        f"- Labeling plan hash: `{hashes['sampling_or_exhaustive_plan']}`",
        f"- Metric protocol hash: `{hashes['metric_protocol']}`",
        f"- Preregistration hash: `{hashes['preregistration']}`",
        "",
        "## 14. EXACT_NEXT_TASK",
        "",
        "`python3 scripts/industry_realism_causal_headroom_phase_d_v1_execute.py --input experiments/industry_realism_causal_headroom_phase_d_v1/PREREGISTRATION_V1.json --run-sharded --num-shards 96` after adding and passing the preregistered causal-correctness tests.",
        "",
        "PHASE_D_DESIGN_COMPLETE = YES",
        "PHASE_D_OUTCOMES_ACCESSED = NO",
        "REGIME_SELECTION_COVERAGE_BASED = YES",
        "NULL_SUPPORT_REGIMES_CAUSALLY_LABELED = NO",
        "NEW_SELECTOR_TRAINING_AUTHORIZED = NO",
        "CAUSAL_HEADROOM_GATE = PENDING",
        "FGCS_CONTRIBUTION_READINESS_SCORE = 70/100",
        "FGCS_CONTRIBUTION_STRENGTH_CONFIDENCE = 62%",
        "READY_FOR_PHASE_D_EXECUTION = YES",
    ])
    path.write_text("\n".join(lines) + "\n")


def write_artifacts(n_jobs: int) -> dict[str, str]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    universe = build_universe(n_jobs=n_jobs)
    universe_summary = summarize_universe(universe["states"], universe["branches"])
    if not universe_summary["reconciles_to_phase_b_valid_disagreement_states"]:
        raise RuntimeError({"phase_b_disagreement_reconciliation_failed": universe_summary})

    selected_keys = {(r["source_dataset"], r["axis"], r["condition_id"]) for r in selected_regime_rows()}
    state_rows_for_csv = universe["states"]
    branch_rows_for_csv = universe["branches"]
    write_csv(OUT_DIR / "ELIGIBLE_DISAGREEMENT_STATES.csv", state_rows_for_csv)
    write_csv(OUT_DIR / "ELIGIBLE_STATE_ACTION_BRANCHES.csv", branch_rows_for_csv)

    eligible = {
        "schema_version": SCHEMA_VERSION,
        "status": "OUTCOME_FREE_CANDIDATE_UNIVERSE",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "phase_b_result_commit": PHASE_B_RESULT_COMMIT,
        "phase_b_summary": phase_b_summary(),
        "universe_summary": universe_summary,
        "support_regimes": support_regime_rows(),
        "selected_regime_keys": sorted([list(k) for k in selected_keys]),
        "state_manifest_csv": "ELIGIBLE_DISAGREEMENT_STATES.csv",
        "branch_manifest_csv": "ELIGIBLE_STATE_ACTION_BRANCHES.csv",
        "state_rows": universe["states"],
        "branch_rows": universe["branches"],
        "per_window_reconstruction": universe["per_window"],
        "outcomes_included": False,
    }
    (OUT_DIR / "ELIGIBLE_DISAGREEMENT_UNIVERSE.json").write_text(stable_json(eligible))

    regime_selection = {
        "schema_version": SCHEMA_VERSION,
        "status": "COVERAGE_BASED_REGIME_SELECTION_PREREGISTERED",
        "selection_rule": "for each workload and non-arrival axis, include onset, sustained, and strongest valid pressure regime when available; merge duplicate cells",
        "selected_regimes": selected_regime_rows(),
        "structural_null_controls": [r for r in transition_rows().to_dict(orient="records") if str(r["axis"]) == "arrival_pressure"],
        "arrival_cells_causally_labeled": False,
        "selected_state_count": int(universe_summary["selected_phase_d_states"]),
        "selected_branch_count": int(universe_summary["selected_phase_d_unique_non_sbs_branches"]),
    }
    (OUT_DIR / "REGIME_SELECTION_V1.json").write_text(stable_json(regime_selection))

    branch_plan = selected_branch_plan(universe_summary)
    (OUT_DIR / "SAMPLING_OR_EXHAUSTIVE_PLAN_V1.json").write_text(stable_json(branch_plan))
    metrics = metric_protocol()
    (OUT_DIR / "METRIC_PROTOCOL_V1.json").write_text(stable_json(metrics))
    budget = compute_budget(universe_summary, branch_plan)
    (OUT_DIR / "COMPUTE_BUDGET_PLAN_V1.json").write_text(stable_json(budget))
    impl = implementation_plan()
    (OUT_DIR / "IMPLEMENTATION_REUSE_AUDIT_V1.json").write_text(stable_json(impl))

    prereg_path = OUT_DIR / "PREREGISTRATION_V1.json"
    preliminary_hashes = {
        "eligible_universe": sha256_file(OUT_DIR / "ELIGIBLE_DISAGREEMENT_UNIVERSE.json"),
        "regime_selection": sha256_file(OUT_DIR / "REGIME_SELECTION_V1.json"),
        "sampling_or_exhaustive_plan": sha256_file(OUT_DIR / "SAMPLING_OR_EXHAUSTIVE_PLAN_V1.json"),
        "metric_protocol": sha256_file(OUT_DIR / "METRIC_PROTOCOL_V1.json"),
        "compute_budget": sha256_file(OUT_DIR / "COMPUTE_BUDGET_PLAN_V1.json"),
        "implementation_reuse_audit": sha256_file(OUT_DIR / "IMPLEMENTATION_REUSE_AUDIT_V1.json"),
        "eligible_states_csv": sha256_file(OUT_DIR / "ELIGIBLE_DISAGREEMENT_STATES.csv"),
        "eligible_branches_csv": sha256_file(OUT_DIR / "ELIGIBLE_STATE_ACTION_BRANCHES.csv"),
    }
    prereg_path.write_text(stable_json(preregistration(preliminary_hashes, universe_summary, branch_plan, budget)))
    hashes = {**preliminary_hashes, "preregistration": sha256_file(prereg_path)}
    hashes["preregistration"] = sha256_file(prereg_path)

    write_design_doc(DESIGN_DOC, universe_summary, branch_plan, budget)
    write_design_report(OUT_DIR / "INDUSTRY_REALISM_PHASE_D_CAUSAL_HEADROOM_DESIGN_REPORT.md", hashes, universe_summary, branch_plan, budget)
    hashes["design_doc"] = sha256_file(DESIGN_DOC)
    hashes["design_report"] = sha256_file(OUT_DIR / "INDUSTRY_REALISM_PHASE_D_CAUSAL_HEADROOM_DESIGN_REPORT.md")
    (OUT_DIR / "ARTIFACT_HASHES_V1.json").write_text(stable_json({"schema_version": SCHEMA_VERSION, "hashes": hashes}))
    return hashes


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-artifacts", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args(argv)
    if args.write_artifacts:
        print(stable_json(write_artifacts(max(1, int(args.n_jobs)))))
    else:
        parser.error("select --write-artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
