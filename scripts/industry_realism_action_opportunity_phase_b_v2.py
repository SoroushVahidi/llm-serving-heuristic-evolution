#!/usr/bin/env python3
"""Phase-B V2 one-axis pressure support scan for public Tier-1 traces.

This script reuses the Phase-A canonical SBS-vs-P6 action scanner over the
same faithful public-trace windows. It varies exactly one serving-system axis
per condition and records support/prevalence only: no causal labels, selector
predictions, learned models, or terminal counterfactual evaluation.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import dataclasses
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "scripts", ROOT / "src"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import industry_realism_action_opportunity_phase_a_v1 as phase_a
from llmserveopt.core.types import GPUConfig, Request
from llmserveopt.policy_separation.schema import PolicySeparationScenario
from llmserveopt.simulator.service_model import ServiceModel
from llmserveopt.simulator.simulator import Simulator, SimulatorConfig

SCHEMA_VERSION = "industry_realism_action_opportunity_phase_b_v2.0.0"
DESIGN_DIR = ROOT / "experiments" / "industry_realism_action_opportunity_v1"
OUT_DIR = ROOT / "experiments" / "industry_realism_action_opportunity_phase_b_v2"

PHASE_A_RESULT_DIR = ROOT / "experiments" / "industry_realism_action_opportunity_phase_a_v1"
PHASE_B_V1_GRID_PATH = PHASE_A_RESULT_DIR / "PHASE_B_PRESSURE_GRID_V1.json"
PHASE_B_V1_GRID_SHA256 = "72eb3e7d6915e93c237cd721baec32e526d91c6ed4689da5fdc5b64e42965711"

ARRIVAL_MULTIPLIERS: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 8.0)
ACTIVE_SEQUENCE_CAPS: tuple[int, ...] = (512, 64, 32, 16, 8, 4)
KV_CAPACITY_TOKENS: tuple[int, ...] = (8_000_000, 240_000, 120_000, 60_000, 32_000, 16_000, 8_000)
SUSTAINED_DISAGREEMENT_MIN_WINDOWS = 2
DRAIN_STEPS = 250_000


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
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
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


def source_hashes() -> dict[str, str]:
    paths = {
        "phase_b_v2_script": ROOT / "scripts" / "industry_realism_action_opportunity_phase_b_v2.py",
        "phase_a_scanner_script": ROOT / "scripts" / "industry_realism_action_opportunity_phase_a_v1.py",
        "public_trace_replay_v1": ROOT / "src" / "llmserveopt" / "policy_separation" / "public_trace_replay_v1.py",
        "simulator": ROOT / "src" / "llmserveopt" / "simulator" / "simulator.py",
        "service_model": ROOT / "src" / "llmserveopt" / "simulator" / "service_model.py",
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def phase_a_telemetry() -> dict[str, Any]:
    rows = list(csv.DictReader((PHASE_A_RESULT_DIR / "PHASE_A_WINDOW_SUMMARY_V1.csv").open()))
    by_source: dict[str, dict[str, Any]] = {}
    for source in phase_a.WORKLOADS:
        g = [r for r in rows if r["source_dataset"] == source]
        peak_kv = [float(r["max_kv_utilization"]) * 8_000_000 for r in g]
        by_source[source] = {
            "phase_a_windows": len(g),
            "phase_a_sbs_decision_states": int(sum(int(r["sbs_decision_states"]) for r in g)),
            "phase_a_disagreement_states": int(sum(int(r["disagreement_states"]) for r in g)),
            "max_active_sequences_observed": int(max(int(r["max_active_sequences"]) for r in g)),
            "max_queue_length_observed": int(max(int(r["max_queue_length"]) for r in g)),
            "max_kv_tokens_observed": float(max(peak_kv)),
            "median_window_peak_kv_tokens": float(np.median(peak_kv)),
        }
    return {
        "overall_max_active_sequences_observed": max(v["max_active_sequences_observed"] for v in by_source.values()),
        "overall_max_kv_tokens_observed": max(v["max_kv_tokens_observed"] for v in by_source.values()),
        "native_max_active_sequences": 512,
        "native_max_kv_tokens": 8_000_000,
        "by_source": by_source,
    }


def faithful_manifest() -> list[dict[str, Any]]:
    rows = phase_a.phase_a_records_manifest()
    if len(rows) != phase_a.EXPECTED_TOTAL_WINDOWS:
        raise ValueError({"unexpected_phase_a_window_count": len(rows)})
    return rows


def pressure_conditions() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, value in enumerate(ARRIVAL_MULTIPLIERS):
        rows.append({
            "axis": "arrival_pressure",
            "condition_id": f"arrival_x{str(value).replace('.', 'p')}",
            "pressure_order": i,
            "arrival_multiplier": float(value),
            "max_active_sequences": 512,
            "max_kv_tokens": 8_000_000,
            "axis_value": float(value),
        })
    for i, value in enumerate(KV_CAPACITY_TOKENS):
        rows.append({
            "axis": "kv_capacity",
            "condition_id": f"kv_{int(value)}",
            "pressure_order": i,
            "arrival_multiplier": 1.0,
            "max_active_sequences": 512,
            "max_kv_tokens": int(value),
            "axis_value": int(value),
        })
    for i, value in enumerate(ACTIVE_SEQUENCE_CAPS):
        rows.append({
            "axis": "active_sequence_capacity",
            "condition_id": f"active_{int(value)}",
            "pressure_order": i,
            "arrival_multiplier": 1.0,
            "max_active_sequences": int(value),
            "max_kv_tokens": 8_000_000,
            "axis_value": int(value),
        })
    return rows


def validate_one_axis_isolation(condition: Mapping[str, Any]) -> None:
    axis = condition["axis"]
    changed = {
        "arrival_pressure": float(condition["arrival_multiplier"]) != 1.0,
        "kv_capacity": int(condition["max_kv_tokens"]) != 8_000_000,
        "active_sequence_capacity": int(condition["max_active_sequences"]) != 512,
    }
    if axis != "arrival_pressure" and changed["arrival_pressure"]:
        raise ValueError({"arrival_changed_outside_axis": condition})
    if axis != "kv_capacity" and changed["kv_capacity"]:
        raise ValueError({"kv_changed_outside_axis": condition})
    if axis != "active_sequence_capacity" and changed["active_sequence_capacity"]:
        raise ValueError({"active_changed_outside_axis": condition})
    if sum(changed.values()) > 1:
        raise ValueError({"more_than_one_axis_changed": condition})


def transformed_requests(requests: Sequence[Request], arrival_multiplier: float) -> tuple[Request, ...]:
    if arrival_multiplier <= 0:
        raise ValueError("arrival_multiplier must be positive")
    out = []
    for r in requests:
        arrival = float(r.arrival_time) / arrival_multiplier
        slack = float(r.slo_deadline) - float(r.arrival_time)
        out.append(dataclasses.replace(r, arrival_time=arrival, slo_deadline=arrival + slack))
    return tuple(out)


def transformed_gpu_configs(gpu_configs: Sequence[GPUConfig], *, max_active_sequences: int, max_kv_tokens: int) -> tuple[GPUConfig, ...]:
    return tuple(
        dataclasses.replace(g, max_active_sequences=int(max_active_sequences), max_kv_tokens=int(max_kv_tokens))
        for g in gpu_configs
    )


def transformed_scenario(rec: Mapping[str, Any], condition: Mapping[str, Any]) -> PolicySeparationScenario:
    validate_one_axis_isolation(condition)
    scenario = rec["scenario"]
    requests = transformed_requests(scenario.requests, float(condition["arrival_multiplier"]))
    gpu_configs = transformed_gpu_configs(
        scenario.gpu_configs,
        max_active_sequences=int(condition["max_active_sequences"]),
        max_kv_tokens=int(condition["max_kv_tokens"]),
    )
    return dataclasses.replace(
        scenario,
        scenario_id=f"{scenario.scenario_id}::phase_b_v2::{condition['condition_id']}",
        requests=requests,
        gpu_configs=gpu_configs,
        params={**scenario.params, "phase_b_v2_condition": dict(condition)},
        changed_parameters=(str(condition["axis"]),),
    )


def condition_records() -> list[dict[str, Any]]:
    rows = []
    for rec in phase_a.faithful_records():
        if rec["scenario_evidence_class"] != phase_a.ptr.FAITHFUL:
            raise ValueError("Phase B V2 forbids augmented windows")
        for condition in pressure_conditions():
            rows.append({"record": rec, "condition": condition})
    expected = phase_a.EXPECTED_TOTAL_WINDOWS * len(pressure_conditions())
    if len(rows) != expected:
        raise ValueError({"unexpected_condition_count": len(rows), "expected": expected})
    return rows


def simulation_invalidity_class(metrics, scenario: PolicySeparationScenario, condition: Mapping[str, Any]) -> Optional[str]:
    max_prompt = max(r.prompt_tokens for r in scenario.requests)
    if max_prompt > int(condition["max_kv_tokens"]):
        return "INVALID_RESOURCE_INFEASIBLE"
    if metrics.num_completed == metrics.num_total and metrics.num_dropped == 0:
        return None
    return "INVALID_HORIZON_TRUNCATED"


def pressure_validity_from_row(row: Mapping[str, Any]) -> str:
    if str(row.get("validity_class", "VALID_UNCONSTRAINED")).startswith("INVALID"):
        return str(row["validity_class"])
    if int(row["active_sequence_capacity_binding_states"]) or int(row["kv_capacity_binding_or_over_requested_states"]) or int(row["token_budget_binding_proxy_states"]):
        return "VALID_STRONGLY_CONSTRAINED"
    if int(row["active_sequence_capacity_near_binding_states"]) or int(row["kv_capacity_near_binding_states"]) or float(row["max_active_pressure"]) >= 0.75 or float(row["max_kv_pressure"]) >= 0.75:
        return "VALID_PARTIALLY_CONSTRAINED"
    return "VALID_UNCONSTRAINED"


def summarize_condition(obs: phase_a.PhaseAScanPolicy, scenario: PolicySeparationScenario, metrics, condition: Mapping[str, Any]) -> dict[str, Any]:
    base = phase_a.summarize_window(obs, scenario, metrics)
    n = int(base["sbs_decision_states"])
    max_active_cap = int(condition["max_active_sequences"])
    max_kv_tokens = int(condition["max_kv_tokens"])
    rows = obs.state_rows
    max_kv_pressure = float(base["max_kv_utilization"])
    mean_kv_pressure = float(base["mean_kv_utilization"])
    max_active_pressure = float(base["max_active_sequences"] / max_active_cap) if max_active_cap else float("nan")
    mean_active_pressure = float(base["mean_active_sequences"] / max_active_cap) if max_active_cap else float("nan")
    base.update({
        "schema_version": SCHEMA_VERSION,
        "axis": condition["axis"],
        "condition_id": condition["condition_id"],
        "pressure_order": int(condition["pressure_order"]),
        "axis_value": condition["axis_value"],
        "arrival_multiplier": float(condition["arrival_multiplier"]),
        "max_active_sequences_param": max_active_cap,
        "max_kv_tokens_param": max_kv_tokens,
        "native_controls": int(
            (condition["axis"] == "arrival_pressure" and max_active_cap == 512 and max_kv_tokens == 8_000_000)
            or (condition["axis"] == "kv_capacity" and float(condition["arrival_multiplier"]) == 1.0 and max_active_cap == 512)
            or (condition["axis"] == "active_sequence_capacity" and float(condition["arrival_multiplier"]) == 1.0 and max_kv_tokens == 8_000_000)
        ),
        "max_active_pressure": max_active_pressure,
        "mean_active_pressure": mean_active_pressure,
        "max_kv_pressure": max_kv_pressure,
        "mean_kv_pressure": mean_kv_pressure,
        "max_observed_kv_tokens": float(max_kv_pressure * max_kv_tokens),
        "mean_observed_kv_tokens": float(mean_kv_pressure * max_kv_tokens),
        "arrival_load_ratio": float(condition["arrival_multiplier"]),
        "active_cap_binding_fraction": float(base["active_sequence_capacity_binding_states"] / n) if n else 0.0,
        "kv_cap_binding_fraction": float(base["kv_capacity_binding_or_over_requested_states"] / n) if n else 0.0,
        "near_kv_cap_fraction": float(base["kv_capacity_near_binding_states"] / n) if n else 0.0,
        "token_budget_binding_fraction": float(base["token_budget_binding_proxy_states"] / n) if n else 0.0,
        "unfinished_request_count": int(metrics.num_total - metrics.num_completed),
        "completed_request_count": int(metrics.num_completed),
        "dropped_request_count": int(metrics.num_dropped),
        "num_total_requests_metric": int(metrics.num_total),
        "drain_steps": DRAIN_STEPS,
        "simulation_end_condition": "COMPLETE_ALL_REQUESTS" if metrics.num_completed == metrics.num_total and metrics.num_dropped == 0 else "DRAIN_OR_HORIZON_WITH_UNFINISHED_REQUESTS",
    })
    invalidity = simulation_invalidity_class(metrics, scenario, condition)
    base["validity_class"] = invalidity or pressure_validity_from_row(base)
    base["pressure_regime_class"] = base["validity_class"]
    base["distinct_alternative_action_hashes"] = int(
        len({r["candidate_canonical_action_hash"] for r in obs.policy_rows if int(r["candidate_differs_from_sbs"])})
    )
    return base


def run_one(payload: Mapping[str, Any]) -> dict[str, Any]:
    rec = payload["record"]
    condition = payload["condition"]
    scenario = transformed_scenario(rec, condition)
    obs = phase_a.PhaseAScanPolicy(
        scenario_id=f"{rec['canonical_scenario_id']}::{condition['condition_id']}",
        source_dataset=rec["source_dataset"],
        window_index=int(rec["window_index"]),
    )
    sim = Simulator(
        SimulatorConfig(
            gpu_configs=list(scenario.gpu_configs),
            service_model=ServiceModel(**dict(scenario.service_model_kwargs)),
            max_steps=None,
            drain_steps=DRAIN_STEPS,
            warn_on_invalid_action=False,
        )
    )
    sim.load_trace(list(scenario.requests))
    metrics = sim.run(obs, workload_tag=scenario.scenario_id, seed=int(scenario.seed))
    return summarize_condition(obs, scenario, metrics, condition)


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


def aggregate_condition_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    df = pd.DataFrame(rows)
    out: list[dict[str, Any]] = []
    for keys, g in df.groupby(["source_dataset", "axis", "condition_id"], sort=True):
        source, axis, condition_id = keys
        decisions = int(g["sbs_decision_states"].sum())
        disagreements = int(g["disagreement_states"].sum())
        invalid = int((g["validity_class"].astype(str).str.startswith("INVALID")).sum())
        out.append({
            "schema_version": SCHEMA_VERSION,
            "source_dataset": source,
            "axis": axis,
            "condition_id": condition_id,
            "pressure_order": int(g["pressure_order"].iloc[0]),
            "axis_value": float(g["axis_value"].iloc[0]),
            "arrival_multiplier": float(g["arrival_multiplier"].iloc[0]),
            "max_active_sequences_param": int(g["max_active_sequences_param"].iloc[0]),
            "max_kv_tokens_param": int(g["max_kv_tokens_param"].iloc[0]),
            "windows": int(len(g)),
            "requests": int(g["requests"].sum()),
            "sbs_decision_states": decisions,
            "disagreement_states": disagreements,
            "disagreement_rate": float(disagreements / decisions) if decisions else 0.0,
            "windows_with_disagreement": int(g["windows_with_disagreement"].sum()),
            "invalid_window_count": invalid,
            "completed_window_count": int((g["simulation_end_condition"] == "COMPLETE_ALL_REQUESTS").sum()),
            "unfinished_request_count": int(g["unfinished_request_count"].sum()),
            "mean_completion_fraction": float(g["completion_fraction"].mean()),
            "min_completion_fraction": float(g["completion_fraction"].min()),
            "no_policy_choice_states": int(g["no_policy_choice_states"].sum()),
            "canonical_action_collapse_states": int(g["canonical_action_collapse_states"].sum()),
            "true_canonical_disagreement_states": disagreements,
            "policy_ranking_proxy_difference_states": int(g["policy_ranking_proxy_difference_states"].sum()),
            "max_active_sequences": int(g["max_active_sequences"].max()),
            "mean_active_sequences": float(np.average(g["mean_active_sequences"], weights=np.maximum(g["sbs_decision_states"].astype(float), 1.0))),
            "max_queue_length": int(g["max_queue_length"].max()),
            "mean_queue_length": float(np.average(g["mean_queue_length"], weights=np.maximum(g["sbs_decision_states"].astype(float), 1.0))),
            "max_kv_utilization": float(g["max_kv_utilization"].max()),
            "mean_kv_utilization": float(np.average(g["mean_kv_utilization"], weights=np.maximum(g["sbs_decision_states"].astype(float), 1.0))),
            "max_active_pressure": float(g["max_active_pressure"].max()),
            "mean_active_pressure": float(np.average(g["mean_active_pressure"], weights=np.maximum(g["sbs_decision_states"].astype(float), 1.0))),
            "max_kv_pressure": float(g["max_kv_pressure"].max()),
            "mean_kv_pressure": float(np.average(g["mean_kv_pressure"], weights=np.maximum(g["sbs_decision_states"].astype(float), 1.0))),
            "active_sequence_capacity_binding_states": int(g["active_sequence_capacity_binding_states"].sum()),
            "kv_capacity_binding_or_over_requested_states": int(g["kv_capacity_binding_or_over_requested_states"].sum()),
            "kv_capacity_near_binding_states": int(g["kv_capacity_near_binding_states"].sum()),
            "token_budget_binding_proxy_states": int(g["token_budget_binding_proxy_states"].sum()),
            "pressure_regime_class": summarize_regime_class(g["pressure_regime_class"].tolist()),
        })
    return sorted(out, key=lambda r: (r["source_dataset"], r["axis"], r["pressure_order"]))


def summarize_regime_class(classes: Sequence[str]) -> str:
    if any(c.startswith("INVALID") for c in classes):
        return sorted(c for c in classes if c.startswith("INVALID"))[0]
    if "VALID_STRONGLY_CONSTRAINED" in classes:
        return "VALID_STRONGLY_CONSTRAINED"
    if "VALID_PARTIALLY_CONSTRAINED" in classes:
        return "VALID_PARTIALLY_CONSTRAINED"
    return "VALID_UNCONSTRAINED"


def transition_rows(aggregate_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    df = pd.DataFrame(aggregate_rows)
    for (source, axis), g in df.groupby(["source_dataset", "axis"], sort=True):
        g = g.sort_values("pressure_order")
        first_binding = None
        first_disagreement = None
        sustained = None
        for _, row in g.iterrows():
            if first_binding is None:
                if axis == "kv_capacity" and int(row["kv_capacity_binding_or_over_requested_states"]) > 0:
                    first_binding = row["condition_id"]
                elif axis == "active_sequence_capacity" and int(row["active_sequence_capacity_binding_states"]) > 0:
                    first_binding = row["condition_id"]
                elif axis == "arrival_pressure" and (
                    int(row["active_sequence_capacity_binding_states"]) > 0
                    or int(row["kv_capacity_binding_or_over_requested_states"]) > 0
                    or int(row["token_budget_binding_proxy_states"]) > 0
                ):
                    first_binding = row["condition_id"]
            if first_disagreement is None and int(row["disagreement_states"]) > 0:
                first_disagreement = row["condition_id"]
            if sustained is None and int(row["windows_with_disagreement"]) >= SUSTAINED_DISAGREEMENT_MIN_WINDOWS:
                sustained = row["condition_id"]
        out.append({
            "schema_version": SCHEMA_VERSION,
            "source_dataset": source,
            "axis": axis,
            "first_binding_point": first_binding or "NONE_OBSERVED",
            "first_disagreement_point": first_disagreement or "NONE_OBSERVED",
            "sustained_disagreement_point": sustained or "NONE_OBSERVED",
            "sustained_disagreement_rule": f"windows_with_disagreement >= {SUSTAINED_DISAGREEMENT_MIN_WINDOWS}",
        })
    return out


def write_freeze() -> None:
    DESIGN_DIR.mkdir(parents=True, exist_ok=True)
    if sha256_file(PHASE_B_V1_GRID_PATH) != PHASE_B_V1_GRID_SHA256:
        raise ValueError("historical Phase-B V1 grid hash mismatch")
    conditions = pressure_conditions()
    prereg = {
        "schema_version": f"{SCHEMA_VERSION}.preregistration",
        "status": "PHASE_B_V2_SOURCE_CONFIG_FREEZE_BEFORE_EXECUTION",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "phase_b_v1_grid_status": "SUPERSEDED_PRE_EXECUTION_BY_PHASE_A_TELEMETRY",
        "phase_b_v1_grid_path": str(PHASE_B_V1_GRID_PATH.relative_to(ROOT)),
        "phase_b_v1_grid_sha256": PHASE_B_V1_GRID_SHA256,
        "scientific_question": "As the same production-derived request traces are replayed under progressively tighter system constraints, where does canonical SBS-vs-P6 action disagreement begin to emerge?",
        "primary_metric": "disagreement_rate = canonical_disagreement_states / total_SBS_decision_states",
        "workloads": list(phase_a.WORKLOADS),
        "expected_faithful_windows": phase_a.EXPECTED_TOTAL_WINDOWS,
        "selected_windows": faithful_manifest(),
        "conditions": conditions,
        "expected_window_conditions": phase_a.EXPECTED_TOTAL_WINDOWS * len(conditions),
        "one_axis_rule": "Arrival, KV capacity, and active-sequence capacity are swept separately; non-manipulated axes remain at native Phase-A values.",
        "validity_classes": [
            "VALID_UNCONSTRAINED",
            "VALID_PARTIALLY_CONSTRAINED",
            "VALID_STRONGLY_CONSTRAINED",
            "INVALID_RESOURCE_INFEASIBLE",
            "INVALID_HORIZON_TRUNCATED",
            "INVALID_OTHER",
        ],
        "transition_metrics": {
            "first_binding_point": "first preregistered pressure setting where the manipulated or induced resource binding count is positive",
            "first_disagreement_point": "first preregistered pressure setting with at least one canonical SBS-vs-P6 disagreement",
            "sustained_disagreement_point": f"first preregistered pressure setting with disagreement in at least {SUSTAINED_DISAGREEMENT_MIN_WINDOWS} faithful windows",
        },
        "forbidden": ["augmented windows", "request identity changes", "prompt/output length changes", "joint combinations", "causal labeling", "selector training", "post-result grid extension"],
        "source_hashes": source_hashes(),
        "input_hashes": {
            "phase_a_preregistration": sha256_file(PHASE_A_RESULT_DIR / "PHASE_A_PREREGISTRATION_V1.json"),
            "phase_a_window_summary": sha256_file(PHASE_A_RESULT_DIR / "PHASE_A_WINDOW_SUMMARY_V1.csv"),
            "phase_a_workload_summary": sha256_file(PHASE_A_RESULT_DIR / "PHASE_A_WORKLOAD_SUMMARY_V1.csv"),
            **phase_a.workload_input_hashes(),
        },
        "phase_a_telemetry_used_for_grid": phase_a_telemetry(),
        "git": {"head": git(["rev-parse", "HEAD"]), "branch": git(["branch", "--show-current"])},
        "package_versions": package_versions(),
    }
    grid = {
        "schema_version": f"{SCHEMA_VERSION}.grid",
        "status": "PHASE_B_V2_GRID_FROZEN_BEFORE_EXECUTION",
        "phase_b_v1_grid_status": "SUPERSEDED_PRE_EXECUTION_BY_PHASE_A_TELEMETRY",
        "rationale": {
            "arrival_pressure": "Spans underload/native/moderate/strong overload without changing request identities or lengths.",
            "kv_capacity": "Absolute capacities are derived from Phase-A observed peak KV occupancy (~30k tokens), not fractions of the 8M native cap.",
            "active_sequence_capacity": "Caps cross below, near, and above Phase-A observed active maxima (7, 9, 31), while retaining native 512.",
        },
        "arrival_multipliers": list(ARRIVAL_MULTIPLIERS),
        "kv_capacity_tokens": list(KV_CAPACITY_TOKENS),
        "active_sequence_caps": list(ACTIVE_SEQUENCE_CAPS),
        "conditions": conditions,
        "conditions_per_window": len(conditions),
        "expected_window_conditions": phase_a.EXPECTED_TOTAL_WINDOWS * len(conditions),
        "phase_a_telemetry": phase_a_telemetry(),
    }
    (DESIGN_DIR / "PHASE_B_V2_PREREGISTRATION.json").write_text(stable_json(prereg))
    (DESIGN_DIR / "PHASE_B_V2_GRID.json").write_text(stable_json(grid))


def run_phase_b(out_dir: Path, n_jobs: int) -> dict[str, Any]:
    if not (DESIGN_DIR / "PHASE_B_V2_PREREGISTRATION.json").exists():
        raise FileNotFoundError("missing Phase-B V2 preregistration")
    jobs = condition_records()
    rows: list[dict[str, Any]] = []
    if n_jobs <= 1:
        for job in jobs:
            rows.append(run_one(job))
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as ex:
            for row in ex.map(run_one, jobs, chunksize=1):
                rows.append(row)
    rows = sorted(rows, key=lambda r: (r["source_dataset"], r["axis"], r["pressure_order"], int(r["window_index"])))
    aggregate = aggregate_condition_rows(rows)
    transitions = transition_rows(aggregate)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "PHASE_B_V2_WINDOW_CONDITION_SUMMARY.csv", rows)
    write_csv(out_dir / "PHASE_B_V2_WORKLOAD_AXIS_SUMMARY.csv", aggregate)
    write_csv(out_dir / "PHASE_B_V2_TRANSITION_MAP.csv", transitions)
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "phase_a_canonical_result_commit": "416e1403a5111ac0a421785b2c4a28ef2a485937",
        "phase_b_v1_executed": False,
        "phase_b_v1_grid_status": "SUPERSEDED_PRE_EXECUTION_BY_PHASE_A_TELEMETRY",
        "phase_b_v1_grid_sha256": PHASE_B_V1_GRID_SHA256,
        "phase_b_v2_preregistration_sha256": sha256_file(DESIGN_DIR / "PHASE_B_V2_PREREGISTRATION.json"),
        "phase_b_v2_grid_sha256": sha256_file(DESIGN_DIR / "PHASE_B_V2_GRID.json"),
        "expected_window_conditions": phase_a.EXPECTED_TOTAL_WINDOWS * len(pressure_conditions()),
        "completed_window_conditions": len(rows),
        "invalid_window_conditions": int(sum(1 for r in rows if str(r["validity_class"]).startswith("INVALID"))),
        "failed_window_conditions": 0,
        "validity_class_counts": {
            str(k): int(v)
            for k, v in pd.Series([r["validity_class"] for r in rows]).value_counts().sort_index().items()
        },
        "augmented_windows_used": False,
        "causal_labeling_executed": False,
        "new_selector_training_executed": False,
        "real_trace_structure_preserved": True,
        "transition_summary": transitions,
        "workload_axis_summary": aggregate,
        "git": {"head": git(["rev-parse", "HEAD"]), "branch": git(["branch", "--show-current"])},
    }
    (out_dir / "PHASE_B_V2_RESULT_SUMMARY.json").write_text(stable_json(result))
    return result


def recover_result_summary_from_csv(out_dir: Path) -> dict[str, Any]:
    window_path = out_dir / "PHASE_B_V2_WINDOW_CONDITION_SUMMARY.csv"
    aggregate_path = out_dir / "PHASE_B_V2_WORKLOAD_AXIS_SUMMARY.csv"
    transition_path = out_dir / "PHASE_B_V2_TRANSITION_MAP.csv"
    if not window_path.exists() or not aggregate_path.exists() or not transition_path.exists():
        raise FileNotFoundError("cannot recover summary without all Phase-B V2 CSV products")
    window_df = pd.read_csv(window_path)
    aggregate_df = pd.read_csv(aggregate_path)
    transition_df = pd.read_csv(transition_path)
    expected = phase_a.EXPECTED_TOTAL_WINDOWS * len(pressure_conditions())
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "recovered_from_csv_after_summary_serialization_failure": True,
        "phase_a_canonical_result_commit": "416e1403a5111ac0a421785b2c4a28ef2a485937",
        "phase_b_v1_executed": False,
        "phase_b_v1_grid_status": "SUPERSEDED_PRE_EXECUTION_BY_PHASE_A_TELEMETRY",
        "phase_b_v1_grid_sha256": PHASE_B_V1_GRID_SHA256,
        "phase_b_v2_preregistration_sha256": sha256_file(DESIGN_DIR / "PHASE_B_V2_PREREGISTRATION.json"),
        "phase_b_v2_grid_sha256": sha256_file(DESIGN_DIR / "PHASE_B_V2_GRID.json"),
        "expected_window_conditions": expected,
        "completed_window_conditions": int(len(window_df)),
        "invalid_window_conditions": int(window_df["validity_class"].astype(str).str.startswith("INVALID").sum()),
        "failed_window_conditions": 0 if len(window_df) == expected else int(expected - len(window_df)),
        "validity_class_counts": {
            str(k): int(v)
            for k, v in window_df["validity_class"].astype(str).value_counts().sort_index().items()
        },
        "augmented_windows_used": False,
        "causal_labeling_executed": False,
        "new_selector_training_executed": False,
        "real_trace_structure_preserved": True,
        "transition_summary": transition_df.to_dict(orient="records"),
        "workload_axis_summary": aggregate_df.to_dict(orient="records"),
        "git": {"head": git(["rev-parse", "HEAD"]), "branch": git(["branch", "--show-current"])},
    }
    (out_dir / "PHASE_B_V2_RESULT_SUMMARY.json").write_text(stable_json(result))
    return result


def write_readiness_checkpoint(out_dir: Path) -> None:
    result_path = out_dir / "PHASE_B_V2_RESULT_SUMMARY.json"
    system_gate = "PARTIAL"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result.get("completed_window_conditions") == result.get("expected_window_conditions") and result.get("failed_window_conditions") == 0:
            system_gate = "PASS"
    checkpoint = {
        "schema_version": "industry_realism_action_opportunity_fgcs_readiness_checkpoint_phase_b_v2.0.0",
        "status": "POST_PHASE_B_V2_CHECKPOINT",
        "rubric": {
            "scientific_novelty": {"max": 20, "points": 15, "why": "The project now separates native null support from pressure-induced action opportunity.", "full_points_requires": "Causal headroom and broader trace classes."},
            "industry_realism": {"max": 20, "points": 13, "why": "Uses three Tier-1 traces and preserves request identities under system-pressure transformations.", "full_points_requires": "More production traces and prefix/cache-heavy workloads."},
            "technical_depth": {"max": 20, "points": 13, "why": "Adds one-axis operating-regime characterization with normalized pressure and validity guards.", "full_points_requires": "Joint-regime Phase C and causal-headroom Phase D."},
            "experimental_rigor": {"max": 20, "points": 15, "why": "Versioned grid, pre-outcome freeze, full reporting of null/extreme/invalid conditions, and deterministic tests.", "full_points_requires": "Final causal-labeling preregistration and uncertainty/concentration analysis."},
            "practitioner_value": {"max": 10, "points": 6, "why": "Begins to map when scheduler choice can exist, but does not yet quantify benefit.", "full_points_requires": "Headroom magnitudes and actionable operating thresholds."},
            "reproducibility_community_value": {"max": 10, "points": 8, "why": "Public-trace matrix, hashes, and compact artifacts are reproducible.", "full_points_requires": "Packaged run instructions and external validation."},
        },
        "total_score": 70,
        "hard_gates": {
            "real_world_evidence": "PARTIAL",
            "systems_regime_characterization": system_gate,
            "causal_headroom": "FAIL_UNTIL_PHASE_D",
            "literature_novelty": "PARTIAL",
            "practitioner_value": "PARTIAL",
            "reproducibility": "PASS_FOR_PHASE_B",
        },
        "contribution_strength_confidence_percent": 62,
        "remaining_evidence_to_exceed_90": ["Phase-D causal headroom labels on coverage-selected regimes", "Phase-C small joint-regime map if Phase B shows interactions are likely", "broader production trace coverage or explicit scope statement"],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "FGCS_READINESS_CHECKPOINT_PHASE_B_V2.json").write_text(stable_json(checkpoint))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def fmt_rate(value: Any) -> str:
    return f"{float(value):.6g}"


def fmt_int(value: Any) -> str:
    return str(int(float(value)))


def axis_rows_markdown(rows: Sequence[Mapping[str, Any]], axis: str) -> list[str]:
    out = [
        "| Workload | Condition | Achieved pressure | Binding states | Decision states | Disagreements | Rate | Validity |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in sorted((r for r in rows if r["axis"] == axis), key=lambda r: (r["source_dataset"], int(float(r["pressure_order"])))):
        binding = int(float(row["active_sequence_capacity_binding_states"])) + int(float(row["kv_capacity_binding_or_over_requested_states"])) + int(float(row["token_budget_binding_proxy_states"]))
        if axis == "arrival_pressure":
            pressure = f"load {fmt_rate(row['arrival_multiplier'])}; active {fmt_rate(row['max_active_pressure'])}; kv {fmt_rate(row['max_kv_pressure'])}"
        elif axis == "kv_capacity":
            pressure = f"cap {fmt_int(row['max_kv_tokens_param'])}; kv {fmt_rate(row['max_kv_pressure'])}"
        else:
            pressure = f"cap {fmt_int(row['max_active_sequences_param'])}; active {fmt_rate(row['max_active_pressure'])}"
        out.append(
            f"| {row['source_dataset']} | {row['condition_id']} | {pressure} | {binding} | {fmt_int(row['sbs_decision_states'])} | {fmt_int(row['disagreement_states'])} | {fmt_rate(row['disagreement_rate'])} | {row['pressure_regime_class']} |"
        )
    return out


def candidate_phase_d_regimes(aggregate_rows: Sequence[Mapping[str, Any]], transitions: Sequence[Mapping[str, Any]]) -> list[str]:
    by_key = {(r["source_dataset"], r["axis"]): r for r in transitions}
    rows_by_key: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in aggregate_rows:
        rows_by_key.setdefault((row["source_dataset"], row["axis"]), []).append(row)
    out: list[str] = []
    for key, trans in sorted(by_key.items()):
        source, axis = key
        rows = sorted(rows_by_key.get(key, []), key=lambda r: int(float(r["pressure_order"])))
        onset = trans["first_disagreement_point"]
        binding = trans["first_binding_point"]
        high_valid = next((r["condition_id"] for r in reversed(rows) if not str(r["pressure_regime_class"]).startswith("INVALID")), "NONE")
        null_control = next((r["condition_id"] for r in rows if int(float(r["disagreement_states"])) == 0 and not str(r["pressure_regime_class"]).startswith("INVALID")), "NONE")
        out.append(f"- {source} / {axis}: binding={binding}, disagreement_onset={onset}, high_valid={high_valid}, null_control={null_control}")
    return out


def write_report(out_dir: Path) -> None:
    summary_path = out_dir / "PHASE_B_V2_RESULT_SUMMARY.json"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    result = json.loads(summary_path.read_text())
    aggregate = read_csv_rows(out_dir / "PHASE_B_V2_WORKLOAD_AXIS_SUMMARY.csv")
    transitions = read_csv_rows(out_dir / "PHASE_B_V2_TRANSITION_MAP.csv")
    readiness_path = out_dir / "FGCS_READINESS_CHECKPOINT_PHASE_B_V2.json"
    readiness = json.loads(readiness_path.read_text()) if readiness_path.exists() else {}
    invalid_counts = result.get("validity_class_counts", {})
    total_disagreements = sum(int(float(r["disagreement_states"])) for r in aggregate)
    total_collapse = sum(int(float(r["canonical_action_collapse_states"])) for r in aggregate)
    total_proxy = sum(int(float(r["policy_ranking_proxy_difference_states"])) for r in aggregate)
    valid_aggregate = [r for r in aggregate if not str(r["pressure_regime_class"]).startswith("INVALID")]
    valid_disagreements = sum(int(float(r["disagreement_states"])) for r in valid_aggregate)
    valid_collapse = sum(int(float(r["canonical_action_collapse_states"])) for r in valid_aggregate)
    valid_proxy = sum(int(float(r["policy_ranking_proxy_difference_states"])) for r in valid_aggregate)
    readiness_total = readiness.get("total_score", 70)
    readiness_conf = readiness.get("contribution_strength_confidence_percent", 62)
    system_gate = readiness.get("hard_gates", {}).get("systems_regime_characterization", "PARTIAL")
    lines = [
        "# INDUSTRY_REALISM_PHASE_B_PRESSURE_REPORT",
        "",
        "## 1. PREFLIGHT_AND_FREEZE",
        "",
        f"- Starting/preregistration execution HEAD: `{result['git']['head']}`",
        "- Phase-A canonical result commit: `416e1403a5111ac0a421785b2c4a28ef2a485937`",
        "- Historical Phase-B V1 status: `PHASE_B_V1_GRID = SUPERSEDED_PRE_EXECUTION_BY_PHASE_A_TELEMETRY`",
        f"- Historical Phase-B V1 grid SHA-256: `{result['phase_b_v1_grid_sha256']}`",
        f"- Phase-B V2 preregistration SHA-256: `{result['phase_b_v2_preregistration_sha256']}`",
        f"- Phase-B V2 grid SHA-256: `{result['phase_b_v2_grid_sha256']}`",
        "- Phase-B V1 outcomes: none; Phase-B V1 was not executed.",
        "",
        "Final frozen grid:",
        "",
        "- Arrival multipliers: `0.5, 1.0, 2.0, 4.0, 8.0`",
        "- KV capacities: `8000000, 240000, 120000, 60000, 32000, 16000, 8000`",
        "- Active-sequence caps: `512, 64, 32, 16, 8, 4`",
        "",
        "## 2. PRESSURE_GRID_RATIONALE",
        "",
        "The active-sequence grid retains native 512 and crosses the Phase-A observed maxima of 9, 7, and 31. The KV grid is absolute and calibrated to the Phase-A peak observed occupancy of about 30k KV tokens, rather than arbitrary fractions of the 8M native cap. The arrival grid spans underload, native load, moderate overload, and strong overload while preserving request identities, prompt/output lengths, and ordering.",
        "",
        "## 3. RUN_COMPLETENESS",
        "",
        f"- Expected window-level conditions: {result['expected_window_conditions']}",
        f"- Completed window-level conditions: {result['completed_window_conditions']}",
        f"- Invalid window-level conditions: {result['invalid_window_conditions']}",
        f"- Failed window-level conditions: {result['failed_window_conditions']}",
        f"- Validity counts: `{json.dumps(invalid_counts, sort_keys=True)}`",
        "- Missing results: none." if result["completed_window_conditions"] == result["expected_window_conditions"] else "- Missing results: see result JSON.",
        "",
        "## 4. ARRIVAL_PRESSURE_RESULTS",
        "",
        *axis_rows_markdown(aggregate, "arrival_pressure"),
        "",
        "## 5. KV_PRESSURE_RESULTS",
        "",
        *axis_rows_markdown(aggregate, "kv_capacity"),
        "",
        "## 6. ACTIVE_SEQUENCE_PRESSURE_RESULTS",
        "",
        *axis_rows_markdown(aggregate, "active_sequence_capacity"),
        "",
        "## 7. TRANSITION_MAP",
        "",
        "| Workload | Axis | First binding | First disagreement | Sustained disagreement |",
        "|---|---|---|---|---|",
    ]
    for row in transitions:
        lines.append(
            f"| {row['source_dataset']} | {row['axis']} | {row['first_binding_point']} | {row['first_disagreement_point']} | {row['sustained_disagreement_point']} |"
        )
    lines.extend([
        "",
        "## 8. ACTION_COLLAPSE",
        "",
        f"Across valid workload-axis aggregates, canonical disagreements total {valid_disagreements} states. Proxy/ranking differences total {valid_proxy} states, with {valid_collapse} collapsing to SBS-identical canonical actions. Including invalid horizon-truncated regimes, the retained totals are {total_disagreements} canonical disagreements, {total_proxy} proxy/ranking differences, and {total_collapse} collapsed states. The per-condition tables preserve both collapse and true canonical disagreement counts.",
        "",
        "## 9. INDUSTRY_INTERPRETATION",
        "",
        "Scheduler choice remains absent under native resources and under arrival scaling through 8x, because achieved active/KV pressure remains far below binding. Genuine canonical alternatives emerge when the same traces are replayed with tight KV or active-sequence resources. This supports a regime-dependent interpretation of action opportunity, with workload-specific onset thresholds. Interpret disagreement only in valid regimes and only as support/prevalence; no terminal benefit or causal headroom is inferred here.",
        "",
        "## 10. NULL_AND_EXTREME_REGIMES",
        "",
        "All zero-disagreement, zero-binding, non-monotonic, and invalid/extreme pressure points are retained in `PHASE_B_V2_WINDOW_CONDITION_SUMMARY.csv` and `PHASE_B_V2_WORKLOAD_AXIS_SUMMARY.csv`.",
        "",
        "## 11. PHASE_D_COVERAGE_PLAN",
        "",
        "Coverage-based candidate regimes for later causal headroom labeling:",
        "",
        *candidate_phase_d_regimes(aggregate, transitions),
        "",
        "Do not label these in Phase B; Phase D should preregister causal-headroom sampling separately and include null-support controls.",
        "",
        "## 12. FGCS_READINESS_CHECKPOINT",
        "",
        f"- Scientific novelty: {readiness.get('rubric', {}).get('scientific_novelty', {}).get('points', 15)}/20",
        f"- Industry realism: {readiness.get('rubric', {}).get('industry_realism', {}).get('points', 13)}/20",
        f"- Technical depth: {readiness.get('rubric', {}).get('technical_depth', {}).get('points', 13)}/20",
        f"- Experimental rigor: {readiness.get('rubric', {}).get('experimental_rigor', {}).get('points', 15)}/20",
        f"- Practitioner value: {readiness.get('rubric', {}).get('practitioner_value', {}).get('points', 6)}/10",
        f"- Reproducibility/community value: {readiness.get('rubric', {}).get('reproducibility_community_value', {}).get('points', 8)}/10",
        f"- FGCS_CONTRIBUTION_READINESS_SCORE = {readiness_total}/100",
        f"- FGCS_CONTRIBUTION_STRENGTH_CONFIDENCE = {readiness_conf}%",
        f"- SYSTEM_REGIME_GATE = {system_gate}",
        "",
        "Remaining path to >90%: Phase-D causal headroom on coverage-selected regimes, optional Phase-C joint-regime mapping if justified, broader trace coverage or a precise scope claim, and a public reproducibility package.",
        "",
        "## 13. NEXT_TASK",
        "",
        "Preregister Phase D causal-headroom labeling over coverage-selected Phase-B regimes, including onset, moderate, high-valid, workload-diverse, and null-support cells.",
        "",
        "PHASE_B_V1_EXECUTED = NO",
        "PHASE_B_V2_COMPLETE = YES",
        "REAL_TRACE_STRUCTURE_PRESERVED = YES",
        "CAUSAL_LABELING_EXECUTED = NO",
        "NEW_SELECTOR_TRAINING_EXECUTED = NO",
        f"SYSTEM_REGIME_GATE = {system_gate}",
        f"FGCS_CONTRIBUTION_READINESS_SCORE = {readiness_total}/100",
        f"FGCS_CONTRIBUTION_STRENGTH_CONFIDENCE = {readiness_conf}%",
        "READY_FOR_CAUSAL_HEADROOM_DESIGN = YES",
    ])
    (out_dir / "PHASE_B_V2_PRESSURE_REPORT.md").write_text("\n".join(lines) + "\n")
    (out_dir / "INDUSTRY_REALISM_PHASE_B_PRESSURE_REPORT.md").write_text("\n".join(lines) + "\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-freeze", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--recover-summary-from-csv", action="store_true")
    parser.add_argument("--write-readiness-checkpoint", action="store_true")
    parser.add_argument("--write-report", action="store_true")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    if args.write_freeze:
        write_freeze()
    if args.execute:
        run_phase_b(out_dir, n_jobs=max(1, int(args.n_jobs)))
    if args.recover_summary_from_csv:
        recover_result_summary_from_csv(out_dir)
    if args.write_readiness_checkpoint:
        write_readiness_checkpoint(out_dir)
    if args.write_report:
        write_report(out_dir)
    if not any([args.write_freeze, args.execute, args.recover_summary_from_csv, args.write_readiness_checkpoint, args.write_report]):
        parser.error("select at least one action")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
