#!/usr/bin/env python3
"""Phase-A native action-opportunity support scan for public Tier-1 traces.

This script measures canonical SBS-vs-P6 action disagreement on faithful
public-trace replay windows only. It does not run terminal counterfactual
continuations, causal labels, selectors, or learned models.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from llmserveopt.core.action import Action
from llmserveopt.policies.base import BasePolicy
from llmserveopt.policy_separation import public_trace_replay_v1 as ptr
from llmserveopt.policy_separation.unified_utility_matrix import _build_policy
from llmserveopt.simulator.service_model import ServiceModel
from llmserveopt.simulator.simulator import Simulator, SimulatorConfig

SCHEMA_VERSION = "industry_realism_action_opportunity_phase_a_v1.0.0"
OUT_DIR = ROOT / "experiments" / "industry_realism_action_opportunity_phase_a_v1"
SBS_POLICY = "kv_constrained_online"
P6_POLICIES: tuple[str, ...] = (
    "full_prefill",
    "chunked_prefill_small",
    "estimated_service_time_first",
    "weighted_fair_share",
    "least_laxity_first",
    "kv_constrained_online",
)
WORKLOADS: tuple[str, ...] = ("azure_2023_code", "azure_2023_conv", "burstgpt")
EXPECTED_WINDOWS_PER_WORKLOAD = 20
EXPECTED_TOTAL_WINDOWS = 60


def stable_json(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=True, separators=(",", ": ")) + "\n"


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


def canonical_action(action: Action) -> str:
    payload = {
        "admit": {int(k): sorted(map(int, v)) for k, v in sorted(action.admit.items()) if v},
        "preempt": {int(k): sorted(map(int, v)) for k, v in sorted(action.preempt.items()) if v},
        "swap": {int(k): sorted(map(int, v)) for k, v in sorted(action.swap.items()) if v},
        "migrate": {
            int(k): sorted((int(a), int(b)) for a, b in v)
            for k, v in sorted(action.migrate.items())
            if v
        },
        "hold_decode": {int(k): sorted(map(int, v)) for k, v in sorted(action.hold_decode.items()) if v},
        "prefill_chunk_override": {int(k): int(v) for k, v in sorted(action.prefill_chunk_override.items())},
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def action_hash(action_full: str) -> str:
    return hashlib.sha256(action_full.encode("utf-8")).hexdigest()


def faithful_records() -> list[dict[str, Any]]:
    records = [r for r in ptr.build_all_scenarios() if r["scenario_evidence_class"] == ptr.FAITHFUL]
    records.sort(key=lambda r: (r["source_dataset"], int(r["window_index"])))
    counts: dict[str, int] = {}
    for r in records:
        if r["scenario_evidence_class"] != ptr.FAITHFUL:
            raise ValueError("Phase A forbids augmented windows")
        if r["source_dataset"] not in WORKLOADS:
            raise ValueError(f"unexpected workload in Phase A: {r['source_dataset']}")
        counts[r["source_dataset"]] = counts.get(r["source_dataset"], 0) + 1
    if counts != {w: EXPECTED_WINDOWS_PER_WORKLOAD for w in WORKLOADS}:
        raise ValueError({"unexpected_faithful_window_counts": counts})
    if len(records) != EXPECTED_TOTAL_WINDOWS:
        raise ValueError({"unexpected_total_windows": len(records)})
    return records


def build_policies() -> dict[str, BasePolicy]:
    return {pid: _build_policy(pid)[0] for pid in P6_POLICIES}


def state_pressure_features(state) -> dict[str, Any]:
    waiting = list(state.waiting_queue)
    gpus = list(state.gpu_states)
    active_count = int(sum(len(g.active_request_ids) for g in gpus))
    total_free_seq = int(sum(max(0, g.free_sequences) for g in gpus))
    total_max_seq = int(sum(g.max_active_sequences for g in gpus))
    total_free_kv = int(sum(max(0, g.free_kv_tokens) for g in gpus))
    waiting_prompt = int(sum(int(r.prompt_tokens) for r in waiting))
    kv_utils = [
        (float(g.current_kv_tokens) / float(g.max_kv_tokens)) if g.max_kv_tokens else 0.0
        for g in gpus
    ]
    total_decode_slots = int(sum(max(0, getattr(g, "decoding_count", 0)) for g in gpus))
    total_prefill_slots = int(sum(max(0, getattr(g, "prefilling_count", 0)) for g in gpus))
    total_batch_budget = int(sum(int(g.max_batch_tokens) for g in gpus))
    token_budget_used = int(
        sum(
            int(getattr(g, "decoding_count", 0)) if int(getattr(g, "decoding_count", 0)) > 0 else len(g.active_request_ids)
            for g in gpus
        )
    )
    return {
        "waiting_queue_count": int(len(waiting)),
        "active_sequences": active_count,
        "total_free_sequence_slots": total_free_seq,
        "total_max_sequence_slots": total_max_seq,
        "kv_utilization_mean": float(np.mean(kv_utils)) if kv_utils else 0.0,
        "kv_utilization_max": float(np.max(kv_utils)) if kv_utils else 0.0,
        "waiting_prompt_token_mass": waiting_prompt,
        "total_free_kv_tokens": total_free_kv,
        "prefilling_count": total_prefill_slots,
        "decoding_count": total_decode_slots,
        "token_budget_used_proxy": token_budget_used,
        "total_batch_token_budget": total_batch_budget,
        "choice_state": int(len(waiting) >= 2),
        "active_sequence_capacity_binding": int(total_max_seq > 0 and total_free_seq == 0),
        "active_sequence_capacity_near_binding": int(total_max_seq > 0 and active_count / total_max_seq >= 0.9),
        "kv_capacity_binding_or_over_requested": int(waiting_prompt > total_free_kv),
        "kv_capacity_near_binding": int(any(u >= 0.9 for u in kv_utils)),
        "token_budget_binding_proxy": int(total_batch_budget > 0 and token_budget_used >= total_batch_budget),
    }


def policy_ranking_proxy_difference(state) -> int:
    waiting = list(state.waiting_queue)
    if not waiting:
        return 0
    tops = {
        min(waiting, key=lambda r: (r.arrival_time, r.request_id)).request_id,
        min(waiting, key=lambda r: (r.prompt_tokens + r.predicted_output_tokens, r.request_id)).request_id,
        min(waiting, key=lambda r: (r.slo_deadline - state.time, r.request_id)).request_id,
        max(waiting, key=lambda r: (r.priority, -r.request_id)).request_id,
    }
    return int(len(tops) > 1)


class PhaseAScanPolicy(BasePolicy):
    name = "industry_realism_action_opportunity_phase_a_v1"

    def __init__(self, scenario_id: str, source_dataset: str, window_index: int) -> None:
        self.scenario_id = scenario_id
        self.source_dataset = source_dataset
        self.window_index = int(window_index)
        self.shadow = build_policies()
        self.state_rows: list[dict[str, Any]] = []
        self.policy_rows: list[dict[str, Any]] = []

    def reset(self) -> None:
        for policy in self.shadow.values():
            if hasattr(policy, "reset"):
                policy.reset()
        self.state_rows = []
        self.policy_rows = []

    def select_action(self, state) -> Action:
        actions = {pid: self.shadow[pid].select_action(copy.deepcopy(state)) for pid in P6_POLICIES}
        canon = {pid: canonical_action(action) for pid, action in actions.items()}
        sbs = canon[SBS_POLICY]
        differing = [pid for pid in P6_POLICIES if pid != SBS_POLICY and canon[pid] != sbs]
        non_sbs_unique = {canon[pid] for pid in P6_POLICIES if pid != SBS_POLICY}
        state_id = f"phase_a::{self.scenario_id}::step{int(state.step)}"
        row = {
            "schema_version": SCHEMA_VERSION,
            "scenario_id": self.scenario_id,
            "source_dataset": self.source_dataset,
            "window_index": self.window_index,
            "state_id": state_id,
            "step": int(state.step),
            "sim_time": float(state.time),
            "sbs_canonical_action_hash": action_hash(sbs),
            "any_non_sbs_policy_differs": int(bool(differing)),
            "n_non_sbs_policies_differ": int(len(differing)),
            "n_distinct_canonical_p6_actions": int(len(set(canon.values()))),
            "n_distinct_non_sbs_canonical_actions": int(len(non_sbs_unique)),
            "differing_policy_ids": ",".join(differing),
            "policy_ranking_proxy_difference": policy_ranking_proxy_difference(state),
            "canonical_action_collapse": int(policy_ranking_proxy_difference(state) and not differing),
        }
        row.update(state_pressure_features(state))
        self.state_rows.append(row)
        for pid in P6_POLICIES:
            if pid == SBS_POLICY:
                continue
            self.policy_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "scenario_id": self.scenario_id,
                    "source_dataset": self.source_dataset,
                    "window_index": self.window_index,
                    "state_id": state_id,
                    "candidate_policy_id": pid,
                    "candidate_differs_from_sbs": int(canon[pid] != sbs),
                    "candidate_canonical_action_hash": action_hash(canon[pid]),
                    "sbs_canonical_action_hash": action_hash(sbs),
                }
            )
        return actions[SBS_POLICY]


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


def summarize_window(obs: PhaseAScanPolicy, scenario, metrics) -> dict[str, Any]:
    rows = obs.state_rows
    n = len(rows)
    disagreement = int(sum(int(r["any_non_sbs_policy_differs"]) for r in rows))
    canonical_collapse = int(sum(int(r["canonical_action_collapse"]) for r in rows))
    no_choice = int(sum(int(r["n_distinct_canonical_p6_actions"] == 1) for r in rows))
    return {
        "schema_version": SCHEMA_VERSION,
        "scenario_id": obs.scenario_id,
        "source_dataset": obs.source_dataset,
        "window_index": obs.window_index,
        "requests": int(len(scenario.requests)),
        "sbs_decision_states": n,
        "disagreement_states": disagreement,
        "disagreement_rate": float(disagreement / n) if n else 0.0,
        "windows_with_disagreement": int(disagreement > 0),
        "no_policy_choice_states": no_choice,
        "canonical_action_collapse_states": canonical_collapse,
        "true_canonical_disagreement_states": disagreement,
        "policy_ranking_proxy_difference_states": int(sum(int(r["policy_ranking_proxy_difference"]) for r in rows)),
        "unique_alternative_canonical_action_hashes": int(
            len({r["candidate_canonical_action_hash"] for r in obs.policy_rows if int(r["candidate_differs_from_sbs"])})
        ),
        "max_active_sequences": int(max((r["active_sequences"] for r in rows), default=0)),
        "mean_active_sequences": float(np.mean([r["active_sequences"] for r in rows])) if rows else 0.0,
        "max_queue_length": int(max((r["waiting_queue_count"] for r in rows), default=0)),
        "mean_queue_length": float(np.mean([r["waiting_queue_count"] for r in rows])) if rows else 0.0,
        "max_kv_utilization": float(max((r["kv_utilization_max"] for r in rows), default=0.0)),
        "mean_kv_utilization": float(np.mean([r["kv_utilization_mean"] for r in rows])) if rows else 0.0,
        "active_sequence_capacity_binding_states": int(sum(int(r["active_sequence_capacity_binding"]) for r in rows)),
        "active_sequence_capacity_near_binding_states": int(sum(int(r["active_sequence_capacity_near_binding"]) for r in rows)),
        "kv_capacity_binding_or_over_requested_states": int(sum(int(r["kv_capacity_binding_or_over_requested"]) for r in rows)),
        "kv_capacity_near_binding_states": int(sum(int(r["kv_capacity_near_binding"]) for r in rows)),
        "token_budget_binding_proxy_states": int(sum(int(r["token_budget_binding_proxy"]) for r in rows)),
        "completion_fraction": float(metrics.completion_fraction),
        "sbs_arrival_normalized_weighted_goodput": float(metrics.arrival_normalized_weighted_goodput),
    }


def aggregate_workloads(window_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    df = pd.DataFrame(window_rows)
    out: list[dict[str, Any]] = []
    for source, g in df.groupby("source_dataset", sort=True):
        decisions = int(g["sbs_decision_states"].sum())
        disagreements = int(g["disagreement_states"].sum())
        requests = int(g["requests"].sum())
        out.append(
            {
                "schema_version": SCHEMA_VERSION,
                "source_dataset": source,
                "windows": int(len(g)),
                "requests": requests,
                "sbs_decision_states": decisions,
                "disagreement_states": disagreements,
                "disagreement_rate": float(disagreements / decisions) if decisions else 0.0,
                "windows_with_disagreement": int(g["windows_with_disagreement"].sum()),
                "window_disagreement_fraction": float(g["windows_with_disagreement"].sum() / len(g)) if len(g) else 0.0,
                "no_policy_choice_states": int(g["no_policy_choice_states"].sum()),
                "canonical_action_collapse_states": int(g["canonical_action_collapse_states"].sum()),
                "true_canonical_disagreement_states": disagreements,
                "policy_ranking_proxy_difference_states": int(g["policy_ranking_proxy_difference_states"].sum()),
                "unique_alternative_canonical_action_hashes": int(g["unique_alternative_canonical_action_hashes"].sum()),
                "max_active_sequences": int(g["max_active_sequences"].max()),
                "mean_active_sequences": float(np.average(g["mean_active_sequences"], weights=g["sbs_decision_states"])),
                "max_queue_length": int(g["max_queue_length"].max()),
                "mean_queue_length": float(np.average(g["mean_queue_length"], weights=g["sbs_decision_states"])),
                "max_kv_utilization": float(g["max_kv_utilization"].max()),
                "mean_kv_utilization": float(np.average(g["mean_kv_utilization"], weights=g["sbs_decision_states"])),
                "active_sequence_capacity_binding_states": int(g["active_sequence_capacity_binding_states"].sum()),
                "active_sequence_capacity_near_binding_states": int(g["active_sequence_capacity_near_binding_states"].sum()),
                "kv_capacity_binding_or_over_requested_states": int(g["kv_capacity_binding_or_over_requested_states"].sum()),
                "kv_capacity_near_binding_states": int(g["kv_capacity_near_binding_states"].sum()),
                "token_budget_binding_proxy_states": int(g["token_budget_binding_proxy_states"].sum()),
            }
        )
    return out


def policy_disagreement_summary(policy_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not policy_rows:
        return []
    df = pd.DataFrame(policy_rows)
    out: list[dict[str, Any]] = []
    for (source, pid), g in df.groupby(["source_dataset", "candidate_policy_id"], sort=True):
        out.append(
            {
                "source_dataset": source,
                "candidate_policy_id": pid,
                "state_policy_rows": int(len(g)),
                "differs_from_sbs_states": int(g["candidate_differs_from_sbs"].astype(int).sum()),
                "differs_from_sbs_rate": float(g["candidate_differs_from_sbs"].astype(int).mean()) if len(g) else 0.0,
                "unique_candidate_action_hashes_when_different": int(
                    g.loc[g["candidate_differs_from_sbs"].astype(int) == 1, "candidate_canonical_action_hash"].nunique()
                ),
            }
        )
    return out


def run_phase_a(out_dir: Path) -> dict[str, Any]:
    records = faithful_records()
    window_rows: list[dict[str, Any]] = []
    all_policy_rows: list[dict[str, Any]] = []
    for rec in records:
        if rec["scenario_evidence_class"] != ptr.FAITHFUL:
            raise ValueError("augmented window reached run path")
        scenario = rec["scenario"]
        obs = PhaseAScanPolicy(
            scenario_id=rec["canonical_scenario_id"],
            source_dataset=rec["source_dataset"],
            window_index=int(rec["window_index"]),
        )
        sim = Simulator(
            SimulatorConfig(
                gpu_configs=list(scenario.gpu_configs),
                service_model=ServiceModel(**dict(scenario.service_model_kwargs)),
                max_steps=None,
                drain_steps=50_000,
            )
        )
        sim.load_trace(list(scenario.requests))
        metrics = sim.run(obs, workload_tag=scenario.scenario_id, seed=int(scenario.seed))
        window_rows.append(summarize_window(obs, scenario, metrics))
        all_policy_rows.extend(obs.policy_rows)

    workload_rows = aggregate_workloads(window_rows)
    policy_rows = policy_disagreement_summary(all_policy_rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "PHASE_A_WINDOW_SUMMARY_V1.csv", window_rows)
    write_csv(out_dir / "PHASE_A_WORKLOAD_SUMMARY_V1.csv", workload_rows)
    write_csv(out_dir / "PHASE_A_POLICY_DISAGREEMENT_SUMMARY_V1.csv", policy_rows)
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "workload_summary": workload_rows,
        "window_count": len(window_rows),
        "workload_count": len(workload_rows),
        "augmented_windows_used": False,
        "causal_labeling_executed": False,
        "new_selector_training_executed": False,
        "git": {"head": git(["rev-parse", "HEAD"]), "branch": git(["branch", "--show-current"])},
    }
    (out_dir / "PHASE_A_RESULT_SUMMARY_V1.json").write_text(stable_json(result))
    return result


def source_hashes() -> dict[str, str]:
    paths = {
        "phase_a_script": ROOT / "scripts" / "industry_realism_action_opportunity_phase_a_v1.py",
        "public_trace_replay_v1": ROOT / "src" / "llmserveopt" / "policy_separation" / "public_trace_replay_v1.py",
        "unified_utility_matrix": ROOT / "src" / "llmserveopt" / "policy_separation" / "unified_utility_matrix.py",
        "simulator": ROOT / "src" / "llmserveopt" / "simulator" / "simulator.py",
        "service_model": ROOT / "src" / "llmserveopt" / "simulator" / "service_model.py",
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def workload_input_hashes() -> dict[str, str]:
    paths = {
        "public_trace_manifest": ROOT / "data" / "public_trace_corpus_v1" / "manifest.json",
        "public_trace_source_coverage": ROOT / "data" / "public_trace_corpus_v1" / "source_coverage.csv",
        "azure_2023_code_records": ROOT / "data" / "public_trace_corpus_v1" / "azure_2023_code" / "records.parquet",
        "azure_2023_conv_records": ROOT / "data" / "public_trace_corpus_v1" / "azure_2023_conv" / "records.parquet",
        "burstgpt_records": ROOT / "data" / "public_trace_corpus_v1" / "burstgpt" / "records.parquet",
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def phase_a_records_manifest() -> list[dict[str, Any]]:
    rows = []
    for rec in faithful_records():
        scenario = rec["scenario"]
        arrivals = [float(r.arrival_time) for r in scenario.requests]
        rows.append(
            {
                "canonical_scenario_id": rec["canonical_scenario_id"],
                "source_dataset": rec["source_dataset"],
                "scenario_evidence_class": rec["scenario_evidence_class"],
                "window_index": int(rec["window_index"]),
                "requests": int(len(scenario.requests)),
                "first_arrival_time": float(min(arrivals)) if arrivals else 0.0,
                "last_arrival_time": float(max(arrivals)) if arrivals else 0.0,
                "applicable_policy_note": "Phase A evaluates all six P6 policies on faithful SBS states; public_trace_replay_v1 faithful-view applicable_policies lists only the two policies that read no controlled annotations.",
            }
        )
    return rows


def write_freeze(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    records = phase_a_records_manifest()
    config = {
        "schema_version": f"{SCHEMA_VERSION}.native_config_freeze",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scheduler": SBS_POLICY,
        "p6_policy_portfolio": list(P6_POLICIES),
        "gpu_server_count": 1,
        "gpu_config": {
            "max_active_sequences": ptr.GPU_CONFIG_KWARGS["max_active_sequences"],
            "max_batch_tokens": ptr.GPU_CONFIG_KWARGS["max_batch_tokens"],
            "max_kv_tokens": ptr.GPU_CONFIG_KWARGS["max_kv_tokens"],
        },
        "service_model_kwargs": dict(ptr.SERVICE_MODEL_KWARGS),
        "simulator_config": {"max_steps": None, "drain_steps": 50000},
        "admission_semantics": "SBS trajectory simulated; all P6 policies queried on the same live SBS ObservableState; canonical actions compared structurally.",
        "resource_pressure_policy": "Use existing public_trace_replay_v1 faithful native baseline; do not tune resources to create disagreement in Phase A.",
    }
    semantics = {
        "schema_version": f"{SCHEMA_VERSION}.replay_semantics",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "faithful_view": {
            "evidence_class": ptr.FAITHFUL,
            "request_ordering": "preserved within each selected source window after source-level chronological normalization",
            "interarrival_timing": "preserved within window after rebasing first arrival to t=0",
            "timestamp_spacing": "preserved as relative_arrival_time differences",
            "prompt_input_tokens": "native prompt_tokens, fill missing with 1 and clip lower bound at 1",
            "generated_output_tokens": "native output_tokens, fill missing with 1 and clip lower bound at 1",
            "conversation_session_structure": "not available for Azure 2023; BurstGPT session_id may exist in corpus but PolicySeparationScenario requests do not expose it",
            "original_burstiness": "preserved within selected windows",
            "window_boundaries": "deterministic 200-request windows selected by public_trace_replay_v1.select_window_indices",
            "time_shift": "each window is rebased so first arrival is 0.0",
            "predicted_output_tokens": "set equal to actual_output_tokens; inert for full_prefill/chunked_prefill, visible to ESTF/KV when Phase A evaluates full P6",
            "slo_deadline": "arrival + 1000.0 in faithful view; inert for faithful-view policies, visible to LLF if full P6 is queried",
            "priority": "uniform 1.0",
            "class_id": "default",
            "capacity_assignment": "single GPU with 512 active sequence cap, 512 max batch tokens, 8,000,000 KV tokens",
            "service_rate_assumptions": "step_size=0.001, prefill modeling enabled, prefill_cost_per_token=1.0, step_token_budget=512, decode/prefill contention enabled, decode_first=False",
        },
        "forbidden_phase_a_inputs": ["augmented windows", "load-scaled windows", "synthetic scenarios", "joint240", "Family-A stress cases"],
    }
    prereg = {
        "schema_version": f"{SCHEMA_VERSION}.preregistration",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "PHASE_A_SOURCE_CONFIG_FREEZE_BEFORE_EXECUTION",
        "primary_question": "Under native replay of locally available Tier-1 production traces, how frequently do SBS and the P6 portfolio produce genuinely different canonical scheduling actions?",
        "primary_metric": "disagreement_rate = disagreement_states / total_SBS_decision_states",
        "workloads": list(WORKLOADS),
        "expected_windows_per_workload": EXPECTED_WINDOWS_PER_WORKLOAD,
        "expected_total_windows": EXPECTED_TOTAL_WINDOWS,
        "selected_windows": records,
        "augmented_windows_allowed": False,
        "causal_labeling_allowed": False,
        "new_selector_training_allowed": False,
        "source_hashes": source_hashes(),
        "input_hashes": workload_input_hashes(),
        "git": {"head": git(["rev-parse", "HEAD"]), "branch": git(["branch", "--show-current"])},
        "package_versions": package_versions(),
    }
    (out_dir / "PHASE_A_NATIVE_CONFIG_FREEZE_V1.json").write_text(stable_json(config))
    (out_dir / "PHASE_A_REPLAY_SEMANTICS_V1.json").write_text(stable_json(semantics))
    (out_dir / "PHASE_A_PREREGISTRATION_V1.json").write_text(stable_json(prereg))
    write_csv(out_dir / "PHASE_A_SELECTED_FAITHFUL_WINDOWS_V1.csv", records)


def write_phase_b_plan(out_dir: Path) -> None:
    plan = {
        "schema_version": "industry_realism_action_opportunity_phase_b_pressure_grid_v1.0.0",
        "status": "DESIGNED_NOT_EXECUTED",
        "principle": "Preserve real request identity, ordering, prompt/output lengths, and window membership; vary one system/environment axis at a time.",
        "workloads": list(WORKLOADS),
        "axes": [
            {"axis": "arrival_pressure_multiplier", "grid": [0.5, 0.75, 1.0, 1.25, 1.5, 2.0], "changes_workload": False},
            {"axis": "kv_capacity_fraction_of_native", "grid": [1.0, 0.5, 0.25, 0.125], "changes_workload": False},
            {"axis": "max_active_sequences_fraction_of_native", "grid": [1.0, 0.5, 0.25, 0.125], "changes_workload": False}
        ],
        "forbidden": ["prompt length scaling", "output length scaling", "augmented windows", "synthetic scenarios", "selector training", "causal labeling during support sweep"],
    }
    (out_dir / "PHASE_B_PRESSURE_GRID_V1.json").write_text(stable_json(plan))


def write_readiness_checkpoint(out_dir: Path) -> None:
    checkpoint = {
        "schema_version": "industry_realism_action_opportunity_fgcs_readiness_checkpoint_v1.0.0",
        "status": "POST_PHASE_A_CHECKPOINT_TEMPLATE_UPDATED_AFTER_RESULTS",
        "rubric": {
            "scientific_novelty": {"max": 20, "points": 12, "why": "Clear pivot from selector rescue to action-opportunity mapping, but only Phase A native support evidence so far.", "full_points_requires": "Regime-transition and causal-headroom evidence across production traces."},
            "industry_realism": {"max": 20, "points": 11, "why": "Uses three local Tier-1 public traces natively, but Azure 2024/Bailian/Mooncake not yet frozen locally.", "full_points_requires": "Broader production traces including prefix/cache-heavy and agentic/coding classes."},
            "technical_depth": {"max": 20, "points": 8, "why": "Action-support instrumentation is useful but not yet resource-pressure matrix or causal labeling.", "full_points_requires": "Phase-B pressure sweeps and Phase-D one-step headroom."},
            "experimental_rigor": {"max": 20, "points": 12, "why": "Pre-outcome design/source/config freeze, tests, exact denominators, and zero-result retention.", "full_points_requires": "Completed Phase B/D with preregistered reporting and uncertainty/concentration analysis."},
            "practitioner_value": {"max": 10, "points": 4, "why": "Can identify whether native traces expose scheduler choice, but not yet thresholds for actionability.", "full_points_requires": "Operating-regime map with pressure thresholds and implications."},
            "reproducibility_community_value": {"max": 10, "points": 8, "why": "Local public-trace artifacts, hashes, tests, and compact summaries are reproducible.", "full_points_requires": "Clean public run package for full Phase A/B/D matrix."}
        },
        "total_score": 55,
        "hard_gates": {
            "real_world_evidence": "PARTIAL",
            "systems_regime_characterization": "FAIL_UNTIL_PHASE_B",
            "causal_headroom": "FAIL_UNTIL_PHASE_D",
            "literature_novelty": "PARTIAL",
            "practitioner_value": "PARTIAL",
            "reproducibility": "PASS_FOR_PHASE_A"
        },
        "contribution_strength_confidence_percent": 45,
        "remaining_evidence_to_exceed_90": ["Phase B pressure-transition map", "Phase D causal headroom on frozen regimes", "broader Tier-1 trace coverage or explicit scope limits", "reproducible public package"]
    }
    (out_dir / "FGCS_READINESS_CHECKPOINT_V1.json").write_text(stable_json(checkpoint))


def assert_freeze_files(out_dir: Path) -> None:
    required = [
        out_dir / "PHASE_A_NATIVE_CONFIG_FREEZE_V1.json",
        out_dir / "PHASE_A_REPLAY_SEMANTICS_V1.json",
        out_dir / "PHASE_A_PREREGISTRATION_V1.json",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError({"missing_phase_a_freeze_files": missing})


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--write-freeze", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--write-phase-b-plan", action="store_true")
    parser.add_argument("--write-readiness-checkpoint", action="store_true")
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    if args.write_freeze:
        write_freeze(out_dir)
    if args.execute:
        assert_freeze_files(out_dir)
        run_phase_a(out_dir)
    if args.write_phase_b_plan:
        write_phase_b_plan(out_dir)
    if args.write_readiness_checkpoint:
        write_readiness_checkpoint(out_dir)
    if not any([args.write_freeze, args.execute, args.write_phase_b_plan, args.write_readiness_checkpoint]):
        parser.error("select at least one action")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
