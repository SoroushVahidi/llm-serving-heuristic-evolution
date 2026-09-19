#!/usr/bin/env python
"""Outcome-blind OOD action-support expansion for SBS override labels."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from llmserveopt.core.action import Action
from llmserveopt.core.types import GPUConfig
from llmserveopt.policies.base import BasePolicy
from llmserveopt.policy_separation.schema import PolicySeparationScenario
from llmserveopt.policy_separation.unified_utility_matrix import _build_policy
from llmserveopt.policy_separation import public_trace_replay_v1 as ptr
from llmserveopt.simulator.service_model import ServiceModel
from llmserveopt.simulator.simulator import Simulator, SimulatorConfig

OUT = ROOT / "experiments" / "sbs_override_fresh_ood_support_expansion_v1"
PREV_SCRIPT = ROOT / "scripts" / "sbs_override_fresh_confirmatory_corpus_v1.py"
PREV_OUT = ROOT / "experiments" / "sbs_override_fresh_confirmatory_corpus_v1"
SCHEMA_VERSION = "sbs_override_fresh_ood_support_expansion_v1.0.0"
SBS_POLICY = "kv_constrained_online"
P6: Tuple[str, ...] = (
    "full_prefill",
    "chunked_prefill_small",
    "estimated_service_time_first",
    "weighted_fair_share",
    "least_laxity_first",
    "kv_constrained_online",
)
LOAD_FACTORS = (1.0, 2.0)


def load_prev():
    spec = importlib.util.spec_from_file_location("sbsfresh_prev", PREV_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {PREV_SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, indent=2, separators=(",", ": ")) + "\n"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_info() -> Dict[str, Any]:
    def run(args: Sequence[str]) -> Optional[str]:
        try:
            return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None

    return {"branch": run(["branch", "--show-current"]), "head": run(["rev-parse", "HEAD"])}


def canonical_action(action: Action) -> str:
    return repr(
        {
            "admit": {int(k): sorted(map(int, v)) for k, v in sorted(action.admit.items()) if v},
            "preempt": {int(k): sorted(map(int, v)) for k, v in sorted(action.preempt.items()) if v},
            "swap": {int(k): sorted(map(int, v)) for k, v in sorted(action.swap.items()) if v},
            "migrate": {int(k): sorted((int(a), int(b)) for a, b in v) for k, v in sorted(action.migrate.items()) if v},
            "hold_decode": {int(k): sorted(map(int, v)) for k, v in sorted(action.hold_decode.items()) if v},
            "prefill_chunk_override": {int(k): int(v) for k, v in sorted(action.prefill_chunk_override.items())},
        }
    )


def build_policies() -> Dict[str, BasePolicy]:
    return {pid: _build_policy(pid)[0] for pid in P6}


def public_trace_augmented_scenarios() -> Dict[str, PolicySeparationScenario]:
    out: Dict[str, PolicySeparationScenario] = {}
    for source in ("azure_2023_conv", "azure_2023_code"):
        df = ptr.load_source_records(source)
        for window_index in ptr.select_window_indices(
            len(df), ptr.WINDOW_SIZE, ptr.WINDOWS_PER_SOURCE
        ):
            window = ptr.extract_window(df, window_index, ptr.WINDOW_SIZE)
            scenario, _field_provenance = ptr.build_scenario_from_window(
                window,
                source=source,
                window_index=window_index,
                evidence_class=ptr.AUGMENTED,
            )
            out[ptr.canonical_scenario_id(source, window_index, ptr.AUGMENTED)] = scenario
    return out


def transform_arrivals(scenario: PolicySeparationScenario, lam: float, scenario_id: str) -> PolicySeparationScenario:
    reqs = list(scenario.requests)
    t0 = min(float(r.arrival_time) for r in reqs)
    new_reqs = []
    for r in reqs:
        new_t = t0 + (float(r.arrival_time) - t0) / float(lam)
        slack = float(r.slo_deadline) - float(r.arrival_time)
        new_reqs.append(replace(r, arrival_time=new_t, slo_deadline=new_t + slack))
    params = dict(scenario.params)
    params.update({"base_scenario_id": scenario.scenario_id, "load_factor": float(lam)})
    return PolicySeparationScenario(
        scenario_id=scenario_id,
        family=f"{scenario.family}_SBS_OOD_EXPANSION",
        template_name="public_trace_replay_v1_augmented_arrival_scaled",
        generator_version=SCHEMA_VERSION,
        seed=int(scenario.seed),
        params=params,
        requests=tuple(new_reqs),
        gpu_configs=scenario.gpu_configs,
        service_model_kwargs=dict(scenario.service_model_kwargs),
        target_policy_family="SBS_OVERRIDE_FRESH_OOD_SUPPORT_EXPANSION_V1",
        expected_qualitative_hypothesis="outcome-blind OOD SBS-vs-P6 action support",
    )


def prepare() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prev = load_prev()
    rows: List[Dict[str, Any]] = []

    # BurstGPT from the already frozen Stage-0/LSSP cache: all 40 windows x two load factors.
    with open(prev.LSSP_WINDOWS_CACHE) as f:
        cache = json.load(f)
    for w in cache["windows"]:
        if str(w["source_family"]) != "burstgpt":
            continue
        for lam in LOAD_FACTORS:
            rows.append(
                {
                    "scenario_id": f"sbsfresh_oodexp_burstgpt_{w['window_id']}_lambda{lam:g}",
                    "source": "burstgpt_v2_lssp_cache",
                    "source_group": "burstgpt_v2",
                    "builder": "lssp_stage0_window_cache_controlled_annotation_v1",
                    "window_id": w["window_id"],
                    "base_scenario_id": "",
                    "source_file": w.get("source_file", ""),
                    "source_file_sha256": w.get("source_file_sha256", ""),
                    "load_factor": lam,
                    "selection_basis": "all staged BurstGPT windows x predeclared load factors",
                }
            )

    # Azure 2023 public trace corpus: augmented-view windows already staged in this repo.
    for sid, scen in sorted(public_trace_augmented_scenarios().items()):
        src = str(scen.params.get("source", ""))
        if src not in {"azure_2023_conv", "azure_2023_code"}:
            continue
        for lam in LOAD_FACTORS:
            rows.append(
                {
                    "scenario_id": f"sbsfresh_oodexp_{sid.replace('::', '_')}_lambda{lam:g}",
                    "source": src,
                    "source_group": "azure_2023_public_trace",
                    "builder": "public_trace_replay_v1_augmented_arrival_scaled",
                    "window_id": scen.params.get("window_index", ""),
                    "base_scenario_id": sid,
                    "source_file": str(ROOT / "data" / "public_trace_corpus_v1" / src / "records.parquet"),
                    "source_file_sha256": sha256_file(ROOT / "data" / "public_trace_corpus_v1" / src / "records.parquet"),
                    "load_factor": lam,
                    "selection_basis": "all staged Azure2023 augmented windows x predeclared load factors",
                }
            )

    pd.DataFrame(rows).to_csv(OUT / "fresh_ood_expansion_candidate_manifest.csv", index=False)
    inventory = [
        {
            "source": "TraceLab",
            "classification": "RELATED_BUT_NOT_EQUIVALENT",
            "path": "not locally staged as raw replay data; prior HF-derived 512-window sweep documented elsewhere",
            "prior_evidence": "near-saturated policy separation/oracle goodput around 1.0 in derived sweep; not equivalent SBS support",
            "replay_compatibility": "requires fresh adapter/re-derivation from raw TraceLab",
        },
        {
            "source": "SwissAI/OpenTela",
            "classification": "NOT_PREVIOUSLY_SCANNED",
            "path": "not found in local/Wulver staged paths during targeted search",
            "prior_evidence": "none found",
            "replay_compatibility": "unknown",
        },
        {
            "source": "Mooncake/Kimi",
            "classification": "UNUSABLE_FOR_THIS_REPLAY",
            "path": "loaders/fixtures exist; trace license/data-provenance caveat",
            "prior_evidence": "related only",
            "replay_compatibility": "internal-only until data license/schema verified",
        },
        {
            "source": "BurstGPT v2",
            "classification": "RELATED_BUT_NOT_EQUIVALENT",
            "path": str(prev.LSSP_WINDOWS_CACHE),
            "prior_evidence": "public replay/load-scaling policy outcomes exist; no equivalent SBS trajectory support manifest found",
            "replay_compatibility": "compatible via existing LSSP controlled annotations",
        },
        {
            "source": "Azure 2023",
            "classification": "RELATED_BUT_NOT_EQUIVALENT",
            "path": str(ROOT / "data" / "public_trace_corpus_v1"),
            "prior_evidence": "public trace replay trajectories exist; no equivalent SBS trajectory support manifest found",
            "replay_compatibility": "compatible via public_trace_replay_v1 augmented view",
        },
    ]
    pd.DataFrame(inventory).to_csv(OUT / "available_ood_source_inventory.csv", index=False)
    prereg = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "confirmatory_blindness": {
            "permitted": [
                "source metadata",
                "arrivals and online request attributes",
                "SBS trajectories",
                "P6 native actions and canonical action equality",
                "state/choice structural diagnostics",
            ],
            "forbidden": ["terminal counterfactual labels", "Q_SBS", "A_SBS", "selector training/tuning", "closed loop"],
        },
        "choice_state_rate_definition": "fraction of SBS decision states with waiting_queue_count >= 2",
        "binding_capacity_definition": "waiting_queue_count > total_free_sequence_slots",
        "manifest": str(OUT / "fresh_ood_expansion_candidate_manifest.csv"),
        "candidate_count": len(rows),
        "source_counts": pd.DataFrame(rows)["source"].value_counts().to_dict(),
        "git": git_info(),
    }
    (OUT / "preregistration_boundary.json").write_text(stable_json(prereg))


def build_scenario(row: Mapping[str, Any]) -> PolicySeparationScenario:
    prev = load_prev()
    if "base_scenario_id" not in row or pd.isna(row.get("base_scenario_id", None)):
        return prev.build_scenario(row)
    if row["builder"] == "lssp_stage0_window_cache_controlled_annotation_v1":
        row2 = {
            "scenario_id": row["scenario_id"],
            "layer": "EXTERNAL_SOURCE_OOD_EXPANSION",
            "source": "burstgpt",
            "window_id": row["window_id"],
            "load_factor": row["load_factor"],
        }
        return prev.build_external_scenario(row2)
    base = public_trace_augmented_scenarios()[str(row["base_scenario_id"])]
    return transform_arrivals(base, float(row["load_factor"]), str(row["scenario_id"]))


def state_features(state) -> Dict[str, Any]:
    waiting = list(state.waiting_queue)
    gpus = list(state.gpu_states)
    active_infos = [r for g in gpus for r in getattr(g, "active_requests_info", [])]
    total_free = sum(max(0, int(g.free_sequences)) for g in gpus)
    free_kv = sum(max(0, int(g.free_kv_tokens)) for g in gpus)
    kv_util = [
        (float(g.current_kv_tokens) / float(g.max_kv_tokens)) if getattr(g, "max_kv_tokens", 0) else 0.0
        for g in gpus
    ]
    slack = [float(r.slo_deadline) - float(state.time) for r in waiting]
    priorities = [float(r.priority) for r in waiting]
    classes = {str(r.class_id) for r in waiting}
    waiting_pred = sum(int(r.predicted_output_tokens) for r in waiting)
    waiting_prompt = sum(int(r.prompt_tokens) for r in waiting)
    proxy_tops = set()
    if waiting:
        proxy_tops.add(min(waiting, key=lambda r: (r.arrival_time, r.request_id)).request_id)
        proxy_tops.add(min(waiting, key=lambda r: (r.predicted_output_tokens + r.prompt_tokens, r.request_id)).request_id)
        proxy_tops.add(min(waiting, key=lambda r: (r.slo_deadline - state.time, r.request_id)).request_id)
        proxy_tops.add(max(waiting, key=lambda r: (r.priority, -r.request_id)).request_id)
    return {
        "waiting_queue_count": int(len(waiting)),
        "active_sequences": int(sum(len(g.active_request_ids) for g in gpus)),
        "prompt_token_mass_waiting": int(waiting_prompt),
        "predicted_remaining_token_mass_waiting": int(waiting_pred),
        "active_predicted_token_mass": int(sum(int(r.predicted_output_tokens) for r in active_infos)),
        "kv_utilization_mean": float(np.mean(kv_util)) if kv_util else 0.0,
        "kv_utilization_max": float(np.max(kv_util)) if kv_util else 0.0,
        "prefilling_count": int(sum(int(getattr(g, "prefilling_count", 0)) for g in gpus)),
        "decoding_count": int(sum(int(getattr(g, "decoding_count", 0)) for g in gpus)),
        "slack_min": float(min(slack)) if slack else None,
        "slack_p50": float(np.median(slack)) if slack else None,
        "priority_unique_count": int(len(set(priorities))),
        "priority_max": float(max(priorities)) if priorities else None,
        "class_unique_count": int(len(classes)),
        "eligible_request_count": int(len(waiting)),
        "total_free_sequence_slots": int(total_free),
        "choice_state": int(len(waiting) >= 2),
        "binding_capacity_choice": int(len(waiting) > total_free),
        "binding_kv_choice": int(waiting_prompt > free_kv),
        "policy_ranking_proxy_difference": int(len(proxy_tops) > 1),
    }


class ScanPolicy(BasePolicy):
    name = "sbs_override_fresh_ood_support_expansion_v1"

    def __init__(self, scenario_id: str) -> None:
        self.scenario_id = scenario_id
        self.shadow = build_policies()
        self.state_rows: List[Dict[str, Any]] = []
        self.policy_rows: List[Dict[str, Any]] = []

    def reset(self) -> None:
        for p in self.shadow.values():
            if hasattr(p, "reset"):
                p.reset()
        self.state_rows = []
        self.policy_rows = []

    def select_action(self, state) -> Action:
        actions = {pid: self.shadow[pid].select_action(copy.deepcopy(state)) for pid in P6}
        c = {pid: canonical_action(a) for pid, a in actions.items()}
        differing = [pid for pid in P6 if pid != SBS_POLICY and c[pid] != c[SBS_POLICY]]
        state_id = f"sbsfresh_oodexp::{self.scenario_id}::{int(state.step)}"
        row = {
            "schema_version": SCHEMA_VERSION,
            "scenario_id": self.scenario_id,
            "state_id": state_id,
            "step": int(state.step),
            "sim_time": float(state.time),
            "any_non_sbs_policy_differs": int(bool(differing)),
            "n_non_sbs_policies_differ": int(len(differing)),
            "n_distinct_canonical_p6_actions_full": int(len(set(c.values()))),
            "differing_policy_ids": ",".join(differing),
        }
        row.update(state_features(state))
        self.state_rows.append(row)
        for pid in P6:
            if pid == SBS_POLICY:
                continue
            self.policy_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "scenario_id": self.scenario_id,
                    "state_id": state_id,
                    "step": int(state.step),
                    "candidate_policy_id": pid,
                    "candidate_differs_from_sbs": int(c[pid] != c[SBS_POLICY]),
                    "candidate_canonical_action_full": c[pid],
                    "sbs_canonical_action_full": c[SBS_POLICY],
                }
            )
        return actions[SBS_POLICY]


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r.keys()}))
        writer.writeheader()
        writer.writerows(rows)


def run_rows(rows: pd.DataFrame, out: Path, shard_index: int, num_shards: int) -> None:
    done = out / "DONE" / f"shard{shard_index:04d}-of-{num_shards:04d}.done"
    failed = out / "FAILED" / f"shard{shard_index:04d}-of-{num_shards:04d}.failed"
    if done.exists():
        return
    state_rows: List[Dict[str, Any]] = []
    policy_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    try:
        for _, row in rows.iterrows():
            rd = row.to_dict()
            scenario = build_scenario(rd)
            obs = ScanPolicy(scenario.scenario_id)
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
            for r in obs.state_rows:
                r.update({"source": rd["source"], "source_group": rd.get("source_group", rd["source"]), "load_factor": rd["load_factor"]})
            for r in obs.policy_rows:
                r.update({"source": rd["source"], "source_group": rd.get("source_group", rd["source"]), "load_factor": rd["load_factor"]})
            state_rows.extend(obs.state_rows)
            policy_rows.extend(obs.policy_rows)
            n_states = len(obs.state_rows)
            n_dis = sum(int(r["any_non_sbs_policy_differs"]) for r in obs.state_rows)
            summary_rows.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "source": rd["source"],
                    "source_group": rd.get("source_group", rd["source"]),
                    "load_factor": rd["load_factor"],
                    "decision_states": n_states,
                    "disagreement_states": n_dis,
                    "disagreement_rate": float(n_dis / n_states) if n_states else 0.0,
                    "choice_states": int(sum(int(r["choice_state"]) for r in obs.state_rows)),
                    "binding_capacity_choice_states": int(sum(int(r["binding_capacity_choice"]) for r in obs.state_rows)),
                    "policy_ranking_proxy_difference_states": int(sum(int(r["policy_ranking_proxy_difference"]) for r in obs.state_rows)),
                    "sbs_anwg": float(metrics.arrival_normalized_weighted_goodput),
                }
            )
        write_csv(out / "state_rows" / f"state_rows.shard{shard_index:04d}-of-{num_shards:04d}.csv", state_rows)
        write_csv(out / "policy_rows" / f"policy_rows.shard{shard_index:04d}-of-{num_shards:04d}.csv", policy_rows)
        write_csv(out / "scenario_summary" / f"scenario_summary.shard{shard_index:04d}-of-{num_shards:04d}.csv", summary_rows)
        done.parent.mkdir(parents=True, exist_ok=True)
        done.write_text(stable_json({"done": True, "n_scenarios": len(rows)}))
    except Exception as exc:
        failed.parent.mkdir(parents=True, exist_ok=True)
        failed.write_text(stable_json({"failed": True, "error": repr(exc)}))
        raise


def scan_shard(shard_index: int, num_shards: int) -> None:
    df = pd.read_csv(OUT / "fresh_ood_expansion_candidate_manifest.csv")
    rows = df.iloc[[i for i in range(len(df)) if i % num_shards == shard_index]].reset_index(drop=True)
    run_rows(rows, OUT / "support_scan", shard_index, num_shards)


def diagnose_prior_shard(shard_index: int, num_shards: int) -> None:
    prev_manifest = pd.read_csv(PREV_OUT / "fresh_candidate_scenario_manifest.csv")
    keep = prev_manifest[prev_manifest["source"].isin(["azure_llm_2024", "bailian_qwen", "fresh_joint_multimechanism_generator_holdout"])].reset_index(drop=True)
    rows = keep.iloc[[i for i in range(len(keep)) if i % num_shards == shard_index]].reset_index(drop=True)
    run_rows(rows, OUT / "zero_support_diagnosis", shard_index, num_shards)


def episode_lengths(flags: Sequence[int]) -> List[int]:
    out: List[int] = []
    cur = 0
    for f in flags:
        if int(f):
            cur += 1
        elif cur:
            out.append(cur)
            cur = 0
    if cur:
        out.append(cur)
    return out


def aggregate_dir(scan_dir: Path, prefix: str) -> Dict[str, Any]:
    scen = pd.concat([pd.read_csv(p) for p in sorted((scan_dir / "scenario_summary").glob("*.csv"))], ignore_index=True)
    state_parts = sorted((scan_dir / "state_rows").glob("*.csv"))
    policy_parts = sorted((scan_dir / "policy_rows").glob("*.csv"))
    source_rows = []
    policy_rows = []
    episode_rows = []
    branch_keys: Dict[str, set[Tuple[str, str]]] = {}
    state_manifest = OUT / f"{prefix}_disagreement_state_manifest.csv"
    policy_manifest = OUT / f"{prefix}_state_policy_action_map.csv"
    for p in (state_manifest, policy_manifest):
        if p.exists():
            p.unlink()
    wrote_state = wrote_policy = False
    for path in state_parts:
        for chunk in pd.read_csv(path, chunksize=50_000):
            for source, g in chunk.groupby("source"):
                source_rows.append(
                    {
                        "source": source,
                        "decision_states": int(len(g)),
                        "disagreement_states": int(g["any_non_sbs_policy_differs"].astype(int).sum()),
                        "choice_states": int(g["choice_state"].astype(int).sum()),
                        "binding_capacity_choice_states": int(g["binding_capacity_choice"].astype(int).sum()),
                        "policy_ranking_proxy_difference_states": int(g["policy_ranking_proxy_difference"].astype(int).sum()),
                        "waiting_queue_count_mean": float(g["waiting_queue_count"].mean()),
                        "waiting_queue_count_p90": float(g["waiting_queue_count"].quantile(0.9)),
                        "active_sequences_mean": float(g["active_sequences"].mean()),
                        "kv_utilization_mean": float(g["kv_utilization_mean"].mean()),
                        "prompt_token_mass_waiting_mean": float(g["prompt_token_mass_waiting"].mean()),
                        "predicted_remaining_token_mass_waiting_mean": float(g["predicted_remaining_token_mass_waiting"].mean()),
                        "priority_unique_count_mean": float(g["priority_unique_count"].mean()),
                        "class_unique_count_mean": float(g["class_unique_count"].mean()),
                    }
                )
            for sid, g in chunk.sort_values(["scenario_id", "step"]).groupby("scenario_id"):
                eps = episode_lengths(g["any_non_sbs_policy_differs"].astype(int).tolist())
                if eps:
                    episode_rows.append({"scenario_id": sid, "source": g["source"].iloc[0], "episodes": len(eps), "max_episode_len": max(eps), "median_episode_len": float(np.median(eps))})
            dis = chunk[chunk["any_non_sbs_policy_differs"].astype(int) == 1]
            if len(dis):
                dis.to_csv(state_manifest, mode="a", index=False, header=not wrote_state)
                wrote_state = True
    if not wrote_state:
        state_manifest.write_text("")
    for path in policy_parts:
        for chunk in pd.read_csv(path, chunksize=50_000):
            dis = chunk[chunk["candidate_differs_from_sbs"].astype(int) == 1]
            if len(dis):
                for source, g in dis.groupby("source"):
                    branch_keys.setdefault(source, set()).update(zip(g["state_id"].astype(str), g["candidate_canonical_action_full"].astype(str)))
                    policy_rows.extend({"source": source, "candidate_policy_id": pid, "count": int(n)} for pid, n in g["candidate_policy_id"].value_counts().items())
                dis.to_csv(policy_manifest, mode="a", index=False, header=not wrote_policy)
                wrote_policy = True
    if not wrote_policy:
        policy_manifest.write_text("")
    source_df = pd.DataFrame(source_rows).groupby("source", as_index=False).sum(numeric_only=True)
    scen_source = scen.groupby("source").agg(scenarios=("scenario_id", "nunique"), scenarios_with_any=("disagreement_states", lambda s: int((s > 0).sum())), median_disagreements=("disagreement_states", "median"), p90_disagreements=("disagreement_states", lambda s: float(s.quantile(0.9))), max_disagreements=("disagreement_states", "max")).reset_index()
    out = source_df.merge(scen_source, on="source", how="outer")
    out["disagreement_rate"] = out["disagreement_states"] / out["decision_states"].clip(lower=1)
    out["choice_state_rate"] = out["choice_states"] / out["decision_states"].clip(lower=1)
    out["binding_capacity_choice_rate"] = out["binding_capacity_choice_states"] / out["decision_states"].clip(lower=1)
    out["policy_ranking_proxy_difference_rate"] = out["policy_ranking_proxy_difference_states"] / out["decision_states"].clip(lower=1)
    out["unique_action_branch_count"] = [len(branch_keys.get(src, set())) for src in out["source"]]
    out.to_csv(OUT / f"{prefix}_support_by_source.csv", index=False)
    pol = pd.DataFrame(policy_rows)
    if len(pol):
        pol.groupby(["source", "candidate_policy_id"], as_index=False)["count"].sum().to_csv(OUT / f"{prefix}_per_policy_disagreement.csv", index=False)
    else:
        (OUT / f"{prefix}_per_policy_disagreement.csv").write_text("")
    pd.DataFrame(episode_rows).to_csv(OUT / f"{prefix}_disagreement_episode_structure.csv", index=False)
    summary = {"support_by_source": out.to_dict(orient="records")}
    (OUT / f"{prefix}_summary.json").write_text(stable_json(summary))
    return summary


def aggregate() -> None:
    aggregate_dir(OUT / "support_scan", "ood_expansion")
    aggregate_dir(OUT / "zero_support_diagnosis", "zero_support_diagnosis")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    scan = sub.add_parser("scan-shard")
    scan.add_argument("--shard-index", type=int, required=True)
    scan.add_argument("--num-shards", type=int, required=True)
    diag = sub.add_parser("diagnose-prior-shard")
    diag.add_argument("--shard-index", type=int, required=True)
    diag.add_argument("--num-shards", type=int, required=True)
    sub.add_parser("aggregate")
    args = ap.parse_args(argv)
    if args.cmd == "prepare":
        prepare()
    elif args.cmd == "scan-shard":
        scan_shard(args.shard_index, args.num_shards)
    elif args.cmd == "diagnose-prior-shard":
        diagnose_prior_shard(args.shard_index, args.num_shards)
    elif args.cmd == "aggregate":
        aggregate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
