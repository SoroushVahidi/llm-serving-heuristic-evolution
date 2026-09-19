#!/usr/bin/env python
"""Fresh SBS override confirmatory corpus v1.

This script freezes an outcome-blind fresh scenario manifest and performs the
cheap SBS action-support scan. It intentionally does not run terminal
branches, fit selectors, or inspect confirmatory benefits.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import os
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from llmserveopt.core.action import Action
from llmserveopt.core.types import GPUConfig
from llmserveopt.policies.base import BasePolicy
from llmserveopt.policies.scoring import DEFAULT_ALPHA, DEFAULT_BETA
from llmserveopt.policy_separation.builders import req
from llmserveopt.policy_separation.schema import PolicySeparationScenario
from llmserveopt.policy_separation.templates_fairness_starvation_v2 import apply_prediction_noise
from llmserveopt.policy_separation.unified_utility_matrix import _build_policy
from llmserveopt.simulator.service_model import ServiceModel
from llmserveopt.simulator.simulator import Simulator, SimulatorConfig

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "sbs_override_fresh_confirmatory_corpus_v1"
MANIFEST = OUT / "fresh_candidate_scenario_manifest.csv"
LSSP_WINDOWS_CACHE = Path(
    os.environ.get(
        "SBS_FRESH_LSSP_WINDOWS_CACHE",
        "/home/soroush/repos/llm-serving-scheduler-lssp-phase12-freeze/"
        "artifacts/pilot_v2_windows_full_cache.json",
    )
)
JOINT_RUNNER = ROOT / "experiments/joint_multimechanism_generalization_v1/run_joint_multimechanism_generalization_v1.py"

SCHEMA_VERSION = "sbs_override_fresh_confirmatory_corpus_v1.0.0"
SUPPORT_SCAN_VERSION = "sbs_override_fresh_support_scan_v1.0.0"
FRESH_ID_SEED = 20260919
FRESH_ID_START_INDEX = 10_000
N_FRESH_ID_SCENARIOS = 80
EXTERNAL_SOURCES = ("azure_llm_2024", "bailian_qwen")
EXTERNAL_LOAD_FACTORS = (1.0, 2.0)
PREDICTION_NOISE_SIGMA = 0.30
SLACK_MULTIPLIER = 1.0
STEP_SIZE = 0.001
SBS_POLICY = "kv_constrained_online"
P6: Tuple[str, ...] = (
    "full_prefill",
    "chunked_prefill_small",
    "estimated_service_time_first",
    "weighted_fair_share",
    "least_laxity_first",
    "kv_constrained_online",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, indent=2, separators=(",", ": ")) + "\n"


def git_info() -> Dict[str, Any]:
    def run(args: Sequence[str]) -> Optional[str]:
        try:
            return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None

    return {
        "branch": run(["branch", "--show-current"]),
        "head": run(["rev-parse", "HEAD"]),
        "status_short": run(["status", "--short"]),
    }


def _load_joint_runner():
    spec = importlib.util.spec_from_file_location("joint_mm_runner_v1_fresh", JOINT_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load joint runner from {JOINT_RUNNER}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def source_inventory() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if LSSP_WINDOWS_CACHE.exists():
        with open(LSSP_WINDOWS_CACHE) as f:
            cache = json.load(f)
        by_source: Dict[str, List[Dict[str, Any]]] = {}
        for w in cache["windows"]:
            by_source.setdefault(str(w["source_family"]), []).append(w)
        for source, windows in sorted(by_source.items()):
            files = sorted({str(w.get("source_file", "")) for w in windows if w.get("source_file")})
            rows.append(
                {
                    "source": source,
                    "physically_available_local": True,
                    "physically_available_wulver": "expected from Stage0 staging; verify before Slurm",
                    "exact_path": str(LSSP_WINDOWS_CACHE),
                    "size_bytes": LSSP_WINDOWS_CACHE.stat().st_size,
                    "loader_exists": True,
                    "previous_policy_separation_run_exists": "related Stage0/ranking-portability runs only",
                    "previously_used_in_joint240_development": False,
                    "contamination_risk": "LOW for terminal labels; workload metadata only inspected",
                    "replay_compatibility": "compatible via controlled-annotation PolicySeparationScenario builder",
                    "license_provenance_concerns": "see source_license/source_url per record",
                    "approx_usable_fresh_windows": len(windows),
                    "source_files": ";".join(files[:4]),
                }
            )
    rows.append(
        {
            "source": "fresh_joint_multimechanism_generator_holdout",
            "physically_available_local": JOINT_RUNNER.exists(),
            "physically_available_wulver": "available after repo sync",
            "exact_path": str(JOINT_RUNNER),
            "size_bytes": JOINT_RUNNER.stat().st_size if JOINT_RUNNER.exists() else 0,
            "loader_exists": JOINT_RUNNER.exists(),
            "previous_policy_separation_run_exists": "joint240 development used same generator family, different IDs/seeds",
            "previously_used_in_joint240_development": False,
            "contamination_risk": "LOW if indices/seeds remain outside 0..239",
            "replay_compatibility": "native PolicySeparationScenario builder",
            "license_provenance_concerns": "synthetic internal generator",
            "approx_usable_fresh_windows": N_FRESH_ID_SCENARIOS,
            "source_files": str(JOINT_RUNNER),
        }
    )
    return rows


def prepare_manifest() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []

    for i in range(N_FRESH_ID_SCENARIOS):
        idx = FRESH_ID_START_INDEX + i
        rows.append(
            {
                "scenario_id": f"sbsfresh_id_joint_mm_{idx:05d}",
                "layer": "FRESH_IN_DISTRIBUTION_GENERATOR_HOLDOUT",
                "source": "fresh_joint_multimechanism_generator_holdout",
                "builder": "joint_multimechanism_generator_v1.0.0",
                "generator_index": idx,
                "fresh_seed": FRESH_ID_SEED,
                "window_id": "",
                "source_file": str(JOINT_RUNNER),
                "source_file_sha256": sha256_file(JOINT_RUNNER) if JOINT_RUNNER.exists() else "",
                "load_factor": 1.0,
                "label_status": "UNLABELED_TERMINAL_OUTCOMES_NOT_INSPECTED",
                "selection_basis": "predeclared fresh index range, outcome-blind",
            }
        )

    with open(LSSP_WINDOWS_CACHE) as f:
        cache = json.load(f)
    for w in cache["windows"]:
        source = str(w["source_family"])
        if source not in EXTERNAL_SOURCES:
            continue
        for lam in EXTERNAL_LOAD_FACTORS:
            rows.append(
                {
                    "scenario_id": f"sbsfresh_ood_{source}_{w['window_id']}_lambda{lam:g}",
                    "layer": "EXTERNAL_SOURCE_OOD",
                    "source": source,
                    "builder": "lssp_stage0_window_cache_controlled_annotation_v1",
                    "generator_index": "",
                    "fresh_seed": FRESH_ID_SEED,
                    "window_id": w["window_id"],
                    "source_file": w.get("source_file", ""),
                    "source_file_sha256": w.get("source_file_sha256", ""),
                    "load_factor": lam,
                    "label_status": "UNLABELED_TERMINAL_OUTCOMES_NOT_INSPECTED",
                    "selection_basis": "all staged Azure2024/Bailian windows x predeclared load factors",
                }
            )

    pd.DataFrame(rows).to_csv(MANIFEST, index=False)
    pd.DataFrame(source_inventory()).to_csv(OUT / "available_fresh_dataset_inventory.csv", index=False)
    prereg = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "development_data": {
            "description": "entire existing joint240 targeted corpus and all OOF results",
            "states": 8888,
            "scenarios": 236,
            "non_sbs_unique_actions": 21858,
            "status": "DEVELOPMENT_DATA_ONLY",
        },
        "confirmatory_data": {
            "scenario_count": len(rows),
            "source_composition": pd.DataFrame(rows)["source"].value_counts().to_dict(),
            "terminal_labels_status": "not generated and not inspected",
            "selector_freeze_rule": "no fresh terminal labels may be used for design/model/threshold choices",
        },
        "support_scan": {
            "policy": SBS_POLICY,
            "shadow_policy_portfolio": list(P6),
            "allowed_before_terminal_labels": True,
            "selection_rule_after_scan": "retain/downweight sources by action-support only, never by terminal benefit",
        },
        "inputs": {
            "manifest": str(MANIFEST),
            "lssp_windows_cache": str(LSSP_WINDOWS_CACHE),
            "lssp_windows_cache_sha256": sha256_file(LSSP_WINDOWS_CACHE),
            "joint_runner": str(JOINT_RUNNER),
            "joint_runner_sha256": sha256_file(JOINT_RUNNER) if JOINT_RUNNER.exists() else None,
        },
        "git": git_info(),
    }
    (OUT / "preregistration_boundary.json").write_text(stable_json(prereg))
    duplication = {
        "classification": "RELATED_BUT_NOT_EQUIVALENT",
        "summary": (
            "Found related public trace/stage0/prospective workload artifacts and "
            "joint240 SBS terminal-label outputs, but no sufficient fresh untouched "
            "SBS-relative targeted terminal-label corpus for this confirmatory stage."
        ),
    }
    (OUT / "duplication_gate_summary.json").write_text(stable_json(duplication))


def _scaled_arrivals(records: Sequence[Mapping[str, Any]], load_factor: float) -> np.ndarray:
    arrivals = np.asarray([float(r["arrival_time_s"]) for r in records], dtype=float)
    order = np.argsort(arrivals, kind="mergesort")
    arrivals = arrivals[order]
    start = float(arrivals[0])
    return order, start + (arrivals - start) / float(load_factor)


def build_external_scenario(row: Mapping[str, Any]) -> PolicySeparationScenario:
    with open(LSSP_WINDOWS_CACHE) as f:
        cache = json.load(f)
    window_by_id = {str(w["window_id"]): w for w in cache["windows"]}
    w = window_by_id[str(row["window_id"])]
    records = list(w["records"])
    order, arrivals = _scaled_arrivals(records, float(row["load_factor"]))
    rng = np.random.default_rng(
        int(hashlib.sha256(str(row["scenario_id"]).encode()).hexdigest()[:16], 16) % (2**32)
    )
    reqs = []
    for new_i, old_i in enumerate(order.tolist()):
        r = records[old_i]
        prompt = int(max(1, r.get("input_tokens") or r.get("prompt_tokens") or 1))
        actual = int(max(1, r.get("output_tokens") or 1))
        pred = int(max(1, round(float(apply_prediction_noise(rng, np.asarray([actual]), PREDICTION_NOISE_SIGMA)[0]))))
        service_est = DEFAULT_ALPHA * prompt + DEFAULT_BETA * pred
        reqs.append(
            req(
                request_id=new_i,
                arrival_time=float(arrivals[new_i]),
                prompt_tokens=prompt,
                predicted_output_tokens=pred,
                actual_output_tokens=actual,
                slo_deadline=float(arrivals[new_i] + service_est * (1.0 + SLACK_MULTIPLIER)),
                priority=1.0,
                class_id=str(row["source"]),
            )
        )
    return PolicySeparationScenario(
        scenario_id=str(row["scenario_id"]),
        family="SBS_OVERRIDE_FRESH_CONFIRMATORY_CORPUS_V1_OOD",
        template_name="lssp_stage0_window_cache_controlled_annotation_v1",
        generator_version=SCHEMA_VERSION,
        seed=int(FRESH_ID_SEED),
        params={
            "source": row["source"],
            "window_id": row["window_id"],
            "load_factor": float(row["load_factor"]),
            "prediction_noise_sigma": PREDICTION_NOISE_SIGMA,
            "slack_multiplier": SLACK_MULTIPLIER,
            "field_provenance": "arrival/input/output native; prediction/deadline/priority/class controlled annotations",
        },
        requests=tuple(reqs),
        gpu_configs=(GPUConfig(gpu_id=0, max_active_sequences=512, max_batch_tokens=512, max_kv_tokens=8_000_000),),
        service_model_kwargs={
            "step_size": STEP_SIZE,
            "enable_prefill_modeling": True,
            "prefill_cost_per_token": 1.0,
            "step_token_budget": 512,
            "enable_decode_prefill_contention": True,
            "decode_first": False,
        },
        target_policy_family="SBS_OVERRIDE_FRESH_CONFIRMATORY_CORPUS_V1",
        expected_qualitative_hypothesis="fresh external workload action support for SBS-relative override labeling",
    )


def build_id_scenario(row: Mapping[str, Any]) -> PolicySeparationScenario:
    mod = _load_joint_runner()
    rng = np.random.default_rng(int(row["fresh_seed"]))
    params = None
    target_idx = int(row["generator_index"])
    for j in range(target_idx - FRESH_ID_START_INDEX + 1):
        params = mod.sample_params(rng, target_idx if j == target_idx - FRESH_ID_START_INDEX else FRESH_ID_START_INDEX + j)
    assert params is not None
    params["joint_id"] = str(row["scenario_id"])
    return mod.build_scenario(params)


def build_scenario(row: Mapping[str, Any]) -> PolicySeparationScenario:
    if row["layer"] == "FRESH_IN_DISTRIBUTION_GENERATOR_HOLDOUT":
        return build_id_scenario(row)
    return build_external_scenario(row)


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


class SupportScanPolicy(BasePolicy):
    name = "sbs_override_fresh_support_scan_v1"

    def __init__(self, scenario_id: str) -> None:
        self.scenario_id = scenario_id
        self.shadow = build_policies()
        self.rows: List[Dict[str, Any]] = []
        self.policy_rows: List[Dict[str, Any]] = []

    def reset(self) -> None:
        for p in self.shadow.values():
            if hasattr(p, "reset"):
                p.reset()
        self.rows = []
        self.policy_rows = []

    def select_action(self, state) -> Action:
        actions = {pid: self.shadow[pid].select_action(copy.deepcopy(state)) for pid in P6}
        sbs = actions[SBS_POLICY]
        c = {pid: canonical_action(a) for pid, a in actions.items()}
        differing = [pid for pid in P6 if pid != SBS_POLICY and c[pid] != c[SBS_POLICY]]
        state_id = f"sbsfresh::{self.scenario_id}::{int(state.step)}"
        self.rows.append(
            {
                "schema_version": SUPPORT_SCAN_VERSION,
                "scenario_id": self.scenario_id,
                "state_id": state_id,
                "step": int(state.step),
                "sim_time": float(state.time),
                "waiting_queue": int(len(state.waiting_queue)),
                "active_sequences": int(sum(len(g.active_request_ids) for g in state.gpu_states)),
                "sbs_canonical_action_full": c[SBS_POLICY],
                "n_distinct_canonical_p6_actions_full": int(len(set(c.values()))),
                "any_non_sbs_policy_differs": int(bool(differing)),
                "n_non_sbs_policies_differ": int(len(differing)),
                "differing_policy_ids": ",".join(differing),
            }
        )
        for pid in P6:
            if pid == SBS_POLICY:
                continue
            self.policy_rows.append(
                {
                    "schema_version": SUPPORT_SCAN_VERSION,
                    "scenario_id": self.scenario_id,
                    "state_id": state_id,
                    "step": int(state.step),
                    "candidate_policy_id": pid,
                    "candidate_canonical_action_full": c[pid],
                    "sbs_canonical_action_full": c[SBS_POLICY],
                    "candidate_differs_from_sbs": int(c[pid] != c[SBS_POLICY]),
                }
            )
        return sbs


def shard_rows(shard_index: int, num_shards: int) -> pd.DataFrame:
    df = pd.read_csv(MANIFEST)
    return df.iloc[[i for i in range(len(df)) if i % int(num_shards) == int(shard_index)]].reset_index(drop=True)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r.keys()}))
        writer.writeheader()
        writer.writerows(rows)


def run_scan_shard(shard_index: int, num_shards: int) -> None:
    out = OUT / "support_scan"
    done = out / "DONE" / f"shard{shard_index:04d}-of-{num_shards:04d}.done"
    failed = out / "FAILED" / f"shard{shard_index:04d}-of-{num_shards:04d}.failed"
    if done.exists():
        return
    rows = shard_rows(shard_index, num_shards)
    state_rows: List[Dict[str, Any]] = []
    policy_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    try:
        for _, row in rows.iterrows():
            scenario = build_scenario(row.to_dict())
            observer = SupportScanPolicy(scenario.scenario_id)
            sim = Simulator(
                SimulatorConfig(
                    gpu_configs=list(scenario.gpu_configs),
                    service_model=ServiceModel(**dict(scenario.service_model_kwargs)),
                    max_steps=None,
                    drain_steps=50_000,
                )
            )
            sim.load_trace(list(scenario.requests))
            metrics = sim.run(observer, workload_tag=scenario.scenario_id, seed=int(scenario.seed))
            for r in observer.rows:
                r.update({"layer": row["layer"], "source": row["source"], "window_id": row.get("window_id", ""), "load_factor": row["load_factor"]})
            for r in observer.policy_rows:
                r.update({"layer": row["layer"], "source": row["source"], "window_id": row.get("window_id", ""), "load_factor": row["load_factor"]})
            state_rows.extend(observer.rows)
            policy_rows.extend(observer.policy_rows)
            n_states = len(observer.rows)
            n_dis = sum(int(r["any_non_sbs_policy_differs"]) for r in observer.rows)
            summary_rows.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "layer": row["layer"],
                    "source": row["source"],
                    "window_id": row.get("window_id", ""),
                    "load_factor": row["load_factor"],
                    "decision_states": n_states,
                    "disagreement_states": n_dis,
                    "disagreement_rate": float(n_dis / n_states) if n_states else 0.0,
                    "sbs_anwg": float(metrics.arrival_normalized_weighted_goodput),
                    "num_completed": int(metrics.num_completed),
                    "num_dropped": int(metrics.num_dropped),
                }
            )
        write_csv(out / "state_rows" / f"state_rows.shard{shard_index:04d}-of-{num_shards:04d}.csv", state_rows)
        write_csv(out / "policy_rows" / f"policy_rows.shard{shard_index:04d}-of-{num_shards:04d}.csv", policy_rows)
        write_csv(out / "scenario_summary" / f"scenario_summary.shard{shard_index:04d}-of-{num_shards:04d}.csv", summary_rows)
        done.parent.mkdir(parents=True, exist_ok=True)
        done.write_text(stable_json({"status": "done", "shard_index": shard_index, "num_shards": num_shards, "n_scenarios": len(rows)}))
    except Exception as exc:
        failed.parent.mkdir(parents=True, exist_ok=True)
        failed.write_text(stable_json({"status": "failed", "error": repr(exc), "shard_index": shard_index, "num_shards": num_shards}))
        raise


def aggregate_scan() -> None:
    out = OUT / "support_scan"
    parts = sorted((out / "scenario_summary").glob("scenario_summary.shard*.csv"))
    if not parts:
        raise FileNotFoundError("no support scan scenario_summary shards found")
    scen = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    state_parts = sorted((out / "state_rows").glob("state_rows.shard*.csv"))
    policy_parts = sorted((out / "policy_rows").glob("policy_rows.shard*.csv"))
    state_manifest = OUT / "fresh_state_manifest.csv"
    policy_manifest = OUT / "fresh_state_policy_action_map.csv"
    for p in (state_manifest, policy_manifest):
        if p.exists():
            p.unlink()

    disagreement_header_written = False
    policy_header_written = False
    branch_keys: set[Tuple[str, str]] = set()
    for path in state_parts:
        for chunk in pd.read_csv(path, chunksize=50_000):
            disagreement_states = chunk[chunk["any_non_sbs_policy_differs"].astype(int) == 1].copy()
            if len(disagreement_states):
                disagreement_states.to_csv(
                    state_manifest,
                    index=False,
                    mode="a",
                    header=not disagreement_header_written,
                )
                disagreement_header_written = True
    if not disagreement_header_written:
        state_manifest.write_text("")

    for path in policy_parts:
        for chunk in pd.read_csv(path, chunksize=50_000):
            state_policy_map = chunk[chunk["candidate_differs_from_sbs"].astype(int) == 1].copy()
            if len(state_policy_map):
                for state_id, action in zip(
                    state_policy_map["state_id"].astype(str),
                    state_policy_map["candidate_canonical_action_full"].astype(str),
                ):
                    branch_keys.add((state_id, action))
                state_policy_map.to_csv(
                    policy_manifest,
                    index=False,
                    mode="a",
                    header=not policy_header_written,
                )
                policy_header_written = True
    if not policy_header_written:
        policy_manifest.write_text("")

    summary = {
        "schema_version": SUPPORT_SCAN_VERSION,
        "scenario_count": int(scen["scenario_id"].nunique()),
        "sbs_decision_states": int(scen["decision_states"].sum()),
        "disagreement_states": int(scen["disagreement_states"].sum()),
        "disagreement_rate": float(scen["disagreement_states"].sum() / max(scen["decision_states"].sum(), 1)),
        "scenarios_with_disagreement": int((scen["disagreement_states"] > 0).sum()),
        "unique_action_branch_count_estimate": int(len(branch_keys)),
        "by_source": scen.groupby("source")[["decision_states", "disagreement_states"]].sum().to_dict(),
    }
    (OUT / "support_scan_summary.json").write_text(stable_json(summary))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    scan = sub.add_parser("scan-shard")
    scan.add_argument("--shard-index", type=int, required=True)
    scan.add_argument("--num-shards", type=int, required=True)
    sub.add_parser("aggregate-scan")
    args = ap.parse_args(argv)
    if args.cmd == "prepare":
        prepare_manifest()
    elif args.cmd == "scan-shard":
        run_scan_shard(args.shard_index, args.num_shards)
    elif args.cmd == "aggregate-scan":
        aggregate_scan()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
