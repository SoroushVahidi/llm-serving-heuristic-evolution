#!/usr/bin/env python
"""Fresh ID-like SBS override confirmatory terminal labels.

Generates terminal labels only. It does not train selectors, summarize target
effects, or run closed-loop scheduling.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from llmserveopt.core.action import Action
from llmserveopt.core.types import ObservableState, Request
from llmserveopt.policies.base import BasePolicy
from llmserveopt.policy_separation.schema import PolicySeparationScenario
from llmserveopt.policy_separation.unified_utility_matrix import _build_policy
from llmserveopt.simulator.service_model import ServiceModel
from llmserveopt.simulator.simulator import Simulator, SimulatorConfig
from llmserveopt.analysis.joint240_same_distribution_adaptive_v1 import P6
from llmserveopt.analysis.joint240_dense_sbs_state_action_v1 import (
    ACTION_DIFF_FEATURES_VERSION,
    LIVE_STATE_FEATURES_VERSION,
    FeatureHistory,
    action_diff_features_v1,
    live_state_features_v1,
    run_one_step_then_sbs_terminal,
)

SBS_POLICY = "kv_constrained_online"
SCHEMA_VERSION = "sbs_override_fresh_id_confirmatory_terminal_label_v1.0.0"
CONFIRMATORY_ROOT = ROOT / "experiments" / "sbs_override_fresh_confirmatory_corpus_v1"
OUT_ROOT = ROOT / "experiments" / "sbs_override_fresh_id_confirmatory_terminal_label_v1"
STATE_MANIFEST = CONFIRMATORY_ROOT / "fresh_state_manifest.full_support_only.csv"
POLICY_ACTION_MAP = CONFIRMATORY_ROOT / "fresh_state_policy_action_map.full_support_only.csv"
CANDIDATE_MANIFEST = CONFIRMATORY_ROOT / "fresh_candidate_scenario_manifest.csv"
SUPPORT_SUMMARY = CONFIRMATORY_ROOT / "fresh_id_clean_manifest_summary.json"


def stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, indent=2, separators=(",", ": ")) + "\n"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(args: Sequence[str]) -> Optional[str]:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def load_fresh_builder():
    path = ROOT / "scripts" / "sbs_override_fresh_confirmatory_corpus_v1.py"
    spec = importlib.util.spec_from_file_location("sbs_fresh_confirmatory_v1", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def action_hash(canonical_full: str) -> str:
    return hashlib.sha256(canonical_full.encode()).hexdigest()[:24]


def fresh_state_id(scenario_id: str, step: int) -> str:
    return f"sbsfresh::{scenario_id}::{int(step)}"


def canonical_action_full_str(action: Action) -> str:
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


def canonical_action_admit_str(action: Action) -> str:
    return repr(tuple(sorted((int(k), tuple(sorted(map(int, v)))) for k, v in action.admit.items() if v)))


def build_p6_policies() -> Dict[str, BasePolicy]:
    return {pid: _build_policy(pid)[0] for pid in P6}


def select_actions_without_mutation(state: ObservableState, policies: Mapping[str, BasePolicy]) -> Dict[str, Action]:
    return {pid: policies[pid].select_action(copy.deepcopy(state)) for pid in P6}


def write_rows_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if not rows:
        tmp.write_text("")
        tmp.replace(path)
        return
    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    states = pd.read_csv(STATE_MANIFEST)
    maps = pd.read_csv(POLICY_ACTION_MAP)
    candidates = pd.read_csv(CANDIDATE_MANIFEST)
    return states, maps, candidates


def validate_inputs() -> Dict[str, Any]:
    states, maps, candidates = load_inputs()
    fresh_candidates = candidates[candidates["source"] == "fresh_joint_multimechanism_generator_holdout"].copy()
    dev_ids = {f"joint_mm_{i:04d}" for i in range(240)}
    scenario_ids = set(states["scenario_id"].astype(str))
    branch_keys = set(zip(maps["state_id"].astype(str), maps["candidate_canonical_action_full"].astype(str)))
    unique_non_sbs = len(branch_keys)
    n_distinct_total = int(states["n_distinct_canonical_p6_actions_full"].astype(int).sum())
    expected = {
        "state_rows": int(len(states)),
        "unique_states": int(states["state_id"].nunique()),
        "duplicate_state_rows": int(states.duplicated("state_id").sum()),
        "scenario_count_with_disagreement": int(states["scenario_id"].nunique()),
        "fresh_candidate_scenarios": int(len(fresh_candidates)),
        "all_sources": states["source"].value_counts().to_dict(),
        "development_overlap_count": int(len(scenario_ids & dev_ids)),
        "policy_action_map_rows": int(len(maps)),
        "policy_action_map_states": int(maps["state_id"].nunique()),
        "unique_non_sbs_action_branches": int(unique_non_sbs),
        "expected_sbs_reference_branches": int(states["state_id"].nunique()),
        "expected_total_unique_terminal_continuations": int(states["state_id"].nunique() + unique_non_sbs),
        "sum_n_distinct_canonical_p6_actions_full": n_distinct_total,
    }
    failures = []
    if expected["duplicate_state_rows"]:
        failures.append("duplicate_state_ids")
    if expected["development_overlap_count"]:
        failures.append("development_scenario_overlap")
    if expected["all_sources"] != {"fresh_joint_multimechanism_generator_holdout": len(states)}:
        failures.append("non_fresh_source_in_state_manifest")
    if expected["policy_action_map_states"] != expected["unique_states"]:
        failures.append("policy_map_state_coverage")
    if n_distinct_total != expected["expected_total_unique_terminal_continuations"]:
        failures.append("branch_count_mismatch")
    expected["validation_failures"] = failures
    return expected


def provenance() -> Dict[str, Any]:
    validation = validate_inputs()
    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": platform.node(),
        "git": {
            "head": _git(["rev-parse", "HEAD"]),
            "branch": _git(["branch", "--show-current"]),
            "status_short": _git(["status", "--short", "--branch"]),
        },
        "duplication_gate": {
            "classification": "NOT_PREVIOUSLY_DONE",
            "note": "Search found joint240 terminal labels and fresh support manifests, but no fresh generator-holdout forced-action terminal labels.",
        },
        "confirmatory_blindness": {
            "development": "all joint240 labels/results, selector development, nested CV, folds, models, thresholds",
            "confirmatory": "80 fresh generator-holdout scenarios; terminal counterfactual outcomes unseen before this campaign",
            "allowed_before_labeling": ["scenario identity", "SBS trajectories", "P6 support", "features", "canonical actions"],
            "forbidden_for_selector_design": ["fresh Q_SBS", "fresh A_SBS", "fresh beneficial/harmful labels"],
        },
        "causal_semantics": "force unique canonical action once at SBS predecision state, then fixed kv_constrained_online continuation",
        "input_paths": {
            "state_manifest": str(STATE_MANIFEST),
            "policy_action_map": str(POLICY_ACTION_MAP),
            "candidate_manifest": str(CANDIDATE_MANIFEST),
            "support_summary": str(SUPPORT_SUMMARY),
        },
        "input_sha256": {
            "state_manifest": sha256_file(STATE_MANIFEST),
            "policy_action_map": sha256_file(POLICY_ACTION_MAP),
            "candidate_manifest": sha256_file(CANDIDATE_MANIFEST),
            "support_summary": sha256_file(SUPPORT_SUMMARY) if SUPPORT_SUMMARY.exists() else None,
        },
        "validation": validation,
        "ood_null_result_preservation": {
            "azure_2023_code": "zero canonical SBS-vs-P6 disagreement",
            "azure_2023_conversation": "zero canonical SBS-vs-P6 disagreement",
            "azure_2024": "zero canonical SBS-vs-P6 disagreement",
            "bailian_qwen": "zero canonical SBS-vs-P6 disagreement",
            "burstgpt_v2": "zero canonical SBS-vs-P6 disagreement",
        },
        "implementation_reuse": [
            "llmserveopt.analysis.joint240_dense_sbs_state_action_v1.run_one_step_then_sbs_terminal",
            "live_state_features_v1",
            "action_diff_features_v1",
            "sbs_override_fresh_confirmatory_corpus_v1.build_scenario",
            "joint240 targeted terminal label DONE/FAILED sharding pattern",
        ],
    }


def scenario_rows_for_shard(manifest: pd.DataFrame, shard_index: int, num_shards: int) -> pd.DataFrame:
    scenarios = sorted(manifest["scenario_id"].astype(str).unique())
    keep = {sid for i, sid in enumerate(scenarios) if i % int(num_shards) == int(shard_index)}
    return manifest[manifest["scenario_id"].astype(str).isin(keep)].copy()


def candidate_by_scenario() -> Dict[str, Mapping[str, Any]]:
    _states, _maps, candidates = load_inputs()
    fresh = candidates[candidates["source"] == "fresh_joint_multimechanism_generator_holdout"].copy()
    return {str(r.scenario_id): dict(r._asdict()) for r in fresh.itertuples(index=False)}


class FreshTargetedTerminalLabelObserver(BasePolicy):
    name = "sbs_override_fresh_id_confirmatory_terminal_label_v1"

    def __init__(
        self,
        *,
        sim_ref: Simulator,
        scenario_id: str,
        seed: int,
        target_steps: set[int],
        target_meta: Mapping[int, Mapping[str, Any]],
        sbs_policy: BasePolicy,
        shadow_policies: Dict[str, BasePolicy],
        all_requests: Sequence[Request],
    ) -> None:
        self.sim_ref = sim_ref
        self.scenario_id = scenario_id
        self.seed = int(seed)
        self.target_steps = target_steps
        self.target_meta = target_meta
        self.sbs_policy = sbs_policy
        self.shadow_policies = shadow_policies
        self.all_requests = all_requests
        self.state_action_rows: List[Dict[str, Any]] = []
        self.state_policy_action_map: List[Dict[str, Any]] = []
        self.hit_steps: set[int] = set()
        self.feature_history = FeatureHistory()

    def reset(self) -> None:
        if hasattr(self.sbs_policy, "reset"):
            self.sbs_policy.reset()
        for policy in self.shadow_policies.values():
            if hasattr(policy, "reset"):
                policy.reset()
        self.state_action_rows = []
        self.state_policy_action_map = []
        self.hit_steps = set()
        self.feature_history = FeatureHistory()

    def select_action(self, state: ObservableState) -> Action:
        step = int(state.step)
        actions = select_actions_without_mutation(state, self.shadow_policies)
        sbs_action = actions[SBS_POLICY]
        state_features = live_state_features_v1(state, history=self.feature_history)
        self.feature_history.update(state, state_features)
        if step not in self.target_steps or step in self.hit_steps:
            return copy.deepcopy(sbs_action)

        state_id = fresh_state_id(self.scenario_id, step)
        meta = dict(self.target_meta.get(step, {}))
        canonical_by_policy = {pid: canonical_action_full_str(action) for pid, action in actions.items()}
        canonical_admit_by_policy = {pid: canonical_action_admit_str(action) for pid, action in actions.items()}
        sbs_full = canonical_by_policy[SBS_POLICY]
        action_by_full: Dict[str, Action] = {}
        policies_by_full: Dict[str, List[str]] = defaultdict(list)
        for pid in P6:
            full = canonical_by_policy[pid]
            action_by_full.setdefault(full, actions[pid])
            policies_by_full[full].append(pid)
        if sbs_full not in action_by_full:
            raise RuntimeError(f"SBS action missing from unique-action set at {state_id}")

        q_by_full: Dict[str, Dict[str, Any]] = {}
        for full, action in sorted(action_by_full.items(), key=lambda kv: (kv[0] != sbs_full, kv[0])):
            q_by_full[full] = run_one_step_then_sbs_terminal(
                self.sim_ref,
                first_action=action,
                continuation=_build_policy(SBS_POLICY)[0],
                all_requests=self.all_requests,
                workload_tag=self.scenario_id,
                seed=self.seed,
            )
        q0 = q_by_full[sbs_full]

        for full, action in sorted(action_by_full.items()):
            policies = sorted(policies_by_full[full], key=P6.index)
            diff = action_diff_features_v1(state, action, sbs_action)
            q = q_by_full[full]
            row = {
                "schema_version": SCHEMA_VERSION,
                "state_feature_version": LIVE_STATE_FEATURES_VERSION,
                "action_diff_feature_version": ACTION_DIFF_FEATURES_VERSION,
                "trajectory_policy": SBS_POLICY,
                "continuation_policy": SBS_POLICY,
                "state_id": state_id,
                "scenario_id": self.scenario_id,
                "step": step,
                "sim_time": float(state.time),
                "seed": int(self.seed),
                "canonical_action_id": action_hash(full),
                "canonical_action_full": full,
                "canonical_action_admit": canonical_action_admit_str(action),
                "sbs_canonical_action_id": action_hash(sbs_full),
                "sbs_canonical_action_full": sbs_full,
                "is_sbs_action": int(full == sbs_full),
                "policies_generating_action": ",".join(policies),
                "n_policies_generating_action": int(len(policies)),
                "n_distinct_canonical_actions": int(len(action_by_full)),
                "differing_policy_ids": str(meta.get("differing_policy_ids", "")),
                "support_scan_schema_version": str(meta.get("schema_version", "")),
                **{f"state__{k}": v for k, v in state_features.items()},
                **{f"action__{k}": v for k, v in diff.items()},
                **q,
                "q0_sbs_anwg": float(q0["q_sbs_anwg"]),
                "q0_sbs_soft": float(q0["q_sbs_soft"]),
                "q0_sbs_wcg": float(q0["q_sbs_wcg"]),
                "q0_sbs_wmt": float(q0["q_sbs_wmt"]),
                "q0_sbs_wnt": float(q0["q_sbs_wnt"]),
                "a_sbs_anwg": float(q["q_sbs_anwg"] - q0["q_sbs_anwg"]),
                "a_sbs_soft": float(q["q_sbs_soft"] - q0["q_sbs_soft"]),
                "a_sbs_wcg": float(q["q_sbs_wcg"] - q0["q_sbs_wcg"]),
                "a_sbs_wmt_improvement": float(q0["q_sbs_wmt"] - q["q_sbs_wmt"]),
                "a_sbs_wnt_improvement": float(q0["q_sbs_wnt"] - q["q_sbs_wnt"]),
            }
            self.state_action_rows.append(row)

        for pid in P6:
            full = canonical_by_policy[pid]
            self.state_policy_action_map.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "state_id": state_id,
                    "scenario_id": self.scenario_id,
                    "step": step,
                    "policy_id": pid,
                    "policy_index": int(P6.index(pid)),
                    "canonical_action_id": action_hash(full),
                    "canonical_action_full": full,
                    "canonical_action_admit": canonical_admit_by_policy[pid],
                    "sbs_canonical_action_id": action_hash(sbs_full),
                    "action_equal_to_sbs": int(full == sbs_full),
                    "is_sbs_policy": int(pid == SBS_POLICY),
                }
            )
        self.hit_steps.add(step)
        return copy.deepcopy(sbs_action)


def run_scenario_labels(scenario: PolicySeparationScenario, target_rows: pd.DataFrame) -> Dict[str, Any]:
    seed = int(getattr(scenario, "seed", scenario.params.get("seed", 0)))
    steps = set(int(x) for x in target_rows["step"].tolist())
    meta = {int(r.step): dict(r._asdict()) for r in target_rows.itertuples(index=False)}
    sim = Simulator(
        SimulatorConfig(
            gpu_configs=list(scenario.gpu_configs),
            service_model=ServiceModel(**dict(scenario.service_model_kwargs)),
            max_steps=None,
            drain_steps=50_000,
        )
    )
    sim.load_trace(list(scenario.requests))
    observer = FreshTargetedTerminalLabelObserver(
        sim_ref=sim,
        scenario_id=scenario.scenario_id,
        seed=seed,
        target_steps=steps,
        target_meta=meta,
        sbs_policy=_build_policy(SBS_POLICY)[0],
        shadow_policies=build_p6_policies(),
        all_requests=list(scenario.requests),
    )
    metrics = sim.run(observer, workload_tag=scenario.scenario_id, seed=seed)
    missing = sorted(steps - observer.hit_steps)
    return {
        "scenario_id": scenario.scenario_id,
        "seed": seed,
        "n_target_states": int(len(steps)),
        "n_hit_states": int(len(observer.hit_steps)),
        "n_missing_states": int(len(missing)),
        "missing_steps": missing,
        "sbs_trajectory_anwg": float(metrics.arrival_normalized_weighted_goodput),
        "state_action_rows": observer.state_action_rows,
        "state_policy_action_map": observer.state_policy_action_map,
    }


def run_label_shard(out_dir: Path, shard_index: int, num_shards: int, max_states: Optional[int] = None) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / f"state_action_rows.shard{shard_index:04d}-of-{num_shards:04d}.csv"
    map_path = out_dir / f"state_policy_action_map.shard{shard_index:04d}-of-{num_shards:04d}.csv"
    summary_path = out_dir / f"summary.shard{shard_index:04d}-of-{num_shards:04d}.json"
    done_path = out_dir / f"DONE.shard{shard_index:04d}-of-{num_shards:04d}"
    failed_path = out_dir / f"FAILED.shard{shard_index:04d}-of-{num_shards:04d}"
    if done_path.exists() and rows_path.exists() and map_path.exists() and summary_path.exists():
        existing = json.loads(summary_path.read_text())
        existing["resume_skipped"] = True
        return existing
    if failed_path.exists():
        failed_path.unlink()

    started = time.perf_counter()
    states, maps, _candidates = load_inputs()
    states = states.sort_values(["scenario_id", "step"]).reset_index(drop=True)
    shard = scenario_rows_for_shard(states, shard_index, num_shards)
    if max_states is not None:
        shard = shard.head(int(max_states))
    by_scenario = candidate_by_scenario()
    builder = load_fresh_builder()
    scenario_rows: List[Dict[str, Any]] = []
    state_action_rows: List[Dict[str, Any]] = []
    policy_map_rows: List[Dict[str, Any]] = []
    for sid, g in shard.groupby("scenario_id", sort=True):
        scenario = builder.build_scenario(by_scenario[str(sid)])
        result = run_scenario_labels(scenario, g)
        scenario_rows.append({k: v for k, v in result.items() if k not in ("state_action_rows", "state_policy_action_map")})
        state_action_rows.extend(result["state_action_rows"])
        policy_map_rows.extend(result["state_policy_action_map"])

    write_rows_csv(rows_path, state_action_rows)
    write_rows_csv(map_path, policy_map_rows)
    failed = [s for s in scenario_rows if int(s["n_missing_states"]) > 0]
    expected_states = int(len(shard))
    observed_states = len({r["state_id"] for r in state_action_rows})
    expected_policy_map_rows = expected_states * len(P6)
    expected_unique_rows = int(shard["n_distinct_canonical_p6_actions_full"].astype(int).sum())
    done = (
        not failed
        and observed_states == expected_states
        and len(policy_map_rows) == expected_policy_map_rows
        and len(state_action_rows) == expected_unique_rows
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "shard_index": int(shard_index),
        "num_shards": int(num_shards),
        "n_input_states": expected_states,
        "n_hit_states": int(observed_states),
        "n_state_action_rows": int(len(state_action_rows)),
        "expected_unique_action_rows": expected_unique_rows,
        "n_state_policy_action_map_rows": int(len(policy_map_rows)),
        "expected_policy_map_rows": expected_policy_map_rows,
        "n_scenarios": int(shard["scenario_id"].nunique()) if len(shard) else 0,
        "n_failed_scenarios": int(len(failed)),
        "failed_scenarios": failed[:10],
        "wall_seconds": float(time.perf_counter() - started),
        "rows_path": str(rows_path),
        "map_path": str(map_path),
        "done": bool(done),
    }
    summary_path.write_text(stable_json(summary))
    (done_path if done else failed_path).write_text(stable_json(summary))
    return summary


def cmd_prepare(args: argparse.Namespace) -> None:
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    prov = provenance()
    if prov["validation"]["validation_failures"]:
        raise SystemExit(stable_json(prov["validation"]))
    (out / "provenance.json").write_text(stable_json(prov))
    print(stable_json(prov))


def cmd_validate(args: argparse.Namespace) -> None:
    print(stable_json(validate_inputs()))


def cmd_run_shard(args: argparse.Namespace) -> None:
    summary = run_label_shard(Path(args.output_dir).resolve(), int(args.shard_index), int(args.num_shards), args.max_states)
    print(stable_json(summary))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default=str(OUT_ROOT / "run_v1"))
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare").set_defaults(func=cmd_prepare)
    sub.add_parser("validate").set_defaults(func=cmd_validate)
    sp = sub.add_parser("run-shard")
    sp.add_argument("--shard-index", type=int, required=True)
    sp.add_argument("--num-shards", type=int, required=True)
    sp.add_argument("--max-states", type=int)
    sp.set_defaults(func=cmd_run_shard)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
