"""Targeted SBS-trajectory terminal-label pilot for joint240 v1.

This experiment labels a preregistered pilot drawn from SBS-trajectory states
where at least one non-SBS P6 policy proposes a different native action.  The
causal unit is the unique canonical action at a state, not the policy name.
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
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..core.action import Action
from ..core.types import ObservableState, Request
from ..policies.base import BasePolicy
from ..policy_separation.schema import PolicySeparationScenario
from ..policy_separation.unified_utility_matrix import _build_policy
from ..simulator.service_model import ServiceModel
from ..simulator.simulator import Simulator, SimulatorConfig
from .decision_criticality_terminal_anwg_joint240_v1 import P6, load_frozen_joint240_context
from .joint240_dense_sbs_state_action_v1 import (
    ACTION_DIFF_FEATURES_VERSION,
    LIVE_STATE_FEATURES_VERSION,
    FeatureHistory,
    action_diff_features_v1,
    canonical_full_action,
    live_state_features_v1,
    run_one_step_then_sbs_terminal,
    sha256_file,
)
from .joint240_sbs_disagreement_scan_v1 import (
    SCHEMA_VERSION as SCAN_SCHEMA_VERSION,
    build_p6_policies,
    canonical_action_admit_str,
    canonical_action_full_str,
    scenario_seed,
    select_actions_without_mutation,
    stable_state_id,
)

ROOT = Path(__file__).resolve().parents[3]
SCAN_ARTIFACT_ROOT = Path(
    os.environ.get(
        "LLMSERVEOPT_JOINT240_SBS_SCAN_ROOT",
        "/home/soroush/llm-serving-heuristic-evolution-joint240-sbs-disagreement-scan-v1",
    )
).resolve()
OUT_DIR = ROOT / "experiments" / "joint240_sbs_targeted_terminal_label_pilot_v1"
SCHEMA_VERSION = "joint240_sbs_targeted_terminal_label_pilot_v1.0.0"
SBS_POLICY = "kv_constrained_online"
PILOT_SEED = 20260919
TARGET_PILOT_STATES = 300
TARGET_PER_FOLD = 60
MAX_STATES_PER_SCENARIO = 2


def _git(args: Sequence[str], cwd: Path = ROOT) -> Optional[str]:
    try:
        return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()
    except Exception:
        return None


def scan_paths(scan_root: Path = SCAN_ARTIFACT_ROOT) -> Dict[str, Path]:
    base = scan_root / "experiments" / "joint240_sbs_disagreement_scan_v1" / "full_v1"
    return {
        "manifest_all_disagreements": base / "manifest_all_disagreements.csv",
        "state_rows": base / "state_rows.csv",
        "summary": base / "summary.json",
    }


def source_provenance(scan_root: Path = SCAN_ARTIFACT_ROOT) -> Dict[str, Any]:
    inputs = scan_paths(scan_root)
    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": platform.node(),
        "git_head": _git(["rev-parse", "HEAD"]),
        "git_branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "git_status_short": _git(["status", "--short", "--branch"]),
        "scan_artifact_root": str(scan_root),
        "scan_schema_version": SCAN_SCHEMA_VERSION,
        "input_sha256": {
            name: sha256_file(path)
            for name, path in inputs.items()
            if path.exists() and path.is_file()
        },
        "input_paths": {name: str(path) for name, path in inputs.items()},
    }


def action_hash(canonical_full: str) -> str:
    return hashlib.sha256(canonical_full.encode()).hexdigest()[:24]


def row_hash(*parts: Any) -> str:
    return hashlib.sha256(":".join(str(p) for p in parts).encode()).hexdigest()


def write_rows_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_scan_disagreement_states(scan_root: Path = SCAN_ARTIFACT_ROOT) -> pd.DataFrame:
    paths = scan_paths(scan_root)
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError({"missing_scan_artifacts": missing})
    usecols = [
        "state_id",
        "scenario_id",
        "fold",
        "step",
        "sim_time",
        "any_non_sbs_policy_differs",
        "n_non_sbs_policies_differ",
        "differing_policy_ids",
        "n_distinct_canonical_p6_actions_full",
    ]
    df = pd.read_csv(paths["state_rows"], usecols=usecols)
    df = df[df["any_non_sbs_policy_differs"].astype(bool)].copy()
    df["fold"] = df["fold"].astype(int)
    df["step"] = df["step"].astype(int)
    df["n_distinct_canonical_p6_actions_full"] = df[
        "n_distinct_canonical_p6_actions_full"
    ].astype(int)
    return add_episode_and_sampling_columns(df)


def add_episode_and_sampling_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["scenario_id", "step"]).reset_index(drop=True).copy()
    episode_ids: List[str] = []
    episode_lengths: Dict[str, int] = {}
    episode_positions: List[int] = []
    scenario_max_step = df.groupby("scenario_id")["step"].transform("max").replace(0, 1)
    df["trajectory_position_bin"] = np.where(
        df["step"] <= scenario_max_step / 3.0,
        "early",
        np.where(df["step"] <= (2.0 * scenario_max_step / 3.0), "middle", "late"),
    )
    for sid, g in df.groupby("scenario_id", sort=True):
        current: List[int] = []
        prev_step: Optional[int] = None
        ep_no = 0
        for idx, step in zip(g.index, g["step"].astype(int).tolist()):
            if prev_step is None or step == prev_step + 1:
                current.append(idx)
            else:
                eid = f"{sid}::ep{ep_no:05d}"
                episode_lengths[eid] = len(current)
                for pos, row_idx in enumerate(current):
                    episode_ids.append((row_idx, eid))  # type: ignore[arg-type]
                    episode_positions.append((row_idx, pos))  # type: ignore[arg-type]
                ep_no += 1
                current = [idx]
            prev_step = step
        if current:
            eid = f"{sid}::ep{ep_no:05d}"
            episode_lengths[eid] = len(current)
            for pos, row_idx in enumerate(current):
                episode_ids.append((row_idx, eid))  # type: ignore[arg-type]
                episode_positions.append((row_idx, pos))  # type: ignore[arg-type]
    ep_map = dict(episode_ids)  # type: ignore[arg-type]
    pos_map = dict(episode_positions)  # type: ignore[arg-type]
    df["disagreement_episode_id"] = [ep_map[i] for i in df.index]
    df["disagreement_episode_length"] = [
        int(episode_lengths[df.loc[i, "disagreement_episode_id"]]) for i in df.index
    ]
    df["disagreement_episode_position"] = [int(pos_map[i]) for i in df.index]
    df["disagreement_episode_type"] = np.where(
        df["disagreement_episode_length"] == 1, "isolated", "multi_step"
    )
    df["pilot_hash"] = [
        row_hash(PILOT_SEED, r.state_id, r.scenario_id, r.step) for r in df.itertuples()
    ]
    return df


def full_manifest_unique_branch_summary(df: pd.DataFrame) -> Dict[str, Any]:
    unique_count = int(df["n_distinct_canonical_p6_actions_full"].sum())
    naive = int(len(df) * len(P6))
    return {
        "states": int(len(df)),
        "naive_policy_branch_count": naive,
        "unique_action_branch_count": unique_count,
        "branches_saved": int(naive - unique_count),
        "fraction_saved": float((naive - unique_count) / naive) if naive else 0.0,
        "distinct_action_count_distribution": {
            str(k): int(v)
            for k, v in df["n_distinct_canonical_p6_actions_full"]
            .value_counts()
            .sort_index()
            .items()
        },
    }


def deterministic_pilot_manifest(
    states: pd.DataFrame,
    *,
    target_states: int = TARGET_PILOT_STATES,
    target_per_fold: int = TARGET_PER_FOLD,
    max_per_scenario: int = MAX_STATES_PER_SCENARIO,
) -> pd.DataFrame:
    df = states.copy()
    df["policy_signature"] = df["differing_policy_ids"].fillna("").astype(str)
    scenario_counts: Counter[str] = Counter()
    selected_indices: List[int] = []
    selected = set()

    def try_add(idx: int, limit: int) -> bool:
        if idx in selected:
            return False
        sid = str(df.loc[idx, "scenario_id"])
        if scenario_counts[sid] >= limit:
            return False
        selected.add(idx)
        selected_indices.append(idx)
        scenario_counts[sid] += 1
        return True

    # Coverage pass: fold x distinct action count x episode type.
    coverage_cols = [
        ["fold", "n_distinct_canonical_p6_actions_full", "disagreement_episode_type"],
        ["fold", "trajectory_position_bin"],
        ["fold", "policy_signature"],
    ]
    for cols in coverage_cols:
        for _, g in df.sort_values("pilot_hash").groupby(cols, sort=True):
            try_add(int(g.index[0]), max_per_scenario)

    # Fill each fold to approximately 60 states while respecting scenario caps.
    for fold, g in df.sort_values("pilot_hash").groupby("fold", sort=True):
        for limit in [max_per_scenario, max_per_scenario + 1, max_per_scenario + 2]:
            while sum(1 for i in selected_indices if int(df.loc[i, "fold"]) == int(fold)) < target_per_fold:
                added = False
                for idx in g.index:
                    if try_add(int(idx), limit):
                        added = True
                        break
                if not added:
                    break
            if sum(1 for i in selected_indices if int(df.loc[i, "fold"]) == int(fold)) >= target_per_fold:
                break

    # If rare coverage made the set larger than target, trim deterministically
    # while keeping at least one row for each preregistered coverage stratum.
    pilot = df.loc[selected_indices].copy()
    if len(pilot) > target_states:
        required = set()
        for cols in coverage_cols:
            for _, g in pilot.sort_values("pilot_hash").groupby(cols, sort=True):
                required.add(int(g.index[0]))
        keep = list(required)
        remaining = [i for i in pilot.sort_values("pilot_hash").index.tolist() if int(i) not in required]
        for idx in remaining:
            if len(keep) >= target_states:
                break
            keep.append(int(idx))
        pilot = df.loc[keep].copy()

    pilot = pilot.sort_values(["fold", "scenario_id", "step"]).reset_index(drop=True)
    pilot["pilot_ord"] = np.arange(len(pilot), dtype=int)
    scenario_ord = {
        sid: i for i, sid in enumerate(sorted(pilot["scenario_id"].astype(str).unique()))
    }
    pilot["pilot_scenario_ord"] = [scenario_ord[str(sid)] for sid in pilot["scenario_id"]]
    pilot["expected_unique_action_branches"] = pilot["n_distinct_canonical_p6_actions_full"].astype(int)
    pilot["selection_seed"] = PILOT_SEED
    return pilot


def build_pilot(out_dir: Path, scan_root: Path = SCAN_ARTIFACT_ROOT) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    states = load_scan_disagreement_states(scan_root)
    pilot = deterministic_pilot_manifest(states)
    pilot_path = out_dir / "pilot_manifest.csv"
    pilot.to_csv(pilot_path, index=False)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "provenance": source_provenance(scan_root),
        "full_manifest": full_manifest_unique_branch_summary(states),
        "pilot": {
            "path": str(pilot_path),
            "sha256": sha256_file(pilot_path),
            "states": int(len(pilot)),
            "scenarios": int(pilot["scenario_id"].nunique()),
            "fold_counts": {
                str(k): int(v) for k, v in pilot["fold"].value_counts().sort_index().items()
            },
            "scenario_max_selected": int(pilot["scenario_id"].value_counts().max()) if len(pilot) else 0,
            "expected_unique_action_branches": int(pilot["expected_unique_action_branches"].sum()),
            "expected_non_sbs_unique_action_branches": int(
                (pilot["expected_unique_action_branches"] - 1).sum()
            ),
            "distinct_action_count_distribution": {
                str(k): int(v)
                for k, v in pilot["n_distinct_canonical_p6_actions_full"]
                .value_counts()
                .sort_index()
                .items()
            },
            "episode_type_counts": {
                str(k): int(v)
                for k, v in pilot["disagreement_episode_type"].value_counts().sort_index().items()
            },
            "position_bin_counts": {
                str(k): int(v)
                for k, v in pilot["trajectory_position_bin"].value_counts().sort_index().items()
            },
        },
    }
    (out_dir / "pilot_manifest_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (out_dir / "provenance.json").write_text(
        json.dumps(source_provenance(scan_root), indent=2, sort_keys=True) + "\n"
    )
    return summary


def shard_scenarios(pilot: pd.DataFrame, shard_index: int, num_shards: int) -> pd.DataFrame:
    scenarios = sorted(pilot["scenario_id"].astype(str).unique())
    keep = {sid for i, sid in enumerate(scenarios) if i % int(num_shards) == int(shard_index)}
    return pilot[pilot["scenario_id"].astype(str).isin(keep)].copy()


@dataclass
class TargetedTerminalLabelObserver(BasePolicy):
    """Executes SBS and terminal-labels preregistered target states."""

    name = "joint240_sbs_targeted_terminal_label_pilot_v1"

    sim_ref: Simulator
    scenario_id: str
    fold: int
    seed: int
    target_steps: set
    target_meta: Mapping[int, Mapping[str, Any]]
    sbs_policy: BasePolicy
    shadow_policies: Dict[str, BasePolicy]
    all_requests: Sequence[Request]

    state_action_rows: List[Dict[str, Any]] = field(default_factory=list)
    state_policy_action_map: List[Dict[str, Any]] = field(default_factory=list)
    hit_steps: set = field(default_factory=set)
    feature_history: FeatureHistory = field(default_factory=FeatureHistory)

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

        state_id = stable_state_id(self.scenario_id, step)
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
            aid = action_hash(full)
            policies = sorted(policies_by_full[full], key=P6.index)
            diff = action_diff_features_v1(state, action, sbs_action)
            q = q_by_full[full]
            is_sbs_action = full == sbs_full
            row = {
                "schema_version": SCHEMA_VERSION,
                "state_feature_version": LIVE_STATE_FEATURES_VERSION,
                "action_diff_feature_version": ACTION_DIFF_FEATURES_VERSION,
                "trajectory_policy": SBS_POLICY,
                "continuation_policy": SBS_POLICY,
                "state_id": state_id,
                "scenario_id": self.scenario_id,
                "fold": int(self.fold),
                "step": step,
                "sim_time": float(state.time),
                "seed": int(self.seed),
                "canonical_action_id": aid,
                "canonical_action_full": full,
                "canonical_action_admit": canonical_action_admit_str(action),
                "sbs_canonical_action_id": action_hash(sbs_full),
                "sbs_canonical_action_full": sbs_full,
                "is_sbs_action": int(is_sbs_action),
                "policies_generating_action": ",".join(policies),
                "n_policies_generating_action": int(len(policies)),
                "n_distinct_canonical_actions": int(len(action_by_full)),
                "policy_signature": str(meta.get("policy_signature", "")),
                "disagreement_episode_id": str(meta.get("disagreement_episode_id", "")),
                "disagreement_episode_type": str(meta.get("disagreement_episode_type", "")),
                "disagreement_episode_length": int(meta.get("disagreement_episode_length", 0)),
                "trajectory_position_bin": str(meta.get("trajectory_position_bin", "")),
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
                    "fold": int(self.fold),
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


def run_scenario_labels(
    scenario: PolicySeparationScenario,
    *,
    fold: int,
    target_rows: pd.DataFrame,
) -> Dict[str, Any]:
    sid = scenario.scenario_id
    seed = scenario_seed(scenario)
    target_steps = set(int(x) for x in target_rows["step"].tolist())
    target_meta = {int(r.step): dict(r._asdict()) for r in target_rows.itertuples(index=False)}
    sim = Simulator(
        SimulatorConfig(
            gpu_configs=list(scenario.gpu_configs),
            service_model=ServiceModel(**dict(scenario.service_model_kwargs)),
            max_steps=80_000,
            drain_steps=20_000,
        )
    )
    sim.load_trace(list(scenario.requests))
    observer = TargetedTerminalLabelObserver(
        sim_ref=sim,
        scenario_id=sid,
        fold=int(fold),
        seed=int(seed),
        target_steps=target_steps,
        target_meta=target_meta,
        sbs_policy=_build_policy(SBS_POLICY)[0],
        shadow_policies=build_p6_policies(),
        all_requests=list(scenario.requests),
    )
    metrics = sim.run(observer, workload_tag=sid, seed=seed)
    missing = sorted(target_steps - observer.hit_steps)
    return {
        "scenario_id": sid,
        "fold": int(fold),
        "seed": int(seed),
        "n_target_states": int(len(target_steps)),
        "n_hit_states": int(len(observer.hit_steps)),
        "n_missing_states": int(len(missing)),
        "missing_steps": missing,
        "sbs_trajectory_anwg": float(metrics.arrival_normalized_weighted_goodput),
        "state_action_rows": observer.state_action_rows,
        "state_policy_action_map": observer.state_policy_action_map,
    }


def run_label_shard(
    *,
    pilot_manifest: Path,
    out_dir: Path,
    shard_index: int = 0,
    num_shards: int = 1,
    max_states: Optional[int] = None,
    resume: bool = True,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / f"state_action_rows.shard{shard_index:04d}-of-{num_shards:04d}.csv"
    map_path = out_dir / f"state_policy_action_map.shard{shard_index:04d}-of-{num_shards:04d}.csv"
    summary_path = out_dir / f"summary.shard{shard_index:04d}-of-{num_shards:04d}.json"
    done_path = out_dir / f"DONE.shard{shard_index:04d}-of-{num_shards:04d}"
    failed_path = out_dir / f"FAILED.shard{shard_index:04d}-of-{num_shards:04d}"
    if resume and done_path.exists() and rows_path.exists() and map_path.exists() and summary_path.exists():
        existing = json.loads(summary_path.read_text())
        existing["resume_skipped"] = True
        return existing
    if failed_path.exists():
        failed_path.unlink()

    started = time.perf_counter()
    pilot = pd.read_csv(pilot_manifest)
    shard = shard_scenarios(pilot, shard_index, num_shards)
    if max_states is not None:
        shard = shard.head(int(max_states))

    ctx = load_frozen_joint240_context()
    scenario_rows: List[Dict[str, Any]] = []
    state_action_rows: List[Dict[str, Any]] = []
    map_rows: List[Dict[str, Any]] = []
    for sid, g in shard.groupby("scenario_id", sort=True):
        result = run_scenario_labels(
            ctx["scenarios"][sid],
            fold=int(g["fold"].iloc[0]),
            target_rows=g,
        )
        scenario_rows.append({k: v for k, v in result.items() if k not in ("state_action_rows", "state_policy_action_map")})
        state_action_rows.extend(result["state_action_rows"])
        map_rows.extend(result["state_policy_action_map"])

    write_rows_csv(rows_path, state_action_rows)
    write_rows_csv(map_path, map_rows)
    failed = [s for s in scenario_rows if int(s["n_missing_states"]) > 0]
    expected_states = int(len(shard))
    observed_states = len({r["state_id"] for r in state_action_rows})
    expected_policy_map = expected_states * len(P6)
    done = not failed and observed_states == expected_states and len(map_rows) == expected_policy_map
    summary = {
        "schema_version": SCHEMA_VERSION,
        "shard_index": int(shard_index),
        "num_shards": int(num_shards),
        "n_input_states": expected_states,
        "n_hit_states": int(observed_states),
        "n_state_action_rows": int(len(state_action_rows)),
        "n_state_policy_action_map_rows": int(len(map_rows)),
        "n_scenarios": int(shard["scenario_id"].nunique()) if len(shard) else 0,
        "expected_unique_action_branches": int(shard["expected_unique_action_branches"].sum()) if "expected_unique_action_branches" in shard else None,
        "expected_policy_map_rows": int(expected_policy_map),
        "n_failed_scenarios": int(len(failed)),
        "failed_scenarios": failed[:20],
        "wall_seconds": float(time.perf_counter() - started),
        "rows_path": str(rows_path),
        "map_path": str(map_path),
        "done": bool(done),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    sentinel = done_path if done else failed_path
    sentinel.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def analyze_completed_pilot(out_dir: Path, num_shards: int) -> Dict[str, Any]:
    missing = [
        str(out_dir / f"DONE.shard{i:04d}-of-{num_shards:04d}")
        for i in range(num_shards)
        if not (out_dir / f"DONE.shard{i:04d}-of-{num_shards:04d}").exists()
    ]
    if missing:
        raise FileNotFoundError({"pilot_not_complete_missing_done": missing})
    row_parts = [
        pd.read_csv(out_dir / f"state_action_rows.shard{i:04d}-of-{num_shards:04d}.csv")
        for i in range(num_shards)
        if (out_dir / f"state_action_rows.shard{i:04d}-of-{num_shards:04d}.csv").stat().st_size
    ]
    map_parts = [
        pd.read_csv(out_dir / f"state_policy_action_map.shard{i:04d}-of-{num_shards:04d}.csv")
        for i in range(num_shards)
        if (out_dir / f"state_policy_action_map.shard{i:04d}-of-{num_shards:04d}.csv").stat().st_size
    ]
    rows = pd.concat(row_parts, ignore_index=True) if row_parts else pd.DataFrame()
    maps = pd.concat(map_parts, ignore_index=True) if map_parts else pd.DataFrame()
    rows.to_csv(out_dir / "state_action_rows.csv", index=False)
    maps.to_csv(out_dir / "state_policy_action_map.csv", index=False)
    non_sbs = rows[rows["is_sbs_action"].astype(int) == 0].copy()
    adv = pd.to_numeric(non_sbs["a_sbs_anwg"], errors="coerce")
    state_best = non_sbs.groupby("state_id")["a_sbs_anwg"].max()
    pos_states = rows.loc[rows["state_id"].isin(state_best[state_best > 0].index)]
    neg_states = non_sbs.groupby("state_id")["a_sbs_anwg"].max()
    harmful_only = neg_states[neg_states < 0].index
    summary = {
        "schema_version": SCHEMA_VERSION,
        "pilot_states": int(rows["state_id"].nunique()) if len(rows) else 0,
        "unique_action_rows": int(len(rows)),
        "non_sbs_unique_action_rows": int(len(non_sbs)),
        "causal_effect_support": {
            "positive": int((adv > 0).sum()),
            "negative": int((adv < 0).sum()),
            "zero": int((adv == 0).sum()),
            "nonzero": int((adv != 0).sum()),
        },
        "state_level_beneficial_opportunity": {
            "best_advantage_gt_0_states": int((state_best > 0).sum()),
            "best_advantage_eq_0_states": int((state_best == 0).sum()),
            "all_differing_actions_harmful_states": int(len(harmful_only)),
        },
        "effective_sample_support": {
            "positive_states": int(pos_states["state_id"].nunique()) if len(pos_states) else 0,
            "positive_scenarios": int(pos_states["scenario_id"].nunique()) if len(pos_states) else 0,
            "positive_folds": int(pos_states["fold"].nunique()) if len(pos_states) else 0,
            "negative_states": int(non_sbs.loc[adv < 0, "state_id"].nunique()) if len(non_sbs) else 0,
            "negative_scenarios": int(non_sbs.loc[adv < 0, "scenario_id"].nunique()) if len(non_sbs) else 0,
            "negative_folds": int(non_sbs.loc[adv < 0, "fold"].nunique()) if len(non_sbs) else 0,
        },
        "rows_path": str(out_dir / "state_action_rows.csv"),
        "map_path": str(out_dir / "state_policy_action_map.csv"),
        "rows_sha256": sha256_file(out_dir / "state_action_rows.csv"),
        "map_sha256": sha256_file(out_dir / "state_policy_action_map.csv"),
    }
    (out_dir / "pilot_analysis_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def cmd_prepare(args: argparse.Namespace) -> None:
    summary = build_pilot(Path(args.output_dir).resolve(), Path(args.scan_root).resolve())
    print(json.dumps(summary, indent=2, sort_keys=True))


def cmd_run_shard(args: argparse.Namespace) -> None:
    summary = run_label_shard(
        pilot_manifest=Path(args.pilot_manifest).resolve(),
        out_dir=Path(args.output_dir).resolve(),
        shard_index=int(args.shard_index),
        num_shards=int(args.num_shards),
        max_states=args.max_states,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def cmd_analyze(args: argparse.Namespace) -> None:
    summary = analyze_completed_pilot(Path(args.output_dir).resolve(), int(args.num_shards))
    print(json.dumps(summary, indent=2, sort_keys=True))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default=str(OUT_DIR / "pilot_v1"))
    p.add_argument("--scan-root", default=str(SCAN_ARTIFACT_ROOT))
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("prepare")
    sp.set_defaults(func=cmd_prepare)
    sp = sub.add_parser("run-shard")
    sp.add_argument("--pilot-manifest", required=True)
    sp.add_argument("--shard-index", type=int, required=True)
    sp.add_argument("--num-shards", type=int, required=True)
    sp.add_argument("--max-states", type=int)
    sp.set_defaults(func=cmd_run_shard)
    sp = sub.add_parser("analyze")
    sp.add_argument("--num-shards", type=int, required=True)
    sp.set_defaults(func=cmd_analyze)
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
