"""SBS-trajectory P6 action-support scan for joint240 v1.

This is a read/compute-only diagnostic. It executes fixed SBS
(`kv_constrained_online`) on each scenario and, before each SBS action is
applied, queries all six P6 native policies on independent state copies.

No terminal counterfactual forks and no learning are performed here.
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
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..core.action import Action
from ..core.types import ObservableState
from ..policies.base import BasePolicy
from ..policy_separation.schema import PolicySeparationScenario
from ..policy_separation.unified_utility_matrix import _build_policy
from ..simulator.service_model import ServiceModel
from ..simulator.simulator import Simulator, SimulatorConfig
from .decision_criticality_terminal_anwg_joint240_v1 import (
    P6,
    load_frozen_joint240_context,
)
from .joint240_dense_sbs_state_action_v1 import (
    ACTION_DIFF_FEATURES_VERSION,
    LIVE_STATE_FEATURES_VERSION,
    FeatureHistory,
    action_diff_features_v1,
    canonical_full_action,
    live_state_features_v1,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "experiments" / "joint240_sbs_disagreement_scan_v1"
SCHEMA_VERSION = "joint240_sbs_disagreement_scan_v1.0.0"
SBS_POLICY = "kv_constrained_online"
PILOT_SCENARIO_COUNT = 4
THIN_EPISODE_SPACING = 10
CONTROL_RATIO = 0.10
MANIFEST_SEED = 20260919


def _git(args: Sequence[str], cwd: Path = ROOT) -> Optional[str]:
    try:
        return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()
    except Exception:
        return None


def stable_state_id(scenario_id: str, step: int) -> str:
    return f"joint240_sbs::{scenario_id}::{int(step)}"


def scenario_seed(scenario: PolicySeparationScenario) -> int:
    return int(getattr(scenario, "seed", scenario.params.get("seed", 0)))


def build_p6_policies() -> Dict[str, BasePolicy]:
    return {pid: _build_policy(pid)[0] for pid in P6}


def canonical_action_full_str(action: Action) -> str:
    return str(canonical_full_action(action))


def canonical_action_admit_str(action: Action) -> str:
    admit = tuple(sorted((int(k), tuple(sorted(map(int, v)))) for k, v in action.admit.items() if v))
    return str(admit)


def select_actions_without_mutation(
    state: ObservableState,
    policies: Mapping[str, BasePolicy],
) -> Dict[str, Action]:
    actions: Dict[str, Action] = {}
    for pid in P6:
        actions[pid] = policies[pid].select_action(copy.deepcopy(state))
    return actions


def bool_to_int(v: bool) -> int:
    return 1 if bool(v) else 0


@dataclass
class SBSTrajectoryScanPolicy(BasePolicy):
    """Executes SBS, records shadow P6 disagreement at each decision point."""

    name = "joint240_sbs_disagreement_scan_v1"

    scenario_id: str
    fold: int
    sbs_policy: BasePolicy
    shadow_policies: Dict[str, BasePolicy]
    max_recorded_steps: Optional[int] = None

    state_rows: List[Dict[str, Any]] = field(default_factory=list)
    policy_rows: List[Dict[str, Any]] = field(default_factory=list)
    history: FeatureHistory = field(default_factory=FeatureHistory)

    def reset(self) -> None:
        if hasattr(self.sbs_policy, "reset"):
            self.sbs_policy.reset()
        for policy in self.shadow_policies.values():
            if hasattr(policy, "reset"):
                policy.reset()
        self.state_rows = []
        self.policy_rows = []
        self.history = FeatureHistory()

    def select_action(self, state: ObservableState) -> Action:
        actions = select_actions_without_mutation(state, self.shadow_policies)
        sbs_action = actions[SBS_POLICY]
        canonical_full = {pid: canonical_action_full_str(a) for pid, a in actions.items()}
        canonical_admit = {pid: canonical_action_admit_str(a) for pid, a in actions.items()}
        sbs_full = canonical_full[SBS_POLICY]
        differing = [pid for pid in P6 if pid != SBS_POLICY and canonical_full[pid] != sbs_full]
        state_id = stable_state_id(self.scenario_id, int(state.step))

        feats = live_state_features_v1(state, history=self.history)
        self.history.update(state, feats)

        row: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "trajectory_policy": SBS_POLICY,
            "scenario_id": self.scenario_id,
            "fold": int(self.fold),
            "step": int(state.step),
            "sim_time": float(state.time),
            "state_id": state_id,
            "sbs_canonical_action_full": sbs_full,
            "sbs_canonical_action_admit": canonical_admit[SBS_POLICY],
            "n_distinct_canonical_p6_actions_full": int(len(set(canonical_full.values()))),
            "n_distinct_canonical_p6_actions_admit": int(len(set(canonical_admit.values()))),
            "any_non_sbs_policy_differs": bool_to_int(bool(differing)),
            "n_non_sbs_policies_differ": int(len(differing)),
            "differing_policy_ids": ",".join(differing),
        }
        for pid in P6:
            row[f"canonical_full__{pid}"] = canonical_full[pid]
            row[f"canonical_admit__{pid}"] = canonical_admit[pid]
            row[f"differs_from_sbs__{pid}"] = bool_to_int(canonical_full[pid] != sbs_full)
        row.update({f"state__{k}": v for k, v in feats.items()})
        self.state_rows.append(row)

        for pid in P6:
            if pid == SBS_POLICY:
                continue
            diff = action_diff_features_v1(state, actions[pid], sbs_action)
            prow = {
                "schema_version": SCHEMA_VERSION,
                "state_feature_version": LIVE_STATE_FEATURES_VERSION,
                "action_diff_feature_version": ACTION_DIFF_FEATURES_VERSION,
                "trajectory_policy": SBS_POLICY,
                "scenario_id": self.scenario_id,
                "fold": int(self.fold),
                "step": int(state.step),
                "sim_time": float(state.time),
                "state_id": state_id,
                "candidate_policy_id": pid,
                "candidate_policy_index": int(P6.index(pid)),
                "sbs_policy_id": SBS_POLICY,
                "candidate_canonical_action_full": canonical_full[pid],
                "candidate_canonical_action_admit": canonical_admit[pid],
                "sbs_canonical_action_full": sbs_full,
                "sbs_canonical_action_admit": canonical_admit[SBS_POLICY],
                "candidate_differs_from_sbs": bool_to_int(canonical_full[pid] != sbs_full),
            }
            prow.update({f"action__{k}": v for k, v in diff.items()})
            self.policy_rows.append(prow)

        # Execute exactly SBS, not any shadow action.
        if self.max_recorded_steps is not None and len(self.state_rows) >= self.max_recorded_steps:
            return sbs_action
        return sbs_action


def run_scenario_scan(
    scenario: PolicySeparationScenario,
    *,
    fold: int,
    max_steps: Optional[int] = None,
) -> Dict[str, Any]:
    sid = scenario.scenario_id
    sbs_policy = _build_policy(SBS_POLICY)[0]
    observer = SBSTrajectoryScanPolicy(
        scenario_id=sid,
        fold=int(fold),
        sbs_policy=sbs_policy,
        shadow_policies=build_p6_policies(),
        max_recorded_steps=max_steps,
    )
    sim = Simulator(
        SimulatorConfig(
            gpu_configs=list(scenario.gpu_configs),
            service_model=ServiceModel(**dict(scenario.service_model_kwargs)),
            max_steps=int(max_steps) if max_steps is not None else 80_000,
            drain_steps=20_000,
        )
    )
    sim.load_trace(list(scenario.requests))
    metrics = sim.run(observer, workload_tag=sid, seed=scenario_seed(scenario))
    return {
        "scenario_id": sid,
        "fold": int(fold),
        "n_state_rows": int(len(observer.state_rows)),
        "n_policy_rows": int(len(observer.policy_rows)),
        "sbs_anwg": float(metrics.arrival_normalized_weighted_goodput),
        "sim_duration": float(metrics.sim_duration),
        "state_rows": observer.state_rows,
        "policy_rows": observer.policy_rows,
    }


def load_context() -> Dict[str, Any]:
    ctx = load_frozen_joint240_context()
    folds = ctx["folds"].set_index("scenario_id")["fold"].astype(int).to_dict()
    ctx["fold_by_scenario"] = folds
    return ctx


def deterministic_pilot_scenarios(ctx: Mapping[str, Any], n: int = PILOT_SCENARIO_COUNT) -> List[str]:
    folds = ctx["folds"].copy()
    folds["_h"] = [
        hashlib.sha256(f"{MANIFEST_SEED}:{sid}".encode()).hexdigest()
        for sid in folds["scenario_id"].astype(str)
    ]
    parts = []
    per_fold = max(1, math.ceil(n / 5))
    for _, g in folds.sort_values("_h").groupby("fold", sort=True):
        parts.extend(g.head(per_fold)["scenario_id"].tolist())
    return sorted(parts[:n])


def scenario_ids_for_shard(scenario_ids: Sequence[str], shard_index: int, num_shards: int) -> List[str]:
    return [sid for i, sid in enumerate(sorted(scenario_ids)) if i % int(num_shards) == int(shard_index)]


def write_rows_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_scan(
    *,
    out_dir: Path,
    scenario_ids: Optional[Sequence[str]] = None,
    shard_index: int = 0,
    num_shards: int = 1,
    max_steps_per_scenario: Optional[int] = None,
    resume: bool = True,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / f"state_rows.shard{shard_index:04d}-of-{num_shards:04d}.csv"
    policy_path = out_dir / f"policy_rows.shard{shard_index:04d}-of-{num_shards:04d}.csv"
    summary_path = out_dir / f"summary.shard{shard_index:04d}-of-{num_shards:04d}.json"
    done_path = out_dir / f"DONE.shard{shard_index:04d}-of-{num_shards:04d}"
    if resume and done_path.exists() and state_path.exists() and policy_path.exists() and summary_path.exists():
        existing = json.loads(summary_path.read_text())
        existing["resume_skipped"] = True
        return existing

    started = time.perf_counter()
    ctx = load_context()
    all_ids = sorted(ctx["scenarios"].keys()) if scenario_ids is None else sorted(scenario_ids)
    shard_ids = scenario_ids_for_shard(all_ids, shard_index, num_shards)
    states: List[Dict[str, Any]] = []
    policies: List[Dict[str, Any]] = []
    scenarios = []
    for sid in shard_ids:
        result = run_scenario_scan(
            ctx["scenarios"][sid],
            fold=int(ctx["fold_by_scenario"][sid]),
            max_steps=max_steps_per_scenario,
        )
        scenarios.append({k: v for k, v in result.items() if k not in ("state_rows", "policy_rows")})
        states.extend(result["state_rows"])
        policies.extend(result["policy_rows"])

    write_rows_csv(state_path, states)
    write_rows_csv(policy_path, policies)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "shard_index": int(shard_index),
        "num_shards": int(num_shards),
        "scenario_ids": list(shard_ids),
        "n_scenarios": int(len(shard_ids)),
        "n_state_rows": int(len(states)),
        "n_policy_rows": int(len(policies)),
        "wall_seconds": float(time.perf_counter() - started),
        "state_rows_path": str(state_path),
        "policy_rows_path": str(policy_path),
        "scenario_summaries": scenarios,
        "done": True,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    done_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def episode_lengths(flags: Sequence[bool]) -> List[int]:
    lengths: List[int] = []
    current = 0
    for flag in flags:
        if flag:
            current += 1
        elif current:
            lengths.append(current)
            current = 0
    if current:
        lengths.append(current)
    return lengths


def longest_true_run(flags: Sequence[bool]) -> int:
    lengths = episode_lengths(flags)
    return max(lengths) if lengths else 0


def first_last_true(steps: Sequence[int], flags: Sequence[bool]) -> Tuple[Optional[int], Optional[int]]:
    vals = [int(s) for s, f in zip(steps, flags) if f]
    if not vals:
        return None, None
    return vals[0], vals[-1]


def summarize_feature_groups(states: pd.DataFrame) -> Dict[str, Any]:
    groups = {
        "queue_load": ["state__waiting_count", "state__active_count", "state__total_in_system_count", "state__waiting_predicted_total_tokens_sum"],
        "kv_pressure": ["state__kv_utilization_mean", "state__kv_utilization_max", "state__projected_kv_pressure_waiting", "state__kv_tokens_free_sum"],
        "prefill_decode": ["state__active_prefilling_count", "state__active_decoding_count", "state__prefill_decode_request_ratio", "state__prefill_decode_token_ratio"],
        "request_size": ["state__admissible_prompt_tokens_mean", "state__admissible_predicted_output_tokens_mean", "state__admissible_predicted_total_tokens_p90"],
        "slack_urgency": ["state__slack_s_min", "state__slack_s_p50", "state__urgent_request_count_slack_le_5s", "state__slo_violation_count_slack_le_0"],
        "priority_fairness": ["state__priority_sum_admissible", "state__priority_max_admissible", "state__priority_entropy_admissible", "state__class_entropy_admissible"],
        "recent_dynamics": ["state__hist_w1_waiting_count_delta", "state__hist_w5_waiting_count_delta", "state__hist_w20_waiting_count_delta", "state__hist_w5_new_request_count"],
    }
    out: Dict[str, Any] = {}
    states = states.copy()
    states["_disagree"] = states["any_non_sbs_policy_differs"].astype(bool)
    for group, cols in groups.items():
        rows = {}
        for col in cols:
            if col not in states:
                continue
            a = pd.to_numeric(states[col], errors="coerce")
            eq = a[~states["_disagree"]]
            dis = a[states["_disagree"]]
            rows[col] = {
                "all_equal_mean": float(eq.mean()) if len(eq) else None,
                "disagreement_mean": float(dis.mean()) if len(dis) else None,
                "all_equal_p50": float(eq.quantile(0.5)) if len(eq) else None,
                "disagreement_p50": float(dis.quantile(0.5)) if len(dis) else None,
            }
        out[group] = rows
    return out


def build_manifests(states: pd.DataFrame, policies: pd.DataFrame, out_dir: Path) -> Dict[str, Any]:
    disagree = states[states["any_non_sbs_policy_differs"].astype(bool)].copy()
    equal = states[~states["any_non_sbs_policy_differs"].astype(bool)].copy()
    all_path = out_dir / "manifest_all_disagreements.csv"
    cols = ["state_id", "scenario_id", "fold", "step", "sim_time", "n_non_sbs_policies_differ", "differing_policy_ids"]
    disagree[cols].to_csv(all_path, index=False)

    thinned_parts = []
    for _, g in disagree.sort_values(["scenario_id", "step"]).groupby("scenario_id", sort=True):
        flags = [True] * len(g)
        # g already contains only disagreement states; split into contiguous step episodes.
        episode: List[int] = []
        prev = None
        indices = list(g.index)
        for idx in indices:
            step = int(g.loc[idx, "step"])
            if prev is None or step == prev + 1:
                episode.append(idx)
            else:
                thinned_parts.extend(episode[0::THIN_EPISODE_SPACING])
                episode = [idx]
            prev = step
        if episode:
            thinned_parts.extend(episode[0::THIN_EPISODE_SPACING])
    thinned = disagree.loc[sorted(set(thinned_parts))].copy() if thinned_parts else disagree.head(0).copy()
    thin_path = out_dir / "manifest_episode_thinned.csv"
    thinned[cols].to_csv(thin_path, index=False)

    controls_n = min(len(equal), int(math.ceil(CONTROL_RATIO * max(1, len(disagree)))))
    if controls_n:
        equal = equal.copy()
        equal["_h"] = [hashlib.sha256(f"{MANIFEST_SEED}:control:{s}".encode()).hexdigest() for s in equal["state_id"]]
        controls = equal.sort_values("_h").head(controls_n).drop(columns=["_h"])
    else:
        controls = equal.head(0)
    controls_path = out_dir / "manifest_all_equal_controls.csv"
    controls[cols].to_csv(controls_path, index=False)

    def manifest_summary(df: pd.DataFrame) -> Dict[str, Any]:
        ids = set(df["state_id"]) if len(df) else set()
        pol = policies[policies["state_id"].isin(ids) & policies["candidate_differs_from_sbs"].astype(bool)]
        return {
            "states": int(len(df)),
            "scenarios": int(df["scenario_id"].nunique()) if len(df) else 0,
            "folds": {str(k): int(v) for k, v in df["fold"].value_counts().sort_index().items()} if len(df) else {},
            "per_policy_action_support": {str(k): int(v) for k, v in pol["candidate_policy_id"].value_counts().sort_index().items()},
            "expected_state_x_6_terminal_branches": int(len(df) * 6),
        }

    return {
        "all_disagreements_path": str(all_path),
        "episode_thinned_path": str(thin_path),
        "all_equal_controls_path": str(controls_path),
        "MANIFEST_ALL_DISAGREEMENTS": manifest_summary(disagree),
        "MANIFEST_EPISODE_THINNED": manifest_summary(thinned),
        "CONTROLS_ALL_EQUAL": manifest_summary(controls),
        "thinning_rule": f"Within each contiguous disagreement episode, keep first state and every {THIN_EPISODE_SPACING}th subsequent state; outcome-blind.",
        "control_rule": f"Deterministic SHA256 sample of all-equal states at ceil({CONTROL_RATIO} * disagreement_states).",
    }


def analyze_outputs(out_dir: Path, *, num_shards: int) -> Dict[str, Any]:
    state_parts = []
    policy_parts = []
    for i in range(num_shards):
        s = out_dir / f"state_rows.shard{i:04d}-of-{num_shards:04d}.csv"
        p = out_dir / f"policy_rows.shard{i:04d}-of-{num_shards:04d}.csv"
        if not s.exists() or not p.exists():
            raise FileNotFoundError(f"missing shard {i}")
        if s.stat().st_size:
            state_parts.append(pd.read_csv(s))
        if p.stat().st_size:
            policy_parts.append(pd.read_csv(p))
    states = pd.concat(state_parts, ignore_index=True) if state_parts else pd.DataFrame()
    policies = pd.concat(policy_parts, ignore_index=True) if policy_parts else pd.DataFrame()
    states.to_csv(out_dir / "state_rows.csv", index=False)
    policies.to_csv(out_dir / "policy_rows.csv", index=False)
    return analyze_frames(states, policies, out_dir)


def analyze_frames(states: pd.DataFrame, policies: pd.DataFrame, out_dir: Path) -> Dict[str, Any]:
    flags = states["any_non_sbs_policy_differs"].astype(bool)
    total = int(len(states))
    diff_count = int(flags.sum())
    n_diff_alts = states["n_non_sbs_policies_differ"].astype(int)
    distinct = states["n_distinct_canonical_p6_actions_full"].astype(int)
    per_policy = {}
    for pid, g in policies.groupby("candidate_policy_id", sort=True):
        diff = g["candidate_differs_from_sbs"].astype(bool)
        by_sc = g.assign(_diff=diff).groupby("scenario_id")["_diff"].sum()
        per_policy[str(pid)] = {
            "disagreement_rows": int(diff.sum()),
            "disagreement_fraction": float(diff.mean()) if len(diff) else 0.0,
            "scenarios_with_any": int((by_sc > 0).sum()),
            "median_disagreements_per_scenario": float(by_sc.median()) if len(by_sc) else 0.0,
            "p90_disagreements_per_scenario": float(by_sc.quantile(0.9)) if len(by_sc) else 0.0,
            "max_disagreements_per_scenario": int(by_sc.max()) if len(by_sc) else 0,
        }
    pairwise = {}
    for a in P6:
        pairwise[a] = {}
        for b in P6:
            pairwise[a][b] = float((states[f"canonical_full__{a}"] != states[f"canonical_full__{b}"]).mean()) if total else 0.0
    scenario_rows = []
    for sid, g in states.sort_values(["scenario_id", "step"]).groupby("scenario_id", sort=True):
        gf = g["any_non_sbs_policy_differs"].astype(bool).tolist()
        first, last = first_last_true(g["step"].tolist(), gf)
        scenario_rows.append({
            "scenario_id": sid,
            "fold": int(g["fold"].iloc[0]),
            "total_sbs_states": int(len(g)),
            "disagreement_states": int(sum(gf)),
            "disagreement_fraction": float(sum(gf) / len(g)) if len(g) else 0.0,
            "policies_providing_disagreements": ",".join(sorted(set(",".join(g.loc[g["any_non_sbs_policy_differs"].astype(bool), "differing_policy_ids"]).split(",")) - {""})),
            "longest_consecutive_disagreement_run": int(longest_true_run(gf)),
            "first_disagreement_step": first,
            "last_disagreement_step": last,
        })
    scenario_df = pd.DataFrame(scenario_rows)
    scenario_df.to_csv(out_dir / "per_scenario_support.csv", index=False)
    fold_rows = []
    for fold, g in states.groupby("fold", sort=True):
        ids = set(g["state_id"])
        pol = policies[policies["state_id"].isin(ids) & policies["candidate_differs_from_sbs"].astype(bool)]
        fold_rows.append({
            "fold": int(fold),
            "scenarios": int(g["scenario_id"].nunique()),
            "total_sbs_states": int(len(g)),
            "disagreement_states": int(g["any_non_sbs_policy_differs"].astype(bool).sum()),
            "disagreement_rate": float(g["any_non_sbs_policy_differs"].astype(bool).mean()) if len(g) else 0.0,
            **{f"policy_disagreements__{pid}": int((pol["candidate_policy_id"] == pid).sum()) for pid in P6 if pid != SBS_POLICY},
        })
    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(out_dir / "per_fold_support.csv", index=False)

    ep_lengths_all = []
    for _, g in states.sort_values(["scenario_id", "step"]).groupby("scenario_id", sort=True):
        ep_lengths_all.extend(episode_lengths(g["any_non_sbs_policy_differs"].astype(bool).tolist()))
    ep = np.asarray(ep_lengths_all, dtype=float)
    state_episode_bins = Counter()
    for l in ep_lengths_all:
        if l == 1:
            state_episode_bins["1"] += l
        elif l <= 5:
            state_episode_bins["2-5"] += l
        elif l <= 20:
            state_episode_bins["6-20"] += l
        else:
            state_episode_bins[">20"] += l
    total_disagreement_states = max(1, int(diff_count))
    episode_summary = {
        "episode_count": int(len(ep_lengths_all)),
        "length_min": float(ep.min()) if len(ep) else None,
        "length_p25": float(np.quantile(ep, 0.25)) if len(ep) else None,
        "length_p50": float(np.quantile(ep, 0.50)) if len(ep) else None,
        "length_p75": float(np.quantile(ep, 0.75)) if len(ep) else None,
        "length_p90": float(np.quantile(ep, 0.90)) if len(ep) else None,
        "length_p95": float(np.quantile(ep, 0.95)) if len(ep) else None,
        "length_max": float(ep.max()) if len(ep) else None,
        "fraction_disagreement_states_by_episode_length": {
            k: float(v / total_disagreement_states) for k, v in sorted(state_episode_bins.items())
        },
    }
    manifests = build_manifests(states, policies, out_dir)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "total_sbs_decision_states": total,
        "global_support": {
            "all_p6_equal_sbs_count": int(total - diff_count),
            "all_p6_equal_sbs_fraction": float((total - diff_count) / total) if total else 0.0,
            "at_least_one_alt_differs_count": diff_count,
            "at_least_one_alt_differs_fraction": float(diff_count / total) if total else 0.0,
            "n_alternatives_differ_distribution": {str(k): int(v) for k, v in n_diff_alts.value_counts().sort_index().items()},
            "n_distinct_canonical_actions_distribution": {str(k): int(v) for k, v in distinct.value_counts().sort_index().items()},
        },
        "per_policy_support": per_policy,
        "pairwise_disagreement_matrix": pairwise,
        "per_scenario_summary_path": str(out_dir / "per_scenario_support.csv"),
        "per_fold_summary_path": str(out_dir / "per_fold_support.csv"),
        "per_fold_support": fold_rows,
        "episode_structure": episode_summary,
        "feature_descriptive_summary": summarize_feature_groups(states),
        "manifests": manifests,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def provenance(out_dir: Path) -> Dict[str, Any]:
    inputs = {
        "split_oof_folds": ROOT / "experiments" / "joint240_same_distribution_adaptive_exploitability_v1" / "split_oof_folds.csv",
        "utility_matrix_wide": ROOT / "experiments" / "joint_multimechanism_generalization_v1" / "utility_matrix_wide.csv",
        "scenario_manifest": ROOT / "experiments" / "joint_multimechanism_generalization_v1" / "scenario_manifest.csv",
    }
    data = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": platform.node(),
        "git_head": _git(["rev-parse", "HEAD"]),
        "git_branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "git_status_short": _git(["status", "--short", "--branch"]),
        "input_sha256": {k: sha256_file(v) for k, v in inputs.items() if v.exists()},
        "out_dir": str(out_dir),
    }
    return data


def write_preregistered_schema(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "provenance.json").write_text(json.dumps(provenance(out_dir), indent=2, sort_keys=True) + "\n")


def cmd_pilot(args: argparse.Namespace) -> None:
    out = Path(args.output_dir).resolve()
    ctx = load_context()
    ids = deterministic_pilot_scenarios(ctx, int(args.n_scenarios))
    write_preregistered_schema(out)
    summary = run_scan(
        out_dir=out,
        scenario_ids=ids,
        max_steps_per_scenario=args.max_steps_per_scenario,
    )
    analysis = analyze_outputs(out, num_shards=1)
    summary["analysis"] = analysis
    print(json.dumps(summary, indent=2, sort_keys=True))


def cmd_run_shard(args: argparse.Namespace) -> None:
    write_preregistered_schema(Path(args.output_dir).resolve())
    ctx = load_context()
    summary = run_scan(
        out_dir=Path(args.output_dir).resolve(),
        scenario_ids=sorted(ctx["scenarios"].keys()),
        shard_index=int(args.shard_index),
        num_shards=int(args.num_shards),
        max_steps_per_scenario=args.max_steps_per_scenario,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def cmd_aggregate(args: argparse.Namespace) -> None:
    summary = analyze_outputs(Path(args.output_dir).resolve(), num_shards=int(args.num_shards))
    print(json.dumps(summary, indent=2, sort_keys=True))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default=str(OUT_DIR))
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("pilot")
    sp.add_argument("--n-scenarios", type=int, default=PILOT_SCENARIO_COUNT)
    sp.add_argument("--max-steps-per-scenario", type=int)
    sp.set_defaults(func=cmd_pilot)
    sp = sub.add_parser("run-shard")
    sp.add_argument("--shard-index", type=int, required=True)
    sp.add_argument("--num-shards", type=int, required=True)
    sp.add_argument("--max-steps-per-scenario", type=int)
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
