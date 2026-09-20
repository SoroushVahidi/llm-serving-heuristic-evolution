#!/usr/bin/env python3
"""Outcome-blind Stage-A predictions for the frozen SBS override V2 selector."""

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
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / "src", ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import joblib
import numpy as np
import pandas as pd

import sbs_override_fresh_confirmatory_corpus_v1 as fresh_builder
from llmserveopt.analysis.joint240_dense_sbs_state_action_v1 import (
    ACTION_DIFF_FEATURES_VERSION,
    LIVE_STATE_FEATURES_VERSION,
    FeatureHistory,
    action_diff_features_v1,
    live_state_features_v1,
)
from llmserveopt.core.action import Action
from llmserveopt.core.types import ObservableState
from llmserveopt.policies.base import BasePolicy
from llmserveopt.policy_separation.unified_utility_matrix import _build_policy
from llmserveopt.simulator.service_model import ServiceModel
from llmserveopt.simulator.simulator import Simulator, SimulatorConfig


SCHEMA_VERSION = "sbs_override_fresh_id_stage_a_predict_v2.0.0"
SELECTOR_JSON_HASH = "1fdd2c0a19234bafd679a2678664119328fa37ae7d51c0178359640c2955de35"
SELECTOR_JOBLIB_HASH = "f882a54b808d9ab8f2c4ec139e19e3096ea5e3152cadf4c0badaa55c9644cfcd"
EXPECTED_SELECTOR_FAMILY = "HIST_GRADIENT_BOOSTING"
EXPECTED_SELECTOR_TAU = 0.005
EXPECTED_FEATURE_COUNT = 165
SBS_POLICY = "kv_constrained_online"
P6 = tuple(fresh_builder.P6)

CONFIRMATORY_ROOT = ROOT / "experiments" / "sbs_override_fresh_confirmatory_corpus_v1"
V2_ROOT = ROOT / "experiments" / "sbs_override_conservative_selector_dev_v2" / "run_v2"
OUT_ROOT = ROOT / "experiments" / "sbs_override_fresh_id_confirmatory_v2"
STAGE_A_ROOT = OUT_ROOT / "stage_a"

DEFAULT_STATE_MANIFEST = CONFIRMATORY_ROOT / "fresh_state_manifest.full_support_only.csv"
DEFAULT_POLICY_MAP = CONFIRMATORY_ROOT / "fresh_state_policy_action_map.full_support_only.csv"
DEFAULT_SCENARIO_MANIFEST = CONFIRMATORY_ROOT / "fresh_candidate_scenario_manifest.csv"
DEFAULT_SELECTOR_JSON = V2_ROOT / "FINAL_CONFIRMATORY_SELECTOR_V2.json"
DEFAULT_SELECTOR_JOBLIB = V2_ROOT / "FINAL_CONFIRMATORY_SELECTOR_V2.joblib"
DEFAULT_STRUCTURAL_FREEZE = STAGE_A_ROOT / "STRUCTURAL_UNIVERSE_FREEZE_V2.json"
DEFAULT_PREDICTIONS = STAGE_A_ROOT / "FROZEN_PREDICTIONS_V2.csv"
DEFAULT_STAGE_A_FREEZE = STAGE_A_ROOT / "STAGE_A_FREEZE.json"

FORBIDDEN_OUTCOME_MARKERS = (
    "q_sbs",
    "a_sbs",
    "terminal",
    "utility",
    "realized",
    "beneficial",
    "harmful",
    "gain",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_json(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=True, separators=(",", ": ")) + "\n"


def write_json(path: Path, obj: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(stable_json(obj))


def git(args: Sequence[str]) -> Optional[str]:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except Exception:
        return None


def package_versions() -> dict[str, str]:
    names = ["numpy", "pandas", "scikit-learn", "joblib"]
    versions = {}
    for name in names:
        try:
            versions[name.replace("-", "_")] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name.replace("-", "_")] = "NOT_INSTALLED"
    versions["python"] = platform.python_version()
    versions["platform"] = platform.platform()
    return versions


def reject_outcome_columns(columns: Sequence[str]) -> None:
    bad = [
        c
        for c in columns
        if any(marker in c.lower() for marker in FORBIDDEN_OUTCOME_MARKERS)
    ]
    if bad:
        raise ValueError({"outcome_columns_forbidden_in_stage_a": bad[:20]})


def canonical_action_full_str(action: Action) -> str:
    return repr(
        {
            "admit": {int(k): sorted(map(int, v)) for k, v in sorted(action.admit.items()) if v},
            "preempt": {int(k): sorted(map(int, v)) for k, v in sorted(action.preempt.items()) if v},
            "swap": {int(k): sorted(map(int, v)) for k, v in sorted(action.swap.items()) if v},
            "migrate": {
                int(k): sorted((int(a), int(b)) for a, b in v)
                for k, v in sorted(action.migrate.items())
                if v
            },
            "hold_decode": {
                int(k): sorted(map(int, v)) for k, v in sorted(action.hold_decode.items()) if v
            },
            "prefill_chunk_override": {
                int(k): int(v) for k, v in sorted(action.prefill_chunk_override.items())
            },
        }
    )


def action_hash(canonical_full: str) -> str:
    return hashlib.sha256(canonical_full.encode("utf-8")).hexdigest()[:24]


def fresh_state_id(scenario_id: str, step: int) -> str:
    return f"sbsfresh::{scenario_id}::{int(step)}"


def load_selector(selector_json: Path, selector_joblib: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    json_hash = sha256_file(selector_json)
    joblib_hash = sha256_file(selector_joblib)
    if json_hash != SELECTOR_JSON_HASH:
        raise ValueError({"selector_json_hash_mismatch": json_hash})
    if joblib_hash != SELECTOR_JOBLIB_HASH:
        raise ValueError({"selector_joblib_hash_mismatch": joblib_hash})
    selector = json.loads(selector_json.read_text())
    bundle = joblib.load(selector_joblib)
    if bundle.get("selector") != selector:
        raise ValueError("selector JSON and joblib embedded selector differ")
    if selector["family"] != EXPECTED_SELECTOR_FAMILY:
        raise ValueError(selector["family"])
    if selector["gate"]["gate"] != "MEAN_THRESHOLD":
        raise ValueError(selector["gate"])
    if float(selector["gate"]["tau"]) != EXPECTED_SELECTOR_TAU:
        raise ValueError(selector["gate"])
    if float(selector["gate"]["margin"]) != 0.0:
        raise ValueError(selector["gate"])
    if len(selector["feature_columns"]) != EXPECTED_FEATURE_COUNT:
        raise ValueError("unexpected selector feature count")
    return selector, bundle


def runtime_feature_order() -> list[str]:
    state = ObservableState(time=0.0, waiting_queue=[], gpu_states=[], completed_count=0, step=0)
    action = Action()
    state_cols = [
        f"state__{name}"
        for name in live_state_features_v1(state, history=FeatureHistory()).keys()
    ]
    action_cols = [
        f"action__{name}"
        for name in action_diff_features_v1(state, action, action).keys()
    ]
    return state_cols + action_cols


def assert_feature_order(selector: Mapping[str, Any]) -> None:
    observed = runtime_feature_order()
    expected = list(selector["feature_columns"])
    if observed != expected:
        first = next(
            (i for i, (a, b) in enumerate(zip(observed, expected)) if a != b),
            None,
        )
        raise ValueError(
            {
                "feature_order_mismatch": {
                    "observed_count": len(observed),
                    "expected_count": len(expected),
                    "first_mismatch": first,
                    "observed": observed[first] if first is not None else None,
                    "expected": expected[first] if first is not None else None,
                }
            }
        )


def load_structural_universe(
    state_manifest: Path,
    policy_map: Path,
    scenario_manifest: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    states = pd.read_csv(state_manifest)
    maps = pd.read_csv(policy_map)
    scenarios = pd.read_csv(scenario_manifest)
    reject_outcome_columns(states.columns)
    reject_outcome_columns(maps.columns)
    reject_outcome_columns(scenarios.columns)

    if len(states) != 2862:
        raise ValueError({"unexpected_clean_state_rows": len(states)})
    if states["state_id"].nunique() != 2862 or states.duplicated("state_id").any():
        raise ValueError("clean manifest must contain exactly 2862 unique states")
    supported = int(states["scenario_id"].nunique())
    if supported != 78:
        raise ValueError({"unexpected_supported_scenarios": supported})
    selected = int(scenarios[scenarios["source"] == "fresh_joint_multimechanism_generator_holdout"]["scenario_id"].nunique())
    if selected != 80:
        raise ValueError({"unexpected_selected_scenarios": selected})
    branch_keys = set(
        zip(maps["state_id"].astype(str), maps["candidate_canonical_action_full"].astype(str))
    )
    if len(branch_keys) != 6996:
        raise ValueError({"unexpected_unique_non_sbs_branches": len(branch_keys)})
    if set(maps["state_id"].astype(str)) != set(states["state_id"].astype(str)):
        raise ValueError("policy map state coverage mismatch")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "state_manifest": str(state_manifest),
        "policy_action_map": str(policy_map),
        "scenario_manifest": str(scenario_manifest),
        "input_sha256": {
            "state_manifest": sha256_file(state_manifest),
            "policy_action_map": sha256_file(policy_map),
            "scenario_manifest": sha256_file(scenario_manifest),
        },
        "raw_disagreement_rows": 2918,
        "raw_duplicate_state_id_rows": 56,
        "clean_unique_disagreement_states": int(len(states)),
        "clean_unique_state_ids": int(states["state_id"].nunique()),
        "selected_scenarios": selected,
        "supported_scenarios": supported,
        "policy_action_map_rows": int(len(maps)),
        "unique_non_sbs_canonical_branches": int(len(branch_keys)),
        "sbs_reference_branches": int(states["state_id"].nunique()),
        "total_terminal_continuations": int(states["state_id"].nunique() + len(branch_keys)),
        "duplicate_handling": "first raw occurrence by state_id, yielding full_support_only clean manifest",
        "confirmatory_label_access": "NOT_ACCESSED",
    }
    return states, maps, scenarios, summary


def write_structural_universe_freeze(args: argparse.Namespace) -> dict[str, Any]:
    _states, _maps, _scenarios, summary = load_structural_universe(
        Path(args.state_manifest),
        Path(args.policy_action_map),
        Path(args.scenario_manifest),
    )
    write_json(Path(args.structural_freeze), summary)
    return summary


def build_p6_policies() -> dict[str, BasePolicy]:
    return {pid: _build_policy(pid)[0] for pid in P6}


def select_actions_without_mutation(
    state: ObservableState, policies: Mapping[str, BasePolicy]
) -> dict[str, Action]:
    return {pid: policies[pid].select_action(copy.deepcopy(state)) for pid in P6}


def predict_scores(bundle: Mapping[str, Any], feature_df: pd.DataFrame, feature_columns: Sequence[str]) -> np.ndarray:
    x = feature_df[list(feature_columns)].to_numpy(dtype=float)
    if bundle["kind"] == "single":
        model = next(iter(bundle["models"].values()))
        return np.asarray(model.predict(x), dtype=float)
    pred = 0.5 * np.asarray(bundle["models"]["extra_trees"].predict(x), dtype=float)
    pred += 0.5 * np.asarray(bundle["models"]["hist_gradient_boosting"].predict(x), dtype=float)
    return pred


class StageAObserver(BasePolicy):
    name = "sbs_override_fresh_id_stage_a_predict_v2"

    def __init__(
        self,
        *,
        scenario_id: str,
        target_steps: set[int],
        expected_branches: Mapping[str, set[str]],
        selector: Mapping[str, Any],
        bundle: Mapping[str, Any],
        shadow_policies: dict[str, BasePolicy],
    ) -> None:
        self.scenario_id = scenario_id
        self.target_steps = set(target_steps)
        self.expected_branches = expected_branches
        self.selector = selector
        self.bundle = bundle
        self.shadow_policies = shadow_policies
        self.history = FeatureHistory()
        self.decisions: list[dict[str, Any]] = []
        self.hit_steps: set[int] = set()

    def reset(self) -> None:
        for policy in self.shadow_policies.values():
            if hasattr(policy, "reset"):
                policy.reset()
        self.history = FeatureHistory()
        self.decisions = []
        self.hit_steps = set()

    def select_action(self, state: ObservableState) -> Action:
        step = int(state.step)
        actions = select_actions_without_mutation(state, self.shadow_policies)
        sbs_action = actions[SBS_POLICY]
        state_features = live_state_features_v1(state, history=self.history)
        self.history.update(state, state_features)
        if step not in self.target_steps or step in self.hit_steps:
            return copy.deepcopy(sbs_action)

        state_id = fresh_state_id(self.scenario_id, step)
        canonical_by_policy = {pid: canonical_action_full_str(action) for pid, action in actions.items()}
        sbs_full = canonical_by_policy[SBS_POLICY]
        action_by_full: dict[str, Action] = {}
        policies_by_full: dict[str, list[str]] = defaultdict(list)
        for pid in P6:
            full = canonical_by_policy[pid]
            action_by_full.setdefault(full, actions[pid])
            policies_by_full[full].append(pid)

        candidate_fulls = sorted(full for full in action_by_full if full != sbs_full)
        expected = set(self.expected_branches.get(state_id, set()))
        if set(candidate_fulls) != expected:
            raise ValueError(
                {
                    "canonical_branch_mismatch": {
                        "state_id": state_id,
                        "observed": len(candidate_fulls),
                        "expected": len(expected),
                    }
                }
            )

        rows: list[dict[str, Any]] = []
        for full in candidate_fulls:
            action = action_by_full[full]
            features = {
                **{f"state__{k}": v for k, v in state_features.items()},
                **{f"action__{k}": v for k, v in action_diff_features_v1(state, action, sbs_action).items()},
            }
            row = {
                "state_id": state_id,
                "scenario_id": self.scenario_id,
                "step": step,
                "candidate_canonical_action_id": action_hash(full),
                "candidate_canonical_action_full": full,
                "policies_generating_action": ",".join(sorted(policies_by_full[full], key=P6.index)),
                **features,
            }
            rows.append(row)

        pred_df = pd.DataFrame(rows)
        feature_columns = list(self.selector["feature_columns"])
        if list(pred_df[feature_columns].columns) != feature_columns:
            raise ValueError("Stage-A feature column order mismatch")
        scores = predict_scores(self.bundle, pred_df, feature_columns)
        pred_df["predicted_advantage"] = scores
        best = pred_df.sort_values(
            ["predicted_advantage", "candidate_canonical_action_id"],
            ascending=[False, True],
        ).iloc[0]
        tau = float(self.selector["gate"]["tau"])
        margin = float(self.selector["gate"]["margin"])
        gate_score = float(best["predicted_advantage"]) - margin
        override = bool(gate_score > tau)
        selected_id = str(best["candidate_canonical_action_id"]) if override else "SBS_ABSTAIN"
        selected_full = str(best["candidate_canonical_action_full"]) if override else ""
        self.decisions.append(
            {
                "schema_version": SCHEMA_VERSION,
                "selector_name": self.selector["name"],
                "selector_family": self.selector["family"],
                "selector_weight_strategy": self.selector["weight_strategy"],
                "gate": self.selector["gate"]["gate"],
                "tau": tau,
                "margin": margin,
                "state_id": state_id,
                "scenario_id": self.scenario_id,
                "step": step,
                "sbs_canonical_action_id": action_hash(sbs_full),
                "sbs_canonical_action_full": sbs_full,
                "n_candidate_non_sbs_actions": int(len(candidate_fulls)),
                "selected_canonical_action_id": selected_id,
                "selected_canonical_action_full": selected_full,
                "override": int(override),
                "abstain": int(not override),
                "selected_predicted_advantage": float(best["predicted_advantage"]),
                "gate_score": gate_score,
                "selector_json_sha256": SELECTOR_JSON_HASH,
                "selector_joblib_sha256": SELECTOR_JOBLIB_HASH,
            }
        )
        self.hit_steps.add(step)
        return copy.deepcopy(sbs_action)


def candidate_by_scenario(scenario_manifest: pd.DataFrame) -> dict[str, Mapping[str, Any]]:
    fresh = scenario_manifest[
        scenario_manifest["source"] == "fresh_joint_multimechanism_generator_holdout"
    ].copy()
    return {str(r.scenario_id): dict(r._asdict()) for r in fresh.itertuples(index=False)}


def generate_predictions(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selector, bundle = load_selector(Path(args.selector_json), Path(args.selector_joblib))
    assert_feature_order(selector)
    states, maps, scenarios, structural = load_structural_universe(
        Path(args.state_manifest),
        Path(args.policy_action_map),
        Path(args.scenario_manifest),
    )
    expected_branches: dict[str, set[str]] = defaultdict(set)
    for row in maps.itertuples(index=False):
        expected_branches[str(row.state_id)].add(str(row.candidate_canonical_action_full))

    by_scenario = candidate_by_scenario(scenarios)
    decisions: list[dict[str, Any]] = []
    for sid, group in states.sort_values(["scenario_id", "step"]).groupby("scenario_id", sort=True):
        scenario = fresh_builder.build_scenario(by_scenario[str(sid)])
        observer = StageAObserver(
            scenario_id=str(sid),
            target_steps=set(int(x) for x in group["step"].tolist()),
            expected_branches=expected_branches,
            selector=selector,
            bundle=bundle,
            shadow_policies=build_p6_policies(),
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
        sim.run(observer, workload_tag=scenario.scenario_id, seed=int(scenario.seed))
        missing = set(int(x) for x in group["step"].tolist()) - observer.hit_steps
        if missing:
            raise RuntimeError({"stage_a_missing_steps": {"scenario_id": str(sid), "steps": sorted(missing)[:10]}})
        decisions.extend(observer.decisions)

    validate_predictions(decisions, states, maps)
    return decisions, structural


def validate_predictions(
    decisions: Sequence[Mapping[str, Any]],
    states: pd.DataFrame,
    maps: pd.DataFrame,
) -> None:
    if len(decisions) != 2862:
        raise ValueError({"stage_a_prediction_row_count": len(decisions)})
    state_ids = [str(r["state_id"]) for r in decisions]
    if len(set(state_ids)) != 2862:
        raise ValueError("Stage-A predictions must contain 2862 unique state IDs")
    if set(state_ids) != set(states["state_id"].astype(str)):
        raise ValueError("Stage-A prediction state universe mismatch")
    if len({str(r["scenario_id"]) for r in decisions}) != 78:
        raise ValueError("Stage-A prediction scenario count mismatch")
    branches = set(
        zip(maps["state_id"].astype(str), maps["candidate_canonical_action_full"].astype(str))
    )
    for row in decisions:
        if int(row["override"]) not in (0, 1) or int(row["abstain"]) not in (0, 1):
            raise ValueError("invalid override/abstain indicator")
        if int(row["override"]) + int(row["abstain"]) != 1:
            raise ValueError("exactly one of override/abstain must be true")
        if int(row["override"]):
            key = (str(row["state_id"]), str(row["selected_canonical_action_full"]))
            if key not in branches:
                raise ValueError({"selected_action_not_in_support": str(row["state_id"])})
        else:
            if row["selected_canonical_action_id"] != "SBS_ABSTAIN":
                raise ValueError("abstention row must use SBS_ABSTAIN")


def write_predictions(path: Path, decisions: Sequence[Mapping[str, Any]], structural_hash: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing Stage-A predictions: {path}")
    rows = []
    for row in decisions:
        out = dict(row)
        out["structural_universe_sha256"] = structural_hash
        rows.append(out)
    fields = [
        "schema_version",
        "selector_name",
        "selector_family",
        "selector_weight_strategy",
        "gate",
        "tau",
        "margin",
        "state_id",
        "scenario_id",
        "step",
        "sbs_canonical_action_id",
        "sbs_canonical_action_full",
        "n_candidate_non_sbs_actions",
        "selected_canonical_action_id",
        "selected_canonical_action_full",
        "override",
        "abstain",
        "selected_predicted_advantage",
        "gate_score",
        "selector_json_sha256",
        "selector_joblib_sha256",
        "structural_universe_sha256",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def write_stage_a_freeze(
    args: argparse.Namespace,
    predictions_path: Path,
    structural_freeze_path: Path,
    decisions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    feature_hash = sha256_text(stable_json(json.loads(Path(args.selector_json).read_text())["feature_columns"]))
    obj = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": " ".join(sys.argv),
        "git": {
            "head": git(["rev-parse", "HEAD"]),
            "branch": git(["branch", "--show-current"]),
            "status_short": git(["status", "--short", "--branch"]),
        },
        "package_versions": package_versions(),
        "stage_a_source": "scripts/sbs_override_fresh_id_stage_a_predict_v2.py",
        "stage_a_source_sha256": sha256_file(ROOT / "scripts" / "sbs_override_fresh_id_stage_a_predict_v2.py"),
        "prediction_path": str(predictions_path),
        "prediction_sha256": sha256_file(predictions_path),
        "row_count": int(len(decisions)),
        "unique_state_count": int(len({str(r["state_id"]) for r in decisions})),
        "supported_scenarios": int(len({str(r["scenario_id"]) for r in decisions})),
        "override_count": int(sum(int(r["override"]) for r in decisions)),
        "abstention_count": int(sum(int(r["abstain"]) for r in decisions)),
        "selector_json_sha256": SELECTOR_JSON_HASH,
        "selector_joblib_sha256": SELECTOR_JOBLIB_HASH,
        "feature_list_sha256": feature_hash,
        "structural_universe_freeze": str(structural_freeze_path),
        "structural_universe_sha256": sha256_file(structural_freeze_path),
        "confirmatory_label_access": "NOT_ACCESSED",
        "fresh_confirmation_executed": False,
    }
    write_json(Path(args.stage_a_freeze), obj)
    return obj


def cmd_freeze_universe(args: argparse.Namespace) -> None:
    obj = write_structural_universe_freeze(args)
    print(stable_json(obj))


def cmd_execute(args: argparse.Namespace) -> None:
    structural = write_structural_universe_freeze(args)
    structural_path = Path(args.structural_freeze)
    structural_hash = sha256_file(structural_path)
    decisions, _structural = generate_predictions(args)
    write_predictions(Path(args.predictions), decisions, structural_hash)
    freeze = write_stage_a_freeze(args, Path(args.predictions), structural_path, decisions)
    print(stable_json(freeze))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-manifest", default=str(DEFAULT_STATE_MANIFEST))
    parser.add_argument("--policy-action-map", default=str(DEFAULT_POLICY_MAP))
    parser.add_argument("--scenario-manifest", default=str(DEFAULT_SCENARIO_MANIFEST))
    parser.add_argument("--selector-json", default=str(DEFAULT_SELECTOR_JSON))
    parser.add_argument("--selector-joblib", default=str(DEFAULT_SELECTOR_JOBLIB))
    parser.add_argument("--structural-freeze", default=str(DEFAULT_STRUCTURAL_FREEZE))
    parser.add_argument("--predictions", default=str(DEFAULT_PREDICTIONS))
    parser.add_argument("--stage-a-freeze", default=str(DEFAULT_STAGE_A_FREEZE))
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("freeze-universe").set_defaults(func=cmd_freeze_universe)
    sub.add_parser("execute").set_defaults(func=cmd_execute)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
