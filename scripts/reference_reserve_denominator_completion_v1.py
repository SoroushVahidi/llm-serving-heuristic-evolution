#!/usr/bin/env python3
"""Post hoc denominator completion for reference_reserve_sensitivity_v1.

The executed sensitivity runner (scripts/reference_reserve_sensitivity_v1.py,
commit 8addcdb) counted disagreement states but never the total number of
reference decision states, so P(D), opportunity-weighted headroom, and the
beneficial decision rate were not produced.  This script supplies ONLY that
denominator, under the counting rule frozen in
experiments/reference_reserve_sensitivity_v1/DENOMINATOR_COMPLETION_V1.md.

Two subcommands:

  count   Replay the same 100 reserve-specific scenarios per reserve, count every
          invocation of the reference scheduler hook, and (as an integrity check)
          re-detect disagreement states.  No state is cloned, no alternative
          action is forced, and no continuation branch or latency counterfactual
          is run.
  derive  Merge the counts with the already completed (hash-verified)
          sensitivity results and the canonical 0.82 support artifacts, evaluate
          the integrity gates, and write the summary/regime/window/provenance
          files.

Nothing in the completed sensitivity results is recomputed or overwritten.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import math
import os
import platform
import socket
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

_ROOT0 = Path(os.environ.get("FRESH_EXECUTION_ROOT", str(Path(__file__).resolve().parents[1])))
for _extra in (_ROOT0, _ROOT0 / "src"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

# Reuse the executed runner's scenario construction, reference-policy construction and
# canonical decision semantics (importing it sets up logging and sys.path).
from scripts import reference_reserve_sensitivity_v1 as rrs

ROOT = rrs.ROOT
phase_a = rrs.phase_a
phase_d = rrs.phase_d
BasePolicy = rrs.BasePolicy
KVConstrainedOnlinePolicy = rrs.KVConstrainedOnlinePolicy
Simulator = rrs.Simulator
SimulatorConfig = rrs.SimulatorConfig
ServiceModel = rrs.ServiceModel
_build_policy = rrs._build_policy

SCHEMA_VERSION = "reference_reserve_denominator_completion_v1.0.0"
CLASSIFICATION = "POST_HOC_DENOMINATOR_COMPLETION"
PROTOCOL_SHA256 = "97b06cf1c96fe6b3f84aaf4355643192a0b400ce206f4c7c041b95d6bc417439"
RESERVES = (0.82, 0.90, 1.00)
RESERVE_DIRS = {0.82: "reserve_082", 0.90: "reserve_090", 1.00: "reserve_100"}
EXPECTED_TOTAL_DISAGREEMENTS = {0.82: 720, 0.90: 466, 1.00: 198}
EXPECTED_SCENARIOS = 100

EXPERIMENT_DIR = ROOT / "experiments" / "reference_reserve_sensitivity_v1"
PROTOCOL_PATH = EXPERIMENT_DIR / "PROTOCOL_V1.json"
AMENDMENT_PATH = EXPERIMENT_DIR / "DENOMINATOR_COMPLETION_V1.md"
DEFAULT_HASHES_PATH = EXPERIMENT_DIR / "denominator_completion_v1" / "EXISTING_SENSITIVITY_RESULT_HASHES_V1.json"
CANONICAL_DIR = ROOT / "experiments" / "fresh_production_latency_headroom_confirmatory_v1"

SCENARIO_KEY = ("source_dataset", "window_index", "axis", "condition_id")
COUNT_FIELDS = [
    "reserve", "source_dataset", "window_index", "axis", "condition_id", "scenario_id",
    "decision_states", "disagreement_states", "sim_step_calls_check", "wall_s",
]


def reserve_dir_name(reserve: float) -> str:
    return RESERVE_DIRS[round(float(reserve), 2)]


def stable_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, separators=(",", ": ")) + "\n"


def write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fields})


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def git_head() -> str | None:
    return rrs.git(["rev-parse", "HEAD"])


def git_dirty_tracked() -> list[str]:
    out = rrs.git(["status", "--porcelain", "--untracked-files=no"])
    return [line for line in (out or "").splitlines() if line.strip()]


# --------------------------------------------------------------------------------------
# Counting observer
# --------------------------------------------------------------------------------------
class DenominatorObserver(BasePolicy):
    """Reference-trajectory observer that counts scheduler decision hooks.

    The detection logic (steps 1-2 and the disagreement test) is the same as
    ``rrs.DynamicObserver.select_action``; the branching step (clone, force one
    alternative, continuation) is deliberately absent.  ``decision_states`` is
    incremented once per invocation of ``select_action``, which the simulator
    performs exactly once per stepped scheduling opportunity -- the same unit the
    original support scan reported as ``sbs_decision_states``.
    """

    name = "denominator_observer"

    def __init__(self, sbs_policy, portfolio_policies):
        self.sbs = sbs_policy
        self.policies = portfolio_policies
        self.decision_states = 0
        self.disagreement_states = 0

    def reset(self):
        if hasattr(self.sbs, "reset"):
            self.sbs.reset()
        for p in self.policies.values():
            if hasattr(p, "reset"):
                p.reset()

    def select_action(self, state):
        self.decision_states += 1

        # 1. Reference (SBS) action, exactly as in DynamicObserver
        sbs_action = self.sbs.select_action(copy.deepcopy(state))
        sbs_canon = phase_a.canonical_action(sbs_action)
        sbs_hash = phase_d.canonical_action_id(sbs_canon)

        # 2. Query the other portfolio policies, exactly as in DynamicObserver
        canon_actions = {}
        for pid, policy in self.policies.items():
            if pid == "kv_constrained_online":
                continue
            if hasattr(policy, "reset"):
                policy.reset()
            act = policy.select_action(copy.deepcopy(state))
            canon = phase_a.canonical_action(act)
            canon_hash = phase_d.canonical_action_id(canon)
            if canon_hash != sbs_hash:
                canon_actions[canon_hash] = canon

        # 3. Integrity-only disagreement tally (no cloning, no forced action)
        if canon_actions:
            self.disagreement_states += 1

        # Continue the main simulation on the reference trajectory
        return sbs_action


def build_portfolio(reserve: float):
    """Reference policy with the reserve, plus the portfolio (same as run_experiment)."""
    sbs_policy = KVConstrainedOnlinePolicy(target_kv_utilization=reserve)
    portfolio_policies = {
        "full_prefill": _build_policy("full_prefill")[0],
        "chunked_prefill_small": _build_policy("chunked_prefill_small")[0],
        "estimated_service_time_first": _build_policy("estimated_service_time_first")[0],
        "weighted_fair_share": _build_policy("weighted_fair_share")[0],
        "least_laxity_first": _build_policy("least_laxity_first")[0],
        "kv_constrained_online": sbs_policy,
    }
    return sbs_policy, portfolio_policies


def build_simulator(scenario) -> Simulator:
    """Simulator configuration identical to rrs.run_experiment."""
    sim = Simulator(SimulatorConfig(
        gpu_configs=list(scenario.gpu_configs),
        service_model=ServiceModel(**dict(scenario.service_model_kwargs)),
        max_steps=None, drain_steps=rrs.phase_b.DRAIN_STEPS, warn_on_invalid_action=True,
    ))
    sim.load_trace(list(scenario.requests))
    return sim


def count_scenario(scenario, reserve: float) -> dict[str, Any]:
    """Replay one reserve-specific reference trajectory and count decision states."""
    sim = build_simulator(scenario)
    sbs_policy, portfolio_policies = build_portfolio(reserve)
    observer = DenominatorObserver(sbs_policy, portfolio_policies)
    t0 = time.time()
    sim.run(observer, workload_tag=scenario.scenario_id, seed=int(scenario.seed))
    return {
        "decision_states": observer.decision_states,
        "disagreement_states": observer.disagreement_states,
        # Independent invariant: the simulator appends one queue-history entry per policy call.
        "sim_step_calls_check": len(sim._waiting_queue_history),
        "wall_s": round(time.time() - t0, 3),
    }


def scenario_grid() -> list[tuple[str, int, str, str]]:
    """The frozen 100 scenarios, in the executed runner's order."""
    design = CANONICAL_DIR
    selected_regimes = json.loads((design / "FRESH_REGIME_SELECTION_V1.json").read_text())["selected_regimes"]
    universe = json.loads((design / "FRESH_WINDOW_UNIVERSE_V1.json").read_text())["workloads"]
    grid = []
    for regime in selected_regimes:
        source = regime["source_dataset"]
        for window_index in universe[source]["selected_window_indices"]:
            grid.append((source, int(window_index), regime["axis"], regime["condition_id"]))
    return grid


# --------------------------------------------------------------------------------------
# Existing-result verification and expectations
# --------------------------------------------------------------------------------------
def verify_existing_results(results_dir: Path, frozen_hashes: Mapping[str, str]) -> list[str]:
    """Return a list of problems (empty when every frozen file hash matches)."""
    problems = []
    for rel, expected in sorted(frozen_hashes.items()):
        path = results_dir / rel
        if not path.exists():
            problems.append(f"missing:{rel}")
        elif rrs.sha256_file(path) != expected:
            problems.append(f"hash_mismatch:{rel}")
    return problems


def load_frozen_hashes(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text())
    return dict(data["sha256_by_relative_path"])


def expected_disagreements_by_scenario(results_dir: Path, reserve: float) -> dict[tuple, int]:
    rows = read_csv(results_dir / reserve_dir_name(reserve) / "disagreement_states.csv")
    counts: dict[tuple, int] = defaultdict(int)
    for r in rows:
        counts[(r["source_dataset"], int(r["window_index"]), r["axis"], r["condition_id"])] += 1
    return dict(counts)


def evaluate_disagreement_gate(count_rows: Sequence[Mapping[str, Any]], expected_by_scenario: Mapping[tuple, int],
                               expected_total: int) -> dict[str, Any]:
    """The recomputed disagreement counts must reproduce the completed run per scenario and in total."""
    total = sum(int(r["disagreement_states"]) for r in count_rows)
    mismatches = []
    for r in count_rows:
        key = (r["source_dataset"], int(r["window_index"]), r["axis"], r["condition_id"])
        exp = expected_by_scenario.get(key, 0)
        if int(r["disagreement_states"]) != exp:
            mismatches.append({"scenario": list(key), "counted": int(r["disagreement_states"]), "expected": exp})
    step_mismatch = [r["scenario_id"] for r in count_rows
                     if int(r["decision_states"]) != int(r["sim_step_calls_check"])]
    passed = (total == expected_total and not mismatches and not step_mismatch
              and len(count_rows) == EXPECTED_SCENARIOS)
    return {
        "status": "PASS" if passed else "FAIL",
        "total_counted": total,
        "total_expected": expected_total,
        "scenarios": len(count_rows),
        "per_scenario_mismatches": mismatches[:20],
        "n_per_scenario_mismatches": len(mismatches),
        "decision_vs_step_call_mismatches": step_mismatch[:20],
    }


# --------------------------------------------------------------------------------------
# count subcommand
# --------------------------------------------------------------------------------------
def cmd_count(args) -> int:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(args.existing_results_dir)
    frozen = load_frozen_hashes(Path(args.frozen_hashes))

    if rrs.sha256_file(PROTOCOL_PATH) != PROTOCOL_SHA256:
        logging.error("PROTOCOL_V1.json hash does not match the frozen protocol hash; aborting.")
        return 3
    problems = verify_existing_results(results_dir, frozen)
    if problems:
        logging.error("Existing sensitivity results failed hash verification: %s", problems[:5])
        return 3
    logging.info("Existing sensitivity result hashes verified (%d files).", len(frozen))

    grid = scenario_grid()
    if len(grid) != EXPECTED_SCENARIOS:
        logging.error("Scenario grid has %d scenarios, expected %d.", len(grid), EXPECTED_SCENARIOS)
        return 3

    started = time.time()
    gates: dict[str, Any] = {}
    reserve_timing: dict[str, float] = {}
    overall_ok = True
    for reserve in args.reserves:
        logging.info("=== Denominator pass for reserve %.2f (%d scenarios) ===", reserve, len(grid))
        t_res = time.time()
        rows = []
        for i, key in enumerate(grid, start=1):
            source, window_index, axis, condition_id = key
            _rec, scenario = rrs.scenario_for_key(key)
            counted = count_scenario(scenario, reserve)
            rows.append({
                "reserve": reserve, "source_dataset": source, "window_index": window_index,
                "axis": axis, "condition_id": condition_id, "scenario_id": scenario.scenario_id, **counted,
            })
            logging.info("reserve %.2f scenario %d/%d %s decisions=%d disagreements=%d",
                         reserve, i, len(grid), key, counted["decision_states"], counted["disagreement_states"])
        write_csv(out_dir / f"counts_{reserve_dir_name(reserve)}.csv", COUNT_FIELDS, rows)
        gate = evaluate_disagreement_gate(rows, expected_disagreements_by_scenario(results_dir, reserve),
                                          EXPECTED_TOTAL_DISAGREEMENTS[round(reserve, 2)])
        gates[f"{reserve:.2f}"] = gate
        reserve_timing[f"{reserve:.2f}"] = round(time.time() - t_res, 1)
        logging.info("DENOMINATOR_TRAJECTORY_INTEGRITY_GATE reserve %.2f: %s (counted %d, expected %d)",
                     reserve, gate["status"], gate["total_counted"], gate["total_expected"])
        overall_ok &= gate["status"] == "PASS"

    provenance = {
        "schema_version": SCHEMA_VERSION,
        "stage": "count",
        "classification": CLASSIFICATION,
        "reserves": [float(r) for r in args.reserves],
        "git_head": git_head(),
        "git_dirty_tracked_files": git_dirty_tracked(),
        "protocol_sha256": rrs.sha256_file(PROTOCOL_PATH),
        "amendment_sha256": rrs.sha256_file(AMENDMENT_PATH) if AMENDMENT_PATH.exists() else None,
        "runner_sha256": rrs.sha256_file(Path(__file__).resolve()),
        "sensitivity_runner_sha256": rrs.sha256_file(ROOT / "scripts" / "reference_reserve_sensitivity_v1.py"),
        "existing_results_dir": str(results_dir),
        "existing_results_verified_files": len(frozen),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "started_epoch": started,
        "duration_s": round(time.time() - started, 1),
        "duration_by_reserve_s": reserve_timing,
        "disagreement_gate_by_reserve": gates,
        "no_counterfactual_execution": True,
    }
    (out_dir / "count_execution_provenance.json").write_text(stable_json(provenance))
    logging.info("Denominator count stage finished; gates ok=%s", overall_ok)
    return 0 if overall_ok else 2


# --------------------------------------------------------------------------------------
# derive subcommand (pure, mechanical)
# --------------------------------------------------------------------------------------
def read_existing_results(results_dir: Path, reserve: float) -> dict[str, Any]:
    d = results_dir / reserve_dir_name(reserve)
    summary = json.loads((d / "result_summary.json").read_text())
    states = read_csv(d / "disagreement_states.csv")
    headroom = [float(r["oracle_headroom"]) for r in states]
    beneficial = sum(int(r["beneficial_opportunity"]) for r in states)
    by_regime: dict[tuple, dict[str, Any]] = {}
    for r in states:
        key = (r["source_dataset"], r["axis"], r["condition_id"])
        g = by_regime.setdefault(key, {"states": 0, "beneficial": 0, "headroom": []})
        g["states"] += 1
        g["beneficial"] += int(r["beneficial_opportunity"])
        g["headroom"].append(float(r["oracle_headroom"]))
    return {
        "summary": summary,
        "states": len(states),
        "beneficial_states": beneficial,
        "mass_sim_ms": math.fsum(headroom) * 1000.0,
        "by_regime": by_regime,
    }


def evaluate_canonical_gate(counts_082: Sequence[Mapping[str, Any]], canonical_dir: Path) -> dict[str, Any]:
    """Compare reserve-0.82 denominators with the exact canonical support/regime artifacts."""
    support_path = canonical_dir / "FRESH_SUPPORT_WINDOW_CONDITIONS_V1.csv"
    regime_path = canonical_dir / "FRESH_LATENCY_WORKLOAD_REGIME_V1.csv"
    if not support_path.exists():
        return {"status": "NOT_AVAILABLE", "reason": f"missing {support_path.name}"}
    canon: dict[tuple, dict[str, int]] = {}
    for r in read_csv(support_path):
        key = (r["source_dataset"], int(r["window_index"]), r["axis"], r["condition_id"])
        canon[key] = {"decisions": int(r["sbs_decision_states"]),
                      "disagreements": int(r["true_canonical_disagreement_states"])}
    window_mismatch = []
    missing = []
    for r in counts_082:
        key = (r["source_dataset"], int(r["window_index"]), r["axis"], r["condition_id"])
        if key not in canon:
            missing.append(list(key))
            continue
        if canon[key]["decisions"] != int(r["decision_states"]) or canon[key]["disagreements"] != int(r["disagreement_states"]):
            window_mismatch.append({"scenario": list(key), "new_decisions": int(r["decision_states"]),
                                    "canonical_decisions": canon[key]["decisions"],
                                    "new_disagreements": int(r["disagreement_states"]),
                                    "canonical_disagreements": canon[key]["disagreements"]})
    # Per-regime totals and P_D against the canonical regime table
    regime_new: dict[tuple, list[int]] = defaultdict(lambda: [0, 0])
    for r in counts_082:
        g = regime_new[(r["source_dataset"], r["axis"], r["condition_id"])]
        g[0] += int(r["decision_states"])
        g[1] += int(r["disagreement_states"])
    regime_mismatch = []
    max_abs_pd_diff = 0.0
    if regime_path.exists():
        for r in read_csv(regime_path):
            key = (r["source_dataset"], r["axis"], r["condition_id"])
            if key not in regime_new:
                continue
            dec, dis = regime_new[key]
            pd_new = dis / dec if dec else 0.0
            diff = abs(pd_new - float(r["P_D"]))
            max_abs_pd_diff = max(max_abs_pd_diff, diff)
            if int(r["states"]) != dis or diff > 1e-12:
                regime_mismatch.append({"regime": list(key), "new_P_D": pd_new, "canonical_P_D": float(r["P_D"]),
                                        "new_states": dis, "canonical_states": int(r["states"])})
    total_new = sum(int(r["decision_states"]) for r in counts_082)
    total_canon = sum(canon[(r["source_dataset"], int(r["window_index"]), r["axis"], r["condition_id"])]["decisions"]
                      for r in counts_082
                      if (r["source_dataset"], int(r["window_index"]), r["axis"], r["condition_id"]) in canon)
    passed = not window_mismatch and not missing and not regime_mismatch and total_new == total_canon
    return {
        "status": "PASS" if passed else "FAIL",
        "scenarios_compared": len(counts_082) - len(missing),
        "missing_in_canonical": missing[:20],
        "window_level_mismatches": window_mismatch[:20],
        "n_window_level_mismatches": len(window_mismatch),
        "regime_level_mismatches": regime_mismatch,
        "total_decisions_new": total_new,
        "total_decisions_canonical": total_canon,
        "max_abs_P_D_difference": max_abs_pd_diff,
        "canonical_sources": [support_path.name, regime_path.name],
    }


def derive_metrics(decisions: int, disagreements: int, beneficial: int, mass_sim_ms: float) -> dict[str, float]:
    """The three mechanical derivations frozen in the amendment."""
    return {
        "P_D": disagreements / decisions,
        "opportunity_weighted_headroom_sim_ms_per_decision": mass_sim_ms / decisions,
        "beneficial_decision_rate": beneficial / decisions,
    }


def cmd_derive(args) -> int:
    counts_dir = Path(args.counts_dir)
    results_dir = Path(args.existing_results_dir)
    out_dir = Path(args.output_dir)
    canonical_dir = Path(args.canonical_dir)
    frozen = load_frozen_hashes(Path(args.frozen_hashes))

    problems = verify_existing_results(results_dir, frozen)
    if problems:
        logging.error("Existing sensitivity results failed hash verification: %s", problems[:5])
        return 3

    counts = {r: read_csv(counts_dir / f"counts_{reserve_dir_name(r)}.csv") for r in RESERVES}
    count_prov = json.loads((counts_dir / "count_execution_provenance.json").read_text())

    gates = {"disagreement_gate_by_reserve": {}, "DENOMINATOR_REPRODUCTION_GATE": None}
    for r in RESERVES:
        gates["disagreement_gate_by_reserve"][f"{r:.2f}"] = evaluate_disagreement_gate(
            counts[r], expected_disagreements_by_scenario(results_dir, r), EXPECTED_TOTAL_DISAGREEMENTS[round(r, 2)])
    gates["DENOMINATOR_TRAJECTORY_INTEGRITY_GATE"] = (
        "PASS" if all(g["status"] == "PASS" for g in gates["disagreement_gate_by_reserve"].values()) else "FAIL")
    gates["DENOMINATOR_REPRODUCTION_GATE"] = evaluate_canonical_gate(counts[0.82], canonical_dir)

    integrity_ok = gates["DENOMINATOR_TRAJECTORY_INTEGRITY_GATE"] == "PASS"
    repro_status = gates["DENOMINATOR_REPRODUCTION_GATE"]["status"]
    derive_allowed = integrity_ok and repro_status in ("PASS", "NOT_AVAILABLE")
    status = "DENOMINATOR_COMPLETION_SUCCESS" if derive_allowed else "DENOMINATOR_INTEGRITY_FAILED"

    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "classification": CLASSIFICATION,
        "status": status,
        "protocol_sha256": PROTOCOL_SHA256,
        "gates": gates,
        "derivations": {
            "P_D": "disagreement_states / total_reference_decision_states",
            "opportunity_weighted_headroom_sim_ms_per_decision": "total_headroom_mass_sim_ms / total_reference_decision_states",
            "beneficial_decision_rate": "beneficial_states / total_reference_decision_states",
            "total_headroom_mass_sim_ms": "math.fsum(oracle_headroom over disagreement_states.csv rows) * 1000",
        },
        "sources": {
            "decision_states": "denominator count stage (counts_reserve_XXX.csv)",
            "disagreement_states_beneficial_states_headroom": "completed sensitivity results (hash-verified)",
            "existing_result_sha256": frozen,
        },
        "reserves": {},
    }
    regime_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []

    if derive_allowed:
        for r in RESERVES:
            existing = read_existing_results(results_dir, r)
            decisions = sum(int(x["decision_states"]) for x in counts[r])
            disagreements = sum(int(x["disagreement_states"]) for x in counts[r])
            beneficial = existing["beneficial_states"]
            if beneficial != int(existing["summary"]["primary"]["beneficial_states"]) or existing["states"] != disagreements:
                raise RuntimeError(f"existing result files internally inconsistent for reserve {r}")
            mass = existing["mass_sim_ms"]
            m = derive_metrics(decisions, disagreements, beneficial, mass)
            per_regime = []
            regime_dec: dict[tuple, list[int]] = defaultdict(lambda: [0, 0])
            for x in counts[r]:
                g = regime_dec[(x["source_dataset"], x["axis"], x["condition_id"])]
                g[0] += int(x["decision_states"])
                g[1] += int(x["disagreement_states"])
            for key in sorted(regime_dec, key=lambda k: (k[0], k[1], k[2])):
                dec, dis = regime_dec[key]
                eg = existing["by_regime"].get(key, {"states": 0, "beneficial": 0, "headroom": []})
                mass_r = math.fsum(eg["headroom"]) * 1000.0
                row = {
                    "reserve": r, "source_dataset": key[0], "axis": key[1], "condition_id": key[2],
                    "decision_states": dec, "disagreement_states": dis, "P_D": dis / dec,
                    "beneficial_states": eg["beneficial"], "total_headroom_mass_sim_ms": mass_r,
                    "opportunity_weighted_headroom_sim_ms_per_decision": mass_r / dec,
                    "beneficial_decision_rate": eg["beneficial"] / dec,
                }
                regime_rows.append(row)
                per_regime.append({k: row[k] for k in ("source_dataset", "axis", "condition_id", "decision_states",
                                                       "disagreement_states", "P_D")})
            for x in counts[r]:
                window_rows.append({
                    "reserve": r, "source_dataset": x["source_dataset"], "window_index": int(x["window_index"]),
                    "axis": x["axis"], "condition_id": x["condition_id"],
                    "decision_states": int(x["decision_states"]), "disagreement_states": int(x["disagreement_states"]),
                    "P_D": int(x["disagreement_states"]) / int(x["decision_states"]),
                })
            summary["reserves"][f"{r:.2f}"] = {
                "total_reference_decision_states": decisions,
                "disagreement_states": disagreements,
                "P_D": m["P_D"],
                "P_D_percent": m["P_D"] * 100.0,
                "beneficial_states": beneficial,
                "P_B_given_D": existing["summary"]["primary"]["P_B_given_D"],
                "total_headroom_mass_sim_ms": mass,
                "opportunity_weighted_headroom_sim_ms_per_decision": m["opportunity_weighted_headroom_sim_ms_per_decision"],
                "beneficial_decision_rate": m["beneficial_decision_rate"],
                "beneficial_decision_rate_percent": m["beneficial_decision_rate"] * 100.0,
                "per_regime": per_regime,
            }
        ref = summary["reserves"]["0.82"]
        for key in ("0.90", "1.00"):
            cur = summary["reserves"][key]
            cur["relative_to_082"] = {
                "decision_states_ratio": cur["total_reference_decision_states"] / ref["total_reference_decision_states"],
                "P_D_ratio": cur["P_D"] / ref["P_D"],
                "opportunity_weighted_headroom_ratio": (cur["opportunity_weighted_headroom_sim_ms_per_decision"]
                                                        / ref["opportunity_weighted_headroom_sim_ms_per_decision"]),
                "beneficial_decision_rate_ratio": cur["beneficial_decision_rate"] / ref["beneficial_decision_rate"],
            }

    (out_dir / "DENOMINATOR_SUMMARY_V1.json").write_text(stable_json(summary))
    reg_fields = ["reserve", "source_dataset", "axis", "condition_id", "decision_states", "disagreement_states", "P_D",
                  "beneficial_states", "total_headroom_mass_sim_ms",
                  "opportunity_weighted_headroom_sim_ms_per_decision", "beneficial_decision_rate"]
    win_fields = ["reserve", "source_dataset", "window_index", "axis", "condition_id", "decision_states",
                  "disagreement_states", "P_D"]
    write_csv(out_dir / "DENOMINATOR_BY_REGIME_V1.csv", reg_fields, regime_rows)
    write_csv(out_dir / "DENOMINATOR_BY_WINDOW_V1.csv", win_fields, window_rows)

    provenance = {
        "schema_version": SCHEMA_VERSION,
        "classification": CLASSIFICATION,
        "count_stage": count_prov,
        "derive_stage": {
            "git_head": git_head(),
            "git_dirty_tracked_files": git_dirty_tracked(),
            "python": platform.python_version(),
            "runner_sha256": rrs.sha256_file(Path(__file__).resolve()),
            "protocol_sha256": rrs.sha256_file(PROTOCOL_PATH),
            "amendment_sha256": rrs.sha256_file(AMENDMENT_PATH) if AMENDMENT_PATH.exists() else None,
            "existing_result_sha256_verified": frozen,
            "count_file_sha256": {p.name: rrs.sha256_file(p) for p in sorted(counts_dir.glob("counts_*.csv"))},
            "output_sha256": {name: rrs.sha256_file(out_dir / name)
                              for name in ("DENOMINATOR_SUMMARY_V1.json", "DENOMINATOR_BY_REGIME_V1.csv",
                                           "DENOMINATOR_BY_WINDOW_V1.csv")},
        },
        "final_status": status,
    }
    (out_dir / "DENOMINATOR_EXECUTION_PROVENANCE_V1.json").write_text(stable_json(provenance))
    logging.info("Derive stage finished: %s", status)
    return 0 if derive_allowed else 2


# --------------------------------------------------------------------------------------
# non-scientific smoke test
# --------------------------------------------------------------------------------------
def run_smoke_test() -> int:
    """Synthetic, non-scientific check: counting, gating and derivation mechanics."""
    logging.info("Denominator-completion smoke test (synthetic, non-scientific)")
    m = derive_metrics(1000, 10, 5, 20.0)
    assert m["P_D"] == 0.01 and m["beneficial_decision_rate"] == 0.005
    assert m["opportunity_weighted_headroom_sim_ms_per_decision"] == 0.02
    rows = [{"source_dataset": "s", "window_index": 1, "axis": "a", "condition_id": "c", "scenario_id": "x",
             "decision_states": 5, "disagreement_states": 2, "sim_step_calls_check": 5, "wall_s": 0.0}]
    assert evaluate_disagreement_gate(rows, {("s", 1, "a", "c"): 2}, 2)["status"] == "FAIL"  # 1 scenario != 100
    tmp = Path("/tmp/test_reserve_denominator_smoke.json")
    tmp.write_text(stable_json({"status": "TEST_OK"}))
    assert json.loads(tmp.read_text())["status"] == "TEST_OK"
    tmp.unlink()
    logging.info("Smoke test passed.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_count = sub.add_parser("count", help="replay reference trajectories and count decision states")
    p_count.add_argument("--reserves", type=float, nargs="+", default=list(RESERVES), choices=list(RESERVES))
    p_count.add_argument("--existing-results-dir", required=True)
    p_count.add_argument("--frozen-hashes", default=str(DEFAULT_HASHES_PATH))
    p_count.add_argument("--output-dir", required=True)
    p_count.set_defaults(func=cmd_count)

    p_derive = sub.add_parser("derive", help="mechanically derive P(D) and unconditional metrics")
    p_derive.add_argument("--counts-dir", required=True)
    p_derive.add_argument("--existing-results-dir", required=True)
    p_derive.add_argument("--frozen-hashes", default=str(DEFAULT_HASHES_PATH))
    p_derive.add_argument("--canonical-dir", default=str(CANONICAL_DIR))
    p_derive.add_argument("--output-dir", required=True)
    p_derive.set_defaults(func=cmd_derive)

    p_smoke = sub.add_parser("smoke-test", help="synthetic non-scientific smoke test")
    p_smoke.set_defaults(func=lambda a: run_smoke_test())

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
