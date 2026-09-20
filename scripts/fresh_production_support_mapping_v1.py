#!/usr/bin/env python3
"""Fresh, latency-blind support mapping for the frozen confirmatory design.

This wrapper reuses the Phase-A scanner and Phase-B V2 pressure semantics on
the preregistered, non-overlapping windows.  It produces support outcomes and
an outcome-blind causal-universe manifest only; it never runs continuations.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "scripts", ROOT / "src"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from scripts import industry_realism_action_opportunity_phase_a_v1 as phase_a
from scripts import industry_realism_action_opportunity_phase_b_v2 as phase_b
from llmserveopt.policy_separation import public_trace_replay_v1 as ptr

DESIGN_DIR = ROOT / "experiments" / "fresh_production_latency_headroom_confirmatory_v1"
OUT_DIR = DESIGN_DIR
WINDOW_UNIVERSE = DESIGN_DIR / "FRESH_WINDOW_UNIVERSE_V1.json"
OVERLAP_AUDIT = DESIGN_DIR / "OVERLAP_AUDIT_V1.json"
SUPPORT_PROTOCOL = DESIGN_DIR / "SUPPORT_PROTOCOL_V1.json"
EXPECTED = 1080
SUSTAINED_MIN_WINDOWS = 2


def stable_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, separators=(",", ": "), default=_json_default) + "\n"


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(args: Sequence[str]) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def frozen_windows() -> dict[str, list[dict[str, Any]]]:
    universe = json.loads(WINDOW_UNIVERSE.read_text())
    result: dict[str, list[dict[str, Any]]] = {}
    for source, spec in universe["workloads"].items():
        result[source] = []
        for index, request_range, source_range in zip(
            spec["selected_window_indices"],
            spec["selected_row_ranges"],
            spec["selected_source_record_id_ranges"],
        ):
            result[source].append({
                "source_dataset": source,
                "window_index": int(index),
                "requests": 200,
                "row_range": list(request_range),
                "source_record_id_range": list(source_range),
            })
    return result


def fresh_records() -> list[dict[str, Any]]:
    windows = frozen_windows()
    records: list[dict[str, Any]] = []
    for source in phase_a.WORKLOADS:
        data = ptr.load_source_records(source)
        for spec in windows[source]:
            window = ptr.extract_window(data, spec["window_index"], ptr.WINDOW_SIZE)
            scenario, provenance = ptr.build_scenario_from_window(
                window,
                source=source,
                window_index=spec["window_index"],
                evidence_class=ptr.FAITHFUL,
            )
            records.append({
                "canonical_scenario_id": f"FRESH_LATENCY_SUPPORT::{source}::w{spec['window_index']}::faithful",
                "source_dataset": source,
                "window_index": spec["window_index"],
                "scenario_evidence_class": ptr.FAITHFUL,
                "scenario": scenario,
                "field_provenance": provenance,
                "source_record_id_range": spec["source_record_id_range"],
                "row_range": spec["row_range"],
            })
    if len(records) != 60:
        raise AssertionError(f"expected 60 fresh windows, got {len(records)}")
    return records


def verify_preflight() -> dict[str, Any]:
    frozen_preregistration = "b4e6c603d0e4d8bbd4ad814341be97f361864a0b"
    try:
        subprocess.check_call(
            ["git", "merge-base", "--is-ancestor", frozen_preregistration, "HEAD"],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("support mapping must descend from the frozen preregistration commit") from exc
    for path in (WINDOW_UNIVERSE, OVERLAP_AUDIT, SUPPORT_PROTOCOL):
        if not path.exists():
            raise FileNotFoundError(path)
    overlap = json.loads(OVERLAP_AUDIT.read_text())
    if int(overlap["global"]["identity_overlap_count"]) != 0:
        raise RuntimeError("fresh windows overlap a prior Phase-A/B/D request identity")
    if overlap["global"].get("fresh_windows_nonoverlapping") is not True:
        raise RuntimeError("frozen overlap proof is not affirmative")
    for name, expected in {
        "window_universe": "9e3b8cd3183afabd243e47c9eef3a863d64663ba9ce2689c0defea17a308228d",
        "overlap_audit": "1cdcca971d39ca3adc6599c8f0d79bfb899eea9c05846f45bee12a92a2c85881",
        "support_protocol": "86eba31e4de30753cee7cf51495301272860daadf12d783ee278b2d4e1506e2e",
    }.items():
        path = {"window_universe": WINDOW_UNIVERSE, "overlap_audit": OVERLAP_AUDIT, "support_protocol": SUPPORT_PROTOCOL}[name]
        if sha256_file(path) != expected:
            raise RuntimeError(f"frozen artifact hash mismatch: {path}")
    records = fresh_records()
    source_ids = {f"{r['source_dataset']}::{r['window_index']}" for r in records}
    if len(source_ids) != 60:
        raise AssertionError("fresh window identity duplication")
    return {
        "starting_commit": git(["rev-parse", "HEAD"]),
        "frozen_preregistration_commit": frozen_preregistration,
        "branch": git(["branch", "--show-current"]),
        "fresh_windows": len(records),
        "requests": sum(len(r["scenario"].requests) for r in records),
        "fresh_request_overlap_with_phase_a_b_d": 0,
        "latency_outcomes_used_for_regime_selection": False,
    }


def condition_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    conditions = phase_b.pressure_conditions()
    jobs = [{"record": record, "condition": condition} for record in records for condition in conditions]
    if len(jobs) != EXPECTED:
        raise AssertionError(f"expected {EXPECTED} support conditions, got {len(jobs)}")
    return jobs


def run_one(job: Mapping[str, Any]) -> dict[str, Any]:
    row = phase_b.run_one(job)
    rec = job["record"]
    row.update({
        "fresh_window_id": f"{rec['source_dataset']}::w{rec['window_index']}",
        "source_record_id_start": rec["source_record_id_range"][0],
        "source_record_id_end": rec["source_record_id_range"][1],
        "source_row_start": rec["row_range"][0],
        "source_row_end": rec["row_range"][1],
        "latency_outcomes_used_for_regime_selection": False,
    })
    return row


def execute_support(records: Sequence[Mapping[str, Any]], n_jobs: int) -> list[dict[str, Any]]:
    import concurrent.futures

    jobs = condition_records(records)
    if n_jobs <= 1:
        rows = [run_one(job) for job in jobs]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
            rows = list(executor.map(run_one, jobs, chunksize=1))
    rows.sort(key=lambda row: (row["source_dataset"], row["axis"], int(row["pressure_order"]), int(row["window_index"])))
    return rows


def aggregate(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(rows)
    output: list[dict[str, Any]] = []
    for (source, axis, condition_id), group in frame.groupby(["source_dataset", "axis", "condition_id"], sort=True):
        decisions = int(group["sbs_decision_states"].sum())
        disagreements = int(group["disagreement_states"].sum())
        output.append({
            "schema_version": "fresh_production_latency_headroom_confirmatory_v1.support_result.0.0",
            "source_dataset": source,
            "axis": axis,
            "condition_id": condition_id,
            "pressure_order": int(group["pressure_order"].iloc[0]),
            "axis_value": float(group["axis_value"].iloc[0]),
            "arrival_multiplier": float(group["arrival_multiplier"].iloc[0]),
            "max_active_sequences_param": int(group["max_active_sequences_param"].iloc[0]),
            "max_kv_tokens_param": int(group["max_kv_tokens_param"].iloc[0]),
            "windows": int(len(group)),
            "requests": int(group["requests"].sum()),
            "sbs_decision_states": decisions,
            "canonical_disagreement_states": disagreements,
            "disagreement_rate": disagreements / decisions if decisions else 0.0,
            "windows_with_disagreement": int(group["windows_with_disagreement"].sum()),
            "valid_window_count": int((~group["validity_class"].astype(str).str.startswith("INVALID")).sum()),
            "invalid_window_count": int(group["validity_class"].astype(str).str.startswith("INVALID").sum()),
            "validity_classes": ",".join(sorted(set(map(str, group["validity_class"])) )),
            "completion_fraction_mean": float(group["completion_fraction"].mean()),
            "unfinished_request_count": int(group["unfinished_request_count"].sum()),
            "max_active_sequences": int(group["max_active_sequences"].max()),
            "mean_active_sequences": float(group["mean_active_sequences"].mean()),
            "max_queue_length": int(group["max_queue_length"].max()),
            "mean_queue_length": float(group["mean_queue_length"].mean()),
            "max_kv_utilization": float(group["max_kv_utilization"].max()),
            "mean_kv_utilization": float(group["mean_kv_utilization"].mean()),
            "max_active_pressure": float(group["max_active_pressure"].max()),
            "mean_active_pressure": float(group["mean_active_pressure"].mean()),
            "max_kv_pressure": float(group["max_kv_pressure"].max()),
            "mean_kv_pressure": float(group["mean_kv_pressure"].mean()),
            "active_cap_binding_states": int(group["active_sequence_capacity_binding_states"].sum()),
            "kv_cap_binding_states": int(group["kv_capacity_binding_or_over_requested_states"].sum()),
            "near_kv_binding_states": int(group["kv_capacity_near_binding_states"].sum()),
            "token_budget_binding_states": int(group["token_budget_binding_proxy_states"].sum()),
            "no_choice_states": int(group["no_policy_choice_states"].sum()),
            "canonical_action_collapse_states": int(group["canonical_action_collapse_states"].sum()),
            "policy_ranking_difference_states": int(group["policy_ranking_proxy_difference_states"].sum()),
            "unique_non_sbs_canonical_actions": int(group["distinct_alternative_action_hashes"].sum()),
            "latency_outcomes_used_for_regime_selection": False,
        })
    return sorted(output, key=lambda row: (row["source_dataset"], row["axis"], row["pressure_order"]))


def transitions(aggregate_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(aggregate_rows)
    output: list[dict[str, Any]] = []
    for (source, axis), group in frame.groupby(["source_dataset", "axis"], sort=True):
        ordered = group.sort_values("pressure_order")
        onset = ordered[ordered["canonical_disagreement_states"] > 0]
        sustained = ordered[ordered["windows_with_disagreement"] >= SUSTAINED_MIN_WINDOWS]
        binding_col = "kv_cap_binding_states" if axis == "kv_capacity" else "active_cap_binding_states" if axis == "active_sequence_capacity" else "token_budget_binding_states"
        binding = ordered[ordered[binding_col] > 0]
        output.append({
            "source_dataset": source,
            "axis": axis,
            "first_binding_point": str(binding.iloc[0]["condition_id"]) if len(binding) else "NONE_OBSERVED",
            "first_disagreement_point": str(onset.iloc[0]["condition_id"]) if len(onset) else "NONE_OBSERVED",
            "sustained_disagreement_point": str(sustained.iloc[0]["condition_id"]) if len(sustained) else "NONE_OBSERVED",
            "sustained_rule": "windows_with_disagreement >= 2",
            "latency_outcomes_used_for_regime_selection": False,
        })
    return output


def select_regimes(aggregate_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(aggregate_rows)
    selected: list[dict[str, Any]] = []
    for (source, axis), group in frame.groupby(["source_dataset", "axis"], sort=True):
        ordered = group.sort_values("pressure_order")
        valid = ~ordered["validity_classes"].astype(str).str.startswith("INVALID")
        valid_rows = ordered[valid]
        roles: dict[str, list[str]] = {}
        onset = valid_rows[valid_rows["canonical_disagreement_states"] > 0]
        sustained = valid_rows[valid_rows["windows_with_disagreement"] >= SUSTAINED_MIN_WINDOWS]
        # Arrival and resource axes with no valid support remain structural
        # controls; strongest-valid is selected only after support exists.
        if not len(onset):
            continue
        strongest = valid_rows.tail(1)
        for role, subset in (("onset", onset), ("sustained", sustained), ("strongest-valid", strongest)):
            if len(subset):
                condition_id = str(subset.iloc[0]["condition_id"])
                roles.setdefault(condition_id, []).append(role)
        for condition_id, role_values in sorted(roles.items(), key=lambda item: int(ordered.loc[ordered["condition_id"] == item[0], "pressure_order"].iloc[0])):
            row = ordered[ordered["condition_id"] == condition_id].iloc[0].to_dict()
            selected.append({
                "source_dataset": source,
                "axis": axis,
                "condition_id": condition_id,
                "pressure_order": int(row["pressure_order"]),
                "roles": sorted(role_values),
                "canonical_disagreement_states": int(row["canonical_disagreement_states"]),
                "validity_classes": row["validity_classes"],
                "latency_outcomes_used_for_regime_selection": False,
            })
    return selected


def selected_state_manifest(records: Sequence[Mapping[str, Any]], selected: Sequence[Mapping[str, Any]], n_jobs: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_key = {(r["source_dataset"], int(r["window_index"])): r for r in records}
    selected_keys = {(r["source_dataset"], r["axis"], r["condition_id"]): r for r in selected}
    jobs = []
    for (source, axis, condition_id), selected_row in selected_keys.items():
        condition = next(c for c in phase_b.pressure_conditions() if c["axis"] == axis and c["condition_id"] == condition_id)
        for window in records:
            if window["source_dataset"] == source:
                jobs.append({"record": window, "condition": condition, "selected_row": selected_row})
    import concurrent.futures
    if n_jobs <= 1:
        observations = [_run_observation(job) for job in jobs]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
            observations = list(executor.map(_run_observation, jobs, chunksize=1))
    states: list[dict[str, Any]] = []
    branch_keys: set[tuple[str, str]] = set()
    branches: list[dict[str, Any]] = []
    for obs in observations:
        rec = obs["record"]
        selection = obs["selected_row"]
        for state in obs["state_rows"]:
            if not int(state["any_non_sbs_policy_differs"]):
                continue
            policy_rows = [row for row in obs["policy_rows"] if row["state_id"] == state["state_id"] and int(row["candidate_differs_from_sbs"])]
            alternatives = sorted({row["candidate_canonical_action_hash"] for row in policy_rows})
            state_key = (state["state_id"], rec["source_dataset"])
            states.append({
                "workload": rec["source_dataset"],
                "fresh_window_id": f"{rec['source_dataset']}::w{rec['window_index']}",
                "window_index": int(rec["window_index"]),
                "source_record_id_start": rec["source_record_id_range"][0],
                "source_record_id_end": rec["source_record_id_range"][1],
                "axis": selection["axis"],
                "condition_id": selection["condition_id"],
                "regime_roles": ",".join(selection["roles"]),
                "state_id": state["state_id"],
                "decision_step": int(state["step"]),
                "decision_time": float(state["sim_time"]),
                "sbs_canonical_action_hash": state["sbs_canonical_action_hash"],
                "unique_non_sbs_canonical_action_hashes": json.dumps(alternatives),
                "n_unique_non_sbs_actions": len(alternatives),
                "p6_policies_with_non_sbs_action": ",".join(sorted(row["candidate_policy_id"] for row in policy_rows)),
                "active_pressure": float(state["active_sequences"] / max(1, int(obs["condition"]["max_active_sequences"]))),
                "kv_pressure": float(state["kv_utilization_max"]),
                "queue_pressure": int(state["waiting_queue_count"]),
                "active_binding": int(state["active_sequence_capacity_binding"]),
                "kv_binding": int(state["kv_capacity_binding_or_over_requested"]),
                "live_state_fingerprint": hashlib.sha256(stable_json(state).encode()).hexdigest(),
                "latency_outcomes_used_for_regime_selection": False,
            })
            for action_hash in alternatives:
                key = (state["state_id"], action_hash)
                if key in branch_keys:
                    continue
                branch_keys.add(key)
                branches.append({
                    "state_id": state["state_id"],
                    "workload": rec["source_dataset"],
                    "fresh_window_id": f"{rec['source_dataset']}::w{rec['window_index']}",
                    "window_index": int(rec["window_index"]),
                    "axis": selection["axis"],
                    "condition_id": selection["condition_id"],
                    "canonical_action_hash": action_hash,
                    "latency_outcomes_used_for_regime_selection": False,
                })
    states.sort(key=lambda row: (row["workload"], row["axis"], row["condition_id"], row["window_index"], row["decision_step"]))
    branches.sort(key=lambda row: (row["workload"], row["axis"], row["condition_id"], row["window_index"], row["state_id"], row["canonical_action_hash"]))
    return states, branches


def _run_observation(job: Mapping[str, Any]) -> dict[str, Any]:
    rec = job["record"]
    condition = job["condition"]
    scenario = phase_b.transformed_scenario(rec, condition)
    obs = phase_a.PhaseAScanPolicy(
        scenario_id=f"{rec['canonical_scenario_id']}::{condition['condition_id']}",
        source_dataset=rec["source_dataset"],
        window_index=int(rec["window_index"]),
    )
    from llmserveopt.simulator.service_model import ServiceModel
    from llmserveopt.simulator.simulator import Simulator, SimulatorConfig
    sim = Simulator(SimulatorConfig(
        gpu_configs=list(scenario.gpu_configs),
        service_model=ServiceModel(**dict(scenario.service_model_kwargs)),
        max_steps=None,
        drain_steps=phase_b.DRAIN_STEPS,
        warn_on_invalid_action=False,
    ))
    sim.load_trace(list(scenario.requests))
    metrics = sim.run(obs, workload_tag=scenario.scenario_id, seed=int(scenario.seed))
    phase_b.summarize_condition(obs, scenario, metrics, condition)
    return {"record": rec, "condition": condition, "selected_row": job["selected_row"], "state_rows": obs.state_rows, "policy_rows": obs.policy_rows}


def rebuild_selection_from_support(n_jobs: int) -> None:
    """Repair/finalize selection from an already complete support table.

    This path never reruns the 1,080 support conditions.  It exists so a
    support-only implementation correction cannot silently reinterpret a
    completed matrix or require latency data.
    """
    preflight = verify_preflight()
    records = fresh_records()
    support_path = OUT_DIR / "FRESH_SUPPORT_WINDOW_CONDITIONS_V1.csv"
    if not support_path.exists():
        raise FileNotFoundError(support_path)
    rows = list(csv.DictReader(support_path.open(newline="")))
    if len(rows) != EXPECTED:
        raise AssertionError(f"existing support table is incomplete: {len(rows)} != {EXPECTED}")
    numeric = {
        "pressure_order", "axis_value", "arrival_multiplier", "max_active_sequences_param",
        "max_kv_tokens_param", "sbs_decision_states", "disagreement_states",
        "windows_with_disagreement", "max_active_sequences", "mean_active_sequences",
        "max_queue_length", "mean_queue_length", "max_kv_utilization", "mean_kv_utilization",
        "max_active_pressure", "mean_active_pressure", "max_kv_pressure", "mean_kv_pressure",
        "active_sequence_capacity_binding_states", "kv_capacity_binding_or_over_requested_states",
        "kv_capacity_near_binding_states", "token_budget_binding_proxy_states",
        "no_policy_choice_states", "canonical_action_collapse_states",
        "policy_ranking_proxy_difference_states", "distinct_alternative_action_hashes",
        "completion_fraction", "unfinished_request_count", "requests",
    }
    for row in rows:
        for key in numeric:
            if key in row:
                row[key] = float(row[key]) if any(token in key for token in ("mean_", "max_", "pressure", "fraction", "axis_value", "multiplier")) else int(float(row[key]))
    aggregate_rows = aggregate(rows)
    selected = select_regimes(aggregate_rows)
    states, branches = selected_state_manifest(records, selected, max(1, n_jobs))
    result_path = OUT_DIR / "FRESH_SUPPORT_RESULT_V1.json"
    result = json.loads(result_path.read_text())
    result["status"] = "SUPPORT_COMPLETE_CAUSAL_UNIVERSE_FROZEN_NO_LATENCY_OUTCOMES"
    result["selected_regimes"] = selected
    result["causal_universe"] = {
        "disagreement_states": len(states),
        "unique_non_sbs_branches": len(branches),
        "sbs_reference_branches": len(states),
        "total_planned_continuations": len(states) + len(branches),
        "labeling_scope": "EXHAUSTIVE" if len(states) + len(branches) <= 50000 else "SAMPLED",
    }
    result["preflight"] = preflight
    result["selection_rebuilt_from_immutable_support_table"] = True
    result["latency_outcomes_used_for_regime_selection"] = False
    (OUT_DIR / "FRESH_SUPPORT_RESULT_V1.json").write_text(stable_json(result))
    write_csv(OUT_DIR / "FRESH_SUPPORT_WORKLOAD_AXIS_SUMMARY_V1.csv", aggregate_rows)
    write_csv(OUT_DIR / "FRESH_REGIME_SELECTION_V1.csv", selected)
    (OUT_DIR / "FRESH_REGIME_SELECTION_V1.json").write_text(stable_json({
        "status": "FROZEN_FROM_SUPPORT_ONLY",
        "selected_regimes": selected,
        "sustained_rule": "windows_with_disagreement >= 2",
        "latency_outcomes_used_for_regime_selection": False,
    }))
    write_csv(OUT_DIR / "FRESH_ELIGIBLE_DISAGREEMENT_STATES_V1.csv", states)
    write_csv(OUT_DIR / "FRESH_ELIGIBLE_DISAGREEMENT_BRANCHES_V1.csv", branches)
    (OUT_DIR / "FRESH_ELIGIBLE_DISAGREEMENT_UNIVERSE_V1.json").write_text(stable_json({
        "status": "OUTCOME_FREEZE_BEFORE_CAUSAL_LABELING",
        "disagreement_states": states,
        "unique_non_sbs_branches": branches,
        "latency_outcomes_used_for_regime_selection": False,
    }))
    (OUT_DIR / "FRESH_CAUSAL_LABELING_PLAN_V1.json").write_text(stable_json({
        "status": "FROZEN_NOT_EXECUTED",
        "population": "all canonical disagreement states in selected fresh regimes",
        "disagreement_states": len(states),
        "unique_non_sbs_branches": len(branches),
        "sbs_reference_branches": len(states),
        "total_planned_continuations": len(states) + len(branches),
        "labeling_scope": "EXHAUSTIVE" if len(states) + len(branches) <= 50000 else "SAMPLED",
        "sampling_outcome_blind": True,
        "latency_outcomes_used_for_regime_selection": False,
    }))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--rebuild-selection", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args(argv)
    if args.rebuild_selection:
        rebuild_selection_from_support(max(1, args.n_jobs))
        return 0
    if not args.execute:
        parser.error("support execution requires --execute or --rebuild-selection")
    preflight = verify_preflight()
    records = fresh_records()
    rows = execute_support(records, max(1, args.n_jobs))
    if len(rows) != EXPECTED:
        raise AssertionError(f"support completeness failure: {len(rows)} != {EXPECTED}")
    aggregate_rows = aggregate(rows)
    transition_rows = transitions(aggregate_rows)
    selected = select_regimes(aggregate_rows)
    states, branches = selected_state_manifest(records, selected, max(1, args.n_jobs))
    references = len(states)
    total_continuations = references + len(branches)
    output = {
        "schema_version": "fresh_production_latency_headroom_confirmatory_v1.support_result.0.0",
        "status": "SUPPORT_COMPLETE_CAUSAL_UNIVERSE_FROZEN_NO_LATENCY_OUTCOMES",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "expected_conditions": EXPECTED,
        "completed_conditions": len(rows),
        "failed_conditions": 0,
        "missing_conditions": 0,
        "duplicate_conditions": len(rows) - len({(r["source_dataset"], r["window_index"], r["axis"], r["condition_id"]) for r in rows}),
        "valid_conditions": sum(not str(r["validity_class"]).startswith("INVALID") for r in rows),
        "invalid_conditions": sum(str(r["validity_class"]).startswith("INVALID") for r in rows),
        "validity_class_counts": pd.Series([r["validity_class"] for r in rows]).value_counts().sort_index().to_dict(),
        "preflight": preflight,
        "transitions": transition_rows,
        "selected_regimes": selected,
        "causal_universe": {
            "disagreement_states": references,
            "unique_non_sbs_branches": len(branches),
            "sbs_reference_branches": references,
            "total_planned_continuations": total_continuations,
            "labeling_scope": "EXHAUSTIVE" if total_continuations <= 50000 else "SAMPLED",
        },
        "latency_outcomes_used_for_regime_selection": False,
        "phase_d_v1_post_hoc_latency_reanalysis_executed": False,
        "phase_d_v1_latency_signs_magnitudes_accessed_after_audit": False,
        "git": {"head": git(["rev-parse", "HEAD"]), "branch": git(["branch", "--show-current"])},
    }
    write_csv(OUT_DIR / "FRESH_SUPPORT_WINDOW_CONDITIONS_V1.csv", rows)
    write_csv(OUT_DIR / "FRESH_SUPPORT_WORKLOAD_AXIS_SUMMARY_V1.csv", aggregate_rows)
    write_csv(OUT_DIR / "FRESH_SUPPORT_TRANSITIONS_V1.csv", transition_rows)
    write_csv(OUT_DIR / "FRESH_ELIGIBLE_DISAGREEMENT_STATES_V1.csv", states)
    write_csv(OUT_DIR / "FRESH_ELIGIBLE_DISAGREEMENT_BRANCHES_V1.csv", branches)
    (OUT_DIR / "FRESH_SUPPORT_RESULT_V1.json").write_text(stable_json(output))
    (OUT_DIR / "FRESH_REGIME_SELECTION_V1.json").write_text(stable_json({
        "status": "FROZEN_FROM_SUPPORT_ONLY",
        "selected_regimes": selected,
        "sustained_rule": "windows_with_disagreement >= 2",
        "latency_outcomes_used_for_regime_selection": False,
    }))
    (OUT_DIR / "FRESH_ELIGIBLE_DISAGREEMENT_UNIVERSE_V1.json").write_text(stable_json({
        "status": "OUTCOME_FREEZE_BEFORE_CAUSAL_LABELING",
        "disagreement_states": states,
        "unique_non_sbs_branches": branches,
        "latency_outcomes_used_for_regime_selection": False,
    }))
    (OUT_DIR / "FRESH_CAUSAL_LABELING_PLAN_V1.json").write_text(stable_json({
        "status": "FROZEN_NOT_EXECUTED",
        "population": "all canonical disagreement states in selected fresh regimes",
        "disagreement_states": references,
        "unique_non_sbs_branches": len(branches),
        "sbs_reference_branches": references,
        "total_planned_continuations": total_continuations,
        "labeling_scope": "EXHAUSTIVE" if total_continuations <= 50000 else "SAMPLED",
        "sampling_outcome_blind": True,
        "latency_outcomes_used_for_regime_selection": False,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
