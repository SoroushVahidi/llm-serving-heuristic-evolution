"""Tests for the post hoc reference-reserve denominator-completion pass (synthetic, non-scientific).

None of these tests reads the scientific windows or reveals any scientific denominator.
"""
from __future__ import annotations

import csv
import inspect
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from scripts import reference_reserve_denominator_completion_v1 as dc  # noqa: E402
from scripts import reference_reserve_sensitivity_v1 as rrs  # noqa: E402
from llmserveopt.core.types import GPUConfig, Request  # noqa: E402
from llmserveopt.simulator.service_model import ServiceModel  # noqa: E402
from llmserveopt.simulator.simulator import Simulator, SimulatorConfig  # noqa: E402


def _requests(n=12, seed=0):
    rnd = random.Random(seed)
    reqs = []
    for i in range(n):
        out = rnd.choice([5, 30, 80])
        reqs.append(Request(
            request_id=i, arrival_time=0.001 * (i // 4), prompt_tokens=rnd.choice([20, 200, 600, 900]),
            predicted_output_tokens=out, actual_output_tokens=out, slo_deadline=1000.0, priority=1.0,
            class_id="default"))
    return reqs


def _sim(max_kv=1500):
    cfg = SimulatorConfig(
        gpu_configs=[GPUConfig(gpu_id=0, max_active_sequences=3, max_batch_tokens=512, max_kv_tokens=max_kv)],
        service_model=ServiceModel(), max_steps=None, drain_steps=2000, warn_on_invalid_action=True)
    sim = Simulator(cfg)
    sim.load_trace(_requests())
    return sim


# ---------------------------------------------------------------------------------------
# Frozen constants / files
# ---------------------------------------------------------------------------------------
def test_frozen_constants_match_repository_files():
    assert rrs.sha256_file(dc.PROTOCOL_PATH) == dc.PROTOCOL_SHA256
    assert dc.AMENDMENT_PATH.exists()
    frozen = dc.load_frozen_hashes(dc.DEFAULT_HASHES_PATH)
    assert len(frozen) == 21
    assert {p.split("/")[0] for p in frozen} == {"reserve_082", "reserve_090", "reserve_100"}
    assert dc.EXPECTED_TOTAL_DISAGREEMENTS == {0.82: 720, 0.90: 466, 1.00: 198}


def test_protocol_not_modified_by_this_pass():
    protocol = json.loads(dc.PROTOCOL_PATH.read_text())
    assert protocol["reserves"] == [0.82, 0.90, 1.00]
    assert "denominator" not in dc.PROTOCOL_PATH.read_text().lower()


# ---------------------------------------------------------------------------------------
# Decision counting
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("reserve", [0.82, 0.90, 1.00])
def test_decision_count_equals_simulator_policy_calls(reserve):
    sim = _sim()
    sbs, port = dc.build_portfolio(reserve)
    obs = dc.DenominatorObserver(sbs, port)
    sim.run(obs, workload_tag="synthetic", seed=0)
    # the simulator appends exactly one queue-history entry immediately before each policy call
    assert obs.decision_states == len(sim._waiting_queue_history) > 0
    assert 0 <= obs.disagreement_states <= obs.decision_states
    assert sbs.target_kv_utilization == reserve


def test_synthetic_disagreement_and_reserve_sensitivity_exist():
    """The synthetic workload must be able to exercise disagreement and reserve-dependent trajectories."""
    results = {}
    for reserve in (0.82, 1.00):
        sim = _sim()
        sbs, port = dc.build_portfolio(reserve)
        obs = dc.DenominatorObserver(sbs, port)
        sim.run(obs, workload_tag="synthetic", seed=0)
        results[reserve] = (obs.decision_states, obs.disagreement_states)
    assert results[0.82][1] > 0
    assert results[0.82] != results[1.00]


@pytest.mark.parametrize("reserve", [0.82, 0.90, 1.00])
def test_detection_matches_original_dynamic_observer(reserve, monkeypatch):
    """Disagreement detection and trajectory must equal rrs.DynamicObserver (branching stubbed out)."""
    monkeypatch.setattr(rrs, "run_branch_with_policy", lambda *a, **k: {
        "mean_latency": 0.0, "median_latency": 0.0, "p95_latency": 0.0, "population_count": 0,
        "completed_count": 0, "completed_ids_hash": "x", "population_ids_hash": "y", "unfinished_count": 0,
        "outside_population_completed_count": 0, "live_fingerprint_unchanged": True, "forced_once": True,
        "continuation_policy": "kv_constrained_online", "policy_forever_switch": False,
        "first_step_warning_count": 0, "first_step_warnings": ""})

    sim_a = _sim()
    sbs_a, port_a = dc.build_portfolio(reserve)
    dyn = rrs.DynamicObserver(sbs_a, port_a, "synthetic", 0, "cond", "axis", sim_a,
                              SimpleNamespace(requests=_requests()))
    calls = {"n": 0}
    orig = dyn.select_action

    def counting(state):
        calls["n"] += 1
        return orig(state)

    dyn.select_action = counting
    sim_a.run(dyn, workload_tag="synthetic", seed=0)

    sim_b = _sim()
    sbs_b, port_b = dc.build_portfolio(reserve)
    den = dc.DenominatorObserver(sbs_b, port_b)
    sim_b.run(den, workload_tag="synthetic", seed=0)

    assert den.decision_states == calls["n"]
    assert den.disagreement_states == dyn.states_found
    assert list(sim_b._waiting_queue_history) == list(sim_a._waiting_queue_history)


def test_count_scenario_returns_consistent_row():
    sim_scenario = SimpleNamespace(
        gpu_configs=[GPUConfig(gpu_id=0, max_active_sequences=3, max_batch_tokens=512, max_kv_tokens=1500)],
        service_model_kwargs={}, requests=_requests(), scenario_id="synthetic", seed=0)
    row = dc.count_scenario(sim_scenario, 0.82)
    assert row["decision_states"] == row["sim_step_calls_check"] > 0


def test_counting_code_contains_no_counterfactual_machinery():
    src = inspect.getsource(dc)
    for banned in ("fork_from_live_simulator", "run_branch_with_policy", "continue_run", "_state_fingerprint"):
        assert banned not in src, banned


def test_simulator_and_portfolio_construction_match_executed_runner():
    exec_src = inspect.getsource(rrs.run_experiment)
    mine = inspect.getsource(dc.build_simulator)
    for fragment in ("max_steps=None", "warn_on_invalid_action=True", "sim.load_trace(list(scenario.requests))"):
        assert fragment in exec_src and fragment in mine, fragment
    assert "phase_b.DRAIN_STEPS" in exec_src and "phase_b.DRAIN_STEPS" in mine
    exec_policies = ("full_prefill", "chunked_prefill_small", "estimated_service_time_first",
                     "weighted_fair_share", "least_laxity_first", "kv_constrained_online")
    _sbs, port = dc.build_portfolio(0.9)
    assert tuple(port) == exec_policies
    for name in exec_policies:
        assert f'"{name}"' in exec_src


def test_scenario_grid_shape_when_design_artifacts_present():
    grid = dc.scenario_grid()
    assert len(grid) == dc.EXPECTED_SCENARIOS
    assert len(set(grid)) == len(grid)


# ---------------------------------------------------------------------------------------
# Gates and mechanical derivations
# ---------------------------------------------------------------------------------------
def _count_row(source, window, axis, cond, dec, dis, reserve=0.82):
    return {"reserve": reserve, "source_dataset": source, "window_index": window, "axis": axis,
            "condition_id": cond, "scenario_id": f"{source}::{window}", "decision_states": dec,
            "disagreement_states": dis, "sim_step_calls_check": dec, "wall_s": 0.0}


def test_derive_metrics_are_mechanical():
    m = dc.derive_metrics(2000, 20, 15, 10.0)
    assert m["P_D"] == 0.01
    assert m["beneficial_decision_rate"] == 0.0075
    assert m["opportunity_weighted_headroom_sim_ms_per_decision"] == 0.005


def test_disagreement_gate_pass_and_fail(monkeypatch):
    monkeypatch.setattr(dc, "EXPECTED_SCENARIOS", 2)
    rows = [_count_row("s", 1, "a", "c", 100, 3), _count_row("s", 2, "a", "c", 200, 4)]
    exp = {("s", 1, "a", "c"): 3, ("s", 2, "a", "c"): 4}
    assert dc.evaluate_disagreement_gate(rows, exp, 7)["status"] == "PASS"
    assert dc.evaluate_disagreement_gate(rows, exp, 8)["status"] == "FAIL"
    assert dc.evaluate_disagreement_gate(rows, {("s", 1, "a", "c"): 3, ("s", 2, "a", "c"): 5}, 7)["status"] == "FAIL"
    bad = [dict(rows[0], sim_step_calls_check=99), rows[1]]
    assert dc.evaluate_disagreement_gate(bad, exp, 7)["status"] == "FAIL"


def test_verify_existing_results_detects_tampering(tmp_path):
    f = tmp_path / "reserve_082" / "x.json"
    f.parent.mkdir()
    f.write_text("{}")
    good = {"reserve_082/x.json": rrs.sha256_file(f)}
    assert dc.verify_existing_results(tmp_path, good) == []
    f.write_text("{ }")
    assert dc.verify_existing_results(tmp_path, good) == ["hash_mismatch:reserve_082/x.json"]
    assert dc.verify_existing_results(tmp_path, {"reserve_082/gone.json": "0"}) == ["missing:reserve_082/gone.json"]


def _write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _build_synthetic_world(tmp_path, monkeypatch, canonical_decisions_082=None, disagreement_override=None):
    """Two scenarios in one regime, three reserves; entirely synthetic."""
    monkeypatch.setattr(dc, "EXPECTED_SCENARIOS", 2)
    monkeypatch.setattr(dc, "EXPECTED_TOTAL_DISAGREEMENTS", {0.82: 5, 0.90: 3, 1.00: 1})
    plan = {  # reserve -> [(window, decisions, disagreements)]
        0.82: [(1, 1000, 3), (2, 3000, 2)],
        0.90: [(1, 900, 2), (2, 2900, 1)],
        1.00: [(1, 800, 1), (2, 2800, 0)],
    }
    results = tmp_path / "existing"
    counts = tmp_path / "counts"
    canon = tmp_path / "canonical"
    hashes = {}
    for r, items in plan.items():
        d = results / dc.reserve_dir_name(r)
        srows, beneficial = [], 0
        for w, _dec, dis in items:
            for k in range(dis):
                ben = int(k % 2 == 0)
                beneficial += ben
                srows.append({"source_dataset": "s", "window_index": w, "axis": "a", "condition_id": "c",
                              "oracle_headroom": 0.001 * (k + 1) * ben, "beneficial_opportunity": ben})
        _write_csv(d / "disagreement_states.csv",
                   ["source_dataset", "window_index", "axis", "condition_id", "oracle_headroom", "beneficial_opportunity"],
                   srows)
        total = len(srows)
        (d / "result_summary.json").write_text(json.dumps({
            "reserve": r, "primary": {"states": total, "beneficial_states": beneficial,
                                       "P_B_given_D": beneficial / total if total else 0.0}}))
        for name in ("disagreement_states.csv", "result_summary.json"):
            hashes[f"{dc.reserve_dir_name(r)}/{name}"] = rrs.sha256_file(d / name)
        rows = []
        for w, dec, dis in items:
            if disagreement_override and r == 0.82 and w == 1:
                dis = disagreement_override
            rows.append(_count_row("s", w, "a", "c", dec, dis, reserve=r))
        _write_csv(counts / f"counts_{dc.reserve_dir_name(r)}.csv", dc.COUNT_FIELDS, rows)
    (counts / "count_execution_provenance.json").write_text(json.dumps({"stage": "count", "slurm_job_id": "synthetic"}))
    frozen_path = tmp_path / "hashes.json"
    frozen_path.write_text(json.dumps({"sha256_by_relative_path": hashes}))
    dec_canon = canonical_decisions_082 or {1: 1000, 2: 3000}
    _write_csv(canon / "FRESH_SUPPORT_WINDOW_CONDITIONS_V1.csv",
               ["source_dataset", "window_index", "axis", "condition_id", "sbs_decision_states", "true_canonical_disagreement_states"],
               [{"source_dataset": "s", "window_index": 1, "axis": "a", "condition_id": "c",
                 "sbs_decision_states": dec_canon[1], "true_canonical_disagreement_states": 3},
                {"source_dataset": "s", "window_index": 2, "axis": "a", "condition_id": "c",
                 "sbs_decision_states": dec_canon[2], "true_canonical_disagreement_states": 2}])
    _write_csv(canon / "FRESH_LATENCY_WORKLOAD_REGIME_V1.csv",
               ["source_dataset", "axis", "condition_id", "states", "P_D"],
               [{"source_dataset": "s", "axis": "a", "condition_id": "c", "states": 5, "P_D": 5 / 4000}])
    return results, counts, canon, frozen_path


def _run_derive(results, counts, canon, frozen_path, out):
    args = SimpleNamespace(counts_dir=str(counts), existing_results_dir=str(results), output_dir=str(out),
                           canonical_dir=str(canon), frozen_hashes=str(frozen_path))
    return dc.cmd_derive(args)


def test_end_to_end_derive_success_and_serialization(tmp_path, monkeypatch):
    results, counts, canon, frozen_path = _build_synthetic_world(tmp_path, monkeypatch)
    out = tmp_path / "out"
    assert _run_derive(results, counts, canon, frozen_path, out) == 0
    summary = json.loads((out / "DENOMINATOR_SUMMARY_V1.json").read_text())
    assert summary["status"] == "DENOMINATOR_COMPLETION_SUCCESS"
    assert summary["classification"] == "POST_HOC_DENOMINATOR_COMPLETION"
    r082 = summary["reserves"]["0.82"]
    assert r082["total_reference_decision_states"] == 4000
    assert r082["disagreement_states"] == 5
    assert r082["P_D"] == 5 / 4000
    assert r082["beneficial_decision_rate"] == r082["beneficial_states"] / 4000
    assert r082["opportunity_weighted_headroom_sim_ms_per_decision"] == pytest.approx(r082["total_headroom_mass_sim_ms"] / 4000)
    assert summary["reserves"]["0.90"]["total_reference_decision_states"] == 3800
    assert summary["reserves"]["1.00"]["relative_to_082"]["decision_states_ratio"] == pytest.approx(3600 / 4000)
    assert summary["gates"]["DENOMINATOR_REPRODUCTION_GATE"]["status"] == "PASS"
    assert summary["gates"]["DENOMINATOR_TRAJECTORY_INTEGRITY_GATE"] == "PASS"
    for name in ("DENOMINATOR_SUMMARY_V1.json", "DENOMINATOR_BY_REGIME_V1.csv",
                 "DENOMINATOR_BY_WINDOW_V1.csv", "DENOMINATOR_EXECUTION_PROVENANCE_V1.json"):
        assert (out / name).exists()
    with (out / "DENOMINATOR_BY_REGIME_V1.csv").open() as f:
        regime = list(csv.DictReader(f))
    assert len(regime) == 3 and {"decision_states", "disagreement_states", "P_D"} <= set(regime[0])
    prov = json.loads((out / "DENOMINATOR_EXECUTION_PROVENANCE_V1.json").read_text())
    assert set(prov["derive_stage"]["output_sha256"]) == {
        "DENOMINATOR_SUMMARY_V1.json", "DENOMINATOR_BY_REGIME_V1.csv", "DENOMINATOR_BY_WINDOW_V1.csv"}
    assert prov["final_status"] == "DENOMINATOR_COMPLETION_SUCCESS"


def test_derive_refuses_when_canonical_denominators_do_not_match(tmp_path, monkeypatch):
    results, counts, canon, frozen_path = _build_synthetic_world(
        tmp_path, monkeypatch, canonical_decisions_082={1: 1001, 2: 3000})
    out = tmp_path / "out"
    assert _run_derive(results, counts, canon, frozen_path, out) == 2
    summary = json.loads((out / "DENOMINATOR_SUMMARY_V1.json").read_text())
    assert summary["status"] == "DENOMINATOR_INTEGRITY_FAILED"
    assert summary["gates"]["DENOMINATOR_REPRODUCTION_GATE"]["status"] == "FAIL"
    assert summary["reserves"] == {}


def test_derive_refuses_when_disagreement_counts_do_not_reproduce(tmp_path, monkeypatch):
    results, counts, canon, frozen_path = _build_synthetic_world(tmp_path, monkeypatch, disagreement_override=4)
    out = tmp_path / "out"
    assert _run_derive(results, counts, canon, frozen_path, out) == 2
    summary = json.loads((out / "DENOMINATOR_SUMMARY_V1.json").read_text())
    assert summary["gates"]["DENOMINATOR_TRAJECTORY_INTEGRITY_GATE"] == "FAIL"
    assert summary["reserves"] == {}


def test_derive_refuses_when_existing_results_are_altered(tmp_path, monkeypatch):
    results, counts, canon, frozen_path = _build_synthetic_world(tmp_path, monkeypatch)
    (results / "reserve_082" / "result_summary.json").write_text("{}")
    assert _run_derive(results, counts, canon, frozen_path, tmp_path / "out") == 3


def test_cli_smoke_test_returns_zero():
    assert dc.main(["smoke-test"]) == 0
