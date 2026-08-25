#!/usr/bin/env python3
"""Policy-blind public-trace stress calibration for external-baseline Pass 1.

Uses FIFO only. Selects smallest time-compression M from a frozen grid using
scheduler-agnostic pressure criteria. Never inspects ANWG winners or Pext
policies.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from llmserveopt.core.types import GPUConfig  # noqa: E402
from llmserveopt.policies.fifo import FIFOPolicy  # noqa: E402
from llmserveopt.policy_separation import public_trace_replay_v1 as ptr  # noqa: E402
from llmserveopt.simulator.service_model import ServiceModel  # noqa: E402
from llmserveopt.simulator.simulator import Simulator, SimulatorConfig  # noqa: E402
from llmserveopt.policies.base import BasePolicy  # noqa: E402

OUT = REPO / "experiments" / "external_baseline_comparison_v1"
PROTOCOL = OUT / "stress_protocol.json"
CAL_DIR = OUT / "stress_calibration"
LOG_DIR = OUT / "logs" / "stress_calibration"


class PressureProbePolicy(BasePolicy):
    """FIFO wrapper that records queue/active/KV pressure each step."""

    name = "fifo_pressure_probe"

    def __init__(self) -> None:
        self.inner = FIFOPolicy()
        self.samples: list[dict] = []

    def select_action(self, state):
        gpu = state.gpu_states[0]
        self.samples.append(
            {
                "step": int(state.step),
                "waiting": len(state.waiting_queue),
                "active": len(gpu.active_request_ids),
                "kv_util": float(gpu.current_kv_tokens) / max(gpu.max_kv_tokens, 1),
            }
        )
        return self.inner.select_action(state)


def load_protocol() -> dict:
    return json.loads(PROTOCOL.read_text())


def summarize_samples(samples: list[dict], completion_fraction: float) -> dict:
    if not samples:
        return {
            "p50_active": 0.0,
            "p99_active": 0.0,
            "max_active": 0.0,
            "frac_steps_queue_positive": 0.0,
            "max_kv_utilization": 0.0,
            "completion_fraction": float(completion_fraction),
            "unfinished_fraction": float(1.0 - completion_fraction),
        }
    act = np.array([s["active"] for s in samples], dtype=float)
    wait = np.array([s["waiting"] for s in samples], dtype=float)
    kv = np.array([s["kv_util"] for s in samples], dtype=float)
    return {
        "p50_active": float(np.quantile(act, 0.50)),
        "p99_active": float(np.quantile(act, 0.99)),
        "max_active": float(np.max(act)),
        "frac_steps_queue_positive": float(np.mean(wait > 0)),
        "max_kv_utilization": float(np.max(kv)),
        "completion_fraction": float(completion_fraction),
        "unfinished_fraction": float(1.0 - completion_fraction),
    }


def compress_scenario(scenario, M: float):
    """Return a shallow-copied scenario with arrivals compressed; SLO slack preserved."""
    from dataclasses import replace
    from llmserveopt.policy_separation.schema import PolicySeparationScenario

    # Preserve absolute slack under compression: deadline' = arrival' + original_slack
    reqs = []
    for r in scenario.requests:
        arr = float(r.arrival_time) / M
        slack = max(0.0, float(r.slo_deadline) - float(r.arrival_time))
        reqs.append(replace(r, arrival_time=arr, slo_deadline=arr + slack))
    params = dict(scenario.params)
    params["time_compression_M"] = float(M)
    return PolicySeparationScenario(
        scenario_id=f"{scenario.scenario_id}__M{int(M) if float(M).is_integer() else M}",
        family=scenario.family,
        template_name=scenario.template_name,
        generator_version=scenario.generator_version,
        seed=scenario.seed,
        params=params,
        requests=tuple(sorted(reqs, key=lambda x: (x.arrival_time, x.request_id))),
        gpu_configs=scenario.gpu_configs,
        service_model_kwargs=dict(scenario.service_model_kwargs),
        target_policy_family=scenario.target_policy_family,
        target_mechanism=scenario.target_mechanism,
        expected_qualitative_hypothesis=scenario.expected_qualitative_hypothesis,
        stress_control_relationship="stress",
    )


def run_one(scenario, M: float) -> dict:
    s = compress_scenario(scenario, M)
    probe = PressureProbePolicy()
    sim = Simulator(
        SimulatorConfig(
            gpu_configs=list(s.gpu_configs),
            service_model=ServiceModel(**s.service_model_kwargs),
            max_steps=200_000,
            drain_steps=50_000,
        )
    )
    sim.load_trace(list(s.requests))
    metrics = sim.run(probe, workload_tag=s.scenario_id, seed=s.seed)
    summary = summarize_samples(probe.samples, float(metrics.completion_fraction))
    summary.update(
        {
            "scenario_id": scenario.scenario_id,
            "M": M,
            "n_requests": len(scenario.requests),
            "anwg_fifo_not_used_for_selection": float(metrics.arrival_normalized_weighted_goodput),
        }
    )
    return summary


def meets_criterion(mean_row: dict, crit: dict) -> bool:
    ok_q = mean_row["frac_steps_queue_positive"] >= crit["mean_frac_steps_queue_positive_min"]
    p99_or = mean_row["p99_active"] >= crit["mean_p99_active_min_or_max_active"]["p99_active"]
    max_or = mean_row["max_active"] >= crit["mean_p99_active_min_or_max_active"]["max_active"]
    ok_act = p99_or or max_or
    ok_comp = mean_row["completion_fraction"] >= crit["mean_completion_fraction_min"]
    ok_kv = mean_row["max_kv_utilization"] < crit["mean_max_kv_utilization_max"]
    return bool(ok_q and ok_act and ok_comp and ok_kv)


def main() -> int:
    CAL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    proto = load_protocol()
    started = datetime.now(timezone.utc).isoformat()
    print(f"[stress_cal] start {started}", flush=True)

    # Build augmented scenarios once; filter calibration subset by window index.
    all_records = ptr.build_all_scenarios()
    aug = [r for r in all_records if r["scenario_evidence_class"] == ptr.AUGMENTED]
    k = int(proto["calibration"]["subset_windows_per_source"])
    selected = []
    per_source: dict[str, list] = {s: [] for s in ptr.SOURCES}
    for r in aug:
        src = r.get("source_dataset")
        scenario = r["scenario"]
        if src is None:
            for cand in ptr.SOURCES:
                if cand in scenario.scenario_id:
                    src = cand
                    break
        if src not in per_source:
            continue
        if len(per_source[src]) >= k:
            continue
        per_source[src].append(scenario)
        selected.append(scenario)
    print(f"[stress_cal] calibration scenarios={len(selected)} (target {k*len(ptr.SOURCES)})", flush=True)
    if len(selected) < k * len(ptr.SOURCES):
        selected = [r["scenario"] for r in aug[: k * len(ptr.SOURCES)]]
        print(f"[stress_cal] fallback selected={len(selected)}", flush=True)

    grid = list(proto["multiplier_grid"])
    rows = []
    for M in grid:
        t0 = time.time()
        print(f"[stress_cal] running M={M} ...", flush=True)
        m_rows = []
        for sc in selected:
            m_rows.append(run_one(sc, float(M)))
        mean_row = {
            "M": M,
            "frac_steps_queue_positive": float(np.mean([r["frac_steps_queue_positive"] for r in m_rows])),
            "p99_active": float(np.mean([r["p99_active"] for r in m_rows])),
            "max_active": float(np.mean([r["max_active"] for r in m_rows])),
            "completion_fraction": float(np.mean([r["completion_fraction"] for r in m_rows])),
            "max_kv_utilization": float(np.mean([r["max_kv_utilization"] for r in m_rows])),
            "n_windows": len(m_rows),
            "elapsed_s": time.time() - t0,
        }
        rows.append({"mean": mean_row, "per_window": m_rows})
        print(
            f"[stress_cal] M={M} queue+={mean_row['frac_steps_queue_positive']:.4f} "
            f"p99_active={mean_row['p99_active']:.2f} max_active={mean_row['max_active']:.2f} "
            f"comp={mean_row['completion_fraction']:.4f} kvmax={mean_row['max_kv_utilization']:.4f} "
            f"elapsed={mean_row['elapsed_s']:.1f}s",
            flush=True,
        )

    crit = proto["calibration"]["selection_criterion"]
    selected_M = None
    reason = None
    for block in rows:
        if meets_criterion(block["mean"], crit):
            selected_M = block["mean"]["M"]
            reason = "smallest_M_satisfying_all_predeclared_pressure_gates"
            break
    if selected_M is None:
        candidates = [b for b in rows if b["mean"]["completion_fraction"] >= crit["mean_completion_fraction_min"]]
        if candidates:
            best = max(candidates, key=lambda b: b["mean"]["frac_steps_queue_positive"])
            selected_M = best["mean"]["M"]
            reason = "fallback_max_queue_positive_among_noncatastrophic"
        else:
            reason = "CALIBRATION_FAILED"

    result = {
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "selected_primary_M": selected_M,
        "selected_reason": reason,
        "means": [b["mean"] for b in rows],
        "policy_blind": True,
        "probe_policy": "fifo",
    }
    out_path = CAL_DIR / "calibration_summary.json"
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    detail_path = CAL_DIR / "calibration_detail.json"
    detail_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")

    proto["status"] = "CALIBRATION_COMPLETE" if selected_M is not None else "CALIBRATION_FAILED"
    proto["selected_primary_M"] = selected_M
    proto["selected_reason"] = reason
    proto["calibration_result_path"] = str(out_path.relative_to(REPO))
    PROTOCOL.write_text(json.dumps(proto, indent=2, sort_keys=True) + "\n")
    print(f"[stress_cal] SELECTED M={selected_M} reason={reason}", flush=True)
    print(f"[stress_cal] wrote {out_path}", flush=True)
    return 0 if selected_M is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
