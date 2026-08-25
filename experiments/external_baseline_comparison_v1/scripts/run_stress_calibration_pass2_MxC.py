#!/usr/bin/env python3
"""Pass-2 policy-blind M×C public-trace stress calibration.

Predeclared BEFORE running (also in stress_protocol / docs):
  PRIMARY: choose lowest-stress (M,C) satisfying ALL of:
    (1) mean frac_steps_queue_positive >= 0.10
    (2) mean(max_active)/C >= 0.25  OR  mean(p99_active)/C >= 0.20
    (3) mean completion_fraction >= 0.80   # not catastrophic
    (4) calibration subset covers BurstGPT + Azure conv + Azure code
  Lowest-stress order: smaller M first, then larger C (less capacity cut).
  Probe policy: FIFO only. No ANWG/winner inspection for selection.
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
from llmserveopt.policies.base import BasePolicy  # noqa: E402
from llmserveopt.policies.fifo import FIFOPolicy  # noqa: E402
from llmserveopt.policy_separation import public_trace_replay_v1 as ptr  # noqa: E402
from llmserveopt.policy_separation.schema import PolicySeparationScenario  # noqa: E402
from llmserveopt.simulator.service_model import ServiceModel  # noqa: E402
from llmserveopt.simulator.simulator import Simulator, SimulatorConfig  # noqa: E402
from dataclasses import replace

OUT = REPO / "experiments" / "external_baseline_comparison_v1"
CAL_DIR = OUT / "stress_calibration" / "pass2_MxC"
LOG_DIR = OUT / "logs" / "stress_calibration_pass2"
PROTOCOL = OUT / "stress_protocol.json"

# Frozen grid (predeclared)
M_GRID = [8, 16, 32]
C_GRID = [256, 128, 64, 32]
COMPLETION_MIN = 0.80
QUEUE_POS_MIN = 0.10
ACTIVE_UTIL_MAX_MIN = 0.25
ACTIVE_UTIL_P99_MIN = 0.20
WINDOWS_PER_SOURCE = 4


class PressureProbePolicy(BasePolicy):
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


def summarize(samples, completion_fraction: float, C: int) -> dict:
    if not samples:
        return {
            "p50_active": 0.0,
            "p99_active": 0.0,
            "max_active": 0.0,
            "frac_steps_queue_positive": 0.0,
            "max_kv_utilization": 0.0,
            "completion_fraction": float(completion_fraction),
            "active_util_max": 0.0,
            "active_util_p99": 0.0,
        }
    act = np.array([s["active"] for s in samples], dtype=float)
    wait = np.array([s["waiting"] for s in samples], dtype=float)
    kv = np.array([s["kv_util"] for s in samples], dtype=float)
    p99 = float(np.quantile(act, 0.99))
    mx = float(np.max(act))
    return {
        "p50_active": float(np.quantile(act, 0.50)),
        "p99_active": p99,
        "max_active": mx,
        "frac_steps_queue_positive": float(np.mean(wait > 0)),
        "max_kv_utilization": float(np.max(kv)),
        "completion_fraction": float(completion_fraction),
        "active_util_max": mx / max(C, 1),
        "active_util_p99": p99 / max(C, 1),
    }


def transform_scenario(scenario, M: float, C: int) -> PolicySeparationScenario:
    reqs = []
    for r in scenario.requests:
        arr = float(r.arrival_time) / M
        slack = max(0.0, float(r.slo_deadline) - float(r.arrival_time))
        reqs.append(replace(r, arrival_time=arr, slo_deadline=arr + slack))
    # Capacity lever: existing GPUConfig fields only
    g0 = scenario.gpu_configs[0]
    gpu = GPUConfig(
        gpu_id=g0.gpu_id,
        max_active_sequences=int(C),
        max_batch_tokens=int(C),  # joint/public convention: count-cap field
        max_kv_tokens=int(g0.max_kv_tokens),
    )
    params = dict(scenario.params)
    params["time_compression_M"] = float(M)
    params["max_active_sequences_override"] = int(C)
    return PolicySeparationScenario(
        scenario_id=f"{scenario.scenario_id}__M{int(M)}_C{C}",
        family=scenario.family,
        template_name=scenario.template_name,
        generator_version=scenario.generator_version,
        seed=scenario.seed,
        params=params,
        requests=tuple(sorted(reqs, key=lambda x: (x.arrival_time, x.request_id))),
        gpu_configs=(gpu,),
        service_model_kwargs=dict(scenario.service_model_kwargs),
        target_policy_family=scenario.target_policy_family,
        target_mechanism=scenario.target_mechanism,
        expected_qualitative_hypothesis=scenario.expected_qualitative_hypothesis,
        stress_control_relationship="stress",
    )


def run_one(scenario, M: float, C: int) -> dict:
    s = transform_scenario(scenario, M, C)
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
    out = summarize(probe.samples, float(metrics.completion_fraction), C)
    out.update({"scenario_id": scenario.scenario_id, "M": M, "C": C, "n_requests": len(scenario.requests)})
    return out


def meets(mean_row: dict) -> bool:
    ok_q = mean_row["frac_steps_queue_positive"] >= QUEUE_POS_MIN
    ok_u = (mean_row["active_util_max"] >= ACTIVE_UTIL_MAX_MIN) or (
        mean_row["active_util_p99"] >= ACTIVE_UTIL_P99_MIN
    )
    ok_c = mean_row["completion_fraction"] >= COMPLETION_MIN
    return bool(ok_q and ok_u and ok_c)


def main() -> int:
    CAL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    print(f"[pass2_cal] start {started}", flush=True)
    print(
        f"[pass2_cal] PREDECLARED gates: queue+>={QUEUE_POS_MIN}, "
        f"util_max>={ACTIVE_UTIL_MAX_MIN}|util_p99>={ACTIVE_UTIL_P99_MIN}, "
        f"comp>={COMPLETION_MIN}; pick smallest M then largest C",
        flush=True,
    )

    all_records = ptr.build_all_scenarios()
    aug = [r for r in all_records if r["scenario_evidence_class"] == ptr.AUGMENTED]
    selected = []
    per_source: dict[str, list] = {s: [] for s in ptr.SOURCES}
    for r in aug:
        src = r["source_dataset"]
        if len(per_source[src]) >= WINDOWS_PER_SOURCE:
            continue
        per_source[src].append(r["scenario"])
        selected.append((src, r["scenario"]))
    sources_present = sorted({s for s, _ in selected})
    print(f"[pass2_cal] n_windows={len(selected)} sources={sources_present}", flush=True)
    assert set(sources_present) == set(ptr.SOURCES), "calibration must cover all three sources"

    # Lowest-stress order: M ascending, C descending
    grid = [(M, C) for M in M_GRID for C in sorted(C_GRID, reverse=True)]
    means = []
    details = []
    for M, C in grid:
        t0 = time.time()
        print(f"[pass2_cal] running M={M} C={C} ...", flush=True)
        rows = [run_one(sc, float(M), int(C)) for _, sc in selected]
        mean_row = {
            "M": M,
            "C": C,
            "frac_steps_queue_positive": float(np.mean([r["frac_steps_queue_positive"] for r in rows])),
            "p99_active": float(np.mean([r["p99_active"] for r in rows])),
            "max_active": float(np.mean([r["max_active"] for r in rows])),
            "completion_fraction": float(np.mean([r["completion_fraction"] for r in rows])),
            "max_kv_utilization": float(np.mean([r["max_kv_utilization"] for r in rows])),
            "active_util_max": float(np.mean([r["active_util_max"] for r in rows])),
            "active_util_p99": float(np.mean([r["active_util_p99"] for r in rows])),
            "n_windows": len(rows),
            "elapsed_s": time.time() - t0,
            "meets_all_gates": False,
        }
        mean_row["meets_all_gates"] = meets(mean_row)
        means.append(mean_row)
        details.append({"mean": mean_row, "per_window": rows})
        print(
            f"[pass2_cal] M={M} C={C} queue+={mean_row['frac_steps_queue_positive']:.4f} "
            f"util_max={mean_row['active_util_max']:.3f} util_p99={mean_row['active_util_p99']:.3f} "
            f"comp={mean_row['completion_fraction']:.4f} meet={mean_row['meets_all_gates']} "
            f"t={mean_row['elapsed_s']:.1f}s",
            flush=True,
        )

    selected_pt = None
    reason = None
    for mean_row in means:
        if mean_row["meets_all_gates"]:
            selected_pt = {"M": mean_row["M"], "C": mean_row["C"]}
            reason = "lowest_stress_M_then_largest_C_satisfying_all_predeclared_gates"
            break

    result = {
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "policy_blind": True,
        "probe_policy": "fifo",
        "predeclared_gates": {
            "queue_pos_min": QUEUE_POS_MIN,
            "active_util_max_min": ACTIVE_UTIL_MAX_MIN,
            "active_util_p99_min": ACTIVE_UTIL_P99_MIN,
            "completion_min": COMPLETION_MIN,
            "pick": "smallest_M_then_largest_C",
        },
        "grid": {"M": M_GRID, "C": C_GRID},
        "sources_present": sources_present,
        "means": means,
        "selected_primary": selected_pt,
        "selected_reason": reason or "NO_POINT_MEETS_ALL_GATES",
        "winner_blind_confirmation": "no ANWG/VTC/vLLM/P6 outcomes inspected for selection",
    }
    (CAL_DIR / "calibration_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (CAL_DIR / "calibration_detail.json").write_text(json.dumps(details, indent=2, sort_keys=True) + "\n")

    # Update protocol only if selected
    proto = json.loads(PROTOCOL.read_text())
    proto["pass2_MxC_calibration"] = {
        "result_path": str((CAL_DIR / "calibration_summary.json").relative_to(REPO)),
        "selected_primary": selected_pt,
        "selected_reason": result["selected_reason"],
        "gates": result["predeclared_gates"],
    }
    if selected_pt is not None:
        proto["status"] = "PUBLIC_TRACE_STRESS_V1_FROZEN"
        proto["workload_name"] = "public_trace_stress_v1"
        proto["selected_primary_M"] = selected_pt["M"]
        proto["selected_primary_C"] = selected_pt["C"]
        proto["selected_reason"] = reason
        proto["gpu_config"] = {
            "max_active_sequences": selected_pt["C"],
            "max_batch_tokens": selected_pt["C"],
            "max_kv_tokens": 8_000_000,
        }
        proto["capacity_lever"] = {
            "type": "max_active_sequences_and_max_batch_tokens",
            "value": selected_pt["C"],
            "preserves_request_contents": True,
        }
    else:
        proto["status"] = "PASS2_CALIBRATION_NO_FEASIBLE_POINT"
    PROTOCOL.write_text(json.dumps(proto, indent=2, sort_keys=True) + "\n")

    print(f"[pass2_cal] SELECTED {selected_pt} reason={result['selected_reason']}", flush=True)
    return 0 if selected_pt is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
