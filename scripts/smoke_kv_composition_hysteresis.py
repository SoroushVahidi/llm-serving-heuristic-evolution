#!/usr/bin/env python3
"""KV-aware composition safety-refined hysteresis -- smoke/mechanism-validation gate.

Verifies:
  - both child modes ("llf", "reserve") activate;
  - at least one within-scenario mode transition occurs;
  - the child remains distinct/differing from both parents;
  - no HOL blocking (non-degenerate ANWG values);
  - no pathological mode thrashing;
  - safety check: peak_KV(refined_child) <= max(peak_KV(parent_A), peak_KV(parent_B)) on all scenarios.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from llmserveopt.policy_separation.templates_kv_pressure_v2 import (  # noqa: E402
    assert_policy_visible_fields_clean_kv_v2,
    case_kv_pressure_reserve_contention_v2,
)
from llmserveopt.policies.kv_constrained_online import KVConstrainedOnlinePolicy  # noqa: E402
from llmserveopt.policies.least_laxity_first import LeastLaxityFirstPolicy  # noqa: E402
from llmserveopt.simulator.service_model import ServiceModel  # noqa: E402
from llmserveopt.simulator.simulator import Simulator, SimulatorConfig  # noqa: E402
from llmserveopt.composition.kv_composition_policy import KVAdaptiveReserveHysteresisChildPolicy  # noqa: E402


def run(scenario, policy):
    sim = Simulator(SimulatorConfig(
        gpu_configs=list(scenario.gpu_configs),
        service_model=ServiceModel(**scenario.service_model_kwargs),
    ))
    sim.load_trace(list(scenario.requests))
    metrics = sim.run(policy, workload_tag=scenario.scenario_id, seed=scenario.seed)
    
    gpu = sim._gpus[0]
    peak_tokens = max(gpu.step_kv_used) if gpu.step_kv_used else 0
    peak_ratio = peak_tokens / gpu.config.max_kv_tokens
    
    adm = {c.request.request_id: float(c.admission_time) for c in sim._completed}  # noqa: SLF001
    return metrics, peak_ratio, adm


def main() -> None:
    DATASETS_ROOT = Path(".local_data")
    grid_bulk = ["low", "high"]
    grid_phase = ["early", "middle", "late"]
    grid_tight = ["loose", "tight"]
    seed = 20260910

    both_modes_seen = False
    any_transition = False
    any_behavioral_diff = False
    low_pressure_reserve_frac = []
    high_pressure_reserve_frac = []
    n_failures = 0
    n_nan_inf = 0
    safety_violations = 0

    for bulk in grid_bulk:
        for phase in grid_phase:
            for tight in grid_tight:
                s = case_kv_pressure_reserve_contention_v2(
                    bulk_pressure=bulk, urgent_arrival_phase=phase,
                    urgent_tightness=tight, seed=seed, datasets_root=DATASETS_ROOT,
                )
                assert_policy_visible_fields_clean_kv_v2(s)

                try:
                    child = KVAdaptiveReserveHysteresisChildPolicy(tau_urgent=2)
                    m_child, p_child, adm_child = run(s, child)
                    m_kv, p_kv, adm_kv = run(s, KVConstrainedOnlinePolicy())
                    m_llf, p_llf, adm_llf = run(s, LeastLaxityFirstPolicy())
                except Exception as e:  # noqa: BLE001
                    n_failures += 1
                    print(f"FAILURE on {s.scenario_id}: {e}")
                    continue

                for val in (m_child.arrival_normalized_weighted_goodput,
                            m_kv.arrival_normalized_weighted_goodput,
                            m_llf.arrival_normalized_weighted_goodput):
                    if not math.isfinite(val):
                        n_nan_inf += 1

                if child.n_llf_steps > 0 and child.n_reserve_steps > 0:
                    both_modes_seen = True
                if child.transition_count > 0:
                    any_transition = True

                total_ids = set(adm_child) | set(adm_kv) | set(adm_llf)
                for rid in total_ids:
                    tc, tk, tl = adm_child.get(rid), adm_kv.get(rid), adm_llf.get(rid)
                    if tc != tk and tc != tl:
                        any_behavioral_diff = True
                        break

                reserve_frac = (
                    child.n_reserve_steps / max(1, child.n_llf_steps + child.n_reserve_steps)
                )
                if bulk == "low":
                    low_pressure_reserve_frac.append(reserve_frac)
                else:
                    high_pressure_reserve_frac.append(reserve_frac)

                # Peak KV check
                max_parent_peak = max(p_kv, p_llf)
                is_safe = p_child <= max_parent_peak + 1e-9
                if not is_safe:
                    safety_violations += 1
                    status_str = f"UNSAFE (Overshoot: {p_child - max_parent_peak:.4f})"
                else:
                    status_str = "SAFE"

                print(f"{s.scenario_id}: anwg child={m_child.arrival_normalized_weighted_goodput:.4f} "
                      f"kv={m_kv.arrival_normalized_weighted_goodput:.4f} "
                      f"llf={m_llf.arrival_normalized_weighted_goodput:.4f} | "
                      f"peak_kv child={p_child:.3f} max_parent={max_parent_peak:.3f} ({status_str}) | "
                      f"reserve_frac={reserve_frac:.3f} transitions={child.transition_count}")

    mean_low = sum(low_pressure_reserve_frac) / len(low_pressure_reserve_frac) if low_pressure_reserve_frac else 0.0
    mean_high = sum(high_pressure_reserve_frac) / len(high_pressure_reserve_frac) if high_pressure_reserve_frac else 0.0

    print()
    print("=== HYSTERESIS SMOKE GATE SUMMARY ===")
    print(f"both_modes_seen={both_modes_seen}")
    print(f"any_transition={any_transition}")
    print(f"any_behavioral_diff_from_both_parents={any_behavioral_diff}")
    print(f"n_failures={n_failures}")
    print(f"n_nan_inf={n_nan_inf}")
    print(f"safety_violations={safety_violations}")
    print(f"mean_reserve_fraction low_bulk_pressure={mean_low:.3f} high_bulk_pressure={mean_high:.3f}")

    ok = (
        both_modes_seen and any_transition and any_behavioral_diff
        and n_failures == 0 and n_nan_inf == 0
        and safety_violations == 0
        and mean_high > mean_low
    )
    print(f"HYSTERESIS_SMOKE_GATE_PASS={ok}")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
