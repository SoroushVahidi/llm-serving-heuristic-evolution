"""Family C Reconstruction v1 -- a new, explicitly-versioned Step-2
evaluation layer for Family C / KV pressure scenarios.

Implements docs/design/FAMILY_C_RECONSTRUCTION_V1.md. NOT historical KV v2
replay -- exact historical replay is confirmed structurally impossible (see
docs/audits/family_c_step2_reconstruction_audit_20260817.md,
FAMILY_C_RECONSTRUCTION_BOUNDED). This module generates the 72 Family-C
scenarios ONCE in the current deterministic environment, serializes their
full request-level content to disk, and evaluates all 6 canonical anchors
by reading that frozen serialization back -- never by re-calling the
generator per policy. This guarantees every policy sees byte-identical
input for a given scenario and decouples evaluation from any further
BurstGPT access.

Data generation only. No selector, no composition, no mechanism attribution.
"""
from __future__ import annotations

import json
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ..core.types import GPUConfig, Request
from ..simulator.simulator import Simulator, SimulatorConfig
from ..simulator.service_model import ServiceModel
from . import unified_utility_matrix as uum

RECONSTRUCTION_VERSION = "CURRENT_RECONSTRUCTED_FAMILY_C_V1"
SOURCE_FAMILY = "FAMILY_C_KV_PRESSURE_V2"

RESULT_FIELDNAMES = [
    "reconstruction_scenario_id",
    "source_family",
    "reconstruction_version",
    "canonical_policy_id",
    "native_to_family_c",
    "degenerate_mechanism",
    "degenerate_reason",
    "status",
    "primary_utility_anwg",
    "secondary_completion_fraction",
    "secondary_unweighted_slo_success_rate",
    "error",
    "builder_version",
]


@dataclass(frozen=True)
class FrozenScenarioReplay:
    """Minimal replay object: everything run_cell-style evaluation needs,
    reconstructed purely from serialized JSONL -- no generator, no BurstGPT
    access."""
    scenario_id: str
    seed: int
    params: Dict[str, Any]
    requests: Tuple[Request, ...]
    gpu_configs: Tuple[GPUConfig, ...]
    service_model_kwargs: Dict[str, Any]


def regenerate_family_c_scenarios(config_path: Path | None = None) -> List[Any]:
    """Generate the 72 Family-C v2 scenarios ONCE, via the frozen,
    unmodified runner's own build_scenarios(). Each PolicySeparationScenario
    is a real object with real Request/GPUConfig tuples -- this is the only
    place BurstGPT is read in this module."""
    import yaml

    config_path = config_path or (uum.ROOT / "configs/kv_pressure_pilot_v2.yaml")
    mod = uum._load_runner_module("scripts/run_policy_separation_kv_pressure_pilot_v1.py")  # noqa: SLF001
    cfg = yaml.safe_load(open(config_path))
    return mod.build_scenarios(
        cfg, template_version="v2", allow_synthetic_tokens=False,
        datasets_root=uum.ROOT / ".local_data",
    )


def serialize_scenarios(scenarios: List[Any], out_path: Path) -> None:
    """Write full request-level content for every scenario, one JSON object
    per line. This is the frozen ground truth all 6 policies replay from."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for s in scenarios:
            row = {
                "scenario_id": s.scenario_id,
                "seed": s.seed,
                "params": s.params,
                "service_model_kwargs": s.service_model_kwargs,
                "gpu_configs": [asdict(g) for g in s.gpu_configs],
                "requests": [asdict(r) for r in s.requests],
            }
            f.write(json.dumps(row, sort_keys=True) + "\n")


def load_serialized_scenarios(path: Path) -> List[FrozenScenarioReplay]:
    """Reconstruct FrozenScenarioReplay objects purely from the serialized
    JSONL. Deliberately imports nothing from templates_kv_pressure_v2,
    templates_prefill_decode, or any BurstGPT resolution/loading code --
    verified by test_family_c_reconstruction_v1.py's
    test_load_serialized_scenarios_never_touches_burstgpt."""
    out: List[FrozenScenarioReplay] = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            out.append(FrozenScenarioReplay(
                scenario_id=row["scenario_id"],
                seed=row["seed"],
                params=row["params"],
                requests=tuple(Request(**r) for r in row["requests"]),
                gpu_configs=tuple(GPUConfig(**g) for g in row["gpu_configs"]),
                service_model_kwargs=row["service_model_kwargs"],
            ))
    return out


def run_cell_reconstruction(scenario: FrozenScenarioReplay, canonical_policy_id: str) -> Dict[str, Any]:
    """Evaluate one (reconstructed scenario, canonical anchor) cell. Never
    raises: failures are captured in the returned row's status/error
    fields. Reuses uum._build_policy for policy construction (the same,
    already-validated logic used for Family A/B) so the ServiceModel-merge
    semantics (and the degenerate_mechanism tagging for full_prefill/
    chunked_prefill_small) stay identical across every Step-2 layer."""
    reconstruction_scenario_id = f"{SOURCE_FAMILY}::{scenario.scenario_id}"
    native = canonical_policy_id in ("kv_constrained_online", "least_laxity_first")
    degenerate = canonical_policy_id in uum.DEGENERATE_MECHANISM_POLICIES  # any non-native family (C included)
    row: Dict[str, Any] = {
        "reconstruction_scenario_id": reconstruction_scenario_id,
        "source_family": SOURCE_FAMILY,
        "reconstruction_version": RECONSTRUCTION_VERSION,
        "canonical_policy_id": canonical_policy_id,
        "native_to_family_c": native,
        "degenerate_mechanism": degenerate,
        "degenerate_reason": uum.DEGENERATE_REASON if degenerate else "",
        "builder_version": uum.BUILDER_VERSION,
    }
    try:
        policy, sm_override = uum._build_policy(canonical_policy_id)  # noqa: SLF001
        merged_sm = dict(scenario.service_model_kwargs)
        merged_sm.update(sm_override)
        sim = Simulator(SimulatorConfig(
            gpu_configs=list(scenario.gpu_configs),
            service_model=ServiceModel(**merged_sm),
        ))
        sim.load_trace(list(scenario.requests))
        metrics = sim.run(policy, workload_tag=scenario.scenario_id, seed=scenario.seed)
        completed = list(sim._completed)  # noqa: SLF001
        n_req = len(scenario.requests)
        n_violated = sum(1 for c in completed if c.slo_violated)
        unweighted = (len(completed) - n_violated) / max(1, n_req)
        row.update({
            "status": "success",
            "primary_utility_anwg": float(metrics.arrival_normalized_weighted_goodput),
            "secondary_completion_fraction": float(metrics.completion_fraction),
            "secondary_unweighted_slo_success_rate": float(unweighted),
            "error": "",
        })
    except Exception as e:  # noqa: BLE001
        row.update({
            "status": "failed",
            "primary_utility_anwg": float("nan"),
            "secondary_completion_fraction": float("nan"),
            "secondary_unweighted_slo_success_rate": float("nan"),
            "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        })
    return row
