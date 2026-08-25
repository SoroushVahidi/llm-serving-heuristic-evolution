#!/usr/bin/env python3
"""Build per-step online regime telemetry by replaying all 176 frozen
MF-PSD scenarios through the simulator with a single neutral policy (FIFO)
wrapped in `TelemetryRecordingPolicy`.

FEASIBILITY STUDY ONLY -- see
docs/audits/online_regime_signal_feasibility_v1_20260817.md. Does not
train a router or a family-specific selector. Does not modify any frozen
scenario, result, or prior audit.

A single policy (FIFO) is used uniformly across all three families so the
resulting telemetry characterizes workload-driven regime signal, not a
particular native policy's own admission dynamics. FIFO is a minimal,
policy-separation-neutral baseline already present in the codebase
(`llmserveopt.policies.fifo.FIFOPolicy`) -- not the native anchor policy of
any of the three families (which are ESTF/WFS, full_prefill/chunked, and
kv_constrained/least_laxity respectively), so it introduces no per-family
scheduling bias into the telemetry.

Requires `LLM_SERVEOPT_BURSTGPT_CSV` for the same deterministic Family A/B
replay already verified in the shared-feature-schema task (exact
scenario-ID match).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from llmserveopt.core.types import GPUConfig, Request  # noqa: E402
from llmserveopt.policies.fifo import FIFOPolicy  # noqa: E402
from llmserveopt.policy_separation.online_regime_signals_v1 import (  # noqa: E402
    SCHEMA_VERSION,
    TELEMETRY_COLUMNS,
    TelemetryRecordingPolicy,
)
from llmserveopt.policy_separation.templates_fairness_starvation_v2 import (  # noqa: E402
    case_fairness_vs_size_v2,
)
from llmserveopt.policy_separation.templates_prefill_decode_v2 import (  # noqa: E402
    case_prefill_decode_ttft_contention,
)
from llmserveopt.simulator.service_model import ServiceModel  # noqa: E402
from llmserveopt.simulator.simulator import Simulator, SimulatorConfig  # noqa: E402

MF_PSD_LONG = REPO_ROOT / "experiments" / "mf_psd_v1" / "mf_psd_long_v1.csv"
FAMILY_C_RECON_SCENARIOS = (
    REPO_ROOT / "experiments" / "family_c_reconstruction_v1" / "family_c_reconstruction_v1_scenarios.jsonl"
)

FAMILY_A = "FAMILY_A_FAIRNESS_STARVATION_V2"
FAMILY_B = "FAMILY_B_PREFILL_DECODE_V2"
FAMILY_C = "FAMILY_C_KV_PRESSURE_V2"


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_head_sha() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _load_mf_psd_scenario_rows() -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    with open(MF_PSD_LONG, newline="") as f:
        for row in csv.DictReader(f):
            out.setdefault(row["canonical_scenario_id"], []).append(row)
    return out


def _replay_family_a(feature_row: Dict[str, str]):
    kwargs = dict(
        target_utilization=float(feature_row["target_utilization"]),
        tenant_weight_skew=float(feature_row["tenant_weight_skew"]),
        favored_tenant_size=feature_row["favored_tenant_size"],
        prediction_noise_sigma=float(feature_row["prediction_noise_sigma"]),
        seed=int(feature_row["seed"]),
        max_active_sequences=int(feature_row["max_active_sequences"]),
    )
    scenario = case_fairness_vs_size_v2(**kwargs)
    return scenario.requests, scenario.gpu_configs, scenario.service_model_kwargs, scenario.scenario_id


def _replay_family_b(feature_row: Dict[str, str]):
    kwargs = dict(
        hog_count=feature_row["hog_count"],
        late_pressure=feature_row["late_pressure"],
        slo_emphasis=feature_row["slo_emphasis"],
        seed=int(feature_row["seed"]),
        n_hog=int(feature_row["n_hog"]),
        n_late=int(feature_row["n_late"]),
        max_active_sequences=int(feature_row["max_active_sequences"]),
        step_token_budget=int(feature_row["step_token_budget"]),
    )
    scenario = case_prefill_decode_ttft_contention(**kwargs)
    return scenario.requests, scenario.gpu_configs, scenario.service_model_kwargs, scenario.scenario_id


def _load_family_c_index() -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    with open(FAMILY_C_RECON_SCENARIOS) as f:
        for line in f:
            d = json.loads(line)
            out[d["scenario_id"]] = d
    return out


def build(output_dir: Path) -> Dict[str, Any]:
    if "LLM_SERVEOPT_BURSTGPT_CSV" not in os.environ:
        raise SystemExit("LLM_SERVEOPT_BURSTGPT_CSV must be set for Family A/B deterministic replay.")
    burstgpt_csv = Path(os.environ["LLM_SERVEOPT_BURSTGPT_CSV"])
    if not burstgpt_csv.is_file():
        raise SystemExit(f"LLM_SERVEOPT_BURSTGPT_CSV does not exist: {burstgpt_csv}")

    scenario_rows_by_id = _load_mf_psd_scenario_rows()
    family_c_index = _load_family_c_index()

    all_rows: List[Dict[str, Any]] = []
    replay_mismatches: List[str] = []
    n_steps_by_family: Dict[str, int] = {FAMILY_A: 0, FAMILY_B: 0, FAMILY_C: 0}
    n_scenarios_by_family: Dict[str, int] = {FAMILY_A: 0, FAMILY_B: 0, FAMILY_C: 0}

    for canonical_scenario_id, rows in sorted(scenario_rows_by_id.items()):
        first = rows[0]
        family = first["mechanism_family"]
        source_scenario_id = first["source_scenario_id"]
        feature_row = json.loads(first["source_scenario_features_json"])

        if family == FAMILY_A:
            requests, gpu_configs, smk, replayed_id = _replay_family_a(feature_row)
        elif family == FAMILY_B:
            requests, gpu_configs, smk, replayed_id = _replay_family_b(feature_row)
        elif family == FAMILY_C:
            d = family_c_index[source_scenario_id]
            requests = tuple(Request(**r) for r in d["requests"])
            gpu_configs = tuple(GPUConfig(**g) for g in d["gpu_configs"])
            smk = d.get("service_model_kwargs", {})
            replayed_id = d["scenario_id"]
        else:
            raise AssertionError(f"unknown mechanism_family {family!r}")

        if replayed_id != source_scenario_id:
            replay_mismatches.append(canonical_scenario_id)
            continue

        sim = Simulator(SimulatorConfig(gpu_configs=list(gpu_configs), service_model=ServiceModel(**smk)))
        sim.load_trace(list(requests))
        base_policy = FIFOPolicy()
        telemetry_policy = TelemetryRecordingPolicy(base_policy, canonical_scenario_id, family)
        sim.run(telemetry_policy, workload_tag=canonical_scenario_id, seed=0)

        for row in telemetry_policy.rows:
            all_rows.append(row.to_flat_dict())
        n_steps_by_family[family] += telemetry_policy.n_steps_observed
        n_scenarios_by_family[family] += 1

    if replay_mismatches:
        raise SystemExit(f"{len(replay_mismatches)} scenarios failed replay verification: {replay_mismatches[:5]}")

    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "online_regime_telemetry_v1.csv"
    with open(table_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(TELEMETRY_COLUMNS))
        writer.writeheader()
        for r in all_rows:
            writer.writerow(r)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "build_git_head_sha": _git_head_sha(),
        "build_timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "replay_policy": "fifo (neutral, non-native to every family)",
        "n_total_rows_recorded_after_dedup": len(all_rows),
        "n_scenarios_by_family": n_scenarios_by_family,
        "n_steps_actually_simulated_by_family": n_steps_by_family,
        "sampling_note": (
            "TelemetryRecordingPolicy records every activity-label "
            "TRANSITION exactly (any of a_active/b_active/c_active "
            "flipping), plus otherwise at least once every "
            "sample_stride_steps=20 raw simulator steps -- a recording-"
            "economy cadence, not a change to which steps the simulator "
            "actually executes or how signals/thresholds are computed. "
            "See TelemetryRecordingPolicy's docstring."
        ),
        "sources_read": {
            "mf_psd_long_v1.csv": {
                "path": str(MF_PSD_LONG.relative_to(REPO_ROOT)),
                "sha256": _sha256_of_file(MF_PSD_LONG),
            },
            "family_c_reconstruction_v1_scenarios.jsonl": {
                "path": str(FAMILY_C_RECON_SCENARIOS.relative_to(REPO_ROOT)),
                "sha256": _sha256_of_file(FAMILY_C_RECON_SCENARIOS),
            },
        },
        "output_files": {"online_regime_telemetry_v1.csv": _sha256_of_file(table_path)},
    }
    manifest_path = output_dir / "online_regime_telemetry_v1_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")

    print(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "experiments" / "online_regime_signal_feasibility_v1",
    )
    args = ap.parse_args()
    build(args.out_dir)


if __name__ == "__main__":
    main()
