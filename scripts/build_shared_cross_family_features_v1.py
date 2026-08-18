#!/usr/bin/env python3
"""Build SHARED_CORE_V1: a deterministic, family-agnostic context-feature
table for all 176 MF-PSD v1 scenarios.

FEATURE-SCHEMA INVESTIGATION ARTIFACT -- see
docs/audits/shared_cross_family_feature_schema_feasibility_v1_20260817.md.
This is a NEW, separate feature table alongside (not replacing) the existing
`experiments/mf_psd_v1/` 33-column family-prefixed schema.

Sources of requests/gpu_configs per family (all frozen, already-committed
artifacts -- no policy is re-run, no workload is regenerated):

  * Family A / Family B: `experiments/mf_psd_v1/mf_psd_long_v1.csv` records
    each scenario's exact source scenario-feature row
    (`source_scenario_features_json`). This build calls the original
    deterministic template function (`case_fairness_vs_size_v2` /
    `case_prefill_decode_ttft_contention`) with the exact recorded
    (seed, params) to regenerate `requests`/`gpu_configs` byte-for-byte
    (verified per-scenario by an exact `scenario_id` match assertion --
    the scenario_id string is a hash-like encoding of every numeric input,
    so a match is strong evidence the replay used identical inputs).
  * Family C: `experiments/family_c_reconstruction_v1/` already stored full
    request-level payloads (`requests`, `gpu_configs`) verbatim -- no
    replay needed, direct deserialization only.

Requires `LLM_SERVEOPT_BURSTGPT_CSV` to point at a local staged BurstGPT CSV
(same dataset used to build `experiments/family_c_reconstruction_v1/`) for
the Family A/B replay step.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(REPO_ROOT / "src"))

from llmserveopt.core.types import GPUConfig, Request  # noqa: E402
from llmserveopt.policy_separation.shared_context_features_v1 import (  # noqa: E402
    SHARED_CORE_V1_FEATURES,
    SHARED_CORE_V1_VERSION,
    compute_shared_context_features,
)
from llmserveopt.policy_separation.templates_fairness_starvation_v2 import (  # noqa: E402
    case_fairness_vs_size_v2,
)
from llmserveopt.policy_separation.templates_prefill_decode_v2 import (  # noqa: E402
    case_prefill_decode_ttft_contention,
)

MF_PSD_LONG = REPO_ROOT / "experiments" / "mf_psd_v1" / "mf_psd_long_v1.csv"
FAMILY_C_RECON_SCENARIOS = (
    REPO_ROOT / "experiments" / "family_c_reconstruction_v1" / "family_c_reconstruction_v1_scenarios.jsonl"
)

FAMILY_A = "FAMILY_A_FAIRNESS_STARVATION_V2"
FAMILY_B = "FAMILY_B_PREFILL_DECODE_V2"
FAMILY_C = "FAMILY_C_KV_PRESSURE_V2"

IDENTITY_COLUMNS = ("canonical_scenario_id", "mechanism_family", "source_scenario_id", "replay_verified")
FORBIDDEN_AUDIT_ONLY_FIELDS = IDENTITY_COLUMNS
OUTPUT_COLUMNS = list(IDENTITY_COLUMNS) + list(SHARED_CORE_V1_FEATURES)


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
    """canonical_scenario_id -> list of long-form rows (one per evaluated
    policy; scenario-level fields are invariant across them, already
    verified by the MF-PSD builder's own validation)."""
    out: Dict[str, List[Dict[str, Any]]] = {}
    with open(MF_PSD_LONG, newline="") as f:
        for row in csv.DictReader(f):
            out.setdefault(row["canonical_scenario_id"], []).append(row)
    return out


def _replay_family_a(feature_row: Dict[str, str]) -> Tuple[Tuple[Request, ...], Tuple[GPUConfig, ...], str]:
    kwargs = dict(
        target_utilization=float(feature_row["target_utilization"]),
        tenant_weight_skew=float(feature_row["tenant_weight_skew"]),
        favored_tenant_size=feature_row["favored_tenant_size"],
        prediction_noise_sigma=float(feature_row["prediction_noise_sigma"]),
        seed=int(feature_row["seed"]),
        max_active_sequences=int(feature_row["max_active_sequences"]),
    )
    scenario = case_fairness_vs_size_v2(**kwargs)
    return scenario.requests, scenario.gpu_configs, scenario.scenario_id


def _replay_family_b(feature_row: Dict[str, str]) -> Tuple[Tuple[Request, ...], Tuple[GPUConfig, ...], str]:
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
    return scenario.requests, scenario.gpu_configs, scenario.scenario_id


def _load_family_c_index() -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    with open(FAMILY_C_RECON_SCENARIOS) as f:
        for line in f:
            d = json.loads(line)
            out[d["scenario_id"]] = d
    return out


def build(output_dir: Path) -> Dict[str, Any]:
    if "LLM_SERVEOPT_BURSTGPT_CSV" not in os.environ:
        raise SystemExit(
            "LLM_SERVEOPT_BURSTGPT_CSV must be set to a local staged BurstGPT CSV "
            "for Family A/B deterministic replay (see module docstring)."
        )
    burstgpt_csv = Path(os.environ["LLM_SERVEOPT_BURSTGPT_CSV"])
    if not burstgpt_csv.is_file():
        raise SystemExit(f"LLM_SERVEOPT_BURSTGPT_CSV does not exist: {burstgpt_csv}")

    scenario_rows_by_id = _load_mf_psd_scenario_rows()
    family_c_index = _load_family_c_index()

    output_rows: List[Dict[str, Any]] = []
    replay_mismatches: List[str] = []
    n_by_family = {FAMILY_A: 0, FAMILY_B: 0, FAMILY_C: 0}

    for canonical_scenario_id, rows in sorted(scenario_rows_by_id.items()):
        first = rows[0]
        family = first["mechanism_family"]
        source_scenario_id = first["source_scenario_id"]
        feature_row = json.loads(first["source_scenario_features_json"])

        if family == FAMILY_A:
            requests, gpu_configs, replayed_id = _replay_family_a(feature_row)
        elif family == FAMILY_B:
            requests, gpu_configs, replayed_id = _replay_family_b(feature_row)
        elif family == FAMILY_C:
            d = family_c_index[source_scenario_id]
            requests = tuple(Request(**r) for r in d["requests"])
            gpu_configs = tuple(GPUConfig(**g) for g in d["gpu_configs"])
            replayed_id = d["scenario_id"]
        else:
            raise AssertionError(f"unknown mechanism_family {family!r}")

        replay_verified = replayed_id == source_scenario_id
        if not replay_verified:
            replay_mismatches.append(canonical_scenario_id)

        feats = compute_shared_context_features(requests, gpu_configs)
        row = {
            "canonical_scenario_id": canonical_scenario_id,
            "mechanism_family": family,
            "source_scenario_id": source_scenario_id,
            "replay_verified": replay_verified,
        }
        row.update(feats)
        output_rows.append(row)
        n_by_family[family] += 1

    if replay_mismatches:
        raise SystemExit(f"{len(replay_mismatches)} scenarios failed replay verification: {replay_mismatches[:5]}")

    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "shared_core_v1_scenarios.csv"
    with open(table_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for r in output_rows:
            writer.writerow(r)

    schema = {
        "schema_version": SHARED_CORE_V1_VERSION,
        "identity_audit_columns": list(IDENTITY_COLUMNS),
        "learnable_feature_allowlist": list(SHARED_CORE_V1_FEATURES),
        "forbidden_audit_only_fields": list(FORBIDDEN_AUDIT_ONLY_FIELDS),
        "output_columns": OUTPUT_COLUMNS,
        "n_scenarios": len(output_rows),
        "n_scenarios_by_family": n_by_family,
        "notes": (
            "Every learnable column is present (non-empty, non-missing) for "
            "every scenario in every family -- no family-prefixed columns, "
            "no structural missingness, no mechanism_family or scenario ID "
            "in the learnable set. See the SHARED_CORE_V1 audit doc for the "
            "formula/units/provenance of each column."
        ),
    }
    schema_path = output_dir / "shared_core_v1_schema.json"
    with open(schema_path, "w") as f:
        json.dump(schema, f, indent=2, sort_keys=True)
        f.write("\n")

    provenance = {
        "schema_version": SHARED_CORE_V1_VERSION,
        "build_git_head_sha": _git_head_sha(),
        "build_timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sources_read": {
            "mf_psd_long_v1.csv": {
                "path": str(MF_PSD_LONG.relative_to(REPO_ROOT)),
                "sha256": _sha256_of_file(MF_PSD_LONG),
            },
            "family_c_reconstruction_v1_scenarios.jsonl": {
                "path": str(FAMILY_C_RECON_SCENARIOS.relative_to(REPO_ROOT)),
                "sha256": _sha256_of_file(FAMILY_C_RECON_SCENARIOS),
            },
            "burstgpt_csv_used_for_replay": {
                "path": str(burstgpt_csv),
                "sha256": _sha256_of_file(burstgpt_csv),
            },
        },
        "n_scenarios_replayed_family_a": n_by_family[FAMILY_A],
        "n_scenarios_replayed_family_b": n_by_family[FAMILY_B],
        "n_scenarios_loaded_family_c_direct": n_by_family[FAMILY_C],
        "all_scenario_id_replay_matches": True,
        "output_files": {
            "shared_core_v1_scenarios.csv": _sha256_of_file(table_path),
            "shared_core_v1_schema.json": _sha256_of_file(schema_path),
        },
    }
    provenance_path = output_dir / "shared_core_v1_provenance.json"
    with open(provenance_path, "w") as f:
        json.dump(provenance, f, indent=2, sort_keys=True)
        f.write("\n")

    summary = {
        "n_scenarios": len(output_rows),
        "n_scenarios_by_family": n_by_family,
        "n_features": len(SHARED_CORE_V1_FEATURES),
        "table_path": str(table_path.relative_to(REPO_ROOT)),
        "schema_path": str(schema_path.relative_to(REPO_ROOT)),
        "provenance_path": str(provenance_path.relative_to(REPO_ROOT)),
    }
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "experiments" / "shared_cross_family_features_v1",
    )
    args = ap.parse_args()
    build(args.out_dir)


if __name__ == "__main__":
    main()
