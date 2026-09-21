"""Intervention ingestion and canonical module-credit target construction.

Expected raw artifact schema
----------------------------
The ingestion layer accepts JSONL, JSON list, or CSV files.  Each row describes
one observed module intervention:

```
state_id, state_features or feat_* columns, base_policy, donor_policy,
module_type, base_reward, donor_reward, intervention_reward,
library_best_reward, source, trace_family, temporal_block, split, seed
```

Optional fields are retained when present: ``compatibility_metadata``,
``base_module_representation``, ``donor_module_representation``,
``base_policy_hash``, ``donor_policy_hash``, ``split_group_key``, raw-trace
range fields, and additional metric dictionaries.

The canonical targets are:

```
C_base   = R(child) - R(base)
C_parent = R(child) - max(R(base), R(donor))
C_env    = R(child) - R(library_best)
```
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ...policies.genome import SchedulerGenomeV1
from ...policies.registry import POLICY_LIBRARY_V2_NAMES
from ...policies.structural_synthesis import map_policy_to_genome
from ..dataset_v2.splits import verify_group_atomicity, verify_no_cross_split_row_range_overlap
from ..suitability.encoders import structural_features

MODULE_CREDIT_COLUMNS: tuple[str, ...] = (
    "state_id",
    "state_features",
    "base_policy",
    "donor_policy",
    "base_policy_hash",
    "donor_policy_hash",
    "module_type",
    "donor_module_representation",
    "base_module_representation",
    "compatibility_metadata",
    "base_reward",
    "donor_reward",
    "intervention_reward",
    "library_best_reward",
    "C_base",
    "C_parent",
    "C_env",
    "source",
    "trace_family",
    "temporal_block",
    "split",
    "seed",
    "donor_predicted_reward",
    "donor_uncertainty",
    "donor_conservative_suitability",
    "base_predicted_reward",
    "base_uncertainty",
    "base_conservative_suitability",
    "predicted_donor_vs_base_advantage",
)

TARGET_COLUMNS = frozenset({"intervention_reward", "C_base", "C_parent", "C_env"})
SUITABILITY_FEATURES = (
    "donor_predicted_reward",
    "donor_uncertainty",
    "donor_conservative_suitability",
    "base_predicted_reward",
    "base_uncertainty",
    "base_conservative_suitability",
    "predicted_donor_vs_base_advantage",
)


class ModuleInterventionDataError(ValueError):
    """Raised when intervention artifacts cannot be converted safely."""


def load_intervention_artifacts(root_or_file: str | Path) -> list[dict[str, Any]]:
    """Load raw intervention rows from a file or directory.

    Directory ingestion is intentionally conservative: only files named with
    ``intervention`` and ending in ``.jsonl``, ``.json``, or ``.csv`` are read.
    """
    path = Path(root_or_file)
    if not path.exists():
        raise FileNotFoundError(path)
    files: list[Path]
    if path.is_dir():
        files = sorted(
            p for p in path.rglob("*")
            if p.is_file() and "intervention" in p.name and p.suffix.lower() in {".jsonl", ".json", ".csv"}
        )
    else:
        files = [path]
    if not files:
        raise ModuleInterventionDataError(f"No intervention artifact files found under {path}")
    rows: list[dict[str, Any]] = []
    for file in files:
        rows.extend(_load_one_file(file))
    return rows


def _load_one_file(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        out = []
        with path.open() as handle:
            for line in handle:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out
    if suffix == ".json":
        payload = json.loads(path.read_text())
        if isinstance(payload, list):
            return [dict(r) for r in payload]
        if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
            return [dict(r) for r in payload["rows"]]
        raise ModuleInterventionDataError(f"Unsupported JSON intervention payload in {path}")
    if suffix == ".csv":
        with path.open(newline="") as handle:
            return [dict(r) for r in csv.DictReader(handle)]
    raise ModuleInterventionDataError(f"Unsupported intervention artifact extension: {path}")


def build_intervention_dataset(
    raw_rows: Sequence[Mapping[str, Any]],
    *,
    suitability_model: Any | None = None,
    suitability_lambda: float = 0.5,
    all_policies: Sequence[str] = POLICY_LIBRARY_V2_NAMES,
) -> list[dict[str, Any]]:
    """Build canonical long-format module-credit rows.

    ``suitability_model`` may be any existing state-policy model implementing
    ``predict_mean`` and ``predict_uncertainty``.  Its predictions are attached
    as input features only; intervention rewards and credit targets are never
    used as model inputs.
    """
    if not raw_rows:
        raise ModuleInterventionDataError("Cannot build module-credit dataset from zero rows")
    out: list[dict[str, Any]] = []
    for raw in raw_rows:
        state_features = _coerce_state_features(raw)
        base = _required_str(raw, "base_policy")
        donor = _required_str(raw, "donor_policy")
        module_type = _required_str(raw, "module_type")
        base_reward = _required_float(raw, "base_reward")
        donor_reward = _required_float(raw, "donor_reward")
        intervention_reward = _required_float(raw, "intervention_reward")
        library_best_reward = _required_float(raw, "library_best_reward")

        base_genome = map_policy_to_genome(base)
        donor_genome = map_policy_to_genome(donor)
        base_module = _module_representation(raw.get("base_module_representation"), base_genome, module_type)
        donor_module = _module_representation(raw.get("donor_module_representation"), donor_genome, module_type)
        compatibility = _coerce_mapping(raw.get("compatibility_metadata")) or _default_compatibility(base_genome, donor_genome, module_type)

        row = {
            "state_id": _required_str(raw, "state_id"),
            "state_features": state_features,
            "base_policy": base,
            "donor_policy": donor,
            "base_policy_hash": str(raw.get("base_policy_hash") or base_genome.stable_hash()),
            "donor_policy_hash": str(raw.get("donor_policy_hash") or donor_genome.stable_hash()),
            "module_type": module_type,
            "donor_module_representation": donor_module,
            "base_module_representation": base_module,
            "compatibility_metadata": compatibility,
            "base_reward": base_reward,
            "donor_reward": donor_reward,
            "intervention_reward": intervention_reward,
            "library_best_reward": library_best_reward,
            "C_base": intervention_reward - base_reward,
            "C_parent": intervention_reward - max(base_reward, donor_reward),
            "C_env": intervention_reward - library_best_reward,
            "source": str(raw.get("source", "unknown")),
            "trace_family": str(raw.get("trace_family", "unknown")),
            "temporal_block": str(raw.get("temporal_block", raw.get("window_idx", ""))),
            "split": str(raw.get("split", "TRAIN")),
            "seed": int(float(raw.get("seed", 0))),
            "split_group_key": str(raw.get("split_group_key", raw.get("state_id"))),
        }
        row.update(_additional_suitability_features(row, suitability_model, suitability_lambda, all_policies))
        for key in (
            "dataset_family",
            "request_plan_ancestor_id",
            "time_slice_pool",
            "time_slice_row_start",
            "time_slice_row_end",
            "window_idx",
        ):
            if key in raw:
                row[key] = raw[key]
        out.append(row)
    out.sort(key=lambda r: (r["state_id"], r["base_policy"], r["donor_policy"], r["module_type"]))
    validate_split_integrity(out)
    validate_no_target_leakage(out)
    return out


def validate_split_integrity(rows: Sequence[Mapping[str, Any]]) -> None:
    """Verify all variants of the same state/split group stay atomic."""
    materialized = [dict(r) for r in rows]
    verify_group_atomicity(materialized, "state_id", "split")
    verify_group_atomicity(materialized, "split_group_key", "split")
    verify_no_cross_split_row_range_overlap(materialized)


def validate_no_target_leakage(rows: Sequence[Mapping[str, Any]]) -> None:
    """Assert target columns are absent from all declared input feature maps."""
    feature_maps = (
        "state_features",
        "donor_module_representation",
        "base_module_representation",
        "compatibility_metadata",
    )
    for row in rows:
        for field in feature_maps:
            payload = row.get(field, {})
            if not isinstance(payload, Mapping):
                continue
            leaked = TARGET_COLUMNS & set(payload)
            if leaked:
                raise ModuleInterventionDataError(f"Target leakage in {field}: {sorted(leaked)}")


def _additional_suitability_features(
    row: Mapping[str, Any],
    model: Any | None,
    lam: float,
    all_policies: Sequence[str],
) -> dict[str, float]:
    if model is None:
        return {name: 0.0 for name in SUITABILITY_FEATURES}
    donor_row = _state_policy_query_row(row, row["donor_policy"], all_policies)
    base_row = _state_policy_query_row(row, row["base_policy"], all_policies)
    donor_mu = float(model.predict_mean([donor_row])[0])
    base_mu = float(model.predict_mean([base_row])[0])
    donor_u = float(model.predict_uncertainty([donor_row])[0])
    base_u = float(model.predict_uncertainty([base_row])[0])
    return {
        "donor_predicted_reward": donor_mu,
        "donor_uncertainty": donor_u,
        "donor_conservative_suitability": donor_mu - float(lam) * donor_u,
        "base_predicted_reward": base_mu,
        "base_uncertainty": base_u,
        "base_conservative_suitability": base_mu - float(lam) * base_u,
        "predicted_donor_vs_base_advantage": donor_mu - base_mu,
    }


def _state_policy_query_row(row: Mapping[str, Any], policy: str, all_policies: Sequence[str]) -> dict[str, Any]:
    genome = map_policy_to_genome(policy)
    return {
        "state_id": row["state_id"],
        "state_features": dict(row["state_features"]),
        "policy_name": policy,
        "policy_representation": structural_features(genome),
        "reward_anwg": 0.0,
    }


def _module_representation(raw: Any, genome: SchedulerGenomeV1, module_type: str) -> dict[str, float]:
    provided = _coerce_mapping(raw)
    if provided:
        return {str(k): float(v) for k, v in provided.items() if _is_number(v)}
    module = getattr(genome, module_type, None)
    if module is None:
        return {
            "module_present": 0.0,
            "module_status_exact": 0.0,
            "module_status_approximate": 0.0,
            "module_status_unsupported": 0.0,
        }
    from .encoders import module_structural_features

    return module_structural_features(module)


def _default_compatibility(base_genome: SchedulerGenomeV1, donor_genome: SchedulerGenomeV1, module_type: str) -> dict[str, float]:
    donor_module = getattr(donor_genome, module_type, None)
    base_module = getattr(base_genome, module_type, None)
    donor_present = donor_module is not None and donor_module.status != "UNSUPPORTED"
    supported = module_type in {"admission_rule", "priority_rule", "prefill_rule", "kv_guard", "fairness_rule"}
    return {
        "compatible": float(bool(donor_present and supported)),
        "donor_module_present": float(donor_module is not None),
        "base_module_present": float(base_module is not None),
        "donor_exact": float(donor_module is not None and donor_module.status == "EXACT"),
        "donor_approximate": float(donor_module is not None and donor_module.status == "APPROXIMATE"),
    }


def _coerce_state_features(row: Mapping[str, Any]) -> dict[str, float]:
    payload = row.get("state_features")
    mapping = _coerce_mapping(payload)
    if mapping:
        return {str(k): float(v) for k, v in mapping.items() if _is_number(v)}
    feat_cols = sorted(k for k in row if str(k).startswith("feat_"))
    if not feat_cols:
        raise ModuleInterventionDataError(f"Row {row.get('state_id')!r} has no state_features or feat_* columns")
    return {str(k): float(row[k]) for k in feat_cols if _is_number(row[k])}


def _coerce_mapping(value: Any) -> dict[str, Any]:
    if value is None or value == "":
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        try:
            payload = json.loads(value)
        except json.JSONDecodeError:
            return {}
        if isinstance(payload, Mapping):
            return dict(payload)
    return {}


def _required_str(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    if value is None or value == "":
        raise ModuleInterventionDataError(f"Missing required field {key!r}")
    return str(value)


def _required_float(row: Mapping[str, Any], key: str) -> float:
    value = row.get(key)
    if not _is_number(value):
        raise ModuleInterventionDataError(f"Missing or nonnumeric required field {key!r}: {value!r}")
    out = float(value)
    if not np.isfinite(out):
        raise ModuleInterventionDataError(f"Non-finite field {key!r}: {value!r}")
    return out


def _is_number(value: Any) -> bool:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return False
    return np.isfinite(f)
