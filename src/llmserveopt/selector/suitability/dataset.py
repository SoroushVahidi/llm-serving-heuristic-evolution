"""Long-format state-policy dataset adapter.

Converts existing wide-format window/policy-vector data (one row per
window plus one row per (window, policy) outcome -- the schema already
produced by scripts/run_local_e2e_smoke.py and the Selector Dataset v2
pilot builder) into one row per (state, deployable policy):

    state_id, state_features, policy_name, policy_hash,
    policy_representation, reward_anwg, completion_fraction,
    completed_request_quality, source, trace_family, temporal_block,
    split, seed

This module does not run the simulator itself -- it is a pure
transformation over already-computed wide-format rows, reusing the
existing causal feature extractor's output and the existing policy
registry/genome mapping. See docs/current/STATE_POLICY_SUITABILITY_SCHEMA.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ...policies.genome import SchedulerGenomeV1
from ...policies.registry import ORACLE_POLICY_NAMES, POLICY_LIBRARY_V2_NAMES
from ...policies.structural_synthesis import map_policy_to_genome
from ..advanced import validate_feature_columns

LONG_FORMAT_COLUMNS: tuple = (
    "state_id",
    "state_features",
    "policy_name",
    "policy_hash",
    "policy_representation",
    "reward_anwg",
    "completion_fraction",
    "completed_request_quality",
    "source",
    "trace_family",
    "temporal_block",
    "split",
    "seed",
)


@dataclass(frozen=True)
class LongFormatRow:
    state_id: str
    state_features: Dict[str, float]
    policy_name: str
    policy_hash: str
    policy_representation: Dict[str, Any]
    reward_anwg: Optional[float]
    completion_fraction: Optional[float]
    completed_request_quality: Optional[float]
    source: str
    trace_family: str
    temporal_block: str
    split: str
    seed: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state_id": self.state_id,
            "state_features": dict(self.state_features),
            "policy_name": self.policy_name,
            "policy_hash": self.policy_hash,
            "policy_representation": dict(self.policy_representation),
            "reward_anwg": self.reward_anwg,
            "completion_fraction": self.completion_fraction,
            "completed_request_quality": self.completed_request_quality,
            "source": self.source,
            "trace_family": self.trace_family,
            "temporal_block": self.temporal_block,
            "split": self.split,
            "seed": self.seed,
        }


def genome_table(policies: Sequence[str]) -> Dict[str, SchedulerGenomeV1]:
    """One SchedulerGenomeV1 per policy name (memoized computation)."""
    return {name: map_policy_to_genome(name) for name in policies}


def _state_features_from_window_row(window_row: Mapping[str, Any]) -> Dict[str, float]:
    feat_cols = sorted(k for k in window_row.keys() if k.startswith("feat_"))
    feat_cols = validate_feature_columns(feat_cols)
    out: Dict[str, float] = {}
    for col in feat_cols:
        value = window_row.get(col)
        try:
            out[col] = float(value) if value is not None else 0.0
        except (TypeError, ValueError):
            out[col] = 0.0
    return out


def build_long_format_rows(
    window_rows: Sequence[Mapping[str, Any]],
    policy_rows: Sequence[Mapping[str, Any]],
    *,
    deployable_policies: Sequence[str] = POLICY_LIBRARY_V2_NAMES,
    source: str,
    trace_family: str,
    seed: int,
    window_idx_field: str = "window_idx",
    split_field: str = "split",
    policy_name_field: str = "policy_name",
    anwg_field: str = "metric_arrival_normalized_weighted_goodput",
    completion_field: str = "metric_completion_fraction",
    quality_field: str = "metric_weighted_goodput",
) -> List[Dict[str, Any]]:
    """Build long-format (state, policy) rows from wide window/policy tables.

    Requirements enforced here:
      - hindsight/oracle policies are rejected if present in
        `deployable_policies` (this function only ever emits deployable
        policy rows);
      - each state's split is inherited exactly from `window_rows` --
        never recomputed, so leakage-safe splits computed upstream
        (selector/dataset_v2/splits.py) are preserved verbatim;
      - output rows are sorted deterministically by (state_id, policy_name);
      - policy_hash is SchedulerGenomeV1.stable_hash(), stable across calls;
      - state_features are restricted to validated causal feat_* columns.
    """
    oracle_overlap = set(deployable_policies) & set(ORACLE_POLICY_NAMES)
    if oracle_overlap:
        raise ValueError(
            f"deployable_policies must not include oracle/hindsight policies: {sorted(oracle_overlap)}"
        )

    genomes = genome_table(deployable_policies)
    from .encoders import structural_features  # local import: avoid a cycle with encoders.py

    representations = {
        name: structural_features(genomes[name]) | {"mapping_status_summary": genomes[name].metadata.get("mapping_status", "UNSUPPORTED")}
        for name in deployable_policies
    }
    hashes = {name: genomes[name].stable_hash() for name in deployable_policies}

    windows_by_idx = {row[window_idx_field]: row for row in window_rows}
    by_window_policy: Dict[Any, Dict[str, Mapping[str, Any]]] = {}
    for prow in policy_rows:
        pname = prow[policy_name_field]
        if pname not in deployable_policies:
            continue
        by_window_policy.setdefault(prow[window_idx_field], {})[pname] = prow

    rows: List[LongFormatRow] = []
    for widx, window_row in windows_by_idx.items():
        state_id = f"{trace_family}__w{widx}"
        state_features = _state_features_from_window_row(window_row)
        split = str(window_row.get(split_field, "UNKNOWN"))
        policy_outcomes = by_window_policy.get(widx, {})
        for policy_name in deployable_policies:
            prow = policy_outcomes.get(policy_name)
            reward = completion = quality = None
            if prow is not None:
                reward = _to_float_or_none(prow.get(anwg_field))
                completion = _to_float_or_none(prow.get(completion_field))
                quality = _to_float_or_none(prow.get(quality_field))
            rows.append(LongFormatRow(
                state_id=state_id,
                state_features=state_features,
                policy_name=policy_name,
                policy_hash=hashes[policy_name],
                policy_representation=representations[policy_name],
                reward_anwg=reward,
                completion_fraction=completion,
                completed_request_quality=quality,
                source=source,
                trace_family=trace_family,
                temporal_block=str(widx),
                split=split,
                seed=seed,
            ))

    rows.sort(key=lambda r: (r.state_id, r.policy_name))

    # No cross-split duplication: a state_id must map to exactly one split.
    split_by_state: Dict[str, str] = {}
    for r in rows:
        prior = split_by_state.setdefault(r.state_id, r.split)
        if prior != r.split:
            raise ValueError(f"state_id {r.state_id!r} has inconsistent split assignment: {prior!r} vs {r.split!r}")

    return [r.to_dict() for r in rows]


def _to_float_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    return f


def rows_with_reward(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Rows where reward_anwg was actually populated (usable for training/eval)."""
    return [dict(r) for r in rows if r.get("reward_anwg") is not None]


def group_by_state(rows: Sequence[Mapping[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        out.setdefault(row["state_id"], []).append(dict(row))
    return out
