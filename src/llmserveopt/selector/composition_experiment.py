"""Utilities for leakage-safe composition experiment preparation."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

PRIMARY_OBJECTIVE = "metric_arrival_normalized_weighted_goodput"


class CompositionExperimentError(ValueError):
    """Raised when a composition experiment setup would be invalid."""


@dataclass(frozen=True)
class UpstreamWorkflowReadiness:
    frontier_root: Path
    policy_library_root: Path
    frontier_final_report_exists: bool
    policy_library_final_report_exists: bool

    @property
    def ready(self) -> bool:
        return self.frontier_final_report_exists and self.policy_library_final_report_exists


def check_upstream_readiness(frontier_root: str | Path, policy_library_root: str | Path) -> UpstreamWorkflowReadiness:
    frontier = Path(frontier_root)
    library = Path(policy_library_root)
    return UpstreamWorkflowReadiness(
        frontier_root=frontier,
        policy_library_root=library,
        frontier_final_report_exists=(frontier / "reports" / "FINAL_REPORT.md").exists(),
        policy_library_final_report_exists=(library / "reports" / "FINAL_POLICY_LIBRARY_REPORT.md").exists()
        or (library / "reports" / "FINAL_REPORT.md").exists(),
    )


def load_policy_vectors_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def assert_no_split_group_leakage(
    rows: Sequence[Mapping[str, str]],
    *,
    split_key: str = "split",
    group_key: str = "split_group_key",
) -> None:
    """Verify each split group appears in at most one split."""
    groups: dict[str, set[str]] = {}
    for row in rows:
        group = row.get(group_key)
        split = row.get(split_key)
        if not group or not split:
            raise CompositionExperimentError(f"Missing {group_key!r} or {split_key!r} in row")
        groups.setdefault(group, set()).add(split)
    leaked = {group: sorted(splits) for group, splits in groups.items() if len(splits) > 1}
    if leaked:
        preview = dict(list(sorted(leaked.items()))[:5])
        raise CompositionExperimentError(f"Split-group leakage detected: {preview}")


def select_best_fixed_policy_from_development(
    rows: Sequence[Mapping[str, str]],
    *,
    development_splits: Iterable[str] = ("TRAIN", "VALIDATION", "ROBUST_DEV"),
    policy_key: str = "policy_name",
    split_key: str = "split",
    metric_key: str = PRIMARY_OBJECTIVE,
) -> tuple[str, dict[str, float]]:
    """Select a fixed policy using development splits only."""
    dev = set(development_splits)
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for row in rows:
        if row.get(split_key) not in dev:
            continue
        policy = row.get(policy_key)
        value = row.get(metric_key)
        if not policy or value in (None, ""):
            continue
        try:
            metric = float(value)
        except ValueError as exc:
            raise CompositionExperimentError(f"Invalid metric value {value!r}") from exc
        totals[policy] = totals.get(policy, 0.0) + metric
        counts[policy] = counts.get(policy, 0) + 1
    if not totals:
        raise CompositionExperimentError("No development rows available for fixed-policy selection")
    means = {policy: totals[policy] / counts[policy] for policy in totals}
    best = max(means.items(), key=lambda item: (item[1], item[0]))[0]
    return best, means


def validate_treatment_selection_does_not_use_heldout(
    selected_from_splits: Iterable[str],
    *,
    forbidden_splits: Iterable[str] = ("ID_TEST", "OOD_TEST", "TEMPORAL_OOD", "CROSS_SOURCE_OOD", "FINAL_OOD"),
) -> None:
    selected = set(selected_from_splits)
    forbidden = set(forbidden_splits)
    overlap = sorted(selected & forbidden)
    if overlap:
        raise CompositionExperimentError(f"Treatment selection used held-out splits: {overlap}")
