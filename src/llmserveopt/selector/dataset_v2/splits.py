"""
Group-aware, leakage-safe splitting for Selector Dataset v2. See
docs/selector_dataset_v2.md §9.

The unit of splitting is a GROUP KEY (e.g. `scenario_family_id` or
`source_trace`), never an individual window/row -- this is what prevents
derived variants of one base scenario/trace from being scattered across
train and test. Assignment is a deterministic hash of the group key
string (no RNG seed involved), so:
  - re-running split assignment with the same group set always reproduces
    the same split,
  - adding a brand-new group later does not reshuffle any existing
    group's assignment (unlike a seeded-shuffle-then-slice approach,
    which reassigns everything whenever the group count changes).
"""
from __future__ import annotations

import hashlib
from typing import Dict, List, Optional, Sequence, Set

TRAIN = "TRAIN"
VALIDATION = "VALIDATION"
ID_TEST = "ID_TEST"
OOD_TEST = "OOD_TEST"

ALL_SPLITS = (TRAIN, VALIDATION, ID_TEST, OOD_TEST)


def _stable_hash_fraction(key: str) -> float:
    """Deterministic, RNG-free [0, 1) fraction derived from a string key."""
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def assign_group_aware_split(
    group_keys: Sequence[str],
    ood_group_keys: Optional[Set[str]] = None,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
) -> Dict[str, str]:
    """Assign every distinct group key to exactly one split. Groups in
    `ood_group_keys` are unconditionally forced to OOD_TEST (e.g. an
    entire held-out workload-source family) regardless of their hash --
    this is how "hold out one entire source family for OOD evaluation"
    (docs/selector_dataset_v2.md §9) is implemented. Every other group is
    assigned TRAIN/VALIDATION/ID_TEST by its stable hash fraction; the
    remainder after `train_frac`+`val_frac` goes to ID_TEST."""
    if not (0.0 < train_frac < 1.0) or not (0.0 < val_frac < 1.0) or train_frac + val_frac >= 1.0:
        raise ValueError(f"train_frac={train_frac}, val_frac={val_frac} must be positive and sum to < 1.0")

    ood_group_keys = ood_group_keys or set()
    assignment: Dict[str, str] = {}
    for gk in group_keys:
        if gk in ood_group_keys:
            assignment[gk] = OOD_TEST
            continue
        frac = _stable_hash_fraction(gk)
        if frac < train_frac:
            assignment[gk] = TRAIN
        elif frac < train_frac + val_frac:
            assignment[gk] = VALIDATION
        else:
            assignment[gk] = ID_TEST
    return assignment


def split_for_group(group_key: str, group_assignment: Dict[str, str]) -> str:
    """Look up a group's split. Raises KeyError (never silently defaults)
    if `group_key` was not part of the assignment -- an unknown group
    indicates a caller bug (a scenario family that was never registered
    with `assign_group_aware_split`), not a case to paper over."""
    if group_key not in group_assignment:
        raise KeyError(
            f"Group key '{group_key}' has no split assignment -- it was never "
            f"passed to assign_group_aware_split(). This must not be silently "
            f"defaulted, since doing so risks leaking an unassigned group into "
            f"an arbitrary split."
        )
    return group_assignment[group_key]


def verify_group_atomicity(rows: List[Dict], group_key_field: str, split_field: str = "split") -> None:
    """Raise ValueError if any group key's rows are scattered across more
    than one split -- the core leakage-prevention invariant for group-aware
    splitting."""
    seen: Dict[str, Set[str]] = {}
    for row in rows:
        gk = row[group_key_field]
        seen.setdefault(gk, set()).add(row[split_field])
    violations = {gk: sorted(splits) for gk, splits in seen.items() if len(splits) > 1}
    if violations:
        raise ValueError(f"Group atomicity violated for '{group_key_field}': {violations}")


def verify_ood_holdout(rows: List[Dict], group_key_field: str, ood_group_keys: Set[str], split_field: str = "split") -> None:
    """Raise ValueError if any row belonging to an OOD-held-out group ever
    appears in a non-OOD_TEST split."""
    violations = []
    for row in rows:
        gk = row[group_key_field]
        if gk in ood_group_keys and row[split_field] != OOD_TEST:
            violations.append((gk, row[split_field]))
    if violations:
        raise ValueError(f"OOD-held-out group(s) leaked into a non-OOD split: {sorted(set(violations))}")


def leakage_safe_split_group_key(row: Dict) -> str:
    """Return the split group for a selector-window row.

    Real-trace windows can be transformed multiple ways while preserving the
    same underlying request row range. Splitting by transform-specific group
    keys lets those sibling windows cross TRAIN/VALIDATION/ID_TEST. Grouping
    by the raw-trace ancestor and temporal pool keeps transformed siblings
    atomic while still forcing newer/OOD pools independently.

    The real-trace key format (``f"{ancestor}__pool_{pool}"``) intentionally
    matches the format originally introduced on
    origin/wulver-final-integration-20260721 (commit c8aee129), not an
    equivalent-but-differently-formatted alternative. Split assignment
    (`assign_group_aware_split`) is a SHA256 hash of this exact string,
    so changing the format changes which split any given group lands in --
    it would silently reshuffle TRAIN/VALIDATION/ID_TEST/OOD_TEST for any
    pilot already regenerated on that lineage since c8aee129.

    For real-trace rows, `request_plan_ancestor_id` and `time_slice_pool`
    are required and never silently substituted with the leaky
    transform-specific `group_key` -- a missing value here is a data-quality
    bug in the upstream window/checkpoint construction, not a case to paper
    over by falling back to the exact grouping this function exists to fix.
    """
    dataset_family = str(row.get("dataset_family", ""))
    if dataset_family == "real_trace":
        ancestor = row.get("request_plan_ancestor_id")
        pool = row.get("time_slice_pool")
        if not ancestor or not pool:
            raise KeyError(
                "real_trace row is missing request_plan_ancestor_id and/or "
                "time_slice_pool -- cannot derive a leakage-safe split group "
                f"key. Refusing to fall back to the transform-specific "
                f"group_key ({row.get('group_key')!r}), since that is the "
                "exact leaky grouping this function exists to avoid. Row: "
                f"{row}"
            )
        return f"{ancestor}__pool_{pool}"
    if row.get("group_key"):
        return str(row["group_key"])
    if row.get("scenario_family_id"):
        return str(row["scenario_family_id"])
    raise KeyError("Cannot derive a leakage-safe split group key from row")


def attach_leakage_safe_split_group_keys(
    rows: List[Dict],
    *,
    output_field: str = "split_group_key",
) -> List[str]:
    """Attach and return leakage-safe split keys for every row."""
    keys: List[str] = []
    for row in rows:
        key = leakage_safe_split_group_key(row)
        row[output_field] = key
        keys.append(key)
    return keys


def verify_no_cross_split_row_range_overlap(
    rows: List[Dict],
    *,
    dataset_family_field: str = "dataset_family",
    ancestor_field: str = "request_plan_ancestor_id",
    pool_field: str = "time_slice_pool",
    row_start_field: str = "time_slice_row_start",
    row_end_field: str = "time_slice_row_end",
    split_field: str = "split",
) -> None:
    """Raise if overlapping raw trace row ranges appear in different splits.

    This catches the cross-transform real-trace leakage mode that ordinary
    group atomicity misses when transform-specific groups reuse the same raw
    request slice.
    """
    real_rows = [
        row for row in rows
        if str(row.get(dataset_family_field, "")) == "real_trace"
        and _has_int(row.get(row_start_field))
        and _has_int(row.get(row_end_field))
    ]
    by_source_pool: Dict[tuple[str, str], List[Dict]] = {}
    for row in real_rows:
        ancestor = str(row.get(ancestor_field, ""))
        pool = str(row.get(pool_field, ""))
        by_source_pool.setdefault((ancestor, pool), []).append(row)

    violations = []
    for (ancestor, pool), group in by_source_pool.items():
        for i, a in enumerate(group):
            a_start, a_end = int(a[row_start_field]), int(a[row_end_field])
            for b in group[i + 1:]:
                b_start, b_end = int(b[row_start_field]), int(b[row_end_field])
                overlap = max(0, min(a_end, b_end) - max(a_start, b_start))
                if overlap <= 0:
                    continue
                if a.get(split_field) != b.get(split_field):
                    violations.append({
                        "ancestor": ancestor,
                        "pool": pool,
                        "a_window": a.get("window_idx", a.get("window_id")),
                        "a_split": a.get(split_field),
                        "b_window": b.get("window_idx", b.get("window_id")),
                        "b_split": b.get(split_field),
                        "overlap_rows": overlap,
                    })
    if violations:
        sample = violations[:5]
        raise ValueError(
            "Cross-split row-range overlap detected for real traces: "
            f"{sample} (total_violations={len(violations)})"
        )


def _has_int(value: object) -> bool:
    try:
        int(value)
        return True
    except (TypeError, ValueError):
        return False
