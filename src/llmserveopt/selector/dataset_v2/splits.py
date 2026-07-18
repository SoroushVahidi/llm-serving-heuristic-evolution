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
