"""Grouped train/val/test splits for Family A v2 composition pilot."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence


@dataclass(frozen=True)
class SplitAssignment:
    train: List[str]
    val: List[str]
    test: List[str]
    ood: List[str]
    logic: str


def assign_family_a_v2_splits(
    scenario_meta: Sequence[Mapping[str, object]],
) -> SplitAssignment:
    """Assign scenario_ids to splits without row-level leakage.

    Primary logic (documented):
    - TRAIN: seed == 20260816, excluding OOD cells
    - VAL:   seed == 20260817 and util in {1.10, 1.30}, excluding OOD cells
    - TEST:  seed == 20260817 and util == 1.50, excluding OOD cells
    - OOD:   favored_tenant_size == \"long\" and tenant_weight_skew == 10.0
             (both seeds; never used for training/model selection)

    OOD holds out the strongest conflict/fairness regime combination.
    """
    by_id = {str(m["scenario_id"]): m for m in scenario_meta}
    train: List[str] = []
    val: List[str] = []
    test: List[str] = []
    ood: List[str] = []

    def _is_ood(m: Mapping[str, object]) -> bool:
        return str(m["favored_tenant_size"]) == "long" and float(m["tenant_weight_skew"]) == 10.0

    for sid, m in sorted(by_id.items()):
        if _is_ood(m):
            ood.append(sid)
            continue
        seed = int(m["seed"])
        util = float(m["target_utilization"])
        if seed == 20260816:
            train.append(sid)
        elif seed == 20260817 and util in {1.1, 1.10, 1.3, 1.30}:
            val.append(sid)
        elif seed == 20260817 and util in {1.5, 1.50}:
            test.append(sid)
        else:
            raise ValueError(f"Unassigned scenario in split logic: {sid} meta={dict(m)}")

    # Integrity: disjoint
    sets = [set(train), set(val), set(test), set(ood)]
    for i, a in enumerate(sets):
        for j, b in enumerate(sets):
            if i >= j:
                continue
            inter = a & b
            if inter:
                raise ValueError(f"Split leakage between buckets {i},{j}: {sorted(inter)[:5]}")

    logic = (
        "train=seed20260816 excluding (long,skew10); "
        "val=seed20260817 util∈{1.1,1.3} excluding OOD; "
        "test=seed20260817 util=1.5 excluding OOD; "
        "ood=(favored=long & skew=10) both seeds"
    )
    return SplitAssignment(train=train, val=val, test=test, ood=ood, logic=logic)


def assert_no_split_leakage(assignment: SplitAssignment) -> None:
    buckets = {
        "train": set(assignment.train),
        "val": set(assignment.val),
        "test": set(assignment.test),
        "ood": set(assignment.ood),
    }
    names = list(buckets)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            inter = buckets[a] & buckets[b]
            if inter:
                raise AssertionError(f"Leakage {a}∩{b}={sorted(inter)[:3]}")
