"""TRAIN/VAL/TEST/OOD splits for the KV-aware composition falsification v1.

Reuses KV v2's own preregistered seed partition (in-sample 20260910-13,
held-out 20260914-15) and subdivides the 4 in-sample seeds further. See
docs/design/KV_COMPOSITION_FALSIFICATION_V1.md section 7.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

TRAIN_SEEDS = (20260910, 20260911)
VAL_SEED = 20260912
TEST_SEED = 20260913
OOD_SEEDS = (20260914, 20260915)
ALL_SEEDS = TRAIN_SEEDS + (VAL_SEED, TEST_SEED) + OOD_SEEDS


@dataclass(frozen=True)
class SplitAssignment:
    train: List[str]
    val: List[str]
    test: List[str]
    ood: List[str]
    logic: str


def _parse_seed(scenario_id: str) -> int:
    """Extract seed from a Family C v2 scenario_id (...".s<seed>")."""
    parts = scenario_id.split(".s")
    return int(parts[-1])


def assign_kv_composition_splits(scenario_ids: Sequence[str]) -> SplitAssignment:
    train: List[str] = []
    val: List[str] = []
    test: List[str] = []
    ood: List[str] = []

    for sid in sorted(scenario_ids):
        seed = _parse_seed(sid)
        if seed in TRAIN_SEEDS:
            train.append(sid)
        elif seed == VAL_SEED:
            val.append(sid)
        elif seed == TEST_SEED:
            test.append(sid)
        elif seed in OOD_SEEDS:
            ood.append(sid)
        else:
            raise ValueError(f"scenario_id {sid!r} has unexpected seed {seed}")

    logic = (
        f"train=seeds {TRAIN_SEEDS}; val=seed {VAL_SEED}; "
        f"test=seed {TEST_SEED}; ood=seeds {OOD_SEEDS} "
        "(identical to KV v2's own in-sample/held-out seed partition)"
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
        for b in names[i + 1:]:
            inter = buckets[a] & buckets[b]
            if inter:
                raise AssertionError(f"Leakage {a}∩{b}: {sorted(inter)[:3]}")
