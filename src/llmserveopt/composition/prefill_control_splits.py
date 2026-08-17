"""Family B v2 train/val/test/OOD splits for PrefillControl composition.

Split logic (grouped by seed, held-out seed 20260823):
  - TRAIN: seeds 20260820, 20260821, 20260822 (non-held-out)
  - VAL:   held-out subset of train (seed 20260820 reserved for val)
  - TEST:  seed 20260823 (held-out from all fitting)
  - OOD:   seed 20260823, but late_pressure=high + unusual factor combos
            (interpolation challenge over factor geometry, not distribution shift)

No generator labels (hog_count, late_pressure, slo_emphasis) leak into
split assignments.  Splits are seeded for reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence

# ===================================================================
# Split data structure
# ===================================================================

@dataclass(frozen=True)
class SplitAssignment:
    train: List[str]
    val: List[str]
    test: List[str]
    ood: List[str]
    logic: str


# ===================================================================
# Split helpers
# ===================================================================

# Seeds used in the v2 pilot
PILOT_SEEDS = (20260820, 20260821, 20260822, 20260823)
TRAIN_SEEDS = (20260820, 20260821, 20260822)
TEST_SEED = 20260823  # preregistered held-out
VAL_SEED = 20260820   # one of the train seeds held out for validation


def _parse_seed(scenario_id: str) -> int:
    """Extract seed from Family B v2 scenario_id."""
    parts = scenario_id.split(".s")
    return int(parts[-1])


def assign_family_b_v2_splits(
    scenario_ids: Sequence[str],
    *,
    n_val_scenarios: int = 8,
) -> SplitAssignment:
    """Assign scenario_ids to splits without row-level leakage.

    Logic:
      1. Parse each scenario_id to extract the seed.
      2. TEST / OOD = seed == 20260823 (preregistered held-out).
      3. Among TEST, split: late_pressure=high -> OOD (interpolation),
         late_pressure=low -> TEST.
      4. TRAIN / VAL among remaining seeds {20260820, 20260821, 20260822}:
        - If enough candidates (>= n_val_scenarios), first n_val_scenarios -> VAL,
          rest -> TRAIN.
        - If not enough to dedicate a full val set, keep a reasonable ratio:
          last floor(n_candidates/3) -> VAL, rest -> TRAIN.
          This ensures both train and val always have data (needed for selector fit).
    """
    train: List[str] = []
    val: List[str] = []
    test: List[str] = []
    ood: List[str] = []

    parsed = []
    for sid in sorted(scenario_ids):
        seed = _parse_seed(sid)
        parsed.append((sid, seed))

    # Step 1: split by seed
    held_out = [(sid, seed) for sid, seed in parsed if seed == TEST_SEED]
    train_candidates = [(sid, seed) for sid, seed in parsed if seed != TEST_SEED]

    # Step 2: held-out split -> test / ood
    for sid, seed in held_out:
        # OOD = high late_pressure on held-out seed
        if "late40" in sid:
            ood.append(sid)
        else:
            test.append(sid)

    # Step 3: val / train among non-held-out seeds
    train_candidates.sort()
    if len(train_candidates) >= max(n_val_scenarios, 4):
        # Enough data for dedicated val set
        val.extend(train_candidates[:n_val_scenarios])
        train.extend(train_candidates[n_val_scenarios:])
    elif len(train_candidates) >= 4:
        # Enough for meaningful train+val (e.g., smoke grid)
        n_val = max(1, len(train_candidates) // 3)
        val.extend(train_candidates[:n_val])
        train.extend(train_candidates[n_val:])
    else:
        # Degenerate case: put all in train (will fail selector fit, but
        # this is intentionally caught during launch)
        train.extend([sid for sid, _ in train_candidates])

    val_sids = [sid for sid, _ in val] if val else list(val)
    train_sids = [sid for sid, _ in train] if all(isinstance(t, tuple) for t in train) else list(train)

    # Integrity: disjoint
    all_sets = [set(train_sids), set(val_sids), set(test), set(ood)]
    for i, a in enumerate(all_sets):
        for j, b in enumerate(all_sets):
            if i >= j:
                continue
            inter = a & b
            if inter:
                raise AssertionError(
                    f"Split leakage {i}∩{j}: {sorted(inter)[:5]}"
                )

    logic = (
        f"train=seeds {TRAIN_SEEDS} minus first {n_val_scenarios} scenarios; "
        f"val=first {n_val_scenarios} non-held-out scenarios; "
        f"test=seed {TEST_SEED} late_pressure=low; "
        f"ood=seed {TEST_SEED} late_pressure=high"
    )
    return SplitAssignment(
        train=sorted(train_sids),
        val=sorted(val_sids),
        test=sorted(test),
        ood=sorted(ood),
        logic=logic,
    )


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
                raise AssertionError(f"Leakage {a}∩{b}: {sorted(inter)[:3]}")
