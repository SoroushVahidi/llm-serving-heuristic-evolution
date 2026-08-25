"""Family-B-Balanced Replication v1 -- frozen, metadata-only held-out
scenario selection (design doc SS 10 of
docs/design/HIERARCHICAL_REGIME_ROUTER_LIVE_REEVAL_V1.md, completed by
docs/design/FAMILY_B_BALANCED_REPLICATION_V1.md).

This module implements ONLY the deterministic selection of the 36-scenario
(12 Family-A + 12 Family-B + 12 Family-C) replication set from
`mf_psd_v1/mf_psd_scenarios_v1.csv`'s existing metadata (mechanism_family,
group_key, canonical_scenario_id, seed, and each row's frozen
train/val/test split assignment from `build_splits`). It never reads or
reasons about any ANWG/utility/policy-performance column -- selection is
governed strictly by which rows were never used to fit Stage-1/Stage-2
(i.e. `split != "train"`), not by any outcome.

IMPLEMENTATION ONLY. This module selects a scenario set; it does not fit
models, does not run the simulator or live harness, and does not compute
or imply any scientific verdict. Launching the replication evaluation
against this frozen set is a separately authorized action.
"""
from __future__ import annotations

from typing import Dict

import pandas as pd

FAMILY_A = "FAMILY_A_FAIRNESS_STARVATION_V2"
FAMILY_B = "FAMILY_B_PREFILL_DECODE_V2"
FAMILY_C = "FAMILY_C_KV_PRESSURE_V2"

# The first of the two frozen Family-C held-out evaluation seeds already
# named in configs/hierarchical_regime_router_v1_gates.json
# (splits.family_c_held_out.seeds = ["20260914", "20260915"]). Selecting
# by this pre-existing, already-frozen designation -- rather than an
# arbitrary new rule -- keeps the replication's Family-C selection
# traceable to a decision made before this task, not invented for it.
FAMILY_C_PRIMARY_HELD_OUT_SEED = 20260914

TARGET_PER_FAMILY = 12


def _non_train_pool(scen: pd.DataFrame, family: str) -> pd.DataFrame:
    return scen[(scen["mechanism_family"] == family) & (scen["split"] != "train")]


def select_family_a_replication(scen: pd.DataFrame) -> pd.DataFrame:
    """12 Family-A scenarios: prefer VAL (never observed by any prior
    evaluation) over TEST (already used in the primary re-evaluation),
    each tier sorted by canonical_scenario_id for a stable, arbitrary-but-
    deterministic tie-break. Minimizes -- without eliminating, since VAL
    alone has only 10 rows -- overlap with the primary 32-scenario TEST
    set."""
    pool = _non_train_pool(scen, FAMILY_A).copy()
    split_priority = {"val": 0, "test": 1}
    pool["_priority"] = pool["split"].map(split_priority)
    pool = pool.sort_values(["_priority", "canonical_scenario_id"])
    return pool.head(TARGET_PER_FAMILY).drop(columns=["_priority"])


def select_family_b_replication(scen: pd.DataFrame) -> pd.DataFrame:
    """12 Family-B scenarios, drawn entirely from VAL (Family B has 0
    TEST scenarios in the primary split, so this pool has zero overlap
    with the primary evaluation by construction). Family B has only 8
    total groups x 4 seeds; VAL holds 4 of those groups (16 rows) -- 12
    of the 16 are selected, sorted by canonical_scenario_id."""
    pool = _non_train_pool(scen, FAMILY_B)
    pool = pool[pool["split"] == "val"].sort_values("canonical_scenario_id")
    return pool.head(TARGET_PER_FAMILY)


def select_family_c_replication(scen: pd.DataFrame) -> pd.DataFrame:
    """12 Family-C scenarios: exactly the primary held-out-eval seed
    (20260914) row from each of Family C's 12 held-out groups. Family C
    has no VAL rows (it is seed-partitioned, not group-partitioned -- see
    hierarchical_regime_router_v1.build_splits docstring), so this pool is
    necessarily a subset of the primary 24-scenario Family-C TEST
    allocation; selecting exactly one (not both) held-out seed per group
    still yields 12 scenario rows never used in TRAIN fitting."""
    pool = _non_train_pool(scen, FAMILY_C)
    pool = pool[(pool["split"] == "test") & (pool["seed"] == FAMILY_C_PRIMARY_HELD_OUT_SEED)]
    pool = pool.sort_values("canonical_scenario_id")
    return pool.head(TARGET_PER_FAMILY)


def select_balanced_replication_set(scen: pd.DataFrame) -> pd.DataFrame:
    """The full frozen 36-scenario (12/12/12) Family-B-Balanced
    Replication set. `scen` must already carry a `split` column from the
    same frozen `build_splits` used by the primary evaluation."""
    parts = [
        select_family_a_replication(scen),
        select_family_b_replication(scen),
        select_family_c_replication(scen),
    ]
    out = pd.concat(parts, ignore_index=False)
    assert len(out) == 3 * TARGET_PER_FAMILY, f"expected {3 * TARGET_PER_FAMILY} scenarios, got {len(out)}"
    assert out["canonical_scenario_id"].is_unique, "duplicate scenario ids in replication set"
    return out


def verify_no_train_leakage(scen: pd.DataFrame, replication_set: pd.DataFrame) -> None:
    """Scenario-row-level (not group-level -- Family C intentionally
    shares group_key archetypes with TRAIN, see select_family_c_replication)
    disjointness check against the frozen primary TRAIN allocation."""
    train_ids = set(scen[scen["split"] == "train"]["canonical_scenario_id"])
    overlap = set(replication_set["canonical_scenario_id"]) & train_ids
    assert not overlap, f"replication set overlaps primary TRAIN scenarios: {overlap}"


def replication_family_counts(replication_set: pd.DataFrame) -> Dict[str, int]:
    return {k: int(v) for k, v in replication_set["mechanism_family"].value_counts().items()}
