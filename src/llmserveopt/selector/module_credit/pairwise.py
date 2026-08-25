"""Pairwise module interaction targets."""
from __future__ import annotations

from typing import Any, Mapping, Sequence


def build_pairwise_interaction_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Build I(m1,m2,x) = R(both)-R(m1)-R(m2)+R(base) rows when artifacts exist.

    Required raw fields per pairwise row: ``module_type_a``, ``module_type_b``,
    ``single_reward_a``, ``single_reward_b``, ``both_reward``, and
    ``base_reward``. Rows without these fields are ignored.
    """
    out = []
    required = ("module_type_a", "module_type_b", "single_reward_a", "single_reward_b", "both_reward", "base_reward")
    for row in rows:
        if not all(k in row for k in required):
            continue
        interaction = float(row["both_reward"]) - float(row["single_reward_a"]) - float(row["single_reward_b"]) + float(row["base_reward"])
        if interaction > 0.01:
            label = "synergistic"
        elif interaction < -0.01:
            label = "antagonistic"
        else:
            label = "additive"
        out.append({
            "state_id": row.get("state_id"),
            "base_policy": row.get("base_policy"),
            "donor_policy_a": row.get("donor_policy_a", row.get("donor_policy")),
            "donor_policy_b": row.get("donor_policy_b", row.get("donor_policy")),
            "module_type_a": row["module_type_a"],
            "module_type_b": row["module_type_b"],
            "interaction": interaction,
            "interaction_class": label,
        })
    return out
