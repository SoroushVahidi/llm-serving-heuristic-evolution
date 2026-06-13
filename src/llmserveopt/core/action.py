"""
Action schema returned by scheduling policies.

An Action maps each GPU ID to the list of request IDs to admit from the
waiting queue into that GPU's active batch during the current step.
An empty mapping (or all-empty lists) is a valid "do nothing" action.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set


@dataclass
class Action:
    admit: Dict[int, List[int]] = field(default_factory=dict)

    def all_admitted_ids(self) -> Set[int]:
        ids: Set[int] = set()
        for req_list in self.admit.values():
            ids.update(req_list)
        return ids

    def is_empty(self) -> bool:
        return all(len(v) == 0 for v in self.admit.values())

    def __repr__(self) -> str:
        total = sum(len(v) for v in self.admit.values())
        return f"Action(total_admitted={total}, by_gpu={self.admit})"
