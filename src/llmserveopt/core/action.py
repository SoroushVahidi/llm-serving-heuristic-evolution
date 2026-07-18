"""
Action schema returned by scheduling policies.

An Action maps each GPU ID to the list of request IDs to admit from the
waiting queue into that GPU's active batch during the current step.
An empty mapping (or all-empty lists) is a valid "do nothing" action.

`preempt` (added for the vllm_faithful baseline; see
docs/vllm_faithful_scheduler_reference.md) is an optional, backward-compatible
extension: it maps each GPU ID to the list of currently-ACTIVE request IDs
that policy wants evicted back to the waiting queue this step, with their
progress discarded (recompute-on-resume semantics -- see
Simulator._apply_action / GPUState.evict). It defaults to empty and every
existing policy leaves it empty, so admit-only behavior is completely
unchanged.

`swap` (added for the distserve_faithful baseline; see
docs/distserve_faithful_scheduler_reference.md) is a second, narrowly-scoped
eviction verb: it maps each GPU ID to the list of currently-ACTIVE (decoding)
request IDs to evict this step WITHOUT discarding progress, re-queuing them
as immediately transfer-ready in the bridge queue rather than the ordinary
waiting queue (see Simulator._apply_action / GPUState.evict(preserve_progress=True)).
DistServe's decode-stage scheduler manages capacity via swap, not recompute,
since by the time a request reaches the decode stage its (potentially
cross-machine) prefill work is a sunk cost not worth re-paying. Defaults to
empty; every pre-existing policy (including vllm_faithful/sarathi_faithful)
leaves it empty, so behavior is completely unchanged for them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set


@dataclass
class Action:
    admit: Dict[int, List[int]] = field(default_factory=dict)
    preempt: Dict[int, List[int]] = field(default_factory=dict)
    swap: Dict[int, List[int]] = field(default_factory=dict)

    def all_admitted_ids(self) -> Set[int]:
        ids: Set[int] = set()
        for req_list in self.admit.values():
            ids.update(req_list)
        return ids

    def all_preempted_ids(self) -> Set[int]:
        ids: Set[int] = set()
        for req_list in self.preempt.values():
            ids.update(req_list)
        return ids

    def all_swapped_ids(self) -> Set[int]:
        ids: Set[int] = set()
        for req_list in self.swap.values():
            ids.update(req_list)
        return ids

    def is_empty(self) -> bool:
        return (
            all(len(v) == 0 for v in self.admit.values())
            and all(len(v) == 0 for v in self.preempt.values())
            and all(len(v) == 0 for v in self.swap.values())
        )

    def __repr__(self) -> str:
        total = sum(len(v) for v in self.admit.values())
        total_preempted = sum(len(v) for v in self.preempt.values())
        total_swapped = sum(len(v) for v in self.swap.values())
        extra = ""
        if total_preempted:
            extra += f", total_preempted={total_preempted}, preempt={self.preempt}"
        if total_swapped:
            extra += f", total_swapped={total_swapped}, swap={self.swap}"
        if extra:
            return f"Action(total_admitted={total}, by_gpu={self.admit}{extra})"
        return f"Action(total_admitted={total}, by_gpu={self.admit})"
