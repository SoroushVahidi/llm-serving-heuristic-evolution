"""Diagnostic instrumentation for decomposing VTC scheduling outcomes into
FAIRNESS-ORDERING effects vs. RESERVATION-ADMISSION effects.

Motivation (see docs/audits/vtc_fairness_benchmark_repair_20260805.md):
the initial smoke evaluation (docs/audits/vtc_initial_integration_20260805.md)
found one workload family where VTC's completion/ANWG collapsed relative
to FIFO, and could not tell from the aggregate metrics alone whether this
was caused by VTC's fairness ordering (the thing actually being evaluated)
or by the official `_can_add_new_req` memory-reservation gate (an
orthogonal mechanism the official code also happens to implement). This
module answers that question by recording, at the finest available
granularity, every accept/reject decision the official code makes.

**Never modifies the official VTCReqQueue/ReqQueue source.** The pinned
clone at ``adapter.provenance.PINNED_COMMIT`` is untouched on disk.
Instrumentation is applied by replacing a single INSTANCE attribute
(`self._vtc._can_add_new_req`) with a thin wrapper that calls the real,
original, unmodified bound method and records its (input, output) before
returning the exact same result -- scheduling behavior (what gets
admitted, in what order, with what cost) is bit-for-bit identical to the
uninstrumented `VTCFairnessPolicy`. This is standard method-wrapping
instrumentation (the same technique `unittest.mock.patch` or a profiling
decorator uses), not a source modification, and it is verified never to
change outcomes by `tests/test_vtc_fairness_diagnostics.py`'s
identical-decisions-under-instrumentation test.

Exposes the exact mechanism responsible for the confound (read directly
from the pinned source, `slora/server/router/vtc_req_queue.py` lines
104-133): `generate_new_batch`'s admission loop tests **at most one**
non-empty-queue (backlogged) tenant per call -- the single least-served
one. If that tenant's head-of-line request fails `_can_add_new_req` (the
worst-case memory-reservation check) or the batch-token-budget check, the
loop `break`s immediately and stops the ENTIRE step's admission, even if
a different, already-backlogged tenant has a smaller request that would
easily fit. This module measures exactly how often that happens
(`reservation_stopped_step` / `budget_stopped_step`) versus how often a
step's outcome was governed purely by fairness ordering among requests
that all passed feasibility (`ordering_governed_step`).
"""
from __future__ import annotations

import types
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from llmserveopt.core.action import Action
from llmserveopt.core.types import ObservableState

from .simulator_policy import VTCFairnessPolicy


@dataclass
class StepDecision:
    step_index: int
    time: float
    n_tenants_backlogged: int          # tenants with a non-empty official waiting queue, before this step's admission
    n_active_before: int                # GPU active_request_ids count, before this step's admission
    candidates_tried: List[Tuple[int, str, bool, str]] = field(default_factory=list)
    # (request_id, tenant, accepted, reason) -- reason in
    # {"admitted", "reservation_rejected", "budget_rejected"}
    admitted_request_ids: List[int] = field(default_factory=list)
    vtc_first_pick_tenant: Optional[str] = None
    fifo_first_pick_tenant: Optional[str] = None

    @property
    def is_contended(self) -> bool:
        return self.n_tenants_backlogged >= 2

    @property
    def disagrees_with_fifo(self) -> bool:
        """True when, at a genuinely contended step, VTC's min-served
        selection differs from what plain FCFS (oldest waiting request
        across all backlogged tenants) would have picked next. Undefined
        (False) at uncontended steps -- with only one candidate, there is
        nothing to disagree about."""
        return (
            self.is_contended
            and self.vtc_first_pick_tenant is not None
            and self.fifo_first_pick_tenant is not None
            and self.vtc_first_pick_tenant != self.fifo_first_pick_tenant
        )

    @property
    def stopped_by_reservation(self) -> bool:
        return bool(self.candidates_tried) and self.candidates_tried[-1][3] == "reservation_rejected"

    @property
    def stopped_by_budget(self) -> bool:
        return bool(self.candidates_tried) and self.candidates_tried[-1][3] == "budget_rejected"

    @property
    def had_unexplored_backlog(self) -> bool:
        """True when the loop stopped (reservation or budget) while more
        than one tenant was backlogged -- i.e. a DIFFERENT tenant's
        request was never even considered this step, because the official
        loop never tries a second candidate after the first rejection."""
        return (self.stopped_by_reservation or self.stopped_by_budget) and self.n_tenants_backlogged > 1


class InstrumentedVTCFairnessPolicy(VTCFairnessPolicy):
    """Drop-in subclass of VTCFairnessPolicy that additionally records a
    StepDecision per select_action call. Diagnostic-only -- see module
    docstring for the exact, verified-inert instrumentation mechanism.
    Never used as the actual policy in the comparative sweep (§8 of the
    repair task) -- only to explain VTCFairnessPolicy's own decisions."""

    name = "vtc_fairness_reference_instrumented"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_log: List[StepDecision] = []
        self._events: List[Tuple[int, str, bool]] = []
        self._install_instrumentation()

    def _install_instrumentation(self) -> None:
        vtc = self._vtc
        original = type(vtc)._can_add_new_req  # unbound function, official + unmodified

        def _wrapped(self_q, req, lora_ranks):
            ok = original(self_q, req, lora_ranks)
            self._events.append((req.request_id, req.adapter_dir, bool(ok)))
            return ok

        vtc._can_add_new_req = types.MethodType(_wrapped, vtc)

    def reset(self) -> None:
        super().reset()
        self.step_log = []
        self._events = []
        # reset() rebuilds self._vtc -- re-install on the new instance.
        if hasattr(self, "_vtc") and self._vtc is not None:
            self._install_instrumentation()

    def select_action(self, state: ObservableState) -> Action:
        self._events = []
        gpu = state.gpu_states[0] if state.gpu_states else None
        n_active_before = len(gpu.active_request_ids) if gpu else 0
        n_tenants_backlogged = sum(1 for dq in self._vtc.user_req_list.values() if len(dq) > 0)

        # What plain FCFS (oldest waiting request, ties by request_id)
        # would pick next -- computed BEFORE calling the real
        # select_action, from the same state.waiting_queue every policy
        # sees, so this is a fair, independent reference point.
        fifo_pick_tenant: Optional[str] = None
        if state.waiting_queue:
            oldest = min(state.waiting_queue, key=lambda r: (r.arrival_time, r.request_id))
            fifo_pick_tenant = self._tenant_of(oldest)

        action = super().select_action(state)

        vtc_pick_tenant = self._events[0][1] if self._events else None

        candidates: List[Tuple[int, str, bool, str]] = []
        admitted_ids = set(action.admit.get(gpu.gpu_id, [])) if gpu else set()
        for rid, tenant, ok in self._events:
            if ok:
                reason = "admitted"
            else:
                # Distinguish reservation vs. batch-token-budget rejection:
                # _can_add_new_req itself returning False is the
                # reservation gate; the compound `and` in the official
                # source also rejects on batch_max_tokens overflow even
                # when _can_add_new_req returns True. Our wrapper only
                # observes _can_add_new_req's own return value, so a
                # "True" here with the request absent from admitted_ids
                # means the SECOND (budget) half of the official `and`
                # rejected it -- distinguished below.
                reason = "reservation_rejected"
            candidates.append((rid, tenant, ok, reason))

        # Re-attribute budget-only rejections: _can_add_new_req said True
        # but the request never made it into admitted_ids and it was the
        # last event (matches the official loop's `else: break` on the
        # combined condition).
        if candidates and candidates[-1][2] is True and candidates[-1][0] not in admitted_ids:
            rid, tenant, ok, _ = candidates[-1]
            candidates[-1] = (rid, tenant, ok, "budget_rejected")

        self.step_log.append(StepDecision(
            step_index=len(self.step_log),
            time=state.time,
            n_tenants_backlogged=n_tenants_backlogged,
            n_active_before=n_active_before,
            candidates_tried=candidates,
            admitted_request_ids=sorted(admitted_ids),
            vtc_first_pick_tenant=vtc_pick_tenant,
            fifo_first_pick_tenant=fifo_pick_tenant,
        ))
        return action

    # ------------------------------------------------------------------
    def decomposition_summary(self) -> Dict[str, float]:
        """Aggregate, human-readable decomposition of this run's steps."""
        n_steps = len(self.step_log)
        n_reservation_stopped = sum(1 for s in self.step_log if s.stopped_by_reservation)
        n_budget_stopped = sum(1 for s in self.step_log if s.stopped_by_budget)
        n_unexplored_backlog = sum(1 for s in self.step_log if s.had_unexplored_backlog)
        n_ordering_governed = sum(
            1 for s in self.step_log
            if s.candidates_tried and not s.stopped_by_reservation and not s.stopped_by_budget
        )
        n_contended_steps = sum(1 for s in self.step_log if s.n_tenants_backlogged >= 2)
        n_disagreements = sum(1 for s in self.step_log if s.disagrees_with_fifo)
        return {
            "n_steps": n_steps,
            "n_contended_steps": n_contended_steps,
            "contention_rate": n_contended_steps / n_steps if n_steps else float("nan"),
            "n_reservation_stopped_steps": n_reservation_stopped,
            "reservation_bind_rate": n_reservation_stopped / n_steps if n_steps else float("nan"),
            "n_budget_stopped_steps": n_budget_stopped,
            "budget_bind_rate": n_budget_stopped / n_steps if n_steps else float("nan"),
            "n_unexplored_backlog_steps": n_unexplored_backlog,
            "unexplored_backlog_rate": n_unexplored_backlog / n_steps if n_steps else float("nan"),
            "n_ordering_governed_steps": n_ordering_governed,
            "ordering_governed_rate": n_ordering_governed / n_steps if n_steps else float("nan"),
            "n_decision_disagreements": n_disagreements,
            "decision_disagreement_rate": (
                n_disagreements / n_contended_steps if n_contended_steps else float("nan")
            ),
        }
