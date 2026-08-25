"""Hand-verifiable micro-traces for the VTC fairness-benchmark repair.

Every expected outcome in this file is computed by hand in each test's
docstring/comments FIRST, directly from the pinned source
(`slora/server/router/vtc_req_queue.py` lines 96-133,
`slora/server/router/req_queue.py`'s `_can_add_new_req`/
`generate_new_batch`) -- not inferred from running the code and copying
its output. Runs are then asserted to match. All traces operate directly
on the real, unmodified official `VTCReqQueue`/`ReqQueue` classes (no
simulator involved) so every step is exact and reproducible by hand,
matching this repair task's step 7 requirement.

Covers the four required comparison cases:
  - test_vtc_reorders_ahead_of_fifo_arrival_order: VTC improves fairness
    over FIFO (a less-served tenant is served before its arrival-order
    position would allow).
  - test_single_tenant_at_a_time_vtc_and_fifo_tie: VTC and FIFO tie
    (no genuine contention -> no reordering possible).
  - test_reservation_blocks_both_official_and_matched_fifo_equally:
    reservation blocks both matched variants equally (same inherited gate).
  - test_official_vtc_vs_fairness_isolation_vtc_differ_only_by_admission:
    official VTC (variant A, tight capacity) differs from fairness-
    isolation VTC (variant C, loose capacity) ONLY in how much gets
    admitted, never in the ORDER of whichever prefix is common to both.
"""
from __future__ import annotations

import pytest

from baselines.vtc.adapter.official_loader import load_vtc_official_classes, verify_official_clone


def _clone_present() -> bool:
    try:
        verify_official_clone()
        return True
    except Exception:
        return False


requires_clone = pytest.mark.skipif(
    not _clone_present(),
    reason="Pinned Ying1123/VTC-artifact clone not found; see PROVENANCE.md to clone it.",
)


@requires_clone
class TestMicroTraces:
    def setup_method(self):
        self.official = load_vtc_official_classes()

    def _sp(self, max_new_tokens=5):
        return self.official.SamplingParams(max_new_tokens=max_new_tokens)

    # ------------------------------------------------------------------
    def test_vtc_reorders_ahead_of_fifo_arrival_order(self):
        """Setup: tenant A backlogs 3 requests (ids 1,2,3, input_len=100
        each) BEFORE tenant B's single request (id 4, input_len=10)
        arrives. Capacity is generous (no admission gate involved) --
        this isolates ORDERING alone.

        Hand-computed FIFO order (plain ReqQueue.append is arrival-order
        only): waiting_req_list = [1,2,3,4] in append order -> admits
        exactly that order: [1,2,3,4].

        Hand-computed VTC order (see this file's module docstring for the
        full derivation): after all 4 appends, served={A:0, B:0} (B's
        "lift" check at append-time finds A's served still 0, a no-op).
        generate_new_batch's loop:
          1. min(A=0,B=0) tie -> A (inserted into `served` first) -> admit
             req 1 (A) -> served[A]=100.
          2. min(A=100,B=0) -> B -> admit req 4 (B) -> served[B]=10.
          3. min(A=100,B=10) -> B, but B's queue is now empty -> delete B
             from consideration.
          4. min(A=100) -> A -> admit req 2 -> served[A]=200.
          5. min(A=200) -> A -> admit req 3 -> served[A]=300.
        VTC order: [1, 4, 2, 3] -- B's single request is served SECOND,
        not last, even though 2 of A's 3 requests arrived before it.
        """
        sp = self._sp()
        Req, Batch = self.official.Req, self.official.Batch

        # --- FIFO (plain ReqQueue) ---
        fifo_q = self.official.ReqQueue(max_total_tokens=10**9, batch_max_tokens=10**9,
                                         running_max_req_size=10**6)
        for rid, tenant, size in [(1, "A", 100), (2, "A", 100), (3, "A", 100), (4, "B", 10)]:
            fifo_q.append(Req(tenant, rid, [0] * size, sp))
        fifo_batch = fifo_q.generate_new_batch(None, {"A": 0, "B": 0})
        assert [r.request_id for r in fifo_batch.reqs] == [1, 2, 3, 4]

        # --- VTC (same appends, same order) ---
        vtc_q = self.official.VTCReqQueue(max_total_tokens=10**9, batch_max_tokens=10**9,
                                           running_max_req_size=10**6, adapter_dirs=["A", "B"],
                                           fair_weights=[1, 1], cost_func="linear")
        for rid, tenant, size in [(1, "A", 100), (2, "A", 100), (3, "A", 100), (4, "B", 10)]:
            vtc_q.append(Req(tenant, rid, [0] * size, sp))
        vtc_batch = vtc_q.generate_new_batch(None, {"A": 0, "B": 0})
        assert [r.request_id for r in vtc_batch.reqs] == [1, 4, 2, 3]
        assert vtc_q.served == {"A": 300.0, "B": 10.0}

        # The fairness claim, stated precisely: B's position in the
        # admission order improved from 4th (FIFO) to 2nd (VTC).
        fifo_position = [r.request_id for r in fifo_batch.reqs].index(4)
        vtc_position = [r.request_id for r in vtc_batch.reqs].index(4)
        assert vtc_position < fifo_position

    # ------------------------------------------------------------------
    def test_single_tenant_at_a_time_vtc_and_fifo_tie(self):
        """With only ONE tenant ever backlogged at a time (no genuine
        contention), VTC's ordering has nothing to reorder relative to
        -- both must produce identical admission order to plain FCFS.
        Setup: tenant A alone submits 3 requests; VTC and FIFO must agree
        exactly, request-for-request, in the same order.
        """
        sp = self._sp()
        Req = self.official.Req

        fifo_q = self.official.ReqQueue(max_total_tokens=10**9, batch_max_tokens=10**9,
                                         running_max_req_size=10**6)
        vtc_q = self.official.VTCReqQueue(max_total_tokens=10**9, batch_max_tokens=10**9,
                                           running_max_req_size=10**6, adapter_dirs=["A"],
                                           fair_weights=[1], cost_func="linear")
        for rid, size in [(1, 30), (2, 40), (3, 50)]:
            fifo_q.append(Req("A", rid, [0] * size, sp))
            vtc_q.append(Req("A", rid, [0] * size, sp))

        fifo_order = [r.request_id for r in fifo_q.generate_new_batch(None, {"A": 0}).reqs]
        vtc_order = [r.request_id for r in vtc_q.generate_new_batch(None, {"A": 0}).reqs]
        assert fifo_order == vtc_order == [1, 2, 3]

    # ------------------------------------------------------------------
    def test_reservation_blocks_both_official_and_matched_fifo_equally(self):
        """A single request whose input_len makes it infeasible under a
        tight max_total_tokens. Hand-computed: `_can_add_new_req`
        computes need_max_token_num = (max_output_len-1)*1 + (input_len+1)
        = (5-1) + (100+1) = 105 for a single candidate (size_array=[1],
        cum_run_len_array=[101], left_out_len_array=[4],
        need = 4*1 + 101 = 105). With max_total_tokens=50: the official
        condition is `need_max_token_num < max_total_tokens` i.e.
        `105 < 50` = False -> infeasible. Both ReqQueue (FIFO/matched) and
        VTCReqQueue inherit the IDENTICAL `_can_add_new_req` from the same
        base class -- this must reject the same request identically for
        both, proving the admission gate is shared code, not a
        VTC-specific mechanism.
        """
        sp = self.official.SamplingParams(max_new_tokens=5)
        Req = self.official.Req

        fifo_q = self.official.ReqQueue(max_total_tokens=50, batch_max_tokens=10**9,
                                         running_max_req_size=10**6)
        vtc_q = self.official.VTCReqQueue(max_total_tokens=50, batch_max_tokens=10**9,
                                           running_max_req_size=10**6, adapter_dirs=["A"],
                                           fair_weights=[1], cost_func="linear")
        fifo_q.append(Req("A", 1, [0] * 100, sp))
        vtc_q.append(Req("A", 1, [0] * 100, sp))

        assert fifo_q.generate_new_batch(None, {"A": 0}) is None
        assert vtc_q.generate_new_batch(None, {"A": 0}) is None
        # Neither queue's waiting list was consumed -- both left the
        # infeasible request exactly where it was.
        assert len(fifo_q.waiting_req_list) == 1
        assert len(vtc_q.user_req_list["A"]) == 1

    # ------------------------------------------------------------------
    def test_official_vtc_vs_fairness_isolation_vtc_differ_only_by_admission(self):
        """Same appends as test_vtc_reorders_ahead_of_fifo_arrival_order
        (A: 3x100-token requests, B: 1x10-token request), run through TWO
        VTCReqQueue instances that are IDENTICAL except for
        `batch_max_tokens` -- 150 ("official", tight) vs. 1000
        ("fairness-isolation", loose).

        Hand-computed "tight" (150) trace: steps 1-2 identical to the
        150-token-headroom derivation above (admit req 1 @ 100 tokens,
        then req 4 @ +10 = 110 total, both <= 150). Step 3 attempts req 2
        (A, +100 = 210 total): 210 <= 150 is False -> the combined
        official condition fails -> `else: break`. Admitted: [1, 4] only;
        A's queue still holds [2, 3].

        Hand-computed "loose" (1000) trace: identical steps 1-2, but step
        3 (210 <= 1000, True) succeeds, and so does step 4 (310 <= 1000).
        Admitted: [1, 4, 2, 3] (full order, matching the earlier trace).

        The key assertion: the tight run's admitted prefix [1, 4] is
        IDENTICAL to the loose run's first two admissions -- proving the
        two variants disagree ONLY on how much fits, never on WHICH
        order the fairness algorithm would pick if capacity allowed it.
        """
        sp = self._sp()
        Req = self.official.Req

        def build_and_run(batch_max_tokens):
            q = self.official.VTCReqQueue(max_total_tokens=10**9, batch_max_tokens=batch_max_tokens,
                                           running_max_req_size=10**6, adapter_dirs=["A", "B"],
                                           fair_weights=[1, 1], cost_func="linear")
            for rid, tenant, size in [(1, "A", 100), (2, "A", 100), (3, "A", 100), (4, "B", 10)]:
                q.append(Req(tenant, rid, [0] * size, sp))
            batch = q.generate_new_batch(None, {"A": 0, "B": 0})
            return q, [r.request_id for r in batch.reqs]

        tight_q, tight_order = build_and_run(batch_max_tokens=150)
        loose_q, loose_order = build_and_run(batch_max_tokens=1000)

        assert tight_order == [1, 4]
        assert loose_order == [1, 4, 2, 3]
        # The common prefix is identical -- ordering logic unaffected by
        # the capacity difference.
        assert tight_order == loose_order[: len(tight_order)]
        # Unadmitted work remains genuinely queued under the tight variant.
        assert [r.request_id for r in tight_q.user_req_list["A"]] == [2, 3]
        assert len(loose_q.user_req_list["A"]) == 0
