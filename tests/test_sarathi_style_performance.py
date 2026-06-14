"""
Performance and correctness regression tests for SarathiStylePolicy.

These tests guard against the O(N²) set-rebuild bug that caused sarathi_style to
hang on the burstgpt_scaled_moderate trace: the inner loop was rebuilding
{rid for gpu_id, rids in admit.items() for rid in rids} on every request,
turning each call to select_action into O(queue² × gpus) work.
"""
from __future__ import annotations

import time

import pytest

from llmserveopt.core.types import ObservableGPUState, ObservableRequest, ObservableState
from llmserveopt.policies.sarathi_style import SarathiStylePolicy


def _make_queue(n: int, prompt_tokens: int = 64) -> list[ObservableRequest]:
    return [
        ObservableRequest(
            request_id=i,
            arrival_time=float(i) * 0.001,
            prompt_tokens=prompt_tokens,
            predicted_output_tokens=128,
            slo_deadline=float(i) * 0.001 + 60.0,
            priority=1.0,
            class_id="medium",
        )
        for i in range(n)
    ]


def _make_gpu(
    gpu_id: int = 0,
    max_active: int = 8,
    max_kv: int = 4096,
    max_batch: int = 512,
    active_ids: list[int] | None = None,
    current_kv: int = 0,
) -> ObservableGPUState:
    active_ids = active_ids or []
    return ObservableGPUState(
        gpu_id=gpu_id,
        max_active_sequences=max_active,
        max_batch_tokens=max_batch,
        max_kv_tokens=max_kv,
        active_request_ids=list(active_ids),
        active_requests_info=[],
        current_kv_tokens=current_kv,
        tokens_decoded_per_request={},
    )


def _make_state(queue: list[ObservableRequest], gpus: list[ObservableGPUState]) -> ObservableState:
    return ObservableState(
        time=0.0,
        waiting_queue=queue,
        gpu_states=gpus,
        completed_count=0,
        step=0,
    )


class TestSarathiStylePerformance:
    """select_action must be fast even with a large persistent queue."""

    def test_large_queue_completes_quickly(self):
        """
        1 000 pending requests × 500 calls to select_action must finish in < 3 s.

        Before the fix this was O(N²) per call; with N=1000 and 500 calls
        that's 500M iterations — it never finishes in a reasonable time.
        After the fix it is O(N) per call and should complete in well under 1 s.
        """
        policy = SarathiStylePolicy(max_prefill_tokens_per_step=512)
        queue = _make_queue(n=1000, prompt_tokens=64)
        gpu = _make_gpu(max_active=2, max_kv=256, max_batch=512)

        start = time.monotonic()
        for _ in range(500):
            state = _make_state(list(queue), [_make_gpu(max_active=2, max_kv=256, max_batch=512)])
            policy.select_action(state)
        elapsed = time.monotonic() - start

        assert elapsed < 3.0, (
            f"select_action took {elapsed:.2f}s for 500 calls with queue=1000 "
            f"— O(N²) regression suspected"
        )

    def test_no_duplicate_assignments_single_gpu(self):
        """No request should appear twice in the admit list for a single GPU."""
        policy = SarathiStylePolicy(max_prefill_tokens_per_step=512)
        queue = _make_queue(n=200, prompt_tokens=32)
        state = _make_state(queue, [_make_gpu(max_active=16, max_kv=8192, max_batch=2048)])

        action = policy.select_action(state)
        for gpu_id, rids in action.admit.items():
            assert len(rids) == len(set(rids)), (
                f"GPU {gpu_id} has duplicate request IDs in admit: {rids}"
            )

    def test_no_duplicate_assignments_multi_gpu(self):
        """No request should be admitted to more than one GPU."""
        policy = SarathiStylePolicy(max_prefill_tokens_per_step=512)
        queue = _make_queue(n=200, prompt_tokens=32)
        gpus = [
            _make_gpu(gpu_id=0, max_active=8, max_kv=4096, max_batch=1024),
            _make_gpu(gpu_id=1, max_active=8, max_kv=4096, max_batch=1024),
        ]
        state = _make_state(queue, gpus)

        action = policy.select_action(state)
        all_admitted: list[int] = []
        for rids in action.admit.values():
            all_admitted.extend(rids)
        assert len(all_admitted) == len(set(all_admitted)), (
            "The same request was admitted to multiple GPUs: "
            f"duplicates = {[r for r in all_admitted if all_admitted.count(r) > 1]}"
        )

    def test_feasibility_kv_limit(self):
        """Admitted requests must not exceed max_kv_tokens."""
        policy = SarathiStylePolicy(max_prefill_tokens_per_step=2048)
        # Large prompt tokens; GPU KV limit is 512
        queue = _make_queue(n=50, prompt_tokens=128)
        gpu = _make_gpu(gpu_id=0, max_active=32, max_kv=512, max_batch=4096)
        state = _make_state(queue, [gpu])

        action = policy.select_action(state)
        rids = action.admit[0]
        admitted_kv = sum(
            req.prompt_tokens for req in queue if req.request_id in set(rids)
        )
        assert admitted_kv <= 512, (
            f"KV limit violated: admitted {admitted_kv} tokens but max is 512"
        )

    def test_feasibility_sequence_limit(self):
        """Admitted requests must not exceed max_active_sequences."""
        policy = SarathiStylePolicy(max_prefill_tokens_per_step=8192)
        queue = _make_queue(n=100, prompt_tokens=32)
        gpu = _make_gpu(gpu_id=0, max_active=4, max_kv=65536, max_batch=65536)
        state = _make_state(queue, [gpu])

        action = policy.select_action(state)
        assert len(action.admit[0]) <= 4, (
            f"Sequence limit violated: admitted {len(action.admit[0])} but max is 4"
        )

    def test_empty_queue_returns_empty_action(self):
        """An empty waiting queue should produce an admit dict with empty lists."""
        policy = SarathiStylePolicy()
        gpu = _make_gpu(gpu_id=0)
        state = _make_state([], [gpu])

        action = policy.select_action(state)
        assert action.admit == {0: []}, f"Expected {{0: []}}, got {action.admit}"

    def test_chunk_budget_halved_with_active_decodes(self):
        """
        When the GPU has active sequences the chunk budget is halved.

        The safety valve means the *first* queue entry is always admitted (to
        prevent starvation).  The halved budget only blocks *subsequent* requests
        that would push the total over the limit.  Verify that on a busy GPU the
        second request is blocked when it would exceed the halved budget, while on
        an idle GPU both requests fit within the full budget.
        """
        policy = SarathiStylePolicy(max_prefill_tokens_per_step=512)
        # Two requests: each 200 tokens.  Full budget=512 fits both (400 ≤ 512).
        # Halved budget=256 fits only one (first=200 ≤ 256, first+second=400 > 256).
        queue = [
            ObservableRequest(
                request_id=i, arrival_time=float(i) * 0.001, prompt_tokens=200,
                predicted_output_tokens=50, slo_deadline=100.0,
                priority=1.0, class_id="medium",
            )
            for i in range(2)
        ]

        # GPU busy → halved budget = 256: only 1 request should be admitted
        gpu_busy = _make_gpu(
            gpu_id=0, max_active=16, max_kv=65536, max_batch=65536,
            active_ids=[999], current_kv=10,
        )
        state_busy = _make_state(list(queue), [gpu_busy])
        action_busy = policy.select_action(state_busy)
        assert len(action_busy.admit[0]) == 1, (
            f"Expected 1 admission on busy GPU (halved budget=256, 200+200>256), "
            f"got {len(action_busy.admit[0])}"
        )

        # GPU idle → full budget = 512: both requests (400 tokens) should be admitted
        gpu_idle = _make_gpu(gpu_id=0, max_active=16, max_kv=65536, max_batch=65536)
        state_idle = _make_state(list(queue), [gpu_idle])
        action_idle = policy.select_action(state_idle)
        assert len(action_idle.admit[0]) == 2, (
            f"Expected 2 admissions on idle GPU (full budget=512, 200+200=400≤512), "
            f"got {len(action_idle.admit[0])}"
        )
