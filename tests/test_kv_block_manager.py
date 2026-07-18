"""Tests for src/llmserveopt/simulator/kv_block_manager.py.

Covers the reusable KV block/page infrastructure independently reimplemented
from vLLM v0.1.0 (see docs/vllm_faithful_scheduler_reference.md). This
module is new and opt-in: none of these tests touch GPUState/Simulator, so
they cannot regress any existing legacy-KV-token behavior.
"""
from __future__ import annotations

import pytest

from llmserveopt.simulator.kv_block_manager import (
    KVBlock,
    KVBlockAllocator,
    KVBlockManagerError,
    KVBlockSpaceManager,
)


# ---------------------------------------------------------------------------
# KVBlockAllocator (low-level free-list allocator)
# ---------------------------------------------------------------------------

def test_allocator_initial_free_count():
    alloc = KVBlockAllocator(num_blocks=4)
    assert alloc.num_free_blocks == 4
    assert alloc.num_used_blocks == 0


def test_allocator_allocate_decrements_free_count():
    alloc = KVBlockAllocator(num_blocks=4)
    block = alloc.allocate()
    assert isinstance(block, KVBlock)
    assert block.ref_count == 1
    assert alloc.num_free_blocks == 3
    assert alloc.num_used_blocks == 1


def test_allocator_exact_capacity_succeeds():
    alloc = KVBlockAllocator(num_blocks=3)
    blocks = [alloc.allocate() for _ in range(3)]
    assert alloc.num_free_blocks == 0
    assert len({b.block_number for b in blocks}) == 3, "block numbers must be unique"


def test_allocator_over_capacity_rejected():
    alloc = KVBlockAllocator(num_blocks=2)
    alloc.allocate()
    alloc.allocate()
    with pytest.raises(KVBlockManagerError, match="Out of memory"):
        alloc.allocate()


def test_allocator_free_returns_block_to_free_list():
    alloc = KVBlockAllocator(num_blocks=2)
    block = alloc.allocate()
    assert alloc.num_free_blocks == 1
    alloc.free(block)
    assert alloc.num_free_blocks == 2
    assert alloc.num_used_blocks == 0


def test_allocator_no_double_free():
    alloc = KVBlockAllocator(num_blocks=2)
    block = alloc.allocate()
    alloc.free(block)
    with pytest.raises(KVBlockManagerError, match="Double free"):
        alloc.free(block)


def test_allocator_repeated_allocate_free_cycles_are_stable():
    alloc = KVBlockAllocator(num_blocks=3)
    for _ in range(50):
        blocks = [alloc.allocate() for _ in range(3)]
        assert alloc.num_free_blocks == 0
        for b in blocks:
            alloc.free(b)
        assert alloc.num_free_blocks == 3
    assert alloc.num_used_blocks == 0


def test_allocator_deterministic_block_reuse_order():
    """Free-list is a stack (LIFO): the most recently freed block is the
    next one allocated. This must be exactly reproducible."""
    alloc = KVBlockAllocator(num_blocks=3)
    _b0, b1, _b2 = alloc.allocate(), alloc.allocate(), alloc.allocate()
    alloc.free(b1)
    reused = alloc.allocate()
    assert reused.block_number == b1.block_number


# ---------------------------------------------------------------------------
# KVBlockSpaceManager (per-request logical block table + capacity checks)
# ---------------------------------------------------------------------------

def test_blocks_needed_ceil_division():
    assert KVBlockSpaceManager.blocks_needed(0, block_size=16) == 0
    assert KVBlockSpaceManager.blocks_needed(1, block_size=16) == 1
    assert KVBlockSpaceManager.blocks_needed(16, block_size=16) == 1
    assert KVBlockSpaceManager.blocks_needed(17, block_size=16) == 2
    assert KVBlockSpaceManager.blocks_needed(32, block_size=16) == 2
    assert KVBlockSpaceManager.blocks_needed(33, block_size=16) == 3


def test_can_allocate_respects_watermark():
    # 10 blocks, watermark=0.1 -> watermark_blocks = 1.
    mgr = KVBlockSpaceManager(block_size=4, num_gpu_blocks=10, watermark=0.1)
    assert mgr.watermark_blocks == 1
    # Requesting 9 blocks worth (36 tokens) would leave exactly 1 free block,
    # satisfying `free - needed >= watermark_blocks` (10 - 9 = 1 >= 1).
    assert mgr.can_allocate(36) is True
    # Requesting 10 blocks worth would leave 0 free, violating the watermark.
    assert mgr.can_allocate(40) is False


def test_allocate_exact_capacity():
    mgr = KVBlockSpaceManager(block_size=4, num_gpu_blocks=4, watermark=0.0)
    mgr.allocate(request_id=1, prompt_tokens=16)  # exactly 4 blocks
    assert mgr.num_blocks_for(1) == 4
    assert mgr.num_free_blocks == 0
    assert mgr.kv_tokens_for(1) == 16
    assert mgr.allocated_kv_capacity_for(1) == 16  # no fragmentation, exact fit


def test_allocate_over_capacity_raises():
    mgr = KVBlockSpaceManager(block_size=4, num_gpu_blocks=2, watermark=0.0)
    with pytest.raises(KVBlockManagerError, match="Out of memory"):
        mgr.allocate(request_id=1, prompt_tokens=100)  # needs 25 blocks, only 2 exist


def test_allocate_twice_for_same_request_raises():
    mgr = KVBlockSpaceManager(block_size=4, num_gpu_blocks=10, watermark=0.0)
    mgr.allocate(request_id=1, prompt_tokens=4)
    with pytest.raises(KVBlockManagerError, match="already has an allocation"):
        mgr.allocate(request_id=1, prompt_tokens=4)


def test_internal_fragmentation_accounting():
    mgr = KVBlockSpaceManager(block_size=16, num_gpu_blocks=10, watermark=0.0)
    mgr.allocate(request_id=1, prompt_tokens=17)  # needs 2 blocks (32 capacity), 15 wasted
    assert mgr.num_blocks_for(1) == 2
    assert mgr.allocated_kv_capacity_for(1) == 32
    assert mgr.internal_fragmentation_tokens() == 32 - 17 == 15

    mgr.allocate(request_id=2, prompt_tokens=16)  # exact fit, 0 wasted
    assert mgr.internal_fragmentation_tokens() == 15 + 0


def test_freeing_releases_blocks_back_to_pool():
    mgr = KVBlockSpaceManager(block_size=4, num_gpu_blocks=4, watermark=0.0)
    mgr.allocate(request_id=1, prompt_tokens=16)
    assert mgr.num_free_blocks == 0
    mgr.free(1)
    assert mgr.num_free_blocks == 4
    assert not mgr.is_allocated(1)


def test_allocated_request_ids_tracks_current_allocations():
    mgr = KVBlockSpaceManager(block_size=4, num_gpu_blocks=8, watermark=0.0)
    assert mgr.allocated_request_ids() == []
    mgr.allocate(request_id=1, prompt_tokens=4)
    mgr.allocate(request_id=2, prompt_tokens=4)
    assert sorted(mgr.allocated_request_ids()) == [1, 2]
    mgr.free(1)
    assert mgr.allocated_request_ids() == [2]


def test_free_unknown_request_is_a_noop():
    mgr = KVBlockSpaceManager(block_size=4, num_gpu_blocks=4, watermark=0.0)
    mgr.free(999)  # never allocated -- must not raise
    mgr.allocate(request_id=1, prompt_tokens=4)
    mgr.free(1)
    mgr.free(1)  # already freed -- must not raise (unlike the raw allocator)


def test_repeated_allocate_free_cycles_no_leak():
    mgr = KVBlockSpaceManager(block_size=4, num_gpu_blocks=8, watermark=0.0)
    for i in range(100):
        mgr.allocate(request_id=i, prompt_tokens=32)  # exactly 8 blocks
        assert mgr.num_free_blocks == 0
        mgr.free(i)
        assert mgr.num_free_blocks == 8
    assert mgr.internal_fragmentation_tokens() == 0


def test_decode_time_growth_no_new_block_within_last_block():
    mgr = KVBlockSpaceManager(block_size=4, num_gpu_blocks=4, watermark=0.0)
    mgr.allocate(request_id=1, prompt_tokens=2)  # 1 block, 2 tokens used of 4
    assert mgr.num_blocks_for(1) == 1
    assert mgr.can_append_slot(1) is True
    mgr.append_slot(1)  # token 3 of 4 -- still fits in the same block
    assert mgr.num_blocks_for(1) == 1
    assert mgr.num_free_blocks == 3  # no new block consumed


def test_decode_time_growth_allocates_new_block_when_full():
    mgr = KVBlockSpaceManager(block_size=4, num_gpu_blocks=4, watermark=0.0)
    mgr.allocate(request_id=1, prompt_tokens=4)  # 1 block, exactly full
    assert mgr.num_free_blocks == 3
    assert mgr.can_append_slot(1) is True  # 3 free blocks available
    mgr.append_slot(1)  # 5th token -> needs a 2nd block
    assert mgr.num_blocks_for(1) == 2
    assert mgr.num_free_blocks == 2
    assert mgr.kv_tokens_for(1) == 5


def test_decode_time_growth_blocked_when_no_free_block_for_new_block():
    mgr = KVBlockSpaceManager(block_size=4, num_gpu_blocks=1, watermark=0.0)
    mgr.allocate(request_id=1, prompt_tokens=4)  # uses the only block, exactly full
    assert mgr.num_free_blocks == 0
    assert mgr.can_append_slot(1) is False  # would need a 2nd block; none free


def test_reset_frees_all_requests():
    mgr = KVBlockSpaceManager(block_size=4, num_gpu_blocks=8, watermark=0.0)
    mgr.allocate(request_id=1, prompt_tokens=4)
    mgr.allocate(request_id=2, prompt_tokens=8)
    assert mgr.num_free_blocks == 5  # 1 block for req 1 + 2 blocks for req 2, 8-3=5
    mgr.reset()
    assert mgr.num_free_blocks == 8
    assert not mgr.is_allocated(1)
    assert not mgr.is_allocated(2)


def test_deterministic_across_repeated_runs():
    """Same sequence of operations on freshly constructed managers must
    produce byte-identical accounting every time (no hidden randomness,
    no dict-ordering dependence)."""
    def run() -> tuple:
        mgr = KVBlockSpaceManager(block_size=4, num_gpu_blocks=8, watermark=0.0)
        mgr.allocate(request_id=1, prompt_tokens=5)
        mgr.allocate(request_id=2, prompt_tokens=3)
        mgr.append_slot(1)
        mgr.free(2)
        mgr.allocate(request_id=3, prompt_tokens=10)
        return (
            mgr.num_free_blocks,
            mgr.num_blocks_for(1),
            mgr.num_blocks_for(3),
            mgr.internal_fragmentation_tokens(),
        )

    results = [run() for _ in range(5)]
    assert len(set(results)) == 1, f"non-deterministic results: {results}"


def test_constructor_rejects_invalid_watermark():
    with pytest.raises(ValueError):
        KVBlockSpaceManager(block_size=4, num_gpu_blocks=4, watermark=1.0)
    with pytest.raises(ValueError):
        KVBlockSpaceManager(block_size=4, num_gpu_blocks=4, watermark=-0.1)


def test_constructor_rejects_invalid_block_size():
    with pytest.raises(ValueError):
        KVBlockSpaceManager(block_size=0, num_gpu_blocks=4, watermark=0.0)
