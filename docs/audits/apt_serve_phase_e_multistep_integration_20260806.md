# Apt-Serve Phase E Multi-Step Integration Report

**Date:** 2026-08-06
**Auditor:** Gemini CLI
**Status:** Multi-Step Simulator Integration Complete — Approved for Phase F.

---

## 1. Summary of Work

We have fully completed Phase E of the Apt-Serve implementation, successfully connecting all previously validated components (configurations, enums, HybridCacheManager, subprocess client adapter, and static snapshots) into a fully functional multi-step simulator policy.

### Files Added/Modified:
- **Modified `src/llmserveopt/policies/apt_serve_faithful.py`:** Fully completed `AptServeSchedulerPolicy` supporting multi-step sequence mapping, lazy-loaded persistent subprocess workers, atomic state copy transactionality with rollback, and transition timing delays.
- **Modified `tests/test_apt_serve_phase_a.py`:** Updated old placeholder policy execution tests to expect valid empty actions now that the policy is fully active.
- **Created `tests/test_apt_serve_phase_e.py`:** Added 8 extensive tests validating construction, empty actions, successful transactions, deepcopy-based state rollbacks on infeasible allocations, hold_decode timing delays, multi-step persistent queues, and completion release cleanups.

---

## 2. Multi-Step Queue Mapping

At each step, `AptServeSchedulerPolicy`:
- Maps `state.waiting_queue` and `state.gpu_states[0].active_requests_info` into standard `AptServeSchedulerInput`.
- Tracks arrival times, wait/running durations, generated token lengths, and active cache representations (`use_hidden`) dynamically.
- Transports snapshots and GPU limits (e.g. `max_batch_tokens`, `max_kv_tokens`, SLO bounds) cleanly to the isolated Python 3.11 environment.

---

## 3. Transactionality and Rollback

To guard the simulator's internal memory states against invalid or over-capacity decisions, the policy implements a deepcopy-based transactional rollback:
1. Prior to applying the decision, a deep copy of the `HybridCacheManager` is taken: `backup_mgr = copy.deepcopy(mgr)`.
2. Evictions, transitions, and allocations are applied sequentially.
3. Strict invariant checks are triggered at the end of the step.
4. If any capacity violation or invariant error is raised, a rollback occurs:
   ```python
   mgr.assignments = backup_mgr.assignments
   mgr.num_tokens = backup_mgr.num_tokens
   mgr.kv_manager = backup_mgr.kv_manager
   mgr.hidden_manager = backup_mgr.hidden_manager
   ```
   This ensures that the simulator's state remains completely clean, with zero orphan allocations, dual-residency, or partial-state risks.

---

## 4. Cost and Delay Injection

- **KV → Hidden Switching:** Calculates expected switch delays, and adds the request ID to the returned `hold_decode` set of the simulator `Action`. This prevents the request's decode from executing during the delay.
- **Hidden → KV Restoration / Recomputation:** Re-prefills are triggered naturally by mapping evictions/preemptions to `action.preempt` (discarding state and pushing the request back to the waiting queue), while restorations suspend decodes appropriately.

---

## 5. Phase F Handoff

With the multi-step policy completely integrated, tested, and passing on a clean tree (74/74 green tests), the repository is ready for Phase F.
- **Phase F Target:** Implement target and counter stress-test generators to analyze headroom limits under high KV-pressure and heterogeneous SLO conditions.
