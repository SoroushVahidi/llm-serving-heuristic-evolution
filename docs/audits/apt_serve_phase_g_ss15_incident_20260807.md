# Apt-Serve Phase G SS15 Incident: Diagnosis and Fix

**Date:** 2026-08-07
**Auditor:** Claude Sonnet 5
**Status:** Root cause identified (two independent bugs), both fixed,
regression-tested, and validated against a compact adversarial stress
grid with zero invariant violations. **The correctness fix is safe to
commit. Resuming the actual overnight sweep is a separate, out-of-scope
follow-up action — this incident does not itself constitute Phase G
completion.**

---

## 1. What happened

The overnight sweep launched in `apt_phase_g_overnight_20260807`
(commit `342a8e0`, run dir
`results/apt_serve_phase_g_overnight_20260807_011542/`) self-terminated
after 152/1599 Stage-1 screening units (9.5%), per its own documented
SS15 rule (`scripts/run_apt_serve_phase_g.py`): on detecting a
systematic Apt-Serve invariant violation, the runner deliberately halts
the entire sweep rather than continue producing data downstream of a
possibly-corrupted mechanism, and writes out whatever was collected so
far. `wrapper_meta.txt` recorded `exit_code=1`.

The triggering failure (`failures.jsonl`, the run's only entry):

```
AptServeCapacityViolation: Transition failed for request 11 to tier
CacheTier.HIDDEN: Insufficient destination hidden capacity
regime: pressure_sustained_overload_baseline, seed=1005,
transition_cost=4x, stage=screening
```

This artifact is preserved as-is at
`results/apt_serve_phase_g_overnight_20260807_011542/` and was not
modified, resumed, or deleted as part of this investigation.

The transactional rollback mechanism worked exactly as designed: the
manager's state was restored from the pre-decision deepcopy backup with
no corruption, and `docs/current/WORK_STATUS.md` / `docs/BASELINE_STATUS.md`
correctly stopped at `PHASE_G_EXPERIMENT_RUNNING` rather than asserting
a scientific conclusion the run itself didn't verify.

## 2. Root cause: two independent bugs, not one

### Bug 1 — transition application order in the adapter (fixed)

`src/llmserveopt/policies/apt_serve_faithful.py`'s `select_action`
applied `decision.cache_assignments` (the client's proposed tier moves)
in **raw Python dict iteration order**. That order reflects each
request's *first touch* in the client's own two-pass bookkeeping
(`scripts/apt_serve/fake_scheduler_worker.py`), not the order its
internal ledger actually depended on. In the failing step, the client's
own accounting was fully self-consistent: it decided to restore request
5 from HIDDEN→KV (freeing 5 hidden blocks, 31/31 used → 26/31), *then*
used that freed room to move request 11 KV→HIDDEN (26+4=30 ≤ 31, fits).
But request 11 had been provisionally touched (and left at "kv") earlier
in the client's pass-1 logic, so its dict key sits *before* request 5's
key — even though request 11's *final* value ("hidden") was only decided
in pass 2, after request 5's release. The adapter, iterating the dict in
that raw order, applied request 11's HIDDEN-consuming move before
request 5's HIDDEN-releasing move, saw only 1 free block instead of 6,
and raised.

Deterministic reproduction: `pressure_sustained_overload_baseline`,
seed=1005, transition_cost=4x, step 35 (see
`tests/test_apt_serve_phase_g.py::test_ss15_known_failing_cell_now_completes`).

### Bug 2 — missing feasibility check in the fake scheduler's preemption fallback (fixed)

Independently, `fake_scheduler_worker.py`'s "evict one relaxed KV
resident to make room for an urgent waiting admission" fallback checked
only that the evicted resident's hidden-tier footprint fit in hidden
capacity — it never checked that the *KV blocks freed by that one
eviction* actually covered the urgent request's need. When they didn't
(e.g. evicting an 8-block resident to admit a 20-block request), the
client still committed the eviction-and-admit, producing a decision that
overcommitted KV capacity even by the client's own ledger. Downstream,
the adapter could only catch this as a raw `KVBlockManagerError` ("Out
of memory") when it tried to actually allocate the new admission — a
correctness bug in the client, not the adapter.

This is a second, distinct failure mode from Bug 1 and did not surface
until Bug 1 was fixed and the simulation could run far enough to reach
it (see `tests/test_apt_serve_phase_g.py::test_fake_worker_preemption_requires_eviction_to_cover_admission`).

### Bug 3 — mutual capacity dependency, found by the post-fix stress grid (fixed)

An initial fix for Bug 1 reordered the adapter's transition loop
statically: apply all HIDDEN→KV releases before all KV→HIDDEN
acquisitions. This resolved the known failing cell, but the compact
pre-resume stress grid (§4) surfaced a **new** failure at
`pressure_sustained_overload_baseline`, seed=9002, all three non-trivial
transition costs: a HIDDEN→KV restore (request 4) needing KV room that
only became free once *other* requests (11, 12) moved KV→HIDDEN in the
*same* decision — exactly the opposite dependency direction from Bug 1,
in the same step. No single static per-request ordering can satisfy
both directions of a mutual dependency at once.

The general fix: decouple every transition's *release* from its
*acquire* entirely. `HybridCacheManager.switch_tier` was split into
`begin_transition_release` (frees the source-tier allocation only) and
`finish_transition_acquire` (allocates the destination-tier, sized from
data captured at release time). The adapter's section 5b now runs a
strict two-phase loop: release every transitioning request first, then
acquire every destination allocation — so by the time any acquisition is
attempted, every release in the batch has already happened, regardless
of which request depends on which. `switch_tier` itself is now a thin
wrapper over the two phases (restoring the release on failure) so its
existing single-call atomic-rollback contract
(`tests/test_apt_serve_phase_b.py::test_transition_insufficient_capacity_rollback`)
is unchanged.

## 3. Fix

**Files changed:**
- `src/llmserveopt/simulator/hybrid_cache_manager.py` — added
  `begin_transition_release`, `finish_transition_acquire`,
  `_restore_release`; `switch_tier` refactored into a thin wrapper over
  the two new methods (external behavior unchanged, verified by existing
  Phase A/B tests).
- `src/llmserveopt/policies/apt_serve_faithful.py` — section 5b of
  `select_action`'s decision-application transaction now runs the
  release phase over every pending transition before the acquire phase,
  instead of iterating `decision.cache_assignments` directly.
- `scripts/apt_serve/fake_scheduler_worker.py` — the urgent-preemption
  fallback now requires `current_kv_blocks - evict_kv_blocks + kv_blocks
  <= max_kv_blocks` before committing to an eviction-and-admit; if not,
  the request is queued (deprioritized) instead, matching the existing
  "no capacity" fallback behavior.

**What was deliberately preserved:**
- SS15 itself (`scripts/run_apt_serve_phase_g.py`'s hard-stop-on-critical-
  invariant-violation behavior) — unchanged.
- Transaction ordering across sections (evictions → transitions →
  admissions → invariant validation) — unchanged.
- Rollback exactness — the deepcopy-backup rollback in
  `select_action` still restores the manager wholesale on any exception;
  verified by `test_genuinely_infeasible_decision_still_rolls_back_exactly`.
- No transitions are silently dropped or clipped — every request named
  in a feasible decision still ends up at its requested tier
  (`test_no_transitions_silently_dropped`). A genuinely infeasible
  decision (net capacity effect negative even under correct sequencing)
  still raises `AptServeCapacityViolation` and still hard-stops the
  sweep via SS15 — this fix narrows *false positives* (feasible
  decisions rejected due to bad application order), it does not weaken
  the invariant check itself.

## 4. Validation

**Exact failing cell:** `pressure_sustained_overload_baseline`,
seed=1005, transition_cost=4x — now completes (36/36 requests, 0
dropped).

**Second failing cell found during validation:**
`pressure_sustained_overload_baseline`, seed=9002, transition_cost ∈
{1x, 2x, 4x} — now completes (36/36, 0 dropped) after the Bug 3 fix.

**Compact adversarial stress grid** (not a Phase G resume — a targeted
mechanism test): `pressure_near_capacity_baseline`,
`pressure_sustained_overload_baseline`, and
`cacheuse_kv_to_hidden_opportunity_near_capacity`, × seeds {1004, 1005,
1006, 9001, 9002, 9003}, × transition costs {1x, 2x, 4x}, apt_serve_faithful
only = 54 cells.
- Before Bug 3's fix: 51/54 OK, 3 FAIL (the seed=9002 mutual-dependency case).
- After Bug 3's fix: **54/54 OK, 0 FAIL**, all 36/36 requests completed, 0 dropped.

**Regression tests added** (`tests/test_apt_serve_phase_e.py`,
`tests/test_apt_serve_phase_g.py`): transition-order-dependent feasible
decisions (both directions), order-independence, mixed eviction +
transition + admission batches, exact-capacity boundary, genuinely
infeasible decisions still roll back exactly, no transitions silently
dropped, the exact failing cell, neighboring seeds/costs, `run_one_unit`
reporting no critical failure, and the fake-worker preemption fix
in isolation. All new tests fail against the pre-fix code and pass
against the fix (verified by temporarily reverting the adapter fix and
re-running).

**Full suite:** 3610 passed, 0 failed (excluding one working-tree-
cleanliness checker test that fails only because this fix was
uncommitted at test time and passes once committed), 62 skipped.

## 5. Performance anomaly (investigated, no fix needed)

The overnight run's `run.log` continued growing for ~90 minutes after
`progress.json`'s last update, and contained 114,482 "admission
rejected: max_kv_tokens exceeded" warnings. Measured
`warnings.warn()` throughput is ~530,000/s, so the warning volume itself
accounts for well under one second — logging overhead is not the cause.
The far more likely explanation: `run_stage()` calls
`ex.shutdown(wait=False, cancel_futures=True)` on detecting the critical
failure, which only cancels *not-yet-started* queued units. The
`ProcessPoolExecutor`'s context-manager `__exit__` still blocks
(`wait=True`) on the up-to-9 other worker processes already mid-flight
on other (regime, seed) pairs, which continue running to completion
before `finalize()` can execute. One of those concurrently-running units
was very likely a genuinely slower regime/seed combination than the
~130s/unit pilot average. This is a structural consequence of using the
executor's default safe shutdown (favoring not losing in-flight work
over fast abort), not a bug, and no change was made.

## 6. Recommended next action

Phase G's correctness blocker is resolved and validated. The overnight
sweep is safe to **resume** from the same run directory once launched
again (as a separate action, not part of this fix) — the partial data
in `results/apt_serve_phase_g_overnight_20260807_011542/` remains valid
only as partial descriptive output per its own `final_summary.json` note,
and should not be treated as a completed or scientifically conclusive
run. Do not mark Phase G complete; that requires the full resumed sweep
plus the deferred bootstrap-CI/mechanism-correlation analysis pass.
