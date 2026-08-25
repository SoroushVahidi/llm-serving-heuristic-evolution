# Family-A Oracle State Fingerprint V1

Date: 2026-08-21

## Purpose

Audit and redesign state fingerprinting so that future DAgger / active-oracle
label reuse is scientifically safe.

This is a design and code-only task. It does not launch simulator sweeps,
generate labels, train models, run tests, or alter the running
`family_a_oracle_dataset_v1_1k` tmux job.

---

## A. Exact Current Fingerprint Contents

The current `stable_state_fingerprint` (from
`scripts/generate_family_a_oracle_policy_v1.py:435`) hashes:

```json
{
  "scenario": "<scenario_id>",
  "step": <int>,
  "estf_request": <estf_contested_request_id>,
  "wfs_request": <wfs_contested_request_id>,
  "features": {
    "feat_queue_length": 12,
    "feat_active_count": 4,
    ...  // 63 rounded feat_* values (rounded to 12 decimal places),
          // sorted by key via json.dumps(sort_keys=True)
  }
}
```

The hash is SHA-256 of the compact JSON (no whitespace separators).

---

## B. Safe for Dataset Deduplication?

**YES.**

The current fingerprint correctly deduplicates ML training rows because:
- It includes the scenario and step (unique decision-point key within a run)
- It includes contested request IDs (distinguishes ESTF vs WFS candidate identity)
- It includes all 63 feature values (the complete ML state representation)

Two D0 rows with the same fingerprint represent the same scheduling decision
state observed by the learned selector. Duplicate rejection is correct.

---

## C. Safe for Oracle-Label Reuse?

**NO.**

The current fingerprint hashes only the *projected 63-feature view* of the
simulator state, not the *complete transition-relevant state*. Two states with
identical feature vectors but differing in unobserved dimensions (queue ordering
within same quantiles, pending-arrival suffix, request progress, GPU live state
details) can produce different oracle labels because the whole-branch oracle
forks on the full simulator state.

Concrete failure modes:

1. **Pending arrivals.** Two states with identical waiting queues and features
   but different unenqueued future arrivals will produce different continuation
   branches and different oracle labels. The 63-feature vector contains no
   pending-arrival information.

2. **Queue ordering.** The quantile features collapse exact queue order into
   summary statistics. Two queues with the same quantiles but different orders
   produce the same feature fingerprint, yet ESTF/WFS sort-key tiebreaking
   (which uses request_id as final tiebreaker) may admit different requests if
   the ordering changes which requests are "first" in sorted order.

3. **Active request progress.** ObservableRequest does not include
   `tokens_decoded`, `prefill_remaining`, `first_token_time`, or
   `transfer_ready_time`. Two GPU states with identical KV utilization and
   active counts but different per-request progress produce the same features,
   yet the continuation branches will diverge in completion times and SLO
   outcomes.

4. **Migrating / relocating requests.** These are partially observable. If the
   feature extractor does not capture their state, two states with different
   migration progress will have the same fingerprint but different oracle
   outcomes.

**Conclusion:** The current fingerprint is adequate for ML deduplication but
insufficient for proving that two states will produce identical oracle labels.

---

## D. All Transition-Relevant State Categories

The following categories of simulator state affect future execution and oracle
branch outcomes. None can be assumed to be captured by the 63-feature vector:

### D1. Core simulation time and identity
| Field | Source | Transition effect |
|-------|--------|-------------------|
| `sim._step` | Simulator | Controls step_size * step = current time; affects deadline checks |
| `sim._time` | Simulator | Absolute time; used in laxity, deadline_slack, arrival enqueueing |
| `sim.config` | Simulator | GPU config, service_model params, max_steps, drain_steps; fixed per scenario but part of oracle semantics |

### D2. Waiting queue contents and ordering
| Field | Source | Transition effect |
|-------|--------|-------------------|
| `sim._waiting` (deque of InternalRequest) | Simulator | Exact queue order matters for FCFS, but policies sort by score. However, the set of available requests matters fully. |
| `sim._waiting_map` | Simulator | Lookup map; redundant with _waiting but used by scheduler |

**Note:** `ObservableState.waiting_queue` contains `ObservableRequest` objects,
which strip `phase`, `tokens_decoded`, `prefill_remaining`, `transfer_ready_time`,
and `migration_destination_gpu_id` from `InternalRequest`. The 63 features
aggregate over these, losing per-request detail.

### D3. Active request progress (on GPUs)
| Field | Source | Transition effect |
|-------|--------|-------------------|
| `GPUState._active` (dict of InternalRequest) | Per GPU | Exact per-request progress (tokens_decoded, prefill_remaining) affects remaining service time and completion order |
| `InternalRequest.phase` | InternalRequest | WAITING, ACTIVE, COMPLETED; determines scheduling eligibility |
| `InternalRequest.tokens_decoded` | InternalRequest | Remaining service = actual_output_tokens - tokens_decoded |
| `InternalRequest.prefill_remaining` | InternalRequest | Affects remaining GPU occupancy time |
| `InternalRequest.first_token_time` | InternalRequest | Affects TTFT metrics, not scheduling decisions, but affects SLO slack |
| `InternalRequest.admission_time` | InternalRequest | Affects queuing delay and latency computation (metrics, not scheduling) |
| `InternalRequest.completion_time` | InternalRequest | Write-once; used for completed request accounting |

### D4. Pending future arrivals
| Field | Source | Transition effect |
|-------|--------|-------------------|
| `sim._pending_arrivals` (suffix not yet enqueued) | Simulator | Entire future request stream; the whole-branch oracle INCLUDES future arrivals in continuation |
| arrival cursor position (implicit `arrival_idx` from `Simulator.run`) | Simulator.run | Determines which arrivals are visible vs future; critical for oracle identity |

**Critical:** Two states with identical observed queues but different
`_pending_arrivals` suffixes will produce different oracle labels. This is the
single largest source of oracle-label mismatch for identical feature vectors.

### D5. GPU live state
| Field | Source | Transition effect |
|-------|--------|-------------------|
| `GPUState._active` (InternalRequest references) | Per GPU | Which requests are actively served, their progress, their KV footprint |
| `GPUState.current_kv_tokens` (computed from _active) | Per GPU | Capacity availability for admissions |
| `GPUState._pending_handoff` | Per GPU | Disaggregated prefill/decode bridge requests awaiting transfer |

### D6. Migrating requests (disaggregated prefill/decode bridge)
| Field | Source | Transition effect |
|-------|--------|-------------------|
| `sim._migrating` (deque of InternalRequest) | Simulator | Requests mid-bridge-transfer between prefill and decode GPUs |
| `InternalRequest.transfer_ready_time` | InternalRequest | When the request becomes eligible for decode GPU admission |

### D7. Relocating requests (live cross-instance migration)
| Field | Source | Transition effect |
|-------|--------|-------------------|
| `sim._relocating` (dict of InternalRequest) | Simulator | Requests mid-migration between GPU instances |
| `InternalRequest.migration_destination_gpu_id` | InternalRequest | Fixed target GPU; determines arrival semantics |
| `InternalRequest.transfer_ready_time` | InternalRequest | When transfer completes and request can be admitted |

### D8. Completed request accounting
| Field | Source | Transition effect |
|-------|--------|-------------------|
| `sim._completed` (list of CompletedRequest) | Simulator | Used in `completed_count` (observable) and in branch continuation to check if the simulation has naturally terminated |

### D9. Diagnostic-only state (no transition effect)
| Field | Source | Transition effect |
|-------|--------|-------------------|
| `sim._util_history` | Simulator | Metrics collection only; read-only after append |
| `sim._batch_history` | Simulator | Metrics collection only |
| `sim._waiting_queue_history` | Simulator | Explicitly documented as diagnostic-only, never consulted by any policy or objective |
| `sim._policy_times` | Simulator | Wall-clock measurement only |
| `sim._idle_skipped` | Simulator | Counter for fast-forward accounting; does not affect simulation semantics |
| `GPUState.step_contention_diagnostics` | Per GPU | Explicitly documented as diagnostic-only |
| `GPUState.step_active_counts` | Per GPU | Diagnostic per-step counters |
| `GPUState.step_kv_used` | Per GPU | Diagnostic per-step counters |

### D10. Native policy internal state
| Field | Source | Transition effect |
|-------|--------|-------------------|
| `EstimatedServiceTimePolicy.alpha`, `.beta` | Policy init | Sorting parameters; fixed per instance, no mutation |
| `WeightedFairSharePolicy.alpha`, `.beta` | Policy init | Scoring parameters; fixed per instance, no mutation |
| `LiveHierarchicalRouterPolicy.trajectory` | Router | Append-only record of past decisions; does not affect select_action |

**No mutable fairness counters, no accumulated service state, no starvation
counters, no previous-selection state exist in ESTF or WFS.** Both are purely
stateless scoring functions over `ObservableState`. This is a key finding for
oracle identity.

### D11. RNG / Determinism
The simulator and branch continuation are **deterministic**. No randomness is
used in:
- `Simulator.run()` scheduling loop
- `_apply_action()` (admission, preemption, eviction, swap, migrate)
- `_advance_decode()` (decode step advancement)
- `_build_observable_state()` (state snapshotting)
- ESTF and WFS `select_action()` (sorting and greedy admission)
- `LiveFork.advance_one_step()` (mirror of run loop)

The only `seed` parameter flows into `Simulator.run()` but is recorded only in
metrics output. No numpy random or Python random is called in the execution path.

**Conclusion: RNG state need not be included in oracle-state fingerprint.**

### D12. Scenario / environment identity (immutable)
| Field | Source | Transition effect |
|-------|--------|-------------------|
| Scenario manifest (utilization, skew, favored size, noise, seed, n_total_jobs, max_active_sequences) | Generator | Fixed per scenario; determines request distribution |
| Arrival trace (sorted list of Request objects) | `sim.load_trace()` | Entire future arrival schedule; part of oracle identity |
| GPU configuration | `SimulatorConfig.gpu_configs` | Capacity, batch tokens, KV tokens, roles |
| Service model | `SimulatorConfig.service_model` | step_size, prefill cost, prefill modeling flags, disaggregation flags |
| Horizon parameters | HORIZON_H constant (1500) | Branch continuation length |

---

## E. Do Pending Arrivals Must Be Part of Oracle Identity?

**YES.**

This is the single most important finding of this audit.

The whole-branch oracle (run by `run_weighted_branch` in the generator) advances
forked simulator states through 1500 extra steps **with future arrivals included**.
The oracle compares `J_ESTF_whole` vs `J_WFS_whole` over this continuation window.

Two pre-decision states with identical:
- scenario, step, contested request IDs
- all 63 feature values (quantiles, means, ratios, history slopes)
- active request progress (tokens_decoded, prefill_remaining)
- waiting queue contents (same requests, same IDs)
- migrating / relocating state
- completed count
- time, step

but **different** pending-arrival suffix (e.g., one has a high-priority SLO-tight
request arriving in 5 steps, the other does not)

will produce **different oracle labels** because the ESTF/WFS sorting and
admission decisions in the continuation branch will diverge when that high-priority
request arrives.

**Oracle-state identity MUST include the full `_pending_arrivals` suffix (all
not-yet-enqueued InternalRequest objects, sorted canonically by request_id) and
the arrival cursor position (sim._step at fork time, which implicitly defines how
many arrivals have been consumed).**

---

## F. Do Mutable Policy State Exist?

**NO.**

Neither ESTF nor WFS maintains mutable internal state across calls to
`select_action`:

- `EstimatedServiceTimeFirstPolicy`: only stores `alpha` and `beta` (constants)
- `WeightedFairSharePolicy`: only stores `alpha` and `beta` (constants)
- Both compute scores purely from the `ObservableState` argument
- No fairness deficit counters, no per-class accumulated service, no starvation
  counters, no tie-breaking history, no previous-selection state

The `LiveHierarchicalRouterPolicy.trajectory` is append-only and never read back
by `select_action`. It does not affect scheduling decisions.

The `scaled_gpu_state` snapshot/restore discipline (in
`family_a_observability_continuation_v1.py`) protects `ObservableGPUState` from
in-place mutation by `select_action`, but this is about observable state isolation,
not policy-side mutable counters.

**Implication for oracle fingerprint:** No policy-internal mutable state needs to
be captured. Only the simulator state is relevant.

---

## G. RNG / Determinism Result

**The simulator and oracle branch continuation are deterministic.** No randomness
is used in the scheduling loop, branch continuation, or policy selection.

**RNG state is NOT needed in the oracle-state fingerprint.**

---

## H. Proposed Canonical Oracle-State Payload

```
ORACLE_STATE_FINGERPRINT_SCHEMA_V1 = "oracle_fp_v1.0"

oracle_state_fingerprint = SHA256(
    canonical_oracle_state_payload(state, simulator)
)

where canonical_oracle_state_payload returns a deterministic byte sequence from:

{
    "schema_version": "oracle_fp_v1.0",
    "simulator_version": "<simulator code version / git commit hash or config hash>",
    "time": <float, not rounded; exact IEEE 754 repr>,
    "step": <int>,
    "arrival_cursor_step": <int; sim._step at observation time, defines arrival_consumed>,
    "pending_arrivals": [
        // All not-yet-enqueued InternalRequest objects, sorted by request_id
        {
            "request_id": <int>,
            "arrival_time": <float>,
            "prompt_tokens": <int>,
            "predicted_output_tokens": <int>,
            "actual_output_tokens": <int>,
            "slo_deadline": <float>,
            "priority": <float>,
            "class_id": "<str>",
            // Internal request state at fork time:
            "phase": "<WAITING|ACTIVE|COMPLETED>",
            "tokens_decoded": <int>,
            "prefill_remaining": <int>,
            "transfer_ready_time": <float>,
            "migration_destination_gpu_id": <int>,
        },
        ...
    ],
    "waiting_queue": [
        // All requests currently in sim._waiting, sorted by request_id
        {
            "request_id": <int>,
            "arrival_time": <float>,
            "prompt_tokens": <int>,
            "predicted_output_tokens": <int>,
            "actual_output_tokens": <int>,
            "slo_deadline": <float>,
            "priority": <float>,
            "class_id": "<str>",
            "phase": "<WAITING|ACTIVE|COMPLETED>",
            "tokens_decoded": <int>,
            "prefill_remaining": <int>,
        },
        ...
    ],
    "active_requests": [
        // All requests in GPUState._active across all GPUs, sorted by (gpu_id, request_id)
        {
            "gpu_id": <int>,
            "request_id": <int>,
            "phase": "<WAITING|ACTIVE|COMPLETED>",
            "tokens_decoded": <int>,
            "prefill_remaining": <int>,
            "admission_time": <float>,
            "first_token_time": <float>,
            "completion_time": <float>,
        },
        ...
    ],
    "gpu_state": [
        // Per-GPU structural state, sorted by gpu_id
        {
            "gpu_id": <int>,
            "max_active_sequences": <int>,
            "max_batch_tokens": <int>,
            "max_kv_tokens": <int>,
            "current_kv_tokens": <int>,
            "prefilling_count": <int>,
            "decoding_count": <int>,
            "active_request_ids": [<int>, ...],  // sorted
            "role": "<str or null>",
            "incoming_migrations": [<int>, ...],  // sorted list of request_ids
        },
        ...
    ],
    "migrating_queue": [
        // Requests in bridge transfer, sorted by request_id
        {
            "request_id": <int>,
            "transfer_ready_time": <float>,
        },
        ...
    ],
    "relocating": [
        // Requests mid-migration, sorted by request_id
        {
            "request_id": <int>,
            "migration_destination_gpu_id": <int>,
            "transfer_ready_time": <float>,
        },
        ...
    ],
    "completed_count": <int>,
    // Environment / semantics identity (immutable, but must match for reuse)
    "environment_id": "<SHA256 of scenario manifest + gpu_configs + service_model + arrival_trace_hash>",
    "oracle_semantics_version": "oracle_branch_v1.0",
    "policy_pair_version": "estf_wfs_v1.0",
    "branch_horizon_version": "horizon_1500_step_v1.0",
    "utility_definition_version": "priority_weighted_slo_safe_v1.0",
    "feature_schema_version": "feat63_v1.0"
}
```

### Determinism guarantees:
- All arrays are sorted by stable, explicit keys (request_id, gpu_id)
- `json.dumps(sort_keys=True, separators=(",", ":"))` for compact canonical form
- Floats use Python's default `repr` which is deterministic for IEEE 754 doubles
- `schema_version` enables future payload structure changes with distinct hash spaces
- `environment_id` binds to the immutable scenario + configuration, computed once
  per scenario at genesis
- `oracle_semantics_version`, `policy_pair_version`, `branch_horizon_version`,
  `utility_definition_version` change whenever the oracle definition changes,
  invalidating all cached labels automatically

---

## I. Fields Explicitly Excluded and Why

| Field | Reason for exclusion |
|-------|---------------------|
| `sim._util_history` | Diagnostic-only metrics; never read by execution or oracle |
| `sim._batch_history` | Diagnostic-only metrics |
| `sim._waiting_queue_history` | Explicitly documented as never consulted by any policy or objective |
| `sim._policy_times` | Wall-clock measurement only |
| `sim._idle_skipped` | Fast-forward counter; does not affect simulation semantics |
| `GPUState.step_contention_diagnostics` | Explicitly documented as diagnostic-only |
| `GPUState.step_active_counts` | Diagnostic per-step counters |
| `GPUState.step_kv_used` | Diagnostic per-step counters |
| `GPUState._pending_handoff` | Empty at observation time (collected by next `_collect_handoffs()` call); transient state that does not affect scheduling decisions |
| `sim.config` (entire object) | Instead, bind to `environment_id` hash which captures the config content; avoids hashing the full object graph |
| `sim._completed` (full list) | Only `completed_count` affects termination check; completed request details are never consulted by policy selection or branch continuation logic (the fork just appends to its own _completed) |
| `sim._gpu_map` | Redundant with `sim._gpus`; deterministic reconstruction from gpu_ids |
| `sim._waiting_map` | Redundant with `sim._waiting`; deterministic reconstruction |
| `sim._time` | Derivable from `sim._step * sim.config.service_model.step_size`; explicitly redundant |
| Python object IDs | Not included anywhere; all sorting is by semantic keys |
| Dict/set iteration order | Eliminated by explicit sorting of all collections |

---

## J. Scenario / Environment / Version Binding

### Environment ID
Compute once per scenario at genesis:

```
environment_id = SHA256(
    json.dumps({
        "scenario_id": "...",
        "target_utilization": ...,
        "tenant_weight_skew": ...,
        "favored_tenant_size": "...",
        "prediction_noise_sigma": ...,
        "seed": ...,
        "n_total_jobs": ...,
        "max_active_sequences": ...,
        "gpu_configs": [sorted gpu config dicts],
        "service_model": {step_size, prefill_cost, enable_prefill_modeling, enable_disaggregation},
        "arrival_trace_hash": SHA256(sorted arrival request tuples),
    }, sort_keys=True, separators=(",", ":"))
)
```

This binds the oracle label to the exact scenario definition, GPU topology, and
arrival trace. Any change in any of these produces a different `environment_id`.

### Version strings (semantic versioning for oracle semantics)

| Version key | Format | When to increment |
|-------------|--------|-------------------|
| `oracle_semantics_version` | `oracle_branch_{horizon}_step_v{N}` | Branch horizon changes, or continuation logic changes |
| `policy_pair_version` | `estf_wfs_v{N}` | ESTF or WFS policy code changes in a way that affects select_action output |
| `branch_horizon_version` | `horizon_{H}_step_v{N}` | HORIZON_H constant changes |
| `utility_definition_version` | `priority_weighted_slo_safe_v{N}` | Label definition changes (epsilon thresholds, weighting, SLO definition) |
| `feature_schema_version` | `feat{M}_v{N}` | 63-feature schema changes (add/remove/rename features) |
| `simulator_version` | `sim_{git_commit_or_config_hash}` | Any simulator code change |

These are NOT included in the raw oracle-state fingerprint hash (which captures
the live state). They are appended to form the **oracle cache key** (see Section K).

---

## K. Exact Oracle Cache-Key Design

```
oracle_cache_key = SHA256(
    oracle_state_fingerprint_bytes
    + oracle_semantics_version.encode()
    + policy_pair_version.encode()
    + branch_horizon_version.encode()
    + utility_definition_version.encode()
)
```

Where `oracle_state_fingerprint_bytes` is the raw SHA-256 digest (32 bytes) of
the current fingerprint, not its hex string. This ensures that appending version
strings produces a domain-separated collision-resistant hash.

**The oracle cache key is the definitive key for label lookup:**
```python
cached_label = oracle_label_cache.get(oracle_cache_key)
```

**The `oracle_state_fingerprint` alone is NOT used for cache lookup.** It is
stored as a metadata column alongside the label but the actual cache key includes
all version strings.

### Relationship to existing columns

| Existing column | New column | Purpose |
|----------------|-----------|---------|
| `state_fingerprint` | `oracle_state_fingerprint` | Stronger state identity for oracle reuse |
| `sample_id` | `oracle_cache_key` | Definitive cache lookup key including versions |

---

## L. Strict SAFE_REUSE / NO_REUSE Rules

### SAFE_REUSE (oracle label may be reused from cache):
1. `oracle_state_fingerprint` matches exactly (hex string comparison)
2. `oracle_semantics_version` matches exactly
3. `policy_pair_version` matches exactly
4. `branch_horizon_version` matches exactly
5. `utility_definition_version` matches exactly
6. `environment_id` matches exactly

All six conditions must hold. This guarantees the cached oracle label was computed
from the identical state, with identical semantics, identical policies, identical
horizon, and identical utility definition.

### NO_REUSE (must compute fresh oracle label):
- Only `state_fingerprint` matches (feature-level identity, not state-level)
- Only `oracle_state_fingerprint` matches but any version string differs
- Near-duplicate state (feature values within epsilon but not exact match)
- `(scenario_id, step)` matches but `oracle_state_fingerprint` differs
- Model features match but pending-arrival suffix differs
- Same arrival cursor but different arrival count in trace
- Changed fairness deficit (N/A for current ESTF/WFS but required for future policies)
- `oracle_cache_key` not present in cache

---

## M. Can D0 Be Post-Hoc Upgraded to Exact Oracle Fingerprints?

**POSTHOC_ORACLE_FINGERPRINT_RECONSTRUCTION_NOT_POSSIBLE**

Current D0 rows do NOT contain enough metadata to reconstruct an exact
`oracle_state_fingerprint` because:

1. **No pending arrivals stored.** D0 rows do not include the `_pending_arrivals`
   suffix (all not-yet-enqueued InternalRequest objects). This information was
   available only in the running simulator at the time of the decision.

2. **No active request progress stored.** D0 rows do not include
   `tokens_decoded`, `prefill_remaining`, `first_token_time`, or
   `transfer_ready_time` for active requests. `ObservableRequest` strips these.

3. **No migrating / relocating state stored.** If any requests were mid-migration
   at decision time, this state is not captured in D0 rows.

4. **No GPU internal state stored.** The exact `GPUState._active` dict content
   (not just observable aggregates) is not in D0.

**Consequence:**
- D0 remains valid as a training dataset (ML features are intact)
- D0 labels should NOT be reused later based solely on the current
  `state_fingerprint` for oracle-label cache lookups
- Future DAgger rows must store `oracle_state_fingerprint` at acquisition time,
  computed from the live fork state before the branch oracle runs
- D0 can still be used for model training; its labels simply cannot serve as
  a deterministic oracle-label cache

---

## N. Proposed Modules / Functions / Tests

### New module: `src/llmserveopt/learning/state_identity.py`

```python
"""Deterministic state identity functions for ML and oracle use."""

from __future__ import annotations

# ---- Feature fingerprint (for ML deduplication) ----

def canonical_feature_payload(row: dict) -> str:
    """Return the canonical JSON string for the current feature fingerprint.
    Same as existing stable_state_fingerprint payload. Compatible with existing D0."""

def feature_fingerprint(row: dict, feature_cols: list[str]) -> str:
    """SHA-256 hex digest of the feature payload. Same as existing stable_state_fingerprint."""

# ---- Oracle state fingerprint (for oracle-label reuse) ----

def canonical_oracle_state_payload(
    state: "ObservableState",
    simulator: "Simulator",
    version_metadata: dict[str, str],
) -> str:
    """Return deterministic canonical JSON string for oracle-state identity.

    Includes:
    - schema_version, simulator_version
    - time, step, arrival_cursor_step
    - pending_arrivals (sorted by request_id)
    - waiting_queue (sorted by request_id)
    - active_requests (sorted by gpu_id, request_id)
    - gpu_state (sorted by gpu_id)
    - migrating_queue (sorted by request_id)
    - relocating (sorted by request_id)
    - completed_count
    - environment_id
    - version strings
    """

def oracle_state_fingerprint(
    state: "ObservableState",
    simulator: "Simulator",
    version_metadata: dict[str, str],
) -> str:
    """SHA-256 hex digest of the oracle state payload."""

def oracle_cache_key(
    state_fp: str,
    oracle_semantics_version: str,
    policy_pair_version: str,
    branch_horizon_version: str,
    utility_definition_version: str,
) -> str:
    """SHA-256 hex digest of (state_fp_bytes + version strings).

    This is the definitive cache lookup key for oracle labels."""

# ---- Environment ID ----

def compute_environment_id(scenario, gpu_configs, service_model, arrival_trace) -> str:
    """SHA-256 of scenario definition, GPU topology, service model, arrival trace."""
```

### Proposed tests: `tests/test_family_a_state_identity_v1.py`

```python
class TestStateIdentity:
    """Lightweight tests for deterministic canonical serialization."""

    def test_same_state_serialized_twice_same_hash(self):
        """A. Same state serialized twice -> same hash."""
        # Create state, serialize twice, assert hashes equal.

    def test_fork_restore_same_hash(self):
        """B. Fork/restore same state -> same hash."""
        # Fork from live simulator, compute fingerprint on both copies, assert equal.

    def test_reorder_dict_same_hash(self):
        """C. Reorder dict/set/request container without semantic change -> same hash."""
        # Verify that sorting by request_id/gpu_id is applied regardless of
        # insertion order.

    def test_change_one_field_different_hash(self):
        """D. Change one transition-relevant field -> different hash."""
        # Modify one InternalRequest field, assert hash differs.

    def test_change_logging_field_same_hash(self):
        """E. Change only logging/non-semantic field -> same hash."""
        # Modify sim._idle_skipped or _waiting_queue_history, assert hash unchanged.

    def test_same_features_different_hidden_state_different_oracle_fp(self):
        """F. Same 63-feature vector but altered hidden transition state:
           - same feature fingerprint allowed
           - different oracle-state fingerprint required."""
        # Craft two states with identical 63 features but different
        # pending_arrivals or active_request progress, assert oracle_fingerprints differ.

    def test_different_arrival_cursor_different_oracle_fp(self):
        """G. Different arrival cursor -> different oracle-state fingerprint."""
        # Same simulator state, different _step values, assert different oracle fingerprints.

    def test_changed_fairness_deficit_different_oracle_fp(self):
        """H. Changed fairness deficit -> different oracle-state fingerprint."""
        # (N/A for current stateless ESTF/WFS, but test the mechanism for future policies.)
```

---

## O. Exact Classifications

### ORACLE_STATE_FINGERPRINT: **ORACLE_STATE_FINGERPRINT_READY**

The simulator state is fully serializable. No unserializable objects, no open
file handles, no network sockets, no thread locks. `copy.deepcopy` already works
on the entire `Simulator` object (used by `fork_from_live_simulator`). The minimal
transition-relevant state is well-defined and tractable to serialize.

### CURRENT FINGERPRINT: **CURRENT_FINGERPRINT_SAFE_FOR_DATASET_DEDUP_ONLY**

The existing `state_fingerprint` is correct and sufficient for ML row deduplication.
It is NOT safe for oracle-label reuse because it omits transition-relevant state
not captured in the 63-feature vector (pending arrivals, active request progress,
migrating/relocating state).

---

## P. Artifact Created

**`docs/design/FAMILY_A_ORACLE_STATE_FINGERPRINT_V1.md`** — this document.

No other repository files are created or modified. No code is implemented.
The `state_identity.py` module and tests are proposed but not yet written,
per the task scope: "design only" unless "clearly independent of active
generation, negligible CPU, does not alter running dataset semantics."
The implementation can be added as a follow-up task.

---

## Q. Real-System Relationship

The exact `oracle_state_fingerprint` is primarily a **simulator/offline-labeling
construct**. A real vLLM deployment does NOT need to compute this hash on every
scheduling decision.

For normal online inference:
- Only the 63 portable features + learned selector are needed
- No oracle label cache exists in production

For DAgger state logging (real-system closed-loop):
- Log enough canonical state metadata to later reproduce/fork or match the state
- The `oracle_state_fingerprint` could be computed on-demand for logging purposes
- Hot-path overhead should remain low: the fingerprint is computed at decision
  points where ESTF/WFS disagree, not on every step

---

## R. Backward Compatibility

The new `oracle_state_fingerprint` and `oracle_cache_key` are **new columns**,
not replacements for `state_fingerprint`.

| Existing column | New column | Relationship |
|----------------|-----------|-------------|
| `state_fingerprint` (SHA-256 of scenario+step+requests+features) | `oracle_state_fingerprint` (SHA-256 of full transition state) | Distinct; newer is strictly stronger |
| `sample_id` | `oracle_cache_key` (SHA-256 of oracle_fp + versions) | Distinct roles |

The existing `state_fingerprint` is never redefined. Its semantics, version,
and collision properties remain unchanged. The new columns carry explicit
version strings and are clearly named to prevent any confusion.

---

## S. Summary of Key Findings

| Finding | Impact |
|---------|--------|
| Current fingerprint adequate for ML dedup | Safe to use existing D0 as-is for training |
| Current fingerprint insufficient for oracle reuse | Must introduce oracle_state_fingerprint + oracle_cache_key |
| Pending arrivals are transition-relevant | MUST be included in oracle fingerprint |
| ESTF/WFS are stateless | No policy mutable state to capture |
| Simulator is deterministic | No RNG state needed |
| 6 diagnostic-only state categories excluded from fingerprint | Reduces fingerprint size, no correctness cost |
| D0 cannot be post-hoc upgraded | D0 labels cannot serve as oracle-label cache |
| ORACLE_STATE_FINGERPRINT_READY | No simulator refactor needed; fully serializable |
| Minimal implementation: one new module, one new test file | Low integration cost |
