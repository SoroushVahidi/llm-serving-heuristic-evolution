# src/llmserveopt/selector

Two generations of selector code live here. Full research narrative:
**[docs/current/SELECTOR_V2.md](../../../docs/current/SELECTOR_V2.md)**.

## v1 vs. v2 -- do not confuse them

- **`selector/` (this directory's top-level files)** -- Selector v1:
  feature extraction, labeling, models (`rule_based`, `decision_tree`,
  `random_forest` + variants), drawing from the full 20-policy portfolio
  (`candidates.py::SELECTOR_CANDIDATES`). Still present, still has passing
  tests, but superseded in active-development focus by v2.
- **`selector/dataset_v2/`** -- Selector v2, the project's current active
  research track. Everything below refers to this subpackage.

## Selector v2 candidate-policy sets -- three, do not conflate them

Defined in `selector/dataset_v2/candidates.py` (see that module's docstring
for the full explanation):

1. `BASELINE_NAMES` (imported from `policies.registry`, 20) -- the full
   historical portfolio, not v2-specific.
2. `MONOLITHIC_DIAGNOSTIC_POLICY_POOL` / `monolithic_candidate_policies()`
   (14 = 3 external + 11 historical) -- a broader diagnostic/exploration
   pool used by earlier (pre-Option-B) pilots. Historical value, not the
   current trainable action space.
3. **`SELECTOR_V2_OPTION_B_POLICIES` (8)** -- **the current, canonical,
   approved Selector v2 trainable action space.** `dataset_v2/calibrated_targeted_pilot.py::CANDIDATE_POLICIES`
   imports this constant directly. When extending or scripting against
   "the Selector v2 candidate set," use this one.

Faithful external baselines (`vllm_faithful`, `vllm_chunked_prefill_faithful`,
`sarathi_faithful`, and the disaggregated/migratory baselines) are
**evaluation-only** -- confirmed genuinely dominated as selector actions
(`docs/selector_v2_faithful_baseline_scope_audit.md`), never part of set 3.

## Dataset v2 source of truth

- **`dataset_v2/slo_calibration.py`** -- policy-independent, per-request SLO
  calibration (`calibrate_window_e2e()`). The fix that made scenarios
  genuinely ANWG-discriminative.
- **`dataset_v2/calibrated_targeted_pilot.py`** -- the current pipeline
  (Option B scope). Generates windows, runs the 8 candidates, produces
  `PolicyOutcomeVector`s.
- **`dataset_v2/builder.py`**, **`schema.py`** -- shared building blocks.
- Objective: **`arrival_normalized_weighted_goodput`** (ANWG) --
  `docs/selector_objective_audit.md`. Not the legacy `weighted_goodput`
  (completed-only denominator, biased toward rejecting more work).

## Current status (read before trusting any pilot output)

The most recent pilot's split construction had a **confirmed leakage bug**
(cross-transform row-range reuse across TRAIN/VALIDATION/ID_TEST). See
`experiments/selector_v2_calibrated_pilot_20260720T163235Z/LEAKAGE_AUDIT.md`
and [docs/current/SELECTOR_V2.md](../../../docs/current/SELECTOR_V2.md) §10
before trusting any held-out numbers from that run. The pipeline's own
`no_leakage` quality gate does not catch this failure mode.
