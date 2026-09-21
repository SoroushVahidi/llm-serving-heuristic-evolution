# DENOMINATOR_COMPLETION_V1

**Classification: POST_HOC_DENOMINATOR_COMPLETION.**
This amendment is *not* preregistered and is *not* pre-specified relative to the earlier sensitivity results. It is written and frozen after those results were observed, and it is recorded as such.

Frozen: 2026-09-21, before the denominator pass is executed.
Original protocol (unchanged): `experiments/reference_reserve_sensitivity_v1/PROTOCOL_V1.json`, sha256
`97b06cf1c96fe6b3f84aaf4355643192a0b400ce206f4c7c041b95d6bc417439`.
Executed sensitivity runner (unchanged): `scripts/reference_reserve_sensitivity_v1.py` at commit
`8addcdb1368f9cb3057d5d3823c8a273e52dcc1f`.

## 1. Why this amendment exists

`PROTOCOL_V1.json` lists `total_reference_decision_states`, `P(D)`, `opportunity_weighted_headroom` and
`beneficial_decision_rate` among its metrics, and `IMPLEMENTATION_V1.md` (section 4) states that total states and
`P(D)` are recorded. The executed runner did **not** count total reference decision states: its observer tallied only
disagreement states. Consequently those four quantities were never produced by Slurm jobs 1303715 / 1303716 / 1303717.
`IMPLEMENTATION_V1.md` is left as committed; this document is the correction of record.

## 2. Frozen statements

1. **Existing scientific outcomes have already been observed.** At the time of this freeze the following are known:
   disagreement states (720 / 466 / 198 for reserves 0.82 / 0.90 / 1.00), beneficial states (590 / 353 / 99),
   P(B_LAT | D), state-weighted and equal-window mean H_LAT, medians, confidence intervals, all-harmful shares,
   action-level counts, concentration shares, and total headroom mass (about 1437.0 / 1408.1 / 23.6 simulated ms).
   The reserve-0.82 disagreement rates `P(D)` also appear, rounded, in the manuscript. The total reference decision
   states for reserves 0.90 and 1.00 have **not** been observed. The exact reserve-0.82 sums were not computed
   before this freeze.
2. **No causal branches will be rerun.** No state is cloned, no alternative action is forced, no continuation branch is
   executed, and no latency counterfactual or H_LAT is computed.
3. **No threshold, policy, reserve value, workload, regime or window changes.** The three reserves (0.82, 0.90, 1.00),
   the six policies, the five selected regimes and the 20 fresh windows per regime (100 scenarios per reserve) are exactly
   those of the completed sensitivity experiment.
4. **The only new quantity measured is the denominator**: the total number of reference decision states on each
   reserve-specific reference trajectory.
5. **Derived quantities are calculated mechanically** (no other formula will be used):

   ```
   P(D)                            = disagreement_states / total_reference_decision_states
   opportunity_weighted_headroom   = total_headroom_mass / total_reference_decision_states
   beneficial_decision_rate        = beneficial_disagreement_states / total_reference_decision_states
   ```

   `disagreement_states`, `beneficial_disagreement_states` and `total_headroom_mass` (the sum of H_LAT over the
   disagreement states, in simulated ms, summed with `math.fsum` over `disagreement_states.csv` and multiplied by 1000)
   are taken from the completed, hash-verified sensitivity results. `P(B_LAT | D)` is copied from
   `result_summary.json`.
6. **The counting rule is frozen before this pass executes** (section 3).

No robustness threshold will be defined. The complete 0.82 / 0.90 / 1.00 curve and effect sizes are reported.

## 3. Exact denominator definition

A *reference decision state* is one invocation of the scheduler hook (`select_action`) by the simulator on the
reserve-specific reference trajectory. This is the same hook, and therefore the same unit, that

* the executed sensitivity runner's `DynamicObserver.select_action` used to detect disagreement, and
* the original support scan (`PhaseAScanPolicy`, phase A / phase B v2) used when it reported `sbs_decision_states`
  (`n = len(state_rows)`, one row per invocation).

Counted: every observer invocation on the reference trajectory, including invocations at which the reference chooses
no admission. **Not** counted: idle-skipped ticks (the simulator does not invoke the hook for them), counterfactual
branch states, and any state after a decision point. The counting observer is invoked exactly once per simulator
step; the runner records, as an independent invariant, that `decision_states` equals the simulator's per-step
queue-history length, and the pass fails its integrity gate otherwise.

The counting observer reuses the executed runner's scenario construction, simulator configuration, reference-policy
construction (`KVConstrainedOnlinePolicy(target_kv_utilization=reserve)`), portfolio and canonical-action detection
verbatim; it omits only the branching step.

## 4. Procedure

**Stage `count`** (`scripts/reference_reserve_denominator_completion_v1.py count`), one sequential Slurm job:

1. Verify the sha256 of `PROTOCOL_V1.json` and of all 21 existing sensitivity result files against
   `denominator_completion_v1/EXISTING_SENSITIVITY_RESULT_HASHES_V1.json`; abort if any differs.
2. For reserves 0.82, 0.90, 1.00, in that order, replay the 100 scenarios, counting decision states and (integrity check
   only) re-detecting disagreement states.
3. Write `counts_reserve_{082,090,100}.csv` and `count_execution_provenance.json` to scratch.

**Stage `derive`** (`... derive`), run on the completed counts:

1. Re-verify the existing-result hashes.
2. Evaluate the gates below.
3. Only if the gates pass (or the 0.82 canonical gate is `NOT_AVAILABLE`), compute the three derived quantities and write
   the output files. If a gate fails, no derived metric is written.

## 5. Gates

* **DENOMINATOR_TRAJECTORY_INTEGRITY_GATE.** The recomputed disagreement counts must reproduce the completed
  experiment: totals 720 (0.82), 466 (0.90), 198 (1.00), *and* the per-scenario counts recorded in each reserve's
  `disagreement_states.csv`, for all 100 scenarios, with `decision_states` equal to the simulator step-call invariant.
  On failure: `DENOMINATOR_INTEGRITY_FAILED`; no P(D) or unconditional metric is derived until the mismatch is explained.
* **DENOMINATOR_REPRODUCTION_GATE (reserve 0.82 only).** The new decision counts must equal the exact canonical
  `sbs_decision_states` (and `true_canonical_disagreement_states`) recorded per source / window / axis / condition in
  `experiments/fresh_production_latency_headroom_confirmatory_v1/FRESH_SUPPORT_WINDOW_CONDITIONS_V1.csv`, and the
  regime-level `P_D` must match `FRESH_LATENCY_WORKLOAD_REGIME_V1.csv` to within 1e-12. Rounded percentages in the PDF
  are not used. On failure the discrepancy is reported and interpretation stops.

## 6. Outputs

Written to `experiments/reference_reserve_sensitivity_v1/denominator_completion_v1/` after execution:

* `DENOMINATOR_SUMMARY_V1.json`, `DENOMINATOR_BY_REGIME_V1.csv`, `DENOMINATOR_BY_WINDOW_V1.csv`,
  `DENOMINATOR_EXECUTION_PROVENANCE_V1.json`.
* Small existing sensitivity artifacts (summaries, provenance, window and regime tables, `disagreement_states.csv`) are
  preserved alongside their frozen hashes; the large `action_effects.csv` files stay in cluster scratch and are
  identified by hash only.

The existing sensitivity results in cluster scratch are read-only inputs and are never overwritten.

## 7. Execution

One CPU-only Slurm job on Wulver (`--partition=general --account=ikoutis --qos=standard`, 1 CPU, 8 GB, no GPU),
processing 0.82 -> 0.90 -> 1.00 sequentially. Execution must occur from the commit that contains this document; the
commit SHA is recorded in `count_execution_provenance.json` and the final provenance file.

## 8. Interpretation constraints

* The pass varies nothing: it changes what is measured (the denominator), not the experiment.
* Rule D of the original protocol still applies: the experiment varies the reserve only, does not isolate
  KV-footprint ordering, and does not establish invariance to other reference policies.
* The disagreement-count interpretation reported before this pass is not revised by it; `P(D)` and the unconditional
  quantities are reported in addition, with counts and rates both shown.
* This is the last planned scientific experiment before manuscript submission. A further experiment would be proposed
  only if this pass exposed an execution or implementation error invalidating its own result.
