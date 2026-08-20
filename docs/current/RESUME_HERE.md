# Resume Here

**Shortest current operational entrypoint.** For the research roadmap, read
[`docs/PROJECT_MAP.md`](../PROJECT_MAP.md). For detailed status, read
[`WORK_STATUS.md`](WORK_STATUS.md). For ordered next actions, read
[`NEXT_ACTIONS.md`](NEXT_ACTIONS.md).

## Current State

| Field | Value |
|---|---|
| Repository | `llm-serving-heuristic-evolution` |
| Branch | `contextual-compositional-heuristics-20260731` |
| Last reconciled SHA | `4dac2201d3b4b08cfbdca61285cd2cda59cb5b31` |
| Last reconciled date | 2026-08-19 |
| Remote | `origin/contextual-compositional-heuristics-20260731` |
| Expected Git state | clean, 0 ahead / 0 behind after `git fetch --prune origin` |
| Canonical roadmap | `docs/PROJECT_MAP.md` |
| Cluster PSD worktree | `/mmfs1/project/ikoutis/sv96/github/llm-serving-heuristic-evolution-policy-separation-v1` |

Resume commands:

```bash
cd /home/soroush/llm-serving-heuristic-evolution
git fetch --prune origin
git status --short --branch
git rev-parse HEAD
git rev-list --left-right --count @{u}...HEAD
python scripts/check_project_handoff_consistency.py
```

## What This Project Is

This project builds toward a verified contextual compositional scheduler system
for LLM inference serving:

```text
policy-separating workloads -> complementary policy library -> contextual selection (multi-family)
        -> mechanism attribution -> bounded envelope
```

The current primary metric is `arrival_normalized_weighted_goodput` (ANWG).

Typed DSL / module-composition infrastructure exists in-repo but is deferred; **within-scenario composition and synthesis have been demoted** as a central hypothesis.
 
 

## Most Recently Completed Work (Structural Reassessment + MF-PSD)

**The higher-level structural reassessment of the composition hypothesis is
COMPLETE.**

- Audit: [`../audits/reassessment_composition_hypothesis_20260817.md`](../audits/reassessment_composition_hypothesis_20260817.md)
- Verdict: **`COMPOSITION_DEMOTED`**. Within-scenario composition/synthesis
  is now exploratory future work, not the project's central hypothesis.
- Revised roadmap: `policy-separating workloads -> complementary policy
  library -> contextual selection (multi-family) -> mechanism attribution
  -> bounded envelope`.

**MF-PSD v1 (Multi-Family Policy Separation Dataset) — revised roadmap Step
1 — is COMPLETE.**

- Audit: [`../audits/multi_family_policy_separation_dataset_v1_20260817.md`](../audits/multi_family_policy_separation_dataset_v1_20260817.md)
- Artifacts: `experiments/mf_psd_v1/` (`mf_psd_long_v1.csv`,
  `mf_psd_scenarios_v1.csv`, `mf_psd_schema_v1.json`,
  `mf_psd_provenance_v1.json`, `mf_psd_build_manifest_v1.json`); builder
  `src/llmserveopt/policy_separation/mf_psd.py`; CLI
  `scripts/build_mf_psd_v1.py`; tests `tests/test_mf_psd_v1.py` (31/31
  passing).
- Verdict: **`MF_PSD_READY`**. Unifies the three `_COMPOSITION_READY`-gate
  sources (Family A v2 fairness/starvation, Family B v2 prefill/decode
  contention, Family C/KV v2 admission control) into one canonical
  long-form utility table (496 rows) and scenario-context table (176
  scenarios), with an explicit machine-readable learnable-feature
  allowlist/forbidden-field denylist, full source-row/scenario
  conservation, zero duplicates, deterministic byte-for-byte rebuild, and
  zero mutation of any frozen source artifact.
- The six-anchor policy matrix was **sparse, not dense** at MF-PSD v1 time
  (each family only evaluated its own 2 anchors) — this is now fully
  resolved, see below.

**Step 2 (unified six-policy utility-matrix evaluation) is now COMPLETE —
the matrix is fully dense.** Three-part path:

1. [`../audits/unified_policy_utility_matrix_v1_20260817.md`](../audits/unified_policy_utility_matrix_v1_20260817.md)
   (design: [`../design/UNIFIED_UTILITY_MATRIX_STEP2_V1.md`](../design/UNIFIED_UTILITY_MATRIX_STEP2_V1.md)) —
   416 new cells evaluated, Family A/B densified to 6/6 anchors; Family C
   blocked (`NEEDS_REFINEMENT`) because its scenario regeneration was
   confirmed not byte-exact (99/144 mismatch vs. frozen native cells,
   reproducing `kv_v2_reproducibility_forensic_20260817.md`). Also found:
   `full_prefill`/`chunked_prefill_small` collapse to one identical
   behavior outside Family B; on all 32 Family-B scenarios,
   `estf`/`wfs`/`least_laxity`/`kv_constrained` are byte-identical to each
   other and to `full_prefill`.
2. [`../audits/family_c_step2_reconstruction_audit_20260817.md`](../audits/family_c_step2_reconstruction_audit_20260817.md) —
   diagnosed the Family-C blocker directly: the KV pilot runner never
   serialized request-level scenario data (unlike Family A/B), and no
   backup copy exists anywhere in this repo's history. Verdict:
   `FAMILY_C_RECONSTRUCTION_BOUNDED` — exact historical replay is
   impossible, but a defensible uniform-reconstruction fallback exists.
3. [`../audits/family_c_reconstruction_v1_and_unified_matrix_completion_20260817.md`](../audits/family_c_reconstruction_v1_and_unified_matrix_completion_20260817.md)
   (design: [`../design/FAMILY_C_RECONSTRUCTION_V1.md`](../design/FAMILY_C_RECONSTRUCTION_V1.md)) —
   built `CURRENT_RECONSTRUCTED_FAMILY_C_V1` (new, explicitly-versioned
   layer, **not** historical replay: generate-once → serialize → replay
   all 6 anchors from the frozen serialization, 432/432 cells succeeded),
   then rebuilt the matrix as v2.

- Artifacts: `experiments/unified_utility_matrix_v2/`
  (`unified_utility_matrix_long_v2.csv`, `_wide_v2.csv`) —
  **176×6 = 1,056/1,056 cells populated, 0 missing.**
- **Final verdict: `UNIFIED_UTILITY_MATRIX_READY`.** 54.0% unique-winner
  rate, positive oracle gain over best-fixed in every family (0.02–0.05),
  no anchor universally dominant. Historical KV v2 evidence and MF-PSD
  v1's frozen Family-C rows are preserved unchanged, never mixed row-wise
  with the reconstruction.
- **Caveat carried forward (not a blocker):** Family B (32/176 scenarios)
  still has near-total policy collapse (5/6 anchors byte-identical) —
  barely dilutes aggregate diversity (54.0% vs. 55.6% without Family B),
  but Step 3's design must account for it.
- **This task did not train a selector or run any composition/synthesis
  experiment** — data generation and audit only, per explicit task scope.

**Step 3 (preregistered multi-family contextual-selector experiment) is
now COMPLETE — `MULTIFAMILY_SELECTOR_NO_GO`.**

- Audit: [`../audits/multifamily_contextual_selector_v1_20260817.md`](../audits/multifamily_contextual_selector_v1_20260817.md)
- Design: [`../design/MULTIFAMILY_CONTEXTUAL_SELECTOR_V1.md`](../design/MULTIFAMILY_CONTEXTUAL_SELECTOR_V1.md)
- Artifacts: `experiments/multifamily_contextual_selector_v1/`; harness
  `src/llmserveopt/selector/multifamily_contextual_selector_v1.py`; CLI
  `scripts/run_multifamily_contextual_selector_v1.py`; tests
  `tests/test_multifamily_contextual_selector_v1.py` (27/27 passing).
- **All 5 preregistered verdict gates failed on the pooled (Regime B)
  holdout** — no trained model beat best-fixed, or even the trivial
  majority baseline. **But within-family (Regime A) selection is strong**
  (near-perfect, 0 regret, on Family A/B holdouts) — the matrix has real
  learnable structure; the failure is specifically in pooling and
  cross-family transfer. Leave-one-family-out (Regime C): only 1/3
  directions win, with a severe collapse (6.2× worse than fixed) when
  Family A is held out.
- **Root cause, directly diagnosed:** `mechanism_family` is predictable at
  **100% accuracy** from the 33-column feature schema alone (every
  feature is family-prefixed with structural missingness). A
  shared-feature-only robustness check (A↔B, Family C has no analog) shows
  the selector cannot beat best-fixed at all once family-identifying
  feature blocks are removed — strong evidence the in-distribution gains
  are driven by family identification, not mechanism understanding.
- **Mechanism attribution (Step 4) remains blocked** — this verdict does
  not justify it.
- Next step (**not started, not authorized**): a separately-scoped
  feature-schema redesign investigation — whether a genuinely
  cross-family-shared feature representation could let a selector
  demonstrate real mechanism-level transfer without depending on family
  identification.

**The feature-schema redesign investigation named above is now
COMPLETE — `SHARED_FEATURE_SCHEMA_NO_GO`.**

- Audit: [`../audits/shared_cross_family_feature_schema_feasibility_v1_20260817.md`](../audits/shared_cross_family_feature_schema_feasibility_v1_20260817.md)
- Artifacts: `experiments/shared_cross_family_features_v1/`; module
  `src/llmserveopt/policy_separation/shared_context_features_v1.py`; build/
  diagnostics scripts `scripts/build_shared_cross_family_features_v1.py`,
  `scripts/analyze_shared_cross_family_features_v1.py`; tests
  `tests/test_shared_cross_family_features_v1.py` (12/12 passing).
- **A genuinely shared, zero-missingness, 17-feature schema
  (SHARED_CORE_V1) was built and replay-verified for all 176/176
  scenarios** (deterministic replay for Family A/B against the frozen
  MF-PSD source, direct load for Family C from the frozen Reconstruction
  v1 artifact) — the schema-construction half of the investigation
  succeeded.
- **But family remains 100% classifiable from SHARED_CORE_V1 alone** — not
  from structural missingness (there is none left), but because the three
  families' workloads occupy almost entirely disjoint regions of the
  shared feature space (range-overlap ≈0 on 15/17 features), and
  cross-family nearest-neighbor scenarios are *not* more utility-consistent
  than random cross-family pairs (mean Spearman −0.038 vs. +0.197 random
  baseline).
- **Independently, the six-policy target itself is not cross-family
  coherent**: `full_prefill`/`chunked_prefill_small` are bit-identical
  outside Family B, and the other four policies collapse to one identical
  value specifically on Family B — no policy is globally meaningful across
  all three families.
- Verdict is `NO_GO` (not `NEEDS_MORE_DATA`) because the target-semantics
  failure alone is sufficient per the frozen decision logic, independent
  of the feature-overlap finding.
- Most defensible next step if pursued (**not started, not authorized**):
  a 3-way mechanism-choice reformulation of the selector target
  (fairness-ranking vs. chunk-control vs. KV-reserve) — directly motivated
  by the target-semantics finding above, though the feature-overlap
  problem is independent and may still block it.

**The mechanism-choice target redesign named above is now
COMPLETE — `MECHANISM_TARGET_NO_GO`.**

- Audit: [`../audits/mechanism_choice_target_feasibility_v1_20260817.md`](../audits/mechanism_choice_target_feasibility_v1_20260817.md)
- Artifacts: `experiments/mechanism_choice_target_feasibility_v1/`; module
  `src/llmserveopt/policy_separation/mechanism_choice_target_v1.py`;
  diagnostics script
  `scripts/analyze_mechanism_choice_target_feasibility_v1.py`; tests
  `tests/test_mechanism_choice_target_v1.py` (8/8 passing).
- **The `kv` mechanism contrast (`|ANWG(kv_constrained_online) −
  ANWG(least_laxity_first)|`) is confounded, not a genuine mechanism-
  relevance signal.** It is *largest* on Family A, which has essentially no
  KV-memory pressure (SHARED_CORE_V1's `token_footprint_per_kv` ≈0.58,
  comfortably under capacity) — larger even than on Family C, KV's own
  native family (footprint ≈7.6, genuine pressure). A within-family
  dose-response check confirms this directly: on Family C, `gain_kv`
  correlates with actual KV pressure (ρ=+0.54, p<1e-6); on Family A it does
  not (ρ=−0.13, p=0.28, no relationship). The contrast instead reflects
  `least_laxity_first`'s general weakness outside its native family.
- This confound corrupts the majority class (56.8% of all 176 scenarios
  argmax to `kv`) of the proposed 3-way target, and target-vs-native-family
  agreement is only 56.25% — well below the ~95% bar that would indicate a
  disguised family classifier, so this is a distinct, more specific failure
  than `MECHANISM_TARGET_FAMILY_PROXY_ONLY`.
- A hypothetical two-stage (mechanism-choice → within-mechanism policy)
  pipeline retains **zero net oracle-approximation advantage** over a
  single fixed global policy (mean regret 0.034 either way), with the
  confounded `kv` bucket carrying the worst regret (0.055) and the largest
  population.
- Only `ranking` shows genuine, non-confounded cross-family activation
  (real signal on both native Family A and non-native Family C); `chunk`
  never activates outside Family B; `kv`'s cross-family reading is an
  artifact.
- Two consecutive redesign attempts (shared features, then mechanism-choice
  target) surfaced two independent, non-overlapping root causes — feature-
  space disjointness, then a contrast confound — rather than converging on
  one fixable issue.
- Next step (**not started, not authorized**): per this task's own stop
  condition, a higher-level reassessment of whether cross-family policy
  transfer is well-posed at all, rather than a third target reformulation.

**The higher-level reassessment named above is now
COMPLETE — `CROSS_FAMILY_TRANSFER_DEMOTED_HIERARCHICAL_ROUTING_READY`.**

- Audit: [`../audits/cross_family_transfer_wellposedness_reassessment_20260817.md`](../audits/cross_family_transfer_wellposedness_reassessment_20260817.md)
- Artifacts: `experiments/cross_family_transfer_wellposedness_reassessment_v1/`;
  script `scripts/analyze_cross_family_task_separation_v1.py`.
- Synthesized all three prior NO_GOs: each failed for a genuinely
  **different** diagnosed reason (structural leakage → geometric
  disjointness → target-contrast confound), which is itself convergent
  evidence against a *universal* per-scenario selector (H1–H4 all
  `SUPPORTED`), not merely three retries of the same problem.
- **But within-family evidence remains strong** (each family's own
  best-fixed-vs-oracle gap is small, 2.0–4.9pp) **and some real
  cross-family structure exists**: `estimated_service_time_first` wins the
  oracle in all three families; policy-ranking similarity is moderately
  positive between A↔B/A↔C (ρ≈0.53–0.59, not significant at n=6 policies).
  This is not enough for a per-scenario universal selector, but is enough
  to motivate one more, qualitatively different attempt.
- **"READY" means ready to be preregistered as the next falsification
  experiment, not validated to work.** Two concrete, named open risks must
  be gated explicitly: (1) Family B's `chunk` regime has no direct
  online-observable proxy in SHARED_CORE_V1 (only an indirect,
  unvalidated request-size-mix correlate) — the router would likely be
  weakest exactly where a wrong decision is most costly (Family B has the
  largest single-family oracle gap, 0.049); (2) every feature validated so
  far is a whole-scenario retrospective aggregate, never tested on
  genuinely online/partial-trajectory state.
- Recommended next experimental question (**not started, not
  authorized**): design (not launch) a hierarchical regime-router +
  family-specific-selector experiment, with the audit's 9 GO/STOP gates
  (§M) frozen in its design doc before any TRAIN/TEST data is touched.

**The online regime-signal feasibility study named above is now
COMPLETE — `ONLINE_REGIME_SIGNALS_READY`.**

- Audit: [`../audits/online_regime_signal_feasibility_v1_20260817.md`](../audits/online_regime_signal_feasibility_v1_20260817.md)
- Artifacts: `experiments/online_regime_signal_feasibility_v1/` (127,319-row
  per-step telemetry, all 176 frozen scenarios replayed through FIFO —
  native to none of the three families — with `TelemetryRecordingPolicy`);
  module `src/llmserveopt/policy_separation/online_regime_signals_v1.py`;
  build/diagnostics scripts `scripts/build_online_regime_telemetry_v1.py`,
  `scripts/analyze_online_regime_telemetry_v1.py`; tests
  `tests/test_online_regime_signals_v1.py` (15/15 passing).
- **Directly resolves both open risks the reassessment named.** Every
  signal is computed from `ObservableState` — the exact pre-decision
  snapshot every real policy already receives before
  `select_action` — reusing (not reimplementing) `causal_context_features`/
  `_prefill_pressure`/`_decode_pressure`/`_kv_pressure`, which are already
  load-bearing live inside real production policies elsewhere in the
  codebase.
- **Family-B contention (the primary gate) is detectable**, but only after
  a documented correction: the first, capacity-normalized contention
  formula (`prefill_pressure × decode_pressure`) never fired even on
  Family B's own scenarios (max_active_sequences=512 is too generous a
  denominator at Family B's ~24-request scale) — its AUROC was still 0.841,
  showing real ranking signal despite the miscalibrated threshold. A
  structurally different, active-fraction-normalized formula
  (`contention_score_v2`) fires on 32/32 Family-B scenarios with **zero
  false positives** anywhere outside Family B (AUROC 0.841, precision
  1.0, recall 0.43).
- **All three activity signals achieve perfect precision** (0 cross-family
  false positives across 127,319 rows) with moderate recall; `kv_pressure`
  is the strongest single feature (AUROC 0.993 for Family C).
- **Zero regime overlap observed anywhere** (no A+B/A+C/B+C/A+B+C rows) —
  supports a **hard top-1** router architecture, not the softer
  multi-label variant — with an explicit caveat that this may partly
  reflect how structurally distinct the three frozen scenario designs are,
  not yet evidence about genuinely blended live traffic.
- Next step (**not started, not authorized**): a separately authorized,
  preregistered hierarchical-router experiment (Stage-1 regime classifier
  on these validated signals + Stage-2 family-specific selectors), gated
  by the reassessment's 9 GO/STOP criteria.

**The hierarchical regime router named above has now been designed,
implemented, AND evaluated on TEST — final verdict `HIERARCHICAL_ROUTER_NO_GO`.**

- Design/preregistration (frozen, unmodified throughout): [`../design/HIERARCHICAL_REGIME_ROUTER_V1.md`](../design/HIERARCHICAL_REGIME_ROUTER_V1.md), `configs/hierarchical_regime_router_v1_gates.json` (commit `078f4f1`).
- Implementation (`HIERARCHICAL_ROUTER_IMPLEMENTATION_COMPLETE`, commit `2923087`): `src/llmserveopt/policy_separation/hierarchical_regime_router_v1.py`, `hierarchical_router_evaluation_v1.py`, `hierarchical_router_gates_v1.py`, `src/llmserveopt/selector/hierarchical_stage2_selectors_v1.py`; 84 focused tests.
- TEST evaluation audit: [`../audits/hierarchical_regime_router_v1_20260818.md`](../audits/hierarchical_regime_router_v1_20260818.md).
- **G4 (Stage-2 preservation) and G5 (beat global fixed) both fail** — mechanically forces `NO_GO` (G4's `elif` branch fires first). Stage-1 (macro-F1 0.989, 0% catastrophic misrouting) and Stage-2 (0 regret standalone) are each individually excellent; the failure is diagnosed as an **integration/measurement-methodology artifact**, not a competence failure: the offline scenario-level majority-vote dispatch approximation used for this evaluation washes out regime activity that is a minority-of-steps phenomenon within a scenario (true for `KV_MEMORY_PRESSURE`), and on this particular TEST split, Stage-2's real headroom (`RANKING_FAIRNESS`'s `skew=1.0` control scenarios) happens to fall exactly on the scenarios Stage-1 correctly, non-leakily declines to route.
- **Family B (`PREFILL_DECODE_CONTENTION`) got 0 TEST scenarios** on this split (8 groups, deterministic hash) — G4/Stage-2-B untested, not merely underpowered.
- One genuine, first-ever `OVERLAP` observation surfaced in the B+C blended microcase (prior feasibility study: 0/127,319) — the router's fallback engaged safely (G9 passes).
- Next step (**not started, not authorized**): build a genuine per-step live-simulation evaluation harness (the offline majority-vote approximation is the diagnosed root cause of the flat-zero result) before any re-evaluation; do not silently re-run under the same preregistration.

**The genuine per-step live-simulation evaluation harness named above is now
COMPLETE, and the live re-evaluation it exists to enable has now been RUN
and FORMALLY GATE-SCORED — final formal verdict `HIERARCHICAL_ROUTER_NO_GO`
(agrees with the ad-hoc `LIVE_REEVAL_CONFIRMS_NO_GO` the run script printed).**

- Harness: [`../audits/hierarchical_router_live_harness_validation_v1_20260818.md`](../audits/hierarchical_router_live_harness_validation_v1_20260818.md)
  (implementation commit `723a39c`); readiness verdict
  `LIVE_HIERARCHICAL_HARNESS_READY` — 6/6 forced-parent equivalence checks
  bit-exact against standalone single-policy runs, and a causal-switch
  microcase directly demonstrates the live trajectory diverging from a
  fallback-only trajectory only after a real Stage-2 switch (i.e. genuinely
  per-step causal, not a majority-vote re-approximation).
- Live re-evaluation preregistration (`6c9ec36`) and run (`9fde981`, fixed in
  `ed74276` — the run script's first launch crashed on a malformed
  `group_resampled_bootstrap_ci` call before any analysis executed; no
  results existed before the fix): [`../audits/hierarchical_regime_router_live_reeval_v1_20260818.md`](../audits/hierarchical_regime_router_live_reeval_v1_20260818.md).
  Result artifact: `experiments/hierarchical_regime_router_live_reeval_v1/live_reeval_results.json`.
- Primary numbers (32-scenario exact TEST split): live ANWG 0.8136 vs. best
  global fixed 0.8075 (`delta_fixed` = 0.00616, 90% CI
  `[0.00055, 0.01140]` — excludes zero but well under the G5 practical-
  significance bar of 0.01) vs. six-policy oracle 0.8506 (oracle-gap
  closure 0.143, well under G6's 0.75 bar). Stage-1 macro-F1 0.989,
  catastrophic misroute 0.0.
- **Formal gate-conformant rescoring (`scripts/rescore_hierarchical_regime_router_live_reeval_v1_gates.py`,
  reusing the same frozen `evaluate_all_gates`/`compute_verdict` implementation
  the TEST evaluation used, not a hand-written substitute): `FORMAL_GATE_VERDICT
  = HIERARCHICAL_ROUTER_NO_GO`.** G5 (beat global fixed) fails outright on its
  mean criterion; G6 (oracle gap closure) also fails. G4 (Stage-2
  preservation), G7 (multi-regime benefit), and G9(a) (Family-C held-out
  delta) could not be mechanically scored from the persisted result artifact
  (it records only the TEST-aggregate live ANWG, not a per-regime
  breakdown) and are reported `NOT_EVALUABLE`, not assumed passing — this
  does not change the verdict, since G5 alone already forces `NO_GO`.
- **Family B (`PREFILL_DECODE_CONTENTION`) again got 0 TEST scenarios** on
  this exact split (same deterministic hash as the approximate evaluation)
  — the live re-evaluation confirms `NO_GO` for two of the three regimes
  (`RANKING_FAIRNESS`, `KV_MEMORY_PRESSURE`); it does **not** constitute a
  three-regime validation. Do not cite it as such.
- Known, transparently-documented provenance defects (not corrected in
  place, to preserve historical evidence): the live-reeval
  `launch_manifest.json`'s `git_sha` field does not resolve to a real git
  object (short-SHA corruption, unrelated to the result's own correctly-
  self-recorded provenance); `fitted_model_hashes.json` omits a
  `KV_MEMORY_PRESSURE` model hash. See the finalized audit for detail.
- **No new scientific experiment is currently running.** Next step (**not
  started, not authorized**): decide whether to pursue a Family-B-specific
  live evaluation (the one regime never TEST-evaluated in this entire
  lineage) or a higher-level reassessment of the hierarchical-routing
  hypothesis itself, analogous to the `dc5757b` composition reassessment.

## Most Recently Completed Work (WS-P / Policy Separation)

**Family B v2 prefill/decode TTFT-contention refinement is COMPLETE.**

- Audit: [`../audits/policy_separation_prefill_decode_pilot_v2_20260817.md`](../audits/policy_separation_prefill_decode_pilot_v2_20260817.md)
- Provenance: [`../../experiments/policy_separation_prefill_decode_pilot_v2_20260817T024204Z/`](../../experiments/policy_separation_prefill_decode_pilot_v2_20260817T024204Z/)
- Family verdict: **`FAMILY_B_COMPOSITION_READY`**
- Two anchors only (`full_prefill` vs `chunked_prefill_small`): 16/15 practical wins at ε=0.01, near-tie 3.1% (v1 was 96%), mean \|Δ\|=0.131, seed agree 0.875, held-out seed bidirectional, mechanism = class TTFT.
- Frozen Family B v1 remains `USEFUL_BUT_NEEDS_REFINEMENT` / `PREFILL_COMPOSITION_NOT_YET_JUSTIFIED` ([`../audits/policy_separation_prefill_decode_pilot_v1_20260817.md`](../audits/policy_separation_prefill_decode_pilot_v1_20260817.md)); do not rewrite that CSV.
- Next WS-P step: **smallest two-parent PrefillControl composition falsification** (not GP / MAP-Elites / LLM synth). Do not run it as part of the v2 audit.

**PrefillControl composition falsification (`full_prefill` vs `chunked_prefill_small`) is now COMPLETE.**

- Audit: [`../audits/family_b_v2_prefill_control_composition_falsification_20260817.md`](../audits/family_b_v2_prefill_control_composition_falsification_20260817.md)
- Provenance: [`../../experiments/prefill_control_composition_v2_20260817T154633Z/`](../../experiments/prefill_control_composition_v2_20260817T154633Z/) (32 scenarios, train=16/val=8/test=4/ood=4, 120/120 success)
- Verdict: **`SELECTION_SUFFICIENT_FOR_THIS_PAIR`**
- A real TRAIN/VAL-fitted contextual top-1 selector reaches the two-parent oracle envelope exactly (0 regret) on both TEST and OOD. The genuinely per-step-dynamic `prefill_control_child` policy (verified not to collapse to any fixed baseline) never beats that selector and never expands the oracle envelope on held-out data. Symbolic distillation / broader module composition / MAP-Elites are **not** justified from this pair alone — see the audit's mechanism analysis for why a different per-step rule remains untested, not falsified.

**Family C v1 KV-pressure reserve pairwise-separation pilot (new mechanism family) is now COMPLETE.**

- Design: [`../design/POLICY_SEPARATION_FAMILY_KV_PRESSURE_V1.md`](../design/POLICY_SEPARATION_FAMILY_KV_PRESSURE_V1.md)
- Audit: [`../audits/family_c_kv_pressure_pairwise_separation_v1_20260817.md`](../audits/family_c_kv_pressure_pairwise_separation_v1_20260817.md)
- Provenance: [`../../experiments/kv_pressure_pilot_v1_20260817T162650Z/`](../../experiments/kv_pressure_pilot_v1_20260817T162650Z/) (32 scenarios, 64/64 success)
- Parents: `kv_constrained_online` (soft KV-occupancy admission reserve) vs `least_laxity_first` (KV-blind laxity-greedy)
- Verdict: **`KV_FAMILY_USEFUL_NEEDS_REFINEMENT`** (5/6 gates pass: bidirectional wins 9-vs-4/32, mechanism activates 28,695 logged deferrals, no twin; tie-rate gate 59.4% did not clear its <50% bound)
- **This is the first family (of ESTF/WFS, PrefillControl, KV-pressure) to demonstrate genuine within-scenario mechanism opportunity**, not just a scenario-level contrast: KV-constrained's advantage over LLF on urgent-tenant SLO attainment is 2× larger when urgent tenants arrive after KV pressure has built up vs before (0.125 vs 0.0625 mean ANWG delta, matched cells) — exactly the structural precondition ESTF/WFS and PrefillControl lacked.
- **This is a pairwise-separation pilot only — no composition work was started or is currently justified.** Next step is refining this family (larger pilot to test whether the tie-rate gate clears with more power), not a composition falsification and not MAP-Elites/GP/distillation/LLM synthesis.

**Family C v2 KV-pressure reserve refinement is now COMPLETE — `KV_FAMILY_COMPOSITION_READY`.**

- Design: [`../design/POLICY_SEPARATION_FAMILY_KV_PRESSURE_V2.md`](../design/POLICY_SEPARATION_FAMILY_KV_PRESSURE_V2.md)
- Audit: [`../audits/family_c_kv_pressure_pairwise_separation_v2_20260817.md`](../audits/family_c_kv_pressure_pairwise_separation_v2_20260817.md)
- Provenance: [`../../experiments/kv_pressure_pilot_v2_20260817T165053Z/`](../../experiments/kv_pressure_pilot_v2_20260817T165053Z/) (72 scenarios, 144/144 success; v1's frozen run untouched)
- v1's tie-rate gap (59.4%) diagnosed to two root causes: coarse ANWG resolution at the v1 population size, and an accidental confound where bulk "background" tenants were themselves often classified urgent by the policy's own threshold. v2 fixed both (population roughly doubled; bulk slack recalibrated) and added a third arrival-phase level — all changes justified against the diagnosis, not tuned toward a preferred outcome (design doc §1-2 documents the full reasoning, including a case where a further "fix" was tried and rejected because it didn't change the qualitative picture).
- **All 10 preregistered gates pass**, including two new ones beyond v1's set: G6 (the within-scenario timing pattern replicates on 2 held-out seeds never used in any calibration decision — it does, at comparable-or-larger magnitude) and G10 (6 of 16 matched scenario cells show a *different practical winner* depending purely on when urgent tenants arrive within the same scenario, holding everything else fixed).
- **This is the first family, of the three studied, to reach `_COMPOSITION_READY`** — stronger motivating evidence for composition than ESTF/WFS or PrefillControl v2 produced, neither of which ever showed a within-scenario-timing dependency (both were already `SELECTION_SUFFICIENT_FOR_THIS_PAIR`, meaning a scenario-level selector was sufficient).
- **Important precision (audit §S):** this shows the *scenario-level optimal parent choice* depends on within-trajectory timing, and that a scenario-level selector alone therefore has less headroom to be sufficient here than in the other two families — it does **not** yet prove a state-dependent child would beat *both* fixed parents on the *same* trajectory. That is exactly what a composition falsification would test.
- **No composition work was started in that task**, per explicit scope. The audit stated what the smallest next composition falsification would look like without running it.

**KV-aware composition falsification v1 is now COMPLETE — `KV_COMPOSITION_INCONCLUSIVE`.**

- Design: [`../design/KV_COMPOSITION_FALSIFICATION_V1.md`](../design/KV_COMPOSITION_FALSIFICATION_V1.md)
- Audit: [`../audits/kv_composition_falsification_v1_20260817.md`](../audits/kv_composition_falsification_v1_20260817.md)
- Provenance: [`../../experiments/kv_composition_falsification_v1_20260817T172446Z/`](../../experiments/kv_composition_falsification_v1_20260817T172446Z/) (72 scenarios, 576/576 success)
- The child (`KVAdaptiveReserveChildPolicy`) delegates every step, unmodified, to `kv_constrained_online` or `least_laxity_first`, chosen from a single online-observable trigger (count of currently-waiting urgent-classified requests ≥ a TRAIN/VAL-fit `tau_urgent`). No new admission logic.
- **This is a qualitatively different outcome from ESTF/WFS and PrefillControl v2's `SELECTION_SUFFICIENT_FOR_THIS_PAIR` verdicts** — 6/8 gates pass with real signal (positive TEST envelope gain, 5/12 TEST scenarios beat both parents by >ε, genuine non-degenerate within-trajectory mode-switching on 24/36 held-out scenarios, directionally-consistent OOD replication), but **G7 (safety) fails**: on 6/36 (16.7%) held-out scenarios the child's peak KV utilization exceeds `max(parent peak utilizations)` by 0.013-0.033 — a composition-specific risk (mode-switching history creates KV states neither pure parent alone reaches) that a pairwise-separation pilot structurally cannot surface. Per the frozen decision rule, G7 failing forces `KV_COMPOSITION_INCONCLUSIVE` regardless of G1-G6.
- **Independent, important finding surfaced during this task's cross-checks (not part of any gate):** re-running the original, unmodified KV v2 pilot runner in the current environment reproduces itself perfectly (0/144 mismatch across independent reruns) but does **not** reproduce the historical frozen KV v2 CSV (99/144 rows mismatch, up to 0.25 ANWG). This falsification's own gates remain valid (all methods compared were computed from one internally-consistent run). **Forensic follow-up complete:** see below.
- **Per task scope, this outcome does not license escalating to a more complex child, MAP-Elites, or synthesis.** The smallest defensible next step (not started) would be a narrowly-rescoped child adding a transition-aware admission cap, re-run through the identical frozen procedure.

**KV v2 reproducibility forensic audit is COMPLETE — `REPRODUCIBILITY_GAP_BOUNDED`.**

- Audit: [`../audits/kv_v2_reproducibility_forensic_20260817.md`](../audits/kv_v2_reproducibility_forensic_20260817.md)
- Root cause **not demonstrated**. Ruled out/narrowed: code drift (zero diff on the entire KV v2 execution path between the historical launch commit `6be526e` and current HEAD), runtime/multiprocessing nondeterminism (current environment is byte-identical-SHA-256-reproducible across independent reruns and across `--workers 1` vs `--workers 4`), and both locally available BurstGPT dataset files (neither reproduces the historical CSV; their derived sampling pools are nearly but not exactly identical — filtered `[1024,3072)` pool length 7335 vs 7337 — demonstrating the pipeline's sensitivity to even a 2-row pool difference without pinning down which file, if either, the historical run actually used).
- Historical mismatch is scientifically material, not bit-level noise: 99/144 cells differ (max `|ΔANWG|`=0.25), and the practical (ε=0.01) parent winner flips on 17/72 (24%) scenarios.
- **Three questions kept explicitly separate:** (1) exact historical KV v2 reproducibility is weakened; (2) the composition falsification's internal validity is **not** weakened — every method it compares was evaluated in one single current-environment run; (3) any future cross-run comparison against the historical v2 numbers requires caution.
- **No historical CSV, verdict, or audit conclusion was rewritten.** `KV_FAMILY_COMPOSITION_READY` (v2) and `KV_COMPOSITION_INCONCLUSIVE` (composition falsification) both stand as originally recorded, with this document added as a standing provenance caveat.
- **Forward-looking guard added:** `scripts/run_policy_separation_kv_pressure_pilot_v1.py` now records additive provenance (git SHA/dirty, config+dataset SHA-256, library versions, result-CSV SHA-256, timestamp) in every future run's `final_summary.json` — behavior-neutral, 14 new focused tests pass, does not affect scenario generation, RNG order, or metrics.

**ESTF↔WFS minimal composition falsification remains COMPLETE.**

- Audit: [`../audits/estf_wfs_composition_falsification_v1_20260816.md`](../audits/estf_wfs_composition_falsification_v1_20260816.md)
- Provenance: [`../../experiments/estf_wfs_composition_falsification_v1_20260816T222108Z/`](../../experiments/estf_wfs_composition_falsification_v1_20260816T222108Z/)
- Verdict: **`SELECTION_SUFFICIENT_FOR_THIS_PAIR`**
- Contextual rank composition does not beat contextual top-1 on TEST; parent
  envelope gain is 0. Symbolic distillation / MAP-Elites / LLM synthesis are
  **not** justified from this pair alone.

Family A v2 Job 1182377 remains validated complementary-parent evidence
(`USEFUL_BUT_NEEDS_REFINEMENT`):
[`../audits/policy_separation_fairness_starvation_pilot_v2_20260816.md`](../audits/policy_separation_fairness_starvation_pilot_v2_20260816.md).

Family A v1 Job 1182306 remains frozen diagnostic evidence
(`USEFUL_DIAGNOSTIC_ONLY` / `REDESIGN_REQUIRED`; historical CSV `anwg` =
unweighted SLO-success, not canonical ANWG):
[`../audits/policy_separation_fairness_starvation_pilot_v1_20260816.md`](../audits/policy_separation_fairness_starvation_pilot_v1_20260816.md).

## Latest Major Result (Apt-Serve/CC thread)

**Apt-Serve Phase G completed.**

- Collection: complete.
- Posthoc analysis: complete with wrapper `exit_code=0`.
- Canonical collection output:
  `results/apt_serve_phase_g_resume_20260807_174028/`.
- Preserved failed SS15 source run:
  `results/apt_serve_phase_g_overnight_20260807_011542/`.
- Canonical analysis output:
  `results/apt_serve_phase_g_analysis_20260809_190000/`.
- Audit:
  [`../audits/apt_serve_phase_g_analysis_20260809.md`](../audits/apt_serve_phase_g_analysis_20260809.md).

Supported interpretation:

- The Phase G dataset is structurally valid.
- Apt-Serve has positive leave-one-out marginal contribution to the policy
  portfolio: mean `0.025219`, grouped bootstrap CI `[0.004099, 0.057757]`.
- Global Apt-vs-best-fixed superiority is not established: mean gap
  `0.012032`, grouped bootstrap CI `[-0.013237, 0.046700]`.
- The best fixed baseline by mean ANWG is `scorpio_style_slo_guard`.
- Apt-Serve is one evaluated external scheduler family and a potential source
  of cache/tier-transition modules, not the whole project.

## Current Project Position

- CC0-CC5: complete; CC5 remains `COMPLETE_REGIME_SPECIFIC`.
- CC6: not started; requires explicit authorization and a scoped design.
- External baselines: current status is centralized in
  [`../BASELINE_STATUS.md`](../BASELINE_STATUS.md).
- Apt-Serve: Phase G analysis is complete; no new Apt-Serve collection job is
  queued.
- WS-P: Family A v2 analyzed; ESTF↔WFS composition =
  `SELECTION_SUFFICIENT_FOR_THIS_PAIR`; Family B (the next mechanism family
  after ESTF/WFS) v1 is `USEFUL_BUT_NEEDS_REFINEMENT` /
  `PREFILL_COMPOSITION_NOT_YET_JUSTIFIED`; v2 is `FAMILY_B_COMPOSITION_READY`;
  PrefillControl composition falsification on the v2 pair = `SELECTION_SUFFICIENT_FOR_THIS_PAIR`
  ([`../audits/family_b_v2_prefill_control_composition_falsification_20260817.md`](../audits/family_b_v2_prefill_control_composition_falsification_20260817.md));
  Family C v2 KV-pressure reserve refinement = `KV_FAMILY_COMPOSITION_READY`
  ([`../audits/family_c_kv_pressure_pairwise_separation_v2_20260817.md`](../audits/family_c_kv_pressure_pairwise_separation_v2_20260817.md));
  v1 pilot remains `KV_FAMILY_USEFUL_NEEDS_REFINEMENT` (frozen, superseded
  by v2, not rewritten). The KV-pressure composition falsification on that
  pair is complete: `KV_COMPOSITION_INCONCLUSIVE`
  ([`../audits/kv_composition_falsification_v1_20260817.md`](../audits/kv_composition_falsification_v1_20260817.md))
  — real envelope-gain signal, blocked specifically by a composition-induced
  KV-safety gate failure, not by absence of signal.

## Exact Next Tasks (two independent threads)

1. **WS-P:** Family B v2 analysis is complete
   ([`../audits/policy_separation_prefill_decode_pilot_v2_20260817.md`](../audits/policy_separation_prefill_decode_pilot_v2_20260817.md)).
   Verdict `FAMILY_B_COMPOSITION_READY`. ESTF↔WFS composition pilot verdict:
   `SELECTION_SUFFICIENT_FOR_THIS_PAIR`
   ([`../audits/estf_wfs_composition_falsification_v1_20260816.md`](../audits/estf_wfs_composition_falsification_v1_20260816.md)).
   PrefillControl composition falsification (`full_prefill` vs
   `chunked_prefill_small`) is COMPLETE, verdict
   `SELECTION_SUFFICIENT_FOR_THIS_PAIR`
   ([`../audits/family_b_v2_prefill_control_composition_falsification_20260817.md`](../audits/family_b_v2_prefill_control_composition_falsification_20260817.md)).
   **Family C v2 KV-pressure reserve** (`kv_constrained_online` vs
   `least_laxity_first`) reached `KV_FAMILY_COMPOSITION_READY`
   ([`../audits/family_c_kv_pressure_pairwise_separation_v2_20260817.md`](../audits/family_c_kv_pressure_pairwise_separation_v2_20260817.md);
   design [`../design/POLICY_SEPARATION_FAMILY_KV_PRESSURE_V2.md`](../design/POLICY_SEPARATION_FAMILY_KV_PRESSURE_V2.md))
   — the first of the three families studied to justify a composition
   falsification. **That falsification has since been run to completion:**
   `KV_COMPOSITION_INCONCLUSIVE`
   ([`../audits/kv_composition_falsification_v1_20260817.md`](../audits/kv_composition_falsification_v1_20260817.md);
   design [`../design/KV_COMPOSITION_FALSIFICATION_V1.md`](../design/KV_COMPOSITION_FALSIFICATION_V1.md)).
   A minimal state-dependent child (delegates every step, unmodified, to one
   of the two frozen parents based on an online-observable urgent-queue-depth
   trigger) showed real signal — positive TEST envelope gain, 5/12 TEST
   scenarios beating both parents by >ε, genuine non-degenerate
   within-trajectory mode-switching on 24/36 held-out scenarios,
   directionally-consistent OOD replication — but the frozen safety gate
   (G7) failed: on 6/36 held-out scenarios the child's peak KV utilization
   exceeded `max(parent peak utilizations)`, a composition-specific risk
   (mode-switching history creates KV states neither pure parent alone
   reaches) that a pairwise-separation pilot cannot surface. Per the frozen
   decision rule this forces `KV_COMPOSITION_INCONCLUSIVE` regardless of the
   otherwise-favorable G1-G6 results. **Do not** escalate to a more complex
   child, MAP-Elites, symbolic distillation, or LLM synthesis from this
    result — per its own audit §Z, the only defensible next step (not
   started) is a narrowly-rescoped child adding a transition-aware admission
   cap, re-run through the identical frozen procedure. **Separately,** this
   task surfaced an unresolved reproducibility gap in the whole KV v1/v2
   evidentiary chain (audit §P) — the current environment cannot reproduce
   the historical frozen KV v2 CSV bit-for-bit even by re-running the
   original unmodified runner; root cause not identified, flagged for a
   dedicated follow-up.

## Family-B Balanced Replication (IMPLEMENTATION_READY)

- Design/artifacts: `experiments/family_b_balanced_replication_v1/` (added in commit `9d8f997`).
- The smoke-synthetic run infrastructure is prepared; the scientific replication across actual scenarios has not yet executed.
- **Status: READY TO RUN — explicit authorization required before launching.** Family-B was the only regime never tested by the hierarchical router's TEST or live evaluations (0 scenarios in both).
- Current provenance-timestamp diffs in `run_smoke_synthetic_results.json` are provenance-only; no scientific result was changed.

## Public Trace Corpus v1 (IMPLEMENTED and KEPT)

- Design: `docs/design/PUBLIC_TRACE_CORPUS_V1.md`
- Schema: `data/public_trace_corpus_v1/schema.json`
- Build script: `scripts/build_public_trace_corpus_v1.py`
- Adapter module: `src/llmserveopt/workloads/public_trace_corpus.py`
- Tests: `tests/test_public_trace_corpus_v1.py` (250 lines, 22 test functions)
- Artifacts: `data/public_trace_corpus_v1/` (manifest, distribution stats, source coverage)
- Commits `84fa31b` + `179a6fe` are **KEPT**: technically sound, aligned with project goals, no frozen scientific artifacts touched.
- Scope: Workload-input layer only (Layers 0-1). Ingests BurstGPT, Azure 2023 conv+code; classifies AgentPerfBench as REAL_SYSTEM_VALIDATION_SOURCE. No policy outcomes, no oracle labels, no paid API use.
- **Important note:** The prior forensic report incorrectly claimed a tracked `public_trace_corpus/` directory was deleted. This was uncommitted local work only. No git-tracked content was removed.
- **Next step:** Layer 2+ (policy replay) is NOT started. Cohere/CloudRift belong only in real-LLM validation (Layer 6).

## Decision-Criticality / Timescale Analysis (PREREGISTERED + IMPLEMENTED, EXPERIMENT NOT RUN)

- Files: `docs/design/DECISION_CRITICALITY_TIMESCALE_TRAINVAL_V1.md`, `src/llmserveopt/analysis/decision_criticality_timescale_trainval_v1.py`, `scripts/run_decision_criticality_timescale_trainval_v1.py`, `tests/test_decision_criticality_timescale_trainval_v1.py`
- Status: design, implementation, and 40 tests are **committed and pushed** (commit `4dac220`, "feat: preregister decision-criticality train-val study"). (At the time of the earlier pass-1/pass-2 audits on 2026-08-19, these files were still untracked, owned by a separate concurrent workstream — that is no longer the case.)
- The actual 144-scenario TRAIN/VAL execution has **not** been run: no `experiments/decision_criticality_timescale_trainval_v1/` output directory or run log exists, and no scientific conclusion has been drawn. Launching it requires separate, explicit authorization, same as Family-B replication. It must not import or use the Family-B held-out replication as evidence (enforced by a runtime + source-text guard against importing `family_b_balanced_replication_v1`).

## Exact Next Tasks (three independent threads)

1. **Family-B replication:** Implementation-ready (added in commit `9d8f997`). Scientific run NOT YET STARTED per task instructions. Requires explicit authorization.
2. **Public Trace Corpus v1:** Implementation complete (commits 84fa31b/179a6fe). No further corpus development until Layer 2+ is explicitly authorized.
3. **Decision-criticality analysis:** Design/implementation/tests committed and pushed (commit `4dac220`). Scientific TRAIN/VAL execution NOT YET STARTED. Requires explicit authorization.

**The two independent main threads remain:**
**WS-P:** Family B v2 analysis → PrefillControl composition → Family C v2 → KV composition falsification — all complete. Family-B balanced replication implementation-ready. See items 1-3 above for next actions.
**Apt-Serve/CC:** Perform the post-Phase-G module-envelope interpretation and decide next module-decomposition/compositional-learning step.

## Do Not Do By Default

- Do not launch Family-B scientific replication without explicit authorization.
- Do not advance public-trace corpus beyond workload-input layer without authorization.
- Do not launch the decision-criticality TRAIN/VAL run without explicit authorization.
- Do not claim Apt-Serve globally beats the best fixed baseline.
- Do not treat Apt-Serve as the project endpoint.
- Do not start CC6 without explicit authorization.
- Do not delete Phase G artifacts or historical negative-result audits.
- Do not start MAP-Elites, selector retraining, or broad synthesis from PSD yet.
- Do not train selectors on Family A v1 rows.
- Do not rewrite Job 1182306 CSV rows.
- Do not use local `results/` absence as proof an experiment never ran; check
  the audit trail.

## Navigation

- Public overview: [`../../README.md`](../../README.md)
- Research roadmap: [`../PROJECT_MAP.md`](../PROJECT_MAP.md)
- Detailed status: [`WORK_STATUS.md`](WORK_STATUS.md)
- Prioritized next actions: [`NEXT_ACTIONS.md`](NEXT_ACTIONS.md)
- External-baseline index: [`../BASELINE_STATUS.md`](../BASELINE_STATUS.md)
- Documentation index: [`../README.md`](../README.md)
