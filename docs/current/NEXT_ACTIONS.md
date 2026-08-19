# Next Actions

This is the current prioritized action list. It must agree with
[`../PROJECT_MAP.md`](../PROJECT_MAP.md), [`RESUME_HERE.md`](RESUME_HERE.md),
[`WORK_STATUS.md`](WORK_STATUS.md), and [`../BASELINE_STATUS.md`](../BASELINE_STATUS.md).

For the full operational narrative, read [`RESUME_HERE.md`](RESUME_HERE.md). This doc is a pointer + delta list.

For historical P0 narrative (superseded as the active P0), see the section at end of this file.

---

## NOW

1. **Decision-criticality/timescale analysis** — Uncommitted parallel workstream in progress. Owned by a separate session. Files: `src/llmserveopt/analysis/`, `docs/design/DECISION_CRITICALITY_TIMESCALE_TRAINVAL_V1.md`, `scripts/run_decision_criticality_timescale_trainval_v1.py`, `tests/test_decision_criticality_timescale_trainval_v1.py`. Do not modify.

2. **Pass-1/Pass-2 documentation reconciliation** — This file was updated on 2026-08-19 (commits in this pass). Check for consistency.

## NEXT

3. **Family-B balanced scientific replication** — Implementation-ready (commit `9d8f997`, HEAD). Smoke-synthetic infrastructure is prepared. The full scientific replication across actual scenarios is NOT YET STARTED. Requires explicit authorization — this was the only regime never tested by hierarchical router (0 scenarios in both TEST and live eval).

4. **Public Trace Corpus v1 — Layer 2+** — Workload-input layer (Layers 0-1) is complete (commits 84fa31b/179a6fe). Next step: policy replay across multiple policies to produce derived policy-separation data (Layer 3). NOT STARTED. Requires authorization.

5. **Provenance-only diffs** — `experiments/family_b_balanced_replication_v1/run_smoke_synthetic_results.json` and `experiments/hierarchical_regime_router_live_reeval_v1/gate_rescoring_v1.json` have only HEAD-SHA and timestamp updates. No scientific content changed. Leave untouched unless the owning session commits them.

## AFTER THAT

6. **Replay public workloads across policy library** — Using the completed Public Trace Corpus v1 as Layer 0-1, run Layer 2 policy replay (BurstGPT + Azure 2023 scenarios across 6+ policies). Produce the same-scenario-multi-policy dataset.

7. **Enrich with state-level separation data** — Layer 4: decision-critical states, alternative actions, counterfactual trajectories. Depends on Layer 2 completion + decision-criticality analysis.

8. **New-policy synthesis/evolution** — Layer 6+: generate or evolve new scheduling policy based on decision-critical mechanism evidence. LONG-TERM GOAL.

## DEFERRED

9. **Arbitrary router redesign** — Hierarchical routing has been evaluated (TEST + live re-eval, both `NO_GO`). No re-design without fresh authorization.

10. **Composition resurrection** — Composition was `COMPOSITION_DEMOTED` per `dc5757b`. Do not revive without fresh justification.

11. **GP / MAP-Elites / LLM-guided synthesis** — Do not start until dataset readiness gates exist (Layers 2-5 complete + decision-criticality analysis showing clear synthesis signal).

## DO NOT START WITHOUT AUTHORIZATION

12. **Family-B scientific replication** — Only after explicit task authorization, not after documentation reconciliation.

13. **Public-trace Layer 2+ policy replay** — Only after explicit task authorization.

14. **Cohere/CloudRift usage** — These belong ONLY to real-LLM validation (final stage D). Never use for coding, literature search, dataset schema design, or code review.

15. **CC6 dynamic adaptation** — After P0-P2 completion only.

16. **Add a fourth workload family** — Until existing three families are fully exhausted.

---

## Historical P0 Narrative (superseded as active P0)

**See [`RESUME_HERE.md`](RESUME_HERE.md) for current operational status.** Below is preserved for historical continuity.

### P0 - Structural Reassessment / MF-PSD (current)

**The higher-level structural reassessment of the composition hypothesis is
COMPLETE:** [`../audits/reassessment_composition_hypothesis_20260817.md`](../audits/reassessment_composition_hypothesis_20260817.md).
**Verdict: `COMPOSITION_DEMOTED`.** Revised roadmap: policy-separating
workloads -> complementary policy library -> contextual selection
(multi-family) -> mechanism attribution -> bounded envelope.

**MF-PSD v1 (revised roadmap Step 1, data unification only) is COMPLETE:**
[`../audits/multi_family_policy_separation_dataset_v1_20260817.md`](../audits/multi_family_policy_separation_dataset_v1_20260817.md).
**Verdict: `MF_PSD_READY`.** Unifies Family A v2 + Family B v2 + Family
C/KV v2 into one canonical long-form utility table (496 rows) + scenario
table (176 scenarios) at `experiments/mf_psd_v1/`, with an explicit
learnable-feature allowlist/forbidden-field denylist, exact conservation,
zero duplicates, deterministic rebuild, and zero mutation of any frozen
source. The six-anchor policy matrix is **sparse** (each family only
evaluated its own 2 anchors) — see the audit's §M/§Q for exactly what
Step 2 requires to build the dense matrix (~704 new policy-scenario
evaluations).

**Step 2 (unified six-policy utility-matrix evaluation) is now COMPLETE —
the matrix is fully dense.**

Path: MF-PSD v1 (sparse 2/6 per family) →
[`unified_policy_utility_matrix_v1_20260817.md`](../audits/unified_policy_utility_matrix_v1_20260817.md)
(Family A/B densified to 6/6; Family C blocked, `NEEDS_REFINEMENT`) →
[`family_c_step2_reconstruction_audit_20260817.md`](../audits/family_c_step2_reconstruction_audit_20260817.md)
(diagnosed the Family-C blocker: exact historical KV v2 replay is
structurally impossible — its runner never serialized request-level data,
unlike Family A/B — verdict `FAMILY_C_RECONSTRUCTION_BOUNDED`) →
[`family_c_reconstruction_v1_and_unified_matrix_completion_20260817.md`](../audits/family_c_reconstruction_v1_and_unified_matrix_completion_20260817.md)
(built `CURRENT_RECONSTRUCTED_FAMILY_C_V1` — a new, explicitly-versioned
layer, **not** historical replay: all 6 anchors, including the 2 native
ones, freshly evaluated on one generate-once, byte-exact-replayable set of
72 scenarios — then rebuilt the matrix as v2).

**Final verdict: `UNIFIED_UTILITY_MATRIX_READY`.**
`experiments/unified_utility_matrix_v2/` — **176×6 = 1,056/1,056 cells
populated, 0 missing, 0 failed.** 54.0% unique-winner rate, positive
oracle gain over best-fixed in every family (0.02–0.05), no anchor
universally dominant. Historical KV v2 evidence (`KV_FAMILY_COMPOSITION_READY`)
and MF-PSD v1's frozen Family-C rows are preserved unchanged and were never
mixed row-wise with the new reconstruction (crosswalk diagnostic in the
completion audit §D: 99/144 real differences vs. history, independently
reproducing the forensic audit's own numbers exactly — expected, not an
error, since this is a fresh sample, not a replay).

**Known caveat carried forward, not resolved and not blocking `READY`:**
Family B (32/176 scenarios) has near-total policy collapse — 5 of 6
anchors are byte-identical to each other on every Family-B scenario (only
`chunked_prefill_small` differs), because none of the four ranking
policies touch the chunk-budget axis Family B isolates. This does not
meaningfully dilute the matrix's aggregate diversity (54.0% unique-winner
with Family B vs. 55.6% without it), but **Step 3's design must account
for it** — e.g. a leave-Family-B-out fold is expected to be uninformative
beyond the `chunked_prefill_small` contrast.

**Step 3 (preregistered multi-family contextual-selector experiment) is
now COMPLETE:**
[`../audits/multifamily_contextual_selector_v1_20260817.md`](../audits/multifamily_contextual_selector_v1_20260817.md)
(design: [`../design/MULTIFAMILY_CONTEXTUAL_SELECTOR_V1.md`](../design/MULTIFAMILY_CONTEXTUAL_SELECTOR_V1.md)).
**Verdict: `MULTIFAMILY_SELECTOR_NO_GO`** (mechanical, all 5 frozen gates
failed on the pooled Regime-B holdout). Key findings: within-family
(Regime A) contextual selection is strong (near-perfect, 0 regret, on
Family A/B holdouts) — the matrix contains real learnable structure — but
pooled (Regime B) and leave-one-family-out (Regime C) generalization both
fail: every trained model underperforms the trivial majority baseline on
the pooled holdout, and LOFO wins only 1/3 directions, with a severe
(6.2×-worse-than-fixed) collapse when Family A is held out. Root cause,
directly diagnosed: `mechanism_family` is predictable at 100% accuracy
from the feature schema alone (every learnable feature is family-prefixed
with structural missingness), and a shared-feature-only robustness check
(A↔B, Family C excluded — no 3-family-shared feature exists) shows the
selector cannot beat best-fixed at all once family-identifying feature
blocks are removed. **Mechanism attribution (Step 4) remains blocked** —
this verdict does not justify it.

**The feature-schema redesign investigation named above is now COMPLETE:**
[`../audits/shared_cross_family_feature_schema_feasibility_v1_20260817.md`](../audits/shared_cross_family_feature_schema_feasibility_v1_20260817.md).
**Verdict: `SHARED_FEATURE_SCHEMA_NO_GO`.** A genuinely shared,
zero-missingness, replay-verified 17-feature schema (SHARED_CORE_V1,
`experiments/shared_cross_family_features_v1/`) was successfully built for
all 176/176 scenarios — the feature-*construction* half of the
investigation succeeded. But two independent problems block a GO: (1)
family remains 100% classifiable from SHARED_CORE_V1 alone, not from
missingness (there is none) but because the three families' workloads
occupy almost entirely disjoint regions of shared feature space
(range-overlap ≈0 on 15/17 features), and cross-family nearest-neighbor
scenarios are *not* more utility-consistent than random cross-family pairs;
(2) the six-policy selector target itself is not cross-family coherent —
`full_prefill`/`chunked_prefill_small` are bit-identical outside Family B,
and the other four policies collapse to one identical value specifically on
Family B, so no policy is globally meaningful. Either finding alone would
block `READY`; the target-semantics finding alone is sufficient to force
`NO_GO` per the frozen decision logic (not `NEEDS_MORE_DATA`), independent
of the feature-overlap finding.

**The 3-way mechanism-choice target redesign named above is now COMPLETE:**
[`../audits/mechanism_choice_target_feasibility_v1_20260817.md`](../audits/mechanism_choice_target_feasibility_v1_20260817.md).
**Verdict: `MECHANISM_TARGET_NO_GO`.** The `kv` mechanism contrast
(`|ANWG(kv_constrained_online) − ANWG(least_laxity_first)|`) is confounded:
it is numerically *largest* on Family A, which has essentially no KV-memory
pressure (SHARED_CORE_V1 `token_footprint_per_kv` ≈0.58, vs. ≈7.6 on
Family C, KV's own native family) — a within-family dose-response check
confirms `gain_kv` tracks real KV pressure on Family C (ρ=+0.54, p<1e-6)
but not on Family A (ρ=−0.13, p=0.28, no relationship). The contrast is
instead picking up `least_laxity_first`'s general weakness outside its
native family. This corrupts the majority class (56.8% of scenarios argmax
to `kv`); target-vs-native-family agreement is only 56.25% (well under the
~95% bar for `MECHANISM_TARGET_FAMILY_PROXY_ONLY`, so this is a distinct,
more specific failure). A hypothetical two-stage mechanism→policy pipeline
retains zero net oracle-approximation advantage over one fixed global
policy, with the confounded `kv` bucket carrying the worst per-bucket
regret and the largest population. Only `ranking` shows genuine
non-confounded cross-family activation; `chunk` never activates outside
Family B.

**The higher-level reassessment named above is now COMPLETE:**
[`../audits/cross_family_transfer_wellposedness_reassessment_20260817.md`](../audits/cross_family_transfer_wellposedness_reassessment_20260817.md).
**Project-level verdict: `CROSS_FAMILY_TRANSFER_DEMOTED_HIERARCHICAL_ROUTING_READY`**
(where "READY" means ready to be *preregistered as the next falsification
experiment*, not validated to work). Synthesizing all three prior NO_GOs:
each failed for a genuinely different diagnosed reason (structural
leakage → geometric disjointness → target-contrast confound) — convergent
evidence against a *universal* per-scenario selector specifically (H1–H4
all `SUPPORTED` in the audit), not three retries of the same problem. But
within-family evidence remains strong (each family's own best-fixed-vs-
oracle gap is small, 2.0–4.9pp) and some real cross-family structure
exists (`estimated_service_time_first` wins the oracle in all three
families; moderate policy-ranking similarity A↔B/A↔C, ρ≈0.53–0.59) —
enough to motivate one more, qualitatively different attempt rather than
stopping selector work entirely or retreating straight to family-specific-
only without testing regime routing first.

**Recommended next experimental question, NOT started, NOT authorized:**
design (not launch) a hierarchical regime-router + family-specific-
selector experiment. Two concrete open risks must be gated explicitly in
that design, not assumed resolved: (1) Family B's `chunk` regime has no
direct online-observable proxy in SHARED_CORE_V1 (only an indirect,
unvalidated request-size-mix correlate), and Family B has the largest
single-family oracle gap (0.049) — the router would likely be weakest
exactly where misrouting is most costly; (2) every validated feature so
far is a whole-scenario retrospective aggregate, never tested on
genuinely online/partial-trajectory state. The audit's §M lists 9
preregistration gates (online-observability discipline, partial-trajectory
validation, routing-accuracy bar, within-family-gain preservation,
beat-global-fixed and approach-family-aware-oracle bars, interpretable
errors, no family-label leakage) to freeze before any TRAIN/TEST data is
touched. Do not start mechanism attribution, composition/synthesis work,
a fourth target reformulation, or a fourth workload family without first
completing this design and its gates.

**The online regime-signal feasibility study named above is now COMPLETE:**
[`../audits/online_regime_signal_feasibility_v1_20260817.md`](../audits/online_regime_signal_feasibility_v1_20260817.md).
**Verdict: `ONLINE_REGIME_SIGNALS_READY`.** Replayed all 176 frozen
scenarios through the real simulator (FIFO, native to none of the three
families) with a non-invasive `TelemetryRecordingPolicy` recording
per-step `ObservableState` — the exact pre-decision snapshot every real
policy already receives — reusing already-production `causal_context_features`/
pressure helpers rather than inventing new ones. Both open risks named
above are directly resolved: (1) a Family-B contention proxy DOES exist at
the online per-step level (`contention_score_v2`, an active-fraction-
normalized formula added after the first capacity-normalized formula was
found to structurally never fire at Family B's small scale — AUROC 0.841,
32/32 scenarios detected, zero false positives outside Family B); (2)
every signal here is validated on genuinely online per-step state, not
whole-scenario retrospective aggregates. All three activity signals
(`a_active`/`b_active_v2`/`c_active`) achieve perfect precision (0
cross-family false positives across 127,319 telemetry rows); `kv_pressure`
is strongest (AUROC 0.993). Zero regime overlap was observed anywhere
(no A+B/A+C/B+C/A+B+C rows) — supports a hard top-1 router, not a
multi-label gate — with the explicit caveat that this may partly reflect
how structurally distinct the three frozen scenario designs are rather
than being proof about genuinely blended live traffic.

**Next action, NOT started, NOT authorized:** design (still not launch) the
hierarchical regime-router + family-specific-selector experiment named
above — two of its 9 preregistration gates (online-observability,
partial-trajectory validity) are now substantially de-risked by this
study's evidence; the remaining gates (routing-accuracy bar,
within-family-gain preservation, beat-global-fixed/approach-family-aware-
oracle bars, interpretable errors, no label leakage) are still untested.
Do not train the router or any family-specific selector, start mechanism
attribution, restart universal-selector work, add a fourth family, or
modify any frozen scientific result without first completing that design.

**The hierarchical router named above is now implemented AND
TEST-evaluated — final verdict `HIERARCHICAL_ROUTER_NO_GO`:**
[`../audits/hierarchical_regime_router_v1_20260818.md`](../audits/hierarchical_regime_router_v1_20260818.md)
(implementation commit `2923087`). G4 (Stage-2 preservation) and G5 (beat
global fixed) both fail on TEST; Stage-1/Stage-2 are each individually
excellent in isolation (macro-F1 0.989, 0 regret standalone) — the
failure is diagnosed as an offline scenario-level majority-vote dispatch
approximation washing out minority-of-steps regime activity, not a
competence failure of either stage. Family B got 0 TEST scenarios on
this split (untested, not underpowered). **Next action, NOT started, NOT
authorized:** build a genuine per-step live-simulation evaluation harness
before any re-evaluation — do not silently re-run under the same
preregistration, do not retune thresholds, do not start composition/
synthesis or a fourth family from this result.

**The live-simulation evaluation harness named above is now COMPLETE
(`723a39c`, verdict `LIVE_HIERARCHICAL_HARNESS_READY` — 6/6 forced-parent
equivalence checks bit-exact, causal-switch microcase confirms genuine
per-step causal dispatch, no majority vote), and the primary live
re-evaluation it exists to enable has now been RUN and FORMALLY
GATE-SCORED:**
[`../audits/hierarchical_regime_router_live_reeval_v1_20260818.md`](../audits/hierarchical_regime_router_live_reeval_v1_20260818.md)
(preregistration `6c9ec36`; the first launch attempt crashed pre-analysis
on a malformed bootstrap-CI call, fixed in `ed74276`; the completed result
is `experiments/hierarchical_regime_router_live_reeval_v1/live_reeval_results.json`).
**Formal verdict (via the same frozen `evaluate_all_gates`/`compute_verdict`
gate evaluator the TEST evaluation used, run through
`scripts/rescore_hierarchical_regime_router_live_reeval_v1_gates.py` —
not a hand-written substitute): `HIERARCHICAL_ROUTER_NO_GO`.** This agrees
with the run script's own ad-hoc-computed `LIVE_REEVAL_CONFIRMS_NO_GO`.
Live ANWG 0.8136 vs. best-global-fixed 0.8075 (`delta_fixed`=0.00616, 90%
CI `[0.00055, 0.01140]` — excludes zero but below the 0.01 practical-
significance gate); oracle-gap closure 0.143 (below the 0.75 gate). G4/G7/
G9(a) are `NOT_EVALUABLE` (the persisted result lacks a per-regime ANWG
breakdown needed to score them) — this does not change the verdict, since
G5 alone already forces `NO_GO`. Family B again got 0 TEST scenarios on
this split — the result confirms `NO_GO` for `RANKING_FAIRNESS` and
`KV_MEMORY_PRESSURE` only, not a three-regime validation. **No new
scientific experiment is currently running. Next action, NOT started, NOT
authorized:** decide between a Family-B-specific live evaluation or a
higher-level reassessment of the hierarchical-routing hypothesis itself.

## P0 - Policy Separation (WS-P) — historical, superseded as the active P0 by the above

**Family B v2 is composition-ready; the two-parent PrefillControl falsification is now COMPLETE (see below).**
Family B (the next mechanism family after ESTF/WFS) v2 audit:
[`../audits/policy_separation_prefill_decode_pilot_v2_20260817.md`](../audits/policy_separation_prefill_decode_pilot_v2_20260817.md)  
**Verdict: `FAMILY_B_COMPOSITION_READY`.** Parents: `full_prefill` vs
`chunked_prefill_small` (16/15 practical wins at ε=0.01; near-tie 3.1%).

Frozen v1 remains `USEFUL_BUT_NEEDS_REFINEMENT` /
`PREFILL_COMPOSITION_NOT_YET_JUSTIFIED`
([`../audits/policy_separation_prefill_decode_pilot_v1_20260817.md`](../audits/policy_separation_prefill_decode_pilot_v1_20260817.md)).
Do not rewrite that CSV or revive decode-priority / large-chunk / adaptive twins.

**PrefillControl composition falsification is now COMPLETE** (`full_prefill` vs
`chunked_prefill_small`):
[`../audits/family_b_v2_prefill_control_composition_falsification_20260817.md`](../audits/family_b_v2_prefill_control_composition_falsification_20260817.md)
**Verdict: `SELECTION_SUFFICIENT_FOR_THIS_PAIR`.** A real TRAIN/VAL-fitted
contextual top-1 selector reaches the two-parent oracle envelope exactly (0
regret) on TEST and OOD; the genuinely per-step-dynamic `prefill_control_child`
(verified not to collapse to a fixed baseline) never beats it and never
expands the envelope. Selection is sufficient for this pair, as with ESTF↔WFS.
Next on this thread: select the next mechanism family / parent pair per the
roadmap — do not re-run this pair unless with a materially different per-step
rule (the tested rule only ever used 3 of its 6 configured chunk options).

**Family C v1 KV-pressure reserve pairwise-separation pilot is COMPLETE and
frozen** (`kv_constrained_online` vs `least_laxity_first`):
[`../audits/family_c_kv_pressure_pairwise_separation_v1_20260817.md`](../audits/family_c_kv_pressure_pairwise_separation_v1_20260817.md)
(design: [`../design/POLICY_SEPARATION_FAMILY_KV_PRESSURE_V1.md`](../design/POLICY_SEPARATION_FAMILY_KV_PRESSURE_V1.md))
**Verdict: `KV_FAMILY_USEFUL_NEEDS_REFINEMENT`.** 5/6 gates pass — bidirectional
wins (9-vs-4/32), mechanism genuinely activates (28,695 logged reserve
deferrals), no policy twin, and (unlike ESTF/WFS and PrefillControl) real
**within-scenario timing evidence**: the reserve's advantage over greedy
admission is 2× larger when urgent latecomers arrive after KV pressure has
built up vs before. Only the tie-rate gate (59.4%, target <50%) did not
clear. Superseded scientifically by v2 below; frozen, not rewritten.

**Family C v2 KV-pressure reserve refinement is now COMPLETE:**
[`../audits/family_c_kv_pressure_pairwise_separation_v2_20260817.md`](../audits/family_c_kv_pressure_pairwise_separation_v2_20260817.md)
(design: [`../design/POLICY_SEPARATION_FAMILY_KV_PRESSURE_V2.md`](../design/POLICY_SEPARATION_FAMILY_KV_PRESSURE_V2.md))
**Verdict: `KV_FAMILY_COMPOSITION_READY`.** v1's tie-rate gap was diagnosed to
coarse ANWG resolution at v1's population size plus an accidental
bulk-tenant-classified-as-urgent confound; both were fixed without touching
parent algorithms or reserve semantics (population roughly doubled,
`BULK_SLACK_S` recalibrated, a third `urgent_arrival_phase` level added,
seeds 4→6 with 2 held out). All 10 preregistered gates pass, including
held-out-seed replication (G6) and within-scenario winner-flip evidence
(G10, 6/16 matched cells show a different practical winner depending only on
urgent-arrival timing) — the first of the three families studied to reach a
`_READY` verdict. This is still a **pairwise-separation pilot only** — no
selector was fit, no child policy was built, no composition work was
started, per explicit task scope.

**KV-aware composition falsification v1 is now COMPLETE:**
[`../audits/kv_composition_falsification_v1_20260817.md`](../audits/kv_composition_falsification_v1_20260817.md)
(design: [`../design/KV_COMPOSITION_FALSIFICATION_V1.md`](../design/KV_COMPOSITION_FALSIFICATION_V1.md))
**Verdict: `KV_COMPOSITION_INCONCLUSIVE`.** A minimal state-dependent child
(`KVAdaptiveReserveChildPolicy`: delegates every step, unmodified, to one of
the two frozen parents, chosen from a single online-observable trigger —
count of currently-waiting urgent-classified requests) showed real signal:
positive TEST envelope gain, 5/12 TEST scenarios beat both parents by >ε,
24/36 held-out scenarios show genuine non-degenerate within-trajectory
mode-switching, directionally-consistent OOD replication. But the frozen
safety gate (G7) failed: child peak KV utilization exceeded
`max(parent peaks)` on 6/36 (16.7%) held-out scenarios — a
composition-specific risk (mode-switching history creates KV states neither
pure parent alone reaches) no pairwise-separation pilot can surface. Per
the frozen decision rule, G7 failing forces `INCONCLUSIVE` regardless of
G1-G6. **Do not** escalate to a more complex child, MAP-Elites, selector
retraining, symbolic distillation, or LLM synthesis from this result — the
audit's §Z smallest next step (not started) is a narrowly-rescoped child
adding a transition-aware admission cap, re-run through the identical
frozen procedure. **Separately (not gated):** this task surfaced an
unresolved reproducibility gap in the whole KV v1/v2 evidentiary chain — the
current environment cannot reproduce the historical frozen KV v2 CSV
bit-for-bit even via the original unmodified runner (99/144 mismatch,
verified not caused by this task's new code). Root cause not identified;
flagged for a dedicated follow-up, not resolved here.

ESTF↔WFS composition falsification is COMPLETE:
[`../audits/estf_wfs_composition_falsification_v1_20260816.md`](../audits/estf_wfs_composition_falsification_v1_20260816.md)  
**Verdict: `SELECTION_SUFFICIENT_FOR_THIS_PAIR`.** Contextual rank composition
did not beat contextual top-1 or expand the parent envelope. More complex
composition, symbolic distillation, MAP-Elites, and LLM synthesis are **not**
justified from that pair alone.

Family A v2 remains validated complementary-parent evidence
([`../audits/policy_separation_fairness_starvation_pilot_v2_20260816.md`](../audits/policy_separation_fairness_starvation_pilot_v2_20260816.md)).

Stop conditions for this thread:

- Do not start GP / MAP-Elites / CMA-ES / QD from ESTF/WFS, PrefillControl, KV-pressure, or from Family B v1 twins.
- Do not start symbolic distillation or Fireworks/Cloudrift LLM APIs yet.
- Do not escalate composition model complexity to rescue ESTF/WFS or PrefillControl `SELECTION_SUFFICIENT_FOR_THIS_PAIR`.
- Do not treat Family B v2 as a completed composition result on its own — the PrefillControl falsification (above) is the completed composition result for that family's anchor pair.
- KV-pressure composition falsification is COMPLETE (`KV_COMPOSITION_INCONCLUSIVE`, blocked by a safety-gate failure, not absence of signal) — do not escalate to a more complex child, or MAP-Elites/selector retraining/symbolic distillation/LLM synthesis from any of the three pairs studied, without explicit authorization.

## P0 - Apt-Serve / module envelope (independent)

**post-Phase-G module-envelope interpretation.**

Inputs:

- Phase G collection:
  `results/apt_serve_phase_g_resume_20260807_174028/`
- Preserved failed SS15 run:
  `results/apt_serve_phase_g_overnight_20260807_011542/`
- Canonical Phase G analysis:
  `results/apt_serve_phase_g_analysis_20260809_190000/`
- Audit:
  [`../audits/apt_serve_phase_g_analysis_20260809.md`](../audits/apt_serve_phase_g_analysis_20260809.md)

Deliverable:

- Decide which Apt-Serve cache/tier-transition mechanisms are candidates for
  typed module decomposition.
- Decide whether those mechanisms should enter a library-envelope evaluation
  tool as module candidates.
- Define the next WS-H/WS-K experiment without launching another broad
  Apt-Serve sweep.

Existing typed DSL / module-composition work remains available as infrastructure;
it does not substitute for Family A v2 on the WS-P thread.

## P1 - Library-Envelope Tooling

Build a standing evaluator for:

- existing-policy marginal contribution `MC_i(x; P)`;
- candidate marginal gain `MG_c(x; P)`;
- grouped bootstrap CIs by regime/context family;
- clear win/tie/loss classification at practical epsilon thresholds.

This should generalize the Phase G analysis pattern beyond Apt-Serve.

## P2 - Module Decomposition

Select one or two mechanisms from the completed external-baseline work and map
them into the typed DSL/module vocabulary. Apt-Serve's tier-transition behavior
is now a candidate input, but it is not the only candidate; Sarathi, VTC,
Llumnix, and DistServe mechanisms should stay visible.

## P3 - CC6 Decision

Only after P0-P2, decide whether CC6 dynamic adaptation should start. If it
starts, keep it restricted to the previously trusted CC5 operating envelope and
retain safe fallback outside that envelope.

## Stop Conditions

- Do not claim Phase G proves global Apt-Serve superiority.
- Do not rerun Phase G collection unless a concrete missing/invalid cell is
  discovered.
- Do not delete Phase G artifacts.
- Do not convert Apt-Serve into the whole project narrative.
- Do not start broad symbolic synthesis until module contribution evidence is
  stronger.
- Do not treat Job 1182306 as BurstGPT-anchored or as using canonical ANWG.
- Do not treat Family A v1 as corpus-ready policy-separation evidence for QD.
- Step 2 is now COMPLETE and dense — `UNIFIED_UTILITY_MATRIX_READY`, see
  `../audits/family_c_reconstruction_v1_and_unified_matrix_completion_20260817.md`.
  Do not re-launch `scripts/build_unified_utility_matrix_v1.py` or
  `scripts/build_family_c_reconstruction_v1.py`; both are resume-safe but
  there is nothing left to compute.
- Do not train a contextual selector, tune selector hyperparameters, do
  pairwise-regret learning, do mechanism attribution, or start any
  composition/synthesis experiment from `experiments/unified_utility_matrix_v2/`
  without a reviewed Step-3 design.
- Do not modify `experiments/mf_psd_v1/`, `experiments/unified_utility_matrix_v1/`,
  `experiments/family_c_reconstruction_v1/`, or `experiments/unified_utility_matrix_v2/`
  in place — any future rebuild should produce a clearly versioned
  successor so each layer remains available as a frozen baseline.
- Do not mix `experiments/mf_psd_v1/`'s frozen historical Family-C native
  rows into `experiments/unified_utility_matrix_v2/` — that matrix's
  Family-C cells come entirely and uniformly from
  `CURRENT_RECONSTRUCTED_FAMILY_C_V1`, by design.
- Historical KV v2 (`KV_FAMILY_COMPOSITION_READY`) remains valid evidence
  for what it was originally used for; do not reinterpret it using the
  reconstruction's different numbers, and do not attempt to further
  resolve the BurstGPT reproducibility gap as a side effect of an
  unrelated task — it is documented, bounded, and no longer blocks Step 2.
- Step 3 design must explicitly account for Family B's near-total
  cross-family policy collapse (see the completion audit §I) — a
  leave-Family-B-out fold is expected to show little signal beyond the
  `chunked_prefill_small` contrast.
