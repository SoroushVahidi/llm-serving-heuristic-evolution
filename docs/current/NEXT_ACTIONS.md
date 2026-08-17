# Next Actions

This is the current prioritized action list. It must agree with
[`../PROJECT_MAP.md`](../PROJECT_MAP.md), [`RESUME_HERE.md`](RESUME_HERE.md),
[`WORK_STATUS.md`](WORK_STATUS.md), and [`../BASELINE_STATUS.md`](../BASELINE_STATUS.md).

## P0 - Structural Reassessment / MF-PSD (current)

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

**Step 2 (unified six-policy utility-matrix evaluation) is COMPLETE for
Family A and Family B, BLOCKED for Family C:**
[`../audits/unified_policy_utility_matrix_v1_20260817.md`](../audits/unified_policy_utility_matrix_v1_20260817.md)
(design: [`../design/UNIFIED_UTILITY_MATRIX_STEP2_V1.md`](../design/UNIFIED_UTILITY_MATRIX_STEP2_V1.md)).
**Verdict: `UNIFIED_UTILITY_MATRIX_NEEDS_REFINEMENT`.** 416 new cells
evaluated (0 failures) at `experiments/unified_utility_matrix_v1/`: Family A
(72 scenarios) and Family B (32 scenarios) are now **fully dense** (6/6
canonical anchors); Family C (72 scenarios) stays at native 2/6 — its 288
cross-family cells are explicit `unsupported_scenario_reconstruction`
placeholders, not silently missing, because Family C / KV v2 scenario
regeneration is confirmed **not** byte-exact (99/144 mismatch against its
own frozen native cells, independently reproducing
[`kv_v2_reproducibility_forensic_20260817.md`](../audits/kv_v2_reproducibility_forensic_20260817.md)).
Two important findings, both explicitly tagged in the data (not hidden):
(1) `full_prefill`/`chunked_prefill_small` collapse to one identical
behavior outside Family B (confirmed: byte-identical ANWG on every Family-A
cell); (2) on **all 32** Family-B scenarios, `estf`/`wfs`/`least_laxity`/
`kv_constrained` are byte-identical to each other and to `full_prefill` —
Family B's cross-family diversity reduces to one contrast
(`chunked_prefill_small` vs. everything else), because none of those four
ranking policies touch the chunk-budget axis Family B was built to
isolate. **Next action, with explicit authorization: investigate the
Family-C/KV-v2 BurstGPT reconstruction gap** (a dedicated task, separate
from selector work) before the matrix can reach `READY`. Do not start
selector training, hyperparameter tuning, pairwise-regret learning,
mechanism attribution, or any composition/synthesis experiment before Step
2 reaches a `READY`/`READY_LOW_DIVERSITY` verdict (per the reassessment
doc's own explicit deferred-items list, §P, and this task's own stop
condition).

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
- Step 2 is now partially complete (Family A/B dense, Family C blocked) —
  see `../audits/unified_policy_utility_matrix_v1_20260817.md`. Do not
  re-launch the full 416-cell evaluation; `scripts/build_unified_utility_matrix_v1.py`
  is resume-safe and will only compute newly-unblocked cells.
- Do not train a contextual selector, tune selector hyperparameters, do
  pairwise-regret learning, do mechanism attribution, or start any
  composition/synthesis experiment from the unified utility matrix before
  Step 2 reaches a `READY`/`READY_LOW_DIVERSITY` verdict.
- Do not modify `experiments/mf_psd_v1/` or `experiments/unified_utility_matrix_v1/`
  in place — any Step 2 v2 rebuild should produce a clearly versioned
  successor so v1 remains available as a frozen baseline.
- Do not attempt to fix the Family-C/KV-v2 BurstGPT reconstruction gap as a
  side effect of an unrelated task; it needs its own dedicated
  investigation before Family C's 288 cross-family cells can be added.
