# Project Map

**Canonical long-term research roadmap.** For a short operational handoff, read
[`docs/current/RESUME_HERE.md`](current/RESUME_HERE.md). For baseline-specific
status, read [`docs/BASELINE_STATUS.md`](BASELINE_STATUS.md). Dated files in
[`docs/audits/`](audits/) are historical evidence, not live status.

Last reconciled: 2026-08-17, after commit
`6be526ebffe4c3eba6428eab27f9adae1835d320`.

## Documentation Authority

1. `README.md` - public overview and navigation.
2. `docs/PROJECT_MAP.md` - this canonical research roadmap.
3. `docs/current/RESUME_HERE.md` - shortest current operational handoff.
4. `docs/current/WORK_STATUS.md` - detailed current status table.
5. `docs/current/NEXT_ACTIONS.md` - prioritized next actions.
6. `docs/BASELINE_STATUS.md` - external-baseline status index.
7. `docs/audits/*` - immutable point-in-time audit trail.

## North Star

The project aims to build a **verified contextual compositional
hyper-heuristic for online LLM inference serving**.

It is not the same thing as:

- choosing the single best fixed scheduler;
- training only a contextual selector over whole policies;
- reproducing any one external paper for its own sake;
- making Apt-Serve the final scheduler.

Those are inputs to the larger system. The durable objective is to learn when
schedulers and modules are useful, compose new deployable policies, verify them,
and measure whether the policy-library envelope expands.

## Architecture

```text
workload/state context
  -> observable feature extraction
  -> policy/module performance modeling
  -> uncertainty, pairwise advantage, marginal contribution
  -> typed DSL / AST
  -> parent and module selection
  -> structural composition / symbolic synthesis
  -> static verification and leakage checks
  -> simulator evaluation
  -> policy-library envelope expansion
  -> iteration
  -> real-system validation
```

Primary metric: `arrival_normalized_weighted_goodput` (ANWG), weighted SLO
goodput normalized by all arriving requests. Completion-conditioned quality is a
secondary diagnostic only.

## Key Objects

Library envelope at context `x`:

```text
E_P(x) = max_{h in P} R_h(x)
```

Existing policy marginal contribution:

```text
MC_i(x; P) = E_P(x) - E_{P \\ {i}}(x)
```

New candidate marginal gain:

```text
MG_c(x; P) = max(R_c(x), E_P(x)) - E_P(x)
```

The next mature research step should use these objects directly rather than
only average policy rankings.

## Workstreams

| ID | Workstream | Current Position |
|---|---|---|
| WS-A | Simulator, GPU/KV model, metrics | Implemented and heavily tested; discriminative-power limitations remain an active scientific concern. |
| WS-B | Workload generation and trace ingestion | Broad synthetic and real-trace support exists; workload families vary strongly in usefulness. |
| WS-C | Internal scheduler/policy library | Mature fixed-policy portfolio, including Policy Library V2 and SCORPIO-style/admission policies. |
| WS-D | Faithful external scheduler integrations | vLLM-family, Sarathi, VTC, Llumnix, DistServe, PARS, and Apt-Serve have point-in-time status in `docs/BASELINE_STATUS.md`. |
| WS-E | Typed heuristic DSL / AST / verification | Implemented through CC3; representative internal policies reconstruct through the DSL. |
| WS-F | Contextual performance / utility learning | Multiple selector lineages exist; CC5 is the strongest current context-conditioned result. |
| WS-G | Pairwise regret and complementarity | Not yet a mature reusable workstream. |
| WS-H | Module decomposition and compositional semantics | Open; earlier broad module-credit work produced negative/weak-signal findings. |
| WS-I | Parent selection / composition gate | CC5 completed as `COMPLETE_REGIME_SPECIFIC`. |
| WS-J | Structural crossover / symbolic synthesis | Infrastructure exists, but broad synthesis is not authorized without stronger module-credit evidence. |
| WS-K | Quality-diversity archive / library-envelope expansion | Needs a standing reusable `MC_i` / `MG_c` evaluation tool. |
| WS-L | Symbolic distillation / deployable children | Partial, mostly selector-oriented; not yet generic for composed policies. |
| WS-M | Uncertainty / abstention / safe fallback | CC5 has a regime-specific fallback gate; generalization remains future work. |
| WS-N | Real-system transfer and validation | Several pilots and Wulver validations exist; not yet a unified transfer package. |
| WS-O | Publication-grade evaluation | Bootstrap/paired-CI patterns exist; they need consolidation into a final evaluation protocol. |
| WS-P | Policy Separation Dataset / decision-boundary characterization | Sobol Pilot v1 COMPLETE+ANALYZED (Job 1182183). Family A v1 diagnostic only (Job 1182306). Family A v2 EXECUTED+ANALYZED (Job 1182377; `USEFUL_BUT_NEEDS_REFINEMENT`). ESTF↔WFS composition falsification COMPLETE (`SELECTION_SUFFICIENT_FOR_THIS_PAIR`; audit `docs/audits/estf_wfs_composition_falsification_v1_20260816.md`) — selection matches/beats simple composition; no envelope expansion. Family B v1 prefill/decode chunk-control ANALYZED (`USEFUL_BUT_NEEDS_REFINEMENT`; `PREFILL_COMPOSITION_NOT_YET_JUSTIFIED`; audit `docs/audits/policy_separation_prefill_decode_pilot_v1_20260817.md`). Family B v2 anchor pair `full_prefill`/`chunked_prefill_small` PrefillControl composition falsification COMPLETE (`SELECTION_SUFFICIENT_FOR_THIS_PAIR`; audit `docs/audits/family_b_v2_prefill_control_composition_falsification_20260817.md`) — fitted top-1 selector matches oracle exactly; genuinely per-step-dynamic child does not expand the envelope. Family C v1 KV-pressure reserve (`kv_constrained_online` vs `least_laxity_first`) pairwise-separation pilot: `KV_FAMILY_USEFUL_NEEDS_REFINEMENT` (audit `docs/audits/family_c_kv_pressure_pairwise_separation_v1_20260817.md`; frozen, superseded by v2) — 5/6 gates pass, tie-rate gate did not clear. **Family C v2** (refined population/calibration/phase-levels/seeds) has since run to completion: **`KV_FAMILY_COMPOSITION_READY`**, all 10 preregistered gates pass, including held-out-seed replication and within-scenario winner-flip evidence (audit `docs/audits/family_c_kv_pressure_pairwise_separation_v2_20260817.md`) — the first of three families studied to reach a `_READY` verdict. Its composition falsification has since run to completion: `KV_COMPOSITION_INCONCLUSIVE` (audit `docs/audits/kv_composition_falsification_v1_20260817.md`) — real envelope-gain signal (positive TEST gain, 5/12 beat-both, non-degenerate within-trajectory mode-switching), blocked by a composition-specific KV-safety gate failure (child peak KV exceeds both parents' peaks on 6/36 held-out scenarios), not by absence of signal. MAP-Elites / distillation / LLM synth not justified yet from any studied pair. Feeds WS-A and WS-F. |

## Current Checkpoint

Contextual composition:

- CC0-CC5 are complete.
- CC5 is `COMPLETE_REGIME_SPECIFIC`: it supports contextual composition inside
  a restricted operating envelope, not universal superiority.
- CC6 is not started and remains gated by an explicit decision to return to
  dynamic adaptation.

External baselines:

- Sarathi, VTC, Llumnix, DistServe, PARS-Serve-2026, vLLM-LTR, and Apt-Serve
  each have evaluated or bounded status in `docs/BASELINE_STATUS.md`.
- Apt-Serve Phase G collection and analysis are complete. The latest audit is
  [`docs/audits/apt_serve_phase_g_analysis_20260809.md`](audits/apt_serve_phase_g_analysis_20260809.md).

Apt-Serve interpretation:

- Supported: Phase G data are structurally valid; Apt-Serve contributes a
  positive leave-one-out marginal portfolio contribution with grouped bootstrap
  CI excluding zero.
- Not established: global Apt-Serve superiority over the best fixed baseline.
  The global Apt-vs-best-fixed grouped bootstrap CI crosses zero.
- Project meaning: Apt-Serve is one external policy family and one source of
  cache/tier-transition mechanisms for future module decomposition.

## Canonical Next Action

**Return from Apt-Serve-specific collection to broader library-envelope and
module-decomposition work.**

Concretely, the next task should review the completed Phase G analysis as input
to WS-H/WS-K:

- identify which Apt-Serve mechanisms are module candidates;
- decide whether those mechanisms expand the library envelope in specific
  contexts;
- design the next module-decomposition/compositional-learning step without
  launching another broad Apt-Serve sweep.

**Parallel thread (WS-P, independent of the above):** Sobol Pilot v1 (Job 1182183)
is complete and analyzed. Family A v1 Job 1182306 remains diagnostic-only
(`USEFUL_DIAGNOSTIC_ONLY` / `REDESIGN_REQUIRED`). Family A v2 Job 1182377 is
executed **and scientifically analyzed**
(`docs/audits/policy_separation_fairness_starvation_pilot_v2_20260816.md`):
verdict `USEFUL_BUT_NEEDS_REFINEMENT`. ESTF↔WFS bidirectional separation is
confirmed under orthogonal size×priority with BurstGPT anchoring and canonical
ANWG. Family B v1 refinement (the next mechanism family after ESTF/WFS) was
analyzed: `docs/audits/policy_separation_prefill_decode_pilot_v1_20260817.md`;
verdict `USEFUL_BUT_NEEDS_REFINEMENT`; `PREFILL_COMPOSITION_NOT_YET_JUSTIFIED`
for v1. The refined **v2** anchor pair (`full_prefill`/`chunked_prefill_small`)
has since had its PrefillControl composition falsification run to completion:
`docs/audits/family_b_v2_prefill_control_composition_falsification_20260817.md`,
verdict `SELECTION_SUFFICIENT_FOR_THIS_PAIR` — a real fitted selector matches
the two-parent oracle exactly; the genuinely per-step-dynamic child does not
expand the envelope. A third mechanism family, Family C KV-pressure reserve
(`kv_constrained_online` vs `least_laxity_first`, selected from a repository
capability audit — the only already-implemented, single-mechanism,
zero-new-simulator-work candidate), had its v1 pairwise-separation pilot run
to completion:
`docs/audits/family_c_kv_pressure_pairwise_separation_v1_20260817.md`
(design: `docs/design/POLICY_SEPARATION_FAMILY_KV_PRESSURE_V1.md`), verdict
`KV_FAMILY_USEFUL_NEEDS_REFINEMENT` — 5/6 gates pass, including the first
within-scenario-timing evidence (the reserve's advantage over greedy
admission is 2× larger when urgent latecomers arrive after KV pressure has
built up vs before) of any family studied so far; only the tie-rate gate did
not clear (v1 is now frozen, superseded scientifically by v2, not rewritten).
**Family C v2** diagnosed v1's tie-rate gap (coarse ANWG resolution at v1's
population size, plus an accidental bulk-tenant-classified-as-urgent
confound) and fixed both without touching parent algorithms or reserve
semantics — population roughly doubled, `BULK_SLACK_S` recalibrated, a third
`urgent_arrival_phase` level added, seeds 4→6 (2 held out). It ran to
completion:
`docs/audits/family_c_kv_pressure_pairwise_separation_v2_20260817.md`
(design: `docs/design/POLICY_SEPARATION_FAMILY_KV_PRESSURE_V2.md`), verdict
**`KV_FAMILY_COMPOSITION_READY`** — all 10 preregistered gates pass,
including two new ones beyond v1: held-out-seed replication (the
timing/pressure pattern replicates on 2 seeds never used in any calibration
decision) and within-scenario winner-flip evidence (6/16 matched scenario
cells show a different practical winner purely as a function of when urgent
tenants arrive). This is the first of the three families studied to reach a
`_READY` verdict. **Its composition falsification has since run to
completion:** `docs/audits/kv_composition_falsification_v1_20260817.md`
(design: `docs/design/KV_COMPOSITION_FALSIFICATION_V1.md`), verdict
**`KV_COMPOSITION_INCONCLUSIVE`**. A minimal state-dependent child
(delegates every step, unmodified, to one of the two frozen parents, chosen
from an online-observable urgent-queue-depth trigger) showed real signal —
positive TEST envelope gain, 5/12 TEST scenarios beating both parents by
>ε, genuine non-degenerate within-trajectory mode-switching on 24/36
held-out scenarios, directionally-consistent OOD replication — but the
frozen safety gate failed: child peak KV utilization exceeded
`max(parent peaks)` on 6/36 held-out scenarios, a composition-specific risk
(mode-switching history creates KV states neither pure parent alone
reaches) no pairwise-separation pilot can surface. Per the frozen decision
rule this forces `INCONCLUSIVE` regardless of the otherwise-favorable
results. **Do not** escalate to a more complex child, MAP-Elites, selector
retraining, symbolic distillation, or LLM synthesis from this result — the
only defensible next step (not started) is a narrowly-rescoped child adding
a transition-aware admission cap, re-run through the identical frozen
procedure. Separately, this falsification surfaced an unresolved
reproducibility gap in the whole KV v1/v2 evidentiary chain (the current
environment cannot reproduce the historical frozen KV v2 CSV bit-for-bit
even via the original unmodified runner) — root cause not identified,
flagged for a dedicated follow-up. Typed DSL/module composition elsewhere in
the repo does not substitute for any of this.


## Stop Conditions

- Do not claim Apt-Serve globally beats the best fixed scheduler from Phase G.
- Do not treat Apt-Serve as the project endpoint.
- Do not start CC6 or broad symbolic synthesis without an explicit decision.
- Do not delete historical negative results; they are part of the research
  record.
- Do not rerun major experiments before checking the relevant audit and current
  status docs.
