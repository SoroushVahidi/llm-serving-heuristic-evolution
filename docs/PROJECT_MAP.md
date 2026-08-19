# Project Map

**Canonical long-term research roadmap.** For a short operational handoff, read
[`docs/current/RESUME_HERE.md`](current/RESUME_HERE.md). For baseline-specific
status, read [`docs/BASELINE_STATUS.md`](BASELINE_STATUS.md). Dated files in
[`docs/audits/`](audits/) are historical evidence, not live status.

Last reconciled: 2026-08-19, after Family-B live replication prep (commit
`9d8f997`), Public Trace Corpus v1 implementation (commits `84fa31b` + `179a6fe`),
and decision-criticality analysis design (uncommitted parallel workstream).

## Documentation Authority

1. `README.md` - public overview and navigation.
2. `docs/PROJECT_MAP.md` - this canonical research roadmap.
3. `docs/current/RESUME_HERE.md` - shortest current operational handoff.
4. `docs/current/WORK_STATUS.md` - detailed current status table.
5. `docs/current/NEXT_ACTIONS.md` - prioritized next actions.
6. `docs/BASELINE_STATUS.md` - external-baseline status index.
7. `docs/audits/*` - immutable point-in-time audit trail.

## North Star

**As of 2026-08-19, this supersedes the pre-reassessment "zero-regret contextual selection"
framing this doc used to lead with** (see `docs/audits/reassessment_composition_hypothesis_20260817.md`
and the three selector/schema/mechanism NO_GOs below for why): the project's ultimate
objective is **not** merely policy selection, composition, or contextual routing between a
fixed policy library.

It is not the same thing as:

- choosing the single best fixed scheduler;
- reproducing any one external paper for its own sake;
- making Apt-Serve the final scheduler;
- training a universal per-scenario selector or router as an end in itself.

Those are inputs to, or evidence-gathering stages for, the larger system, not its endpoint.
A multi-family heuristic library and rigorous per-family performance-boundary separation are
durable prerequisites; contextual selection/routing (flat/pooled selector, hierarchical
router) has been tried and demoted at every tier tested so far (see the Current Checkpoint
below) — the finding is that selection/routing alone is not the mechanism that will get the
project to its objective, not that selection/routing was a wasted step.

**Desired end state:**

```text
NEW WORKLOAD / INPUT / SERVING STATE
  -> characterize context
  -> identify where existing policies disagree/fail
  -> identify decision-critical mechanisms
  -> generate / evolve / construct a NEW scheduling policy tailored to that context
  -> verify it against strong existing policies
  -> validate selected results using real LLM serving
```

### Timelines

**Short-term:** Build a strong policy-separating dataset (COMPLETE — MF-PSD v1 + unified utility matrix + three NO_GOs).

**Mid-term:** Mechanism attribution + decision-critical state modeling (NOT STARTED — gated on public-trace corpus completion and policy replay).

**Long-term:** New-policy synthesis/evolution (NOT STARTED — depends on Layers 2-5 of public-trace corpus + decision-criticality analysis).

**Final validation:** Real LLM serving experiments (NOT STARTED — Cohere/CloudRift reserved for this stage only).

### Dataset Layers

Seven layers, each with its own scope, ownership, and gating dependency. This
supersedes the earlier four-bucket (A-D) framing this doc briefly used —
folded into one numbering here so the layer used in status tables
(`WORK_STATUS.md`, `EXPERIMENT_INDEX.md`) and the layer used here always
agree:

- **Layer 0 — public trace provenance:** raw third-party dataset sources
  (BurstGPT MIT, Azure 2023 conv/code CC-BY-4.0, AgentPerfBench classified
  `REAL_SYSTEM_VALIDATION_SOURCE`), license/provenance tracked, not
  redistributed beyond what each license permits. **COMPLETE**
  (`data/raw/**`, commits `84fa31b`/`179a6fe`).
- **Layer 1 — normalized real public workload inputs:** the canonical
  schema-conformant corpus built from Layer 0. **COMPLETE**
  (`data/public_trace_corpus_v1/`, `docs/design/PUBLIC_TRACE_CORPUS_V1.md`).
- **Layer 2 — canonical replay scenarios:** Layer-1 traces turned into
  concrete simulator scenarios (policy-agnostic). **NOT STARTED.**
- **Layer 3 — same-scenario multi-policy outcomes:** each Layer-2 scenario
  run under the full policy library; scenario-level utility/regret recorded.
  **NOT STARTED for public-trace scenarios** — but this is exactly what
  MF-PSD v1 + Unified Utility Matrix v2 already did for the synthetic Family
  A/B/C scenario set (**COMPLETE** for that set; the public-trace corpus has
  not yet gone through this step).
- **Layer 4 — step-level actions/trajectories:** per-step action records
  during Layer-3 replay, not just scenario-level aggregate utility.
  **NOT STARTED.**
- **Layer 5 — counterfactual decision-criticality/mechanism annotations:**
  decision-critical-state identification, counterfactual action divergence,
  mechanism attribution, built from Layer 4. **NOT STARTED** — this is the
  layer the decision-criticality/timescale analysis workstream (currently
  prepared-only, uncommitted, owned separately) is designed to eventually
  produce evidence toward.
- **Layer 6 — small real-LLM validation subset:** a small subset of
  synthesized-policy results validated against real hosted LLM serving
  (Cohere/CloudRift). **NOT STARTED. Only stage where Cohere/CloudRift
  belong** — never for coding, literature search, dataset schema design, or
  code review.

### New-Policy Synthesis Path

```
PUBLIC TRACES
    ↓
Layer 0: PROVENANCE-TRACKED RAW TRACES (COMPLETE)
    ↓
Layer 1: CANONICAL WORKLOAD CORPUS (COMPLETE per commits 84fa31b/179a6fe)
    ↓
Layer 2: CANONICAL REPLAY SCENARIOS (NOT STARTED)
    ↓
Layer 3: SAME-SCENARIO MULTI-POLICY OUTCOMES (COMPLETE for MF-PSD's synthetic
         scenario set via MF-PSD v1 + Unified Utility Matrix v2; NOT STARTED
         for public-trace scenarios)
    ↓
Layer 4: STEP-LEVEL ACTIONS/TRAJECTORIES (NOT STARTED)
    ↓
Layer 5: DECISION-CRITICALITY / MECHANISM ATTRIBUTION (NOT STARTED)
    ↓
NEW POLICY SYNTHESIS / EVOLUTION (NOT STARTED — LONG-TERM GOAL)
    ↓
SIMULATOR VALIDATION
    ↓
Layer 6: REAL-LLM VALIDATION (Cohere/CloudRift only here)
```

Selector/router experiments are **evidence-gathering stages**, not the final goal. The flat/pooled selector NO_GO, shared-feature NO_GO, mechanism-choice target NO_GO, and cross-family demotion all converge on the same conclusion: a universal per-scenario selector is unlikely to work globally, but per-family/statedependent policy construction has real potential when grounded in decision-critical mechanism evidence rather than aggregate features.

## Architecture

```text
policy-separating workloads
  -> complementary policy library
  -> workload/state context
  -> observable feature extraction
  -> policy/module performance modeling
  -> contextual selection (multi-family)
  -> uncertainty, pairwise regret, marginal contribution
  -> mechanism attribution
  -> bounded envelope
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
| WS-F | Contextual performance / utility learning | MF-PSD v1 (`MF_PSD_READY`) and the dense Unified Utility Matrix v2 (`UNIFIED_UTILITY_MATRIX_READY`) are both complete. The flat/pooled multi-family selector trained on them returned `MULTIFAMILY_SELECTOR_NO_GO`, followed by two further NO_GOs (shared-feature schema, mechanism-choice target) and a formal demotion (`CROSS_FAMILY_TRANSFER_DEMOTED_HIERARCHICAL_ROUTING_READY`). A hierarchical regime router was then built and evaluated at both TEST and live per-step granularity; both returned `HIERARCHICAL_ROUTER_NO_GO`. See `docs/current/WORK_STATUS.md` for the full chain and current next actions. |
| WS-G | Pairwise regret and complementarity | Ready for evaluation alongside direct utility prediction. |
| WS-H | Module decomposition and compositional semantics | **COMPOSITION_DEMOTED**. Composition/synthesis is exploratory future work, deferred. |
| WS-I | Parent selection / composition gate | Composition falsification found selection sufficient (not beaten by dynamic composition) for A, B, and C. Contextual selection/routing was subsequently tried at flat/pooled and hierarchical granularity and demoted at every tier tested (see WS-F) — see the North Star section above: selection/routing is evidence-gathering, not the project's final objective. |
| WS-J | Structural crossover / symbolic synthesis | Explicitly deferred. Not supported by current heuristic evidence. |
| WS-K | Quality-diversity archive / library-envelope expansion | Paused. Focus shifts to mapping multi-family boundaries. |
| WS-L | Symbolic distillation / deployable children | Partial, mostly selector-oriented; not yet generic for composed policies. |
| WS-M | Uncertainty / abstention / safe fallback | CC5 has a regime-specific fallback gate; generalization remains future work. |
| WS-N | Real-system transfer and validation | Several pilots and Wulver validations exist; not yet a unified transfer package. |
| WS-O | Publication-grade evaluation | Bootstrap/paired-CI patterns exist; they need consolidation into a final evaluation protocol. |
| WS-P | Policy Separation Dataset / decision-boundary characterization | Sobol Pilot v1 COMPLETE+ANALYZED (Job 1182183). Family A v1 diagnostic only (Job 1182306). Family A v2 EXECUTED+ANALYZED (Job 1182377; `USEFUL_BUT_NEEDS_REFINEMENT`). ESTF↔WFS composition falsification COMPLETE (`SELECTION_SUFFICIENT_FOR_THIS_PAIR`; audit `docs/audits/estf_wfs_composition_falsification_v1_20260816.md`) — selection matches/beats simple composition; no envelope expansion. Family B v1 prefill/decode chunk-control ANALYZED (`USEFUL_BUT_NEEDS_REFINEMENT`; `PREFILL_COMPOSITION_NOT_YET_JUSTIFIED`; audit `docs/audits/policy_separation_prefill_decode_pilot_v1_20260817.md`). Family B v2 anchor pair `full_prefill`/`chunked_prefill_small` PrefillControl composition falsification COMPLETE (`SELECTION_SUFFICIENT_FOR_THIS_PAIR`; audit `docs/audits/family_b_v2_prefill_control_composition_falsification_20260817.md`) — fitted top-1 selector matches oracle exactly; genuinely per-step-dynamic child does not expand the envelope. Family C v1 KV-pressure reserve (`kv_constrained_online` vs `least_laxity_first`) pairwise-separation pilot: `KV_FAMILY_USEFUL_NEEDS_REFINEMENT` (audit `docs/audits/family_c_kv_pressure_pairwise_separation_v1_20260817.md`; frozen, superseded by v2) — 5/6 gates pass, tie-rate gate did not clear. **Family C v2** (refined population/calibration/phase-levels/seeds) has since run to completion: **`KV_FAMILY_COMPOSITION_READY`**, all 10 preregistered gates pass, including held-out-seed replication and within-scenario winner-flip evidence (audit `docs/audits/family_c_kv_pressure_pairwise_separation_v2_20260817.md`) — the first of three families studied to reach a `_READY` verdict. Its composition falsification has since run to completion: `KV_COMPOSITION_INCONCLUSIVE` (audit `docs/audits/kv_composition_falsification_v1_20260817.md`) — real envelope-gain signal (positive TEST gain, 5/12 beat-both, non-degenerate within-trajectory mode-switching), blocked by a composition-specific KV-safety gate failure (child peak KV exceeds both parents' peaks on 6/36 held-out scenarios), not by absence of signal. MAP-Elites / distillation / LLM synth not justified yet from any studied pair. **These three families' evidence fed the higher-level structural reassessment** (`docs/audits/reassessment_composition_hypothesis_20260817.md`, `COMPOSITION_DEMOTED`) and were then unified into **MF-PSD v1** (`docs/audits/multi_family_policy_separation_dataset_v1_20260817.md`, `MF_PSD_READY`, `experiments/mf_psd_v1/`) — 496-row long-form utility table + 176-scenario context table, sparse six-anchor coverage (not yet a dense matrix; Step 2, not started). Feeds WS-A and WS-F. |

## Current Checkpoint

**This section covers only the CC0-CC5/Apt-Serve Phase G checkpoint (through
~2026-08-09) — it predates the entire MF-PSD → selector-NO_GOs →
hierarchical-router → Family-B-replication lineage summarized above and in
[`docs/current/WORK_STATUS.md`](current/WORK_STATUS.md). Kept as historical
narrative for the CC/Apt-Serve thread specifically, not as the full current
state.**

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

**UPDATE (2026-08-19): Steps 2 and 3 below are now COMPLETE, and the
revised roadmap they describe has itself been superseded by a further
hierarchical-routing attempt and its own NO_GO. Read
[`docs/current/WORK_STATUS.md`](current/WORK_STATUS.md) for the authoritative
current state and next action; the paragraph below is kept as historical
narrative for how the project reached Step 1, not as the live next action.**
In one line: MF-PSD v1 (Step 1) → Unified Utility Matrix v2 (Step 2,
`UNIFIED_UTILITY_MATRIX_READY`) → flat/pooled selector (Step 3,
`MULTIFAMILY_SELECTOR_NO_GO`) → shared-feature-schema and mechanism-target
redesigns (both NO_GO) → cross-family transfer reassessment
(`CROSS_FAMILY_TRANSFER_DEMOTED_HIERARCHICAL_ROUTING_READY`) → hierarchical
regime router, TEST and live re-evaluation (both `HIERARCHICAL_ROUTER_NO_GO`)
→ Family-B balanced replication (implementation-ready, scientific run not
yet authorized) and Public Trace Corpus v1 (workload-input layer complete)
as the two currently queued independent threads — see
[`docs/current/NEXT_ACTIONS.md`](current/NEXT_ACTIONS.md) for the exact list.

**Primary thread (historical narrative): the higher-level structural reassessment
(`docs/audits/reassessment_composition_hypothesis_20260817.md`,
`COMPOSITION_DEMOTED`) set a revised roadmap — policy-separating workloads
-> complementary policy library -> contextual selection (multi-family) ->
mechanism attribution -> bounded envelope. Its Step 1, MF-PSD v1
(`docs/audits/multi_family_policy_separation_dataset_v1_20260817.md`,
`MF_PSD_READY`, `experiments/mf_psd_v1/`), is complete: a 496-row canonical
long-form utility table + 176-scenario context table unifying Family A v2 +
Family B v2 + Family C/KV v2, with an explicit learnable-feature
allowlist/denylist and full provenance. The six-anchor policy matrix is
sparse, not dense. The next action, with explicit authorization, is Step 2
— unified six-policy utility-matrix evaluation (see the MF-PSD audit's §M/§Q
for the exact ~704 new policy-scenario evaluations required). Do not start
selector training, hyperparameter tuning, pairwise-regret learning,
mechanism attribution, or composition/synthesis before Step 2 is complete.**

**Secondary/independent thread:** Return from Apt-Serve-specific collection to broader library-envelope and module-decomposition work
(WS-H is now
`COMPOSITION_DEMOTED` at the project level, so this thread's scope is
narrower than before — module decomposition remains a legitimate exploratory
input, not the central hypothesis).

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
