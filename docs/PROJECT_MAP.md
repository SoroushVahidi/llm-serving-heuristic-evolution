# Project Map

**Canonical long-term research roadmap.** For a short operational handoff, read
[`docs/current/RESUME_HERE.md`](current/RESUME_HERE.md). For baseline-specific
status, read [`docs/BASELINE_STATUS.md`](BASELINE_STATUS.md). Dated files in
[`docs/audits/`](audits/) are historical evidence, not live status.

Last reconciled: 2026-08-16, after commit
`b1181c6380029254080397c161d5dd281bbd6d89`.

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
| WS-P | Policy Separation Dataset / decision-boundary characterization | Sobol Pilot v1 COMPLETE+ANALYZED (Job 1182183). Family A v1 diagnostic only (Job 1182306). Family A v2 EXECUTED+ANALYZED (Job 1182377; `USEFUL_BUT_NEEDS_REFINEMENT`). ESTF↔WFS composition falsification COMPLETE (`SELECTION_SUFFICIENT_FOR_THIS_PAIR`; audit `docs/audits/estf_wfs_composition_falsification_v1_20260816.md`) — selection matches/beats simple composition; no envelope expansion. Next mechanism family: Family B v1 prefill/decode chunk-control EXECUTED 720/720 (design `docs/design/POLICY_SEPARATION_FAMILY_PREFILL_DECODE_V1.md`; analysis pending). MAP-Elites / distillation / LLM synth not justified from ESTF/WFS. Feeds WS-A and WS-F. |

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
ANWG. Next WS-P step: **analyze Family B v1** (the next mechanism family:
prefill/decode chunk-control; design
`docs/design/POLICY_SEPARATION_FAMILY_PREFILL_DECODE_V1.md`) after the full
pilot — not MAP-Elites, selector retraining, or new composition experiments. Typed
DSL/module composition elsewhere in the repo does not substitute for this.


## Stop Conditions

- Do not claim Apt-Serve globally beats the best fixed scheduler from Phase G.
- Do not treat Apt-Serve as the project endpoint.
- Do not start CC6 or broad symbolic synthesis without an explicit decision.
- Do not delete historical negative results; they are part of the research
  record.
- Do not rerun major experiments before checking the relevant audit and current
  status docs.
