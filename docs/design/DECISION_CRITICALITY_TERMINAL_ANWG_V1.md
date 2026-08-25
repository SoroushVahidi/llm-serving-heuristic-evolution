# Decision-Criticality Terminal-ANWG v1 — Design / Preregistration

Date: 2026-08-24  
Status: **FROZEN BEFORE SCORING** (runtime implementation notes appended 2026-08-25, pre-full-run)  
Parent: `docs/design/DECISION_CRITICALITY_TIMESCALE_TRAINVAL_V1.md`  
Feasibility: `docs/current/decision_criticality_h_critical_feasibility_and_evidence_20260824.md`

## 0. Scope

TRAIN/VAL-only diagnostic. No TEST. No Family-B replication. No selector/router retraining.
No manuscript edits. Does not overwrite
`experiments/decision_criticality_timescale_trainval_v1/`.

Reuses `fork_from_live_simulator` / `LiveFork` / frozen live hierarchical router.

## 1. Primary estimand

For reference trajectory and decision time \(t\):

\[
C_t = \mathrm{ANWG}(\text{CF branch}) - \mathrm{ANWG}(\text{reference terminal})
\]

**Reference terminal:** untouched live-router trajectory to normal completion → terminal ANWG.

**CF branch** (from identical pre-decision simulator state at \(t\)):

1. Apply exactly one alternative native-pair action \(a_{\mathrm{alt}}\).
2. Return control to a **clone** of the live hierarchical router whose FSM/dwell state matches
   the reference router **after** its step-\(t\) `select_action` (so subsequent routing uses the
   same policy class, adapting to the counterfactual state).
3. Continue to natural termination via `Simulator.continue_run` (identical idle/drain/handoff
   semantics to `Simulator.run`).
4. Identical future arrivals (fork copies the not-yet-enqueued suffix).
5. Terminal ANWG via `compute_metrics` with full scenario request list as denominator.

**Not used as primary:** H=10 completed-count, alt-policy continuation, scenario-level policy swaps.

## 2. Diagnostic hypotheses (frozen before outcomes)

- **H1 Sparse terminal criticality:** A minority of evaluated states accounts for a disproportionate
  share of \(\sum |\Delta\mathrm{ANWG}|\).
- **H2 Disagreement insufficient:** Native disagreement has imperfect precision for
  \(|\Delta\mathrm{ANWG}| > 0\) (and for practical thresholds 0.001 / 0.005 / 0.01).
- **H3 Proxy mismatch:** H=10 completed-count criticality from v1 does not perfectly identify
  terminal-ANWG critical states.
- **H4 Closed-loop persistence:** Some nonzero-\(C_t\) interventions induce multi-step state
  divergence before any reconvergence.

This study is **diagnostic**; it does not declare project GO/NO-GO.

## 3. Data / acquisition (frozen before outcomes)

- Corpus: same 144 TRAIN/VAL MF-PSD scenarios as v1 (A64 / B32 / C48).
- **DISAGREEMENT:** all active-regime steps where canonicalized \(a_{\mathrm{ref}} \neq a_{\mathrm{alt}}\)
  for the regime's native pair; keep the **first**
  `MAX_DISAGREEMENT_PER_SCENARIO = 5` per scenario (trajectory order; outcome-blind).
- **AGREEMENT_CONTROL:** among active-regime steps with agreeing actions, take the
  **first** `MAX_AGREEMENT_CONTROL_PER_SCENARIO = 3` in trajectory order (outcome-blind;
  same structural rule as disagreement caps). Online evaluation cannot rewind, so a
  hash-priority subsample without a second pass is not used. `CONTROL_SEED = 20260824`
  is retained for provenance / future offline subsample extensions.
- If a scenario has fewer disagreements/agreements than the caps, take all.
- **Reference-action replay control:** once per scenario (at the first evaluated intervention,
  if any): fork, apply \(a_{\mathrm{ref}}\), continue with router clone → must match reference
  terminal ANWG within `ANWG_EQ_ATOL = 1e-12`.
- **Action canonicalization:** admit-set only (parent v1 semantics). Prefill-chunk differences
  alone do **not** count as DISAGREEMENT; they may still appear under AGREEMENT_CONTROL when
  the alt native policy applies a different `prefill_chunk_override`.

## 4. Causal validity checklist

1. Bit-identical pre-intervention fingerprint vs live sim before fork mutations.
2. Live sim fingerprint unchanged after CF/REF-replay forks.
3. Future arrivals from deep-copied pending suffix.
4. Only step-\(t\) forced action differs initially between CF and REF-replay.
5. Continuation policy = cloned live hierarchical router (same Stage-1/2 weights).
6. Alt action from native pair only; no actual-output oracle for ranking.
7. Invalid actions: simulator `_apply_action` semantics unchanged.
8. Mandatory REF-replay ≡ reference terminal ANWG.

## 5. Analysis outputs

Prevalence; concentration curves on \(|\Delta\mathrm{ANWG}|\) and on \(\max(\Delta\mathrm{ANWG},0)\);
scenario-aggregated mass; disagreement vs agreement controls; join to v1 H10 completion proxy;
closed-loop divergence proxies (completion / duration / utility); temporal run lengths of
ANWG-critical states; scenario-grouped bootstrap (n=1000, seed `BOOTSTRAP_SEED = 20260825`)
for mean \(|\Delta\mathrm{ANWG}|\) and top-5% mass share.

## 6. Runtime / implementation notes (pre-full-run, outcome-blind)

- Terminal continuation uses `Simulator.continue_run` after `fork_from_live_simulator`'s forced
  first action (not `LiveFork` step-to-end), matching reference idle/drain/handoff semantics.
- Stage-2 feature rows are scenario-level constants; `LiveHierarchicalRouterPolicy` caches the
  Stage-2 policy id per regime (bit-identical to re-predicting every step; ~20× speedup on
  long Family-A active trajectories).
- Smoke (A high-skew / B / C) verified REF-replay match and fork isolation before full TRAIN/VAL.
- Full run: named tmux `decision_criticality_terminal_anwg_v1`. Do not overwrite v1 artifacts.
