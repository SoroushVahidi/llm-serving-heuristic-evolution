# Hierarchical Regime Router v1 — Live Closed-Loop Re-Evaluation: Finalized Audit

Status: **FINAL**. Closes the formal gate-scoring gap identified by a
repository-wide audit (2026-08-19). Does not rerun any simulation, retrain
any model, or change any threshold/feature/split/dwell/fallback semantics.

## A. Original Approximate Verdict

The first (approximate) TEST evaluation of the hierarchical regime router
(design/impl commits `078f4f1`/`2923087`, evaluation script
`scripts/run_hierarchical_regime_router_v1_test_evaluation.py`, result
`experiments/hierarchical_regime_router_v1_test_evaluation/test_evaluation_results.json`)
returned **`HIERARCHICAL_ROUTER_NO_GO`** via the canonical
`evaluate_all_gates`/`compute_verdict` gate evaluator. G4 (Stage-2
preservation) failed outright (value 0.0 on Regime `RANKING_FAIRNESS`, the
only regime with a computable fraction; threshold 0.90), which alone
forces `NO_GO` under the frozen decision rule. Root cause, as diagnosed in
[`hierarchical_regime_router_v1_20260818.md`](hierarchical_regime_router_v1_20260818.md):
the evaluation dispatched to ONE majority-vote regime label per scenario
(the "approximate" contract), washing out minority-of-steps regime
activity — a measurement-methodology artifact, not a competence failure of
either stage. Family B (`PREFILL_DECODE_CONTENTION`) had 0 TEST scenarios
on the frozen split (its 8 groups all hashed into TRAIN/VAL). Named next
step: "build a genuine per-step live-simulation evaluation harness ... do
not silently re-run under the same preregistration."

## B. Live Harness Readiness

`723a39c` ("feat: add live closed-loop hierarchical router harness") built
`src/llmserveopt/policy_separation/hierarchical_router_live_harness_v1.py`.
Readiness audit:
[`hierarchical_router_live_harness_validation_v1_20260818.md`](hierarchical_router_live_harness_validation_v1_20260818.md),
verdict `LIVE_HIERARCHICAL_HARNESS_READY`. Independently re-confirmed by
this task's own code reading: `LiveHierarchicalRouterPolicy.select_action`
is invoked fresh every step inside the **unmodified** `Simulator.run()`
loop → Stage-1 predicts on the current-step's `ObservableState` → the
incremental dwell/fallback FSM (proven bit-identical to the frozen batch
FSM) → Stage-2 selector or fixed fallback policy → the chosen native
policy's real `select_action`, which the simulator applies, causally
producing the next state. No majority vote exists in this module (enforced
by AST regression tests). 6/6 forced-parent equivalence tests pass
bit-exact against standalone single-policy runs. A causal-switch microcase
demonstrates the live trajectory diverging from a fallback-only trajectory
only after a genuine Stage-2 switch.

## C. Live Result Integrity

`experiments/hierarchical_regime_router_live_reeval_v1/live_reeval_results.json`
(2,762 bytes, SHA-256 `6ff89399cc63b76309f234abb3df8620f11c9bcd98a5f56d23e1b85250043952`):
parseable JSON; `split_counts` = train 118 / val 26 / test 32, matching the
frozen split (8 Family A + 0 Family B + 24 Family C, per
`launch_manifest.json`'s `expected_family_*_count` fields, independently
hand-reproducible from `build_splits(mf_psd_scenarios_v1.csv)` — see the
repository audit's Section K); `preregistration_integrity.git_head_sha` =
`ed742769a108547c6b08d06590c902dca1698ebf` (this task's own HEAD before any
edit) with `git_tree_dirty: false`; `config`/`telemetry`/`scenario`
checksums match the current frozen files exactly (independently
recomputed, see repository audit Sections F/P). `run.log` at repo root
matches the result file byte-for-byte; `crash.log` is empty; no
`hierarchical-live-reeval` tmux session or python process is running. The
result file itself does **not** carry a per-scenario or per-regime
breakdown — only TEST-aggregate scalars — which is the source of the G4/
G7/G9(a) gap in Section G below.

## D. Execution / Provenance History

| Step | Commit | Outcome |
|---|---|---|
| Preregistration | `6c9ec36` | Design doc + gates frozen |
| Setup | `9fde981` | Launch manifest + fitted-model-hash scaffold written |
| First launch attempt | (uncommitted) | **Crashed before any analysis ran** — no `run.log`, no live process, no per-scenario output ever existed for this attempt |
| Fix | `ed74276` | Corrected `group_resampled_bootstrap_ci` call signature; added AST regression test (`tests/test_run_hierarchical_regime_router_live_reeval_v1.py`) pinning the interface |
| Corrected run | (post-`ed74276`, untracked result until this task) | Completed 2026-08-18T22:27:49Z; result self-attests `git_head_sha = ed74276...`, `git_tree_dirty = false` |

## E. Bootstrap-Call Bug / Fix History

`group_resampled_bootstrap_ci(df, hierarchy, fixed, *, n_boot, ci)`
(`hierarchical_router_evaluation_v1.py:281`) requires 3 positional
arguments and an `n_boot` keyword. The pre-fix call passed only 2
positional arguments (`delta_method, test_df["group_key"]`) plus an
unsupported `n_resamples` keyword — a `TypeError` at call time, before any
scenario was ever run. `ed74276` corrected both calls to
`group_resampled_bootstrap_ci(test_df, live_col, approx_col/best_fixed,
n_boot=5000, ci=0.90)`, unchanged `n_boot`/`ci` values, and preserved
grouping via `test_df`'s `group_key`. Repository-audit-independent check:
no other caller in the repository has this bug pattern (only 2 production
callers + 1 unit test exist for this function, all now correct).

## F. Main Live Metrics (from the persisted result artifact only)

| Metric | Value |
|---|---|
| Mean ANWG — approximate (old contract) | 0.807479668003565 |
| Mean ANWG — live (new contract) | 0.8136377562388593 |
| Mean ANWG — best global fixed (`weighted_fair_share`) | 0.807479668003565 |
| Mean ANWG — six-policy oracle | 0.8506364193404634 |
| `delta_method` (live − approximate) | 0.006158088235294129 |
| `delta_method` 90% CI | [0.0005514705882353164, 0.011397058823529423] |
| `delta_fixed` (live − best global fixed) | 0.006158088235294129 |
| `delta_fixed` 90% CI | [0.0005514705882353164, 0.011397058823529423] |
| Regret to six-policy oracle | 0.03699866310160427 |
| Oracle gap closure | 0.1426911907066834 |
| Stage-1 accuracy | 0.9921507064364207 |
| Stage-1 macro-F1 (present classes only) | 0.9886585377383629 |
| Catastrophic misroute rate | 0.0 |
| Routing fractions (A / B / C / fallback) | 0.3541 / 0.0000 / 0.1190 / 0.5268 |
| Switching rate per 1000 steps | 2.856 |
| Total transitions | 908 |
| Dwell violations | 0 |

`delta_method` and `delta_fixed` are numerically identical (15 significant
digits) because `baseline_a_anwg` (best global fixed) literally returns the
`weighted_fair_share` column, and the old approximate baseline also falls
back to that same column whenever its majority-vote regime label isn't in
`{A,B,C}` — consistent with zero of the 32 TEST scenarios having an
electable majority regime under the old approximation (this is the
measurement artifact the live harness exists to fix; not independently
re-executed to confirm, but consistent with all code paths inspected).
`fraction_fallback = 0.527` is expected, not a defect: `NONE`/`OVERLAP`
regimes always dispatch to the fixed fallback policy by design.

## G. Formal G1–G9 Table

Computed via `scripts/rescore_hierarchical_regime_router_live_reeval_v1_gates.py`,
which calls ONLY the canonical `evaluate_all_gates`/`compute_verdict`
implementation in `hierarchical_router_gates_v1.py` — the same evaluator
already used correctly by the approximate TEST evaluation — never a
hand-written substitute. Full output:
`experiments/hierarchical_regime_router_live_reeval_v1/gate_rescoring_v1.json`.

| Gate | Metric | Threshold | Measured | Result |
|---|---|---|---|---|
| G1 | Online input validity | == 1.0 | 1.0 | **PASS** |
| G2 | Router quality (macro-F1) | ≥ 0.90 | 0.9887 | **PASS** |
| G3 | Catastrophic misrouting | ≤ 0.05 | 0.0 | **PASS** |
| G4 | Stage-2 preservation (min per-regime fraction) | ≥ 0.90 | — | **NOT_EVALUABLE** |
| G5 | Beat global fixed (mean AND CI-lower>0) | > 0.01 | 0.00616 (CI lower 0.00055) | **FAIL** (mean criterion fails; CI criterion alone would pass) |
| G6 | Oracle gap closure | ≥ 0.75 | 0.1427 | **FAIL** (non-critical) |
| G7 | Multi-regime benefit | ≥ 2 of 3 | — | **NOT_EVALUABLE** (non-critical) |
| G8 | Interpretable/non-leaking errors | (a) == 0 leakage; (b) qualitative | (a) 0 leakage | (a) **PASS**; (b) not reviewed → overall not-strictly-pass, but does **not** independently force NO_GO (only G8(a) can) |
| G9 | Robustness | (a) ≥ 0 on Family-C held-out; (b) ≤ 0.10 blended | — / 0.0097 (reused, see below) | **NOT_EVALUABLE** (non-critical) |

**G4/G7/G9(a) are `NOT_EVALUABLE`, not manufactured**, because
`live_reeval_results.json` persists only the TEST-aggregate
`mean_anwg_live` scalar, not a per-scenario or per-regime breakdown. Each
of these three gates requires per-regime live ANWG (G4: standalone-vs-
integrated regret by regime; G7: per-regime live-vs-fixed delta; G9(a):
the same restricted to `KV_MEMORY_PRESSURE`, which is Family C's entire
held-out-eval-seed TEST allocation). Recovering this would require
re-executing the live harness, which is explicitly out of scope for this
rescoring task (no simulation reruns). This is a distinct, newly-surfaced
artifact-completeness gap in `run_hierarchical_regime_router_live_reeval_v1.py`,
separate from the already-known Family-B-has-0-TEST-scenarios limitation
(Regime B would have been `NOT_EVALUABLE` for G4 either way; Regimes A and
C are *additionally* `NOT_EVALUABLE` here specifically because of this
persistence gap).

G9(b)'s blended-microcase catastrophic rate (0.0097, sample not too small)
was **reused, not recomputed**, from the sibling approximate-TEST-evaluation
artifact (`experiments/hierarchical_regime_router_v1_test_evaluation/test_evaluation_results.json`).
This is valid because blended microcases exercise only the frozen Stage-1
router against synthetic FIFO-simulated probes, independent of the
live-vs-approximate Stage-2 evaluation contract, and
`hierarchical_regime_router_v1.py`/`configs/hierarchical_regime_router_v1_gates.json`
are verified byte-identical between the two evaluation runs (`git diff
--stat 2923087 ed74276 -- ...` is empty).

## H. Historical Script Verdict

`LIVE_REEVAL_CONFIRMS_NO_GO`, printed by
`scripts/run_hierarchical_regime_router_live_reeval_v1.py`'s own hand-rolled
4-branch if/else (lines ~350–360 as of `ed74276`), which the script's
comments explicitly flag as provisional: *"G1-G9 will be rescored manually
or with the gate function. For now, let's assign verdict based on primary
numbers, assuming gates pass if dm > 0.01."* This ad-hoc logic uses a
different verdict vocabulary (`LIVE_REEVAL_{CONFIRMS_NO_GO,
IMPROVES_METHOD_BUT_NO_END_TO_END_GAIN, SUPPORTS_HIERARCHY, INCONCLUSIVE}`)
than the canonical framework's (`HIERARCHICAL_ROUTER_{NO_GO,
ROUTING_WORKS_SELECTION_NO_GAIN, INCONCLUSIVE, GO}`) — the two are not the
same function and were never claimed to be identical namespaces.

## I. Formal Canonical Verdict

**`FORMAL_GATE_VERDICT = HIERARCHICAL_ROUTER_NO_GO`.**

Derivation via `compute_verdict`: G1 passes and G8(a) passes, so the first
NO_GO branch does not fire. G2 and G3 both pass, so the second does not
fire. G4 is `NOT_EVALUABLE` (`passed = None`, not `False`), so the direct
G4-failure branch does not fire either. G5 fails on its mean criterion
(0.00616 not > 0.01); because G4's `passed` is `None` rather than
literal `True`, the softer `ROUTING_WORKS_SELECTION_NO_GAIN` branch (which
requires G2, G3, **and G4** to all be `True`) does not apply, so
`compute_verdict` falls through to its final `NO_GO` branch.

## J. Do H and I Agree?

**Yes — both land in the NO_GO family.** The ad-hoc script's
`LIVE_REEVAL_CONFIRMS_NO_GO` and the formal evaluator's
`HIERARCHICAL_ROUTER_NO_GO` agree directionally: the hierarchy does not
beat the best global fixed policy by a practically significant margin on
this TEST split. The formal scoring closes the methodology gap the
repository audit identified — the verdict was directionally correct all
along, but is now backed by the actual frozen 9-gate framework rather than
an ad-hoc numeric threshold check, and the gap in per-regime data (G4/G7/
G9a) that the ad-hoc script's simplicity had been silently sidestepping is
now explicit and documented rather than invisible.

## K. Family-B Zero-TEST Limitation

`PREFILL_DECODE_CONTENTION` (Family B) again received exactly 0 TEST
scenarios on this run (same deterministic `sha256(group_key) mod 100`
split as the approximate evaluation — all 8 Family-B groups hash into
TRAIN/VAL, a preregistered/expected outcome, not a runtime surprise; see
`launch_manifest.json`'s `expected_family_b_count: 0`). **This live
re-evaluation validates two of the three regimes
(`RANKING_FAIRNESS`, `KV_MEMORY_PRESSURE`), not all three.** Any citation
of this result must not claim three-regime validation.

## L. What Is Established

- The live closed-loop harness is a genuine, causally-correct, non-
  majority-vote implementation (code- and test-verified).
- The primary live re-evaluation has been executed to completion under the
  exact preregistered 32-scenario split, with self-consistent, checksum-
  verified provenance.
- Hierarchical routing, evaluated live (not approximately), still does not
  clear the preregistered practical-significance bar (G5) or the oracle-
  gap-closure bar (G6) — the CI excludes zero (some real, small, positive
  effect exists) but the effect is too small to constitute a `GO`.
- The formal, gate-framework-conformant verdict is now on record and
  agrees with the run script's own provisional read.
- Stage-1 routing quality remains excellent (macro-F1 0.989, 0
  catastrophic misroutes) independent of the end-to-end verdict.

## M. What Is Not Established

- Whether Family B (`PREFILL_DECODE_CONTENTION`) routing would change the
  overall picture — it has never been TEST-evaluated anywhere in this
  lineage (approximate or live).
- G4 (Stage-2 preservation), G7 (multi-regime benefit), and G9(a)
  (Family-C held-out robustness) for the live run specifically — the
  persisted result artifact lacks the per-regime breakdown needed to score
  them, and this task did not re-execute the harness to recover it.
- G8(b) (qualitative "every misrouting cluster attributable to input
  ambiguity, not forbidden fields") — still requires human review, as in
  the original TEST evaluation; not auto-computable.
- Whether re-executing the live harness with per-regime output persistence
  added would change G4/G7/G9(a)'s values (they are silent, not assumed
  favorable).

## N. Remaining Provenance Caveats

- `experiments/hierarchical_regime_router_live_reeval_v1/launch_manifest.json`'s
  `git_sha` field (`6c9ec364dcc7b1373570258169e6b72648fb7cd1`) does not
  resolve to a real git object — it diverges from the actual preregistration
  commit (`6c9ec36387744e24a3a84a62d312ebbee4fafc55`) after the shared
  7-character prefix, consistent with a hand-padded rather than
  `git rev-parse`-derived SHA. This is a defect in the setup-time manifest
  writer, not in the result's own self-recorded provenance (which
  correctly carries the real HEAD `ed74276...` — see Section C). **Not
  corrected in place**, to preserve the historical record; flagged here as
  a standing caveat, matching the "preserve, don't silently rewrite"
  policy already established by the KV v2 reproducibility forensic audit.
- `fitted_model_hashes.json` in the same directory records `stage1_model_hash`
  and 2 of 3 `stage2_model_hashes` (`RANKING_FAIRNESS`,
  `PREFILL_DECODE_CONTENTION`) but omits `KV_MEMORY_PRESSURE`, even though
  that regime's Stage-2 result is `EVALUATED` in the primary result. Also
  **not corrected in place**, for the same reason.
- The live-reeval result artifact was untracked in git prior to this task
  and has now been committed as part of this reconciliation (Section T of
  the companion final report).

## O. Exact Next Scientific Action

Not begun by this task, per its own stop condition. Two candidate
directions are named (not started, not authorized): (1) a scoped,
preregistered Family-B-specific live evaluation — the one regime never
TEST-evaluated anywhere in this lineage — or (2) a higher-level
reassessment of the hierarchical-routing hypothesis itself, analogous in
kind to the `dc5757b` composition reassessment that preceded MF-PSD. Choosing
between them requires explicit authorization, not a default continuation.
