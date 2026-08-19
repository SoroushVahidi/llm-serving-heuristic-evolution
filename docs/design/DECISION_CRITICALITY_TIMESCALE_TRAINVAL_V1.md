# Decision-Criticality & Regime-Timescale Diagnostic v1 — TRAIN/VAL-Only Design and Preregistration

Date: 2026-08-19

## 0. Scope

**DESIGN / PREREGISTRATION ONLY, TRAIN/VAL METHODOLOGY DIAGNOSTIC.**
No Stage-1 router is retrained, no Stage-2 selector is retuned, no threshold or dwell value is
changed, no new policy is added, no TEST-split row (scenario or telemetry) is read, and no new
project-level scientific verdict is computed or implied by this document or by the analysis it
authorizes. This is a diagnostic/methodology study over TRAIN and VAL data only, intended to
characterize three previously unresolved hypotheses (H-CRITICAL, H-TIMESCALE, H-CEILING) named in
the task that authorized this document. It does not supersede, reopen, or contradict the frozen
`HIERARCHICAL_ROUTER_NO_GO` verdict (`docs/audits/hierarchical_regime_router_v1_20260818.md`) or
the frozen `LIVE_REEVAL_CONFIRMS_NO_GO` verdict
(`experiments/hierarchical_regime_router_live_reeval_v1/live_reeval_results.json`,
`gate_rescoring_v1.json`). The preregistered Family-B held-out live replication
(`docs/design/FAMILY_B_BALANCED_REPLICATION_V1.md`) is **NOT** run, read, or referenced by this
analysis — this document and its analysis are strictly TRAIN/VAL.

---

## 1. Precisely Defining the Diagnostic Questions

This diagnostic answers three questions left open by the frozen `LIVE_REEVAL_CONFIRMS_NO_GO`
result, using TRAIN/VAL data only (never TEST, never the Family-B replication set):

- **H-CRITICAL**: What fraction of scheduling steps are actually "decision-critical" — i.e. the
  two native policies in the currently active regime's pair propose different actions, and that
  difference has a measurable downstream causal consequence?
- **H-TIMESCALE**: How does the empirical duration of A/B/C regime episodes compare with the
  frozen `dwell=20` reaction floor? Are most performance-relevant episodes shorter than the
  router's own minimum reaction time?
- **H-CEILING**: Even where the router routes correctly, how much opportunity does the underlying
  native-pair policy library actually offer? If the two candidates in a pair almost always behave
  identically, no routing scheme (however accurate) can produce a large end-to-end gain.

This is explicitly **NOT**:
- A new scientific TEST evaluation.
- A retraining, rethresholding, or dwell-tuning exercise.
- A composition/synthesis proposal.
- A launch of the Family-B held-out replication.

---

## 2. Frozen Elements (Must Remain Identical to the Live Re-eval System)

Every element of the hierarchical router system under diagnostic observation is frozen to exactly
match `HIERARCHICAL_REGIME_ROUTER_V1` / `HIERARCHICAL_REGIME_ROUTER_LIVE_REEVAL_V1`:

- **Stage-1 Router**: `Stage1Router` (multiclass logistic regression), fit exclusively on TRAIN
  telemetry (`online_regime_telemetry_v1.csv`, rows whose `split == "train"`), unmodified from
  `hierarchical_regime_router_v1.Stage1Router`.
- **Stage-1 Inputs** (the four frozen "regime signals" the task names): `contention_score_v2`,
  `priority_skew`, `kv_pressure`, `queue_length` (`STAGE1_INPUT_COLUMNS`).
- **Stage-2 Selectors**: `fit_all_stage2_selectors`, fit exclusively on TRAIN scenario-level rows,
  unmodified from `hierarchical_stage2_selectors_v1.py`.
- **Frozen native pairs** (Stage-2 candidates, `STAGE2_CANDIDATES`):
  - Family/Regime A (`RANKING_FAIRNESS`): `estimated_service_time_first` vs. `weighted_fair_share`.
  - Family/Regime B (`PREFILL_DECODE_CONTENTION`): `full_prefill` vs. `chunked_prefill_small`.
  - Family/Regime C (`KV_MEMORY_PRESSURE`): `kv_constrained_online` vs. `least_laxity_first`.
- **FSM**: `IncrementalDwellFallbackFSM` / `apply_dwell_and_fallback`, `dwell = DWELL_MINIMUM_STEPS
  = 20` (read-only reference value throughout; never modified, never swept).
- **Live per-step driver**: `LiveHierarchicalRouterPolicy` / `run_live_scenario`
  (`hierarchical_router_live_harness_v1.py`), unmodified, used verbatim to generate every
  reference (actually-executed) trajectory this diagnostic observes.
- **No post-hoc tuning**: no thresholds, models, training parameters, dwell value, or splits are
  altered based on any output of this diagnostic.

---

## 3. Exact Scope — Splits, Families, Data

- **Allowed splits: TRAIN and VAL only.** Every scenario, telemetry row, and trajectory this
  analysis reads or produces must have `split in {"train", "val"}` under the frozen
  `hierarchical_regime_router_v1.build_splits` assignment
  (`experiments/mf_psd_v1/mf_psd_scenarios_v1.csv`, joined the same way
  `hierarchical_router_evaluation_v1.load_scenario_level_dataset` already does).
- **TEST is categorically forbidden**: no TEST-split scenario is ever rebuilt or simulated; no
  TEST-split telemetry row is ever read; the code enforces this with an explicit split guard
  (`assert_trainval_only`) raising on any `"test"` value, tested directly
  (`tests/test_decision_criticality_timescale_trainval_v1.py`).
- **Families**: A (`FAMILY_A_FAIRNESS_STARVATION_V2`), B (`FAMILY_B_PREFILL_DECODE_V2`), C
  (`FAMILY_C_KV_PRESSURE_V2`) — exactly the three MF-PSD mechanism families already wired to
  Regimes A/B/C.
- **TRAIN/VAL scenario counts** (frozen `build_splits`, verified before launch):
  TRAIN = 118 (A=54, B=16, C=48), VAL = 26 (A=10, B=16, C=0) → **144 TRAIN/VAL scenarios total**
  (A=64, B=32, C=48). Family C has no VAL rows by construction (`build_splits`'s seed-based
  partition puts all non-held-out-seed Family-C rows in TRAIN); this is a pre-existing, frozen
  property of the split, not something this diagnostic can or should change.
- **Family-B data dependency**: rebuilding real Family-B TRAIN/VAL scenarios requires the staged
  BurstGPT token-length corpus (`case_prefill_decode_ttft_contention(..., datasets_root=...)`,
  resolved via `resolve_burstgpt_path`). This diagnostic passes `datasets_root=<repo>/.local_data`
  explicitly (confirmed present in this environment: `.local_data/burstgpt_v2/raw/`) — the same
  BurstGPT-backed non-synthetic construction already used for the frozen MF-PSD Family-B dataset
  itself. It never falls back to `allow_synthetic_tokens=True` for TRAIN/VAL scenario replay (that
  flag is used only by this diagnostic's own small, explicitly-synthetic unit-test fixtures, never
  for the real 144-scenario TRAIN/VAL sweep).
- **No model tuning, no threshold tuning, no split changes** anywhere in this diagnostic.

---

## 4. Router-Input / Causal-Evaluation-Output Separation (Section 4 of the authorizing task)

Two strictly separated categories of information:

- **ROUTER INPUTS** (online, current/past-only): the four Stage-1 signal columns, computed from
  the live `ObservableState` at step `t` only (`compute_regime_signals`, unchanged). Nothing this
  diagnostic computes is ever fed back into Stage-1, Stage-2, or the FSM.
- **CAUSAL EVALUATION OUTPUTS** (may inspect controlled future counterfactual branches): action
  disagreement, immediate state divergence, short-horizon causal divergence, and (bounded, see §9)
  full-trajectory branch outcomes. These are diagnostic-only, written to
  `experiments/decision_criticality_timescale_trainval_v1/` and never read back by any model,
  router, or selector code path. This separation is enforced structurally (the diagnostic module
  never imports itself into `hierarchical_regime_router_v1.py` / `hierarchical_stage2_selectors_v1.py`
  / `hierarchical_router_live_harness_v1.py`) and is checked by a dedicated test.

---

## 5. Counterfactual Methodology — Frozen BEFORE Scoring

### A. Reference trajectory
For each TRAIN/VAL scenario, the **reference (actually-executed) trajectory** is the real,
unmodified `run_live_scenario` output: the frozen `LiveHierarchicalRouterPolicy`, wrapping the
frozen Stage-1/Stage-2 models, driving the real `Simulator` end-to-end via `Simulator.run()`,
exactly as `hierarchical_router_live_harness_v1.py` already does for the primary live re-evaluation
and the (not-run-here) Family-B replication. This is the only trajectory whose actions are ever
actually applied to the canonical simulator; every counterfactual branch operates on an isolated
clone and can never mutate it.

### B. ACTION_DISAGREEMENT (item A)
At every step where the FSM-resolved `effective_regime` is one of the three active regimes (A, B,
or C — i.e. a native pair is genuinely in play), the diagnostic additionally computes what the
**other** native-pair candidate would have proposed from an *independent deep copy* of that same
step's `ObservableState` (the router's actually-chosen policy's action is never altered — this is
a pure read-only shadow computation). Two actions are compared **canonically**: by their
`{gpu_id: sorted(admit_ids)}` mapping (`preempt`/`swap`/`migrate`/`hold_decode` are always empty
for all six frozen native policies — verified in §7 below — so `admit` alone is a complete,
non-lossy canonicalization for this policy set). A disagreement is recorded whenever the two
canonical mappings differ.

### C. Counterfactual forking mechanism (items B/C — IMMEDIATE_STATE_DIVERGENCE /
SHORT_HORIZON_CAUSAL_DIVERGENCE)
Forking is performed **only on steps where ACTION_DISAGREEMENT is true** (if the two candidates
propose identical actions, `state_(t+1)` is provably identical by the simulator's determinism, so
no fork is needed or performed for those steps — this is a computational-cost reduction, not an
approximation).

On a disagreement step, the diagnostic builds a **lightweight fork**: a new `Simulator` object
(`Simulator.__new__(Simulator)`, never routed through `__init__`/`Simulator.run()`) whose mutable
per-step state (`_gpus`, `_waiting`/`_waiting_map`, `_migrating`/`_migrating_map`, `_relocating`,
and the *not-yet-enqueued suffix* of `_pending_arrivals`) is `copy.deepcopy`'d from the live
reference simulator at that exact step, and whose read-only/shared state (`config`,
already-consumed `_pending_arrivals` prefix, `_completed`) is either shared by reference (proven
never mutated by `_apply_action`/`_advance_decode`, the only two methods ever called on a fork) or
shallow-copied. The fork then applies the **alternative** native policy's action via the simulator's
own unmodified `_apply_action`/`_advance_decode` methods (zero reimplementation of admission/decode
logic — the exact same canonical code path the real run uses) and continues to be driven by that
same alternative policy, in lockstep with the real run, for `H` further steps (§F). The real
reference trajectory is never paused, altered, or re-derived to produce this fork — it continues
forward on its own, unaffected, in the same `sim.run()` call.

`tests/test_decision_criticality_timescale_trainval_v1.py` asserts, by identity and by before/after
state fingerprinting, that no fork ever mutates the real reference simulator's `_gpus`, `_waiting`,
`_pending_arrivals`, or `_completed` containers.

### D. FULL_TRAJECTORY_OPPORTUNITY (item D) — bounded, preregistered
Running every disagreement-step fork to literal scenario completion is not computationally
feasible at the full TRAIN/VAL scale (144 scenarios, some >30,000 steps; §8 measured a single
Family-A reference run at ~19s/31,396 steps). Per the task's own instruction ("only where
computationally feasible and preregistered"), this diagnostic performs a **bounded** version:
- At most **3 full-trajectory branch attempts per family per scenario** (a scenario contributes at
  most 3 Family-A-relevant, 3 Family-B-relevant, 3 Family-C-relevant full branches — in practice
  only its own family is ever relevant, so at most 3 per scenario), chosen as the **first 3
  disagreement steps** encountered in that scenario's reference trajectory (a fixed, outcome-blind
  rule — not selected by observed divergence magnitude).
- Each such branch continues for at most **`FULL_TRAJECTORY_MAX_EXTRA_STEPS = 3000`** additional
  steps beyond the fork point (or until the branch's own queue+active-set empties with no more
  arrivals, whichever comes first) — a fixed, preregistered cap, not tuned on any observed result.
- This yields a **bounded proxy** for full-trajectory opportunity, not literal completion for very
  long scenarios; every reported full-trajectory-branch metric is explicitly labeled
  `bounded_horizon_steps` in the output so it is never confused with a true whole-scenario oracle
  value.

### E. Immediate + short-horizon comparison points
Both the 1-step (`IMMEDIATE_STATE_DIVERGENCE`) and `H`-step (`SHORT_HORIZON_CAUSAL_DIVERGENCE`)
comparisons are read off the **same** fork lineage: the fork's own state after 1 step, and after
`H` steps, each compared against the real reference trajectory's actual state at the same absolute
step number (which continues to be produced by the unmodified, uninterrupted real run).

### F. Counterfactual horizon `H` — frozen BEFORE scoring
**`H = 10` scheduling steps**, fixed here in advance of running any diagnostic analysis, chosen
only for being (a) large enough to reveal short-horizon causal consequences beyond the immediate
step and (b) small enough (half of `dwell=20`) to keep exhaustive per-disagreement-step forking
computationally tractable at full TRAIN/VAL scale without any per-scenario sampling. **`H` is not
selected based on any observed performance, divergence rate, or other result of this diagnostic.**

### G. Divergence metrics computed at each comparison point (item C)
Transparent, non-learned diagnostics only (no invented "importance score"):
- `queue_length` absolute difference.
- Total active-request-count absolute difference (summed over GPUs).
- Mean KV-utilization absolute difference (mean over GPUs of `current_kv_tokens / max_kv_tokens`).
- Newly-completed-request-count absolute difference within the window (an online-computable
  completion-divergence proxy; a literal ANWG value is **not** computed for a partial window since
  ANWG requires whole-run normalization constants not legitimately available mid-trajectory — this
  is a deliberate, documented omission, not an oversight).
- Count of steps within the window where the fork's and the real run's *next* chosen action would
  also differ (a cheap re-application of the same canonical-action comparison, computed only for
  the fork's own driving policy vs. what the real run's driving policy is doing at the same step —
  diagnostic-only signal, not a new metric family).

---

## 6. Regime-Episode Timescale Analysis (task §5)

Episodes are contiguous runs of a per-step **activity state** derived from the trajectory's own
raw activity-label columns (`a_active`, `b_active_v2`, `c_active`, already recorded by
`LiveHierarchicalRouterPolicy._log_step` — computed online from `compute_activity_labels`, never
from `effective_regime`, so this measures the *raw* regime-episode timescale independent of the
FSM/dwell smoothing):

```
NONE        : not a_active and not b_active_v2 and not c_active
OVERLAP     : more than one of {a_active, b_active_v2, c_active} true
A_active    : only a_active true
B_active_v2 : only b_active_v2 true
C_active    : only c_active true
```

For every TRAIN/VAL scenario's trajectory, contiguous runs of this label are identified
(`(label != label.shift()).cumsum()` grouping). Reported per family (A/B/C) and pooled:
count, min, p10, p25, median, p75, p90, p95, max episode length (in steps); fraction of episodes
with length `<5`, `<10`, `<20`, `=20`, `>20`, `>40`; and fraction of **total active steps**
(not episode count) contained in episodes shorter than `dwell=20` — reported separately from
episode-count fractions specifically because a family with many short episodes but low traffic
share is scientifically different from one where short episodes carry most of the active-step
mass, per the task's explicit instruction.

---

## 7. Dwell-Latency Diagnostic (task §6)

Computed over the **FSM-resolved `effective_regime`** trajectory column (the router's own smoothed
regime state, not the raw activity labels of §6 — this is what actually gates Stage-2 dispatch and
what `dwell=20` actually constrains). For every contiguous active-regime (A/B/C) episode of length
`L` steps, starting at (scenario-relative) step `s`:
- `activation_step = s`
- `earliest_switch_eligible_step = s + dwell` (`dwell = DWELL_MINIMUM_STEPS = 20`, read-only)
- `episode_end_step = s + L - 1`
- `ends_before_switch_eligible = episode_end_step < earliest_switch_eligible_step`
- `useful_active_steps_remaining_after_eligibility = max(0, episode_end_step -
  earliest_switch_eligible_step + 1)` if not ended-before-eligible, else `0`

Classification (fixed thresholds, preregistered, not tuned on results):
- **`UNREACHABLE_UNDER_DWELL20`**: `L < dwell` (episode ends before the router could ever act on a
  reassessment even if the FSM allowed instant switching).
- **`PARTIALLY_REACTABLE`**: `dwell <= L < 2 * dwell` (eligible for at least one step, but with
  less than one further full dwell window of steps remaining to exploit that eligibility).
- **`FULLY_REACTABLE`**: `L >= 2 * dwell` (eligible with at least one full dwell window remaining
  after eligibility).

`dwell` itself is never altered and no sweep over its value is performed anywhere in this
diagnostic (task explicit prohibition, §6/§18).

---

## 8. Action-Disagreement Rate & Causal Importance (task §7/§8)

For every family/native pair, reported over all TRAIN/VAL steps where that regime's Stage-2 is
consulted (`effective_regime` active): total evaluated steps, identical-action steps,
different-action steps, disagreement fraction; and the same fraction conditioned on the *raw*
regime-activity label (§6) being active vs. inactive (using the raw label, since the FSM/dwell
state is itself downstream of the router-under-test and conditioning on it would partially
confound the comparison).

For disagreement steps, the causal-importance diagnostics of §5G are aggregated: one-step
divergence rate (fraction of disagreement forks where any of the four divergence metrics is
nonzero after 1 step), `H`-step divergence rate (same, after `H=10` steps), and per-metric mean
absolute divergence. All transparent, formula-based aggregates — no learned "importance score" is
introduced anywhere.

---

## 9. Minority-But-Critical Episodes (task §9) & Family-B Primary Diagnostic (task §10)

Within TRAIN/VAL only, a raw-activity episode (§6) is flagged **minority-but-critical** if both:
(a) it is a minority-length episode for its family (`episode_length < median(episode_length)` for
that family, computed *before* looking at any divergence outcome — the median itself is a property
of the episode-length distribution, not of divergence, so this is not outcome-selection), and
(b) at least one step inside the episode is an `ACTION_DISAGREEMENT` step (§5B). Counts and
representative `(canonical_scenario_id, episode_start_step, episode_end_step)` triples are reported
per family. These are never selected or filtered using any TEST-split ANWG value (categorically
inaccessible under §3) or any held-out Family-B replication outcome (categorically inaccessible
under §0/§3).

Family B additionally receives the full task-§10 diagnostic battery (episode duration distribution,
fraction of B episodes shorter than `dwell=20`, fraction of B-active disagreement steps, causal
divergence rate after those disagreements, whether B behavior concentrates in short episodes, and
whether majority/plurality routing — i.e. the original scenario-majority-vote approximation this
whole hierarchical-router research line was built to move past — would miss those episodes),
computed entirely from the 32 real (BurstGPT-backed, `datasets_root`-resolved — §3) TRAIN/VAL
Family-B scenarios. **The preregistered held-out Family-B-balanced replication set and its results
are never read, referenced, or used by any part of this computation.**

---

## 10. Policy-Library Ceiling Diagnostic (task §11)

For each regime/native pair, on TRAIN/VAL only: fraction of evaluated active steps with identical
vs. differing actions (§8, restated here as the ceiling framing); fraction of differing-action
steps with measurable downstream consequence (§8's divergence-rate figures); and, from the bounded
full-trajectory branches of §5D, the maximum observed **bounded-horizon** scenario-level
improvement from following the alternative native policy instead of the one the router/reference
chose at that disagreement point, plus the mean such bounded-horizon oracle opportunity across all
attempted branches. Explicitly labeled as a bounded-horizon proxy (§5D), not a full-scenario
oracle. No new policy is added; only the two existing frozen candidates per pair are ever compared.

---

## 11. Provenance Requirements

Every run of this diagnostic records, in
`experiments/decision_criticality_timescale_trainval_v1/`:
- Git HEAD SHA and dirty-tree flag at launch.
- Exact invoking command.
- SHA-256 of this design doc.
- SHA-256 of `configs/hierarchical_regime_router_v1_gates.json` (frozen gate config, read-only
  reference — not evaluated as a verdict by this diagnostic).
- SHA-256 of `experiments/mf_psd_v1/mf_psd_scenarios_v1.csv` and
  `experiments/online_regime_signal_feasibility_v1/online_regime_telemetry_v1.csv`.
- The frozen `H`, `dwell`, and full-trajectory bound constants (§5F, §7, §5D).
- Python executable path and package versions (numpy, pandas, scikit-learn).
- UTC start/end timestamps.
- TRAIN/VAL scenario counts actually processed, per family, with an explicit assertion that the
  count matches §3's frozen expectation (A=64, B=32, C=48) before any result is written.

---

## 12. Pre-Launch Verification Plan

Before the long run is launched, the executing runner asserts:
1. **Split guard**: every scenario id fed to the diagnostic has `split in {"train","val"}` under
   the frozen `build_splits`; any `"test"` id raises immediately.
2. **No replication access**: the module never imports `family_b_balanced_replication_v1` and never
   opens any file under `experiments/family_b_balanced_replication_v1/`.
3. **Fork isolation**: forking a running simulator never mutates the original's `_gpus`, `_waiting`,
   `_pending_arrivals`, or `_completed` (identity + fingerprint checks).
4. **Canonical-action purity**: all six frozen native policies used here have empty
   `preempt`/`swap`/`migrate`/`hold_decode` on every action they return (so `admit`-only
   canonicalization is complete) — checked directly, not assumed.
5. **Deterministic replay**: running the same TRAIN/VAL scenario's diagnostic twice yields
   bit-identical action-disagreement and episode-segmentation output.
6. **Focused test suite passes** (`tests/test_decision_criticality_timescale_trainval_v1.py`)
   before the long run is launched.

---

## 13. Standing Long-Running Job Rule

This diagnostic branches simulator state at potentially many steps across 144 TRAIN/VAL scenarios
and may run for a long time. Per the task authorizing this document:
1. Launched in a dedicated, named tmux session:
   `decision_criticality_timescale_trainval_v1`.
2. Logs and results written under `experiments/decision_criticality_timescale_trainval_v1/` and
   `logs/decision_criticality_timescale_trainval_v1.log`.
3. Monitored for at most ~3 minutes after launch to confirm the process is alive, the log is
   growing, scenario/step progress is advancing, resource usage is healthy, and no immediate
   failure storm has occurred — then monitoring stops and the job is left running. No result
   interpretation, verdict, or composition step follows from this document or its launch.
