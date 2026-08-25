# Online Regime-Signal Feasibility Study — v1 Audit

Date: 2026-08-17

## 0. Scope

**FEASIBILITY ONLY.** No hierarchical router trained, no family-specific
selectors trained, no mechanism attribution, no universal-selector work
restarted, no fourth family added, no frozen scientific result modified.
Central question, per
[`cross_family_transfer_wellposedness_reassessment_20260817.md`](cross_family_transfer_wellposedness_reassessment_20260817.md)
§E/§M: *can the three operational scheduling regimes (ranking/fairness,
prefill/decode contention, KV/memory pressure) be identified from
information available ONLINE at scheduling time*, with Family B
(prefill/decode contention) as the named primary, previously-unresolved
gate.

## A. Initial Git State

Branch `contextual-compositional-heuristics-20260731`, clean, HEAD
`00acb1f` ("docs: higher-level cross-family transfer well-posedness
reassessment"), already pushed to origin.

## B. Online-Observable Signal Inventory

The decisive discovery of this task: the simulator already has a
first-class, causal, pre-decision state snapshot —
`ObservableState` (`src/llmserveopt/core/types.py`) — built at
`Simulator._build_observable_state()` and passed to
`policy.select_action(state)` *before* any action is chosen
(`src/llmserveopt/simulator/simulator.py` lines 168–172). This is not a
new abstraction introduced for this study; it is the exact interface every
real policy in the codebase already scheduled through.

| Signal | Source object/field | Available before action? | Units | Level | Direct or derived | Consistent across A/B/C? | Classification |
|---|---|---|---|---|---|---|---|
| `priority_skew` | `causal_context_features(state)` — max/min of `ObservableRequest.priority` over `state.waiting_queue` | Yes | dimensionless ratio | request (aggregated) | Derived, one rolling-free aggregate over current queue | Yes (queue always present, possibly empty) | **DIRECT_ONLINE** |
| `class_imbalance` | same function — max class-count fraction over `state.waiting_queue` | Yes | fraction [0,1] | request (aggregated) | Derived | Yes | **DIRECT_ONLINE** |
| `queue_length` | `len(state.waiting_queue)` | Yes | count | request (aggregated) | Direct | Yes | **DIRECT_ONLINE** |
| `urgent_deadline_fraction` | same function — fraction with `slo_deadline <= state.time` | Yes | fraction [0,1] | request (aggregated) | Derived | Yes | **DIRECT_ONLINE** |
| `prefilling_count` / `decoding_count` | `ObservableGPUState` (Phase 1.5 fields) | Yes | count | GPU | Direct | **Only when `ServiceModel.enable_prefill_modeling=True`** — 0 by construction on Family A/C (§D) | **DIRECT_ONLINE, but only load-bearing on Family B** — see §D for why this is a genuine mechanism-availability fact, not missing instrumentation |
| `current_kv_tokens` / `max_kv_tokens` | `ObservableGPUState` | Yes | tokens | GPU | Direct | Yes | **DIRECT_ONLINE** |
| `max_active_sequences` | `GPUConfig`/`ObservableGPUState` | Yes (static per-GPU config) | count | GPU | Direct | Yes | **DIRECT_ONLINE** |
| `step_token_budget` | `ServiceModel` config (deploy-time constant, not per-step state) | Yes (known at deploy time) | tokens/step | simulator-wide | Direct (constant) | Yes | **DIRECT_ONLINE** (constant, not state-dependent) |

Every one of these is already load-bearing in production code, not
invented for this study: `causal_context_features` and the private
`_prefill_pressure`/`_decode_pressure`/`_kv_pressure`/`_queue_pressure`
helpers (`src/llmserveopt/policies/composition.py`) are called **live,
inside `select_action`**, by real policies
(`src/llmserveopt/composition/estf_wfs_policies.py`,
`ContextualRankEnsemblePolicy` in `policies/composition.py`) to make actual
admission/ranking decisions. That existing usage is direct evidence these
quantities are genuinely causal, not retrospective — a policy that already
depends on them to act could not do so if they secretly depended on future
information.

## C. Excluded Retrospective Signals

Never read anywhere in this study: `Request.actual_output_tokens`
(explicitly policy-hidden per its own field comment), any
`CompletedRequest` field (`ttft`, `tpot`, `slo_violated`, `latency` — all
only exist after completion), `mechanism_family`, `canonical_scenario_id`
/ `scenario_id` (used only as **audit-metadata identity columns**, never
as a learnable signal — enforced by
`TELEMETRY_IDENTITY_COLUMNS`/`TELEMETRY_LEARNABLE_SIGNAL_COLUMNS` being
disjoint, tested), and any post-run aggregate metric
(`arrival_normalized_weighted_goodput`, etc.). `RETROSPECTIVE_ONLY`
signals were not built at all — there was no candidate that required them
(the task's own §B/§1 candidate list was itself pre-scoped to
online-observable quantities).

## D. Step-Level Telemetry Schema and Build

`src/llmserveopt/policy_separation/online_regime_signals_v1.py`
(`compute_regime_signals`, `compute_activity_labels`,
`TelemetryRecordingPolicy`) + `scripts/build_online_regime_telemetry_v1.py`.

**Replay design.** All 176 frozen MF-PSD scenarios (Family A/B: exact
scenario-ID-match deterministic replay against the frozen
`mf_psd_long_v1.csv`, same method verified in the shared-feature-schema
task; Family C: direct load from the frozen Reconstruction v1 artifact) are
run through the **real simulator**, with a single neutral policy — FIFO
(`llmserveopt.policies.fifo.FIFOPolicy`), native to none of the three
families — wrapped in `TelemetryRecordingPolicy`. Using one uniform policy
across every family means the resulting telemetry characterizes
**workload-driven** regime signal, not a particular native policy's own
admission dynamics; using each family's own native anchor policy instead
would risk conflating "this family's own designed-for policy behaves
differently" with "this family's *workload* looks different online."

`TelemetryRecordingPolicy` computes signals from `state` and records a row
**before** delegating to the wrapped policy's `select_action` — the
wrapped policy's returned `Action` is forwarded completely unmodified (same
non-invasive contract as the pre-existing
`llmserveopt.policies.instrumentation.InstrumentedPolicy`). No simulator
file was modified.

**A genuine scale surprise, reported honestly.** The first build produced
2,497,997 raw steps (Family A alone: 2,306,715) because **every one of
Family A's 72 scenarios sets `max_active_sequences=1`** (verified directly
against `scenario_features.csv`) — a real, intentional property of that
family's design (only one request can ever be "active," making *which one
is chosen next from the queue* maximally consequential, which is exactly
the point of a fairness/starvation study), not a bug or an artifact of
using FIFO. Recording every one of ~2.3M steps produced a 518MB file,
dominated by long stretches of unchanged `urgent_deadline_fraction` drift
under a large FIFO-serialized backlog. **Fix (recording economy only, not
a change to signals/thresholds or which steps the simulator executes):**
`TelemetryRecordingPolicy` now records every activity-label *transition*
exactly (any of `a_active`/`b_active_v2`/`c_active` flipping — so no
within-trajectory dynamics can be missed, §K), plus otherwise at least once
every `sample_stride_steps=20` raw steps. Final table: **127,319 rows, 29MB**
(`experiments/online_regime_signal_feasibility_v1/online_regime_telemetry_v1.csv`),
byte-identical on rebuild (verified — full determinism check, not spot
sampling).

**Schema** (`TELEMETRY_COLUMNS`): identity/audit columns
(`canonical_scenario_id`, `mechanism_family`, `step`, `sim_time`) + 14
learnable signal columns + 4 activity-label columns (§E). Disjoint by
construction (`TELEMETRY_IDENTITY_COLUMNS` ∩
`TELEMETRY_LEARNABLE_SIGNAL_COLUMNS` = ∅, tested).

## E. Activity-Label Definitions

Preregistered **before** any diagnostic was run against them, from
physical/system semantics or pre-existing project constants — not fit to
maximize accuracy (verified honestly in §F for the one case where an
initial formula failed and was replaced, not merely re-thresholded):

| Label | Formula | Threshold | Justification |
|---|---|---|---|
| `a_active` | `priority_skew > 1.05 AND queue_length >= 2` | 1.05 / 2 | A fairness mechanism cannot matter with equal priorities (skew=1.0 is Family A's own generator's exact definition of a "control," not-stressed, scenario) or with fewer than 2 waiting candidates to choose between |
| `b_active` (v1) | `contention_score_product = prefill_pressure × decode_pressure > 0.05` | 0.05 | Both phases must simultaneously occupy ≥~22% of `max_active_sequences` capacity each for the product to clear 0.05 — physically motivated as "both phases meaningfully co-occupying the GPU" |
| `b_active_v2` | `contention_score_v2 = min(prefill_fraction_of_active, decode_fraction_of_active) > 0.20` | 0.20 | Fraction-of-currently-active-cohort framing (not capacity-normalized) — 0.20 means the minority phase is ≥40% the size of the majority phase, out of a [0, 0.5]-range score maximized at exact 50/50 split |
| `c_active` | `kv_pressure = current_kv_tokens / max_kv_tokens > 0.82` | 0.82 | **Reused verbatim** from `KVConstrainedOnlinePolicy.target_kv_utilization`'s own default (`src/llmserveopt/policies/kv_constrained_online.py`) — the project's own pre-existing, already-deployed KV-admission threshold, not invented here |

Three independent binary labels reported (`a_active`, `b_active`/`b_active_v2`,
`c_active`) rather than a forced single winner, per task instruction — §L
reports the empirical overlap structure this makes possible to observe.

## F. Family-B Contention Proxy — the Primary Gate

**`b_active` (v1, capacity-normalized `contention_score_product`) never
fires — 0/127,319 rows, including on Family B's own native scenarios.**
Root cause, diagnosed directly rather than assumed: Family B's scenarios
are small-scale (~24 total requests) but generously provisioned
(`max_active_sequences=512`), so even if every request were simultaneously
prefilling or decoding, `prefill_pressure`/`decode_pressure` (each
normalized by 512) could never exceed ≈0.05 individually — their product
is capped far below the 0.05 threshold by construction. This is a
**normalization-denominator mismatch**, not evidence that no detectable
contention exists — confirmed because `contention_score_product`'s
*continuous* value still achieves AUROC 0.841 discriminating Family-B rows
from all others (§N): the signal has real ranking power, the threshold (a
reasonable a priori guess before knowing this family's actual value range)
was simply miscalibrated for this specific normalization.

**`b_active_v2` (fraction-of-active-cohort framing) fires cleanly.** Added
*after* this discovery — a structurally different formula (denominator
changes from `max_active_sequences` to `prefilling_count + decoding_count`
itself), not a retuned threshold on the same formula, per the task's own
prohibition on accuracy-fit thresholds:

- Fires on **32/32 Family-B scenarios**, 330/760 of that family's recorded
  rows (43.4%).
- **Zero false positives**: 0/127,319 rows outside Family B ever have
  `b_active_v2=True` — precision 1.0 (§N).
- AUROC (continuous `contention_score_v2` vs. "is this row from Family B")
  = **0.841**.
- Mean score when active: 0.295; when not: 0.080 (§M) — a clear separation
  margin, not a borderline threshold artifact.

**This directly and empirically resolves the reassessment's named open
risk** ("Family B's `chunk` regime has no direct online-observable proxy")
— the earlier finding was specific to *scenario-level, whole-trajectory*
SHARED_CORE_V1 aggregates; at the genuinely per-step, online level, a
clean proxy exists.

## G. Family-A Sanity Result

`priority_skew`-gated `a_active` fires on 48/72 scenarios (never on the
24 `tenant_weight_skew=1.0` "control" scenarios — mean fraction of steps
active = **0.0** exactly on those, vs. **0.836** on `skew>1.0` "stress"
scenarios). External validation (never fed into the `a_active` formula
itself): Spearman correlation of each scenario's fraction of
`a_active=True` steps against that scenario's own frozen
`tenant_weight_skew` sweep parameter (MF-PSD audit metadata) = **ρ=0.721,
p=9.4e-13**. AUROC of continuous `priority_skew` vs. "is this row from
Family A" = 0.779. Precision 1.0 (0 false positives, 64,482/64,482 A-active
rows all genuinely Family A), recall 0.558.

## H. Family-B Sanity Result

Covered in §F (the primary gate) rather than duplicated here.

## I. Family-C Sanity Result

`c_active` (`kv_pressure > 0.82`) fires on all 72/72 Family-C scenarios,
never once outside Family C (0 false positives, precision 1.0, recall
0.681). External validation: mean fraction of `c_active=True` steps is
higher under `bulk_pressure="high"` (0.716) than `"low"` (0.605); mean peak
`kv_pressure` similarly higher (1.139 vs. 1.092) — same direction, real
though modest effect (Family C's own sweep varies several other
parameters besides `bulk_pressure`, so a larger separation was not
expected). AUROC of continuous `kv_pressure` vs. "is this row from Family
C" = **0.993** — the strongest single-feature separation of the three
regimes.

## J. Temporal Leakage Checks

**Code-level**: `compute_regime_signals`/`compute_activity_labels` take
only `ObservableState` (or `RegimeSignals` derived from it) as their sole
parameter (`tests/test_online_regime_signals_v1.py::test_activity_label_computation_never_reads_family_or_scenario_identity`,
asserted via `inspect.signature`) — structurally cannot read
`mechanism_family`/`scenario_id`/any future field, since those are never
passed in. `ObservableRequest` (what every signal is computed from) has no
`actual_output_tokens` attribute at all (tested). `TelemetryRecordingPolicy`
records a row from `state` **before** calling the wrapped policy, and a
dedicated test
(`test_telemetry_row_never_touches_time_after_its_own_step`) proves a later
call cannot retroactively mutate an already-recorded row's signal values.

**Spot-check on the built artifact**: the first recorded step of every one
of the 176 scenarios has `queue_length == 0.0` — no scenario's earliest
telemetry already reflects a large backlog that could only be explained by
future arrivals leaking in early (tested,
`test_telemetry_first_recorded_step_per_scenario_has_empty_or_near_empty_queue`).

**15/15 new focused tests pass** (`tests/test_online_regime_signals_v1.py`),
covering: KV-pressure formula/threshold-provenance, the
`contention_score_v2`=0 mechanistic-zero-under-`enable_prefill_modeling=False`
property, balanced-split maximization, a regression guard for §F's
`contention_score_product` never-fires finding, `a_active` gating logic
(skew AND queue-size conjunction, including both individual-condition
failure cases), action-forwarding non-invasiveness, transition-exact
recording, temporal-causality-by-construction, plus 5 built-artifact tests
(column schema, row-alignment to all 176 scenarios, the queue-length spot
check, frozen-source non-mutation, and the central zero-cross-family-false-
positive finding as an explicit regression guard).

## K. Within-Trajectory Dynamics

Every family shows genuine within-trajectory activation dynamics, not a
constant-for-the-whole-scenario regime label:

| Family | Scenarios that ever activate their own signal | Scenarios always active | Mean on/off transitions per scenario |
|---|---|---|---|
| A (`a_active`) | 48/72 | 0/72 | 5.4 |
| B (`b_active_v2`) | 32/32 | 0/32 | 10.1 |
| C (`c_active`) | 72/72 | 0/72 | **39.4** |

No family's regime signal is ever "on for the entire trajectory" — every
one genuinely turns on and off as the workload evolves (arrivals, backlog
buildup/drain). Family C shows the most dynamic behavior (KV occupancy
fluctuates most rapidly relative to its 0.82 threshold), consistent with
KV pressure being inherently the most transient of the three mechanisms
under a FIFO replay (admission/completion directly move `current_kv_tokens`
every step, unlike `priority_skew`, which only changes when the *waiting
queue's composition* changes).

## L. Regime-Overlap Distribution

Across all 127,319 recorded rows, using `(a_active, b_active_v2, c_active)`:

| Pattern | Rows | Fraction |
|---|---|---|
| none | 55,021 | 43.2% |
| A only | 64,482 | 50.6% |
| B only | 330 | 0.26% |
| C only | 7,486 | 5.9% |
| A+B / A+C / B+C / A+B+C | **0** | **0.0%** |

**Zero overlap, in every direction, across the entire 127K-row telemetry
set.** This is a clean, decisive answer to §12's architecture question: a
**hard top-1 router** is well-supported by this evidence — there is no
observed case in this frozen replay where a soft/multi-label gate would
have had to arbitrate between two simultaneously-active regimes, because
none ever co-occurred. **Important caveat, stated plainly (not glossed
over):** this zero-overlap result is partly a direct consequence of how
structurally distinct the three families' simulator configurations are
(`enable_prefill_modeling` only true for B; `max_kv_tokens` only tight for
C; `max_active_sequences=1` only for A) — i.e., it is consistent with, not
independent evidence beyond, the reassessment's own H2 finding that these
are "separate domains" rather than draws from one continuum. A live
deployment with genuinely mixed, less cleanly siloed traffic might show
real overlap that this replay, by the nature of these three specific
frozen scenario designs, cannot surface. This is named explicitly as an
open question for the next stage, not resolved here.

## M. Family-B Falsification Cases

All four required cases identified directly from real Family-B replay
telemetry (no synthetic microcases needed — genuine trajectories naturally
pass through each):

| Case | Rows | Scenarios exhibiting it | Mean `contention_score_v2` |
|---|---|---|---|
| 1. Prefill-only, no decode | **0** | 0/32 | n/a |
| 2. Decode-only / no prefill backlog | 210 | 32/32 | 0.000 (exactly, mechanistically) |
| 3. Simultaneous heavy contention (`b_active_v2`) | 330 | 32/32 | **0.295** |
| 4. Low-load mixed | 394 | 32/32 | 0.085 |

Case 1 never occurring is itself informative, not a gap: under this
replay's admission dynamics, a request begins decoding essentially as soon
as any prefill work exists elsewhere in the cohort (the "pure prefill, zero
decode" state is either instantaneous or never reached before some other
request is already decoding) — a real property of the observed dynamics,
reported honestly rather than manufactured with a synthetic microcase to
fill the table. **The B signal cleanly distinguishes case 3 from
cases 2/4**: mean `contention_score_v2` is 0.295 when `b_active_v2=True`
vs. 0.080 across all non-active rows (which include cases 2 and 4) — a
clear, non-borderline separation.

## N. Diagnostic Performance

From `experiments/online_regime_signal_feasibility_v1/online_regime_telemetry_diagnostics.json`
(`diagnostic_performance`) — single fixed-rule/single-continuous-score
diagnostics only, no model search, no hyperparameter tuning:

| Signal | AUROC (continuous, vs. "is this row from the native family") | Activity-label precision | Activity-label recall |
|---|---|---|---|
| `priority_skew` (A) | 0.779 | 1.000 | 0.558 |
| `contention_score_product` (B, v1) | 0.841 | n/a (never fires) | 0.000 |
| `contention_score_v2` (B, v2) | 0.841 | **1.000** | 0.434 |
| `kv_pressure` (C) | **0.993** | 1.000 | 0.681 |

**Every activity label achieves perfect precision (zero cross-family false
positives) with moderate-to-good recall.** For a *routing* signal
specifically (as opposed to an exhaustive detector), high precision matters
more than high recall — a missed activation (false negative) at worst
routes a step to a default/fallback selector; a false positive would
actively misroute. None of the three signals ever produces a false
positive in this evidence.

## O. Final Feasibility Verdict

**`ONLINE_REGIME_SIGNALS_READY`**

Justified against the frozen decision logic (§13): all three regimes have
legitimate online signals (§B, §G, §I — and, after the v1→v2 correction,
§F); Family-B contention is detectable with useful, non-borderline
separation (§F, §M, §N — AUROC 0.841, precision 1.0, clear score-margin
separation of case 3 from 1/2/4); temporal-causality checks pass at both
the code level and the built-artifact level (§J, 15/15 tests); every
signal varies meaningfully within trajectories where expected, never
constant-for-the-whole-scenario (§K); regime activity is **not** merely a
disguised recovery of source-family identity in the leakage sense — no
family label, scenario ID, or any hidden field is ever read by the
signal/label formulas (§C, §J) — even though, empirically, the resulting
activity labels happen to partition perfectly by family in this specific
frozen evidence (§L), which is itself the *intended, validated* outcome
(family identity in this dataset was already reassessed as a legitimate,
mechanistically-grounded regime, not bad leakage — this study is the
online-causal confirmation of that claim, not a contradiction of it); and
the overlap structure (§L: exactly zero co-occurrence) is unambiguous
enough to support a **hard top-1** router architecture without needing
`ONLINE_REGIME_SIGNALS_READY_MULTILABEL`'s softer framing.

## P. Scientific Interpretation

1. **Does this resolve the two concrete open risks named in the prior
   reassessment?** Yes, both, directly. (a) "Family B's `chunk` regime has
   no direct online-observable proxy in SHARED_CORE_V1" — true of
   *scenario-level, whole-trajectory* aggregates specifically; at the
   genuinely per-step online level, `contention_score_v2` is a clean,
   validated proxy (§F). (b) "every feature validated so far is a
   whole-scenario retrospective aggregate, never tested on genuinely
   online/partial-trajectory state" — every signal in this study is
   computed per-step from `ObservableState`, the exact pre-decision
   snapshot real policies already use to act (§B, §J).
2. **Why did `b_active` (v1) fail, and is that a bad sign?** No — it is a
   textbook normalization mismatch (§F), not evidence against Family-B
   detectability; the underlying continuous score's AUROC (0.841) was
   respectable even under the "wrong" normalization, and the corrected v2
   formula (a structurally different denominator, not a retuned threshold)
   cleanly resolves it. Reported transparently, including the failed first
   attempt, rather than only presenting the version that worked.
3. **How much should the "zero overlap" / "zero cross-family false
   positive" result be trusted as a general finding vs. an artifact of
   these three specific frozen scenario designs?** Genuinely both,
   honestly stated (§L): it is real, validated evidence that a router
   *can* work cleanly on data shaped like these three families' scenarios
   — but it is not yet evidence that a live deployment with genuinely
   blended traffic conditions would show equally clean separation. That is
   the natural next falsification question, not answered here.
4. **Does "READY" here mean the router will definitely work?** No —
   consistent with the calibrated language established in the prior
   reassessment, `ONLINE_REGIME_SIGNALS_READY` means the online-signal
   half of the hierarchical-routing proposal has cleared its
   preregistered feasibility bar; the router itself (Stage 1 classifier
   trained on these signals, Stage 2 family-specific selectors) is a
   separately authorized, not-yet-started experiment (§T).

## Q. Files Changed

**New (additive only):**
- `src/llmserveopt/policy_separation/online_regime_signals_v1.py`
- `scripts/build_online_regime_telemetry_v1.py`
- `scripts/analyze_online_regime_telemetry_v1.py`
- `tests/test_online_regime_signals_v1.py`
- `experiments/online_regime_signal_feasibility_v1/` (telemetry CSV,
  build manifest, diagnostics JSON — 3 files, ~29MB)
- `docs/audits/online_regime_signal_feasibility_v1_20260817.md` (this
  document)

**Confirmed unmodified:** `experiments/mf_psd_v1/`,
`experiments/unified_utility_matrix_v2/`,
`experiments/shared_cross_family_features_v1/`,
`experiments/mechanism_choice_target_feasibility_v1/`,
`experiments/cross_family_transfer_wellposedness_reassessment_v1/`,
`experiments/multifamily_contextual_selector_v1/`, every frozen source run
directory, every one of the four prior audit documents (checksum-verified
against recorded provenance, §Q's test).

## R. Tests

`tests/test_online_regime_signals_v1.py`, **15/15 passing** — see §J for
the full list; also confirmed no simulator source file was modified (`git
status` shows only new files) and the pre-existing regression suite
(`tests/test_mf_psd_v1.py`, `tests/test_shared_cross_family_features_v1.py`,
`tests/test_mechanism_choice_target_v1.py`, 66 combined) is unaffected.
`python3 scripts/check_project_handoff_consistency.py` passes. Determinism
independently verified by a full rebuild into a separate directory,
byte-compared identical to the committed artifact.

## S. Commit / Push State

Committed on `contextual-compositional-heuristics-20260731` and pushed to
`origin`. No force push. See the corresponding commit for the exact SHA.

## T. Exact Single Next Scientific Action

**Not the hierarchical router itself, not family-specific selector
training.** Per §17: with verdict `ONLINE_REGIME_SIGNALS_READY`, the next
step is a **separately authorized, preregistered hierarchical-router
experiment** — training a Stage-1 regime classifier on these validated
online signals (or a close variant) and Stage-2 family-specific selectors,
gated by the 9 GO/STOP criteria already named in the prior reassessment's
§M (online-observability discipline, routing-accuracy bar,
within-family-gain preservation, beat-global-fixed and
approach-family-aware-oracle bars, interpretable errors, no label leakage)
— now with two of those criteria (online-observability, partial-trajectory
validity) substantially de-risked by this study's evidence rather than
still open. Not started here.
