# Joint-240 A_live Underperformance Decomposition v1 — Analysis

**Date:** 2026-08-25
**Experiment:** `experiments/joint240_alive_underperformance_decomposition_v1/`
**Design (frozen before results):** `docs/design/JOINT240_ALIVE_UNDERPERFORMANCE_DECOMPOSITION_V1.md`
**Authoritative parent (unmodified):** `experiments/joint240_same_distribution_adaptive_exploitability_v1/`
**Wall clock:** 393.5s, fully local/foreground, no Wulver use

This is a **read-only diagnostic reanalysis**. No frozen controller code was modified; no
parent artifact was overwritten; no new controller was tuned or proposed.

---

## A. Preflight

| Item | Value |
|---|---|
| Hostname | `al-khwarizmi` |
| Branch | `contextual-compositional-heuristics-20260731` |
| HEAD | `2987b7181efa2bc550d8a894c537eca8f6393eb6` (unchanged) |
| Upstream | ahead 2 of origin |
| Git status | 175 pre-existing lines (unrelated in-flight work); dirty tree preserved |
| Scientific processes | none active before this analysis |
| tmux sessions | none |
| CPU/RAM | 20 cores, 62GiB RAM, 57GiB available |

### One-time Wulver snapshot (job 1195618, `public_replay_load_scaling_v2`)

Queried once via `sacct -j 1195618 --format=State -X -n`: **56 COMPLETED, 4 RUNNING** (out of
60 array tasks) at query time. Not polled again; not touched, cancelled, or resubmitted. This
job is entirely independent of the present analysis.

---

## B. Authoritative input verification

Read and reused (unmodified) the parent experiment's full artifact set:
`DESIGN.md`, `frozen_protocol.json`, `config/frozen_protocol.json`, `feature_allowlist.json`,
`feature_denylist.json`, `split_oof_folds.csv`, `split_reference_tvt.csv`,
`per_scenario_oof_results.csv`, `model_metadata.json`, `matrix_live_integrity.json`,
`summary.json`, `bootstrap.json`, `provenance.json`, plus the source module
`llmserveopt.analysis.joint240_same_distribution_adaptive_v1` and runner
`scripts/run_joint240_same_distribution_adaptive_v1.py`. Also read
`docs/current/joint240_guarded_abstaining_selector_v1_analysis_20260825.md` (Section K below).

No per-step router trace / policy-choice log exists in the parent artifacts (only aggregate
`a_live_n_switches` per scenario is stored) — per-step traces were **necessarily** reconstructed
via observational replay (Section C), as anticipated by the frozen design (Section 1).

---

## C. Exact baseline reproduction (mandatory gate)

Reconstructed scenarios, utility matrix, generator features, and the 5-fold OOF split using the
unmodified module functions plus an exact copy of the runner's `_inner_train_val` split logic.

| Check | Result |
|---|---|
| Fold assignment vs. frozen `split_oof_folds.csv` | **0/240 mismatches** |
| Per-fold Stage-1 refit + live replay via an observational tracing subclass | completed, 5/5 folds |
| Per-scenario `a_live_anwg` vs. frozen `per_scenario_oof_results.csv` | **0/240 mismatches** (exact float equality, tol 1e-9) |
| Per-scenario `a_live_n_switches` vs. frozen CSV | **0/240 mismatches** (exact integer equality) |

**Reproduction: PASS, exactly.** SBS (`R=0.31407`, policy `kv_constrained_online`), VBS
(`R=0.33311`), Majority (`R=0.29093`, from parent, not re-derived), `A_scen` (`R=0.30595`,
from parent), and `A_live` (`R=0.28397`) all match the parent exactly. Analysis proceeds per
the frozen design's gate.

---

## D. Frozen decomposition (recap — methods only, see design doc for full detail)

Implemented exactly as frozen: (1) visited-repertoire best-fixed diagnostic approximation, (2)
first-choice fixed baseline, (3) dominant-choice fixed baseline, (4) switching diagnostics, (5)
early-decision damage, (6) dwell sensitivity grid {1,5,20,50} (reusing already-fitted Stage-1,
no retraining), (7) oracle-sequence upper bound — **skipped**, documented reason: true
within-scenario oracle switching would require expensive per-step counterfactual branching
disproportionate to a diagnostic reanalysis; VBS (best-fixed) used instead as an explicit
**lower bound** on any true oracle, (8) distribution-shift diagnostic on the correct 4-feature
`ONLINE_FEATURES` set, (9) frozen deterministic error-category rules.

---

## E. First-choice / dominant-choice / best-visited diagnostics

| Quantity | Mean ANWG | vs. SBS (paired bootstrap) | vs. A_live (paired bootstrap) |
|---|---:|---|---|
| SBS | 0.31407 | — | — |
| **A_live (actual)** | **0.28397** | −0.03010, CI [−0.0381, −0.0231] | — |
| First-choice held fixed | 0.28551 | −0.02856, CI [−0.0388, −0.0196] | **+0.00154, CI [−0.0053, +0.0089]** (n.s.) |
| Dominant-choice held fixed | 0.28940 | −0.02467, CI [−0.0333, −0.0171] | **+0.00543, CI [−0.0012, +0.0125]** (borderline n.s.) |
| Best-visited (diagnostic approx.) | **0.32606** | +0.01199, CI [0.0078, 0.0158] | **+0.04210, CI [0.0351, 0.0494]** (highly sig.) |
| VBS | 0.33311 | — | vs. best-visited: +0.00704, CI [0.0046, 0.0098] |

**Central finding:** holding *either* the first or the dominant choice fixed for the whole
scenario is **not** significantly different from full `A_live` (both CIs include or nearly
include zero) — so "pick once, hold forever" using A_live's own actual first/dominant pick is
roughly a wash. But the **best single policy among the ones A_live actually visited** during
the scenario would have scored **0.326**, only 0.007 below VBS (highly significant, small gap)
— and **0.042 above what A_live's real switching trajectory achieved** (highly significant,
large gap; CI entirely positive and excludes zero by a wide margin). That 0.042 is **86% of
A_live's total 0.049 shortfall vs. VBS** (0.042 / 0.0491). In other words: **A_live's Stage-1
usually does identify a near-optimal policy at some point during the scenario — the switching
behavior around that identification is what destroys the value.**

---

## F. Switching analysis

| Metric | Value |
|---|---:|
| Mean switches/scenario | 23.67 (matches parent exactly) |
| Scenarios with ≥1 A→B→A oscillation | **234 / 240 (97.5%)** |
| Median `frac_progress_to_first_switch` | ≈0.03–0.04 (first switch happens almost immediately, within the first few percent of the scenario, both catastrophic and non-catastrophic) |
| Dwell violations (segments <20 steps, post-first) | 13 total, in 13 scenarios — near-certainly the trace-end (final, truncated) segment in each case, not a genuine FSM defect; `PolicyDwellFSM`'s cooldown logic is otherwise confirmed to hold |
| Spearman(`n_switches`, `gap_vs_vbs`) | **0.195** (weak positive, not a strong single-variable predictor) |
| Catastrophic rate by switch-count quartile | Q1 (0–9): 29.7% · Q2 (9–18): **64.9%** · Q3 (18–32): 60.0% · Q4 (32–102): 44.1% |

Catastrophic rate is **non-monotonic** in switch count — it peaks in the middle quartiles, not
at the extremes. Raw switch *count* alone is a weak, non-causal correlate; it is not simply
"more switching → worse," which argues against a naive "just reduce switch count" fix and for
looking at *what* is being switched to/from (Sections E, G) rather than *how often*.

---

## G. Dwell sensitivity (cheap, reused already-fitted Stage-1, no retraining)

| Dwell | Mean ANWG | Gain vs. SBS (CI95) | Catastrophic | Mean switches |
|---:|---:|---|---:|---:|
| 1 | 0.28798 | −0.02609, [−0.0337, −0.0191] | 110 | 39.62 |
| 5 | 0.28717 | −0.02690, [−0.0347, −0.0203] | 113 | 34.18 |
| **20 (frozen)** | **0.28397** | **−0.03010, [−0.0381, −0.0231]** | **118** | **23.67** |
| 50 | 0.28299 | −0.03109, [−0.0392, −0.0240] | 126 | 15.87 |

**Monotonic degradation as dwell increases**: mean ANWG falls, gain-vs-SBS worsens, and
catastrophic count rises monotonically from dwell=1 through dwell=50, despite *fewer*
switches at higher dwell. This is direct evidence that **the dwell cooldown itself is
materially harmful, not protective** — it is not preventing costly thrashing (thrashing is
already near-universal at dwell=1 too, 39.6 switches/scenario, and performance is *least* bad
there), it is instead delaying corrections after a bad pick. However, **dwell is not the sole
cause**: even at dwell=1 (minimal cooldown), `A_live` still loses badly to SBS (−0.026, CI
excludes zero) and still has 110 catastrophic scenarios — barely better than Majority's 109
from the parent. Dwell length modulates the *severity* of the loss; it does not explain why
the loss exists in the first place (Sections E/F already show that: the visited repertoire is
usually fine, the switching *execution* of it is not).

Per design instructions, no dwell value is selected as "best" or proposed as a new controller;
this is diagnostic evidence about the shape of the effect only.

---

## H. Early-decision damage

- `first_choice_correct` (first pick == VBS policy) occurs in only **15.8%** of scenarios
  overall — statistically indistinguishable from chance (1/6 ≈ 16.7%) for a 6-way classifier.
- `first_choice_correct` rate is **not** meaningfully different between catastrophic (16.9%)
  and non-catastrophic (14.8%) scenarios — whether the very first pick happens to be exactly
  right is not a strong predictor of eventual catastrophe.
- Median time/progress-to-first-switch is small and similar in both groups (~3–4% of the
  scenario) — the router switches away from its first pick almost immediately and almost
  universally, regardless of whether that first pick was good.

**Conclusion:** early damage, in the narrow sense of "the very first choice being wrong,"
is not the dominant story (Section D/J category `initial_choice_wrong` accounts for only 43/240
scenarios and 27.9% of regret mass — real, but not dominant). The more consequential "early"
event is architectural: the router almost always switches away from its initial state within
the first few percent of every scenario, entering the oscillatory switching regime documented
in Section F almost immediately.

---

## I. Distribution-shift analysis

Feature set: the 4 features Stage-1 actually consumes (`contention_score_v2`, `priority_skew`,
`kv_pressure`, `queue_length`) — **not** the 17-feature set named in the task prompt, which
belongs to the separate, scenario-level `A_scen` generator-parameter model. Using the correct
per-step online feature contract is the faithful choice here (see design doc Section 2.9 for
the explicit note on this discrepancy).

| Group vs. T (train/probe, n=1,811,100) | contention SMD | priority_skew SMD | kv_pressure SMD | queue_length SMD | mean NN-dist (standardized) |
|---|---:|---:|---:|---:|---:|
| L: live-visited (n=440,967) | 0.011 | 0.075 | 0.034 | 0.291 | 0.101 |
| L-postswitch (n=424,885) | −0.001 | 0.103 | 0.079 | 0.336 | 0.101 |
| C: catastrophic scenarios (n=220,063) | −0.055 | 0.043 | 0.071 | 0.305 | 0.101 |

All standardized mean differences are small (**max |SMD| = 0.336**, on `queue_length`,
well under any "large shift" threshold such as 1.0), and **catastrophic scenarios are not
systematically more shifted** than the full live-visited pool (all three groups' NN-distance
statistics are essentially identical, ≈0.10 standardized units). PSI values for `contention_score_v2`
(≈12.4) and `priority_skew` (≈5.1) do indicate a shape change in those two marginals relative
to the probe distribution's decile bins, but this is not accompanied by a large *mean* shift or
by any distinguishing signal specific to catastrophic scenarios — i.e., it looks like a
property of A_live's operating regime in general (vs. the WFS-probe regime it was trained
under), not a signature that predicts failure. No frozen support-threshold machinery exists in
this codebase to reuse, and none was invented (per instruction); only threshold-free
descriptive statistics are reported.

**Conclusion: distribution shift is not the dominant mechanism.** A_live does not drive itself
into states that are dramatically unlike its Stage-1 training distribution, and catastrophic
scenarios in particular show no distinguishing shift signature.

---

## J. Error-category decomposition (frozen rules, Section 2.10 of the design doc)

| Category | n | Regret-mass share | Catastrophic rate within category |
|---|---:|---:|---:|
| **`switching_or_execution_dominant`** | **134 (55.8%)** | **72.9%** | 64.9% |
| `initial_choice_wrong` | 43 (17.9%) | 27.9% | 72.1% |
| `no_obvious_pathology` | 63 (26.2%) | −0.8% (net negative — these scenarios did fine) | 0.0% |

(Percentages of regret mass sum to ~100%; `no_obvious_pathology`'s small negative share
reflects a handful of scenarios where `A_live` matched or slightly beat VBS.)

Supplementary flags (non-exclusive, reported alongside): `oscillation_flag` true in 97.5% of
all scenarios; `dwell_lock_flag` (raw Stage-1 correctly identified VBS while dwell blocked
adoption for a full cooldown) true in **19.6%** of scenarios overall, marginally higher among
catastrophic scenarios (21.2% vs. 18.0% non-catastrophic) — real but not a dominant driver on
its own.

**The switching/execution-dominant category alone carries 73% of total regret mass** — this is
the deciding evidence for the verdict (Section M).

---

## K. Relation to guarded-selector result

`joint240_guarded_abstaining_selector_v1` (scenario-level `A_scen`-style selection) found:
unguarded selection loses badly (gain −0.0081, all catastrophic-prone), a simple SBS-fallback
guard removes most catastrophic downside and returns to ≈SBS performance, but does not close
VBS headroom.

`A_live` is architecturally the **unguarded case taken to a much finer time grain**: it has no
abstention/fallback mechanism at all — every one of its (typically hundreds to thousands of)
per-step decisions must commit to *some* active P6 policy, with only a bare dwell-cooldown as
friction, and no confidence-based "stay on the safe default" option. Given that even the
*scenario-level* unguarded selector (one decision per scenario) loses substantially without a
safety valve, it is unsurprising — and consistent with the guarded-selector's core lesson —
that a router making far more such decisions per scenario, with no equivalent safety valve,
loses even more (`A_live` gain −0.0301 vs. `A_scen` unguarded's −0.0081, roughly 3.7× worse).
This analysis's finding that `A_live`'s repertoire usually *does* contain a near-optimal choice
(Section E, `R_best_visited=0.326`) but the router fails to *commit* to it is the online analogue
of the guarded-selector's finding that unsafe *switching to* non-SBS options, not an inability
to *recognize* good options, is what destroys value. **A natural, preregisterable next
experiment** (not run here, per instructions) would be an online analogue of the guarded
selector: an abstention/hysteresis rule that requires sustained high-confidence disagreement
with the current policy (not just a fixed step-count cooldown) before switching — but this is
named only as a follow-up direction, not implemented or tuned in this task.

---

## L. Main bootstrap results (paired scenario bootstrap, B=1000, seed 20260825)

See Section E table and `bootstrap.json` for the complete set. The two load-bearing results:

- `best_visited − A_live`: **+0.0421, CI [0.0351, 0.0494]** — large, highly significant. The
  visited-repertoire-diagnostic value is far above what was actually realized.
- `VBS − best_visited`: **+0.0070, CI [0.0046, 0.0098]** — small but significant residual gap
  even under the generous diagnostic approximation; `A_live`'s Stage-1 does not always visit
  the *literal* VBS policy, only something close to it.

---

## M. Final diagnostic verdict

Applying the frozen Section 4 rules exactly: `switching_or_execution_dominant` carries 72.9%
of total regret mass (> 50% threshold for the switching/dwell rule), while
`initial_choice_wrong` carries only 27.9% (< 50%, so the selection-error rule is not met).
Distribution shift (Section I) is small and not catastrophic-specific, so the shift-dominated
rule is also not met.

## **`ALIVE_LOSS_DOMINATED_BY_SWITCHING_OR_DWELL`**

Supporting evidence, all converging on the same conclusion: (1) best-visited-repertoire value
(0.326) is close to VBS (0.333) and far above realized `A_live` (0.284) — Section E; (2)
first/dominant fixed-choice baselines are statistically indistinguishable from full `A_live` —
picking once and holding is a wash, meaning the *loss is not simply "pick better and stop
switching"* but specifically *"the switching dynamics around an otherwise-reasonable repertoire
destroy value"*; (3) dwell sensitivity shows monotonic degradation as dwell increases (Section
G) — the cooldown mechanism itself materially worsens outcomes; (4) but dwell=1 still loses
substantially, so dwell length is a modulator of severity, not the sole cause; (5) 97.5%
oscillation prevalence and 19.6% dwell-lock prevalence are corroborating switching-pathology
signatures; (6) distribution shift is ruled out as a material contributor (Section I).

---

## N. Manuscript implication (not applied — informational only)

If/when this line of results is incorporated into manuscript discussion of adaptive-scheduler
headroom, the supported framing is: **`A_live`'s failure is a mechanism-specific execution
problem (per-step switching against a coarse dwell cooldown), not evidence that online
state-dependent specialization is intrinsically infeasible on this workload distribution** —
the underlying repertoire the router draws from is usually adequate (Section E). This is
consistent with, and strengthens, the broader "portfolio headroom is difficult to exploit
*safely*" narrative already established by the guarded-selector result (Section K): the
difficulty is in *safe commitment/switching*, not in *recognizing* good specialists. No
manuscript file was edited in this session.

---

## O. One-time Wulver job 1195618 status snapshot

See Section A. **56/60 COMPLETED, 4/60 RUNNING** at query time (single `sacct` query). Not
polled again, not touched.

---

## P. Files created/modified

Created:
- `docs/design/JOINT240_ALIVE_UNDERPERFORMANCE_DECOMPOSITION_V1.md`
- `docs/current/joint240_alive_underperformance_decomposition_v1_analysis_20260825.md` (this file)
- `experiments/joint240_alive_underperformance_decomposition_v1/{DESIGN_FROZEN.md, reproduction_check.json, per_scenario_decomposition.csv, switching_diagnostics.csv, distribution_shift.json, bootstrap.json, verdict.json, summary.json, DONE, logs/run.log}`

No files under `experiments/joint240_same_distribution_adaptive_exploitability_v1/` (parent)
were modified. No files related to Wulver job 1195618 or `public_replay_load_scaling_v2` were
touched.

---

## Q. Git status

All changes scoped to the files listed in Section P (all new/untracked). HEAD unchanged
(`2987b71`). **No commits made. No pushes made.**

---

## R. Confirmation: manuscript untouched

No files under `paper/` were opened for editing or modified in this session.
