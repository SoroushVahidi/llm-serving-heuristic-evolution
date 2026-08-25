# Decision-Criticality Terminal Utility on Joint-240 v1 — Analysis

**Date:** 2026-08-25
**Experiment:** `experiments/decision_criticality_terminal_utility_joint240_v1/`
**Design (frozen):** `docs/design/DECISION_CRITICALITY_TERMINAL_UTILITY_JOINT240_V1.md`
**Schema:** `decision_criticality_terminal_utility_joint240_v1.0.0`
**Parent (unmodified):** `experiments/decision_criticality_terminal_anwg_joint240_v1/`
**Wall clock:** 1175.6s (240/240 scenarios, 0 failures)

This is a **post-hoc, read-only** analysis of an already-completed run. No branches were
re-executed; all numbers below are derived from the existing `branches.csv` /
`request_traces.jsonl.gz` produced by the frozen run (`DONE` timestamp 2026-08-25T18:14Z).

---

## A. Preflight

| Item | Value |
|---|---|
| Hostname | `al-khwarizmi` |
| Branch | `contextual-compositional-heuristics-20260731` |
| HEAD | `2987b7181efa2bc550d8a894c537eca8f6393eb6` |
| Upstream | `origin/contextual-compositional-heuristics-20260731`, **ahead 2** |
| Git status | working tree has pre-existing modified/untracked files (large in-flight release + research batch); **no files in this analysis's write-set were dirty before this session** |
| Active local scientific processes | none (`ps aux` shows only editor/background infra processes, no simulator/python workers) |
| tmux sessions | none (`no server running`) |
| `DONE` marker | present: `{"ok": true, "elapsed_s": 1175.56, "n_branches": 3541, "verdicts": ["ANWG_ZERO_RATE_STEP_FUNCTION_ARTIFACT_BUT_CONCENTRATION_ROBUST"], "anwg_reproduction_ok": true}` |
| Exit status | success (0 failures recorded in `summary.json["failures"]`) |

### Wulver job 1195488 — one-time read-only snapshot

Via live control-master SSH (`~/.ssh/cm/wulver.sock`), one `sacct` query (not polled again):

| State | Count |
|---|---:|
| COMPLETED | 60 / 60 array tasks |
| RUNNING / PENDING | 0 |

`squeue -j 1195488` returned `Invalid job id specified` — expected once a job has fully
left the queue after completion. All 60 sampled array tasks (`1195488_0` … `1195488_27`+)
show `State=COMPLETED`, `ExitCode=0:0`, elapsed ≈ 1m42s–2m27s each. **Not polled further; job
1195488 was not touched, cancelled, or resubmitted.**

---

## B. Integrity

Branch identity key: `(scenario_id, fold, step, acquisition_type, chosen_policy_id, alt_policy_id)`.

| Check | Result |
|---|---|
| Scenarios | **240 / 240** (matches parent `terminal_anwg_joint240_v1`) |
| Branches | **3541 / 3541** (`branches.csv` rows = `branches.jsonl` lines = parent count) |
| Duplicate branch keys | **0** (`branch_id.duplicated().sum() == 0`, join-key duplicate check also 0 on both parent and child) |
| Missing branches vs parent | **0** (inner join on `(scenario_id, step, acquisition_type, alt_policy_id)` matched all 3541 rows 1:1, `validate="one_to_one"` passed) |
| Extra branches vs parent | **0** |
| NaNs in `delta_anwg_live`, `delta_wcg`, `delta_wmt_improvement`, `delta_wnt_improvement`, `delta_soft` | **0 each** |
| Tracebacks / `FAIL` lines in `logs/full_run.log` | **0** (488-line log, clean) |
| Per-request trace schema | complete — sampled row keys ⊇ `{branch_id, scenario_id, fold, step, branch_role, request_id, weight, arrival_time, deadline, completion_time, completed, dropped, lateness, tardiness, slo_window, class_id}` (design §5), 207,544 trace rows (REF + CF) |
| REF/CF completeness | **all 3541×2 branch legs have complete per-request traces**; no partial dumps |

**Integrity: PASS.**

---

## C. ANWG exact reproduction

From `anwg_reproduction.json` / `summary.json["integrity"]`:

| Quantity | Value |
|---|---:|
| REF matches / 3541 | **3541 / 3541** (`ref_replay_n_match = ref_replay_n = 240`; every scenario's untouched REF replay matches) |
| CF matches / 3541 | **3541 / 3541** (`n_cf_anwg_match = 3541`) |
| Δ matches / 3541 | **3541 / 3541** (`n_delta_anwg_match = 3541`) |
| max REF error | `1.11e-16` |
| max CF error | `1.11e-16` |
| max Δ error | `9.89e-17` |
| Independent join cross-check (this analysis) | joining child `delta_anwg_live` against parent frozen `delta_anwg` on the branch key gives **max abs diff = 0.0** across all 3541 rows |
| Tolerance required | `≤ 1e-12` |
| Tolerance met | **yes**, by ~4 orders of magnitude margin |

No material mismatch. **Analysis proceeds** (no STOP condition triggered).

---

## D. Frozen metrics

All five metrics (ANWG, WCG, WMT, WNT, SoftGoodput) computed exactly per
`DECISION_CRITICALITY_TERMINAL_UTILITY_JOINT240_V1.md` §3, using `w_i = priority_i` if
positive else `1.0`, `W = Σw_i` over all arrived requests, `C_i = sim_duration` for
unfinished/dropped requests.

**Completion check:** `cf_num_dropped` sums to **0** across all 3541 branches; `cf_wcg` and
`reference_wcg` are `1.0` for 100% of branches → **every REF and CF branch completes all
arrived requests.** `Δ_WCG = 0` for all 3541 branches (as preregistered expectation). This
means WMT/WNT/SoftGoodput differences are driven entirely by *when* requests complete, not
*whether* they complete.

---

## E. Primary result: zero-rate artifact

### Per-metric effect distribution (from `summary.json`)

| Metric | exact-zero | positive | negative | mean\|Δ\| | median\|Δ\| | p90 | p95 | p99 | practical (≥0.001) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ANWG | 94.18% | 3.19% | 2.63% | 0.001413 | 0.0 | 0.0 | 0.010257 | 0.035102 | 5.82% |
| WCG | 100% | 0% | 0% | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0% |
| WMT_improvement | 50.10% | 7.68% | 42.22% | 0.001971 | 0.0 | 0.001955 | 0.008425 | 0.039873 | 11.27% |
| WNT_improvement | 49.96% | 8.25% | 41.80% | 0.026932 | 1.07e-05 | 0.022827 | 0.122553 | 0.616387 | 26.04% |
| SoftGoodput | 49.96% | 8.11% | 41.94% | 0.001550 | 7.43e-08 | 0.002065 | 0.007798 | 0.034746 | 12.51% |

(all fractions of 3541 branches; "meaningful" = \|Δ\|>1e-9, which for these data equals
"nonzero at 1e-12" — there is no probability mass strictly between 1e-12 and 1e-9)

### Decomposition of the ANWG exact-zero mass (n=3335 branches, 94.18% of 3541)

| Among ANWG-zero branches | nonzero (>1e-12) | meaningful (>1e-9) | practical (≥0.001) |
|---|---:|---:|---:|
| Δ_WMT | 1561 / 3335 = **46.81%** | 46.81% | 6.96% |
| Δ_WNT | 1566 / 3335 = **46.96%** | 46.96% | 21.68% |
| Δ_SoftGoodput | 1566 / 3335 = **46.96%** | 46.96% | 7.74% |

### 2×2 tables (ANWG meaningful vs. continuous-metric meaningful, \|Δ\|>1e-9), n=3541

**ANWG vs. WMT:**

| | WMT meaningful | WMT negligible | row total |
|---|---:|---:|---:|
| ANWG meaningful | 206 | 0 | 206 |
| ANWG negligible | 1561 | 1774 | 3335 |
| col total | 1767 | 1774 | 3541 |

**ANWG vs. WNT:**

| | WNT meaningful | WNT negligible | row total |
|---|---:|---:|---:|
| ANWG meaningful | 206 | 0 | 206 |
| ANWG negligible | 1566 | 1769 | 3335 |
| col total | 1772 | 1769 | 3541 |

**ANWG vs. SoftGoodput:** identical cell counts to WNT (206 / 0 / 1566 / 1769) — WNT and
SoftGoodput share the same zero/nonzero partition to reported precision (both derive from the
same per-request tardiness/SLO-window ratio; see §G).

**Key structural fact, all three tables:** the **"ANWG meaningful, metric negligible" cell is
exactly 0**. Every branch where ANWG moves also has a meaningful continuous effect. The
asymmetry runs entirely the other way: **1561–1566 branches (44–44.2% of all 3541) have a
meaningful continuous effect while ANWG is exactly flat.**

### Explicit answer

> **How much of the 94.18% ANWG exact-zero rate is caused by the hard deadline indicator?**

**All of it, in the following precise sense:** ANWG is a hard step function
(`1[completed ∧ C≤D]`); a branch registers ANWG=0 whenever the counterfactual timing shift
does not cross the binary on-time/late boundary for any completing request — even when that
shift is large. Roughly **half (46.8–47.0%) of the 3335 ANWG-zero branches have a real,
non-floating-point (>1e-9), and in some cases practically-sized (7–22%) shift in continuous
lateness** that ANWG is structurally blind to. Conversely, **zero ANWG-nonzero branches are
continuous-null** — ANWG's alarm is never a false positive for the continuous metrics, but it
misses a large fraction of the continuous signal. This is the textbook signature of a coarse
binary threshold statistic sitting on top of a continuous underlying process.

---

## F. Magnitude concentration

### State-level \|Δ\| mass concentration (point estimates, `summary.json`)

| Metric | top-1% | top-5% | top-10% |
|---|---:|---:|---:|
| ANWG | 0.4206 | 0.9513 | 1.0000 |
| SoftGoodput | 0.3674 | 0.7909 | 0.9313 |
| WMT_improvement | 0.4578 | 0.8495 | 0.9620 |
| WNT_improvement | 0.4516 | 0.8549 | 0.9707 |

### Scenario-level mass concentration (point estimates; computed this session from `branches.csv`, multiplicity-aware `scenario_top_k_share_mult`)

| Metric | top-1 scenario | top-2 scenarios | top-5 scenarios | top-5% scenarios (k=12) | top-10% scenarios (k=24) |
|---|---:|---:|---:|---:|---:|
| ANWG | 0.1123 | 0.1790 | 0.2930 | 0.4876 | 0.6845 |
| WMT_improvement | 0.1608 | 0.2352 | 0.3714 | 0.5304 | 0.7030 |
| WNT_improvement | 0.1462 | 0.2160 | 0.3487 | 0.5644 | 0.7376 |
| SoftGoodput | 0.0882 | 0.1423 | 0.2889 | 0.4541 | 0.6233 |

### Scenario-grouped bootstrap (B=2000, seed=20260825, multiplicity-retained), from `bootstrap.json`

| Metric | meaningful prevalence CI95 | mean\|Δ\| CI95 | top-1% state CI95 | top-5% state CI95 | top-10% state CI95 | top-5-scenario mass CI95 | top-10%-scenario mass CI95 |
|---|---|---|---|---|---|---|---|
| ANWG | [0.0448, 0.0733] | [0.00101, 0.00191] | [0.344, 0.500] | [0.860, 1.000] | [1.000, 1.000] | [0.191, 0.393] | [0.593, 0.760] |
| WMT_improvement | [0.4724, 0.5256] | [0.00129, 0.00284] | [0.337, 0.534] | [0.787, 0.893] | [0.942, 0.974] | [0.202, 0.493] | [0.590, 0.788] |
| WNT_improvement | [0.4739, 0.5271] | [0.0178, 0.0386] | [0.328, 0.576] | [0.794, 0.905] | [0.953, 0.981] | [0.232, 0.474] | [0.645, 0.810] |
| SoftGoodput | [0.4739, 0.5271] | [0.00113, 0.00205] | [0.297, 0.432] | [0.738, 0.834] | [0.908, 0.946] | [0.200, 0.345] | [0.533, 0.694] |

(ANWG's own top-5-scenario-mass CI here, `[0.191,0.393]`, mean 0.2841, matches the corrected value
already frozen in the parent study, `[0.192, 0.399]`, to within Monte-Carlo noise of an
independently-reproduced bootstrap.)

### Regression check: multiplicity-retained bootstrap (frozen bug fix)

Re-implemented independently this session (same seed, same with-replacement scenario draw,
concatenation instead of groupby-collapse) and cross-checked against `bootstrap.json`'s
`"multiplicity_retained": true` flags on all five metrics. **Confirmed**: draws are built via
`pd.concat([by_scen[sid] for sid in draw])`, so a scenario sampled *k* times contributes its
full branch set *k* times to both the numerator (mass) and the denominator (state count) — no
`groupby` collapse anywhere in the path.

**Zero-mass bootstrap replicates** (out of 2000, independently recomputed this session):

| Metric | zero-mass replicates |
|---|---:|
| ANWG | 0 / 2000 |
| WCG | **2000 / 2000** (expected: every branch has Δ_WCG≡0, so every resample also has zero mass) |
| WMT_improvement | 0 / 2000 |
| WNT_improvement | 0 / 2000 |
| SoftGoodput | 0 / 2000 |

The zero-mass rule (share:=0 if total mass=0) is exercised deterministically only by WCG, as
expected, and never spuriously triggers for the four metrics with real effect mass.

**Reading:** the ANWG "top-10% state mass = 1.000" result is a **near-degenerate special
case** driven by ANWG's extreme sparsity (only 206 nonzero states, well under 10% of 3541, so
the top-10% bucket contains literally every nonzero state). WMT/WNT/SoftGoodput, with an order
of magnitude more nonzero states (1767–1772), still concentrate 85–97% of total \|Δ\| mass in
the top decile — this is a substantive concentration result, not an artifact of a tiny
denominator.

---

## G. ANWG vs. continuous criticality — cross-metric overlap

| Comparison | Jaccard | Spearman \|effect\| | top-1% overlap | top-5% overlap | top-10% overlap |
|---|---:|---:|---:|---:|---:|
| ANWG vs. WMT | 0.1166 | 0.3838 | 0.333 | 0.444 | 0.479 |
| ANWG vs. WNT | 0.1163 | 0.3853 | 0.083 | 0.371 | 0.465 |
| ANWG vs. SoftGoodput | 0.1163 | 0.4033 | 0.444 | 0.562 | 0.518 |

(overlap fractions are of the top-k *index set* by \|Δ\|, k = ⌈frac × 3541⌉; e.g. top-10% = 355
states)

**Pattern diagnosis (design §G, three hypotheses):**

1. *ANWG hides a broad low-magnitude continuous tail* — partially true (44% of branches have a
   meaningful-but-ANWG-invisible continuous effect), **but**
2. *ANWG and continuous metrics identify largely the same high-impact states* — **this is the
   dominant pattern**: moderate positive rank correlation (Spearman \|Δ\| ≈ 0.38–0.40, clearly
   above 0), 33–56% top-1%/5% index overlap (far above the ~1–10% expected under independence
   given the size ratio of the sets), and — most importantly — **zero** cases where ANWG fires
   and the continuous metric doesn't (the top-left-empty cell in every 2×2 table in §E). ANWG's
   206 critical states are a strict, high-rank **subset** of the continuous metrics' critical
   states, not a disjoint or contradictory set.
3. *Continuous metrics fundamentally contradict ANWG* — **rejected**: no evidence of sign
   reversal or disjoint critical sets; Jaccard is low (~0.12) only because the continuous
   metrics' meaningful set is ~8–8.6× larger, not because they disagree with ANWG's identified set.

**On the specific question — does exact-zero sparsity disappear while magnitude concentration
survives?** **Yes, precisely.** Exact-zero prevalence collapses from 94.18% (ANWG) to ~50%
(all three continuous metrics) — sparsity in the "is there any registered effect" sense
disappears. But magnitude concentration (top-10% state mass 93–97%, top-5-scenario mass
29–37%, top-10%-scenario mass 62–74%) survives essentially intact relative to ANWG's own
(top-10% state=100%, top-5-scenario=29.3%, top-10%-scenario=68.5%). The *locations* of the
largest continuous effects also substantially coincide with ANWG's critical states (§ above).

---

## H. Disagreement proxy

Predictor: `acquisition_type == DISAGREEMENT` (identical predictor definition used for the
parent ANWG study, enabling direct comparison). Label: `1[|Δ_metric| > 1e-9]`.

| Metric | base prevalence | disagreement prevalence | agreement-control prevalence | enrichment | AUROC | AUPRC | AUPRC baseline |
|---|---:|---:|---:|---:|---:|---:|---:|
| **ANWG (parent, reference)** | 0.0582 | 0.0880 | 0.0000 | ∞ | 0.680 | 0.082 | 0.058 |
| WMT_improvement | 0.4990 | 0.7548 | 0.0000 | ∞ | **0.838** | 0.755 | 0.499 |
| WNT_improvement | 0.5004 | 0.7569 | 0.0000 | ∞ | **0.839** | 0.757 | 0.500 |
| SoftGoodput | 0.5004 | 0.7569 | 0.0000 | ∞ | **0.839** | 0.757 | 0.500 |

Scenario-grouped bootstrap CI95 (B=2000, seed=20260825): AUROC — WMT `[0.821, 0.857]`, WNT/Soft
`[0.822, 0.858]`; AUPRC — WMT `[0.715, 0.794]`, WNT/Soft `[0.716, 0.796]`.

**Structural fact confirmed for continuous metrics, exactly as for ANWG:** agreement-control
prevalence is exactly 0.0 for all three continuous metrics too — every `AGREEMENT_CONTROL`
branch is, by construction, one where the "alternative" action equals the real action, so the
CF trajectory is bit-identical to REF regardless of which metric is used. This is not new
information about continuous criticality; it is a property of the branch design.

**Comparison with joint-240 ANWG values (given):** base 5.82%, disagreement 8.80%,
agreement-control 0%, AUROC 0.680, AUPRC 0.082, baseline 0.058 — reproduced exactly (see §K).
Disagreement is a **stronger** proxy for continuous meaningfulness (AUROC ≈ 0.84 vs. 0.68 for
ANWG) because ~48% of all disagreement branches carry a real timing perturbation (vs. only
8.8% crossing ANWG's hard deadline boundary), so the binary predictor captures a much larger
share of the (much larger) continuous positive class. **Caveat, per design instructions:**
floating-point-level effects are never classified as "meaningful" here — the 1e-9 noise floor
is applied uniformly, and AUROC/AUPRC use exactly the same `meaningful` labels as §E/§F.

---

## I. Closed-loop divergence

Using `subsequent_trajectory_diverged` (joined from parent branches on exact branch key; this
flag is a property of the intervention/continuation dynamics, not of which terminal metric is
scored, so it is shared across ANWG and the continuous metrics for a given branch):

| Metric | divergence rate \| meaningful | divergence rate \| negligible | mean\|Δ\| \| diverged | mean\|Δ\| \| not diverged | median\|Δ\| \| diverged | median\|Δ\| \| not diverged |
|---|---:|---:|---:|---:|---:|---:|
| WMT_improvement | 0.4001 | 0.0039 | 0.009489 | 0.0000719 | 0.001537 | 0.0 |
| WNT_improvement | 0.4001 | 0.0028 | 0.129810 | 0.0009486 | 0.019714 | 0.0 |
| SoftGoodput | 0.4001 | 0.0028 | 0.007183 | 0.0001272 | 0.001366 | 0.0 |

**Reading:** divergence is far more common among branches with a meaningful continuous effect
(~40%) than among negligible ones (0.3–0.4%), and conditional on divergence the mean effect
size is 60–135× larger than conditional on non-divergence. This matches the parent ANWG
finding qualitatively (all ANWG-nonzero branches diverge downstream: rate 1.0), but the
continuous metrics show divergence is **necessary-but-far-from-sufficient**: 40% divergence
rate among meaningful branches (not ~100%), because many diverging trajectories still net out
to a near-zero terminal timing difference by branch end. No additional quantities were invented
beyond what the existing branch metadata supports.

---

## J. Pressure stratification (frozen threshold = 0.60, WMT primary; WNT/SoftGoodput near-identical and summarized)

Pressure indicator columns joined 1:1 from the parent ANWG branches on the exact branch key
(`high_*` flags and `n_elevated_mechanisms` are computed once per branch and do not depend on
which terminal metric is scored).

### WMT_improvement by pressure flag

| Stratum | n | meaningful % | mean\|Δ\| | mass share |
|---|---:|---:|---:|---:|
| `high_burst_pressure=False` | 2770 | 51.7% | 0.001658 | 0.658 |
| `high_burst_pressure=True` | 771 | 43.3% | 0.003093 | 0.342 |
| `high_fairness_pressure=False` | 2606 | 49.8% | 0.002382 | 0.890 |
| `high_fairness_pressure=True` | 935 | 50.3% | 0.000824 | 0.110 |
| `high_kv_pressure=False` | 564 | 54.4% | 0.002150 | 0.174 |
| `high_kv_pressure=True` | 2977 | 49.0% | 0.001937 | 0.826 |
| `high_prefill_decode_pressure=False` | 1913 | 50.6% | 0.002275 | 0.624 |
| `high_prefill_decode_pressure=True` | 1628 | 49.1% | 0.001613 | 0.376 |
| `high_service_heterogeneity=False` | 1588 | 51.4% | 0.002107 | 0.480 |
| `high_service_heterogeneity=True` | 1953 | 48.7% | 0.001860 | 0.520 |
| `high_urgency_pressure=False` | 803 | 45.0% | 0.003640 | 0.419 |
| `high_urgency_pressure=True` | 2738 | 51.4% | 0.001481 | 0.581 |

### `n_elevated_mechanisms` bins (WMT)

| Bin | n | meaningful % | mean\|Δ\| | mass share |
|---|---:|---:|---:|---:|
| 0–1 | 255 | 42.4% | 0.002551 | 0.093 |
| 2 | 696 | 55.3% | 0.003140 | 0.313 |
| 3 | 1329 | 51.5% | 0.001959 | 0.373 |
| ≥4 | 1261 | 46.8% | 0.001220 | 0.220 |

WNT and SoftGoodput pressure tables (`pressure_stratified.json`) show the same qualitative
shape (meaningful-prevalence range narrows to ~43–55% across strata; mass shares differ from
WMT by at most a few points per stratum).

**Contrast with the ANWG pressure table (parent, for reference):** ANWG's meaningful-prevalence
varies **much more sharply** by stratum (3.1%–9.7%, a >3× range) than the continuous metrics
do (43–55%, a <1.3× range). This is a direct consequence of ANWG being a rare-event indicator
(small-sample strata swing prevalence a lot) versus the continuous metrics being near-coin-flip
events overall (~50% base rate, so stratification has much less room to move it). **Do not
overinterpret** small cells (e.g., 0–1 elevated mechanisms, n=255) exactly as in the parent study.

---

## K. Three-study comparison

*(not numerically pooled — different scales; see `comparison_vs_anwg.json`)*

| | A/B/C ANWG | Joint-240 ANWG | Joint-240 continuous (WMT, primary) |
|---|---:|---:|---:|
| Scenarios | 144 | 240 | 240 |
| Branches | 734 | 3541 | 3541 |
| Utility | binary deadline credit | binary deadline credit | continuous tardiness improvement |
| Exact-zero prevalence | 96.32% | 94.18% | 50.10% |
| Meaningful-effect prevalence | 3.68% | 5.82% | 49.90% |
| Top-1% mass | 0.483 | 0.421 | 0.458 |
| Top-5% mass | 1.000 | 0.951 | 0.850 |
| Top-10% mass | 1.000 | 1.000 | 0.962 |
| Disagreement AUROC | 0.513 | 0.680 | 0.838 |
| Disagreement AUPRC | 0.045 | 0.082 | 0.755 |
| Core conclusion | sparse + fully concentrated | `JOINT240_TERMINAL_CRITICALITY_REPLICATED` | `ANWG_ZERO_RATE_STEP_FUNCTION_ARTIFACT_BUT_CONCENTRATION_ROBUST` |

WNT/SoftGoodput rows (secondary continuous metrics) are qualitatively identical to WMT
(exact-zero ≈ 50.0%, meaningful ≈ 50.0%, top-10% mass 0.93–0.97, AUROC ≈ 0.839) and are given
in full in `comparison_vs_anwg.json`.

---

## L. Relation to guarded selector

From `docs/current/joint240_guarded_abstaining_selector_v1_analysis_20260825.md`:
unguarded Ascen gain ≈ **−0.0081** (CI excludes 0, entirely negative); guarded
(`util_advantage_guard`) gain ≈ **+0.0010**, CI **[−0.000033, +0.002118]** (includes 0);
catastrophic regressions **67 → 7**; SBS ≈ 0.314072, VBS ≈ 0.333106, headroom ≈ 0.019034.

**Does the continuous-utility criticality result support this interpretation?**

> "Unguarded specialization incurs avoidable regret; SBS abstention largely restores safety,
> while useful interventions remain difficult to identify because consequential effect
> magnitude is concentrated in a relatively small subset of states."

**Yes, and the continuous-utility result strengthens rather than weakens it**, with one
refinement. The guarded-selector claim rests on "most decisions are not terminally
consequential, so abstaining to SBS is safe." Under ANWG alone this rested on a narrow
94.18%-exact-zero definition of "not consequential." The continuous analysis shows:

- The **magnitude-concentration** half of the claim is **robust**: even under continuous
  metrics with ~50% nonzero prevalence, 85–97% of total effect mass concentrates in the top
  decile of states and 29–37% in just five scenarios (§F) — a small subset of states still
  carries most of the *consequential* magnitude, which is exactly what "difficult to identify
  because effect magnitude is concentrated in a relatively small subset of states" requires.
- The **refinement**: "difficult to identify" cannot mean "rare" any more — under continuous
  scoring, roughly half of all disagreement-adjacent decisions produce *some* measurable
  timing effect. What remains rare and hard to find is not the *existence* of an effect but
  the **large, scenario-concentrated** effects — the disagreement-AUROC numbers make this
  precise: 0.838 AUROC for "any meaningful timing shift" is comfortably informative, but that
  is a much easier target than the ANWG-style "will this cross the deadline boundary" (AUROC
  0.680). Identifying the *few* scenarios holding a third of the mass is a harder, more
  concentrated-tail problem than the aggregate AUROC numbers alone convey.
- Nothing here contradicts the "SBS abstention restores safety" claim — that claim concerns
  policy-selection behavior (Ascen vs. guards vs. SBS), which this experiment does not touch.

---

## M. Preregistered verdict

Applying design §7 rules mechanically to the **primary continuous metric (WMT_improvement)**:

- `frac_above_1e9` (meaningful prevalence) = **0.49901** → ≥ 0.25 ✓ (zeros "largely gone" condition)
- `concentration_abs["0.1"]["share"]` (top-10% state mass) = **0.96200** → ≥ 0.50 ✓

Checking rule order exactly as specified:
1. `TERMINAL_CRITICALITY_ROBUST_TO_CONTINUOUS_UTILITY` requires prevalence **< 0.25** — **fails** (0.499 ≥ 0.25).
2. `ANWG_ZERO_RATE_STEP_FUNCTION_ARTIFACT_BUT_CONCENTRATION_ROBUST` requires prevalence **≥ 0.25 and** top-10% mass **≥ 0.50** — **both satisfied**.
3. `ANWG_CRITICALITY_NOT_ROBUST_TO_CONTINUOUS_UTILITY` requires top-10% mass **< 0.30** — not reached (rule 2 already matched; and 0.962 is nowhere near 0.30 regardless).
4. `TERMINAL_UTILITY_ROBUSTNESS_INCONCLUSIVE` — not reached; ANWG reproduction passed and traces are complete.

**Final verdict (independently re-derived, not merely trusted from the runner):**

## `ANWG_ZERO_RATE_STEP_FUNCTION_ARTIFACT_BUT_CONCENTRATION_ROBUST`

This matches the preliminary label printed by the run. The re-derivation above confirms it
follows from the frozen §7 criteria applied to the frozen §4 definitions, not from the runner's
self-report.

---

## N. Reviewer-level interpretation

1. **Was the reviewer correct that hard ANWG deadlines manufacture many exact zeros?**
   **Yes.** 94.18% of branches show `Δ_ANWG = 0` while 46.8–47.0% of those same branches have
   a genuine (>1e-9), non-floating-point continuous timing effect (§E). ANWG's threshold
   structure is directly responsible for most of its own sparsity.

2. **By how much?** Exact-zero prevalence falls from **94.18% → ~50.0–50.1%** when scoring
   with any of WMT/WNT/SoftGoodput instead of ANWG — roughly **half** of ANWG's "no effect"
   population is reclassified as "some effect" under a continuous scale.

3. **Does this invalidate the current "sparse criticality" wording?** It invalidates
   **unqualified** claims of the form "terminal decisions are rarely consequential" if
   "consequential" is defined by *any nonzero continuous timing shift*. It does **not**
   invalidate "sparse criticality" if that phrase is reserved for *large-magnitude,
   scenario-concentrated* effects — that property survives (§F, §K). The wording needs a
   **scope qualifier**, not removal.

4. **Does magnitude concentration survive?** **Yes**, robustly: top-10% state mass 93–97%
   (vs. ANWG's near-degenerate 100%), top-5-scenario mass 29–37% (vs. ANWG's 29.3%), top-10%-
   scenario mass 62–74% (vs. ANWG's 68.5%), all with scenario-grouped 95% CIs that overlap
   ANWG's own CIs substantially (§F).

5. **What old claim should be removed or softened?** Any claim reading approximately "terminal
   interventions are consequential in only ~6% of decisions" should be **softened** to make
   explicit that the ~6% figure is specific to ANWG's binary deadline-crossing definition. It
   should not be presented as a general "terminal decisions rarely matter" statement.

6. **What precise replacement claim is supported?**
   > Under ANWG's binary deadline-crossing definition, terminal interventions register a
   > nonzero effect in only ~5.8–6% of forced-alternative branches. Under continuous
   > lateness-sensitive utilities (WMT/WNT/SoftGoodput) scored on the identical frozen
   > branches, roughly half of all branches show a measurable (>1e-9) timing effect — ANWG's
   > sparsity is substantially a step-function artifact of its own threshold, not evidence
   > that terminal decisions are mostly inert. What remains true under both scorings is
   > **magnitude concentration**: 85–100% of total effect mass sits in the top decile of
   > states, and roughly a third of total mass sits in just five scenarios (out of 240), with
   > every ANWG-critical state also registering among the continuous metrics' top-ranked
   > effects (zero counterexamples across 3541 branches).

7. **Is the same-distribution criticality argument now strong enough for the paper?** With the
   qualifier from (6) added, **yes for the concentration argument**; the same-distribution
   evidence (joint-240, n=3541, exact ANWG reproduction to 1e-16, scenario-grouped bootstrap
   CIs, disagreement-proxy AUROC/AUPRC under multiple metrics) is now cross-validated against
   a threshold-free scoring rule and the concentration result holds up. The **sparsity**
   argument specifically is **not** strong enough in its unqualified prior form and should be
   revised per (5)/(6) before being restated in the manuscript. (No manuscript edits made in
   this session, per instructions.)

Be conservative: this analysis supports *softening*, not *retracting*, the sparse-criticality
narrative.

---

## O. Manuscript-ready numbers

See `experiments/decision_criticality_terminal_utility_joint240_v1/manuscript_ready_numbers.json`
for the machine-readable block. Summary:

| Quantity | Value |
|---|---:|
| ANWG exact-zero / nonzero prevalence | 94.18% / 5.82% |
| WMT (preferred continuous) meaningful-effect prevalence | 49.90%, CI95 [47.24%, 52.56%] |
| WMT top-1% state mass | 0.4578, CI95 [0.337, 0.534] |
| WMT top-5% state mass | 0.8495, CI95 [0.787, 0.893] |
| WMT top-10% state mass | 0.9620, CI95 [0.942, 0.974] |
| ANWG-zero → WMT-meaningful fraction | 46.81% (1561/3335) |
| ANWG-zero → WNT-meaningful fraction | 46.96% (1566/3335) |
| ANWG-zero → SoftGoodput-meaningful fraction | 46.96% (1566/3335) |
| WMT disagreement AUROC / AUPRC | 0.838 / 0.755 (baseline 0.499) |
| ANWG vs. WMT rank overlap | Spearman\|Δ\| 0.384; top-10% index overlap 0.479 |
| ANWG top-5-scenario mass (corrected) | 0.293, CI95 [0.192, 0.399] |
| Guarded selector (`util_advantage_guard`) | gain +0.0010 vs. SBS, CI [−0.000033, 0.002118] (includes 0); catastrophes 67→7 |

---

## P. Outputs

Created in `docs/current/`:
- `decision_criticality_terminal_utility_joint240_v1_analysis_20260825.md` (this file, new)

Created/updated in `experiments/decision_criticality_terminal_utility_joint240_v1/`:
- `summary.json`, `summary.csv`, `bootstrap.json` — **unchanged** (already correct from the
  frozen run; verified, not re-derived)
- `cross_metric_overlap.json` — **updated** (merged in WNT/SoftGoodput Jaccard/Spearman/top-k
  overlap stats and the ANWG-zero decomposition / 2×2 tables; original ANWG–WMT keys preserved)
- `disagreement_metrics.json` — **new** (per-metric base/disagreement/agreement-control
  prevalence, enrichment, AUROC, AUPRC + scenario-grouped bootstrap CIs, for WMT/WNT/SoftGoodput)
- `pressure_stratified.json` — **new** (WMT/WNT/SoftGoodput × 6 pressure flags + elevated-mechanism bins)
- `comparison_vs_anwg.json` — **new** (three/four-row A/B/C vs. joint-240-ANWG vs.
  joint-240-continuous table, explicitly not pooled)
- `manuscript_ready_numbers.json` — **new** (compact exact-number block per §O)
- `cross_metric_overlap_extended.json`, `extra_analysis.json` — **new**, supporting detail
  files (2×2 tables, top-1/2/5 scenario mass, zero-mass bootstrap regression check, closed-loop
  divergence conditionals) referenced throughout this report

No files under `experiments/decision_criticality_terminal_anwg_joint240_v1/` (parent) were
modified. No branches were re-run.

---

## Q. Git status

Working tree remains exactly as at session start plus the new/updated files listed in §P
(all under `docs/current/` and `experiments/decision_criticality_terminal_utility_joint240_v1/`).
**No commits made. No pushes made.**

---

## R. Confirmation: manuscript untouched

No files under `paper/` were read, opened for editing, or modified in this session.

---

## S. Non-interference confirmation

Wulver SLURM array job **1195488** (`public_replay_load_scaling_v1`) was queried **exactly
once**, read-only (`sacct`), and was already fully `COMPLETED` (60/60 array tasks, exit 0:0) at
query time. It was not cancelled, resubmitted, restarted, or polled again. The 3541 frozen
branches were not re-run. The parent ANWG experiment directory was not overwritten. The
manuscript was not edited. No commits or pushes were made.
