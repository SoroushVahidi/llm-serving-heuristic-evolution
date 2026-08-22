# Decision-Criticality & Regime-Timescale Diagnostic v1 — TRAIN/VAL Scientific Analysis

Date: 2026-08-20
Analysis-only pass over the completed run at
`experiments/decision_criticality_timescale_trainval_v1/`. No code modified, no experiment
rerun, nothing committed/pushed. This document is the only repository file created or modified
by this analysis.

---

## 1. Executive verdict

The completed diagnostic (144/144 TRAIN/VAL scenarios, 0 failures) shows a real but **narrow and
family-concentrated** decision-criticality signal. Native-policy disagreement is rare overall
(0.05%–1.26% of evaluated steps depending on family), is **exactly zero** for
`PREFILL_DECODE_CONTENTION` (Family B, all 32 real BurstGPT-backed scenarios), and where it does
occur, only `RANKING_FAIRNESS` (Family A) shows a corroborated positive average headroom
(+0.886 completions/branch, bounded horizon). `KV_MEMORY_PRESSURE` (Family C) disagreement is
frequent and large in short-horizon state-divergence magnitude (88.4% of total severity mass)
but its bounded-horizon ceiling is **slightly negative on average** (-0.204), i.e. the
router-chosen policy already tends to be right — this matches, independently, the frozen TEST-side
finding in `docs/audits/hierarchical_regime_router_v1_20260818.md` that Family C's native-pair
oracle is degenerate (`kv_constrained_online` wins every TEST row). Timescales are not the
bottleneck for Families A/C (episodes long relative to `dwell=20`); Family B's episodes are
short enough to make reaction-time mismatch plausible, but the point is moot since B never
disagrees at all in this data. Mechanism attribution to observable router-input state
(`queue_length`, `kv_pressure`, `contention_score`, `priority_skew`) is **not possible** from the
retained causal-outputs artifact — it was structurally never logged there (by the design's own
router-input/causal-output separation, §4 of the design doc). **Classification:
`MIXED_SYNTHESIS_SIGNAL`. Recommended next step:
`RUN_PARTIAL_OBSERVABILITY_OR_CONTINUATION_DIAGNOSTIC`.**

---

## 2. Integrity summary

- File sizes: `decision_criticality_timescale_trainval_v1_results.json` = 14,407 B;
  `scenario_summaries.csv` = 22,182 B (144 rows × 11 cols); `disagreement_and_divergence_events.csv`
  = 149,915,714 B (885,875 rows × 12 cols).
- Scenario counts: observed = expected exactly. A=64, B=32, C=48, total=144
  (`trainval_scenario_counts` in results JSON matches `scenario_summaries.csv.mechanism_family`
  value_counts exactly). Split: 118 train / 26 val, **0 TEST rows** (`split` column has only
  `{train, val}` values — confirms §3's split guard held).
- No duplicate `canonical_scenario_id` (0/144). No duplicate `(scenario, step)` keys in the
  events CSV (0 among 876,839 base per-step rows). No null `canonical_scenario_id`, `step`,
  `regime`, or `chosen_policy_id` anywhere in the events file.
- Events-CSV row structure verified: 876,839 "base" rows (one per evaluated active-regime step,
  `disagree ∈ {True, False}`, `horizon = NaN`) + 9,036 "divergence" rows (2 per disagreement event
  × 4,518 events, `horizon ∈ {1.0, 10.0}`, `disagree = NaN` on these rows by construction — this is
  the file's actual two-row-family schema, not corruption).
- Cross-check: sum of `scenario_summaries.csv`'s per-family `effective_regime_count__*` columns =
  876,839, exactly matching the events-CSV base-row count. `n_disagreement_steps` sums to 4,518,
  matching `disagreement_rates.*.different_action_steps` summed across the three regimes in the
  results JSON (973 + 0 + 3,545). Independent CSV-derived aggregates reproduce the JSON's
  precomputed aggregates exactly — no discrepancy found.
- Family-B dependency guard: `family_b_primary_diagnostic.n_family_b_scenarios = 32`, computed
  "entirely from real TRAIN/VAL Family-B scenarios (BurstGPT-backed, `datasets_root`-resolved)"
  per the JSON's own note — matches §3's non-synthetic requirement.
- Frozen constants used (from `provenance`): `dwell_reference=20`, `horizon_h=10`,
  `full_trajectory_max_branches_per_scenario=3`, `full_trajectory_max_extra_steps=3000`.
  `git_head_sha=8e1223b...` (matches current HEAD), `git_tree_dirty=true` (expected — matches the
  session's known uncommitted files).
- `n_scenarios_failed=0`, `failures=[]`. No malformed/corrupt lines found.

**Structural integrity: clean.** No repair needed.

---

## 3. H-CRITICAL

Design doc's H-CRITICAL is operationalized in the implementation via `disagree` (canonical
action mismatch) and the four divergence metrics (`queue_length`, `active_count`,
`kv_utilization`, `completed_count` absolute diffs) at horizons 1 and 10 — there is **no
separate "consequential" flag or preregistered pass threshold** in either the design doc or the
implementation; "consequential" is operationalized here as nonzero divergence / nonzero
completed-count difference, reported at face value.

| Family (native pair) | Steps evaluated | Disagreement steps | Disagreement fraction | Scenarios w/ ≥1 disagreement |
|---|---|---|---|---|
| RANKING_FAIRNESS (A) | 796,415 | 3,545 | 0.445% | 44/64 (68.75%) |
| PREFILL_DECODE_CONTENTION (B) | 2,901 | 0 | 0.000% | 0/32 (0%) |
| KV_MEMORY_PRESSURE (C) | 77,523 | 973 | 1.255% | 41/48 (85.4%) |
| **Total** | 876,839 | 4,518 | 0.515% | 85/144 (59.0%) |

Immediate/short-horizon state divergence following a disagreement is **near-universal**
(`any_nonzero_divergence_rate = 1.0` at both horizon 1 and horizon 10, both families) — expected,
since a different admitted-request set trivially perturbs queue/active/KV state. The more
meaningful "consequence" signal — a difference in **completed-request count** within the H=10
window — is rare:

| Family | mean `completed_count_abs_diff` @H=10 | fraction of events with c10 > 0 | max c10 |
|---|---|---|---|
| RANKING_FAIRNESS | 0.0175 | 1.75% | 1 |
| KV_MEMORY_PRESSURE | 0.0000 | 0.00% | 0 |

So within a 10-step window, throughput consequence essentially never manifests for either family
— it is not visible at this horizon. Genuine consequence only appears in the bounded
full-trajectory branches (§5, up to 3,000 steps): mean **+0.886** completions/branch for A vs.
mean **-0.204** for C (signed `alt_minus_chosen`). Combined-magnitude "severity" (unweighted sum
of the four H=10 abs-diff metrics, an exploratory proxy only) is far larger per-event for C (mean
1.85) than A (mean 0.067), and C events account for 88.4% of total severity mass despite having
fewer events (973 vs. 3,545) — i.e. Family C disagreements are individually large-magnitude
short-horizon state perturbations that do **not** translate into positive long-run headroom on
average.

**Assessment**: consequential disagreement is not "ubiquitous tiny noise" (it is genuinely rare —
≤1.3% of steps) and not concentrated in a handful of scenarios only (59% of the 144 scenarios have
≥1 disagreement event; §6 gives full concentration figures) — but its downstream *value* is
markedly family-asymmetric: real for A, near-degenerate/slightly negative for C, entirely absent
for B.

---

## 4. H-TIMESCALE

Raw-activity episode-length distributions (`episode_timescales`, computed from
`a_active`/`b_active_v2`/`c_active`, independent of FSM/dwell):

| Family | n episodes | median | p10 | p90 | max | % > dwell(20) | % of *active steps* in episodes < dwell |
|---|---|---|---|---|---|---|---|
| A_active | 902 | 223 | 40.1 | 662 | 32,892 | 95.3% | 0.05% |
| B_active_v2 | 159 | 4 | 1.8 | 92.8 | 193 | 20.8% | 12.7% |
| C_active | 933 | 68 | 10.0 | 227.4 | 449 | 81.2% | 1.73% |

FSM-resolved (`effective_regime`) dwell-latency classification (`dwell_latency`):

| Family | episodes | fully_reactable | partially_reactable | unreachable_under_dwell20 |
|---|---|---|---|---|
| RANKING_FAIRNESS | 900 | 90.1% | 5.9% | 4.0% |
| KV_MEMORY_PRESSURE | 777 | 69.5% | 18.3% | 12.2% |
| PREFILL_DECODE_CONTENTION | 76 | 35.5% | 15.8% | **48.7%** |

**A.** Are consequential regimes mostly too short for a 20-step controller? — **Only for Family B.**
Its median episode length is 4 steps and 48.7% of its episodes end before the router could ever
exploit switch-eligibility. Families A and C are the opposite: median 223 and 68 steps
respectively, with 90.1%/69.5% fully reactable.

**B.** Are they long enough for mechanism-aware state control to be plausible? — Yes for A and C.
A's episodes in particular are an order of magnitude longer than `dwell=20`.

**C.** Could prior hierarchical-router failure plausibly be explained by reaction-time mismatch? —
**Plausible specifically for Family B's structural episode-shortness**, but this diagnostic finds
**zero action disagreement ever recorded for Family B** (§3) — so there is no disagreement event
for a faster-reacting controller to have exploited in the first place; the timescale-mismatch
story cannot be the operative explanation for *this* study's B result, because B's failure mode
(if any) is upstream of decision-criticality — the two native B candidates apparently never
diverge in chosen action across all 32 real TRAIN/VAL scenarios.

**D.** Or are episodes so long dwell time was unlikely the bottleneck? — For A specifically, yes:
median 223 steps is >11× the dwell floor, and 95.3% of A episodes clear it comfortably. Dwell
mismatch is a weak explanation for any A-related routing shortfall.

This is an explanatory hypothesis only, not a causal claim — no counterfactual "what if dwell
were shorter" branch was run (correctly, per the design doc's explicit prohibition on sweeping
`dwell`).

---

## 5. H-CEILING

Ceiling/headroom definition (design doc §5D/§10): the maximum and mean **bounded-horizon**
(≤3,000 extra steps or scenario-queue-drain, whichever first) signed difference in completed-request
count between continuing with the alternative native policy vs. the router's actually-chosen one,
sampled at the first 3 disagreement steps per scenario per family (not outcome-selected).

| Family | branches attempted | mean (alt − chosen) | max (alt − chosen) | fraction differing action (ceiling exists at all) |
|---|---|---|---|---|
| RANKING_FAIRNESS | 132 | **+0.886** | +6 | 0.445% |
| KV_MEMORY_PRESSURE | 98 | **-0.204** | +1 | 1.255% |
| PREFILL_DECODE_CONTENTION | 0 | n/a (no disagreement ever) | n/a | 0.000% |

**Canonical repo cross-reference** (`docs/audits/hierarchical_regime_router_v1_20260818.md`,
§4 "Stage-2 TEST metrics", frozen TEST-split audit, independent of this TRAIN/VAL study):
RANKING_FAIRNESS standalone Stage-2 gain (oracle routing, n=8 TEST) = **+0.0302** ANWG (nonzero,
positive); KV_MEMORY_PRESSURE standalone gain (n=24 TEST) = **0.0000**, explicitly because
"the native-pair oracle and the best-fixed policy coincide exactly row-for-row
(`kv_constrained_online` wins every time)". This TRAIN/VAL diagnostic's finding — Family A has
real positive average headroom, Family C's native pair is near-degenerate with the router's
`kv_constrained_online` choice already dominant — **independently corroborates** that frozen
TEST-side result via a completely different (counterfactual-forking, TRAIN/VAL) methodology. No
other directly comparable canonical portfolio-envelope number for this exact native-pair
comparison was found in repo docs; the overall system-level ΔANWG=0.0000 (D−A) and oracle-gap-closure
=0.0000 from the same audit are end-to-end router numbers, not native-pair ceiling numbers, so are
cited as context only, not as a like-for-like comparison.

**Answer**: yes, there is nonzero theoretical local decision headroom — but it exists almost
entirely in Family A (RANKING_FAIRNESS), is modest in magnitude (mean <1 completion per branch,
sampled from only the first 3 disagreement events per scenario), and is corroborated by an
independent prior TEST-split audit. Family C offers essentially no ceiling on average despite
producing the most frequent and largest-magnitude short-horizon divergence. Family B offers no
ceiling because it has no disagreement to begin with.

---

## 6. Concentration

Per-scenario "severity" (unweighted sum of the four H=10 abs-diff metrics; exploratory magnitude
proxy, not a validated metric) across the 85 scenarios with ≥1 disagreement:

| Top-k scenarios (of 85 with disagreement) | share of total severity mass |
|---|---|
| top 1% (1 scenario) | 15.0% |
| top 5% (5 scenarios) | 51.0% |
| top 10% (9 scenarios) | 70.6% |
| top 20% (17 scenarios) | 86.4% |

5 scenarios (out of 144 total, 85 with any disagreement) account for 50% of total severity mass.
By raw **event count** (rather than severity), concentration is markedly less extreme: top 5% of
scenarios = 14.3% of events, top 20% = 44.8% — a broader base of scenarios each contributing a
modest number of disagreement events, with severity dominated by a smaller subset of
(overwhelmingly Family-C) scenarios producing large-magnitude but low-value events. Family
contribution to total severity mass: **KV_MEMORY_PRESSURE = 88.4%, RANKING_FAIRNESS = 11.6%**
(PREFILL_DECODE_CONTENTION = 0%, no events). One family (C) dominates the raw magnitude signal,
but — per §5 — not the family that carries genuine positive ceiling value; that is the reverse
of what raw severity alone would suggest, which is itself a notable finding: **magnitude of
short-horizon state divergence is not a reliable proxy for long-run consequence value.**

---

## 7. Mechanism/feature associations

**Available fields in the retained causal-outputs artifact**: `regime` (family), `step`
(position in scenario), `chosen_policy_id`, `alt_policy_id`, `disagree`, and the four divergence
magnitudes at horizons 1/10. **Not available**: any router-input observable state at the
disagreement step itself (`queue_length`, `kv_pressure`, `contention_score`, `priority_skew`,
raw KV/active/queue values) — these are structurally excluded from the causal-evaluation-output
artifact by the design's explicit router-input/causal-output separation (§4 of the design doc:
"nothing this diagnostic computes is ever fed back into Stage-1... never imports itself into
`hierarchical_regime_router_v1.py`"). That separation was built to prevent the diagnostic from
leaking into the router, but as a side effect it also means the retained CSV **cannot support a
genuine feature-driven mechanism model** linking disagreement/consequence to the router's own
input signals — a real, structural limitation of this artifact, not a shortcut taken in this
analysis. No decision tree / logistic model was fit for this reason: with only `regime` and
`step`-position as usable covariates, any such model would be a near-tautological re-derivation
of the family split already reported above, not genuine mechanism attribution.

What the two available covariates do show:
- **Family/regime** is overwhelmingly the strongest available "feature": it alone predicts
  severity magnitude (C ≫ A, §6), ceiling sign (A positive, C negative, §5), and even whether
  disagreement occurs at all (B: never).
- **Relative position within scenario** (`step / n_steps`) differs by family: KV_MEMORY_PRESSURE
  disagreements cluster early (mean relative position 0.204, median 0.195, max 0.417) — consistent
  with KV-pressure buildup being front-loaded in these scenario constructions — while
  RANKING_FAIRNESS disagreements are spread across the full scenario duration (mean 0.458, median
  0.438, IQR 0.26–0.66).

**Conclusion**: consequential disagreement is predictable at the *family* level (a compact,
1-variable "mechanism") but not below that from what this artifact retains. A genuine
sub-family mechanism account (e.g. "high `kv_pressure` + high `queue_length` predicts negative
ceiling") would require re-deriving router-input state at each logged disagreement step from the
raw trajectory/telemetry — out of scope for a read-only analysis pass and not attempted here.

---

## 8. Temporal structure

Contiguous-step disagreement "bursts" (runs of consecutive disagreement steps within a scenario):
**3,613 bursts total, 99.14% of length 1** (isolated single-step spikes). Median gap between
successive disagreement steps within a scenario (where >1 exist) is 169 steps (mean 270, max
3,103) — i.e., disagreement events are **sparse, scattered, isolated spikes superimposed on**
the much longer regime-activity episodes described in §4, not sustained multi-step "critical
windows" a controller could latch onto and dwell within. This matters directly for synthesis
feasibility: even where an episode is long enough (§4) for a `dwell=20` controller to react, the
underlying disagreement signal within that episode is almost always a single transient step, not
an extended critical regime — weakening (though not eliminating) the case that a state-dwelling
controller would have many actionable within-episode reaction opportunities. No systematic
clustering near overload buildup or KV-pressure peaks could be tested directly (no raw KV/queue
state retained per §7), beyond the coarse early-vs-spread-out family-level positional pattern
already reported in §7.

---

## 9. Policy-direction analysis

| Family | chosen policy at disagreement steps | alt policy | split |
|---|---|---|---|
| KV_MEMORY_PRESSURE | `kv_constrained_online` | `least_laxity_first` | **100% / 0%** (973/973 vs. 0/973) |
| RANKING_FAIRNESS | `weighted_fair_share` (2,177) / `estimated_service_time_first` (1,368) | (reverse) | **61.4% / 38.6%** |
| PREFILL_DECODE_CONTENTION | n/a | n/a | no disagreement events |

For KV_MEMORY_PRESSURE, `kv_constrained_online` is the router's chosen policy at **every single**
disagreement event in TRAIN/VAL — `least_laxity_first` is never the live pick when the two
disagree. Combined with the near-zero/negative average ceiling (§5) and the independent TEST-side
finding that `kv_constrained_online` "wins every time" on the oracle comparison, this is strong,
convergent evidence of **one-sided dominance**, not complementary mechanisms, for this family.

For RANKING_FAIRNESS, the router's chosen policy genuinely varies (61.4%/38.6% split) across
disagreement events, and the bounded-horizon ceiling is positive on average — consistent with
**state-dependent, non-one-sided** advantage between the two candidates, the more promising
pattern for synthesis. This sign-variation was not further decomposed by scenario/state because
the retained artifact does not carry the state features needed to explain *when* each side wins
(§7).

---

## 10. Synthesis-feasibility classification: `MIXED_SYNTHESIS_SIGNAL`

Justification:
- **Supports STRONG**: a real, nonzero, cross-validated (independent TEST-audit-corroborated)
  positive-headroom signal exists in Family A; that family's episode timescales comfortably
  clear the dwell floor; the router's chosen policy genuinely varies at Family-A disagreement
  points (not one-sided), consistent with complementary mechanisms.
- **Weighs against STRONG, toward MIXED**: two of three families offer no synthesis
  opportunity — Family B has zero disagreement ever recorded, Family C's large, frequent
  short-horizon divergence does not convert into positive average ceiling (it's marginally
  negative — the current policy is already usually right). The overall disagreement rate is tiny
  (≤1.3% of steps in any family). 99.1% of disagreement events are isolated single-step spikes,
  not extended critical episodes, weakening the "state-dwelling controller" story even where
  timescales are otherwise favorable. Mechanism attribution to observable state — the
  "mechanistically interpretable" leg of the central scientific question — is **not achievable**
  from the retained artifact; only the coarse family-level association is available.
- **Not WEAK**: the Family-A signal is not tiny/noise (mean +0.886, max +6 completions per
  branch, corroborated independently), is not confined to a pathological handful of scenarios
  (44/64 Family-A scenarios show ≥1 disagreement), and shows genuine bidirectional policy
  preference rather than one policy always winning.

## 11. Recommended next direction: `RUN_PARTIAL_OBSERVABILITY_OR_CONTINUATION_DIAGNOSTIC`

Both of this option's stated trigger conditions are met: (a) local consequence exists (Family A)
but is **not well explained by currently observable state** — the retained artifact structurally
lacks the router-input features needed for mechanism attribution (§7); and (b) **continuation
dependence appears important** — 99.1% of disagreement events are isolated single-step spikes
(§8) embedded in much longer regime episodes, so understanding whether/how a single-step
disagreement's consequence depends on what happens in the surrounding episode (rather than being
assessable as an independent point event) is an open, load-bearing question before committing to
a designed child policy. `DESIGN_FIRST_DISAGREEMENT_GUIDED_CHILD` would be premature: it presumes
a mechanism account this study does not yet provide. `ABANDON_DISAGREEMENT_GUIDED_SYNTHESIS` would
discard the one genuine, corroborated signal (Family A) without cause. `NEED_RESULT_INTEGRITY_REPAIR`
does not apply — the data is structurally clean (§2).

---

## 12. Limitations

- "Consequential disagreement" has no preregistered numeric threshold in the design doc or
  implementation; this analysis reports raw divergence/consequence magnitudes rather than a
  binary pass/fail, by design (the design doc explicitly avoids inventing an "importance score").
- The bounded full-trajectory ceiling (§5) samples only the **first 3** disagreement steps per
  scenario per family (132 of 3,545 Family-A disagreement events; 98 of 973 Family-C events) — a
  fixed, outcome-blind, preregistered rule, but it means the §5 mean/max figures are not a
  full-population estimate over all 4,518 disagreement events, only over this bounded sample.
  Extrapolating the mean headroom to the full disagreement population was not attempted.
  `bounded_horizon` results (≤3,000 extra steps) are explicitly not full-scenario-completion
  oracle values.
  the horizon-10 "any_nonzero_divergence_rate ≈ 1.0" figures are near-tautological (a different
  admitted-request set almost always perturbs at least one of queue/active/KV state immediately)
  and were not treated as meaningful consequence evidence on their own — completed-count
  divergence and the bounded full-trajectory branches were used instead as the substantive
  consequence signals.
- No raw router-input state (queue_length, kv_pressure, contention_score, priority_skew) is
  retained at disagreement steps in the causal-outputs artifact (§7); mechanism attribution below
  the family level is not possible from this run's outputs alone.
- Family C's apparent relative-position clustering near scenario start (§7) is descriptive only;
  no causal claim about "buildup phase" is made, since the underlying KV-pressure trajectory
  itself was not re-derived here.
- One relative-position value for RANKING_FAIRNESS events exceeded 1.0 (max 1.113) when computed
  as `step / n_steps` against `scenario_summaries.csv`'s `n_steps` column — a minor step-indexing
  or n_steps-accounting artifact between the two output files that does not affect any of the
  quantitative results reported here (it affects at most the descriptive relative-position stat,
  not counts, rates, or ceiling values) and was not investigated further under the read-only
  analysis-only constraint.
- This is a TRAIN/VAL-only diagnostic; per its own design and per §13, it says nothing about
  TEST-split or held-out Family-B replication performance, and none of that data was read here.

---

## 13. Novelty-aware interpretation

Relative to closest prior methods (GPI/successor features, MAMBA multi-expert improvement, VIPER
critical-state extraction, BCOL counterfactual decisions, scheduling hyper-heuristics), none of
the individual ingredients used here — local action-advantage estimation, critical-state
identification, cross-policy stitching, multi-expert improvement, or decision-tree-style
interpretability — would themselves be a novel contribution. The chain this research line is
actually pursuing is: **policy disagreement → downstream consequence → mechanism attribution →
standalone interpretable scheduling rule → positive marginal portfolio contribution.**

This experiment establishes, with real (not tautological) evidence, the first link
(disagreement occurs, non-trivially, for 2 of 3 families) and partially establishes the second
(downstream consequence: real and corroborated for Family A; near-degenerate for Family C; absent
for Family B). It does **not** establish the third link (mechanism attribution to observable
state) — that is structurally blocked by what this artifact retained (§7) — and therefore cannot
yet speak to the fourth or fifth links (a standalone rule, or its marginal portfolio value). The
completed experiment supports moving to the *next* link in the chain (mechanism/observability),
not skipping ahead to rule synthesis; §10/§11's `MIXED_SYNTHESIS_SIGNAL` /
`RUN_PARTIAL_OBSERVABILITY_OR_CONTINUATION_DIAGNOSTIC` verdict reflects exactly this state.

---

## 14. Public-trace context (not analyzed here)

Layer 3 of `public_trace_replay_v1` completed 480/480 cells with integrity `ok=true`; Layer 4
completed 480/480 trajectories. These are available for later external validation but were not
read or analyzed beyond this note, and are not mixed into any TRAIN/VAL conclusion above.

---

## 15. Exact artifact paths and reproducible analysis commands

Source artifacts read (unmodified):
- `docs/design/DECISION_CRITICALITY_TIMESCALE_TRAINVAL_V1.md`
- `experiments/decision_criticality_timescale_trainval_v1/decision_criticality_timescale_trainval_v1_results.json`
- `experiments/decision_criticality_timescale_trainval_v1/scenario_summaries.csv`
- `experiments/decision_criticality_timescale_trainval_v1/disagreement_and_divergence_events.csv`
- `docs/audits/hierarchical_regime_router_v1_20260818.md` (canonical TEST-side cross-reference, §5)

Reproducible commands (read-only; all figures in this report were derived this way):

```bash
# Top-level results (verbatim JSON fields)
python3 -c "import json; print(json.dumps(json.load(open('experiments/decision_criticality_timescale_trainval_v1/decision_criticality_timescale_trainval_v1_results.json')), indent=2))"

# Scenario-level integrity + per-family disagreement summary
python3 - <<'PY'
import pandas as pd
df = pd.read_csv('experiments/decision_criticality_timescale_trainval_v1/scenario_summaries.csv')
print(df['split'].value_counts())
print(df.groupby('mechanism_family')['n_disagreement_steps'].agg(['count','sum','mean','median','max']))
print(df.assign(has_dis=df['n_disagreement_steps']>0).groupby('mechanism_family')['has_dis'].mean())
PY

# Event-level H-CRITICAL / H-CEILING / concentration / temporal / policy-direction analysis
python3 - <<'PY'
import pandas as pd, numpy as np
df = pd.read_csv('experiments/decision_criticality_timescale_trainval_v1/disagreement_and_divergence_events.csv')
base = df[df['disagree'].notna()].copy(); base['disagree'] = base['disagree'].astype(bool)
h10 = df[df['horizon']==10.0].reset_index(drop=True)
h1  = df[df['horizon']==1.0].reset_index(drop=True)
dis = base[base['disagree']==True][['canonical_scenario_id','step','regime','chosen_policy_id','alt_policy_id']].reset_index(drop=True)
ev = dis.copy()
ev['c10'] = h10['completed_count_abs_diff']; ev['severity10'] = h10[['queue_length_abs_diff','active_count_abs_diff','kv_utilization_abs_diff','completed_count_abs_diff']].fillna(0).sum(axis=1)
print(ev.groupby('regime')['severity10'].describe())
print(ev.groupby('regime').apply(lambda g: (g['c10']>0).mean()))
print(ev.groupby('regime')[['chosen_policy_id']].value_counts())
scen = ev.groupby('canonical_scenario_id')['severity10'].sum().sort_values(ascending=False)
print((scen.iloc[:5].sum()/scen.sum()))  # top-5-scenario severity share
# temporal burst analysis
dis_sorted = dis.sort_values(['canonical_scenario_id','step'])
dis_sorted['gap'] = dis_sorted.groupby('canonical_scenario_id')['step'].diff()
dis_sorted['newburst'] = (dis_sorted['gap']!=1) | (dis_sorted['gap'].isna())
dis_sorted['burst_id'] = dis_sorted.groupby('canonical_scenario_id')['newburst'].cumsum()
burst_sizes = dis_sorted.groupby(['canonical_scenario_id','burst_id']).size()
print((burst_sizes==1).mean())  # fraction isolated single-step spikes
PY
```
