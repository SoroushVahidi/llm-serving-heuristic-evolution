# Family-A Contested-Request Value Diagnosis

Date: 2026-08-21

Diagnostic-only pass over the completed `family_a_contested_request_value_diagnosis`
extraction (`experiments/family_a_contested_request_value_diagnosis/`). No new
scientific simulation was run, no controller/policy/simulator code was
touched, no rollout horizon was changed, no TEST data was read, no
DAgger/RL was run, and nothing was staged/committed/pushed. All numbers
below come from one deterministic, read-only analysis script
(`scripts/analyze_family_a_contested_request_value_diagnosis.py`) over the
already-extracted CSVs plus the pre-existing 91-event repaired artifact.

---

## 1. Executive Verdict

Classification: **`CONTESTED_VALUE_SIGNAL_PARTIAL`**

Next step: **`KEEP_AS_DIAGNOSTIC_ONLY`**

The extraction resolves the disagreement structure with unusual cleanliness:
every one of the 91 disagreement events contests **exactly one** ESTF-only
request against **exactly one** WFS-only request (no common admissions, no
asymmetric set sizes). Those two specific requests are drawn from
systematically different populations (WFS-only requests are longer,
higher-priority, and further past naive feasibility; Cohen's d for
`predicted_service_proxy` is −3.4 in `favlong` alone), and their eventual
fates reveal a clean, causally-interpretable, regime-consistent mechanism
split: **ESTF's contested value is completion protection** (its admitted
request completes ~100% of the time under its own continuation vs. 56–65%
under the other policy's), while **WFS's contested value is SLO protection**
(its admitted request's post-completion SLO-success rate is 78–86% under its
own continuation vs. 45–65% under ESTF's). The contested pair also explains
a materially large share of each event's whole-branch raw-completed-count
delta (median 33%, mean 47%, p90 = 100%).

However, when this causal information is turned into simple, coefficient-free
diagnostic proxies (Section 9), none of them materially or reliably improves
alignment with the native-continuation outcome beyond what a trivial
regime-conditioned or majority-class guess already achieves within each
`favlong`/`favshort` stratum (Section 10–11, Section 14). The signal is real
and concentrated, but not yet reduced to a working, validated formula — this
is exactly the `PARTIAL` definition ("some consistent evidence exists, but
explanation/alignment is incomplete"), not `STRONG` (which requires the
causal proxy to *demonstrably* improve alignment) and not `NO_GO` (which
requires the contested requests to not explain much, which is false here).

---

## 2. Integrity (independent reconfirmation)

| Check | Result |
|---|---|
| events rows | 91 |
| contested-request rows | 182 |
| event-key set matches existing 91-event artifact exactly | **true** |
| duplicate `event_id` | 0 |
| duplicate `(event_id, contested_side, request_id)` | 0 |
| null identity fields (`event_id`/`canonical_scenario_id`/`split`/`step`/`contested_side`/`request_id`) | 0 |
| all 16 expected `br_{estf_estf,wfs_wfs,wfs_estf,estf_wfs}_{completed,completion_time,slo_violated,weighted_contribution}` columns present | **true** |
| split values | `{train, val}` only, both files |
| TEST rows | 0 / 0 |

No integrity failure. `CONTESTED_VALUE_DIAGNOSIS_BLOCKED_BY_INTEGRITY` does
not apply.

---

## 3. Contested-Set Semantics

`n_estf_only` and `n_wfs_only` are **exactly 1** in all 91/91 events;
`n_common` is **exactly 0** in all 91/91 events. There are **zero**
events with asymmetric set size. This extraction therefore yields a strict
**one ESTF-only vs. one WFS-only request per event** structure, not a
general multi-request set — the "specific contested request" the task asks
about is, in this data, always a single well-defined pair per event (91 ESTF-only
+ 91 WFS-only + 0 common = 182 contested rows, matching the file exactly).

---

## 4. ESTF-Only vs. WFS-Only Request Properties

Mean (Cohen's d, ESTF-only minus WFS-only) over causal pre-decision features,
`ALL` / `favlong` / `favshort`:

| Feature | ALL: ESTF-only vs WFS-only mean | ALL d | favlong: ESTF-only vs WFS-only mean | favlong d | favshort: ESTF-only vs WFS-only mean | favshort d |
|---|---:|---:|---:|---:|---:|---:|
| `priority`/`weight` | 2.75 vs 4.96 | −0.70 | **1.00 vs 7.00** | **−3.44** | 6.13 vs 1.00 | **+3.41** |
| `prompt_tokens` | 288 vs 907 | −2.21 | 257 vs 959 | −2.28 | 348 vs 805 | −2.38 |
| `predicted_output_tokens` | 117 vs 407 | −2.55 | 99 vs 450 | −3.04 | 152 vs 323 | −2.17 |
| `predicted_service_proxy` | 261 vs 860 | −3.41 | 228 vs 930 | **−3.87** | 326 vs 726 | −3.93 |
| `queue_age` | 0.41 vs 0.46 | −0.11 | 0.46 vs 0.34 | +0.34 | 0.32 vs 0.70 | −0.65 |
| `slo_deadline` | 10.31 vs 8.03 | +0.38 | 9.42 vs 2.54 | +5.73 | 12.03 vs 18.66 | −1.69 |
| `deadline_slack_if_admitted_now` | −256 vs −857 | +3.40 | −220 vs −929 | +3.91 | −326 vs −718 | +3.86 |
| `feasible_if_admitted_now` (fraction True) | 0.0 vs 0.0 | — | 0.0 vs 0.0 | — | 0.0 vs 0.0 | — |

Two findings stand out:

1. **`priority` cleanly and near-perfectly tracks the split in each regime**:
   in `favlong`, every single ESTF-only request has `priority=1.0` (std=0)
   and every WFS-only request has `priority≥5` (mean 7.0); in `favshort` the
   pattern flips (ESTF-only mean 6.1, WFS-only exactly 1.0, std=0). WFS-only
   requests are also always the larger/longer ones (`prompt_tokens`,
   `predicted_output_tokens`, `predicted_service_proxy` all show |d| > 2 in
   every stratum). This matches the repaired-analysis report's finding
   (`docs/current/family_a_observability_continuation_v1_repaired_analysis_20260820.md`
   §15) that service/request-size distribution dominates the mechanism
   signal, with fairness/priority secondary but present.
2. **`feasible_if_admitted_now` is `False` for literally every one of the
   182 contested rows, in every regime.** By the time ESTF and WFS disagree
   at these sampled events, both candidates are already past the strict
   `deadline_slack(now, full_service_proxy) >= 0` feasibility bar if newly
   admitted from a clean slate — a genuinely queue-saturated regime, not a
   marginal-feasibility one. This degenerate value has direct consequences
   for the diagnostic proxies built in Section 9.

`queue_age` (available starvation/age proxy) is *not* dramatically larger for
WFS-only requests overall (d=−0.11 ALL; even slightly *smaller* in `favlong`,
d=+0.34) — the WFS-only advantage is driven by priority and size, not
raw waiting time. No true per-request fairness/class-deficit field exists in
this extraction (only `class_id`, no deficit ratio attached per row); this is
a documented limitation (§18).

---

## 5. Eventual Branch Outcomes

Per contested side, per branch (all regimes pooled, n=91 per cell):

| Side | Branch | Completion prob. | SLO success \| completed | Mean weighted contribution |
|---|---|---:|---:|---:|
| ESTF-only | `br_estf_estf` (own native) | **1.000** | 0.989 | 2.692 |
| ESTF-only | `br_wfs_wfs` (other's native) | 0.560 | 0.882 | 1.275 |
| ESTF-only | `br_wfs_estf` (WFS admits, ESTF continues) | 1.000 | 0.923 | 2.198 |
| ESTF-only | `br_estf_wfs` (ESTF admits, WFS continues) | **1.000** | 0.989 | 2.692 |
| WFS-only | `br_wfs_wfs` (own native) | **1.000** | 0.857 | 4.022 |
| WFS-only | `br_estf_estf` (other's native) | 0.934 | 0.647 | 2.253 |
| WFS-only | `br_wfs_estf` (WFS admits, ESTF continues) | **1.000** | 0.857 | 4.022 |
| WFS-only | `br_estf_wfs` (ESTF admits, WFS continues) | 0.978 | 0.809 | 3.571 |

**A structural finding that reframes Sections 7–8**: `br_wfs_wfs` and
`br_wfs_estf` are **bit-identical** on every outcome field for every one of
the 91 WFS-only rows (and likewise `br_estf_estf`≡`br_estf_wfs` for every
ESTF-only row). This is not a bug — Family-A GPUs have
`max_active_sequences=1` and the simulator has no preemption, so once a
specific request is admitted onto its slot, **its own eventual fate is
already sealed**; which policy continues scheduling *other* requests
afterward cannot change it. The real, non-degenerate lever is not "which
policy continues" but **"which policy admits it, and how soon"** — isolated
by comparing a request's own native branch against the *other* policy's
native branch, used throughout Sections 6–8 below.

**Cross-branch rescue**: ESTF-only requests complete only 56.0% of the time
if left to WFS's own native path (44% never finish within the bounded
1500-step window) — completion itself is the bottleneck. WFS-only requests
complete 93.4% of the time even under ESTF's own native path, but their SLO
success rate given completion collapses from 85.7% (own/WFS native) to
64.7% (ESTF native) — **timeliness**, not raw completion, is the
bottleneck. This is the clearest single result in this diagnosis: **the two
policies protect contested requests along two different outcome axes.**

---

## 6. Value Concentration

Definition (documented precisely so the fraction is reproducible):

```
contested_raw_delta(event) =
    sum_{i in estf_only} [1(completed under br_estf_estf) - 1(completed under br_wfs_wfs)]
  + sum_{i in wfs_only}  [1(completed under br_wfs_wfs)  - 1(completed under br_estf_estf)]

delta_native_whole_branch_raw(event) = br_estf_estf_completed - br_wfs_wfs_completed
    (whole-branch total raw completed-request COUNT, joined by (canonical_scenario_id, step)
     from the pre-existing 91-event artifact -- the only whole-branch signal available;
     no whole-branch WEIGHTED/SLO total exists in either artifact)

explained_fraction = contested_raw_delta / delta_native_whole_branch_raw
    (defined only where the denominator is nonzero)
```

| Quantity | Value |
|---|---:|
| events with nonzero denominator | 59/91 |
| events with zero denominator | 32/91 (of which 4 still have nonzero `contested_raw_delta` — see note) |
| `explained_fraction` mean | 0.469 |
| `explained_fraction` median | **0.333** |
| `explained_fraction` p75 | 1.00 |
| `explained_fraction` p90 | **1.00** |
| fraction of events with \|explained_fraction\| ≥ 0.5 | 50.8% |
| fraction of events with \|explained_fraction\| ≥ 1.0 | 33.9% |
| top single event's share of total \|weighted contested delta\| | 3.8% |
| top single request's share of total \|weighted contested diff\| | 3.4% |

Zero-denominator handling (explicit, not dropped): 32/91 events have
`delta_native_whole_branch_raw == 0` (the whole branch produced the same
raw completed-request count under both native continuations). Of those, 4
events still have a *nonzero* `contested_raw_delta` — meaning the contested
pair's own outcome differs even though it is offset by an equal-and-opposite
change elsewhere in the branch; these 4 are reported separately, not folded
into `explained_fraction`, and not silently discarded.

**Interpretation**: value is not concentrated in a handful of outlier
events or requests (top event/request share ≈ 3–4%, i.e. broadly spread) —
but *within* a typical event, the two contested requests alone account for
a median 33% / mean 47% of that event's entire whole-branch raw-completion
delta, and in a full third of nonzero-denominator events they account for
**100% or more** (the rest of the branch's completions net out to zero
difference). This is real, non-trivial concentration, though bounded to the
raw-completed-count unit — no whole-branch weighted/SLO denominator exists
in either extraction artifact to compute a weighted version of this
fraction (documented limitation, §18).

---

## 7. WFS-Protection Hypothesis Test (`favlong`)

Central hypothesis: in `favlong`, WFS wins by protecting a specific
WFS-only contested request with high downstream value.

Using `admission_native_effect = weighted_contribution(br_wfs_wfs) −
weighted_contribution(br_estf_estf)` (own-native minus other-native, per
§5's reframing; continuation-only effect is verified degenerate — identically
zero for all 60 rows):

| Quantity | Value |
|---|---:|
| n (WFS-only, favlong) | 60 |
| `admission_native_effect` mean | +2.667 |
| `admission_native_effect` median | 0.0 |
| fraction positive | 36.7% |
| fraction negative | **0.0%** |
| completion prob. own-native (`br_wfs_wfs`) | 100.0% |
| completion prob. other-native (`br_estf_estf`) | 91.7% |
| **SLO success rate given completed, own-native** | **78.3%** |
| **SLO success rate given completed, other-native** | **45.5%** |

**Answer: yes, there is a clean, observable, causal protection signature —
but it is an SLO-timeliness signature, not a raw-completion signature.**
`admission_native_effect` is never negative (WFS-only requests are never
worse off under WFS's own path) and the SLO-success gap (78.3% vs. 45.5%,
+33pp) is large and one-directional. Correlations of `admission_native_effect`
with causal pre-decision features (Spearman, favlong WFS-only, n=60):

| Feature | ρ | p |
|---|---:|---:|
| `priority` | +0.220 | 0.091 |
| `queue_age` | **−0.306** | **0.017** |
| `deadline_slack_if_admitted_now` | −0.137 | 0.298 |
| `predicted_service_proxy` | +0.143 | 0.275 |

Only `queue_age` reaches significance, and its sign is the **opposite** of
the naive "protects the most-starved request" story (younger-arrived
WFS-only requests show a *larger* protection effect). `priority` trends in
the expected direction but does not reach significance at n=60. There is
therefore a real protection mechanism, but it is not cleanly reducible to
"protect whichever request is oldest/most starved" — a nuance any future
formula must not paper over.

---

## 8. ESTF-Useful Hypothesis Test (`favshort`)

Same reframing (continuation-only effect verified degenerate for all 31
favshort ESTF-only rows): `admission_native_effect = weighted_contribution
(br_estf_estf) − weighted_contribution(br_wfs_wfs)`.

| Quantity | Value |
|---|---:|
| n (ESTF-only, favshort) | 31 |
| `admission_native_effect` mean | +3.226 |
| `admission_native_effect` median | +5.0 |
| fraction positive | 51.6% |
| completion prob. own-native (`br_estf_estf`) | 100.0% |
| completion prob. other-native (`br_wfs_wfs`) | **64.5%** |
| feasible-if-admitted-now fraction | 0.0% |

**Answer: yes, ESTF's benefit in `favshort` is real and completion-driven,
not a fairness-insensitivity story.** ESTF-only requests complete 100% of
the time on their own native path but only 64.5% of the time if left to
WFS's — over a third never finish within the bounded window. Correlations
(Spearman, favshort ESTF-only, n=31):

| Feature | ρ | p |
|---|---:|---:|
| `priority` | +0.284 | 0.121 |
| `queue_age` | −0.064 | 0.734 |
| `deadline_slack_if_admitted_now` | −0.347 | 0.056 |
| `predicted_service_proxy` | +0.344 | 0.058 |
| `prompt_tokens` | +0.006 | 0.976 |
| `predicted_output_tokens` | **+0.355** | **0.049** |

`predicted_output_tokens` is (marginally) significant and **positive** —
larger, not smaller, ESTF-only jobs benefit more from ESTF's prompt
admission, the opposite of a pure "ESTF only helps trivially-short jobs"
story. `deadline_slack`/`predicted_service_proxy` trend the same direction
(more urgent / more service-hungry jobs benefit more), consistent with a
single mechanism: **ESTF's contested value in `favshort` is rescuing jobs
that would otherwise plausibly time out under WFS's admission ordering**,
not merely skimming trivially cheap work. This does **not** fully explain
why ESTF is useful in `favshort` and WFS in `favlong` from the *same* small
feature space with a single shared rule — the two regimes' contested
populations differ almost categorically on `priority` (favshort ESTF-only
mean priority 6.1 vs. favlong ESTF-only priority fixed at 1.0), so any
unified proxy must use priority/weight, and priority alone is not
significantly predictive of the effect size within either stratum (§7, §8) —
it separates the *populations*, not the *effect magnitude* within a
population.

---

## 9. Contested-Only Diagnostic Proxies (definitions)

No fitted free coefficients; each proxy scores a side by summing a single
causal quantity (or ratio of two) over that side's contested request(s) and
compares ESTF-only's score to WFS-only's score.

| Proxy | Formula (per side) | Status |
|---|---|---|
| **A. OLD_COMPLETION_ONLY** | whole-branch `br_estf_estf_completed − br_wfs_wfs_completed` (joined from the existing 91-event artifact) | Computed, but **circular as a "ground truth" comparison** — see §10 caveat |
| **B. FAILED_AGGREGATE_PROGRESS** | `V_inflight` over *every* in-flight request at branch terminal state | **Not recomputable** from this artifact (only contested-request outcomes were persisted, not full terminal `ObservableState`); cited from the existing offline-alignment result instead (§13) |
| **C. CONTESTED_PRIORITY_FEASIBILITY** | `Σ weight_i · feasible_i` | Computed — **degenerate**: `feasible_if_admitted_now` is `False` for all 182 rows (§4), so every event scores 0–0 and the proxy collapses to a constant tie |
| **D. CONTESTED_VALUE_PER_REMAINING_SERVICE** | `Σ weight_i / predicted_service_proxy_i` | Computed |
| **E. CONTESTED_AGE_PROTECTION** (substitutes for a true fairness/class-deficit term, which is not available per-request in this extraction — documented substitution) | `Σ weight_i · queue_age_i` | Computed |

---

## 10. Long-Run Alignment

Ground truth = sign of `delta_native_whole_branch_raw` (ESTF if >0, WFS if
<0, TIE if =0), i.e. the same whole-branch raw-completed-count native signal
used throughout the prior Family-A diagnostic chain.

**Important caveat on Proxy A**: `margin_A_completion_only` is defined
directly from `delta_native_whole_branch_raw`, i.e. it *is* the ground-truth
signal at its own native scale. Its "perfect" agreement (balanced accuracy
1.0, ROC-AUC 1.0 everywhere) is a tautology, not evidence the old objective
is good — it merely confirms the old objective's within-window raw-count
computation is internally self-consistent. It is **not** independent
evidence about true full-scenario ANWG alignment (which the prior
`family_a_rollout_value_limit_diagnosis_20260820.md` §6/§10 already showed
this same raw-completion signal is *poorly* aligned with in `favlong`). It
is retained here only to show what "the old signal, restricted to the
contested pair" looks like relative to itself — not as a meaningful
baseline for C/D/E.

| Proxy | Stratum | Balanced acc. (3-class) | ROC-AUC | Macro F1 | Spearman ρ vs. `delta_native` | ESTF pref. share |
|---|---|---:|---:|---:|---:|---:|
| C_PRIORITY_FEASIBILITY | ALL | 0.333 | — | 0.173 | — (degenerate) | 0.0% |
| C_PRIORITY_FEASIBILITY | favlong | 0.500 | — | 0.178 | — | 0.0% |
| C_PRIORITY_FEASIBILITY | favshort | 0.333 | — | 0.253 | — | 0.0% |
| D_VALUE_PER_SERVICE | ALL | **0.086** | 0.103 | 0.101 | **−0.548** (p=1.9e-8) | 45.1% |
| D_VALUE_PER_SERVICE | favlong | **0.043** | — | 0.047 | **−0.320** (p=0.013) | 16.7% |
| D_VALUE_PER_SERVICE | favshort | 0.333 | 0.545 | 0.175 | −0.002 (p=0.99) | 100.0% |
| E_AGE_PROTECTION | ALL | **0.063** | 0.052 | 0.086 | **−0.406** (p=6.4e-5) | 29.7% |
| E_AGE_PROTECTION | favlong | **0.043** | — | 0.052 | −0.098 (p=0.46) | 6.7% |
| E_AGE_PROTECTION | favshort | 0.212 | 0.273 | 0.137 | −0.073 (p=0.69) | 74.2% |

**FAVSHORT**: none of C/D/E clear their stratum's own majority-class
balanced accuracy of 0.333 (C and D tie it exactly; E falls below it).
**FAVLONG**: none of C/D/E clear their stratum's own majority-class
balanced accuracy of 0.500 (C ties it; D and E fall well below it, and D's
Spearman correlation with `delta_native` is *significantly negative* —
anti-aligned with the raw-count native signal).

**Critical question answered**: no — none of the three coefficient-free
contested-only proxies tried materially improves `favlong` alignment while
preserving `favshort`; two of the three (D, E) actively **underperform
chance** on this ground truth in `favlong`. A caveat applies to reading this
too literally against D/E, though (§17 limitations): raw completed-count is
a *documented-flawed* full-scenario-ANWG surrogate specifically in `favlong`
(prior diagnosis, cited above), so a proxy's disagreement with it in
`favlong` is not unambiguous proof of proxy failure — it is ambiguous
without an independent per-event weighted/ANWG ground truth, which does not
exist in either artifact.

---

## 11. Collapse / Triviality Check

| Rule | ALL bal. acc. | favlong bal. acc. | favshort bal. acc. |
|---|---:|---:|---:|
| always-WFS | 0.333 | 0.000 | 0.333 |
| always-ESTF | 0.333 | 0.500 | 0.333 |
| regime-label-equivalent (favlong→WFS, favshort→ESTF) | 0.063 | 0.000 | 0.333 |
| priority-only (`Σweight`, no feasibility/age) | 0.063 | 0.000 | 0.333 |
| fairness/age-only (`Σqueue_age`, no weight) | **0.546** | 0.351 | 0.455 |

None of C/D/E reduces to always-WFS or always-ESTF (all show mixed
ESTF/WFS preference shares, §10). None reduces to the hidden `favlong`/
`favshort` regime-identity rule either — that rule itself scores worst of
all (0.063 ALL), so no proxy is "secretly reconstructing" it. The
noteworthy result here is the opposite direction: the naive,
un-constructed **fairness/age-only baseline (E without the weight term)
outperforms every purpose-built proxy including E itself** on ALL
(bal. acc. 0.546, ROC-AUC 0.741, though its Spearman correlation is not
significant at p=0.083) — none of the three constructed contested-only
proxies clears this simplest possible baseline.

---

## 12. Causal Availability Audit

| Variable | Classification |
|---|---|
| `priority` / `weight` | `ONLINE_CAUSAL` |
| `prompt_tokens`, `predicted_output_tokens` | `ONLINE_CAUSAL` |
| `predicted_service_proxy` | `ONLINE_CAUSAL` (derived from `ONLINE_CAUSAL` fields via `scoring.py::predicted_service_proxy`) |
| `queue_age` | `ONLINE_CAUSAL` |
| `deadline_slack_if_admitted_now` / `feasible_if_admitted_now` | `ONLINE_CAUSAL` (`scoring.py::deadline_slack` at decision time) |
| `class_id` | `ONLINE_CAUSAL` |
| `contested_side` (estf_only/wfs_only label) | `ONLINE_CAUSAL` *in principle* — requires scoring both ESTF and WFS on the identical pre-decision snapshot, which is a real, no-future-information online computation, just not free (2× scoring cost per eligible decision) |
| `br_*_completed` / `_completion_time` / `_slo_violated` / `_weighted_contribution` | `FUTURE_OUTCOME` — ground-truth/label only, never usable inside a deployable proxy |
| `canonical_scenario_id` / `split` / `step` / `event_id` | `EXPERIMENT_METADATA` |
| `favlong` / `favshort` (parsed from `canonical_scenario_id`) | `EXPERIMENT_METADATA` — analysis stratum only, forbidden as a runtime input |

Every variable used in proxies C/D/E is `ONLINE_CAUSAL`. No proxy tried
requires hidden or future information — the alignment failure (§10) is a
formula-quality problem, not a leakage problem.

---

## 13. Explaining the Failed Aggregate-Progress Value

`docs/current/family_a_terminal_value_v1_analysis_20260820.md` found that
crediting *all* in-flight requests' progress (`V_inflight`, gated by
`feasible_i`) moved branch preference toward ESTF, not WFS, in `favlong` —
the opposite of intent — with 128–199 WFS→ESTF flips vs. 1–35 ESTF→WFS
flips across horizons. The prior report's diagnostic hypothesis (§14 there)
was that ESTF's many small admissions accrue diluted aggregate credit that
outweighs WFS's concentrated credit on fewer, larger protected jobs.

**This artifact cannot directly recompute that aggregate quantity**
(§9, Proxy B note) — it only persisted outcomes for the two contested
requests per event, not the full terminal `ObservableState` needed to sum
`progress_fraction · feasible · weight` over every in-flight request. Two
pieces of new, corroborating (not independently sufficient) evidence from
this extraction:

1. **§4**: `feasible_if_admitted_now` is `False` for 100% of both ESTF-only
   and WFS-only contested requests, in every regime — these are, by
   construction, exactly the highest-contention items in an already
   deadline-saturated queue. `V_inflight`'s own `feasible_i` gate is a
   *different* computation (evaluated at branch terminal time using
   *remaining* service, not full service, from a clean decision point), so
   this is not a proof of identical mechanism — but it is directionally
   consistent with the specific high-value WFS-protected item being at
   meaningful risk of being gated to zero credit by any strict feasibility
   filter, which would suppress exactly the credit the redesign needed to
   concentrate.
2. **§7/§8**: the real per-request value split is completion-vs-SLO, not a
   uniform "progress" quantity. `V_inflight`'s `progress_fraction_i` term
   rewards partial completion identically regardless of whether that
   progress will ultimately meet the deadline — it cannot distinguish "on
   track to finish, on time" (WFS's actual contested value) from "on track
   to finish, late" (still counted at the same partial-progress rate),
   which is consistent with, though does not independently prove, why a
   uniform in-flight-progress credit failed to reproduce WFS's SLO-specific
   protection value.

Verdict: this data is **consistent with, and adds diagnostic color to, but
does not independently confirm** the prior aggregate-dilution explanation.
The originally-cited flip-count/offline-alignment evidence
(`family_a_terminal_value_v1_analysis_20260820.md` §5) remains the primary,
already-established evidence for the `TERMINAL_VALUE_OFFLINE_NO_GO` result.

---

## 14. Regime-Level vs. Request-Level Signal

Because `favlong` and `favshort` strata hold regime constant by
construction, any proxy balanced accuracy *within* a stratum that exceeds
that stratum's own majority-class baseline cannot be explained by regime
identity alone.

| Stratum | Majority-class baseline bal. acc. | C bal. acc. | D bal. acc. | E bal. acc. |
|---|---:|---:|---:|---:|
| favlong (n=60; labels ESTF=47, TIE=13, WFS=0) | 0.500 | 0.500 | 0.043 | 0.043 |
| favshort (n=31; labels TIE=19, ESTF=11, WFS=1) | 0.333 | 0.333 | 0.333 | 0.212 |

**None of C/D/E exceeds its own stratum's majority baseline.** By this
test, none of the three constructed contested-request proxies demonstrates
request-level signal beyond simply knowing the regime (indeed, D and E
underperform even that trivial baseline in `favlong`). This is the clearest
single piece of evidence against `CONTESTED_VALUE_SIGNAL_STRONG`: the
*descriptive* mechanism (§4–§8) is genuinely request-level and
well-evidenced, but none of the specific, simple formulas constructed here
successfully operationalizes it into a working predictive signal.

---

## 15. Exact Classification

**`CONTESTED_VALUE_SIGNAL_PARTIAL`**

- Contested requests explain a substantial, non-trivial share of long-run
  (whole-branch, native-continuation) policy difference: median 33% / mean
  47% / p90 100% of each event's raw-completed-count delta (§6) — satisfies
  the concentration half of `STRONG`.
- A clean, causally-grounded, statistically supported dual mechanism exists
  (completion-protection for ESTF, SLO-protection for WFS; §5, §7, §8) —
  satisfies the "consistent evidence" bar for `PARTIAL`.
- But none of the three coefficient-free contested-only proxies built from
  currently-available causal fields improves `favlong` alignment without
  also failing to beat trivial baselines, and none beats its own stratum's
  majority-class baseline (§10, §11, §14) — this is exactly what
  disqualifies `STRONG` ("causal contested features meaningfully improve
  long-run direction alignment") without being severe enough to justify
  `NO_GO` ("contested requests do not explain much... or causal variables
  do not improve alignment", read narrowly as "no such variable could work"
  — several individual causal features (`queue_age` in favlong,
  `predicted_output_tokens`/`deadline_slack` in favshort) *do* correlate
  significantly with the per-request admission-native effect at the
  univariate level, §7–§8; it is specifically the three simple linear
  aggregation formulas tried that fail, not the underlying causal
  information).

---

## 16. Exact Next Step

**`KEEP_AS_DIAGNOSTIC_ONLY`**

The concentration and mechanism findings (§4–§8, §13) are valuable,
well-evidenced diagnostic knowledge about *why* Family-A ESTF/WFS
disagreement matters and *how* the two policies protect different outcome
axes for their respective contested requests. They are not, on their own,
sufficient evidence to justify committing to a specific
`JUSTIFY_CONTESTED_TERMINAL_VALUE_V2` formulation — the one class of simple,
coefficient-free proxy formulas tried here did not survive the alignment
gate (§10, §14), and the significant univariate correlations found (§7,
§8) point toward a *more carefully constructed* future formula (e.g. one
that separately tracks completion-risk and SLO-risk rather than a single
blended score) rather than validating any formula tried in this diagnosis.
Per the task's own instruction, this next step is named, not executed.

---

## 17. Limitations

- The whole-branch "ground truth" (`delta_native_whole_branch_raw`) is a
  **raw completed-request count**, not a priority/SLO-weighted quantity —
  the only whole-branch signal available in either artifact. It is a
  documented-flawed surrogate for true full-scenario ANWG specifically in
  `favlong` (prior diagnosis chain, cited throughout). Low or negative
  agreement between a causal proxy and this signal in `favlong` is
  therefore ambiguous: it could mean the proxy is wrong, or it could mean
  the proxy is (correctly) diverging from a signal already known to be
  misleading there. No independent per-event weighted/ANWG ground truth
  exists to disambiguate this without new simulation, which this task
  prohibits.
- Proxy A (`OLD_COMPLETION_ONLY`) is defined directly from the same signal
  used as ground truth in §10; its "agreement" metrics are a tautology, not
  independent evidence, and are reported only for completeness (flagged
  explicitly at every use).
- Proxy B (`FAILED_AGGREGATE_PROGRESS`) could not be recomputed from this
  artifact at all (only contested-request outcomes were persisted, not full
  terminal branch state); §13's treatment is corroborating, not conclusive.
- The continuation-only isolation originally planned for §7/§8 (holding
  first action fixed, varying only the continuation policy) is verified
  **degenerate** (identically zero) for every row, due to Family-A's
  single-active-sequence-per-GPU, no-preemption structure. This is a real
  finding, but it means §7/§8 answer a different (and, on reflection, more
  useful) question than originally posed: not "does continuation protect a
  request already admitted" but "does being admitted by the favoring
  policy versus the other policy's own path change the outcome."
- No true per-request fairness/class-deficit-ratio field exists in this
  extraction (`class_id` is present, a deficit *value* is not); Proxy E
  substitutes `queue_age` (age-based aging proxy), a documented, explicit
  substitution, not the literally-named `CONTESTED_FAIRNESS_PROTECTION`.
- `favshort`'s ground-truth label distribution is heavily skewed (19 TIE,
  11 ESTF, only 1 WFS out of 31) — any `favshort` balanced-accuracy number
  here rests on a single WFS-labeled event and should be read as directional,
  not as a well-powered estimate.
- n=91 events (32/64 scenarios); same finite, synthetic Family-A scenario
  family as every prior study in this chain; TRAIN/VAL only.
- Sample sizes for the within-regime correlation tests are small (n=60
  favlong, n=31 favshort); several reported correlations are only
  marginally significant (p≈0.05–0.09) and should not be treated as
  strongly powered results.

---

## 18. Reproducible Commands / Artifacts

Inputs read (all pre-existing, none modified):

- `experiments/family_a_contested_request_value_diagnosis/contested_events.csv`
- `experiments/family_a_contested_request_value_diagnosis/contested_requests.csv`
- `experiments/family_a_contested_request_value_diagnosis/integrity_check.json`
- `experiments/family_a_observability_continuation_v1/family_a_observability_continuation_events.csv`
- `docs/current/family_a_terminal_value_v1_analysis_20260820.md`
- `docs/design/FAMILY_A_TERMINAL_VALUE_V1.md`
- `docs/current/family_a_rollout_value_limit_diagnosis_20260820.md`
- `docs/current/family_a_receding_horizon_oracle_v1_analysis_20260820.md`
- `docs/current/family_a_observability_continuation_v1_repaired_analysis_20260820.md`
- `scripts/diagnose_family_a_contested_request_value.py` (extraction script; read for exact branch/feature semantics, not rerun)

New artifact created by this diagnosis (deterministic, read-only over the
above; no simulation):

- `scripts/analyze_family_a_contested_request_value_diagnosis.py`
- `experiments/family_a_contested_request_value_diagnosis/contested_request_value_diagnosis_summary.json`
- `experiments/family_a_contested_request_value_diagnosis/contested_events_with_diagnosis_scores.csv`
- This report: `docs/current/family_a_contested_request_value_diagnosis_20260821.md`

Reproduce: `python3 scripts/analyze_family_a_contested_request_value_diagnosis.py`
(pure pandas/numpy/scipy/sklearn over existing CSVs; no simulator import, no
RNG, no new scenario execution; wall clock < 5s).
