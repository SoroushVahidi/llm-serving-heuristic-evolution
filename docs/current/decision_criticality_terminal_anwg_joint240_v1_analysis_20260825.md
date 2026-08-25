# Decision-Criticality Terminal-ANWG on Joint-240 v1 — Analysis

**Date:** 2026-08-25  
**Experiment:** `experiments/decision_criticality_terminal_anwg_joint240_v1/`  
**Design (frozen):** `docs/design/DECISION_CRITICALITY_TERMINAL_ANWG_JOINT240_V1.md`  
**Schema:** `decision_criticality_terminal_anwg_joint240_v1.0.0`  
**Wall clock:** 1456.5s (240/240 scenarios, 0 failures)

## Final preregistered verdict

**`JOINT240_TERMINAL_CRITICALITY_REPLICATED`**

Frozen criteria satisfied:
- nonzero events \(n_{\mathrm{nz}}=206\) ≥ 10;
- sparse: \(p_{\mathrm{nz}}=5.82\%\) < 15%;
- concentrated: top-10% state \(|\Delta|\) mass share = 1.000 ≥ 0.50;
- disagreement AUROC = 0.680 < 0.70.

Additive label `JOINT240_DISAGREEMENT_PROXY_USEFUL` is **not** triggered (AUROC < 0.70 despite AUPRC > base prevalence and AUROC CI excluding 0.5).

---

## Estimand (unchanged)

Continuation-policy-conditional one-step terminal effect under OOF Alive:

\[
\Delta_{\mathrm{terminal}} = \mathrm{ANWG}_{\mathrm{cf}} - \mathrm{ANWG}_{\mathrm{ref}}
\]

One forced alternative P6-native action at \(t\), then **both** branches continue with the same cloned Alive router; identical future arrivals. **Not** a policy-independent Q-value.

---

## Integrity validation

| Check | Result |
|---|---|
| `DONE` | present; `EXIT_CODE=0` |
| Log traceback / NaN / serialization / integrity fail | **none** |
| Scenarios attempted / succeeded | **240 / 240** |
| Scenario IDs vs frozen manifest | **exact match** |
| Branches (csv = jsonl) | **3541** |
| Duplicate `(scenario, step, acquisition, alt)` | **0** |
| NaN \(\Delta\) | **0** |
| Live fingerprint unchanged | **all True** |
| REF-replay | **240 / 240 match**, max absolute mismatch **0.0** |

Integrity: **PASS**. No re-run required.

---

## Primary joint-240 results

| Quantity | Value |
|---|---:|
| Scenarios | 240 |
| Acquired branches | 3541 (disagreement 2341; agreement-control 1200) |
| \(\Delta=0\) | 3335 (94.18%) |
| \(\Delta>0\) | 113 (3.19%) |
| \(\Delta<0\) | 93 (2.63%) |
| \(\|\Delta\|\ge 0.01\) | 186 (5.25%) |
| Mean / median \(\Delta\) | +0.000293 / 0.000000 |
| Mean / median \(\|\Delta\|\) | 0.001413 / 0.000000 |
| \(\|\Delta\|\) p90 / p95 / p99 | 0.000000 / 0.010257 / 0.035102 |

### Scenario prevalence

| | |
|---|---:|
| Scenarios with ≥1 nonzero | 85 / 240 (35.4%) |
| Scenarios with ≥1 positive | 48 (20.0%) |
| Scenarios with ≥1 negative | 47 (19.6%) |
| Mean nonzero branches / scenario | 0.858 |
| Nonzero/scenario quantiles (p50/p75/p90/p95/max) | 0 / 1 / 3 / 4.0 / 10 |

By acquisition: **all 206 nonzero effects are in DISAGREEMENT states**; agreement-control nonzero prevalence = **0**.

---

## Concentration + scenario-grouped bootstrap (B=2000, seed 20260825)

> **Correction (2026-08-25):** top-5 scenario mass CI fixed (multiplicity retained). Old buggy CI [0.297, 0.562] excluded the point estimate; corrected CI **[0.192, 0.399]**. State-level concentration CIs were already valid.

Zero-mass bootstrap replicates: **0** / 2000 (preregistered rule: share:=0 if total mass=0; unused here).

### State-level \(|\Delta|\) mass

| Share | Point | Bootstrap mean | CI95 |
|---|---:|---:|---|
| Top 1% | 0.4206 | 0.4185 | [0.3400, 0.5041] |
| Top 5% | 0.9513 | 0.9478 | [0.8560, 1.0000] |
| Top 10% | 1.0000 | 1.0000 | [1.0000, 1.0000] |

### Positive-effect concentration \(\max(\Delta,0)\)

| Share | Point |
|---|---:|
| Top 1% | 0.6266 |
| Top 5% | 1.0000 |
| Top 10% | 1.0000 |

### Scenario-aggregated \(|\Delta|\) mass

| | Point | Bootstrap CI (where computed) |
|---|---:|---|
| Top 1 scenario | 0.1123 | — |
| Top 2 | 0.1790 | — |
| Top 5 | 0.2930 | **corrected** mean 0.2865; CI95 [0.1924, 0.3993] (old buggy CI [0.2969, 0.5623] excluded the point; groupby-collapse bug)
| Top 5% scenarios | 0.4876 | — |
| Top 10% scenarios | 0.6845 | mean 0.6747; CI [0.5930, 0.7602] |

### Other CIs

| Estimand | mean | CI95 |
|---|---:|---|
| Nonzero prevalence | 0.0581 | [0.0444, 0.0734] |
| Mean \(\|\Delta\|\) | 0.001415 | [0.000979, 0.001939] |

---

## Disagreement as criticality proxy

Predict \(1[\|\Delta\|>10^{-12}]\) from `acquisition_type==DISAGREEMENT`.

| Metric | Value |
|---|---:|
| Base prevalence | 0.0582 |
| Prevalence \| disagreement | 0.0880 |
| Prevalence \| agreement-control | 0.0000 |
| Enrichment | **∞** (agreement-control has **zero** nonzero events) |
| AUROC | 0.6799 |
| AUROC CI95 | [0.6767, 0.6836] |
| AUPRC | 0.0819 |
| AUPRC CI95 | [0.0643, 0.1308] |
| AUPRC baseline (= prevalence) | 0.0582 |

**Interpretation:** Disagreement is a **necessary but weak** filter in this sample: every nonzero effect is a disagreement state (perfect recall among acquired states; enrichment infinite vs controls), yet most disagreements are still null (only 8.8% nonzero). AUROC ≈ 0.68 is above chance but below the preregistered “useful proxy” bar (0.70). AUPRC (0.082) only modestly exceeds prevalence (0.058).

---

## H10 proxy

**`H10 proxy unavailable for joint-240 under the frozen definition`.**

H10 completed-count events come from the A/B/C TRAIN/VAL timescale corpus and do not share `(scenario_id, step)` with joint-240 Alive trajectories. No new proxy was invented.

---

## Closed-loop trajectory divergence

| | Nonzero \(\|\Delta\|\) | Zero \(\|\Delta\|\) |
|---|---:|---:|
| Downstream divergence rate | 1.000 | 0.152 |
| Mean post-fork steps | 1722.2 | 1642.3 |
| Median post-fork steps | 1409.0 | 1122.0 |
| Range post-fork steps | [232, 6205] | [157, 7163] |
| Mean intervention step | 273.0 | 251.9 |
| Median intervention step | 215.0 | 159.0 |

Intervention-step quantiles (all branches): p10=20.0, p50=164.0, p90=631.0.

Nonzero effects occur slightly **later** on average than zero-effect forks (median step 215 vs 159) and **always** induce downstream trajectory divergence under the recorded proxies.

---

## Pressure-stratified analysis (frozen threshold 0.60)

Mass shares are fractions of total \(\sum\|\Delta\|\).

### Binary high-pressure flags

| Stratum | n | nonzero % | mean \|Δ\| | mass share |
|---|---:|---:|---:|---:|
| `high_burst_pressure=False` | 2770 | 6.10 | 0.001402 | 0.776 |
| `high_burst_pressure=True` | 771 | 4.80 | 0.001453 | 0.224 |
| `high_fairness_pressure=False` | 2606 | 6.79 | 0.001610 | 0.838 |
| `high_fairness_pressure=True` | 935 | 3.10 | 0.000866 | 0.162 |
| `high_kv_pressure=False` | 564 | 5.14 | 0.001311 | 0.148 |
| `high_kv_pressure=True` | 2977 | 5.95 | 0.001433 | 0.852 |
| `high_prefill_decode_pressure=False` | 1913 | 7.48 | 0.001798 | 0.687 |
| `high_prefill_decode_pressure=True` | 1628 | 3.87 | 0.000961 | 0.313 |
| `high_service_heterogeneity=False` | 1588 | 4.35 | 0.001026 | 0.325 |
| `high_service_heterogeneity=True` | 1953 | 7.01 | 0.001729 | 0.675 |
| `high_urgency_pressure=False` | 803 | 9.71 | 0.003197 | 0.513 |
| `high_urgency_pressure=True` | 2738 | 4.67 | 0.000890 | 0.487 |

### `n_elevated_mechanisms` bins

| Bin | n | nonzero % | mean \|Δ\| | mass share |
|---|---:|---:|---:|---:|
| 0_1 | 255 | 8.63 | 0.002106 | 0.107 |
| 2 | 696 | 8.48 | 0.002270 | 0.316 |
| 3 | 1329 | 6.09 | 0.001493 | 0.397 |
| ge4 | 1261 | 3.49 | 0.000716 | 0.181 |

Pattern: nonzero prevalence and mean \|Δ\| tend to be **higher** in lower multi-mechanism bins (0–2) than in ≥4; high-urgency and high-fairness strata show **lower** nonzero rates than their complements. Do not overinterpret small cells (e.g. 0–1 elevated, n=255 branches).

---

## Comparison with A/B/C terminal-ANWG v1 (not pooled)

| | A/B/C TRAIN/VAL | Joint-240 |
|---|---:|---:|
| Scenarios | 144 | 240 |
| Acquired states | 734 | 3541 |
| Nonzero % | 3.68 | 5.82 |
| Positive % | 2.04 | 3.19 |
| Negative % | 1.63 | 2.63 |
| Mean \|Δ\| | 0.001153 | 0.001413 |
| Top-1% state mass | 0.483 | 0.421 |
| Top-5% state mass | 1.000 | 0.951 |
| Top-5 scenario mass (joint) / top scen (A/B/C via prior summary) | — | 0.293 |
| Disagreement AUROC | 0.513 | 0.680 |
| Disagreement AUPRC | 0.045 | 0.082 |
| H10 Spearman | 0.005028827439722593 | unavailable |
| Divergence among nonzero | 1.0 | 1.0 |
| Verdict | sparse+concentrated (prior study) | **REPLICATED** |

**Relative assessment:** Joint-240 **replicates** the A/B/C qualitative pattern (sparse, highly concentrated terminal effects; disagreement insufficient as a precise criticality detector), with a larger n (3541 vs 734) and same-distribution workload as Section 4.2. Nonzero % is slightly higher (5.8% vs 3.7%); concentration remains extreme. This is **not** a contradiction and **not** inconclusive.

---

## Relation to guarded abstaining selector

From `joint240_guarded_abstaining_selector_v1` (`['GUARDED_SELECTOR_RECOVERS_SBS']`):

- Best guarded util-advantage gain vs SBS ≈ **+0.001022**, CI [-0.000033, +0.002118] (includes 0).
- Catastrophic regressions: Ascen **67** → guarded **7**.
- VBS–SBS headroom (~0.019) remains largely unexploited.

### How the two results fit

1. **Terminal criticality is sparse on the same joint-240 Alive trajectories** where Alive underperforms SBS — supporting a mechanism that only a minority of decisions move terminal ANWG.
2. **Guarded selection recovering ≈SBS** is consistent with “most decisions are not terminally critical”: abstaining to SBS avoids unsafe specialist swaps without needing to identify the rare critical forks online.
3. Neither result shows that adaptive methods **close** portfolio headroom; criticality concentration explains *why unguarded switching is dangerous*, not how to capture VBS.
4. Disagreement alone remains too crude for routing (AUROC < 0.70; most disagreements are null).

### Implications for manuscript revision (do not edit yet)

- Strengthen Section 4.2 with **same-distribution** terminal-criticality evidence (joint-240), not only A/B/C TRAIN/VAL.
- Soften any implication that “selectors inevitably destroy value”: report the **guarded ≈ SBS** result honestly.
- Keep the claim that **portfolio headroom is hard to exploit** (Alive/Ascen fail; guarded does not beat SBS; VBS gap remains).
- Frame criticality as **continuation-policy-conditional**, not Q-values.
- Report uncertainty (scenario-grouped CIs) for concentration; avoid overselling disagreement AUROC.
- Optionally add a brief “unguarded vs abstaining selector” paragraph linking catastrophic Ascen regressions to unsafe non-SBS picks.

---

## Artifacts

- `summary.json`, `summary.csv`, `bootstrap.json`
- `pressure_stratified.json`, `comparison_vs_abc.json`
- `scenario_effect_stats.csv`, `concentration_curve.csv`
- `branches.csv` / `branches.jsonl` (unchanged run outputs)
- Prior A/B/C experiment **not overwritten**
- Manuscript **not edited**
