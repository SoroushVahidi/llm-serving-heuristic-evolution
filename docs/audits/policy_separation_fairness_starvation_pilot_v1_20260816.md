# Policy Separation Fairness and Starvation Pilot v1 — Scientific Audit

**Date:** 2026-08-16  
**Classification:** `STRUCTURALLY_VALID` execution; scientific verdict
`USEFUL_DIAGNOSTIC_ONLY` / `REDESIGN_REQUIRED` for corpus use  
**Scope:** Frozen historical Job 1182306 only. No rerun. No MAP-Elites.
No selector retraining.

---

## 1. Provenance

| Field | Value |
|---|---|
| Slurm Job | `1182306` |
| Scratch root | `/mmfs1/scratch/ikoutis/sv96/policy_separation_fairness_starvation_pilot_20260816T211029Z_1182306` |
| Git provenance copy | `experiments/policy_separation_fairness_starvation_pilot_20260816T211029Z_1182306/` |
| Git HEAD at run | `8b0fc6c7a88d5a596e33ae1088936f659ad1ee63` |
| Config | `configs/policy_separation_fairness_starvation_pilot_v1.yaml` |
| Generator | `src/llmserveopt/policy_separation/templates_fairness_starvation.py` |
| Analysis script | `scripts/analyze_policy_separation_fairness_starvation_pilot.py` |
| Analysis artifacts | `experiments/…_1182306/analysis/` |

**Integrity (execution):** 120 scenarios × 4 policies = **480/480** successes,
0 failures (`final_summary.json`). Duplicate `(scenario_id, policy_name)` keys:
0. Grid product matches 120.

**Policies:** `fifo`, `estimated_service_time_first`, `aging_priority`,
`weighted_fair_share`.

**Scenario coordinates** are recovered by parsing
`fs.util{U}.skew{S}.vol{V}.s{SEED}` (documented + unit-tested). Family A v1
did not emit `scenario_features.csv`.

---

## 2. Critical caveats (do not overclaim)

1. **Historical primary scalar is not canonical ANWG.**  
   CSV column `anwg` is **unweighted SLO-success**
   `(completed_without_SLO_violation) / n_loaded`.  
   Canonical `RunMetrics.arrival_normalized_weighted_goodput` was **not**
   written by this runner. All primary-scalar analyses below use that
   historical unweighted SLO-success field exactly as stored.

2. **Token lengths were synthetic**, not BurstGPT-anchored.  
   Job-time BurstGPT filename miss → `synthetic_lognormal_fallback`.

3. **Perfect size predictions:** `predicted_output_tokens == actual_output_tokens`.

4. **Single-slot contention:** `max_active_sequences = 1`.

5. Frozen CSV rows are **not rewritten**. Future clarified runs use
   `unweighted_slo_success_rate` + canonical ANWG columns.

---

## 3. Methods

Reproducible analysis via:

```bash
python scripts/analyze_policy_separation_fairness_starvation_pilot.py \
  --run-dir experiments/policy_separation_fairness_starvation_pilot_20260816T211029Z_1182306
```

Primary scalar \(R_p(x)\): historical unweighted SLO-success.  
Best-vs-second margin: \(\max_p R_p -\) second-ranked \(R\).  
Near-tie at \(\varepsilon\): ≥2 policies within \(\varepsilon\) of the max.  
Pairwise \(\Delta_{ij}(x) = R_i(x) - R_j(x)\); material win if \(\Delta > \varepsilon\).

Epsilons: \(\{0, 0.001, 0.005, 0.01\}\).

---

## 4. Winner / headroom results

### Overall (n = 120)

| Quantity | Value |
|---|---|
| Unique-winner scenarios | 52 (**all** `aging_priority`) |
| Exact multi-way ties (\(\varepsilon=0\)) | 68 (rate **0.5667**) |
| Near-tie rate \(\varepsilon=0.001\) | 0.5667 |
| Near-tie rate \(\varepsilon=0.005\) | 0.5667 |
| Near-tie rate \(\varepsilon=0.01\) | **0.6000** (72/120) |
| Mean best-vs-second margin | **0.0275** |
| Median margin | 0.0 |
| Fraction margin > 0 | 0.433 |
| Fraction margin > 0.01 | **0.400** |

No scenario has a unique winner other than `aging_priority`.

### Mean historical unweighted SLO-success by policy

| Policy | Mean | Median | Mean interactive viol. rate | Mean bulk viol. rate | Mean JFI |
|---|---:|---:|---:|---:|---:|
| `aging_priority` | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 |
| `estimated_service_time_first` | 0.9719 | 1.0000 | 0.0000 | 0.0352 | 0.9990 |
| `weighted_fair_share` | 0.9594 | 0.9917 | 0.0697 | 0.0345 | 0.9945 |
| `fifo` | 0.9025 | 0.9083 | 0.4940 | 0.0000 | 0.8199 |

### Winners by utilization

| util | Exact ties | Unique `aging_priority` | Mean margin | Frac margin > 0.01 |
|---:|---:|---:|---:|---:|
| 0.5 | 24/24 | 0 | 0.0000 | 0.00 |
| 0.8 | 24/24 | 0 | 0.0000 | 0.00 |
| 1.0 | 20/24 | 4 | 0.0014 | 0.00 |
| 1.2 | 0/24 | 24 | 0.0403 | 1.00 |
| 1.5 | 0/24 | 24 | 0.0958 | 1.00 |

**Interpretation:** useful primary-scalar separation appears almost entirely as
**Aging vs the rest under overload**. Low/moderate load is flat.

Artifacts: `analysis/per_scenario_winners.csv`, `analysis/analysis_summary.json`.

---

## 5. Pairwise advantage / bidirectional separation

At \(\varepsilon=0.01\) (primary historical scalar):

| Pair (i vs j) | Mean \(\Delta_{ij}\) | i≻j | j≻i | Bidirectional? |
|---|---:|---:|---:|---|
| Aging vs FIFO | +0.0975 | 96 | 0 | No |
| Aging vs ESTF | +0.0281 | 48 | 0 | No |
| Aging vs WFS | +0.0406 | 54 | 0 | No |
| ESTF vs FIFO | +0.0694 | 88 | 4 | **Yes** (weak reverse) |
| ESTF vs WFS | +0.0125 | 39 | **0** | **No** |
| FIFO vs WFS | −0.0569 | 6 | 86 | **Yes** |

### ESTF ↔ WFS (design target)

| Metric | Value |
|---|---|
| WFS beats ESTF (any \(\varepsilon\in\{0,0.001,0.005,0.01\}\)) | **0 / 120** |
| ESTF beats WFS (\(\varepsilon=0\)) | 50 |
| Exact equal | 70 |
| Bidirectional at \(\varepsilon=0.01\) | **False** |

**Conclusion:** Family A v1 does **not** create a fairness-vs-size bidirectional
niche between WFS and ESTF on the historical primary scalar.

Artifacts: `analysis/pairwise_summary.csv`, `analysis/pairwise_deltas.csv`.

---

## 6. Seed stability

Cells: 60 unique `(util, skew, vol)` × 2 seeds.

| Metric | Value |
|---|---|
| Winner-set agreement | 48/60 (**0.80**) |
| Best-policy agreement | 48/60 (**0.80**) |
| Unstable cells (winner-set, best, or key pairwise sign disagree) | 19 |

Seed disagreement exists but is secondary to the structural design confounds
below. Artifact: `analysis/seed_stability.csv`, `analysis/unstable_cells.csv`.

---

## 7. Tenant-level fairness / starvation

### FIFO
- Interactive violation rate rises sharply with utilization: ~0.019 (util 0.5) →
  **0.951** (util 1.5); bulk violations remain **0**.
- Mechanism: arrival-order service under single-slot contention lets long bulk
  work delay tight-deadline interactive requests (convoy / deadline mismatch).

### ESTF
- Interactive violations: **0 in all 120 scenarios**.
- Bulk violations: mean rate ~0.035.
- Mechanism: interactive jobs are **shorter**; with perfect predictions ESTF
  prefers them, so it does **not** starve interactive work in this generator.

### Weighted Fair Share
- Responds to skew: interactive violation rate falls ~0.179 (skew=1) →
  **0.011** (skew=10); primary success rises ~0.940 → 0.970.
- Never exceeds ESTF on the historical primary scalar (0 WFS≻ESTF scenarios).
- Mechanism: priority weighting helps interactive compliance under skew, but
  because size already aligns with priority, ESTF captures most of the same
  wins on unweighted success.

### Aging Priority
- Historical success **1.0 on all 120 scenarios**; interactive and bulk
  violations both 0; JFI 1.0.
- Mechanism: aged priority + laxity under this SLO/capacity grid eliminates all
  measured violations → **ceiling / saturation**, not a mapped crossover surface.

JFI is reported as recorded; it is **not** a substitute for missing canonical
ANWG. Artifacts: `analysis/policy_fairness_overall.csv`,
`analysis/fairness_surfaces.csv`.

---

## 8. Hypothesis audit

| ID | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| H1 | FIFO causes interactive starvation/violations under contention | **CONFIRMED** | Mean interactive viol. rate 0.494; 104/120 scenarios with interactive violations; rate → 0.95 at util 1.5 |
| H2 | ESTF starves interactive work | **CONTRADICTED** | ESTF interactive violations = 0/120 |
| H3 | WFS responds to tenant-weight skew | **CONFIRMED** | Interactive viol. 0.179→0.011 and success 0.940→0.970 as skew 1→10 |
| H4 | WFS creates a regime where it materially beats ESTF | **CONTRADICTED** | WFS≻ESTF = 0 at all tested \(\varepsilon\); ESTF≻WFS = 50 (ε=0) |
| H5 | Aging mitigates starvation | **CONFIRMED** (with saturation caveat) | Aging success = 1.0 everywhere; 0 violations |
| H6 | Family A v1 creates useful fairness-vs-size bidirectional decision boundaries | **CONTRADICTED** | ESTF↔WFS not bidirectional; unique winners only Aging; near-tie ε=0.01 = 60% |
| H7 | Increasing utilization increases useful policy separation | **CONFIRMED** (qualified) | Mean margin 0 at util≤0.8 → 0.040 (1.2) → 0.096 (1.5); but separation is Aging-monopoly, not ESTF↔WFS |
| H8 | Interactive volume fraction materially shifts winner boundaries | **AMBIGUOUS / weak** | At high util Aging remains sole unique winner; vol modulates FIFO/ESTF levels but not unique-winner identity |

---

## 9. Design confounds (separate causes)

| Confound | Type | Effect on conclusions |
|---|---|---|
| Size–priority **collinearity** (interactive=short+high; bulk=long+low) | Generator artifact | Collapses intended fairness↔size conflict; ESTF already protects interactive |
| Perfect output-length predictions | Generator artifact | Makes ESTF oracle-SJ; removes prediction-error niche |
| `max_active_sequences=1` | Generator choice | Clean ordering effects; may amplify Aging/laxity success |
| Synthetic token lengths | Provenance limitation | External validity weaker than BurstGPT-anchored claims |
| Historical `anwg` ≠ canonical ANWG | Metric limitation | Cannot close weighted / arrival-normalized objective questions from this CSV |
| Aging ceiling (success=1 everywhere) | Policy behavior × grid | Removes discriminativeness; dominates high-util unique wins |

**Actual policy behavior that remains real under these confounds:** FIFO
interactive failure; WFS skew sensitivity; Aging’s perfect compliance on this
grid; ESTF’s interactive protection when short=high-priority.

---

## 10. Scientific verdict for Family A v1

**Verdict: `USEFUL_DIAGNOSTIC_ONLY` + `REDESIGN_REQUIRED`.**

Not `CORPUS_READY` for selector training or MAP-Elites descriptors.

| Use | Recommendation |
|---|---|
| Historical / audit evidence | **Retain forever** (frozen provenance) |
| Contribute training rows to selectors | **No** (misleading mechanism; noncanonical primary; synthetic lengths) |
| Exclude from training, keep for audit | **Yes** |
| Supersede scientifically by Family A v2 | **Yes** (do not delete v1) |

---

## 11. Family A v2 — GO / design (not implemented here)

**Decision: GO for Family A v2 redesign + new job id** after this audit.

### Required design changes
1. **Orthogonal 2×2 size × priority** (break collinearity):
   - short/high, short/low, long/high, long/low  
2. **BurstGPT token-shape anchoring** (path discovery already fixed in tree)  
3. **Canonical ANWG as primary**; keep unweighted SLO-success + tenant
   violation rates + JFI as secondary  
4. **Imperfect service-time predictions** (non-zero inversion / noise)  
5. **Equal-weight (skew=1) controls** retained  
6. **Anti-saturation tuning** so Aging is not universally perfect (tighter
   interactive SLOs and/or contention regime where headroom exists among
   multiple specialists)

### Success criteria (GO for using A v2 as PSD family)
- Near-tie rate at \(\varepsilon=0.01\) **materially below** v1’s **0.60**  
- **Bidirectional** fairness-vs-size separation (e.g. WFS≻ESTF and ESTF≻WFS
  each in a non-trivial fraction of stress cells)  
- **No universal dominant** policy (Aging not success=1 on all scenarios)  
- Seed-stable winner / pairwise signs on a majority of grid cells  
- Canonical ANWG present in results; BurstGPT (or explicit documented fallback)
  recorded in scenario provenance  

### Stop criteria
- Still size∥priority in practice  
- WFS never beats ESTF under stress  
- Aging still saturates globally  
- Near-tie remains ≥ ~0.5 at \(\varepsilon=0.01\)

---

## 12. Exact next scientific action

**Design and implement Family A v2** (orthogonal size×priority, BurstGPT,
canonical ANWG, prediction noise, anti-saturation), then run a **new** pilot
job. Do **not** start MAP-Elites, selector retraining, symbolic distillation,
LLM-guided synthesis, or large real-vLLM campaigns.

Parallel Apt-Serve/CC module-envelope interpretation remains independent and
does not replace this PSD step.

---

## 13. Analysis artifact index

Under `experiments/policy_separation_fairness_starvation_pilot_20260816T211029Z_1182306/analysis/`:

- `analysis_summary.json`
- `per_scenario_winners.csv`
- `pairwise_summary.csv`
- `pairwise_deltas.csv`
- `seed_stability.csv`
- `unstable_cells.csv`
- `policy_fairness_overall.csv`
- `fairness_surfaces.csv`
