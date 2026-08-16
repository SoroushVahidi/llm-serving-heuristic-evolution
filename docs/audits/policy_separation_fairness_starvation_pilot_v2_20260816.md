# Family A v2 Fairness-vs-Size Pilot — Scientific Audit

**Date:** 2026-08-16  
**Job:** 1182377 (successful relaunch after failed Job 1182373)  
**Provenance:** [`experiments/policy_separation_fairness_starvation_pilot_v2_20260816T220113Z_1182377/`](../../experiments/policy_separation_fairness_starvation_pilot_v2_20260816T220113Z_1182377/)  
**Design:** [`docs/design/POLICY_SEPARATION_FAMILY_A_V2.md`](../design/POLICY_SEPARATION_FAMILY_A_V2.md)  
**Predecessor v1:** Job 1182306 — `USEFUL_DIAGNOSTIC_ONLY` / `REDESIGN_REQUIRED`  
**Primary metric:** canonical `arrival_normalized_weighted_goodput`  
**Verdict:** `USEFUL_BUT_NEEDS_REFINEMENT`

## 1. Integrity / provenance

| Check | Result |
|---|---|
| Evaluations | 72 × 4 = **288**; `n_failed=0` |
| Duplicates | 0 |
| `wrapper_exit_code` | 0 |
| BurstGPT | **72/72** `burstgpt_staged`; path `BurstGPT_without_fails_1.csv` |
| Synthetic fallback | none |
| Primary column | `arrival_normalized_weighted_goodput` present; no ambiguous `anwg` |
| Factors | util∈{1.1,1.3,1.5} × skew∈{1,5,10} × fav∈{short,long} × noise∈{0,0.3} × 2 seeds |
| Git SHA at run | `16ad5d3e5af2e02516dfc42cc0825fa8eb7cbf38` |

Job 1182373 failed on BurstGPT header mismatch (`Request tokens` vs legacy names); fixed via shared schema detector before relaunch. Failed scratch retained for provenance only.

## 2. Headline quantitative findings

| Quantity | v2 (canonical ANWG) | v1 (historical unweighted SLO-success) |
|---|---:|---:|
| Exact-tie rate | **12.5%** (9/72) | 56.7% |
| Near-tie ε=0.01 | **18.1%** (13/72) | 60.0% |
| Mean best−second margin | **0.046** | 0.0275 |
| Unique winners | WFS 29, ESTF 23, Aging 11, FIFO 0 | Aging 52 only |
| Aging perfect rate | **8.3%** (6/72) | 100% (120/120) |
| ESTF≻WFS / WFS≻ESTF @ε=0.01 | **26 / 29** | 39 / **0** |
| Bidirectional ESTF↔WFS @ε=0.01 | **yes** | no |
| Token source | BurstGPT staged | synthetic fallback |
| Primary metric | canonical ANWG | noncanonical `anwg` alias |

Winner entropy (unique winners only): **1.49 bits** (three policies share niches).

## 3. Pre-registered hypotheses (H1–H10)

| ID | Result | Evidence |
|---|---|---|
| H1 FIFO suffers under contention | **CONFIRM** | FIFO mean ANWG **0.284**; favored violation rate **0.894**; **0** unique wins |
| H2 ESTF benefits short jobs regardless of priority | **CONFIRM** | Under `favored=long`, ESTF other (short) violation rate **0.000** vs favored **0.636** |
| H3 WFS responds to weights when favored is long | **CONFIRM** (partial on ANWG) | Conflict-cell favored violation rate: skew1 **0.581** → skew5 **0.513** → skew10 **0.474**. Mean WFS ANWG is non-monotonic (0.688 → 0.547 → 0.549), so weight response is clearest on tenant SLO, not always on ANWG |
| H4 Cells where ESTF≻WFS | **CONFIRM** | 26 cells @ε=0.01 (mostly `favored=short`) |
| H5 Cells where WFS≻ESTF | **CONFIRM** | 29 cells @ε=0.01 (almost entirely `favored=long`) |
| H6 Bidirectional ESTF↔WFS | **CONFIRM** | Both directions nonempty at ε∈{0,0.001,0.005,0.01} |
| H7 Aging avoids starvation, not universal | **CONFIRM** | Perfect ANWG **6/72**; unique wins **11/72**; mean ANWG **0.588** |
| H8 Near-tie ≪ v1 60% | **CONFIRM** | **18.1%** at ε=0.01 |
| H9 Seed stability | **MIXED / near-miss** | Winner-set agree **72.2%** (26/36 cells) vs preregistered ~75%; 10 unstable cells |
| H10 Real fairness-vs-size tradeoff | **CONFIRM** | Distinct ESTF/WFS niches by `favored_tenant_size`; WFS mean JFI **0.946** vs ESTF **0.888**; rankings disagree |

## 4. Regime structure

### Short-favored (aligned size∥priority; v1-like)

- Unique wins: ESTF 20, Aging 9, WFS **0**, ties 7  
- ESTF↔WFS @ε=0.01: ESTF≻WFS **23**, WFS≻ESTF **0**  
- Expected: when size and priority agree, size-based ESTF dominates fairness-aware WFS.

### Long-favored (orthogonal conflict)

- Unique wins: WFS **29**, ESTF 3, Aging 2, ties 2  
- ESTF↔WFS @ε=0.01: ESTF≻WFS **3**, WFS≻ESTF **29**  
- This is the scientifically critical regime that v1 could not create.

### Utilization / noise / skew

- Higher util increases ESTF unique wins (util 1.1→1.5: ESTF 2→13) and reduces Aging monopoly.  
- Noise σ=0.3 reduces ESTF unique wins (15→8) vs accurate control; WFS stays strong (15→14).  
- Higher skew shifts unique wins toward WFS (skew1: ESTF12/WFS7 → skew10: ESTF3/WFS11).

## 5. v1 confound checklist

| v1 confound | Fixed in v2? |
|---|---|
| Size∥priority collinearity | **Yes** — explicit `favored_tenant_size` orthogonal axis |
| No WFS≻ESTF niches | **Yes** — 29 cells @ε=0.01 |
| ~60% ε=0.01 near-ties | **Yes** — 18.1% |
| Aging universal dominance | **Yes** — 8.3% perfect; not sole unique winner |
| Synthetic tokens | **Yes** — BurstGPT staged only |
| Noncanonical ANWG | **Yes** — canonical primary column |

## 6. Verdict and corpus decision

**Verdict: `USEFUL_BUT_NEEDS_REFINEMENT`**

Family A v2 **can be included** in the policy-separation dataset as the fairness-vs-size family: it produces genuine, interpretable ESTF↔WFS complementarity under controlled orthogonality, with BurstGPT anchoring and canonical ANWG.

We are **scientifically justified in proceeding to the next mechanism family**. Remaining issues are refinement-scale, not redesign-scale:

1. Seed winner-set agreement (72%) slightly below the preregistered ~75% bar.  
2. Under conflict, WFS preferred-tenant violation rates respond to skew, but canonical ANWG is not monotone in skew (weighting/SLO interaction).  
3. Bidirectional separation is driven by the conflict half-grid; aligned short-favored cells remain ESTF-dominated (expected, but limits niche diversity there).  
4. Pilot-scale only (288 evals); not a full envelope map.

Optional later refinements (not blockers): more seeds, a skew×util slice focused on ANWG monotonicity, or a mild expansion of conflict-cell density.

## 7. Non-goals (still deferred)

MAP-Elites, selector retraining, module composition, symbolic distillation, LLM-guided evolution, large real-vLLM runs.
