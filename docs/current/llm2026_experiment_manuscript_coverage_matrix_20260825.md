# LLM 2026 — Experiment → Manuscript Coverage Matrix (2026-08-25)

**Host:** `al-khwarizmi`  
**Branch:** `contextual-compositional-heuristics-20260731` @ `2987b718…` (ahead 2)  
**Manuscript:** `paper/llm2026/main.tex` (built 15 pages)

## One-time SBS status

| Item | Value |
|---|---|
| PID 2896942 | **COMPLETED** (no longer running) |
| Progress | 240/240; wrote 3541 SBS-continuation rows; `missing_keys_total=0` |
| Verdict (artifact) | `CONTINUATION_SENSITIVE` |
| Wulver squeue | empty |

## Coverage matrix

| Experiment | Question | Status | Headline | Coverage | Relevance | Recommendation |
|---|---|---|---|---|---|---|
| `joint240_same_distribution_adaptive_exploitability_v1` | Same-dist A_scen/A_live vs SBS/VBS | COMPLETED | A_scen/A_live below SBS | FULL | CENTRAL | keep |
| `joint240_strong_learned_selector_v1` | Stronger nonlinear utility selector | COMPLETED_NOT_IN_MANUSCRIPT → **now integrated** | A_hgb=0.3145; gain +0.00047 [−0.0023,+0.0033]; GapClosure 0.025; 34/240 cat. | FULL (after this revision) | CENTRAL | INTEGRATE_MAIN_TEXT |
| `decision_criticality_terminal_anwg_joint240_v1` | Terminal criticality on joint-240 (Alive cont.) | COMPLETED_NOT_IN_MANUSCRIPT → **now primary** | 3541 states; 206 nz (5.82%); top1% mass 42.1%; AUROC 0.680 | FULL (after this revision) | CENTRAL | INTEGRATE_MAIN_TEXT |
| `decision_criticality_terminal_anwg_v1` (A/B/C) | Controlled terminal criticality | ALREADY_IN_MANUSCRIPT (was primary) | 734/27; sparse/concentrated | PARTIAL (now corroborating) | SUPPORTING | keep as secondary |
| `joint240_terminal_criticality_sbs_continuation_v1` | Continuation robustness | COMPLETED (just finished) | SBS nz 17.4%; Spearman\|C\| 0.221; Jacc@1% 0.108; CONTINUATION_SENSITIVE | PARTIAL (one-sentence caveat) | CENTRAL | INTEGRATE_MAIN_TEXT (concise) |
| `decision_criticality_terminal_utility_joint240_v1` | Continuous utility estimand | COMPLETED_NOT_IN_MANUSCRIPT | ANWG zero rate partly step-function; concentration robust under WMT | NONE | SUPPORTING | REPOSITORY_ONLY (avoid bloat) |
| `joint240_guarded_abstaining_selector_v1` | Guarded abstention | COMPLETED_NOT_IN_MANUSCRIPT | Best guard ≈SBS (gain CI includes 0); fewer catastrophes | PARTIAL (limitations clause) | SUPPORTING | INTEGRATE_LIMITATIONS |
| `joint240_alive_underperformance_decomposition_v1` | Why A_live fails | COMPLETED_NOT_IN_MANUSCRIPT | decomposition of live gap | NONE | SUPPORTING | REPOSITORY_ONLY |
| `public_replay_load_scaling_v1/v2` | Stressed public replay | COMPLETED | non-separating; v2 gate_pass | PARTIAL (limitations) | SUPPORTING | INTEGRATE_LIMITATIONS (already) |
| native vLLM Family-B / token-budget | Real-system boundary | ALREADY_IN_MANUSCRIPT | semantic mismatch + token-budget tradeoff | FULL | CENTRAL | keep |
| portfolio GP / Family-A sweeps | constructive / search | ALREADY_IN_MANUSCRIPT / SUPERSEDED mix | bounded negatives | PARTIAL | SUPPORTING | keep terse |

## Exact validated numbers used in manuscript

### A_hgb (`joint240_strong_learned_selector_v1/summary.json`)
- ANWG = **0.31454046851943507** → 0.3145
- Gain vs SBS = **+0.000468799** → +0.0005; CI **[−0.002289, +0.003346]**
- GapClosure = **0.02463** → +0.025
- Catastrophic = **34/240**
- VBS-winner accuracy = **0.2625**
- A_hgb−A_scen = **+0.008594 [0.003806, 0.013866]**
- A_hgb−A_live = **+0.030574 [0.023230, 0.038584]**

### Joint-240 Alive terminal criticality
- n = **3541**; nonzero = **206**; prevalence = **5.8176%**
- Scenario-clustered bootstrap (frozen artifact, B=2000, seed 20260825):
  - prevalence CI **[4.438%, 7.337%]**
  - top-1% mass point **42.062%**, CI **[34.003%, 50.414%]**
  - mean |Δ| **0.001415 [0.000979, 0.001939]**
  - AUROC **0.67991 [0.67668, 0.68360]**
  - AUPRC **0.08191 [0.06432, 0.13079]**; no-skill **0.05818**
- H10: **NOT AVAILABLE**

### SBS continuation (completed; concise integration)
- n paired = **3541**; SBS nonzero = **617** (17.42%)
- both zero 2819; both nz 101; only Alive 105; only SBS 516
- Spearman |C| = **0.221**; top-1% Jaccard **0.108**
- Label: **PARTLY_CONTINUATION_DEPENDENT** / artifact `CONTINUATION_SENSITIVE`

## Replication vs A/B/C
**PARTIALLY_REPLICATES** sparse/concentrated structure on same workload (higher nz rate 5.8% vs 3.7%; concentration still strong; disagreement proxy better than chance but not precise).

## Reviewer issue status (both reviews)

| Issue | Status |
|---|---|
| 1. lightweight adaptive baselines only | MOSTLY_RESOLVED (A_hgb added) |
| 2. stronger learned selector missing | RESOLVED |
| 3. joint-240 vs A/B/C disconnect | RESOLVED (joint-240 primary) |
| 4. only 27 nonzero events | MOSTLY_RESOLVED (206 on joint-240) |
| 5. concentration uncertainty | RESOLVED (CIs reported) |
| 6. AUPRC / imbalance | RESOLVED (AUPRC + no-skill) |
| 7. continuation-policy dependence | MOSTLY_RESOLVED (SBS arm; sensitive) |
| 8. loose causal terminology | MOSTLY_RESOLVED (continuation-conditional wording) |
| 9. public-trace saturation | PARTIALLY_RESOLVED (stressed; still non-separating) |
| 10. synthetic workload | UNRESOLVED (inherent) |
| 11. native vLLM breadth | PARTIALLY_RESOLVED (one config) |
| 12. H-MAS/SOLA/… empirical comparisons | UNRESOLVED (discussed, not reproduced) |
| 13. faithful LTR comparison | UNRESOLVED (explicit non-claim) |
