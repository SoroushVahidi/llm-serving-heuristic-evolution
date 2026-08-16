# ESTF↔WFS Minimal Composition Falsification — Scientific Audit

**Date:** 2026-08-16  
**Verdict:** `SELECTION_SUFFICIENT_FOR_THIS_PAIR`  
**Run:** [`experiments/estf_wfs_composition_falsification_v1_20260816T222108Z/`](../../experiments/estf_wfs_composition_falsification_v1_20260816T222108Z/)  
**Design:** [`docs/design/ESTF_WFS_COMPOSITION_FALSIFICATION_V1.md`](../design/ESTF_WFS_COMPOSITION_FALSIFICATION_V1.md)  
**Parents:** Family A v2 Job 1182377 (`estimated_service_time_first`, `weighted_fair_share`)  
**Primary metric:** canonical `arrival_normalized_weighted_goodput`  
**tmux:** `estf-wfs-comp-pilot` (completed; elapsed ≈ 559 s)

## 1. Provenance / integrity

| Check | Result |
|---|---|
| Child evaluations | **252/252** success; 0 failures |
| Parent reference rows | 84 (42 scenarios × ESTF/WFS from Family A CSV); 0 mismatches |
| Duplicates | 0 |
| `summary.json` | present |
| Splits | train **30**, val **20**, test **10**, OOD **12** |
| BurstGPT | required; Family A features all `burstgpt_staged`; local staged CSV via `--datasets-root .local_data` |
| Hidden generator features | denylisted; unit-tested |
| Ranking semantics | normalized Borda ESTF/WFS ranks → `deterministic_place` |
| α=1 / α=0 identity | unit-tested against named experts |

Raw `composition_results.csv` is frozen evidence; not rewritten.

## 2. Composition semantics (unchanged)

For waiting set \(Q\):

1. ESTF and WFS orderings from `rank_with_named_expert`
2. Normalized ranks in \([0,1]\) (best = 1)
3. \(\mathrm{score}(r)=\alpha\cdot\mathrm{rank}_{ESTF}(r)+(1-\alpha)\cdot\mathrm{rank}_{WFS}(r)\)
4. Deterministic tie-break; existing placement path

Contextual models use **scenario-level observable summaries** (not util/favored-size/seed/noise labels). Per-run α/choice is therefore constant within a scenario (switch_count = 0), which is intentional for this minimal pilot.

## 3. Split / leakage controls

- TRAIN: seed 20260816, exclude OOD  
- VAL: seed 20260817, util ∈ {1.1, 1.3}, exclude OOD  
- TEST: seed 20260817, util = 1.5, exclude OOD  
- OOD: `favored=long` ∧ `skew=10` (both seeds)

Models selected on VAL only (logreg selector val_acc **0.30**; alpha logreg proxy_acc **0.40**). Held-out TEST/OOD never used for fitting.

## 4. Held-out method table (canonical ANWG)

### TEST (n=10)

| Method | Mean ANWG | Median | Mean regret vs oracle | Env gain \(G\) | \(G_{\varepsilon=0.01}\) | Beat both (>0 / >0.01) | Lose both |
|---|---:|---:|---:|---:|---:|---:|---:|
| ESTF parent | 0.7205 | 0.7750 | 0.0211 | 0 | 0 | 0 / 0 | 0 |
| WFS parent | 0.7178 | 0.7208 | 0.0237 | 0 | 0 | 0 / 0 | 0 |
| Best fixed parent (oracle) | 0.7416 | — | 0 | — | — | — | — |
| Static α=0.25 | 0.7164 | 0.7125 | 0.0251 | 0.0014 | 0.0004 | 1 / 1 | 1 |
| Static α=0.50 | 0.7170 | 0.7167 | 0.0246 | 0.0006 | 0.0000 | 1 / 0 | 1 |
| Static α=0.75 | 0.7142 | 0.7250 | 0.0273 | 0 | 0 | 0 / 0 | 1 |
| Contextual top-1 | **0.7034** | 0.7458 | 0.0382 | 0 | 0 | 0 / 0 | 0 |
| Contextual α composition | **0.7006** | 0.7250 | 0.0410 | **0** | **0** | **0 / 0** | 1 |
| Hard if/else | 0.7278 | 0.7750 | 0.0137 | 0 | 0 | 0 / 0 | 0 |

Best fixed parent mean = mean of per-scenario \(\max(R_{ESTF},R_{WFS})\).

### OOD (n=12; long-favored + skew=10)

| Method | Mean ANWG | Median | Mean regret | Env gain \(G\) | \(G_{\varepsilon}\) | Beat both (>0 / >0.01) | Lose both |
|---|---:|---:|---:|---:|---:|---:|---:|
| ESTF parent | 0.4217 | 0.3864 | 0.1277 | 0 | 0 | 0 / 0 | 0 |
| WFS parent | **0.5494** | 0.5136 | **0.0000** | 0 | 0 | 0 / 0 | 0 |
| Static α=0.25 | 0.5198 | 0.4848 | 0.0295 | 0 | 0 | 0 / 0 | 0 |
| Static α=0.50 | 0.5172 | 0.4856 | 0.0322 | 0.0052 | 0.0037 | 3 / 1 | 1 |
| Static α=0.75 | 0.4611 | 0.4583 | 0.0883 | 0.0010 | 0.0002 | 1 / 1 | 2 |
| Contextual top-1 | 0.4797 | 0.4568 | 0.0697 | 0 | 0 | 0 / 0 | 0 |
| Contextual α composition | 0.5106 | 0.4568 | 0.0388 | **0** | **0** | **0 / 0** | 0 |
| Hard if/else | 0.5494 | 0.5136 | 0.0000 | 0 | 0 | 0 / 0 | 0 |

On OOD, **WFS is the oracle on every scenario** (WFS regret = 0). Hard if/else collapses to WFS everywhere and matches the oracle mean. Contextual composition improves on top-1 mean but **never expands the parent envelope**.

## 5. Decisive test: composition vs top-1

| Split | Mean Δ (comp − top1) | 95% bootstrap CI | Wins/losses/ties @ ε=0.01 |
|---|---:|---|---|
| TEST | **−0.0028** | [−0.0211, +0.0114] | 2 / 1 / 7 |
| OOD | +0.0309 | [0.0000, +0.0677] | 3 / 0 / 9 |

Composition does **not** beat top-1 on TEST by a practical ε=0.01 margin; the CI includes zero and negative values. Envelope gain for contextual composition is **exactly 0** on TEST and OOD (CI [0,0]).

## 6. Alpha behavior (collapse / blending)

| Split | Mean α | Near 0 (≤0.05) | Near 1 (≥0.95) | Intermediate | Switch count |
|---|---:|---:|---:|---:|---:|
| TEST | 0.65 | 20% | 40% | 40% | 0 |
| OOD | 0.375 | 50% | 25% | 25% | 0 |

α is not purely {0,1}, but intermediate weights **do not produce envelope expansion**. Effective behavior remains within the parent hull. Top-1 choices on TEST are balanced (5 ESTF / 5 WFS); on OOD balanced (6/6) despite WFS oracle dominance — selector is weak (val acc 0.30).

## 7. Static vs contextual

- Fixed α blends are comparable to parents on TEST; tiny sporadic envelope bumps (α=0.25: 1 scenario >0.01 over both).  
- Contextual α does **not** improve on the best static blends or on top-1 on TEST.  
- State-dependent weights are **not** justified for this pair from this pilot.

## 8. ID vs OOD

| Quantity | TEST | OOD |
|---|---:|---:|
| Top-1 mean regret | 0.038 | 0.070 |
| Composition mean regret | 0.041 | 0.039 |
| Composition envelope gain | 0 | 0 |
| Composition beat-both @0.01 | 0 | 0 |

OOD is a hard fairness-conflict regime where WFS dominates. Composition does not unlock new rankings beyond parents; selector underperforms relative to always-WFS / hard-conditional-WFS.

## 9. Safety / stability

- No feasibility failures; all 252 child runs succeeded.  
- Switch counts = 0 (scenario-frozen features; no intra-episode oscillation).  
- No evidence of new starvation modes beyond parent behavior; hard-conditional OOD matches WFS fairness profile.  
- Determinism: α=0/1 ordering identity covered by unit tests; parent CSV reuse is bit-exact.

## 10. Explicit answers

| Question | Answer |
|---|---|
| A. Composition beyond selection? | **No** on held-out TEST (Δ≈−0.003; CI crosses 0) |
| B. Child beats both parents? | Contextual α: **never** on TEST/OOD at >0.01; static α=0.25: 1 TEST cell only |
| C. Envelope expansion? | Contextual α: **no** (\(G=0\)). Static: negligible |
| D. Robust on OOD? | No envelope gain; WFS already optimal |
| E. Intermediate weights meaningful? | Present (~25–40%) but **not useful** for envelope |
| F. Justify symbolic distillation? | **No** |

## 11. Verdict

**`SELECTION_SUFFICIENT_FOR_THIS_PAIR`**

Independently confirmed against the design GO criteria:

- contextual composition does **not** beat contextual top-1 by ≥0.01 on TEST;  
- composition envelope gain is 0 with CI pinned at 0;  
- no held-out scenario where contextual composition beats both parents by >0.01;  
- OOD shows no envelope expansion (WFS is already oracle).

Therefore: **more complex composition, symbolic distillation from this teacher, MAP-Elites, and LLM-guided synthesis are NOT justified from the ESTF/WFS pair alone.**

## 12. Implications for the research hypothesis

Family A v2 successfully established **complementary parents** (bidirectional ESTF↔WFS niches). This pilot falsifies the claim that a **simple rank blend / contextual α** over those parents expands the library envelope beyond selection on this corpus. Complementarity of parents ≠ automatic value of composition.

## 13. Exact next scientific action

1. Keep Family A v2 as validated PSD fairness-vs-size evidence.  
2. **Design/execute the next mechanism family** (new complementary parent pair), **or** a different composition interface only if newly motivated by that family.  
3. Do **not** start MAP-Elites, symbolic distillation, or Fireworks/Cloudrift synthesis from ESTF/WFS composition.

Negative result preserved as scientific evidence.
