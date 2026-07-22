# State-Policy Suitability: Scientific Report

**Date:** 2026-07-21/22
**Branch:** `wulver-selector-v2-and-composition-integrated`
**Code:** `src/llmserveopt/selector/suitability/` (schema/API: [STATE_POLICY_SUITABILITY_SCHEMA.md](STATE_POLICY_SUITABILITY_SCHEMA.md))
**Data:** `scripts/build_state_policy_suitability_fixture.py` output, 32 windows (STRONGLY/MODERATELY_DISCRIMINATIVE synthetic controlled-stress windows only, via the existing `selector/dataset_v2/calibrated_targeted_pilot.py` generators + `discriminativeness.py` classifier), full 27-policy reward vector per window, 864 long-format rows. 19 TRAIN / 6 VALIDATION / 7 TEST windows.
**Evaluation:** `scripts/run_state_policy_suitability_report.py`.

## Scope and honesty constraints

This is a **small-scale, preliminary** evaluation (32 windows, 7 held-out TEST states) -- a first proof-of-concept that the joint state-policy suitability formulation is implementable and connects real 27-policy data to a working selector, not a scaled scientific claim. Every result below should be read with that sample size in mind. Two design choices structurally limit what this run can say:

1. The fixture keeps only STRONGLY/MODERATELY_DISCRIMINATIVE windows by construction, so **there are no near-tie states in this dataset** -- the "are improvements concentrated on meaningful-margin states?" question cannot be answered from this run (every TEST state already has a true top-2 margin > 0.01).
2. Only 6 of 27 policies have a real (EXACT/APPROXIMATE) genome mapping (see schema doc); the other 21 share a structurally near-identical `UNSUPPORTED` placeholder representation. This is the dominant confound in every structural-generalization result below and is called out explicitly where it matters.

## 1. Is joint state-policy modeling technically viable?

Yes. `f(x, pi) -> (predicted_reward, uncertainty)` is implemented, fits and predicts over the real 27-policy registry with real simulator-derived rewards, and the full pipeline (real trace -> causal features -> full 27-policy reward vector -> joint model -> suitability -> selected policy) runs end-to-end deterministically in seconds on CPU (verified by the integrated E2E smoke test).

## 2. Which encoding is strongest?

Reward-prediction MAE on the 7 TEST states (189 rows):

| Encoding | MAE | RMSE |
|---|---:|---:|
| Model 1 (identity) | 0.0956 | 0.1272 |
| Model 2 (structural only) | 0.2749 | 0.2917 |
| Model 3 (hybrid) | **0.0902** | **0.1237** |
| Independent per-policy (baseline) | 0.1280 | 0.1484 |

**Hybrid is best, identity is close second, structural-only is markedly worst** (~3x worse than identity/hybrid). Both joint encodings (identity, hybrid) beat the independent-per-policy baseline -- answering question 1 of §13 (joint modeling does beat independent per-policy regression here, modestly, in raw prediction error).

## 3. Does structural information improve prediction?

Marginally, only on top of identity: hybrid (0.0902) edges out pure identity (0.0956), a ~5.6% MAE improvement. Structural information **alone** does not currently carry enough signal to compete with identity -- expected, given only 6/27 policies have a real genome mapping; the model cannot distinguish among the 21 structurally-placeholder policies from structure alone.

For **selection** (not just prediction), the gap is much starker: the structural-only selector achieved `policy_match_accuracy = 0.143` (1/7) and `gap_closed_fraction = 0.0` -- it performed identically to just picking the single best-fixed-on-train policy every time. Structural-only selection currently provides **no value over a fixed baseline** at this policy-mapping coverage level.

## 4. Does uncertainty-aware selection improve regret?

**No, not in this evaluation, and this is a genuine (not a bug) finding.** At the default `lambda=0.5`, the per-tree-variance uncertainty term was too small relative to reward differences to change any of the 7 selections versus pure mean-argmax (`lambda=0`) -- both already achieve zero regret on this small TEST set. Sweeping lambda:

| lambda | mean regret to oracle | policy-match accuracy |
|---:|---:|---:|
| 0.0 | 0.000 | 1.00 |
| 0.5 (default) | 0.000 | 1.00 |
| 1.0 | 0.000 | 1.00 |
| 2.0 | 0.231 | 0.57 |
| 5.0 | 0.495 | 0.00 |

Larger lambda actively **hurts** regret here. Plausible explanation: with only 19 TRAIN states, per-tree prediction variance is a noisy signal not well-correlated with genuine predictive error at this scale, so penalizing it just pushes the selector toward lower-variance, lower-reward policies. This needs re-testing at larger scale before drawing a real conclusion either way -- flagged as a specific next step, not a verdict that uncertainty-aware suitability doesn't work.

## 5. How much oracle gap is closed?

On TEST: identity, hybrid, independent-per-policy, and both existing `selector/advanced.py` baselines (regressor and classifier) all reach `gap_closed_fraction = 1.0` (full oracle match, zero regret) at `lambda=0.5`. The structural-only model and the naive fixed-best-train-policy baseline both reach `gap_closed_fraction = 0.0`. Given `n=7` TEST states with clear margins (mean spread 0.63 ANWG between best and worst policy per state), this "perfect" result for the strong encodings is expected at this sample size and margin regime -- it demonstrates the mechanism works, not that the problem is solved at scale.

## 6. What happens on meaningful-margin states?

Cannot be answered from this run (see Scope note above) -- every TEST state already has a true top-2 margin > 0.01 by construction of the discriminative fixture. Re-running with a mixed discriminative/near-tie fixture is a concrete next step.

## 7. Does held-out-policy generalization show real signal?

**Mixed, genuinely mixed -- not a clean yes or no.**

| Held-out policy | Genome mapping | Hybrid-model MAE | Nearest-structural-policy MAE | Global-mean MAE | Verdict |
|---|---|---:|---:|---:|---|
| `weighted_shortest_processing` | EXACT | **0.038** | 0.089 | 0.284 | **Clear win** -- hybrid model beats both baselines by a wide margin. |
| `scorpio_style_slo_guard` | APPROXIMATE | 0.487 | 0.531 | **0.280** | **No generalization benefit** -- hybrid model is worse than a flat global mean. |
| `fifo` | UNSUPPORTED placeholder | 0.027 | 0.050 | 0.218 | Numerically good, but **not attributable to genuine structural transfer** -- `fifo`'s structural representation is the same generic placeholder shared by ~20 other policies, so this likely reflects state-feature-driven prediction of a typical reward level, not learned "fifo-ness". |

The one policy with a real, EXACT structural mapping (WSP) shows a genuine, unconfounded generalization win. The one policy with an APPROXIMATE mapping (SCORPIO) shows no benefit at all -- consistent with (and cross-validated by) the held-out-family result below, where SCORPIO is again the worst-generalizing member. The UNSUPPORTED-placeholder case cannot be used as evidence either way given its representation carries no real distinguishing information.

**Honest conclusion: structural generalization works when the genome mapping is faithful (EXACT), does not yet work when the mapping is a rough approximation, and is untestable for the 21 policies that only have a generic placeholder.** Improving genome-mapping coverage and fidelity is the clear leverage point before trusting this representation to score genuinely novel synthesized policies.

## 8. Held-out-family results

Family: `kv_memory_pressure` (from `docs/current/policy_component_matrix.json`, not an invented grouping) -- `best_fit`, `greedy_token_fill`, `kv_constrained_online`, `scorpio_style_slo_guard`, `sola_style_state_aware`, `vllm_style_token_budget` (6 policies, all held out together).

| | structural MAE | hybrid MAE | global-mean MAE |
|---|---:|---:|---:|
| Overall | 0.256 | **0.115** | 0.256 |

Per-policy hybrid MAE ranges from 0.021 (`best_fit`, `greedy_token_fill`) to 0.470 (`scorpio_style_slo_guard`) -- the same policy that failed to generalize in the single-held-out-policy pilot fails again here, a useful internal consistency check.

**Caveat that must not be glossed over:** this result is confounded. The 6 held-out policies' states are the *same* windows used to train the other 21 policies, so the hybrid model has already seen every TEST state's features 21 times (once per non-held-out policy at that state). A large part of the apparent "generalization" is plausibly the model learning *state difficulty* (this window is easy/hard for policies in general), not genuinely transferable *policy structure*. This confound cannot be resolved without a fixture where held-out-family states are also disjoint from training states -- another concrete next step.

## 9. Is the representation promising enough for later synthesized-policy scoring?

**Not yet, conditionally.** The infrastructure is real and works end-to-end. The one clean, unconfounded generalization signal (WSP, EXACT mapping) is genuinely encouraging -- it shows the *mechanism* can work when the input representation is faithful. But two of three held-out-policy tests (SCORPIO's APPROXIMATE mapping failing outright, and FIFO's UNSUPPORTED-placeholder result being uninterpretable) show the *current* representation is not yet reliable enough to trust for scoring a genuinely new, never-before-simulated synthesized policy -- which will, by construction, look more like one of the 21 placeholder policies than like the 6 faithfully-mapped ones unless genome-mapping coverage is expanded first.

## STATE_POLICY_MODEL_STATUS = STRUCTURAL_GENERALIZATION_WEAK

The selection mechanism itself is not the weak point (identity/hybrid selectors reach the oracle envelope on this small test, and joint modeling beats independent-per-policy prediction). The specific piece needed for the next stage -- genuine structural generalization to unseen policies, the prerequisite for scoring synthesized children -- is inconsistent: real for the one EXACT-mapped policy tested, absent for the one APPROXIMATE-mapped policy tested, and untestable for the 21 placeholder-mapped policies. `STRUCTURAL_GENERALIZATION_WEAK` reflects exactly this: not `NO_SIGNAL` (there is a real, unconfounded positive case), not `STRONG_SIGNAL` or `SELECTION_SIGNAL_ONLY` (the structural piece specifically -- not just selection -- is the thing being asked about, and it is weak/inconsistent, not simply absent).

## Recommended next steps (not undertaken here)

1. Expand `map_policy_to_genome` coverage beyond 6/27 policies -- the single highest-leverage fix given the WSP-vs-SCORPIO-vs-placeholder pattern above.
2. Re-run held-out-family evaluation with training states disjoint from held-out-family states, to remove the state-difficulty confound.
3. Re-run at larger scale (more windows) with a *mixed* discriminative/near-tie fixture to test the meaningful-margin and uncertainty-lambda questions this run structurally could not answer.
4. Re-examine the uncertainty formulation (per-tree variance) at larger training-set sizes before concluding it is or isn't useful for conservative suitability.

---

## Re-Evaluation After Genome Expansion (2026-07-22)

Following recommendation 1 above, `map_policy_to_genome` coverage was expanded from 6/27 to 19/27 policies (10 EXACT + 9 APPROXIMATE), verified by validation, compilation, hash-uniqueness, and (for 4 EXACT mappings) exact behavioral reconstruction-matching against the native policy on real deterministic test states. Full per-policy audit: [POLICY_GENOME_COVERAGE_AUDIT.md](POLICY_GENOME_COVERAGE_AUDIT.md). The experiment was re-run on the same 32-window fixture (identical seed, identical rewards -- only `policy_representation` changed) with held-out policies chosen to cover 5 representative families plus a general baseline, all with faithful (non-placeholder) mappings: `edf` (SLO-aware), `weighted_shortest_processing` (shortest-work), `adaptive_chunked_prefill` (prefill-aware), `kv_constrained_online` (KV-aware), `aging_priority` (fairness/aging), `fifo` (general baseline).

### Reward-prediction quality (unchanged in aggregate)

| Encoding | MAE (before expansion) | MAE (after expansion) |
|---|---:|---:|
| Model 1 (identity) | 0.0956 | 0.0956 (unchanged, as expected -- identity doesn't use the genome) |
| Model 2 (structural only) | 0.2749 | 0.2747 (essentially unchanged) |
| Model 3 (hybrid) | 0.0902 | 0.0942 (slightly worse) |

**This is an important, honest null result on its own:** expanding genome coverage did *not* improve raw prediction accuracy for the specific policies in this TEST split, because those policies were already well-represented via *identity* (they appear directly in training) -- genome coverage should be judged by held-out-*policy* generalization, not in-sample prediction quality, and the two questions must not be conflated.

### Held-out-policy results (6 policies, all faithfully mapped)

| Held-out policy | Family | Mapping | Hybrid MAE | Nearest-structural-policy MAE | Global-mean MAE |
|---|---|---|---:|---:|---:|
| `edf` | SLO-aware | EXACT | 0.155 | 0.120 | 0.373 |
| `weighted_shortest_processing` | shortest-work | EXACT | 0.047 | 0.000 | 0.284 |
| `adaptive_chunked_prefill` | prefill-aware | APPROXIMATE | 0.066 | 0.000 | 0.203 |
| `kv_constrained_online` | KV-aware | APPROXIMATE | 0.097 | 0.002 | 0.284 |
| `aging_priority` | fairness/aging | APPROXIMATE | 0.149 | 0.078 | 0.286 |
| `fifo` | general baseline | EXACT | 0.025 | 0.000 | 0.218 |

**Every one of the 6 beats global-mean substantially** (a clean, consistent win, unlike the pre-expansion run where this was only true for 1/3 policies tested). **But the nearest-structural-policy baseline now wins in all 6 cases**, several near-exactly (0.000-0.002) -- because expanding coverage gave the structural feature space many more genuinely similar neighbors to be "nearest" to. This is a real, positive finding about the *structural representation's* informativeness, and a real, honest finding about a *limitation of the specific RF-based hybrid model*: a smoothing ensemble regressor does not automatically exploit a near-perfect single nearest neighbor as sharply as direct nearest-neighbor lookup would. A distance-weighted or k-NN-based suitability model is a concrete candidate for closing this gap -- not attempted here.

### Held-out-family results

Two families evaluated: `slo_deadline_handling` (9/10 members now faithfully mapped, up from ~3/10 before) and `kv_memory_pressure` (mixed coverage, as before). Both show the same pattern as the single-held-out-policy pilots: hybrid clearly beats global mean (0.117 vs 0.260 for `slo_deadline_handling`; 0.111 vs 0.256 for `kv_memory_pressure`), and `scorpio_style_slo_guard` is consistently the worst-generalizing member in both families (hybrid MAE 0.467 and 0.443 respectively) -- a real cross-validation of the single-policy finding, not a coincidence.

### Structural-distance diagnostic (new)

Across all 171 pairs of the 19 faithfully-mapped policies, structural distance and mean-absolute-reward disagreement correlate at **r = 0.559** (Pearson, correlational only -- not a causal claim). The identical-genome pair (`fifo`, `first_fit`) sits at exactly zero distance and zero disagreement; the most structurally distant pairs all involve `scorpio_style_slo_guard` and show the largest disagreement. This is genuine, quantified evidence that the expanded structural feature space captures real behavioral information.

### Answering the four required questions (task §10)

1. **Did broader faithful genome coverage improve structural generalization?** Mixed. The *representation's* informativeness clearly improved (r=0.559 structural-distance correlation; every held-out policy's nearest-neighbor baseline became dramatically stronger). The *trained hybrid model's* own advantage over that baseline did not improve, and in most cases the model no longer beats the nearest-neighbor baseline where it used to (WSP: beat it before, loses to it now that a near-identical neighbor exists).
2. **Did structural encoding become useful for unseen-policy reward prediction?** Yes, as a *distance metric* (nearest-neighbor baseline). Not yet, as a *feature space fed into an RF regressor* -- the model formulation itself is now a visible bottleneck, not just representation coverage.
3. **Is the prior `STRUCTURAL_GENERALIZATION_WEAK` result mostly attributable to poor representation coverage?** Partially. Coverage was clearly *a* factor (the nearest-neighbor evidence is unambiguous), but not the whole story: `scorpio_style_slo_guard`'s APPROXIMATE mapping still fails to generalize regardless of how much coverage exists elsewhere, and the joint hybrid model's relative advantage over simple baselines did not improve with more coverage.
4. **Does weak structural transfer persist despite better coverage?** Yes, specifically in the *modeling* sense: the RF-based joint hybrid regressor still does not reliably outperform a much simpler nearest-structural-neighbor copy baseline, even though the underlying structural representation is now demonstrably more informative (both by the nearest-neighbor evidence and the r=0.559 distance-disagreement correlation).

### POLICY_GENOME_STATUS = PARTIAL_BUT_USEFUL

19/27 (70.4%) faithful coverage, verified (not merely claimed) via validation, compilation, hash-uniqueness, and behavioral reconstruction-matching for the simplest EXACT cases. Not `STRONG_COVERAGE` (8/27 remain genuinely unmappable given documented DSL constraints -- fixed placement strategy, determinism, variable whitelist). Not weak either: the coverage increase produced a measurable, real improvement in structural-distance informativeness (r=0.559, dramatically stronger nearest-neighbor baselines), which is exactly the kind of evidence "useful" should require.

### STRUCTURAL_GENERALIZATION_STATUS = IMPROVED_BUT_WEAK

Not `STILL_WEAK` -- there is genuine, measurable improvement in the structural representation itself (§ diagnostics above), which was absent before. Not `IMPROVED_STRONG_SIGNAL` -- the specific joint suitability model (RF-based hybrid regression) implemented so far does not yet reliably capitalize on that improved representation; it now loses to a naive nearest-structural-neighbor baseline in cases where, before this expansion, it used to win. The bottleneck has shifted from "representation coverage" to "model formulation" -- a distance-aware or k-NN-augmented suitability model is the concrete next candidate, not attempted in this pass.
