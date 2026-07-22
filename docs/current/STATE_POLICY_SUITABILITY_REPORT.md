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
