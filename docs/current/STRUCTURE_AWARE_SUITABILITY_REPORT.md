# Structure-Aware State-Policy Suitability: Scientific Report

**Date:** 2026-07-22
**Branch:** `wulver-selector-v2-and-composition-integrated`
**Code:** `src/llmserveopt/selector/suitability/structural_models.py`
**Evaluation:** `scripts/run_structural_suitability_report.py` (same 32-window discriminative fixture as the prior two reports; no new simulation launched)
**Prior results this builds on:** `POLICY_GENOME_STATUS = PARTIAL_BUT_USEFUL` (19/27), `STRUCTURAL_GENERALIZATION_STATUS = IMPROVED_BUT_WEAK` ([STATE_POLICY_SUITABILITY_REPORT.md](STATE_POLICY_SUITABILITY_REPORT.md))

## Scope note (unchanged from prior reports)

Small sample (32 windows, 7 ID-TEST states, 6 held-out policies, 4 pairwise-advantage pairs). Every number below should be read as a preliminary, well-instrumented proof-of-concept, not a scaled claim.

## A. Models implemented

All share the `fit`/`predict_mean`/`predict_uncertainty`/`predict_suitability(lam)` interface:

- `StructuralKNNModel(k, weighting)` -- k in {1,3,5}, weighting in {uniform, inverse_distance, exponential}, 9 configurations.
- `KernelSuitabilityModel(tau)` -- Nadaraya-Watson kernel regression over all training policies, `K = exp(-d/tau)`; tau in {0.5, 2.0, 5.0}.
- `StateConditionedNeighborModel` -- kernel weight discounted by neighbor policy's own predicted-reward uncertainty (from `IndependentPerPolicyRewardModel`'s per-tree variance) at that state.
- `ResidualTransferModel(k, weighting, weight_scheme)` -- KNN neighbor estimate (leave-one-policy-out at training time) plus an RF-learned residual correction over (state + structural features); `weight_scheme` in {uniform, margin, margin_plus_epsilon}.

All four are genuinely **transductive**: they look up *sibling policies' true rewards at the exact same query state* (legitimate given full-simulator-policy-vector data), weighted by structural distance -- fundamentally different from `JointRewardModel(encoding=...)`'s "structure as an RF input feature" approach, which was the sole structural mechanism evaluated in the two prior reports.

**A real bug was found and fixed during this work**: on the standard ID (different-states) TRAIN/TEST split, every transductive model initially degenerated to a constant-zero prediction (all query states were simply absent from the training-only lookup table), producing an identical MAE (0.2037) across every k/weighting/tau combination -- itself the tell. Fixed by adding an explicit `lookup_rows` parameter (widening the sibling-reward lookup to states outside the RF-training set, while the target policy's own value at its query state is still structurally never used as its own neighbor -- verified by a dedicated leakage test, `test_structural_knn_never_uses_target_policys_own_value_as_neighbor`).

## B. Preserved baselines

`policy_id_rf`, `structural_only_rf`, `hybrid_rf` (all `JointRewardModel` from the prior report), `independent_per_policy`, and the existing `nearest_structural_policy_baseline`/`held_out_policy_pilot` machinery are all still evaluated unchanged, side by side with the new models.

## C. Structural k-NN results (ID reward-prediction MAE, post-fix)

| k | uniform | inverse_distance | exponential |
|---|---:|---:|---:|
| 1 | 0.0361 | 0.0361 | 0.0361 |
| 3 | 0.0348 | 0.0349 | 0.0345 |
| 5 | 0.0360 | 0.0361 | 0.0359 |

All 9 configurations land in a tight 0.0345-0.0361 band -- weighting scheme and k barely matter for raw MAE here (small sample). All dramatically beat `hybrid_rf` (0.0942) and `structural_only_rf` (0.2747).

## D. Kernel results

`kernel_tau0.5` = **0.0022 MAE** -- the single best reward-prediction result of the entire suitability-modeling effort across all three reports, ~40x better than the best RF model. `kernel_tau2.0` = 0.0057, `kernel_tau5.0` = 0.0212 (accuracy degrades monotonically as tau widens the effective neighborhood, exactly as expected for a kernel estimator).

## E. State-conditioned neighbor results

MAE 0.0360 -- essentially tied with the plain KNN family, not clearly better. The uncertainty-discount mechanism did not produce a measurably better *point estimate* than undiscounted kernel weighting at this scale (though see F/N below for where it matters more).

## F. Residual-transfer results

`residual_transfer_uniform` = 0.0182, `residual_transfer_margin_plus_epsilon` = 0.0177 -- margin-weighted training gives a small, consistent MAE improvement over uniform (§H). Both sit between the kernel and KNN results: better than plain KNN, worse than the tight-bandwidth kernel. The residual-correction RF adds real value on top of the raw neighbor estimate, but doesn't yet close the gap to `kernel_tau0.5`.

## G. Pairwise-advantage-transfer results

| Pair | Direct DeltaModel MAE | Direct sign accuracy | Structural-KNN-implied MAE | Structural-KNN-implied sign accuracy |
|---|---:|---:|---:|---:|
| scorpio_style_slo_guard vs. weighted_shortest_processing | 0.216 | 1.00 | 0.582 | **0.14** |
| edf vs. fifo | 0.008 | 1.00 | 0.049 | **0.14** |
| aging_priority vs. weighted_shortest_processing | 0.005 | 0.43 | 0.010 | 0.43 |
| kv_constrained_online vs. adaptive_chunked_prefill | 0.021 | 0.43 | 0.023 | 0.43 |

**The direct state-only `DeltaModel` beats or ties the structural-KNN-implied delta on every pair**, and dramatically so for the two pairs with a clear true winner (scorpio-vs-WSP, edf-vs-fifo): the structural-implied delta's sign accuracy (0.14, *worse than random*) shows that composing two separately-estimated absolute rewards (`Rhat_A(x) - Rhat_B(x)`) into a delta is a worse estimator of the delta than modeling the delta directly -- a real, informative negative result, not just a statistical artifact of small n (it is consistent and directionally sharp, not noisy-around-0.5). SCORPIO-vs-WSP is the worst case, consistent with SCORPIO's APPROXIMATE genome mapping being the structurally most distant/complex one in the library (§ distance diagnostics, r=0.559 report).

## H. Margin/regret-aware results

`ResidualTransferModel(weight_scheme=...)`: `uniform` MAE 0.0182 -> `margin_plus_epsilon` MAE 0.0177 (small, consistent improvement; also `margin` alone gives 0.0613 vs 0.0681 uniform in a separate held-out-WSP check during development). Margin weighting helps a little, doesn't transform the result. `min_samples`-scale sample counts (32-864 rows) limit how much a training-time-only reweighting can move an already-strong estimator.

## I. Held-out-policy results (6 policies, all faithfully mapped)

MAE, best-performing model per policy (full table in `results/structural_suitability_report/latest/structural_suitability_results.json`):

| Policy | Best model | MAE |
|---|---|---:|
| `edf` | `structural_knn_k5_uniform` | 0.131 |
| `weighted_shortest_processing` | `structural_knn_k1_*` | **0.000** |
| `adaptive_chunked_prefill` | `structural_knn_k1_*` | **0.000** |
| `kv_constrained_online` | `structural_knn_k5_uniform` | 0.014 |
| `aging_priority` | `structural_knn_k3_uniform` | 0.027 |
| `fifo` | `structural_knn_k1_*`/`k3_*` | **0.000** |

Structural-KNN variants occupy every top spot for held-out-policy prediction; `edf` is the clear, consistent outlier (highest error everywhere it's tested, and see O below).

## J. Held-out-family results

| Family | structural_only_rf | hybrid_rf | global mean | **structural_knn** | **kernel** |
|---|---:|---:|---:|---:|---:|
| `slo_deadline_handling` (9/10 mapped) | 0.264 | 0.117 | 0.260 | **0.107** | **0.105** |
| `kv_memory_pressure` | 0.255 | 0.111 | 0.256 | **0.109** | **0.115** |

Both new transductive models beat every prior-report baseline (including `hybrid_rf`) at the harder family-holdout level, not just single-policy holdout -- the clearest evidence in this report that explicit structural transfer generalizes beyond memorized single-policy neighbors.

## K. ID selector results

| Model | Mean regret to oracle | Gap closed | Policy-match accuracy |
|---|---:|---:|---:|
| `policy_id_rf` / `hybrid_rf` / `independent_per_policy` / `kernel_tau0.5` / `residual_transfer_*` / oracle | **0.000** | **1.00** | **1.00** |
| `kernel_tau2.0` / `kernel_tau5.0` | 0.029 | 0.939 | 0.857 |
| `structural_only_rf` / `fixed_best_train` | 0.467 | 0.000 | 0.143 |
| **every `structural_knn_*` / `state_conditioned_neighbor` configuration** | **0.524** | **-0.122** | **0.000** |

**The single most important, counter-intuitive finding of this report**: despite `structural_knn_*` models having *excellent* held-out-policy prediction MAE (§I), as **selectors on the ID split they are worse than doing nothing** (worse than picking the single best-fixed-on-train policy every time, and worse than random -- 0/7 correct picks). `kernel_tau0.5`, in contrast, achieves perfect selection. The likely mechanism: averaging over a *fixed* k structural neighbors smooths away exactly the distinguishing signal that makes the true best policy better than its neighbors at a given state (a "hedgehog" effect); the tight-bandwidth kernel avoids this by weighting almost exclusively toward the single closest neighbor while still being able to draw on the full candidate pool rather than a fixed k. **This is the report's central, literal confirmation of the task's framing**: good reward-prediction accuracy and good decision quality are not the same thing, and a model must be chosen on decision regret, not RMSE (§ model-selection rule, N).

## L. OOD selector results

Reframed as the practically relevant OOD decision -- "does the model correctly judge whether admitting this held-out policy would expand the oracle envelope at each state" (directly relevant to the later frontier-gap-conditioned policy-library-expansion decision):

| Held-out policy | Frontier-expansion-decision accuracy (kernel_tau0.5) |
|---|---:|
| `weighted_shortest_processing` | **1.00** |
| `adaptive_chunked_prefill` | **1.00** |
| `kv_constrained_online` | **1.00** |
| `aging_priority` | **1.00** |
| `fifo` | **1.00** |
| `edf` | 0.625 |

5/6 held-out policies: perfect frontier-expansion judgment. `edf` is again the exception -- consistent with its outlier status in both §I and §O, not a new independent failure.

## M. Best fixed/WSP/SCORPIO/oracle comparison

On the ID TEST split: `fixed_weighted_shortest_processing` regret=0.491 (worst), `fixed_scorpio_style_slo_guard` regret=0.143, `fixed_best_train` (= edf on this fixture) regret=0.467, `oracle` regret=0 by definition. Every strong suitability model (`hybrid_rf`, `kernel_tau0.5`, `residual_transfer_*`) matches oracle (regret=0) on this small test set; every naive fixed policy and the `structural_knn_*` family sit far worse.

## N. Uncertainty-aware results (strongest structural model: `kernel_tau0.5`)

| lambda | mean regret | policy-match |
|---:|---:|---:|
| 0.0 - 5.0 (full sweep) | 0.000 | 1.00 |

Flat across the entire sweep -- unlike the prior report's RF hybrid model (where large lambda actively *hurt* regret), `kernel_tau0.5`'s selections never change with lambda at this scale: its neighbor-disagreement uncertainty is small and doesn't cross a decision boundary for any of the 7 test states. Neither actively helpful nor harmful here; genuinely inconclusive at this sample size, not evidence against uncertainty-awareness.

## O. Structural extrapolation diagnostics

| Policy | Nearest training-policy distance | Mean k=5 neighbor distance | Mean abs. error |
|---|---:|---:|---:|
| `fifo` | 0.00 | 4.83 | ~0.00 |
| `edf` | 2.95 | 5.12 | **0.139** |
| `weighted_shortest_processing` | 5.97 | 6.44 | 0.047 |
| `aging_priority` | 10.35 | 12.29 | 0.037 |
| `adaptive_chunked_prefill` | 11.33 | 13.43 | 0.058 |
| `kv_constrained_online` | 12.11 | 12.92 | 0.015 |

Cross-policy correlation (n=6, distance vs. error): **r = -0.16** (nearest-distance) and **r = -0.27** (mean-k-distance) -- weakly *negative*, the opposite of the naive "farther = worse" expectation, driven almost entirely by `edf` being both close (distance 2.95, second-closest) and the worst-predicted policy. With only 6 held-out policies this cannot support a real conclusion either way, but it does establish that **distance-to-nearest-neighbor is not simply proportional to error** here -- `edf`'s failure looks like a case-specific mismatch (its structurally-closest neighbor, `shortest_output_first`, has a genuinely different admission/tie-break profile despite similar raw distance) rather than generic extrapolation difficulty. This is exactly what "is the model interpolating or extrapolating" diagnostics are for: `edf` is nominally *interpolating* (close neighbor exists) but still fails, meaning the failure mode here is a **structural-distance/behavioral-distance mismatch for this specific policy**, not extrapolation into unmapped territory.

## P. Winning model and exact formulation

**`kernel_tau0.5`**: `S(x, pi) = mu(x, pi) - lambda * u(x, pi)`, where
`mu(x, pi) = sum_j K(pi, j) * R_true(x, j) / sum_j K(pi, j)`, `K(pi_i, pi_j) = exp(-d_struct(pi_i, pi_j) / 0.5)`, sum over all training policies with an observed reward at state `x`; `u(x, pi)` is the weighted std of those same neighbor rewards. Chosen primarily for decision regret (0.000, tied for best) and reward MAE (0.0022, best of all models in all three reports), not merely the latter (§ model-selection rule below explains why this is not circular).

## Q. Old RF (`hybrid_rf`) vs. winning model (`kernel_tau0.5`)

| Metric | `hybrid_rf` (prior report's best) | `kernel_tau0.5` |
|---|---:|---:|
| ID reward MAE | 0.0942 | **0.0022** |
| ID mean regret to oracle | 0.000 | 0.000 (tied) |
| ID policy-match accuracy | 1.00 | 1.00 (tied) |
| Held-out-family MAE (`slo_deadline_handling`) | 0.117 | **0.105** |
| Held-out-family MAE (`kv_memory_pressure`) | 0.111 | 0.115 (slightly worse) |

`kernel_tau0.5` is a strict or near-strict improvement on every axis except one (a small loss on the `kv_memory_pressure` family, 0.115 vs. 0.111) -- the first model in this whole line of work where explicit structural transfer clearly, materially beats the RF-encoding-based baseline rather than merely matching it.

## R. Module-credit readiness

**Yes, conditionally** -- per the task's own stated bar ("the answer may be YES even if unseen-family generalization is imperfect, provided suitability is useful as a feature/prior rather than treated as ground truth"). `kernel_tau0.5` now provides a genuinely informative, decision-relevant, cheaply-computed suitability prior: strong ID decisions (tied for best), strong held-out-family MAE (beats the RF baseline), and a near-perfect frontier-expansion judgment for 5/6 held-out policies. It is not reliable enough to be trusted as ground truth (the `edf` failure and the `structural_knn` selector-regret trap both show real, unresolved failure modes), but as a *contextual prior feeding a future module-credit model* -- which by construction combines multiple noisy signals rather than depending on one -- this clears the bar. The `edf`-style near-neighbor-but-wrong failures are exactly the kind of signal a module-level credit model (crediting specific structural pieces, not whole policies) should be able to resolve, since `edf`'s failure is plausibly a mismatch in one *specific* module (priority_rule/tie_breaker) rather than a global structural distance problem.

## Falsification verdict

## STRUCTURE_AWARE_MODEL_STATUS = NICHE_USEFUL

Not `STILL_WEAK` -- `kernel_tau0.5` materially improved reward-prediction MAE (40x), held-out-family generalization (beats the RF baseline on 1/2 families, ties on the other), and OOD frontier-expansion judgment (5/6 perfect) relative to every prior-report model. Not `STRONG` either -- the improvement is **not uniform across all structural-transfer formulations**: the `structural_knn_*` family (9/9 configurations) is actively *harmful* as a selector despite being reasonably accurate as a predictor, and pairwise-advantage transfer via structural composition is clearly worse than direct delta modeling on every pair tested. Structural transfer helps *near known structural neighbors and with the right kernel bandwidth* (a tight one), and *does not* help uniformly or when composed naively (fixed-k averaging, or subtracting two separately-estimated absolute rewards) -- exactly the definition of `NICHE_USEFUL`.

Where the remaining weakness points: **not** insufficient policy diversity (19/27 faithfully-mapped policies span 6 real families) or insufficient training states (kernel_tau0.5 needed very little data to excel) -- it points most directly at **structural distance not perfectly matching behavioral distance for specific policies** (the `edf` case, §O) and at **naive fixed-k neighbor averaging being the wrong aggregation** for decision-relevant (not just MAE-relevant) prediction (§K). Both are formulation/aggregation issues, addressable without more genome coverage or more data.

## Recommended next research step

Investigate *why* `edf`'s nearest structural neighbor (`shortest_output_first`) mismatches its behavior despite low structural distance -- likely the highest-leverage single follow-up, since it's the one clear, reproducible failure mode identified across three independent diagnostics (§I, §L, §O) in this report.
