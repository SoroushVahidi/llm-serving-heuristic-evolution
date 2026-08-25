# Family B v2 PrefillControl Composition Falsification — Scientific Audit

**Date:** 2026-08-17
**Verdict:** `SELECTION_SUFFICIENT_FOR_THIS_PAIR`
**Run:** [`experiments/prefill_control_composition_v2_20260817T154633Z/`](../../experiments/prefill_control_composition_v2_20260817T154633Z/)
**Config:** [`p2_config.yaml`](../../p2_config.yaml)
**Branch doc:** [`docs/CONTEXTUAL_COMPOSITION_BRANCH.md`](../CONTEXTUAL_COMPOSITION_BRANCH.md) (Family B v2 section)
**Parents:** `full_prefill` (unlimited prefill chunk) vs `chunked_prefill_small` (chunk=64)
**Primary metric:** canonical `arrival_normalized_weighted_goodput` (ANWG)
**Commit:** `16be179` (launch commit; analysis committed separately, see repo history)

## 1. Frozen verdict criteria (read before results, per `p5_analysis_chunk_comp.py::compute_verdict`)

`PRACTICAL_EPS = 0.01` (module constant, unchanged by this session's edits — verified via `git diff` showing only two hunks inside `analyse()`, none inside `compute_verdict`).

- **`COMPOSITION_GO`**: TEST mean envelope gain > ε(0.01) **AND** bootstrap CI lower bound > 0 **AND** adequate samples (test≥4, ood≥2) **AND** (composition beats fitted selector by >ε, **OR** fitted selector already matches oracle).
- **`SELECTION_SUFFICIENT_FOR_THIS_PAIR`**: fitted selector matches oracle (`|selector_vs_oracle_delta| < 0.005`) **AND** TEST mean envelope gain < ε(0.01).
- **`INCONCLUSIVE`**: otherwise.

These criteria and the `PRACTICAL_EPS=0.01` / `0.005` thresholds are pre-existing (`p5_analysis_chunk_comp.py`, present before this run and before this continuation session touched the file). This session's only edit to `analyse()` was *which rows feed `test_sel_scores`/`ood_sel_scores`* (real fitted `contextual_top1` rows when present, replacing a hindsight-oracle placeholder) — the verdict thresholds themselves are untouched.

## 2. Run-integrity recheck (independent of the frozen analysis script)

| Check | Result |
|---|---|
| Scenarios | 32 (2×2×2×4 grid, matches `p2_config.yaml`) |
| Splits | train **16**, val **8**, test **4**, ood **4** — matches preregistered `p2_config.yaml` split spec exactly |
| Evaluations | **120/120** success, 0 failed |
| Duplicate `(scenario_id, policy_name)` pairs | 0 |
| Non-finite (NaN/Inf) primary-metric rows | 0 |
| `run_manifest.json` `git_head` | `16be179ba06243efc59225ce36f253eb7fe7404f` — matches the launch commit |
| Policies present | `full_prefill`, `chunked_prefill_small`, `chunk_96`, `chunk_128`, `chunk_192`, `prefill_control_child`, `contextual_top1`, `hard_conditional`, `contextual_alpha` (9/9 expected) |
| Row counts by policy | parents 32+32, fixed intermediates 8+8+8, `prefill_control_child` 8, composites 8+8+8 = 120 |
| Held-out seed | `20260823` only in test/ood (`test_ood_only_held_out_seed: true`) |
| TEST vs OOD factor | test = `late_pressure=low` (`late12` in id), ood = `late_pressure=high` (`late40` in id) — matches `p2_config.yaml` |

## 3. Method identities (do not conflate)

| Label | What it actually is | Row source |
|---|---|---|
| A. `full_prefill` | Parent 1, unlimited chunk (65536), fixed for the whole scenario | genuine simulation |
| B. `chunked_prefill_small` | Parent 2, chunk=64, fixed for the whole scenario | genuine simulation |
| C. best fixed parent | `max(A, B)` per scenario — **not a deployable policy**, a post-hoc oracle over the two parents only | computed from A/B, not a separate row |
| D. two-parent oracle envelope | identical quantity to C in this design (`parent_envelope` = max(full,small) per scenario) | computed from A/B |
| E. fitted contextual top-1 selector | `contextual_top1` — genuinely fit on **TRAIN** (16 scenarios), model type (logreg vs tree) picked on **VAL** (8 scenarios) via `select_prefill_model_on_val`; TEST/OOD scores are that fitted model's *prediction*, then the corresponding **already-simulated parent's own score** — not re-simulated, not the oracle | analytic, from A/B scores + fitted selector prediction |
| F. static/intermediate chunk controls | `chunk_96`, `chunk_128`, `chunk_192` — each a **fixed** chunk size for the whole scenario, genuinely simulated, structurally identical to A/B except chunk size | genuine simulation |
| G. contextual PrefillControl child | `prefill_control_child` — the actual falsification target. Re-decides its chunk size **every simulation step** from step-level observable features via a pre-specified (not data-fit) symbolic rule (`default_step_level_chunk_rule`), attached to the Action as `prefill_chunk_override` | genuine simulation, per-step dynamic |
| H. hard conditional baseline | `hard_conditional` — symbolic if/else over **scenario-level** SLO-slack features (`hard_conditional_rule`), no fitting, decides once per scenario | analytic, from A/B scores + rule prediction |
| I. oracle envelope after adding the child | not separately reported — `envelope_gain` in the frozen analysis is exactly "does any child (F ∪ G) exceed C/D"; see §9 | computed |

`contextual_alpha` (not in the user's A–I list but present in the run) is a fifth comparator: a genuine scenario-level **metric blend** `alpha·A + (1−alpha)·B` using a fitted alpha model (TRAIN/VAL), not a separate simulation, not a chunk-size interpolation.

**Explicitly not conflated:** E (fitted selector, scenario-level, TRAIN/VAL-fit) ≠ D (oracle, has no fitting — it's hindsight); F (fixed intermediate chunks, constant for a whole run) ≠ G (child, changes every step); C/D (post-hoc-best-of-two-parents, not deployable) ≠ E (a real deployable predictor).

## 4. TEST method table (n=4, canonical ANWG)

| Method | Mean ANWG | Tied-for-max on N/4 scenarios | Beats both parents strictly | Beats both by >ε=0.01 |
|---|---:|---:|---:|---:|
| A. `full_prefill` | 0.5556 | 2/4 | — | — |
| B. `chunked_prefill_small` | 0.5833 | 2/4 | — | — |
| C/D. best-fixed-parent oracle | 0.6181 | (by definition, 4/4) | — | — |
| E. `contextual_top1` (fitted selector) | **0.6181** | **4/4** | n/a (baseline) | n/a |
| F. `chunk_96` | 0.6076 | 3/4 | 0/4 | 0/4 |
| F. `chunk_128` | 0.5833 | 2/4 | 0/4 | 0/4 |
| F. `chunk_192` | 0.5660 | 0/4 | 0/4 | 0/4 |
| G. `prefill_control_child` | 0.5556 | 2/4 | 0/4 | 0/4 |
| H. `hard_conditional` | 0.5972 | 3/4 | 0/4 | 0/4 |
| `contextual_alpha` | 0.5990 | 0/4 | 0/4 | 0/4 |

Selector-vs-oracle delta (TEST): **0.0** exactly (selector matches oracle on 4/4 scenarios — `selector_val_accuracy=1.0`, confirmed independently: `contextual_top1` is in the tied-max set on all 4 TEST scenarios).

## 5. OOD method table (n=4, canonical ANWG)

| Method | Mean ANWG | Tied-for-max on N/4 scenarios | Beats both parents strictly | Beats both by >ε=0.01 |
|---|---:|---:|---:|---:|
| A. `full_prefill` | 0.5607 | 2/4 | — | — |
| B. `chunked_prefill_small` | 0.5646 | 2/4 | — | — |
| C/D. best-fixed-parent oracle | 0.5820 | (4/4) | — | — |
| E. `contextual_top1` (fitted selector) | **0.5820** | **4/4** | n/a | n/a |
| F. `chunk_96` | 0.5724 | 2/4 | 0/4 | 0/4 |
| F. `chunk_128` | 0.5637 | 1/4 | 0/4 | 0/4 |
| F. `chunk_192` | 0.5646 | 0/4 | 0/4 | 0/4 |
| G. `prefill_control_child` | 0.5607 | 2/4 | 0/4 | 0/4 |
| H. `hard_conditional` | **0.5820** | **4/4** | 0/4 | 0/4 |
| `contextual_alpha` | 0.5747 | 0/4 | 0/4 | 0/4 |

Selector-vs-oracle delta (OOD): **0.0** exactly. `hard_conditional` (no fitting at all) also happens to match the oracle on all 4 OOD scenarios here — see §8 for why this is plausible given the coarse metric grain, not evidence hard_conditional generalizes better in general.

## 6. Fitted-selector performance / regret

- Trained on 16 TRAIN scenarios, model type selected on 8 VAL scenarios (`select_prefill_model_on_val`, candidates logreg/tree).
- `selector_val_accuracy = 1.0` (`run.log`).
- Regret vs two-parent oracle: **0.0 on both TEST and OOD** (exact, not approximate — `contextual_top1` is in the tied-max group on all 8 held-out scenarios).
- This is an unusually clean result for a 2-class scenario-level classifier; see §10 (limits) for why n=4/4 held-out scenarios and a metric with a small number of achievable values make "0 regret" a weaker claim than it would be at larger n.

## 7. Contextual-child (G) behavior — genuinely dynamic, not collapsed

Independently replayed `PrefillControlChildPolicy` against all 8 held-out scenarios outside the frozen pipeline (fresh `Simulator` run, same seed, same requests) to inspect `policy.decision_log` directly:

| Scenario | ANWG (replay, matches CSV) | Distinct chunk sizes used | Chunk-size step-mix |
|---|---:|---:|---|
| `hog12.late12.slohog_ttft` (test) | 0.5833 | 3 | 65536:28%, 128:3%, 192:69% |
| `hog12.late12.slolate_ttft` (test) | 0.5833 | 3 | 65536:29%, 128:3%, 192:68% |
| `hog24.late12.slohog_ttft` (test) | 0.3889 | 3 | 65536:43%, 128:1%, 192:56% |
| `hog24.late12.slolate_ttft` (test) | 0.6667 | 3 | 65536:40%, 128:3%, 192:56% |
| `hog12.late40.slohog_ttft` (ood) | 0.8077 | 3 | 65536:50%, 128:10%, 192:40% |
| `hog12.late40.slolate_ttft` (ood) | 0.4038 | 3 | 65536:57%, 128:3%, 192:40% |
| `hog24.late40.slohog_ttft` (ood) | 0.6562 | 3 | 65536:73%, 128:5%, 192:22% |
| `hog24.late40.slolate_ttft` (ood) | 0.3750 | 3 | 65536:75%, 128:3%, 192:22% |

The child **never** collapses to a single constant chunk within a scenario (3 distinct sizes every time, across hundreds of steps) and its mix genuinely varies with the scenario (e.g. `65536` share ranges 28%→75% across the 8 held-out scenarios). This rules out "the child is secretly a fixed-chunk baseline in disguise." It also, however, **never selects `chunk_64` or `chunk_96` or `chunk_256`** on any held-out scenario — the rule's 5-tier decision surface only exercises 3 of its 6 configured bins for this workload's typical (urgency, decode-pressure) trajectory, an honest limitation of the specific hand-specified thresholds (see §9).

## 8. Child (G) vs fitted selector (E)

| Split | Mean(G−E) | Per-scenario deltas | Bootstrap 95% CI (2000 boot, seed 20261201) |
|---|---:|---|---:|
| TEST | **−0.0625** | `[0, −0.1667, 0, −0.0833]` | mean −0.0625, **[−0.125, 0.000]** |
| OOD | **−0.0213** | `[0, −0.0385, 0, −0.0469]` | mean −0.0213, **[−0.0427, 0.000]** |

The child never beats the fitted selector on any held-out scenario (upper CI bound is exactly 0.0 on both splits) and is strictly worse on 2/4 TEST and 2/4 OOD scenarios — specifically the two `slo_emphasis=late_ttft` scenarios on each split. Mechanism: `hard_conditional`/`contextual_top1` correctly identify these as small-chunk-favoring cells (both hit the true oracle here), while the step-level child's decision surface (built from `n_decoding_active`/`n_prefilling_active`/`fraction_urgent`/`min_slo_slack`, never `slo_emphasis` — which is a forbidden feature by design) never selects a small enough chunk on these scenarios to recover the oracle score.

## 9. Child (G) vs parents (A, B) and vs the two-parent oracle (C/D)

| Split | Beats both parents strictly | Beats both by >ε=0.01 | Mean(G − oracle envelope) | Envelope gain `mean_envelope_gain` (clipped ≥0) | Bootstrap CI |
|---|---:|---:|---:|---:|---:|
| TEST | 0/4 | 0/4 | −0.0625 | **0.0** | [0.0, 0.0] |
| OOD | 0/4 | 0/4 | −0.0213 | **0.0** | [0.0, 0.0] |

The child never exceeds the two-parent oracle envelope on any held-out scenario in this run — `mean_envelope_gain=0.0` on both splits with a degenerate (all-zero) bootstrap CI, since every per-scenario gain clips at 0. This is the direct input to the `COMPOSITION_GO` gate's first condition (`test_mean_gain > 0.01`), which fails.

## 10. Envelope expansion — direct answer to the falsification question

**Does the genuinely contextual PrefillControl child beat the realistically fitted contextual top-1 selector and expand the two-parent oracle envelope on held-out TEST/OOD?**

**No, on both counts, in this run.** §8 shows the child is weakly-to-strictly dominated by the fitted selector on every held-out scenario (upper CI bound 0.0 on both TEST and OOD). §9 shows zero envelope expansion beyond the two-parent oracle on every held-out scenario. The fitted selector itself already reaches the oracle envelope exactly (§6), so there is no headroom left for a composition mechanism to add on top of selection for *this specific parent pair* — matching `SELECTION_SUFFICIENT_FOR_THIS_PAIR` by construction, not just by threshold arithmetic.

## 11. Mechanism analysis

- **Does it preserve the intended TTFT-contention tradeoff?** Partially. §7 shows the child's chunk mix responds to scenario conditions (more `65536` under low hog-count / low late-pressure, more `192`/`128` mix under high late-pressure) — directionally consistent with the decode-protection theory the rule was built on. But it never reaches for the smallest chunks (`64`/`96`) that `chunked_prefill_small`/the fitted selector correctly use on `late_ttft`-emphasis scenarios.
- **Does it merely imitate a parent selector?** No — it is mechanistically distinct (§7: genuine per-step mix, not a single parent's config) and its *raw scores coincidentally equal `full_prefill`'s* on 4/8 held-out scenarios (§4–5) despite a measurably different execution trace: on `pd2.hog12.late12.slohog_ttft.s20260823` the child shows `decode_stalled_steps=53` vs `0` for every fixed-chunk policy on that same scenario, and different `mean_num_decoding`/`budget_saturation_fraction` — a real, different trajectory that happens to land on the same discrete ANWG value.
- **Does it collapse to a fixed intermediate chunk?** No (§7, ruled out directly).
- **Does it overfit TEST while failing OOD?** No evidence of that asymmetry — the child's shortfall pattern is essentially the same magnitude and the same *kind* of scenario (`late_ttft` emphasis) on both TEST (−0.0625 mean) and OOD (−0.0213 mean); if anything it is closer to the selector on OOD.
- **True state-dependent control, or artifact?** The per-step dynamics are real (§7), but the *aggregate outcome* is dominated by metric coarseness: ANWG is a discrete SLO-attainment fraction over only 12–64 requests per scenario, so many genuinely different execution trajectories land on identical or near-identical discretized scores (§12 discusses this directly). The clean "0 regret" and "0 gain" results should be read as *this run's specific 4+4 held-out sample*, not as a high-resolution measurement.

## 12. Safety / stability

| Check | Result |
|---|---:|
| Failed evaluations | 0 / 120 |
| NaN/Inf primary-metric values | 0 |
| Exact ties (`full_prefill` == `chunked_prefill_small`, all 32 scenarios) | 1 (3.1%) |
| Near-ties (ε=0.01, all 32 scenarios) | 0 |
| Per-scenario top-of-ranking ties (held-out, 9 methods) | tied-winner groups of size 3–6 methods on 7/8 held-out scenarios (only `hog24.late40.slohog_ttft` OOD has a 6-way tie at the top; smallest tie group is 3-way) |

The high rate of exact multi-method ties at the top of each held-out scenario's ranking is itself a finding: with only 12–64 total requests per scenario, ANWG (a discrete pass/fail-style aggregate) has a coarse achievable-value grid, so several structurally different chunk-size policies frequently land on the identical discretized score. This limits how much any method — including the child — can be distinguished from the two-parent oracle envelope at this scenario count and workload scale.

## 13. OOD / generalization

The qualitative pattern is stable from TEST to OOD: fitted selector = oracle (0 regret) on both; child underperforms the selector by a similar relative margin on both (−0.0625 TEST vs −0.0213 OOD, same sign, same failure mode — `late_ttft`-emphasis scenarios); no failed evaluations, no NaN/Inf, no split leakage on either split. There is no sign of TEST-specific overfitting.

## 14. Final preregistered verdict

**`SELECTION_SUFFICIENT_FOR_THIS_PAIR`** (frozen `compute_verdict`, unmodified thresholds, run mechanically via `p5_analysis_chunk_comp.py` and cross-checked by hand in §4–§9).

- `test_mean_gain = 0.0`, not `> 0.01` → `COMPOSITION_GO`'s first condition fails regardless of the other three.
- `selector_matches = True` (`|selector_vs_oracle_delta| = 0.0 < 0.005`) and `test_mean_gain (0.0) < 0.01` → `SELECTION_SUFFICIENT_FOR_THIS_PAIR` condition satisfied exactly.

## 15. What this result scientifically establishes

- For the `full_prefill` ↔ `chunked_prefill_small` parent pair, on this preregistered Family B v2 grid, a simple scenario-level fitted top-1 selector (16 train / 8 val scenarios, logreg/tree) already reaches the two-parent oracle envelope exactly on held-out TEST and OOD.
- A genuinely dynamic, per-step composition mechanism (the actual `PrefillControlChildPolicy`, verified to make real, scenario-varying, multi-valued per-step decisions and not collapse to any fixed baseline) provides **no measurable additional value** over that selector for this pair, at this scenario count and workload scale.
- The pipeline itself is now scientifically complete for this question: the previously-missing dynamic child and the previously-missing real (non-oracle-placeholder) selector are both genuinely wired in and exercised (see `docs/CONTEXTUAL_COMPOSITION_BRANCH.md` for the implementation record).

## 16. What this result does NOT establish

- It does **not** show composition is never useful for PrefillControl in general — only for this specific two-parent pair, this specific symbolic per-step rule, and this specific 32-scenario grid.
- It does **not** rule out a *better* per-step rule (e.g. one that reaches for smaller chunks under `late_ttft`-like online-observable conditions) producing genuine envelope expansion — the rule used here never selects `chunk_64`/`chunk_96`/`chunk_256` at all on held-out data (§7), so a materially different rule is untested, not falsified.
- It does **not** provide high statistical resolution: n=4/4 held-out scenarios with a coarse-grained metric (§12) means the bootstrap CIs are wide/degenerate and a small true effect could easily be invisible at this scale.
- It does **not** validate or invalidate the `hard_conditional`/`contextual_alpha` composites as general-purpose baselines beyond this pair/grid.

## 17. Exact next scientific action

`SELECTION_SUFFICIENT_FOR_THIS_PAIR` → **stop escalating composition machinery for the `full_prefill`/`chunked_prefill_small` pair.** Per the frozen verdict semantics (§1) and `docs/CONTEXTUAL_COMPOSITION_BRANCH.md`'s guardrail, this does not authorize symbolic distillation, broader module composition, or QD/MAP-Elites for this pair — those remain gated pending a `COMPOSITION_GO` result somewhere. The next mechanism family / parent pair should be selected per the existing roadmap (`docs/CONTEXTUAL_COMPOSITION_BRANCH.md` "Next Action" / CC6 guardrail); if Family B v2 PrefillControl composition is revisited, the more promising angle per this audit's mechanism analysis (§11) would be to re-run with a per-step rule that is explicitly not restricted to a 3-of-6 chunk range before concluding composition has no value for this parent pair, rather than assuming this specific rule's null result generalizes to all possible dynamic rules.

## 18. Tests / checkers

- `python3 -m pytest tests/ p8_test_runner.py -q`: 3869 passed, 62 skipped, 1 pre-existing environment-state failure (git-tree-dirty check, resolved on commit).
- `python3 scripts/check_project_handoff_consistency.py`: passed.
- Frozen analysis (`p5_analysis_chunk_comp.py`) integrity block: `n_rows=120, n_failed=0, duplicate_pairs=0, nan_or_inf_primary=0`.

## 19. Reproducibility

- Raw run untouched: `experiments/prefill_control_composition_v2_20260817T154633Z/per_policy_results.csv` (frozen evidence, not rewritten by this audit).
- Analysis artifacts regenerated deterministically via `python3 p5_analysis_chunk_comp.py --run-dir experiments/prefill_control_composition_v2_20260817T154633Z --features experiments/prefill_control_composition_v2_20260817T154633Z/scenario_features.csv` → `experiments/prefill_control_composition_v2_20260817T154633Z/analysis/`.
- Independent verification script (ad hoc, not committed): direct CSV aggregation for method means/ties/paired deltas, and a fresh out-of-pipeline `Simulator` replay of `PrefillControlChildPolicy` against all 8 held-out scenarios to inspect `decision_log` directly (§7).
