# Selector v2 (Canonical Synthesis)

This is the missing single narrative for Selector v2. The detailed audits
below are the authoritative source for specifics; this doc's job is to put
them in order so a new reader doesn't have to reconstruct the sequence from
12 separate documents.

## The story, in order

### 1. The old objective rewarded rejection

The legacy `weighted_goodput` metric divided by **completed requests only**.
A policy that completed 20% of arrivals with perfect SLO attainment on those
could outscore a policy that completed 100% with a lower per-completion
attainment rate. `docs/audits/phase2b14_metric_definition_audit.md`,
`docs/selector_objective_audit.md`.

### 2. Corrected objective: ANWG

`arrival_normalized_weighted_goodput` (ANWG) divides by **all arriving
requests**, not just completed ones -- rejected/dropped/unfinished requests
get zero numerator credit but still count in the denominator. This is now
the primary objective for all Selector v2 work. The old `weighted_goodput`
field is retained (not deleted) as a distinctly-named "conditional quality
of completions" metric -- see `docs/selector_objective_audit.md` for the
exact, deliberate distinction.

### 3. Dataset v2 infrastructure

`docs/selector_dataset_v2.md` -- window-based feature extraction, per-policy
outcome vectors (never reduced to hard labels prematurely), group-aware
leakage-safe splits (TRAIN/VALIDATION/ID_TEST/OOD_TEST), and 10 automated
quality gates (discriminativeness, oracle headroom, no-leakage, real-trace
representation, etc.).

### 4. Scenario redesign

Early pilots found near-total window ties (see
`docs/selector_dataset_v2_validity_after_chunked_prefill_baseline.md`,
which is itself a historical checkpoint -- its "not ready" verdict was
superseded, see step 7). Root-caused to two issues: a dead `decode_first`
simulator branch (fixed, see
`docs/decode_prefill_contention_execution_model.md`) and, later, a fixed
`slo_deadline` construction that made almost every scenario trivially easy
or trivially impossible.

### 5. Faithful baseline scope audit -> Option B

`docs/selector_v2_faithful_baseline_scope_audit.md` -- a seven-angle
investigation (reproduction check, execution-health audit, loss
decomposition, pairwise specialization search, objective decomposition,
admission-control fairness audit, resource/normalization audit) concluding
the 3 faithful monolithic baselines are genuinely dominated under ANWG, not
victims of a bug or an unfair comparison. **Decision: `SELECTOR_SCOPE_DECISION
= OPTION B`** -- the Selector v2 trainable action space is the 8
historical-monolithic policies (see [BASELINES.md](BASELINES.md) §B);
faithful baselines remain external evaluation references (Protocol C,
`docs/external_baseline_integration.md`), never selector actions.

### 6. Contention-model investigation

`docs/selector_v2_contention_frontier_search.md` -- root-caused a 300/300
tied-window result to the same dead `decode_first` branch above; historical,
superseded by step 7's SLO-calibration fix for the *ranking* question
specifically.

### 7. SLO calibration fix

`docs/selector_v2_slo_calibrated_frontier_search.md` -- introduced
policy-independent, per-request SLO calibration
(`selector/dataset_v2/slo_calibration.py::calibrate_window_e2e()`). This
reversed the "not ready" verdict from step 4: genuinely
ANWG-discriminative windows went from 0/900 to 16.6%/910.

### 8. Calibrated targeted pilot (current, HEAD)

`src/llmserveopt/selector/dataset_v2/calibrated_targeted_pilot.py`
(`scripts/build_selector_dataset_v2_calibrated_targeted_pilot.py`) --
implements Option B end to end: 250 retained windows, real-trace + synthetic
scenario diversity, a genuinely newer non-overlapping real-trace time slice
reserved for OOD, group-aware splits, full 8-policy utility vectors.

## 9. Current result (verifiable claims only)

Run: `experiments/selector_v2_calibrated_pilot_20260720T163235Z/` (finished
locally, not yet committed to git).

**Pipeline's own automated checks:** all 10 quality gates passed, including
`no_leakage: {"passed": true, "detail": "verified"}` (from
`quality_gates.json`, read directly).

**Held-out performance is mixed, not a clean win:**

| Split | n windows | Regressor ANWG vs. best-fixed (0.664 TRAIN baseline) | Classifier vs. best-fixed |
|---|---|---|---|
| TRAIN | 125 | 0.695 (+0.031) | 0.684 (+0.020) |
| VALIDATION | 51 | 0.673 (**-0.012**) | 0.647 (**-0.037**) |
| ID_TEST | 43 | 0.665 (+0.029) | 0.625 (-0.011) |
| OOD_TEST | 31 | 0.227 (**-0.010**) | 0.205 (**-0.031**) |

(best-fixed = `weighted_shortest_processing` throughout; OOD_TEST's absolute
numbers are low for *every* policy including oracle at 0.263 -- a harder
regime overall, not a selector-specific collapse.)

## 10. Open question: leakage (unresolved, not confirmed)

A concern has been raised that the pilot's non-OOD splits may have
cross-transform / row-ancestry leakage that could explain the weak
VALIDATION result. **This has not been independently verified.** The
pipeline's own `no_leakage` gate reports `passed: true`, and no independent
leakage-audit artifact exists in the repository as of this writing. Given
the mixed results, an independent audit of this specific question is the
right next step -- but until one is performed and produces an actual
finding, this document states only that held-out generalization is
currently weak/mixed and the cause is undiagnosed, not that leakage is a
confirmed fact.

## 11. Current status statement

**Selector v2 has not yet demonstrated a clean, confirmed win over
best-fixed on held-out data.** It wins on TRAIN (expected) and, for the
regressor only, on ID_TEST; it loses on VALIDATION for both model variants
and on OOD_TEST for both model variants. Do not cite this pilot as a
finished result. External-baseline (faithful, Protocol C) comparison
against this selector has not yet been run and would be premature before
this generalization question is resolved.

## 12. Next step

See [NEXT_STEPS.md](NEXT_STEPS.md). In order: independently audit for
leakage -> if found, fix and regenerate a clean pilot; if not found,
investigate the alternative explanations (training-set size, genuine
regime shift) -> only then decide whether to scale Dataset v2 generation
and train a final selector.
