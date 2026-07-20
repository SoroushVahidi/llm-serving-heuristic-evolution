# Next Steps (Canonical)

Short and actionable. See [SELECTOR_V2.md](SELECTOR_V2.md) and
[PROJECT_STATUS.md](PROJECT_STATUS.md) for the full context behind each step.

## Recommended sequence

1. **Independently audit the calibrated pilot for leakage.** The pipeline's
   own `no_leakage` gate reports `passed: true`, but the mixed
   VALIDATION/ID_TEST results (§9 of [SELECTOR_V2.md](SELECTOR_V2.md)) are
   reason enough to check independently rather than accept the automated
   gate alone. This has **not been done yet** -- it is the actual first
   step, not an assumed-complete precondition.
2. If leakage is found: fix the split construction and regenerate a clean
   250-500-window calibrated pilot.
   If leakage is *not* found: investigate the alternative explanations --
   training-set size (125 windows is small for 49 features), or a genuine
   regime shift specific to the OOD split (every policy's ANWG collapses
   there, not just the selector's).
3. Verify zero leakage on whichever pilot you proceed with (don't just trust
   the automated gate a second time either -- confirm by inspection).
4. Retrain the **same** prototype selector (no hyperparameter tuning yet --
   keep this an apples-to-apples check, not a fishing expedition).
5. Evaluate cleanly on VALIDATION / ID_TEST / OOD_TEST.
6. **Stop condition:** if held-out (VALIDATION and OOD_TEST) still loses to
   best-fixed after steps 1-5, do not scale Dataset v2 generation and do not
   claim selector superiority. Go back to diagnosing why (feature set,
   model choice, training-set size, action-space scope) rather than
   generating more data on the same recipe.
7. If held-out cleanly beats best-fixed: scale Dataset v2 generation.
8. Train a final Selector v2 model on the scaled dataset.
9. Compare the final trained selector against:
   - the best fixed historical policy (already tracked throughout),
   - the 3 faithful monolithic external baselines (evaluation-only,
     Protocol C),
   - the disaggregated/migratory external baselines under their own
     topology-aware comparison (Protocol C, `docs/external_baseline_integration.md`).
10. Only after 9 produces a clean, held-out-confirmed result: prepare
    manuscript claims, using `docs/result_claims.md`'s safe/unsafe-claim
    conventions as the template.

## Explicit stop conditions

- Do not scale Dataset v2 generation while OOD/held-out generalization
  remains negative.
- Do not claim Selector v2 "beats best fixed" before a clean, leakage-
  audited held-out result exists.
- Do not compare the trained selector against the faithful external
  baselines (Protocol C) before step 6's stop condition is cleared -- that
  comparison is only meaningful once there is a selector worth comparing.
