# Leakage audit -- CONFIRMED, independently reproduced

**Status: CONFIRMED.** Cross-transform row-range reuse across the non-OOD
splits (TRAIN/VALIDATION/ID_TEST) of this pilot. Computed directly from this
directory's own `retained_windows.csv` -- reproducible via:

```bash
python scripts/audit_selector_v2_calibrated_pilot_leakage.py \
    --pilot-dir experiments/selector_v2_calibrated_pilot_20260720T163235Z
```

Full machine-readable output: `leakage_audit.json` (in this directory).

## What was found

- **19 cross-split row-range overlap pairs**, involving **27 distinct
  windows** out of 48 real-trace "historical"-pool windows (56%).
- Split-pair breakdown: TRAIN-VALIDATION (8 pairs), ID_TEST-TRAIN (9 pairs),
  ID_TEST-VALIDATION (2 pairs).
- Mechanism: the same underlying source-trace row range (e.g. rows 217-361
  of `burstgpt_scaled_moderate`) was drawn multiple times under different
  transforms (`representative`, `compressed_tight`, `burst_kv`,
  `noise_underpredict` -- see `calibrated_targeted_pilot.py::REAL_TRACE_TRANSFORMS`)
  and the resulting windows were assigned to *different* splits, because the
  pipeline's group-key (`source_trace x transform x pool`) treats each
  transform as an independent group. The underlying request content (prompt
  lengths, base ordering) is identical across such a pair; only arrival
  timing scale / prediction noise differs.
- **OOD_TEST is unaffected**: zero overlap pairs involve OOD_TEST, and zero
  `historical<->ood_reserved` pool violations were found -- the last-15%
  OOD reservation's row-disjointness holds as designed.

## Why the pipeline's own gate missed this

`quality_gates.json`'s `no_leakage: {"passed": true}` check verifies that
each *group* (`source_trace x transform x pool`) is wholly assigned to one
split -- which is true by construction and cannot fail. It does not check
whether *different* groups drawing from the *same underlying rows* land in
different splits. That is the gap this audit fills.

## What this does and doesn't mean

- **Does mean**: VALIDATION and ID_TEST results from this pilot
  (`selector_metrics.json`) should not be treated as clean, independent
  held-out evaluation -- some windows in those splits share underlying
  request content with TRAIN windows.
- **Does not mean**: this explains OOD_TEST's separately weak result (mean
  ANWG ~0.14-0.26 for every policy there, including oracle) -- that split
  has no overlap with TRAIN by this mechanism and its weakness is a
  different, still-open question (see `docs/current/SELECTOR_V2.md`).
- **Does not mean**: the dataset-generation pipeline is unfixable -- the
  fix is straightforward (group by underlying row range across transforms,
  not just by transform name) and should be applied before the next pilot.

See `docs/current/SELECTOR_V2.md` and `docs/current/PROJECT_STATUS.md` for
how this is reflected in the project's current-status documentation.
