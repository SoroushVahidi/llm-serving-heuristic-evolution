# CC5 Final Operating Envelope Report

Date: 2026-08-03/04
Branch: `contextual-compositional-heuristics-20260731`
Starting SHA: `f5a4f82d54111a656e5f49c554c2b41974de5349`
New SHA: `33e832c2b2d6cc8f8ce3405f3ffc19f80e3cae2c`
Canonical issue: [#5](https://github.com/SoroushVahidi/llm-serving-heuristic-evolution/issues/5).
tmux session: `cc5_uncertainty_regime`; log: `logs/cc5_uncertainty_regime_finalization_20260804_024058.log`.
Reference run: `results/cc5_final_operating_envelope/20260804T024524Z/`
(untracked/local; reproducible via `bash results/cc5_final_operating_envelope/20260804T024524Z/replay_commands.sh`).

## 1. Verdict, Up Front

**Final CC5 classification: `COMPLETE_REGIME_SPECIFIC`.**

The frozen, deterministic operating-envelope system statistically beats
best fixed policy (paired 95% CI `[+0.0074, +0.0235]`, p<0.0001) and
remains competitive with the hard selector (paired 95% CI
`[+0.0020, +0.0199]`, p=0.0207), with **zero completion violations** and a
non-empty operating envelope (7 of 12 regimes) derived entirely from
development-split evidence. Its point-estimate edge over best global
composition (+0.0019 ANWG, 0.4044 vs 0.4025) is **not** statistically
distinguishable from zero on the paired held-out comparison (95% CI
`[-0.0044, +0.0083]`, p=0.5654) -- full-context superiority over best
global composition is **not** established. Per the task's own framing,
this is not a failure: CC5 is not required to beat global composition in
every regime, and the correct, honest classification for "beats fixed and
selector significantly, ties global, non-empty validated envelope, zero
completion violations" is `COMPLETE_REGIME_SPECIFIC`, not `COMPLETE_FULL`.

CC5 is closed as **COMPLETE -- REGIME_SPECIFIC**. CC6 is queued in a
**restricted** form (see §6).

## 2. State Verification

Confirmed at start of this query:

* branch `contextual-compositional-heuristics-20260731`;
* HEAD `f5a4f82d54111a656e5f49c554c2b41974de5349` (expected starting SHA);
* local/remote synchronized at that SHA (0 ahead / 0 behind) before work
  began, working tree clean;
* `scripts/check_contextual_composition_status.py` passed;
* `scripts/check_contextual_composition_status.py --resume-readiness`
  passed;
* CC5 `IN PROGRESS`; issue #5 OPEN; CC6 (issue #6) BLOCKED;
* CC4b + CC5-retry + CC5-uncertainty-regime artifacts present and
  `validate_cc4_dataset` clean.

## 3. Paired Statistical Analysis

Performed on the **unrestricted contextual predictor** (existing
OOD+uncertainty gate, completion-safe hybrid fallback -- the same
deployable point evaluated in the CC5 uncertainty/regime refinement,
reproduced deterministically at seed=0) against each baseline, on the 76
held-out evaluation windows (`ID_TEST` n=50, `OOD_TEST` n=26), never
adjusted after computation. Methods: paired bootstrap CI of the mean
difference (5,000 resamples of window indices, jointly resampled so
`predictor[w] - baseline[w]` pairing is preserved), a two-sided paired
sign-flip permutation test (10,000 permutations), Cohen's d on the paired
differences, and window-level win/tie/loss counts (tie = `|diff| <
0.005`, matching CC4's own near-tie threshold column).

| Comparison | Subset | n | Mean diff | 95% CI | p (perm) | Cohen's d | W/T/L |
|---|---|---|---|---|---|---|---|
| vs best_global | overall | 76 | +0.0019 | [-0.0044, +0.0083] | 0.565 | 0.07 | 22/40/14 |
| vs best_global | non_near_tie | 35 | +0.0032 | [-0.0066, +0.0131] | 0.546 | 0.10 | 11/16/8 |
| vs best_global | ID only | 50 | +0.0016 | [-0.0065, +0.0095] | 0.714 | 0.05 | 13/27/10 |
| vs best_global | OOD only | 26 | +0.0024 | [-0.0077, +0.0118] | 0.635 | 0.09 | 9/13/4 |
| vs best_fixed | overall | 76 | +0.0149 | [+0.0074, +0.0235] | <0.001 | 0.42 | 30/36/10 |
| vs best_fixed | non_near_tie | 35 | +0.0232 | [+0.0107, +0.0379] | 0.001 | 0.55 | 20/11/4 |
| vs best_fixed | ID only | 50 | +0.0198 | [+0.0097, +0.0321] | 0.001 | 0.48 | 24/19/7 |
| vs best_fixed | OOD only | 26 | +0.0054 | [-0.0016, +0.0128] | 0.164 | 0.28 | 6/17/3 |
| vs hard_selector | overall | 76 | +0.0106 | [+0.0020, +0.0199] | 0.021 | 0.27 | 29/24/23 |
| vs hard_selector | non_near_tie | 35 | +0.0221 | [+0.0077, +0.0382] | 0.008 | 0.46 | 19/4/12 |
| vs hard_selector | ID only | 50 | +0.0117 | [+0.0008, +0.0242] | 0.060 | 0.27 | 19/16/15 |
| vs hard_selector | OOD only | 26 | +0.0084 | [-0.0016, +0.0210] | 0.183 | 0.27 | 10/8/8 |

**Answering the task's core question:** the 0.0006-0.0019 predictor-global
point difference (unrestricted or frozen) is **not** statistically
distinguishable from zero at any subset (ID, OOD, non-near-tie, or
overall) -- the paired 95% CI always straddles zero, and permutation
p-values are all >0.5. In contrast, the predictor's advantage over best
fixed is significant everywhere except OOD-only (where n=26 and the
fallback dominates via envelope/OOD gating), and its advantage over the
hard selector is significant overall and on ID/non-near-tie subsets. Full
per-regime paired tables: `paired_regime_analysis.csv`.

## 4. Frozen Operating Envelope

**Derivation methodology (development-split evidence only, never touching
evaluation-split windows):** for each of the 30 development windows
(`TRAIN` + `VALIDATION`), leave-one-development-window-out (LOWO):
refit the point model (`gradient_boosting`, unchanged LOWO-CV selection
criterion) on the other 29 development windows and record its actual
ANWG on the held-out window; separately refit `best_fixed_policy` and
`best_global_composition` on those same other 29 development windows and
record their actual ANWG on the same held-out window. This gives a
paired, leakage-free (evaluation-split-free) LOWO comparison per
development window. A regime enters the trusted envelope only if it has
**>= 2 development windows** (never a single-sample fluke) **and** mean
LOWO predictor ANWG >= mean LOWO best-global-composition ANWG across that
regime's development windows. Regimes with **zero** development windows
(pure-OOD-only regimes, by CC4b's own split design) are excluded by
construction -- there is no development evidence to trust them on.

| Regime | Dev windows | LOWO predictor | LOWO global | Trust? |
|---|---|---|---|---|
| azure_conversation_like | 0 | -- | -- | No (no dev windows) |
| burst_transition | 3 | 0.5827 | 0.5827 | **Yes** |
| burstgpt_derived | 0 | -- | -- | No (no dev windows) |
| kv_pressure | 3 | 0.1417 | 0.1270 | **Yes** |
| long_output | 3 | 0.0086 | 0.0086 | **Yes** |
| long_prompt | 3 | 0.1926 | 0.1987 | No (below global) |
| mixed_slo | 3 | 0.6550 | 0.6734 | No (below global) |
| prediction_noise | 3 | 0.5772 | 0.5772 | **Yes** |
| priority_conflict | 3 | 0.5182 | 0.5382 | No (below global) |
| saturated | 3 | 0.3970 | 0.3773 | **Yes** |
| selective_admission_trap | 3 | 0.2486 | 0.2415 | **Yes** |
| underloaded | 3 | 0.9855 | 0.9855 | **Yes** |

**Frozen trusted envelope (envelope_version=1):** `burst_transition`,
`kv_pressure`, `long_output`, `prediction_noise`, `saturated`,
`selective_admission_trap`, `underloaded` (7 of 12 regimes). This is
broader than the task's stated expected-initial envelope
(`kv_pressure`, `saturated`); the dev-only LOWO evidence supports trusting
five additional regimes without touching held-out data. `long_prompt`,
`mixed_slo`, and `priority_conflict` are excluded because LOWO predictor
underperforms LOWO best-global there; `azure_conversation_like` and
`burstgpt_derived` are excluded structurally (zero development windows --
by CC4b's own split design these are OOD-only regimes the predictor never
sees in training).

**Deployable policy (deterministic, versioned, logged):**

```python
def select_with_frozen_envelope(gate, artifact, ds, causal_row, *, fallback):
    in_envelope = gate.in_envelope(regime)                       # envelope_version=1
    decision = select_composition_with_fallback(                 # existing OOD+uncertainty gate
        artifact, ds, causal_row, gate_mode="ood_or_uncertainty", fallback_override=fallback,
    )
    used_predictor = in_envelope and not decision["abstained"]
    selected = decision["model_recommended_candidate_id"] if used_predictor else fallback.select(regime)
    # fallback_reason logs "regime_outside_envelope", "ood", "high_uncertainty", or a combination
```

i.e. *use the contextual predictor when the regime is inside the
validated envelope AND calibrated uncertainty/OOD checks pass; otherwise
fall back to the validation-tuned completion-safe choice between
best-global and best-fixed* (`fit_completion_safe_fallback_rules`,
unchanged from the uncertainty/regime refinement, fit on `VALIDATION`
only). `FrozenEnvelopeGate` carries `schema_version=1`,
`envelope_version=1`, `dataset_config_hash`, and `dev_window_count`;
`assert_envelope_compatible` rejects a stale/missing gate. Implementation:
`src/llmserveopt/experiments/cc5_final_operating_envelope.py`.

## 5. Held-Out Evaluation Of The Frozen System (Touched Exactly Once)

| Metric | Value |
|---|---|
| Overall ANWG | 0.4044 [0.3377, 0.4701] (n=76) |
| ID ANWG | 0.4293 [0.3537, 0.5073] (n=50) |
| OOD ANWG | 0.3565 [0.2394, 0.4820] (n=26) |
| Non-near-tie ANWG | 0.3164 [0.2471, 0.3893] (n=35) |
| Completion fraction | 0.9118 |
| Completion violations | **0** |
| Abstention / fallback rate | 0.5789 (44/76) |
| Inference overhead | 0.20 ms/window mean |

### System comparison (all six systems, same 76 held-out windows)

| System | ANWG | 95% CI |
|---|---|---|
| best_fixed_policy | 0.3895 | [0.3233, 0.4556] |
| hard_selector | 0.3938 | [0.3281, 0.4597] |
| best_global_composition | 0.4025 | [0.3365, 0.4703] |
| unrestricted_contextual_predictor | 0.4019 | [0.3357, 0.4671] |
| **frozen_regime_specific_system** | **0.4044** | [0.3377, 0.4701] |
| oracle_composition | 0.4273 | [0.3618, 0.4939] |

The frozen envelope system's point estimate (0.4044) is the best of the
five non-oracle systems and edges out the unrestricted predictor (0.4019)
-- restricting to validated regimes and forcing fallback elsewhere
removed exactly the three regimes (`long_prompt`, `mixed_slo`,
`priority_conflict`) where the unrestricted predictor lost to global
composition, without sacrificing performance where the predictor was
strong.

### Per-regime (frozen system)

| Regime | n | In envelope | Frozen ANWG | Global ANWG | Fixed ANWG | Fallback rate |
|---|---|---|---|---|---|---|
| kv_pressure | 7 | Yes | 0.2241 | 0.2048 | 0.1992 | 0.00 |
| saturated | 7 | Yes | 0.3575 | 0.3459 | 0.3418 | 0.00 |
| burst_transition | 7 | Yes | 0.4835 | 0.4900 | 0.4343 | 0.29 |
| prediction_noise | 7 | Yes | 0.5735 | 0.5786 | 0.5550 | 0.14 |
| selective_admission_trap | 7 | Yes | 0.2429 | 0.2549 | 0.2324 | 0.14 |
| long_output | 7 | Yes | 0.0134 | 0.0077 | 0.0134 | 0.86 |
| underloaded | 7 | Yes | 0.9391 | 0.9391 | 0.9391 | 1.00 |
| long_prompt | 7 | No | 0.2549 | 0.2611 | 0.2549 | 1.00 |
| mixed_slo | 7 | No | 0.7379 | 0.7379 | 0.7059 | 1.00 |
| priority_conflict | 7 | No | 0.4974 | 0.4974 | 0.4867 | 1.00 |
| azure_conversation_like | 3 | No | 0.1277 | 0.1099 | 0.1277 | 1.00 |
| burstgpt_derived | 3 | No | 0.0262 | 0.0131 | 0.0262 | 1.00 |

Full table: `per_regime_summaries.csv`; per-window: `per_window_predictions.csv`.

## 6. Deployment Semantics

* **Determinism:** `select_with_frozen_envelope` makes no random draws at
  inference; the same `(gate, artifact, causal_row)` always returns the
  same decision (verified by test; `inference_overhead_s` is the only
  non-reproduced field, being a wall-clock measurement).
* **Versioning:** `FrozenEnvelopeGate.schema_version` (artifact-format
  version, currently 1) and `.envelope_version` (which specific envelope
  definition, currently 1) are both carried in every decision record and
  in `manifest.json` / `envelope_definition.json`.
* **Staleness rejection:** `assert_envelope_compatible` raises `CC5Error`
  on a missing gate or a `schema_version` mismatch (tested).
* **Logging:** every decision's `fallback_reason` records
  `regime_outside_envelope`, `ood`, `high_uncertainty`, or a combination;
  `used_predictor` / `abstained` / `in_envelope` / `uncertainty_ood_ok`
  are all logged per window in `per_window_predictions.csv`.
* **No held-out tuning:** the envelope (§4) uses only development-split
  LOWO evidence; the fallback rules
  (`fit_completion_safe_fallback_rules`) use only `VALIDATION`; neither
  touches `ID_TEST`/`OOD_TEST` until the single, final evaluation in §5.

## 7. Limitations (Statistically Documented)

* The frozen system's advantage over **best global composition** is a
  point-estimate edge only (+0.0019 to +00032 depending on subset); it is
  **not** statistically significant in any subset tested (paired CIs
  straddle zero, p >= 0.55 everywhere). Do not claim the frozen system
  "beats" global composition -- it ties it, with a marginal favorable
  point estimate.
* n=76 held-out windows (n=26 OOD) is a modest sample; per-regime paired
  tests (`paired_regime_analysis.csv`) are underpowered at n=7 per regime
  and mostly do not reach significance individually even where the
  aggregate advantage over fixed/hard-selector does.
* The envelope was derived from only 3 development windows per ID regime
  (`TRAIN`=2, `VALIDATION`=1) -- adequate to clear the `>= 2 windows`
  bar and to reproduce a directionally consistent LOWO signal, but not a
  large-sample guarantee; a regime could flip sides of the `>=` boundary
  under a different (but still legitimate) random dev/eval split.
* OOD-only regimes (`azure_conversation_like`, `burstgpt_derived`) have
  zero development windows and are excluded from the envelope by
  construction -- this is a structural limitation of CC4b's split design,
  not a discovered failure of the predictor.
* Fallback/abstention rate is high (57.9%) -- the frozen system is
  conservative by design; this is the intended trade-off for the
  completion-safety and no-held-out-tuning guarantees, not a defect.

## 8. Final CC5 Verdict

Applied `determine_final_cc5_verdict` (paired-statistics-based, not
point-estimate-only):

* completion_violations = 0
* beats_fixed (paired, significant) = **True**
* beats_global_overall (paired, significant) = **False**
* competitive_with_hard_selector (paired, significant win) = **True**
* trusted_regimes non-empty = **True** (7 regimes)

-> **`COMPLETE_REGIME_SPECIFIC`**. Full-context superiority over best
global composition was **not** established; the validated, restricted
envelope above is the deployable scope.

## 9. Restricted CC6 Entry Conditions

Per the task's instruction, CC6 is queued **only** in this restricted
form (issue #6 updated accordingly, still marked not-started):

> Evaluate controlled temporal adaptation only inside the validated CC5
> operating envelope (`burst_transition`, `kv_pressure`, `long_output`,
> `prediction_noise`, `saturated`, `selective_admission_trap`,
> `underloaded`), with hysteresis and fallback. Do not enable contextual
> switching in unsupported regimes (`azure_conversation_like`,
> `burstgpt_derived`, `long_prompt`, `mixed_slo`, `priority_conflict`).

CC6 implementation itself is **not** started in this query.

## 10. Artifacts

`results/cc5_final_operating_envelope/20260804T024524Z/` contains:

* `calibration_manifest.json` (reused calibrator, unchanged from
  uncertainty/regime refinement)
* `dev_lowo_table.csv` -- per-development-window LOWO comparison (§4)
* `envelope_definition.json` / `envelope_definition.csv` -- frozen gate +
  regime-by-regime derivation table
* `paired_statistical_analysis.csv` -- full paired-stats table (§3),
  both for the unrestricted predictor and the frozen system
* `paired_regime_analysis.csv` -- same, broken out per regime
* `per_window_predictions.csv`, `per_regime_summaries.csv`,
  `system_comparison.csv`
* `manifest.json`, `verdict.json`, `model_card.md`, `replay_commands.sh`

## 11. Tests

Focused tests in `tests/test_cc5_final_operating_envelope.py` (19 tests):
paired-statistics correctness (identical sequences -> zero, consistent
shift -> significant, deterministic permutation test, win/tie/loss sum,
Cohen's d edge case), envelope leakage-freedom (LOWO table never touches
evaluation windows, zero-dev-window regimes excluded by construction,
minimum-window rule, determinism), frozen-gate versioning
(stale/missing-gate rejection), deterministic + logged decisions,
regime-outside-envelope always falls back, in-envelope windows still
respect the uncertainty/OOD gate, and final-verdict logic (rejects
`COMPLETE_FULL` on a non-significant global comparison, `STOP_OR_REDESIGN`
on a completion violation, `INCONCLUSIVE` on an empty envelope).
