# Phase 2B.9 Failure Case Summary

**Phase:** 2B.9  
**Date:** 2026-06-25  
**Experiment:** `phase2b9_selector_robustness`  
**Config:** `configs/phase2b9_selector_robustness.yaml`  
**Seeds (held-out group):** 3, 4, 5  

---

## Summary

Phase 2B.9 held-out evaluation identified **one unresolved failure case** for the repaired
rule selector. All Phase 2B.7/2B.8 development failure cases remain resolved on the dev group.

| failure_id | workload | selector_policy | selector_wg | best_fixed | best_wg | delta | status |
|---|---|---|---|---|---|---|---|
| fail_004 | `heldout_very_high_noise_s4` | `admission_control` | 0.970 | `edf` | 0.993 | **−0.023** | **unresolved** |

**Phase 2B.9 status:** 1 unresolved held-out failure. No LLM escalation in Phase 2B.9.  
**Phase 2B.8 status:** ✓ Phase 2B.7 dev failures (fail_001–003) remain resolved on dev group.

---

## fail_004: heldout_very_high_noise_s4

### Observed behavior

| Field | Value |
|-------|-------|
| Workload tag | `heldout_very_high_noise_s4` |
| Regime | Very high prediction noise (90% `prediction_noise_rel`) |
| Seed | 4 (held-out seed; not in dev seeds 0–2) |
| Windows | 2 |
| Selector choice | `admission_control` |
| Selector mean WG | 0.970 |
| Best fixed policy | `edf` |
| Best fixed WG | 0.993 |
| Gap | ≈ −0.023 (−2.3 pp) |

### Suspected pattern

1. **Very high prediction noise (90%)** — harder than the dev workload (`dev_high_prediction_noise`
   at 70% noise), where the selector correctly picks `admission_control` and matches best fixed.
2. **Admission-control over-selection** — Rule 2 (`pred_output_cv > 1.0 → admission_control`)
   fires as designed, but at 90% noise with seed 4 the admission filter may be too aggressive,
   dropping or delaying requests that `edf` would serve in time.
3. **EDF better under this noise regime** — for this specific seed/window mix, deadline-aware
   FIFO-style urgency (`edf`) outperforms admission throttling. The optimal policy is not
   admission control despite high CV.

### Comparison with sibling held-out seeds

| Workload | Selector policy | Selector WG | Best fixed | Best fixed WG | Match? |
|----------|-----------------|-------------|------------|---------------|--------|
| `heldout_very_high_noise_s3` | `admission_control` | 0.996 | `admission_control` | 0.996 | ✓ |
| `heldout_very_high_noise_s4` | `admission_control` | 0.970 | `edf` | 0.993 | ✗ |
| `heldout_very_high_noise_s5` | `admission_control` | 0.983 | `edf` | 0.990 | partial (−0.007) |

The failure is **seed-sensitive** at the 90% noise boundary — not a systematic collapse, but
a real gap that should not be hidden.

### Root cause hypothesis

Rule 2 treats `pred_output_cv > 1.0` as a binary switch to `admission_control`. At extreme
noise levels, CV remains high across seeds, but the **cost** of admission filtering varies with
arrival realizations. Seed 4 produces windows where AC's drop/delay behavior hurts more than
EDF's steady deadline sorting.

Possible fixes (future work — not in Phase 2B.9 scope):

- Add a noise **gradient** rule: at CV just above 1.0, prefer `edf` or `slo_slack_score`; reserve
  `admission_control` for CV ≫ 1.0 or combined with high `recent_slo_violation_rate`.
- Calibrate Rule 2 threshold on held-out noise sweeps (70%, 90%, 95%) without tuning on dev seeds.
- LLM-assisted rule synthesis (1–2 API calls, ledger-logged) if manual threshold tuning stalls.

---

## Resolved on Dev Group (Regression Check)

All four Phase 2B.7/2B.8 development workloads pass on seeds 0–2:

| Dev workload | Selector WG | Best fixed WG | Match best fixed? |
|--------------|-------------|---------------|-------------------|
| `dev_kv_pressure_decode_heavy` | 0.662 | 0.662 (WSP) | ✓ |
| `dev_overloaded_mixed_slo` | 0.987 | 0.987 (EDF) | ✓ |
| `dev_high_prediction_noise` | 0.984 | 0.986 (EDF) | partial (−0.002) |
| `dev_overloaded_prefill_heavy` | 1.000 | 1.000 (FIFO) | ✓ |

---

## Registry Cross-Reference

This failure is **not** yet added to `results/failure_cases/failure_case_registry.csv`
(gitignored template at `docs/templates/failure_case_registry.csv`). Add at next registry update:

```
fail_004,heldout_very_high_noise_s4,admission_control,0.970,edf,0.993,-0.023,unresolved,very_high_noise_admission_overselection,phase2b9
```

---

## Next Steps

1. Do **not** claim zero selector failures in publication text.
2. Prioritize SCORPIO-style SLO guard and noise-regime sweeps before further rule tuning.
3. If LLM escalation is used for Rule 2 refinement, limit to 1–2 calls and log in API ledger.
