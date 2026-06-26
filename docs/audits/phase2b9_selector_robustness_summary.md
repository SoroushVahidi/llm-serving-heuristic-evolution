# Phase 2B.9 Selector Robustness Summary

**Phase:** 2B.9  
**Date:** 2026-06-25  
**Branch:** `phase2b9-selector-robustness-and-suite-freeze`  
**Config:** `configs/phase2b9_selector_robustness.yaml`  
**Runner:** `scripts/run_phase2b9_selector_robustness.py`  
**Log:** `logs/phase2b9/phase2b9_selector_robustness.log` (gitignored)  
**Full results:** `results/phase2b9_selector_robustness/` (gitignored; summaries in this doc)

---

## Experiment Purpose

Phase 2B.9 is the **first held-out generalization test** for the Phase 2B.8 repaired rule selector.
Phase 2B.8 repaired the rule dispatch table using the same four development workloads that
motivated the repair (Phase 2B.7 failure cases). Phase 2B.9 asks whether that repair
generalizes to workloads and seeds **not used** during rule design.

The experiment also freezes comparison-suite decisions (external baselines, datasets) and
audits selector training-data sufficiency (`docs/audits/phase2b9_selector_training_audit.md`).

**Selectors evaluated in this run:**
- `rule_based` — Phase 2B.8 repaired rule selector (primary result)
- `random_forest` / `decision_tree` — **not loaded** in this run (Phase 2A.4 model artifacts
  absent at `results/phase2a4_2b4_final_eval/selector_models/`). RF/DT Phase 2A.4 test-split
  numbers (WG ≈ 0.828) remain historical evidence only; they were **not re-evaluated** on the
  Phase 2B.9 dev/held-out workload suite.

---

## Workload Groups

### Development / regression (4 workloads, seeds 0–2)

Same workloads as Phase 2B.7/2B.8 — confirms Phase 2B.8 fixes still hold:

| Tag | Regime |
|-----|--------|
| `dev_overloaded_mixed_slo` | Overloaded mixed-SLO (Phase 2B.7 fail_001) |
| `dev_high_prediction_noise` | 70% prediction noise (Phase 2B.7 fail_002) |
| `dev_kv_pressure_decode_heavy` | KV-pressure decode-heavy (Phase 2B.7 fail_003) |
| `dev_overloaded_prefill_heavy` | Prefill-heavy (underloaded; tracking only) |

**Windows:** 27

### Held-out robustness (5 workloads, seeds 3–5)

Not used to design or tune any selector rule:

| Tag | Regime |
|-----|--------|
| `heldout_moderate_kv_pressure` | output_mean=150 (near KV threshold=200) |
| `heldout_very_high_noise` | 90% prediction noise (harder than dev) |
| `heldout_prefill_overloaded` | Prefill-heavy at arrival_rate=80 (actually overloaded) |
| `heldout_bursty_mixed_slo` | Bursty arrivals + mixed SLO |
| `heldout_burstgpt_smoke` | BurstGPT real-trace smoke (400 requests) |

**Windows:** 33

**Overall windows:** 60

---

## Repaired Rule Selector Results

| Group | Windows | Rule selector WG | Accuracy vs per-window best |
|-------|---------|------------------|----------------------------|
| Dev | 27 | **0.917** | 25.9% |
| Held-out | 33 | **0.979** | 6.1% |
| Overall | 60 | **0.951** | 15.0% |

**Policy distribution (rule selector):**

| Group | slo_slack_score | admission_control | weighted_shortest_processing |
|-------|-----------------|-------------------|------------------------------|
| Dev | 15 | 6 | 6 |
| Held-out | 25 | 6 | 2 |

---

## Best Fixed Baseline Results

| Group | Best fixed policy | Best fixed WG |
|-------|-------------------|---------------|
| Dev | `weighted_shortest_processing` | **0.893** |
| Held-out | `edf` | **0.970** |
| Overall | `weighted_shortest_processing` | **0.922** |

---

## Gaps

| Metric | Dev | Held-out | Overall |
|--------|-----|----------|---------|
| Selector WG − best fixed WG | **+0.024** | **+0.008** | **+0.028** |
| Selector WG − per-window oracle/reference WG | −0.001 | **−0.005** | −0.003 |

*Per-window oracle/reference* = mean WG of the best deployable policy per window (oracle_srtf excluded).

---

## Remaining Failure Case

| Field | Value |
|-------|-------|
| Workload | `heldout_very_high_noise_s4` |
| Selector choice | `admission_control` |
| Selector WG | **0.970** |
| Best fixed | `edf` |
| Best fixed WG | **0.993** |
| Gap | **≈ −0.023** |
| Status | Unresolved — see `docs/audits/phase2b9_failure_cases_summary.md` |

All other held-out workload×seed combinations matched or beat best fixed.

---

## Interpretation

### Does the repaired rule selector generalize?

**Yes.** On held-out workloads (33 windows, seeds 3–5, five new regime families), the repaired
rule selector achieves WG = **0.979**, beating best fixed (WG = 0.970, `edf`) by **+0.008**
and landing within **0.5 pp** of per-window oracle/reference (gap = **−0.005**).

This is the first evidence that the Phase 2B.8 repair generalizes **beyond** the four
Phase 2B.7/2B.8 development workloads. The result is **promising** and supports the
workload-adaptive selector as a potential paper contribution.

### Publication readiness caveats

This is **not** final publication evidence:

1. **Modern external baselines** (SCORPIO-style SLO guard, KV-cache-aware scheduler,
   FairBatching, PARS-style LTR, PROSERVE SlideBatching) are not yet implemented.
   See `docs/external_baseline_decision.md`.
2. **Broader real-trace datasets** (full BurstGPT, Azure 2023, LongBench, LMSYS length stats)
   are not yet ingested. See `docs/dataset_workload_decision.md`.
3. **RF/DT selectors** were trained on ~30 windows without KV-pressure or high-noise regimes.
   They were **not re-run** in Phase 2B.9 (model artifacts missing). Historical Phase 2A.4
   test WG = 0.828 must not be cited as Phase 2B.9 generalization evidence.
4. **One unresolved failure** (`heldout_very_high_noise_s4`) suggests admission-control
   over-selection at extreme noise; reserved for future rule/LLM-synthesis work.
5. Dev-group improvement (+0.024 vs best fixed) uses the **same workloads** that motivated
   the repair — treat as regression confirmation, not independent generalization.

### Recommended claims (safe wording)

- "On 60 evaluation windows spanning 9 workload families, the repaired feature-based rule
  selector achieves weighted goodput within 0.5 pp of per-window best-fixed on held-out
  workloads and beats best-fixed by +2.8 pp overall."
- "Held-out generalization (5 new workloads, seeds 3–5) supports the selector design; final
  claims require modern baselines and real-trace evaluation."

---

## Related Documents

| Document | Purpose |
|----------|---------|
| `docs/audits/phase2b9_selector_training_audit.md` | Training data sufficiency and leakage |
| `docs/audits/phase2b9_failure_cases_summary.md` | Unresolved failure cases |
| `docs/audits/phase2b8_rule_selector_repair_summary.md` | Rule repair that Phase 2B.9 validates |
| `docs/external_baseline_decision.md` | Must-add baselines before submission |
| `docs/dataset_workload_decision.md` | Must-use datasets before submission |
