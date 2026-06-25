# Phase 2B.7 Resume Audit

**Date:** 2026-06-25  
**Branch:** `phase2b7-overload-failure-mining`  
**Audited by:** Claude Sonnet 4.6 (session resume)

---

## What was already done

Phase 2B.7 was **fully completed** in commit `992fe11` before this resume session.

### Completed items

| Item | Status | Evidence |
|---|---|---|
| Branch `phase2b7-overload-failure-mining` exists | ✓ | local branch present |
| Uncommitted work | None | `git status` clean |
| Phase 2B.7 commit exists | ✓ | `992fe11` |
| `results/phase2b7_overload_failure_mining/` populated | ✓ | 4 workload dirs + summary files |
| `docs/audits/phase2b7_overload_failure_mining_summary.md` | ✓ | 148-line summary with per-workload breakdown |
| `docs/audits/phase2b7_failure_cases_summary.md` | ✓ | 3 failure cases documented |
| `results/failure_cases/failure_case_registry.csv` | ✓ | 3 entries (fail_001–fail_003) |
| AdmissionControl unit fix | ✓ | `src/llmserveopt/policies/admission_control.py` fixed |
| Admission control unit tests (18 new) | ✓ | `tests/test_admission_control_policy.py` |
| Multi-bin batching direct unit tests (18 new) | ✓ | `tests/test_multi_bin_batching_policy.py` |
| `configs/phase2b7_overload_failure_mining.yaml` | ✓ | 4 workloads × 19 policies × 3 seeds |
| `docs/research_status.md` updated for Phase 2B.7 | ✓ | sections added |
| `scripts/report_research_status.py` updated | ✓ | unit-fix + failure-case detection added |
| `docs/baselines.md` updated | ✓ | admission-control unit fix documented |
| `docs/audits/admission_control_threshold_calibration_summary.md` updated | ✓ | |
| 770 tests passing | ✓ | confirmed in resume session |

### Missing at resume time

- `docs/audits/phase2b7_resume_audit.md` — this file (being created now)
- Branch not yet pushed to `origin`

---

## What was missing

Only the resume audit document and the remote push were outstanding.

---

## What was decided / completed in this resume session

1. Confirmed all 770 tests still pass.
2. Confirmed the phase2b7 commit (`992fe11`) includes all required deliverables.
3. Created this resume audit document.
4. Pushed `phase2b7-overload-failure-mining` to `origin`.

---

## Uncommitted work at resume

None. Working tree was clean on entry.

---

## Branch status

- Branch existed locally: **yes** (created in the prior session)
- Branch existed on remote: **no** (pushed in this resume session)
- Base commit: `a6df363` (phase2b6-fair-sweep-failure-audit)

---

## Results status

Results were already present under `results/phase2b7_overload_failure_mining/`:

```
overloaded_mixed_slo/
high_prediction_noise/
overloaded_prefill_heavy/
kv_pressure_decode_heavy/
per_run.csv
per_run.jsonl
summary.csv
summary.json
README.md
```

The `results/failure_cases/failure_case_registry.csv` already contained 3 entries.

---

## Key findings (from prior session, summarized here)

- **AdmissionControl unit fix**: laxity now computed in seconds via `step_size`; threshold=0.0s
  correctly filters infeasible requests; default remains `inf` (no filtering).
- **Best fixed baseline**: `weighted_shortest_processing` (mean WG=0.827 across 4 workloads)
- **Selector failure**: Rule-based selector always fires Rule 1 → `least_laxity_first` (WG≈0.540);
  delta vs best fixed = −0.287.
- **3 failure cases**: all `wrong_rule_fired` pattern; Rule 1 threshold too broad.
- **Recommended next step**: Phase 2B.8 — LLM-assisted rule synthesis targeting the 3 failure patterns,
  or implementation of modern external baselines.
