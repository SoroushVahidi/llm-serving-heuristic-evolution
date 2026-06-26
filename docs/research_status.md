# Research Status

**Last updated:** 2026-06-26  
**Current branch:** `phase2b13-selector-training-after-diversity`  
**Current phase:** Phase 2B.13 — Selector training on diversified workload suite

---

## Quick Facts

| Item | Value |
|---|---|
| Deployable scheduling policies | **20** |
| Non-deployable oracle policies | **1** (`oracle_srtf`) |
| Selector candidate policies | **20** (= deployable baselines) |
| Implemented selector models | 3 (`rule_based`, `decision_tree`, `random_forest`) |
| Test count | **992 passing**, 2 skipped (Phase 2B.13) |
| Phase 2B.13 windows evaluated | **256** (60 regression + 196 diversity) |
| SCORPIO label fraction (Phase 2B.13) | **43.75%** (112/256) |
| RF/DT training feasible | **Yes** (256≥200; 7 policies ≥10 wins; top=43.75%<85%) |
| RF test WG (heldout) | **0.9975** (gap=+0.0000 vs best fixed SCORPIO) |
| DT test WG (heldout) | 0.9752 (gap=−0.0224 vs best fixed; not recommended) |

---

## Important Branches / Commits

| Branch | Commit | Description |
|---|---|---|
| `phase2a4-2b4-final-eval` | `9ed8f71` | Final frozen evaluation (Phase 2A.4/2B.4) |
| `phase2b5-external-baselines` | `e1c6c01` | External baseline audit + completion_fraction metric |
| `phase2b5-admission-rule-selector-status` | `5d2afb6` | Admission control policy + feature rule selector |
| `phase2b6-fair-sweep-failure-audit` | `a6df363` | Fair sweep, failure tracking, correctness audit |
| `phase2b7-overload-failure-mining` | `992fe11` | Unit fix, overloaded sweep, multi-bin tests, failure registry |
| `phase2b8-rule-selector-repair` | `429e96e` | Rule selector repair: KV-pressure guard, noise guard, slo_slack_score for tight SLO |
| `phase2b9-selector-robustness-and-suite-freeze` | `5fe977b` | Selector robustness audit, held-out generalization, suite freeze |
| `phase2b10-scorpio-slo-guard` | `a9921b9` | SCORPIO-style SLO guard baseline (20th deployable policy) |
| `phase2b11-scorpio-selector-integration` | `6de9e2b` | SCORPIO integrated into rule selector; 3 new routing rules |
| `phase2b12-workload-diversity-selector-labels` | `93b6da7` | Workload diversity sweep for selector label analysis |
| `phase2b13-selector-training-after-diversity` | current | Selector training on diversified suite; RF trained; Rule 5 repaired |
| `main` | stale | Do not use; all work is on phase branches |

---

## Completed Work

### Phase 1 — Simulator + Baselines
- Deterministic step-based LLM serving simulator
- 10 Phase 1 baselines (FIFO, EDF, SOF, SPF, greedy fill, first/best fit, multi-bin, least loaded, random)

### Phase 1.5 — Serving-Style Baselines + Metrics
- 6 serving-style baselines: Orca-style, vLLM-style, Sarathi-style, SplitFuse-style, SLO-slack, WSPT
- TTFT / TPOT metrics
- BurstGPT + ShareGPT trace loading

### Phase 1.7 — Calibration
- RTX 5060 Ti calibrated with Qwen2.5-0.5B (prefill MAPE 9%, decode MAPE 12%)

### Phase 2A — Selector + Objective
- Selector dataset, features, labels, training
- RF selector WG=0.828 (+3pp vs. FIFO)
- Frozen shortlist and held-out CI computation

### Phase 2B — LLM Heuristic Search
- DSL verifier, LLM generation loop, controlled search
- Best generated heuristic: WG=0.9595 CI=[0.00, 0.27]

### Phase 2A.3B — Deadline/Laxity Baselines
- `least_laxity_first` (LLF) — 14 unit tests
- `estimated_service_time_first` (PARS-like SJF proxy) — 16 unit tests

### Phase 2B.5 — External Baseline Coverage
- External baseline audit (`docs/external_baseline_coverage_report.md`)
- Dataset/workload plan (`docs/dataset_workload_plan.md`)
- `completion_fraction` and `num_total` added to `RunMetrics`
- `admission_control` policy added (19th deployable baseline)
- `RuleBasedSelector` upgraded from FIFO placeholder to feature-based dispatch
- `scripts/report_research_status.py` for code-derived status
- 710 tests passing

### Phase 2B.6 — Fair Sweep + Failure Tracking
- Experiment/accounting infrastructure: committed CSV templates + `docs/experiment_tracking.md`
- Per-policy correctness audit for 11 baselines (`docs/external_baseline_correctness_audit.md`)
- `AdmissionControlPolicy` threshold calibration (`docs/audits/admission_control_threshold_calibration_summary.md`)
- Fair sweep: all 19 deployable policies × 3 workloads × 2 seeds (`configs/phase2b6_fair_sweep.yaml`)
- Sweep finding: underloaded workloads (10s, 10-20 req/s) produce zero WG differentiation — all policies tie
- `report_research_status.py` updated: template existence checks + `--check` mode validates templates
- New tests: leakage/fairness audit + registry template tests

### Phase 2B.13 — Selector Training After Diversity (current)

- **Goal:** Extend Phase 2B.12 suite to ≥200 windows, train RF and DT selectors, apply minimal
  rule selector repair (Rule 5: prefill-heavy → admission_control instead of sarathi_style).
- **Config:** `configs/phase2b13_selector_training_after_diversity.yaml` (25 workloads)
- **Runner:** `scripts/run_phase2b13_selector_training_after_diversity.py`
- **Log:** `logs/phase2b13/phase2b13_selector_training.log`
- **Tests:** 73 new tests in `tests/test_phase2b13_selector_training.py`; 992 total
- **tmux session:** `phase2b13_selector_training` (completed, EXIT_CODE=0, ~854s)
- **Audit docs:** `docs/audits/phase2b13_selector_training_summary.md`, `docs/audits/phase2b13_failure_cases_summary.md`

#### Phase 2B.13 Key Results

| Metric | Value |
|--------|-------|
| Total windows | **256** (60 regression + 196 diversity) |
| RF/DT training | **FEASIBLE** (all 3 criteria pass) |
| Best fixed WG (overall) | **0.9959** (SCORPIO, all groups) |
| Rule selector WG (overall) | **0.9694** (gap −0.0265) |
| Rule repaired WG (overall) | **0.9694** (no WG change; Rule 5 affects 1/256 windows) |
| RF WG (overall) | **0.9970** (gap +0.0011 vs best fixed) |
| RF WG heldout (test) | **0.9975** (gap +0.0000 vs best fixed) |
| DT WG heldout (test) | 0.9752 (gap −0.0224; not recommended) |
| SCORPIO label fraction | **43.75%** (112/256) |
| All-complete windows | **238/256 (93%)** — WG≈1.0 for all policies |
| Meaningful WG gap windows | 18/256 (7%) |
| Distinct oracle policies | **11** (7 win ≥10 windows) |
| Rule repair: Rule 5 | sarathi_style → admission_control (1/256 windows affected) |

#### Phase 2B.13 Selector Comparison Table

| Group | N | Best Fixed | Rule | Rule Repaired | RF | DT |
|---|---|---|---|---|---|---|
| dev | 27 | 0.9878 | 0.9168 | 0.9168 | 0.9881 | 0.9878 |
| heldout | 33 | 0.9975 | 0.9803 | 0.9803 | **0.9975** | 0.9752 |
| regression | 60 | 0.9932 | 0.9518 | 0.9518 | **0.9933** | 0.9809 |
| diversity | 196 | 0.9967 | 0.9747 | 0.9747 | **0.9982** | 0.9946 |
| overall | 256 | 0.9959 | 0.9694 | 0.9694 | **0.9970** | 0.9914 |

#### Phase 2B.13 Key Findings

1. **RF trained and feasible:** 256 windows; 7 policies win ≥10 windows each; top (SCORPIO) = 43.75%.
2. **RF ≈ best fixed:** RF test WG = 0.9975 matches best fixed SCORPIO exactly (+0.0000 gap).
   RF provides no WG improvement over always-SCORPIO in this all-complete regime.
3. **All-complete problem confirmed:** 238/256 windows (93%) have WG≈1.0 for all policies.
   Labels encode tie-breaking, not genuine WG differentiation. RF learns secondary-metric
   distinctions that happen to be WG-neutral.
4. **Rule repair WG-neutral:** Only 1/256 windows was dispatched to sarathi_style (oracle=AC);
   repair is semantically correct but has zero WG impact.
5. **DT inadequate:** DT test WG = 0.9752 (gap −0.0224). max_depth=8 insufficient for 11-class.
   RF is the correct ML selector; DT is discarded.

#### Phase 2B.13 Failure Cases

| ID | Description | Status |
|----|-------------|--------|
| fail_007 | Rule selector under-dispatches SCORPIO (offline artifact) | Carry-forward |
| fail_008 | Missing rule targets: best_fit, mbb, SOF, estST | Carry-forward (RF covers) |
| fail_009 | sarathi_style wrong for prefill-heavy | **Resolved** (Rule 5 → AC) |
| fail_010 | 172 < 200 window threshold | **Resolved** (256 windows) |
| fail_011 | All-complete tie-breaking labels | Partially addressed |
| fail_012 | All-complete regime limits RF signal (93% trivial) | Open — documented |
| fail_013 | DT underperforms on heldout (WG=0.9752) | Known — DT discarded |
| fail_014 | New workloads still mostly all-complete | Known — workload design issue |

### Phase 2B.12 — Workload Diversity for Selector Label Analysis

- **Goal:** Build ~200-window evaluation suite spanning diverse regimes (load, SLO pressure,
  token structure, KV pressure, noise, priority) where non-SCORPIO policies can win.
- **Motivation:** Phase 2B.11 found SCORPIO dominates all 60 Phase 2B.9/2B.10 windows →
  RF/DT training infeasible ("always choose SCORPIO").
- **Design:** 9 regression workloads (seeds 0-5, same as Phase 2B.11) + 14 new diversity
  workloads (seeds 6-9) targeting sarathi, WSP, EDF, AC as expected winners in different regimes.
- **Config:** `configs/phase2b12_workload_diversity_selector_labels.yaml` (23 workloads total)
- **Runner:** `scripts/run_phase2b12_workload_diversity_selector_labels.py`
- **Log:** `logs/phase2b12/phase2b12_workload_diversity.log`
- **Design doc:** `docs/audits/phase2b12_workload_diversity_design.md`
- **Results:** `results/phase2b12_workload_diversity_selector_labels/` (gitignored; see audit docs)
- **tmux session:** `phase2b12_workload_diversity` (completed, EXIT_CODE=0, ~638s)
- **Tests:** 36 new tests in `tests/test_phase2b12_workload_diversity.py`; 919 total

#### Phase 2B.12 Key Results

| Metric | Value |
|--------|-------|
| Total windows evaluated | **172** (60 regression + 112 diversity) |
| Deployable policies | **20** |
| Selector candidates | **20** |
| oracle_srtf excluded | Yes |
| Dev rule selector WG | 0.9168 (unchanged from Phase 2B.11) |
| Heldout rule selector WG | 0.9803 |
| Overall rule selector WG | **0.9721** |
| Best fixed WG overall | **0.9956** (SCORPIO, all groups) |
| Gap vs best fixed | **−0.024** overall (−0.041 regression, −0.014 diversity) |
| SCORPIO label fraction (overall) | **45.9%** (down from 100% in Phase 2B.11) |
| SCORPIO label fraction (regression) | 76.7% |
| SCORPIO label fraction (diversity) | 29.5% |
| Non-SCORPIO policies winning ≥10 windows | **5**: AC(29), best_fit(14), edf(14), SOF(13), estST(10) |
| Total distinct policies as oracle labels | **9** |
| RF/DT training feasible | **No — 172 < 200 window threshold** |
| Passes policy spread criterion | Yes (6 policies ≥10 wins) |
| Passes concentration criterion | Yes (top=45.9% < 85%) |

#### Phase 2B.12 Key Findings

1. **Label diversity substantially improved:** SCORPIO wins 45.9% of overall windows (vs 100%
   in Phase 2B.11). 9 distinct policies appear as oracle labels across 172 windows.
2. **RF/DT training NOT done:** 172 windows falls just short of 200-window threshold. Policy
   spread and concentration criteria both pass. ~28 more windows needed.
3. **Unexpected prefill winner:** `admission_control` wins all 16 prefill-heavy windows
   (designed for `sarathi_style`). AC's urgency sort outperforms chunked prefill under this WG objective.
4. **Throughput-packing gap:** `best_fit` (14×), `multi_bin_batching` (9×), and
   `estimated_service_time_first` (10×) win in loose-SLO / high-load regimes but are not in
   the current rule selector's policy choices.
5. **Many diversity wins are tie-breaking:** Most diversity workloads achieve WG=1.000 for all
   policies (underloaded/all-complete). Label diversity in these windows reflects tie-breaking
   order, not genuine performance differentiation.
6. **Rule selector regression confirmed identical:** dev WG=0.9168, heldout WG=0.9803 match
   Phase 2B.11 exactly on regression workloads.
7. **SCORPIO remains best fixed overall:** Even at 45.9% label frequency, SCORPIO's mean WG
   (0.9956) is the highest of any fixed policy, because it outperforms alternatives by large
   margins on overloaded windows and is competitive (or tied) elsewhere.

#### Phase 2B.12 Failure Cases

| ID | Description | Status |
|----|-------------|--------|
| fail_007 | Rule selector under-dispatches SCORPIO (2/172 vs 79/172 oracle) — offline artifact | Partially deferred |
| fail_008 | Missing rule targets: best_fit, multi_bin_batching, SOF, estST | Open |
| fail_009 | sarathi_style rule target wrong; AC wins prefill-heavy | Open |
| fail_010 | 172 < 200 window threshold; RF/DT training blocked | Open |
| fail_011 | All-complete diversity windows have tie-breaking labels | Open |

### Phase 2B.11 — SCORPIO Selector Integration
- **Rule selector update:** 3 new routing rules integrate `scorpio_style_slo_guard` into `RuleBasedSelector`
  - Rule 0: overloaded tight-SLO + recent violations → `scorpio_style_slo_guard`
  - Rule 2a: very high noise (pred_output_cv > 2.0) → `scorpio_style_slo_guard` (fail_004 fix)
  - Rule 3: standalone recent violations → `scorpio_style_slo_guard` (was AC)
- **Selector policy choices:** 7 (was 6); `scorpio_style_slo_guard` added to `_POLICY_CHOICES`
- **Experiment:** Phase 2B.9/2B.10 workload suite re-run with updated selector
- **Tests:** 43 rule selector tests, 20 Phase 2B.11 tests; see `tests/test_rule_based_selector.py`, `tests/test_phase2b11_scorpio_selector_integration.py`
- Config: `configs/phase2b11_scorpio_selector_integration.yaml`
- Runner: `scripts/run_phase2b11_scorpio_selector_integration.py`
- Log: `logs/phase2b11/phase2b11_scorpio_selector_integration.log`
- Summary: `docs/audits/phase2b11_scorpio_selector_integration_summary.md`

### Phase 2B.10 — SCORPIO-Style SLO Guard
- **New policy:** `scorpio_style_slo_guard` — SCORPIO-inspired TTFT/TPOT guard with credit throttling
- **Registry:** 20 deployable policies, 20 selector candidates; `oracle_srtf` still excluded
- **Comparison results:** SCORPIO-style WG dev=0.988, held-out=0.998, overall=0.993; becomes best fixed baseline; rule selector no longer beats best fixed (gap −0.042 overall). See `docs/audits/phase2b10_scorpio_slo_guard_summary.md`
- **Failure cases:** `docs/audits/phase2b10_failure_cases_summary.md` (selector vs SCORPIO gap; high-noise s4 fixed for SCORPIO)

### Phase 2B.9 — Selector Robustness Audit and Suite Freeze
- **Selector training data audit**: Phase 2A.4 RF/DT use ~30 training windows (critically small for 19-class).
  KV-pressure and high-noise regimes missing from training data.
  See `docs/audits/phase2b9_selector_training_audit.md` for full analysis.
- **Broader robustness experiment**: 4 dev workloads (same as Phase 2B.7/2B.8, seeds 0–2) +
  5 held-out workloads (new regimes, seeds 3–5). Config: `configs/phase2b9_selector_robustness.yaml`.
  Runner: `scripts/run_phase2b9_selector_robustness.py`. Log: `logs/phase2b9/`.
- **External baseline decisions**: 5 must-add baselines identified (SCORPIO-style, KV-cache-aware,
  FairBatching, PARS-LTR, PROSERVE SlideBatching). See `docs/external_baseline_decision.md`.
- **Dataset/workload decisions**: BurstGPT (full), Azure LLM Inference 2023, LMSYS-Chat-1M,
  LongBench, and calibrated synthetics are must-use before publication.
  See `docs/dataset_workload_decision.md`.
- **Caveats and claims**:
  - Phase 2B.8 repaired rule selector matched best fixed baseline on 4 **development** workloads.
    This is **not** a final generalization claim (same workloads used for rule design and evaluation).
  - Phase 2B.9 evaluates the selector on held-out workloads (5 new + different seeds) for the first time.
  - RF/DT selectors have 30 training windows; insufficient for strong publication claims.
  - Final publication requires: ≥200 training windows, KV-pressure/noise regimes in training,
    real-trace BurstGPT/Azure data, and must-add external baselines.
- **Tests added**: `tests/test_phase2b9_selector_robustness.py`
  (rule selector dispatch table, oracle exclusion, config validation, doc existence checks)
- **Robustness results analyzed**: `docs/audits/phase2b9_selector_robustness_summary.md`
  - Dev WG=0.917, held-out WG=0.979, overall WG=0.951 (rule selector)
  - Beats best fixed on dev (+0.024) and held-out (+0.008); within 0.5 pp of per-window oracle on held-out
  - RF/DT not re-evaluated (Phase 2A.4 model artifacts absent in this run)
- **Failure cases**: `docs/audits/phase2b9_failure_cases_summary.md`
  - 1 unresolved: `heldout_very_high_noise_s4` (AC vs EDF, gap ≈ −0.023)
- Log: `logs/phase2b9/phase2b9_selector_robustness.log` (gitignored)

### Phase 2B.8 — Rule Selector Repair Under KV Pressure
- **Root cause**: Phase 2B.7 showed Rule 1 (`tight_slo/min_slack → least_laxity_first`) fired for
  all 3 differentiated workloads; LLF catastrophic under KV pressure (WG=0.101 vs WSP=0.477)
- **Repair**: Three changes to `RuleBasedSelector.predict_one()`:
  1. New Rule 1 (elevated): `mean_pred_output_tokens > 200 OR kv_utilization > 0.7 → weighted_shortest_processing`
     (KV pressure proxy: large outputs fill KV cache; WSP avoids urgency-induced cascade)
  2. New Rule 2: `pred_output_cv > 1.0 → admission_control`
     (High noise: laxity estimates unreliable under 70%+ prediction noise)
  3. Rule 4 (was Rule 1): tight SLO now → `slo_slack_score` instead of `least_laxity_first`
     (composite urgency+throughput score; avoids LLF throughput collapse under overload)
- **Policy choices reduced**: `_POLICY_CHOICES` from 7 to 6 (removed `vllm_style_token_budget` and `least_laxity_first`)
- Tests: 32 rule selector tests pass (including 3 failure-case regression tests)
- Total tests: 784 passing
- Sweep: same 4 workloads × 19 policies × 3 seeds as Phase 2B.7 (apples-to-apples)

### Phase 2B.7 — Unit Fix + Overloaded Failure Mining
- **Unit fix**: `AdmissionControlPolicy._laxity()` now unit-consistent (seconds); `laxity_threshold` in seconds
- New parameter `step_size=0.001` for service proxy → seconds conversion
- **Multi-bin unit tests**: `tests/test_multi_bin_batching_policy.py` (18 tests)
- **Overloaded sweep**: all 19 policies × 4 regimes × 3 seeds (`configs/phase2b7_overload_failure_mining.yaml`)
  - 3/4 workloads show substantial policy differentiation (WG range 0.43–0.56)
  - `prefill_heavy_small` still underloaded (all tie WG=1.0)
- **3 failure cases identified** in `results/failure_cases/failure_case_registry.csv`
  - Root cause: Rule 1 (`min_slack < 1.0s`) fires for all overloaded workloads → always picks LLF
  - LLF catastrophically bad in kv_pressure_decode_heavy (WG=0.101 vs best=0.477)
- Best fixed baseline overall: `weighted_shortest_processing` WG=0.827
- Rule-based selector overall WG ≈ 0.540 (−0.287 vs best fixed)
- Admission control (post unit-fix, threshold=inf): wins high_prediction_noise (rank 1), loses kv_pressure (rank last)

---

## Implemented Deployable Scheduling Policies (20)

All 20 are in `BASELINE_NAMES` and `SELECTOR_CANDIDATE_NAMES`.  
None have access to `actual_output_tokens` (oracle excluded).

| # | Policy name | Category | SLO-aware | KV-budget aware |
|---|---|---|---|---|
| 1 | `fifo` | Classical | No | No |
| 2 | `edf` | Classical | Yes | No |
| 3 | `shortest_output_first` | SRPT-style | No | No |
| 4 | `shortest_prompt_first` | Heuristic | No | No |
| 5 | `greedy_token_fill` | Packing | No | Yes |
| 6 | `least_loaded` | Load balancing | No | No |
| 7 | `multi_bin_batching` | Batching | No | No |
| 8 | `random_feasible` | Stochastic | No | No |
| 9 | `first_fit` | Packing | No | Yes |
| 10 | `best_fit` | Packing | No | Yes |
| 11 | `orca_style` | Serving-style (inspired) | Partial | Yes |
| 12 | `vllm_style_token_budget` | Serving-style (inspired) | No | Yes |
| 13 | `sarathi_style` | Serving-style (inspired) | No | Yes |
| 14 | `splitfuse_style` | Serving-style (inspired) | No | Yes |
| 15 | `slo_slack_score` | Composite | Yes | No |
| 16 | `weighted_shortest_processing` | WSPT | No | No |
| 17 | `least_laxity_first` | Deadline/laxity | Yes | No |
| 18 | `estimated_service_time_first` | SJF proxy | Partial | No |
| 19 | `admission_control` | Admission control | Yes | Yes |
| 20 | `scorpio_style_slo_guard` | SCORPIO-inspired SLO guard | Yes | Yes |

---

## Non-Deployable Oracle Policies (1)

| Policy | File | Notes |
|---|---|---|
| `oracle_srtf` | `policies/oracle.py` | Hindsight SRTF oracle. Uses `actual_output_tokens`. **Never** in selector candidates. Access via `make_oracle_policy()` only. Always emits `UserWarning`. |

The selector must never choose `oracle_srtf`. This invariant is enforced by `selector/candidates.py` and tested by multiple tests.

---

## Selector Candidate Set

`SELECTOR_CANDIDATES` = `BASELINE_NAMES` minus `ORACLE_POLICY_NAMES`.

Currently: **20 candidates** (all 20 deployable policies above).

Verified at import time by `selector/candidates.py` assertion loop.

---

## Implemented Selector Models (3)

| Model | File | Type | Notes |
|---|---|---|---|
| `rule_based` | `selector/models.py:RuleBasedSelector` | Feature-based rules | No training needed. Dispatches to 7 different policies based on workload features. Was a FIFO placeholder before Phase 2B.5. |
| `decision_tree` | `selector/models.py:DecisionTreeSelector` | sklearn DT | max_depth=8, min_samples_leaf=20. Requires sklearn. |
| `random_forest` | `selector/models.py:RandomForestSelector` | sklearn RF | n_estimators=200, max_depth=10. Primary Phase 2A.4 selector. WG=0.828. |

### Rule-Based Selector Dispatch Logic (Phase 2B.5)

The selector picks policies based on these observable features (in priority order):

1. `fraction_tight_slo > 0.4` or `min_slack < 1.0` → `least_laxity_first`
2. `recent_slo_violation_rate > 0.3` → `admission_control`
3. `kv_utilization > 0.7` → `vllm_style_token_budget`
4. `mean_prompt_tokens > 512` or `p95_prompt_tokens > 1024` → `sarathi_style`
5. `mean_pred_output_tokens < 64` and `pred_output_cv < 0.5` → `estimated_service_time_first`
6. `burstiness_cv > 1.5` → `slo_slack_score`
7. default → `edf`

---

## Missing External Baselines

| Baseline | Status | Notes |
|---|---|---|
| CP-SAT / ILP oracle | ❌ Missing | Requires `ortools`; only useful for micro-traces (< 20 reqs) |
| True PARS (learning-to-rank) | ❌ Not implemented | Our `estimated_service_time_first` is a simplified proxy; true PARS needs learned ranking |
| Preemption-based SJF | ❌ Not implemented | Requires preemption support in simulator (Phase 2+ scope) |

---

## Missing Workloads / Datasets

| Dataset | Status | Notes |
|---|---|---|
| LMSYS-Chat-1M / ShareGPT raw | ❌ Not downloaded | Loader exists in `workloads/sharegpt.py`; data download pending |
| LongBench | ❌ Not downloaded | Synthetic long-context generator exists; LongBench data not needed yet |

All synthetic and BurstGPT workloads already exist in repo.

---

## Reporting Gaps

| Gap | Status | Notes |
|---|---|---|
| `completion_fraction` in reports | ✅ Added (Phase 2B.5) | `num_total` and `completion_fraction` now in `RunMetrics` and CSV output |
| Selective vs. forced-admission labels | Partial | Field added; report headers not yet updated |
| Multi-bin batching unit tests | ⚠️ Registry check only | Dedicated `test_multi_bin_batching_policy.py` not yet added |

---

## API-Credit Policy

**No paid API calls in Phase 2B.5 (or Phase 2B.5-admission).**

CloudRift, Cohere, OpenAI, Gemini, Mistral, and all other paid APIs are not called in:
- Any test
- Any script run during CI or development
- Any default script behavior

Paid API usage is gated behind explicit `--use-llm` flags or dedicated scripts.

---

## Phase 2B.6 Findings

| Finding | Detail |
|---|---|
| Cheap sweep (10s, 10-20 req/s): zero differentiation | All 19 policies achieve WG ≈ 0.9977-1.0 under underloaded conditions |
| AdmissionControl unit mismatch | `_laxity()` mixes seconds and steps; default `threshold=inf` is safe |
| Threshold calibration: `threshold=200.0` | Drops ~21% of requests, achieves WG=1.0, zero SLO violations |
| Historical rule_based failure (Phase 2A.4) | FIFO placeholder: WG=0.597 vs. best_fixed 0.798 (−0.20); now fixed in Phase 2B.5 |

---

## Admission/Completion Accounting Limitation

The simulator tracks:
- `num_total` = all arrivals
- `num_completed` = requests serviced before drain
- `num_dropped` = arrivals − completed (never serviced OR timed out)
- `completion_fraction` = num_completed / num_total

**Not tracked:**
- `num_admitted` = requests explicitly accepted for service in some step
- `num_rejected` = requests explicitly filtered by admission control policy
- `admission_fraction` = num_admitted / num_total
- `conditional_completion_fraction` = num_completed / max(num_admitted, 1)

**Limitation:** In the current simulator, "dropped" conflates requests filtered by admission
control with requests that arrived but were never scheduled before simulation ended. The 
simulator does not distinguish these. For `admission_control` with `threshold=inf` (default),
`num_dropped` = requests that ran out of simulation time. For `threshold=0.0s`, dropped
includes genuinely-filtered infeasible requests — but the metrics don't distinguish them.

**TODO:** Add `num_rejected` tracking when `AdmissionControlPolicy` explicitly filters a request
(not just delays it). This requires simulator-level hooks or policy instrumentation.

---

## Next Planned Tasks

### Completed in Phase 2B.9 ✅
- Selector training data sufficiency and leakage audit (`docs/audits/phase2b9_selector_training_audit.md`)
- Broader robustness experiment (9 workloads: 4 dev + 5 heldout, seeds 0–5)
- Robustness results analysis (`docs/audits/phase2b9_selector_robustness_summary.md`)
- Failure case documentation (`docs/audits/phase2b9_failure_cases_summary.md`)
- External baseline decision document (`docs/external_baseline_decision.md`)
- Dataset/workload decision document (`docs/dataset_workload_decision.md`)
- Tests for oracle exclusion, rule dispatch, config validation, doc existence

### Remaining Before Submission
1. **Update rule selector / RF-DT for 20 policies** — route to `scorpio_style_slo_guard` where best (Phase 2B.10 shows −0.042 overall gap vs new best fixed); re-run robustness.
2. **Expand selector training data** — add KV-pressure + high-noise + real-trace windows to reach ≥200 training windows; re-train and re-evaluate RF/DT on Phase 2B.9/2B.10 suite.
3. **Implement remaining must-add baselines** (see `docs/external_baseline_decision.md` section B):
   - B.3 KV-cache-aware scheduler (WAIT/Jaillet-style)
   - B.4 FairBatching
   - B.1 PARS-style LTR (requires training data)
   - B.5 PROSERVE SlideBatching
   - ~~B.2 SCORPIO-style~~ ✅ implemented Phase 2B.10
4. **Ingest real-trace datasets** (see `docs/dataset_workload_decision.md` section B):
   - BurstGPT full dataset (currently using 10k subset only)
   - Azure LLM Inference 2023 / Splitwise trace
   - LMSYS-Chat-1M length statistics for calibrated synthetic
   - LongBench for long-context stress workloads
5. **LLM escalation** (CloudRift/Cohere): if further rule improvement needed,
   generalization analysis, synthesize updated rules. Log in API ledger. Limit: 1–2 calls per pattern.
