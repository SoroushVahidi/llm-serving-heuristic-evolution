# Hierarchical Regime Router v1 — Live Closed-Loop Scientific Re-evaluation Design and Preregistration

Date: 2026-08-18

## 0. Scope

**DESIGN / PREREGISTRATION ONLY.** 
No Stage-1 router is retrained, no Stage-2 selector is retuned, no new models are fitted on the TEST split, no scientific run is launched, and no new scientific TEST verdict is computed or implied in this document. 

This document defines and freezes every scientific and methodological choice for the live closed-loop re-evaluation of the Hierarchical Regime Router v1. This follows the completion and validation of the live closed-loop evaluation harness (verdict `LIVE_HIERARCHICAL_HARNESS_READY`, documented in [`hierarchical_router_live_harness_validation_v1_20260818.md`](../audits/hierarchical_router_live_harness_validation_v1_20260818.md)), which resolved the scenario-level majority-vote integration bottleneck identified in the original approximate evaluation (verdict `HIERARCHICAL_ROUTER_NO_GO`, documented in [`hierarchical_regime_router_v1_20260818.md`](../audits/hierarchical_regime_router_v1_20260818.md)).

The frozen original `HIERARCHICAL_ROUTER_NO_GO` verdict, the original approximate evaluation audit, the original `HIERARCHICAL_REGIME_ROUTER_V1.md` design, the original gates JSON, and the frozen router/policy implementations **MUST and DO remain preserved exactly as written**. Nothing here alters them; this is a brand new, separately pre-registered evaluation protocol designed to isolate methodology as the independent variable.

---

## 1. Precisely Defining the Scientific Question

The primary scientific question this re-evaluation will answer is:

> **Does the same frozen hierarchical router that received `HIERARCHICAL_ROUTER_NO_GO` under the offline scenario-majority approximation show useful end-to-end behavior when evaluated causally in a genuine per-step closed-loop simulator?**

This is strictly a **METHODOLOGY RE-EVALUATION**. It is designed to test the majority-vote artifact hypothesis—namely, that the offline evaluation's collapse of a step-by-step trajectory into a single modal label destroyed critical minority-regime signals (especially under KV-memory pressure), thereby masking the true capability of the hierarchical router.

It is **NOT** and must not involve:
- A new router model or newly trained weights
- A retuned router or modified thresholds
- A new policy library or newly added heuristics
- A new Stage-1 classifier architecture
- A new Stage-2 selector structure
- A new dwell rule or fallback mechanism

---

## 2. Frozen Elements (Must Remain Identical)

To preserve the scientific integrity of the comparison, every single element of the hierarchical router system under test is frozen to match the original `HIERARCHICAL_REGIME_ROUTER_V1` specification exactly:

### A. Stage-1 Router
- **Model Class & Settings**: Multiclass Logistic Regression (`multiclass_classifier_5way`), fit exclusively on TRAIN telemetry.
- **Stage-1 Inputs**: Exactly the four online-observable columns:
  1. `contention_score_v2`
  2. `priority_skew`
  3. `kv_pressure`
  4. `queue_length`
- **Stage-1 Target Semantics**:
  - `RANKING_FAIRNESS` (A)
  - `PREFILL_DECODE_CONTENTION` (B)
  - `KV_MEMORY_PRESSURE` (C)
  - `NONE` (Fallback)
  - `OVERLAP` (Fallback)
- **Activity Thresholds**:
  - `PRIORITY_SKEW_THRESHOLD = 1.05`
  - `CONTENTION_SCORE_V2_THRESHOLD = 0.20`
  - `KV_PRESSURE_THRESHOLD = 0.82`
  - `MIN_CONFLICT_QUEUE = 2`

### B. Stage-2 Native Pairs
Only the native policies within each active regime are eligible candidates. The router is barred from cross-regime routing, preserving the native-pair constraint:
- **Regime A (`RANKING_FAIRNESS`)**: `estimated_service_time_first` vs. `weighted_fair_share`.
- **Regime B (`PREFILL_DECODE_CONTENTION`)**: `full_prefill` vs. `chunked_prefill_small`.
- **Regime C (`KV_MEMORY_PRESSURE`)**: `kv_constrained_online` vs. `least_laxity_first`.

### C. FSM & Fallback Rules
- **Dwell Minimum Steps**: Exactly `dwell = 20` (unexempt for active regimes, exempt for fallback NONE/OVERLAP transitions as frozen).
- **NONE Fallback**: Dispatches to best global fixed policy, `weighted_fair_share`.
- **OVERLAP Fallback**: Dispatches to best global fixed policy, `weighted_fair_share`.
- **No Confidence Fallback**: Dispatches to best global fixed policy, `weighted_fair_share`.

### D. No Post-Hoc Tuning
No thresholds, models, training parameters, or features shall be altered based on the original approximate TEST outcome or any initial live TRAIN/VAL smoke results.

---

## 3. Handling the Family-B TEST Gap (Do Not Silently Fix)

The original evaluation split had a documented limitation:
```
Family B (PREFILL_DECODE_CONTENTION) TEST scenarios = 0
```
Because of this, Regime B was completely unrepresented in the TEST set, making G4's Stage-2-B evaluation and G2's ground-truth B prediction accuracy mathematically `NOT_EVALUABLE`.

We **MUST NOT** silently alter or redraw the primary TEST split to inject Family-B scenarios. Doing so would violate the "apples-to-apples" comparison against the original approximate run, introducing multiple confounding variables (different scenarios, different seeds, different sample sizes) and preventing us from isolating the evaluation methodology as the single independent variable.

Instead, we define **TWO scientifically distinct analyses**:

1. **PRIMARY RE-EVALUATION (Exact Split)**:
   - Uses the exact same frozen TRAIN, VAL, and TEST split definitions as the original evaluation.
   - Evaluates this exact split using the validated live causal closed-loop harness.
   - Purpose: Direct methodology comparison to isolate the majority-vote approximation artifact.
2. **SECONDARY REPLICATION (Family-B-Balanced)**:
   - A separately preregistered, family-balanced, held-out replication partition.
   - Includes a non-zero allocation of Family-B scenarios to evaluate Regime-B under live-harness conditions.
   - Purpose: Answer whether the live hierarchy works across all three regimes when Family B is represented.
   - **Crucial Rule**: This secondary replication must *never* overwrite, replace, or be blended into the primary TEST results. Their results and conclusions must remain strictly separated.

---

## 4. Primary Re-evaluation — Exact Split Definition

The Primary Re-evaluation is frozen to use the exact same splits, grouping unit, seeds, models, and thresholds as the original approximate run:
- **Grouping Unit**: `group_key` (ensures scenario lineage/seed is group-disjoint).
- **TEST Split Boundary**: Deterministic hash `sha256(group_key) mod 100` >= 80.
- **TEST Scenarios (n=32)**: Exactly the 32 scenarios (24 Family C, 8 Family A, 0 Family B) generated by the frozen split builder.
- **Model Checkpoints**: The original Stage-1 and Stage-2 models fit on TRAIN telemetry and TRAIN scenarios respectively, with hashes verified to be unchanged.

By holding the scenarios, models, and thresholds identical, any difference in ANWG or routing behavior is mathematically attributable solely to the transition from *scenario-level majority-vote approximation* to *per-step live closed-loop causal execution*.

---

## 5. Primary Comparison Pairs

For every scenario in the 32-scenario TEST set, the future run will execute the live harness and record the causal trajectory. We will compile a scenario-by-scenario paired table comparing:

- **A. OLD (Approximate Hierarchy)**: The pre-computed results from `test_evaluation_results.json` (where the router's decision was collapsed into a single majority policy for the whole scenario).
- **B. NEW (Live Hierarchy)**: The real ANWG computed via the live closed-loop harness (`LiveHierarchicalRouterPolicy` running inside the unmodified `Simulator`).
- **C. Best Global Fixed**: The baseline `weighted_fair_share` policy evaluated directly on the same scenario.
- **D. Regime-Specific Fixed-Best**: The within-family fixed-best baselines (`weighted_fair_share` for A, `kv_constrained_online` for C).
- **E. Global Six-Policy Oracle**: The maximum ANWG achievable by running all six policies in isolation for the entire scenario.
- **F. Relevant Family-Aware/Oracle Upper Bound**: The native-pair Stage-2 oracle.

The core scientific contrast is:
$$\Delta_{\text{methodology}} = \text{ANWG}(\text{Live Hierarchy}) - \text{ANWG}(\text{Approximate Hierarchy})$$

If $\Delta_{\text{methodology}} > 0$ is statistically significant, it directly supports the hypothesis that the majority-vote integration was a destructive measurement artifact.

---

## 6. Primary and Secondary Evaluation Metrics

To capture both utility outcomes and fine-grained live-routing dynamics, the re-evaluation will report:

### A. Utility Metrics (Primary)
- **Mean Arrival-Normalized Weighted Goodput (ANWG)**:
  - Live hierarchy mean ANWG ($\mu_{\text{live}}$)
  - Approximate hierarchy mean ANWG ($\mu_{\text{approx}}$)
  - Delta methodology: $\Delta_{\text{method}} = \mu_{\text{live}} - \mu_{\text{approx}}$
  - Delta vs. Fixed-Best: $\Delta_{\text{fixed}} = \mu_{\text{live}} - \mu_{\text{best-global-fixed}}$
  - Regret to Global Oracle: $\text{Regret}_{\text{oracle}} = \mu_{\text{oracle}} - \mu_{\text{live}}$
  - Oracle Gap Closed: Fraction of the gap between Best Global Fixed and the Oracle closed by the Live Hierarchy.
- **Bootstrap Confidence Interval**: A 90% group-resampled bootstrap confidence interval (16 groups, 5,000 draws) over $\Delta_{\text{method}}$ and $\Delta_{\text{fixed}}$.

### B. Router Classifier Metrics (Stage-1)
- **Accuracy**: Fraction of steps where Stage-1 predictions match the ground-truth activity label.
- **Macro-F1**: Evaluated over the 3 classes present in the primary TEST set (`RANKING_FAIRNESS`, `KV_MEMORY_PRESSURE`, `NONE`).
- **Catastrophic Misroute Rate**: Rate of active-to-active misrouting (A $\leftrightarrow$ C) excluding fallbacks.

### C. Selector Metrics (Stage-2)
- **Per-Regime Regret**: Standalone regret of Stage-2 selectors on active steps.
- **$\epsilon$-Optimal Accuracy**: Standalone selection accuracy of Stage-2 on active steps.

### D. Live Dynamics Metrics
- **Routing Distribution**: Fraction of steps routed to A, B, C, and Fallbacks (`NONE`, `OVERLAP`, No-Confidence).
- **Transition Metrics**: Total transition count, switching rate per 1,000 steps, and dwell-minimum violations.
- **Minority-Regime Episodes**: Count of episodes where a minority regime is entered and successfully maintained for at least `dwell=20` steps.
- **Minority Period Utility Contribution**: Summed ANWG contributions or localized request-level completion rates during minority routing periods to verify performance-critical interventions.

---

## 7. Key Methodology Hypothesis (Preregistered)

We preregister the core methodology hypothesis, **H-METHOD**:

> **H-METHOD**: The live closed-loop hierarchy will materially outperform the scenario-majority approximation on scenarios where the true underlying regime changes during the trajectory or where a minority performance-critical regime is active.

### Identification of "Methodology-Sensitive" Scenarios
To prevent post-hoc causal overclaiming, we define how "methodology-sensitive" scenarios are identified **strictly from router trajectory logs**, without using utility or performance outcomes:

A scenario is classified as **Regime-Dynamic** if its live-routing trajectory log satisfies either:
1. **Minority-Regime Presence**: The FSM-resolved effective regime is an active, non-fallback regime (A, B, or C) for at least $K = 20$ steps (matching the dwell minimum), but this regime is *not* the plurality/majority predicted regime of that scenario's trajectory.
2. **Trajectory Regime-Switching**: The FSM-resolved active regime changes at least once during the run (e.g., transitioning from `NONE` to `KV_MEMORY_PRESSURE`, or `RANKING_FAIRNESS` to `KV_MEMORY_PRESSURE`), excluding purely reverting back to `NONE`.

Under **H-METHOD**, we expect $\Delta_{\text{methodology}} > 0$ to be heavily concentrated in these Regime-Dynamic scenarios, while remaining approximately 0 on Regime-Static scenarios (where a single regime dominates 100% of the steps, meaning the majority-vote approximation was a faithful surrogate).

---

## 8. Primary Re-evaluation Verdict Logic

We define a brand new, clean evaluation verdict namespace to prevent confusion with the original approximate run. The final verdict of the live closed-loop re-evaluation will be assigned mechanically based on the following frozen logic:

```
                  +-----------------------------------+
                  |      Execute Primary TEST (n=32)   |
                  +-----------------------------------+
                                    |
                                    v
                  +-----------------------------------+
                  |  Compute Mean ANWG Delta:         |
                  |  D_method = Live - Approx         |
                  |  D_fixed  = Live - Global-Best    |
                  +-----------------------------------+
                                    |
            +-----------------------+-----------------------+
            | D_method >= 0.01      | D_method >= 0.01      | D_method < 0.01
            | D_fixed  >= 0.01      | D_fixed  < 0.01       | D_fixed  < 0.01
            | 90% CI excludes 0     | OR CI includes 0      |
            | G1,G2,G3,G4(A),G8,G9  |                       |
            | all PASS. No leaks.   |                       |
            v                       v                       v
+-----------------------+ +-----------------------+ +-----------------------+
|     LIVE_REEVAL_      | |     LIVE_REEVAL_      | |     LIVE_REEVAL_      |
|  SUPPORTS_HIERARCHY   | |   IMPROVES_METHOD_    | |    CONFIRMS_NO_GO     |
|                       | |  BUT_NO_END_TO_END  | |                       |
+-----------------------+ +-----------------------+ +-----------------------+
```

### Verdict Definitions

1. **`LIVE_REEVAL_SUPPORTS_HIERARCHY`**
   - **Conditions**:
     - $\Delta_{\text{method}} \ge 0.01$ (Live evaluation materially improves over the majority-vote approximation).
     - $\Delta_{\text{fixed}} \ge 0.01$ (Live hierarchy beats the best global fixed baseline by a practical margin).
     - The 90% bootstrap CI of $\Delta_{\text{fixed}}$ excludes 0 (lower bound $> 0$).
     - The original gates G1, G2, G3, G4 (evaluated on Regime A only, since Regime B is empty), G8, and G9 all **PASS** under live execution.
     - 0 dwell violations and 0 temporal leakage observed during the run.
   - **Interpretation**: The hierarchical router is scientifically validated. The prior negative verdict was 100% an artifact of the approximate evaluation methodology.

2. **`LIVE_REEVAL_IMPROVES_METHOD_BUT_NO_END_TO_END_GAIN`**
   - **Conditions**:
     - $\Delta_{\text{method}} \ge 0.01$ (The live harness successfully captures active routing that the approximation destroyed).
     - **BUT** $\Delta_{\text{fixed}} < 0.01$ or the 90% CI includes 0 (the live hierarchy still fails to beat the best global fixed policy by a meaningful, statistically sound margin).
   - **Interpretation**: The live harness successfully corrected the measurement methodology, but the hierarchical router v1 still does not provide sufficient end-to-end advantage to justify deployment over a simple fixed baseline.

3. **`LIVE_REEVAL_CONFIRMS_NO_GO`**
   - **Conditions**:
     - $\Delta_{\text{method}} < 0.01$ (The live evaluation does not materially alter the reported ANWG vs. the approximation).
     - **AND** the live hierarchy remains worse than or identical to the global fixed baseline ($\Delta_{\text{fixed}} < 0.01$).
   - **Interpretation**: The live evaluation confirms that the hierarchical router does not provide a benefit, and the prior `NO_GO` verdict was a correct reflection of router/selector capability rather than a measurement artifact.

4. **`LIVE_REEVAL_INCONCLUSIVE`**
   - **Conditions**:
     - The absence of Family-B scenarios or severe statistical noise (highly overlapping or wide bootstrap CIs) prevents a clear distinction between the above categories.

---

## 9. Preservation of Original G1–G9 as Secondary Checks

The original frozen gates G1–G9 will be scored under live execution as secondary checks. We **WILL NOT** modify or rewrite any of the original thresholds (e.g., G2's Macro-F1 $\ge 0.90$, G5's delta $\ge 0.01$, G3's misroute rate $\le 0.05$).

We will explicitly report:
- Which gates changed status (e.g., FAIL $\rightarrow$ PASS) solely because evaluation is now live.
- Which gates remained unchanged.
- Which gates are mathematically `NOT_EVALUABLE` due to the zero Family-B TEST representation.

This secondary analysis is crucial for tracing the exact causal pathways of any performance changes.

---

## 10. Family-B-Balanced Replication Design

To address the complete absence of Family-B scenarios in the primary split, we design a secondary, family-balanced held-out replication test set.

### A. Replication Split Properties
- **Linage-Disjoint Grouping**: Built using the same `group_key` grouping to prevent any data leakage.
- **Family Balance**: Must contain exactly equal representations of all three mechanism families:
  - 12 Family-A scenarios
  - 12 Family-B scenarios
  - 12 Family-C scenarios
  - **Total $n = 36$ held-out scenarios**
- **No Threshold/Model Tuning**: The existing Stage-1 and Stage-2 models will be evaluated *completely out-of-the-box* on this replication split. No weights, thresholds, or hyperparameters will be tuned or refitted.

### B. Role of the Secondary Replication
The replication is designed to answer: *Does the live hierarchy work when Family B is actually represented?*
- If the Primary Re-evaluation yields `LIVE_REEVAL_IMPROVES_METHOD_BUT_NO_END_TO_END_GAIN` (due to zero Family B and zero standalone Stage-2-C gain on its 24 scenarios), but the Balanced Replication shows strong, statistically significant gains across all three families, it provides a crucial scientific bridge.
- **Strict Separation Rule**: The replication results must be compiled in a separate section of the future report and must *never* be merged into the primary 32-scenario TEST results.

---

## 11. Blended-Regime Replication

The live-harness validation confirmed the first empirical observation of a `B+C` overlap under blended scenarios. To test the robustness of the router's hard-routing and fallback semantics in the presence of blended traffic, we preregister a small robustness analysis:

### A. Blended Scenarios
We will evaluate the live harness on four pre-defined blended-regime microcase templates (with 10 seeds each):
- **A+B** (Ranking + Prefill-Decode Contention)
- **A+C** (Ranking + KV Pressure)
- **B+C** (Prefill-Decode Contention + KV Pressure)
- **A+B+C** (All three active)

### B. Constraints
- **Zero Semantic Changes**: We **MUST NOT** modify the hard-routing or fallback semantics (FSM, fallback to `weighted_fair_share` on `OVERLAP` or `NONE`) even if high overlap is observed.
- **Reporting**: We will record and report the exact empirical rate of `OVERLAP` predictions and catastrophic misroutes. This acts as a stress test for the boundaries of the hard-routing assumption.

---

## 12. Evaluation Provenance Requirements

Any future execution of this live re-evaluation must automatically record and output an immutable provenance block to guarantee scientific reproducibility. The output JSON/markdown report must contain:

- **Launch Git SHA**: Exact HEAD commit hash at the time of execution.
- **Dirty-Tree Flag**: Boolean indicating if there are any uncommitted local changes.
- **Exact Command**: The precise shell command used to invoke the evaluation.
- **Config SHA-256**: The SHA-256 hash of `configs/hierarchical_regime_router_v1_gates.json`.
- **Gate JSON SHA-256**: Verified identical to the frozen design-time gate config.
- **Model Identity & Hashes**: Checksum hashes of the trained Stage-1 and Stage-2 model files.
- **Dataset Checksums**: Verification of `telemetry_by_scenario` and `mf_psd_scenarios_v1.csv` checksums.
- **Live-Harness Implementation SHA**: SHA-256 hash of `src/llmserveopt/policy_separation/hierarchical_router_live_harness_v1.py`.
- **Output Result SHA-256**: Computed hash of the generated results JSON.
- **Trajectory Log SHA-256**: Computed hash of the step-by-step trajectory CSVs.
- **Package Versions**: Active python, pandas, numpy, scikit-learn, and jinja2 versions.
- **Execution Timestamp**: UTC timestamp of the run.

This separates the original approximate runs and the new live runs with absolute, audit-proof clarity.

---

## 13. Pre-Launch Verification Plan

Before triggering the scientific re-evaluation, the executing runner must run a diagnostic check suite asserting:
1. **Split Identity**: Assert that the 32 scenarios selected for the primary live run match the original 32 TEST scenario IDs exactly.
2. **TEST Holdout**: Assert that no scenario in the TEST split was exposed to Stage-1 or Stage-2 fitting.
3. **Model Immutability**: Verify that the Stage-1 and Stage-2 model weights match the original pre-evaluation checkpoints exactly.
4. **Harness Integrity**: Verify that `tests/test_hierarchical_router_live_harness_v1.py` passes 26/26 tests.
5. **No Majority-Vote Leakage**: Verify that no part of the live-evaluation routing path imports or references `scenario_regime_from_telemetry` or the majority-vote logic.
6. **Forced-Parent Equivalence**: Re-run and assert 6/6 bit-exact match for the direct policy baselines.
7. **Dwell/Fallback Constancy**: Confirm `dwell=20` and fallback policies are unchanged.
8. **Deterministic Replay**: Run a test scenario twice and assert that both the ANWG and the per-step trajectory logs are bit-identical.
9. **Frozen Audit Protection**: Run a git diff check ensuring no changes were made to `docs/audits/hierarchical_regime_router_v1_20260818.md` or `docs/design/HIERARCHICAL_REGIME_ROUTER_V1.md`.

---

## 14. Standing Long-Running Job Rule

The scientific re-evaluation involves executing 32 full-trajectory simulator runs (and up to 36 for the balanced replication). Depending on scenario length, this may take several minutes to hours.

To ensure stability, the following operational rule **MUST** be applied during launch:

1. **Named Tmux Session**: The evaluation run must be launched in a dedicated, named tmux session (e.g., `tmux new -s hierarchical-live-reeval`).
2. **Limited Initial Monitoring**: After launching the job, the agent will monitor the process for **at most 3 minutes** to verify:
   - The process is alive (PID active).
   - Trajectory logs are actively being written to disk and files are advancing.
   - The scenario progress counter is increasing.
   - No immediate "failure storm" (e.g., instant crashes or import errors) occurs.
3. **Proactive Stop**: Once initial health is verified within the 3-minute window, the agent **MUST STOP monitoring and yield control to the user**, leaving the job running safely in the background. The agent will not wait for full completion or repeatedly poll the status.
