# H-CRITICAL Feasibility Audit & Evidence Reconciliation — 2026-08-24

**Mode:** Discover → reuse → verify. No parallel criticality framework. No manuscript edits.  
**HEAD:** `2987b7181efa2bc550d8a894c537eca8f6393eb6` (dirty tree preserved).

---

## 1. Existing infrastructure (do not duplicate)

| Asset | Path | Status |
|---|---|---|
| Preregistration | `docs/design/DECISION_CRITICALITY_TIMESCALE_TRAINVAL_V1.md` | Frozen |
| Implementation | `src/llmserveopt/analysis/decision_criticality_timescale_trainval_v1.py` | Fork + shadow disagreement |
| Runner | `scripts/run_decision_criticality_timescale_trainval_v1.py` | Used for full run |
| Tests | `tests/test_decision_criticality_timescale_trainval_v1.py` | 40 tests (committed) |
| Completed run | `experiments/decision_criticality_timescale_trainval_v1/` | **144/144 TRAIN/VAL, 0 fails** (~3.5 h) |
| Prior analysis | `docs/current/decision_criticality_timescale_trainval_v1_analysis_20260820.md` | `MIXED_SYNTHESIS_SIGNAL` |
| Downstream reuse of forks | `family_a_observability_continuation_v1`, receding-horizon oracle, DAgger/oracle datasets | Same `fork_from_live_simulator` |

**WORK_STATUS.md is stale** when it says the scientific run was `NOT_STARTED` — artifacts and the 2026-08-20 analysis already exist.

---

## 2. Counterfactual-feasibility audit

### Simulator capabilities already present

| Capability | Status |
|---|---|
| Exact mutable-state fork (`deepcopy` of `_gpus`, waiting, future arrivals) | **Yes** — `fork_from_live_simulator` |
| Deterministic paired prefix (reference continues unaltered) | **Yes** |
| Forced alternative action at step \(t\) | **Yes** |
| Continue fork with a chosen policy | **Yes** — `LiveFork.advance_one_step` / `run_bounded_rollout` |
| Official `Simulator.snapshot`/`restore` API | **No** native API; fork pattern is the supported substitute |
| One-step alt action → **return to reference policy** | **Feasible** with existing fork (apply alt once, then drive fork with reference policy) — **not** the preregistered estimand of v1 |
| Terminal **ANWG** on a fork | **Feasible** only if the branch runs to natural completion (or a documented full-horizon bound) with the same arrival process; v1 deliberately used **completed-count**, not ANWG, for partial windows |

### Estimand actually executed in v1 (labeled, not silent)

\[
\hat C^{\text{v1}}(s_t) \;=\; \mathbb{1}[\text{native pair disagree at }t]
\times \text{(state / completion divergence under alt-policy continuation)}
\]

- **Acquisition:** disagreement-only (outcome-blind for full-trajectory sample: first 3 disagreements/scenario).  
- **Short horizon:** alt policy for \(H=10\) vs reference trajectory (lockstep comparison).  
- **Bounded “ceiling”:** alt vs chosen **policy-switch** up to 3000 extra steps; metric = **completed-count** delta.  
- **Explicit omission:** mid-trajectory ANWG (design §5G).

### Ideal estimand from the authorizing task (not yet run)

\[
C(s_t)=V(s_t,a_{\text{alt}})-V(s_t,a_{\text{ref}})
\]

with one-step intervention, return to \(a_{\text{ref}}\) thereafter, terminal ANWG, identical future arrivals.

**Verdict:** `COUNTERFACTUAL_INFRASTRUCTURE_FEASIBLE`;  
`ANWG_ONE_STEP_THEN_RETURN_ESTIMAND_NOT_YET_EXECUTED`.  
Do **not** treat v1 completed-count / policy-switch results as if they were the ideal ANWG estimand.

---

## 3. Dataset / split (honored)

- TRAIN/VAL only: 144 scenarios (A=64, B=32, C=48); **0 TEST**.  
- Family-B replication held out (guarded).  
- Joint-240 SBS/VBS headroom (0.314072 / 0.333106 / 0.019034) is **scenario-level portfolio** evidence — not the same corpus/split as this state-level diagnostic. Connect narratively, not by pooling TEST joint outcomes into this TRAIN/VAL study.

---

## 4. What prior work already answered

| Question | Answered by v1? | Result (summary) |
|---|---|---|
| Are decision disagreements rare? | Yes | 0.515% of 876,839 active-regime steps (4,518 events) |
| Does disagreement ⇒ immediate closed-loop state change? | Yes | `any_nonzero_divergence_rate = 1.0` @ H=1 and H=10 |
| Does disagreement ⇒ short-horizon completion change? | Yes | Only **62/4518 (1.37%)** have `completed_count_abs_diff>0` @ H=10; **all** in Family A |
| Is recoverable short-horizon completion mass concentrated? | Yes (proxy) | See §5 |
| Timescale of high-value decisions | Yes | 99.1% disagreement bursts length 1 |
| Positive bounded ceiling | Partial | Family A mean +0.886 completions/branch; Family C −0.204; Family B n/a (0 disagreements) |
| Ideal ANWG one-step criticality curve | **No** | Not executed |
| Utility-weighted router accuracy on same states | **No** | Scope mismatch (router F1 on TEST telemetry vs criticality on TRAIN/VAL forks) |

---

## 5. Regret/criticality concentration (recomputed from frozen events)

**Utility proxy (pre-specified as closest recoverable without rerun):**  
`H=10 completed_count_abs_diff` on disagreement forks; agreement steps score **0** by determinism (identical actions ⇒ identical next state).

Source: `pass5_concentration_reanalysis.json`.

### Among all 876,839 evaluated steps

| Top fraction of steps | Share of total H10 completion-abs mass |
|---|---|
| top 1% | **100%** |
| top 5% … 50% | **100%** |

Interpretation: only **62** steps carry any mass; 62 ≪ 1% of steps ⇒ extreme concentration.  
**Statement supported:** “The top 1% of evaluated queue steps account for 100% of this short-horizon completion-consequence mass.”

### Among 4,518 disagreement events

| Top fraction | Share of completion-abs mass |
|---|---|
| top 1% (46) | **74.2%** |
| top 5% (226) | **100%** |

### Among all 144 scenarios

| Top fraction of scenarios | Share of completion-abs mass |
|---|---|
| top 1% (2) | 29.0% |
| top 5% (8) | 72.6% |
| top 10% (15) | 95.2% |
| top 20% (29) | 100% |

**Taxonomy:** `CRITICALITY_CONCENTRATED` (for this proxy).  
Not diffuse.

---

## 6. Disagreement ≠ utility criticality

| Quantity | Value |
|---|---|
| P(H10 completion-critical \| disagree) | **0.0137** |
| P(disagree \| H10 completion-critical) | **1.0** (by acquisition — forks only on disagree) |
| Frac disagree with ~zero H10 completion impact | **98.6%** |
| Immediate state divergence \| disagree | **100%** |

**Taxonomy:** `DISAGREEMENT_PROXY_WEAK` for **utility/completion** consequence;  
disagreement remains a valid **acquisition** signal and a perfect predictor of **state** divergence.

This is the core manuscript-relevant distinction: behavioral disagreement overstates useful adaptive leverage.

---

## 7. Classification accuracy vs utility (scope-honest)

Frozen hierarchical router (separate evaluation): macro-F1 ≈ **0.9887**, live re-eval confirms NO_GO on ANWG gain.

This criticality study does **not** share the same split/state table as that F1 number.  
Therefore: **cannot** legitimately compute “accuracy on critical states” or “utility-weighted F1” by naively joining artifacts.

**Compatible qualitative claim (already in audits):** high regime-classification accuracy can coexist with near-zero deployable ANWG gain because (a) native-pair ceilings are tiny/one-sided outside Family A, and (b) decision disagreements that matter for completions are extremely rare.

---

## 8. Closed-loop divergence & timescale

- Closed-loop: **confirmed** — every disagreement fork changes queue/active/KV state within 1–10 steps.  
- Timescale: critical **disagreements** are **isolated single-step spikes** (99.1% bursts length 1), sitting inside much longer regime-activity episodes (Family A median episode 223 steps ≫ dwell=20).  
- Implication: coarse per-scenario or slow regime routing is structurally mismatched to the temporal grain of the rare consequential disagreements — even when episodes are long enough for dwell.

**Taxonomy:** `CLOSED_LOOP_EFFECT_CONFIRMED`.

---

## 9. Statistical caution

- Within-scenario steps are **not** independent; scenario-level summaries (144) are the honest unit for uncertainty.  
- Per-branch signed full-trajectory deltas were **aggregated then discarded** (not persisted row-wise) — cannot bootstrap branch-level ANWG-like curves without a versioned re-run that logs them.  
- H10 completion-abs concentration uses a **sparse count** (mass=62); treat as descriptive concentration, not a precise ANWG fraction CI.

---

## 10. Scientific interpretation (H-CRITICAL)

**Supported (with labeled estimand):** Recoverable **short-horizon completion consequence** under native-pair counterfactual forks is concentrated in a tiny minority of online steps; most steps—and even most disagreement steps—are utility-inert at H=10; closed-loop state effects are ubiquitous given disagreement but do not imply completion/ANWG value.

**Not established:** Exact ANWG fraction captured by top-k% states under one-step-then-return interventions on joint-240 or TRAIN/VAL.

**Manuscript explanation of exploitability gap:** **Materially strengthened** as a mechanism diagnosis (concentration + disagreement≠utility + closed-loop), without claiming a new selector/scheduler.

---

## 11. Recommended next action (only if author-authorized)

Versioned extension **reusing** `fork_from_live_simulator` (no parallel stack):

1. Preregister `C(s)=ANWG` with **one-step alt → return to reference policy**, TRAIN/VAL only.  
2. Acquisition: existing disagreement set ∪ outcome-blind random control states.  
3. Persist per-branch rows (scenario_id, step, ΔANWG, trajectory fingerprints).  
4. Smoke → small Family-A subset → full only if cost justifies.  
5. Then compute true ANWG concentration curve + scenario-bootstrap CIs.

Until then: cite v1 + this reconciliation; do not invent a new selector.
