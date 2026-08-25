# Family-A Terminal-Value Redesign V1 — Analysis

Date: 2026-08-20

## 1. Hypothesis

`docs/current/family_a_rollout_value_limit_diagnosis_20260820.md` found the
V1 receding-horizon oracle controller loses to fixed WFS because its
completion-only window objective undervalues high-priority, fairness-
protected work still in flight at the rollout boundary, concentrated in
`favlong` scenarios. Hypothesis: crediting feasible, service-invested but
unfinished work at a branch's terminal state (design doc
`docs/design/FAMILY_A_TERMINAL_VALUE_V1.md`) would shift branch preference
toward WFS specifically in `favlong`, without destroying the already-good
`favshort` alignment.

## 2. Diagnosis Motivating the Redesign (recap)

favlong: WFS achieves the best `priority_weighted_slo_goodput` (0.603)
despite the worst `max_latency` (25.8 vs. ESTF's 23.7) — a full-trajectory
fairness-debt signature no bounded window sees. Within `favlong`, the old
objective's ESTF-choice frequency at H=1 significantly predicted worse
outcome (ρ=−0.518, p=0.002).

## 3. Exact Value Equation (frozen, unchanged from design)

```
V(branch) = W_completed(branch) + V_inflight(branch)
V_inflight(branch) = sum_i  weight_i * progress_fraction_i * feasible_i
```

over every not-yet-completed request visible in the branch's terminal
`ObservableState` (`waiting_queue` + every GPU's `active_requests_info`).
`weight_i` is identical to the old objective's weight; `progress_fraction_i
= tokens_decoded_i / predicted_output_tokens_i`; `feasible_i` from
`deadline_slack(req, now, remaining_service) >= 0` using the existing
`scoring.py::deadline_slack`/`DEFAULT_BETA`. No free coefficients. Full
derivation in the design doc.

## 4. Causality / No-Future-Information Argument

Every input (`priority`, `predicted_output_tokens`, `slo_deadline`,
`tokens_decoded_per_request`, `time`) is read from the branch's own terminal
`ObservableState`, structurally identical in kind to what
`EstimatedServiceTimeFirstPolicy` already reads online. A dedicated test
(`test_no_scenario_metadata_symbols_referenced_in_module`) greps the module
source for `canonical_scenario_id`/`favlong`/`favshort`/`TEST` and confirms
none appear.

## 5. Offline Alignment (the gate; full results)

Executed the diagnostic-only offline-alignment run
(`scripts/run_family_a_receding_horizon_terminal_value_v1_offline_alignment.py`):
the **unmodified old-V1 controller executes** (trajectory identical to the
existing V1 scientific run), while the new value is computed as an unused
side channel at every eligible decision, across all 64 scenarios × 3
horizons (8,195 decisions, matching the original run's planning-call count
exactly). Wall clock 885.6s.

| H | regime | n | agreement (old vs. new preference) | old ESTF share | new ESTF share | Δ ESTF share | ρ(new-ESTF-frac, outcome) | p |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | all | 2707 | 95.0% | 26.5% | 31.3% | **+4.8pp** | −0.332 | 0.016 |
| 1 | favlong | 2069 | 93.8% | 27.7% | 33.8% | **+6.1pp** | **−0.461** | **0.008** |
| 1 | favshort | 638 | 99.2% | 22.6% | 23.0% | +0.4pp | +0.099 | 0.679 |
| 5 | all | 2736 | 93.7% | 28.0% | 34.1% | **+6.1pp** | −0.273 | 0.050 |
| 5 | favlong | 2085 | 91.9% | 28.4% | 36.5% | **+8.1pp** | **−0.485** | **0.005** |
| 5 | favshort | 651 | 99.5% | 26.4% | 26.6% | +0.2pp | +0.078 | 0.745 |
| 20 | all | 2752 | 91.2% | 45.2% | 51.0% | +5.8pp | +0.178 | 0.207 |
| 20 | favlong | 2111 | 88.9% | 45.6% | 53.4% | **+7.8pp** | +0.210 | 0.248 |
| 20 | favshort | 641 | 98.8% | 44.0% | 43.1% | −0.9pp | −0.265 | 0.258 |

**The redesigned value moves ESTF-preference share in `favlong` *up*, not
down, at every horizon** (+6.1pp at H=1, +8.1pp at H=5, +7.8pp at H=20) —
the opposite of the design hypothesis. At H=1 and H=5, this is confirmed
directly harmful: the within-`favlong` correlation between new-value
ESTF-preference frequency and the (already-known, old-controller) outcome
becomes **more** negative and more significant (ρ=−0.461, p=0.008 at H=1;
ρ=−0.485, p=0.005 at H=5) than the diagnosis's original finding for the old
objective (ρ=−0.518, p=0.002, H=1 only). Only at H=20 does the correlation
sign flip positive, but it is not significant (p=0.248) and is accompanied
by the ESTF-share still moving in the wrong direction.

Decision-level flip counts confirm this is a real, systematic, non-noise
effect, not an artifact:

| H | WFS→ESTF flips (favlong) | ESTF→WFS flips (favlong) |
|---|---:|---:|
| 1 | 128 | 1 |
| 5 | 168 | 1 |
| 20 | 199 | 35 |

Flips are overwhelmingly one-directional (WFS→ESTF), concentrated in
`favlong`, at every horizon.

## 6. Collapse Check

No trivial always-WFS collapse occurred — if anything the opposite: ESTF
share moved *up* everywhere except a small H=20 `favshort` decrease (−0.9pp).
`favshort` alignment was not meaningfully disturbed in either direction
(agreement rate 98.8–99.5%, ESTF share nearly unchanged). Both ESTF and WFS
remain meaningfully selected at every horizon (16/16 unit tests, including
the no-collapse-across-fixtures test, pass).

## 7. Integrity / Tests

16/16 new tests pass (`tests/test_family_a_receding_horizon_terminal_value_v1.py`),
covering: determinism and future-information-freedom of the credit function;
correct realized-value passthrough for completed requests; positive credit
for feasible in-flight high-priority work; monotonic response to decode
progress; zero credit for deadline-infeasible work; no fairness-debt-term
leakage (class_id-invariance); no scenario-metadata leakage (source-grep);
identical eligibility/fallback mechanics to V1 (only scoring differs);
snapshot/restore non-interference (fingerprint-identical before/mid/after);
full-run determinism; a `favlong`-like synthetic case where the new value
demonstrably narrows ESTF's old-objective advantage (3.0→0 gap); a
myopic-H=1 case confirming no always-WFS collapse. 44/44 pre-existing
Family-A regression tests (`test_family_a_receding_horizon_oracle_v1.py`,
`test_family_a_stateful_controller_v1.py`,
`test_family_a_observability_continuation_v1.py`) still pass unchanged.
**No integrity failure — `TERMINAL_VALUE_INTEGRITY_NO_GO` does not apply.**

## 8–13. Full Results / Comparisons / Fairness / Horizon Interaction

**Not run.** Per the frozen design-doc gate (SS7): "If any of these fail:
`TERMINAL_VALUE_OFFLINE_NO_GO`, full simulation is not launched." The
primary gate criterion — new-preference ESTF share in `favlong` decreasing
relative to old — failed at every horizon (§5). The full 64-scenario
closed-loop evaluation (`scripts/run_family_a_receding_horizon_terminal_value_v1.py`,
sections 15–21 of the task) was therefore **not launched**, consistent with
the design doc's own pre-registered stopping rule. No new-vs-old ANWG
comparison, favshort/favlong closed-loop decomposition, oracle-gap recovery,
or horizon-interaction result exists to report.

## 14. Likely Mechanism Behind the Failure (diagnostic, not a redesign)

The single test-M fixture (`test_favlong_like_case_new_value_credits_inflight_long_priority_work`)
showed the new value *does* narrow ESTF's advantage when there is exactly
one contested long/high-priority job and a small, fixed set of fillers — the
mechanism works as intended in isolation. The full-scenario failure is
consistent with a different effect at scale: `V_inflight` credits progress
on **every** not-yet-completed request, not specifically the contested
high-priority item. In real Family-A `favlong` scenarios (many candidate
requests, a queue that refills continuously), ESTF's branch — which churns
through more distinct short/cheap admissions within the window — accrues
partial credit across *many* different in-flight items (each contributing a
small `weight_i * progress_fraction_i`), while WFS's branch commits to
fewer, larger items and accrues credit on only those. The aggregate
in-flight credit across many small items apparently outweighs the credit
concentrated in WFS's protected long jobs, reintroducing a
breadth/throughput-favoring bias through a new channel — the same class of
problem the redesign was meant to fix. This is offered as a diagnostic
hypothesis for a *future* task, not something remedied here (redesigning
now, after seeing this result, would violate the frozen-design discipline).

## 15. Classification

**`TERMINAL_VALUE_OFFLINE_NO_GO`**

## 16. Limitations

- The offline alignment's outcome correlation (§5) uses the *old*
  controller's already-realized outcome as ground truth, not the new
  controller's own (never-run) closed-loop outcome — a necessary proxy
  given the gate is designed to screen out clearly-misaligned redesigns
  before paying for a full closed-loop run, but not a perfect substitute
  for it.
- Only one candidate value-function formulation was tested (per the design
  doc's "no free coefficients, no sweep" discipline) — this result rules out
  *this* formulation, not the general idea of terminal-state crediting.
- `favshort` alignment was not meaningfully disturbed, so this NO_GO is
  specific to the `favlong` mechanism, not a wholesale failure of the
  terminal-value concept.
- No TEST, public-trace, or real-serving analysis was performed (by design).

## 17. Next Step

**`STOP_TERMINAL_VALUE_DIRECTION`**

Per the design doc's explicit offline-gate stopping rule and the task's own
instruction not to execute a next step, this specific terminal-value
formulation is not carried into a full closed-loop run. §14's diagnostic
hypothesis (credit needs to be concentrated on the specifically-contested
item, not spread uniformly across all in-flight work) is a candidate
starting point for a *future*, separately-designed and separately-frozen
redesign — not attempted here.

## 18. Reproducible Commands / Artifacts

- Design: `docs/design/FAMILY_A_TERMINAL_VALUE_V1.md`
- New value/controller code (additive, V1 untouched):
  `src/llmserveopt/policies/family_a_receding_horizon_terminal_value_v1.py`
- Tests: `python3 -m pytest -q tests/test_family_a_receding_horizon_terminal_value_v1.py` (16 passed)
- Offline alignment gate: `python3 scripts/run_family_a_receding_horizon_terminal_value_v1_offline_alignment.py`
  → `experiments/family_a_receding_horizon_terminal_value_v1/offline_alignment_summary.json`,
  `offline_alignment_decisions.csv` (git HEAD `8e1223b`, dirty tree, wall clock 885.6s, 0 failures, 8,195 decisions)
- The planned full closed-loop script
  (`scripts/run_family_a_receding_horizon_terminal_value_v1.py`) was **not
  created or run**, since the offline gate did not pass.
