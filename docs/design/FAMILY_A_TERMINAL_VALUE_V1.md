# Family-A Receding-Horizon Terminal-Value Redesign V1

Date: 2026-08-20

Status: frozen before offline alignment evaluation. Not modified after
scientific results are known, except to append a clearly marked execution
record.

## 1. Hypothesis

`docs/current/family_a_rollout_value_limit_diagnosis_20260820.md` found that
the V1 receding-horizon oracle controller (`FAMILY_A_RECEDING_HORIZON_ORACLE_V1.md`)
loses to fixed WFS because its window objective — a sum over requests that
**fully complete** within the bounded rollout window — undervalues
high-priority, fairness-protected work that is still in flight (admitted,
receiving service, but not yet finished) at the window boundary. This is
concentrated almost entirely in `favlong` scenarios, where WFS achieves the
best priority-weighted SLO outcome despite the worst raw latency (a
fairness-debt-protection signature that plays out over the full ~32,000-step
scenario, not a ≤220-step window), while the window objective, built on a
raw-completion-favoring signal, systematically prefers ESTF there.

**Hypothesis under test**: crediting a rollout branch's terminal state for
the value of unfinished-but-feasible, service-invested work — not just fully
completed requests — will make the rollout's branch comparison better
aligned with true full-scenario ANWG, specifically in `favlong`, without
destroying its already-good alignment in `favshort`.

## 2. Exact Old Objective (unchanged, reused verbatim)

From `FAMILY_A_RECEDING_HORIZON_ORACLE_V1.md` SS5
(`src/llmserveopt/policies/family_a_receding_horizon_oracle_v1.py::_window_weighted_slo_goodput`):

```
W_completed(branch) = sum_i  weight_i * 1[not slo_violated_i]
weight_i = request.priority if request.priority > 0 else 1.0
```

summed over requests that **complete** strictly within the branch's own
bounded window (candidate segment of up to `H` steps, then a common
`WeightedFairSharePolicy` continuation of up to `COMMON_CONTINUATION_BUDGET
= 200` further steps). This term is retained **unchanged** in the new value
— the redesign is additive, not a replacement.

## 3. Diagnosed Failure (recap)

- Raw-completion-based counterfactual signals (both this window objective
  and the prior `family_a_stateful_controller_v1`'s `sign(Delta_native)`
  training label) structurally favor ESTF-style throughput: in the existing
  91-event repaired counterfactual data, ESTF-first-then-own-continuation
  beats ESTF-first-then-WFS-continuation in raw completions in 51/91 events
  and never loses; WFS-first-then-own-continuation beats
  WFS-first-then-ESTF-continuation in only 1/91 (diagnosis §6).
- In `favlong`, WFS achieves the best `priority_weighted_slo_goodput`
  (0.603) despite the *worst* `max_latency` (25.8 vs. ESTF's 23.7) —
  evidence of a full-trajectory fairness-debt mechanism invisible to a
  ≤220-step window (diagnosis §11).
- Within `favlong` alone, higher ESTF-choice frequency at `H=1`
  significantly predicts *worse* scenario outcome (Spearman ρ=−0.518,
  p=0.002) — the window objective is actively misleading there, not merely
  noisy (diagnosis §4).

## 4. New Terminal-Value Equation

```
V(branch) = W_completed(branch) + V_inflight(branch)

V_inflight(branch) = sum_i  weight_i * progress_fraction_i * feasible_i

    over every request i visible in the branch's terminal ObservableState
    (built via `shell._build_observable_state()` at the branch's window
    boundary) that has NOT completed within the window — i.e. every request
    in `waiting_queue` and every `active_requests_info` entry across all
    `gpu_states`.

weight_i            = request.priority if request.priority > 0 else 1.0
                       (identical definition to W_completed's weight_i —
                       this keeps V_inflight in the same completion-equivalent
                       units as the existing objective; no new scale)

tokens_decoded_i     = gpu_states[*].tokens_decoded_per_request.get(request_id, 0)
                       (0 for any request in `waiting_queue`, or admitted but
                       still in prefill)

progress_fraction_i  = clip(tokens_decoded_i / max(predicted_output_tokens_i, 1), 0, 1)

remaining_tokens_i   = max(predicted_output_tokens_i - tokens_decoded_i, 0)

remaining_service_i  = DEFAULT_BETA * remaining_tokens_i
                       (decode-only remaining-service estimate; DEFAULT_BETA
                       is the existing repo constant from
                       `policies/scoring.py`, already used by
                       `EstimatedServiceTimeFirstPolicy` and every other
                       service-time-aware policy in this repo — not a new
                       constant)

feasible_i           = 1 if deadline_slack(req, now=branch_terminal_time,
                              service_proxy=remaining_service_i) >= 0
                       else 0
                       (`deadline_slack` is the existing, unmodified
                       function in `policies/scoring.py`:
                       `slo_deadline - now - service_proxy`)
```

No new free coefficients are introduced. `progress_fraction_i ∈ [0,1]` and
`feasible_i ∈ {0,1}` are both unitless, causally-derived multipliers on the
same `weight_i` the existing objective already uses; there is nothing to
tune and therefore no sensitivity sweep is required or performed (per the
task's anti-tuning instruction, SS7).

### Why component F (explicit fairness-debt term) was excluded

The task's candidate list (SS5) included a separate fairness/class-deficit
terminal term. It is deliberately **not** included here: WFS's
priority-weighted SLO protection in `favlong` (§3) is a *consequence* of
which specific high-priority requests it keeps in service, and
`V_inflight` already credits exactly that — a high-priority in-flight
request under WFS's admission pattern earns more `V_inflight` credit than
the same slot occupied by a low-priority one under ESTF's. Adding a second,
separately-scaled fairness-debt term on top would double-count the same
underlying mechanism through an arbitrary second channel and would violate
the "keep the formulation small" instruction (SS5) without a demonstrated
need. This is a design decision, not an oversight, and is left open for the
diagnostic tests (test G) to fail loudly if that assumption turns out to be
load-bearing.

## 5. Causal-Availability Argument (no future information)

Every quantity in `V_inflight` is read from the branch's own terminal
`ObservableState`, which is built the same way (`Simulator._build_observable_state`)
as every real online decision already uses:

- `weight_i`, `predicted_output_tokens_i`, `slo_deadline_i` — fields of
  `ObservableRequest`, already exposed to every online policy today.
- `tokens_decoded_i` — `ObservableGPUState.tokens_decoded_per_request`,
  already exposed to every online policy today (used by no current policy,
  but present in the same struct ESTF/WFS already read).
- `branch_terminal_time` — `ObservableState.time` at the fork's own current
  simulated step; never real-world/actual scenario time beyond the branch's
  own simulated horizon.

No field uses `actual_output_tokens` (forbidden per `BasePolicy`'s own
docstring), no field is scenario-identity-derived (`canonical_scenario_id`,
`favlong`/`favshort`, utilization/skew/noise are never read), and nothing
here reads past the branch's own simulated window boundary. This is
structurally identical in kind to what `EstimatedServiceTimeFirstPolicy`
already does online (reading `ObservableRequest`/`ObservableGPUState`
fields to score a proxy) — only the *terminal window* it's evaluated at is
new (the fork's end-of-window state, not the real current step).

## 6. Controller Mechanics (unchanged)

Identical to `FAMILY_A_RECEDING_HORIZON_ORACLE_V1.md` in every respect
except scoring:

- Same eligibility gate (`actions_disagree(ESTF, WFS)` on identical
  pre-decision snapshot).
- Same WFS fallback outside the candidate region / on empty queue.
- Same two candidates (ESTF-first, WFS-first).
- Same `H ∈ {1, 5, 20}`.
- Same common continuation (`WeightedFairSharePolicy`, budget 200).
- Same execute-first-action-then-replan semantics.
- Same 64 Family-A TRAIN/VAL scenarios, same simulator.

Only the branch-comparison rule changes: `V(ESTF branch) > V(WFS branch)` →
execute ESTF's first action, else WFS's (ties → WFS, same convention as V1).

## 7. Offline Alignment Gate (before any full simulation)

A diagnostic-only re-run of the (unmodified) V1 eligibility/fork mechanics
computes **both** `W_completed` (old) and `V` (new) for every branch at
every eligible decision across all 64 TRAIN/VAL scenarios × 3 horizons,
while the *executed* action still follows old-V1 preference (so this run is
directly comparable to, and reuses the same trajectories as, the original
V1 scientific run — no execution-path change during this phase, satisfying
the task's "controller mechanics remain fixed" instruction even during
offline analysis).

Reported per horizon, split into ALL / `favshort` / `favlong`:

- old-preference vs. new-preference agreement rate
- ESTF/WFS selection-share shift (old → new)
- rank correlation of `(V_estf − V_wfs)` margin against
  `(controller_ANWG − WFS_ANWG)` at scenario level (same statistic already
  computed in the diagnosis, §10, for the old objective — directly
  comparable)

**GO requires all of**:
- New-preference ESTF share in `favlong` decreases relative to old (moves
  toward WFS, where diagnosis says appropriate) by a non-trivial margin.
- New-preference ESTF share in `favshort` does not collapse to near-zero
  (both candidates remain meaningfully selected — no trivial always-WFS
  solution, SS9 of the task).
- No `favlong`/`favshort` metadata is read by the value function (verified
  by code inspection + a dedicated test, not just by claim).
- No integrity issue (snapshot/restore, determinism) in the new code path.

If any of these fail: **`TERMINAL_VALUE_OFFLINE_NO_GO`**, full simulation is
not launched.

## 8. Safety Criteria (unchanged from V1)

Same frozen tolerances as `FAMILY_A_RECEDING_HORIZON_ORACLE_V1.md` SS10:
`completion_fraction`/`weighted_completion_fraction` must not fall more than
0.02 absolute below `min(mean ESTF, mean WFS)`; `slo_violation_rate` must
not rise more than 0.02 absolute above `max(mean ESTF, mean WFS)`.

## 9. TRAIN/VAL Only

Identical scenario table (`fac.load_family_a_trainval_scenario_table()`,
64 scenarios, 54 train/10 val) as every prior Family-A study in this chain.
No TEST scenario is loaded, generated, or read anywhere in this task.

## 10. GO/NO_GO Criteria (full closed-loop, frozen)

Let `best_fixed_mean = max(mean ANWG_ESTF, mean ANWG_WFS)` and
`oracle_gap = mean(native_pair_envelope) − best_fixed_mean` (as in V1).

**`TERMINAL_VALUE_POSITIVE_SIGNAL`** requires all of:
- New controller's best-arm mean ANWG clearly exceeds the old V1 rollout's
  best-arm mean ANWG, with more per-scenario wins than losses.
- New controller's best-arm mean ANWG either beats `best_fixed_mean`, or
  shows convincing, stable positive `recovered_fraction` with more wins
  than losses vs. best-fixed (not required to fully beat WFS to qualify —
  the task explicitly allows "at minimum shows convincing positive
  oracle-gap recovery with stable paired evidence").
- Both ESTF and WFS remain meaningfully selected (no collapse).
- `favshort`/`favlong` behavior becomes more appropriate (ESTF share drops
  in `favlong`, stays meaningful in `favshort`).
- Safety tolerances (§8) hold.

**`TERMINAL_VALUE_MIXED_SIGNAL`**: alignment and closed-loop result both
improve over old V1, but still do not reliably beat `best_fixed_mean` and
`recovered_fraction` is not stably positive.

**`TERMINAL_VALUE_NO_GO`**: no meaningful closed-loop improvement over old
V1, or the improvement is explained by collapse toward WFS (§7's no-collapse
check), or safety fails.

**`TERMINAL_VALUE_OFFLINE_NO_GO`**: §7 gate fails; full simulation not run.

**`TERMINAL_VALUE_INTEGRITY_NO_GO`**: implementation/instrumentation defect
found (snapshot/restore, determinism, metadata leakage).

## 11. Deliverables

- `src/llmserveopt/policies/family_a_receding_horizon_terminal_value_v1.py`
  (additive; does not modify the frozen V1 controller file)
- `scripts/run_family_a_receding_horizon_terminal_value_v1_offline_alignment.py`
- `scripts/run_family_a_receding_horizon_terminal_value_v1.py`
- `tests/test_family_a_receding_horizon_terminal_value_v1.py`
- `experiments/family_a_receding_horizon_terminal_value_v1/`
- `docs/current/family_a_terminal_value_v1_analysis_20260820.md`

---

## Execution Record (appended after results are known)

Offline alignment gate (SS7) executed 2026-08-20: FAILED. New-preference
ESTF share in `favlong` increased (not decreased) at every horizon
(+6.1pp H=1, +8.1pp H=5, +7.8pp H=20). Classification:
`TERMINAL_VALUE_OFFLINE_NO_GO`. Full closed-loop simulation was not
launched, per this document's own pre-registered stopping rule. See
`docs/current/family_a_terminal_value_v1_analysis_20260820.md` for the full
scientific record.
