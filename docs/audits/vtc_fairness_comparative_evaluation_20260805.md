# VTC Fairness-Validated Comparative Evaluation — 2026-08-05

Companion to `docs/audits/vtc_fairness_benchmark_repair_20260805.md` (the
repair methodology). This document is the comparative-sweep results,
independent verification, and scientific classification decision.

## Sweep configuration

Ran only after `python scripts/check_vtc_fairness_headroom.py --all`
reported **ALL FAMILIES PASS**. Six policies × six repaired
fairness-extension workload families × three deterministic seeds (offsets
0/1/2, applied to each workload generator's own base seed) = 108 runs,
`RECOMMENDED_GPU_CONFIG` throughout. No selector/regression policy
included (explicitly out of scope). Runtime: 2m04s.

Policies: `official_vtc` (variant A), `matched_admission_fifo` (variant
B), `fairness_isolation_vtc` (variant C), `fifo` (native gate),
`shortest_prompt_first` (throughput-oriented), `scorpio_style_slo_guard`
(SLO/admission-oriented).

Full raw results:
`baselines/vtc/sweep_results/vtc_fairness_comparative_sweep_20260805.json`.

## Results (averaged across 3 seeds)

| Family | Policy | ANWG | Jain (checkpoint) | Disparity | Max starvation (s) | p95 queuing delay (s) |
|---|---|---|---|---|---|---|
| balanced_tenants | official_vtc | 1.000 | 0.995 | 5430 | 1.23 | 1.861 |
| balanced_tenants | matched_admission_fifo | 1.000 | 0.993 | 6613 | 1.27 | 1.641 |
| balanced_tenants | fifo | 1.000 | 0.993 | 6613 | 1.27 | 1.641 |
| balanced_tenants | shortest_prompt_first | 1.000 | 0.994 | 6054 | 1.18 | 4.435 |
| balanced_tenants | scorpio_style_slo_guard | 1.000 | 0.994 | 5889 | 1.30 | 1.875 |
| **one_heavy_hitter** | **official_vtc** | 1.000 | **0.521** | 60588 | 2.76 | 1.352 |
| one_heavy_hitter | matched_admission_fifo | 1.000 | 0.448 | 69798 | 2.89 | 4.432 |
| one_heavy_hitter | fifo | 1.000 | 0.448 | 69798 | 2.89 | 4.432 |
| one_heavy_hitter | shortest_prompt_first | 1.000 | 0.467 | 61549 | 3.07 | 12.727 |
| one_heavy_hitter | scorpio_style_slo_guard | 1.000 | 0.446 | 74113 | 3.05 | 7.295 |
| **heterogeneous_token_sizes** | **official_vtc** | 1.000 | **0.775** | 46419 | 1.25 | 5.842 |
| heterogeneous_token_sizes | matched_admission_fifo | 1.000 | 0.688 | 53459 | 2.12 | 10.503 |
| heterogeneous_token_sizes | fifo | 1.000 | 0.688 | 53459 | 2.12 | 10.503 |
| heterogeneous_token_sizes | shortest_prompt_first | 1.000 | 0.636 | 42312 | 26.93 | 16.208 |
| heterogeneous_token_sizes | scorpio_style_slo_guard | 1.000 | 0.657 | 72973 | 1.54 | 7.441 |
| bursty_tenant | official_vtc | 1.000 | 0.929 | 22060 | 7.39 | 5.052 |
| bursty_tenant | matched_admission_fifo | 1.000 | **0.935** | 24310 | 8.05 | 7.958 |
| bursty_tenant | fifo | 1.000 | 0.935 | 24310 | 8.05 | 7.958 |
| bursty_tenant | shortest_prompt_first | 1.000 | 0.932 | 21729 | 7.22 | 15.683 |
| bursty_tenant | scorpio_style_slo_guard | 1.000 | 0.925 | 27753 | 7.66 | 11.566 |
| returning_inactive_tenant | official_vtc | 1.000 | 0.960 | 31910 | 34.96 | 0.991 |
| returning_inactive_tenant | *(all 6 policies)* | 1.000 | 0.960 | 31910 | ~35 | ~1.0 |
| **priority_fairness_conflict** | **official_vtc** | **0.680** | **1.000** | 2387 | 0.48 | 1.519 |
| priority_fairness_conflict | matched_admission_fifo | 0.715 | 0.998 | 4662 | 0.63 | 1.349 |
| priority_fairness_conflict | fifo | 0.715 | 0.998 | 4662 | 0.63 | 1.349 |
| priority_fairness_conflict | shortest_prompt_first | 0.884 | 0.998 | 4929 | 0.59 | 3.403 |
| priority_fairness_conflict | scorpio_style_slo_guard | **0.984** | 0.997 | 5885 | 0.71 | 1.501 |

`official_vtc` and `fairness_isolation_vtc` are numerically identical (or
within noise) in every family — confirms the repair worked: at
`RECOMMENDED_GPU_CONFIG`, the admission gate no longer dominates enough to
separate variants A and C. `matched_admission_fifo` and `fifo` are
likewise identical everywhere, for the same reason.

## Independent verification

`scripts/verify_vtc_fairness_sweep.py` reran all 108 combinations and
independently recomputed completion fraction, ANWG (re-derived from its
definition in `src/llmserveopt/core/metrics.py`, not trusted from
`RunMetrics`), Jain's index, service disparity, starvation intervals, and
p95 queuing delay — using fresh implementations, never importing the
sweep script's own aggregation functions.

```
Checked 108 (family, seed, policy) combinations.
Mismatches: 0
```

**Zero unexplained mismatches.** The simulator-reported ANWG and the
independently-recomputed ANWG also agreed on every run (an internal
consistency check, not just agreement with the sweep script).

**Independent win/tie/loss tally** (Jain's index, checkpoint-based, win =
strictly at or within 1e-9 of the best policy for that family+seed; tie =
within 0.01 of the best; loss = otherwise; 18 family×seed combinations
total):

| Policy | Win | Tie | Loss |
|---|---|---|---|
| **official_vtc** | **13** | 4 | 1 |
| **fairness_isolation_vtc** | **13** | 4 | 1 |
| fifo | 5 | 6 | 7 |
| matched_admission_fifo | 5 | 6 | 7 |
| shortest_prompt_first | 5 | 5 | 8 |
| scorpio_style_slo_guard | 4 | 7 | 7 |

**Leakage check:** structurally guaranteed, not just spot-checked —
`ObservableRequest` (`src/llmserveopt/core/types.py`) has no
`actual_output_tokens` field at all, and every policy above only ever
receives `ObservableRequest` instances via `BasePolicy.select_action`,
never the raw `Request`.

## Scientific decision

**Does official VTC improve fairness when reservation is not the dominant
factor?** Yes, measurably and independently verified. VTC achieves the
strictly-best or tied-best Jain's index in 17 of 18 family×seed
combinations (13 outright wins + 4 ties), losing only in `bursty_tenant`
(0.929 vs. matched-admission FIFO's 0.935 — a small, real, disclosed
negative result, not hidden). The two families most directly designed to
test VTC's headline claim show large, unambiguous wins: `one_heavy_hitter`
(Jain 0.521 vs. next-best 0.467, a full policy family below) and
`heterogeneous_token_sizes` (0.775 vs. next-best 0.688).

**How much throughput or ANWG does it sacrifice?** In 5 of 6 families,
essentially none — ANWG stays at 1.000 for every policy (this simulator's
generous drain absorbs all demand eventually regardless of scheduling
order). The exception is `priority_fairness_conflict`, engineered
specifically to expose VTC's blindness to `priority`/`slo_deadline`:
VTC's ANWG (0.680) is the WORST of all six policies (SCORPIO, SLO-aware,
achieves 0.984), and its tight-SLO tenant's violation rate is 38.1%
(SCORPIO: 0.0%) in exchange for the best raw fairness (Jain 1.000). This
is a real, bounded, well-characterized trade-off, not a universal
throughput tax — it appears only when SLO-awareness and per-tenant
fairness are placed in direct conflict, exactly as designed.

**Are gains caused by ordering, reservation, or both?** Ordering, cleanly
isolated. `official_vtc` and `fairness_isolation_vtc` (identical ordering,
different admission capacity) are numerically indistinguishable across
the entire sweep, and `admission_gate_bind_rate` stayed under 0.02 in
every family (headroom-gated). The repair's entire point — separating
these two effects — succeeded: the fairness wins reported here are not an
artifact of the admission gate.

**Does VTC provide unique wins or a genuinely distinct behavioral niche?**
Yes to both. Unique wins: in `one_heavy_hitter`/`heterogeneous_token_sizes`,
no other policy in the comparison comes close to VTC's Jain's index — the
next-best policy in each family is 0.05-0.10 lower. Distinct niche: VTC is
the only policy in this comparison implementing per-tenant
virtual-service-counter accounting with a work-conserving,
min-served-first admission rule and an explicit counter-lift-on-return
mechanism — structurally unlike FIFO (arrival order), SPF (prompt-length
order), or SCORPIO (SLO-laxity order). The four hand-verified micro-traces
(`tests/test_vtc_micro_traces.py`) demonstrate this mechanism concretely
and exactly.

**Is the behavior useful for the project's foundational heuristic
library?** Conditionally yes — for a FAIRNESS-oriented composition
context specifically, not as a general ANWG-maximizing candidate. The
`priority_fairness_conflict` result is the load-bearing caveat: VTC
optimizes a genuinely different objective (equalized service) that can
actively conflict with this project's primary existing objective
(SLO-weighted goodput) when the two are placed in tension. It is not a
drop-in improvement to the existing 20-policy portfolio's objective; it is
a different objective the portfolio does not currently represent at all.

**Can VTC's fairness accounting be represented as reusable primitives?**
Yes, in principle: a per-tenant virtual-cost counter + min-cost-first
admission ordering + a counter-lift-on-return rule is a compact,
well-specified mechanism (fully characterized by the micro-traces) that
could be reimplemented natively (not merely wrapped) as a composable
scheduling primitive. This was not attempted in this task (explicitly out
of scope — no registration this task) and would be genuinely new
implementation work, not a re-export of the wrapped adapter.

## Final classification

**FOUNDATIONAL_CANDIDATE**

Justification: independently-verified, statistically broad (17/18
family×seed win-or-tie rate) fairness improvement, a cleanly isolated
mechanism (ordering, not admission), a genuinely distinct and reusable
algorithmic niche, and a well-characterized, bounded trade-off rather than
an unexplained one. This classification is explicitly **scoped to VTC's
fairness objective as a candidate primitive for future fairness-aware
composition**, not a claim that VTC (or a wrapped/native equivalent of it)
should replace or be blended into the existing ANWG-maximizing 20-policy
portfolio without further, separate evaluation of that specific question.

**Foundational-library eligibility: YES** (scoped as above).
**Registration decision: NOT REGISTERED this task**, per explicit
instruction — eligibility and registration are deliberately kept as
separate decisions here.

## Limitations

- Evaluated only against 6 custom-built synthetic fairness-extension
  workloads, not the canonical suite (structurally incompatible — no
  tenant concept) and not a real multi-tenant production trace.
- `returning_inactive_tenant`'s counter-lift mechanism is exercised
  (nonzero `decision_disagreement_rate`, confirmed directly by micro-trace)
  but its AGGREGATE effect on this specific workload's checkpoint-measured
  Jain's index is negligible (all 6 policies converge to ~0.960) — a
  genuine, disclosed null result at the workload/checkpoint level tested,
  not evidence the mechanism doesn't work (the micro-trace and per-step
  diagnostics prove it does).
- Single monolithic GPU only (inherited adapter limitation, see
  `baselines/vtc/PROVENANCE.md`).
- `cost_func="linear"` only (the official `"profile"` cost function
  remains out of scope, hardware-specific).

## Exact next action

If VTC's fairness primitive is to be pursued toward foundational-library
registration, the next step is a native (non-wrapped) reimplementation of
the same three-part mechanism (virtual-cost counter, min-cost-first
ordering, counter-lift-on-return) as a first-class simulator policy, so it
can compose with the existing portfolio the way every other foundational
policy does — a separate, explicitly-scoped task, not attempted here.
