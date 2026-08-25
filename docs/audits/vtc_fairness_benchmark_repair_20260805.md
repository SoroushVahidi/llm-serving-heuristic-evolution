# VTC Fairness-Benchmark Repair — 2026-08-05

Repairs the VTC evaluation design so it cleanly measures VTC's fairness
mechanism rather than being dominated by an admission-gate confound. Full
comparative results and independent verification are in the companion
document, `docs/audits/vtc_fairness_comparative_evaluation_20260805.md`.
This document covers the diagnosis and the repair methodology: the
smoke-test confound decomposition, the three labeled experimental
variants, the redesigned fairness-extension workloads, the headroom
gates, and micro-trace verification.

## 0. Repository state at start

- Branch: `contextual-compositional-heuristics-20260731`
- Starting SHA: `55e0da97c106f29ea9059f344d0e1de9798be8f8`
- Upstream: `origin/contextual-compositional-heuristics-20260731`, 0 ahead / 0 behind, clean working tree
- `python scripts/check_contextual_composition_status.py` — passed
- `python scripts/check_contextual_composition_status.py --resume-readiness` — passed
- VTC confirmed `EVALUATION_ONLY` in `docs/BASELINE_STATUS.md`
- No other jobs/tmux sessions running for this repository

## 1. What was wrong, restated precisely

`docs/audits/vtc_initial_integration_20260805.md`'s smoke evaluation found
that at the capacity used (`max_active_sequences=8, max_batch_tokens=1024,
max_kv_tokens=4096`), FIFO/`shortest_prompt_first`/`scorpio_style_slo_guard`
were indistinguishable from VTC in 5 of 6 fairness-extension families, and
the one that diverged (`heterogeneous_token_sizes`) looked like a
confound rather than a fairness signal, without being able to say
precisely why.

## 2. Reproduction and decomposition (task step 2)

`baselines/vtc/adapter/diagnostics.py`'s `InstrumentedVTCFairnessPolicy`
wraps `VTCReqQueue._can_add_new_req` at the INSTANCE level (never edits
the pinned source file) purely to record every accept/reject decision
official code makes, per step: which tenant was tried, whether it was
admitted, and — if not — whether the official reservation formula itself
rejected it or the batch-token-budget check did. Verified inert (produces
bit-for-bit identical scheduling outcomes with and without instrumentation
— `tests/test_vtc_fairness_headroom.py::TestDiagnosticsInstrumentationIsInert`)
before trusting any of its output.

`scripts/decompose_vtc_smoke_confound.py` reran the original 6 families
at the original capacity through this instrumentation. Results (see that
script's output, archived in this repo's history):

| Family | FIFO contention rate | VTC contention rate | reservation_bind | budget_bind | unexplored_backlog |
|---|---|---|---|---|---|
| balanced_tenants | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| one_heavy_hitter | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| **heterogeneous_token_sizes** | 0.000 | **0.986** | **0.000** | **0.992** | **0.986** |
| bursty_tenant | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| returning_inactive_tenant | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| priority_fairness_conflict | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

**Both original claims confirmed quantitatively:**

1. **Insufficient contention in 5/6 families.** FIFO's own per-step
   contention rate (fraction of admission steps with >=2 tenants
   simultaneously backlogged) was 0.000 in every family except
   `heterogeneous_token_sizes` — scheduling ORDER never had to break a
   real tie between competing tenants, so any two policies were
   guaranteed to look identical regardless of their fairness properties.
   Root cause: per-tenant arrival rates (2-5 req/s combined) were below
   the effective single-slot service rate given ~100-token mean outputs.

2. **`heterogeneous_token_sizes` was reservation/budget-dominated, not
   ordering-dominated** — but more precisely than originally stated:
   `reservation_bind_rate=0.000` (the official `_can_add_new_req`
   worst-case-reservation FORMULA never actually rejected anything),
   while `budget_bind_rate=0.992` (99.2% of steps stopped on the
   simpler `new_batch_total_tokens + req.input_len <= self.batch_max_tokens`
   check). This is a **sharper, corrected finding** vs. the original
   audit's "reservation gate" framing.

**Root cause of the budget-bind pathology — a units mismatch, isolated by
direct code inspection:** this simulator's native `BasePolicy._feasible_on_gpu`
(used by every non-VTC policy) reads `GPUConfig.max_batch_tokens` as a
per-step ACTIVE-REQUEST-COUNT cap (`new_batch = new_count`, a documented
Phase-1 simplification — see `src/llmserveopt/policies/base.py`), while the
official `VTCReqQueue`/`ReqQueue` code reads the identically-named field as
a real cumulative PROMPT-TOKEN budget. Feeding the same numeric value
(1024) into both interpretations gives a ~1024-concurrent-request cap for
native policies (never binding, since `max_active_sequences=8` is tighter)
and a genuine ~1024-cumulative-token cap for official-code policies —
which a single 900-token `long_prompts` request nearly exhausts by itself.
`max_active_sequences` and `max_kv_tokens` are consistently interpreted by
both code paths; only `max_batch_tokens` differs.

**Direct confirmation this is admission-driven, not ordering-driven:**
running `MatchedAdmissionFIFOPolicy` (FCFS ordering, same official gate)
on `heterogeneous_token_sizes` at the original capacity produced
`completion_fraction=0.036` — bit-for-bit identical to official VTC's
`0.036` — while plain, native-gate FIFO achieved `1.000`. Since variant B
shares VTC's admission gate but not its ordering, and reproduces VTC's
exact failure, the failure cannot be attributed to fairness ordering.

## 3. Three labeled experimental variants (task step 3)

Implemented in `baselines/vtc/adapter/variants.py` (full rationale in its
module docstring):

- **A. Official VTC** (`official_vtc` / `vtc_fairness_reference`):
  `VTCFairnessPolicy` with `batch_token_budget_override=None` — real VTC
  ordering, real official admission gate, this project's GPUConfig numbers
  used exactly as designed.
- **B. Matched-admission FIFO** (`matched_admission_fifo`): plain
  FCFS-by-arrival, via the official, unmodified `ReqQueue` base class
  `VTCReqQueue` itself subclasses — not a hand-matched approximation of
  the gate, the EXACT SAME code path. New file:
  `baselines/vtc/adapter/matched_admission_fifo_policy.py`.
- **C. Fairness-isolation VTC** (`fairness_isolation_vtc`): same
  `VTCFairnessPolicy`, with `batch_token_budget_override=2048` — still
  100% unmodified official code; only the numeric capacity argument fed
  to it differs from variant A. `2048` was chosen empirically: it
  comfortably exceeds every family's maximum single-request prompt size
  (1353 tokens) while remaining non-trivial relative to
  `max_active_sequences`-scale demand (raising it further to 4096/8192
  produced no additional change in any family's completion fraction,
  confirming it is a real, non-vacuous constraint, not disguised
  unlimited capacity).

Direct verification on `heterogeneous_token_sizes` at the ORIGINAL
capacity (`batch_max_tokens=1024`): variant A completion=0.036, variant B
completion=0.036 (identical to A), variant C (override=2048)
completion=1.000. This is the core decomposition result: identical
ordering + identical gate + different capacity → completion collapses or
doesn't, in lockstep with the capacity number, never the ordering rule.

## 4. Redesigned contention-valid fairness workloads (task step 4)

`baselines/vtc/fairness_workloads.py` was rewritten (all 6 families kept,
same names, same purposes) with:

- **A single shared, deliberately-chosen capacity**,
  `RECOMMENDED_GPU_CONFIG = GPUConfig(max_active_sequences=3,
  max_batch_tokens=4096, max_kv_tokens=16384)`. `max_active_sequences=3`
  is the SOLE deliberate contention-inducing knob (consistently
  interpreted as a request count by every policy, native or official).
  `max_batch_tokens=4096` is set generously above every family's maximum
  single-request prompt size (2031 tokens, in the repaired
  `heterogeneous_token_sizes`) so it never becomes a confound under
  EITHER interpretation — this is what the task's own requirement #5
  ("heterogeneous request sizes with sufficient memory headroom") calls
  for directly.
- **Retuned arrival rates** (roughly 2-5x higher per-tenant rates than
  the original smoke test), verified via a FIFO-only contention-rate
  probe (`>=2 tenants simultaneously backlogged` fraction) for every
  family before being accepted.
- **A tightened SLO** for `priority_fairness_conflict`'s tight-SLO tenant
  (1.0s slack, down from 3.0s) — the original 3.0s slack never actually
  bound under FIFO (0.000 violation rate) at the new contention level, so
  the family did not discriminate on the axis it was designed to test;
  1.0s produces a genuinely mixed FIFO violation rate (0.603).
- **Full per-family spec** (tenant count, arrival process, prompt/output
  distributions, capacity, SLOs, priorities, expected queue depth,
  expected fairness behavior, why reservation should/shouldn't bind) in
  each generator function's docstring, per the task's explicit
  requirement.

Measured FIFO contention rates under the repaired design (full detail:
`scripts/check_vtc_fairness_headroom.py --all`):

| Family | Contention rate | Notes |
|---|---|---|
| balanced_tenants | 0.742 | sanity-check family |
| one_heavy_hitter | 0.304 | |
| heterogeneous_token_sizes | 0.947 | reservation now non-binding (0.014) |
| bursty_tenant | 0.280 | |
| returning_inactive_tenant | 0.234 overall / 0.603+ in overlap windows | contention concentrated in 2 windows by design |
| priority_fairness_conflict | 0.936 | |

## 5. Fairness-headroom gates (task step 5) and checker (task step 6)

`scripts/check_vtc_fairness_headroom.py` implements PASS/FAIL gates with
thresholds justified against the measured ranges above (each threshold
chosen with comfortable margin both below the smallest passing measured
value and above the ORIGINAL broken regime's near-zero values):

**Universal gates** (every family): `contention_rate >= 0.15` (or, for
`returning_inactive_tenant`, `windowed_contention_rate >= 0.50` within its
two overlap windows — its contention is concentrated by design, so the
flat per-run rate understates it); `n_contended_steps >= 500` (absolute
sample-size floor; measured range 9304-40836); `admission_gate_bind_rate
<= 0.05` (reservation + budget bind rates combined; measured range
0.009-0.019+0.000, vs. the original broken regime's 0.992);
`decision_disagreement_rate >= 0.005` (VTC's min-served pick must
genuinely differ from what plain FCFS — oldest waiting request across
backlogged tenants — would pick, at a meaningful fraction of contended
steps; measured range 0.010-0.021).

**Family-specific gates:** `one_heavy_hitter`/`heterogeneous_token_sizes`
additionally require `fifo_jain_at_checkpoint <= 0.90` (FIFO must show
real disparity for VTC to have something to correct; measured 0.426/0.703).
`priority_fairness_conflict` additionally requires the tight-SLO tenant's
FIFO violation rate to be genuinely mixed, `(0.10, 0.90)` exclusive
(measured 0.603 — neither floor nor ceiling).
`returning_inactive_tenant` additionally requires
`continuous_demand_before_return > 0` — the counter-lift mechanism is only
meaningfully exercised if the other tenant actually accrued real service
while `returning` was idle (measured: 133,439 tokens).

**Result: `python scripts/check_vtc_fairness_headroom.py --all` → ALL 6
FAMILIES PASS**, every applicable gate.

15 tests in `tests/test_vtc_fairness_headroom.py` (16 including the
diagnostics-inertness test) cover: insufficient contention (2 synthetic
negative cases), reservation domination (1 synthetic negative case, a
direct reproduction of the original confound at `batch_max_tokens=64`),
valid headroom (all 6 real families, parametrized), the returning-tenant
counter-lift precondition (both a positive case against the real family
and a synthetic negative case), and deterministic output (repeated runs
produce byte-identical reports).

## 6. Micro-trace verification (task step 7)

`tests/test_vtc_micro_traces.py` — 4 tests, all against the raw,
unmodified official `VTCReqQueue`/`ReqQueue` classes directly (no
simulator), with expected outcomes computed BY HAND in each test's
docstring before being asserted:

1. **VTC improves fairness over FIFO**: tenant A backlogs 3 requests
   before tenant B's single request arrives; FIFO admits
   `[A1,A2,A3,B1]` (pure arrival order); VTC admits `[A1,B1,A2,A3]` — B's
   request moves from 4th position to 2nd, hand-derived from the
   min-served-first loop and the `served`-dict insertion-order tie-break.
2. **VTC and FIFO tie**: a single tenant alone — no genuine contention
   exists, so both produce identical order `[1,2,3]` by construction.
3. **Reservation blocks both matched variants equally**: a single
   101-token-equivalent request against `max_total_tokens=50`
   (hand-computed `need_max_token_num=105 >= 50` → infeasible) is
   rejected identically by `ReqQueue` and `VTCReqQueue`, since both
   inherit the exact same `_can_add_new_req`.
4. **Official VTC differs from fairness-isolation VTC only by
   admission**: the SAME appends as trace 1, run through two
   `VTCReqQueue` instances differing ONLY in `batch_max_tokens` (150 vs.
   1000) — the tight variant admits `[1,4]` and stops (hand-computed:
   `210 <= 150` is False), the loose variant admits `[1,4,2,3]` (full
   order); the tight variant's admitted prefix is IDENTICAL to the loose
   variant's first two admissions, proving the two disagree only on how
   much fits, never on which order the algorithm would pick.

All 4 hand-derivations matched the actual code's output on first
execution (one test needed a trivial `deque([]) != []` comparison fix,
not a derivation error).

## Files added/modified in this repair pass

```
baselines/vtc/adapter/
  diagnostics.py                       # NEW: instrumented decomposition wrapper
  matched_admission_fifo_policy.py     # NEW: variant B
  variants.py                          # NEW: labeled A/B/C factories
  simulator_policy.py                  # MODIFIED: added batch_token_budget_override param
baselines/vtc/fairness_workloads.py    # REWRITTEN: retuned capacity/rates, full specs
scripts/
  decompose_vtc_smoke_confound.py      # NEW
  check_vtc_fairness_headroom.py       # NEW
  run_vtc_fairness_comparative_sweep.py  # NEW
  verify_vtc_fairness_sweep.py         # NEW
tests/
  test_vtc_fairness_headroom.py        # NEW (16 tests)
  test_vtc_micro_traces.py             # NEW (4 tests)
```

See `docs/audits/vtc_fairness_comparative_evaluation_20260805.md` for the
full comparative-sweep results, independent verification, and the
scientific classification decision this repair enabled.
