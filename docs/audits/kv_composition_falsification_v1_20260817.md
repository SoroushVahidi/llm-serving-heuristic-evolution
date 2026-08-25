# KV-Aware Composition Falsification v1 — Audit

**Date:** 2026-08-17
**Verdict:** `KV_COMPOSITION_INCONCLUSIVE`
**Run:** [`experiments/kv_composition_falsification_v1_20260817T172446Z/`](../../experiments/kv_composition_falsification_v1_20260817T172446Z/)
**Design (frozen before scoring):** [`docs/design/KV_COMPOSITION_FALSIFICATION_V1.md`](../design/KV_COMPOSITION_FALSIFICATION_V1.md)
**Launch commit:** `5b59306` (design + code + tests committed before the run)
**Parents (unmodified):** `kv_constrained_online`, `least_laxity_first`

This is the first composition falsification attempted for a pair whose
pairwise-separation pilot reached `_COMPOSITION_READY` (KV v2). The verdict
is driven by a genuine, specific, previously-undetected finding — not by
absence of signal. Full detail in sections below.

## A. Pre-composition KV v2 evidence audit

Summarized in design doc §1 (full detail there). Key points:

- v2's own 10/10 gates and G5/G10 timing evidence are a **between-scenario
  aggregate** finding (`urgent_arrival_phase` is a scenario-generation
  parameter).
- New re-simulation work (this task, before any composition code was
  written) directly measured `sim._gpus[0].step_kv_used` and per-request
  admission times on several frozen v2 scenarios and found: (1) raw KV
  occupancy saturates almost immediately (~83-100%) throughout the whole
  trajectory in the signal regime — **not** a usable within-trajectory
  switch signal, confirming v2's own §D finding first-hand; (2) the count of
  currently-waiting urgent-classified requests genuinely varies within a
  single trajectory (0 to 13 in one directly-measured scenario, 16% of steps
  at exactly 0); (3) the actual mechanistic differentiator is
  `kv_constrained_online`'s strict urgent-first two-tier sort vs
  `least_laxity_first`'s continuous-laxity sort; (4) individual urgent
  requests' SLO outcomes vary systematically by their own arrival order
  *within* single re-simulated scenarios (e.g. seed 20260912/middle:
  `least_laxity_first` succeeds on request 1 and 9, fails on the other 8).

## B. Evidence that within-scenario composition is genuinely plausible

**Plausible, not proven** (design doc §1G) — stated precisely: the
mechanism (urgent-vs-bulk queue contention at the moment a KV slot frees up)
plausibly operates continuously within any one run, but this was not
literally read off a continuous trajectory in v2's frozen gates. This
falsification's own G1 result (§I below) is the actual confirming test.

## C. Child mechanism and why it is minimal

`KVAdaptiveReserveChildPolicy`: every `select_action()` call delegates,
unmodified, to a fresh-instance call of `KVConstrainedOnlinePolicy` or
`LeastLaxityFirstPolicy`. Mode = `"reserve"` if the count of currently-
waiting urgent-classified requests (using `kv_constrained_online`'s own
`urgent_laxity_seconds` threshold — no new mechanism) is ≥ `tau_urgent`,
else `"llf"`. One free parameter. No new admission logic is invented.

## D. Observable vs forbidden features

Design doc §1E / `kv_composition_features.FORBIDDEN_FEATURE_KEYS`. Only
`ObservableRequest`/`ObservableState`/`ObservableGPUState` fields used.

## E. TRAIN / VAL / TEST / OOD split

Reuses KV v2's own preregistered seed partition: TRAIN=`{20260910,20260911}`
(24 scenarios), VAL=`{20260912}` (12), TEST=`{20260913}` (12),
OOD=`{20260914,20260915}` (24). Identical scenario grid to KV v2 (72 total).

## F. Threshold/rule fitting procedure and result

`tau_urgent ∈ {1,2,3}` fit on TRAIN (mean ANWG: `{1: 0.8794, 2: 0.8831, 3:
0.8773}` → argmax `tau=2`). VAL confirmation check: `{1: 0.8490, 2: 0.8441,
3: 0.8333}` — **VAL slightly preferred `tau=1` over TRAIN's argmax**
(`val_confirmed_not_worse_than_runner_up = False`). Honestly reported: the
runner's frozen procedure proceeds with TRAIN's argmax regardless (no abort
branch was specified in the design), and the disagreement is small (all
three candidates are within ≈0.016 ANWG of each other on VAL, ≈0.006 on
TRAIN) — this reads as **the tiny threshold grid not being sharply resolved
by 24 TRAIN / 12 VAL scenarios**, not as a large or alarming inconsistency,
but it is a real limitation of the fitting procedure at this sample size,
disclosed here rather than glossed over. `tau_urgent=2` is the frozen value
actually used for every TEST/OOD score below (`child_threshold.json`,
written before any TEST/OOD scenario was simulated for the child).

## G. Preregistered hypotheses — outcome

- **H1** (switch signal non-constant on held-out data): **supported** — see
  §I.
- **H2** (child beats parent_oracle on some held-out scenarios by >ε):
  **supported** — 5/12 TEST, 3/24 OOD (§N).
- **H3** (selector-sufficiency null): **partially supported but
  uninformative** — see §T's caveat on the selector collapsing to a constant
  classifier in this data.
- **H4** (non-degeneracy): **supported** — §I.

## H. Preregistered gates — mechanical result

| Gate | Result | Detail |
|---|---|---|
| G1 Non-degeneracy | **PASS** | 24/36 held-out (TEST∪OOD) scenarios show both modes active, ≥1 transition, ≥1 admission decision differing from both fixed-mode replays (design doc §8) — far above the "≥1" threshold |
| G2 TEST envelope expansion | **PASS** | `G_ε(child;P)` on TEST = 0.01887 > 0 |
| G3 CI support | **FAIL** | Paired bootstrap 95% CI on TEST `child−E_P`: mean=0.00980, **lo=−0.01912**, hi=0.03775 — lower bound does not exceed 0 |
| G4 Beats both parents | **PASS** | 5/12 TEST scenarios have `child > kv_constrained_online+ε` **and** `child > least_laxity_first+ε` simultaneously (3/24 on OOD) |
| G5 Beats selector | **PASS (via OR clause)** | `mean(child−contextual_top1)` on TEST = 0.00980 (just under ε=0.01, so the direct clause narrowly misses) — but `contextual_top1`'s gap to `parent_oracle` on TEST is exactly 0.0 (< 0.005), and G2 passed, satisfying the design doc's OR clause |
| G6 OOD directional replication | **PASS** | TEST mean(child−E_P)=+0.00980, OOD mean(child−E_P)=+0.00037 — same sign, though OOD magnitude is much smaller |
| G7 Safety/feasibility | **FAIL** | 0 failed evals, 0 duplicate rows, 0 NaN/Inf, 0 completion-fraction regressions, leakage guard passed — **but** child's peak-KV-utilization-ratio exceeds `max(parent peak ratios)` on **6/36 (16.7%)** held-out scenarios (magnitude: exceeds by 0.013–0.033 of an already->100%-capacity baseline both frozen parents themselves exhibit — see §V) |
| G8 Sample adequacy | **PASS** | TEST=12≥8, OOD=24≥8 |

**Mechanical verdict application (design doc §6):** `KV_COMPOSITION_GO`
requires all of G1-G8 — **not met** (G3, G7 fail).
`KV_COMPOSITION_INCONCLUSIVE` triggers explicitly "if G1 fails... G7 fails,
or G3's CI is too wide..." — **G7 failing alone forces this verdict**,
applied exactly as frozen before this result was seen.

## I. Non-degeneracy definition — result

24/36 held-out scenarios pass all three sub-conditions simultaneously (both
modes active, ≥1 transition, ≥1 admission decision differing from both pure
replays). Representative examples (scenario / n_llf_steps / n_reserve_steps
/ transitions / n_differ-from-both-parents):
`kvp2.bulk24.phasemiddle.tighttight.s20260913` — 1026/2537/4/23;
`kvp2.bulk24.phaselate.tightloose.s20260914` — 1537/1284/2/25. The 12
degenerate held-out scenarios are concentrated in the `bulk_pressure=high,
seed=20260915` sub-block, where `n_admission_decisions_differ_from_both_
parents=0` despite both modes being active in several — i.e., the child's
*mode choice* differs from a fixed replay but happens to produce the exact
same admission *outcome* on that particular seed's request stream. This is
expected occasionally and does not itself indicate a bug (confirmed by the
focused test suite's separate, directly-verified example, §K).

## J. Implementation changes

New (`src/llmserveopt/composition/`): `kv_composition_features.py`,
`kv_composition_splits.py`, `kv_composition_policy.py`,
`kv_composition_metrics.py`. New: `scripts/run_kv_composition_falsification_v1.py`,
`scripts/smoke_kv_composition_falsification_v1.py`,
`configs/kv_composition_{falsification,smoke}_v1.yaml`. Zero changes to
`kv_constrained_online.py` / `least_laxity_first.py` (contract test, §K).

## K. Tests

26 new focused tests (`tests/test_kv_composition_falsification_v1.py`), all
pass: parent-contract-unchanged, forbidden-feature-key leakage guard (×3),
scenario determinism/uniqueness/leakage-guard, `n_urgent_waiting` genuinely
varies within one trajectory, child low-pressure-control mostly-llf-mode,
high-pressure-signal-cell activates reserve mode, child can transition and
differ from both parents, **child KV-overflow no worse than both parents**
(revised during test-writing after discovering the pre-existing baseline
property, §8b of the design doc — the same property responsible for G7's
failure above, caught early by direct measurement rather than assumed away),
reset-clears-instrumentation, selector dataflow signature guard (no
test/ood parameter exists to leak into), selector/hard-rule
no-leakage/validity, envelope/bootstrap/oracle-regret metric unit tests,
tau_urgent grid sanity. 64/64 total across
`test_policy_separation_kv_pressure_v1.py` + `_v2.py` +
`test_kv_composition_falsification_v1.py`.

## L. Smoke/calibration history

One round (`scripts/smoke_kv_composition_falsification_v1.py`, 12 cells, 1
seed, `tau_urgent=2` — the design-doc-anchored default, not tuned to this
result): `both_modes_seen=True`, `any_transition=True`,
`any_behavioral_diff_from_both_parents=True`, `n_failures=0`, `n_nan_inf=0`,
`mean_reserve_fraction`: `low_bulk_pressure=0.329` vs
`high_bulk_pressure=0.707` (regime coverage confirmed, not winner-direction
tuned). A separate wiring dry-run (6 scenarios, all 6 seeds, single factor
cell disjoint from any smoke/final-run threshold decision) confirmed the
full runner pipeline end-to-end before freezing — no threshold or code was
changed as a result of anything observed in either check.

## M. Launch-gate verdict

**PASS.** Both mechanism-activation and regime-coverage criteria met; no
manufactured workload; proceeded to the frozen full run.

## N. Frozen launch SHA/config

Commit `5b59306` (design doc + all composition code + tests, committed
**before** the full run was launched). Config:
`configs/kv_composition_falsification_v1.yaml` (identical grid to KV v2's
own `kv_pressure_pilot_v2.yaml`). `datasets_root=.local_data` (staged
BurstGPT, `BurstGPT_without_fails_1.csv`, md5
`a68f7783b3b2d143b6cd1f102163d0f2`).

## O. Full-run command and run directory

```
tmux new-session -d -s kv_composition_falsification_v1 \
  "python scripts/run_kv_composition_falsification_v1.py \
   --config configs/kv_composition_falsification_v1.yaml \
   --run-dir experiments/kv_composition_falsification_v1_20260817T172446Z \
   --workers 4 --datasets-root .local_data"
```

Run directory: `experiments/kv_composition_falsification_v1_20260817T172446Z/`.
Completed naturally in ~4 seconds (well under the 3-minute monitoring
window); verified directly per task item 13's fallback clause.

## P. Run integrity

72 scenarios, 576 rows (72 × 8 methods), **0 failed**, 0 duplicate
`(scenario_id, method_name)` pairs, 0 NaN/Inf, split sizes exactly as
designed (train=24, val=12, test=12, ood=24).

**Important, independently-discovered finding (not part of any preregistered
gate, but directly relevant to the whole KV family's provenance):** cross-
checking this run's `kv_constrained_online`/`least_laxity_first` parent
scores against the frozen, committed KV v2 CSV
(`experiments/kv_pressure_pilot_v2_20260817T165053Z/per_policy_results.csv`)
on the identical 144 `(scenario_id, policy)` pairs found **99/144 (69%)
mismatched**, up to 0.25 ANWG apart. Verified this is **not** a bug in the
new composition code: re-running the *original, unmodified* KV v2 pilot
runner (`scripts/run_policy_separation_kv_pressure_pilot_v1.py
--template-version v2`) against the exact same frozen config in the current
environment reproduces the same 99/144 mismatch against the committed CSV,
while being perfectly self-reproducible across two independent
single-worker reruns (0/144 mismatch). This means **the current environment
deterministically reproduces itself but not the historical frozen v2 CSV** —
a genuine, previously-undetected reproducibility gap somewhere between the
BurstGPT data snapshot / library versions used when v2 was originally run
and now. Root cause not identified (no dataset checksum was recorded in the
original v1/v2 run provenance to compare against); flagged here as an open
finding relevant to the whole KV v1/v2 evidentiary chain, not just this
falsification. **This falsification's own gates remain valid** because
every method compared here (parents, child, selector, oracle) was computed
from one single, internally self-consistent run/data snapshot — the
mismatch only affects comparability against the *historical* v2 numbers,
not the internal validity of this run's own relative comparisons. See §Z
for the recommended follow-up.

## Q. TEST results

n=12 (seed 20260913). Mean ANWG: `kv_constrained_online`=0.8517,
`least_laxity_first`=0.7676, `parent_oracle`=0.8517 (kv wins/ties every TEST
scenario — see §S), `best_fixed_parent`=0.8517, `contextual_top1`=0.8517,
`hard_conditional`=0.7676, `kv_adaptive_reserve_child`=0.8615,
`oracle_after_child`=0.8748. Envelope gain `G`=0.02304, `G_ε`=0.01887.
Bootstrap CI on `child−E_P`: mean=0.00980, 95% CI=[−0.01912, 0.03775].
Beats-both-parents: 5/12. Selector regret to `parent_oracle`: 0 (selector
== `kv_constrained_online` on every TEST scenario, §T).

## R. OOD results

n=24 (seeds 20260914-15). Mean ANWG: `kv_constrained_online`=0.8615,
`least_laxity_first`=0.7331, `parent_oracle`=0.8615, `best_fixed_parent`=0.8615,
`contextual_top1`=0.8615, `hard_conditional`=0.7331,
`kv_adaptive_reserve_child`=0.8619, `oracle_after_child`=0.8676. Envelope
gain `G`=0.00613, `G_ε`=0.00488. Bootstrap CI on `child−E_P`: mean=0.00037,
95% CI=[−0.00944, 0.01017]. Beats-both-parents: 3/24. **TEST and OOD are
reported separately, never averaged**, per task instruction.

## S. Parent-envelope expansion

Positive on both TEST (+0.019 practical gain) and OOD (+0.005 practical
gain), directionally consistent (G6) but an order of magnitude weaker on
OOD. **Important caveat:** in this run's 72-scenario data,
`kv_constrained_online` achieves the max of the two parents on **every
single scenario** (`parent_oracle` mean exactly equals
`kv_constrained_online` mean on all four splits) — `least_laxity_first`
never wins outright anywhere in this specific dataset (0/72, vs the
historical v2 CSV's 4/48 — directly attributable to the reproducibility gap
in §P, not a property of the mechanism itself). This means "child beats the
envelope" and "child beats `kv_constrained_online`" are the *same statement*
in this run — the envelope-expansion evidence here is real but narrower in
character than the design doc anticipated (it does not additionally
demonstrate the child navigating genuine llf-favoring vs kv-favoring
sub-regions, because no held-out scenario in this run actually favors llf).

## T. Selector-vs-child comparison

`contextual_top1` collapsed to a constant classifier in this run (TRAIN
labels are single-class — `kv_constrained_online` wins every TRAIN scenario
by >ε, so the fitting code's single-class fallback, `_ConstantClassifier`,
triggered; `selector_val_accuracy=1.0` trivially). `hard_conditional`
similarly resolves to `least_laxity_first` on every scenario (its
`fraction_urgent_waiting≥0.15 and n_queued≥10` condition is never
simultaneously true on this grid). Consequently `contextual_top1 ≡
kv_constrained_online ≡ parent_oracle` and `hard_conditional ≡
least_laxity_first` exactly, everywhere, in this run. G5 ("beats
selector") is technically satisfied (§H) but **is not a meaningful
selector-vs-composition comparison here** — it reduces to "child beats
kv_constrained_online," already captured by G4. This is a direct
consequence of §S's dominance finding, itself downstream of §P's
reproducibility gap, not a new independent result.

## U. Within-scenario mechanism evidence

Both the design doc's direct re-simulation work (§A) and this run's own
instrumentation (§I) support genuine within-trajectory switching: on
non-degenerate held-out scenarios the child spends a real mix of steps in
each mode (mean reserve-fraction: 0.498 TEST, 0.471 OOD — not
collapsed to one mode), transitions multiple times per scenario (mean 4.7
TEST, 3.6 OOD), and produces admission decisions distinguishable from both
pure-parent replays on the majority of held-out scenarios (24/36).

## V. Safety/feasibility

Completion fraction: 1.0 on every held-out scenario for every method (0
regressions). 0 failures/duplicates/NaN/Inf/leakage-guard failures. **KV
peak-utilization-ratio check (§H, G7): fails on 6/36 (16.7%) held-out
scenarios.** All 6 violations are small in magnitude (child exceeds
`max(parent peaks)` by 0.013-0.033, e.g.
`kvp2.bulk24.phasemiddle.tighttight.s20260913`: child=1.198 vs
max(kv=1.147, llf=1.166)=1.166) and occur against a baseline where *both*
frozen parents already routinely exceed nominal 100% capacity by up to 20%
(a pre-existing simulator/policy property — KV usage grows during decode
past the admission-time capacity check, unrelated to this composition
work, §8b of the design doc). This is nonetheless a real, novel,
mechanistically-plausible finding specific to composition: alternating
between two different admission policies mid-trajectory can put the system
into a KV-pressure state that *neither pure policy alone* would reach,
because the identity of currently-active requests at any given step depends
on the *history* of mode choices, not just the current mode. This is
exactly the kind of composition-specific risk a pairwise-separation pilot
(which only ever runs one fixed policy per scenario) structurally cannot
surface — a genuinely new piece of evidence from this falsification.

## W. Final mechanical verdict

**`KV_COMPOSITION_INCONCLUSIVE`** — applied exactly per the frozen decision
rule (design doc §6): G7 (safety) fails, which explicitly forces this
verdict regardless of G1-G6's largely favorable results. Not
`KV_COMPOSITION_GO` (G3, G7 unmet). Not
`KV_SELECTION_SUFFICIENT_FOR_THIS_PAIR` (that verdict requires G2 or G4 to
*fail*, i.e., requires the child to show no envelope-gain signal at all —
the opposite of what happened here; G2 and G4 both pass).

## X. Scientific interpretation

This is a **qualitatively different outcome from the two prior
`SELECTION_SUFFICIENT_FOR_THIS_PAIR` verdicts** (ESTF/WFS, PrefillControl
v2) and is not simply "another negative result." The evidence here shows
real promise (positive envelope gain on TEST, 5/12 and 3/24 beat-both
counts, directionally-consistent OOD replication, genuine non-degenerate
within-trajectory mode-switching on two-thirds of held-out scenarios) that
is specifically blocked by a concrete, mechanistically-understood, and
plausibly *fixable* problem: naive mode-switching can transiently push KV
pressure higher than either parent alone would allow. This is not evidence
that within-scenario composition is impossible for this pair — it is
evidence that **the frozen fitting procedure was too permissive about the
child's transition behavior** (no smoothing/hysteresis or transition-aware
admission cap was included in this deliberately minimal v1 child, per the
task's explicit "keep the child simple" instruction). Per task item 15,
this outcome does **not** license escalating to a more complex child,
MAP-Elites, or synthesis — it licenses, at most, a narrowly-scoped
follow-up that directly targets the diagnosed safety mechanism (e.g., a
transition-aware admission cap), evaluated with the same rigor, not a
broader search.

Separately and independently of the composition question: §P's
reproducibility-gap finding is a real, standing infrastructure concern for
the whole KV family's evidentiary chain (v1 and v2 alike) that this task
did not set out to find and does not resolve — flagged for a dedicated
follow-up (§Z), not silently absorbed into this result.

## Y. Git commit/push state

Design + code + tests: `5b59306` (pushed as part of this task's final
commit sequence). This audit + status-doc updates: committed and pushed
separately (see final report §Y for exact SHAs). Frozen v1/v2 KV artifacts:
untouched.

## Z. Exact next action

Two independent, unstarted threads, neither of which is composition
escalation (task item 15's stop condition applies):

1. **Composition-adjacent (smallest defensible next step):** a narrowly
   re-scoped `KVAdaptiveReserveChildPolicy` variant that adds a
   transition-aware admission cap (e.g., suppress admission for one step
   immediately after a mode transition, or require `tau_urgent` to persist
   for ≥2 consecutive steps before switching) specifically targeting the
   §V safety finding — re-run through the identical frozen falsification
   procedure, not a broader search.
2. **Infrastructure (independent, arguably higher priority):** investigate
   and resolve §P's reproducibility gap — record a BurstGPT dataset
   checksum (and library version manifest) in every future pilot's
   provenance, and determine whether the historical v1/v2 KV CSVs need a
   documented caveat or a from-scratch reproducibility re-verification.

Neither is started in this task.
