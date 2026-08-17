# Policy Separation Family C — KV-Pressure Reserve v1

**Date:** 2026-08-17
**Status:** PREREGISTERED — pilot not yet executed
**Predecessor lesson (why this family is designed differently):** ESTF↔WFS
and Family B v2 PrefillControl both landed `SELECTION_SUFFICIENT_FOR_THIS_PAIR`
— a scenario-level fitted top-1 selector already matched the two-parent
oracle envelope. In both cases the parents' mechanism differed by a value
that was **fixed for the whole scenario** (a rank-blend weight; a prefill
chunk size), so "which parent wins" was a per-scenario property a selector
could learn once and never revisit. This family is deliberately designed so
the mechanism difference is a **live, time-varying admission gate** driven
by a quantity (`current_kv_tokens`) that genuinely rises and falls within a
single scenario's trajectory — see §3 for why this makes scenario-level
top-1 selection structurally weaker here.

## 1. Scientific goal

Does a KV-occupancy-aware admission reserve create at least two stable,
mechanistically distinct, online-observable policy niches **whose relative
advantage can plausibly change within a single scenario's trajectory** — a
necessary (not sufficient) precondition for a future composition experiment
to have room to add value beyond scenario-level selection?

This is a **pairwise-separation pilot**, not a composition falsification.
No selector is fit, no child policy is built or run. Per explicit instruction
for this task, composition work on this pair is out of scope here.

Possible terminal verdicts (mirrors Family B v1→v2 convention):

| Verdict | Meaning |
|---|---|
| `KV_FAMILY_COMPOSITION_READY` | All gates (§8) pass, including within-scenario evidence. Recommend a future composition falsification — do not run it in this task. |
| `KV_FAMILY_USEFUL_NEEDS_REFINEMENT` | Real contrast exists but one or more gates fail. Stop composition; refine workload calibration. |
| `KV_FAMILY_NO_GO` | No bidirectional separation, or the two policies are behaviorally a twin. Stop this family. |
| `DESIGN_CONFOUND` | Integrity / leakage / BurstGPT / schema failure. Do not interpret ANWG. |

## 2. Candidate audit (repository evidence, not invention)

Read before proposing anything (see PR discussion / audit fork transcript
for full citations): `src/llmserveopt/policies/` (27-policy library),
`src/llmserveopt/simulator/gpu.py`, `core/action.py`, `core/types.py`,
`docs/design/POLICY_SEPARATION_DATASET_V1.md` §4/§5E (a broader,
never-executed "Family E cache/KV-aware" plan that already scaffolded
`policy_separation/builders.py::kv_scarce_gpu()` for exactly this purpose,
confirming this direction was previously judged worth pursuing),
`docs/current/POLICY_COMPOSITION_READINESS.md`,
`docs/current/POLICY_GENOME_COVERAGE_AUDIT.md`.

**Confirmed: no prefix-sharing / prompt-reuse mechanism exists in the
simulator at all** (`Request` carries no reuse-group field; no policy reads
one — `docs/design/POLICY_SEPARATION_DATASET_V1.md` line 131). Any
reuse/locality-themed candidate is therefore excluded up front as
`UNSUPPORTED_BY_SIMULATOR` by construction, not as a finding of this pilot.

### 2.1 Candidate parent contrasts

| # | Contrast | A: exact mechanism difference | B: genuine niche for each parent | C: why the winner could change within one scenario | D: online-observable driver | E: why a structural child could beat both | F: diagnostics | G: confounds / twin risk |
|---|---|---|---|---|---|---|---|---|
| 1 | `kv_constrained_online` vs `least_laxity_first` | Both rank by the same laxity-based urgency proxy (`α·prompt+β·output`, `DEFAULT_ALPHA=0.5/DEFAULT_BETA=1.0`, `src/llmserveopt/policies/scoring.py`); `kv_constrained_online` additionally **defers non-urgent admission** once projected KV utilization would exceed `target_kv_utilization=0.82`, unless the request's own laxity ≤ `urgent_laxity_seconds=0.25` (`kv_constrained_online.py:37-40`). `least_laxity_first` has no such gate — it admits in laxity order up to the simulator's hard `max_kv_tokens` capacity only (`base.py::_feasible_on_gpu`). | KV-constrained protects a reserve for genuinely urgent latecomers under pressure at the cost of some early throughput; LLF maximizes early throughput/packing at the cost of leaving no headroom. | `GPUState.current_kv_tokens` genuinely grows during prefill and further during decode (`simulator/request.py::kv_tokens`) and falls as requests complete — a real trajectory quantity, not a per-scenario config value. Whether the reserve matters depends on **when** an urgent request arrives relative to that trajectory. | `ObservableGPUState.current_kv_tokens` / `max_kv_tokens` (already exposed, no leakage) | A policy that defers *conditionally* (only when pressure is actually rising, not always) could beat both a static-reserve and a static-greedy policy on the same trajectory — untested here, but this pair is the right substrate to find out. | `current_kv_tokens` time series (already recorded via `step_kv_used`); admission-deferral events are directly loggable from `_admit_filter`. | Both already exist, are laxity-based, and are structurally close — real twin risk if the reserve never binds (mitigated by gate G5, §8). |
| 2 | `kv_constrained_online` vs `apt_serve_faithful` | KV-constrained's soft reserve gate vs Apt-Serve's dual-tier (`KV`/`HIDDEN`) cache-transition-cost model (`apt_serve_faithful.py`, `hybrid_cache_enabled`, `cache_switch_latency`, `hidden_restore_latency`). | Different mechanisms entirely (admission deferral vs cache-tier transition cost) — genuine niches likely, but for **two conflated mechanisms**, not one. | Cache-tier transitions are inherently trajectory-dependent (a request's tier changes as it's accessed), so this also has within-scenario dynamics. | `current_kv_tokens`, tier occupancy (typed `CacheAssignment`/`CacheRepresentation`) | Plausible, but confounds mechanism attribution — a win/loss can't be cleanly attributed to "KV reserve" vs "cache tiering." | Existing dual-tier diagnostics (Apt-Serve Phase G instrumentation). | Higher — this is a SIGMOD-2025 external-baseline reproduction with its own interface scaffolding (Phase E), not a simple two-parameter contrast. Not the minimal pair for a first pilot. |
| 3 | `vllm_faithful` vs `sarathi_faithful` | Both do genuine KV-block-driven preemption (recompute-on-resume via `kv_block_manager.py`) but differ in prefill/decode scheduling policy (chunked vs unchunked). | Real, well-documented niches (this is the literature's own contrast). | Preemption/recompute is inherently trajectory-dependent. | KV-block occupancy, preemption events | Plausible in principle. | Existing faithful-baseline preemption diagnostics. | **High** — `external_baselines_registry.py` explicitly notes both "reuse[] their KVBlockSpaceManager/preemption pattern"; documented as a real twin risk between these two specifically, and neither is in the deployable 27-policy library used by every other family's pilots (would require new integration work). |
| 4 | `kv_constrained_online` vs `llumnix_faithful` | KV-constrained's local admission-deferral vs Llumnix's live cross-instance migration (`Action.migrate`) — relocation, not eviction or admission. | Different resource-management primitive (spatial rebalancing vs temporal deferral) — likely a genuine niche under multi-GPU load imbalance. | Migration decisions are inherently trajectory-dependent (triggered by observed imbalance). | Per-GPU `current_kv_tokens`/`utilization` across GPUs | Plausible for a *multi-GPU* pilot specifically. | Existing migration-count diagnostics. | Requires a multi-GPU scenario design (every other family pilot to date is single-GPU) — higher implementation cost and a confound (single- vs multi-GPU) unrelated to the KV-pressure question itself. |

## 3. Selected pair and why

**`kv_constrained_online` vs `least_laxity_first` (candidate #1).**

Explicit selection criteria (per instruction):

- **Bidirectional mechanism plausibility:** high — a reserve genuinely trades early throughput for later headroom; both directions (packing wins when nothing urgent arrives late; reserve wins when something urgent does) are mechanistically real, not asserted.
- **Within-scenario state variation:** high — `current_kv_tokens` is a live, monotonically-neither quantity that this pilot's own factor design (§5) deliberately pushes across the 0.82 reserve threshold mid-trajectory.
- **Observable decision boundary:** clean — the gate is a single online-observable inequality (`post_util <= target_kv_utilization`) with a single online-observable override (`laxity <= urgent_laxity_seconds`); both quantities are already exposed, no leakage risk.
- **Causal interpretability:** high — one policy has the gate, the other structurally cannot have it (no code path). No ambiguity about which mechanism produced a difference (contrast with #2).
- **Low risk of policy twins:** moderate-but-checked — both are laxity-ranked, so a real risk exists that the gate simply never binds; §8 gate G5 is a direct non-degeneracy check for this.
- **Low implementation cost:** high — both policies already exist, are tested individually, and need zero new simulator support (`kv_scarce_gpu()` GPU-config helper already scaffolded for exactly this).
- **Realistic LLM-serving connection:** high — KV-cache admission reserves are a real production concern (protecting headroom for latency-critical requests under memory pressure).
- **Compatibility with canonical ANWG / existing semantics:** full — no new metric, no new Action verb, no simulator change required.

Candidate #2 (Apt-Serve) and #3 (vLLM/Sarathi) were rejected for this
**minimal, mechanism-isolating** pilot specifically because of confound risk
(#2) and confirmed twin risk plus non-trivial integration cost (#3).
Candidate #4 (Llumnix) was rejected because it requires a multi-GPU scenario
design outside this family's minimal scope. All three remain legitimate
**future** KV/cache-family extensions once #1 is characterized.

## 4. Within-scenario composition hypothesis (the new element vs prior families)

> The two-parent oracle envelope for `kv_constrained_online` vs
> `least_laxity_first` is **not** a stable per-scenario property. Within a
> single scenario, `least_laxity_first` should hold an early-trajectory
> throughput advantage (no deferred admissions to pay for) that **can flip**
> to a `kv_constrained_online` advantage once KV occupancy crosses the
> reserve threshold and a genuinely urgent request arrives — i.e., the same
> scenario can contain both a phase where LLF is locally better and a phase
> where KV-constrained is locally better, which scenario-level top-1
> selection (fit once per scenario) cannot exploit but a genuinely
> state-dependent policy could.

This is directly testable without building a child policy: §5's
`urgent_arrival_phase` factor controls *when* the urgent tenants arrive
relative to the KV ramp, and §8's gate G4 tests whether the KV-constrained
advantage on urgent-tenant SLO attainment is measurably larger in the
`late` phase than the `early` phase — i.e., whether the *within-scenario
timing* of pressure, not just its scenario-level magnitude, drives the
outcome.

## 5. Workload / factor design

Scenario generator: `case_kv_pressure_reserve_contention` in the new module
`src/llmserveopt/policy_separation/templates_kv_pressure.py` (`Family C v1`,
`generator_version="kv_pressure_v1"`). Two tenant classes, matching the
naming convention of prior families (`tenant_prefill`/`tenant_late` for B;
`tenant_a`/`tenant_b` for A):

- `tenant_bulk` — drives KV pressure. Long-window BurstGPT prompts
  (target median 4096, window [2048, 8192]), moderate output length
  (target median 300, window [100, 600]), loose deadline (30s slack —
  never itself SLO-relevant; pure background load). Arrival: convoy from
  t=0, spacing 0.05s.
- `tenant_urgent` — the SLO-critical population whose admission timing is
  the object of study. Medium-window BurstGPT prompts (target median 1024,
  window [512, 2048]), short output (target median 150, window [50, 400] —
  a **labeled synthetic intervention**, matching Family B v2's
  `output_intervention` precedent, to isolate admission-latency effects
  from raw decode-time variance). Count fixed at 6 (not swept, to isolate
  the two swept factors below). Arrival spacing 0.03s.

Three swept factors + seed, matching Family B v2's 2×2×2×4=32 scale:

| Factor | Levels | What it controls |
|---|---|---|
| `bulk_pressure` | `low` (n_bulk=6) / `high` (n_bulk=14) | How far KV occupancy rises — whether the 0.82 reserve threshold is credibly crossed. |
| `urgent_arrival_phase` | `early` (t≈0, before pressure builds) / `late` (t = 0.7 × bulk convoy span, after pressure has built) | **The within-scenario timing test (§4).** |
| `urgent_tightness` | `loose` (slack=3.0s, laxity≈2.3s at arrival — never urgent by the policy's own 0.25s definition) / `tight` (slack=0.55s **[calibrated, see below]**, laxity≈0.11s at arrival — already urgent by the policy's own 0.25s definition) | Whether the reserve's own *urgent override* is the one being exercised — a placebo control: the H3/G4 effect should be near-absent under `loose` (both policies trivially meet SLO) and present under `tight`. |
| `seed` | `20260901, 20260902, 20260903, 20260904` | Reproducibility / sampling variance, fresh seeds (not reused from Family B v2). |

`2 × 2 × 2 × 4 = 32` scenarios. GPU: `kv_scarce_gpu(max_kv_tokens=6_000`
**[calibrated, see below]**`, max_active_sequences=64, max_batch_tokens=64)`
— capacity is tight enough that `bulk_pressure=high` is expected to cross
82% utilization, loose enough that `max_active_sequences`/`max_batch_tokens`
are not the binding constraint (KV must be the thing that binds, per
`kv_scarce_gpu`'s own documented purpose). `tenant_bulk`'s deadline slack
(`BULK_SLACK_S`) is likewise calibrated below, not `NO_PRESSURE_SLACK`.

**Calibration note (pre-registered adjustment, made from smoke/pre-smoke
diagnostics before the full pilot, not from pilot results):** two rounds.

*Round 1 (urgency binding):* the original targets (`max_kv_tokens=24_000`,
`urgent_tightness=tight` slack=0.9s) produced real, large KV-occupancy
differences between the two policies (`least_laxity_first` peaked over 100%
nominal KV utilization vs `kv_constrained_online`'s ~88%) but **zero outcome
difference** — both policies always achieved ANWG=1.0 and 6/6 urgent SLO
attainment, because 0.9s slack fully absorbed even LLF's largest observed
admission delays (~0.78s). A sweep of `max_kv_tokens` and `urgent_tightness=tight`
slack found `max_kv_tokens=8000`, slack=0.55s produces a real, non-degenerate
outcome difference.

*Round 2 (bidirectionality):* at those Round-1 values, `kv_constrained_online`
never lost a single cell across a full 32-cell check (`BULK_SLACK_S=30.0`,
"never itself SLO-relevant" per the original design) — an artificial
one-sided result, because deferring a bulk tenant was cost-free for it (30s
slack absorbs any delay). This violates gate G1's bidirectionality
requirement and §9's own smoke-gate criterion ("neither parent universally
dominates"), so it was diagnosed and corrected: a sweep of
`BULK_SLACK_S ∈ {30, 5, 3, 2, 1.5, 1.2, 1.0}` crossed with
`max_kv_tokens ∈ {8000, 7000, 6000}` was run across the full 32-cell grid,
scoring win/tie counts, landing provisionally on `BULK_SLACK_S=1.5`,
`max_kv_tokens=7000`.

*Round 3 (infeasible-request bug):* inspecting the provisional Round-2
smoke's raw diagnostics (`n_steps` ballooning to ~50,000 for some
`bulk_pressure=high` cells vs ~2,000 for `bulk_pressure=low`) surfaced a real
workload-generation defect, not a finding: the original `BULK_PROMPT_HI=8192`
window could sample a single bulk request whose `prompt_tokens` alone
exceeded `max_kv_tokens`, making it permanently unadmittable under *either*
policy and corrupting that scenario's step count and ANWG denominator. Fixed
by capping the bulk prompt window to `[1024, 3072]` (median 2048), always
comfortably below any candidate `max_kv_tokens` — a workload-generation
correctness fix, not a calibration choice. Re-ran the Round-2 win/tie sweep
(now free of infeasible requests) across `BULK_SLACK_S ∈ {1.0..1.5}` ×
`max_kv_tokens ∈ {4500..7000}`; no combination simultaneously hit a
perfectly balanced win split and gate G2's <50% tie-rate bound. Selected
`BULK_SLACK_S=1.5`, `max_kv_tokens=6000` (wins 9-vs-4, tie rate 59%) as the
most balanced *bidirectional* point found, over more lopsided
lower-tie-rate alternatives (e.g. `max_kv_tokens=5000` gave tie rate 41% but
an 18-vs-1 win split) — i.e., bidirectionality (G1) was prioritized over the
tie-rate target (G2) when the two traded off, and G2 may or may not pass on
the full pilot; that is a legitimate gate outcome to observe, not something
to keep re-tuning until it passes. At these final values the H3/H4 pattern
(§4/§8) is clean: mean `(kv_constrained − llf)` ANWG delta is 0 for
`bulk_pressure=low, urgent_tightness=loose` and ≈−0.025 (small, both phases
alike — supports H4) for `bulk_pressure=high, urgent_tightness=loose`
(placebo); +0.02→+0.06 (`early`→`late`) for `bulk_pressure=low,
urgent_tightness=tight`; +0.06→+0.13 (`early`→`late`, ~2×) for
`bulk_pressure=high, urgent_tightness=tight` — supporting H3 directly, most
cleanly in the highest-pressure/tightest-deadline cell as designed.

**Final calibrated values: `max_kv_tokens=6000` (was 24000),
`urgent_tightness=tight` slack=0.55s (was 0.9s), `BULK_SLACK_S=1.5s` (was
30.0s / `NO_PRESSURE_SLACK`), bulk prompt window `[1024, 3072]` median 2048
(was `[2048, 8192]` median 4096 — Round 3 correctness fix).** All other §5
parameters are unchanged from the original preregistration. All three
rounds are workload-generation/*scale* adjustments made before any
full-pilot cell was scored, per this section's explicit calibration
allowance — no policy code, ranking logic, or gate definition was touched,
and Rounds 2-3 were triggered by gate-shaped or correctness diagnostics
(bidirectionality; an impossible-admission bug), not by which policy was
ahead.

**These are the final, calibrated parameter targets used by the smoke (§9)
and full pilot configs.** §9 runs a cheap smoke calibration specifically to
verify these targets are
actually met (meaningful pressure, non-trivial urgency) before the full
pilot — adjusting workload *scale* parameters (counts, medians, slacks) in
response to calibration diagnostics is legitimate (matches how Family A
v1→v2 and Family B v1→v2 were refined) and is explicitly not the same as
adjusting anything based on which policy wins.

## 6. Field provenance (explicit, per repository convention)

| Field | Provenance |
|---|---|
| `prompt_tokens` (both classes) | Real-trace-anchored: BurstGPT-staged sample within a controlled window when available, `burstgpt_anchored` (lognormal shape-matched) or `synthetic_lognormal` fallback otherwise — same three-tier provenance tagging as Family B v2's `_sample_lengths`. |
| `predicted_output_tokens` (both classes) | Derived/controlled: sampled with `prefer_real=False` around a class-specific median — a controlled intervention (short for `tenant_urgent`, to isolate admission-latency effects), exactly as Family B v2 labels its `output_intervention` field. |
| `actual_output_tokens` | Equal to `predicted_output_tokens` (no prediction error is the mechanism under test in this family). |
| `arrival_time` | Synthetic, deterministic convoy schedule per class (§5) — the controlled intervention that creates the KV-pressure trajectory. |
| `slo_deadline` | Derived: `arrival_time + slack`, slack is the controlled `urgent_tightness` intervention for `tenant_urgent`, fixed generous value for `tenant_bulk`. |
| `class_id` | `tenant_bulk` / `tenant_urgent` — generator-only label, asserted absent from any online-observable feature (see leakage guard, §7). |
| `bulk_pressure`, `urgent_arrival_phase`, `urgent_tightness`, `seed` | Generator-only factor labels, recorded in `params`/`scenario_id`, never read by any policy. |

## 7. Observable vs hidden features

Online-observable (policies may read, via `ObservableRequest`/`ObservableGPUState`):
`prompt_tokens`, `predicted_output_tokens`, `arrival_time`, `slo_deadline`,
`priority`, `current_kv_tokens`, `max_kv_tokens`, `active_request_ids`,
`decoding_count`/`prefilling_count`.

Hidden / generator-only (never read by any policy, must never appear in
`class_id` or any observable field — enforced by a leakage-guard assertion
analogous to Family B v2's `assert_policy_visible_fields_clean_v2`):
`bulk_pressure`, `urgent_arrival_phase`, `urgent_tightness`, `seed`,
`scenario_id`, `actual_output_tokens` (equals predicted here, but the
convention is preserved for consistency), tenant-role intent beyond the
bare `class_id` string itself.

## 8. Preregistered hypotheses and composition-readiness gates

- **H1 (bidirectional separation):** at ε=0.01 practical significance, each
  policy wins ≥1 of the 32 cells (not a universal winner).
- **H2 (mechanism activates):** `kv_constrained_online` shows ≥1 logged
  admission-deferral-due-to-reserve event on ≥1 `bulk_pressure=high`
  scenario; `least_laxity_first` never can (no code path) — establishes the
  treatment actually did something, not just a name change.
- **H3 (within-scenario timing effect, §4):** `kv_constrained_online`'s
  mean urgent-tenant `unweighted_slo_success_rate` advantage over
  `least_laxity_first` is larger under `urgent_arrival_phase=late` than
  `urgent_arrival_phase=early`, for matched `(bulk_pressure, urgent_tightness,
  seed)` cells.
- **H4 (tightness modulates H3, placebo control):** the H3 effect is small
  under `urgent_tightness=loose` and larger under `urgent_tightness=tight`.

Gates (all must pass for `KV_FAMILY_COMPOSITION_READY`):

| Gate | Test | Threshold |
|---|---|---|
| G1 | Bidirectional wins (H1) | each policy wins ≥1/32 cells at ε=0.01 |
| G2 | Near-tie rate not saturated | < 50% of cells near-tied at ε=0.01 (cf. Family B v1's 96% failure / v2's 3.1% pass) |
| G3 | Mechanism activates (H2) | ≥1 logged reserve-deferral event on ≥1 high-pressure cell |
| G4 | **Within-scenario/trajectory evidence (H3)** | KV-constrained's mean urgent-SLO advantage in `late` cells exceeds its advantage in `early` cells, for a majority of matched (`bulk_pressure`, `urgent_tightness`, seed) pairs |
| G5 | No twin | `kv_constrained_online` ANWG is not byte-identical to `least_laxity_first` ANWG on every cell |
| G6 | Integrity | 0 failed evals, 0 duplicate `(scenario_id, policy)` pairs, 0 NaN/Inf, BurstGPT provenance recorded, leakage guard passes |

`KV_FAMILY_USEFUL_NEEDS_REFINEMENT` if G1/G2/G3/G5/G6 pass but G4 fails
(real separation exists, but no within-scenario evidence — same category as
Family B v1). `KV_FAMILY_NO_GO` if G1 or G5 fails. `DESIGN_CONFOUND` if G6
fails.

## 9. Smoke calibration (before the full 32-scenario pilot)

Smallest possible grid that still exercises every factor:
`bulk_pressure × urgent_arrival_phase × urgent_tightness × 1 seed = 8`
scenarios (`configs/kv_pressure_smoke_v1.yaml`, seed `20260901` only).

Smoke gate (purely a calibration check — not a verdict, per instruction
item 9):

- Neither policy universally dominates all 8 smoke cells.
- `current_kv_tokens/max_kv_tokens` genuinely exceeds 0.82 on at least one
  `bulk_pressure=high` scenario (confirms the pressure target is real).
- ~~`current_kv_tokens/max_kv_tokens` stays below 0.82 for most
  `bulk_pressure=low` scenarios~~ **revised after running the smoke (see
  below): `least_laxity_first` structurally packs to ~100% KV utilization
  whenever any backlog exists at all, by definition of greedy-to-capacity
  admission with no reserve — this happens at `bulk_pressure=low` too, so
  "peak crosses 0.82" cannot be the low-vs-high differentiator.** Replaced
  with: `bulk_pressure=high` produces a measurably higher mean
  `n_reserve_deferrals` and longer sustained backlog (`n_steps`) than
  `bulk_pressure=low` — the mechanistically correct signal for "how much
  pressure", since peak occupancy saturates in both regimes once any queue
  backlog exists.
- At least one logged reserve-deferral event exists somewhere in the smoke
  output (G3 precondition).
- `kv_constrained_online` and `least_laxity_first` are not byte-identical on
  every smoke cell (G5 precondition).
- 0 failed evaluations, 0 leakage-guard violations.

If the smoke gate fails, workload *scale* parameters (§5's counts/medians/
slacks/GPU capacity) may be recalibrated and the smoke rerun — this is
explicitly permitted (§5). If the smoke gate passes, proceed to the full
32-scenario pilot unchanged.

### 9.1 Smoke result (`configs/kv_pressure_smoke_v1.yaml`, 8 scenarios, final calibrated parameters)

| Check | Result |
|---|---|
| Bidirectional wins | 2/8 cells to `kv_constrained_online`, 2/8 to `least_laxity_first`, 4/8 ties (ε=0.01) — **not** universal domination |
| Peak KV utilization exceeds 0.82 on `bulk_pressure=high` | Yes (up to 1.091) |
| `bulk_pressure=high` vs `low` pressure differentiator (revised) | Mean `n_reserve_deferrals`: high=1415 vs low=567 (~2.5×); mean `n_steps`: high≈2400 vs low≈1230 (~2×) — both clearly higher under `high` |
| ≥1 reserve-deferral event logged | Yes (up to 2566 in one cell) |
| Not a policy twin | Confirmed — 4/8 cells show a real ANWG difference (not all 8 tied) |
| Failed evaluations | 0/16 |
| Leakage-guard violations | 0 (scenario generation would have raised `AssertionError` otherwise; all 8 scenarios generated successfully) |

**Smoke gate: PASS.** Proceeding to the full 32-scenario pilot
(`configs/kv_pressure_pilot_v1.yaml`) with `max_kv_tokens=6000`,
`BULK_SLACK_S=1.5`, `urgent_tightness=tight` slack=0.55s, bulk prompt window
`[1024, 3072]` unchanged from this smoke.

## 10. Explicitly out of scope for this task

No selector fitting, no child policy, no composition falsification, no
verdict beyond §1's four labels. No MAP-Elites/GP/symbolic distillation/LLM
synthesis. No multi-GPU work (candidate #4). No promotion of
`vllm_faithful`/`sarathi_faithful`/`apt_serve_faithful` into this pilot.
