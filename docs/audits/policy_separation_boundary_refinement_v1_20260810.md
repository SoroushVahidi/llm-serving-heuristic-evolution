# Policy Separation Boundary Refinement v1 -- Scientific Audit

## Provenance

- Slurm job: `1171116`
- Run directory: `/mmfs1/scratch/ikoutis/sv96/policy_separation_boundary_refinement_20260810T134748Z_1171116/`
- Repo checkout: `/mmfs1/project/ikoutis/sv96/github/llm-serving-heuristic-evolution-policy-separation-v1`
- Git branch: `policy-separation-v1-wulver-20260809`, HEAD/experiment SHA `a8c2dd564680c3b6d763ad0966e1c56dac7aa880` (verified identical to `origin/contextual-compositional-heuristics-20260731` both at launch and at time of this audit -- no code drift)
- Script/config: `scripts/run_policy_separation_boundary_refinement.py`, `configs/policy_separation_boundary_refinement_v1.yaml`
- Predecessor: job `1170116`, audited in `docs/audits/policy_separation_three_case_v1_20260810.md`; mechanism-audit prerequisite for Study C in `docs/audits/policy_separation_edf_admission_mechanism_20260810.md`
- Wall time: 5:01, 8 CPU workers, no GPU
- Scope: three targeted studies refining the three-case diagnostic's generators around the boundaries it found. NOT the full 5-family/25-template corpus, NOT MAP-Elites/Sobol/selector training.

## Integrity

- 3,300 scenarios, 15,660 (scenario, policy) evaluations, `n_completed=15660`, `n_failed=0`, no `failures.jsonl`
- Independently re-verified (not just trusting the run's own aggregation): 0 duplicate `(scenario_id, policy_name)` keys, 0 duplicate `scenario_id`, 0 NaN/Inf ANWG, ANWG bounded in [0,1]
- Per-study scenario/eval counts match the config's own scale-check comment exactly: Study A 840 scenarios / 3,360 evals, Study B 1,980 / 9,900, Study C 480 / 2,400
- Stress/control pairing verified complete (every `pair_id` in Study A and C has both a `stress` and a `control` scenario; Study B has exactly 1,800 stress + 180 control rows, matching its 1-control-per-11-inversion-levels design)
- Leakage check: Study B's recorded `rank_agreement_kendall_tau`/`rank_agreement_spearman` sweep smoothly and exactly from +1.0 (`inversion_fraction=0.0`) to -1.0 (`inversion_fraction=1.0`), confirming the generator's rank-inversion construction is correct and that no policy-visible field silently encodes ground truth
- **Classification: STRUCTURALLY_VALID**

## Study A -- FCFS convoy boundary

- At `max_active_sequences=1`, offset=0.0 gives mean(ESTF-FIFO)=+0.348 (n=60 across all (ratio, n_short, seed) cells, sign_consistency=100%). Every positive offset tested (1e-4 through 1e-2) gives **exactly** 0.0 (std=0.0, 60/60 ties) -- a deterministic, bit-identical collapse, not a statistical near-zero.
- This is mathematically explained by the simulator's discrete step_size=0.001s event ordering: a request is only enqueued once `current_time >= arrival_time`, and with a single admission slot already claimed by the long job at t=0, any positive offset (even far below one step) still isn't visible until after that slot is taken, leaving no policy any admission choice to make. TTFT corroborates: ESTF/WSP/SOF's mean TTFT is 0.363s at offset=0 vs FIFO's 1.936s, and all four converge to ~1.92s the instant offset>0.
- At `max_active_sequences=4` (representative cell ratio=32, n_short=32), separation **persists** across the entire offset range with CIs excluding zero at every point (e.g. offset=0.01: mean=0.079, 95% CI=[0.052, 0.106]) and 100% sign consistency (10/10 seeds) throughout -- this is the general FCFS convoy mechanism, distinguishable from the mas=1 single-slot arrival-order artifact.
- Heterogeneity ratio (32x vs 128x) has **zero** measurable effect on the ANWG gap at offset=0 (identical mean 0.348371 and std 0.183294 to 6 decimal places) -- uninformative for this metric in this range.
- Short-job count is a real, monotonic effect: 16->0.588, 32->0.303, 64->0.154 (gap x n_short is roughly constant, ~9.6), consistent with a fixed absolute number of starved short jobs diluted across a larger burst.
- Control role (short-burst-first) at offset=0 reduces the gap to +0.014, confirming the stress/control pairing correctly isolates arrival order as the mechanism.

## Study B -- prediction-inversion decision surface (primary)

- `target_utilization=0.30` behaves as a clean, correctly-uninformative control: FIFO wins 319/330 cells, winner entropy is lowest (0.24 bits), and CIs include zero almost everywhere.
- The richest decision-boundary region is `target_utilization` 0.65-0.95, where winner entropy peaks (1.86 bits at util=0.85) and four distinct policies (fifo, estimated_service_time_first, aging_priority, shortest_output_first) each win somewhere in the grid.
- Critical inversion thresholds (linear interpolation of the mean ESTF-FIFO gap across the 11-point inversion grid) exist **only under strong heterogeneity**: util=0.50 -> q*~=0.35, util=0.65 -> q*~=0.36, util=0.85 -> q*~=0.64, util=0.95 -> q*~=0.65. Under moderate heterogeneity, the advantage stays positive across the entire sampled inversion range at every load level tested -- no crossing is found, not because the search failed but because FIFO never becomes preferable within [0,1] at these (moderate-heterogeneity, load) combinations.
- The threshold is not simply monotonic in load: it jumps from ~0.35 to ~0.65 as load rises from 0.50-0.65 to 0.85-0.95, meaning **higher queue pressure makes accurate prediction more valuable** (a larger initial advantage that survives more inversion) under strong heterogeneity specifically.
- Family-level oracle headroom: mean 0.0154 (close to job 1170116's single-anchor estimate of ~0.0177, now resolved into a full surface), 43.4% of scenarios with headroom >0.005 (identically >0.01 too, an artifact of ANWG's coarse ~1/60-job quantization, not a data error). Headroom peaks at util=0.85 (mean 0.0248, 56.4% of scenarios >0.005) -- the single most exploitable region found in either experiment. Unique winners = 4, near-tie rate = 66.8%.

## Study C -- EDF / SCORPIO / admission_control mechanism

- Prerequisite code audit (`docs/audits/policy_separation_edf_admission_mechanism_20260810.md`): `edf` never rejects; `admission_control`'s laxity filter is inert at its default `laxity_threshold=inf`; `scorpio_style_slo_guard` has three live mechanisms (finite-threshold laxity/TTFT filter, guard-active exclusion + credit-budget throttle, decode-penalized composite score).
- SCORPIO beats EDF in 230/240 stress cells overall (95.8%), 199/200 (99.5%) once `fraction_impossible>0`, and margin is **exactly** 0.0 in all 240 control (loosened-deadline) cells -- a perfectly clean control confirming the guard logic never fires without genuine pressure.
- Crossover (fraction_impossible=0.0): overload_factor ~= 0.90. Once any impossible jobs are present, SCORPIO already wins at the lowest tested overload (0.9). Notably, even at fraction_impossible=0.0, SCORPIO wins substantially once overload_factor exceeds 1.0 (mean margin 0.15 -> 0.37 as overload rises 1.0 -> 1.4) -- the mechanism is not specific to the deliberately-constructed "impossible" job class; generic overload alone creates emergent unsalvageable jobs too.
- Causal signal (not full ablation): SCORPIO's `completion_fraction` margin is strongly *negative* vs EDF (down to -0.80 at fraction_impossible=0.8) while its `slo_violation_rate` margin is also strongly negative (down to -0.98, i.e. far fewer violations); `num_dropped` averages ~11.9/30 jobs for SCORPIO vs **exactly 0** for every other policy in the roster (fifo, edf, least_laxity_first, admission_control). This directly evidences active admission exclusion (not merely re-ranking) as a real contributor, but **cannot causally separate the filter's contribution from the credit-budget throttle's or the composite score's** without an ablation run, which this experiment did not perform.
- `admission_control` vs `edf`: 94.0% exactly bit-identical (451/480), small residual (mean -0.0008, std 0.011) explained by genuine laxity-order vs deadline-order divergence when predicted service time varies across requests -- **near-equivalent**, not structurally identical.

## Corpus-wide equivalence findings

- **`estimated_service_time_first` / `weighted_shortest_processing`: exactly equivalent in this corpus** -- 2,820/2,820 (100%) bit-identical ANWG across both Study A and Study B. Causally explained: `weighted_shortest_processing_score = predicted_service_proxy(req) / priority`, and every template in both experiments uses the `builders.req()` default `priority=1.0` (never overridden), with matching default alpha/beta -- WSP's sort key collapses to exactly ESTF's. This is parameter-induced, not a bug, and would plausibly diverge in a future corpus with heterogeneous request priority.

## Replication vs job 1170116

| Mechanism | Classification | Notes |
|---|---|---|
| FCFS convoy | REFINED | Diagnostic's pooled +0.42 (ratios [8,32,128]) and refinement's pooled +0.348 (ratios [32,128], different short-count grid) are not directly comparable pools, but both confirm the same clean offset=0 separation; refinement additionally proves the offset>0 collapse is exactly deterministic and isolates the mas=1-specific artifact from the general mechanism (mas=4). |
| Prediction inversion | REPLICATED + REFINED | Diagnostic's single anchor (+0.113 -> -0.010, strong heterogeneity/high load) reproduces almost exactly inside the refined surface (util=0.85, strong: +0.132 -> -0.010); refinement adds the full crossing-threshold surface and the qualification that crossings only occur under strong heterogeneity. |
| EDF overload | REPLICATED | 120/120 -> 199/200 (near-identical win rate), 0-margin controls reconfirmed; refinement adds the crossover-overload location and the overload-alone (no impossible jobs) generalization. |
| admission_control ~= EDF | REPLICATED + QUANTIFIED + CAUSALLY EXPLAINED | 94% exact match measured directly; mechanism audit explains why. |

## Dataset-generator dimension classification

- **Validated primary**: offered load / target_utilization, prediction-rank quality, heterogeneity (categorical gate on whether a crossing exists at all), overload_factor, fraction_impossible, max_active_sequences (categorical), arrival_offset (categorical mechanism switch, NOT a smooth numeric axis)
- **Useful secondary**: short-job burst size (rescales magnitude, does not flip ranking)
- **Control-only**: arrival-order-reversal role, loosened-deadline role, inversion_fraction=0.0 role
- **Currently uninformative** (in the ranges tested): FCFS convoy heterogeneity *ratio* magnitude for ANWG specifically (32x vs 128x show zero difference)
- **Need more mechanism work** (present in the simulator, untouched by either experiment): tenant/fairness dimensions, KV pressure, prefix reuse, prefill/decode ratio

## Sobol / next-stage readiness

Both completed experiments together justify a **modest** Sobol/space-filling pilot restricted to the validated dimensions above, with `arrival_offset` kept as a separate categorical template rather than folded into continuous Sobol coordinates (its discontinuity would waste sampling budget). Not yet justified: MAP-Elites/QD execution (should follow, not precede, a Sobol characterization pass), selector retraining, or module-level causal-ablation interventions (needed to resolve Study C's filter-vs-throttle-vs-ranking attribution gap, but out of scope until the decision-boundary map is denser).

## Selector / module / MAP-Elites limitations (explicit, not to be overclaimed)

- No selector has been trained or evaluated against this corpus; "exploitable headroom exists" (Study B, up to 0.0248 mean at util=0.85) is a necessary but not sufficient condition for a selector to realize it.
- No module-level ablation has been run on SCORPIO's three mechanisms; its ANWG/completion/SLO-violation/num_dropped pattern is consistent with active filtering+throttling contributing, but the relative contribution of each mechanism is unresolved.
- No MAP-Elites or other QD search has been started; Study B's non-flat entropy/headroom-by-load pattern is suggestive of a QD-worthy landscape but has not been exercised with actual QD machinery.
- No real-system (hardware) validation was performed in either experiment; all findings are simulator-internal.
