# Canonical Discriminative Benchmark Suite — Design — 2026-08-04

## Motivation

The completed vLLM-LTR comparative evaluation
(`docs/audits/vllm_ltr_first_comparative_evaluation_20260804.md`)
established a clean, independently-verified, but scientifically limiting
result: on real WildChat-1M text, `fifo` and the hindsight-optimal
`oracle_srtf` tie **exactly** (ANWG=0.9957, identical completion/SLO
outcomes every seed). Eight of ten policies tie. vLLM-LTR makes genuinely
different per-request ranking decisions (Spearman agreement with
EST/SOF ≈0.35–0.48, not near 1.0) but cannot demonstrate any benefit from
that distinctness, because the workload gives *no* ordering policy any
room to differ — diagnosed mechanistically in
`docs/audits/ordering_workload_headroom_audit_20260804.md`: generous GPU
capacity relative to prompt sizes makes admission "contended but
non-exclusionary" (49.8% of decisions see ≥2 queued requests, but only
0.54% of those decisions would have admitted a different SET under a
hindsight-SRTF reordering), and SLO slack (2–20s) is loose relative to
typical service time (~0.1–0.5s).

**This means the benchmark, not the scheduler, was the limiting factor.**
This task formalizes and extends that prior audit into a versioned,
reproducible, headroom-validated canonical benchmark suite — the thing
every future baseline/selector/composition comparison in this repo should
be run against, so a null result can be trusted as "this method has no
value here" rather than "this benchmark cannot show value for *anything*
here."

## Benchmark philosophy

1. **A benchmark must be proven discriminative before it is trusted.**
   Every synthetic family here is validated by an inexpensive,
   GPU-free, learned-inference-free headroom check (§4) *before* being
   accepted into the suite — the same principle the prior ordering-
   headroom audit established, now made a reusable, documented gate
   rather than a one-off diagnostic.
2. **Real data is a control, not a competitor.** WildChat-1M is retained
   unmodified as the negative/control workload (per this task's explicit
   constraint) — not because it is a bad benchmark in general, but because
   its own well-characterized null result is itself a useful, reproducible
   floor: any future workload-design regression can be checked against
   "does this still look like the WildChat control."
3. **Rejections are findings, not failures.** Two of the nine synthetic
   families designed for this suite (`overloaded_queue`,
   `kv_budget_pressure`) failed the headroom gate — both for scientifically
   interesting, specific, documented reasons (§5), not silently discarded.
4. **Isolate one phenomenon per family where possible; measure overlap
   when not.** §6 (diversity analysis) checks this directly rather than
   assuming it.
5. **No method-improvement leakage.** This suite is designed and validated
   using only pre-existing, deployable "foundational" policies
   (`fifo`/`edf`/`estimated_service_time_first`/`shortest_output_first`/
   `weighted_shortest_processing`/`scorpio_style_slo_guard`) plus the
   non-deployable `oracle_srtf` hindsight ceiling — no learned selector, no
   contextual composition, no CC5/CC6 code is touched, imported, or run by
   any script in this task.

## Scientific rationale for acceptance thresholds

The headroom gate (reused from the prior audit, `scripts/check_ordering_workload_headroom.py`,
imported not duplicated) requires ALL of:

| Check | Threshold | Justification |
|---|---|---|
| `disagreement_fraction_nonzero` | `> 0` | SRTF and FIFO must choose differently at least once. The WildChat control's own 0.54% (nonzero but tiny) proves this alone is too weak a bar. |
| `anwg_gap_meaningful` | `fifo_srtf_anwg_gap >= 0.01` | 1 percentage point — an order of magnitude above the WildChat control's exact 0.0 gap; small enough not to require an extreme regime, large enough not to be noise. Empirically, this suite's accepted families range 0.017–0.53, well above the bar; the two rejected families are −0.009 and 0.003, both clearly below it. |
| `queue_contention_sufficient` | `queue_contention_fraction >= 0.05` | At least 5% of decisions must see a real multi-request choice. The WildChat control's own 49.8% shows contention alone is easy to obtain — this mainly guards against a workload with almost no simultaneous arrivals. |
| `not_degenerate_tie` | `all_four_policies_bit_identical_anwg == False` on EVERY seed | Direct, cheap detector for the exact WildChat failure pattern this suite exists to avoid reproducing. |

**Acceptance is decided on the MEAN across 3 seeds** (not a single seed),
more robust than the prior audit's single-seed smoke pass — this changed
the outcome for `prediction_noise_regime`, which failed on seed 0 alone
(gap=0.0041) in the earlier single-seed pass but passes on the 3-seed mean
(gap=0.0354) once seed-to-seed noise is averaged out.

**New composite diagnostic (not a gate criterion by itself):**
`headroom_score` — an equal-weighted mean of four independently normalized
signals (`fifo_srtf_anwg_gap` / 0.10 ceiling, `queue_contention_fraction`
directly, `fifo_srtf_decision_disagreement_fraction` / 0.05 ceiling,
`normalized_entropy` — see below), each clipped to [0, 1]. This is
reported for every family (accepted or not) as a continuous ranking
signal, complementing the binary gate. The 0.10/0.05 reference ceilings
are scoring-scale choices (roughly the middle of this suite's own observed
range), not hard requirements — documented as such rather than presented
as literature-derived constants, since no directly comparable prior
threshold exists for this repo's specific ANWG metric.

**New metric: `ordering_entropy_bits`.** Shannon entropy (bits) of the
SRTF-rank of the request FIFO actually admits first, at each contended
(queue_size≥2) decision. Low entropy (peaked near rank 0) means arrival
order already tracks SRTF's preference; high entropy means arrival order
and service-time order are closer to independent. **Empirically shown to
be necessary but not sufficient as a headroom signal on its own**: the
WildChat control's own `normalized_entropy = 0.79` (fairly high — arrival
order and SRTF order ARE substantially shuffled) coexists with an ANWG gap
of exactly 0.0 — the shuffling essentially never crosses a
feasibility/deadline threshold that changes an outcome. This is why
entropy is one component of `headroom_score`, not a gate criterion by
itself. (Caveat: WildChat's entropy was computed over only 17 actual
admission events with queue_size≥2 — few enough that the entropy estimate
itself is noisy; treat as directional, not precise.)

## Generator description

`scripts/generate_canonical_benchmark_suite.py` (new, self-contained;
imports only pre-existing `llmserveopt.workloads.synthetic`/`policies`/
`simulator` modules and the prior audit's `check_ordering_workload_headroom.py`
— no CC5/CC6/contextual-composition code). Nine `WorkloadFamily` dataclass
instances, each specifying: `WorkloadConfig` (arrival process, prompt/
output-length distributions, prediction noise, SLO classes = deadline +
priority distribution), GPU capacity/concurrency overrides, service-model
overrides, plus narrative metadata (`target_phenomenon`, `hypothesis`,
`expected_divergent_policies`, `expected_winner`). Determinism:
`llmserveopt.workloads.synthetic.generate_workload()` is a pure function of
`(WorkloadConfig, seed)` via `numpy.random.default_rng(seed)` — verified
byte-identical dataset output for identical inputs
(`tests/test_canonical_benchmark_suite_generator.py`). No request ever
carries `actual_output_tokens` to any deployable policy at runtime (only
`predicted_output_tokens`, itself independently noise-perturbed from
`actual_output_tokens` — verified by a dedicated leakage-guard test).

Outputs, under `benchmarks/canonical_suite/<family>/`:
`seed_<n>.json` (one row per request, all `Request` fields), `manifest.json`
(workload config, GPU/service-model config, per-seed SHA-256 dataset
hashes, full validation record). Suite-level: `suite_manifest.json`,
`foundational_comparison.json`, `diversity_analysis.json`.

## Validation metrics

Per §4 of the task: `fifo_srtf_anwg_gap`, `fifo_srtf_decision_disagreement_fraction`,
`queue_contention_fraction`, `service_time_cv`, `deadline_slack_cv`,
`ordering_entropy_bits` (+ `normalized_entropy`), `headroom_score`. All
computed per-seed (3 seeds) and averaged for the accept/reject decision;
full per-seed detail retained in each family's `manifest.json`.

## Accepted workloads (7 of 9 synthetic families)

| Family | Target phenomenon | Mean ANWG gap | Headroom score | Expected winner | Actual winner (foundational comparison) |
|---|---|---|---|---|---|
| `staggered_heterogeneous` | head-of-line blocking, scarce concurrency | 0.0665 | 0.560 | oracle_srtf | **scorpio_style_slo_guard** |
| `burst_independent_lengths` | EST-vs-SOF disagreement → outcome difference | 0.5257 | 0.807 | oracle_srtf | **shortest_output_first** |
| `mixed_tight_deadlines` | SRTF-vs-deadline tradeoff | 0.0174 | 0.392 | edf/llf (not in this cheap set) | **estimated_service_time_first** |
| `priority_vs_service_time_conflict` | priority-vs-size conflict | 0.0684 | 0.563 | weighted_shortest_processing | **scorpio_style_slo_guard** |
| `prediction_noise_regime` | predictor-quality robustness | 0.0354 | 0.446 | oracle_srtf | **oracle_srtf** |
| `long_output_tail` | tail-heaviness | 0.2807 | 0.739 | oracle_srtf | **scorpio_style_slo_guard** |
| `burst_arrivals_isolated` | pure arrival burstiness | 0.3814 | 0.835 | oracle_srtf | **scorpio_style_slo_guard** |

**The single most important finding from the foundational-heuristic
comparison** (§7): `scorpio_style_slo_guard` — the WORST policy on the
WildChat control (0.9743 vs. 0.9957 for every other policy, the entire
motivating finding for this task) — is the **BEST** policy in 4 of 7
accepted synthetic families, and #2 in a 5th, sometimes even beating
`oracle_srtf` (negative regret: −0.021 on `burst_arrivals_isolated`,
−0.171 on `long_output_tail`, −0.079 on `priority_vs_service_time_conflict`,
−0.034 on `staggered_heterogeneous`). This is not a contradiction — it is
exactly the phenomenon this whole benchmark-design task exists to expose:
**admission-control-style guarding is actively harmful under the WildChat
control's near-uncontended regime (needlessly dropping requests that would
have been fine) but genuinely valuable under the real contention every
accepted synthetic family provides.** A benchmark suite that only included
WildChat would have permanently hidden this.

`fifo`/`edf` are the worst or tied-worst policy in every single accepted
family (regret 0.017–0.53) — confirming these families give real,
substantial reordering value to capture, unlike the WildChat control where
they tied everything.

## Rejected workloads (2 of 9 synthetic families) — documented, not discarded

- **`overloaded_queue`** (mean ANWG gap **−0.0093**, headroom_score 0.422):
  `oracle_srtf` (`OracleShortestJobFirstPolicy`, a hindsight
  shortest-job-first ceiling) is **not** guaranteed optimal for this
  repo's priority-weighted, deadline-based ANWG objective — it is optimal
  for mean flow time. Under genuine overload, greedily admitting short
  jobs first can starve an early-arriving request that could have met its
  own deadline if served promptly, converting an avoidable violation into
  a real one. `fifo` scored *above* `oracle_srtf` here. This is a real,
  useful, informative finding (documented rather than silently tuned away)
  — future use of `oracle_srtf` as "the" ceiling anywhere in this repo
  should carry this caveat.
- **`kv_budget_pressure`** (mean ANWG gap **0.0033**, headroom_score
  0.347, 99%+ queue contention but ~0% FIFO/SRTF disagreement): with a
  uniformly large prompt distribution and a tight `max_kv_tokens` budget,
  at most 1–2 large requests ever fit concurrently regardless of *which*
  one is tried first — KV exhaustion blocks admission in an
  order-INDEPENDENT way here. This family needs an intentionally
  heterogeneous prompt-size mix (some requests fit the KV budget alone,
  some don't) so admission order actually determines *which combination*
  fits, rather than a uniformly-large distribution where "who goes first"
  doesn't change the outcome. Flagged as a concrete next-iteration
  hypothesis, not abandoned.

## Diversity analysis

**Behavioral similarity** (Pearson correlation of each policy's mean-ANWG
vector across the 7 accepted families + control): `fifo`/`edf` are
perfectly correlated (1.00 — behaviorally indistinguishable across this
entire suite, as expected since neither uses predicted length or priority
information this suite's SLO/priority structure would differentiate on).
`estimated_service_time_first`/`shortest_output_first`/
`weighted_shortest_processing`/`oracle_srtf` are all highly correlated
(0.99–1.00) — expected, since all four are "prefer smaller/faster jobs"
variants differing only in weighting. **`scorpio_style_slo_guard` is the
least correlated with every other policy (0.79–0.91)** — direct
quantitative confirmation that it is the behaviorally distinct policy in
this suite (consistent with its rank-flipping behavior above).

**Workload feature-space diversity** (z-scored Euclidean distance over
`[anwg_gap, disagreement_fraction, contention_fraction, service_time_cv,
deadline_slack_cv, prompt_predicted_output_correlation]`):
**`staggered_heterogeneous` and `priority_vs_service_time_conflict` are by
far the closest pair (distance 0.93 — every other pair is ≥1.68)**,
meaning these two families may be measuring substantially overlapping
phenomena despite their different design intent (head-of-line blocking vs.
priority conflict). This is flagged honestly as a redundancy candidate for
suite trimming (§8), not hidden. `burst_independent_lengths` and
`burst_arrivals_isolated` are moderately close (1.87, both bursty-arrival
families) but clearly more distinct than the pair above.
`headroom_score` spread across all 9 (including rejected) families:
min=0.347, max=0.835, std=0.174 — a reasonably wide, non-degenerate range.

## Foundational-heuristic comparison (benchmark characterization, not a publication experiment)

Full per-workload rankings, win/tie/loss, unique wins, and regret-vs-oracle
are in `benchmarks/canonical_suite/foundational_comparison.json` (summary
table above). Only `fifo`/`edf`/`estimated_service_time_first`/
`shortest_output_first`/`weighted_shortest_processing`/
`scorpio_style_slo_guard` + `oracle_srtf` were run — no learned selector,
no contextual composition, no CC5, 3 seeds, all 8 workloads (7 accepted +
WildChat control).

## Limitations

- Acceptance/rejection and all comparison numbers use 3 seeds — enough to
  average out clear single-seed noise (as `prediction_noise_regime`
  demonstrated) but not a publication-grade multi-seed study.
- `headroom_score`'s 0.10/0.05 reference ceilings are scoring-scale
  choices calibrated to this suite's own observed range, not derived from
  external literature (none directly applicable to this repo's specific
  ANWG metric was found) — documented as such, not presented as more
  authoritative than it is.
- `ordering_entropy_bits` can be estimated from very few samples on
  low-contention workloads (17 for the WildChat control) — noisy at that
  sample size; treat as directional evidence, not a precise estimate.
- `kv_budget_pressure`'s rejection suggests a concrete redesign
  (heterogeneous, not uniform, prompt sizes) that was not attempted in
  this task (would require another generation+validation cycle, out of
  scope for "benchmark design, not an experiment campaign").
- The `staggered_heterogeneous`/`priority_vs_service_time_conflict`
  near-duplication (§6) was detected but not resolved (e.g. by redesigning
  one of them) — flagged for the next iteration.
- This task's foundational comparison is a characterization exercise
  (§7 of the task explicitly calls it that) — it does not include
  statistical significance testing beyond simple win/tie/loss counts, and
  should not be cited as a rigorous multi-seed policy comparison the way
  the vLLM-LTR evaluation's bootstrap-CI analysis is.

## Future extensions

- Redesign `kv_budget_pressure` with heterogeneous prompt sizes (concrete
  hypothesis already stated above) and re-validate.
- Resolve or consolidate the `staggered_heterogeneous`/
  `priority_vs_service_time_conflict` near-duplicate pair.
- Extend the foundational comparison to more seeds once a specific
  baseline/selector needs a publication-grade result on this suite (this
  task's own 3-seed characterization is not that).
- Add EDF/LLF and a real priority-aware policy to the (currently 4-policy)
  headroom checker itself, now that `mixed_tight_deadlines` and
  `priority_vs_service_time_conflict` specifically motivate them.

## Recommended mandatory benchmark suite

- **Quick smoke suite** (CI-speed sanity check, <2 min): `wildchat_control`
  (subset, e.g. 60 requests) + `burst_arrivals_isolated` (highest headroom
  score, 0.835, fastest-generating accepted family) — confirms a change
  doesn't silently collapse ordering headroom.
- **Required workloads** (every future baseline/selector comparison):
  all 7 accepted synthetic families + `wildchat_control` — 8 workloads
  total, ≈75s generation/validation + a few minutes per policy set
  compared (this task's own full run, 9 families × 3 seeds ×
  [headroom+entropy+7-policy comparison], took 5m32s total on this
  workstation).
- **Optional workloads**: a redesigned `kv_budget_pressure` v2 and a
  consolidated `staggered_heterogeneous`/`priority_vs_service_time_conflict`
  replacement, once the Future Extensions above are addressed.
- **Control workload**: `wildchat_control` — mandatory in every
  comparison, always reported alongside the synthetic suite, never
  dropped.
- **Publication suite**: the 8 required workloads above, run at ≥5 seeds
  each with the FULL policy set (including any learned selector/
  composition under test) plus the paired-bootstrap-CI methodology
  established in the vLLM-LTR evaluation
  (`scripts/verify_vllm_ltr_comparison_results.py`'s independent-
  verification pattern) — not run in this task.
- **Large-scale suite**: publication suite × more seeds (10+) and/or
  larger `duration`/request counts per family, for a final pre-registration-
  style result.

**Estimated runtime:**
- **Local workstation** (this machine, 20 vCPU, no GPU needed for
  foundational-only comparisons): quick smoke suite <10s; required suite
  generation+validation+comparison ≈5.5 min (measured, this task); a
  publication-suite run adding a learned selector/vLLM-LTR-style policy
  should budget per the vLLM-LTR recovery's own measured cost
  (`regression_anwg_selector`'s live per-step dispatch was the dominant
  cost there, ~38 min for a 3-seed, 300-request run at 4.16ms/call
  post-fix) — i.e. plan for the SLOWEST candidate policy's per-call cost
  to dominate, not the cheap foundational policies measured here.
- **NJIT Wolverine cluster**: not used or required for this task (no GPU,
  no expensive experiments per the task's own constraints); would only
  become relevant for a large-scale, many-seed, many-policy publication
  run of a genuinely expensive candidate (e.g. a learned selector or the
  vLLM-LTR checkpoint) across the full 8-workload suite — estimate by
  linearly scaling this task's local-workstation numbers by
  (seeds × policies_including_expensive_ones), not attempted here.
