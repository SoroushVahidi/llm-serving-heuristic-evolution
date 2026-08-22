# Public Trace Replay V1 Scientific Analysis

Date: 2026-08-20

Analysis-only pass over the completed Public Trace Replay v1 artifacts. No replay was run or
rerun, no scheduler/policy code was modified, no controller was created, no TEST-driven design was
performed, canonical replay outputs were not altered, and nothing was staged, committed, pushed,
stashed, or reset.

---

## 1. Executive verdict

Classification: `PUBLIC_TRACE_NEAR_DEGENERACY`.

The completed public-trace replay is structurally valid, but under the frozen replay configuration
it does **not** expose meaningful natural policy separation. Every faithful-view cell and every
controlled-annotation six-policy cell has:

- `arrival_normalized_weighted_goodput = 1.0`
- `completion_fraction = 1.0`
- `slo_violation_rate = 0.0`

The six-policy envelope mean is `1.0`, the best-fixed mean is `1.0`, and the envelope gain over the
best fixed policy is exactly `0.0` in all `60/60` annotated public windows. Source identity does
not change the preferred policy because all policies tie on the primary utility in all sources.

Trajectory logs explain the degeneracy: the replayed public windows are lightly loaded relative to
the frozen capacity (`max_active_sequences = 512`, `max_kv_tokens = 8,000,000`). Across
`8,236,824` trajectory steps, active count has median `1`, p99 `5`, and max `31` (`6.05%` of
active-sequence capacity). Max KV utilization has median `0.0001875`, p99 `0.001167`, and max
`0.003802`. Queue-positive steps exist but are rare (`0.975%` step-weighted overall), and no cell
ever reaches active-sequence capacity.

This weakens any claim that adaptive scheduling is generally useful on ordinary public traces under
this frozen configuration. It strengthens the diagnostic/falsification contribution by showing why
mechanism-targeted benchmarks such as MF-PSD are necessary: public-average replay can hide scheduler
differences that controlled stress regimes expose.

---

## 2. Integrity

Canonical inputs:

- Layer-2 manifest: `experiments/public_trace_replay_v1/layer2_scenario_manifest.json`
- Layer-3 checkpoint: `experiments/public_trace_replay_v1/layer3_checkpoint.jsonl`
- Layer-3 integrity: `experiments/public_trace_replay_v1/layer3_checkpoint_integrity_report.json`
- Layer-3 provenance: `experiments/public_trace_replay_v1/layer3_provenance.json`
- Layer-4 trajectories: `experiments/public_trace_replay_v1/trajectories/`
- Corpus: `data/public_trace_corpus_v1/`

Structural counts:

| Check | Observed |
|---|---:|
| base windows | 60 |
| scenario-view records | 120 |
| faithful scenarios | 60 |
| controlled-annotation scenarios | 60 |
| expected cells | 480 |
| checkpoint rows | 480 |
| successful cells | 480 |
| failed cells | 0 |
| faithful cells | 120 |
| controlled-annotation cells | 360 |
| trajectory parquet files | 480 |
| integrity `ok` | true |

Layer-3 provenance records the exact replay command as:

`python3 scripts/run_public_trace_replay_v1.py`

with run start `2026-08-20T04:14:30Z`, run end `2026-08-20T04:17:46Z`, seed `20260820`,
prediction-noise sigma `0.30`, slack multiplier `1.0`, window size `200`, and `20` windows/source.

No structural mismatch was found.

---

## 3. Source characterization

The replay sources are BurstGPT, Azure 2023 conversation, and Azure 2023 code. AgentPerfBench is
real-system validation metadata only; it is not a per-request replay source here.

Selected replay windows contain exactly `200` requests each. Source-level workload shape differs:

| Source | windows | mean window duration s | median duration s | prompt mean/window | output mean/window | mean interarrival/window | zero-interarrival fraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| BurstGPT | 20 | 2518.750 | 212.000 | 984.314 | 147.034 | 12.657 | 0.376 |
| Azure 2023 conversation | 20 | 33.871 | 32.046 | 1172.108 | 195.004 | 0.170 | 0.000 |
| Azure 2023 code | 20 | 73.795 | 52.879 | 2047.440 | 28.584 | 0.371 | 0.000 |

Full-source distribution highlights:

| Source | full requests | prompt p50/p90 | output p50/p90 | interarrival mean | interarrival p50/p90/p99 |
|---|---:|---:|---:|---:|---:|
| BurstGPT | 1,404,294 | 262 / 1824 | 36 / 276 | 3.753 | 1.000 / 4.000 / 54.000 |
| Azure 2023 conversation | 19,366 | 1020 / 2734 | 129 / 424 | 0.181 | 0.118 / 0.421 / 0.889 |
| Azure 2023 code | 8,819 | 1469 / 5186 | 13 / 55 | 0.390 | 0.064 / 0.273 / 2.205 |

The sources are heterogeneous in prompt/output shape and arrival cadence, but that heterogeneity
does not translate into primary-utility policy separation under the frozen replay.

---

## 4. Faithful two-policy result

Evidence class: `PUBLIC_TRACE_FAITHFUL`.

Policies:

- `full_prefill`
- `chunked_prefill_small`

Primary utility:

| Metric | full_prefill | chunked_prefill_small |
|---|---:|---:|
| n windows | 60 | 60 |
| mean ANWG | 1.000 | 1.000 |
| median ANWG | 1.000 | 1.000 |
| std | 0.000 | 0.000 |
| min / max | 1.000 / 1.000 | 1.000 / 1.000 |
| completion fraction mean | 1.000 | 1.000 |

Paired difference is reported as `chunked_prefill_small - full_prefill`:

| Quantity | Value |
|---|---:|
| mean paired difference | 0.000 |
| median paired difference | 0.000 |
| bootstrap 95% CI | [0.000, 0.000] |
| chunked wins / ties / losses | 0 / 60 / 0 |
| exact tie fraction | 1.000 |
| near-tie fraction using inherited `0.01` convention | 1.000 |

Per-source paired mean differences are also exactly `0.000` for BurstGPT, Azure conversation, and
Azure code.

Latency differs despite identical utility:

| Difference: chunked - full | Mean |
|---|---:|
| mean latency | +0.018040 |
| p95 latency | +0.025096 |
| mean queueing delay | 0.000000 |
| mean TTFT | +0.018040 |
| SLO violation rate | 0.000000 |

Answer: naturally observed public workloads do not meaningfully distinguish full prefill from
chunked prefill on ANWG/completion under this replay configuration. Chunked prefill changes timing
and action traces, but not primary utility.

---

## 5. Annotated six-policy result

Evidence class: `PUBLIC_TRACE_DERIVED_WITH_CONTROLLED_ANNOTATIONS`.

The six-policy ANWG matrix is fully degenerate:

| Policy | mean ANWG | median | std | min | max | source means |
|---|---:|---:|---:|---:|---:|---|
| `chunked_prefill_small` | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | all sources 1.000 |
| `estimated_service_time_first` | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | all sources 1.000 |
| `full_prefill` | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | all sources 1.000 |
| `kv_constrained_online` | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | all sources 1.000 |
| `least_laxity_first` | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | all sources 1.000 |
| `weighted_fair_share` | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | all sources 1.000 |

Every policy is rank 1 in every annotated window because every policy ties exactly.

Latency/resource secondary means:

| Policy group | mean latency | p95 latency | mean queueing delay | mean TTFT | SLO violation |
|---|---:|---:|---:|---:|---:|
| `chunked_prefill_small` | 0.146382 | 0.348196 | 0.000335 | 0.023841 | 0.000 |
| all five non-chunked policies | 0.128342 | 0.323100 | 0.000335 | 0.005802 | 0.000 |

The controlled annotations are present, but the replayed workloads do not make them binding enough
to affect ANWG.

---

## 6. Best fixed policy

The deterministic `argmax` over policy means returns `chunked_prefill_small`, but this is not a
scientific unique winner: all six policies have identical mean ANWG `1.000`.

Best fixed by source is likewise non-unique:

| Source | deterministic argmax | best-fixed mean |
|---|---|---:|
| BurstGPT | `chunked_prefill_small` | 1.000 |
| Azure 2023 conversation | `chunked_prefill_small` | 1.000 |
| Azure 2023 code | `chunked_prefill_small` | 1.000 |

There is no universal fixed winner in the meaningful sense; there is a universal exact tie.

---

## 7. Oracle / portfolio envelope

For annotated windows:

`E_P(x) = max_h R_h(x)` over the six-policy portfolio.

| Quantity | Value |
|---|---:|
| envelope mean ANWG | 1.000 |
| best-fixed mean ANWG | 1.000 |
| mean envelope gain over best fixed | 0.000 |
| median gain | 0.000 |
| p90 / p95 / max gain | 0.000 / 0.000 / 0.000 |
| positive-gain window fraction | 0/60 = 0.000 |
| gain > inherited `0.01` convention | 0/60 = 0.000 |
| bootstrap 95% CI for mean gain | [0.000, 0.000] |

Source-specific envelope gains are also exactly `0.000` for all three sources.

This is one of the main results: public-trace six-policy oracle headroom is absent under the frozen
replay.

---

## 8. Ties and winners

Annotated six-policy view:

| Quantity | Value |
|---|---:|
| windows | 60 |
| unique-winner windows | 0 |
| unique-winner fraction | 0.000 |
| tie windows | 60 |
| tie fraction | 1.000 |
| tie multiplicity | six-way tie in 60/60 |
| distinct unique winners | 0 |
| fractional winner entropy | 2.585 bits |

The entropy number is high only because exact ties split fractional winner credit equally across all
six policies. It does not indicate real policy preference diversity.

---

## 9. Pairwise separation

All pairwise ANWG separations are exactly zero in the annotated view.

| Pair class | mean abs diff | median abs diff | max abs diff | nonzero-window fraction |
|---|---:|---:|---:|---:|
| every one of 15 policy pairs | 0.000 | 0.000 | 0.000 | 0.000 |

Specific requested pairs:

| Pair | wins / ties / losses | mean abs diff |
|---|---:|---:|
| `estimated_service_time_first` vs `weighted_fair_share` | 0 / 60 / 0 | 0.000 |
| `kv_constrained_online` vs `weighted_fair_share` | 0 / 60 / 0 | 0.000 |
| `chunked_prefill_small` vs `full_prefill` | 0 / 60 / 0 | 0.000 |

ESTF/WFS disagreement does not appear in public-trace primary utility. KV does not differ
materially. Prefill policies differ in timing/action traces but not utility.

---

## 10. Source heterogeneity

Source identity changes workload shape, but not policy ranking, winner identity, or headroom.

| Source | policy means | tie fraction | unique-winner fraction | envelope gain |
|---|---|---:|---:|---:|
| BurstGPT | all six 1.000 | 1.000 | 0.000 | 0.000 |
| Azure 2023 conversation | all six 1.000 | 1.000 | 0.000 | 0.000 |
| Azure 2023 code | all six 1.000 | 1.000 | 0.000 | 0.000 |

No source has a clear policy winner on ANWG. Workload source identity does not change the best
policy under this configuration.

---

## 11. MF-PSD comparison

MF-PSD / Unified Utility Matrix v2:

- `176` scenarios
- six-policy dense matrix
- unique-winner rate `101/176 = 57.4%`
- tie rate `75/176 = 42.6%`
- best fixed: `weighted_fair_share`, mean ANWG `0.782943`
- six-policy envelope mean `0.816567`
- mean envelope gain over best fixed `0.033624`
- positive envelope-gain fraction `56.8%`
- gain > inherited `0.01` convention in `55.1%`
- strongest pair by mean absolute separation:
  `chunked_prefill_small` vs `weighted_fair_share`, mean abs diff `0.233618`

Public Trace Replay v1 annotated view:

- `60` windows
- unique-winner rate `0/60 = 0.0%`
- tie rate `60/60 = 100.0%`
- all six policies mean ANWG `1.0`
- six-policy envelope gain over best fixed `0.0`
- every pairwise ANWG difference exactly `0.0`

Central comparison: MF-PSD is not merely replaying public-average behavior. It intentionally
amplifies controlled, mechanism-targeted stress regimes that are largely absent from these frozen
public trace windows. That does not invalidate MF-PSD; it clarifies its role. MF-PSD is a
policy-separation benchmark, not a claim that ordinary public traces ubiquitously induce scheduler
headroom.

---

## 12. Trajectory and contention analysis

Layer-4 trajectory totals:

- trajectory files: `480`
- step rows: `8,236,824`
- capacity: `max_active_sequences=512`, `max_batch_tokens=512`, `max_kv_tokens=8,000,000`

Overall step-weighted contention:

| Metric | p50 | p90 | p95 | p99 | max | mean |
|---|---:|---:|---:|---:|---:|---:|
| queue length | 0 | 0 | 0 | 0 | 31 | 0.011655 |
| active count | 1 | 3 | 3 | 5 | 31 | 1.544484 |
| active/capacity | 0.001953 | 0.005859 | 0.005859 | 0.009766 | 0.060547 | 0.003017 |
| max KV utilization | 0.000188 | 0.000542 | 0.000748 | 0.001167 | 0.003802 | 0.000269 |

Additional contention indicators:

| Quantity | Value |
|---|---:|
| queue-positive step fraction | 0.009747 |
| active-at-capacity step fraction | 0.000000 |
| active >= 90% capacity step fraction | 0.000000 |
| cells ever queue-positive | 480/480 |
| cells ever active-at-capacity | 0/480 |
| max active observed | 31/512 |
| max KV utilization observed | 0.003802 |

By source:

| Source | queue-positive steps | queue p99/max | active p99/max | max KV p99/max |
|---|---:|---:|---:|---:|
| BurstGPT | 0.005576 | 0 / 31 | 7 / 31 | 0.001336 / 0.003802 |
| Azure 2023 conversation | 0.008494 | 0 / 2 | 5 / 7 | 0.000921 / 0.001915 |
| Azure 2023 code | 0.031292 | 1 / 4 | 5 / 10 | 0.001522 / 0.003777 |

All sources have occasional queue-positive steps, but the system is nowhere near the active or KV
limits that would make most scheduling choices consequential for ANWG.

---

## 13. Degeneracy explanation

The degeneracy appears primarily utility-level, with both action-level and true-action degeneracy
depending on the policy pair:

- Faithful view: `chunked_prefill_small` and `full_prefill` have different action traces in
  `60/60` windows, but identical ANWG/completion in `60/60`.
- Annotated view: `chunked_prefill_small` differs in action trace from each non-chunked policy in
  `60/60` windows, but still has identical ANWG/completion.
- Annotated view: the five non-chunked policies
  (`full_prefill`, `estimated_service_time_first`, `least_laxity_first`, `kv_constrained_online`,
  `weighted_fair_share`) have identical admitted-request traces in `60/60` windows.

Therefore:

- full-vs-chunked degeneracy is mostly "different actions, same final utility";
- ESTF/LLF/KV/WFS/full-prefill degeneracy is mostly "same actions and same utility";
- all requests complete, controlled deadlines do not bind, class variation is inert within each
  single-source window, and KV pressure is tiny.

---

## 14. Annotation sensitivity without new runs

Controlled annotations were not altered or rerun. Reconstructed Layer-2 annotations from the frozen
builder show:

| Annotation | Observed behavior |
|---|---|
| priority | uniform `1.0` |
| class_id | `source_dataset`; constant within each single-source window |
| prediction noise | sigma `0.30`; prediction/actual ratio median `1.0`, p90 `1.479`, p99 `2.0` |
| deadline | `arrival + 2 * service_est`; deadline-service multiple exactly `2.0` |

Prediction noise varies materially, but it does not affect primary utility because the workload is
too lightly loaded. WFS receives no within-window class contrast because class is constant for a
single-source window. LLF and KV receive derived deadlines, but those deadlines do not bind: SLO
violation rate is `0.0` for every cell. Priority is constant by design, so it cannot induce
priority-based policy separation.

---

## 15. Statistical consolidation

All primary paired effects are exactly zero, so bootstrap CIs are also exactly zero:

| Comparison | mean paired diff | bootstrap 95% CI | win/tie/loss |
|---|---:|---:|---:|
| faithful chunked - full | 0.000 | [0.000, 0.000] | 0 / 60 / 0 |
| annotated envelope - best fixed | 0.000 | [0.000, 0.000] | 0 positive / 60 zero |
| ESTF - WFS | 0.000 | not needed; all exact zero | 0 / 60 / 0 |
| KV - WFS | 0.000 | not needed; all exact zero | 0 / 60 / 0 |

The analysis has only `60` base windows, `20` per source, selected deterministically and evenly
spaced. Because the same windows are evaluated under multiple policies, all policy comparisons are
paired by window. No p-value is needed to interpret exact degeneracy.

---

## 16. Central falsification questions

**Q1. Do public traces show meaningful six-policy separation?**
No. All six policies tie exactly on ANWG in all `60/60` annotated windows.

**Q2. Is there meaningful oracle headroom over best fixed?**
No. Envelope gain over best fixed is exactly `0.0` in all windows.

**Q3. Does source identity alter preferred policy?**
No. BurstGPT, Azure conversation, and Azure code all show exact six-way ANWG ties.

**Q4. Are public traces mostly too lightly loaded for scheduling policy to matter?**
Yes under the frozen configuration. Active-count p99 is `5`, max is `31`, capacity is `512`, and
KV utilization is below `0.004` even at maximum.

**Q5. Does MF-PSD represent realistic recurring regimes, rare stress regimes, or amplified
policy-separation regimes?**
This evidence supports the third description: MF-PSD is an intentionally amplified,
mechanism-targeted policy-separation benchmark. These public windows do not show that such regimes
are common under the frozen replay.

**Q6. Does the public-trace result strengthen or weaken the claim that adaptive scheduling is
generally useful?**
It weakens that claim. There is no public-trace oracle headroom for adaptation to exploit here.

**Q7. Does it strengthen the benchmark/falsification contribution?**
Yes. It shows why a diagnostic benchmark is valuable: naive public-average replay can be
uninformative because realistic traces may be underloaded or utility-saturated.

---

## 17. Exact classification

`PUBLIC_TRACE_NEAR_DEGENERACY`

Rationale:

- policies mostly tie: all windows exact ANWG ties;
- oracle gain is negligible: exactly zero;
- contention is weak relative to capacity;
- public workloads rarely expose scheduler differences under the frozen configuration.

---

## 18. Implications for contribution

The public-trace result does not support a new scheduler or broad adaptive-scheduling claim. It
supports a diagnostic/falsification paper framed around:

1. MF-PSD exposes controlled scheduler-separation regimes.
2. Unified matrix quantifies contextual/oracle headroom in those regimes.
3. Adaptive selectors/controllers repeatedly fail to convert offline signal into safe closed-loop
   value.
4. Public trace replay shows that ordinary public windows can be utility-saturated and therefore
   hide scheduler differences.

If the paper claims external relevance, it should be careful: the result supports the need for
stress/diagnostic benchmarks, not ubiquity of public-trace policy separation.

---

## 19. Benchmark / dataset implications

Ready pieces:

- MF-PSD v1 scenarios and provenance.
- Unified Utility Matrix v2.
- Public Trace Corpus v1.
- Public Trace Replay v1 Layer-2/3/4 artifacts.
- Integrity reports and provenance.
- Adaptive-controller and selector `NO_GO` analyses.

Remaining release work:

- package public-trace analysis summaries in a stable derived-analysis artifact if desired;
- document that faithful public evidence covers only two policies;
- document controlled-annotation semantics for the remaining four policies;
- include trajectory-level contention summaries so users know why public replay is degenerate;
- keep raw canonical replay outputs immutable and release derived summaries separately.

The combination is coherent as a reusable diagnostic benchmark/dataset contribution, provided the
limitations and evidence-class split are explicit.

---

## 20. External-baseline implications

Existing literature/baseline audits already identify VTC, vLLM-LTR, PARS, FSP, and T-SRPT as
important comparators for constructive scheduler claims. This public-trace result reduces the value
of immediately running those baselines on the same frozen public windows: if all six current
policies already achieve ANWG/completion `1.0`, additional baselines are likely to tie on the
primary metric unless the replay configuration changes.

Because changing the replay configuration would be a new design task, not this analysis, the
minimum useful external-baseline work now is positioning and comparison against existing audit
evidence, not another underloaded public-trace run.

---

## 21. Limitations

- Only 3 public trace sources are included.
- Only 20 base windows per source are replayed.
- The replay configuration is frozen and may underload the system.
- The replay is simulator-based, not a native production scheduler.
- The faithful view supports only 2 policies.
- The remaining 4 policies require controlled annotations.
- Priority is fixed at `1.0`.
- `class_id = source_dataset`, which is constant within each single-source window.
- Deadlines are derived, not native.
- Prediction noise is synthetic.
- `max_active_sequences=512` and the system capacity may be too high for these windows.
- Public traces may not contain stress regimes.
- No claim is made that these sources represent all LLM workloads.
- No public-trace data was used for training, tuning, or designing a controller.

---

## 22. Artifacts and reproducible commands

Files created by this analysis task:

- `src/llmserveopt/analysis/public_trace_replay_v1_analysis.py`
- `scripts/analyze_public_trace_replay_v1.py`
- `tests/test_public_trace_replay_v1_analysis.py`
- `docs/current/public_trace_replay_v1_analysis_20260820.md`

Commands:

```bash
python -m pytest -q tests/test_public_trace_replay_v1_analysis.py
python scripts/analyze_public_trace_replay_v1.py
```

Test result:

`5 passed`

The second command reads the completed checkpoint, manifests, corpus parquet files, unified matrix,
and trajectory parquets, then prints the deterministic analysis summary to stdout. It does not run
or rerun replay.
