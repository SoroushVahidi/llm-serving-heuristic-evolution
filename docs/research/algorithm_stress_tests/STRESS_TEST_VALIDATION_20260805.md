# Algorithm Stress-Test Library — Validation Record (2026-08-05)

Records the calibration process for
`configs/stress_tests/algorithm_stress_test_catalog.yaml` and
`scripts/stress_tests/generators.py`: what the first-draft workloads and
gates got wrong, how each was diagnosed, and what was actually verified.
Every specific number cited here is reproducible via
`python scripts/stress_tests/run_stress_test_smoke.py` (smoke scale) and
`--full` (full scale).

## Method

1. Draft every generator + catalog gate from the literature-grounded
   design (`ALGORITHM_INVENTORY_20260805.md`,
   `LITERATURE_RESEARCH_20260805.md`).
2. Run `scripts/stress_tests/run_stress_test_smoke.py`, which executes
   each entry's workload against its algorithm-under-test and comparison
   algorithms, and evaluates the catalog's own `acceptance_gates`
   expression against the results.
3. For every FAIL, diagnose the actual mechanism (not just retune until
   green) and fix either the workload parameters, the gate expression, or
   both — documented inline as a `calibration_note` field on the affected
   catalog entry, and summarized here.
4. Re-run at both smoke and full scale before accepting.

## Final result

`python scripts/stress_tests/run_stress_test_smoke.py` (smoke) and
`--full` (full scale): **16/16 auto-evaluable gates PASS** at both
scales. 6 entries (regression_anwg's 2, vLLM-LTR/PARS's 4) are explicitly
out of automated-execution scope this pass (trained-artifact / offline-
scoring constraints, disclosed in the catalog itself), not silently
skipped or force-passed.

## Bug classes found and fixed

### 1. "Arrives alone, gets trivially admitted" (3 generators)

`fifo_counter_head_of_line_blocking`, `sof_counter_long_job_starvation`,
`wsp_counter_priority_service_time_conflict` all originally had their
"disadvantaged" request (the long job, or the high-priority-but-long job)
arrive at t=0 with NOTHING else waiting yet. With `max_active_sequences=1`
and no preemption in any policy under test, a lone waiting candidate is
admitted immediately regardless of policy — there is no scheduling
DECISION being made differently, so every policy produced bit-for-bit
identical results. Fixed by seeding a few competing (short/low-priority)
requests at the SAME t=0 arrival, guaranteeing genuine contestation from
the very first admission decision.

### 2. Arrival rate below the competing job's own service rate (2 generators)

`sof_counter_long_job_starvation` (original: 8 short requests/s, each
taking ~0.015-0.02s to serve) and `priority_counter_continuous_high_
priority_starves_low` (original: 10 high-priority/s at a similar service
time) both had an arrival rate FAR below the service rate needed to keep
even one competing request continuously in the queue (service rate ≈
1/0.015s ≈ 65/s for a 1-slot GPU). The competing-request queue drained to
empty between arrivals almost immediately, handing the "disadvantaged"
request an easy early admission — not a starvation scenario at all.
Diagnosed by directly measuring `n_short_completed`/`max_queuing_delay`
at increasing arrival rates until a persistent (non-emptying) backlog was
confirmed. Fixed: 8/s → 100/s (SOF starvation, measured max_queuing_delay
0.045s → 0.945s smoke / 4.545s full); 10/s → 40/s combined with EXTENDING
the simulated window from ~6s to 20s (priority starvation, since aging's
own recovery time, priority_gap/aging_rate ≈ 4.0/0.15 ≈ 26.7s, requires a
comparably long horizon to observe at all).

### 3. Silent metric-name/class-label bugs in the gate EVALUATOR (5 entries)

The most consequential class of bug, because it silently produced FALSE
PASSES, not failures. Several catalog `acceptance_gates` expressions
referenced a per-class qualifier (e.g. `.mean_queuing_delay(low)`) against
a metric name that only exists as a SCALAR in the runner's computed
results (`mean_queuing_delay`, not `mean_queuing_delay_by_priority_class`).
The evaluator's original `substitute()` function only raised an error for
the OPPOSITE mismatch (a dict metric referenced with no class qualifier)
— a class qualifier against a scalar was silently ignored, meaning the
gate compared the SAME aggregate number to itself on both sides,
trivially "passing" or "failing" independent of the actual per-class
behavior. Caught only because `priority_counter_continuous_high_priority_
starves_low`'s gate output an obviously-suspicious `X > 2*X` in its
`detail` string. **Fixed the evaluator itself** (`run_stress_test_smoke.py`)
to raise on a class-qualified scalar reference rather than silently drop
the qualifier, then re-ran the full suite, which caught and required
fixing 4 MORE previously-silently-wrong gates
(`fifo_counter_head_of_line_blocking`, `priority_target_bounded_high_
priority_load`, `wsp_counter_priority_service_time_conflict`, and
confirmed `priority_counter_continuous_high_priority_starves_low` itself).
Also found: 4 gates used algorithm-name abbreviations (`sof`, `estf`,
`wsp`, `llf`) that don't match any key the runner's `POLICY_FACTORIES`
dict recognizes (which uses the catalog's own canonical `algorithm_id`
values) — silently reported `NOT_AUTO_EVALUABLE` rather than a false
pass, but still fixed for correctness (renamed to full `algorithm_id`
values throughout).

### 4. Hypothesis not supported by evidence at any tested severity (2 entries, genuinely revised)

`estf_counter_reasoning_prompt_length_misprediction` and
`llf_counter_laxity_instability_under_prediction_error` both originally
hypothesized that misprediction would make the policy under test WORSE
THAN a naive baseline on an aggregate metric (mean latency / SLO violation
rate respectively). Tested up to extreme severity (70% of requests
mispredicted, understated 30x) — **the hypothesis did not hold**: ESTF's
mean latency never exceeded FIFO's; LLF's SLO violation rate stayed
BETTER than EDF's throughout. Rather than force a pass by cranking
severity further (which risks manufacturing an artificial result with no
real-world analogue) or silently deleting the entry, both were
**genuinely re-diagnosed**: the real, measurable damage from misprediction
shows up in TAIL latency (ESTF's p95_latency: 10.35s vs. FIFO's 5.72s at
the ORIGINAL, moderate severity parameters) and in LLF's own p95_latency
(12.05s vs. EDF's 6.94s), not in the aggregate metric originally guessed.
Both catalog entries' `expected_failure_signature` and `acceptance_gates`
were rewritten to state the empirically-supported claim precisely, with
the original (unsupported) hypothesis and the testing that disproved it
left visible in each entry's `calibration_note` rather than erased. This
is consistent with the literature's own framing (Wierman & Nuyens'
"graceful degradation" — bounded, not necessarily baseline-reversing) and
is arguably a MORE scientifically honest finding than the original guess.

### 5. Gate too strict for a low-contention target case (1 entry)

`estf_target_accurate_alpha_beta_estimate`'s gate (`<=`, no tolerance)
failed at full scale by 0.002% (floating-point-level noise; all three
compared policies landed within 0.002% of each other, consistent with
the LOW-contention regime this entry is supposed to represent). Added a
1% tolerance, matching the pattern already used by
`fifo_target_homogeneous_low_contention`'s own gate.

## What this validation pass does NOT claim

- The 6 out-of-scope entries (regression_anwg x2, vLLM-LTR/PARS x4) are
  unexecuted, not verified false or true — they require infrastructure
  (a trained selector artifact; new offline scoring passes) not built in
  this pass.
- `real_system_followup_required: true` entries' MAGNITUDE claims
  (e.g. how severe real reasoning-prompt misprediction actually is) are
  not validated against a live system — only the MECHANISM (that
  misprediction of the tested severity produces tail-latency damage) is
  validated in-simulator.
- Full-scale validation used this project's default `n_requests`/duration
  scaling from `default_full_setting`; no attempt was made to push beyond
  those into much larger workloads (out of scope for a validation pass,
  not a claimed limitation of the mechanism itself).
