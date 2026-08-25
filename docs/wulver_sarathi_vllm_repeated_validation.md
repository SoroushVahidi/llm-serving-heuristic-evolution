# Repeated-Trial Sarathi vs vLLM Runtime Validation (Wulver, A100)

Follow-up to `docs/wulver_vllm_kv_pressure_results.md`'s single-run Sarathi
runtime validation (job 1111723 vs vLLM job 1111706). That comparison
showed a real, measured Sarathi E2E advantage on the
`active_decode_plus_arriving_prefill` scenario, but was one run per system
— this document repeats it N=5 times per system, independently, to check
whether that effect (and the rest of the comparison) is stable or was
noise.

## Hardware, model, configuration (unchanged from the validated single run)

- GPU: NVIDIA A100-SXM4-80GB, 1 per trial (Wulver `gpu` partition)
- Model: `mistralai/Mistral-7B-Instruct-v0.1`
- Sarathi-Serve: vendored fork commit `96f9911`, built on a compute node
  (job 1111574), `dtype` left at the fork's own default (float16, per the
  fix validated in job 1111723) — **not forced to bfloat16**
- Sarathi config: `gpu-memory-utilization=0.35`, `max-num-seqs=16`,
  `chunk-size=512`, `block-size=16`, `max-model-len=16384`
- vLLM: 0.24.0, `gpu-memory-utilization=0.35`, `max-num-seqs=16`,
  `max-num-batched-tokens=512`, `block-size=16`, `max-model-len=16384`,
  chunked prefill enabled
- No runtime configuration was changed from the validated single-run
  jobs. The only new parameter is `--trial-index`, which is recorded in
  output metadata only and does not affect request generation.

## Number of repetitions

**N=5 per system** (10 GPU jobs total), the "if cheap" target — all 10
Slurm array tasks (5 Sarathi, 5 vLLM) completed successfully
(`exit 0:0`) well within their bounded walltime, so 5 was affordable, not
just the fallback minimum of 3.

## Scenarios (5 of the original 6; `short_context_control` dropped)

The 5 scientifically prioritized scenarios, unchanged in shape from the
single-run comparison (`docs/wulver_vllm_kv_pressure_results.md`):
`long_prompt_moderate_output`, `active_decode_plus_arriving_prefill`,
`prefill_heavy_burst`, `mixed_prompt_lengths`, `kv_pressure`.

## Prompt/seed methodology (why prompts do NOT vary across trials)

Every trial, for both systems, uses byte-identical prompts: the same fixed
`seed=20260719` and the same per-request `variant_index` scheme already
used in the single-run comparison (`scripts/run_sarathi_gpu_smoke_and_validation.py`'s
`make_scenarios()` / `scripts/run_gpu_external_validity_audit.py`'s
`build_mistral_match_scenarios()` — neither was modified). This is a
deliberate methodological choice, stated explicitly here: with prompt
content held constant and `temperature=0.0` on both systems, any
trial-to-trial variance in the measured metrics is attributable to
system/execution-level noise (GPU scheduling jitter, kernel launch
variance, minor floating-point nondeterminism, network/Slurm scheduling)
rather than workload-content variation. This isolates the question
actually being asked — "if I rerun this exact comparison, do I get the
same qualitative result?" — from a different question (whether the effect
generalizes across different request content), which this experiment does
not address. Each of the 5 trials per system used a genuinely independent
process: a fresh engine build (Sarathi) or a fresh server start (vLLM),
not repeated measurements sharing one warm process/engine instance.

## Execution

Two independent Slurm job arrays (`--array=0-4`), one full A100 per array
task:

- Sarathi: `scripts/slurm/wulver_sarathi_repeated_trials_array.sbatch`,
  array job **1111988** (tasks 1111988_0 .. 1111988_4)
- vLLM: `scripts/slurm/wulver_vllm_repeated_trials_array.sbatch`, array job
  **1111989** (tasks 1111989_0 .. 1111989_4)
- CPU-only postprocessing (`--partition=general`, no GPU),
  `--dependency=afterok:1111988,afterok:1111989`:
  `scripts/slurm/wulver_repeated_trials_postprocess.sbatch`, job
  **1111990** — ran `scripts/analyze_repeated_trials.py`

All 11 jobs (10 GPU array tasks + 1 CPU postprocess) completed with exit
code `0:0`.

## Statistical method

For each scenario, across the 5 matched trial pairs (same `trial_index` on
both systems): mean, median, sample standard deviation, p50, and p95
(reported for completeness; with n=5, p95 is only loosely informative — a
single order statistic, not a robust tail estimate). For the primary
metric of interest (E2E latency), a **paired bootstrap** (10,000
resamples, fixed seed `20260719` for reproducibility, resampling trial
*pairs* to preserve the Sarathi/vLLM matching) of the mean
(vLLM − Sarathi) difference, reporting a 95% percentile CI.

**Robustness classification rule** (stated explicitly in
`scripts/analyze_repeated_trials.py`, not a formal significance test):

- **ROBUST**: Sarathi wins E2E in ≥80% of trials (≥4/5) AND the bootstrap
  95% CI for the mean difference excludes zero in Sarathi's favor.
- **SUGGESTIVE**: Sarathi wins E2E in ≥60% of trials, CI may include zero.
- **NOT_REPRODUCED**: otherwise.

This is deliberately conservative language, not a p-value. With n=5,
formal significance testing is not attempted; the win-count + bootstrap-CI
combination is a simple, pre-stated, falsifiable rule, chosen so the
classification can't be quietly adjusted after seeing the data.

## Results

| Scenario | N | Sarathi E2E wins | vLLM E2E wins | Mean diff (vLLM−Sarathi, s) | 95% CI | Robustness |
|---|---:|---:|---:|---:|---|---|
| long_prompt_moderate_output | 5 | 0 | 5 | −0.2555 | [−0.298, −0.213] | NOT_REPRODUCED (for Sarathi) |
| **active_decode_plus_arriving_prefill** | 5 | **5** | 0 | **+1.0172** | **[0.990, 1.036]** | **ROBUST** |
| prefill_heavy_burst | 5 | 0 | 5 | −0.1466 | [−0.157, −0.137] | NOT_REPRODUCED (for Sarathi) |
| mixed_prompt_lengths | 5 | 0 | 5 | −0.2052 | [−0.257, −0.161] | NOT_REPRODUCED (for Sarathi) |
| **kv_pressure** | 5 | **5** | 0 | **+0.8360** | **[0.769, 0.903]** | **ROBUST** |

Full per-metric (TTFT/TPOT/E2E) descriptive statistics:
`experiments/gpu_external_validity/sarathi_vllm_repeated_trials/repeated_trials_summary.csv`.

**Note on the classification's asymmetry**: the rule above only tests
"is Sarathi's E2E advantage robust" (matching what the task set out to
check). The three `NOT_REPRODUCED` rows are not "no effect" — they are a
**robust vLLM advantage**: vLLM wins E2E 5/5 in all three, with bootstrap
CIs just as tight and just as clearly excluding zero (in vLLM's favor).
Read the table as "which system wins, robustly, per scenario" rather than
"Sarathi wins vs. inconclusive."

### `active_decode_plus_arriving_prefill`: now ROBUST, not just a single-run observation

This is the headline result. The single-run comparison showed vLLM
marginally ahead on TTFT (0.151s vs 0.168s) but Sarathi 2.7x ahead on E2E
(0.612s vs 1.647s). Across 5 independent trials: Sarathi wins E2E in
**all 5**, with means of 0.631s (Sarathi, stdev 0.027s) vs 1.649s (vLLM,
stdev 0.011s) — both systems individually low-variance, and the gap
between them (mean diff 1.017s, 95% CI [0.990s, 1.036s]) is an order of
magnitude larger than either system's own trial-to-trial noise. This is
the clearest, most defensible real-hardware evidence in this whole
investigation for Sarathi's stall-free decode-protection claim.

### `kv_pressure`: ROBUST — a new finding beyond the single run

The single run showed only a modest Sarathi E2E edge (12.852s vs 13.616s,
a 764ms /5.9% difference) — not flagged as a headline finding at the time.
Across 5 trials it is **robust**: Sarathi wins 5/5, mean diff 0.836s in
Sarathi's favor, CI [0.769s, 0.903s], with both systems' individual stdevs
under 70ms. This scenario (long context + long decode, concurrency 12,
matched to `stress_kv_pressure`) is a second, independently robust regime
worth highlighting alongside the decode-protection scenario.

### The other three scenarios: robust vLLM advantage

`long_prompt_moderate_output`, `prefill_heavy_burst`, and
`mixed_prompt_lengths` all show vLLM winning E2E in 5/5 trials with tight
CIs excluding zero. These are not scenarios with a "Sarathi advantage that
didn't reproduce" — they are scenarios with a robust *vLLM* advantage, on
this hardware/model/configuration.

## Implications for simulator calibration

`docs/wulver_vllm_kv_pressure_results.md`'s session-3 finding — that
`vllm_faithful`/`sarathi_faithful` picked the wrong E2E winner specifically
on `active_decode_plus_arriving_prefill` (comparing a single real run
against the simulator) — is strengthened by this repeated-trial result:
that scenario's real-hardware Sarathi advantage is now confirmed ROBUST,
not a single noisy measurement, which means the simulator's mismatch there
is a mismatch against a *stable* real effect, not an artifact of comparing
against one unlucky/lucky run. This raises the priority of understanding
that specific scheduling dynamic (an already-decoding sequence competing
with a newly arriving long prefill for the next scheduling slot) before
trusting the simulator for scenarios shaped like it. The newly discovered
robust `kv_pressure` advantage gives a second, independent target the
simulator should be checked against once `vllm_chunked_prefill_faithful`
(design audit: `docs/vllm_chunked_prefill_faithful_design_audit.md`)
exists.

## Safe claims

- On this hardware/model/configuration, Sarathi's E2E advantage on
  `active_decode_plus_arriving_prefill` is robust across 5 independent
  trials (5/5 win rate, bootstrap 95% CI excludes zero, gap an order of
  magnitude larger than either system's own measurement noise).
- The same holds for `kv_pressure` (5/5, CI excludes zero) — a robust
  Sarathi E2E advantage not identified as such from the single-run data
  alone.
- vLLM has an equally robust E2E advantage on `long_prompt_moderate_output`,
  `prefill_heavy_burst`, and `mixed_prompt_lengths` (5/5 each, CIs exclude
  zero) on this same hardware/model/configuration.
- Trial-to-trial measurement noise for this deterministic
  (`temperature=0.0`, matched prompts) workload is small relative to the
  between-system differences in every one of the 5 scenarios tested —
  each system's own stdev is a small fraction of the gap to the other
  system.

## Unsafe claims

- Do NOT claim these results generalize beyond
  `mistralai/Mistral-7B-Instruct-v0.1` on A100-SXM4-80GB at this exact
  configuration (`gpu-memory-utilization=0.35`, `max-num-seqs=16`,
  matched per-step token budget of 512) — no other model, GPU, or config
  was tested here.
- Do NOT treat the bootstrap 95% CIs as a formal significance test or
  p-value — n=5 is small, and the classification rule is a stated,
  pre-registered-in-code heuristic, not inferential statistics with
  established error-rate guarantees.
- Do NOT claim the underlying scheduling-algorithm *mechanism* has been
  identified for any of the 5 scenarios — this document establishes *that*
  each effect is stable, not *why*, mechanistically, at the
  scheduler-code level.
- Do NOT use this as grounds to resume Selector Dataset v2 generation —
  see `docs/wulver_vllm_kv_pressure_results.md`'s standing
  `SELECTOR_DATASET_V2_DECISION = MORE_RUNTIME_VALIDATION_REQUIRED`,
  unchanged by this document. If anything, a robust, quantified real
  effect the simulator gets wrong is a stronger reason to wait, not a
  weaker one.

## Canonical artifacts

- `experiments/gpu_external_validity/sarathi_vllm_repeated_trials/repeated_trials_summary.csv`
  — full per-scenario/per-metric descriptive statistics
- `experiments/gpu_external_validity/sarathi_vllm_repeated_trials/repeated_trials_summary.json`
  — same, machine-readable
- `experiments/gpu_external_validity/sarathi_vllm_repeated_trials/bootstrap_comparison.json`
  — per-scenario paired bootstrap results and robustness classification
- `experiments/gpu_external_validity/sarathi_vllm_repeated_trials/repeated_trials_report.md`
  — the generated report this document's Results section summarizes
- `experiments/gpu_external_validity/sarathi_vllm_repeated_trials/postprocess_report.txt`
  — job manifest (array job IDs, `sacct` states/exit codes for all 10 GPU
  trials)
- Raw per-trial outputs (server logs, full request-level JSONL, individual
  `scenario_results.json` per trial) are on Wulver scratch storage, not
  committed: `/mmfs1/scratch/ikoutis/sv96/sarathi_repeated_trial_1111988_{0..4}/`
  and `/mmfs1/scratch/ikoutis/sv96/vllm_repeated_trial_1111989_{0..4}/`
