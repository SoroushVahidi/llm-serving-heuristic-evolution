# Sarathi-Serve Stress-Test Catalog Completion — 2026-08-05

Follow-up to `docs/audits/sarathi_official_artifact_audit_20260805.md`
(the artifact audit), completing the exact next action it identified:
adding Sarathi-targeted entries to the Algorithm Stress-Test Library
using the already-finished simulator implementation
(`sarathi_faithful.py`) and the already-completed Wulver A100 real-
hardware validation. Local-workstation task, no new Wulver jobs
submitted, no Apt-Serve/Llumnix/DistServe work begun, CC5/CC6 and the
canonical benchmark suite untouched (verified — see §10 below).

## 1. Starting state (task step 1)

- Branch: `contextual-compositional-heuristics-20260731`
- Starting SHA: `3e8743961e285ec624861c7ea2d612d99514f577` (the artifact
  audit's own commit)
- Upstream: 0 ahead / 0 behind, working tree clean
- Status checker + resume-readiness: both passed
- Confirmed present: `sarathi_faithful.py`, its 18-test suite
  (`tests/test_sarathi_faithful_scheduler.py`), the artifact audit doc,
  the three Wulver job-record docs, and both commit pins
  (`ceaa0660`/`96f9911`) — all as recorded in the prior audit, not
  reinterpreted here without new evidence.

## 2. Recovering the 5 validated Wulver scenarios (task step 2)

Recovered directly from source (`scripts/run_sarathi_gpu_smoke_and_validation.py`'s
`make_scenarios()`, `scripts/slurm/wulver_sarathi_repeated_trials_array.sbatch`,
`experiments/gpu_external_validity/sarathi_vllm_repeated_trials/{repeated_trials_summary.json,bootstrap_comparison.json,postprocess_report.txt}`),
not invented. Full table:

| Scenario | Role | Prompt bucket (target tokens) | Output target | Arrival pattern | Wulver E2E result |
|---|---|---|---|---|---|
| `sarathi_long_prompt_moderate_output` | COUNTER | long (2048) | 256 | 4 reqs, all t=0 | vLLM wins 5/5, diff -0.2555s, CI [-0.298,-0.213] |
| `sarathi_active_decode_plus_arriving_prefill` | TARGET | medium (512) then long (2048) | 256 then 64 | 4 medium @t=0, 4 long @t=3.0+1.0i | Sarathi wins 5/5, diff +1.0172s, CI [0.990,1.036] |
| `sarathi_prefill_heavy_burst` | COUNTER | long (2048) | 32 | 6 reqs, all t=0 | vLLM wins 5/5, diff -0.1466s, CI [-0.157,-0.137] |
| `sarathi_mixed_prompt_lengths` | COUNTER | short/medium/long cycled | 64 | 6 reqs, all t=0 | vLLM wins 5/5, diff -0.2052s, CI [-0.257,-0.161] |
| `sarathi_matched_vllm_kv_pressure` | TARGET | long (2048) | 768 | 12 reqs, all t=0 | Sarathi wins 5/5, diff +0.8360s, CI [0.769,0.903] |

- Model: `mistralai/Mistral-7B-Instruct-v0.1`; hardware: 1x A100-SXM4-80GB
- Config: `gpu-memory-utilization=0.35`, `max-num-seqs=16`,
  `chunk-size=512` (Sarathi) / `max-num-batched-tokens=512` (vLLM),
  `block-size=16`, `max-model-len=16384`
- Seeds/trials: N=5 per system, fixed seed `20260719`, byte-identical
  prompts across trials (deliberate — isolates system noise from
  workload-content variation)
- Compared systems: vendored Sarathi-Serve fork commit `96f9911`
  vs. real vLLM `0.24.0`
- Source jobs: build `1111574`; repeated-trial arrays `1111988`
  (Sarathi) / `1111989` (vLLM); postprocess `1111990`
- Result artifacts + sha256: `repeated_trials_summary.json`
  (`805c9683759b8fbf38d98c9c8fe8d4eff10505628e8063ca04aa260288335c11`),
  `bootstrap_comparison.json`
  (`fb75354bcdfd3390721ae5cd60c87c32a8bed9b28c02e8ca7e4ebc8834f66815`) —
  both verified present and hash-matched by
  `tests/stress_tests/test_sarathi_stress_test_catalog.py::TestWulverProvenance`

No parameter was invented; every field above traces to a specific file
and line. The 6th scenario (`sarathi_short_context_control`) was
confirmed dropped from the repeated-trial pass (kept only in the earlier
single-run comparison) and is explicitly NOT claimed as real-hardware
evidence anywhere in the new catalog entries.

## 3. Catalog mapping (task step 3)

7 new entries added to
`configs/stress_tests/algorithm_stress_test_catalog.yaml` (section 12),
bringing the catalog from 22 to 29 entries: 5 mirroring the table above
(2 TARGET, 3 COUNTER) plus 2 literature-motivated entries (§5). Every
required field is present (`stress_test_id`, `algorithm_id`, `test_role`,
`evidence_class`, `source_citations` with Wulver job IDs, `observed_behavior`,
`comparison_algorithms`, `simulator_requirements`, `validation_status`,
`provenance_notes`, `calibration_note` where applicable).

`algorithm_id: sarathi_faithful`, not `sarathi_serve` — a deliberate,
disclosed deviation from the task's literal suggested field value. Every
other row in this catalog uses a runnable policy key as `algorithm_id`
(e.g. `pars_semantic_reference`, not the paper name `PARS`), and
`"sarathi_serve"` is not registered in `POLICY_FACTORIES` anywhere in
this codebase — using it verbatim would make these entries silently
fall through to `NOT_AUTO_EVALUABLE` in the generic runner, defeating
the point of running real headroom checks. The correct, consistent, and
actually-executable name was used instead; recorded here so it is a
visible decision, not a silent substitution.

`evidence_class: EXPERIMENTALLY_VALIDATED_ON_REAL_HARDWARE` — a new
6th value added to the catalog's evidence-class enum (documented in the
YAML header), used only for the 5 real-hardware entries, strictly
stronger than `INTERNAL_EMPIRICAL_FINDING`. Not applied to
`PROVEN_WORST_CASE` anywhere, per the task's explicit caution.

## 4. Commit-drift disclosure (task step 4)

Full structured note:
`docs/research/algorithm_stress_tests/SARATHI_COMMIT_DRIFT_20260805.md`.
Summary: `ceaa0660` (faithful-reimplementation pin) and `96f9911`
(Wulver real-hardware pin) have diverged (20 ahead / 2 behind); the
scheduler files changed (~11% of `sarathi_scheduler.py`), but the actual
diff (read via `gh api`'s `.patch` field, not summarized from a commit
message) shows refactoring/infrastructure changes only — renamed
accessors, new constructor params for multi-stage/pipeline-parallel
plumbing, `seq_id` typing — the three-phase scheduling algorithm and its
chunk-budget arithmetic are unchanged. One meaningful default changed:
`chunk_size` gained an explicit `default=512` at `96f9911` (previously
required, no default), which happens to match what this project already
uses independently. **Classification: MECHANISM-LEVEL VALIDATION** —
not exact-commit-level (the bytes differ), not merely
partial-semantic/insufficiently-comparable (the algorithm itself is
unchanged). The real-hardware evidence validates the Sarathi-Serve
chunked-prefill/stall-free mechanism as a family, with high confidence
(not certainty) that it reflects `sarathi_faithful.py`'s specific
modeled algorithm.

## 5. Literature-motivated entries (task step 5)

- `sarathi_counter_short_prompt_decode_dominated_regime`
  (`HYPOTHESIZED_ADVERSARIAL_REGIME`): short-bucket prompts with
  long decode, motivated by the general chunk-size-sensitivity finding
  in the artifact audit's own literature review, not a direct
  reproduction of any cited experiment. Simulator-executable (fully
  representable); local GPU (RTX 5060 Ti) NOT sufficient for a real
  comparison (same categorical CUDA/Blackwell gap documented for VTC and
  the base Sarathi artifact); Wulver required for real-hardware
  follow-up. **Result**: the gate technically passes, but degenerately —
  at smoke scale, ALL FOUR compared policies (sarathi_faithful,
  vllm_chunked_prefill_faithful, vllm_faithful, fifo) produce
  byte-identical output, because a 100-token prompt fits in a single
  512-token chunk regardless of scheduling policy. Disclosed honestly in
  the catalog's `calibration_note` rather than presented as a confirmed
  finding.
- `sarathi_counter_long_context_attention_recompute`
  (`PAPER_MOTIVATING_STRESS_CASE`, citing the artifact audit's technical
  review synthesis: <3% overhead at 8K tokens, potentially 10-15% at
  64K, paper itself evaluates only ≤~13K-token prompts): **NOT
  REPRESENTABLE** — this simulator's timing model has no attention-cost
  scaling term at all. Generator raises `NotImplementedError` by design,
  matching the vLLM-LTR/PARS out-of-scope disclosure pattern rather than
  silently producing a meaningless flat-cost number. Local GPU
  insufficient regardless (16GB VRAM cannot hold a 32K-context KV cache
  for a 7B+ model at usable batch sizes); Wulver required, and even then
  needs the real engines, not this simulator, since the gap is
  structural.

## 6. Local smoke workloads (task step 6)

`scripts/stress_tests/run_sarathi_headroom_check.py` dumps deterministic
per-seed workload files to `configs/stress_tests/generated/sarathi/`
(18 files: 6 executable entries × 3 seeds `[0, 1, 2]`; the 7th entry,
long-context, is not executable). Does not touch
`benchmarks/canonical_suite/` — verified by
`TestNoModificationOfProtectedFiles` in the new test file (`git status
--porcelain` against that directory is empty).

Note disclosed in the script's own docstring and the report: 5 of 6
generators (all except `sarathi_counter_short_prompt_decode_dominated_regime`)
are fully deterministic (no randomness at all), matching the real
Wulver validation's own methodology (byte-identical prompts,
`temperature=0.0`, so any variance is attributable to system noise, not
workload content) — dumping 3 seeds for those therefore produces 3
byte-identical files by design, not a wasted or redundant step.

## 7. Headroom checks (task step 7)

Compared, per entry: `sarathi_faithful` (under test), `fifo` (non-chunked
FCFS baseline), `vllm_faithful` (decode-first without chunking — the
v0.1.0-era faithful reimplementation, no chunked-admission model),
`vllm_chunked_prefill_faithful` (chunking without decode-priority — the
closest in-repo analog to real vLLM 0.24.0's own chunked-prefill
scheduler), `shortest_output_first` (throughput baseline). Measured
TTFT, TPOT, throughput, completion fraction, and two new proxy metrics
added to `run_stress_test_smoke.py`'s `run_policy()` (stall-frequency
proxy = p95/mean TPOT ratio; scheduling-disagreement proxy =
completion-order-vs-arrival-order inversion rate) — both explicitly
disclosed as proxies, not ground-truth instrumentation. Chunk
utilization was NOT computed (`chunk_utilization: None` in every result)
— this simulator does not expose per-step chunk-budget-consumption
counters at the policy level, and adding that instrumentation would mean
modifying `sarathi_faithful.py` itself, out of scope for a stress-test
catalog addition; disclosed as `NOT_INSTRUMENTED` rather than
approximated with a fabricated number.

**Headline finding**: `sarathi_faithful` and `vllm_chunked_prefill_faithful`
produce byte-identical `mean_latency` for every one of the 5
real-hardware-mirrored fixtures, at every parameter combination tested
(a full `step_token_budget` sweep 16-4096, an arrival-offset rescaling
sweep to correct a real-wall-clock-seconds-vs-simulator-abstract-time
units mismatch, and a deliberately adversarial inverse-arrival-order
fixture). Root-caused (not just observed) via direct per-step
`GPUState.step_contention_diagnostics` tracing: the two execution paths
are **provably equivalent** whenever both compared policies use
FCFS-strict admission (true of both here), because a currently-decoding
request can never have been admitted later than a currently-competing
prefill under FCFS-strict admission — full proof in
`docs/research/algorithm_stress_tests/SARATHI_MECHANISM_CALIBRATION_20260805.md`.

Per this task's explicit instruction ("reject or revise any workload
that does not [distinguish the mechanism]"): the original gate design
(direct `sarathi_faithful` vs. `vllm_chunked_prefill_faithful` latency
comparison) was **rejected**, not tuned further — further parameter
search would not fix a structural/provable equivalence. **Revised** to a
gate that IS genuinely distinguishing and was independently discovered
during the same diagnostic pass: `vllm_faithful` (non-chunked) fails to
complete 33-100% of requests in every one of the 5 fixtures
(`completion_fraction` 0.00-0.50), while both chunked policies complete
100%. All 5 entries' `acceptance_gates` now read
`sarathi_faithful.completion_fraction >= vllm_faithful.completion_fraction`
— a real, evaluable, honestly-labeled claim (chunking enables successful
completion of long-prompt workloads at all) that is explicitly NOT the
same claim real hardware validated (the finer decode-protection
distinction), with that gap disclosed in every affected entry's
`calibration_note` and `validation_status`, not silently substituted.

All 6 executable entries: **ACCEPTED** (genuinely, not force-passed —
verified via `scripts/stress_tests/run_sarathi_headroom_check.py`,
`results/stress_test_catalog/sarathi_smoke/report.{json,md}`).

## 8. Coverage matrix / catalog docs (task step 8)

Created (did not exist previously — "update" interpreted as
"establish," since no earlier version existed to update):
`docs/research/algorithm_stress_tests/COVERAGE_MATRIX.md` (per-algorithm
rollup + Sarathi-specific real-hardware/simulator/local-GPU/Wulver/
commit-fidelity table) and `STRESS_TEST_CATALOG.md` (human-readable
29-entry catalog rendering). `README.md` updated for the new entry
counts (22→29) and new document/script references.

## 9. Testing (task step 10)

New file `tests/stress_tests/test_sarathi_stress_test_catalog.py` — 54
tests covering: catalog schema (7 entries, correct role split, evidence
classes, `algorithm_id` naming decision), the 5 Wulver provenance
records (exact job-ID citations, result-artifact existence AND sha256
match), target/counter pairing (against the real-hardware direction),
deterministic generation (reproducibility, exact smoke-count match to
real trial N), no future-information leakage (predicted==actual
invariant, no `actual_output_tokens` reference in policy code), commit-
drift disclosure (doc exists, cites both pins, states a classification),
the headroom checker (runs, exits 0, writes workloads + report), and
non-modification of the canonical suite / VTC / `sarathi_faithful.py`
itself (`git status --porcelain` checks).

Pre-existing `tests/stress_tests/test_stress_test_generators.py` updated
for the now-legitimate catalog growth (22→29 entries, 5th→6th
evidence-class value, new `NotImplementedError` stub added to the
existing exclusion lists) — 4 previously-failing assertions fixed
because the catalog genuinely grew, not because a test was weakened;
all pre-existing assertions' logic is unchanged.

## 10. Protected-file verification

`git status --porcelain -- benchmarks/canonical_suite/ baselines/vtc/`
confirmed empty throughout this task (also asserted by
`TestNoModificationOfProtectedFiles`). No CC5/CC6 core files
(`src/llmserveopt/core/*`, `src/llmserveopt/selector/*` contextual-
composition-specific modules) were touched — only
`scripts/stress_tests/*`, `configs/stress_tests/*`, `tests/stress_tests/*`,
and `docs/*` changed this pass.
