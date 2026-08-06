# Sarathi-Serve Commit-Drift Compatibility Note — 2026-08-05

Structured note requested as part of the Sarathi stress-test catalog
completion task: does the real-hardware Wulver validation
(`docs/wulver_sarathi_vllm_repeated_validation.md`) validate the EXACT
scheduler `sarathi_faithful.py` models, or only the broader Sarathi
mechanism family? Answered with evidence, not asserted.

## The two pins in play

| Pin | Commit | Date | Used for |
|---|---|---|---|
| Paper-era snapshot | `ceaa0660ea2487976101a8167aad5c8046e85b27` (branch `osdi-sarathi-serve`) | 2024-06-04 | `sarathi_faithful.py`'s scheduler-logic source, per `docs/sarathi_faithful_scheduler_reference.md` |
| Real-hardware build | `96f9911790ecc00af12ee9fae47cb8fa9ba0d199` (`main` tip at validation time) | 2026-01-08 | Vendored fork built and run on Wulver A100 (jobs 1111574, 1111988-1111990) |

`git compare ceaa0660...96f9911` (via `gh api`): **diverged**, 20 commits
ahead / 2 behind — not a simple fast-forward from the paper-era pin.

## Files changed in scheduler-relevant paths

Confirmed via `gh api .../compare/...`'s file list, restricted to
scheduling-relevant paths:

- `sarathi/core/scheduler/sarathi_scheduler.py` (+18/−9 of 238 lines, ~11%)
- `sarathi/core/scheduler/base_scheduler.py`
- `sarathi/core/scheduler/vllm_scheduler.py`, `orca_scheduler.py`,
  `faster_transformer_scheduler.py`, `simple_chunking_scheduler.py`
  (sibling schedulers in the same file, not used by `sarathi_faithful.py`)
- `sarathi/core/block_space_manager/base_block_space_manager.py`
- `sarathi/core/datatypes/scheduler_output.py`
- `sarathi/config.py` → **deleted entirely** (546 lines), replaced by a
  package: `sarathi/config/config.py`, `sarathi/config/base_poly_config.py`

## Whether chunked-prefill semantics changed

**No, not at the algorithm level.** The actual diff of
`sarathi_scheduler.py` (read directly via `gh api`'s `.patch` field, not
summarized from a commit message) shows:

- Constructor signature grew two new required parameters
  (`model_config: ModelConfig`, `parallel_config: ParallelConfig`),
  consistent with broader multi-stage/pipeline-parallel infrastructure
  added elsewhere in the repo (MoE support, pipeline-parallel fixes — per
  `docs/sarathi_faithful_scheduler_reference.md`'s own "do not use main
  blindly" warning, written before this specific diff was pulled).
- `seq.get_num_prompt_tokens_processed()` renamed to
  `seq.get_num_prompt_tokens_stage_processed()`, and
  `seq.prompt_processing_finished` renamed to
  `seq.prompt_stage_processing_finished` — a "stage" concept was added
  (almost certainly pipeline-parallel-stage tracking), but the **call
  sites and control flow are unchanged**: the three-phase `_schedule()`
  loop (decode-slot reservation → resume mid-prefill → admit new,
  chunk-budget-bounded) is byte-identical in structure to the pinned
  `ceaa0660` version documented in
  `docs/sarathi_faithful_scheduler_reference.md`.
- `seq_id` typed `List[int]` → `List[str]` (a typing/infra change, not a
  scheduling-logic change).
- `self.prompt_limit = self.scheduler_config.max_model_len` line removed
  (this attribute was never read anywhere in the faithful
  reimplementation's modeled logic — `sarathi_faithful.py` already
  documents prompt-length rejection as explicitly excluded, "silently
  drops requests longer than max_model_len... omitted rather than
  approximated" — so this removal does not affect fidelity either way).

**Conclusion: the core chunked-prefill scheduling algorithm
(`sarathi_faithful.py`'s entire basis) is unchanged between the two
pins.** The diff is refactoring/infrastructure (multi-stage plumbing,
typing), not an algorithmic change.

## Whether defaults changed

**Yes, one meaningful default was added.** At `ceaa0660`,
`SarathiSchedulerConfig.chunk_size` was a required `Optional[int]` field
with **no built-in default** (confirmed directly in
`docs/sarathi_faithful_scheduler_reference.md`: "`SarathiSchedulerConfig`
itself declares `chunk_size` as a required `Optional[int]` with no
built-in default"). At `96f9911` (`sarathi/config/config.py:246-248`,
read directly via `gh api`), `chunk_size` now has an explicit
**`default=512`**. This is convergent, not concerning: 512 is exactly
the value this project's `sarathi_faithful.py` already selected as its
own default, independently, from the paper's own OSDI evaluation scripts
(`table-6/scheduling_ablation.sh`, `figure-9/prefill_chunking_overhead_runs.sh`
— "512 the most frequently used value"). The Wulver validation's own
config (`chunk-size=512`) also matches. No default-value discrepancy
affects any claim made in this project.

## Whether metrics or CLI behavior changed

Not directly audited beyond the scheduler/config files above (out of
scope for this note — the Wulver validation's own harness
(`scripts/run_sarathi_gpu_smoke_and_validation.py`) measures TTFT/E2E/
output-token-count from Python-level wall-clock timestamps around
`LLMEngine.step()` calls, independent of the vendored fork's own metrics
or CLI surface, so this project's measurements are not exposed to any
metrics/CLI drift between the pins even if it exists).

## Classification

**MECHANISM-LEVEL VALIDATION** (one of: exact-commit-level / mechanism-level /
partial-semantic / insufficiently-comparable — selecting mechanism-level,
with evidence, not by default).

Not exact-commit-level: the literal bytes of `sarathi_scheduler.py` at
`ceaa0660` and `96f9911` differ (confirmed diff above), so a claim of
"the Wulver results validate exactly the code `sarathi_faithful.py`
models" would be overclaiming. Not merely partial-semantic or
insufficiently-comparable either: the diff is refactoring/infrastructure
only — no change to the three-phase scheduling algorithm, no change to
the chunk-budget arithmetic (`_get_seq_next_num_prefill_tokens`'s
`min(...)` formula is untouched apart from the renamed accessor), and
the one default-value change (`chunk_size=512`) matches what this
project already uses independently. The real-hardware evidence
therefore validates the **Sarathi-Serve chunked-prefill/stall-free
scheduling mechanism as a family**, with high confidence that
`sarathi_faithful.py`'s specific pinned algorithm is what was exercised
(same control flow, same budget formula, same default), but without the
byte-for-byte identity that would license an "exact reproduction"
claim.

## Practical implication for the 5 catalog entries

None of the 5 real-hardware-mirrored catalog entries' `evidence_class`
(`EXPERIMENTALLY_VALIDATED_ON_REAL_HARDWARE`) is weakened by this
finding — that classification describes what happened on real Wulver
A100 hardware, using the real (albeit `96f9911`-pinned, not
`ceaa0660`-pinned) Sarathi-Serve system, which is unaffected by any
in-repo pin choice. What this note bounds is a *different* claim this
project must NOT make: that `sarathi_faithful.py`'s simulator behavior
should be expected to numerically reproduce the Wulver results — a
claim this project was never making in the first place (the simulator
studiously avoids claiming hardware-timing fidelity;
`sarathi_faithful.py`'s own docstring: "NOT... a full Sarathi-Serve
performance model"), and one further undermined on independent grounds
by `docs/research/algorithm_stress_tests/SARATHI_MECHANISM_CALIBRATION_20260805.md`'s
finding that the simulator's decode-protection mechanism cannot
currently distinguish `sarathi_faithful` from a shared-contention
scheduler for any FCFS-admitted workload, regardless of pin.
