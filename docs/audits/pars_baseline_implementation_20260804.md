# PARS Baseline Implementation — 2026-08-04

> **Naming note:** this project's docs also separately reference an
> unrelated, earlier "PARS" (Zheng et al., NeurIPS 2023 — now called
> **PARS-2023** in `docs/external_baseline_coverage_report.md` to avoid
> collision). Everything in this document is about the different paper
> named in the Summary below, referred to in prose elsewhere in this
> project as **PARS-Serve-2026** (see `docs/BASELINE_STATUS.md`). Code
> identifiers (`baselines/pars/`, `pars_semantic_reference`) are
> unchanged. See `baselines/pars/PROVENANCE.md` for the full
> disambiguation rationale.

## Summary

Integrated the official PARS ("Prompt-Aware Scheduling for Low-Latency LLM
Serving" / "Ranking Before Serving: Low-Latency LLM Serving via Pairwise
Learning-to-Rank") code as an evaluation-only external baseline, following
the same pattern already established for vLLM-LTR
(`docs/audits/vllm_ltr_baseline_audit_20260804.md`), with one material
difference: **no pretrained checkpoint is shipped by the official repo** —
a real checkpoint was trained locally using the official, unmodified
training script.

## Step 1 — Locate and verify

**Official paper:** Tao, Zhang, Dearing, Wang, Fan, Papka, Lan, *"Ranking
Before Serving: Low-Latency LLM Serving via Pairwise Learning-to-Rank,"*
ISC High Performance 2026 (arXiv:2510.03243). v1 (2025-09-25) was titled
*"Prompt-Aware Scheduling for Low-Latency LLM Serving"* — the exact title
this task's request used; retitled in v3 (2026-06-26, ISC camera-ready).
Affiliations: University of Illinois Chicago, Argonne National Laboratory.

**Official repository:** `https://github.com/SPEAR-UIC/PARS` — verified
real and public via `gh api repos/SPEAR-UIC/PARS` (not assumed from a
search snippet: the exact URL was extracted directly from the arXiv PDF's
own embedded hyperlink via a fetch-and-analyze pass, then independently
confirmed to exist, be public, and contain real content via the GitHub
API). Pinned commit `fd4e125b65bb73aef5eccafa79c2509434be61ec` (3 commits
total: initial release 2026-03-22, checkpoint-files README clarification
2026-03-22, citation update 2026-07-24).

**License:** **none.** Verified via the full recursive git tree — no
`LICENSE`/`LICENSE.md`/`COPYING` file anywhere, and the GitHub API's own
`license` field is `null`. This is the central risk of this baseline; see
`baselines/pars/PROVENANCE.md`'s License section for the full explanation
and the explicit, user-directed 2026-08-04 decision to proceed with local,
non-commercial research use while never committing or redistributing the
official source.

**Checkpoint:** none shipped. The repo is training *code* only
(`data_preprocess/`, `predictor_train/`, `predictor_serving/`) — a real
`bert-base-uncased`-based pairwise ranker was trained locally using the
official, unmodified `predictor_train/scripts/train_pairwise_bert.py`.

**Dependencies:** `datasets`, `tqdm` (preprocessing); `torch`,
`transformers`, `tqdm` (training); `fastapi`, `uvicorn`, `torch`,
`transformers` (serving — not used by this evaluation, see Deviations).
All already present in this project's environment.

**Supported models:** `bert-base-uncased` is the only encoder the official
training script defaults to and the only one used here (unmodified
default, not evaluated as a hyperparameter choice).

**Inference pipeline:** official `PairwiseRanker` (BERT encoder + one
`Linear(hidden_size, 1)` head on the pooler/`[CLS]` output) →
`predictor_serving`'s FastAPI `/score` endpoint in the official repo;
this evaluation calls the identical `.score()` method directly in-process
instead (see Deviations).

## Step 2 — Integration strategy

Reused as much official code as possible, exactly per instruction:

- **Training**: the official `data_preprocess/scripts/preprocess_alpaca_gpt4.py`
  and `predictor_train/scripts/train_pairwise_bert.py` were run **unmodified**,
  with **unmodified default hyperparameters** (`bert-base-uncased`,
  `max_length=128`, `batch_size=128`, `num_epochs=3`, `learning_rate=2e-5`,
  `margin=1.0`, `warmup_ratio=0.1`, `weight_decay=0.01`, `seed=42`). No
  flag was overridden.
- **Architecture**: `baselines/pars/adapter/checkpoint_loader.py` never
  redefines the `PairwiseRanker` class — it dynamically imports the exact
  class object from the pinned local clone at runtime
  (`_import_pairwise_ranker_class`), verified by a dedicated test
  (`tests/test_pars_baseline_adapter.py::TestOfficialCodeReuseNotVendored`)
  that fails if this project's own source ever contains a
  `class PairwiseRanker` definition.
- **Where an adapter was unavoidable** (documented, not hidden): (1) no
  FastAPI server — the same `PairwiseRanker.score()` method is called
  in-process for offline batch scoring, exactly mirroring how
  `baselines/vllm_ltr/adapter/offline_scoring.py` already handles
  vLLM-LTR; (2) a simulator policy wrapper
  (`baselines/pars/adapter/simulator_policy.py`), since the official repo
  has no simulator integration at all — its README explicitly describes
  integration as narrative-only ("assign request priorities through the
  platform scheduler ... vLLM's priority scheduler").

## Step 3 — Simulator adapter

`PARSSemanticReferencePolicy` (`baselines/pars/adapter/simulator_policy.py`):
admits requests in PARS-ranked order given a precomputed
`{request_id: score}` map. Does **not** modify `Request` or
`ObservableRequest` (verified:
`tests/test_pars_baseline_adapter.py::TestNoLeakage`), does **not** modify
simulator semantics (uses only the existing `Action`/`select_action`
interface every other policy uses), and is **not** registered as a
selector candidate anywhere (`SELECTOR_ELIGIBLE = False`, verified against
`BASELINE_NAMES`, `SELECTOR_CANDIDATE_NAMES`, `POLICY_LIBRARY_V2_NAMES`,
`EXTERNAL_BASELINE_NAMES` — `tests/test_pars_baseline_adapter.py::TestSelectorScopeInvariants`).

**Ranking direction, derived and verified, not assumed:** the official
`MarginRankingLoss(score_A, score_B, target)` construction, with
`target = +1` when prompt_A's real response is longer, trains the model so
`score_A > score_B` is rewarded exactly when response_A is longer — a
**higher PARS score predicts a LONGER response**. For SJF-style
scheduling, `order_by_pars_score` therefore sorts **ascending**
(shortest-predicted-first) — the mirror image of vLLM-LTR's descending
rule (`tests/test_pars_baseline_adapter.py::TestRankingSemanticEquivalence::test_opposite_direction_from_vllm_ltr`
cross-checks this directly against `order_by_ltr_score` on identical
input). There is no official scheduler-integration source code to
reproduce exactly (unlike vLLM-LTR's pinned `vllm/core/scheduler.py`
excerpt) — this derivation is the best available grounding, and is
recorded as such.

## Step 4 — Fidelity verification

Training completed in tmux session `pars_training`: 3 epochs,
`best_val_accuracy=0.9141` (epoch 2), checkpoint written to
`results/pars_official/predictor_train/alpaca_gpt4_bert/best_model.pt`,
SHA256 `d54be0871ebc9f2c2538b4e53da7f45cb57ae678563488822cdc1694bc33eb27`
(matches the pinned expected hash `d54be087...c33eb27` byte-for-byte).
Full per-epoch history in
`results/pars_official/predictor_train/alpaca_gpt4_bert/metrics.json`.

With the trained checkpoint in place,
`tests/test_pars_checkpoint_fidelity_gpu.py` (`LLMSERVEOPT_RUN_GPU_TESTS=1`,
requires torch/transformers + the pinned official clone + this checkpoint)
passes 10/10:
`TestArchitectureFidelity` (encoder is real `bert-base-uncased`,
hidden_size=768; score head is a single `Linear(768, 1)` unit; max_length
matches the official default of 128), `TestDeterministicScoring`
(repeated calls bit-identical; batched scoring matches singleton scoring),
`TestRealScoringBehavior` (scores are finite; diverse prompts produce
diverse scores — the checkpoint is not collapsed/degenerate), and
`TestLongPromptTruncation` (prompts over max_length do not crash,
truncation never exceeds max_length, long-prompt scoring is deterministic).

Verified independent of the trained checkpoint (architecture/interface
level, `tests/test_pars_baseline_adapter.py`, 22/22 passing):
- Ranking semantics (ascending order, tie-breaking stability, determinism,
  missing-score rejection).
- No-leakage guarantees (`ObservableRequest` has no `actual_output_tokens`
  field; `select_action`'s source contains no reference to it).
- Checkpoint-loader failure modes (missing official clone, stale clone
  commit, missing checkpoint file — all raise typed errors, no silent
  fallback).
- Selector-scope invariants (not in any registry).
- Official-code-reuse invariants (no vendored `PairwiseRanker` definition
  anywhere in this project; the official clone lives outside this repo's
  tree).
- Truncation: the official serving/training code already explicitly
  passes `max_length=128` in every tokenizer call (confirmed by reading
  `serve_predictor_score.py`/`train_pairwise_bert.py` directly) — unlike
  vLLM-LTR's OPT tokenizer, `bert-base-uncased`'s tokenizer does not carry
  the `model_max_length` "unset" sentinel bug class that required a fix
  there; this adapter's `PARSPredictorHandle.score_batch` preserves the
  same explicit-`max_length` call pattern.

## Deviations from the official pipeline (all documented, none in the official training code itself)

1. **No FastAPI server** — offline in-process batch scoring instead (see
   Step 2).
2. **No simulator integration in the official repo** — an adapter was
   unavoidable (see Step 3).
3. **Ranking direction inferred from the training objective**, not copied
   from an official scheduler source (no such source is public — see
   Step 3's ranking-direction note).
4. **Locally-trained checkpoint**, not an official pretrained release (see
   Step 1) — trained on `vicgalle/alpaca-gpt4` (CC BY-NC 4.0), chosen for
   domain realism (general instruction-following text, closest of the 4
   official dataset options to WildChat's general-chat domain) over the
   more permissively licensed but narrow-domain `code`/`math` options —
   see `baselines/pars/PROVENANCE.md` for the full license comparison and
   rationale. `lmsys` was excluded per this project's pre-existing
   restrictive-license policy (unrelated to this task).
5. **Canonical-suite text substitution** (evaluation-script-level, not an
   adapter deviation): the canonical benchmark suite's synthetic workloads
   carry no prompt text (only token counts) — PARS requires real text.
   Each synthetic request is matched to the real WildChat prompt with the
   closest `bert-base-uncased` token count (nearest-neighbor,
   `scripts/score_pars_eval_datasets.py`), and PARS scores that real
   matched text. Necessarily reuses WildChat prompts many times over;
   quantified per-family in each `matching_manifest.json`, never hidden.

## Results

The comparative evaluation (WildChat control + all 7 accepted
canonical-suite families, 3 seeds × 10 policies each, 60,830 total
request-level rows) completed via a mix of the original run (3 families)
and per-family timeout recovery (5 families) — see
`docs/audits/pars_first_comparative_evaluation_20260804.md` for the full
recovery narrative, independent verification, and final classification.

**Summary:** PARS-Serve-2026 never ranks in the top 3 of 10 policies in
any of the 8 evaluated workloads (best rank observed: 5th of 10, in
`mixed_tight_deadlines`; typical rank 7th of 10), records **zero unique
wins** across all 8 families, and is statistically significantly worse
than the best policy in 5 of 8 families (non-overlapping bootstrap CIs).
It is statistically significantly *better* than FIFO/EDF in 3 of 8
families (`burst_independent_lengths`, `long_output_tail`,
`burst_arrivals_isolated`) — all burst/long-tail-heavy regimes — showing
its length-prediction signal is not vacuous, just consistently dominated
by simpler heuristics (`shortest_output_first`,
`estimated_service_time_first`) and by the best adaptive/fixed policies
already in this project's library (`scorpio_style_slo_guard`,
`regression_anwg_selector`). Completion fraction is 1.0 for PARS in every
family (it never drops requests; its shortfall is entirely in
latency-weighted goodput, not completion). On the one workload directly
comparable to vLLM-LTR (WildChat control — the only workload vLLM-LTR's
own comparative evaluation was run on), PARS and vLLM-LTR are
statistically indistinguishable (both 0.9957 ANWG, tied with most fixed
policies) because that control workload is itself non-discriminative
(see `benchmarks/canonical_suite/suite_manifest.json`'s own
`control_workload_headroom` metrics). **Final classification:
EVALUATION_ONLY** — real, verified, working infrastructure and a genuine
(if modest) signal above trivial baselines in burst regimes, but no case
for foundational-library / selector-candidate promotion given zero
unique wins and consistent domination by cheaper existing policies.
