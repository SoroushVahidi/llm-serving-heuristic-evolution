# vLLM-LTR Baseline-Integration Audit

**Date:** 2026-08-04 (scaffold pass + completion pass, same day)
**Branch:** `contextual-compositional-heuristics-20260731`
**Tests:** 25/25 CPU tests pass (`tests/test_vllm_ltr_baseline_adapter.py`); 13/13 real-checkpoint GPU tests pass (`tests/test_vllm_ltr_checkpoint_fidelity_gpu.py`, `LLMSERVEOPT_RUN_GPU_TESTS=1`); full non-live suite unaffected (see §9)
**Scope:** Baseline-completion phase. **CC5/CC6 logic and results are untouched** — see `docs/contextual_composition_roadmap.md` for that track; nothing here touches it.

This document supersedes the same-day scaffold pass: every "untested
assumption" from that pass has now been checked against the real official
checkpoint. Section numbers below are reorganized around the completion
task's own requirements (docs correction, checkpoint acquisition,
architectural fidelity, offline pipeline, semantic equivalence, overhead,
recommendation).

---

## 1. Documentation correction: venue

**Corrected.** "Efficient LLM Scheduling by Learning to Rank" is a
**NeurIPS 2024 main-conference paper**, not merely an arXiv preprint.
Confirmed independently (not just trusting the task's assertion) via:

- `https://proceedings.neurips.cc/paper_files/paper/2024/hash/6c8985579293e0209bdaa4f21bb1d237-Abstract-Conference.html`
- `https://papers.nips.cc/paper_files/paper/2024/hash/6c8985579293e0209bdaa4f21bb1d237-Abstract-Conference.html`
- The official repository's own title has since been updated to
  "**[NeurIPS 2024]** Efficient LLM Scheduling by Learning to Rank" —
  independent confirmation beyond the repo's own README, which cites only
  the arXiv preprint.

Updated in `adapter/provenance.py` (`PAPER_VENUE`,
`PAPER_NEURIPS_PROCEEDINGS_URL`), `PROVENANCE.md`, `docs/baselines.md`, and
`docs/external_baseline_decision.md`. The arXiv id (`2408.15792`) is kept as
a supplementary preprint identifier, not the venue.

---

## 2. Official checkpoint: acquisition and provenance

Downloaded via `huggingface_hub.snapshot_download` (through the new
`adapter.checkpoint_loader.download_and_provision_checkpoint`), **outside
the git repository** (HF cache, `~/.cache/huggingface`) — weight files are
never committed; only hashes, revision, and code are.

| Field | Value |
|---|---|
| HF repo | `LLM-ltr/OPT-Predictors` |
| Repo revision | `39df2b41ffe88d5ed967c6035d3838b5b5960379` |
| Model card | **none** (`README.md` → "Entry not found") |
| License | **not declared** on the HF repo (no tag, no model card, no `LICENSE` file among its 43 files) — see the explicit caveat in `CHECKPOINT_PROVENANCE.md`. The *code* repo is Apache-2.0; that does not automatically cover separately-hosted weights. Treated as unconfirmed, not assumed permissive. |
| Variants downloaded | classification (`opt-125m-llama3-8b-sharegpt-class-trainbucket820-b32`, `num_labels=10`) and regression (`opt-125m-llama3-8b-sharegpt-score-trainbucket10-b32`, `num_labels=1`) — both ShareGPT-trained, 125M backbone |
| `model.safetensors` size | 250,516,416 bytes (~239 MiB) per variant |
| sha256 hashes | recorded in `CHECKPOINT_PROVENANCE.md` (both variants, both files) |
| Weights modified? | No — read-only load throughout; a regression test (`TestCheckpointProvenanceDocFormat`) locks the hash strings' *format* (64 hex chars) after an actual transcription bug (one hex digit truncated on all four hashes) was caught and fixed during this pass — see §8. |

**Full record:** `baselines/vllm_ltr/CHECKPOINT_PROVENANCE.md`.

---

## 3. Architectural fidelity: checkpoint vs. reconstructed HF architecture

**Result: exact match, both variants.**

- `raw safetensors state dict keys == fresh AutoModelForSequenceClassification(same config).state_dict() keys` — set difference empty in both directions (197/197 keys for the classification variant).
- Zero shape mismatches across all 197 keys.
- `config.json` declares `"architectures": ["OPTForSequenceClassification"]` — the checkpoint **is** a stock HF `OPTForSequenceClassification` export, not a vLLM-fork-specific serialization. This confirms the scaffold pass's hypothesis (built from reading pinned source only) against the real artifact.
- Automated, reproducible: `tests/test_vllm_ltr_checkpoint_fidelity_gpu.py::TestArchitectureFidelity`.

### Real deviation found and fixed: tokenizer

**The checkpoint repository ships no tokenizer files** (`tokenizer.json`,
`vocab.json`, `merges.txt`, `tokenizer_config.json`) for any of its 12
variants — only `config.json` + `model.safetensors` + `usage_config.json`.
`AutoTokenizer.from_pretrained(checkpoint_dir)` does **not** raise an error
in this situation: it silently falls back to a generic GPT-2-style
tokenizer (`pad_token_id=None`, `bos/eos/unk="<|endoftext|>"`) that is
**wrong** for this checkpoint — mismatched vocabulary, and scoring would
have proceeded without ever erroring, just silently producing meaningless
scores from mistokenized input.

**Fix applied to `adapter/checkpoint_loader.py`:** read the checkpoint's raw
`config.json` *before* `from_pretrained()` mutates `_name_or_path`, extract
the recorded base pretrained model (`facebook/opt-125m` for both variants
downloaded here), load the tokenizer from there instead, and verify its
`pad_token_id` matches the checkpoint's own `config.pad_token_id` (both are
`1`) — raising `StaleArtifactError` if they don't. This is exactly the
class of silent-wrong-result failure mode a fidelity audit is supposed to
catch; it would not have been found without actually downloading and
loading the real checkpoint.

---

## 4. Offline scoring pipeline

`baselines/vllm_ltr/adapter/offline_scoring.py` (new this pass):

- `score_prompts_offline(handle, id_to_prompt, batch_size)` — scores every
  `(request_id, prompt_text)` pair once, using only prompt text (no
  `actual_output_tokens`, not even accepted as a parameter).
- `save_score_cache` / `load_score_cache` — JSON persistence.
- `scores_only(cache, id_to_prompt=None)` — extracts the plain
  `{request_id: score}` map `VLLMLTRSemanticReferencePolicy` consumes; when
  given `id_to_prompt`, verifies every cached entry's stored sha256 prompt
  hash still matches the current prompt text, raising `StaleScoreCacheError`
  on any mismatch (the "prompt hash" integrity key the task asked for,
  layered on top of request-id keying since `ObservableRequest` itself has
  nowhere to carry a hash).
- **No simulator request objects were modified** — `Request`/`ObservableRequest`
  in `src/llmserveopt/core/types.py` are byte-for-byte unchanged from before
  this task. The pipeline is entirely external; `VLLMLTRSemanticReferencePolicy`
  (unchanged from the scaffold pass) still requires an externally-supplied
  `{request_id: score}` map at construction, with no fallback for a missing
  score (`MissingScoreError`).
- **No future-information leakage:** input is prompt text only; verified by
  the existing structural test (`ObservableRequest` has no
  `actual_output_tokens` field to leak) plus the new
  `TestOfflineScoringPipelineEndToEnd` tests exercising the real checkpoint
  end-to-end (score → cache → reload → rank), including the stale-hash
  rejection path.

---

## 5. Semantic equivalence: quantitative agreement

Two independent forms of "compare against the official implementation"
were performed; a third (live differential against the actually-served
vLLM-fork engine) remains infeasible and is disclosed as a limitation, not
silently skipped.

### 5a. Independent recomputation cross-check (performed, bit-exact)

For both checkpoint variants, two completely separate code paths were run
on the same real ShareGPT text and compared:

- **Path A** (what the adapter uses): `transformers.AutoModelForSequenceClassification.from_pretrained(...)` — HF's own pooling/head logic.
- **Path B** (independent reimplementation): `transformers.AutoModel` backbone-only forward (bypassing HF's classification wrapper entirely) + the checkpoint's real `score.weight` tensor applied *by hand* via matrix multiply, with last-non-pad-token pooling implemented from scratch from the attention mask — replicating the pinned source's `compute_logits` formula (`official_reference/opt_predictor_head_excerpt.md`) from first principles, not by trusting HF's wrapper to do the right thing.

**Result: `(logits_a - logits_b).abs().max() == 0.0` for both variants** — bit-exact agreement in float16, on real text. This is strong evidence that the adapter's understanding of the official score-extraction formula is exactly correct, independent of the convenience wrapper. Reproducible: `TestSemanticEquivalence` in the GPU test file.

### 5b. Ranking/tie-break equivalence (performed, exact — carried over from the scaffold pass)

`ranking_adapter.order_by_ltr_score` was already verified byte-for-byte against the literal official sort formula (`sorted(reqs, key=lambda req: -req.aux_model_score)`, stable) on synthetic scores in the scaffold pass (24 tests, still passing). This pass adds an end-to-end version using **real** regression-variant scores on real ShareGPT text (`TestOfflineScoringPipelineEndToEnd::test_score_cache_roundtrip_and_ranking`): scored → cached → reloaded → ranked, and the resulting order is exactly descending by score.

### 5c. Live official-engine differential (NOT performed — disclosed limitation)

Running the actual forked `vllm-ltr` engine (real paged-attention scheduler, `AUXLLMEngine`, `_get_ltr_ordered_requests` executing for real) requires building the fork's compiled CUDA extensions (`vllm/csrc`) from source against a pinned older toolchain (torch 2.2.1 / CUDA 12.1), in an environment whose driver/toolkit (CUDA 13.0, torch 2.12) is substantially newer. This was judged out of scope: a multi-step, failure-prone native build, not a "download and run" step, and not necessary to establish the two facts that actually matter for using this baseline correctly — (1) the checkpoint's math is being computed correctly (§5a, bit-exact), and (2) the ranking rule is being applied correctly (§5b, exact). **This is the one deviation the audit cannot close without a substantially larger, riskier effort**, and is reported as such rather than glossed over.

### Quantitative agreement summary

| Check | Sample | Result |
|---|---|---|
| Independent recomputation (classification) | 9 real ShareGPT prompts | 0.0 max abs diff (bit-exact, fp16) |
| Independent recomputation (regression) | 9 real ShareGPT prompts | 0.0 max abs diff (bit-exact, fp16) |
| Ranking order vs. official sort formula | synthetic, n=200 batched | 100% order agreement, all ties stable |
| Batched vs. singleton scoring | 9 real prompts, regression variant | agree to <2e-3 abs (fp16 padding-order noise — see §7), not bit-identical |
| Deterministic across repeated runs | 9 real prompts | 100% identical (both variants) |

---

## 6. Real scoring behavior on real text — an honest finding, not oversold

Using the 9 genuine human-authored prompts in `tests/fixtures/sharegpt_tiny.json` (the only real ShareGPT-style text already vetted/committed in this repo — the full `ShareGPT_V3_unfiltered_cleaned_split.json` corpus is not staged locally, see `data/raw/sharegpt/.gitkeep` and `external/datasets/sharegpt.md`'s own license-verification caveat):

- **Classification variant** (`num_labels=10`, argmax-reduced): **collapses to the same top bin (9) for all 9 prompts.** The pre-argmax logits *do* vary meaningfully by prompt (e.g. top-bin logit 8.19 vs. 5.23 vs. 8.23 across different prompts) — only the argmax reduction discards that signal on this short-prompt sample. This is a real, reproducible property of this checkpoint variant on short inputs, not an adapter defect (confirmed via the bit-exact cross-check in §5a using the *same* raw logits).
- **Regression variant** (`num_labels=1`, raw logit): **9/9 distinct scores** on the same sample — retains the discrimination the classification variant's argmax throws away.

**Recommendation for future ranking use:** the regression variant
(`opt-125m-llama3-8b-sharegpt-score-trainbucket10-b32`) is the more useful
of the two for an actual ranking comparison on short-to-medium prompts;
`adapter/provenance.py` documents both and defaults nothing silently — the
caller picks the variant explicitly via
`download_and_provision_checkpoint(subfolder=...)`.

**Explicit limitation on this finding:** n=9 is small. It demonstrates the
mechanism works end-to-end on genuine text and surfaces a real
variant-selection consideration; it is not a statistically powered claim
about either variant's ranking quality. A larger, licensed ShareGPT sample
would be needed for that — out of scope here (see §10).

---

## 7. Resource and overhead measurement (real, on this session's hardware)

Hardware: RTX 5060 Ti (per `nvidia-smi`), driver 580.159.03, CUDA 13.0. Measured with the regression-variant handle, batch of 32 (9 real prompts cycled), `torch==2.12.0+cu130`, `transformers==5.8.1`.

| Batch size | CPU mean | CPU per-request | GPU mean | GPU per-request |
|---|---|---|---|---|
| 1 | 47.3 ms | 47.3 ms | 60.2 ms (2.3 ms p50 — see note) | 60.2 ms |
| 4 | 90.3 ms | 22.6 ms | 15.5 ms (3.2 ms p50) | 3.9 ms |
| 8 | 361.3 ms | 45.2 ms | 2.9 ms | 0.36 ms |
| 16 | 714.5 ms | 44.6 ms | 3.1 ms | 0.20 ms |
| 32 | 1344.3 ms | 42.0 ms | 4.9 ms | 0.15 ms |
| 64 | — | — | 8.1 ms | 0.13 ms |

- **Note on GPU bs=1 mean vs. p50:** the mean is dominated by first-call CUDA kernel warm-up noise even after a tokenizer-only warmup loop; p50 (2.27 ms) is the more representative steady-state figure. Later batch sizes are stable mean≈p50.
- **Peak GPU memory:** 438 MB at batch 32 (125M-parameter model; well within any single-GPU budget).
- **CPU process RSS after a CPU run:** ~1080 MB (includes Python/torch/transformers overhead, not just model weights).
- **Batch-size sensitivity:** GPU per-request cost drops ~450x from batch 1 (p50-adjusted) to batch 64; CPU per-request cost is roughly flat (~42-47 ms) across batch sizes 4-32 — CPU is compute-bound per-token regardless of batching at this model size, GPU is latency-bound at small batches.
- **Comparison to existing policies:** every one of the 20 registered simulator policies' `select_action()` calls is O(request count) pure-Python arithmetic — microseconds, not milliseconds. Even the fastest GPU-batched LTR scoring here (0.13-0.36 ms/request) is 2-4 orders of magnitude slower per request than any existing policy, and that cost is paid **once per request, offline, before the simulator run** (per the caching design in §4) rather than per simulation step — the asymmetry noted in the original scaffold pass's resource audit still holds and is now quantified rather than estimated.

Reproducible (sanity-bounded, not exact-value-pinned, since latency is hardware-dependent): `TestOverhead` in the GPU test file.

---

## 8. A real bug caught and fixed during this pass

Documented for completeness, since finding and fixing this is itself part
of the fidelity story:

1. **`OPTPredictorHandle.score_batch` didn't move tokenizer output to the
   model's device.** Worked silently on CPU (the default) but raised
   `RuntimeError: Expected all tensors to be on the same device` the moment
   the model was moved to GPU — caught by `TestOverhead` the first time it
   ran. Fixed: tokenizer output is now moved to
   `next(self.model.parameters()).device` before the forward pass.
2. **`download_and_provision_checkpoint` was not idempotent**: re-running it
   against an already-downloaded directory picked up its own
   previously-written `vllm_ltr_provenance.json` sidecar as if it were a
   checkpoint artifact and hashed it, changing the sidecar's own recorded
   hash of itself on every rerun. Fixed: the sidecar's own filename is now
   excluded from the file-hash listing.
3. **Transcription bug, not a code bug:** all four sha256 hashes originally
   written into `CHECKPOINT_PROVENANCE.md` were one hex character short (63
   instead of 64) from a manual copy-paste error while drafting that file.
   Caught by writing a small script to read the actual sidecar JSON and
   diff against the doc, rather than trusting the transcription. Fixed, and
   locked with `TestCheckpointProvenanceDocFormat` so it cannot silently
   recur.

None of these affect the classification/ranking logic verified in §5 — (1)
and (2) are correctness/idempotency bugs in verification tooling caught by
running it for real, (3) is a documentation-accuracy bug. All three are the
kind of thing a "download and actually run it" pass is specifically for.

---

## 9. Fidelity classification and evaluation-readiness recommendation

**Per the task's four fidelity labels:**

- Not **"official"** outright — the tokenizer had to be sourced separately
  and one dependency-adaptation layer (plain `transformers` instead of the
  vLLM fork's engine) sits between the real weights and the score.
- Not **"proxy"** or **"cite-only"** — real weights, real forward pass, real
  bit-exact-verified formula.
- **Closest label: "adapted official."** The predictor is the real,
  hash-verified official checkpoint, running its exact architecture and
  score-extraction formula (bit-exact-verified independently), through a
  necessary and fully-disclosed adaptation layer (plain `transformers`
  instead of the vLLM engine, and a tokenizer sourced from the base model
  since the checkpoint ships none).
- `adapter/provenance.py::FIDELITY_LABEL` updated accordingly:
  `"evaluation-ready external baseline (offline-scored; official checkpoint verified)"`.

**Evaluation-readiness verdict: YES, as an offline-scored external
baseline** — every one of the completion task's requirements (1-7) is now
satisfied:

1. ✅ Venue corrected (§1).
2. ✅ Official checkpoint downloaded, hashed, provenance recorded, never modified (§2).
3. ✅ Architectural fidelity confirmed exactly (§3), one real deviation (tokenizer) found and fixed.
4. ✅ Offline-only scoring pipeline built; simulator request objects untouched; no leakage (§4).
5. ✅ Semantic equivalence validated quantitatively where feasible; the one infeasible check (live official-engine differential) is disclosed, not hidden (§5).
6. ✅ Overhead measured on real hardware, CPU and GPU, with batch-size sensitivity (§7).
7. ✅ This document.

**One structural scope boundary remains, by the task's own design, not as
a shortfall:** requirement 4 explicitly says *"Do NOT modify simulator
request objects."* `ObservableRequest` therefore still cannot carry prompt
text, so `VLLMLTRSemanticReferencePolicy` still cannot be driven purely
from a live `ObservableState` — it requires an externally-precomputed score
map, exactly as designed. This means: **ready to run in any simulator sweep
where a caller supplies real per-request prompt text and precomputed
scores; not runnable against this repo's existing synthetic workload
generators (Poisson/bursty/heavy-tail), which never carried prompt text to
begin with.** This is a pre-existing dataset-availability gap, not
something this completion pass could or should have closed under its own
constraints.

**Still explicitly not promoted:** not in `BASELINE_NAMES`,
`SELECTOR_CANDIDATE_NAMES`, `POLICY_LIBRARY_V2_NAMES`, or
`EXTERNAL_BASELINE_REGISTRY` — per the original task's "do not add to the
main selector candidate set yet" instruction, still in force and
re-confirmed by `TestSelectorScopeInvariants` (unchanged, still passing).

---

## 10. Remaining work

1. Assemble a real, licensed, larger (n≥100) ShareGPT (or equivalent)
   prompt-text-carrying dataset — the only thing standing between "verified
   offline pipeline" and "an actual head-to-head ranking comparison
   against `estimated_service_time_first`/other policies." Must resolve
   the license-verification caveat already flagged in
   `external/datasets/sharegpt.md` before broad use.
2. If a live official-engine differential is ever judged worth the build
   risk: build the pinned vllm-ltr fork's CUDA extensions in an isolated
   environment matching its pinned toolchain (torch 2.2.1/CUDA 12.1) and
   compare `_get_ltr_ordered_requests`'s real output against this adapter's
   on the same trace.
3. Only after (1): decide whether to register `vllm_ltr_semantic_reference`
   in `EXTERNAL_BASELINE_REGISTRY` (evaluation-only, `selector_eligible=False`)
   for head-to-head comparison — deferred per the original task's explicit
   scope limit, not attempted here.

## Exact next baseline

None. Do not begin VTC or any other new baseline implementation per this
task's explicit instruction.

## Exact next action

Item (1) above: assemble a larger, license-cleared real prompt-text dataset
before attempting any comparative evaluation run.
