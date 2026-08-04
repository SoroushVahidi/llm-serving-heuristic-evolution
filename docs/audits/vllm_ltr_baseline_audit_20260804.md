# vLLM-LTR Baseline-Integration Audit

**Date:** 2026-08-04
**Branch:** `contextual-compositional-heuristics-20260731`
**Tests:** 23/23 new tests pass (`tests/test_vllm_ltr_baseline_adapter.py`); full non-live suite unaffected (see §8)
**Scope:** Baseline-integration phase begun. **CC5/CC6 logic and results are untouched** — see `docs/contextual_composition_roadmap.md` for that track; this document and all files it describes are entirely outside it.

---

## 1. Official source

| Field | Value |
|---|---|
| Repository | https://github.com/hao-ai-lab/vllm-ltr |
| Pinned commit | `13bbf6ff3dab661791d41362551b089e5f77c91c` (tip of `main`, 2024-10-31) |
| Why this pin | No tags/releases exist in the repository (`git tag` is empty); this is the tip of `main` at inspection time — the correct fallback per "prefer a release; if none exists, pin a commit and explain why." |
| License | Apache License 2.0 |
| Paper | Fu, Yichao; Zhu, Siqi; Su, Runlong; Qiao, Aurick; Stoica, Ion; Zhang, Hao. *"Efficient LLM Scheduling by Learning to Rank."* arXiv:2408.15792 (2024). No venue beyond the arXiv preprint is asserted (do not upgrade to a conference name without independent verification). |
| Checkpoints | `LLM-ltr/OPT-Predictors` (Hugging Face); 125M and 350M OPT-backbone variants, regression and classification config flavors |
| Training dataset | `LLM-ltr/Llama3-Trace` (Hugging Face) |

Full manifest, environment versions (Python 3.10, PyTorch 2.2.1, CUDA 12.1,
flash-attn etc.), and architecture details: `baselines/vllm_ltr/PROVENANCE.md`.

---

## 2. Architecture (from reading the pinned commit, not just the README)

- **Ranking model:** `OPTForSequenceClassification` in
  `vllm/model_executor/models/opt.py` = a standard OPT transformer backbone
  (`OPTModel`) + one `nn.Linear(word_embed_proj_dim, num_labels, bias=False)`
  score head. No pooling/MLP/dropout at inference.
- **Score semantics:** for `num_labels > 1` (the classification config), the
  head's logits are reduced via `argmax` to a single bin index (ordinal
  classifier over output-length bins); for `num_labels == 1` (regression
  config) the raw logit is used directly.
- **Scheduler integration:**
  `vllm/core/scheduler.py::_get_ltr_ordered_requests` sorts
  `waiting + running + swapped` by `-req.aux_model_score` (descending score
  = highest priority), via Python's stable `sorted()` (ties keep input
  order). Scores are computed once per request via a second, dedicated vLLM
  engine (`AUXLLMEngine`) and cached (`need_aux_model_score()` /
  `obtain_aux_scores()`), not recomputed every step.
- Literal pinned-source excerpts (citation only, never imported):
  `baselines/vllm_ltr/official_reference/scheduler_ranking_excerpt.md`,
  `baselines/vllm_ltr/official_reference/opt_predictor_head_excerpt.md`.

**Key finding used to design the loader:** the pinned commit's
`OPTForSequenceClassification` is field-for-field identical to
HuggingFace `transformers`' own `OPTForSequenceClassification` (verified by
diffing against `transformers` v4.30.0's `modeling_opt.py`: same
`self.model = OPTModel(config)`, same
`self.score = nn.Linear(config.word_embed_proj_dim, num_labels, bias=False)`).
The vLLM fork's custom class exists only to plug into vLLM's paged-attention
serving plumbing, not to change the ranking computation. This means the
official checkpoint should be loadable through plain `transformers`
(`AutoModelForSequenceClassification`), with no custom weight-key
remapping — **this specific claim has not been verified against the real
checkpoint bytes** (no ~GB-scale network fetch was performed for this
scaffold); see §9.

---

## 3. Integration classification

None of the six single-label categories in the task instructions fits
cleanly on its own; the honest answer is a **structural split**, driven by
one finding:

> **The official predictor's only input is tokenized prompt *text*. This
> project's simulator (`src/llmserveopt/core/types.py::Request`/
> `ObservableRequest`) represents every request as integer token *counts*
> only (`prompt_tokens: int`) — there is no prompt text or token-ID field
> anywhere in the data model.**

Consequences:

- **The predictor artifact itself** (OPT backbone + linear score head):
  classified as **"official predictor reused with a simulator adapter"** —
  it can be loaded and run *independently* of vLLM (no vLLM internals
  needed, per the `transformers`-compatibility finding above), *given real
  prompt text*. This is fully implemented in
  `baselines/vllm_ltr/adapter/checkpoint_loader.py`, gated behind an
  optional dependency and a provenance sidecar.
- **The scheduling/ranking rule** (`-aux_model_score` descending sort,
  stable tie-break): classified as **"official code usable only as a
  semantic reference"** for the vLLM-engine-specific plumbing around it
  (paged KV, `AttentionMetadata`, `LogitsProcessor`, the separate
  `AUXLLMEngine`) — none of that is vendored or executed — but the *rule
  itself* is simple enough to reproduce exactly and is implemented
  faithfully in `baselines/vllm_ltr/adapter/ranking_adapter.py`.
- **Wiring the real predictor into the live discrete-event simulator loop**
  (i.e., a `BasePolicy.select_action()` that calls the OPT model on each
  step): **not currently possible** — `ObservableRequest` has nothing to
  tokenize. This is a gap in the simulator's data model, not a limitation
  of the official code or this adapter. `simulator_policy.py` handles this
  explicitly (see §4) rather than papering over it.

### Answers to the task's specific questions

| Question | Answer |
|---|---|
| Which official files can be reused unchanged? | None verbatim (vLLM-engine plumbing is not vendored); the *architecture* (backbone + head shape) is reproduced via plain `transformers`, and the *ranking rule* is reproduced as ~5 lines of pure Python. Both are cited, not copied, at the file/line level in `official_reference/`. |
| Which pieces depend on vLLM internals? | The scheduler's `AttentionMetadata`/`LogitsProcessor`/`Sampler`/`AUXLLMEngine` plumbing — all serving-efficiency machinery, not part of the ranking computation itself. |
| Can the learned ranker run independently? | Yes, in principle, given real prompt text and the checkpoint — via plain `transformers`, no vLLM needed. Untested against real weights in this environment (no torch/transformers installed; no checkpoint downloaded — see §9). |
| What exact input features does it require? | Tokenized prompt text (`input_ids`) only. No other feature channel exists in the official architecture. |
| What output does it produce? | A single scalar per request: an output-length bin index (classification config, `num_labels > 1`) or a raw regression value (`num_labels == 1`). |
| How is its output converted into request order? | `sorted(requests, key=lambda r: -score)`, stable (Python's `sorted`) — reproduced exactly in `ranking_adapter.order_by_ltr_score`. |
| Can its ranking semantics be reproduced inside our simulator without replacing the official model? | The *sort rule*: yes, exactly. The *score*: no — it structurally requires real prompt text the simulator doesn't carry, so it cannot be reproduced *or* replaced with a substitute inside the simulator; scores must come from outside (see §4). |
| Vendoring / submodule / pinned dependency / isolated environment? | A pinned **optional dependency** (`torch`, `transformers` — new `vllm_ltr` extra in `pyproject.toml`), plus a provenance-sidecar convention for the checkpoint. No git submodule, no vendored C++/CUDA, no isolated environment needed, because only the backbone+head (plain HF-compatible) matters, not the full vLLM fork build. |

---

## 4. What was implemented

```
baselines/vllm_ltr/
  PROVENANCE.md                                   # manifest (source of truth)
  official_reference/
    scheduler_ranking_excerpt.md                   # pinned citation, not executable
    opt_predictor_head_excerpt.md                  # pinned citation, not executable
  adapter/
    provenance.py                                  # importable constants mirroring the manifest
    errors.py                                       # MissingDependencyError, MissingCheckpointError,
                                                      # StaleArtifactError, VersionMismatchError, MissingScoreError
    checkpoint_loader.py                            # optional torch/transformers loader + provenance-sidecar gate
    ranking_adapter.py                              # faithful -score-descending stable sort
    simulator_policy.py                             # VLLMLTRSemanticReferencePolicy(BasePolicy)
tests/test_vllm_ltr_baseline_adapter.py             # 23 tests, see §5
```

`VLLMLTRSemanticReferencePolicy` takes a precomputed `{request_id: score}`
map at construction (produced *offline*, before a simulator run, by
tokenizing each request's real prompt text and calling
`OPTPredictorHandle.score()`) and admits requests in official-ranked order
each step. A request missing from the score map raises `MissingScoreError`
— there is no fallback heuristic, satisfying the "do not silently replace
the ranker" constraint. It reads no oracle information: `ObservableRequest`
structurally has no `actual_output_tokens` field, and the policy never
reads `predicted_output_tokens` either (locked by
`TestNoLeakage.test_policy_never_reads_predicted_output_tokens_either`).

**Not implemented / explicitly out of scope for this scaffold:** actually
downloading and loading the real `LLM-ltr/OPT-Predictors` checkpoint bytes
(no torch/transformers installed in this environment; no network fetch of
GB-scale weights performed — see §9), and any offline scoring pipeline that
would tokenize a real dataset's prompt text and populate the score map for
a full simulator run.

---

## 5. Fidelity verification

All 23 tests in `tests/test_vllm_ltr_baseline_adapter.py` pass. Coverage:

- **Feature construction:** N/A at the simulator-wiring layer (no live
  feature construction happens — see §3); the checkpoint loader's expected
  input (tokenized prompt text) is documented and asserted structurally.
- **Ranking-input / score equivalence:** `TestRankingSemanticEquivalence`
  cross-checks the adapter's sort against the literal official key formula
  (`sorted(reqs, key=lambda req: -req.aux_model_score)`) applied to
  identical synthetic inputs — orders match exactly.
- **Ranking-order equivalence:** same tests, plus a 200-request batching
  test confirming per-score-bin stability at larger scale.
- **Deterministic tie-breaking:** `test_deterministic_tie_breaking_preserves_input_order`
  and `test_deterministic_across_repeated_calls`.
- **Batching behavior:** `test_batching_many_requests_preserves_stable_partial_ties`.
- **No inference-time label leakage:** `TestNoLeakage` — structural
  (`ObservableRequest` has no `actual_output_tokens` field at all) and
  behavioral (the policy prefers the *injected score*, not
  `predicted_output_tokens`, when the two disagree).
- **Missing-checkpoint behavior:** `test_missing_dependency_or_missing_checkpoint`
  (genuinely exercises `MissingDependencyError` in this environment, since
  torch/transformers are not installed — not mocked).
- **Version-mismatch rejection:** `test_version_mismatch_rejected`.
- **Stale-artifact rejection:** `test_stale_artifact_rejected_when_sidecar_absent`,
  `test_stale_artifact_rejected_when_pinned_commit_differs`.
- **Selector-scope invariants:** `TestSelectorScopeInvariants` — not in
  `BASELINE_NAMES`, `SELECTOR_CANDIDATE_NAMES`, `POLICY_LIBRARY_V2_NAMES`,
  or `EXTERNAL_BASELINE_NAMES`; `src/llmserveopt` has zero references to
  `baselines.vllm_ltr` (grep-verified test).

### Fidelity label

**`FIDELITY_LABEL = "official predictor reused with a simulator adapter (offline-only)"`**
(`adapter/provenance.py`). Not labeled plain "official" (the checkpoint
itself is untested against real weights) or "faithful" (that label is
reserved, by this repo's convention, for the six `*_faithful` scheduling
*policies* that run entirely inside the simulator — this baseline's
predictor cannot run inside the simulator loop at all, live). Not "proxy"
or "cite-only" either: the ranking *rule* is exactly reproduced and the
predictor architecture is a real, loadable artifact once real prompt text
and the checkpoint are supplied out-of-band.

### Known deviations

1. **Source of divergence:** structural/data-model, not numerical or
   architectural. The simulator's `ObservableRequest` has no prompt text —
   the official predictor cannot be invoked live inside `select_action()`.
2. **Untested claim:** the "loadable via plain `transformers`, no custom
   weight remapping" hypothesis (§2) is architecturally well-supported
   (identical class shape, verified against `transformers` v4.30.0 source)
   but has not been checked against the actual `LLM-ltr/OPT-Predictors`
   checkpoint files.
3. **Scope of the ranking rule:** the official rule reorders
   `waiting + running + swapped` combined; this simulator's policies only
   choose admission order from `waiting_queue` per step (already-admitted
   requests are advanced by the service model, not reordered by the
   policy). The adapter reproduces the rule over whatever sequence it's
   given rather than the three-queue concatenation — see
   `ranking_adapter.py`'s module docstring.

---

## 6. Resource and comparability audit

| Question | Answer |
|---|---|
| GPU required? | Not for this scaffold's tests (pure Python + optional CPU-side `transformers` inference). Real checkpoint inference (125M/350M OPT forward pass per request) is feasible on CPU but would be far slower than any of the existing 20 simulator-side policies, which are O(1) Python per request. |
| CPU inference possible? | Yes, architecturally (small OPT variants) — not attempted here. |
| Expected per-request inference overhead | Not measured (no checkpoint loaded). Order-of-magnitude expectation: milliseconds-to-tens-of-milliseconds per request on CPU for a 125M-350M parameter transformer forward pass over a short prompt — several orders of magnitude more than any existing policy's `select_action()` call. |
| Training cost | Per official docs: 350M predictor needs ~80GB GPU memory to train, 125M needs ~40GB. Not attempted; training is out of scope for this baseline-integration phase. |
| Checkpoint size | Not measured (not downloaded). Expect roughly the OPT-125M/350M base-model size plus a small linear head (~250MB-1.4GB range, typical for these HF checkpoint sizes — not verified). |
| Compatible with current simulator workloads? | Only if a workload also carries real prompt text (e.g., the raw ShareGPT/BurstGPT text ingested in Phase 1.7A, before it was reduced to `prompt_tokens` counts) — not the synthetic Poisson/bursty generators, which never had text. |
| Fair to compare under the same resource budget as the other 20 policies? | No, not as a live per-step policy — it would require an offline preprocessing pass (tokenize + score every request in a dataset before the simulator run even starts) that none of the other 20 policies need. Any future comparison must account for that asymmetric setup cost explicitly. |
| Evaluation-only or future selector candidate? | Evaluation-only for now, and only once the offline scoring pipeline exists. Explicitly excluded from the selector candidate set per the task instructions and locked by `TestSelectorScopeInvariants`. |

No GPU jobs were run for this scaffold.

---

## 7. Documentation updated

- `docs/baselines.md` — new "Baseline-integration scaffolds (evaluation-only, not yet runnable end-to-end)" section documenting vLLM-LTR's status.
- `docs/external_baseline_decision.md` — §B.1 status line updated (scaffold added; not yet a runnable selector-eligible baseline), mirroring the §B.2 SCORPIO precedent; §F checklist row updated.
- `docs/roadmap.md` — one additive line appended to the existing "Current track" numbered list (item 7); the CC5/CC6 banner (lines 1-13) and the historical phase table are untouched.
- **Not touched:** anything under the CC-scoped canonical files (`docs/contextual_composition_roadmap.md`, `docs/contextual_composition_decisions.md`, `docs/START_HERE_CONTEXTUAL_COMPOSITION.md`, `docs/CONTEXTUAL_COMPOSITION_BRANCH.md`, `docs/RESUME_CONTEXTUAL_COMPOSITION.md`, `scripts/check_contextual_composition_status.py`), nor `docs/current/PROJECT_STATUS.md`'s unrelated Wolverine/pause narrative.

---

## 8. Validation run

```
python -m compileall -q src scripts tests         # PASS
python scripts/check_contextual_composition_status.py                    # "contextual composition status check passed"
python scripts/check_contextual_composition_status.py --resume-readiness # "contextual composition resume-readiness check passed"
pytest --collect-only -q                          # collected cleanly, no errors
pytest tests/test_vllm_ltr_baseline_adapter.py -q # 23 passed
```

(Full non-live suite result recorded in the final commit summary printed to
the user; see that output for the exact pass count.)

---

## 9. Remaining work

1. Download the real `LLM-ltr/OPT-Predictors` checkpoint and verify the
   "loadable via plain `transformers`, no custom weight remapping"
   hypothesis against actual weight files (requires network access +
   ~GB-scale storage; not done here per "do not run expensive jobs unless
   already safe").
2. Install the `vllm_ltr` optional extra (`torch`, `transformers`) in a
   dedicated environment and re-run `TestCheckpointLoaderFailureModes` to
   confirm the `MissingCheckpointError` (rather than `MissingDependencyError`)
   path, plus a real end-to-end `OPTPredictorHandle.score()` call.
3. Build an offline scoring pipeline: for a real-text dataset (e.g. the
   Phase 1.7A ShareGPT ingestion, before token-count reduction), tokenize
   each prompt, run it through the checkpoint, and produce the
   `{request_id: score}` map `VLLMLTRSemanticReferencePolicy` requires.
4. Only after (1)-(3): consider whether `vllm_ltr_semantic_reference`
   should become an `EXTERNAL_BASELINE_REGISTRY` entry (evaluation-only,
   `selector_eligible=False`) for head-to-head comparison against
   `estimated_service_time_first` — decide fidelity-class naming at that
   point (this scaffold does not need `FidelityClass` to gain a new member
   yet).

## Exact next action

Do not begin VTC implementation. Next baseline-integration action is item
(1) above: download and verify the official checkpoint against the
`transformers`-compatibility hypothesis in an environment with network
access and the `vllm_ltr` extra installed.
