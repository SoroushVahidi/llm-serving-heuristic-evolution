# vLLM-LTR — Provenance Manifest

**Baseline phase:** started 2026-08-04, branch `contextual-compositional-heuristics-20260731`.
This manifest is the single source of truth for pinned-source facts; code in
`adapter/provenance.py` mirrors these values as importable constants and
tests lock them so this file cannot silently drift from the code.

## Official source

- **Repository:** https://github.com/hao-ai-lab/vllm-ltr
- **Pinned commit:** `13bbf6ff3dab661791d41362551b089e5f77c91c` (branch `main`, 2024-10-31)
- **Why this commit:** the repository has no tags/releases (`git tag` is
  empty); this is the tip of `main` at inspection time, matching the
  "no release exists → pin a specific commit" instruction. Re-verified via
  `git ls-remote https://github.com/hao-ai-lab/vllm-ltr.git HEAD` at
  inspection time.
- **License:** Apache License 2.0 (repository `LICENSE` file, standard ASF text).
- **Paper:** Fu, Yichao; Zhu, Siqi; Su, Runlong; Qiao, Aurick; Stoica, Ion;
  Zhang, Hao. *"Efficient LLM Scheduling by Learning to Rank."*
  **NeurIPS 2024 (main conference)**. Confirmed 2026-08-04 via
  `proceedings.neurips.cc`/`papers.nips.cc` (paper id
  `6c8985579293e0209bdaa4f21bb1d237`) and the official repository's own
  title, "[NeurIPS 2024] Efficient LLM Scheduling by Learning to Rank" —
  independent of the repo's README, which only cites the preprint. The
  arXiv identifier (`2408.15792`) is retained as a supplementary preprint
  reference, not as the venue.

## Environment (per official `README.md`)

- **Python:** 3.10 (conda env)
- **PyTorch:** 2.2.1, torchvision 0.17.1, torchaudio 2.2.1, `pytorch-cuda=12.1`
- **vLLM:** not a released version — a modified **fork** of vLLM (this
  repository *is* the fork; there is no separate upstream version pin beyond
  the commit above). Structurally closest to upstream vLLM ~v0.4.x based on
  the scheduler/executor module layout, but no explicit upstream vLLM
  version string appears in the repo.
- **Key extra deps:** `flash-attn`, `numpy==1.25.2`, `fschat`, `accelerate`,
  `gcsfs`, `scikit-learn`, `scipy`, `matplotlib`, `evaluate`.
- **Training framework:** [allRank](https://github.com/allegro/allRank),
  vendored under `train/allrank/` in the official repo.

## Model checkpoints

- **Hugging Face:** `LLM-ltr/OPT-Predictors`
- Two predictor sizes referenced in `train/README.md`: **125M** (needs ~40GB
  GPU memory to *train*) and **350M** (needs ~80GB GPU memory to *train*).
  Inference memory footprint is far smaller (this is a training-time figure
  from the official docs) but no official inference-memory figure is given.
- Config variants exist for both **regression** (`config_prefill_opt.txt`,
  `config_prefill_opt_350m.txt`) and **classification**
  (`config_prefill_opt_classify.txt`, `config_prefill_opt_350m_classify.txt`)
  training targets — see `train/configs/`.
- **Downloaded and verified 2026-08-04:** both the classification and
  regression 125M/ShareGPT variants. Exact revision, sha256 hashes, tokenizer
  gotcha, and verified-environment record: `CHECKPOINT_PROVENANCE.md`. Full
  fidelity results: `docs/audits/vllm_ltr_baseline_audit_20260804.md`.

## Datasets

- **Training data:** `LLM-ltr/Llama3-Trace` (Hugging Face dataset), downloaded
  via `huggingface-cli download LLM-ltr/Llama3-Trace --local-dir jsonfiles
  --repo-type dataset` per `train/README.md`.
- **Benchmarks:** `benchmarks/` directory (not inspected in depth for this
  scaffold — see "Remaining work" in the audit doc).

## Architecture (verified by reading pinned-commit source, not the README)

- **Ranking model:** `OPTForSequenceClassification`
  (`vllm/model_executor/models/opt.py`, pinned commit) = a standard OPT
  transformer backbone (`OPTModel`, HF-weight-compatible) + a single
  `nn.Linear(config.word_embed_proj_dim, config.num_labels, bias=False)`
  "score" head. No pairwise/listwise loss module lives in the *inference*
  path — that is training-time-only (`train/allrank/`); at inference the
  head just projects the final hidden state to `num_labels` logits.
- **Score extraction:** `compute_logits` slices the head's output to
  `[:, :num_labels]` and (when `num_labels > 1`) takes `argmax(dim=-1)` —
  i.e. the predictor is used as an **ordinal classifier over output-length
  bins**, not a continuous regressor, in the classification config variant.
  `sample()` then records `logits[:, 0].tolist()` as `aux_model_scores`
  (see `official_reference/opt_predictor_head_excerpt.md` for the exact
  pinned lines).
- **Scheduler integration:** `vllm/core/scheduler.py::_get_ltr_ordered_requests`
  sorts `waiting + running + swapped` by `-req.aux_model_score` (descending
  score = highest priority; ties broken by Python's stable sort, i.e. by
  whatever order the three queues were concatenated in). See
  `official_reference/scheduler_ranking_excerpt.md`.
- **Score plumbing:** `Scheduler.aux_model` is a second, separate vLLM engine
  instance (`AUXLLMEngine`, `vllm/engine/aux_llm_engine.py`) running only the
  predictor model, invoked once per newly-arrived request
  (`need_aux_model_score()` / `obtain_aux_scores(...)`) — this is genuine
  extra inference (a full transformer forward pass per request), not a
  cheap heuristic.

## Critical structural finding (drives the integration classification)

The official predictor's **only input is the tokenized prompt text**
(`input_ids` through the OPT backbone) — there is no other feature channel.
This project's simulator (`src/llmserveopt/core/types.py`) represents every
request as `Request`/`ObservableRequest` with **integer token *counts* only**
(`prompt_tokens: int`); it does **not** carry raw prompt text or token IDs
anywhere in its data model. Consequently the real OPT predictor **cannot be
invoked from inside the live discrete-event simulator loop** as it exists
today — there is nothing to tokenize. This is a gap in the simulator's data
model, not a limitation of the official code or of this adapter. See the
audit doc `docs/audits/vllm_ltr_baseline_audit_20260804.md` §3 for the full
classification consequence and `adapter/simulator_policy.py`'s docstring for
how the wrapper handles this explicitly (offline-precomputed score injection,
not live text access).
