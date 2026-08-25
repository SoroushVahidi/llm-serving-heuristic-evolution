# vLLM-LTR — Checkpoint Provenance

Downloaded and verified 2026-08-04. Weights are never modified; this file
records exactly what was fetched and how it was validated. The weight
files themselves are **not** committed to this repository (large binaries;
they live in the local Hugging Face cache and are reproducible from the
hashes below) — only this provenance record, the sidecar-writing code
(`adapter/checkpoint_loader.py`), and the verification results are
committed.

## Source

- **Hugging Face repo:** `LLM-ltr/OPT-Predictors`
- **Repo revision (Hub API `sha`):** `39df2b41ffe88d5ed967c6035d3838b5b5960379`
- **Repo last modified:** 2024-10-14T02:01:42Z (per Hub API)
- **Model card:** none exists (`README.md` request returns "Entry not found").
- **License:** **not declared on the Hugging Face repo** — no license tag,
  no model card, no `LICENSE` file among the repo's 43 files (verified via
  the Hub API's file listing). The *code* repository
  (`github.com/hao-ai-lab/vllm-ltr`) is Apache-2.0, but that does not
  automatically extend to model weights hosted in a separate repository
  unless explicitly stated, and no such statement exists here. **Treat the
  checkpoint's license as unconfirmed** — permissive intent can reasonably
  be inferred from the paper/code being fully open-source, but this is not
  a substitute for an explicit grant. Any external publication or
  redistribution of derived scores should note this gap.
- **Repo structure:** 12 checkpoint variants total (`opt-125m`/`opt-350m` ×
  `lmsys`/`sharegpt` training data × classification/regression config ×
  varying `trainbucket` sizes), each `<variant>/finetuned/{config.json,
  model.safetensors}` + `<variant>/usage_config.json`. **No tokenizer files
  are shipped for any variant** — see "Tokenizer" below.

## Variants downloaded

Both variants use the ShareGPT-trained, 125M-backbone, `trainbucket`-largest
(classification) / only-available (regression) configuration — the
smallest, most directly comparable pair to this project's own ShareGPT
ingestion (Phase 1.7A) and the pinned source's two training config flavors
(`config_prefill_opt_classify.txt` / `config_prefill_opt.txt`).

### Classification variant

- **Subfolder:** `opt-125m-llama3-8b-sharegpt-class-trainbucket820-b32`
- `config.json` sha256: `ee39f1275f293885dc69a9445b6b7201d0cefd17ce90c220e7ff0762e1b996f6`
- `model.safetensors` sha256: `6dcc1923b86baa4c807d3acde4b4bed57ccd215a04bd67b0ab8215d5bd406781`
- `model.safetensors` size: 250,516,416 bytes (~239 MiB)
- Config: `architectures=["OPTForSequenceClassification"]`, `num_labels=10`,
  `word_embed_proj_dim=768`, `pad_token_id=1`, `torch_dtype=float16`,
  exported with `transformers_version=4.45.2`.

### Regression variant

- **Subfolder:** `opt-125m-llama3-8b-sharegpt-score-trainbucket10-b32`
- `config.json` sha256: `cafd33fb03dcab5a7c4eb88ab8736d1f85f9a9f336ce9d5c371736f142b2a245`
- `model.safetensors` sha256: `37756a0384984065028983c6bb32fac8357dd1df6773a8061939cde8dadea284`
- Config: `architectures=["OPTForSequenceClassification"]`, `num_labels=1`,
  `word_embed_proj_dim=768`, `pad_token_id=1`, `torch_dtype=float16`.

Reproduce either download with:
```python
from baselines.vllm_ltr.adapter.checkpoint_loader import download_and_provision_checkpoint
download_and_provision_checkpoint(
    repo_id="LLM-ltr/OPT-Predictors",
    subfolder="opt-125m-llama3-8b-sharegpt-score-trainbucket10-b32",  # or ...-class-trainbucket820-b32
    revision="39df2b41ffe88d5ed967c6035d3838b5b5960379",
    local_dir="unused",  # snapshot_download manages the real cache location
    verified_environments=[{"torch_version": "2.2.1", "transformers_version": "4.45.2"}],
)
```

## Tokenizer

**Finding:** the checkpoint repository ships no tokenizer files
(`tokenizer.json`, `vocab.json`, `merges.txt`, `tokenizer_config.json`) for
any of its 12 variants. `AutoTokenizer.from_pretrained(checkpoint_dir)`
does **not** error in this situation — it silently falls back to a
GPT-2-style tokenizer with `pad_token_id=None`, `bos/eos/unk="<|endoftext|>"`,
which is **wrong** for this checkpoint (mismatched vocabulary and no pad
token at all). The correct tokenizer is the base pretrained model recorded
in each variant's own `config.json._name_or_path` field
(`facebook/opt-125m` for both variants downloaded here), whose
`pad_token_id=1` matches the checkpoint's own `config.pad_token_id=1`
exactly. `adapter/checkpoint_loader.py::load_opt_predictor_from_local`
reads the raw `config.json` before `from_pretrained()` mutates
`_name_or_path`, loads the tokenizer from that recorded base model, and
raises `StaleArtifactError` if the resulting `pad_token_id` doesn't match
the checkpoint's — this silent-fallback failure mode is exactly the kind of
thing that would otherwise produce plausible-looking but wrong scores.

## Environments this checkpoint has been verified to load and score
correctly under

Recorded in each downloaded variant's `vllm_ltr_provenance.json` sidecar
(not committed — regenerated by `download_and_provision_checkpoint`):

1. `torch==2.2.1`, `transformers==4.45.2` — the checkpoint's own recorded
   export environment (`config.json.transformers_version`); not
   independently re-run in this session (no torch 2.2.1 environment
   available here), included because it's the environment the artifact
   itself claims.
2. `torch==2.12.0+cu130`, `transformers==5.8.1` — **actually re-run** in
   this session: both variants loaded cleanly (exact state-dict key/shape
   match, see the audit doc §3) and produced deterministic, sane-shaped
   scores on real text (see the audit doc §4-5).
