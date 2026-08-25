# WildChat-1M Dataset Provenance

## Selection status

**Selected** as the prompt-bearing dataset for the first vLLM-LTR
comparative evaluation (`docs/audits/vllm_ltr_first_comparative_evaluation_20260804.md`).
See "Why WildChat over LMSYS-Chat-1M / ShareGPT" below for the decision
rationale.

## Source

**Dataset**: WildChat-1M
**Official owner**: Allen Institute for AI (AI2)
**HF repo**: `allenai/WildChat-1M`
**Paper**: Zhao et al., "WildChat: 1M ChatGPT Interaction Logs in the Wild"
(arXiv:2405.01470)
**Point of contact**: Yuntian Deng

## License

**ODC-BY** (Open Data Commons Attribution License). Permits copying,
distribution, and adaptation of the dataset with attribution to the source.
This is materially more permissive than LMSYS-Chat-1M's custom license
(which explicitly prohibits redistribution to third parties) and more
verifiable than the unofficial ShareGPT mirrors already flagged as
license-uncertain in `external/datasets/sharegpt.md`.

## Access method

**Ungated** — no license-click-through or access-request step, unlike
`lmsys/lmsys-chat-1m` (custom-license, gated, requires accepting terms in
the HF web UI before `HfApi`/`datasets` calls succeed). Downloadable via
`huggingface_hub`/`datasets` with only a standard HF token (read scope).
This repo's `HF_TOKEN` was confirmed to have to access both repos
(2026-08-04), but WildChat's ungated status makes it the reproducible
choice for anyone re-running this pipeline without first requesting LMSYS
access.

**Pinned revision**: `7d6490e462285cf85d91eabea0f9a954fbddcd1f` (the `main`
branch tip as of 2026-08-04; `HfApi.dataset_info("allenai/WildChat-1M").sha`).
All ingestion in this repo pins this exact commit — never `revision="main"`.

## Fields available

Top-level per-row (one row = one full conversation):

| Field | Type | Notes |
|---|---|---|
| `conversation_hash` | string | Stable content hash, unique per conversation — used as this repo's ingestion sort/sampling key |
| `model` | string | ChatGPT model that served the conversation |
| `timestamp` | timestamp | |
| `conversation` | list of turn dicts | Full turn sequence, see below |
| `turn` | int | Number of turns in the conversation (turn=1 → one user + one assistant message) |
| `language` | string | Detected language (top-level; may reflect the last recorded turn — this repo reads `conversation[0]["language"]` instead, see below) |
| `openai_moderation` | list of dicts | Per-turn OpenAI moderation API output |
| `detoxify_moderation` | list of dicts | Per-turn Detoxify scores |
| `toxic` | bool | Top-level toxicity flag |
| `redacted` | bool | Top-level PII-redaction flag |
| `state`, `country`, `hashed_ip`, `header` | various | Demographic/network metadata (hashed IP, not raw) |

Per-turn (`conversation[i]`): `content`, `role` (`"user"`/`"assistant"`),
`language`, `toxic`, `redacted`, `country`, `state`, `hashed_ip`, `header`,
`timestamp`, `turn_identifier`.

Confirmed by direct row inspection (`hub_repo_details` dataset_preview,
2026-08-04): `turn` is the **conversation's own turn count**, not a
per-row index — `turn=1` selects genuinely single-exchange conversations,
which is exactly this pipeline's target shape (one prompt, one real
response), the same shape `workloads/sharegpt.py` already expects.

## Prompt-text quality and conversation structure

Real, human-authored ChatGPT prompts collected in the wild (not
crowdsourced/templated) — high lexical and topical diversity, spanning
casual chat, coding, creative writing, and (per the moderation columns)
some low-prevalence unsafe content. This repo's ingestion filters on
`conversation[0]["toxic"] is False` and `conversation[0]["redacted"] is
False` (see `scripts/ingest_wildchat_eval_dataset.py`) to exclude both.

## Language coverage

74 languages represented dataset-wide (per HF dataset card). This repo's
selected evaluation sample is filtered to `conversation[0]["language"] ==
"English"` for a controlled, single-language comparison — see
`docs/audits/vllm_ltr_first_comparative_evaluation_20260804.md` for the
rationale (avoids conflating cross-lingual tokenization-length effects
with scheduling-policy behavior in a first evaluation pass).

## Safety / privacy notes

- IP addresses are hashed (SHA-256-style 64-hex-char strings), never raw.
- `redacted=True` rows have had detected PII replaced with placeholders by
  the dataset's own pipeline; this repo's ingestion **excludes** redacted
  rows rather than using placeholder-mangled text as "real" prompt text.
- `toxic=True` rows are excluded from the evaluation sample (both via the
  dataset's own `toxic` flag and, transitively, the `openai_moderation`/
  `detoxify_moderation` columns are available for any future finer-grained
  filtering but are not additionally thresholded here).
- Geographic metadata (state/country) is retained in the raw dataset but
  **never used or propagated** by this repo's ingestion — only
  `conversation[0]["content"]` (prompt text), `conversation[1]["content"]`
  (response text, for real-token-count-derived synthetic-output modeling
  only, never given to deployable policies), and `conversation_hash` are
  read.

## Compatibility with existing simulator request fields

Directly compatible with `src/llmserveopt/workloads/sharegpt.py`'s
existing `{"conversations": [{"from": "human"/"gpt", "value": ...}]}`
shape — this repo's ingestion script re-emits sampled WildChat rows in
that exact shape (`wildchat_eval_sharegpt_shaped.json`) so
`convert_sharegpt_to_requests()` (unmodified) derives `prompt_tokens` /
`actual_output_tokens` from real text exactly as it already does for
ShareGPT, with `arrival_time` / `predicted_output_tokens` / `slo_deadline`
/ `priority` / `class_id` remaining synthetic (via the same
`augment_trace()` path). See `src/llmserveopt/core/types.py::Request`.

## Redistribution

ODC-BY explicitly **permits** redistribution with attribution. This repo
nonetheless follows its existing `data/raw/*` / `data/processed/*`
`.gitignore` convention (see `.gitignore` lines 27-30) and does **not**
commit the downloaded/sampled dataset content itself — only the
deterministic ingestion script, the pinned revision, and the
reproducibility manifest schema are committed. Anyone with the pinned
revision and this repo's ingestion script can regenerate byte-identical
output.

## Why WildChat over LMSYS-Chat-1M / ShareGPT

| Criterion | WildChat-1M | LMSYS-Chat-1M | ShareGPT (unofficial mirrors) |
|---|---|---|---|
| Official source | AI2 (yes) | LMSYS (yes) | No single official source — `anon8231489123/ShareGPT_Vicuna_unfiltered` etc. are third-party re-hosts of scraped sharegpt.com data |
| License | ODC-BY (clear, permissive) | Custom, restricts third-party redistribution | Uncertain / disputed (flagged in `external/datasets/sharegpt.md`, "verify current license status before use") |
| Access | Ungated | Gated (must accept terms) | No formal gating, but no formal license grant either |
| Provenance verified this session | Yes (`hub_repo_details`, `HfApi.dataset_info`) | Yes (access confirmed working) | Not independently verifiable — no canonical repo |

WildChat-1M is the only candidate that is simultaneously **official**,
**license-clear**, and **reproducibly accessible without a manual gating
step**. LMSYS-Chat-1M would also have worked (this session's token has
`canReadGatedRepos=True` and successfully called `dataset_info` on it) but
its license and gating make it a worse choice for a pipeline meant to be
re-run by others without first requesting dataset access. ShareGPT is
rejected per this task's explicit instruction not to use an unofficial
mirror without verified provenance/licensing — unchanged from the caveat
already on record in `external/datasets/sharegpt.md`.

## Download date

Selected and pinned: 2026-08-04. See
`docs/audits/vllm_ltr_first_comparative_evaluation_20260804.md` for the
full selection writeup and `scripts/ingest_wildchat_eval_dataset.py` for
the exact reproducible ingestion command.
