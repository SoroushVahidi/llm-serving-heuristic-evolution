# PARS Baseline — Provenance

> **Naming disambiguation (added 2026-08-04):** this repository's own
> `docs/external_baseline_coverage_report.md` had already, since an
> earlier phase, informally used "PARS" as shorthand for a *different,
> unrelated* paper (Zheng et al., NeurIPS 2023, "Response Length
> Perception and Sequence Scheduling," approximated internally by
> `estimated_service_time_first` and never given an official-code
> integration) — that earlier usage is now called **PARS-2023** in this
> project's docs to avoid collision. Everything in this file and
> `baselines/pars/` refers to the *different* paper this task actually
> integrates (Tao et al., ISC 2026, official repo `SPEAR-UIC/PARS`),
> referred to in prose as **PARS-Serve-2026** in this project's newer
> docs (`docs/BASELINE_STATUS.md`,
> `docs/audits/pars_baseline_implementation_20260804.md`). Code
> identifiers (the `baselines/pars/` package, the `pars_semantic_reference`
> policy name, file paths) are **left unchanged** — only prose/narrative
> text uses the disambiguated names, to avoid destabilizing the
> already-built, about-to-run evaluation pipeline with a code rename.

## Official source

**Paper:** Yiheng Tao, Yihe Zhang, Matthew Dearing, Xin Wang, Yuping Fan,
Michael E. Papka, Zhiling Lan, *"Ranking Before Serving: Low-Latency LLM
Serving via Pairwise Learning-to-Rank"*, Proc. of ISC High Performance
2026 (June 22–26, 2026, Hamburg, Germany). arXiv:2510.03243 — v1
(2025-09-25) was titled *"Prompt-Aware Scheduling for Low-Latency LLM
Serving"* (the title this task's request used); retitled in v3
(2026-06-26, the ISC camera-ready) to the "Ranking Before Serving" title.
Same paper, same authors, same arXiv ID throughout. Affiliations:
University of Illinois Chicago and Argonne National Laboratory.

**Repository:** `https://github.com/SPEAR-UIC/PARS` (public GitHub org
`SPEAR-UIC`). Verified via `gh api repos/SPEAR-UIC/PARS` — real, public,
3 commits (`eff51953bc` initial release 2026-03-22, `a403524027` checkpoint
clarification 2026-03-22, `fd4e125b65` citation update 2026-07-24).

**Pinned commit:** `fd4e125b65bb73aef5eccafa79c2509434be61ec` (HEAD of
`main` at clone time, 2026-08-04). Cloned to
`/home/soroush/.cache/external_baselines/PARS` — **outside this repo's git
tree**, never committed here (see License below for why).

## License — verified, and the central risk this baseline carries

**No LICENSE file exists anywhere in the repository.** Verified via the
full recursive git tree (`gh api repos/SPEAR-UIC/PARS/git/trees/main?recursive=1`)
— only `.gitignore`, `README.md`, and the three stage directories
(`data_preprocess/`, `predictor_train/`, `predictor_serving/`) exist; no
`LICENSE`, `LICENSE.md`, `COPYING`, or license field in `README.md`. The
GitHub API's own `license` field for the repo is `null`.

**Consequence:** under default copyright, code with no explicit license
grant is "all rights reserved" by the authors — GitHub's terms permit
viewing and forking, but do not themselves grant a license to use, modify,
or redistribute the code. This is a materially worse situation than any
other external baseline in this repo (contrast `baselines/vllm_ltr/`,
Apache-2.0; WildChat, ODC-BY; even LMSYS-Chat-1M, which at least has a
*stated*, if restrictive, custom license).

**Decision (explicit, user-directed 2026-08-04):** proceed with full
official integration — train the official, unmodified code locally for
non-commercial academic research/evaluation purposes, and:
- **never commit, vendor, or redistribute the official repository's
  source files** in this project's git history (the clone lives entirely
  outside this repo's tree, referenced only by URL + pinned commit hash);
  this project's own adapter code only *dynamically imports* the official
  `PairwiseRanker` class at runtime from the external clone path (see
  `adapter/checkpoint_loader.py`) rather than duplicating its source;
- **the trained checkpoint (a derived artifact) is also never committed**
  — it lives under `results/pars_official/` (gitignored, `results/*`),
  consistent with how `baselines/vllm_ltr/` keeps its (differently-
  licensed, Apache-2.0) checkpoint out of git too;
- this baseline's status is explicitly **evaluation-only, license-
  encumbered** everywhere it is documented — never presented as freely
  redistributable, and this repo's own results derived from it (ANWG
  numbers, comparisons) are safe to report (they are *our* measurements,
  not a redistribution of their code/data), but the underlying official
  code/checkpoint should not be assumed reusable by a third party without
  contacting the authors.

## Training dataset — chosen and its license

The official repo supports 4 GPT4-based dataset pipelines (see
`data_preprocess/README.md`):

| Pipeline | HF dataset | License (verified) |
|---|---|---|
| `alpaca` | `vicgalle/alpaca-gpt4` | **CC BY-NC 4.0** (non-commercial) |
| `code` | `theblackcat102/evol-codealpaca-v1` | Apache-2.0 |
| `lmsys` | `lmsys/lmsys-chat-1m` (model=gpt-4, turn=1) | Custom, restricts third-party redistribution — **already flagged and avoided elsewhere in this repo** (see `external/datasets/wildchat.md`'s WildChat-vs-LMSYS comparison) |
| `math` | `TIGER-Lab/MATH-plus` | MIT |

**Chosen: `alpaca` (`vicgalle/alpaca-gpt4`).** Not the most permissively
licensed option (`code`/`math` are Apache-2.0/MIT, fully unrestricted) —
chosen instead for **domain realism**: `vicgalle/alpaca-gpt4` is diverse,
general-purpose instruction-following text (52,002 examples: explanations,
writing, general Q&A, etc.), the closest available match among the 4
official options to WildChat's general real-chat domain this baseline
will ultimately be evaluated against. `code`/`math` are narrow, single-
domain datasets (programming tasks / math problems only) that would train
a response-length predictor specialized to a domain this evaluation never
tests, confounding "PARS the method underperforms" with "this predictor
was trained on the wrong domain." `lmsys` was excluded per this repo's
pre-existing license policy (see above), independent of this task.

**CC BY-NC 4.0 (non-commercial) is accepted here** because this project's
use is non-commercial academic research (training a baseline scheduler
predictor to compare against other non-commercial research baselines) —
squarely within what CC BY-NC permits. This does not license any future
commercial use of the resulting checkpoint or this evaluation's derived
artifacts; that would require separately contacting the dataset's
publishers.

### Recovered dataset provenance (added 2026-08-04, while training was
still in progress -- recovered via a lightweight, read-only HF Hub API
call, not by re-downloading or touching the training job)

- **HF repository:** `vicgalle/alpaca-gpt4`
- **Resolved revision (`HfApi().dataset_info(...).sha`):** `f7e3ded725cb81e8e564e32feb12860f376f2b51`
- **Dataset `last_modified`:** 2024-02-10T10:03:45Z; **`created_at`:** 2023-04-07T16:22:59Z
  -- last changed well over a year before this evaluation (2026-08-04),
  so this resolved revision is, with very high confidence, the exact
  content the original (unpinned) `load_dataset("vicgalle/alpaca-gpt4")`
  call actually fetched -- but this was recovered *after the fact*, not
  pinned *at* download time (the official `preprocess_alpaca_gpt4.py`
  script has no `revision=` parameter to pin in the first place; patching
  that in would deviate from "run the official script unmodified").
- **License:** CC BY-NC 4.0 (verified via the HF dataset card; see above).
- **Preprocessing command (official, unmodified):**
  `python scripts/preprocess_alpaca_gpt4.py --output-dir results/pars_official/data_preprocess/alpaca_gpt4`
  (defaults: `similarity-threshold=0.2`, `train-num-pairs=10`,
  `val-num-pairs=10`, `test-size=0.1`, `seed=42`).
- **Training command (official, unmodified):** see "Official training
  hyperparameters used" below.
- **Download/preprocessing timestamp:** 2026-08-04, ~14:03 EDT (local
  workstation clock; see `results/pars_official/data_preprocess/alpaca_gpt4/`
  file mtimes for the precise moment).
- **Local cache path (non-portable, this machine only):**
  `results/pars_official/data_preprocess/alpaca_gpt4/` (gitignored,
  `results/*` -- never committed).
- **Row counts (recomputed directly from the preprocessed files, not
  merely quoted from the script's own printed log):** `train_data.json`
  46,801 rows; `val_data.json` 5,201 rows; `train_pairs_length_diff_0.2.json`
  468,010 pairs; `val_pairs_length_diff_0.2.json` 52,010 pairs.
- **SHA-256 of the preprocessed output files** (recorded for this run's
  own reproducibility record -- these files themselves are gitignored,
  never committed, so these hashes are the only durable record of their
  exact content):
  - `train_data.json`: `a4656651ba398a028252b00bac6b67350958d4834b7878226a2bb4eef7ff672d`
  - `val_data.json`: `aca1be76b6df59717ada7c484e0693e5ffcb238595446e74695b55d23f3c188f`
  - `train_pairs_length_diff_0.2.json`: `8bb8ca8c3a7cc93a2279bb652da1fb07cda595821508d7c2cb6c9f38eab3e663`
  - `val_pairs_length_diff_0.2.json`: `2cf94dc630a5cd396ec530c9802b0caa6c1c8666f1c354c1cac0a2eee70810c5`

## Architecture (verified directly from the official training/serving scripts)

`PairwiseRanker` (`predictor_train/scripts/train_pairwise_bert.py` and
`predictor_serving/scripts/serve_predictor_score.py` — functionally
identical class definition in both, confirmed via a structural diff of the
class body; the only difference is `forward()`'s parameter casing,
`input_ids_A/B` in the training script vs. `input_ids_a/b` in the serving
script, purely cosmetic since both are called positionally): a
`bert-base-uncased` encoder (via `transformers.AutoModel`) + a single
linear layer (`hidden_size -> 1`) on the pooler output (falls back to
`last_hidden_state[:, 0]`, i.e. the `[CLS]` token, if no pooler exists).
Trained with `MarginRankingLoss` on prompt pairs, target = `+1` if
`prompt_A`'s real response is longer than `prompt_B`'s, else `-1`. **A
higher PARS score means the model predicts a LONGER response** — for
scheduling purposes (shortest-job-first-style prioritization), a
deployable policy must sort by **ascending** PARS score, not descending
(verified directly from the loss construction, not assumed).

## Official training hyperparameters used (unmodified defaults)

`model_name=bert-base-uncased`, `max_length=128`, `batch_size=128`,
`num_epochs=3`, `learning_rate=2e-5`, `margin=1.0`, `warmup_ratio=0.1`,
`weight_decay=0.01`, `seed=42` (all defaults from
`train_pairwise_bert.py`'s `argparse` definitions — none overridden).

## Preprocessing (unmodified, official script)

`data_preprocess/scripts/preprocess_alpaca_gpt4.py`, defaults
(`similarity-threshold=0.2`, `train-num-pairs=10`, `val-num-pairs=10`,
`test-size=0.1`, `seed=42`) — produced 46,801 train rows / 5,201 val rows
/ 468,010 train pairs / 52,010 val pairs from the 52,002-row source
dataset. Output under `results/pars_official/data_preprocess/alpaca_gpt4/`
(gitignored).

## Deviations from the official pipeline (documented, all in Step 3+ of this evaluation, none in the official code itself)

1. **No FastAPI server.** `predictor_serving/scripts/serve_predictor_score.py`
   wraps the model in a FastAPI HTTP service (`/score`, `/score_batch`,
   `/compare`, `/compare_batch`). This evaluation calls the identical
   `PairwiseRanker.score()` method directly in-process (offline batch
   scoring, exactly mirroring how `baselines/vllm_ltr/adapter/offline_scoring.py`
   already handles vLLM-LTR) — this is an execution-context change only;
   the scoring computation itself is unmodified official code, dynamically
   imported from the pinned clone.
2. **No simulator integration in the official repo.** The official README
   explicitly describes integration as "assign request priorities through
   the platform scheduler ... use the official benchmark code of vLLM."
   This repo's simulator is not vLLM — an adapter (`adapter/simulator_policy.py`)
   is unavoidable, exactly as it was for vLLM-LTR. It is evaluation-only,
   not selector-eligible (see `adapter/simulator_policy.py`'s own
   `SELECTOR_ELIGIBLE = False`).
