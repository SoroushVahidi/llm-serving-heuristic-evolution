# Branch and PARS-Readiness Audit — 2026-08-04

**Scope note:** audit-only, read-only throughout. No files modified, no
commits, no interaction with the active `pars_training` tmux session
beyond read-only `capture-pane`. This document itself is written but
**not committed** per the task's explicit instruction ("do not commit or
push unless explicitly instructed after the audit").

## 1. Repository state

- **Current branch:** `contextual-compositional-heuristics-20260731`
- **Current SHA:** `4972dd5766d9588daaf6a792ec01f141f08df106`
- **Upstream branch:** `origin/contextual-compositional-heuristics-20260731`
- **Ahead/behind:** 0 ahead / 0 behind — **local and remote are synchronized.**
- **Working-tree status:** clean of tracked modifications; untracked files
  present (see below) — no staged files.
- **Untracked files** (all PARS work-in-progress, correctly not yet
  committed per the standing "do not commit until PARS integration is
  complete" instruction):
  - `baselines/pars/` (adapter package + `PROVENANCE.md`)
  - `docs/audits/pars_baseline_implementation_20260804.md`
  - `scripts/run_pars_first_comparative_evaluation.py`
  - `scripts/score_pars_eval_datasets.py`
  - `scripts/verify_pars_comparison_results.py`
  - `tests/test_pars_baseline_adapter.py`
  - `tests/test_pars_checkpoint_fidelity_gpu.py`
- **Ignored-but-relevant experiment artifacts:** `results/pars_official/**`
  (gitignored via `results/*`, confirmed with `git check-ignore -v`) —
  contains `data_preprocess/alpaca_gpt4/*.json` (preprocessed pairwise
  training data), `predictor_train/alpaca_gpt4_bert/best_model.pt`
  (438,015,898 bytes, saved after epoch 1 — a new `best_model.pt` will
  overwrite this if a later epoch's validation accuracy exceeds epoch 1's
  0.9114), and two training logs (one empty 0-byte leftover from the
  first, XET-stalled training attempt that was killed and restarted; one
  live, growing log for the current successful run).
- **Active tmux sessions related to this repo:** `pars_training` (active,
  the subject of §2); `cc3_dsl_verifier`, `cc4_oracle_dataset`,
  `cc4b_cc5_retry`, `cc5_contextual_predictor`, `cc5_uncertainty_regime`
  (all from completed, already-committed CC3-CC5 work, 2026-08-03 —
  presumed idle/stale shells, not verified to still be running live
  processes, out of scope to interrogate further under a read-only audit
  of the *current* PARS-training-adjacent state); `vllm_ltr_comparison_recovery`,
  `vllm_ltr_eval_dataset` (from the already-completed, already-committed
  vLLM-LTR recovery work earlier today — likely idle); `vllm-our-method-comparison-pilot`,
  `vllm-scaled-baseline-comparison` (older, from 2026-07-03, unrelated to
  today's work).

## 2. Running PARS job

- **tmux session:** `pars_training`
- **Process status:** running, healthy (`ps`: state `Sl+`, 100% CPU,
  `nvidia-smi`: 89% GPU utilization, 14,447/16,311 MiB VRAM used)
- **Current epoch/progress:** Epoch 2/3, 50% (1,827/3,657 steps)
- **Elapsed runtime:** ~68 minutes (process `ELAPSED` ≈ 4,081s at last
  check; training log timestamped start `2026-08-04T18:10:52Z`)
- **Estimated remaining runtime:** ~19 more minutes to finish epoch 2
  (at the observed steady ~1.53 it/s) + ~40 minutes for epoch 3 + a few
  minutes of validation-set evaluation per epoch (3 total) ≈ **~60-65
  more minutes**, i.e. total training runtime ≈ 2h05m-2h10m.
- **Latest training loss:** 0.113 (mid-epoch-2 batch; epoch 1 ended at
  train_loss=0.2463 overall, 0.0777 on its last batch — normal, expected
  batch-to-batch noise, not divergence).
- **Latest validation loss/accuracy:** epoch 1 only so far —
  `val_loss=0.2341`, `val_accuracy=0.9114` (91.14% pairwise ranking
  accuracy on held-out data — a strong, healthy result for a first epoch,
  consistent with the model genuinely learning the length-ordering task
  rather than memorizing or collapsing).
- **Best checkpoint so far:** `results/pars_official/predictor_train/alpaca_gpt4_bert/best_model.pt`
  (saved after epoch 1, since epoch 1 is the only completed epoch and its
  validation accuracy became the new best by definition).
- **Current checkpoint files:** only `best_model.pt` exists yet (438,015,898
  bytes) — `last_model.pt` and `metrics.json` are written only once, at
  the very end of `main()` after all 3 epochs finish (verified by reading
  `train_pairwise_bert.py`'s control flow directly, not assumed).
- **Log path:** `results/pars_official/train_log_20260804T181052Z.log`
  (the live, current run's log; a second, empty
  `train_log_20260804T180356Z.log` is a harmless leftover from the first
  attempt, which was killed after an HF Hub `xet`-backend download stall
  before any training began — see §4).
- **Health assessment:** healthy. Loss is decreasing smoothly, GPU
  utilization is high and steady, no exceptions, no NaN/inf losses
  observed in the log, and the epoch-1 validation accuracy (91.14%) is
  well above chance (50%) and in a plausible range for this task —
  nothing suggests invalid results in progress.
- **Warnings observed (both already known, both benign, neither indicates
  a problem):** (1) the `BertModel LOAD REPORT` "UNEXPECTED" keys
  (`cls.predictions.*`, `cls.seq_relationship.*`) — expected, since
  `AutoModel` (not `AutoModelForPreTraining`) intentionally drops BERT's
  pretraining-only MLM/NSP heads; (2) none observed from this run itself
  (the "Token indices sequence length is longer than..." warning seen
  earlier in this session came from this project's own offline-scoring
  *smoke test* against a fake checkpoint, not from the official training
  script).

## 3. PARS integration audit

All independently verified (not merely asserted) via direct inspection of
the cloned repository, its git history, `gh api`, and this project's own
adapter code:

| Item | Verified value |
|---|---|
| Official repository URL | `https://github.com/SPEAR-UIC/PARS` (confirmed real/public via `gh api repos/SPEAR-UIC/PARS`) |
| Pinned official commit | `fd4e125b65bb73aef5eccafa79c2509434be61ec` (3 commits total in the repo's entire history) |
| Upstream license status | **NONE** — no `LICENSE`/`LICENSE.md`/`COPYING` anywhere in the full recursive git tree; GitHub API `license` field is `null`. Documented explicitly in `baselines/pars/PROVENANCE.md`, not hidden. Proceeding was an explicit, user-directed decision (2026-08-04) after this gap was surfaced. |
| Training dataset and license | `vicgalle/alpaca-gpt4`, **CC BY-NC 4.0** (non-commercial) — verified via the HF dataset card, not assumed. Chosen over the official repo's two more-permissively-licensed options (`theblackcat102/evol-codealpaca-v1`, Apache-2.0; `TIGER-Lab/MATH-plus`, MIT) for domain realism (general instruction-following text, closer to WildChat's general-chat domain than code- or math-only data); `lmsys/lmsys-chat-1m` was excluded per this project's pre-existing restrictive-license policy. |
| Dataset revision/hash | No pinned HF dataset revision was recorded at preprocessing time (the official `preprocess_alpaca_gpt4.py` script calls `load_dataset("vicgalle/alpaca-gpt4")` without a `revision=` argument) — **this is a real, confirmed gap**, not yet closed; see §4. |
| Preprocessing command | `python scripts/preprocess_alpaca_gpt4.py --output-dir .../alpaca_gpt4` (official, unmodified defaults: `similarity-threshold=0.2`, `train-num-pairs=10`, `val-num-pairs=10`, `test-size=0.1`, `seed=42`) — produced 46,801 train rows / 5,201 val rows / 468,010 train pairs / 52,010 val pairs. |
| Training command | `python scripts/train_pairwise_bert.py --train-file ... --val-file ... --output-dir ...` (official, unmodified defaults: `bert-base-uncased`, `max_length=128`, `batch_size=128`, `num_epochs=3`, `learning_rate=2e-5`, `margin=1.0`, `warmup_ratio=0.1`, `weight_decay=0.01`, `seed=42`) — currently running, see §2. |
| Model architecture | `PairwiseRanker`: `bert-base-uncased` encoder + `Linear(hidden_size=768, 1)` on pooler/`[CLS]` output. Confirmed by dynamically importing the class from the pinned clone and inspecting `handle.model.encoder.config`/`handle.model.fc`, not by re-reading source alone. |
| Ranking direction | Higher PARS score = longer predicted response (derived from the `MarginRankingLoss(score_A, score_B, target=+1 if len_A>len_B)` construction) → **ascending** score order = shortest-first scheduling priority. Verified against the opposite (descending) convention vLLM-LTR uses, via a direct cross-check test (`test_opposite_direction_from_vllm_ltr`). |
| Tokenizer | `bert-base-uncased`'s own `AutoTokenizer`, `max_length=128` passed explicitly in every tokenizer call in both the official code and this adapter — confirmed by reading `serve_predictor_score.py`/`train_pairwise_bert.py` directly. |
| Checkpoint-selection rule | Official script saves `best_model.pt` whenever the current epoch's validation accuracy exceeds the running best (`if val_accuracy > best_val_accuracy: save`), and unconditionally saves `last_model.pt` once at the very end — this project's evaluation uses `best_model.pt`. |
| Adapter implementation | `baselines/pars/adapter/{errors,provenance,checkpoint_loader,ranking_adapter,offline_scoring,simulator_policy}.py` — present, `py_compile`-clean, 22/22 unit tests passing. |
| Dynamic import behavior | `checkpoint_loader._import_pairwise_ranker_class` loads `PairwiseRanker` via `importlib.util.spec_from_file_location` directly from the pinned external clone at runtime — confirmed no `class PairwiseRanker` string appears anywhere under `baselines/pars/` (`grep`, empty result) and a dedicated regression test (`TestOfficialCodeReuseNotVendored`) locks this in. |
| No upstream source vendoring | Confirmed — the official clone lives at `/home/soroush/.cache/external_baselines/PARS`, entirely outside this repository's directory tree (`test_official_clone_path_is_outside_this_repo` asserts this structurally, not just as a convention). |
| Evaluation-only registration status | `SELECTOR_ELIGIBLE = False` in `simulator_policy.py`. |
| No selector/composition-library registration | Confirmed fresh (not just via the unit test): `pars_semantic_reference` is absent from `BASELINE_NAMES`, `SELECTOR_CANDIDATE_NAMES`, `POLICY_LIBRARY_V2_NAMES`, and `EXTERNAL_BASELINE_NAMES` (checked directly against the live registry objects). |
| No future-information leakage | `ObservableRequest` (the only state `select_action()` receives) has no `actual_output_tokens` field; `PARSSemanticReferencePolicy.select_action`'s source contains no reference to that field (structural, not just behavioral, guard — `test_policy_select_action_never_receives_actual_output_tokens`). |

### Classification

**Official-code reproduction with locally trained checkpoint.**

Not "official checkpoint execution" (no official checkpoint exists to
execute). Not "adapted official implementation" in the sense of modified
official logic (every official script ran completely unmodified — the
*adapter* layer around it, not the official code itself, is what's new,
exactly analogous to vLLM-LTR's own classification). Not a "faithful
independent reimplementation" (nothing was reimplemented — the actual
official `PairwiseRanker` class, training loop, and hyperparameters ran
verbatim). Not a "proxy" (no hand-written substitute heuristic stands in
for the model). Not "incomplete" (all 3 required scripts — preprocess,
train, and the score-extraction logic — have run/will run to completion
using official code).

## 4. Fidelity risks

**Confirmed deviations (real, disclosed, all already documented in
`PROVENANCE.md`/the implementation doc):**
- No official pretrained checkpoint exists — a checkpoint was trained
  locally instead of downloaded (the single largest fidelity risk: this
  evaluates "PARS's method, correctly trained," not "the authors' own
  released weights," since none exist).
- Training dataset substituted from the authors' own (unpublished, per
  the released repo) training data to one of the 4 officially-supported
  *alternative* dataset options (`vicgalle/alpaca-gpt4`) — the released
  repo never claims `alpaca-gpt4` is what the paper's own reported numbers
  used, only that it is one of 4 supported reproduction paths.
- Canonical-suite evaluation requires real-text substitution (nearest
  BERT-token-count WildChat-prompt matching) since those synthetic
  workloads carry no prompt text — a necessary, disclosed adaptation, not
  a hidden one.

**Unverified assumptions (real gaps, not yet closed — flagged, not
silently accepted):**
- **No HF dataset revision was pinned** when `vicgalle/alpaca-gpt4` was
  downloaded (§3) — unlike this project's own WildChat/vLLM-LTR-checkpoint
  provenance discipline (which always pins and records an exact revision
  SHA), the official PARS preprocessing script itself doesn't accept a
  `revision` argument, and this evaluation did not patch that in. If the
  dataset's `main` branch changes upstream, this run would not be
  byte-for-byte reproducible from the command alone — only from the
  locally-cached/serialized `train_data.json`/`val_data.json`/pairwise
  files this project already wrote to `results/pars_official/data_preprocess/alpaca_gpt4/`
  (gitignored, so those files are not preserved beyond this local run
  either). **Recommendation:** record the dataset's resolved commit SHA
  after the fact (`HfApi().dataset_info("vicgalle/alpaca-gpt4").sha`) into
  the provenance doc once training completes, even though the
  preprocessing script itself can't be told to pin it without patching
  official code (which this task's "reuse official code unmodified"
  principle argues against).
- The authors' own original training data/hyperparameter search (if any
  beyond the released defaults) is unknown — this evaluation trusts the
  released script's defaults are a faithful reproduction path, since no
  paper-reported hyperparameter table was cross-checked against the code
  defaults (out of scope for this task; the code defaults are what's
  actually released and runnable).

**Harmless environment differences:**
- torch 2.12.0+cu130 / transformers 5.8.1 in this environment vs.
  whatever the original authors used (not stated in the repo) — the
  official training script has no version pins in its `requirements.txt`
  (bare `torch`, `transformers`), so there is no specific version to
  match or diverge from.
- `HF_HUB_DISABLE_XET=1` was required to work around an unrelated HF Hub
  CDN transfer stall (confirmed by direct reproduction: the download
  resumed and completed normally once XET was disabled) — an
  infrastructure workaround, not a change to any PARS-relevant behavior.

**Potentially result-changing differences (the ones that matter most for
interpreting final numbers):**
- The locally-trained checkpoint's quality (91.14% val accuracy after
  epoch 1, final number pending) stands in for whatever quality the
  paper's own (unreleased) checkpoint achieves — any comparison of this
  evaluation's ANWG numbers against the *paper's own reported latency
  results* would be invalid (different training data, different metric
  entirely — ANWG vs. the paper's p50/p99 latency reduction); this
  evaluation only supports comparing *this trained instance of PARS's
  method* against this project's other policies, not against the paper's
  own headline numbers.
- Ranking-direction inference (§3) is derived from the loss function's
  mathematics, not from a pinned official scheduler-integration source
  (none is public) — low risk (the derivation is unambiguous algebra, not
  a judgment call), but structurally different in evidentiary strength
  from vLLM-LTR's literal pinned-source-excerpt reproduction.

## 5. Benchmark readiness

| Check | Status |
|---|---|
| WildChat control availability | ✅ `data/processed/wildchat/wildchat_eval_prompts_by_id.json` (300 prompts) + `wildchat_eval_sharegpt_shaped.json` present and already used successfully for the vLLM-LTR evaluation. |
| Seven canonical benchmark families | ✅ `benchmarks/canonical_suite/{staggered_heterogeneous,burst_independent_lengths,mixed_tight_deadlines,priority_vs_service_time_conflict,prediction_noise_regime,long_output_tail,burst_arrivals_isolated}/` all present with `seed_{0,1,2}.json` + `manifest.json` each. |
| Deterministic seeds | ✅ Seeds 0/1/2 fixed throughout; canonical-suite datasets are pre-generated, static, committed files (not regenerated per run) — confirmed via `git log` showing exactly one commit (`4972dd5`) ever touched `benchmarks/canonical_suite/`. |
| Raw request-level output support | ✅ `scripts/run_pars_first_comparative_evaluation.py` writes `request_level_outcomes.csv` per workload (same schema as the vLLM-LTR eval script, smoke-tested). |
| Bootstrap CI support | ✅ Reuses (imports, does not duplicate) `compute_bootstrap_ci` from the vLLM-LTR eval script. |
| Independent verification support | ✅ `scripts/verify_pars_comparison_results.py`, reuses (imports) the generic, workload-agnostic recomputation functions from `scripts/verify_vllm_ltr_comparison_results.py`; smoke-tested end-to-end with 0 mismatches after two real bugs (hardcoded cache paths, missing `--max-requests` propagation) were found and fixed during that smoke test. |
| Ranking-agreement analysis | ✅ `cross_check_pars_ranking_agreement` (PARS-vs-EST/SOF Spearman), independently reimplemented, not reused from the eval script's own computation. |
| vLLM-LTR comparison support | ✅ vLLM-LTR's own WildChat results are already complete, independently verified, and committed (`docs/audits/vllm_ltr_first_comparative_evaluation_20260804.md`) — directly comparable on the WildChat leg. **Not** comparable on the canonical-suite legs — vLLM-LTR was never evaluated there (it predates the canonical suite's existence); this scope limitation must be stated explicitly when the comparison is written up, not glossed over. |
| Identical workload reuse across policies | ✅ Every policy in a given (workload, seed) runs the identical `Request` list — verified by construction (`run_one_workload` builds `requests` once per seed, passes the same list to every policy). |
| No metric leakage | ✅ `ObservableRequest` carries no `actual_output_tokens`; independently re-verified in §3. |
| No benchmark tuning based on final results | ✅ Confirmed via `git log` (§7 below) — the canonical suite's family definitions and accept/reject decisions were finalized in a single commit, before any PARS-specific evaluation existed to tune toward. |

**Blockers before the full evaluation can run:** none structural. The
only blocker is **training not yet complete** (§2) — the real checkpoint
is required before real scores/results can be produced; no fabricated or
placeholder run should substitute for this.

## 6. Existing baseline inventory

| Baseline | Implementation status | Official-code status | Fidelity classification | Evaluation status | Canonical benchmark status | Foundational-library status |
|---|---|---|---|---|---|---|
| **vLLM (the serving framework itself)** | Not integrated as a runnable engine in this repo — this project's simulator is a discrete-event abstraction, not vLLM itself. | N/A | N/A | N/A | N/A | N/A |
| **vLLM-LTR** | Complete | Official (checkpoint + adapter, `baselines/vllm_ltr/`) | Official checkpoint execution (real, hash-verified, architecturally verified pretrained checkpoint) | Complete, independently verified — WildChat only | Not run (predates the canonical suite) | EVALUATION_ONLY (explicit classification in `docs/audits/vllm_ltr_first_comparative_evaluation_20260804.md`) |
| **PARS** | In progress (training running, §2) | Official-code reproduction with locally trained checkpoint (§3) — no official checkpoint exists | Official-code reproduction with locally trained checkpoint | Not yet run (blocked on training completion) | Not yet run | Not yet determined |
| **Sarathi-Serve** | Implemented as `sarathi_style` | Style/inspired — explicitly NOT an official reproduction (no official code integrated) | Proxy/inspired | Part of this repo's standard internal policy comparisons | N/A (internal policy, always available) | Foundational (in `BASELINE_NAMES`/policy library) |
| **DistServe** | Referenced (`docs/distserve_faithful_scheduler_reference.md` exists) | None (no external-baseline table entry; no official code integrated per `docs/external_baseline_coverage_report.md` line 486) | Not applicable — not integrated as an external baseline | Not evaluated as an external baseline | N/A | Not applicable |
| **Llumnix** | Referenced (`docs/llumnix_faithful_scheduler_reference.md` exists) | Not confirmed as an integrated official-code baseline in this audit's scope (reference doc exists; full status not re-verified here — out of scope for a PARS-focused audit) | Unverified in this pass | Unverified in this pass | N/A | Unverified in this pass |
| **SLAI/RAD** | Referenced (`docs/slai_faithful_scheduler_reference.md` exists) | Not confirmed as integrated official code in this audit's scope | Unverified in this pass | Unverified in this pass | N/A | Unverified in this pass |
| **VTC** | Not started | N/A | N/A | N/A | N/A | N/A — explicitly **not** to be started per this and prior tasks' standing instructions |
| **JITServe** | Not implemented (`docs/external_baseline_coverage_report.md` line 487: "Tempo / JITServe — None") | None | N/A | N/A | N/A | N/A |
| **Apt-Serve** | Not implemented (same table, line 488: "None") | None | N/A | N/A | N/A | N/A |
| **HyGen** | Not found anywhere in this repo's docs or source (grep across `docs/*.md` and `src/llmserveopt/policies/*.py` returned no matches) | None | N/A | N/A | N/A | N/A |

**Naming-collision finding (real, worth flagging explicitly, not a
fabrication):** this repo's own `docs/external_baseline_coverage_report.md`
(line 320, from earlier Phase 2A.3B work) already used the label "PARS" to
refer to **Zheng et al.'s NeurIPS 2023 "Response Length Perception and
Sequence Scheduling"** paper (approximated internally as
`estimated_service_time_first`, itself explicitly labeled "NOT a
reproduction of PARS"). That is a **different paper** from the one this
task's "PARS" refers to (Tao et al., ISC 2026, arXiv:2510.03243, official
repo `SPEAR-UIC/PARS`) — the two happen to share the same informal
abbreviation. This is a pre-existing internal-naming ambiguity, not
something introduced by this integration, but it should be disambiguated
in any future write-up that references "PARS" without a full citation
(recommend: rename the older internal usage to its full paper title or a
distinct shorthand like "RLP-2023" in future edits, since `PARS` as an
unqualified term now collides).

## 7. Canonical benchmark audit

Verified against `benchmarks/canonical_suite/suite_manifest.json` and
`git log`:

- **9 synthetic families + WildChat control:** confirmed
  (`suite_manifest.json["families"]` lists exactly 9 names; `control_workload`
  field names `wildchat_control` explicitly, pointing at the real,
  un-regenerated WildChat data).
- **7 accepted, 2 rejected:** confirmed (`accepted`/`rejected` arrays sum
  to 9, no overlap).
- **Deterministic generation:** confirmed — `generate_workload()` is seed-
  pure (verified by this project's own `tests/test_canonical_benchmark_suite_generator.py`,
  7/7 passing at the time of that commit) and the committed
  `seed_{0,1,2}.json` files are static artifacts, not regenerated per
  evaluation run.
- **Headroom gate:** confirmed present and documented
  (`docs/audits/canonical_benchmark_suite_design_20260804.md`'s
  acceptance-threshold table — 4 checks, all independently justified).
- **Accepted workload manifests:** all 7 accepted families' `manifest.json`
  files present, each recording the family's own GPU/service-model
  overrides, validation record, and per-seed dataset SHA-256 hashes.
- **Foundational-policy comparison artifacts:** `benchmarks/canonical_suite/foundational_comparison.json`
  and `diversity_analysis.json` present (from the suite's own §7
  characterization pass — fifo/edf/est/sof/wsp/scorpio/oracle_srtf only,
  no learned selector, no PARS, no vLLM-LTR — confirming that comparison
  predates and is independent of this PARS integration).
- **No accidental benchmark bias toward PARS or the project's own
  method:** confirmed — `git log --oneline --all -- benchmarks/canonical_suite/
  configs/workload_headroom_candidates/ scripts/generate_canonical_benchmark_suite.py
  scripts/check_ordering_workload_headroom.py` returns exactly **one**
  commit (`4972dd5`) across all of those paths, on **all** branches/refs
  (`--all` flag). The suite's family definitions, acceptance thresholds,
  and accept/reject decisions were authored and finalized entirely before
  any PARS-specific code existed in this repository (PARS work began
  fresh in this session, after `4972dd5` was already committed and
  pushed) — there is no commit history showing the benchmark being
  retuned after seeing PARS (or any other candidate's) results.
- **No benchmark definitions modified after acceptance:** confirmed by
  the same single-commit finding above — nothing has touched these paths
  since.

## 8. Contextual-composition roadmap consistency

- **CC5 status:** `docs/contextual_composition_roadmap.md` (the
  authoritative, actively-maintained roadmap for this branch, per its own
  banner) correctly shows CC5 = `COMPLETE` (`COMPLETE_REGIME_SPECIFIC`),
  with the full statistical detail (beats best-fixed and hard-selector at
  p<0.0001/p=0.021; edge over `best_global_composition` not
  statistically distinguishable from zero, p=0.5654) preserved
  accurately.
- **CC6 status:** correctly shown as `NEXT`/queued, explicitly
  **RESTRICTED** to the CC5 trusted envelope (`burst_transition`,
  `kv_pressure`, `long_output`, `prediction_noise`, `saturated`,
  `selective_admission_trap`, `underloaded`), not yet started.
- **Resume document:** `docs/RESUME_CONTEXTUAL_COMPOSITION.md` is
  consistent with the authoritative roadmap (correctly directs to the CC5
  final-operating-envelope report as "current status").
- **Decision log / issue states:** GitHub issue #5 (CC5) is referenced as
  closed in the roadmap table; issue #6 (CC6) is open/queued — consistent
  with the roadmap's own text, not independently re-queried against the
  live GitHub issue tracker in this pass (would require an API call
  outside this audit's read-only-repo scope).
- **Stale/contradictory documentation found (real, pre-existing, NOT
  introduced by this session's PARS or benchmark-suite work):**
  `docs/roadmap.md`'s own top banner (lines 3-11, dated 2026-08-03) still
  reads *"CC5 is `IN PROGRESS`* (targeted dataset expansion + rerun
  underway)"* and points at the CC4b/CC5-retry report as "current status"
  — this is **stale**. The banner itself already correctly disclaims that
  `contextual_composition_roadmap.md` (not this file) is the active
  roadmap and that "the document below remains a historical... roadmap...
  do not treat it as the current roadmap," which somewhat mitigates the
  staleness (a reader is directed away from the stale part), but the
  banner's own status line was not updated when CC5 actually closed. This
  predates the current session's work (this session's own edits to
  `docs/roadmap.md` were confined to adding the vLLM-LTR result item much
  further down the file, around line 46-65 — verified by diff — and did
  not touch this banner). **Recommendation, not executed in this
  audit-only pass:** update `docs/roadmap.md`'s banner to say CC5 is
  `COMPLETE (COMPLETE_REGIME_SPECIFIC)` and CC6 is `NEXT (RESTRICTED)`,
  matching `contextual_composition_roadmap.md`.
- **No corruption from the baseline-integration work:** neither the
  vLLM-LTR work, the ordering-headroom audit, the canonical benchmark
  suite, nor the in-progress PARS work has touched any CC5/CC6-related
  file, config, or result — confirmed via `git log --oneline -- docs/contextual_composition_roadmap.md
  docs/RESUME_CONTEXTUAL_COMPOSITION.md` showing no commits from today's
  session touching those paths.

## 9. Test and code-health audit

```
$ python -m compileall -q src scripts tests
(clean, exit 0)

$ python scripts/check_contextual_composition_status.py
contextual composition status check passed

$ python scripts/check_contextual_composition_status.py --resume-readiness
ERROR: working tree is not clean
(exit 1 -- EXPECTED: caused entirely by the untracked PARS work-in-progress
files listed in §1, which are intentionally not yet committed. Not a real
failure; will resolve once PARS work is committed per the standing plan.)

$ pytest --collect-only -q
3319 tests collected, 0 collection errors
```

## Summary of findings requiring attention (none are blockers to continuing PARS)

1. No HF dataset revision was pinned for `vicgalle/alpaca-gpt4` (§4) —
   recommend recording the resolved commit SHA post-hoc once training
   completes.
2. Pre-existing internal-naming collision: two different papers have both
   been informally called "PARS" in this repo's history (§6) — recommend
   disambiguating in future docs, not urgent.
3. Pre-existing stale banner in `docs/roadmap.md` (§8) — recommend a
   one-line update, not urgent, not caused by this session's work.
4. `resume-readiness` check fails only due to intentionally-uncommitted
   PARS work-in-progress (§9) — expected, will resolve on commit.

**No blockers to continuing the PARS evaluation once training completes.**
