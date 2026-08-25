# KV v2 Reproducibility Forensic Audit

**Date:** 2026-08-17
**Verdict:** `REPRODUCIBILITY_GAP_BOUNDED`
**Trigger:** discovered as a side effect of cross-checking the KV-aware composition
falsification v1 against the frozen KV v2 pairwise-separation pilot.
**Scope:** diagnosis only. No historical CSV, verdict, policy, child, or
simulator semantics were modified.

## A. Historical frozen-run identity

- Run path: `experiments/kv_pressure_pilot_v2_20260817T165053Z/`
- Launch/commit SHA: `6be526e` (the commit that added the run directory,
  `templates_kv_pressure_v2.py`, and the extended runner together)
- Config: `configs/kv_pressure_pilot_v2.yaml` (grid: `bulk_pressure`
  low/high, `urgent_arrival_phase` early/middle/late, `urgent_tightness`
  loose/tight, seeds `20260910-15`, `max_kv_tokens=6000`)
- Runner/template: `scripts/run_policy_separation_kv_pressure_pilot_v1.py
  --template-version v2` → `case_kv_pressure_reserve_contention_v2`
- Historically recorded dataset information: **none**. `final_summary.json`
  and `run.log` for this run record only `template_version`, scenario/task
  counts, `max_kv_tokens`, policy names, and `held_out_seeds` — no dataset
  path, no dataset checksum, no Python/library versions, no exact command
  line (in particular, whether `--datasets-root` or `--allow-synthetic-tokens`
  was passed is **not recoverable** from anything committed).
- Historical CSV SHA-256: `59ef1d3276b01427a20bb940504f47dcd460c106332267d8c5dcf4ed39267a7a`

## B. Current reproduction identity

- HEAD: `855dd12`
- Exact reproduction command:
  ```
  python scripts/run_policy_separation_kv_pressure_pilot_v1.py \
    --config configs/kv_pressure_pilot_v2.yaml \
    --run-dir <scratch> \
    --template-version v2 --workers 1 --datasets-root .local_data
  ```
- Config: identical file, unchanged since `6be526e` (verified, §D)
- Runner/template: identical files, unchanged since `6be526e` (verified, §D)
- Resolved dataset path: `.local_data/burstgpt_v2/raw/BurstGPT_without_fails_1.csv`
  (via `_get_staged_burstgpt_path`'s `{datasets_root}/burstgpt_v2/raw/`
  resolution; no `LLM_SERVEOPT_BURSTGPT_CSV` env override was set)
- Fresh reproduction CSV SHA-256: `e6303d854b2e4f158cd388a85a86833e1902a2b717f3e4c8a9edf894f3285bf2`

## C. Mismatch characterization

Merged on `(scenario_id, policy_name)`, all 144 historical rows matched to
144 current rows (no missing keys either direction):

| Metric | Value |
|---|---|
| Exact match (`\|ΔANWG\| < 1e-12`) | 45/144 |
| Floating-point-only difference (`1e-12 ≤ \|ΔANWG\| < 1e-6`) | 0/144 |
| Real difference (`\|ΔANWG\| ≥ 1e-6`) | 99/144 |
| Max `\|ΔANWG\|` | 0.250000 |
| Mean `\|ΔANWG\|` over the 99 mismatches | 0.092810 |
| `bulk_n` / `urgent_n` differences | 0/144 (population **counts** match exactly — these are deterministic, not BurstGPT-content-dependent) |
| `completion_fraction` differences | 0/144 |
| `n_steps` differences | 144/144 (max Δ = 2119 steps) |
| `peak_kv_utilization` differences | 144/144 (max Δ = 0.1898) |
| Every row's `status` | `success` in both (0 status differences, 0 failures either side) |

**This is scientifically material, not merely bit-level.** Practical winner
at ε=0.01 (`kv_constrained_online` vs `least_laxity_first`) changes on
**17/72 (24%) scenarios**. Aggregate winner distribution over all 72
scenarios: historical `{kv: 45, tie: 22, llf: 5}`; current `{kv: 52, tie:
20, llf: 0}` — **`least_laxity_first` wins zero scenarios outright in the
current environment**, versus 5 historically. Because population counts
(`bulk_n`/`urgent_n`) and `completion_fraction` are identical while
step-count/KV-occupancy-trajectory metrics differ substantially, the
divergence is localized to **sampled prompt/output token values**
(BurstGPT-derived), not to request counts, arrival mechanics, or admission
outcomes downstream of a fixed workload.

**Effect on historical gates:** the frozen KV v2 audit's own gates were
computed and frozen using the historical CSV as committed — that CSV is
unchanged and its gate computations remain as documented. This forensic
audit does not recompute or challenge those gate values; it only
establishes that the historical CSV **cannot currently be reproduced from
scratch** with the evidence available (§I).

## D. Code provenance

`git diff 6be526e HEAD` restricted to every file on the KV v2 execution
path — `templates_kv_pressure_v2.py`, `templates_kv_pressure.py`,
`templates_prefill_decode.py` (shared BurstGPT loading/sampling helpers),
`templates_fairness_starvation.py`/`_v2.py` (shared path resolution),
`kv_constrained_online.py`, `least_laxity_first.py`,
`policy_library_v2_helpers.py`, `scoring.py`, the entire
`src/llmserveopt/simulator/` package, and
`scripts/run_policy_separation_kv_pressure_pilot_v1.py` — returns **zero
changed lines**. Inspected scope: every module reachable from scenario
generation through simulation to CSV output for this pilot. Within this
scope, code drift is **not** a demonstrated explanation. This does not
prove no code anywhere in the repository changed in a way that could
matter (dependency code outside the repo is a separate question, §H), only
that the KV v2 pilot's own tracked execution path is byte-identical to what
was committed alongside the historical run.

## E. Determinism

Two independent current full reruns (`--workers 1`, identical command,
identical `.local_data` dataset) produced **byte-identical** output files:
SHA-256 `e6303d854b2e4f158cd388a85a86833e1902a2b717f3e4c8a9edf894f3285bf2`
on both. A third rerun at `--workers 4` (default) produced the same 99/144
mismatch count against history as `--workers 1` (§F). **Ordinary runtime
nondeterminism is not supported as the explanation** — the current
pipeline is fully deterministic given fixed code+data+config.

## F. Multiprocessing

`--workers 1` and `--workers 4` reruns against the historical CSV both
show the same 99/144 mismatch count and the same max `|ΔANWG|`. Since
worker count has zero effect on the size or character of the
historical-vs-current gap, **multiprocessing/task-scheduling order is
ruled out** as the explanation.

## G. Dataset forensic analysis

Two real candidate BurstGPT files exist on this machine (no others found
by a repository-wide + common-path search):

| Candidate | Resolved path | SHA-256 | Rows | Size | mtime |
|---|---|---|---|---|---|
| A | `.local_data/burstgpt_v2/raw/BurstGPT_without_fails_1.csv` | `a4d068a7113ec0290e74063a1b3447dc6001a30e4298eb313581b71006dda1f4` | 1,404,295 | 51,429,517 B | 2026-08-16 18:16 |
| B | `data/raw/burstgpt/BurstGPT_1.csv` | `46fc9480ef0b748ecb2b51d512ff08c196b031782cbe6f78e28044d768e86d5a` | 1,429,738 | 50,853,373 B | 2026-06-13 18:11 |

Both were tested as the resolved dataset (B via `LLM_SERVEOPT_BURSTGPT_CSV`
override, which takes resolution priority): **neither reproduces the
historical frozen CSV** (candidate A: 99/144 mismatch, as reported
throughout; candidate B: 104/144 mismatch — slightly worse, not better).

`_load_burstgpt_arrays` reads only the first `nrows=30000` rows of
whichever file is resolved. Comparing the two candidates' resulting pool
arrays over those first 30,000 rows: the first 5 `prompts`/`outputs`
values are **identical** between A and B, and the filtered
`[1024, 3072)` prompt-window pool length used by
`templates_kv_pressure_v2`'s bulk-tenant sampling is **7335 (A) vs 7337
(B)** — nearly but not exactly identical. **Simple alternate-file
substitution between these two specific candidates is therefore not a
demonstrated smoking gun** — both are close to each other but neither
matches history, and the pipeline's sensitivity to even a 2-row pool-length
difference (via `rng.choice(real, size=count)`, whose output depends on
`len(real)`) means a small, currently-unidentified perturbation of this
kind remains a live, evidence-compatible hypothesis (§H) even though it
was not pinned down to either available file.

## H. Remaining plausible source classes

Evidence-compatible, not proven:

1. **Unavailable historical exact data bytes/staging state** — the
   BurstGPT file actually resolved at historical run time may have been a
   third variant (different staging, different row order/filter epoch)
   that no longer exists on this machine and was never checksummed.
2. **Unrecorded historical preprocessing/staging state** — `.local_data`
   could have been re-staged between the historical run and now without
   the file's own mtime necessarily reflecting every intermediate state
   (only current mtime is observable; no historical mtime was recorded at
   run time either).
3. **Unrecorded dependency/environment state** — no Python/numpy/pandas/
   sklearn version was recorded for the historical run; `pd.read_csv` /
   `pd.to_numeric(...).dropna()` behavior inside `_load_burstgpt_arrays`
   is capable in principle of differing across pandas versions in ways
   that would shift the filtered pool length even from byte-identical file
   content, which (per §G) is sufficient to fully explain this
   magnitude of divergence.
4. **Unrecorded generated-input/sample state** — no intermediate sampled
   array (the actual `prompt_tokens`/`predicted_output_tokens` per request)
   was preserved for the historical run, so the exact sampled values used
   cannot be directly compared, only inferred from downstream metrics.

No other candidate class is evidence-supported; this list is not extended
further.

## I. Scientific impact — kept as three separate questions

1. **Exact reproducibility of historical KV v2:** weakened. The historical
   `experiments/kv_pressure_pilot_v2_20260817T165053Z/per_policy_results.csv`
   cannot currently be regenerated bit-for-bit, or even within practical
   tolerance (max `|ΔANWG|`=0.25, 24% of scenarios change practical
   winner) from the same committed code and config.
2. **Internal validity of the KV-aware composition falsification v1:**
   **not weakened.** Every method compared in that experiment — both
   parents, the child, the selector, the hard-conditional rule, the oracle
   — was evaluated in one single run, under one single current
   environment and one single dataset snapshot (`.local_data`, SHA-256
   `a4d068a7...`). The gap identified here is a *cross-run* comparison
   problem, not a within-run inconsistency; the composition falsification's
   own preregistered gates (G1-G8) remain validly computed from internally
   self-consistent data.
3. **Cross-run comparison (historical KV v2 ↔ current composition-era
   environment):** requires caution. Any statement of the form "the
   composition run's parent scores match/exceed what v2 reported" should
   not be made without re-deriving both sides from the same environment —
   this audit's own composition falsification report already avoided this
   by computing its own parent baselines fresh (per point 2) rather than
   citing v2's historical numbers directly.

## J. Verdict

**`REPRODUCIBILITY_GAP_BOUNDED`** — exact root cause has not been
demonstrated (§H lists remaining, evidence-compatible classes without
adjudicating among them); the major *ordinary* explanations (code drift,
runtime/multiprocessing nondeterminism, either of the two locally
available BurstGPT files) have been tested directly and ruled out or
narrowed; the current environment is demonstrably self-reproducible
(§E, byte-identical SHA-256 across independent reruns); historical
reconstruction remains incomplete. **Not** `REPRODUCIBILITY_GAP_RESOLVED`.

No historical CSV, verdict, or audit conclusion is rewritten by this
document. The historical KV v2 pairwise-separation pilot's own frozen
result and `KV_FAMILY_COMPOSITION_READY` verdict stand as originally
recorded, with this document added as a standing provenance caveat.
