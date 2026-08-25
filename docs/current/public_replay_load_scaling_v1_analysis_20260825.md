# Public Replay Load Scaling v1 — Analysis (HALTED: material integrity failure found)

**Date:** 2026-08-25
**Experiment:** `experiments/public_replay_load_scaling_v1/`
**Design (frozen):** `docs/design/PUBLIC_REPLAY_LOAD_SCALING_V1.md`
**SLURM job:** array `1195488`, 60/60 tasks `COMPLETED`, exit `0:0` for every task

**Bottom line up front:** the SLURM run completed cleanly and the data copied back
byte-for-byte intact, but the experiment's own **mandatory λ=1 reproduction check
(design §9, item 1) fails materially**: 66/360 (18.3%) of λ=1 P6 cells deviate from the
required `ANWG=1.0, completion=1.0` by up to **0.98 absolute** — nowhere close to the
required `1e-6` tolerance. Root cause identified precisely (below): a hardcoded simulator
step-budget in the new harness that is absent from the authoritative prior harness. Per the
frozen design's own rule and the explicit operating instructions for this task, **analysis is
STOPPED before interpreting the load curve (sections F–O below were not performed).** No new
SLURM job was submitted. No frozen artifacts were modified.

---

## A. Preflight

| Item | Value |
|---|---|
| Hostname | `al-khwarizmi` |
| Branch | `contextual-compositional-heuristics-20260731` |
| HEAD | `2987b7181efa2bc550d8a894c537eca8f6393eb6` |
| Upstream | `origin/contextual-compositional-heuristics-20260731`, ahead 2 |
| Git status | 168 pre-existing modified/untracked entries (large in-flight release + research batch, unrelated to this task); no files touched before this session |
| Scientific processes | none running |
| tmux sessions | none |

### Wulver status (one read-only check, control-master path)

```
ssh -S ~/.ssh/cm/wulver.sock sv96@wulver.njit.edu
export PATH=/apps/slurm/current/bin:$PATH
sacct -j 1195488 --format=JobID,JobName,State,ExitCode,Elapsed,Start,End -X
```

- **60/60 array tasks `COMPLETED`**, `ExitCode=0:0` for all 60
- No `FAILED`/`CANCELLED`/`TIMEOUT`/`OOM` states anywhere
- Elapsed times ranged ~17s–2m42s per task (heterogeneous window sizes/policies)
- Remote experiment directory confirmed to exist:
  `/mmfs1/scratch/ikoutis/sv96/llm-serving-heuristic-evolution/public_replay_load_scaling_v1/experiments/public_replay_load_scaling_v1/`
- Not polled again after this single check. Job 1195488 was not touched, cancelled, or resubmitted.

---

## B. Remote completeness audit

Verified remotely (via ssh, read-only `ls`/`wc`/`du`) before copy-back:

| Check | Result |
|---|---:|
| `cells_window_*.jsonl` files | **60** |
| `provenance_task_*.json` files | **60** |
| Total data rows across all cell files | **3840** (matches `60×8×8`) |
| Per-file row count | **64 each** (60/60 files) |
| Source distribution (from filenames) | **20 burstgpt / 20 azure_2023_conv / 20 azure_2023_code** — exact 20/20/20 |
| Tracebacks/errors in `logs/*.err`, `*.out` | **0** |
| Total remote output size | 16 MB (cells+provenance) + 152 KB (SLURM logs) |

Locally, after copy-back (see §C), the following were additionally verified against the full
3840-row dataset:

| Check | Result |
|---|---:|
| Duplicate `(canonical_scenario_id, load_factor_lambda, policy_id)` keys | **0** (3840 unique keys) |
| NaN values across all numeric fields, all rows | **0** |
| `status` field values | **3840/3840 `"success"`**, 0 explicit errors |
| Load factors present | exactly `{1,2,4,8,16,32,64,128}` |
| Policy IDs present | exactly the 8 frozen Pext IDs |
| GPU capacity tuples seen | exactly one: `(512, 512, 8000000)` at every cell |
| `n_requests_base` / `n_requests_scaled` | **200 at every one of 3840 cells** |
| Accidental duplicate task execution | none (60 provenance files, 60 distinct `task_index` values 0–59, 60 distinct cell files) |

All structural/schema-level checks pass. **The problem found below is a scientific-content
defect, not a structural/serialization one** — which is precisely why it survives the "no
NaN / no error / no duplicate" checks and requires the mandatory λ=1 reproduction check to
surface it.

---

## C. Copy-back integrity

Copied via `rsync` over the verified control-master SSH path
(`ssh -S ~/.ssh/cm/wulver.sock`) into `experiments/public_replay_load_scaling_v1/`
(pre-existing `DESIGN_FROZEN.md` and `smoke/` preserved; no unrelated repository files
touched). Remote files were **not deleted**.

**Copy-back manifest:** `experiments/public_replay_load_scaling_v1/copyback_manifest.json`
— 121 files (60 cell files + 60 provenance files + `DESIGN_FROZEN.md`), SHA256 computed
**remotely first** (`sha256sum` on the login node) and then re-verified **locally** after
transfer:

| Check | Result |
|---|---:|
| Files in manifest | 121 |
| SHA256 mismatches (remote vs. local, post-transfer) | **0 / 121** |
| Expected cell count | 3840 |
| Actual cell count (local, post-transfer) | **3840** |

`experiments/public_replay_load_scaling_v1/remote_sha256.txt` (remote-computed hashes) and
`remote_logs/` (all 120 SLURM `.out`/`.err` files) were also copied for the permanent record.
**No remote files were deleted or modified.**

---

## D. Frozen-design integrity

Cross-checked `docs/design/PUBLIC_REPLAY_LOAD_SCALING_V1.md` and
`experiments/public_replay_load_scaling_v1/DESIGN_FROZEN.md` (byte-identical to the design
doc, confirmed) against the completed run and the harness source
(`src/llmserveopt/policy_separation/public_replay_load_scaling_v1.py`):

| Frozen requirement | Verified |
|---|---|
| Exactly 60 windows | ✅ (§B) |
| Exact 20/20/20 source split | ✅ (§B) |
| Exact load grid `{1,2,4,8,16,32,64,128}` | ✅ (§B) |
| Exact 8-policy Pext set | ✅ (§B) |
| Fixed capacity (512/512/8,000,000) at every λ | ✅ (§B, single tuple observed) |
| Only inter-arrival times scaled | ✅ by code inspection: `transform_arrival_only()` uses `dataclasses.replace(r, arrival_time=..., slo_deadline=...)` — every other `Request` field (`prompt_tokens`, `actual_output_tokens`, `predicted_output_tokens`, `class_id`, `priority`, `request_id`) is passed through untouched by construction |
| Request counts unchanged | ✅ `n_requests_base == n_requests_scaled == 200` at all 3840 cells |
| Request ordering unchanged | ✅ by construction (the transform is a strictly increasing affine map of `arrival_time` for `lambda>0`); re-verified by the 20/20 passing local unit tests in `tests/test_public_replay_load_scaling_v1.py` |
| Input/output token lengths unchanged | ✅ by code inspection (not touched by `replace()`) |
| Deadline/SLO slack convention unchanged | ✅ by code inspection: `slack = max(0, deadline - arrival); new_deadline = new_arrival + slack` exactly matches design §3 |
| ANWG unchanged | ✅ `metrics.arrival_normalized_weighted_goodput` read directly from `llmserveopt.core.metrics`, no redefinition in this experiment's code |
| Epsilon definition unchanged | not exercised yet (portfolio-level analysis not reached — see halt below); no redefinition found in code |
| Local unit tests | **20/20 passed** (`tests/test_public_replay_load_scaling_v1.py`) — covers transform correctness, matrix completeness/uniqueness, and slack/ordering invariants, but **does not exercise full-duration simulation-to-completion**, which is exactly where the defect below lives |

**Everything the design explicitly promises about the *transform* is upheld exactly.** The
defect found in §E is in the **simulator invocation parameters** used by `evaluate_cell()`
(engineering harness code), not in the transform itself and not in the ANWG/epsilon
definitions.

---

## E. λ=1 reproduction — MATERIAL FAILURE (mandatory stop-gate triggered)

Compared every λ=1 cell against the authoritative prior public replay
(`experiments/public_trace_replay_v1/layer3_checkpoint.jsonl`), which recorded
`ANWG=1.0, num_completed=200/200` for all 6 P6 policies on all 60 windows.

| Quantity | Value |
|---|---:|
| P6 λ=1 cells checked | 360 (60 windows × 6 P6 policies) |
| Matches (`|ANWG−1|≤1e-6` and `|completion−1|≤1e-6`) | **294 / 360 (81.7%)** |
| **Failures** | **66 / 360 (18.3%)** |
| Max ANWG absolute error from 1.0 | **0.98** |
| Max completion-fraction absolute error from 1.0 | **0.98** |
| Distinct windows failing | **11 / 60** (10 BurstGPT, 1 Azure-code; 0 Azure-conversation) |
| Required tolerance | `1e-6` |

Example failing windows (ANWG at λ=1, identical across all 6 P6 policies + VTC within each
window — ruling out a policy-specific bug):

| Window | ANWG @ λ=1 (should be 1.0) |
|---|---:|
| `burstgpt::w0` | **0.02** |
| `burstgpt::w4563` | 0.025 |
| `burstgpt::w1404` | 0.085 |
| `burstgpt::w5616` | 0.12 |
| `burstgpt::w5967` | 0.395 |
| `burstgpt::w4914` | 0.595 |
| `burstgpt::w3861` | 0.66 |
| `burstgpt::w1755` | 0.735 |
| `burstgpt::w351` | 0.785 |
| `azure_2023_code::w4` | 0.84 |
| `burstgpt::w6669` | 0.92 |

Every affected window shows **request counts other than completed+dropped < 200** — i.e.
some fraction of the 200 requests never even reach a terminal state before the simulation
stops (e.g. `burstgpt::w0`: only 4/200 requests complete, 0 dropped, 196 simply never
processed).

### Root cause (identified precisely)

`evaluate_cell()` in `src/llmserveopt/policy_separation/public_replay_load_scaling_v1.py`
(lines 104–105, 317–323) hardcodes:

```python
SIM_MAX_STEPS = 200_000
SIM_DRAIN_STEPS = 50_000
...
sim = Simulator(SimulatorConfig(..., max_steps=SIM_MAX_STEPS, drain_steps=SIM_DRAIN_STEPS))
```

The base scenario's `service_model_kwargs["step_size"] = 0.001` (inherited unchanged from
`public_trace_replay_v1`, confirmed identical at every λ). `200_000` steps at `step_size=0.001`
gives an **effective simulated-wall-clock ceiling of ≈200–210 seconds — independent of λ**.

By contrast, the **authoritative** prior harness
(`public_trace_replay_v1.evaluate_scenario_policy`, line ~327) constructs
`SimulatorConfig(gpu_configs=..., service_model=...)` **without any `max_steps`/`drain_steps`
override**, so it uses `SimulatorConfig`'s own default (`max_steps: Optional[int] = None`,
i.e. **unbounded** — `src/llmserveopt/simulator/simulator.py:53`). That harness runs each
scenario to natural completion (all arrivals processed + idle-drain), which is exactly why
the frozen `layer3_checkpoint.jsonl` shows `ANWG=1.0, 200/200 completed` for every one of
these 11 windows.

`burstgpt::w0`'s real arrival span is **34,064 seconds (~9.5 hours)**; at λ=1 only requests
arriving in the first ~200s of that span (4–5 of 200) are ever admitted before the simulator
halts. This is **not** a policy-specific bug — all 7 policies checked at λ=1 give
bit-identical ANWG/completion per affected window, consistent with a shared harness ceiling.

**Critically, this is not confined to λ=1.** The ceiling is a *fixed real-time budget*
(~200–210s), not a step count that scales with λ. As λ grows, the *scaled* span
(`original_span / λ`) shrinks, so different windows clear the ceiling at different λ
thresholds — but `burstgpt::w0` (span 34,064s) requires `λ > ~170` to fit under ~200s, which
is **above the entire preregistered grid (max 128)**. Quantified across all 3840 cells:

| λ | cells incomplete / 480 | % |
|---:|---:|---:|
| 1 | 88 | 18.3% |
| 2 | 48 | 10.0% |
| 4 | 40 | 8.3% |
| 8 | 32 | 6.7% |
| 16 | 24 | 5.0% |
| 32 | 8 | 1.7% |
| 64 | 8 | 1.7% |
| 128 | 8 | 1.7% |

**256 / 3840 cells (6.67%) across the *entire* preregistered grid are silently truncated.**
One window (`burstgpt::w0`) is corrupted at **every single preregistered λ**, including the
top of the grid. All 256 affected cells carry `status: "success"` with no NaN and no
recorded error — the corruption is invisible to design checks §9/§10 as literally worded
(no crash, no NaN) and is only caught by the mandatory λ=1 ANWG/completion reproduction check.

**Per the frozen design (§9): "If check 1 fails materially, the run stops before
interpreting higher lambda values."** This check fails materially (18.3% of λ=1 P6 cells,
up to 0.98 absolute error). **Per the operating instructions for this task, analysis is
therefore STOPPED here. Sections F–O below were not performed.**

---

## F–O. Load curve, operating regimes, headroom, ranking diversity, P6-vs-external,
## source-stratified analysis, pressure/separation relation, prior-claim diagnosis,
## stressed-public comparison, uncertainty — **NOT PERFORMED (halted per §E)**

Interpreting the load curve on data where 6.67% of all cells (concentrated non-uniformly:
18.3% at λ=1 down to 1.7% at λ=128, and 100% of one window's cells at *every* λ) are silently
truncated would bias exactly the question this experiment is designed to answer: "does
increasing load reveal scheduler separation?" A truncated cell's low ANWG/completion is an
artifact of the harness's simulated-time ceiling, not evidence about scheduler behavior under
load — and because the artifact's prevalence *itself* falls monotonically as λ increases
(88→8 cells), any observed "improvement" in completion/ANWG with increasing λ would be
**partially or wholly a mechanical consequence of fewer truncation artifacts**, not a
scientific signal about load-induced scheduler separation. Reporting a load curve, headroom
values, winner counts, or source-stratified results computed over this data would either (a)
require silently excluding a non-preregistered subset of cells — which is exactly the kind of
post-hoc, outcome-dependent redefinition the frozen design and operating instructions
prohibit — or (b) present contaminated numbers as if they answered the reviewer's question.
Neither is acceptable, so no such analysis was produced.

Restoring interpretability requires **the harness be fixed and the data regenerated**
(remediation described above) — a decision reserved for explicit authorization, not taken
unilaterally in this session.

---

## P. Preregistered verdict

Applying design §10 exactly:

- `PUBLIC_LOAD_SCALING_REVEALS_POLICY_SEPARATION` — cannot be assessed; would require valid
  higher-λ interpretation, which is blocked.
- `PUBLIC_LOAD_SCALING_REMAINS_NONDISCRIMINATIVE` — cannot be assessed for the same reason.
- `PUBLIC_LOAD_SCALING_ONLY_COLLAPSE` — cannot be assessed for the same reason.
- **`PUBLIC_LOAD_SCALING_INCONCLUSIVE`** — *"technical failures or trace/simulator
  incompatibility prevent interpretation."* **This applies exactly**: a technical
  incompatibility between the new harness's hardcoded simulator step budget and the real
  arrival span of a subset of windows (independently confirmed via the mandatory λ=1 check
  against the authoritative prior harness) prevents interpretation of the load curve.

## **Final verdict: `PUBLIC_LOAD_SCALING_INCONCLUSIVE`**

This is not a redefinition of criteria after seeing results — it is the literal, pre-existing
fourth option in the frozen design, triggered by the frozen design's own mandatory check.

---

## Q. Reviewer-level interpretation

1. **Was the reviewer correct that the previous public replay was effectively underloaded?**
   This experiment cannot yet confirm or refute that, because the instrument built to test it
   is itself broken for a material fraction of cells. (The *prior* `public_trace_stress_v1`
   finding of near-zero pressure at the base λ=1/base-capacity configuration is untouched by
   this issue and stands on its own.)
2. **Does policy separation appear at higher load?** Unknown — not assessable from this run.
3. **Is there a meaningful non-collapse operating region?** Unknown — not assessable.
4. **Does public-trace evidence now independently support the synthetic joint-240 portfolio
   story?** Not yet — this leg of evidence is currently unusable pending remediation.
5. **What external-validity limitation remains?** In addition to the pre-existing limitations
   already disclosed in the design (proxy vLLM policy, augmented-not-faithful evidence class),
   there is now a **known engineering limitation**: the harness's fixed simulated-time budget
   cannot currently evaluate the full real-world span of some public-trace windows (esp.
   long-tailed BurstGPT windows) at any preregistered λ.
6. **Does this sufficiently answer the reviewer criticism for the current paper?** **No, not
   yet.** The experiment as executed cannot be cited to answer the reviewer's underload
   concern. It should not be used in the manuscript until re-run with the harness fix and
   re-verified against the mandatory λ=1 check.

Conservative summary: **this run demonstrates the experiment was executed, orchestrated, and
copied back correctly, and it demonstrates precisely why the results cannot yet be trusted —
but it does not yet answer the scientific question the experiment was designed to answer.**

---

## R. Integration with the two new internal results

From the two prior analyses this session had already completed:

1. **Guarded selection** (`docs/current/joint240_guarded_abstaining_selector_v1_analysis_20260825.md`):
   recovers ≈SBS-level safety (gain +0.0010, CI includes zero), catastrophes 67→7, does not
   materially close the ≈0.019 VBS–SBS headroom.
2. **Continuous terminal utility** (`docs/current/decision_criticality_terminal_utility_joint240_v1_analysis_20260825.md`):
   ANWG's hard deadline threshold manufactures most of its own 94.18% exact-zero rate; ~47%
   of ANWG-zero branches carry a real continuous timing effect; magnitude concentration
   (top-10% state mass 93–97%) survives; disagreement-proxy AUROC improves substantially
   under continuous scoring (0.84 vs. 0.68).
3. **Public load scaling** (this analysis): **inconclusive** — the harness has a confirmed,
   precisely-diagnosed integrity defect (fixed ~200s simulated-time ceiling vs. real BurstGPT
   window spans up to 34,064s) that corrupts 6.67% of cells non-uniformly across the load
   grid and fails the experiment's own mandatory λ=1 reproduction check by up to 0.98
   absolute. **No public-trace evidence about load-induced policy separation can be added to
   the manuscript from this run.**

**Strongest honest unified story right now:** the two synthetic joint-240 results (guarded
selection, continuous utility) are mutually reinforcing and both usable: terminal decisions
are sparser-looking under ANWG than under continuous scoring, but the magnitude that *does*
concentrate is genuinely concentrated and genuinely hard to exploit online, which is
consistent with unguarded selection destroying value and guarded selection merely
recovering safety rather than capturing headroom. **The public-trace leg of "is this
externally valid beyond synthetic joint-240" is currently a gap, not a third supporting
pillar** — the reviewer's underload criticism has been *engaged with* (a real, isolated,
non-confounded load-scaling experiment was designed and executed) but not yet *answered*,
because the instrument needs a harness fix before its output can be trusted. This should be
described honestly in the manuscript as **ongoing/blocked work**, not cited as evidence
either for or against load-induced separation. (No manuscript edits were made.)

---

## S. Manuscript-ready numbers

**None of the public-load-scaling numbers are manuscript-ready.** Presenting any load-curve,
headroom, or winner-diversity number from this run — favorable or unfavorable — would be
citing data known to be partially corrupted. The only numbers that are solid and could be
cited (as *methodology/integrity*, not *results*) are:

| Quantity | Value |
|---|---:|
| SLURM completion | 60/60 tasks, exit 0:0 |
| Matrix completeness | 3840/3840 cells, 0 duplicates, 0 NaNs |
| Copy-back integrity | 121/121 files SHA256-verified |
| λ=1 mandatory-check failure rate | 66/360 P6 cells (18.3%), max error 0.98 |
| Cells silently truncated, full grid | 256/3840 (6.67%), spanning λ=1 through λ=128 |
| Windows affected | 11/60 (10 BurstGPT, 1 Azure-code) |
| Worst-case window still corrupted at top of grid | `burstgpt::w0` (span 34,064s, needs λ>170 to clear the harness ceiling) |

---

## T. Files created/modified

Created in `experiments/public_replay_load_scaling_v1/` (copy-back + audit artifacts):
- 60× `cells_window_*.jsonl`, 60× `provenance_task_*.json` (copied from remote, verified byte-identical)
- `remote_sha256.txt`, `remote_logs/` (60×`.out` + 60×`.err`, copied from remote)
- `copyback_manifest.json` — per-file remote/local size + SHA256 + match status
- `integrity_report.json` — full integrity findings, root-cause analysis, corruption scope
- `full_results.csv` — raw 3840-row cell-level dataset (factual copy, no derived interpretation)
- `summary.json` — run-level status, verdict, and halt rationale

Created in `docs/current/`:
- `public_replay_load_scaling_v1_analysis_20260825.md` (this file)

**Deliberately NOT created** (would require interpreting data known to be partially
corrupted, per the mandatory stop-gate in §E): `per_load_summary.csv`,
`source_stratified_summary.csv`, `bootstrap.json`, `winner_counts.json`,
`pressure_separation.json`, `external_policy_increment.json`,
`manuscript_ready_numbers.json` (a stub answer is given in §S instead of a populated file, to
avoid a file that looks authoritative but isn't).

`experiments/public_replay_load_scaling_v1/DESIGN_FROZEN.md` and `smoke/` were **not**
modified (pre-existing, unchanged; rsync did not touch them since remote copies are
byte-identical).

---

## U. Git status

Working tree changes are scoped to the files listed in §T (all under
`experiments/public_replay_load_scaling_v1/` and `docs/current/`). No other repository files
were touched. **No commits made. No pushes made.**

---

## Confirmation: manuscript untouched

No files under `paper/` were opened for editing or modified in this session.

## Confirmation: no new SLURM job submitted

Job 1195488 was queried exactly once (read-only `sacct`) and was not cancelled, resubmitted,
or interfered with. No new SLURM job was submitted, per the explicit instruction to STOP and
report first upon finding a genuine artifact defect — which this session did.
