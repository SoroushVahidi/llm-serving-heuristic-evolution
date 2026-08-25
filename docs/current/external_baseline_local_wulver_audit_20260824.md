# External-Baseline Local + Wulver Recovery / Gap Audit

**Date:** 2026-08-24  
**Mode:** AUDIT ONLY (no experiments launched, no jobs submitted, no scientific code/config changes, no commit/push)  
**Primary repo:** `/home/soroush/llm-serving-heuristic-evolution`  
**Verdict:** `EXTERNAL_BASELINE_AUDIT_COMPLETE_WITH_UNRESOLVED_ITEMS`

---

## A. LOCAL PRE-FLIGHT

| Field | Value |
|---|---|
| Hostname | `al-khwarizmi` |
| Repository | `/home/soroush/llm-serving-heuristic-evolution` |
| Branch | `contextual-compositional-heuristics-20260731` |
| HEAD | `2987b7181efa2bc550d8a894c537eca8f6393eb6` |
| Upstream | `origin/contextual-compositional-heuristics-20260731` |
| Ahead/behind | **ahead 2** / behind 0 (at audit time) |
| Dirty tree | Yes — large modified + untracked experiment/docs/paper/scripts set; **preserved** |
| Worktrees | Primary only for this repo (`git worktree list` shows this checkout) |
| Git locks | None observed |
| Tmux | Not required for audit conclusion |
| Nearby copies | `/home/soroush/llm-serving-heuristic-evolution-local-artifacts`; other LLM projects under `/home/soroush/` (frontier-allocation, consistency-aware-llm-rankin, mwfas staging) — scanned for sola/wait/qlm clones: **no official SOLA/WAIT/QLM repos found** |
| Mounts | Standard local home; no live Wulver FS mount required (SSH used) |

---

## B. WULVER ACCESS / PRE-FLIGHT

| Field | Value |
|---|---|
| SSH alias | `Host login02` → `login02.tartan.njit.edu`, User `sv96` (also historical `wulver.njit.edu`) |
| Working connection | Control master: `ssh -S ~/.ssh/wulver-control sv96@wulver.njit.edu` (lands on `login02`) |
| Remote user / home | `sv96` / `$HOME` → `/home/sv96` (also `/mmfs1/home/sv96`) |
| Project | `/mmfs1/project/ikoutis/sv96/` (`github/`, `vendor/`, `llmserveopt-data/`, archives) |
| Scratch | `/mmfs1/scratch/ikoutis/sv96/` (policy-separation pilots, Sarathi/vLLM GPU validations, Apt-Serve probes) |
| Active SLURM | `squeue -u sv96`: **empty** at audit time |
| Main clone HEAD (Wulver) | `/mmfs1/project/ikoutis/sv96/github/llm-serving-heuristic-evolution` on branch `wulver-policy-composition-readiness` @ `c8aee12` (older/different from local research HEAD) |
| Extra clones | `llm-serving-heuristic-evolution-cc-probe`, `...-policy-separation-v1`, SwissAI repair clone — **contain `baselines/vtc` + `baselines/vllm_ltr` + VTC sweep JSON (108 rows)** |
| Vendor | `/mmfs1/project/ikoutis/sv96/vendor/{sarathi-serve,apt-serve}` |

No private keys or credentials printed.

---

## C. SEARCH COVERAGE

Searched aliases for: vLLM / VTC / SOLA / LTR / WAIT / KV-constrained / Sarathi / QLM / DistServe / Llumnix / Mooncake / FastServe / Splitwise / BurstGPT / Azure / Family A–C / joint-240 / stress/scaling / Pext.

Locations:

- Local primary tree: `baselines/`, `src/llmserveopt/policies/`, `experiments/`, `results/`, `docs/audits/`, `docs/current/`, `configs/stress_tests/`, `paper/`, `scripts/`, `tests/`
- Local nearby homes under `/home/soroush/` (depth-limited)
- Git history (`git log --all --grep` for VTC/LTR/SOLA/external baseline/utility matrix)
- Wulver project/scratch/home via SSH find + directory listing + VTC JSON integrity
- `docs/BASELINE_STATUS.md`, `docs/external_baseline_decision.md`

---

## D. GIT-HISTORY FINDINGS

Commits establishing external baselines (still discoverable; not checked out into dirty worktree):

| Theme | Example commits | Implication |
|---|---|---|
| VTC | `07c79d9`, `05e0da9` | Official adapter + fairness sweep |
| vLLM-LTR | `7a19a0e`, `d64ed68`, `aea0249` | Checkpoint-backed LTR + verified comparative eval |
| Sarathi | `3e87439`, `74c7fc8` | Artifact audit + stress coverage |
| Apt-Serve / Llumnix / DistServe | multiple Aug 2026 baseline commits | Faithful sims + evaluations |
| Unified / joint matrices | `bda0755`, `43c88e1`, … | **P6-only** matrices; no Pext expansion commit found |

No historical commit found that adds VTC/SOLA/LTR columns to joint-240 or Families A/B/C under current ANWG Pext protocol.

---

## E. LOCAL FILESYSTEM FINDINGS (HIGH SIGNAL)

### Mandatory methods

| Method | Implementation | Completed results (prior regimes) | On current paper workloads? |
|---|---|---|---|
| Native vLLM | `experiments/real_vllm_mechanism_validation_v1/` (0.27.x / Qwen2.5-0.5B / RTX 5060 Ti); older `experiments/real_llm/`, `gpu_external_validity/` | Semantic validation + chunk-budget probe (T512 vs T4096) | **No** Pext comparison on joint/public/Families |
| VTC | `baselines/vtc/` official `VTCReqQueue` adapter; scripts `run_vtc_fairness_comparative_sweep.py` | `baselines/vtc/sweep_results/vtc_fairness_comparative_sweep_20260805.json` (108 runs); audits under `docs/audits/vtc_*` | **No** joint-240 / A–C / BurstGPT–Azure paper cells |
| SOLA | `src/llmserveopt/policies/sola_style_state_aware.py` only | Style heuristic; docstring: **not** a SOLA reproduction | Faithful SOLA **missing** |
| LTR | `baselines/vllm_ltr/` official checkpoint adapter | `results/vllm_ltr_first_comparative_evaluation/` (30 cells, ANWG present, WildChat) | **No** joint-240 / Families / public-trace Pext cells |

### Strong optional / related

| Method | Local status |
|---|---|
| Sarathi | Faithful + style policies; Wulver GPU validation docs; stress catalog entries |
| DistServe / Llumnix | Faithful policies + Aug 2026 comparative audits (`FOUNDATIONAL_CANDIDATE*`) |
| Apt-Serve | Large Phase G collection + analysis (portfolio marginal contribution; not mandatory Pext) |
| WAIT / Nested WAIT | Prose in `docs/external_baseline_decision.md` §B.3 only |
| QLM | Related-work / decision-doc only |
| `kv_constrained_online` | **Internal P6 member**, not Jaillet/WAIT external algorithm |

### Workloads

| Workload | Evidence |
|---|---|
| Joint 240 | `experiments/joint_multimechanism_generalization_v1/` — 240×6=1440 cells; SBS=0.314072; VBS=0.333106; headroom=0.019034; winners 46/5/45/35/59/50 — **P6 only** |
| Unified matrix v2 | Same six `anwg__*` columns only |
| Public BurstGPT/Azure replay | Completed; **ANWG=1.0 all cells** (near-degeneracy) — docs/current analysis 20260820 |
| Stressed BurstGPT (June) | `results/burstgpt_scaled_*` — old 14-policy latency suite (`vllm_style_token_budget`, `sarathi_style`, …); **no ANWG primary; not P6/Pext** |
| Families A/B/C | Extensive internal P6 / selector / oracle work; **no VTC/SOLA/LTR/native-vLLM external columns on frozen family matrices** |

---

## F. WULVER FILESYSTEM FINDINGS

| Finding | Path / note |
|---|---|
| VTC + LTR code + VTC 108-row sweep | Present in `*-cc-probe` and `*-policy-separation-v1` clones (mirrors local) |
| Sarathi / vLLM GPU jobs | Scratch: `sarathi_*`, `vllm_kv_pressure_*`, `vllm_repeated_trial_*` |
| Policy-separation pilots | Scratch + local copies; fairness/prefill-decode/Sobol — **internal policies**, not VTC/SOLA/LTR on joint-240 |
| No scratch dirs named | `*vtc*`, `*sola*`, `*ltr*`, `*joint*`, `*burst*`, `*azure*` at shallow search (beyond policy-sep / sarathi / vllm GPU) |
| No expanded Pext matrix | No `utility_matrix` with >6 ANWG external columns found on Wulver |
| Archive curiosity | `adaptive-reasoning-archive-.../canonical_external_baseline_closure_20260424T000020Z/` — **different project/era**; not current Pext |

---

## G–O. METHOD DOSSIERS (COMPRESSED)

Status codes: I* implementation / E* execution / R* result (see task §9).  
Compatibility vs frozen paper Pext (ANWG, P6 vs Pext, joint/Families/public stress).

### G. Native vLLM default / chunked

- **Local:** I5 semantic validation (Family-B analogue; disable vs enable chunked prefill; T512/T4096 budget probe). E4/R4 for **mechanism fidelity**, not portfolio comparison.
- **Wulver:** Extensive historical Sarathi↔vLLM GPU validations + KV-pressure vLLM runs (different question / older configs).
- **Best:** I5 / E4 / R4 (validation) — **E0 / R0 for Pext comparison cells**.
- **Action:** `RERUN_REQUIRED` (sim-side proxy + selective native cells) / native full-trace comparison is HIGH cost → prefer simulator `vllm_style` + targeted native Family-B confirmation already done.

### H. VTC

- **Local:** I4–I5 official adapter; E5/R4 fairness family sweep (108 = 6 policies × 6 families × 3 seeds); ANWG in JSON; classified `FOUNDATIONAL_CANDIDATE` / `EVALUATION_ONLY`.
- **Wulver:** Same sweep JSON present in project clones.
- **Workloads done:** Dedicated fairness micro/families — **not** joint-240, not A/B/C paper manifests, not BurstGPT/Azure stress.
- **WFS ≠ VTC.**
- **Action:** `RERUN_REQUIRED` on mandatory common cells using existing adapter (do **not** re-implement).

### I. SOLA

- **Local:** I2 `sola_style_state_aware` only; explicitly non-faithful.
- **Wulver:** No official SOLA clone / results found.
- **Action:** `IMPLEMENT_AND_RUN` for mandatory claim **or** downgrade manuscript wording to style-inspired (author decision). Strict audit: **missing faithful SOLA**.

### J. Learning-to-Rank (vLLM-LTR)

- **Local:** I4–I5 checkpoint-backed; E5/R4 WildChat comparative (10 policies × 3 seeds); ANWG present.
- **Not** ESTF alone — real LTR baseline exists, wrong workload for Pext.
- **Action:** `RERUN_REQUIRED` on paper workloads (adapter reuse).

### K. WAIT / Nested WAIT

- I1 prose only. **Action:** `OPTIONAL_NO_ACTION` (do not block paper).

### L. External KV scheduler (Jaillet / WAIT-class)

- Internal `kv_constrained_online` is P6 member (inspired), **not** external paper algorithm.  
- **Action:** `RELATED_WORK_ONLY` / `OPTIONAL_NO_ACTION` for external Jaillet/WAIT unless author expands scope.

### M. Sarathi

- I4–I5 + Wulver GPU evidence. Optional for mechanism coverage (Family B).  
- **Action:** `OPTIONAL_NO_ACTION` for mandatory Pext; cheap to include in Family-B mechanism table if desired (`REUSE_NOW` for prior mechanism claims only).

### N. QLM

- I0–I1. **Action:** `OPTIONAL_NO_ACTION`.

### O. Architecture-level (FastServe, DistServe, Llumnix, Mooncake, Splitwise)

- DistServe/Llumnix: faithful sims + evals → `RELATED_WORK_ONLY` / optional foundational.
- FastServe/Mooncake/Splitwise: cite-only. **Action:** `RELATED_WORK_ONLY`.

---

## P–W. WORKLOAD DOSSIERS (COMPRESSED)

### P. BurstGPT

- Public-window replay: complete P6, **saturated ANWG=1.0** — cannot discriminate.
- June scaled calibrated runs: complete but **obsolete policy set + non-ANWG primary**.
- **External policies on BurstGPT under current protocol:** **none**.

### Q/R. Azure conversation / code

- Same public-trace package: P6 complete, saturated; no VTC/SOLA/LTR/native comparison cells for Pext.

### S. Stressed real-trace

- Stress **catalog** exists (`configs/stress_tests/algorithm_stress_test_catalog.yaml`) including vllm_ltr targets.
- Scaled BurstGPT results exist but obsolete vs Pext.
- **No** verified scheduler-discriminating public-trace cell under current ANWG+P6/Pext protocol.  
- **Action needed:** design+run **one** discriminating stress config (time compression / rate multiplier / SLO tighten) — Group 2/3.

### T. Family A (fairness/completion/SLO)

- Rich internal evidence; VTC prior fairness sweep is **related but not same manifests**.  
- Missing: VTC (+ optionally SOLA/LTR) on Family-A compatible cells.

### U. Family B (prefill/decode)

- Native vLLM semantic validation **reuse for mechanism claim**.  
- Missing: external columns in Family-B utility matrix for Pext; Sarathi optional.

### V. Family C (KV pressure)

- Internal KV pilots + `kv_constrained_online` in P6.  
- Missing external WAIT/Jaillet; native vLLM KV-pressure on Wulver is **older hardware validation**, not current Family-C Pext matrix.

### W. Joint-240 expanded portfolio

- **No** external baseline columns. No Pext matrix anywhere local or Wulver.  
- Expanding to Pext requires **new cells** for each mandatory external method (reuse adapters where I≥4).

---

## X. MASTER BASELINE × WORKLOAD MATRIX

Standardized **Action** values only.

| Method | Workload | Local impl | Wulver impl | Prior exec | Result | Canonical path | Compatibility | Action |
|---|---|---|---|---|---|---|---|---|
| vLLM native default/chunked | Family-B analogue (local GPU) | I5 | I4 (older GPU) | E4 | R4 | `experiments/real_vllm_mechanism_validation_v1/` | Mechanism validation ≠ Pext | **REUSE_NOW** (semantic claim only) |
| vLLM native / style | Joint 240 | I3 style policies exist historically | — | E0 Pext | R0 | — | Need cells | **RERUN_REQUIRED** |
| vLLM native / style | BurstGPT/Azure/stress | Partial old | GPU hist. | E3 old / E4 sat. public | R2–R3 | public replay; `results/burstgpt_scaled_*` | Public sat. / scaled obsolete | **RERUN_REQUIRED** |
| VTC | Fairness families (Aug 5) | I5 | I5 copy | E5 | R4 | `baselines/vtc/sweep_results/vtc_fairness_comparative_sweep_20260805.json` | Wrong workload for Pext | **REUSE_NOW** (fairness appendix / provenance only) |
| VTC | Joint 240 / Fam A / public stress | I5 adapter | I5 adapter | E0 | R0 | `baselines/vtc/` | Need run | **RERUN_REQUIRED** |
| SOLA faithful | Any paper WL | I2 style only | I0 | E0 | R0 | `sola_style_state_aware.py` | Insufficient fidelity | **IMPLEMENT_AND_RUN** |
| LTR (vllm_ltr) | WildChat eval | I5 | I4 tests/code | E5 | R4 | `results/vllm_ltr_first_comparative_evaluation/` | Wrong WL | **REUSE_NOW** (appendix/provenance) |
| LTR | Joint / Fam A / public | I5 adapter | I4 | E0 | R0 | `baselines/vllm_ltr/` | Need run | **RERUN_REQUIRED** |
| WAIT/Nested WAIT | Any | I1 | I0 | E0 | R0 | decision doc | — | **OPTIONAL_NO_ACTION** |
| External KV paper algo | Any | I2 internal only | — | E4 as P6 member | R5 in joint | `kv_constrained_online` | Not external | **RELATED_WORK_ONLY** |
| Sarathi | Mechanism / stress | I5 | I5 GPU | E5 | R4 | audits + scratch | Optional | **OPTIONAL_NO_ACTION** |
| QLM | Any | I0 | I0 | E0 | R0 | — | — | **OPTIONAL_NO_ACTION** |
| DistServe/Llumnix | Prior comparative | I4 | copies | E5 | R4 | `docs/audits/*` | Related | **RELATED_WORK_ONLY** |
| FastServe/Mooncake/Splitwise | — | I1 | I0 | E0 | R0 | cites | — | **RELATED_WORK_ONLY** |
| P6 portfolio | Joint 240 | I5 | — | E5 | R5 | `experiments/joint_multimechanism_generalization_v1/` | Canonical | **REUSE_NOW** |
| P6 | Public BurstGPT/Azure | I5 | — | E5 | R4 | public_trace_replay_v1 | Saturated — not discriminative | **RECOMPUTE_METRICS_ONLY** N/A; need stress **RERUN_REQUIRED** |

---

## Y. GROUP 1 — REUSE NOW

1. **P6 joint-240 matrix** — `experiments/joint_multimechanism_generalization_v1/` (1440 cells; SBS/VBS/headroom/winners frozen). No rerun.
2. **Native vLLM semantic validation** — Family-B analogue + chunk-budget probe under `experiments/real_vllm_mechanism_validation_v1/`. Supports mechanism fidelity text, **not** Pext table.
3. **VTC fairness sweep (108)** — provenance / optional appendix; **not** Pext joint cells.
4. **vLLM-LTR WildChat comparative** — provenance / optional appendix; **not** Pext joint cells.
5. **Sarathi/Llumnix/DistServe/Apt-Serve prior evals** — related-work / optional mechanism citations only.

---

## Z. GROUP 2 — RERUN / COMPLETE ONLY

| Item | Method | Workload | Impl path | Missing | Env | Cost |
|---|---|---|---|---|---|---|
| Z1 | VTC | Joint 240 | `baselines/vtc/` | 240 (+seeds if required) ANWG cells vs P6 | Local or Wulver **CPU** | **MEDIUM** |
| Z2 | VTC | Family A (mechanism-relevant subset) | same | Family-A cells under current manifests/overlays | CPU | **SMALL–MEDIUM** |
| Z3 | LTR | Joint 240 | `baselines/vllm_ltr/` | 240 cells; may need offline scores for joint prompts | CPU (+ scoring pass) | **MEDIUM–LARGE** |
| Z4 | LTR | Family A subset (length heterogeneity) | same | Compatible family cells | CPU | **MEDIUM** |
| Z5 | vLLM-style / native proxy | Joint 240 + Fam B | existing `vllm_style*` / registry | Pext columns; native full joint **not** required if sim proxy accepted | CPU; optional local GPU spot-check | **MEDIUM** |
| Z6 | Discriminating public-trace stress | P6 + externals | public replay harness + stress generators | One load/SLO/time-compression config that breaks ANWG=1.0 | CPU first | **MEDIUM** |
| Z7 | BurstGPT / Azure (stressed) | VTC+LTR+vLLM-style | adapters | Common-benchmark cells after stress works | CPU | **MEDIUM** |

Do **not** launch until authorized.

---

## AA. GROUP 3 — IMPLEMENT + RUN

| Item | Method | Workload | Why insufficient | Source | Env | Impl difficulty | Run cost |
|---|---|---|---|---|---|---|---|
| AA1 | **Faithful SOLA** | Fam A (SLO) + joint subset or full joint if feasible | Only `sola_style_*`; not paper algorithm | Need official artifact/paper algorithm port (none cloned) | CPU | **HIGH** | MEDIUM |
| AA2 | (Conditional) Native vLLM on stressed public traces | BurstGPT/Azure stress | Semantic validation ≠ baseline comparison | local vLLM 0.27.x install | **Local GPU** / Wulver GPU | MEDIUM wiring | **LARGE** — only if sim proxy rejected |

Optional high-cost architecture systems **not** in Group 3.

---

## AB. MINIMUM MANDATORY EXPERIMENT SET STILL MISSING

**Common benchmark (all feasible mandatory methods):**

1. Joint-240 ANWG columns for: **VTC**, **LTR**, **vLLM default/chunked (sim-faithful or native-proxy)**.  
2. At least one **stressed** BurstGPT **or** Azure configuration where ANWG < 1 and policies separate, with same three methods (+ P6).  
3. **Faithful SOLA** on Family-A (SLO) + enough joint/public cells to enter Pext envelope analysis — **or** author demotes SOLA from mandatory.

**Mechanism-specific:**

- Fam A: VTC (+ SOLA if kept mandatory); LTR if length conflict present.  
- Fam B: vLLM default vs chunked (reuse native validation + sim columns).  
- Fam C: internal KV already in P6; external WAIT **not** mandatory.

---

## AC. OPTIONAL EXPERIMENTS THAT SHOULD NOT BLOCK THE PAPER

- WAIT / Nested WAIT / Jaillet full reproduction  
- QLM  
- Full Sarathi-Serve / DistServe / Llumnix / Apt-Serve re-runs on joint-240  
- FastServe / Mooncake / Splitwise  
- Exhaustive method × every workload Cartesian product  
- Native vLLM full joint-240 (GPU-prohibitive)

---

## AD. PRIORITIZED EXECUTION PLAN (WHEN AUTHORIZED)

1. **Reuse** Group 1 artifacts; refresh analysis tables only if needed (no new sims).  
2. **Stress public-trace calibration** (CPU): find one discriminating load/SLO/compression setting (Z6) — unlocks BurstGPT/Azure cells.  
3. **Batch CPU Pext expansion:** VTC + vLLM-style on joint-240 (Z1, Z5) in one harness.  
4. **LTR scoring + joint/family cells** (Z3–Z4) — may parallelize after prompt manifest freeze.  
5. **Family-A VTC (+ SOLA if implemented)** mechanism table.  
6. **Implement SOLA last** among mandatory (AA1), unless author removes from frozen list.  
7. Native GPU comparison only for disputed cells (AA2).

Safe batching: VTC∥vLLM-style on shared scenario manifest; LTR after aux scores cached; SOLA independent once adapter exists.

---

## AE. ARTIFACT INTEGRITY CHECK (LIGHTWEIGHT)

| Artifact | Check | Result |
|---|---|---|
| Joint long/wide | 1440 cells; 6 policies; 240 scenarios; integrity JSON matches | **PASS** |
| Joint SBS/VBS | 0.314072 / 0.333106 / headroom 0.019034 | **PASS** |
| VTC sweep | 108 rows; 6 policies × 6 families × 3 seeds; ANWG ∈ [0.396, 1.0] | **PASS** |
| LTR metrics | 30 rows; 10 policies; seeds 0–2; ANWG column present | **PASS** |
| Public trace | Documented ANWG=1.0 all annotated cells | **PASS** (degenerate) |
| BurstGPT scaled | 14 old policies; latency metrics; no ANWG | **PASS as obsolete** |
| No silent Pext matrix | No >6-policy ANWG joint matrix found | **PASS** (gap confirmed) |

---

## AF. UNRESOLVED QUESTIONS

1. Will the paper accept **simulator-faithful vLLM-style** as M1 for joint/public, with native reserved for Family-B semantic validation (already done)?  
2. Is **faithful SOLA** still mandatory, or may `sola_style` / LLF-adjacent wording replace it?  
3. Exact seed/replication policy for new Pext cells (match joint-240 single-seed vs multi-seed)?  
4. LTR: can joint-240 prompts be scored offline with the pinned checkpoint without domain-shift NO_GO?  
5. Preferred stress transform for public traces (rate ×k vs time compression vs SLO tighten)?  
6. Wulver main clone is on an **older branch** than local research HEAD — which tree is execution source of truth when authorized?

---

## AG. GIT / SCIENTIFIC SAFETY STATUS

- No experiment launched; no SLURM submit; no GPU/CPU scientific runs.  
- No code/config mutation for science; **this audit markdown is the only intended new doc artifact**.  
- No commit, push, pull, merge, rebase, reset, stash, clean, or delete.  
- Dirty/untracked local work **preserved**.  
- Wulver inspected read-only via existing SSH control socket.

---

## AH. FINAL VERDICT

**`EXTERNAL_BASELINE_AUDIT_COMPLETE_WITH_UNRESOLVED_ITEMS`**

**Bottom line:** Strong prior implementations exist for **VTC** and **vLLM-LTR**, and strong **native vLLM semantic validation** exists, but **no mandatory external method has a current-paper-compatible completed Pext matrix** on joint-240 / Families / discriminating public traces. **Faithful SOLA is missing.** Stressed public-trace discrimination for ANWG is also still missing. Minimum path = reuse adapters + CPU Pext expansion + one stress unlock + SOLA implement-or-demote.
