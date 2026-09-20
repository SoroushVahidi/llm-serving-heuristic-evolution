# FGCS Submission Roadmap and Handoff

**STATUS:** ACTIVE / AUTHOR-REVIEW STAGE

**CANONICAL CURRENT-STATE DOCUMENT:**
`docs/current/FGCS_SUBMISSION_ROADMAP_20260920.md`

**CURRENT JOURNAL TARGET:**
Future Generation Computer Systems (FGCS)

**CURRENT MANUSCRIPT TITLE:**
When Does LLM-Serving Scheduler Adaptation Matter?
Action Opportunity and Causal Headroom in Production-Derived Replay

> This file is the **single authoritative source of truth** for the current
> FGCS submission effort. If any other document (including historical "FINAL"
> or "release-ready" labels) conflicts with this file on the *current*
> submission state, treat this file as current and the other as historical.

---

## Current Git State (verified 2026-09-20)

| Item | Value |
|---|---|
| Default branch | `main` @ `617d6dce` |
| Canonical historical/scientific integration branch | `contextual-compositional-heuristics-20260731` @ `76a3cf3e` (in sync with origin) |
| Current FGCS manuscript revision branch | `revise/fgcs-prior-reviewer-risk-closure-20260920` @ `b5ffe4f5` |
| Revision branch pushed? | **NO** (local only, 3 commits ahead of the canonical branch) |
| Revision worktree | `/home/soroush/llm-serving-heuristic-evolution-worktrees/fgcs-prior-reviewer-risk-closure-20260920` (clean) |
| Author-review PDF on `main` | YES — `paper/llm_scheduler_adaptation_causal_headroom.pdf` (commit `617d6dce`), SHA-256 `c810cfb1b1dd41d086810f9806fcd5fd91ff1730e0638e6600c741255f581fda`, 15 pages, built from revision commit `b5ffe4f5` |

The revision branch contains the manuscript source (`paper/llm2026/main.tex`),
the direct public-repository-link update, and all reviewer-risk-audit fixes.
It has **not** been merged into `main` and **not** been pushed.

---

## Status Labels

Used consistently throughout this roadmap:

- **DONE** — completed and verified; do not redo.
- **READY** — complete and verified, ready to be consumed by a later stage.
- **PENDING** — known remaining work, not yet done.
- **BLOCKED** — cannot proceed until a specific dependency is resolved.
- **OPTIONAL** — nice-to-have; skip if not needed for submission.
- **HISTORICAL_ONLY** — retained for provenance; not part of the current FGCS submission line.
- **DO_NOT_REOPEN_WITHOUT_NEW_EVIDENCE** — settled decision; only revisit with new evidence.

---

## 1. Executive Scientific Summary

The current FGCS paper asks a staged question:

```text
production-derived workload
  -> resource pressure
  -> canonical executable scheduler disagreement  P(D)
  -> default-relative one-step causal latency headroom  P(B_LAT | D)
  -> is adaptive-scheduling complexity justified?
```

It is a **measurement** paper about *when* adaptation matters, not a systems
paper that ships a new scheduler. The paper does **not** primarily propose:

- a new scheduler;
- a learned deployable selector;
- a universal adaptive policy;
- scheduler-superiority results over all modern systems.

### Central evidence (frozen)

- **Native production-derived replay is action-null under abundant resources.**
  Three workloads (Azure 2023 code, Azure 2023 conversation, BurstGPT) produced
  0/99,992, 0/461,985, and 0/440,461 canonical disagreements respectively under
  an abundant-resource baseline.
- **Tight KV-capacity and active-sequence pressure create canonical
  disagreement** (arrival scaling up to 8x does not).
- **Fresh untouched-window causal study:** 720 disagreement states; 590
  beneficial states; P(B_LAT | D) = 0.8194; mean oracle latency headroom
  1.9958 ms; clustered 95% CI [0.1846, 3.4974] ms; bootstrap seed 20260920.
- **ANWG objective sensitivity:** terminal ANWG saturated at 1.0 in the earlier
  objective study, demonstrating objective choice matters — not intervention
  failure.
- **Bounded real-vLLM validation:** mechanism/pressure-action correspondence
  only (one RTX 5060 Ti, vLLM 0.27.1, Qwen2.5-0.5B). It does **not** establish
  real-system one-step causal headroom or deployed-selector gains.

**NEW EXPERIMENTS REQUIRED BEFORE SUBMISSION = NO**
(unless repository evidence contradicts this; nothing currently does).

---

## 2. Scientific Claim Boundaries

**SUPPORTED:**
- production-derived abundance-null result;
- pressure-to-disagreement transition;
- fresh selected-regime default-relative one-step latency headroom;
- objective sensitivity (ANWG saturation);
- bounded real-vLLM pressure/action correspondence.

**NOT CLAIMED** (future agents must not broaden the manuscript into any of
these):
- universal external validity;
- all schedulers;
- all workloads;
- all GPU architectures;
- learned predictability;
- deployed selector performance;
- real-system recovery of 1.9958 ms;
- universal scheduler superiority;
- proof that adaptation is always useful;
- proof that adaptation is never useful.

---

## 3. Completed Quality and Integrity Audits

| Audit | Final status | Reopen? |
|---|---|---|
| A. Prior-reviewer-risk audit | contribution novelty LOW, causal LOW, baseline LOW, actionability LOW, statistical LOW, generalization MEDIUM-but-bounded; no new experiment required | DO_NOT_REOPEN_WITHOUT_NEW_EVIDENCE |
| B. Repetition / visual / text-consistency audit | repetition LOW, argument structure STRONG, text-figure PASS, text-table PASS, figure-table PASS, visual PASS (minor refinement items tracked in Section 6) | PENDING (visual items only) |
| C. Historical test failures | six failures in `tests/test_unified_utility_matrix_v1.py` from unavailable BurstGPT data path / environment dependency; deterministic historical-pipeline failures; scientific impact on FGCS = NONE; full-suite rerun unnecessary | DO_NOT_REOPEN_WITHOUT_NEW_EVIDENCE |
| D. Secret audit | CloudRift key exists only in local gitignored `results/.../environment.txt`; never committed; never pushed; no Git-history purge required; public secret scan PASS | DO_NOT_REOPEN_WITHOUT_NEW_EVIDENCE |
| E. Data/code availability audit | direct public repo URL in manuscript revision; public repo holds source, configs, derived artifacts, result tables, manifests, figure scripts, reproducibility docs; raw traces remain upstream | READY |
| F. Duplicate-submission / overlap audit | FGCS manuscript distinct from scheduler-ranking portability, module-intervention, and frontier-allocation manuscripts; earlier LLM 2026 paper is a direct withdrawn predecessor; disclose accurately if asked | READY |

---

## 4. Historical LLM 2026 Manuscript Status

**LLM2026_STATUS = HISTORICAL_WITHDRAWN_PREDECESSOR**

The repository still contains `paper/llm2026/` and historical documents calling
"The Exploitability Gap in LLM-Serving Scheduler Portfolios" "final" or
"release-ready." These are **HISTORICAL_ONLY** relative to the current FGCS line:

- The historical LLM 2026 conference manuscript was **submitted (2026-08-25),
  accepted, then withdrawn before publication**. No proceedings/DOI/publication
  record was found.
- The **current FGCS manuscript supersedes** the withdrawn conference manuscript
  as the active submission.
- Historical provenance is preserved; these files are **not deleted** and their
  "final" labels are not rewritten (they were accurate at the time).

When FGCS asks about previous submissions or related manuscripts, disclose the
withdrawn LLM 2026 predecessor accurately (see Section 6 item K and the
Stage 8 cover letter).

---

## 5. Author-Review PDF

- **Path (public):** `paper/llm_scheduler_adaptation_causal_headroom.pdf`
- **Branch:** `main`
- **Commit:** `617d6dce`
- **SHA-256:** `c810cfb1b1dd41d086810f9806fcd5fd91ff1730e0638e6600c741255f581fda`
- **Pages:** 15
- **Built from:** revision commit `b5ffe4f5`

> **AUTHOR-REVIEW PDF ≠ FINAL SUBMISSION PDF.** This review copy still precedes
> final disclosures/acknowledgments, FGCS template conversion, and visual
> polish. It exists for author visual review only.

---

## 6. Remaining Manuscript Work

**All items below are PENDING. Do NOT fix them in the repository-organization
query that created this roadmap.** Each is recorded so the next agent works
one issue at a time.

### A. Scientific / numeric verification

| ID | Issue | Priority | Next action |
|---|---|---|---|
| A1 | **Table 3 window-count discrepancy:** visible per-row windows are 2, 15, 14, 17, 1 (sum = 49), but the caption reportedly states row-window sum = 67. | HIGH | Verify against the canonical artifact **before** changing anything; do not guess the correct value. |
| A2 | Clarify pooled weighting/composition of the 720-state causal aggregate if needed for reviewer clarity. | MEDIUM | Check against `FRESH_LATENCY_CAUSAL_RESULT_V1.json` / bootstrap artifact. |

### B. Venue / template compliance

| ID | Issue | Priority | Next action |
|---|---|---|---|
| B3 | Verify current official FGCS Guide for Authors from authoritative sources. | HIGH | Fetch Elsevier/FGCS official page; record requirements. |
| B4 | Confirm current Elsevier/FGCS manuscript template/class. | HIGH | Identify required `.cls`/template. |
| B5 | Convert from historical LNCS formatting to the required FGCS/Elsevier format. | HIGH | Do not perform until B3/B4 verified. |
| B6 | Re-check page count AFTER conversion. | MEDIUM | Build and count. |
| B7 | Verify abstract word/format constraints. | MEDIUM | Check FGCS limits. |
| B8 | Verify citation/reference style. | MEDIUM | Check FGCS/Elsevier style. |
| B9 | Verify highlights requirement. | MEDIUM | Check FGCS requirement. |
| B10 | Verify cover-letter and submission-portal requirements. | MEDIUM | Record at submission-prep time. |

### C. Abstract

| ID | Issue | Priority | Next action |
|---|---|---|---|
| C11 | Make abstract fully self-contained. | HIGH | Edit in Stage 3. |
| C12 | Avoid unexplained acronyms/notation (ANWG, SLO, KV, P(B_LAT \| D)) unless defined. | HIGH | Define or replace in abstract. |
| C13 | Prefer plain-language "590/720 = 81.9%" where appropriate. | MEDIUM | Apply in abstract. |

### D. Repository-report language in manuscript

| ID | Issue | Priority | Next action |
|---|---|---|---|
| D14 | Replace internal labels (Phase-B V2, Phase-D V1, LATENCY_HEADROOM_CONFIRMED, PARTIAL_SUPPORT) with publication-quality prose. | MEDIUM | Rewrite in Stage 3. |
| D15 | Remove/rewrite repository-management language in the Conclusion (e.g., "The shortest path forward is final manuscript and artifact audit..."). | MEDIUM | Rewrite in Stage 3. |

### E. Terminology / definitions

| ID | Issue | Priority | Next action |
|---|---|---|---|
| E16 | Define/redefine at first body use: SBS, P6, KV cache, TTFT, E2E latency, "low-late"/"high-late" regime labels. | MEDIUM | Add definitions in Stage 3. |
| E17 | Replace raw repo config IDs in reader-facing prose/figures (active_8 -> active-sequence cap = 8; kv_16000 -> KV-capacity setting = 16,000), retaining reproducibility mapping. | MEDIUM | Apply in Stage 3/5. |

### F. Figures

| ID | Issue | Priority | Next action |
|---|---|---|---|
| F18 | Figure 5: top-left annotation collides with top frame; rightmost Azure-code kv_16000 label too close to marker/frame; labels too small. Increase size + deliberate offsets/margins. | MEDIUM | Fix in Stage 5 (after template known). |
| F19 | Figure 2: x-axis category labels small/crowded; consider cleaner multiline labels/sizing. | MEDIUM | Fix in Stage 5. |
| F20 | Re-inspect every figure at actual print size after template conversion. | MEDIUM | Stage 5/6. |

### G. Tables

| ID | Issue | Priority | Next action |
|---|---|---|---|
| G21 | Table 2 too compressed; header columns run together; increase row spacing + column clarity; replace raw config IDs. | MEDIUM | Stage 5. |
| G22 | Table 1 small/dense; consider full-width in final Elsevier layout. | MEDIUM | Stage 5/6. |
| G23 | Table 3: resolve 49-vs-67 discrepancy first, then row spacing. | HIGH (depends on A1) | After A1. |
| G24 | Apply adequate row spacing consistently (e.g., appropriate `\arraystretch`) without wasting page space. | LOW | Stage 5. |

### H. Equations

| ID | Issue | Priority | Next action |
|---|---|---|---|
| H25 | Audit all display mathematics. | MEDIUM | Stage 3. |
| H26 | The displayed P(B_LAT \| D) = 590/720 = ... is standalone but unnumbered. | MEDIUM | Make inline or number consistently. |
| H27 | Ensure every argumentative displayed equation is numbered/referenced consistently. | MEDIUM | Stage 3. |

### I. References

| ID | Issue | Priority | Next action |
|---|---|---|---|
| I29 | Full bibliography metadata audit against authoritative sources. | HIGH | Stage 3. |
| I30 | Verify the claim attached to every citation. | HIGH | Stage 3. |
| I31 | Re-check known stale metadata: ServeGen, LMetric. | HIGH | Stage 3. |
| I32 | Prefer <= ~4 references at the end of one sentence; split claims for attribution clarity. | LOW | Stage 3. |

### J. Acknowledgments / funding / AI disclosure

| ID | Issue | Priority | Next action |
|---|---|---|---|
| J33 | Current acknowledgment is incomplete. | HIGH | Stage 4. |
| J34 | Final acknowledgment should consider: Professor Ioannis Koutis; NJIT Wulver; Anders Borum / Secure ShellFish; author's mother. | HIGH | Stage 4. |
| J35 | CloudRift: include as **in-kind computational/tool support**; do NOT call it conventional funding; CloudRift models were project-associated; no dollar amount needed. | HIGH | Stage 4. |
| J36 | Cohere: do NOT claim Cohere funded/supported this specific paper based on current provenance. | HIGH | Stage 4. |
| J37 | Final AI declaration must distinguish manuscript-preparation (ChatGPT product-level, OpenAI Codex, Google Gemini) from research/software-process (Codex, CloudRift-hosted models, Gemini where project-associated). | HIGH | Stage 4. |
| J38 | Verify current Elsevier AI disclosure language/placement before applying. | HIGH | Stage 2/4. |

### K. Previous submission disclosure

| ID | Issue | Priority | Next action |
|---|---|---|---|
| K39 | Prepare cover-letter wording: an earlier conference version was submitted to LLM 2026; it was withdrawn before publication; the present manuscript is substantially revised/reframed; there is no simultaneous submission. | MEDIUM | Stage 8. |

---

## 7. Closed / Do-Not-Reopen Items

**DO_NOT_REOPEN_WITHOUT_NEW_EVIDENCE:**

- no new broad trace experiment;
- no new adaptive-selector training;
- no requirement to reproduce every modern adaptive scheduler as a direct baseline;
- no need to rerun the full historical test suite merely to green the six environment-dependent UUM failures;
- no Git-history secret purge (key never entered Git);
- no need to rotate/revoke the CloudRift key for repository-exposure reasons;
- no claim that the real-vLLM experiment establishes one-step causal headroom;
- no claim that 81.9% applies to all scheduler decisions.

This section exists to stop future agents from reopening already-settled work.

---

## 8. Repository Map

| Path | Purpose | Classification |
|---|---|---|
| `paper/llm2026/` | Historical LLM 2026 manuscript package (source, PDF, figures, scripts). Also the working location of the current FGCS `main.tex` on the revision branch. | HISTORICAL_MIXED |
| `paper/llm_scheduler_adaptation_causal_headroom.pdf` | Current FGCS author-review PDF (on `main`). | CURRENT_SUPPORTING |
| `src/` | Library code: simulator, policies, DSL, selector, workloads. | ACTIVE_CURRENT |
| `scripts/` | Experiment runners, analysis scripts, maintenance tools. | ACTIVE_CURRENT |
| `configs/` | YAML/JSON experiment and calibration configs. | CURRENT_SUPPORTING |
| `experiments/` | Committed experiment artifacts and curated provenance (principal FGCS evidence lives here). | CURRENT_SUPPORTING |
| `data/public_trace_corpus_v1/` | Derived public-trace corpus manifests/schema/stats (raw records excluded). | EXTERNAL_OR_DERIVED |
| `docs/current/` | Current status, this roadmap, resumepoint docs. | ACTIVE_CURRENT |
| `docs/` | Roadmap, design docs, historical audits. | HISTORICAL_MIXED |
| `results/` | Local generated outputs; gitignored except selected provenance. | LOCAL_GENERATED |
| `REPRODUCIBILITY.md` | Reproducibility entry point + FGCS reproducibility matrix. | CURRENT_SUPPORTING |

---

## 9. Canonical Evidence Pointers

Exact repository paths (verified tracked in Git) for the paper's principal
claims. A new agent should jump directly to these instead of searching:

- **Native production-derived replay (abundance-null):**
  `experiments/industry_realism_action_opportunity_phase_a_v1/`
  - `PHASE_A_NATIVE_REPLAY_REPORT_V1.md`
  - `PHASE_A_POLICY_DISAGREEMENT_SUMMARY_V1.csv`
  - `PHASE_A_RESULT_SUMMARY_V1.json`
- **Pressure-transition (Phase B V2):**
  `experiments/industry_realism_action_opportunity_phase_b_v2/`
  - `PHASE_B_V2_PRESSURE_REPORT.md`
  - `PHASE_B_V2_TRANSITION_MAP.csv`
  - `PHASE_B_V2_RESULT_SUMMARY.json`
- **Fresh latency headroom (causal confirmation):**
  `experiments/fresh_production_latency_headroom_confirmatory_v1/`
  - `FRESH_LATENCY_CAUSAL_RESULT_V1.json`
  - `FRESH_LATENCY_BOOTSTRAP_V1.json`
  - `FRESH_LATENCY_CONFIRMATORY_REPORT_V1.md`
  - `OVERLAP_AUDIT_V1.json` (fresh window non-overlap proof)
  - `FRESH_SUPPORT_RESULT_HASHES_V1.json`
- **Causal state / action manifests:**
  - `experiments/sbs_override_fresh_confirmatory_corpus_v1/fresh_state_manifest.csv`
  - `experiments/sbs_override_fresh_id_confirmatory_terminal_label_v1/provenance_freeze_1299925/`
    (Wulver job 1299925 source provenance, `PROVENANCE_FREEZE_1299925.md`, `local_sha256.txt`)
  - `experiments/sbs_override_fresh_id_confirmatory_v2/stage_b/one_shot_20260919/`
    (`ONE_SHOT_CONFIRMATORY_REPORT_V2.md`, `CONFIRMATORY_RESULT_V2.json`)
- **Bootstrap analysis:**
  - `experiments/fresh_production_latency_headroom_confirmatory_v1/FRESH_LATENCY_BOOTSTRAP_V1.json`
  - `experiments/industry_realism_causal_headroom_phase_d_v1/PHASE_D_BOOTSTRAP_SUMMARY_V1.json`
- **ANWG objective-sensitivity audit:**
  `experiments/decision_criticality_terminal_anwg_v1/` (`summary.json`, `DESIGN_FROZEN.md`)
- **Real-vLLM validation:**
  `experiments/real_vllm_pressure_action_validation_v1/`
  (`REAL_VLLM_VALIDATION_RESULT_V1.json`, `REAL_VLLM_INFRASTRUCTURE_AUDIT_V1.json`, `PREREGISTRATION_V1.json`)
- **Figure-generation scripts:**
  `paper/llm2026/scripts/` (e.g., `plot_fgcs_figures.py`)
- **Current manuscript source (revision branch):**
  `paper/llm2026/main.tex` (on `revise/fgcs-prior-reviewer-risk-closure-20260920`)
- **Current review PDF (default branch):**
  `paper/llm_scheduler_adaptation_causal_headroom.pdf` (on `main`)
- **Reproducibility matrix / documentation:**
  `REPRODUCIBILITY.md`, `docs/current/FGCS_REPRODUCIBILITY_MATRIX_V1.md`
- **Public trace corpus (derived):**
  `data/public_trace_corpus_v1/manifest.json` (+ `schema.json`, `source_coverage.csv`, `distribution_stats.json`)

---

## 10. Potentially Confusing Historical / Current Docs

| Path | Classification | Action |
|---|---|---|
| `README.md` "## Manuscript" section | POTENTIALLY_CONFUSING | Updated to point to the active FGCS manuscript and mark LLM 2026 as withdrawn predecessor. |
| `docs/README.md` (index) | POTENTIALLY_CONFUSING | One-line pointer added to this canonical FGCS roadmap. |
| `paper/llm2026/README.md` | HISTORICAL_ACCURATE | Left unchanged (documents the historical llm2026 package). |
| `docs/current/RESUME_HERE.md` | CURRENT_BUT_STALE | Not edited (research-ops entry point, not submission status); superseded for submission state by this roadmap. |
| `docs/PROJECT_MAP.md` | HISTORICAL_ACCURATE (research roadmap) | Not edited; this file is the submission authority. |

---

## 11. Ordered Submission Roadmap

- **Stage 0 — Repository / Handoff:** DONE (this query).
- **Stage 1 — Resolve factual/numeric blockers:** PENDING.
  - Table 3 window-count discrepancy (A1); any similar exact consistency issue found during verification.
- **Stage 2 — Verify FGCS official requirements:** PENDING.
  - Guide for Authors, template, page/abstract/style/highlights requirements (B3–B10, J38).
- **Stage 3 — Apply manuscript scientific/editorial corrections:** PENDING.
  - terminology, abstract, internal phase labels, conclusion, equations, definitions, reference correctness (A2, C, D, E, H, I).
- **Stage 4 — Apply acknowledgments/disclosures:** PENDING.
  - Koutis, Wulver, mother, ShellFish, CloudRift, AI declaration, competing-interest consistency (J, K).
- **Stage 5 — Visual refinement:** PENDING.
  - Figure 5, Figure 2, Table 1, Table 2, Table 3, table row spacing, full PDF inspection (F, G).
- **Stage 6 — FGCS format conversion:** PENDING.
  - Elsevier/FGCS template, page count, final typography/layout (B5–B6).
  - *Ordering note:* official template verification (Stage 2) and conversion
    (Stage 6) should logically **precede** final visual refinement at print size;
    therefore perform coarse visual fixes in Stage 5 only as cheap as is safe,
    and do the definitive figure/table inspection at actual FGCS print size in
    Stage 6. If template verification shows large layout changes, re-do Stage 5
    items after Stage 6.
- **Stage 7 — Final integrity audit:** PENDING.
  - repository/manuscript consistency, citations, numbers, figures/tables, secrets, duplicate-submission disclosure, direct repository URL, PDF visual QA.
- **Stage 8 — Submission package:** PENDING.
  - highlights, cover letter (incl. withdrawn-predecessor disclosure), source archive, declarations, metadata, any supplementary files.
- **Stage 9 — Author approval and integration:** PENDING.
  - author visually approves final PDF; merge/integrate approved revision; publish a clearly named final PDF; optional release/tag only after approval.

---

## 12. Next Agent Instructions

1. **Read this roadmap first.** It is the single source of truth for the FGCS submission.
2. **Do not start new experiments.** `NEW EXPERIMENTS REQUIRED BEFORE SUBMISSION = NO`.
3. **Do not merge branches yet.** The revision branch stays separate until author approval (Stage 9).
4. **Do not trust old "FINAL" labels** in historical LLM 2026 docs as current submission status. They are HISTORICAL_ONLY.
5. **Verify actual Git state before changes** (`git fetch --prune`, `git status`, `git worktree list`).
6. **The next substantive task is the Table 3 numeric discrepancy / factual consistency check** (Stage 1, item A1). Verify against the canonical artifact before changing anything; do not guess.
7. **Work one issue at a time.**
8. **Preserve the one-query-at-a-time workflow with the user.**
9. **After every substantive change:** build the PDF, verify affected numbers, record the branch/SHA, and update this roadmap.
