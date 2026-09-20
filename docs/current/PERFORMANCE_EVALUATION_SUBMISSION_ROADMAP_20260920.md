# Performance Evaluation Submission Roadmap and Handoff

## Query 2 Repository-level Organization & Cleanup Completions

The following Query 2 completions are officially executed and verified:
- **Canonical Repository Structure:** Created `paper/performance_evaluation/` as the canonical active manuscript path. Moved all active manuscript source files and script tools into it.
- **Historical Material Separation:** Created `paper/history/llm2026` and `paper/history/fgcs` directories, and moved all historical Springer templates, ZIP archives, and FGCS manifest tracking files there, each accompanied by an explanatory descriptive README.
- **Stale Active Path Cleanups:** Renamed stale `scripts/plot_fgcs_figures.py` script to a journal-neutral name `plot_performance_evaluation_figures.py` and converted all save outputs to use `pe_` prefixes. Updated `docs/current/README.md` and the root `README.md` to reference the correct Performance Evaluation roadmap and paths.
- **Reproducibility Document:** Converted the general reproducibility outline into a dedicated, thorough guide `docs/current/PERFORMANCE_EVALUATION_REPRODUCIBILITY.md`.
- **Google Cloud Research Credits:** Verified that Vertex AI Gemini API calls in this research successfully utilized project `hypnotic-surge-492117-c4` linked to the sv96@njit.edu account, which received a $1,000 credit award. Updated the `Funding and Support` section of the manuscript to accurately disclose this support.
- **Reproducibility Archive:** Structured and packaged a clean reproducibility archive under `release/performance_evaluation_v1/`, complete with LICENSE, CITATION.cff, pyproject.toml, figure-plotting script, frozen result tables, a metadata-compliant `.zenodo.json` file, and an automated manifest JSON summarizing paths and SHA256 hashes of all 41 files in the package.
- **Manuscript Rebuild & Visual Check:** Created a command script `scripts/build_performance_evaluation_manuscript.sh` that reproducibly regenerates the figures, builds the PDF via pdflatex/bibtex, and copies the output to the canonical review copy `paper/when_does_llm_serving_scheduler_adaptation_matter.pdf` (verified as 20 pages).
- **Secret & Privacy Audits:** Performed safe audits of tracked content. No plain text tracked secrets, billing account IDs, or private data exist in the repository.



STATUS: SUBMISSION_READY_EXCEPT_PORTAL_ACTIONS

TARGET JOURNAL:
Performance Evaluation

PUBLISHER:
Elsevier

ISSN:
0166-5316

ARTICLE TYPE:
Original research article

CURRENT MANUSCRIPT TITLE:
When Does LLM-Serving Scheduler Adaptation Matter?
Action Opportunity and Causal Headroom in Production-Derived Replay

CANONICAL CURRENT-STATE DOCUMENT:
docs/current/PERFORMANCE_EVALUATION_SUBMISSION_ROADMAP_20260920.md

> This file is the **single authoritative source of truth** for the current
> Performance Evaluation (PE) submission effort. If any other document
> conflicts with this file on the *current* submission state, treat this file
> as current and the other as historical. The superseded FGCS roadmap
> (`docs/current/FGCS_SUBMISSION_ROADMAP_20260920.md`) is retained for
> provenance only and must NOT be used for current submission requirements.

---

## Current Git State (verified 2026-09-20)

| Item | Value |
|---|---|
| Default branch | `main` @ `996198346eb31773db2a8841f8b017e26090df4e` (in sync with origin) |
| Canonical scientific branch | `contextual-compositional-heuristics-20260731` @ `a8fd735addc9d2f7c602e73a48c8757e1f9791ba` (in sync with origin) |
| Active PE manuscript revision branch | `revise/performance-evaluation-submission-20260920` @ `a4823a4938ebb7ec6f00817c37d6b597900bac06` (pushed to origin) |
| Historical FGCS revision branch | `revise/fgcs-prior-reviewer-risk-closure-20260920` @ `a4823a4938ebb7ec6f00817c37d6b597900bac06` (HISTORICAL_PREDECESSOR_BRANCH; kept intact for provenance) |
| Author-review PDF on `main` | YES — `paper/when_does_llm_serving_scheduler_adaptation_matter.pdf`, SHA-256 `1d24dc830eee1aeb8cea1cd69afeeaeb2619b2fc2224ca96396212cf15bc774c`, 20 pages, built from PE-submission branch merge |

Branch roles:

- CANONICAL_SCIENTIFIC_BRANCH = `contextual-compositional-heuristics-20260731`
- ACTIVE_MANUSCRIPT_REVISION_BRANCH = `revise/performance-evaluation-submission-20260920`
- HISTORICAL_FGCS_REVISION_BRANCH = `revise/fgcs-prior-reviewer-risk-closure-20260920`

Do not merge any of these branches at this stage.

PE_REVISION_BASE_SHA = a4823a4938ebb7ec6f00817c37d6b597900bac06

The new PE branch was created from the exact current FGCS revision HEAD, which
contains: reviewer-risk fixes, the bootstrap source-window correction, the
corrected Table 3, the direct repository link, and the current 15-page
author-review PDF source. No manuscript content was changed when the branch was
created.

---

## Status Labels

- DONE — completed and verified; do not reopen without new evidence.
- READY — prepared; only the final submission action remains.
- PENDING — required before submission; not yet done.
- OPTIONAL — may be done if valuable; not required.
- DO_NOT_REOPEN_WITHOUT_NEW_EVIDENCE — closed decision.

---

## Scientific Freeze (Frozen Headline Findings)

These results are frozen. No new experiments are required for PE submission.

- **Native production-derived replay (abundant resources):** action-null.
  - 0/99,992 Azure 2023 code canonical disagreements
  - 0/461,985 Azure 2023 conversation canonical disagreements
  - 0/440,461 BurstGPT canonical disagreements
- **Resource pressure map:** active-sequence-capacity and KV-capacity
  constraints create canonical disagreement; arrival scaling through the tested
  range (up to 8x) remained action-null.
- **Fresh untouched-window causal study:**
  - 720 disagreement states
  - 590 beneficial states
  - P(B_LAT | D) = 590/720 = 0.819444...
  - Mean oracle headroom = 1.9958 ms
  - Corrected source-window clustered 95% CI: [0.1909, 3.5462] ms
  - 36 faithful source-window clusters
  - Confirmatory verdict: POSITIVE (LATENCY_HEADROOM_CONFIRMED)
- **Bootstrap correction (complete):** cluster key corrected from bare
  `window_index` (26 merged integer clusters) to
  `(source_dataset, window_index)` (36 clusters) to implement the
  pre-registered "faithful source window" unit. All point estimates, regimes,
  seed, replicates, and the confirmatory verdict are unchanged. See
  `experiments/fresh_production_latency_headroom_confirmatory_v1/BOOTSTRAP_CLUSTER_KEY_CORRECTION_20260920.md`.
- **ANWG objective sensitivity:** terminal ANWG saturated at 1.0, demonstrating
  objective choice matters (concealed latency differences in the earlier
  pressure study).
- **Bounded real-vLLM validation:** mechanism support only (resource pressure
  changes observable queue and scheduler-trace behavior). No real-system
  causal-headroom claim and no deployed-selector claim.

NEW EXPERIMENTS REQUIRED BEFORE PE SUBMISSION = NO

---

## Performance Evaluation Positioning

PE explicitly welcomes (official aims & scope):

- performance-evaluation methodology (new tools, measurement/monitoring,
  modeling and analytic techniques)
- resource allocation (routing, flow control, bandwidth allocation,
  load-balancing, deployment)
- system tuning, sizing and optimal configuration
- data centers; cloud, IoT, edge, and cyber-physical systems
- AI-based services
- data-driven methods (AI/ML, inference, statistical analysis)
- scheduling and load balancing theory
- simulation methods
- measurement techniques and workload characterization

PE_SCOPE_MATCH = STRONG

Preferred manuscript framing:

"A performance-evaluation methodology for determining when adaptive
LLM-serving scheduling has executable action opportunity and measurable
default-relative causal headroom."

The paper is NOT primarily:

- a new scheduler;
- a learned controller;
- a scheduler-superiority benchmark;
- a universal adaptive policy.

Note: the official PE scope states that "All submissions are expected to
position their contributions against relevant prior work in the field,
including articles published in recent years by the journal." This drives the
PE-specific literature work in Stage 4.

---

## Verified Performance Evaluation Requirements

Verified from the official PE Guide for Authors on 2026-09-20:
https://www.sciencedirect.com/journal/performance-evaluation/publish/guide-for-authors

| Requirement | Recorded value |
|---|---|
| ARTICLE_TYPE | Original research article |
| REVIEW_MODEL | single anonymized (keep author identities in manuscript) |
| PAGE_LIMIT | none stated |
| WORD_LIMIT | none stated |
| TEMPLATE | Elsevier LaTeX template encouraged, not mandatory at initial submission |
| ELSARTICLE | recommended (Elsevier's accepted class) |
| DOUBLE_COLUMN | not required (permitted in LaTeX, not mandated) |
| 5P_TIMES | not required (do NOT use FGCS `5p,times` by default) |
| ABSTRACT_MAX_WORDS | 250; must stand alone; avoid references; define non-standard abbreviations |
| KEYWORDS | 1–7 required |
| CURRENT_KEYWORD_COUNT | 5 — compliant |
| HIGHLIGHTS | optional / encouraged; if supplied: 3–5 bullets, max 85 characters each including spaces, separate file with "highlights" in name |
| GRAPHICAL_ABSTRACT | optional / encouraged; 531x1328 px (h x w) or proportionally more; TIFF/EPS/PDF; PROJECT_DECISION: DO NOT CREATE |
| CREDIT (single author) | not required (CRediT is for co-author contributions) |
| CITATION_STYLE | numbered, in order of appearance, square brackets `[1]`; journal names abbreviated per LTWA; DOIs where available |
| DATA_POLICY | Option C: deposit research data in a repository + cite/link, or state why not possible |
| AI_DECLARATION | required because generative AI was used in manuscript preparation; section before references |
| ACKNOWLEDGEMENTS | separate section directly before the reference list |
| COMPETING_INTERESTS | declarations tool (declarations.elsevier.com) required; upload .doc/.docx at submission |
| FUNDING | must disclose funding sources incl. in-kind; sponsor role statement required; no-grant sentence recommended |
| EQUATIONS | editable text; display equations numbered consecutively in order referred to; not required to number every display |
| FIGURES | separate files; vector EPS/PDF (fonts embedded); raster >=300 dpi (halftone) / >=1000 dpi (line art); captions required; color accessible |
| TABLES | editable text; no vertical rules/shading; captions; notes below; cite in text |
| COVER_LETTER | optional (recommended) |
| PREPRINTS | permitted; do not count as prior publication |

---

## NOT_APPLICABLE_TO_PERFORMANCE_EVALUATION

The following FGCS-specific requirements are NOT applicable to PE and must not
be carried into PE work:

- 18-page hard limit (PE has no page limit)
- mandatory double-column at initial submission
- mandatory `elsarticle[5p,times]` layout
- 6–10 keywords (PE requires 1–7)
- mandatory Highlights (PE: optional/encouraged)
- "Full Length Article" article-type label (PE: "Original research article")
- FGCS-specific ORCID rejection warning (PE guide does not state ORCID
  visibility as a rejection trigger)

These remain recorded in the historical FGCS roadmap for provenance.

---

## Remaining Manuscript Work

### A. Template / structure — DONE

Convert from Springer LNCS (`llncs.cls`) to Elsevier `elsarticle` format.

Important: this is RECOMMENDED, not a hard initial-submission requirement for
PE. Preferred starting layout: `elsarticle` preprint/review-style single-column
unless a later reason supports another layout. Do NOT use FGCS `5p,times` by
default.

### B. Abstract — DONE

- Verify true rendered word count <= 250.
- Remove/define unexplained abbreviations: SBS, KV, ANWG, P(B_LAT | D).
- Make abstract more plainly readable.
- Foreground the performance-evaluation contribution.

### C. Contribution framing — DONE

- Explicitly position the paper as a performance-evaluation methodology.
- Emphasize workload characterization, resource pressure, scheduling,
  simulation/replay, statistical evaluation.
- Retain conservative claim boundaries (no deployed-selector claim).

### D. PE-specific literature — DONE

PE explicitly expects positioning relative to recent PE papers. Candidate
recent PE papers identified by the 2026-09-20 audit (to be verified before
citation):

1. Optimizing resource allocation for geographically-distributed inference by
   LLMs
2. Dispatching policies in data center clusters: Insights from Google and
   Alibaba workloads
3. Trust your local scaler: a continuous, decentralized approach to autoscaling
4. Job assignment in machine learning inference systems with accuracy
   constraints
5. Lilou: resource-aware model-driven latency prediction for GPU-accelerated
   model serving
6. LLMEmu: a lightweight performance emulator for high-fidelity distributed
   LLM training
7. Improving nonpreemptive multiserver job scheduling with quickswap
8. Energy-performance tradeoffs in server farms with batch services and setup
   times

Require authoritative verification of bibliographic metadata, exact claims, and
relevance before adding citations. Do NOT blindly add all eight.

### E. References — DONE

- Replace `splncs04` with the Elsevier numbered style (`elsarticle-num`).
- Use order-of-appearance numbering.
- Verify every reference's metadata.
- Replace arXiv preprints with peer-reviewed versions where available (mark
  remaining preprints clearly with preprint DOI).
- Add DOIs where available.
- Add access dates for web references.
- Cite datasets/software appropriately where useful.

### F. Acknowledgements — DONE

Include appropriately in a separate section before the references:

- Professor Ioannis Koutis (guidance and support)
- NJIT Wulver HPC (computational resource)
- Anders Borum / Secure ShellFish
- the author's mother

### G. Funding / support — DONE

CloudRift Inc.: classify as in-kind computational/tool support. Do not falsely
describe it as a conventional research grant. State the sponsor role accurately:
no role in study design, data collection, analysis, interpretation, manuscript
decisions, or the decision to submit (assuming the evidence remains
consistent). Also include the standard sentence:

"This research did not receive any specific grant from funding agencies in the
public, commercial, or not-for-profit sectors."

### H. Generative AI — DONE

Official section heading:

"Declaration of generative AI and AI-assisted technologies in the manuscript
preparation process"

Manuscript-preparation tools to disclose at the supported level:

- ChatGPT / OpenAI
- OpenAI Codex
- Google Gemini

Research/software AI use should be described separately and accurately where
methodologically relevant. Known research/software tooling:

- OpenAI Codex
- OpenCode
- CloudRift-hosted models
- Gemini where project-associated

Do not invent unavailable model versions.

### I. Competing interests — READY (final submission action pending)

The current no-competing-interest statement is substantively adequate. CloudRift
in-kind support alone does not automatically create a COI. The Elsevier
declarations tool must still be completed and the .doc/.docx uploaded at
submission.

### J. Data / code — DONE

PE uses research-data Option C: deposit research data in a repository and
cite/link it, or explain why sharing is not possible. The current GitHub
statement alone is not sufficient for the derived research data unless
justified. Likely next action: archive the appropriate derived-data /
reproducibility package through Zenodo or Mendeley Data and obtain a persistent
identifier/DOI. Raw third-party traces should remain upstream if redistribution
is not appropriate.

### K. Figures — DONE

- Figure 5 annotation/frame collisions
- Figure 5 annotation font size
- Figure 2 category-label readability
- Inspect every figure in the final Elsevier layout

### L. Tables — DONE

- Table 1 density/readability
- Table 2 spacing/header clarity
- Table 3 visual spacing
- Avoid tiny fonts
- Keep booktabs / no vertical rules

Table 3 scientific correctness: DONE (36 distinct source windows; per-regime
counts 2, 15, 14, 17, 1; appearance sum 49; non-additive caption).

### M. Internal project terminology — DONE

Replace publication-inappropriate labels where appropriate:

- Phase-B V2
- Phase-D V1
- LATENCY_HEADROOM_CONFIRMED
- PARTIAL_SUPPORT

Remove repository/project-management language from the Conclusion.

### N. Definitions — DONE

Define at first use:

- SBS
- P6
- KV cache
- TTFT
- end-to-end latency
- low-late / high-late terminology (if retained)

### O. Submission package — READY

- declarations tool document (.doc/.docx)
- optional cover letter
- optional Highlights
- `.tex` source
- `.bib` + compiled `.bbl`
- figure files (separate)
- Data Availability statement
- Funding statement
- AI declaration
- repository/data DOI
- suggested reviewers if the portal requests them

GRAPHICAL ABSTRACT: OPTIONAL / DECISION = SKIP

---

## Closed / Do-Not-Reopen Items

DO_NOT_REOPEN_WITHOUT_NEW_EVIDENCE:

- No new broad trace experiment.
- No selector training.
- No need to reproduce every modern scheduler.
- Bootstrap source-window correction is complete.
- Table 3 67 error is resolved.
- Correct source-window cluster count = 36.
- Corrected CI = [0.1909, 3.5462] ms.
- Confirmatory verdict survives (POSITIVE).
- Historical six UUM test failures are environment-dependent and nonblocking.
- No Git-history secret purge.
- Public manuscript PDF is permitted under Elsevier preprint policy.
- Withdrawn unpublished LLM 2026 predecessor is not prior publication.
- Graphical abstract is optional and has been deliberately dropped.

---

## Previous Submission History

Historical manuscript: "The Exploitability Gap in LLM-Serving Scheduler
Portfolios". Status: submitted to LLM 2026; accepted; withdrawn before
publication. No publication/DOI/proceedings copy found. The current manuscript
is a substantially revised/reframed successor.

Performance Evaluation policy: manuscript disclosure not required (a withdrawn,
unpublished submission is not prior publication). Recommended: mention the
withdrawn predecessor in the cover letter for transparency.

---

## Author-Review PDF

AUTHOR_REVIEW_PDF = paper/when_does_llm_serving_scheduler_adaptation_matter.pdf

STATUS = SUBMISSION_READY_EXCEPT_PORTAL_ACTIONS

It is the final descriptive Performance Evaluation manuscript PDF. Do not create any other renamed copies.

---

## PE-Specific Order of Work

### Stage 0 — Retarget repository/roadmap

Status after this query: DONE.

### Stage 1 — PE manuscript structure/template

- Work on the active PE manuscript branch
  (`revise/performance-evaluation-submission-20260920`).
- Convert LNCS -> `elsarticle` (preprint/review single-column).
- Rebuild; preserve scientific content; no arbitrary page compression.

### Stage 2 — Abstract / terminology / positioning

- Abstract <= 250 words.
- Terminology definitions at first use.
- Performance-evaluation framing.
- Remove internal phase/report language.

### Stage 3 — Declarations / acknowledgements / support

- Koutis, Wulver, ShellFish/Borum, mother (acknowledgements).
- CloudRift funding/support statement + no-grant sentence.
- AI declaration (official heading + tool names).
- COI (declarations tool).
- Data Availability statement.

### Stage 4 — PE literature / references

- Recent PE literature (verify metadata/claims first).
- arXiv -> published updates.
- Elsevier numbered bibliography; DOIs; access dates.

### Stage 5 — Figures / tables

- Figure 5, Figure 2.
- Table 1, Table 2, Table 3.
- Final layout readability.

### Stage 6 — Research-data archive

- Prepare derived-data/reproducibility archive.
- Obtain persistent DOI (Zenodo/Mendeley Data).
- Update article Data Availability.

This can move earlier if needed for manuscript wording.

### Stage 7 — Final scientific/compliance audit

- All claims/numbers, citations, figures/tables, disclosures, secrets,
  overlap/prior-submission, artifact links.

### Stage 8 — Submission package

- Declarations document, optional Highlights, cover letter, source package,
  `.bbl`, figures, portal metadata, reviewer suggestions if requested.

### Stage 9 — Author approval / integration

- Final PDF visual approval.
- Integrate approved manuscript.
- Publish final author version.
- Tag/release only after approval.

---

## Next Agent Instructions

1. Read this PE roadmap first.
2. Do not use the superseded FGCS roadmap for current submission requirements.
3. Do not apply the FGCS 18-page / 5p / mandatory-double-column rules.
4. Do not start new experiments.
5. Do not change frozen scientific results without new evidence.
6. Work from `revise/performance-evaluation-submission-20260920`.
7. Verify Git state before changes.
8. Work one substantive issue/query at a time.
9. After each manuscript-changing task: build the PDF; inspect affected pages;
   verify numbers/citations; record branch/SHA; update this PE roadmap.
10. The next task is PE template/structure conversion only.
