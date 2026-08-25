# LLM 2026 Submission Readiness Audit

Date: 2026-08-24

Manuscript: `paper/llm2026/main.tex`

## A. Venue Compliance

- Format: official Springer LNCS template, `llncs.cls`, `splncs04.bst`.
- Page limit: LLM 2026 full/regular paper permits 12-15 one-column Springer pages including figures, tables, and references.
- Current PDF: 15 pages, within the official maximum but with no spare page margin.
- Abstract: 88 words, self-contained, no citations, no undefined abbreviations.
- Keywords: 4, within the LLM maximum of 5 and Springer 3-6 guidance.
- Acknowledgments, data/code availability, generative-AI disclosure, and competing-interest statement are present.
- Author metadata: placeholder remains and must be replaced before non-blind submission.

## B. Final Paper Thesis

LLM-serving scheduler portfolios exhibit meaningful workload-dependent complementarity and VBS/oracle headroom, including under jointly varying synthetic workload pressures. However, the tested contextual selection, routing, support-expansion, guarded-composition, and typed-synthesis mechanisms expose a persistent gap between oracle opportunity and realized deployable gain. Real-vLLM validation further shows that simulator scheduling mechanisms require engine-level semantic correspondence checks.

## C. Contribution Hierarchy

1. Portfolio complementarity and oracle headroom under controlled and joint workloads.
2. Exploitability-gap characterization using SBS/VBS methodology specialized to LLM-serving portfolios.
3. Constructive falsification beyond selection: support expansion, guarded composition, and typed synthesis fail frozen gates under tested formulations.
4. Real-serving semantic validation: direct simulator-to-vLLM mapping fails, while native vLLM token-budget control shows a reproducible tradeoff.

## D. Dataset Provenance

- External datasets actually used in final reported public-trace replay: BurstGPT and Azure 2023 LLM inference traces.
- Synthetic/project-generated workloads provide the main scientific evidence.
- Existing public Hugging Face datasets under `SoroushVahidi` are related earlier artifacts, not final manuscript source artifacts.

## E. Code/Data Availability

The manuscript includes a GitHub availability statement for `https://github.com/SoroushVahidi/llm-serving-heuristic-evolution`, plus citations to BurstGPT and Azure/Splitwise for public-trace-derived replay windows.

## F. Acknowledgments

The manuscript thanks Professor Ioannis Koutis, Anders Borum for Secure ShellFish access, and the author's mother. The Secure ShellFish access is described as workflow support, not funding.

## G. AI Disclosure

The manuscript discloses use of ChatGPT, Claude, Codex, and Perplexity AI for organization, language refinement, literature-search planning, software-development tasks, and workflow support. It states that the author reviewed and verified scientific decisions, results, interpretations, citations, and final content.

## H. Baseline Audit

The six policies are described as repository-specific, mechanism-diverse baselines. The manuscript does not claim faithful reproduction of Sarathi-Serve, VTC, WAIT, SOLA, Mooncake, or other external production systems.

## I. Citation Audit

- Bibliography uses primary or authoritative sources where available.
- Added public-trace citations: BurstGPT KDD 2025, Azure 2023 public dataset page, and Splitwise ISCA 2024.
- Citation clusters were split so that dense claims are tied to specific propositions.
- No Wikipedia/blog/SciSpace citations are used.

## J. Terminology/Definition Audit

Definitions are present for scheduler portfolio, SBS, VBS, oracle headroom, exploitability gap, ANWG, workload families, KV pressure, chunked prefill, and native token budget. Baseline names are consistently used as repository policy labels rather than external-system reproductions.

## K. Equation Audit

Displayed equations are numbered and labeled:

- Eq. (1): ANWG.
- Eq. (2): SBS.
- Eq. (3): VBS/envelope.
- Eq. (4): oracle headroom.

All are referenced in prose and define symbols locally.

## L. Figure Audit

- `paper/llm2026/figures/joint_complementarity.pdf`: vector PDF, readable at included size, no clipping observed.
- `paper/llm2026/figures/vllm_semantic_validation.pdf`: vector PDF, readable at included size, no clipping observed.
- No figure-revision request file was needed.

## M. Table Audit

- Table 1: compact six-policy mechanism matrix; readable at `\scriptsize`.
- Table 2: frozen-gate summary; dense but readable and central to the paper's argument.
- Tables avoid claiming external-system reproduction.

## N. Claim-Evidence Audit

Claims were checked against `docs/current/llm2026_claim_evidence_ledger_20260824.md`. Wording remains bounded to tested methods, evaluated workloads, and the tested vLLM 0.27.1 configuration.

## O. Number Audit

Numbers in the manuscript were cross-checked against `docs/current/llm2026_number_source_of_truth_20260824.md`; final-pass additions were recorded there.

## P. Abstract Audit

The abstract is 88 words, self-contained, no citations, no unexplained SBS/VBS/ANWG abbreviations, and includes one headline number: 0.0190 arrival-normalized weighted-goodput oracle headroom.

## Q. Author Metadata Status

`Anonymous Author(s)` and anonymous affiliation/email placeholders remain. Before final non-blind submission, the author must supply author name(s), affiliations, city/country, email, corresponding-author details, and ORCID if desired. If LLM 2026 double-blind review applies due to program-committee authorship, acknowledgments and public repository links must be anonymized for review.

## R. Funding/Conflict Status

No funding source was inferred from repository evidence. Secure ShellFish access is acknowledged as an in-kind workflow resource, not funding. The manuscript includes a no-competing-interests declaration.

## S. Page-Count Status

Current PDF is exactly 15 LNCS pages including references. It complies with the official maximum but leaves no spare margin; avoid adding prose before submission.

## T. Compile Status

`cd paper/llm2026 && tectonic --keep-logs main.tex` succeeds. Remaining warnings are nonfatal overfull/underfull boxes; no undefined citations or references were found in the final search.

## U. Reviewer-Style Weaknesses

- Main workloads are synthetic, even after joint generalization.
- Real-vLLM validation uses one vLLM version, one small model, one GPU, and one mechanism family.
- The six-policy portfolio is mechanism-diverse but not exhaustive.
- Negative adaptive results are bounded to tested formulations.
- Page budget is tight.

## V. Submission Blockers

No scientific blocker was found. The remaining pre-submission actions are author-side metadata/anonymization checks and repository/artifact-publication logistics.

## W. Final Readiness Verdict

`READY_FOR_AUTHOR_REVIEW`

