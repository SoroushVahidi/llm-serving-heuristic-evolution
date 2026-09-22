# QUERY 4 of 6 — Performance Evaluation compliance and submission-metadata finalization

Date: 2026-09-21. Base: `334c307` on `revise/peva-metadata-consistency-20260921`. No push, no Zenodo action, no experiments.
Guide for Authors source: `docs/current/PERFORMANCE_EVALUATION_SUBMISSION_ROADMAP_20260920.md` ("Verified Performance Evaluation Requirements", read from the official guide on 2026-09-20).

## 1. Preflight and baseline (before edits)

| Item | Value |
|---|---|
| HEAD / worktree | `334c307`, clean |
| Pages / tables / figures | 34 / 7 (all `tabular`) / 4 |
| Manuscript tests | 79 pass (73 `test_manuscript_*` + 6 `test_peva_prepackage_readiness`) |
| Claim manifest | 80 claims, 0 problems |
| Undefined citations/references | 0 |
| `main.tex` SHA-256 (334c307) | `4a95e197bf58cb3a646efcdbf82d1ae95654c10e45f7a3fe762042b1ef568b08` |
| PDF SHA-256, tracked review copy | `8b6a8c961ed4a97734ad779730468565bab29d3d3d7f60a77750521e6d5767fb` |
| Title | When Does LLM-Serving Scheduler Adaptation Matter? Action Opportunity and Causal Headroom in Production-Derived Replay |
| Keywords (6) | LLM serving; request scheduling; performance evaluation; workload replay; resource contention; causal evaluation |

Abstract word count: **247** by whitespace split (MS-Word style), 247 by the words in the built PDF, 247 by the repository test's own counter. Strict tokenizers that also split hyphenated compounds and en-dash ranges give 254-259; no journal count does this, so no trimming was done. If the portal shows a higher number, trim the redundant "scheduler" in the first sentence and similar single words, not content.

After edits: `main.tex` `96a158d7268fd0cc...`, rebuilt PDF (34 pages, 538063 bytes) SHA-256 prefix `f468c51bb4e336c8`. Rebuilt PDFs differ byte-wise between builds only by timestamps and the `\today` footer date. The final hashes are for Query 5.

## 2. Compliance matrix

| # | Requirement | Status | Evidence |
|---|---|---|---|
| 1 | Article title | PASS | Unchanged; no compliance defect |
| 2 | Author name | PASS | Soroush Vahidi |
| 3 | Full affiliation and postal address | PASS | Dept. of Computer Science, Ying Wu College of Computing, New Jersey Institute of Technology, University Heights, Newark, NJ 07102, USA |
| 4 | Country | PASS | USA |
| 5 | Corresponding-author designation | PASS | `\corref{cor1}`, rendered footnote "Corresponding author." |
| 6 | Corresponding-author email | PASS | `sv96@njit.edu` (rendered "Email address: sv96@njit.edu (Soroush Vahidi)") |
| 7 | Abstract <= 250 words, no citations | PASS | 247; no `\cite`, no brackets; LLM and KV defined on first use |
| 8 | 1-7 keywords | PASS | 6, English, each <= 3 words |
| 9 | Numbered sections | PASS | Sections 1-8 numbered; declarations and Appendix A are unnumbered by design |
| 10 | Editable tables | PASS | 7 LaTeX `tabular`/`tabularx`; none is an image |
| 11 | Figure captions | PASS | 4 vector PDFs, each with a caption |
| 12 | Acknowledgements immediately before references | PASS | Last section before `\bibliography` |
| 13 | Funding disclosure | PASS | See section 5 below |
| 14 | Competing-interest declaration | FIXED | Heading case aligned to Elsevier's "Declaration of competing interest"; Elsevier standard sentence retained |
| 15 | CRediT statement | PASS | See section 6 below |
| 16 | Data/code availability | PASS as written, **conditional on Query 5 pushes** | See section 7 |
| 17 | GenAI disclosure | FIXED | Added "verified the outputs" and "No AI tool is listed as an author"; test-pinned phrases kept |
| 18 | Preprint labeling | PASS | Footer "Preprint submitted to Elsevier" is the elsarticle default; the paper is not posted as a preprint. Cited arXiv works are labelled "arXiv preprint" |
| 19 | Reference/citation consistency | PASS; 5 DOIs added | 38 cited = 38 in `.bib`, none uncited or undefined |
| 20 | Editable LaTeX source | PASS | `main.tex`, `main.bbl`, `references.bib`, 4 vector figures; source ZIP builds |
| 21 | Highlights (encouraged, not mandatory) | PASS, verify-only | See section 9 |
| 22 | Graphical abstract | NOT_APPLICABLE | Optional; not supplied |
| 23 | Elsevier declarations-tool document | NOT_APPLICABLE here | Portal step; the text is in `submission/declarations.md` |

BLOCKERs: none for the manuscript text. One external-state item (repository push) is listed for Query 5.

## 3. Title page

Verified in the rendered PDF: author, four-line affiliation, `Newark, NJ 07102, USA`, `*` corresponding-author mark, email. No obsolete affiliation and no placeholder (grep over the manuscript and `submission_docs/`). The cover letter carries the short form "New Jersey Institute of Technology, Newark, NJ, USA", which is acceptable for a letter.

## 4. Abstract and keywords

The canonical primary result is unchanged: 720 states, 590 (81.9%), mean 2.00 ms, 95% window-clustered CI 0.19-3.55 ms. "Simulated" and "simulator" qualify every latency claim, and the vLLM probe is stated not to validate the simulated latency effect. The abstract does not mention the reserve analysis and does not contradict it ("reference-conditional"). Keywords are compact and indexable; "causal evaluation" matches the title's "Causal Headroom". Kept.

## 5. Funding, acknowledgements

* Wording is correct: in-kind support; "Google Cloud Research Credits Program (USD 1,000 in cloud credits)" (source: roadmap, credit award verified 2026-09-20); CloudRift "computational/tooling support"; "No monetary research grant was received from these organizations"; no grant numbers; no promotional language; both providers stated to have had no role in design, collection, analysis, interpretation, manuscript decisions, or the decision to submit.
* **Author to confirm:** the manuscript and the roadmap say "CloudRift Inc."; this query's text says "CloudRift AI". No local evidence settles the legal name, so nothing was changed.
* The roadmap also suggested Elsevier's generic no-grant sentence. It was not added: the author must confirm no other grant exists.
* Acknowledgements placement is correct (last, before references). "the author's mother" is neutral and unchanged.

## 6. CRediT

Ten roles listed: Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data curation, Visualization, Writing - original draft, Writing - review & editing. No Supervision, Funding acquisition, Resources or Project administration claimed. Matches `credit_authorship_statement.txt`. The author still has to confirm, as the roles cannot be verified from the repository.

## 7. Data and Code Availability: sentence-by-sentence audit

| Sentence | True today? |
|---|---|
| Code, derived artifacts, configs, reproducibility materials are in the GitHub repository | Yes for `main` (2bdf4b7 includes e6fe60a, the commit the Zenodo record links to). Repo is **PUBLIC** (`gh repo view`, 2026-09-21) |
| v1.1.0 package archived on Zenodo [28] | Yes: record 22866983, published 2026-09-21, open, MIT |
| Archive contains confirmatory artifacts, robustness outputs, continuation shards, figure code, claim manifest, simulator, analysis code | Yes: checked against `MANIFEST.json` (289 shard files under `provenance_archive/.../continuation_shards/`, robustness dir, `src/`, `scripts/`) |
| Reserve analysis and denominator completion post-date that archive | Yes: the archive tree has no reserve material |
| Their protocol, results and provenance are available in the GitHub repository under `experiments/reference_reserve_sensitivity_v1/` | **NOT YET TRUE.** `origin/main` = 2bdf4b7 has no such directory. The remote reserve branch is at ec19963 (protocol/implementation/denominator freeze only, no results). The results commit 327736b and the manuscript branch (10 commits ahead of main) are unpushed |
| Raw third-party traces are not redistributed | Yes |

### Query-5 action list (availability)

1. Push `experiment/reference-reserve-sensitivity-20260921` (adds 327736b, one commit, keeps the protocol -> freeze -> results history public).
2. Push `revise/peva-metadata-consistency-20260921`. Its merge base with `main` is 2bdf4b7, so `main` can be fast-forwarded to it. Its reserve directory is byte-identical to the experiment branch tip plus `PROVENANCE_ON_MANUSCRIPT_BRANCH.md`; the two branches carry the same results under different hashes (cherry-pick d43e572).
3. Put the reserve directory on the default branch (fast-forward or PR of the manuscript branch). Optionally add a tag such as `performance-evaluation-submission-v1`.
4. Verify anonymously afterwards: `git ls-remote origin main` equals the final commit; `curl -sI https://raw.githubusercontent.com/SoroushVahidi/llm-serving-heuristic-evolution/main/experiments/reference_reserve_sensitivity_v1/results_v1/reserve_090/result_summary.json` returns 200; the same for `paper/performance_evaluation/main.tex`.
5. Repository visibility: **no change needed** (already public). The auto-memory note that it is private is outdated.
6. Zenodo: **a new version is not required.** The sentence is truthful once step 3 is done because it says "GitHub repository only". A new version is only desirable for permanence; if one is created, the sentence, `references.bib` [28], README, CITATION.cff, the cover letter's DOI mention and `scripts/build_performance_evaluation_submission.py` all need the new version DOI.
7. Optional metadata-only edit on record 22866983 (no new version): its description still says "preregistered fresh causal study", while the manuscript uses "pre-specified". The archived tree keeps the historical file names, as the paper README documents.
8. Manuscript wording: unchanged if step 3 is done. If only a branch is pushed, add the branch or tag name to the sentence.

## 8. DOI and archive consistency

| DOI | Meaning | Where used |
|---|---|---|
| 10.5281/zenodo.22866983 | v1.1.0 version DOI (published 2026-09-21) | manuscript [28], README, CITATION.cff, cover letter, package scripts |
| 10.5281/zenodo.22865293 | concept DOI | manuscript [28] (as "concept DOI"), CITATION.cff, cover letter |
| 10.5281/zenodo.22865294 | superseded v1.0.0 | historical contexts only (README, readiness doc, archived v1.1.0 `CITATION.cff` and `.zenodo.json`) |

The manuscript cites the version DOI plus the concept DOI; no file presents 22865294 as current. The v1.1.0 archive predates the reserve experiment (statement 4 above is accurate). Its internal `CITATION.cff` still lists v1.0.0, which the paper README already documents as a known limitation.

## 9. References and preprints

* Preprints labelled "arXiv preprint": P-PAS (2608.15171), Nixon et al. (2608.13573), Jaillet et al. (2502.07115). arXiv's API shows no journal reference or DOI for any of them today (2026-09-21).
* Published venues in use: Bari = PACMMAC 9(3), doi 10.1145/3771574; LMetric = OSDI 26; Strata = OSDI 26; MorphServe = MLSys 8 (2026); Beyond Prediction = ICML 2026 (arXiv journal_ref agrees).
* **FIXED:** five DOIs added, each checked against Crossref for title, venue, volume, pages and year: vLLM (SOSP '23) 10.1145/3600006.3613165; Rice 1976 10.1016/S0065-2458(08)60520-3; Gomes-Selman 2001 10.1016/S0004-3702(00)00081-3; SATzilla 10.1613/jair.2490; AutoFolio 10.1613/jair.4726. The remaining entries without a DOI are USENIX, MLSys, ICLR and ICML papers or arXiv preprints, which have none.
* Seven pre-existing BibTeX "empty pages" warnings are for conference entries without page ranges.

## 10. Back matter

Rendered order: Conclusion, Appendix A, GenAI declaration, CRediT, competing interest, Funding, Data and Code Availability, Acknowledgements, References. The only journal constraint recorded in the guide is Acknowledgements directly before References, which holds. No reordering.

## 11. Highlights and cover letter (verify only, nothing regenerated)

* `submission_docs/` and `submission/` copies are byte-identical (md5 `b3f06b09...`): 5 bullets, 58-67 characters (limit 85), consistent with the frozen manuscript. Bullet 3 says "latency headroom" without "simulated"; not a contradiction, left as is.
* **Watch:** `paper/performance_evaluation_highlights.txt` (repo root of `paper/`, linked from `docs/current/README.md`) is a different, older text with markdown "- " prefixes, 76-83 characters, "82%" and "oracle". It does not contradict the manuscript but is not the `submission/` text. Confirm which file was uploaded.
* Cover letter: title and journal correct; numbers match (about one million states, 720 fresh Azure states, median 0.20 ms); no reserve assertion, so no contradiction. Its "archived in the published v1.1.0 Zenodo dataset" refers only to items that are in the archive.

## 12. Submission-system inventory

| Field | Value |
|---|---|
| Title | as in section 1 |
| Article type | Original research article |
| Author / corresponding author | Soroush Vahidi, sv96@njit.edu |
| Affiliation | as in section 3 |
| Abstract | text of `\begin{abstract}` (247 words) |
| Keywords | six, section 1 |
| Funding | Funding and Support section (in-kind; no grant number) |
| Competing interests | none declared (Elsevier standard sentence) |
| Data availability | Zenodo 10.5281/zenodo.22866983 + public GitHub; reserve materials GitHub only, after Query 5 push |
| Code availability | same |
| Preprint status | not posted as a preprint |
| Portal-only, not filled here | ORCID, suggested reviewers, declarations-tool upload |

## 13. Source-ZIP check

Extracted into `mktemp -d /tmp/peva-submission-check.XXXXXX` and compared: ZIP `main.tex`, `main.bbl`, `references.bib` were identical to the canonical files at 334c307. The directory was removed by exact path after printing it and checking the prefix. **The ZIP and `submission/` are now stale** because of the Query 4 edits.

## 14. Edits made and freeze check

| File | Change |
|---|---|
| `main.tex` | GenAI sentence extended; competing-interest heading case |
| `references.bib` | 5 verified DOIs |
| `main.bbl` | regenerated (DOI lines only) |
| `scripts/build_performance_evaluation_submission.py` | heading lookup case, to match |

Versus 334c307: added/removed numeric tokens in `main.tex` = none; figures identical; no table, equation or `\includegraphics` line touched; Eq. (4), CI and reserve results unchanged. After edits: 34 pages, 79 tests pass, 80 claims 0 problems, 0 undefined citations or references.

## 15. Query-5 required actions (consolidated)

1. Availability push and verification: section 7, steps 1-4 and 8.
2. Layout and pagination (Query 5 scope); rebuild after any `main.tex` change.
3. Then run `scripts/build_performance_evaluation_submission.py`. Caution: it does `shutil.rmtree` on `paper/performance_evaluation/submission/` and recreates it, copying highlights, cover letter, CRediT and reviewers files back from `submission_docs/`. Confirm those four files exist there and `git status` is clean first.
4. Verify the new source ZIP in a `mktemp` directory (script does this itself); record ZIP SHA-256, `main.tex` SHA-256, PDF SHA-256 and page count in `SOURCE_ZIP_VERIFICATION.json` and the manifest.
5. Copy the final PDF to `paper/when_does_llm_serving_scheduler_adaptation_matter.pdf` (the tracked review copy predates the Query 4 edits) and record the final page count. The title-page footer date is `\today`, so build on the submission day.
6. Visual inspection of the final PDF (title page, back matter, references with new DOIs).
7. Author confirmations: CloudRift legal name, no-other-grant sentence, CRediT roles, which highlights file was uploaded, optional Zenodo description edit.
