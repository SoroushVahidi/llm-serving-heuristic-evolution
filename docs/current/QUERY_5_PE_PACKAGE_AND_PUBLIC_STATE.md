# QUERY 5 of 6 — final package, rendering and public-availability synchronization

Date: 2026-09-21. Scientific content frozen at 334c307 (Query 2); compliance commit 132e2fa (Query 4 + CloudRift/CRediT corrections).
The commit that contains this file is the final Query-5 commit; read its SHA with `git log -1 -- docs/current/QUERY_5_PE_PACKAGE_AND_PUBLIC_STATE.md`. A file cannot record its own commit hash.

## 1. Query-4 edits preserved and committed

Preflight state: HEAD 334c307, edits uncommitted, matching the Query-4 report exactly (GenAI sentence, heading case, five DOIs, builder heading lookup). Committed in 132e2fa together with the Query-5 wording fixes below. Nothing was lost.

## 2. CloudRift correction

The award email (author-supplied) names the **CloudRift AI Builder Grant**, USD 1,000 in compute credits.

| Location | Class | Action |
|---|---|---|
| `paper/performance_evaluation/main.tex` (Funding and Support) | active | rewritten |
| `scripts/build_performance_evaluation_submission.py` (checklist row) | active | rewritten |
| `submission/main.tex`, `declarations.md`, `SUBMISSION_CHECKLIST.md` | derived | regenerated |
| `docs/current/PERFORMANCE_EVALUATION_SUBMISSION_ROADMAP_20260920.md` section G | current roadmap | updated |
| `release/performance_evaluation_v1_1_0/paper/.../main.tex` | archived in published Zenodo v1.1.0 | left; historical |
| `docs/current/QUERY_4_...md` | historical record of the open question | left |

New text: "This work was supported by Google Cloud Research Credits (USD 1,000 in cloud credits) and by USD 1,000 in compute credits through the CloudRift AI Builder Grant. Both awards were provided as computing credits, not as cash. Neither Google nor CloudRift had any role in the study design, data collection, analysis, interpretation, manuscript preparation decisions, or the decision to submit the article for publication." No "CloudRift Inc." remains in any active file (`git grep`).
The sentence "No monetary research grant was received from these organizations" was replaced by the credits-not-cash sentence, which is more accurate for a named "grant".

## 3. CRediT

Eleven roles: Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data curation, Visualization, Project administration, Writing - original draft, Writing - review & editing. The previous statement had ten; **Project administration was added** because the author's Query-5 role list includes it. Supervision, Funding acquisition and Resources are not claimed.

## 4. Highlights and cover letter

Neither was regenerated or replaced.
* Canonical local reference: `submission_docs/performance_evaluation_highlights.txt` = `submission/performance_evaluation_highlights.txt` (SHA-256 `d3b01cfa...`) and `submission_docs/cover_letter.txt` = `submission/cover_letter.txt`.
* Builder source: the script copies these four files from `submission_docs/`, never from `paper/performance_evaluation_highlights.txt`. Verified by md5 before and after the rebuild: highlights, cover letter and suggested reviewers byte-identical.
* **Stale:** `paper/performance_evaluation_highlights.txt` (different, older text, `ac6b4d46...`). Kept in place but unused. `docs/current/README.md` now links to the canonical file instead.
* Cover letter re-read: title and journal correct; no assertion about the reserve analysis, so no contradiction.

## 5. Experiment branch

`experiment/reference-reserve-sensitivity-20260921` (primary worktree, clean): 327736b, one commit ahead of the remote tip ec19963, fast-forward, 33 files, all under `experiments/reference_reserve_sensitivity_v1/`, largest file 156 KB, no manuscript files, no credentials. The `slurm/*.sbatch` files carry the HPC account name and scratch path, which already appear in 49 and 190 files on public `main`. Pushed without force; `git ls-remote` shows 327736b; 18 result files present under `results_v1/`.

## 6. Public-repository safety audit (everything that becomes public on `main`)

`origin/main` (2bdf4b7) is an ancestor of the manuscript branch: fast-forward possible, no divergence. The diff is 76 files, the largest blob 0.5 MB (the PDF). Over 5,181 added text lines: no API keys, tokens, passwords, bearer strings or private-key headers; the only e-mail is the author's public sv96@njit.edu; no absolute `/home/soroush` paths. One pre-existing observation: `docs/current/PERFORMANCE_EVALUATION_SUBMISSION_ROADMAP_20260920.md` names a Google Cloud project ID; it is already public on `main` and is an identifier, not a credential.

## 7. Layout and presentation

* All 34 pages of the final PDF were rendered and inspected (contact sheets plus page 1 and 28 at reading size). Page 1, Table 1 (p. 7), Tables 2-7, Figures 1-4, the back matter and the references are intact: no clipped text, no separated caption, no `??`, no blank page.
* Two safe layout attempts to reach 33 pages (breakable `\texttt` path; removing `\raggedright` from the Data paragraph) did not change the count, and the second produced a stretched, hyphenated line. Both were reverted; `main.tex` is exactly as committed. **34 pages is final.** Page 34 holds only reference [38] (40 words).
* Presentation-trap scan (AI-like or vague wording, repeated caveats, table headers, page-1 framing): no concrete instance found; no prose changed.
* `sim. ms` labels and measured vLLM milliseconds remain distinct; all seven tables and four figures unchanged.

## 8. Zenodo

No new version, no metadata edit. The record's description still says "preregistered" (optional cleanup, not done). The manuscript already separates the archived v1.1.0 material from the reserve analysis available on GitHub.

## 9. Derived files and package

Regenerated by `scripts/build_performance_evaluation_submission.py` from the frozen sources: review PDF, `submission/{main.tex,references.bib,main.bbl}`, four figures, source ZIP, manifest, checklist, verification JSON, `declarations.md`. Builder checks all pass (no absolute paths, figures resolve, clean build, no undefined references, extracted-build text equals the review PDF, abstract renders). The builder's checklist rows for the title-page block, CRediT and acknowledgements were changed from USER_CONFIRM to READY because the author has confirmed them; "Prior-submission disclosure" stays USER_CONFIRM.

Independent check: the ZIP was extracted into `mktemp -d /tmp/peva-submission-check.XXXXXX` (path printed, prefix checked), built from scratch (34 pages, 0 undefined), `main.tex`/`main.bbl`/`references.bib` and four figures identical to canonical, no absolute paths, text equal to the review PDF; only that directory was then removed.

## 10. Tests and freeze

`build_claim_manifest.py --check`: 80 claims, 0 problems. 79 manuscript tests pass. 0 undefined citations. No builder-specific test exists in `tests/`. Versus 334c307: the only new numeric token is the second "1,000" of the CloudRift credit amount; all 7 tables, 4 equations (incl. Eq. 4), the three `tabular` blocks, the figures, `FINAL_CLAIM_MANIFEST.json` and `experiments/` are identical; 590/720, the primary CI and all reserve values unchanged. Abstract: 247 words.

## 11. Page-count documentation

Updated 30 -> 34 in `docs/PEVA_PREPACKAGE_READINESS.md` and `paper/performance_evaluation/README.md`. Left as historical: `docs/current/FINAL_MANUSCRIPT_FREEZE.md` (30 pages, correct when written) and the roadmap's earlier "20 pages".

## 12. Final artifact hashes (SHA-256)

| Artifact | SHA-256 |
|---|---|
| `paper/performance_evaluation/main.pdf` = tracked review PDF `paper/when_does_llm_serving_scheduler_adaptation_matter.pdf` (34 pages) | `e0f780c2772b66a6aaa963a02423aeef23ac74542d1b153b59d4ea2d845f3975` |
| `main.tex` (= `submission/main.tex`) | `7b00dd2dab37b5980f2ad556cd4332fe928d8d68a991545efcdb5e7e2521c426` |
| `references.bib` (= `submission/references.bib`) | `f45756813a9257922d9c22aa4f6b712aac02166026ca0f0eb8a1dee11efa0292` |
| `main.bbl` (= `submission/main.bbl`) | `f73b4d2706d3f6d97f41ed8c468a7da3c0060f8d3635689bbbb8d0c129415cb9` |
| `submission/performance_evaluation_submission_sources.zip` (8 files, 187,108 bytes) | `cfba8623fc4f5703c2e427a0fa1ece04cd1e0e13cbc5162c78a4865ebf594f31` |
| `submission/declarations.md` | `fc2b5db325405ffcbbfde62115945de2c29909e3d0aff7d04e65dd7d5965615c` |
| canonical highlights (`submission_docs/` = `submission/`) | `d3b01cfa2acead5a621debc6e7c6bffdbda781d7ce302880b4e5a92d983e6ea2` |
| cover letter (`submission/`) | `84893d165866fcf31ec6b3eaabe0bd62322c3f55b0f4ece5810eb9a0c4d6f17a` |
| stale `paper/performance_evaluation_highlights.txt` (unused) | `ac6b4d46c301975be0910fd33fcbb321f3f0f7d4876131bee24691c2ab24c424` |

Git: compliance commit `132e2fa`; experiment branch tip `327736b`; remote `main` before Query 5 `2bdf4b7`. Final commit, remote `main` and the availability check are appended in the Query-5 report returned to the author.

## 13. Availability statement, sentence by sentence (after the push)

Recorded in the final report; the statement itself is unchanged: GitHub repository public; v1.1.0 archived on Zenodo record 22866983 (contents verified in Query 4); reserve protocol, results and provenance under `experiments/reference_reserve_sensitivity_v1/` on `main`; raw third-party traces not redistributed.

## 14. Remaining items for the author (not blockers)

* The Elsevier declarations-tool document and ORCID are portal steps.
* "Prior-submission disclosure" (withdrawn LLM 2026 predecessor) is still an author decision; the cover letter is already uploaded.
* Optional: Zenodo description wording "preregistered" -> "pre-specified"; a lone reference on page 34.
