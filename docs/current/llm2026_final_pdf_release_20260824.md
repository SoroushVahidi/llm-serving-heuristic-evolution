# LLM 2026 Final PDF Release Record

Date: 2026-08-24

## Manuscript

- Title: `The Exploitability Gap in LLM-Serving Scheduler Portfolios`
- Final PDF path in research worktree: `paper/llm2026/main.pdf`
- Final PDF path on `origin/main`: `paper/llm2026/main.pdf`
- Final page count: **15 LNCS pages**
- Compile command: `cd paper/llm2026 && tectonic --keep-logs main.tex`
- Compile result: exit code 0; no undefined citations/references, no missing figures, no fatal LaTeX errors. Nonfatal underfull/overfull warnings remain and were judged visually harmless.
- Final PDF SHA-256:
  `78aa2dee802866b83e655482534169ab1a3fd4168d1ff3ca1daffc0608ef1b07`

## Figure Status at Release

- **Figure 1:** unchanged in this pass; readable at final LNCS size on page 6.
- **Figure 2:** schematic cell text increased to ~11 pt matplotlib (~7.4 pt effective at `0.92\textwidth`) with shortened labels (`large chunk`, `64-token chunk`, `budget 512`, `bundled change`); dot-and-whisker panel preserved.

## Research Worktree

- Branch: `contextual-compositional-heuristics-20260731`
- HEAD: `2987b7181efa2bc550d8a894c537eca8f6393eb6` (unchanged)
- Upstream: `origin/contextual-compositional-heuristics-20260731`
- Ahead/behind at release time: `2 ahead / 0 behind`
- Research worktree status: dirty/untracked research and manuscript artifacts preserved; no research-branch commit or push performed.

## Origin/Main PDF-Only Update

- Pre-update `origin/main` SHA:
  `19532c294d81c991f09e2907657c5b6efbf7dd1b`
- Temporary worktree path:
  `/tmp/llm-serving-main-pdf-final-20260824` (removed after verification)
- Temporary branch:
  `llm2026-final-pdf-20260824` (deleted after verification)
- Release commit SHA:
  `4771b69d6c6d10fcaa903bdd9d51631cfbf238f3`
- Release commit message:
  `Update LLM 2026 manuscript PDF`
- Commit contents:
  `paper/llm2026/main.pdf` only (141,373 → 153,719 bytes)
- Post-update `origin/main` SHA:
  `4771b69d6c6d10fcaa903bdd9d51631cfbf238f3`
- Remote PDF checksum verification:
  `git show origin/main:paper/llm2026/main.pdf | sha256sum` matched local final PDF SHA-256 exactly.

## Final PDF Approval

- Status: `FINAL_PDF_APPROVED_FOR_RELEASE`

## Remaining Human Actions

- Replace placeholder author metadata only if venue submission route requires changes.
- Make the GitHub repository public before submission if the Data and Code Availability URL must resolve for reviewers.
- Perform venue submission upload using the approved PDF at `paper/llm2026/main.pdf` or the identical copy on `origin/main`.
- No additional scientific experiments are required for submission.

## Repository Public Status

The repository still needs to be made public before submission if reviewers are expected to access the cited GitHub URL.
