# LLM 2026 PDF Release Record

Date: 2026-08-24

## Manuscript

- Title: `The Exploitability Gap in LLM-Serving Scheduler Portfolios`
- Final PDF path in research worktree: `paper/llm2026/main.pdf`
- Final PDF path on `main`: `paper/llm2026/main.pdf`
- Final page count: 15 LNCS pages
- Compile command: `cd paper/llm2026 && tectonic --keep-logs main.tex`
- Compile result: exit code 0; no undefined citations/references, no missing
  figures, no fatal LaTeX errors. Nonfatal underfull/overfull warnings remain.
- Final PDF SHA-256:
  `9562010c898f9614ca2d226031e6f23a9f7d168fcf0a9cf6afd1dd2de6686eb4`

## Research Worktree

- Branch: `contextual-compositional-heuristics-20260731`
- HEAD: `2987b7181efa2bc550d8a894c537eca8f6393eb6`
- Upstream: `origin/contextual-compositional-heuristics-20260731`
- Ahead/behind at release time: `2 ahead / 0 behind`
- Research worktree status: existing dirty/untracked research and manuscript
  artifacts preserved; no research-branch commit or push was performed.

## Review / Author Mode

- Review-mode status: `NORMAL_IDENTIFIED_SUBMISSION_EXPECTED`
- Basis: official LLM 2026 guidance indicates ordinary submissions include
  author information, while PC-authored papers receive double-blind handling.
  Repository/manuscript context did not provide evidence that the manuscript
  author is an LLM 2026 program-committee member.
- Author metadata in released PDF: identified single-author block:
  `Soroush Vahidi`, New Jersey Institute of Technology, Newark, NJ, USA,
  `sv96@njit.edu`.
- ORCID: not included; no confirmed ORCID was found in repository context.

## Origin/Main Release

- Pre-push `origin/main` SHA:
  `738605f949ec7928eb0dc0354c5933ef20faab20`
- Temporary worktree path:
  `/tmp/llm-serving-main-pdf-release-20260824`
- Temporary branch:
  `llm2026-pdf-release-20260824`
- Release commit SHA:
  `19532c294d81c991f09e2907657c5b6efbf7dd1b`
- Release commit message:
  `Add LLM 2026 manuscript PDF`
- Commit contents:
  `paper/llm2026/main.pdf` only
- Post-push `origin/main` SHA:
  `19532c294d81c991f09e2907657c5b6efbf7dd1b`
- Remote PDF checksum verification:
  `git show origin/main:paper/llm2026/main.pdf | sha256sum` matched the final
  local PDF SHA-256.
- Temporary worktree cleanup: worktree removed after clean status verification;
  temporary local branch deleted after confirming merge to `origin/main`.

## Remaining Release Notes

- Repository-public status: the repository still needs to be made public before
  submission if the Data and Code Availability URL is expected to resolve for
  reviewers.
- Hugging Face status: no Hugging Face upload was performed. Existing related
  Hugging Face datasets are not cited as final-paper artifact sources. A
  dedicated artifact release remains optional.
- External datasets cited by manuscript: BurstGPT and Azure LLM Inference Trace
  2023 conversation/code splits.
- No experiments, simulations, GP runs, vLLM/GPU work, Wulver/Vulver jobs,
  external API calls, or scientific-threshold changes were performed in this
  release task.
