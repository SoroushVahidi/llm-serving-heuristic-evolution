# LLM 2026 Manuscript Publication — 2026-08-25

This is a publication note for the finalized manuscript snapshot pushed to
`main` on 2026-08-25. It records provenance only; it is not a research status
document and does not supersede any doc under the working branch's
`docs/current/`.

## Source

- Source branch: `contextual-compositional-heuristics-20260731`
- Source HEAD: `2987b7181efa2bc550d8a894c537eca8f6393eb6`
- Published from: a temporary worktree checked out from `origin/main`
  (`4771b69d6c6d10fcaa903bdd9d51631cfbf238f3`), not a merge of the source
  branch's history.

## Manuscript

- Title: *The Exploitability Gap in LLM-Serving Scheduler Portfolios*
- Final PDF path: `paper/llm2026/main.pdf`
- Final PDF SHA-256: `e0b81ef2182e989f10457472bb297f8bb6f39f44ec264fea18041eb2ae10fc0e`
- Page count: 15 (LNCS, within the 12-15 page venue limit)
- Build command: `cd paper/llm2026 && tectonic --keep-logs main.tex`
- Build result: exit 0, no undefined references or citations

## Isolated correctness fixes included in this push

Two variable-shadowing bugs in `scripts/run_decision_criticality_terminal_anwg_v1.py`
(`analyze()` reused a local name across the positive-gain concentration
computation and an unrelated AUROC scratch array, and separately across the
prevalence-summary dict and a burst-length-tracking loop variable), plus two
new regression tests in
`tests/test_decision_criticality_terminal_anwg_v1.py` that would have caught
each. The frozen experiment artifacts under `experiments/` were not modified
and are not part of this push.

## Note on PDF hash reproducibility

Rebuilding `main.tex` with `tectonic` embeds a fresh `CreationDate` in the PDF
on every build, so the SHA-256 of `main.pdf` will differ across rebuilds even
when the source is unchanged. The page count, text content, and structure are
the invariant to check, not the hash.
