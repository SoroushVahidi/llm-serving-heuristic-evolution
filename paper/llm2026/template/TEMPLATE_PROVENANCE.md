# Springer Proceedings Template Provenance

Date integrated: 2026-08-24

## Remote Source

- Remote repository: `https://github.com/SoroushVahidi/llm-serving-heuristic-evolution`
- Remote ref containing upload: `origin/main`
- Remote commit: `738605f949ec7928eb0dc0354c5933ef20faab20`
- Remote path: `LaTeX2e+Proceedings+Template+ZIP.zip`
- Extraction method: `git show origin/main:LaTeX2e+Proceedings+Template+ZIP.zip > paper/llm2026/template/source/LaTeX2e+Proceedings+Template+ZIP.zip`
- Note: the remote upload is on `origin/main`, not on the current working branch upstream. No pull, merge, rebase, checkout, or branch switch was used.

## Local Preservation

- Untouched archive: `paper/llm2026/template/source/LaTeX2e+Proceedings+Template+ZIP.zip`
- Untouched extracted package: `paper/llm2026/template/official/`
- Working compile copies:
  - `paper/llm2026/llncs.cls`
  - `paper/llm2026/splncs04.bst`

## Checksums

| File | SHA256 |
|---|---|
| `template/source/LaTeX2e+Proceedings+Template+ZIP.zip` | `7cc8efaa4f6e7ea8d17069c37a192c6023170f1e60f59509f3bb00591dcaf5de` |
| `template/official/llncs.cls` | `a3cfe775b394aba8db8fbb54b8920ecbb12f4532cf787cd6d9b04712f58d0d1a` |
| `template/official/splncs04.bst` | `f36c3a17e5304a692706359aafa9de709395a085e579eb47c027095aeaa35174` |

## Package Identity

The extracted package identifies itself as Springer's LaTeX2e package for
Lecture Notes in Computer Science (LNCS) and other proceedings book series.
It contains:

- `llncs.cls`
- `splncs04.bst`
- `samplepaper.tex`
- `llncsdoc.pdf`
- `readme.txt`
- `history.txt`
- `fig1.eps`

The sample uses `\documentclass[runningheads]{llncs}`, places keywords inside
the `abstract` environment, uses `\maketitle`, `\inst{}` author affiliations,
and recommends BibTeX style `splncs04`.

## Format Conclusion

The official package is a Springer LNCS / Computer Science proceedings
template. It is one-column by default. The earlier local two-column article
draft scaffold has been superseded for manuscript formatting by this official
template. Page-budget planning should now use the venue's one-column Springer
format allowance unless a later venue instruction explicitly requires a
different review format.
