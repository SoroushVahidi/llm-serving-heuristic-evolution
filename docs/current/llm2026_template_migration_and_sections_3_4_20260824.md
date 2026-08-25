# LLM 2026 Template Migration and Sections 3-4 Draft Audit

Date: 2026-08-24

Scope: manuscript/template work only. No simulations, selector/router training,
GP, vLLM/GPU work, Wulver/Vulver work, TEST/FINAL use, DEV redesign, threshold
changes, push, merge, rebase, reset, clean, stash, or checkout were performed.

## 1. Pre-Flight

- Branch: `contextual-compositional-heuristics-20260731`
- HEAD: `2987b7181efa2bc550d8a894c537eca8f6393eb6`
- Upstream: `origin/contextual-compositional-heuristics-20260731`
- Ahead/behind after fetch versus upstream: `2 0`
- Existing dirty tracked files before this task were preserved:
  - `docs/current/NEXT_ACTIONS.md`
  - `docs/current/RESUME_HERE.md`
  - `docs/current/WORK_STATUS.md`
- Existing untracked scientific/manuscript artifacts were preserved.
- Git lock state: `.git/index.lock` was present during inspection. It was not
  removed or modified.

## 2. Remote Fetch / Divergence

`git fetch origin` was used; no pull, merge, rebase, checkout, or branch switch
was used.

- Current upstream branch remote tip:
  `origin/contextual-compositional-heuristics-20260731` =
  `8e1223beb58fd4d296061b6b48e3ba493714108f`
- Remote template upload was found on:
  `origin/main` = `738605f949ec7928eb0dc0354c5933ef20faab20`
- Remote commit subject: `Add files via upload`
- Relevant uploaded file: `LaTeX2e+Proceedings+Template+ZIP.zip`
- The `origin/main` diff also contains unrelated deletes/modifications, so only
  the template ZIP was materialized locally.

## 3. Template Copy / Verification

Copied only:

```bash
git show origin/main:LaTeX2e+Proceedings+Template+ZIP.zip \
  > paper/llm2026/template/source/LaTeX2e+Proceedings+Template+ZIP.zip
```

Extracted into `paper/llm2026/template/official/` and left that extracted
official package untouched.

The package contains:

- `llncs.cls`
- `splncs04.bst`
- `samplepaper.tex`
- `llncsdoc.pdf`
- `readme.txt`
- `history.txt`
- `fig1.eps`

Template identity: Springer's LaTeX2e package for LNCS and other Computer
Science proceedings series.

The sample paper confirms:

- class: `\documentclass[runningheads]{llncs}`
- author affiliation syntax: `\author{... \inst{...}}`,
  `\institute{...}`
- keywords are placed inside `abstract`
- bibliography style: `splncs04`
- default layout: one-column LNCS, not the previous local two-column article
  draft scaffold.

Full checksums and provenance are recorded in:
`paper/llm2026/template/TEMPLATE_PROVENANCE.md`.

## 4. Official Format / Page-Limit Interpretation

The previous planning artifact targeted an 8-page two-column draft because no
official template was locally available. The uploaded package is one-column
LNCS. The venue constraints recorded in
`docs/current/llm2026_full_manuscript_architecture_20260824.md` state that
full/regular papers allow either 12-15 pages in one-column Springer Nature
format or 6-8 pages in two-column format, including figures, tables, and
references.

Since the official uploaded package is one-column LNCS, current planning should
use the 12-15 page one-column Springer basis unless later venue instructions
override it. The old 8-page two-column budget is superseded for the current
manuscript file.

## 5. Draft Preservation

The pre-migration draft was copied to:

`paper/llm2026/archive/pre_official_template/`

Preserved files:

- `main.tex`
- `references.bib`
- `README.md`
- `main.pdf`

## 6. Manuscript Migration

Updated:

- `paper/llm2026/main.tex`
- `paper/llm2026/references.bib`
- `paper/llm2026/main.pdf`

Added compile working copies:

- `paper/llm2026/llncs.cls`
- `paper/llm2026/splncs04.bst`

The manuscript now uses the official LNCS class and `splncs04` bibliography
style. The existing title, abstract, Sections 1-2, thesis, and contribution
structure were preserved in substance and adapted to LNCS syntax.

Remaining manuscript placeholders:

- Section 5: `Constructive Falsification Beyond Selection`
- Section 6: `Real-Serving Semantic Validation`
- Section 7: `Implications, Limitations, Related Work, and Conclusion`
- acknowledgments

These were intentionally not drafted in this task.

## 7. Bibliography Status

Started `paper/llm2026/references.bib` with verified primary-source or
standard references for:

- Orca / iteration-level LLM serving
- vLLM / PagedAttention
- Sarathi-Serve / chunked prefill
- DistServe / prefill-decode disaggregation
- Vidur
- VTC / fairness in LLM serving
- KV-constrained LLM inference scheduling
- Rice algorithm selection
- SATzilla
- Koza genetic programming
- Montana strongly typed GP
- WFQ/GPS fairness
- SRPT/SPT scheduling

Only citations needed by Sections 1-2 currently appear in the compiled
references. Later sections should add the GP/STGP and real-serving citations
where appropriate.

## 8. Section 3 Summary

Drafted `3. Complementarity Beyond Handcrafted Families`.

Main content:

- controlled A/B/C mechanism-isolation summary;
- public-trace saturation as bounded motivation for stress workloads;
- joint 240-scenario multi-mechanism result as the main breadth evidence;
- transition from oracle opportunity to exploitability.

Key numbers included:

- controlled matrix: 176 scenarios, 1,056 policy cells, 54.0% unique-winner
  rate;
- Family A/B/C headrooms: 0.021742, 0.049433, 0.020261;
- public replay: 60/60 windows tied at ANWG 1.0, envelope gain 0.0;
- joint workload: 240 scenarios, 1,440 cells, 92.9% with >=2 elevated
  pressures, 175/240 with >=3;
- winner counts: LLF 59, KV constrained 50, full prefill 46, ESTF 45, WFS 35,
  chunked prefill 5;
- joint headroom: 0.019034, bootstrap 95% CI [0.015988, 0.022433];
- mixed-mechanism gain share: 93.1% from >=2 pressures, 63.3% from >=3.

Claim-safety wording is bounded to the evaluated synthetic distribution.

## 9. Joint Complementarity Figure

Created:

- `paper/llm2026/figures/joint_complementarity.pdf`
- `paper/llm2026/figures/joint_complementarity.png`

Source artifacts:

- `experiments/joint_multimechanism_generalization_v1/winner_summary.json`
- `experiments/joint_multimechanism_generalization_v1/oracle_summary.json`
- `experiments/joint_multimechanism_generalization_v1/coverage_summary.json`
- `experiments/joint_multimechanism_generalization_v1/utility_matrix_wide.csv`

Panels:

1. per-scenario winner distribution;
2. oracle-gain distribution with mean and 0.01 markers;
3. elevated-mechanism pressure count distribution and gain-share annotations.

No simulations were rerun.

## 10. Section 4 Summary

Drafted `4. The Exploitability Gap`.

Main content:

- contextual selector failure despite within-family learnability;
- shared-feature rescue failure;
- hierarchical router as the strongest detectability-vs-utility example;
- evidence-supported reasons classification can diverge from utility;
- compact frozen-gate table.

Key numbers included:

- pooled selector regret 0.0463 versus best-fixed 0.0233 and majority 0.0127;
- LOFO-A regret 0.4786 versus fixed 0.0767;
- router macro-F1 0.9887;
- router live gain +0.00616 ANWG, below +0.01 gate;
- router oracle-gap closure 0.143;
- Wulver: 24,314 unique fingerprints, 156 novel domains, 1/104 DEV rows closer;
- guarded composition: 3.11% WFS-regret reduction, mechanism-ordering failure;
- typed structural crossover: 4,320 TRAIN candidate-scenario evaluations,
  best crossover mean MG 0.0.

## 11. Frozen-Gate Table

Created as Table 2 in `paper/llm2026/main.tex`.

Rows:

- pooled/shared selector;
- hierarchical router;
- target-free Wulver support expansion;
- guarded ESTF/WFS composition;
- typed structural crossover.

Known layout issue: in one-column LNCS, Table 2 is dense. It compiles, but
future compression should either shorten cells further or move some rows to an
appendix/supplement if available.

## 12. Claim / Number Ledger Updates

Updated:

- `docs/current/llm2026_number_source_of_truth_20260824.md`
- `docs/current/llm2026_claim_evidence_ledger_20260824.md`

Added manuscript-location addenda for all numeric claims inserted into Sections
3-4.

## 13. Compile Status

Command:

```bash
cd paper/llm2026
tectonic main.tex
```

Result: success.

Output:

- `paper/llm2026/main.pdf`

Warnings:

- overfull boxes in abstract/introduction and long policy/table text;
- underfull boxes in narrow table columns.

No fatal errors remain. BibTeX ran successfully through Tectonic.

## 14. Page Budget

Compiled PDF page count: 10 pages, letter paper, one-column LNCS.

Approximate current allocation:

- title/abstract/Sections 1-2: pages 1-5;
- Section 3 starts on page 6;
- Section 4 continues through page 9;
- Section 5-7 placeholders and references appear on pages 9-10.

Planning implication:

- Under a 12-page LNCS target, the draft is already tight and Sections 5-7 must
  be very compact or Sections 1-4 must be compressed.
- Under a 15-page LNCS maximum, there is enough room for Sections 5-7 and a
  real-vLLM figure, but compression is still advisable.

## 15. Next Manuscript Task

Draft only:

1. Section 5: `Constructive Falsification Beyond Selection`
2. Section 6: `Real-Serving Semantic Validation`

Do not run new science. Keep Section 5 short because Table 2 already carries
the no-go ledger. Section 6 should focus on the three-stage real-vLLM story and
native budget result.
