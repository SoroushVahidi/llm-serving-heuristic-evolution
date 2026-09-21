# Performance Evaluation manuscript: pre-package readiness

Audit date: 2026-09-20. Branch `manuscript/peva-final-visual-freeze-20260920`, created from
`63418512a84543afc83d09c06a3de65bf170fc31` (`manuscript/peva-final-scientific-cleanup-20260920`).
Target journal: *Performance Evaluation* (Elsevier). Source: `paper/performance_evaluation/main.tex`
(`elsarticle`, preprint, 12 pt; 30 pages, letter paper, text width 390 pt = 5.40 in).

No experiment was run, no scientific result, artifact or preregistered analysis was changed, and neither Zenodo, `main`
nor the submission package was touched.

FINAL_VERDICT: MANUSCRIPT_READY_FOR_ARCHIVE_AND_PACKAGING

Status vocabulary: `READY` means finished for this pass; `BLOCKED_FOR_QUERY_8` means it cannot be finished before the
archive and submission package exist.

## 1. Checklist

### Manuscript

| Item | Status | Note |
|---|---|---|
| Title | READY | Unchanged. |
| Abstract | READY | About 225 words (limit 250); every number is checked by `tests/test_manuscript_presentation.py`. |
| Keywords | READY | Six keywords. |
| Introduction | READY | Contributions and RQs unchanged. |
| Related Work | READY | Table 1 unchanged; see section 5. |
| Methods | READY | Line reflowed; Eq. (3) now one numbered display of two aligned lines. |
| Results | READY | Section 5 text now lists all three workloads' first-disagreement caps (it had omitted BurstGPT cap 16). |
| Discussion | READY | Idiom removed ("leaves on the table"); limitations lead-in made accurate. |
| Conclusion | READY | Unchanged. |
| References | READY | 27 cited, 27 in the bibliography, 27 in `main.bbl`; see section 6. |
| AI declaration | READY | Verified against the repository history; see section 4. |
| Acknowledgements | READY | See the author-attention item A1. |
| Data Availability | READY | Accurate for the current archive; the DOI and citation update is BLOCKED_FOR_QUERY_8 (section 7). |

### Visuals

| Item | Status | Note |
|---|---|---|
| Figure 1 (evidence chain) | READY | Redrawn: grayscale, 7.5 pt, dashed box for the unmeasured stage. |
| Figure 2 (native versus pressure prevalence) | READY | Horizontal bars, counts in the labels, values read from the artifacts. |
| Figure 3 (active-cap transition) | READY | Reversed axis is intentional and now stated; percent units; no `1e-3` multiplier. |
| Figure 4 (regime characterization) | READY | Same data; drawn at final size. |
| Figure 5 (regime map) | READY | Same data; plain tick labels (0.01, 0.1, 1). |
| Figure 6 (distribution, concentration) | READY | Same data; legend no longer overlaps a label. |
| Tables 1-6 | READY | Table 3 overflow removed; Table 6 widened to the text width. |

### Provenance

| Item | Status | Note |
|---|---|---|
| Frozen primary artifacts | READY | Unchanged; every quoted number recomputed (`FINAL_CLAIM_MANIFEST.json`, 46 claims). |
| Corrected derivative | READY | Unchanged (`..._corrected/`); the manuscript never reads the two defective columns. |
| Robustness outputs | READY | Unchanged (`..._robustness/`), recomputed by `scripts/robustness_numbers.py`. |
| Claim manifest | READY | `paper/performance_evaluation/FINAL_CLAIM_MANIFEST.json`, checked by `tests/test_manuscript_claim_manifest.py`. |

### Packaging items deferred to Query 8

| Item | Status |
|---|---|
| Merge / integration into `main` | BLOCKED_FOR_QUERY_8 |
| Zenodo new version | BLOCKED_FOR_QUERY_8 |
| Dataset / software citation | BLOCKED_FOR_QUERY_8 |
| `submission/main.tex` regeneration | BLOCKED_FOR_QUERY_8 |
| Source zip | BLOCKED_FOR_QUERY_8 |
| Highlights | BLOCKED_FOR_QUERY_8 |
| Cover letter | BLOCKED_FOR_QUERY_8 |
| Submission checklist | BLOCKED_FOR_QUERY_8 |
| Elsevier declarations tool | BLOCKED_FOR_QUERY_8 |
| Suggested reviewers | BLOCKED_FOR_QUERY_8 |

## 2. Figure record

All figures are drawn by three scripts that share `paper/performance_evaluation/scripts/figstyle.py`, are included with
`\includegraphics[scale=1]`, and therefore print at the sizes below. Every PDF is vector, has no raster image, embeds
DejaVu Sans as TrueType (no Type 3), and has a bounding box equal to the canvas (an assertion in `figstyle.save`
rejects clipped artists). Every PNG is 300 dpi and has identical R, G and B channels, so no hue carries information.

| Figure | File | Size (in) | Smallest text |
|---|---|---|---|
| 1 | `pe_pipeline` | 5.39 x 1.45 | 7.5 pt |
| 2 | `pe_disagreement_rates` | 5.39 x 3.05 | 7.5 pt |
| 3 | `pe_pressure_transition` | 4.70 x 2.55 | 7.5 pt |
| 4 | `pe_fresh_headroom` | 5.39 x 2.75 | 7.5 pt |
| 5 | `pe_regime_map` | 5.00 x 3.40 | 7.0 pt |
| 6 | `pe_robustness` | 5.39 x 2.75 | 7.0 pt |

The only text below 7 pt is the mathtext subscript `LAT` (5.3-5.6 pt, 70 % of the parent), which is inherent to the
subscript notation. Before this pass Figures 1-3 were 8.1, 7.9 and 6.3 in wide and were shrunk to 65-79 %, which put
most of their text at 6-7 pt and stored it as Type 3 fonts; Figure 4 was shrunk to 89 %.

Encoding shared by all figures: dark gray fill = KV capacity, white fill = active-sequence cap; hatching and line style
carry every other distinction; panel labels "(a)", "(b)"; workload names "Azure code", "Azure conv.", "BurstGPT";
regime names "active cap n", "KV 16,000"; prevalence always "$P(D)$ (% of SBS decision states)".

### Defects found and fixed while restyling

* **Figure 2 plotted a hand-typed value that matched no artifact.** The BurstGPT cap-8 bar was 0.0003; the Phase B
  artifact gives 1.71e-4 (76 of 444,469 states). The other three pressure values were correct to their typed
  precision. Figures 2 and 3 now read every value from `PHASE_A_WORKLOAD_SUMMARY_V1.csv` and
  `PHASE_B_V2_WORKLOAD_AXIS_SUMMARY.csv`. No manuscript sentence quoted the wrong value.
* **Figure 3's reversed axis is scientifically intentional.** The artifact's own `pressure_order` runs 512, 64, 32, 16,
  8, 4, i.e. increasing pressure. It is retained, the ticks are the six measured settings, and the axis label and
  caption say "tighter caps to the right".
* Section 5 said active-sequence pressure created disagreement "at active-cap 8 or 4"; BurstGPT first disagrees at
  cap 16 (Table 2). Corrected, and the text now says KV disagreement first appears at 16,000 tokens in all three
  workloads (Azure conversation first binds at 8,000 but first disagrees at 16,000).
* "Window w11" was used before it was tied to the Azure-code trace; it is now "Azure-code window w11" in the text, in
  Table 5 and in the Figure 6 caption.

## 3. LaTeX warnings

| Warning | Disposition |
|---|---|
| Table 3, 1.0 pt overfull | Fixed: "Windows" column widened from 0.100 to 0.110 of the line width, "P(B given D)" narrowed to keep the total. |
| Methods paragraph, underfull (badness 2119) | Fixed by reflowing one clause ("20 windows from each of ..."). |
| Bibliography entries [11] (three lines) and [16] | Fixed: the bibliography is set ragged-right, so the long Azure URL no longer stretches. |
| Data Availability long URL | Fixed with the standard `url` package settings (`\urlstyle{same}`, breaks only at `/`, `-`, `_`) and a locally ragged paragraph. No manual break characters. |
| Page 1, 2.6 pt overfull "while `\output` is active" | Harmless, intentionally retained. It is the class's own first-page footer ("Preprint submitted to Elsevier" and the date); it is invisible in the PDF, and adding `\journal{}` does not change it. |

The final build (`pdflatex`, `bibtex`, two more `pdflatex` passes) has no unresolved reference, no undefined citation
and no other box warning.

## 4. Generative-AI declaration

Text: "the author used ChatGPT (OpenAI), Codex (OpenAI), Gemini (Google), and Claude (Anthropic) in order to assist with
organization, language editing, literature-search planning, and drafting analysis, figure, and test code. After using
these tools, the author reviewed and edited the content as needed and takes full responsibility for the content of the
publication." It follows the Elsevier wording (tool, purpose, review and editing, responsibility).

* Claude is retained: the manuscript commits `7b10293`, `be1dc58`, `9070278`, `6341851` and the robustness and
  correction commits `5c8a78c`, `1fa9468` carry a `Co-Authored-By: Claude` trailer, and the tasks named there (analysis and test code, figure code, manuscript
  editing) are covered by the stated purposes.
* ChatGPT, Codex and Gemini are the tools listed in `docs/current/PERFORMANCE_EVALUATION_SUBMISSION_ROADMAP_20260920.md`
  (section H) as disclosed at the supported level. No tool was added and none states that a tool decided anything
  scientifically. `tests/test_peva_prepackage_readiness.py` pins this.
* The figures carry no AI note: their code is deterministic plotting code over frozen artifacts, covered by the
  declaration's "figure and test code".

## 5. Related-work decision (Table 1)

The four *Performance Evaluation* papers (refs [7]-[10]) stay in prose only. Only title-level information was
available (no abstracts or full texts in the repository), so a row would need cells for method, workload and evidence
that cannot be verified. The prose already states only what the titles support.

## 6. Reference audit

* Every `\cite` key exists in `references.bib`; every entry is cited; `main.bbl` has 27 items.
* The four PEVA entries carry volume, article number and DOI, pinned by
  `tests/test_manuscript_presentation.py`: [7] 167:102463, [8] 172:102551, [9] 171:102539, [10] 167:102451. These were
  not re-verified against publisher records in this pass (no web research was done). [9] pairs volume 171 (2026) with a
  DOI containing 2025; confirm this against the Elsevier record at proof stage.
* Preprints [20], [21], [23] are marked "arXiv preprint"; [11] is a web dataset.
* Software/data citation gaps for Query 8 are in section 7.

## 7. Data Availability and what Query 8 must do

The Zenodo record 10.5281/zenodo.22865294 was built from tag `performance-evaluation-v1.0.0` (commit `6abada6`,
2026-09-20 19:27). It contains the frozen native, pressure and fresh-causal artifacts and the older figure script. It
predates the robustness outputs (commit `5c8a78c`, 20:50), the corrected derivative (`1fa9468`, 21:04), the shared
figure code, the claim manifest and every figure and text change of the manuscript. The manuscript previously said the
"derived research data and reproducibility package" were archived there, which implied otherwise. It now says the
archive holds the frozen confirmatory artifacts and predates the robustness analysis, the corrected derivative and the
final figure code, which are in the GitHub repository only.

   write the highlights and cover letter; complete the declarations tool and the suggested-reviewer list.

## 8. Items for the author (not blockers)

* **A1.** The Acknowledgements say the author thanks "his mother". The pronoun is the author's own text and was left
  unchanged; confirm it. The reference to Professor Koutis no longer uses a pronoun.
* **A2.** Elsevier journals often ask for a CRediT authorship statement in the declarations tool; for a single-author
  paper this is a one-line entry (Query 8).
* **A3.** Page 2 begins with a large blank band above "1. Introduction". This is the `elsarticle` preprint layout;
  Elsevier typesets the accepted paper itself.
