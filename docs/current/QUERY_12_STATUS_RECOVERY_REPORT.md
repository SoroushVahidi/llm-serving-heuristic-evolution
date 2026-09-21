# QUERY_12 status recovery report

Generated 2026-09-21 ~12:47 EDT by a read-only audit session. This is the only file that session created; nothing was edited, committed, pushed, or launched.

## Headline

**Query 12 ran to completion and was committed as `0ad9a86` (2026-09-21 12:43:08 EDT) on `release/peva-v1-20260920`. It has not been pushed.** The session that ran it (`llm-serving-heuristic-evolution-ee`, id `e601d1`, transcript `cbefa9ac-…jsonl`) delivered its final 34-section report at ~12:43:52 and is now idle. The working tree is clean.

The question "how much was done?" changed while this audit ran. When the audit began (12:37) the Query 12 session was still mid-execution: `main.tex` and `main.pdf` were changing every 1–2 minutes. It committed at 12:43. The early snapshots are discarded here; every number below was re-verified against the committed state `0ad9a86` unless labelled otherwise.

Evidence labels used below:
- **VERIFIED**: re-checked by this audit against the repository or a fresh run.
- **RECORD**: stated in the Query 12 audit document or final message; not independently re-checked.
- **TRACE**: established from the Query 12 session transcript (tool calls it actually made).

---

## 1. CURRENT_REPOSITORY_STATE (VERIFIED)

| Item | Value |
|---|---|
| Branch | `release/peva-v1-20260920` |
| HEAD | `0ad9a86673a67aa6fd3905bd791dc00ec37a9fd7` |
| Working tree | Clean (`git status --short` empty) before this report was written |
| Commits after `5887cb4` | 1: `0ad9a86 paper: coherence, fairness disclosure and compression pass (QUERY_12)` |
| vs `origin/main` | 0 behind, 17 ahead (was 16 before `0ad9a86`) |
| Upstream | None configured for this branch |
| On any remote branch? | No (`git branch -r --contains HEAD` empty) |
| Tags at HEAD | None |
| `.git/index.lock` | Absent |

## 2. QUERY_12_STARTING_POINT_RECOVERED (VERIFIED + TRACE)

- Query 12 started from `5887cb4` with a clean tree, 34 pages, `main.tex` sha256 `7d2ee7ff…` (TRACE, first command at 16:12:42Z = 12:12 EDT). The prompt was submitted at 12:10:55 EDT.
- Reflog: `5887cb4` (11:13:01) → `0ad9a86` (12:43:08). Nothing else intervened.
- The session is the same conversation that ran Queries 9, 10 and 11 (started ~10:03 EDT). Query 12 is its fourth prompt.
- **No unrelated work is mixed into the commit.** All 13 changed files are manuscript, manuscript-script, test, doc, or the new audit record.

## 3. EXISTING_QUERY_12_WORK_PRODUCTS

| Product | Location | State |
|---|---|---|
| Audit record (policy-information table, fairness table, 14-topic consistency matrix, structure diff, literature notes, journal table, packaging to-do, weaknesses) | `docs/current/QUERY_12_COHERENCE_FAIRNESS_AND_COMPRESSION_AUDIT.md` (134 lines) | Committed in `0ad9a86` |
| Restructured manuscript | `paper/performance_evaluation/main.tex` | Committed |
| Rebuilt PDF (31 pages, built 12:42:37) | `paper/performance_evaluation/main.pdf` + `.aux/.bbl/.blg/.log/.spl` | **Gitignored, not tracked.** It exists only locally |
| Claim manifest and builder | `FINAL_CLAIM_MANIFEST.json`, `scripts/build_claim_manifest.py` | Committed |
| Memory entry | `~/.claude/projects/.../memory/query12_final_coherence_pass.md` (+ index line) | Outside the repo |
| Assembly/patch scripts (scratch) | Session scratchpad: `assemble.py`, `patch2.py`, `patch3.py`; transient `/tmp/main_keep.tex`, `/tmp/main_keep2.tex` | Not in the repo; disposable |

No file named `QUERY_12_*` other than the audit record and this report exists. No stray logs or temp summaries were left in the repo.

## 4. RUNNING_JOBS (VERIFIED)

- **None related to Query 12.** The Query 12 session ran no long job (no tmux/SLURM). It ran `pdflatex`, `bibtex` and `pytest` in the foreground. Each finished in seconds.
- `tmux ls`: `njit-afs` (created Sep 6) and `real-vllm-engineering` (created Aug 28, different repo). Both predate Query 12 and are unrelated.
- `squeue` is not installed on this host, so SLURM could not be queried from here.
- No `pytest`, `latexmk`, `pdflatex`, `vllm` or simulator process was running at the end.
- While the Query 12 session was still active (12:37–12:43) it was monitored for about 6 minutes. It was healthy and progressing (34→32→31 pages, tests green) until it committed and went idle.

## 5. QUERY_12_34_ITEM_STATUS_TABLE

Statuses use the strict scale you asked for. Where a claim rests only on the session's own report, the rationale says so.

| # | Item | Status | Evidence / rationale |
|---|---|---|---|
| 1 | STARTING_STATE | DONE | Branch/HEAD/34 pages/sha recorded at start (TRACE) |
| 2 | PRESPECIFICATION_TERMINOLOGY | DONE | "preregistered" 28→1 (the survivor sits in the sentence explaining the choice); "pre-specified" 0→18; timeline paragraph added; builder/README/reproducibility doc/manifest updated (VERIFIED) |
| 3 | POLICY_INFORMATION_FAIRNESS | DONE | Per-policy table exists in audit record §2 and Methods now discloses exact-length use. I spot-checked SBS, ESTF and LLF code (VERIFIED); WFS and prefill rows are RECORD |
| 4 | SIX_POLICY_COMPARISON_FAIRNESS | DONE | Audit record §3 covers state, arrivals, hardware, candidates, canonicalization, continuation, horizon, information asymmetry. It cites code and protocol fields I did not re-open (RECORD) |
| 5 | SBS_SELECTION_WORDING | DONE | "unrelated" removed; now names the same-project, same-simulator 240-scenario benchmark; fixed before fresh evaluation; not claimed optimal (VERIFIED, see §11) |
| 6 | CENTRAL_CONTRIBUTION | DONE | Abstract, Introduction, Contributions, Discussion "Implications" and Conclusion all frame a measurement, not a scheduler (VERIFIED by reading) |
| 7 | NOVELTY_LANGUAGE | DONE | Intro: "This paper proposes no new scheduler."; heuristics, replay, KV constraints, bootstrap named as established; five contributions match the NEW list (VERIFIED) |
| 8 | CROSS_SECTION_CONTRADICTIONS | DONE | 14-topic matrix in audit record §5; 3 MAJOR + 3 MINOR issues found and fixed. My spot-checks of A–H, L–N found no residual contradiction |
| 9 | REPOSITORY_REPORT_TONE | DONE | "frozen", "artifact", "manifest" removed from prose; "phase" 5→1 (the survivor is legitimate "phase separation"). Effect was modest because the prose was already mostly clean; "earlier" ×4 remains |
| 10 | REPETITION_REMOVED | PARTIAL | Discussion cut 1,015→790 words, but my counts of the nine tracked ideas are essentially unchanged (88.4% ×4, 0.20 ms ×5, "lightly loaded" ×5, 590 ×8, one-step-oracle ×7). Trimming happened, de-duplication of key statements largely did not |
| 11 | ABSTRACT | DONE | 247→237 words, ≤250. Two words above the 215–235 target |
| 12 | ORGANIZATION_CHANGES | PARTIAL | Sections renamed and consolidated, appendix added. The comparison with "several" recent PEVA articles was one article fetched (Lilou, UMass PDF); no notes stored anywhere |
| 13 | METHODS_RESULTS_BOUNDARY | DONE | Methods restructured to 5 neutral subsections; estimands stay in Methods. Residual judgement call: Methods still carries the endpoint rationale and the cluster-key disclosure |
| 14 | RESULTS_DISCUSSION_BOUNDARY | DONE | Interval diagnostics moved to Appendix A; Discussion no longer re-quotes §6 leave-one-out numbers |
| 15 | PAGE_COMPRESSION | DONE | 34→31 pages, the top of the 29–31 target. No font/margin/spacing lines changed in the diff (VERIFIED). Float specifier was loosened from `[tbp]` to `[!htbp]` |
| 16 | APPENDIX_DECISION | DONE | Appendix A holds bootstrap/BCa/jackknife diagnostics only; primary result, thresholds, concentration and reference dependence stay in the body |
| 17 | TABLE_AUDIT | PARTIAL | Six tables checked from contact sheets at 40–60 dpi plus zooms; no written per-table log. See §16 |
| 18 | FIGURE_AUDIT | PARTIAL | Four figures viewed at ≤60 dpi. "Legible at actual size" is not established at that resolution |
| 19 | MATHEMATICAL_NOTATION | DONE | 5 display equations, all numbered; the unnumbered `P(B\|D)=590/720` display was made inline; unused `Q_SBS` removed (VERIFIED) |
| 20 | DEFINITION_AUDIT | PARTIAL | Final report confirms only 5 terms plus the absence of "no-choice iteration". Of the 13 requested terms, the rest are unreported |
| 21 | LITERATURE_TERMINOLOGY | PARTIAL | vLLM engine-arg docs and the Sarathi-Serve abstract were fetched fresh (TRACE). The prompt also listed queueing, service-time estimate and SLO/deadline terminology, which the record does not address |
| 22 | CITATION_VERIFICATION | PARTIAL | **No web fetches for any of the named sources occurred during Query 12** (TRACE). Verification was done earlier in Queries 9–11 (38 web calls). See §18 |
| 23 | LIMITATIONS | PARTIAL | Length essentially unchanged (351→354 words); exact-output-length disclosure strengthened (1→2 mentions). The audit record's "limitations shortened" is not borne out by word count |
| 24 | CONCLUSION | DONE | 315→219 words (reported 220), inside the 150–220 target; ends on the main takeaway; no new statistics |
| 25 | PRACTICAL_IMPACT | DONE | "Measurement-first practice, not a scheduler recommendation"; "demonstrates no production savings, deployed-selector benefit, or capacity-planning gain" |
| 26 | JOURNAL_COMPLIANCE | PARTIAL | Abstract, keywords, numbering, AI declaration OK. CRediT is not in `main.tex`; the official guide returned 403; the appendix heading is unnumbered |
| 27 | FULL_VISUAL_PDF_CHECK | PARTIAL | All 31 pages seen only as 50 dpi contact sheets; higher-resolution zooms cover just a few pages. No written page log |
| 28 | CLAIM_MANIFEST_AND_TESTS | DONE | Manifest 60 claims/0 problems; the session's exact 10-file set gives 111 passed (VERIFIED) |
| 29 | NO_NEW_EXPERIMENTS_VERIFICATION | DONE | NO_NEW_EXPERIMENTS_DETECTED (VERIFIED, §22) |
| 30 | FILES_CHANGED | DONE | 13 files, all in-scope (§6) |
| 31 | LONG_RUNNING_JOBS | DONE | None |
| 32 | GIT_STATE | DONE | Committed, not pushed, `submission/` and `release/` untouched (VERIFIED) |
| 33 | REMAINING_MANUSCRIPT_WEAKNESSES | DONE | Audit record §10 lists them honestly (simulator uncalibrated, four of six policies oracle-informed, SBS untuned, Azure-only causal evidence, timestamp-based pre-specification, title-level PEVA descriptions) |
| 34 | FINAL_MANUSCRIPT_ASSESSMENT | DONE | Final message §34: ready for packaging refresh on acceptance; CRediT in `main.tex` is the one open compliance item |

Tally: DONE 25, PARTIAL 9, NOT_STARTED 0, UNCLEAR 0, BLOCKED 0.

## 6. FILES_CHANGED_SINCE_5887CB4 (VERIFIED)

13 files, +936 / −843.

| Status | File |
|---|---|
| M | `docs/current/PERFORMANCE_EVALUATION_REPRODUCIBILITY.md` (+7: terminology decision) |
| **A** | `docs/current/QUERY_12_COHERENCE_FAIRNESS_AND_COMPRESSION_AUDIT.md` |
| M | `paper/performance_evaluation/FINAL_CLAIM_MANIFEST.json` |
| M | `paper/performance_evaluation/README.md` (1 line: "pre-specified") |
| M | `paper/performance_evaluation/main.tex` |
| M | `paper/performance_evaluation/scripts/build_claim_manifest.py` |
| M | `paper/performance_evaluation/scripts/reference_policy_numbers.py` |
| M | `paper/performance_evaluation/scripts/robustness_numbers.py` |
| M | `tests/test_manuscript_discussion_conclusion.py` |
| M | `tests/test_manuscript_figures.py` |
| M | `tests/test_manuscript_presentation.py` |
| M | `tests/test_manuscript_robustness_numbers.py` |
| M | `tests/test_manuscript_scientific_corrections.py` |

Not changed: `references.bib`, `submission/`, `release/`, `paper/performance_evaluation/submission/`, `experiments/`, `data/`, `src/`, `scripts/`.

## 7. COMMITS_SINCE_5887CB4

```
0ad9a86  2026-09-21 12:43:08 -0400  paper: coherence, fairness disclosure and compression pass (QUERY_12)
```
Co-authored-by trailer present. Not pushed. The commit message is accurate against everything I checked.

## 8. MANUSCRIPT_CURRENT_PAGE_COUNT (VERIFIED)

**31 pages** (the local `main.pdf`, built 12:42:37, after the last `main.tex` edit at 12:41:11; the committed `main.tex` is byte-identical to the working file). Start: 34. Intermediate: 33, 32. LaTeX log: one overfull `\hbox` (2.6 pt, page 1), which the session says predates this work; no undefined references.

## 9. ABSTRACT_CURRENT_STATE (VERIFIED)

- **237 words** (method validated: the same script gives 247 for `5887cb4`, matching the figure in your prompt). Changed after `5887cb4`: yes. ≤250: yes. Target 215–235: **missed by 2 words**.
- Acronyms: only LLM and KV; both are defined in the abstract. No citations.
- Preserves the BurstGPT qualification, the light-load interpretation, the fresh causal result (590/720, 2.00 ms, CI 0.19–3.55 ms), and the concentration/harm qualification (median 0.20 ms, one of 36 windows holds 88%, 14% all-harmful). "pre-specified" replaces "preregistered".
- Reads as a self-contained argument rather than a list of corrections.

## 10. POLICY_FAIRNESS_AUDIT_STATE

**DONE.** The requested table is in audit record §2 (`Policy | information | true length? | deadline? | priority? | KV? | oracle info? | disclosed?`).

Findings (RECORD, with my spot-checks marked):
- `ObservableRequest` omits `actual_output_tokens` by construction, but in these experiments `predicted_output_tokens == true output length`, so the "prediction" is exact.
- Four of six policies use it: ESTF (service estimate), LLF (slack = deadline − now − estimate), WFS (share ÷ estimated steps), and the reference SBS (`kv_cost = prompt + 0.25·predicted_output`, VERIFIED at `kv_constrained_online.py:34`). The two prefill variants use arrival order only.
- Deadline and priority are uninformative (1000 s deadline, priority 1.0, one class).
- All six read the same `ObservableState`. The asymmetry lies in use, not access. Relative to deployment, four of six are oracle-informed.
- Name "estimated-service-first" kept; Methods now says the estimates are exact and calls the four rules "oracle-informed heuristics rather than deployable schedulers". Limitations repeats it once.
- Comparison fairness (same state / arrivals / hardware / candidates / canonicalization / continuation / horizon): fair in setup, not equally realistic. Documented in audit record §3 against `scripts/industry_realism_action_opportunity_phase_a_v1.py` and the protocol `intervention` block; I did not re-open those.

## 11. PRESPECIFICATION_TERMINOLOGY_STATE

**DONE.** Independently confirmed from git:

| Event | Commit | Time (EDT) |
|---|---|---|
| Protocol frozen (`PREREGISTRATION_V1.json` + 3 protocol files) | `b4e6c60` | 2026-09-19 22:12:26 |
| Support map frozen | `38e02bc` | 2026-09-19 23:00:38 |
| Results executed | `b196c3e` | 2026-09-20 00:26:14 (**2 h 14 min** after protocol) |
| Cluster-key correction | `a8fd735` | 2026-09-20 17:38:45 |

- External registry: none. Searches for OSF / AsPredicted / registered-report in `docs`, the manuscript and the protocol JSON returned nothing (TRACE).
- Repository visibility: **public now** (GitHub API, TRACE); the earlier project memory says it was created private. Visibility history and the push time of `b4e6c60` **cannot be verified** (the events API no longer returns that push). The pre-specification claim therefore rests on author-controlled commit timestamps plus content hashes (`ARTIFACT_HASHES_V1.json`). The manuscript states this.
- Manuscript wording now: "…so we call the analysis pre-specified rather than preregistered." The one remaining "preregistered" is inside that sentence.
- **New disclosure found and added:** the first bootstrap keyed clusters on a bare `window_index` (26 clusters), silently merging 10 cross-workload window pairs. It was corrected to `(source_dataset, window_index)` (36 clusters) after outcomes were known. CI moved [0.185, 3.497]→[0.191, 3.546] ms; the decision is unchanged. The manuscript never mentioned this before; it is now in Methods and registered as manifest claim `prespec.protocol_timeline_and_cluster_correction`.
- Historic filenames keep "PREREGISTRATION"; the reproducibility guide records why.
- SBS wording (committed text): the SBS "had the highest mean goodput among the six policies on an earlier benchmark of 240 multi-mechanism scenarios from the same project, run with the same simulator and policy implementations on scenarios other than these trace windows. It was fixed before the fresh evaluation and is not selected, tuned, or changed using fresh outcomes; it is best only among these six policies on that benchmark and is not claimed to be optimal or strong here." "unrelated" is gone. "earlier" remains, in Methods and Limitations.

## 12. CONTRADICTION_AUDIT_STATE

Matrix is in audit record §5. **Status per topic** (all "found/fixed" per RECORD; the last column is this audit's spot-check):

| Topic | Contradiction found? | Fixed? | Spot-check |
|---|---|---|---|
| A BurstGPT | No residual | n/a | Consistent in Abstract, Intro (2×), Results, Discussion, Limitations |
| B arrival scaling | No residual | n/a | "lightly loaded", peak KV <1%, in Abstract, Results, Discussion, Conclusion |
| C physical binding | No residual | n/a | Defined once in Methods; used in Table 2 and Discussion |
| D SBS reserve | **MAJOR** ("unrelated benchmark") | Yes | 0.82 reserve consistent in Methods, Results, Discussion, Limitations |
| E beneficial states | No | n/a | 590/720 (81.9%) consistent |
| F effect magnitude | No | n/a | 2.00 ms mean / 0.20 ms median consistent |
| G relative effect | No | n/a | 12.8% >10% in Results and Table; absent from Abstract and Conclusion |
| H concentration | No | n/a | 88.4% / 88% / 1.27 consistent |
| I VBS | No | n/a | "illustration only; six-policy gap not measured" |
| J diversity | No | n/a | 1.15 alternatives, ESTF≡WFS, full≡chunked |
| K causal interpretation | No | n/a | "local counterfactual", ex post best alternative |
| L vLLM | No | n/a | "bounded", "does not validate" |
| M workload realism | No | n/a | Limitations "Workload overlays" and "External validity" |
| N pre-specification | **MAJOR ×2** (unqualified "preregistered"; undisclosed cluster fix) | Yes | Consistent after edit |

Also fixed: MINOR `Q_SBS` defined-but-unused; MINOR unnumbered `P(B|D)` display; MINOR "reference-dependent/-conditional" drift. No CRITICAL found.

## 13. TONE_AND_REPETITION_STATE

- **Tone: DONE, modest.** Repository-style words in the scientific prose at `5887cb4` vs `0ad9a86`: frozen 1→0, artifact 1→0, manifest 1→0, phase 5→1 (legitimate "phase separation"). `repository`/`committed` rose 0→2 each, deliberately, in the pre-specification paragraph. Section headings lost their "Results:" prefixes. Report-like prose that remains: "earlier" ×4 (benchmark, causal run, post hoc analysis), and "post hoc" ×7 (a legitimate statistical term).
- **Repetition: PARTIAL.** Whole-body counts (regex, crude), `5887cb4` → `0ad9a86`: 88.4% 4→4; 0.20 ms 5→5; "lightly loaded" 5→5; 590 8→8; one-step-oracle 7→7; "never queued" 5→4; "all five other" 3→2; "not validate" 3→2 (the only three that moved, each by one). Removed: the Conclusion's list of limitations, the Methods copy of "disagreement ≠ benefit", and the Discussion's re-quoted §6 numbers. Remains: the 88.4%/0.20 ms/lightly-loaded statements each appear in Introduction, Results, Discussion and Conclusion. The "disagreement ≠ benefit" theme still occurs 5× inside Discussion alone.

## 14. ORGANIZATION_STATE

Committed top level (9 numbered sections + unnumbered back matter):

1 Introduction (+ Contributions) · 2 Related Work and Positioning · 3 Methods (Workloads and simulator; Scheduling policies and reference; Action opportunity and one-step headroom; Endpoints and pre-specification; Statistical analysis) · 4 Where Disagreement Appears (Native replay; Capacity pressure) · 5 Causal Headroom on Untouched Windows (Support; Primary result; Secondary outcomes) · 6 Concentration, Sensitivity, and Reference Dependence · 7 A Bounded vLLM System Correspondence Probe · 8 Discussion (5 subsections) · 9 Conclusion · Appendix A. Interval Diagnostics · AI declaration · Competing interests · Acknowledgements · Funding · Data and Code Availability.

Before: 10 numbered sections, five of them "Results: …", Methods with 6 retrospective-order subsections (incl. "Preregistered secondary outcomes", "Post hoc robustness analyses"). Comparison with recent Performance Evaluation articles: **one article** (Lilou, UMass PDF) was fetched and its headings read (TRACE, 12:14). Its findings were not recorded in any note or in the audit record, so the comparison's result cannot be recovered. Status PARTIAL.

## 15. PAGE_COMPRESSION_STATE

- 34 → 33 → 32 → **31** pages (target 29–31; achieved at the upper end).
- Removed: Figure 3 (six-point pressure-transition curve), Figure 4 (per-regime headroom), old Table 7 (vLLM correspondence, folded into two sentences), one unnumbered equation. vLLM section 332→253 words including the table's markup; its prose alone went 234→253, because the table's content was folded into two sentences. Six captions shortened by 7–27 words (`fresh` 103→76, `regimemap` 77→55, `disagreement` 64→44, `sensitivity` 84→64, `thresholds` 68→61, `robust` 79→72).
- Moved: interval diagnostics (10⁶-replicate percentile, BCa, jackknife) to Appendix A. Pipeline figure moved to the Introduction.
- Not changed: font, margins, row spacing, figure sizes (VERIFIED: no such lines in the diff). Float placement loosened `[tbp]`→`[!htbp]`, which affects page breaks, not sizes.
- Body words (comment-stripped TeX, markup included): 9,204 → 8,513 (about −7.5%).
- **Decision to review:** the two figures and one table were removed as "redundant". I did not verify that no unique content was lost. The removed table was a five-row qualitative correspondence with no numbers.

## 16. TABLE_FIGURE_VISUAL_AUDIT_STATE

**Now 6 tables and 4 figures** (was 7 and 6; your prompt said "7 tables").

Tables: `closest`, `transition`, `fresh`, `secondary`, `thresholds`, `sensitivity`. Figures: `pipeline`, `disagreement`, `regimemap`, `robust`.

What the transcript shows: contact sheets of every page at 40 dpi (pre-final layout), then 6 contact sheets at 50 dpi for the final 31-page layout, plus zooms at 45–60 dpi of pages 4, 12–15, and 13–14 in the final layout. Row-spacing and font checks rest on the diff (no `arraystretch`/`tabcolsep`/font-size change) more than on inspection. Table 4's rows were simplified after an awkward wrap.

Not established: legibility at actual size, per-table unit/significant-digit review, per-figure grayscale check. At ≤60 dpi a page is ~500 px wide. **No visual-check log was saved.** The session's final message asserts these checks passed; the transcript supports contact-sheet-level inspection only.

## 17. MATHEMATICAL_DEFINITION_STATE

- 5 display equations, all `equation` environments and numbered: `eq:chain`, `eq:disagreement` D(s), `eq:latency` (A_LAT, H_LAT), `eq:decomposition`, `eq:anwg`. None unnumbered; no `\[`/`$$` displays (VERIFIED).
- `eq:disagreement` and `eq:anwg` are numbered but never cross-referenced with `\ref` (also true at `5887cb4`); not an error.
- The requested per-symbol audit (D(s), A_LAT, H_LAT, B_LAT, ANWG, aggregation estimands) is asserted done in the final message ("defined at first use") but has no written record. No inconsistency found in my reading.
- Definition audit: PARTIAL. Only 5 of 13 terms are confirmed by the session. My first-use spot-check: SBS defined in the Introduction; ANWG defined with its equation; physical binding and reserve threshold defined at first use in Methods; "no-choice iteration" appears nowhere (also absent at `5887cb4`). **Weaker points:** "regime" is used in the Introduction before its Methods definition; "disagreement state" is used informally in the Introduction; "fresh window" is never formally defined (the text says "untouched windows").

## 18. LITERATURE_CITATION_STATE

- **Fresh Query 12 web checks (6 calls, TRACE):** Performance Evaluation Guide for Authors (HTTP 403); a search for the same (excerpts only); Lilou paper (PEVA structure); Sarathi-Serve arXiv abstract (terminology); vLLM engine-args docs (terminology); a search for Choudhury–Joshi–Wang (cited as `choudhury2025job`).
- **Not fetched during Query 12:** Preble, Jaillet et al., Vidur, Mitzenmacher–Shahout, MorphServe, ServeGen, Libra, Llumnix, SOLA, Mooncake, PEVA papers. These were verified earlier in Queries 9–11 (same conversation, 38 web calls; recorded in the Query 10/11 documents). The Query 12 record's phrase "re-checked against sources" therefore means re-read against earlier evidence, not fresh fetches.
- `references.bib` is unchanged in Query 12. The Related Work section was lightly edited (653→640 words).
- All seven named keys (`preble2025`, `jaillet2025online`, `vidur2024`, `mitzenmacher2025queueing`, `morphserve2025`, `servegen2025`, `libra2026`) are cited in the committed manuscript. No `\cite` exceeds 4 keys (RECORD).
- PEVA-paper descriptions remain title-level (Crossref exposes no abstracts).
- Unchecked: whether any Related Work sentence edited in Query 12 says more than its source supports. Unlikely given the small edit, but not tested.

## 19. LIMITATIONS_CONCLUSION_STATE

- **Limitations:** 10 bolded items, 354 words (351 at `5887cb4`, so slightly longer, not shorter). Direct tone (no "unfortunately", "severe", "we make no claim"). **True-output-length disclosure is present in two places** ("Policy portfolio": four use exact lengths; "Workload overlays": predicted output lengths equal true lengths, so "length-aware rules face no prediction error"). Not shortened despite the audit record's claim.
- **Conclusion (219 words by my count, 220 reported):** *"This paper asked whether an LLM-serving scheduler ever faces an executable alternative worth taking. … Adaptation is justified by where the measured opportunity lies, not by its average."* It restates 2.00 ms, 0.20 ms, 0.33 ms and 88.4% (all already in Results; no new statistic), lists no limitations, and ends on the main takeaway. One sentence is ambiguous: "the reference alone differs from the rest of the portfolio" restates the abstract's "all five other policies differ from it".
- Practical impact: "The study demonstrates no production savings, deployed-selector benefit, or capacity-planning gain." I found nothing stronger than the evidence. One hedged bullet ("a fixed scheduler may be the appropriate policy") is the strongest practical statement.

## 20. JOURNAL_COMPLIANCE_STATE

| Item | State |
|---|---|
| Abstract ≤250 | 237 ✔ |
| Keywords (1–7) | 6 ✔ |
| Numbered sections | 9 ✔; Appendix A uses an **unnumbered** heading (elsarticle's `\appendix` forces a page break) |
| Display equations | 5/5 numbered ✔ |
| Citation style | numeric (`elsarticle-num`) ✔ |
| Tables / figures | 6 / 4 |
| AI declaration | Present ✔ |
| Competing interests, Acknowledgements, Funding | Present ✔ |
| Data and Code Availability | Present, but wording still awaits the release refresh (per record §9) |
| **CRediT** | **Only in `submission/credit_authorship_statement.txt`, not in `main.tex`. Open** |
| Editable source | `main.tex` ✔ |
| Page count | 31 |
| Page limit | **NO EXPLICIT PAGE LIMIT FOUND IN THE CURRENT AUTHOR GUIDANCE CHECKED.** Caveat: the official guide returned HTTP 403 to the fetcher in this query. The statement rests on search excerpts (250-word abstract; 3–5 highlights ≤85 characters) and on the earlier roadmap note "PE has no page limit" |

## 21. CLAIM_MANIFEST_AND_TEST_STATE (VERIFIED, read-only runs)

- `build_claim_manifest.py --check`: **60 claims checked, 0 problems** (run without writing, at 12:39 and 12:44).
- Net claim change vs `5887cb4`: 60→60. Dropped `figure3.azure_code_active_cap`; added `prespec.protocol_timeline_and_cluster_correction`. The audit record's "61 → 60" describes an intermediate build; it is not the net figure.
- Tests, the session's exact 10-file set: **111 passed** (reproduced, 11 s). `tests/test_manuscript_*.py` alone: 73 passed.
- Tests changed: 5 files modified, **0 added**. Deliberate changes: section titles, Discussion number pins, table count 7→6, figure count 6→4, "fixed" for "frozen", "pre-specified". **One test deleted** (`test_figure_3_axis_direction_is_documented_and_matches_the_artifact`, tied to the removed figure) and one renamed (`…six_figures…`→`…four_figures…`). Assertion lines removed vs added: 13 vs 10 (crude count). The deletions correspond to removed content. Whether that meets "do not weaken tests" is your call.
- `main.tex` will differ from the manifest's recorded sha if edited again, and the `--check` note says so.

## 22. NO_NEW_EXPERIMENTS_CHECK

**NO_NEW_EXPERIMENTS_DETECTED.**
- No file under `results/`, `experiments/`, `data/` or `configs/` was modified after the Query 12 prompt (12:10:55). `git diff 5887cb4..HEAD` contains none.
- The session made 86 Bash calls: git, grep/sed, Python text-editing scripts, `pdflatex`/`bibtex`, `pdftoppm`, `pytest` on manuscript tests, and read-only `gh api`. A keyword scan flagged 5 commands; all were manuscript-editing heredocs containing the word "simulator".
- No tmux/SLURM launch, no vLLM process. Simulator and policy source under `src/` unchanged.

## 23. LONG_RUNNING_JOBS

None started by Query 12. See §4. This audit started no long job.

## 24. GIT_FILE_INTEGRITY

- All pre-existing changes were preserved. This audit ran no `checkout`, `reset`, `stash`, `restore` or `clean`.
- Inside the Query 12 session (not this audit): one `git checkout HEAD -- build_claim_manifest.py` at 12:34 restored the builder after a bad slice, and its edits were then re-applied. The committed builder is correct (manifest check passes). No work was lost.
- `main.pdf` and LaTeX build files are gitignored: they will not appear in `git status` and are not in the commit.
- This audit ran the manifest check and tests with `PYTHONDONTWRITEBYTECODE=1 -p no:cacheprovider`, so it did not write `.pyc` or `.pytest_cache`.
- This report is the only new file, and it is uncommitted.

## 25. QUERY_12_PROGRESS_PERCENT

**≈ 90%** (weighted by importance, not by file count).

Every one of the 34 requested sections was produced, the manuscript is committed, the manifest and 111 tests pass, and no experiment ran. The remaining ~10% is depth, not missing work: (a) the visual check was at contact-sheet resolution with no page log; (b) literature checks in this pass were terminology-level only, and source re-verification relied on earlier queries; (c) the organization comparison used one article and saved no notes; (d) the definition audit is unreported for 8 of 13 terms; (e) the repetition removal was partial; (f) two figures and a table were removed and the "redundant" judgement is unreviewed.

## 26. LAST_CONFIRMED_COMPLETED_STEP

Commit `0ad9a86` at 12:43:08 EDT, memory entry written, and the final 34-section report delivered at ~12:43:52. The session is idle.

## 27. NEXT_UNFINISHED_STEP

Your acceptance decision on the manuscript pass. Per the Query 12 rules, packaging (`submission/`, `release/`, Zenodo, release tag, remote push) stays frozen until you accept it. Open compliance item: add a CRediT section to `main.tex`.

## 28. SAFE_RESUME_ACTION

Do not re-run Query 12; it is finished. If you want to continue, first read `docs/current/QUERY_12_COHERENCE_FAIRNESS_AND_COMPRESSION_AUDIT.md`, then `git show --stat 0ad9a86`, then open the 31-page PDF. Decide accept vs. revise before any packaging work.

If you choose to revise, the narrow follow-ups this audit suggests (none performed) are:
1. A ≥150 dpi page-by-page pass over the six tables and four figures, with a saved log.
2. Confirm that removing Figures 3 and 4 and old Table 7 loses no unique content.
3. Add the CRediT section to `main.tex`.
4. Fix the "regime" and "disagreement state" first-use ordering; optionally define "fresh window".
5. If wanted, a fresh source check for the Related Work sentences edited in Query 12.

After acceptance, the packaging query should cover the stale items listed in audit record §9: highlights line 2, cover-letter "preregistered", Data and Code Availability wording, the regenerated `submission/` and `release/` copies, and the tracked review PDF.

## 29. OPEN_UNCERTAINTIES

1. **Pre-specification timing is unverifiable independently.** It rests on commit timestamps and content hashes; the push time of `b4e6c60` and the repository's visibility history could not be recovered. Your earlier project memory says the repository was private, while the GitHub API now reports it public.
2. **Visual quality is unproven at reading resolution** (≤60 dpi). Overfull-box and float-placement checks come from the LaTeX log, not from page inspection.
3. **Citation verification was not refreshed** in Query 12 (see §18). The audit record's wording overstates this.
4. **Removed content.** Two figures, one table, one test and one manifest claim were deleted. I did not confirm their information survives elsewhere.
5. **Limitations and repetition.** Both improved less than the audit record implies (§13, §19).
6. **Unnumbered Appendix A** against the "numbered sections" guidance is a reviewer judgement call.
7. **Length targets:** abstract 2 words over 215–235; conclusion at the 220 ceiling.
8. **The page-limit finding is inconclusive**, because the official guide could not be fetched.
9. **Stale project memory.** `github_repo.md` still says the repository is private and `main` is stale. I did not edit it, because this task allowed only this one file.
