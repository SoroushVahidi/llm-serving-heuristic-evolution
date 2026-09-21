# QUERY_9 — Whole-manuscript coherence and scientific-writing audit

Audit target: `paper/performance_evaluation/main.tex` at branch `release/peva-v1-20260920` (HEAD `77cdde9`).
Relative to the visual-freeze commit `cfc5a66`, the only manuscript change is the Data and Code Availability
paragraph (Zenodo v1.1.0 wording); `paper/performance_evaluation/submission/main.tex` and
`release/peva_submission_final/manuscript/main.tex` were byte-identical to the canonical file before editing.
No scientific number, estimand, figure or dataset is changed by this audit.

## Part 1 — Diagnosis (written after one continuous read, before any edit)

**Central question.** Does the live system ever reach a state where another executable scheduling action exists, and
is taking it worth anything?

**Answer.** Under abundant resources: no alternative action in ~1.0 million decision states. Capacity binding
(active-sequence or KV caps, not arrival-rate scaling) creates alternatives. Where they exist, forcing one lowers
latency in 81.9% of 720 fresh states, but the effect is small at the median (0.20 ms) and concentrated (one window holds
88.4% of the total). So adaptation should be justified stage by stage before selector complexity is added.

**The 2–4 real insights.** (1) Opportunity is a property of *which resource binds*, not of load. (2) Disagreement,
benefit, and material benefit are three different stages that each can fail. (3) The sign of headroom is robust; its
magnitude is concentrated, so a mean alone misleads.

**Does it read as one article?** The abstract, introduction chain (Eq. 1) and conclusion do. The middle does not always.
The body still carries the seams of the research phases that produced it.

### Findings

| # | Problem | Where | Kind |
|---|---------|-------|------|
| F1 | RQ2 and Contribution 2 say the pressure transition "reproduces on untouched windows". The records show it reproduced for Azure code and conversation but **not for BurstGPT** (0 disagreement states in all 360 valid fresh BurstGPT conditions; the original windows had disagreement at cap 16/8 and KV 16,000). The body never reports the fresh support map, and Limitations attributes the missing BurstGPT causal regime to "no selected regime" rather than to the absence of disagreement. | Intro RQ2/C2, Sec. 5–6, Limitations | Mismatched claim / omission |
| F2 | The ANWG-saturation material is a separate results section placed *after* the robustness analysis, yet its only job is to justify the choice of latency as primary endpoint, which the reader needs *before* the fresh results. It refers to "the earlier pressure study" and "the earlier post hoc latency analysis", which the reader never met. | Sec. 8 + Methods "Objective audit" | Narrative fragmentation, report style |
| F3 | The Discussion re-derives Section 7 almost paragraph for paragraph (means, CIs, concentration, interval diagnostics), so the same 10+ numbers appear in Abstract, Intro, Sec. 6, Sec. 7, Sec. 10 and Conclusion. | Sec. 10.1 | Repetition |
| F4 | The "Performance Evaluation neighbours" paragraph appears twice (Intro and Related Work) with the same four citations. "Relation to modern adaptive systems" (Discussion) re-lists the Related Work. | Sec. 1, 2, 10.5 | Repetition |
| F5 | "Reproducibility and Artifact Scope" duplicates Data and Code Availability, sits between Discussion and Conclusion, and mentions "earlier selector studies" (repository history). | Sec. 11 | Repository-report prose |
| F6 | Four-tier evidence hierarchy: Tier 4 is never used, Tier 2 is never distinguished from Tier 3 in the results. It is scaffolding from the research plan. | Sec. 3.1 | Procedural detail |
| F7 | Report-style vocabulary: "campaign", "verdict", "historical", "the causal labeling plan selected", "the earlier … study", "frozen …" (dozens), "returned empty dictionaries". Terminology drift: "preregistered" / "pre-specified" / "pre-frozen". | throughout | Style |
| F8 | "P(D) is often below one percent in the selected regimes" understates: Table 3 shows it is below 0.5% in **every** selected regime (max 0.456%). | Sec. 6 | Imprecise claim |
| F9 | How the five causal regimes were chosen (mechanically, from the support map, without latency) is never stated, though it is what makes the confirmatory claim credible. | Sec. 3.1 | Omission |
| F10 | The same caveat ("opportunity/headroom does not prove a deployed selector is worthwhile"; "vLLM does not validate magnitudes") is restated 5–6 times. | Abstract, Intro, C5, Sec. 6.1, Sec. 8, Sec. 10 | Redundant caveats |
| F11 | Excess precision and log-like phrasing in results ("0.001995791 s (1.9958 ms)" plus a second seconds/ms interval pair). | Sec. 6 | Procedural detail |

### What was checked and found consistent

All 720/590/589 counts, the regime table sums (states 720, windows 36, headroom share 100.00), the 1.996 ms
state-weighted mean recomputed from the regime means, the 1.93 ms KV-16,000 contribution, 457 and 289 leave-one-out
state counts, the ~six-fold aggregation ratio, the ~1.0 million native decision states, and the 1,080 = 926 + 38 + 96 + 20
condition classes.

## Part 2 — Changes made (`paper/performance_evaluation/main.tex`)

| Finding | Change |
|---------|--------|
| F1 | New subsection 6.1 "Support on untouched windows" reports the fresh support map: Azure onsets reproduce (KV 16,000; cap 8 code / 4 conversation), arrival scaling action-null, **BurstGPT shows no disagreement at any pressure setting**. RQ2, Contribution 2, abstract, intro, Discussion 9.3, Limitations ("Workloads") and Conclusion now say so. Sec. 5 is labelled as the *original* windows. New manifest claim `fresh.support_transfer` (recomputed from `FRESH_SUPPORT_WINDOW_CONDITIONS_V1.csv`). |
| F2 | ANWG saturation moved from a post-robustness results section into Methods 3.5 "Choice of primary endpoint" (label `sec:endpoint`), where it justifies latency as the endpoint. "Earlier study" wording anchored to "an earlier causal run on the original pressure windows". Old Sec. 8 removed. |
| F3 | Discussion 9.1 no longer re-derives the interval diagnostics; it points to Sec. 7. Numbers that interpret the result (median, weighted means, 0.5/2 ms shares, w11/KV concentration, leave-one-out means) stay. |
| F4 | Intro's "Performance Evaluation neighbours" paragraph reduced to one sentence pointing at Sec. 2. "Relation to modern adaptive systems" subsection removed (restated Sec. 2). |
| F5 | "Reproducibility and Artifact Scope" section removed. Its unique content (provenance, non-overlap check, manifests, vLLM environment) is one sentence at the end of Methods 3.1; the seed was already in Methods. |
| F6 | Four-tier hierarchy replaced by a plain statement of which experiment answers which RQ. |
| F7 | "campaign", "verdict", "historical", "labeling plan", "pre-frozen"/"pre-specified", "returned empty dictionaries" removed or replaced; "frozen" reduced from ~20 to 5 uses (one is pinned by a test). |
| F8 | "often below one percent" -> "below 0.5% in every selected regime (Table 3)". |
| F9 | Methods now states how the five regimes were chosen (mechanically from the support map, no latency outcome). |
| F10 | Duplicate scope caveats removed from Sec. 6.2 and Sec. 10; caveats stay in abstract, Sec. 8, Limitations, Conclusion. |
| F11 | Intro results paragraph shortened. "Supports five practices" -> "suggests". |

Abstract: 225 -> 241 words (limit 250). Pages: 30 -> 30. Build has no undefined references; the one overfull box (page 1) predates this audit.

**Tests updated (structure only, no numbers relaxed except two):** section-order and slice markers in
`test_manuscript_robustness_numbers.py`; Discussion slice end, subsection list, and the vLLM-section check in
`test_manuscript_discussion_conclusion.py`. The Discussion no longer re-quotes the 10^6-replicate percentile and BCa
intervals (they remain in Sec. 7.4 and are still checked there). 78/78 manuscript tests pass; manifest `--check`: 47
claims, 0 problems; `main.bbl` regenerated (citation order changed because the intro no longer cites the four
Performance Evaluation papers first).

## Part 3 — Not changed / needs a decision

1. **Derived copies are now stale.** `paper/performance_evaluation/submission/` and
   `release/peva_submission_final/` (main.tex, main.bbl, PDF, source zip, checksums, MANIFEST) still hold the
   pre-audit text. Regenerate them with the existing builders after you accept the edits; not done here because it
   rewrites checksums and archive metadata.
2. **Data and Code Availability untouched.** It still says the v1.1.0 materials are "in the GitHub repository only until
   the v1.1.0 version of the archive is published". That sentence will be wrong once the archive is published.
3. **Report inaccuracy found.** `FRESH_SUPPORT_MAPPING_REPORT_V1.md` §4 says BurstGPT `kv_8000` is invalid/truncated.
   The CSV shows all 360 BurstGPT conditions valid; the 21 invalid conditions are Azure code (20) and Azure
   conversation (1). The manuscript follows the CSV. The report was not edited.
4. Judgment calls left alone: the citation-dense first paragraph of Related Work, the "not X, but Y" cadence in several
   sentences, and the level of detail in Sec. 7.4 and Sec. 8. Section labels `sec:phasea`/`sec:phaseb` are internal and
   invisible to readers.
5. The query text supplied was cut off in its Section 3 examples list, so any later sections of it (4 onward) were not seen.
