# LLM 2026 Venue Compliance Audit

Date: 2026-08-24

Scope: final submission-preparation audit for `paper/llm2026/main.tex`.

## Sources Checked

- Official LLM 2026 paper categories page: `https://www.american-cse.org/LLM2026/paper_categories`
- Official LLM 2026 paper submission page: `https://www.american-cse.org/LLM2026/paper_submission`
- Official LLM 2026 review-process page: `https://www.american-cse.org/LLM2026/paper_review_pocess`
- Springer LNCS author/editor information: `https://link.springer.com/series/558/information-for-authors-and-editors`
- Springer Nature manuscript guidelines: `https://www.springernature.com/gp/authors/publish-a-book/manuscript-guidelines`
- Springer Nature AI guidance: `https://www.springernature.com/gp/group/ai/ai-guidance-for-our-researchers-and-communities`
- Springer Nature book publishing policies: `https://www.springernature.com/gp/policies/book-publishing-policies`

## Compliance Table

| Requirement | Source | Manuscript implication | Status |
|---|---|---|---|
| Full/regular research paper page limit is 12-15 pages in one-column Springer Nature formatting, or 6-8 pages in two-column formatting; figures, tables, and references count. | LLM 2026 paper categories page | Use LNCS one-column template and keep `main.pdf` <=15 pages including references. | PASS: compiled PDF is 15 pages. |
| Review-submission first page should include title, author names, affiliations, city/country, email/contact author, about-100-word abstract, and up to five topical keywords. | LLM 2026 paper submission page | Current draft has title, abstract, keywords, and LNCS-compatible author block. | ACTION_NEEDED: author block is still placeholder and must be replaced unless blind review is required. |
| Review process is peer review by 2-4 peers; PC-authored papers are reviewed double-blind. | LLM 2026 review-process page | Default appears non-anonymous, but double-blind may apply if an author is on the program committee. | ACTION_NEEDED: author must confirm whether any PC-conflict double-blind rule applies. |
| Springer LNCS proceedings templates are the expected computer-science proceedings template. | Springer LNCS author/editor information | Use `llncs.cls` and `splncs04.bst`. | PASS: official template is integrated. |
| Supplementary material can be added to SpringerLink. | Springer LNCS author/editor information | Supplemental artifact release is allowed in principle, subject to venue upload workflow. | PASS/OPTIONAL. |
| Springer manuscript guidelines encourage abstracts <=200 words and 3-6 keywords; LLM site asks about 100 words and max five keywords. | Springer guidelines and LLM submission page | Keep abstract near 100 words and <=5 keywords. | PASS: abstract is 88 words, 4 keywords. |
| AI tools must not be listed as authors; AI use should be transparently declared where applicable, and authors remain accountable. | Springer Nature AI guidance and manuscript guidelines | Add generative-AI disclosure in credits/acknowledgment area. | PASS. |
| Competing interests should be declared where required. | Springer Nature book publishing policies / LNCS template credits | Include disclosure of interests. | PASS: no competing interests statement present. |
| Acknowledgments are supported by LNCS `credits` area; blind-review implications depend on review mode. | LNCS template and LLM review policy | Include acknowledgments for non-blind submission; remove or anonymize if PC-conflict double-blind applies. | PASS with author-confirmation caveat. |
| Code/data availability is not explicitly required by the LLM page, but is compatible with Springer-style declarations and improves reproducibility. | Springer policies and LNCS credits convention | Include public GitHub statement only if non-anonymous submission is acceptable. | PASS with anonymization caveat. |

## Unresolved Venue Requirements

- `VENUE_REQUIREMENT_UNVERIFIED`: exact EasyChair/upload checklist fields were not available from the static site audit.
- `ACTION_NEEDED`: replace anonymous author metadata before non-blind submission.
- `ACTION_NEEDED`: if the author is subject to the LLM 2026 PC-authored double-blind route, remove or anonymize acknowledgments and public repository links for the review submission.

