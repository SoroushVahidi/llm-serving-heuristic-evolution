# LLM 2026 Author Metadata Checklist

Date: 2026-08-24

Manuscript: `paper/llm2026/main.tex`

## Current Status

The manuscript currently preserves an anonymous LNCS author block:

- `Anonymous Author(s)`
- `Anonymous Institution, City, Country`
- `anonymous@example.com`

Do not deanonymize until the applicable LLM 2026 review mode is confirmed.

## Anonymization Decision

Official LLM 2026 review text states that papers whose authors include one or
more program-committee members are evaluated using the double-blinded review
process.

`AUTHOR_CONFIRMATION_REQUIRED`

Question for the author:

- Does any author of this paper serve on the LLM 2026 program committee or
  otherwise fall under the venue's PC-conflict double-blind route?

## If Normal Identified Submission Applies

Replace the anonymous block with:

- [ ] Exact author name(s)
- [ ] Author order
- [ ] Affiliation(s)
- [ ] Department/lab if desired
- [ ] Institution
- [ ] City
- [ ] Country
- [ ] Email address for each author if required
- [ ] Contact/corresponding author marked as required by the portal
- [ ] ORCID(s), if supported or desired
- [ ] `\authorrunning{...}` shortened appropriately
- [ ] `\titlerunning{...}` checked after author insertion

Keep acknowledgments, Data and Code Availability, Generative AI Use, and
Disclosure of Interests in the review manuscript.

## If PC-Conflict Double-Blind Submission Applies

Preserve or strengthen anonymization:

- [ ] Keep anonymous author block.
- [ ] Remove or anonymize the public GitHub URL in Data and Code Availability.
- [ ] Remove acknowledgments for review or replace with a blind-review-safe
      placeholder if the portal/template permits.
- [ ] Ensure generated PDF metadata does not reveal author identity.
- [ ] Ensure supplemental files, if any, do not reveal author identity.

The non-blind author block should be restored only for camera-ready submission.

