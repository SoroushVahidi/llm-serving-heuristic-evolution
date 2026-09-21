# USER_CONFIRMATIONS — items requiring the author's decision

These are the only items that need the author's explicit decision before
submission. Everything else (technical verification, builds, checksums,
Zenodo metadata) is machine-verified and listed in the release report.

Technical items are NOT listed here: manuscript builds (30 pages, 78/78
tests), claim manifest (46/46), robustness/correction tests (27/27), source
ZIP independent compile, release bundle self-check, highlights character
counts, and the absence of placeholder/fake DOIs are all verified.

---

## 1. Corresponding author email

The manuscript front matter lists the author and affiliation; the
corresponding-author email is entered in the submission portal and appears
in `cover_letter.txt` as **sv96@njit.edu** (taken from the repository git
author metadata).

**Confirm:** is sv96@njit.edu the address to use as corresponding author?
ORCID (optional, portal-only) is not in the manuscript.

## 2. CRediT authorship contribution statement (sole author)

Candidate statement (see `declarations/credit_authorship_statement.txt`):

> Soroush Vahidi: Conceptualization, Methodology, Software, Validation,
> Formal analysis, Investigation, Data curation, Visualization, Writing –
> original draft, Writing – review & editing.

Deliberately NOT claimed (no project-record support): Funding acquisition,
Supervision, Resources, Project administration.

**Confirm:** the listed roles, and that none of the excluded roles apply.

## 3. Acknowledgement sentence

Current text (page 26): "The author thanks his mother for her continued
support."

**Confirm:** keep unchanged (it is grammatical and appears intentional), or
edit.

## 4. Pronoun consistency

The competing-interest declaration uses "The author declares that **they**
have no known competing financial interests…"; the acknowledgements use
"**his** mother".

**Confirm:** leave both as-is, or harmonize (e.g., "the author" /
"their") to a single style.

## 5. Prior-submission disclosure wording

The cover letter states: "This manuscript is not under consideration for
publication elsewhere." It does not mention the withdrawn, unpublished
LLM 2026 predecessor. The repository roadmap suggested optional
transparency about a prior withdrawal.

**Confirm:** the current wording, or add a sentence about the prior
withdrawn submission (only if it was formally submitted and withdrawn).

## 6. Suggested reviewers (optional)

Suggested reviewers are optional for Performance Evaluation. Four verified
candidates are prepared in `reviewer_suggestions/suggested_reviewers.md`
(name, affiliation, institutional email, expertise match, conflict screen,
sources; verified 2026-09-20):

1. Gustavo de Veciana — UT Austin — deveciana@utexas.edu
2. Jim G. Dai — Cornell University — jd694@cornell.edu
3. Michael Mitzenmacher — Harvard University — michaelm@eecs.harvard.edu
4. Ana Klimovic — ETH Zurich — aklimovic@ethz.ch

**Confirm:** whether to enter them in the portal, and re-check that none is
on the journal's editorial board or otherwise conflicted (not checkable
from the repository).

---

## Non-blocking note on the Zenodo archive DOI

Zenodo's publish endpoint was unavailable during QUERY_8 and is still
returning 404 in QUERY_9 (service-side). The manuscript Data Availability
therefore names the **published v1.0.0 record**
(10.5281/zenodo.22865294; concept 10.5281/zenodo.22865293) and states that
the v1.1.0 version — containing the robustness outputs, the corrected
derivative and the continuation shards — is published in the same record
under its Zenodo-assigned DOI, with the GitHub repository carrying all
materials in the interim. When Zenodo's publish service recovers, the
assigned v1.1.0 DOI should be inserted into `references.bib`, the Data
Availability text, `CITATION.cff`, the release README, and the submission
package, then the manuscript and package rebuilt (a metadata-only change;
no scientific content changes). This does not block submission, since no
nonexistent DOI is cited anywhere.
