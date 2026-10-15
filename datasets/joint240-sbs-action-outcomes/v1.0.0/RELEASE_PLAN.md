# Conditional GitHub/Zenodo release plan

Do not execute this plan without explicit authorization.

## Editorial context

The associated manuscript is **PEVA-D-26-00474**, formally submitted to
Performance Evaluation on 2026-09-22. The author supplied a 2026-10-07
notification saying it was currently under review. The latest live editorial
status remains unconfirmed in this preflight. Do not confuse it with
PEVA-D-26-00487, a separate rejection. The public preprint is
https://ssrn.com/abstract=7584825.

## Conditional sequence

1. Obtain author approval for the proposed CC BY 4.0 data license, attribution,
   exact contents, version, and release timing. Use `PUBLIC_RELEASE_ALLOWLIST.md`
   as the file-level boundary.
2. Keep the Joint-240 supplement explicitly historical, exploratory, synthetic,
   and simulator-based. Do not merge it into the production-derived
   confirmatory v1.1.0 manuscript archive or describe it as deployed
   scheduler evidence.
3. If the author elects to associate it with PEVA-D-26-00474, confirm with the
   editor/portal whether it should be supplied as supplementary material or
   cited as an external repository dataset. Elsevier's general research-data
   policy encourages appropriate sharing and persistent citation, but does not
   by itself require or prohibit this release. Do not contact the journal
   without authorization.
4. Add one historical Joint-240 supplement directory to the existing
   `SoroushVahidi/llm-serving-heuristic-evolution` repository using the
   approved version and license. Do not alter the active manuscript archive
   silently.
5. After GitHub review, create one matching Zenodo archive with the same
   allowlist, version, attribution, and public checksums. Obtain a DOI and
   update the manuscript data-availability statement only if the author
   chooses to connect this supplement to the active submission.

## Metric interpretation

For every retained alternative row, `q_sbs_*` is the terminal metric after
the candidate action and fixed SBS continuation, `q0_sbs_*` is the SBS
reference metric for the same state, and `a_sbs_* = q_sbs_* - q0_sbs_*`.
The candidate publishes `q0` in each outcome row, so the omitted public
scenario utility table is not required to interpret or reconstruct the
within-state contrast. For cross-checking, the already public repository
utility matrix has one SBS utility per scenario; it may be cited or obtained
from the repository rather than copied into this supplement.

The original campaign covered 240 scenarios; this candidate contains rows
from 236 scenarios because four scenarios had no disagreement and therefore
no retained labeled states.

## Citation and attribution text

Suggested citation:

> S. Vahidi, *Joint-240 SBS Disagreement and One-Step Counterfactual Action Outcomes*, historical
> exploratory synthetic simulator dataset, version `1.0.0` (author-approved
> release pending), GitHub repository
> `SoroushVahidi/llm-serving-heuristic-evolution`; matching Zenodo record to
> be assigned.

Suggested attribution:

> Copyright © 2026 Soroush Vahidi. Joint-240 SBS alternative-action outcomes
> and synthetic scenario metadata. Licensed under CC BY 4.0, **only if and
> after the author approves and the released copy includes the corresponding
> license notice**. BurstGPT and Azure 2023 are upstream references only; no
> third-party trace records are redistributed.

No GitHub commit, push, tag, release, Zenodo upload, or publication was
performed in Phase 6C or this Phase 6D preflight.
