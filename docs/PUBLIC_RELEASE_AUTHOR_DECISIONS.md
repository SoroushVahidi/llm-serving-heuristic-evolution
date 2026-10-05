# Public Release — Author Decisions Only

> **Historical document.** This pre-publication decision list records the project state as of 2026-08-24 and is retained for provenance; its present-tense statements ("current", "active", "next") describe that date. The repository has since been made public, and its instructions are not current release steps. For the current study, start at [`docs/current/README.md`](current/README.md).

Date: 2026-08-24

Technical preparation (manifest, allowlist, dry-run, docs) is complete.
Check boxes below that **only the author** can resolve. Do not flip GitHub
visibility until these are decided.

## Release strategy

- [ ] Confirm **identified** vs **anonymized** public-artifact strategy
- [ ] Confirm venue/public-artifact policy **externally** (not determined from this repo alone)
- [ ] Approve Strategy A (identified) **or** request Strategy B sanitization list

## Data / licensing

- [ ] Confirm third-party dataset redistribution rights for BurstGPT and Azure 2023 traces
- [ ] Confirm raw CSVs remain **excluded** (recommended) vs explicitly redistributed with attribution
- [ ] Confirm treatment of `data/public_trace_corpus_v1/*/records.parquet`

## Content inclusion

- [ ] Confirm whether `paper/` directory (source + PDF) should be public
- [ ] Confirm whether historical `docs/audits/` should be included (recommended YES)
- [ ] Confirm exclusion of operational `docs/current` handoffs (`RESUME_HERE`, `NEXT_ACTIONS`, `WORK_STATUS`, `*HANDOFF*`)
- [ ] Confirm exclusion of root scratch scripts (`p2`–`p8`)
- [ ] Confirm treatment of `artifacts/` (default EXCLUDE)
- [ ] Confirm treatment of `datasets/` (default EXCLUDE)

## Security / history

- [ ] Confirm no local `.env` / credentials will be force-added
- [ ] Confirm masked token audit logs stay unpublished
- [ ] Confirm any required credential rotation/history cleanup (none found as full secrets in this audit; still author-verify)

## Final publication

- [ ] Confirm final release branch/tag contents match dry-run allowlist
- [ ] Approve making the GitHub repository public
- [ ] Choose timing: public now vs after acceptance / camera-ready

## Pointers

- Manifest: `docs/PUBLIC_RELEASE_MANIFEST.md`
- Allowlist: `docs/PUBLIC_RELEASE_ALLOWLIST.txt`
- Excludelist: `docs/PUBLIC_RELEASE_EXCLUDELIST.txt`
- Data policy: `docs/DATA_RELEASE_POLICY.md`
- Dry-run tree (local): `/tmp/llm-serving-public-release-dryrun-20260824`
