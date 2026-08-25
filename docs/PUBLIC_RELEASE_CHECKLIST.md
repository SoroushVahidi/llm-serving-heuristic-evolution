# Public Release Checklist

Date: 2026-08-24

Repository: `https://github.com/SoroushVahidi/llm-serving-heuristic-evolution`

This checklist does **not** change GitHub visibility, create releases, push
commits, or upload artifacts.

Venue anonymity policy for any simultaneous paper submission **cannot be
determined solely from repository-local evidence**. Authors must apply the
target venue's rules separately. The manuscript currently includes identified
author metadata under `paper/llm2026/`.

---

## SAFE FOR PUBLIC RELEASE

(After author confirmation of remaining blockers below.)

- [x] MIT `LICENSE` present
- [x] `CITATION.cff` present (software citation; paper venue/DOI not claimed)
- [x] Public README / REPRODUCIBILITY / repository map / results index
- [x] `.gitignore` covers `.env`, caches, most `results/`, logs, HF caches
- [x] No full API keys / private key blocks found in working-tree text scan
- [x] Core simulator smoke path documented and verified (`scripts/smoke_test.py`)
- [x] Paper PDF present at `paper/llm2026/main.pdf`
- [x] Frozen experiment summaries under `experiments/` for the paper evidence chain

## NEEDS AUTHOR DECISION

- [ ] Whether to make the GitHub repository public now vs after acceptance /
      camera-ready
- [ ] Whether double-blind review requires a separate anonymous artifact mirror
      (author name, institution, and GitHub identity are present in README,
      CITATION.cff, manuscript, and remote URL)
- [ ] Whether to keep or relocate root scratch files (`p2_*` … `p8_*`,
      `opencode.json`) before a public tag
- [ ] Whether `docs/current/` operational handoffs (`RESUME_HERE`,
      `NEXT_ACTIONS`, agent notes) should ship publicly or move behind an
      `archive/` / release-branch filter
- [ ] Redistribution of any raw BurstGPT / Azure traces vs manifests-only
- [ ] Whether cluster provenance paths (`/mmfs1/...`, `/home/...`) inside
      historical JSON should be redacted in a release branch
- [ ] Whether `artifacts/` Wulver bundles should be included in a public tag
- [ ] ORCID / citation year updates once the paper is accepted

## BLOCKER BEFORE PUBLIC RELEASE

- [ ] **Author review of secrets:** confirm no untracked local `.env` or
      credential files will be force-added; confirm masked token audit logs
      under gitignored `results/baseline_api_audit/` stay unpublished
- [ ] **Third-party data license confirmation** before publishing any raw
      traces or non-manifest redistributable tables
- [ ] **Visibility decision:** do not flip GitHub to public until the above
      decisions are recorded
- [ ] **No force-push / history rewrite** if any secret were ever committed
      historically (none found in this scan as full secrets; still verify with
      author tooling if unsure)

---

## Active-submission / anonymity notes

Present and identifying:

- Author name and institution in `paper/llm2026/main.tex` / PDF
- Author in `CITATION.cff` and README citation block
- Public GitHub username in repository URL
- Acknowledgments in manuscript credits

If a double-blind route is required, prepare an anonymized package separately;
do not assume this repository tree is anonymous.

## Related docs

- Manifest: `docs/PUBLIC_RELEASE_MANIFEST.md`
- Allowlist: `docs/PUBLIC_RELEASE_ALLOWLIST.txt`
- Excludelist: `docs/PUBLIC_RELEASE_EXCLUDELIST.txt`
- Data policy: `docs/DATA_RELEASE_POLICY.md`
- Author decisions: `docs/PUBLIC_RELEASE_AUTHOR_DECISIONS.md`
- Prior manuscript-oriented checklist:
  `docs/current/llm2026_public_release_checklist_20260824.md`
- Readiness audit:
  `docs/current/public_repository_readiness_audit_20260824.md`
- Dry-run tree (local only): `/tmp/llm-serving-public-release-dryrun-20260824`