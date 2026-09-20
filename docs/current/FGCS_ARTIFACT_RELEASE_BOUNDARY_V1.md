# FGCS Artifact Release Boundary V1

## Compact artifacts intended for repository or release

- Source code and tests.
- Experiment protocols, manifests, hashes, and compact result CSV/JSON files.
- Figure-generation code and current figures.
- Manuscript source, bibliography, build manifest, and compiled PDF.

## Local or upstream-controlled data

Raw Azure traces, raw BurstGPT copies, and other upstream datasets are not
redistributed by default. Their local acquisition and license notes are
documented in `docs/DATA_RELEASE_POLICY.md` and
`docs/PUBLIC_RELEASE_MANIFEST.md`.

## Large local-only outputs

Family-A decision logs, model checkpoints, fresh raw-run directories, and Wulver
scratch outputs remain local or are candidates for an external archive. They
are not copied into Git by this cleanup.

Derived artifacts with uncertain upstream licensing require an explicit release
decision before publication.
