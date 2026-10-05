# Zenodo deposit helpers (historical)

Two one-off scripts used on 2026-09-20 to create and publish the **v1.0.0**
Zenodo record of the current study's reproducibility archive
([10.5281/zenodo.22865294](https://doi.org/10.5281/zenodo.22865294)). The
v1.1.0 record ([10.5281/zenodo.22866983](https://doi.org/10.5281/zenodo.22866983))
supersedes it. Nothing in the build, the tests, or the archives depends on these
scripts. They are kept for provenance.

| Script | What it did |
|---|---|
| `create_zenodo_draft.py` | Created a draft deposition from `release/performance_evaluation_v1/.zenodo.json` and printed its id, reserved DOI and upload bucket. |
| `upload_publish.py` | Uploaded `release/performance_evaluation_v1.zip` to a hard-coded bucket and published deposition `22865294`. |

Both scripts read `ZENODO_API_TOKEN` from the environment and use paths relative
to the repository root. **Do not run `upload_publish.py`**: it targets the
v1.0.0 deposition, which is already published. The scripts sat at the
repository root until 2026-10-04.
