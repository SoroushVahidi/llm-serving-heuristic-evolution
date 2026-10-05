# release/

Reproducibility archives of the current study. The files inside each archive
are hash-locked by that archive's own manifest. Do not edit them; corrections
belong in new, separately documented files outside the archive.

| Path | What it is |
|---|---|
| `performance_evaluation_v1_1_0.zip` | **Current archive**, v1.1.0, published on Zenodo as [10.5281/zenodo.22866983](https://doi.org/10.5281/zenodo.22866983). |
| `performance_evaluation_v1_1_0/` | The same archive unpacked. `MANIFEST.json` and `SHA256SUMS.txt` list every file with its hash. |
| `performance_evaluation_v1.zip`, `performance_evaluation_v1/` | Superseded v1.0.0 archive ([10.5281/zenodo.22865294](https://doi.org/10.5281/zenodo.22865294)), kept for provenance. |

Verify v1.1.0 from the archive root:

```bash
cd release/performance_evaluation_v1_1_0     # or an unpacked copy of the ZIP
python3 verify_release.py
```

On 2026-10-04 the verifier passed 7/7 both on the tracked directory and on an
unpacked copy of the committed ZIP. The environment matched the versions
pinned in `ENVIRONMENT.json` (numpy 2.3.5, pandas 3.0.2). Other library
versions were not tested here.

The post hoc reference-reserve sensitivity analysis came after v1.1.0. It is
only in this repository, under
[`../experiments/reference_reserve_sensitivity_v1/`](../experiments/reference_reserve_sensitivity_v1/).
