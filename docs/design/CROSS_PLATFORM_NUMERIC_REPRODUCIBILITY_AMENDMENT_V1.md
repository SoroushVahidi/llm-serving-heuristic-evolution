# CROSS_PLATFORM_NUMERIC_REPRODUCIBILITY_AMENDMENT_V1

## Trigger

The preregistered Wulver profile for `JOINT240_SBS_TARGETED_TERMINAL_LABEL_FULL_V1` completed successfully on Slurm job `1298651` before the full campaign was launched. State/action identities, canonical action hashes, policy maps, and row keys matched the original local pilot shard artifacts exactly, but some floating-valued features and terminal metrics differed at machine precision.

Observed maximum absolute numeric difference in the profile:

`7.105427357601002e-15`

Largest observed differing field:

`state__priority_sum_admissible`, local `27.554136526567152`, Wulver `27.554136526567145`.

Terminal metric differences were smaller, for example `q_sbs_anwg` differed by at most `2.7755575615628914e-16`.

This amendment is made before the full campaign and before any model training.

## Environment Evidence

Local environment:

- host: `al-khwarizmi`
- architecture: `x86_64`
- Python: `3.12.3`
- NumPy: `2.3.5`
- pandas: `3.0.2`
- SciPy: `1.17.1`
- scikit-learn: `1.8.0`
- glibc: `2.39`
- OpenBLAS: `0.3.30`, Haswell backend

Wulver profile environment:

- profile compute host: `n0006`
- login/profile architecture: `x86_64`
- Python: `3.10.20`
- NumPy: `2.2.6`
- pandas: `2.3.3`
- SciPy: `1.15.3`
- scikit-learn: `1.7.2`
- glibc: `2.34`
- OpenBLAS: `0.3.29`, SkylakeX backend

The difference pattern is consistent with cross-platform floating-point reduction and serialization effects, not with different simulator states or action semantics.

## Exact Invariants

The following must match exactly for cross-platform reproduction:

- state IDs;
- scenario IDs;
- folds;
- steps;
- policy IDs;
- canonical action IDs and hashes;
- canonical action contents;
- policy-to-action mappings;
- branch counts;
- row keys;
- discrete and categorical fields;
- boolean fields;
- integer-valued simulator state fields;
- missingness pattern;
- advantage sign for any `|A_SBS| > 1e-12`.

Any violation of these exact invariants is a hard reproduction failure.

## Numeric Equivalence Rule

Floating-valued deterministic simulator, utility, and feature fields are cross-platform equivalent if:

`abs_diff <= 1e-12 OR relative_diff <= 1e-12`

For advantages near zero, absolute tolerance governs. Zero/nonzero classification for cross-platform reproduction checks uses `1e-12` as the numerical zero threshold.

This tolerance is only a reproducibility-equivalence rule. It does not redefine the scientific target. Stored `Q_SBS` and `A_SBS` values remain the computed floating values from the run that generated the dataset.

## Validation Gate

`CROSS_PLATFORM_REPRODUCTION_PASS` requires:

- exact equality for all exact invariants above;
- zero numeric fields exceeding the frozen `1e-12` tolerance;
- no sign disagreement for `|A_SBS| > 1e-12`;
- no missingness mismatch.

Otherwise the verdict is `CROSS_PLATFORM_REPRODUCTION_FAIL`, and the full campaign must not launch.
