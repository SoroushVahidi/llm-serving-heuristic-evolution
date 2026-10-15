# Usage

All examples assume the dataset root
`datasets/joint240-sbs-action-outcomes/v1.0.0/` (GitHub) or the unpacked Zenodo
directory / Hugging Face dataset snapshot root.

Validation uses only the Python 3.9+ standard library. Optional analysis
examples use `pandas` and `matplotlib` (see `requirements-optional.txt`).

## Download

### GitHub

```bash
git clone https://github.com/SoroushVahidi/llm-serving-heuristic-evolution.git
cd llm-serving-heuristic-evolution/datasets/joint240-sbs-action-outcomes/v1.0.0
```

Or download the directory from:
https://github.com/SoroushVahidi/llm-serving-heuristic-evolution/tree/main/datasets/joint240-sbs-action-outcomes/v1.0.0

### Zenodo

Download the version record https://doi.org/10.5281/zenodo.23398347 and unpack
`joint240-sbs-action-outcomes-v1.0.0.zip`, then enter
`datasets/joint240-sbs-action-outcomes/v1.0.0/`.

### Hugging Face

```bash
huggingface-cli download SoroushVahidi/joint240-sbs-action-outcomes --repo-type dataset --local-dir joint240-sbs-action-outcomes
cd joint240-sbs-action-outcomes
```

## Integrity checks

```bash
python3 validation/validate_public.py
sha256sum -c PUBLIC_RELEASE_CHECKSUMS.sha256
```

Expected validator effect-sign totals: positive=2573, negative=3624, null=15657.

## Load tables (pandas)

```python
import pandas as pd

states = pd.read_csv("data/state_context.csv")
outcomes = pd.read_csv("data/alternative_action_outcomes.csv")
scenarios = pd.read_csv("metadata/scenario_metadata.csv")
policies = pd.read_csv("metadata/policy_metadata.csv")

assert len(states) == 8886
assert len(outcomes) == 21854
assert len(scenarios) == 236
assert len(policies) == 6
```

## Schema and key integrity

```python
assert states["state_id"].is_unique
assert outcomes.duplicated(["state_id", "canonical_action_id"]).sum() == 0
assert set(outcomes["state_id"]).issubset(set(states["state_id"]))
assert set(states["scenario_id"]) == set(scenarios["scenario_id"])
assert (outcomes["is_sbs_action"] == 0).all()
```

## Join states with action outcomes

```python
joined = states.merge(
    outcomes,
    on=["state_id", "scenario_id", "fold", "step", "sim_time"],
    how="inner",
    validate="one_to_many",
)
print(joined.shape)  # one row per retained alternative at a retained state
```

## Reproduce effect-sign counts

```python
eps = 1e-12
pos = (outcomes["a_sbs_anwg"] > eps).sum()
neg = (outcomes["a_sbs_anwg"] < -eps).sum()
nul = (outcomes["a_sbs_anwg"].abs() <= eps).sum()
assert (pos, neg, nul) == (2573, 3624, 15657)
print({"positive": int(pos), "negative": int(neg), "null": int(nul)})
```

## Summarize advantages without treating rows as independent

```python
# First reduce to one statistic per scenario, then summarize.
scenario_mean = outcomes.groupby("scenario_id")["a_sbs_anwg"].mean()
print(scenario_mean.describe())
```

## Group by scenario and policy signature

```python
by_scenario = outcomes.groupby("scenario_id").size().sort_values(ascending=False)
by_policy = outcomes.groupby("policies_generating_action").size().sort_values(ascending=False)
print(by_scenario.head())
print(by_policy.head())
```

## Scenario-clustered analysis pattern

```python
import numpy as np

# Example: scenario-level mean ANWG advantage, then percentile interval
# over scenarios (illustrative; not the historical 10k bootstrap).
stat = outcomes.groupby("scenario_id")["a_sbs_anwg"].mean().to_numpy()
print(np.quantile(stat, [0.025, 0.5, 0.975]))
```

Do **not** use the 21,854 outcome rows as independent replicates for
confidence intervals.

## Verify stored effect identities

```python
import numpy as np

assert np.allclose(
    outcomes["a_sbs_anwg"],
    outcomes["q_sbs_anwg"] - outcomes["q0_sbs_anwg"],
    atol=1e-10,
)
assert np.allclose(
    outcomes["a_sbs_wmt_improvement"],
    outcomes["q0_sbs_wmt"] - outcomes["q_sbs_wmt"],
    atol=1e-10,
)
```

## Illustrative visualization

```python
import matplotlib.pyplot as plt

means = outcomes.groupby("scenario_id")["a_sbs_anwg"].mean().sort_values()
ax = means.plot(kind="bar", figsize=(14, 4), color="steelblue")
ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel("Mean alternative-minus-SBS ANWG")
ax.set_xlabel("scenario_id (sorted)")
plt.tight_layout()
plt.show()
```

## Machine-learning splits

This release does **not** include predefined train/test splits. Historical
five-fold learnability aggregates were excluded because they are already public.
If you create new splits, split by `scenario_id`, never by row, and document
them separately.
