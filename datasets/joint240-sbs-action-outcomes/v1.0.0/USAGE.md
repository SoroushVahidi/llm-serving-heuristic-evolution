# Usage

The release uses CSV and Python's standard library only. Download the GitHub
directory, Zenodo archive, or Hugging Face mirror after publication, then run
from the dataset root:

```bash
python3 validation/validate_public.py
```

The scientific tables can be loaded with pandas:

```python
import pandas as pd
states = pd.read_csv("data/state_context.csv")
outcomes = pd.read_csv("data/alternative_action_outcomes.csv")
scenarios = pd.read_csv("metadata/scenario_metadata.csv")
joined = states.merge(outcomes, on=["state_id", "scenario_id", "fold", "step", "sim_time"])
```

Summarize effects without treating rows as independent:

```python
outcomes["effect_class"] = outcomes["a_sbs_anwg"].abs().le(1e-12).map(
    {True: "null", False: "non-null"}
)
outcomes.loc[outcomes["a_sbs_anwg"] > 1e-12, "effect_class"] = "positive"
outcomes.loc[outcomes["a_sbs_anwg"] < -1e-12, "effect_class"] = "negative"
print(outcomes["effect_class"].value_counts())
scenario_summary = outcomes.groupby("scenario_id")["a_sbs_anwg"].agg(["count", "mean"])
```

Group by policy signatures when studying action generation:

```python
print(outcomes.groupby("policies_generating_action").size().sort_values(ascending=False))
```

For scenario-clustered analysis, first compute one statistic per scenario,
then summarize or bootstrap those 236 scenario statistics. Do not use the
21,854 outcome rows as independent replicates. There is no predefined
machine-learning train/test split in this release; the historical five-fold
learnability records were excluded.

An illustrative plot:

```python
import matplotlib.pyplot as plt
means = outcomes.groupby("scenario_id")["a_sbs_anwg"].mean().sort_values()
means.plot(kind="bar", figsize=(14, 4), color="steelblue")
plt.axhline(0, color="black", linewidth=0.8)
plt.ylabel("Mean alternative-minus-SBS ANWG")
plt.tight_layout()
plt.show()
```

The public validator checks row counts, keys, foreign keys, finite values,
`a = q - q0`, effect-sign counts, scenario coverage, schema headers, and
relative-path checksums. It does not reproduce the simulator.
