# Joint-240 SBS Disagreement and One-Step Counterfactual Action Outcomes

**Version:** 1.0.0  
**Author:** Soroush Vahidi (New Jersey Institute of Technology)  
**License (data):** CC BY 4.0  
**Resource type:** Historical exploratory synthetic-simulator research dataset

## 1. Short abstract

This dataset releases a compact, normalized subset of a 240-scenario synthetic
simulator study of when a fixed single-best scheduler (SBS) proposes a
different action from other policies, and what happens when one alternative
action is forced for one decision before returning to SBS. It contains
**8,886 retained labeled state-context records** and **21,854 retained
alternative-action outcomes** from **236 of the original 240 independently
seeded scenarios**. Effects are preserved in all three historical classes:
**2,573 positive, 3,624 negative, and 15,657 null** under the original
`1e-12` ANWG sign convention.

This is **not** the active Performance Evaluation manuscript’s
production-derived confirmatory evidence. It is historical exploratory
simulator evidence used to motivate a fixed SBS reference.

## 2. Scientific motivation

Adaptive scheduling can help only when an executable alternative exists and
has favorable downstream consequences. The Joint-240 campaign isolates
disagreement states on an SBS trajectory and measures one-step counterfactual
terminal utilities under fixed SBS continuation. The goal is to characterize
conditional action opportunity and its mixed sign structure, including harmful
and null effects.

## 3. Research questions

1. How often do six scheduling policies disagree at an SBS predecision state?
2. What is the terminal simulator effect of forcing each distinct alternative
   full canonical action once and then continuing with SBS?
3. Which predecision state features and action-difference contexts accompany
   those conditional opportunities?

## 4. Overview of the original Joint-240 experiment

The campaign generated **240 independently seeded synthetic scenarios**. A
scan ran fixed `kv_constrained_online` (SBS) and, at each SBS predecision
state, queried five other P6 policies on copied observable state while
executing only SBS. That scan produced **453,016** unique SBS-trajectory
predecision states. A later targeted-label campaign took all **8,888**
disagreement states, forced each unique full canonical action once, then
continued under fixed SBS. Learnability aggregates from a later offline
analysis are already public and are **not** redistributed here.

## 5. How the 240 scenarios were generated

Scenarios come from the project’s joint multi-mechanism synthetic generator
(`joint_multimechanism_generalization_v1` scenario manifest). Each scenario
combines synthetic arrivals, request lengths, priorities, deadlines, service
heterogeneity, and resource-pressure mechanisms under a frozen seed. Generator
parameters for the 236 represented scenarios are in
`metadata/scenario_metadata.csv`. No prompts and no raw third-party trace rows
are included.

## 6. Meaning of SBS and the six scheduling policies

**SBS** means the fixed scan-trajectory reference policy
`kv_constrained_online`. The six P6 policies are:

| Policy ID | Role |
|---|---|
| `kv_constrained_online` | SBS reference / scan trajectory |
| `full_prefill` | Candidate |
| `chunked_prefill_small` | Candidate |
| `estimated_service_time_first` | Candidate |
| `weighted_fair_share` | Candidate |
| `least_laxity_first` | Candidate |

See `metadata/policy_metadata.csv`.

## 7. Definition of scheduler/action disagreement

At an SBS predecision state, the scan records the full canonical action of SBS
and of each candidate policy on the same observable state. **Disagreement**
means at least one non-SBS policy proposes a different **full canonical
action** from SBS (`any_non_sbs_policy_differs`).

## 8. How disagreement states were selected

The terminal campaign selected **all** SBS scan states with disagreement: the
full disagreement manifest of **8,888** states spanning **236** scenarios and
all five frozen folds. Equal-action states (444,128) were not terminal-labeled
because, under this deterministic one-action-then-SBS estimand, they have zero
override advantage by construction.

## 9. How the 8,888 labeled states were obtained

For each disagreement state, every unique full canonical action among the six
policies was forced once as a one-step intervention; the remainder of the
episode continued with fixed SBS under the same future arrivals, configuration,
and seed. Terminal utilities were recorded as `Q_SBS(s,a)`.

## 10. Why only 8,886 labeled states appear in this release

Ninety-eight historical `(scenario_id, step)` keys have the same 95 numeric
physical features as earlier Alive-acquired states, but hidden simulator-state
identity cannot be resolved from the preserved archive. This release
**excludes all 98 keys**, including **2 labeled states** and **4 linked
terminal rows**. Therefore **8,886** labeled states and **21,854** alternative
outcomes remain. Already-public SBS reference utilities, public learnability
aggregates, and public-equivalent alternatives are also excluded.

## 11. Counterfactual intervention and continuation semantics

For a retained alternative action `a` at state `s`:

1. Force `a` for that one decision.
2. Continue the episode with fixed SBS thereafter.
3. Record terminal metrics as `q_sbs_*`.
4. Compare to the same-state SBS reference terminal metrics `q0_sbs_*`.

There is no Alive continuation, no candidate-policy forever switch, no H-step
hold, and no claim of a deployed closed-loop controller.

## 12. Definition of all terminal outcomes

Primary and companion terminal metrics:

* `anwg` — arrival-normalized weighted goodput (primary)
* `soft` — soft goodput
* `wcg` — weighted completion goodput / completion metric
* `wmt` — weighted mean time
* `wnt` — weighted normalized time

Also retained: `num_completed`, `num_dropped`, `sim_duration`, `extra_steps`,
and `live_fingerprint_unchanged` (continuation integrity flag; always true in
this release).

## 13. Meaning of positive, negative, and null outcomes

For ANWG/soft/WCG:

```text
a_sbs_* = q_sbs_* - q0_sbs_*
```

For time metrics stored as improvements:

```text
a_sbs_wmt_improvement = q0_sbs_wmt - q_sbs_wmt
a_sbs_wnt_improvement = q0_sbs_wnt - q_sbs_wnt
```

Positive ANWG advantage means higher terminal ANWG than SBS at that state;
negative means worse; null means absolute ANWG difference at or below the
original `1e-12` classification threshold. Harmful and null effects are first-class
scientific content.

## 14. Important numerical results

| Quantity | Value |
|---|---:|
| Original independently seeded scenarios | 240 |
| Scenarios represented in this release | 236 |
| Scenarios with no retained SBS disagreement | 4 |
| Original scan predecision states | 453,016 |
| Original labeled disagreement states | 8,888 |
| Retained labeled states | 8,886 |
| Retained alternative-action outcomes | 21,854 |
| Positive / negative / null ANWG effects | 2,573 / 3,624 / 15,657 |
| Historically ambiguous keys excluded | 98 |
| Offline selector held-out mean ANWG gain (historical) | 0.0001959413 |
| Scenario-bootstrap 95% CI (historical) | [-0.000207663, 0.000585958] (includes zero) |

Independent experimental units are **scenarios**, not rows. States and action
forks within a scenario are correlated.

## 15. Limitations and selection bias

Labels were acquired only where policies disagreed with SBS. Outcome
frequencies must not be extrapolated to arbitrary SBS states. The setting is a
deterministic synthetic simulator. No production-calibrated latency claim and
no confirmed closed-loop selector benefit follow from these tables. See
`LIMITATIONS.md`.

## 16. Relationship to earlier published experiments

Already-public scenario-level SBS utilities, public learnability aggregates,
and earlier public-equivalent terminal rows are excluded rather than
reissued. The bounded overlap audit is summarized in
`PUBLIC_RECORD_OVERLAP.md`. Dense-study legacy records and fitted predictions
are out of scope.

## 17. Relationship to the active Performance Evaluation manuscript

The active manuscript is *When Does LLM-Serving Scheduler Adaptation Matter?
Action Opportunity and Causal Headroom in Production-Derived Replay*
(Performance Evaluation, manuscript **PEVA-D-26-00474**; reported under review
on 2026-10-07; preprint https://ssrn.com/abstract=7584825). That manuscript
uses the earlier Joint-240 study to motivate a fixed SBS reference, but its
primary confirmatory outcomes are based on separate production-derived replay.
**Do not attribute the manuscript’s confirmatory claims to this dataset.**
This dataset must not be described as the manuscript’s production-derived
confirmatory evidence. The manuscript files themselves are not modified by
this release.

Performance Evaluation’s guide for authors applies research-data Option C
(deposit research data in a relevant repository). This independent historical
dataset deposit is compatible with that policy when kept clearly separate from
the confirmatory archive.

## 18. Recommended research uses

* Reproducing the conditional one-step SBS-relative action-outcome estimand
* Scenario-clustered exploratory analyses and negative-result analysis
* Studying disagreement-conditioned opportunity filters
* Developing overlap-aware simulator methods and integrity checks
* Teaching/examples of counterfactual continuation semantics

## 19. Uses and conclusions not supported by the data

* Claiming production performance or production-calibrated latency
* Treating each row as an independent experimental replicate
* Claiming a confirmed deployed or closed-loop selector improvement
* Presenting these tables as the PE manuscript’s confirmatory evidence
* Assuming predefined ML train/test splits in this release (historical folds
  were excluded as already-public aggregates)

## 20. File inventory and instructions

| Path | Rows | Role |
|---|---:|---|
| `data/state_context.csv` | 8,886 | Retained labeled SBS disagreement contexts + 95 features |
| `data/alternative_action_outcomes.csv` | 21,854 | Retained alternative-action terminal outcomes |
| `metadata/scenario_metadata.csv` | 236 | Synthetic scenario generator parameters + fold |
| `metadata/policy_metadata.csv` | 6 | Policy roles / continuation notes |
| `SCHEMA.md` | — | Column dictionary |
| `PROVENANCE.md` | — | Lineage and exclusions |
| `LIMITATIONS.md` | — | Scientific limits |
| `USAGE.md` | — | Load / join / summarize examples |
| `PUBLIC_RECORD_OVERLAP.md` | — | Public-overlap declaration |
| `CITATION.md` / `CITATION.bib` | — | Citation text |
| `ATTRIBUTION.md` / `LICENSE_DATA_CC_BY_4.0.txt` | — | Data license |
| `CHANGELOG.md` | — | Version history |
| `validation/validate_public.py` | — | Public-only integrity checks |
| `PUBLIC_RELEASE_CHECKSUMS.sha256` | — | SHA-256 manifest (excludes itself) |

Frozen scientific SHA-256 digests:

```text
5152ec0a2e139ccad7b9edceab571d51ca5bd2a5b9f93c2275344d64b62bdb50  data/state_context.csv
af9e60fe749538cfbb1b0b7f1e999408ff3a5811c3f90ba5eb8b39c53b3a92fd  data/alternative_action_outcomes.csv
d3ccde80792a80f466c687cdef60eb94537181d7f99e7859fbec3f6f3b0c53e1  metadata/scenario_metadata.csv
9b46d5940a0d0ac898be114cc4be7a877172e01206f80488965baa7671300793  metadata/policy_metadata.csv
```

Validate after download:

```bash
python3 validation/validate_public.py
# or:
sha256sum -c PUBLIC_RELEASE_CHECKSUMS.sha256
```

Requirements: Python 3.9+ standard library for validation; optional
`pandas>=1.5` and `matplotlib>=3.5` for `USAGE.md` examples
(`requirements-optional.txt`).

## 21. Citation instructions

Prefer the Zenodo version DOI for exact reproduction. See `CITATION.md` and
`CITATION.bib`. Do not overwrite or confuse this dataset citation with the
repository-root `CITATION.cff`, which refers to the broader research software
project.

## 22. License and attribution

The four scientific CSV tables and the accompanying release documentation in
this directory are licensed under **CC BY 4.0** (see
`LICENSE_DATA_CC_BY_4.0.txt` and `ATTRIBUTION.md`). The broader
`llm-serving-heuristic-evolution` repository software remains under its
existing **MIT** license. Upstream BurstGPT and Azure 2023 traces are
**not redistributed** and are not relicensed by this release.

## 23. Reproducibility and provenance links

* Design (scan): repository path
  `docs/design/JOINT240_SBS_DISAGREEMENT_SCAN_V1.md` on branch
  `experiment/joint240-sbs-disagreement-scan-v1`
* Design (labels):
  `docs/design/JOINT240_SBS_TARGETED_TERMINAL_LABEL_FULL_V1.md` on branch
  `experiment/joint240-sbs-targeted-terminal-label-full-v1`
* Commit hashes and frozen input digests: `PROVENANCE.md`

## 24. Version history

* **1.0.0 (2026-10-15)** — Initial public release of the normalized historical
  Joint-240 SBS disagreement / one-step counterfactual supplement.

## 25. Contact

Soroush Vahidi, New Jersey Institute of Technology, via the public issues and
contact channels of:
https://github.com/SoroushVahidi/llm-serving-heuristic-evolution

## 26. Persistent release locations

* **GitHub (canonical source tree):**
  https://github.com/SoroushVahidi/llm-serving-heuristic-evolution/tree/main/datasets/joint240-sbs-action-outcomes/v1.0.0
* **Zenodo (version DOI):** https://doi.org/10.5281/zenodo.23398347
* **Zenodo (concept DOI):** https://doi.org/10.5281/zenodo.23398346
* **Hugging Face (identical mirror):**
  https://huggingface.co/datasets/SoroushVahidi/joint240-sbs-action-outcomes

The Zenodo **version DOI** identifies this exact 1.0.0 package. The **concept
DOI** resolves to the dataset family / latest version.
