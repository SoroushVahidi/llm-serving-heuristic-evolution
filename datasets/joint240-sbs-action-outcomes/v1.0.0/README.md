# Joint-240 SBS Disagreement and One-Step Counterfactual Action Outcomes

**Version 1.0.0 — historical exploratory synthetic simulator dataset**

This dataset contains a compact, normalized subset of a 240-scenario study of
when a fixed single-best scheduler (SBS) proposes an action different from
other policies, and what happens when one alternative action is forced for one
decision before returning to SBS. It contains **8,886 retained state records**
and **21,854 retained alternative-action outcomes** from **236 of the original
240 scenarios**.

The dataset is not the active Performance Evaluation manuscript's
production-derived confirmatory dataset. It is historical exploratory
simulator evidence. It is not evidence of a deployed scheduler improvement,
an independent experiment per row, or a validated closed-loop selector.

## Motivation and research questions

Adaptive scheduling can help only when an executable alternative exists and
has favorable downstream consequences. The Joint-240 study asks:

1. How often do six scheduling policies disagree at an SBS predecision state?
2. What is the terminal simulator effect of forcing each distinct alternative
   action once and then continuing with SBS?
3. Which state/context and action-difference features describe those
   conditional opportunities?

## Original experiment

The campaign generated **240 independently seeded synthetic scenarios** using
the project's joint workload generator. Each scenario combines synthetic
arrivals, request lengths, priorities, deadlines, service heterogeneity, and
resource-pressure mechanisms. No prompts or raw third-party trace rows are
included. The scan followed `kv_constrained_online` as SBS and recorded
predecision state features while querying five other policies on copied
observable state:

* `kv_constrained_online` (SBS reference);
* `full_prefill`;
* `chunked_prefill_small`;
* `estimated_service_time_first`;
* `weighted_fair_share`;
* `least_laxity_first`.

An action disagreement means that at least one of the five non-SBS policies
proposed a different **full canonical action** from SBS. The terminal campaign
selected all **8,888** SBS scan states with disagreement, forced each unique
full canonical action once, and continued the same future arrivals and
configuration under fixed SBS. Four scenarios had no retained SBS
disagreement; therefore this release covers 236 scenarios.

## Release filtering

The original scan contained 453,016 predecision states, of which 8,888 were
labeled. This release retains 8,886 labeled states and excludes the two
labeled members of **98 historically ambiguous state keys** whose 95 physical
features matched earlier Alive-acquired states but whose hidden simulator
identity cannot be resolved. The release also excludes the four linked
terminal rows, all 8,888 already-public SBS reference utilities, the 260
already-public learnability aggregates, and earlier-public equivalent records.

The original alternative table had 21,858 rows. After the ambiguity and
public-overlap exclusions, **21,854** alternative outcomes remain.

## Outcome semantics

For each retained state/action row:

* `q_sbs_*` is the terminal metric after forcing the candidate action and
  continuing with SBS.
* `q0_sbs_*` is the terminal metric for the same state under the SBS action.
* `a_sbs_* = q_sbs_* - q0_sbs_*`.

Positive ANWG advantage means the forced alternative has higher
arrival-normalized weighted goodput; negative means worse; null means absolute
ANWG difference at or below the original `1e-12` classification threshold.
The release preserves all three classes: **2,573 positive, 3,624 negative,
and 15,657 null** ANWG effects. ANWG is dimensionless; weighted mean and
normalized-time metrics retain their simulator units as documented in
`SCHEMA.md`.

## Scientific interpretation and limitations

The independent units are scenarios, not rows: 240 for the original scan and
236 for the conditional labeled population. States and action forks within a
scenario/episode are correlated. Labels were acquired conditional on
disagreement, so outcome frequencies must not be generalized to arbitrary SBS
states. The offline selector's held-out mean gain was `0.0001959413` ANWG
with scenario-bootstrap 95% interval `[-0.000207663, 0.000585958]`, which
includes zero. See `LIMITATIONS.md`.

Recommended uses include reproducing the conditional action-outcome
estimand, scenario-clustered exploratory analyses, negative-result analysis,
and development of overlap-aware simulator methods. Unsupported uses include
claiming production performance, training/evaluating a confirmed deployed
controller, treating rows as independent replications, or presenting these
historical results as the manuscript's production-derived confirmation.

## Files and integrity

| File | Rows | SHA-256 |
|---|---:|---|
| `data/state_context.csv` | 8,886 | `5152ec0a2e139ccad7b9edceab571d51ca5bd2a5b9f93c2275344d64b62bdb50` |
| `data/alternative_action_outcomes.csv` | 21,854 | `af9e60fe749538cfbb1b0b7f1e999408ff3a5811c3f90ba5eb8b39c53b3a92fd` |
| `metadata/scenario_metadata.csv` | 236 | `d3ccde80792a80f466c687cdef60eb94537181d7f99e7859fbec3f6f3b0c53e1` |
| `metadata/policy_metadata.csv` | 6 | `9b46d5940a0d0ac898be114cc4be7a877172e01206f80488965baa7671300793` |

Run the public-only validator:

```bash
python3 validation/validate_public.py
```

It uses only the release directory, Python 3.9+ standard library, and
`PUBLIC_RELEASE_CHECKSUMS.sha256`; it does not access private experiment
outputs. The four CSVs are the scientific data. The original 3.66 GB scan and
4.37 GB provenance archive remain separate backups and are not replaced by
this derivative.

## Citation, license, and provenance

Use `CITATION.md` and `CITATION.bib` for citation text. The proposed data
license is CC BY 4.0 for the author-owned synthetic dataset files only,
subject to author approval; see `LICENSE_DATA_CC_BY_4.0.txt` and
`ATTRIBUTION.md`. The broader repository software remains MIT-licensed.
BurstGPT and Azure 2023 are upstream references only and are not included.

The canonical source repository is
https://github.com/SoroushVahidi/llm-serving-heuristic-evolution. The intended
permanent dataset record is Zenodo version 1.0.0 at
https://doi.org/10.5281/zenodo.23398347. The intended identical mirror is
https://huggingface.co/datasets/SoroushVahidi/joint240-sbs-action-outcomes.
Until those records exist, do not treat placeholder URLs or DOI text as
published identifiers.

## Version history

* `1.0.0` — initial normalized release; 8,886 states, 21,854 alternative
  outcomes, 236 scenario rows, six policy rows; 98 ambiguous state keys and
  previously public material excluded.

Contact: Soroush Vahidi, New Jersey Institute of Technology, through the
public contact and issue channels of the canonical GitHub repository.
