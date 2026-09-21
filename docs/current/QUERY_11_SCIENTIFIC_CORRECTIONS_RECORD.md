# QUERY_11 — Scientific corrections from the peer-review audit (record)

Scope: corrections to `paper/performance_evaluation/main.tex` in response to
`QUERY_10_EXTERNAL_PEER_REVIEW_AUDIT.md`. No simulation, workload execution, counterfactual campaign, vLLM run, or new
baseline was executed; every new number is recomputed from completed artifacts by
`paper/performance_evaluation/scripts/reference_policy_numbers.py` and registered in `FINAL_CLAIM_MANIFEST.json` (60 claims).
Starting state: branch `release/peva-v1-20260920`, HEAD `77cdde98`, `main.tex` SHA-256 `8d1661c4…3e40e`, 30 pages.

## Corrections to the numbers in QUERY_10 that this query was asked to verify

| Statement in the query | Verified value | Note |
|---|---|---|
| KV-regime p90 relative reduction ≈ 19% | **16.1%** (overall p90 11.3%) | QUERY_10 used the mislabeled `mean_ref_latency` column of the frozen state file; corrected derivative used now |
| 15.4% of 720 states above 10% | **12.8% (92 states)**, all in the Azure-code KV-16,000 regime | same cause |
| 99.5–100% of decision states are no-choice iterations | true, but it is the complement of P(D) by definition | not used as evidence of light load |
| ≤ 0.44% peak KV utilization; mean queue ≤ 0.09 | fresh windows: 0.44%, workload mean queue 0.090; **original windows: 0.70%, 0.081** | the manuscript's arrival result (Sec. 5) comes from the original windows |
| 512/720, 94.8%; 414/524; 102/720 | confirmed | 511 of the 512 are in the KV regimes; in 445 the five policies agree on one alternative |

## Preregistered secondary metrics

| Metric (METRIC_PROTOCOL_V1) | Preregistered? | Reported before | Existing result | Action |
|---|---|---|---|---|
| P(B_LAT \| D) | yes (primary state metric) | yes | 590/720 | none |
| Relative oracle headroom H_REL | yes (secondary) | no | median 0.21%, p90 11.3%; 92 states > 10% | reported (Table 4, Sec. 6.4, 9.1) |
| State structure: all harmful / all zero / mixed | yes | no | 102 / 26 (+2 zero-and-harmful) / 22 of 720 | reported (Table 4) |
| Action-level positive / negative / zero fractions | yes | no | 660 / 141 / 30 of 831 | reported (Table 4) |
| Action-level mean, median, quantiles, best gain, worst harm | yes | no | in action-level CSV | left in artifacts (no added interpretation) |
| Mean/median of positive H_LAT | yes | partly (median 0.29 ms among 589 positive states) | per-regime values in result JSON | left in artifacts |
| p95 latency effect | yes ("secondary only") | no | stored per branch/regime | not reported; disclosed in Methods 3.5 |
| ANWG continuity metric | yes | no | `NOT_CAPTURED_IN_EXECUTION_OUTPUT` | not available; disclosed in Methods 3.5 |

## Portfolio functional diversity (from existing artifacts and code)

- `full_prefill` and `chunked_prefill_small` use the same class (`GreedyArrivalPrefillControlPolicy`); they differ only in the execution chunk size.
- `class_id = "default"` and `priority = uniform 1.0`, so WFS's class-share term is constant; it differs from SBS in exactly the same 519 states as ESTF.
- Deadlines are arrival + 1000 s; the SBS urgency exemption (slack < 0.25 s) never applies (largest mean continuation latency 0.36 s).
- `predicted_output_tokens` equals the true length (`PHASE_A_REPLAY_SEMANTICS_V1.json`); the manuscript formerly said actual lengths were "hidden from policies".
- 610 of 720 disagreement states have exactly one distinct alternative action; mean 1.15.

## VBS comparison

An exact SBS-versus-VBS comparison on the paper's faithful-view windows cannot be supported: `experiments/public_trace_replay_v1` ran only
`full_prefill` and `chunked_prefill_small` whole-window in the faithful view; all six policies were run only under the controlled-annotation
(augmented) view, which is a different input. The manuscript claim was weakened to an illustration: on the same 60 native windows the chunked
variant is slower than the full variant in 60/60 windows (median +14.2%) although both issue the same action from any state.
(Side observation, not in the manuscript: in the augmented view the five non-chunked policies have identical whole-window latency in all 60 windows.)

## Bibliography

Verified and corrected: MorphServe (title, 7 authors), ServeGen (8 authors, pp. 1845–1859), Libra (pp. 1243–1258). Verified via Crossref:
Choudhury et al., Yildiz et al., Lilou, Ali et al. (title, authors, volume, article number; no abstracts, so title-level descriptions kept).
OpenTela's "operational traces" is supported by its paper (it releases a production trace) but Table 1 was rebuilt without an evaluation-setting column.
Added: Preble (ICLR 2025), Jaillet et al. (arXiv:2502.07115), Mitzenmacher and Shahout (Stochastic Systems 15(3):195–219, 2025), Vidur (MLSys 2024).
Not added after inspection: none of the four was dropped.

## Remaining stale copies (deliberately not regenerated)

`paper/performance_evaluation/submission/`, `release/peva_submission_final/`, `release/performance_evaluation_v1_1_0/` and the tracked
`paper/when_does_llm_serving_scheduler_adaptation_matter.pdf` still hold pre-QUERY_9 text. The highlights file lives in those copies.
