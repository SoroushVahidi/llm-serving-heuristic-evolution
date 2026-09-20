# FGCS Local Artifact Retention Index V1

This index records large local artifacts that must not be removed by ordinary
repository cleanup.

| Path | Approx. size | Purpose | Canonical summary? | Disposition |
|---|---:|---|---|---|
| `experiments/family_a_pi0_closed_loop_final_v1/decision_rows.csv` | 1.79 GB | Family-A closed-loop decision output | Historical analysis/docs | NEEDS_QUERY_3_DECISION |
| `experiments/family_a_receding_horizon_oracle_v1/family_a_receding_horizon_oracle_v1_decision_logs.csv` | 867 MB | Receding-horizon oracle logs | Historical analysis/docs | NEEDS_QUERY_3_DECISION |
| `experiments/family_a_wulver_medium_sweep_v1/merged/candidate_states.jsonl` | 316 MB | Family-A Wulver sweep states | Historical analysis/docs | EXTERNAL_ARCHIVE_CANDIDATE |
| `results/pars_official/predictor_train/alpaca_gpt4_bert/{best,last}_model.pt` | 438 MB each | Historical selector checkpoints | Historical selector results | KEEP_LOCAL_PROVENANCE |
| `experiments/decision_criticality_timescale_trainval_v1/disagreement_and_divergence_events.csv` | 150 MB | Decision-criticality analysis | Historical analysis | KEEP_LOCAL_PROVENANCE |
| `experiments/sbs_override_fresh_ood_support_expansion_v1/support_scan/` | 100+ MB shards | Fresh/OOD support exploration | Partially summarized | NEEDS_QUERY_3_DECISION |
| `worktrees/fresh-latency-confirmatory-v1/` raw staging | local worktree-sized | Fresh causal execution provenance | Canonical compact result exists | KEEP_LOCAL_PROVENANCE |
| `worktrees/fresh-latency-execution-v1/` raw staging | local worktree-sized | Fresh causal execution provenance | Canonical compact result exists | KEEP_LOCAL_PROVENANCE |

No listed artifact is copied into Git or deleted by Query 2.

## Query-3 archival convention

The Query-3 local archive is outside the repository at a machine-local path
named `llm-serving-heuristic-evolution-local-provenance/fgcs-finalization-20260920/`.
Its `LOCAL_PROVENANCE_ARCHIVE_MANIFEST_V1.json`, file inventory, and SHA-256
lists are the authoritative records for the archived copies. Exact local paths
are intentionally kept outside Git.
