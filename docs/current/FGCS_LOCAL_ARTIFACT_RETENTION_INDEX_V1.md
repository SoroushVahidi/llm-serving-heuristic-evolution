# FGCS Local Artifact Retention Index V1

This index records large local artifacts that must not be removed by ordinary
repository cleanup.

| Path | Approx. size | Purpose | Canonical summary? | Disposition |
|---|---:|---|---|---|
| `experiments/family_a_pi0_closed_loop_final_v1/decision_rows.csv` | 1.79 GB | Family-A closed-loop decision output | Historical analysis/docs | LOCAL_RAW_ARTIFACTS_RETAIN |
| `experiments/family_a_receding_horizon_oracle_v1/family_a_receding_horizon_oracle_v1_decision_logs.csv` | 867 MB | Receding-horizon oracle logs | Historical analysis/docs | LOCAL_RAW_ARTIFACTS_RETAIN |
| `experiments/family_a_wulver_medium_sweep_v1/merged/candidate_states.jsonl` | 316 MB | Family-A Wulver sweep states | Historical analysis/docs | LOCAL_RAW_ARTIFACTS_RETAIN |
| `results/pars_official/predictor_train/alpaca_gpt4_bert/{best,last}_model.pt` | 438 MB each | Historical selector checkpoints | Historical selector results | HISTORICAL_PROVENANCE_RETAIN |
| `experiments/decision_criticality_timescale_trainval_v1/disagreement_and_divergence_events.csv` | 150 MB | Decision-criticality analysis | Historical analysis | HISTORICAL_PROVENANCE_RETAIN |
| `experiments/sbs_override_fresh_ood_support_expansion_v1/support_scan/` | 100+ MB shards | Fresh/OOD support exploration | Partially summarized | LOCAL_RAW_ARTIFACTS_RETAIN |
| `worktrees/fresh-latency-confirmatory-v1/` raw staging | local worktree-sized | Fresh causal execution provenance | Canonical compact result exists | LOCAL_RAW_ARTIFACTS_RETAIN |
| `worktrees/fresh-latency-execution-v1/` raw staging | local worktree-sized | Fresh causal execution provenance | Canonical compact result exists | LOCAL_RAW_ARTIFACTS_RETAIN |

No listed artifact is copied into Git or deleted by Query 2 or Query 3.

## Query-3 archival convention

The Query-3 local archive is outside the repository at a machine-local path
named `llm-serving-heuristic-evolution-local-provenance/fgcs-finalization-20260920/`.
Its `LOCAL_PROVENANCE_ARCHIVE_MANIFEST_V1.json`, file inventory, and SHA-256
lists are the authoritative records for the archived copies. Exact local paths
are intentionally kept outside Git.

## Query-3 final disposition

All formerly unresolved repository-finalization decisions in this index are
closed as local or historical retention decisions. They do not block the final
FGCS submission-format audit because the compact manuscript evidence is tracked
in Git and the large/raw artifacts are indexed, hashed, and retained outside
Git. Retained local provenance is not a release defect.
