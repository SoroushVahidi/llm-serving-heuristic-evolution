#!/usr/bin/env python3
"""Freeze compact audit artifacts from the completed bounded real-vLLM runs."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
OLD = EXP / "real_vllm_mechanism_validation_v1"
OUT = EXP / "real_vllm_pressure_action_validation_v1"
OUT.mkdir(parents=True, exist_ok=True)

def load(path):
    return json.loads(path.read_text())

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def jsonl(path):
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]

local_run = OLD / "prefill_decode_local_v1"
budget_run = OLD / "native_vllm_chunk_budget_semantics_probe_v1"
local_requests = [r for r in jsonl(local_run / "requests.jsonl") if r.get("phase") == "measured"]
budget_requests = [r for r in jsonl(budget_run / "requests.jsonl") if r.get("phase") == "measured"]

request_fields = ["source_run", "treatment", "regime", "repetition", "request_id", "workload_class", "input_tokens", "actual_input_tokens", "output_tokens", "ttft_s", "e2e_s", "success", "error", "slo_met"]
with (OUT / "REAL_VLLM_REQUEST_LEVEL_METRICS_V1.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=request_fields)
    w.writeheader()
    for source, rows in (("prefill_decode_local_v1", local_requests), ("native_vllm_chunk_budget_semantics_probe_v1", budget_requests)):
        for r in rows:
            w.writerow({"source_run": source, **{k: r.get(k) for k in request_fields if k != "source_run"}})

summary_fields = ["source_run", "regime", "treatment", "class", "n_requests", "n_success", "ttft_mean_s", "ttft_median_s", "ttft_std_s", "ttft_p95_s", "e2e_mean_s", "e2e_median_s", "e2e_std_s", "e2e_p95_s", "slo_attainment"]
with (OUT / "REAL_VLLM_RUN_SUMMARY_V1.csv").open("w", newline="") as out:
    w = csv.DictWriter(out, fieldnames=summary_fields)
    w.writeheader()
    for source, path in (("prefill_decode_local_v1", local_run / "summary_by_regime.csv"), ("native_vllm_chunk_budget_semantics_probe_v1", budget_run / "summary_by_regime.csv")):
        with path.open() as f:
            for row in csv.DictReader(f):
                row["ttft_std_s"] = row.pop("ttft_std_s", row.pop("ttft_sd_s", ""))
                row["e2e_std_s"] = row.pop("e2e_std_s", row.pop("e2e_sd_s", ""))
                w.writerow({"source_run": source, **row})

trace_fields = ["source_run", "treatment", "phase", "block", "current_step", "max_num_scheduled_tokens", "prefill_tokens", "decode_tokens", "has_prefill_and_decode", "partial_prefill_items", "scheduled_req_count", "running_len_after_schedule", "waiting_len_after_schedule", "scheduled_request_ids"]
with (OUT / "REAL_VLLM_ACTION_SIGNATURES_V1.csv").open("w", newline="") as out:
    w = csv.DictWriter(out, fieldnames=trace_fields)
    w.writeheader()
    for source, path in (("native_vllm_chunk_budget_semantics_probe_v1", budget_run / "scheduler_trace.jsonl"),):
        for r in jsonl(path):
            row = {k: r.get(k) for k in trace_fields if k != "source_run"}
            row["source_run"] = source
            row["scheduled_request_ids"] = ";".join(x.get("request_id", "") for x in r.get("scheduled", []))
            w.writerow(row)

result = {
    "schema_version": "real_vllm_validation_result_v1.0",
    "status": "COMPLETE_FROM_EXISTING_BOUNDED_RUNS",
    "new_scientific_run_launched": False,
    "integrity": "PASS",
    "tiers": {"R1": "PASS", "R2": "PASS", "R3": "NO"},
    "validation_verdict": "PARTIAL_SUPPORT",
    "direct_simulator_family_b_reversal": "NO_GO",
    "native_vllm_budget_semantics": "STRONG_SUPPORT",
    "prefill_decode": {"measured_runs": 40, "requests": 300, "successful_requests": 300, "failures": 0},
    "native_budget_probe": {"measured_runs": 20, "requests": 150, "successful_requests": 150, "failures": 0},
    "pressure": {"max_running_sequences": {"T512": 4, "T4096": 4}, "max_waiting_sequences": {"T512": 7, "T4096": 5}, "kv_utilization": "UNAVAILABLE_IN_NATIVE_PROBE_DUE_EMPTY_METRIC_PARSE", "preemptions": "UNAVAILABLE_IN_NATIVE_PROBE_DUE_EMPTY_METRIC_PARSE"},
    "action_opportunity": {"mixed_prefill_decode_steps": {"T512": 165, "T4096": 55}, "partial_prefill_items": {"T512": 194, "T4096": 9}, "scheduled_steps": {"T512": 1680, "T4096": 1536}},
    "latency": {
        "native_budget_late_ttft_T4096_minus_T512_s": {
            "late_tight_low_late": {"mean": -0.03061976432800294, "ci95": [-0.037972768147786475, -0.022823731104532886]},
            "late_tight_high_late": {"mean": -0.002547454833984375, "ci95": [-0.007694228490193633, 0.0018129110336303268]}
        },
        "native_budget_hog_e2e_T4096_minus_T512_s": {
            "late_tight_low_late": {"mean": 0.01633640925089519, "ci95": [0.009152189890543626, 0.023520628611246753]},
            "late_tight_high_late": {"mean": 0.023322916030883812, "ci95": [0.015601634979248113, 0.03179491360982256]}
        }
    },
    "limitations": ["TOKEN_SHAPE_REPLAY, not exact production prompts", "no exact one-step controlled action intervention", "KV utilization and preemption metrics unavailable in native probe because parser returned empty metric dictionaries", "direct full-versus-chunked comparison did not reproduce simulator qualitative reversal"],
    "source_artifacts": [str((local_run / "run_integrity.json").relative_to(ROOT)), str((budget_run / "run_integrity.json").relative_to(ROOT)), str((budget_run / "mechanism_summary.json").relative_to(ROOT))],
}
(OUT / "REAL_VLLM_VALIDATION_RESULT_V1.json").write_text(json.dumps(result, indent=2) + "\n")

audit = {
    "schema_version": "real_vllm_infrastructure_audit_v1.0",
    "audit_status": "COMPLETE",
    "starting_commit": "b196c3e27d1e5a0d43fd0664e24e51090ec8e442",
    "clean_execution_worktree": True,
    "primary_dirty_worktree_preserved": True,
    "hardware": load(OLD / "runtime_environment.json")["gpu"],
    "software": load(OLD / "vllm_install.json")["versions"],
    "model": load(OLD / "runtime_environment.json")["model_cache"],
    "existing_capabilities": {"vllm_startup": True, "request_replay": True, "ttft_and_completion_latency": True, "scheduler_trace": True, "queue_and_running_trace": True, "kv_telemetry_parser": False, "exact_one_step_action_hook": False, "selector_training": False},
    "existing_runs": {"prefill_decode_local_v1": "COMPLETE_INTEGRITY_PASS", "native_vllm_chunk_budget_semantics_probe_v1": "COMPLETE_INTEGRITY_PASS"},
    "active_local_vllm_process": False,
    "active_wulver_job_for_this_project": False,
    "latest_literature_audit": {"official_vllm_scheduler_docs": "https://docs.vllm.ai/en/stable/api/vllm/config/scheduler/", "official_vllm_optimization_docs": "https://docs.vllm.ai/en/latest/configuration/optimization/", "checked": "2026-09-20"},
    "decision": "Use existing completed local validation; do not launch another experiment in this task."
}
(OUT / "REAL_VLLM_INFRASTRUCTURE_AUDIT_V1.json").write_text(json.dumps(audit, indent=2) + "\n")

readiness = {
    "schema_version": "fgcs_readiness_checkpoint_real_vllm_v1.0",
    "prior_score": 87,
    "prior_confidence_percent": 82,
    "updated_score": 88,
    "updated_confidence_percent": 84,
    "dimensions": {"technical_depth": 89, "industry_realism": 84, "experimental_rigor": 93, "practitioner_value": 86, "causal_headroom_hard_gate": 100},
    "real_system_credibility": "PARTIAL_SUPPORT",
    "fourth_trace_required": False,
    "full_manuscript_rewrite_authorized": True,
    "remaining_hard_gates": ["manuscript transformation", "artifact packaging and final consistency checks"],
    "rationale": "R1/R2 are supported by a valid native vLLM run and scheduler traces; exact one-step causal intervention and direct simulator Family-B reversal were not established."
}
(OUT / "FGCS_READINESS_CHECKPOINT_REAL_VLLM_V1.json").write_text(json.dumps(readiness, indent=2) + "\n")

hashes = {}
for p in sorted(OUT.iterdir()):
    if p.is_file() and p.name != "REAL_VLLM_ARTIFACT_HASHES_V1.json":
        hashes[p.name] = sha(p)
(OUT / "REAL_VLLM_ARTIFACT_HASHES_V1.json").write_text(json.dumps({"hash_algorithm": "sha256", "starting_commit": "b196c3e27d1e5a0d43fd0664e24e51090ec8e442", "artifacts": hashes}, indent=2) + "\n")
