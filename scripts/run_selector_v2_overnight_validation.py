#!/usr/bin/env python3
"""Overnight orchestrator: targeted decode/prefill-contention fixtures ->
mechanism validation -> gated specialization search -> (if gates pass)
Selector Dataset v2 targeted pilot -> prototype selector training/eval ->
external-baseline comparison plan -> testing.

Designed to run unattended for up to ~8.5 hours inside a detached tmux
session (see scripts/run_selector_v2_overnight_validation.sh). Every phase
is checkpointed to `experiments/selector_v2_overnight_<timestamp>/
phase_status.json`; re-running this script resumes from the last
completed phase rather than recomputing it. Any phase that raises stops
the pipeline cleanly (never lets a crash propagate to a bare traceback
with no final report) and writes final_summary.md explaining what
happened.

Reuses existing, already-vetted infrastructure wherever possible rather
than reimplementing it:
  - src/llmserveopt/selector/dataset_v2/{schema,discriminativeness,
    candidates,splits}.py for data model / gates / policy roster / splits.
  - src/llmserveopt/selector/dataset_v2/builder.py's
    `run_candidate_policy_on_window` for actually executing one policy on
    one window (kept per-policy so each candidate can be given its OWN
    ServiceModel -- required for the decode_first mechanism to mean
    anything; see docs/decode_prefill_contention_execution_model.md).
  - scripts/train_policy_selector.py / evaluate_policy_selector.py (CLI
    subprocess reuse) for Phase 6/7.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

DEADLINE_SECONDS_DEFAULT = 8 * 3600 + 15 * 60  # 8h15m soft deadline (task budget is 8h30m)
PROTECTED_PATHS = [
    "experiments/real_llm/vllm_healthcheck_20260703T171021Z/server.log",
    "experiments/gpu_external_validity/vllm_qwen05b_stress2_20260718T2212/server.log",
    "experiments/gpu_external_validity/vllm_qwen15b_stress_20260718T2158/server.log",
]


def _log(out_dir: Path, msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(out_dir / "logs" / "orchestrator.log", "a") as f:
        f.write(line + "\n")


def _load_status(out_dir: Path) -> Dict:
    p = out_dir / "phase_status.json"
    if p.exists():
        return json.loads(p.read_text())
    return {"phases": {}, "start_time": time.time(), "deadline_seconds": DEADLINE_SECONDS_DEFAULT}


def _save_status(out_dir: Path, status: Dict) -> None:
    (out_dir / "phase_status.json").write_text(json.dumps(status, indent=2, sort_keys=True))


def _deadline_exceeded(status: Dict) -> bool:
    elapsed = time.time() - status["start_time"]
    return elapsed >= status["deadline_seconds"]


def _run_git(args: List[str]) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


# ---------------------------------------------------------------------------
# Phase 0: pre-flight / repo protection
# ---------------------------------------------------------------------------

def phase0_preflight(out_dir: Path) -> Dict:
    branch = _run_git(["branch", "--show-current"])
    head = _run_git(["rev-parse", "HEAD"])
    status = _run_git(["status", "--short"])
    diff_check = subprocess.run(["git", "diff", "--check"], cwd=ROOT, capture_output=True, text=True)

    protected_hashes = {}
    import hashlib
    for rel in PROTECTED_PATHS:
        p = ROOT / rel
        if p.exists():
            protected_hashes[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
        else:
            protected_hashes[rel] = None

    result = {
        "branch": branch,
        "head": head,
        "git_status_short": status,
        "git_diff_check_returncode": diff_check.returncode,
        "protected_file_hashes_before": protected_hashes,
    }
    return result


# ---------------------------------------------------------------------------
# Phase 1: build targeted contention fixtures
# ---------------------------------------------------------------------------

def phase1_build_fixtures(out_dir: Path) -> Dict:
    from llmserveopt.selector.dataset_v2.contention_fixtures import all_fixtures

    fixtures_dir = ROOT / "experiments" / "runtime_validation_benchmark_pack" / "contention_targets"
    fixtures_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for fx in all_fixtures():
        payload = {
            "fixture_id": fx.fixture_id,
            "description": fx.description,
            "interaction_shape": fx.interaction_shape,
            "step_token_budget": fx.step_token_budget,
            "max_prefill_chunk_tokens": fx.max_prefill_chunk_tokens,
            "max_kv_tokens": fx.max_kv_tokens,
            "max_active_sequences": fx.max_active_sequences,
            "prefill_cost_per_token": fx.prefill_cost_per_token,
            "expected_to_diverge": fx.expected_to_diverge,
            "requests": [
                {
                    "request_id": r.request_id, "arrival_time": r.arrival_time,
                    "prompt_tokens": r.prompt_tokens,
                    "predicted_output_tokens": r.predicted_output_tokens,
                    "actual_output_tokens": r.actual_output_tokens,
                }
                for r in fx.requests
            ],
        }
        out_path = fixtures_dir / f"{fx.fixture_id}.json"
        out_path.write_text(json.dumps(payload, indent=2))
        written.append(str(out_path.relative_to(ROOT)))

    readme = fixtures_dir / "README.md"
    readme.write_text(
        "# Targeted decode/prefill-contention fixtures\n\n"
        "Generated by scripts/run_selector_v2_overnight_validation.py Phase 1.\n"
        "See src/llmserveopt/selector/dataset_v2/contention_fixtures.py for the\n"
        "full construction logic and the documented, empirically-derived finding\n"
        "about which interaction shapes actually produce divergence under this\n"
        "simulator's strict FCFS-by-arrival-time contention model.\n"
    )
    return {"fixtures_written": written, "fixtures_dir": str(fixtures_dir.relative_to(ROOT))}


# ---------------------------------------------------------------------------
# Phase 2: mechanism validation on targeted fixtures
# ---------------------------------------------------------------------------

def _service_model_for_policy(policy_name: str, budget: int, chunk: int) -> "ServiceModel":
    from llmserveopt.simulator.service_model import ServiceModel
    decode_first = policy_name != "vllm_chunked_prefill_faithful"
    return ServiceModel(
        enable_prefill_modeling=True, decode_first=decode_first,
        enable_decode_prefill_contention=True,
        step_token_budget=budget, max_prefill_chunk_tokens=chunk,
    )


def phase2_mechanism_validation(out_dir: Path) -> Dict:
    from llmserveopt.core.types import GPUConfig
    from llmserveopt.evaluation.run_policy import run_policy
    from llmserveopt.policies.sarathi_faithful import SarathiFaithfulPolicy
    from llmserveopt.policies.vllm_chunked_prefill_faithful import VLLMChunkedPrefillFaithfulPolicy
    from llmserveopt.policies.vllm_faithful import VLLMFaithfulPolicy
    from llmserveopt.selector.dataset_v2.contention_fixtures import all_fixtures

    ADMIT_CHUNK = 100_000  # decouple policy-level admission chunking from execution-level contention
    rows = []
    for fx in all_fixtures():
        gpu_configs = [GPUConfig(0, max_active_sequences=fx.max_active_sequences,
                                  max_batch_tokens=1_000_000, max_kv_tokens=fx.max_kv_tokens)]
        results = {}
        for pname, cls, kw in [
            ("sarathi_faithful", SarathiFaithfulPolicy, dict(chunk_size=ADMIT_CHUNK)),
            ("vllm_chunked_prefill_faithful", VLLMChunkedPrefillFaithfulPolicy,
             dict(max_num_batched_tokens=ADMIT_CHUNK)),
            ("vllm_faithful", VLLMFaithfulPolicy, dict(max_num_batched_tokens=ADMIT_CHUNK)),
        ]:
            sm = _service_model_for_policy(pname, fx.step_token_budget, fx.max_prefill_chunk_tokens)
            try:
                policy = cls(**kw)
            except TypeError:
                policy = cls()
            m = run_policy(policy=policy, requests=list(fx.requests), gpu_configs=gpu_configs,
                            service_model=sm, workload_tag=fx.fixture_id, seed=1, drain_steps=20_000)
            results[pname] = m

        sarathi_e2e = results["sarathi_faithful"].mean_latency
        vllm_ck_e2e = results["vllm_chunked_prefill_faithful"].mean_latency
        diverges = abs(sarathi_e2e - vllm_ck_e2e) > 1e-9
        if diverges:
            classification = "SARATHI_ADVANTAGE" if sarathi_e2e < vllm_ck_e2e else "VLLM_ADVANTAGE"
        else:
            classification = "NEAR_TIE"

        row = {
            "fixture_id": fx.fixture_id, "interaction_shape": fx.interaction_shape,
            "expected_to_diverge": fx.expected_to_diverge,
            "observed_diverges": diverges, "classification": classification,
        }
        for pname, m in results.items():
            row[f"{pname}_mean_latency"] = m.mean_latency
            row[f"{pname}_mean_ttft"] = m.mean_ttft
            row[f"{pname}_completion_fraction"] = m.completion_fraction
            row[f"{pname}_num_completed"] = m.num_completed
        rows.append(row)

    csv_path = out_dir / "contention_fixture_results.csv"
    if rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    n_diverging = sum(1 for r in rows if r["observed_diverges"])
    sarathi_wins = sum(1 for r in rows if r["classification"] == "SARATHI_ADVANTAGE")
    vllm_wins = sum(1 for r in rows if r["classification"] == "VLLM_ADVANTAGE")
    return {
        "rows": rows, "n_fixtures": len(rows), "n_diverging": n_diverging,
        "sarathi_advantage_count": sarathi_wins, "vllm_advantage_count": vllm_wins,
        "csv_path": str(csv_path.relative_to(ROOT)),
        "mechanism_question_answer": (
            "sarathi_faithful outperforms vllm_chunked_prefill_faithful on "
            f"{sarathi_wins}/{len(rows)} targeted fixtures; vllm_chunked_prefill_faithful "
            f"outperforms on {vllm_wins}/{len(rows)}; {len(rows) - n_diverging}/{len(rows)} tie."
        ),
    }


# ---------------------------------------------------------------------------
# Phase 3: bounded robustness search around contention boundaries
# ---------------------------------------------------------------------------

def phase3_specialization_search(out_dir: Path, n_candidates: int, search_seed: int) -> Dict:
    import random

    from llmserveopt.core.types import GPUConfig, Request
    from llmserveopt.evaluation.run_policy import run_policy
    from llmserveopt.policies.registry import make_policy
    from llmserveopt.policies.sarathi_faithful import SarathiFaithfulPolicy
    from llmserveopt.policies.vllm_chunked_prefill_faithful import VLLMChunkedPrefillFaithfulPolicy
    from llmserveopt.policies.vllm_faithful import VLLMFaithfulPolicy
    from llmserveopt.selector.dataset_v2.discriminativeness import (
        STANDARD_OBJECTIVES, PRIMARY_SELECTOR_OBJECTIVE, compute_discriminativeness,
    )
    from llmserveopt.selector.dataset_v2.schema import PolicyOutcomeVector
    from llmserveopt.selector.dataset_v2.builder import metrics_to_outcome_vector

    rng = random.Random(search_seed)
    ADMIT_CHUNK = 100_000
    FAITHFUL_POLICIES = ["vllm_faithful", "sarathi_faithful", "vllm_chunked_prefill_faithful"]
    CHEAP_HISTORICAL = ["fifo", "edf", "scorpio_style_slo_guard", "admission_control",
                         "weighted_shortest_processing", "estimated_service_time_first",
                         "best_fit", "multi_bin_batching"]

    def _make_policy(name: str):
        if name == "sarathi_faithful":
            return SarathiFaithfulPolicy(chunk_size=ADMIT_CHUNK)
        if name == "vllm_chunked_prefill_faithful":
            return VLLMChunkedPrefillFaithfulPolicy(max_num_batched_tokens=ADMIT_CHUNK)
        if name == "vllm_faithful":
            return VLLMFaithfulPolicy(max_num_batched_tokens=ADMIT_CHUNK)
        return make_policy(name)

    def _random_window(idx: int) -> Dict:
        hog_prompt = rng.choice([2000, 4000, 8000, 12000])
        n_runners = rng.randint(2, 8)
        runner_output = rng.choice([5, 10, 20, 40])
        budget = 512 + rng.choice([1, 2, 3, 5, 8])
        chunk = 512
        arrival_gap = rng.choice([0.001, 0.002, 0.005])
        reqs = [Request(request_id=0, arrival_time=0.0, prompt_tokens=hog_prompt,
                         predicted_output_tokens=1, actual_output_tokens=1,
                         slo_deadline=1000.0, priority=1.0, class_id="search")]
        for i in range(1, n_runners + 1):
            reqs.append(Request(
                request_id=i, arrival_time=arrival_gap, prompt_tokens=rng.randint(1, 40),
                predicted_output_tokens=runner_output, actual_output_tokens=runner_output,
                slo_deadline=1000.0, priority=1.0, class_id="search",
            ))
        return dict(requests=reqs, budget=budget, chunk=chunk, hog_prompt=hog_prompt,
                    n_runners=n_runners, runner_output=runner_output, arrival_gap=arrival_gap)

    all_window_outcomes: List[List[PolicyOutcomeVector]] = []
    window_meta: List[Dict] = []
    for i in range(n_candidates):
        w = _random_window(i)
        gpu_configs = [GPUConfig(0, max_active_sequences=64, max_batch_tokens=1_000_000, max_kv_tokens=200_000)]
        outcomes: List[PolicyOutcomeVector] = []
        for pname in FAITHFUL_POLICIES + CHEAP_HISTORICAL:
            sm = _service_model_for_policy(pname, w["budget"], w["chunk"])
            try:
                policy = _make_policy(pname)
            except Exception:
                continue
            try:
                m = run_policy(policy=policy, requests=list(w["requests"]), gpu_configs=gpu_configs,
                                service_model=sm, workload_tag=f"search_{i}", seed=search_seed + i,
                                drain_steps=5_000)
            except Exception:
                continue
            outcomes.append(metrics_to_outcome_vector(pname, m, {"admit": 0, "preempt": 0, "swap": 0, "migrate": 0},
                                                        gpu_count=1))
        if len(outcomes) < 2:
            continue
        all_window_outcomes.append(outcomes)
        window_meta.append(w)

    primary_obj = next(o for o in STANDARD_OBJECTIVES if o.name == PRIMARY_SELECTOR_OBJECTIVE)
    disc_rows = []
    win_counts: Dict[str, int] = {}
    strong_win_counts: Dict[str, int] = {}
    for outcomes in all_window_outcomes:
        disc = compute_discriminativeness(outcomes, primary_obj)
        if disc is None:
            continue
        disc_rows.append(asdict(disc))
        win_counts[disc.best_policy] = win_counts.get(disc.best_policy, 0) + 1
        if disc.classification == "STRONGLY_DISCRIMINATIVE":
            strong_win_counts[disc.best_policy] = strong_win_counts.get(disc.best_policy, 0) + 1

    n_windows = len(disc_rows)
    all_equiv = sum(1 for d in disc_rows if d["classification"] == "ALL_COMPLETE_OR_EFFECTIVELY_TIED")
    all_equiv_fraction = (all_equiv / n_windows) if n_windows else 1.0
    total_strong = sum(strong_win_counts.values())
    top_strong_share = (max(strong_win_counts.values()) / total_strong) if total_strong else 0.0

    values_by_policy: Dict[str, List[float]] = {}
    for outcomes in all_window_outcomes:
        for o in outcomes:
            v = primary_obj.extractor(o)
            if v is not None:
                values_by_policy.setdefault(o.policy_name, []).append(v)
    best_fixed_name, best_fixed_mean = None, None
    for name, vals in values_by_policy.items():
        m = sum(vals) / len(vals)
        if best_fixed_mean is None or m > best_fixed_mean:
            best_fixed_name, best_fixed_mean = name, m

    oracle_vals, best_fixed_vals = [], []
    for outcomes in all_window_outcomes:
        vals = {o.policy_name: primary_obj.extractor(o) for o in outcomes if primary_obj.extractor(o) is not None}
        if not vals:
            continue
        oracle_vals.append(max(vals.values()))
        if best_fixed_name in vals:
            best_fixed_vals.append(vals[best_fixed_name])
    oracle_headroom = (
        (sum(oracle_vals) / len(oracle_vals)) - (sum(best_fixed_vals) / len(best_fixed_vals))
        if oracle_vals and best_fixed_vals and len(oracle_vals) == len(best_fixed_vals) else None
    )
    disc_oracle_vals, disc_best_fixed_vals = [], []
    for outcomes, d in zip(all_window_outcomes, disc_rows):
        if d["classification"] not in ("STRONGLY_DISCRIMINATIVE", "MODERATELY_DISCRIMINATIVE"):
            continue
        vals = {o.policy_name: primary_obj.extractor(o) for o in outcomes if primary_obj.extractor(o) is not None}
        if not vals or best_fixed_name not in vals:
            continue
        disc_oracle_vals.append(max(vals.values()))
        disc_best_fixed_vals.append(vals[best_fixed_name])
    discriminative_oracle_headroom = (
        (sum(disc_oracle_vals) / len(disc_oracle_vals)) - (sum(disc_best_fixed_vals) / len(disc_best_fixed_vals))
        if disc_oracle_vals else None
    )

    csv_path = out_dir / "specialization_search.csv"
    with open(csv_path, "w", newline="") as f:
        fieldnames = ["window_idx"] + list(disc_rows[0].keys()) if disc_rows else ["window_idx"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i, d in enumerate(disc_rows):
            w.writerow({"window_idx": i, **d})

    gates = {
        "at_least_3_policies_win": len(win_counts) >= 3,
        "no_policy_dominates_strong_wins_gt_85pct": top_strong_share <= 0.85,
        "oracle_headroom_gte_0.01": (oracle_headroom or 0.0) >= 0.01,
        "discriminative_oracle_headroom_gte_0.03": (discriminative_oracle_headroom or 0.0) >= 0.03,
        "all_equivalent_fraction_lt_0.40": all_equiv_fraction < 0.40,
    }
    all_gates_pass = all(gates.values())

    return {
        "n_candidates_attempted": n_candidates, "n_windows_scored": n_windows,
        "win_distribution": win_counts, "strong_win_distribution": strong_win_counts,
        "all_equivalent_fraction": all_equiv_fraction, "top_policy_strong_win_share": top_strong_share,
        "best_fixed_policy": best_fixed_name, "oracle_headroom": oracle_headroom,
        "discriminative_oracle_headroom": discriminative_oracle_headroom,
        "gates": gates, "all_gates_pass": all_gates_pass,
        "csv_path": str(csv_path.relative_to(ROOT)),
    }


# ---------------------------------------------------------------------------
# Phase 4: targeted Selector Dataset v2 pilot (GATED on phase 3)
# ---------------------------------------------------------------------------

def phase4_dataset_pilot(out_dir: Path, target_windows: int) -> Dict:
    pilot_out = out_dir / "selector_dataset_v2_pilot"
    pilot_out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(ROOT / "scripts" / "build_selector_dataset_v2_redesigned_pilot.py"),
        "--output-dir", str(pilot_out.relative_to(ROOT)),
        "--topology-class", "monolithic",
        "--target-windows", str(target_windows),
        "--seeds", "20260720", "20260721",
        "--search-seed", "20260720",
    ]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=3 * 3600)
    (out_dir / "logs" / "phase4_build_pilot_stdout.log").write_text(proc.stdout)
    (out_dir / "logs" / "phase4_build_pilot_stderr.log").write_text(proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"build_selector_dataset_v2_redesigned_pilot.py failed (rc={proc.returncode}); "
                            f"see logs/phase4_build_pilot_stderr.log")
    manifest_path = pilot_out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    return {
        "pilot_output_dir": str(pilot_out.relative_to(ROOT)),
        "manifest": manifest,
        "note": (
            "Reuses the existing, already-vetted build_selector_dataset_v2_redesigned_pilot.py "
            "pipeline unmodified (BurstGPT/Azure/synthetic scenario sources, existing quality-"
            "gate computation). This pilot does NOT yet incorporate the new contention_fixtures.py "
            "family directly (that generator produces standalone Request lists, not "
            "ScenarioFamilySpec entries in scenario_redesign.py) -- see Phase 2's "
            "contention_fixture_results.csv for the targeted-fixture mechanism results instead. "
            "Folding contention_fixtures.py into scenario_redesign.py as a first-class family is "
            "flagged as follow-up work, not done here."
        ),
    }


# ---------------------------------------------------------------------------
# Phase 5: quality gates (reuses Phase 4's own manifest quality_gate_results)
# ---------------------------------------------------------------------------

def phase5_quality_gates(out_dir: Path, phase3_result: Dict, phase4_result: Optional[Dict]) -> Dict:
    gates = {"phase3_search_gates": phase3_result["gates"]}
    proceed_to_training = phase3_result["all_gates_pass"]
    if phase4_result is not None:
        gates["phase4_pilot_gates"] = phase4_result["manifest"].get("quality_gate_results", {})
        proceed_to_training = proceed_to_training and all(gates["phase4_pilot_gates"].values())
    gates["proceed_to_selector_training"] = proceed_to_training
    (out_dir / "quality_gates.json").write_text(json.dumps(gates, indent=2))
    return gates


# ---------------------------------------------------------------------------
# Phase 6/7: train + evaluate prototype selector (subprocess reuse)
# ---------------------------------------------------------------------------

def phase6_train_selector(out_dir: Path, dataset_csv: Path) -> Dict:
    models_out = out_dir / "selector_models"
    cmd = [
        sys.executable, str(ROOT / "scripts" / "train_policy_selector.py"),
        "--dataset", str(dataset_csv), "--output", str(models_out),
        "--model-types", "rule_based,decision_tree,random_forest",
    ]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=3600)
    (out_dir / "logs" / "phase6_train_stdout.log").write_text(proc.stdout)
    (out_dir / "logs" / "phase6_train_stderr.log").write_text(proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"train_policy_selector.py failed (rc={proc.returncode})")
    return {"models_dir": str(models_out.relative_to(ROOT))}


def phase7_evaluate_selector(out_dir: Path, models_dir: Path, dataset_csv: Path) -> Dict:
    eval_out = out_dir / "selector_evaluation"
    cmd = [
        sys.executable, str(ROOT / "scripts" / "evaluate_policy_selector.py"),
        "--models-dir", str(models_dir), "--test-dataset", str(dataset_csv), "--output", str(eval_out),
    ]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=3600)
    (out_dir / "logs" / "phase7_eval_stdout.log").write_text(proc.stdout)
    (out_dir / "logs" / "phase7_eval_stderr.log").write_text(proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"evaluate_policy_selector.py failed (rc={proc.returncode})")
    metrics_path = eval_out / "evaluation_summary.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    (out_dir / "selector_metrics.json").write_text(json.dumps(metrics, indent=2))
    return {"eval_dir": str(eval_out.relative_to(ROOT)), "metrics": metrics}


# ---------------------------------------------------------------------------
# Phase 8: external-baseline comparison plan (static doc, no execution)
# ---------------------------------------------------------------------------

def phase8_comparison_plan(out_dir: Path) -> Dict:
    plan_path = out_dir / "external_baseline_comparison_plan.md"
    plan_path.write_text(
        "# External-baseline comparison plan (Selector v2 contention validation pilot)\n\n"
        "Per architecture-native Protocol C (docs/external_baseline_integration.md), the\n"
        "trained selector prototype is compared ONLY against policies in its own valid\n"
        "topology class unless resource/topology normalization is explicitly applied.\n\n"
        "## Monolithic selector comparison (the selector's own class)\n"
        "- vllm_faithful\n- vllm_chunked_prefill_faithful\n- sarathi_faithful\n"
        "- historical monolithic baselines (fifo, edf, scorpio_style_slo_guard, "
        "admission_control, weighted_shortest_processing, estimated_service_time_first, "
        "best_fit, multi_bin_batching)\n- per-scenario oracle\n\n"
        "## Disaggregated reference comparison (reported separately, NOT as a flat baseline)\n"
        "- distserve_faithful\n- tetriinfer_paper_reimplementation\n\n"
        "## Migratory reference comparison (reported separately, NOT as a flat baseline)\n"
        "- llumnix_faithful\n\n"
        "These non-monolithic policies are structurally incompatible with the monolithic\n"
        "selector's own topology and are NOT run head-to-head against it as if they were\n"
        "peers; they are reported under their own architecture-native Protocol C configs\n"
        "for context only, per docs/external_baseline_integration.md.\n"
    )
    return {"plan_path": str(plan_path.relative_to(ROOT))}


# ---------------------------------------------------------------------------
# Phase 9: testing
# ---------------------------------------------------------------------------

def phase9_testing(out_dir: Path) -> Dict:
    results = {}
    test_targets = [
        "tests/test_decode_prefill_contention_execution.py",
        "tests/test_vllm_chunked_prefill_faithful_scheduler.py",
        "tests/test_sarathi_faithful_scheduler.py",
        "tests/test_selector_dataset_v2.py",
        "tests/test_selector_evaluation.py",
        "tests/test_selector_models.py",
        "tests/test_external_baseline_integration.py",
    ]
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *test_targets],
        cwd=ROOT, capture_output=True, text=True, timeout=1800,
    )
    (out_dir / "logs" / "phase9_targeted_tests.log").write_text(proc.stdout + proc.stderr)
    results["targeted_tests_returncode"] = proc.returncode
    results["targeted_tests_tail"] = proc.stdout[-3000:]

    proc_full = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-m", "not gpu"],
        cwd=ROOT, capture_output=True, text=True, timeout=3600,
    )
    (out_dir / "logs" / "phase9_full_suite.log").write_text(proc_full.stdout + proc_full.stderr)
    results["full_suite_returncode"] = proc_full.returncode
    results["full_suite_tail"] = proc_full.stdout[-3000:]

    diff_check = subprocess.run(["git", "diff", "--check"], cwd=ROOT, capture_output=True, text=True)
    results["git_diff_check_returncode"] = diff_check.returncode
    results["git_diff_check_output"] = diff_check.stdout

    return results


# ---------------------------------------------------------------------------
# Phase 12: git commits
# ---------------------------------------------------------------------------

def phase12_git_commit(out_dir: Path, message: str, paths: List[str]) -> Dict:
    existing = [p for p in paths if (ROOT / p).exists()]
    if not existing:
        return {"committed": False, "reason": "no paths existed"}
    subprocess.run(["git", "add", *existing], cwd=ROOT, check=True)
    diff_cached = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT)
    if diff_cached.returncode == 0:
        return {"committed": False, "reason": "nothing staged"}
    full_message = message + "\n\nCo-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>\n"
    subprocess.run(["git", "commit", "-m", full_message], cwd=ROOT, check=True)
    sha = _run_git(["rev-parse", "HEAD"])
    return {"committed": True, "sha": sha}


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

PHASE_ORDER = [
    "phase0_preflight", "phase1_build_fixtures", "phase2_mechanism_validation",
    "phase3_specialization_search", "phase4_dataset_pilot", "phase5_quality_gates",
    "phase6_train_selector", "phase7_evaluate_selector", "phase8_comparison_plan",
    "phase9_testing",
]


def _write_final_summary(out_dir: Path, status: Dict, stopped_reason: str) -> None:
    lines = ["# Selector v2 overnight contention-validation pilot -- final summary", ""]
    lines.append(f"Stopped: {stopped_reason}")
    lines.append(f"Elapsed: {round(time.time() - status['start_time'], 1)}s")
    lines.append("")
    for phase in PHASE_ORDER:
        entry = status["phases"].get(phase)
        if entry is None:
            lines.append(f"- {phase}: NOT STARTED")
            continue
        lines.append(f"- {phase}: {entry['status']}")
        if entry["status"] == "failed":
            lines.append(f"  error: {entry.get('error', '')}")
    (out_dir / "final_summary.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--deadline-seconds", type=int, default=DEADLINE_SECONDS_DEFAULT)
    parser.add_argument("--search-candidates", type=int, default=300)
    parser.add_argument("--search-seed", type=int, default=20260720)
    parser.add_argument("--pilot-target-windows", type=int, default=300)
    args = parser.parse_args()

    if args.output_dir is None:
        ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        out_dir = ROOT / "experiments" / f"selector_v2_overnight_{ts}"
    else:
        out_dir = ROOT / args.output_dir
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)

    status = _load_status(out_dir)
    status["deadline_seconds"] = args.deadline_seconds
    _save_status(out_dir, status)
    _log(out_dir, f"orchestrator starting; output_dir={out_dir}")

    def _run_phase(name, fn, *fargs, **fkwargs):
        entry = status["phases"].get(name)
        if entry and entry.get("status") == "completed":
            _log(out_dir, f"{name}: already completed, skipping (resume)")
            return entry.get("result")
        if _deadline_exceeded(status):
            _log(out_dir, f"{name}: deadline exceeded, not starting")
            status["phases"][name] = {"status": "skipped_deadline"}
            _save_status(out_dir, status)
            return None
        _log(out_dir, f"{name}: starting")
        t0 = time.time()
        try:
            result = fn(*fargs, **fkwargs)
            status["phases"][name] = {
                "status": "completed", "elapsed_s": round(time.time() - t0, 1), "result": result,
            }
            _save_status(out_dir, status)
            _log(out_dir, f"{name}: completed in {round(time.time()-t0,1)}s")
            return result
        except Exception as exc:
            tb = traceback.format_exc()
            status["phases"][name] = {
                "status": "failed", "elapsed_s": round(time.time() - t0, 1),
                "error": str(exc), "traceback": tb,
            }
            _save_status(out_dir, status)
            _log(out_dir, f"{name}: FAILED: {exc}")
            (out_dir / "logs" / f"{name}_traceback.log").write_text(tb)
            return None

    p0 = _run_phase("phase0_preflight", phase0_preflight, out_dir)
    if p0 is None:
        _write_final_summary(out_dir, status, "phase0_preflight failed or deadline exceeded")
        return 1

    p1 = _run_phase("phase1_build_fixtures", phase1_build_fixtures, out_dir)
    if p1 is not None:
        phase12_git_commit(
            out_dir, "Add targeted prefill/decode contention fixtures",
            ["src/llmserveopt/selector/dataset_v2/contention_fixtures.py",
             "experiments/runtime_validation_benchmark_pack/contention_targets/"],
        )

    p2 = _run_phase("phase2_mechanism_validation", phase2_mechanism_validation, out_dir)

    p3 = _run_phase(
        "phase3_specialization_search", phase3_specialization_search,
        out_dir, args.search_candidates, args.search_seed,
    )
    if p3 is not None:
        phase12_git_commit(
            out_dir, "Add contention specialization search",
            [str((out_dir / "specialization_search.csv").relative_to(ROOT)),
             str((out_dir / "contention_fixture_results.csv").relative_to(ROOT)),
             str((out_dir / "phase_status.json").relative_to(ROOT))],
        )

    p4 = None
    if p3 is not None and p3["all_gates_pass"] and not _deadline_exceeded(status):
        p4 = _run_phase("phase4_dataset_pilot", phase4_dataset_pilot, out_dir, args.pilot_target_windows)
        if p4 is not None:
            phase12_git_commit(
                out_dir, "Generate corrected-objective Selector Dataset v2 targeted pilot",
                [p4["pilot_output_dir"]],
            )
    else:
        _log(out_dir, "phase3 gates did not pass (or deadline exceeded) -- skipping phase4 dataset pilot")
        status["phases"]["phase4_dataset_pilot"] = {"status": "skipped_gate_failure"}
        _save_status(out_dir, status)

    p5 = _run_phase("phase5_quality_gates", phase5_quality_gates, out_dir, p3 or {"gates": {}, "all_gates_pass": False}, p4)

    p6 = p7 = None
    if p5 is not None and p5.get("proceed_to_selector_training") and p4 is not None:
        pilot_csv = ROOT / p4["pilot_output_dir"] / "selector_dataset_v2_corrected_objective_pilot.csv"
        if pilot_csv.exists():
            p6 = _run_phase("phase6_train_selector", phase6_train_selector, out_dir, pilot_csv)
            if p6 is not None:
                p7 = _run_phase("phase7_evaluate_selector", phase7_evaluate_selector, out_dir,
                                 ROOT / p6["models_dir"], pilot_csv)
                if p7 is not None:
                    phase12_git_commit(
                        out_dir, "Add Selector v2 prototype training and evaluation",
                        [p6["models_dir"], p7["eval_dir"],
                         str((out_dir / "selector_metrics.json").relative_to(ROOT))],
                    )
    else:
        _log(out_dir, "quality gates did not clear all_gates_pass -- skipping selector training/eval")
        status["phases"].setdefault("phase6_train_selector", {"status": "skipped_gate_failure"})
        status["phases"].setdefault("phase7_evaluate_selector", {"status": "skipped_gate_failure"})
        _save_status(out_dir, status)

    _run_phase("phase8_comparison_plan", phase8_comparison_plan, out_dir)
    _run_phase("phase9_testing", phase9_testing, out_dir)

    _write_final_summary(out_dir, status, "all applicable phases attempted")
    _log(out_dir, "orchestrator finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
