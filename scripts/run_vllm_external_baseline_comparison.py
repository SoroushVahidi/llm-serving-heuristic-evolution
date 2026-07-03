#!/usr/bin/env python3
"""
External-admission-control comparison on top of a real vLLM server.

Compares this project's own scheduling policies (src/llmserveopt/policies/),
used here as CLIENT-SIDE admission controllers gating a fixed concurrency
budget, against naive direct submission — all issuing real HTTP requests to
the SAME vLLM server, over the IDENTICAL fixed request plan (arrivals,
prompts, priorities, deadlines, target output lengths) per policy, so
differences in outcome are attributable to admission ORDER, not workload
variance.

**What this is:** an external admission-control layer in front of vLLM.
This project's policies (`select_action(ObservableState) -> Action`) were
built for the discrete-event simulator, where a GPU's `ObservableGPUState`
reflects true internal batch/KV-cache state. Here, `ObservableGPUState` is
reconstructed from only what a client can observe: how many of ITS OWN
requests are currently in flight (bounded by a chosen concurrency budget).
This is a legitimate, different measurement — admission-order effects under
a client-side concurrency cap — not a reproduction of vLLM's own internal
batching/scheduling, which remains invisible from outside (see
docs/real_llm_simulator_integration_plan.md and
docs/vllm_real_serving_external_baseline_pilot.md for the exact boundary).

**What is NOT wired (do not fake):** the trained selector and any
"generated heuristic" from Phase 2B require simulator-internal features
(KV pressure, batch composition, etc.) that this client-side harness cannot
observe against a real server. Requesting them via --policies raises a
clear error rather than silently substituting a fixed baseline.

**"vllm_direct" vs. the policies:** vllm_direct submits requests in strict
arrival order bounded by the same concurrency cap, with NO ObservableState/
Action machinery at all. It is expected to behave identically to `fifo`
(which also admits oldest-arrived-first) — any observed difference between
them indicates overhead or a bug in the policy-admission loop itself, which
is itself a useful sanity signal, not noise to discard.

No hosted API is ever called (no Cohere/Gemini/OpenAI/Azure). The only
network target is a vLLM server (local subprocess or --server-url).

Usage (dry-run):
    python scripts/run_vllm_external_baseline_comparison.py \\
        --dry-run --output-dir experiments/real_llm/vllm_baseline_comparison_pilot_DRYRUN

Usage (mock, no network):
    python scripts/run_vllm_external_baseline_comparison.py \\
        --mock --output-dir /tmp/x

Usage (live, against an already-running vLLM server):
    python scripts/run_vllm_external_baseline_comparison.py \\
        --allow-live-server --server-url http://127.0.0.1:8000 \\
        --model Qwen/Qwen2.5-0.5B-Instruct \\
        --output-dir experiments/real_llm/vllm_baseline_comparison_pilot_<timestamp>
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from llmserveopt.real_llm import calibration_common as cc  # noqa: E402
from llmserveopt.core.types import (  # noqa: E402
    Request, ObservableRequest, ObservableGPUState, ObservableState,
)
from llmserveopt.policies.registry import make_policy  # noqa: E402
from llmserveopt.workloads.synthetic import DEFAULT_SLO_CLASSES, SLOClass  # noqa: E402
import run_vllm_serving_baseline_pilot as vllm_mod  # noqa: E402

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# Policies this harness can actually run against a real server today.
WIRED_POLICIES = (
    "vllm_direct", "fifo", "edf", "shortest_output_first",
    "least_laxity_first", "estimated_service_time_first",
)

# "vllm_default" is the natural name for "no external admission control at
# all" from the vLLM side; it is mechanically identical to vllm_direct here
# (see the module docstring's "vllm_direct vs. the policies" section).
POLICY_ALIASES = {"vllm_default": "vllm_direct", "llf": "least_laxity_first", "estf": "estimated_service_time_first"}


def normalize_policy_name(name: str) -> str:
    return POLICY_ALIASES.get(name, name)

# Requested-but-not-implemented policies and why, surfaced verbatim to the
# user rather than silently ignored or faked. Investigated (not assumed)
# before this task concluded they aren't safely wirable today:
#   - Most of the selector's 18 features (src/llmserveopt/selector/
#     features.py: queue_length, active_sequence_count, free_sequence_ratio,
#     prompt/output-length stats, slack stats, arrival-rate/burstiness,
#     recent SLO violation rate) ARE reconstructable from this harness's own
#     client-side bookkeeping. Only kv_utilization has no honest client-side
#     substitute without scraping vLLM's /metrics endpoint (not implemented
#     here).
#   - More fundamentally: the only serialized model artifacts on disk
#     (results/phase2a2_selector_dataset/, phase2a3_selector_eval/,
#     phase2a4_2b4_final_eval/ -- all *.joblib) were trained under the
#     PRE-CORRECTION objective that Phase 2B.14's metric audit found flawed
#     (completed-only WG denominator). The corrected, validated retraining
#     (Phase 2B.15/16 -- the one actually described as "best" going
#     forward) was evaluated only in-memory by one-off scripts and was
#     never persisted as a loadable model. Loading a pre-correction
#     artifact and calling it "our best selector" would misrepresent this
#     project's own findings, so neither policy is wired here.
NOT_WIRED_POLICIES = {
    "generated_heuristic": (
        "No current, methodologically-valid model artifact exists to load "
        "(see the module-level comment above): the only serialized "
        "*.joblib files on disk predate the Phase 2B.14 objective "
        "correction. Wiring this safely would require re-running the "
        "Phase 2B.15/16 corrected-objective training pipeline to produce a "
        "fresh artifact, then building a feature adapter for the ~17 of 18 "
        "features that ARE client-observable plus an honest placeholder "
        "(or /metrics scrape) for kv_utilization. Neither is done here."
    ),
    "best_generated": "alias of generated_heuristic -- see that entry.",
    "selector": (
        "Same gap as generated_heuristic: no current corrected-objective "
        "selector model was ever persisted to disk. Not wired."
    ),
}


# ---------------------------------------------------------------------------
# Plan construction (identical across every policy)
# ---------------------------------------------------------------------------

@dataclass
class PlanRow:
    request_id: int
    prompt_bucket: str
    target_output_tokens: int
    concurrency_level: int
    request_index: int
    intended_prompt_tokens: int
    priority: float
    slo_slack_seconds: float
    class_id: str
    prompt_text: str = field(repr=False)


def _assign_slo_class(rng, slo_classes: List[SLOClass]) -> SLOClass:
    weights = [c.weight for c in slo_classes]
    total = sum(weights)
    r = rng.random() * total
    acc = 0.0
    for c, w in zip(slo_classes, weights):
        acc += w
        if r <= acc:
            return c
    return slo_classes[-1]


def build_request_plan(
    prompt_buckets: List[str], target_output_tokens_list: List[int],
    concurrency_list: List[int], requests_per_cell: int, seed: int,
    slo_classes: List[SLOClass] = DEFAULT_SLO_CLASSES,
) -> List[PlanRow]:
    import random
    from itertools import product

    rng = random.Random(seed)
    plan: List[PlanRow] = []
    request_id = 0
    for bucket, target, concurrency in product(prompt_buckets, target_output_tokens_list, concurrency_list):
        for i in range(requests_per_cell):
            prompt_text = cc.build_length_targeted_prompt(bucket, target, seed, i)
            cls = _assign_slo_class(rng, slo_classes)
            plan.append(PlanRow(
                request_id=request_id, prompt_bucket=bucket, target_output_tokens=target,
                concurrency_level=concurrency, request_index=i,
                intended_prompt_tokens=cc.approx_token_count(prompt_text),
                priority=cls.priority, slo_slack_seconds=cls.slo_slack, class_id=cls.class_id,
                prompt_text=prompt_text,
            ))
            request_id += 1
    return plan


def run_warmup(out_dir: Path, *, model: str, base_url: Optional[str], mock: bool, timeout_s: float) -> None:
    """One short/target=64 and one medium/target=128 request at
    concurrency=1, run before any measured policy loop, to absorb vLLM's
    one-time JIT kernel compilation latency spike under --enforce-eager
    (see experiments/real_llm/vllm_healthcheck_*/healthcheck.md). Written
    to its own files, never mixed into requests.jsonl / policy metrics."""
    warmup_plan = [
        PlanRow(
            request_id=-1, prompt_bucket="short", target_output_tokens=64, concurrency_level=1,
            request_index=0, intended_prompt_tokens=cc.approx_token_count(cc.build_length_targeted_prompt("short", 64, 0, 0)),
            priority=1.0, slo_slack_seconds=10.0, class_id="warmup",
            prompt_text=cc.build_length_targeted_prompt("short", 64, 0, 0),
        ),
        PlanRow(
            request_id=-2, prompt_bucket="medium", target_output_tokens=128, concurrency_level=1,
            request_index=0, intended_prompt_tokens=cc.approx_token_count(cc.build_length_targeted_prompt("medium", 128, 0, 0)),
            priority=1.0, slo_slack_seconds=10.0, class_id="warmup",
            prompt_text=cc.build_length_targeted_prompt("medium", 128, 0, 0),
        ),
    ]
    rows = []
    for planned_row in warmup_plan:
        t0 = time.monotonic()
        try:
            out = _dispatch(planned_row, model=model, base_url=base_url, mock=mock, timeout_s=timeout_s)
            rows.append({
                "request_id": planned_row.request_id, "prompt_bucket": planned_row.prompt_bucket,
                "target_output_tokens": planned_row.target_output_tokens, "status": "success",
                "ttft_seconds": out.get("ttft_seconds"),
                "server_request_latency_seconds": out.get("server_request_latency_seconds"),
                "total_wall_time_seconds": round(time.monotonic() - t0, 4),
                "output_tokens": out.get("output_tokens"), "finish_reason": out.get("finish_reason"),
            })
        except Exception as exc:  # noqa: BLE001
            rows.append({
                "request_id": planned_row.request_id, "prompt_bucket": planned_row.prompt_bucket,
                "target_output_tokens": planned_row.target_output_tokens, "status": "error",
                "error_type": type(exc).__name__, "error_message": str(exc)[:500],
                "total_wall_time_seconds": round(time.monotonic() - t0, 4),
            })

    with open(out_dir / "warmup_requests.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    lines = [
        "# Warm-up phase — NOT counted in policy metrics",
        "",
        "One short/target=64 and one medium/target=128 request at concurrency=1,",
        "run before any measured policy loop to absorb vLLM's one-time JIT kernel",
        "compilation latency spike under `--enforce-eager` (observed in",
        "`experiments/real_llm/vllm_healthcheck_*/healthcheck.md`: first request",
        "needed ~180s, subsequent requests ~0.3s). These requests are excluded",
        "from `requests.jsonl` and every `aggregate_by_*.csv` / policy metric.",
        "",
        "| request_id | bucket | target | status | ttft_s | server_latency_s | wall_s |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['request_id']} | {row['prompt_bucket']} | {row['target_output_tokens']} | "
            f"{row['status']} | {row.get('ttft_seconds')} | {row.get('server_request_latency_seconds')} | "
            f"{row['total_wall_time_seconds']} |"
        )
    (out_dir / "warmup_summary.md").write_text("\n".join(lines) + "\n")


def write_request_plan(plan: List[PlanRow], out_dir: Path) -> None:
    with open(out_dir / "request_plan.jsonl", "w") as f:
        for row in plan:
            f.write(json.dumps(asdict(row)) + "\n")


# ---------------------------------------------------------------------------
# Per-cell, per-policy external-admission execution against vLLM
# ---------------------------------------------------------------------------

@dataclass
class ComparisonResultRow:
    policy: str
    request_id: int
    prompt_bucket: str
    target_output_tokens: int
    concurrency_level: int
    class_id: str
    priority: float
    arrival_time_s: float
    admission_time_s: float
    completion_time_s: Optional[float]
    queuing_delay_s: float
    ttft_seconds: Optional[float]
    server_request_latency_seconds: Optional[float]
    total_wall_time_seconds: Optional[float]
    slo_deadline_s: float
    slo_violated: Optional[bool]
    output_tokens: Optional[float]
    status: str
    error_type: Optional[str]
    error_message: Optional[str]


def _dispatch(planned_row: PlanRow, *, model: str, base_url: Optional[str], mock: bool, timeout_s: float) -> Dict[str, Any]:
    fake_planned = vllm_mod.VllmPlannedRequest(
        request_id=str(planned_row.request_id), prompt_bucket=planned_row.prompt_bucket,
        target_output_tokens=planned_row.target_output_tokens,
        concurrency_level=planned_row.concurrency_level, request_index=planned_row.request_index,
        intended_prompt_tokens=planned_row.intended_prompt_tokens, prompt_text=planned_row.prompt_text,
    )
    if mock:
        return vllm_mod.mock_call(fake_planned)
    return vllm_mod.query_vllm_completion(
        base_url, model, planned_row.prompt_text,
        max_tokens=planned_row.target_output_tokens * 2, timeout_s=timeout_s,
    )


def run_cell_for_policy(
    policy_name: str, cell_plan: List[PlanRow], concurrency: int, *,
    model: str, base_url: Optional[str], mock: bool, timeout_s: float,
) -> List[ComparisonResultRow]:
    policy = None if policy_name == "vllm_direct" else make_policy(policy_name)

    requests = [
        Request(
            request_id=row.request_id, arrival_time=0.0, prompt_tokens=row.intended_prompt_tokens,
            predicted_output_tokens=row.target_output_tokens,
            actual_output_tokens=row.target_output_tokens,  # placeholder; real value recorded post-hoc below
            slo_deadline=row.slo_slack_seconds, priority=row.priority, class_id=row.class_id,
        )
        for row in cell_plan
    ]
    by_id = {row.request_id: row for row in cell_plan}
    waiting: List[ObservableRequest] = [ObservableRequest.from_request(r) for r in requests]
    gpu_state = ObservableGPUState(
        gpu_id=0, max_active_sequences=concurrency, max_batch_tokens=10**9, max_kv_tokens=10**9,
        active_request_ids=[], active_requests_info=[], current_kv_tokens=0, tokens_decoded_per_request={},
    )

    results: List[ComparisonResultRow] = []
    active: Dict[int, Tuple[concurrent.futures.Future, float]] = {}
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=max(1, concurrency))
    t0 = time.monotonic()
    completed_count = 0
    step = 0

    try:
        while waiting or active:
            now = time.monotonic() - t0
            if policy is not None:
                # Policies mutate gpu.active_request_ids/current_kv_tokens as
                # internal bookkeeping ("avoid over-admission within the same
                # action" — see e.g. fifo.py). That must land on a disposable
                # snapshot, never on our own authoritative gpu_state, or a
                # completed request's slot leaks permanently (only visible
                # once concurrency < requests pending, e.g. concurrency=1
                # with 2+ requests in a cell).
                gpu_snapshot = ObservableGPUState(
                    gpu_id=gpu_state.gpu_id, max_active_sequences=gpu_state.max_active_sequences,
                    max_batch_tokens=gpu_state.max_batch_tokens, max_kv_tokens=gpu_state.max_kv_tokens,
                    active_request_ids=list(gpu_state.active_request_ids),
                    active_requests_info=list(gpu_state.active_requests_info),
                    current_kv_tokens=gpu_state.current_kv_tokens,
                    tokens_decoded_per_request=dict(gpu_state.tokens_decoded_per_request),
                    prefilling_count=gpu_state.prefilling_count, decoding_count=gpu_state.decoding_count,
                )
                state = ObservableState(
                    time=now, waiting_queue=list(waiting), gpu_states=[gpu_snapshot],
                    completed_count=completed_count, step=step,
                )
                action = policy.select_action(state)
                admitted_ids = action.all_admitted_ids()
            else:  # vllm_direct: strict arrival-order admission bounded by concurrency
                free = gpu_state.max_active_sequences - len(gpu_state.active_request_ids)
                admitted_ids = {r.request_id for r in waiting[:max(0, free)]}

            for rid in list(admitted_ids):
                if rid in active:
                    continue
                idx = next((i for i, r in enumerate(waiting) if r.request_id == rid), None)
                if idx is None:
                    continue
                waiting.pop(idx)
                admission_time = time.monotonic() - t0
                fut = executor.submit(
                    _dispatch, by_id[rid], model=model, base_url=base_url, mock=mock, timeout_s=timeout_s,
                )
                active[rid] = (fut, admission_time)
                gpu_state.active_request_ids.append(rid)

            if not active:
                break  # nothing admitted and nothing pending — avoid a busy loop

            futures = [f for f, _ in active.values()]
            done, _ = concurrent.futures.wait(futures, timeout=timeout_s + 5, return_when=concurrent.futures.FIRST_COMPLETED)

            for rid, (fut, admission_time) in list(active.items()):
                if fut not in done:
                    continue
                row = by_id[rid]
                completion_time = time.monotonic() - t0
                try:
                    out = fut.result()
                    output_tokens = out.get("output_tokens")
                    results.append(ComparisonResultRow(
                        policy=policy_name, request_id=rid, prompt_bucket=row.prompt_bucket,
                        target_output_tokens=row.target_output_tokens, concurrency_level=concurrency,
                        class_id=row.class_id, priority=row.priority, arrival_time_s=0.0,
                        admission_time_s=round(admission_time, 4), completion_time_s=round(completion_time, 4),
                        queuing_delay_s=round(admission_time, 4),
                        ttft_seconds=out.get("ttft_seconds"),
                        server_request_latency_seconds=out.get("server_request_latency_seconds"),
                        total_wall_time_seconds=round(completion_time, 4),
                        slo_deadline_s=row.slo_slack_seconds,
                        slo_violated=completion_time > row.slo_slack_seconds,
                        output_tokens=output_tokens, status="success",
                        error_type=None, error_message=None,
                    ))
                except Exception as exc:  # noqa: BLE001
                    status = "timeout" if "timed out" in str(exc).lower() else "error"
                    results.append(ComparisonResultRow(
                        policy=policy_name, request_id=rid, prompt_bucket=row.prompt_bucket,
                        target_output_tokens=row.target_output_tokens, concurrency_level=concurrency,
                        class_id=row.class_id, priority=row.priority, arrival_time_s=0.0,
                        admission_time_s=round(admission_time, 4), completion_time_s=None,
                        queuing_delay_s=round(admission_time, 4),
                        ttft_seconds=None, server_request_latency_seconds=None,
                        total_wall_time_seconds=round(completion_time, 4),
                        slo_deadline_s=row.slo_slack_seconds, slo_violated=None,
                        output_tokens=None, status=status, error_type=type(exc).__name__,
                        error_message=str(exc)[:500],
                    ))
                del active[rid]
                gpu_state.active_request_ids.remove(rid)
                completed_count += 1
            step += 1
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    return results


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_policy_metrics(rows: List[Dict[str, Any]], policy_wall_clock_s: float) -> Dict[str, Any]:
    total = len(rows)
    completed = [r for r in rows if r["status"] == "success"]
    failed = [r for r in rows if r["status"] in ("error", "timeout")]

    def stats(values: List[float]) -> Dict[str, Optional[float]]:
        values = [v for v in values if v is not None]
        if not values:
            return {"count": 0, "mean": None, "p50": None, "p95": None, "p99": None}
        return {
            "count": len(values), "mean": sum(values) / len(values),
            "p50": cc._percentile(values, 0.50), "p95": cc._percentile(values, 0.95),
            "p99": cc._percentile(values, 0.99),
        }

    ttft_stats = stats([r["ttft_seconds"] for r in completed])
    latency_stats = stats([r["server_request_latency_seconds"] for r in completed])
    wall_stats = stats([r["total_wall_time_seconds"] for r in completed])
    output_tokens = [r["output_tokens"] for r in completed if r.get("output_tokens")]

    weights = [r["priority"] if r["priority"] > 0 else 1.0 for r in completed]
    met = [0.0 if r["slo_violated"] else 1.0 for r in completed]
    total_weight = sum(weights)
    conditional_wg = (sum(w * m for w, m in zip(weights, met)) / total_weight) if total_weight > 0 else 0.0
    completion_fraction = (len(completed) / total) if total > 0 else float("nan")
    # Arrival-normalized weighted goodput: same convention as
    # scripts/run_phase2b14_metric_audit_scorpio_ablation.py's
    # arrival_normalized_wg = completion_fraction * conditional_WG — the
    # corrected objective (denominator over ALL arrivals, not completed-only).
    arrival_normalized_wg = completion_fraction * conditional_wg if total > 0 else None

    slo_violated_flags = [r["slo_violated"] for r in completed if r["slo_violated"] is not None]
    slo_violation_rate = (sum(slo_violated_flags) / len(slo_violated_flags)) if slo_violated_flags else None

    request_throughput = (len(completed) / policy_wall_clock_s) if policy_wall_clock_s > 0 else None
    token_throughput = (sum(output_tokens) / policy_wall_clock_s) if (policy_wall_clock_s > 0 and output_tokens) else None

    return {
        "n_total": total,
        "n_completed": len(completed),
        "n_failed": len(failed),
        "completion_fraction": completion_fraction,
        "ttft_stats": ttft_stats,
        "server_latency_stats": latency_stats,
        "total_wall_time_stats": wall_stats,
        "mean_output_tokens": (sum(output_tokens) / len(output_tokens)) if output_tokens else None,
        "conditional_weighted_goodput": conditional_wg,
        "arrival_normalized_weighted_goodput": arrival_normalized_wg,
        "slo_violation_rate_among_completed": slo_violation_rate,
        "request_throughput_per_sec": request_throughput,
        "output_token_throughput_per_sec": token_throughput,
        "policy_wall_clock_seconds": policy_wall_clock_s,
    }


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_outputs(
    out_dir: Path, all_rows: List[Dict[str, Any]], per_policy_wall_clock: Dict[str, float], cfg: Dict[str, Any],
) -> Dict[str, Any]:
    import pandas as pd

    with open(out_dir / "requests.jsonl", "w") as f:
        for row in all_rows:
            f.write(json.dumps(row) + "\n")

    errors = [r for r in all_rows if r["status"] != "success"]
    with open(out_dir / "errors.jsonl", "w") as f:
        for row in errors:
            f.write(json.dumps(row) + "\n")

    df = pd.DataFrame(all_rows) if all_rows else pd.DataFrame()
    policies = sorted({r["policy"] for r in all_rows})

    per_policy_metrics = {
        policy: compute_policy_metrics([r for r in all_rows if r["policy"] == policy], per_policy_wall_clock.get(policy, 0.0))
        for policy in policies
    }

    def _group_rows(group_cols: List[str]) -> List[Dict[str, Any]]:
        if df.empty:
            return []
        records = []
        for keys, sub in df.groupby(group_cols):
            keys = keys if isinstance(keys, tuple) else (keys,)
            sub_success = sub[sub["status"] == "success"]
            rec = dict(zip(group_cols, keys))
            rec["n_total"] = len(sub)
            rec["n_success"] = len(sub_success)
            rec["n_failed"] = int((sub["status"].isin(["error", "timeout"])).sum())
            lat = sub_success["server_request_latency_seconds"].dropna().tolist()
            ttft = sub_success["ttft_seconds"].dropna().tolist()
            rec["mean_latency_s"] = (sum(lat) / len(lat)) if lat else None
            rec["mean_ttft_s"] = (sum(ttft) / len(ttft)) if ttft else None
            records.append(rec)
        return records

    by_policy = [{"policy": p, **per_policy_metrics[p]} for p in policies]
    pd.DataFrame(by_policy).to_csv(out_dir / "aggregate_by_policy.csv", index=False)

    pd.DataFrame(_group_rows(["policy", "concurrency_level"])).to_csv(out_dir / "aggregate_by_concurrency.csv", index=False)
    pd.DataFrame(_group_rows(["policy", "target_output_tokens"])).to_csv(out_dir / "aggregate_by_target_output_tokens.csv", index=False)
    pd.DataFrame(_group_rows(["policy", "prompt_bucket"])).to_csv(out_dir / "aggregate_by_prompt_bucket.csv", index=False)

    overall = {
        "total_records": len(all_rows),
        "policies": policies,
        "per_policy": per_policy_metrics,
    }
    (out_dir / "summary.json").write_text(json.dumps(overall, indent=2))
    write_summary_md(out_dir, overall, cfg)
    return overall


def write_summary_md(out_dir: Path, overall: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    lines = [
        "# vLLM External-Admission Baseline Comparison — Summary",
        "",
        f"**Model:** `{cfg.get('model')}`",
        f"**Run status:** {cfg.get('run_status')}",
        f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
    ]
    if cfg.get("run_status") != "completed":
        lines += [
            "**No real vLLM server was used for this run.** "
            f"See `run_config.json` (`run_status: {cfg.get('run_status')}`).",
            "",
        ]
    lines += [
        "## Policies compared",
        "",
        "| Policy | n_total | n_completed | n_failed | Arrival-norm. WG | SLO violation (completed) | Mean TTFT (s) | Mean server latency (s) | Req/s |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for policy, m in overall["per_policy"].items():
        anwg = m["arrival_normalized_weighted_goodput"]
        slo = m["slo_violation_rate_among_completed"]
        lines.append(
            f"| {policy} | {m['n_total']} | {m['n_completed']} | {m['n_failed']} | "
            f"{anwg:.4f} | {(slo if slo is not None else float('nan')):.4f} | "
            f"{(m['ttft_stats']['mean'] or float('nan')):.4f} | "
            f"{(m['server_latency_stats']['mean'] or float('nan')):.4f} | "
            f"{(m['request_throughput_per_sec'] or float('nan')):.3f} |"
        )
    lines += [
        "",
        "Policies not compared: " + ", ".join(NOT_WIRED_POLICIES.keys()) + " — see "
        "`docs/vllm_real_serving_external_baseline_pilot.md` for why.",
        "",
        "See `aggregate_by_policy.csv`, `aggregate_by_concurrency.csv`, "
        "`aggregate_by_target_output_tokens.csv`, `aggregate_by_prompt_bucket.csv` "
        "for breakdowns, and `errors.jsonl` for failure detail.",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")


def write_manifest_and_repro(out_dir: Path, plan: List[PlanRow], cfg: Dict[str, Any]) -> None:
    git_info = vllm_mod._git_info(ROOT)
    (out_dir / "manifest.json").write_text(json.dumps({
        **git_info,
        "planned_requests_per_policy": len(plan),
        "policies": cfg["policies"],
        "cells": sorted({(r.prompt_bucket, r.target_output_tokens, r.concurrency_level) for r in plan}),
    }, indent=2))
    vllm_mod.write_reproducibility_md(out_dir, cfg, git_info)


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def csv_str_list(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-live-server", action="store_true")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--server-url", "--base-url", dest="server_url", default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--policies", type=csv_str_list, default=list(WIRED_POLICIES))
    parser.add_argument("--prompt-buckets", type=csv_str_list, default=["short", "medium"])
    parser.add_argument("--target-output-tokens-list", type=cc.csv_int_list, default=[64, 128])
    parser.add_argument("--concurrency-list", type=cc.csv_int_list, default=[1, 2, 4])
    parser.add_argument("--requests-per-cell", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--max-total-requests", type=int, default=1000)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument(
        "--stream", action="store_true",
        help="No-op: live requests always use vLLM's streaming SSE endpoint "
        "(needed to measure TTFT). Accepted for CLI compatibility.",
    )
    parser.add_argument(
        "--warmup", action="store_true",
        help="Run one short/target=64 and one medium/target=128 request at "
        "concurrency=1 before measurement, to absorb any one-time JIT "
        "kernel compilation latency spike. Not counted in policy metrics; "
        "written to warmup_requests.jsonl/warmup_summary.md instead.",
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def repo_path(root: Path, raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else root / p


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    args.policies = [normalize_policy_name(p) for p in args.policies]

    unknown = set(args.policies) - set(WIRED_POLICIES) - set(NOT_WIRED_POLICIES.keys())
    if unknown:
        print(f"ERROR: unknown policies: {sorted(unknown)}. Known: {WIRED_POLICIES}", file=sys.stderr)
        return 2
    requested_not_wired = set(args.policies) & set(NOT_WIRED_POLICIES.keys())
    if requested_not_wired:
        print("ERROR: the following requested policies are not wired to a live server:", file=sys.stderr)
        for name in sorted(requested_not_wired):
            print(f"  {name}: {NOT_WIRED_POLICIES[name]}", file=sys.stderr)
        print(f"Use --policies with a subset of: {WIRED_POLICIES}", file=sys.stderr)
        return 8

    if not args.dry_run and not args.allow_live_server and not args.mock:
        print("ERROR: specify --dry-run, --mock, or --allow-live-server.", file=sys.stderr)
        return 2
    if args.allow_live_server and not args.mock and not args.server_url:
        print("ERROR: --allow-live-server requires --server-url.", file=sys.stderr)
        return 2

    out_dir = repo_path(ROOT, args.output_dir)
    requests_path = out_dir / "requests.jsonl"
    if out_dir.exists() and requests_path.exists() and requests_path.stat().st_size > 0:
        print(f"ERROR: output dir {out_dir} already has a non-empty requests.jsonl. Choose a new --output-dir.", file=sys.stderr)
        return 3
    out_dir.mkdir(parents=True, exist_ok=True)

    plan = build_request_plan(
        args.prompt_buckets, args.target_output_tokens_list, args.concurrency_list,
        args.requests_per_cell, args.seed,
    )
    total_planned = len(plan) * len(args.policies)
    if total_planned > args.max_total_requests:
        print(
            f"HARD CAP VIOLATION: planned {total_planned} requests ({len(plan)} x "
            f"{len(args.policies)} policies) exceeds --max-total-requests={args.max_total_requests}",
            file=sys.stderr,
        )
        return 4

    run_status = "planned_only"
    if args.allow_live_server and not args.mock:
        run_status = "completed"
    elif args.mock:
        run_status = "completed_mock"
    elif args.dry_run:
        run_status = "planned_only" if vllm_mod.vllm_cli_available() else "planned_only_vllm_not_installed"

    cfg = {
        "model": args.model, "policies": args.policies, "prompt_buckets": args.prompt_buckets,
        "target_output_tokens_list": args.target_output_tokens_list, "concurrency_list": args.concurrency_list,
        "requests_per_cell": args.requests_per_cell, "timeout_seconds": args.timeout_seconds,
        "max_total_requests": args.max_total_requests, "fail_fast": args.fail_fast, "seed": args.seed,
        "mock": args.mock, "server_url": args.server_url, "run_status": run_status,
        "not_wired_policies": NOT_WIRED_POLICIES,
    }
    write_request_plan(plan, out_dir)
    write_manifest_and_repro(out_dir, plan, cfg)
    (out_dir / "run_config.json").write_text(json.dumps(cfg, indent=2))

    print("vLLM external-admission baseline comparison")
    print(f"  output_dir:        {out_dir}")
    print(f"  planned per policy: {len(plan)}")
    print(f"  policies:          {args.policies}")
    print(f"  run_status:        {run_status}")

    if args.dry_run and not args.allow_live_server and not args.mock:
        write_summary_md(out_dir, {"per_policy": {}}, cfg)
        (out_dir / "summary.json").write_text(json.dumps({"per_policy": {}}, indent=2))
        print("  No vLLM server was launched or queried (dry-run).")
        return 0

    if args.server_url:
        try:
            req = urllib.request.Request(f"{args.server_url.rstrip('/')}/v1/models")
            with urllib.request.urlopen(req, timeout=10) as resp:
                (out_dir / "server_status.json").write_text(resp.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            (out_dir / "server_status.json").write_text(json.dumps({"error": str(exc)}))

    if args.warmup:
        print("  Running warm-up phase (not counted in policy metrics)...")
        run_warmup(out_dir, model=args.model, base_url=args.server_url, mock=args.mock, timeout_s=args.timeout_seconds)

    cells: Dict[Tuple[str, int, int], List[PlanRow]] = {}
    for row in plan:
        cells.setdefault((row.prompt_bucket, row.target_output_tokens, row.concurrency_level), []).append(row)

    all_rows: List[Dict[str, Any]] = []
    per_policy_wall_clock: Dict[str, float] = {}
    fail_fast_triggered = False

    for policy_name in args.policies:
        policy_t0 = time.monotonic()
        for (bucket, target, concurrency), cell_plan in cells.items():
            if fail_fast_triggered:
                break
            cell_results = run_cell_for_policy(
                policy_name, cell_plan, concurrency, model=args.model,
                base_url=args.server_url, mock=args.mock, timeout_s=args.timeout_seconds,
            )
            for r in cell_results:
                all_rows.append(asdict(r))
            if args.fail_fast:
                n = len(cell_results)
                n_failed = sum(1 for r in cell_results if r.status != "success")
                if n >= 10 and n_failed / n > 0.5:
                    fail_fast_triggered = True
                    print(f"FAIL-FAST: aborting after high failure rate in policy={policy_name}", file=sys.stderr)
                    break
        per_policy_wall_clock[policy_name] = time.monotonic() - policy_t0

    overall = write_outputs(out_dir, all_rows, per_policy_wall_clock, cfg)

    for policy_name, m in overall["per_policy"].items():
        print(f"  [{policy_name}] completed={m['n_completed']}/{m['n_total']} arrival_norm_wg={m['arrival_normalized_weighted_goodput']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
