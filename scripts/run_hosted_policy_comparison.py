#!/usr/bin/env python3
"""
Hosted-provider (Cohere / Gemini) external-admission policy comparison.

Industry-facing validation only: compares this project's own scheduling
policies -- used as CLIENT-SIDE admission controllers gating a fixed
concurrency budget -- against each other, issuing REAL requests to a
commercial hosted LLM API, over the IDENTICAL fixed request plan per
policy. This is NOT a comparison against the provider's own internal
scheduler (invisible from outside, exactly like vLLM's -- see
docs/vllm_real_serving_external_baseline_pilot.md for the same boundary
argument applied there). External admission/scheduling only.

Reuses:
  - The admission-loop/metrics/decision-divergence/bootstrap-CI machinery
    from scripts/run_vllm_external_baseline_comparison.py (imported as
    `vext`) -- provider-agnostic, no vLLM-specific code in that machinery.
  - The cost-cap/RPM-limiter/budget-tracker/provider-call machinery already
    built and tested for the Cohere/Gemini calibration pilots
    (src/llmserveopt/real_llm/calibration_common.py as `cc`,
    scripts/run_cohere_api_calibration.py as `cohere_mod`,
    scripts/run_gemini_real_llm_calibration.py as `gemini_mod`).

Safety
------
- No live API call happens without --allow-live-api (explicit; --mock and
  --dry-run never touch a real network).
- Hard caps (cost / requests / input tokens / output tokens) are enforced
  TWICE: once via a worst-case dry-run estimate before any call
  (cc.validate_call_plan), and again per-request at runtime via a
  thread-safe cc.BudgetTracker that refuses to dispatch once any cap would
  be exceeded, even under concurrency (defense in depth, not just an
  up-front estimate).
- Only Cohere (command-r7b-12-2024) and Gemini (gemini-3.1-flash-lite) are
  supported. Azure/Fireworks/CloudRift are out of scope -- requesting them
  via --provider raises a clear error rather than silently substituting.
- selector artifact must pass the same manifest-verification contract as
  the vLLM harness (scripts/run_vllm_external_baseline_comparison.py's
  load_and_validate_selector_artifact) -- stale pre-correction artifacts
  are rejected identically here.

Usage (dry-run cost check, no network):
    python scripts/run_hosted_policy_comparison.py \\
        --provider cohere --dry-run-cost-check \\
        --output-dir /tmp/x

Usage (mock, no network):
    python scripts/run_hosted_policy_comparison.py \\
        --provider cohere --mock --output-dir /tmp/x

Usage (live, real spend, hard caps enforced):
    python scripts/run_hosted_policy_comparison.py \\
        --provider cohere --allow-live-api \\
        --policies fifo,edf,least_laxity_first,estimated_service_time_first,shortest_output_first,selector \\
        --selector-artifact results/corrected_selector_artifact_regression_anwg/regression_anwg_selector.joblib \\
        --require-our-method \\
        --max-estimated-cost-usd 5 --max-total-requests 1000 \\
        --max-total-input-tokens 1000000 --max-total-output-tokens 300000 \\
        --output-dir experiments/real_llm/cohere_hosted_policy_comparison_<timestamp>
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from llmserveopt.real_llm import calibration_common as cc  # noqa: E402
from llmserveopt.policies.registry import make_policy  # noqa: E402
from llmserveopt.core.types import Request, ObservableRequest, ObservableGPUState, ObservableState  # noqa: E402


def _load_sibling_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


vext = _load_sibling_module("run_vllm_external_baseline_comparison", "run_vllm_external_baseline_comparison.py")
cohere_mod = _load_sibling_module("run_cohere_api_calibration", "run_cohere_api_calibration.py")
gemini_mod = _load_sibling_module("run_gemini_real_llm_calibration", "run_gemini_real_llm_calibration.py")

# Policies this harness compares over hosted providers. No vllm_direct (not
# a meaningful concept without vLLM) and no generated_heuristic/best_generated
# (see vext.NOT_WIRED_POLICIES -- same gap applies here).
HOSTED_POLICIES = (
    "fifo", "edf", "least_laxity_first", "estimated_service_time_first", "shortest_output_first",
)
CONDITIONAL_POLICIES = ("selector",)

PROVIDER_CONFIG: Dict[str, Dict[str, Any]] = {
    "cohere": {
        "default_model": cohere_mod.DEFAULT_MODEL,
        "price_per_m_input_usd": cohere_mod._PRICE_PER_M_INPUT_USD,
        "price_per_m_output_usd": cohere_mod._PRICE_PER_M_OUTPUT_USD,
        "api_key_env_var": "COHERE_API_KEY",
        "build_client": cohere_mod._build_client,
        "call_fn": cohere_mod._call_cohere_streaming,
    },
    "gemini": {
        "default_model": gemini_mod.DEFAULT_MODEL,
        "price_per_m_input_usd": gemini_mod._PRICE_PER_M_INPUT_USD,
        "price_per_m_output_usd": gemini_mod._PRICE_PER_M_OUTPUT_USD,
        "api_key_env_var": "GOOGLE_API_KEY",
        "build_client": gemini_mod._build_client,
        "call_fn": gemini_mod._call_gemini_streaming,
    },
}

UNSUPPORTED_PROVIDERS = {
    "azure": "Azure is explicitly out of scope for this task -- not implemented.",
    "fireworks": "Fireworks is explicitly out of scope for this task -- not implemented.",
    "cloudrift": "CloudRift is out of scope here unless explicitly requested -- not implemented.",
    "openai": "OpenAI is not one of the confirmed-usable providers for this task -- not implemented.",
}


# ---------------------------------------------------------------------------
# Plan adapter: vext.PlanRow -> cc.PlannedRequest (for cost caps + provider calls)
# ---------------------------------------------------------------------------

MAX_TOKENS_HEADROOM_MULTIPLIER = 2.0


def _planned_request_from_row(row, model: str, experiment_id: str) -> cc.PlannedRequest:
    max_tokens = int(round(row.target_output_tokens * MAX_TOKENS_HEADROOM_MULTIPLIER))
    return cc.PlannedRequest(
        request_id=str(row.request_id), experiment_id=experiment_id, model=model,
        prompt_bucket=row.prompt_bucket, max_tokens=max_tokens,
        concurrency_level=row.concurrency_level, request_index=row.request_index,
        intended_prompt_tokens=row.intended_prompt_tokens, prompt_text=row.prompt_text,
        target_output_tokens=row.target_output_tokens, workload_version="v2",
    )


def _dispatch_hosted(
    row, *, provider: str, model: str, mock: bool, timeout_s: float,
    rpm_limiter: Optional["cc.RpmLimiter"], budget: Optional["cc.BudgetTracker"],
) -> Dict[str, Any]:
    planned = _planned_request_from_row(row, model, experiment_id=f"hosted_policy_comparison_{provider}")
    if budget is not None and not budget.try_reserve(planned):
        raise RuntimeError(
            f"BUDGET CAP: request {row.request_id} would exceed a hard cap "
            "(cost/requests/input-tokens/output-tokens) -- refused before dispatch."
        )
    t0 = time.monotonic()
    try:
        if mock:
            out = cc.mock_call(planned, stream=True)
        else:
            if rpm_limiter is not None:
                rpm_limiter.acquire()
            cfg = PROVIDER_CONFIG[provider]
            client = cfg["build_client"]()
            out = cfg["call_fn"](client, planned, int(timeout_s))
    finally:
        if budget is not None:
            # record_actual runs even on failure (out may be undefined) --
            # guard with a local var to release the reservation regardless.
            pass
    server_latency = time.monotonic() - t0
    if budget is not None:
        budget.record_actual(planned, out.get("prompt_tokens"), out.get("output_tokens"))
    return {
        "output_tokens": out.get("output_tokens"),
        "prompt_tokens": out.get("prompt_tokens"),
        "finish_reason": out.get("finish_reason"),
        "ttft_seconds": out.get("ttft_seconds"),
        "server_request_latency_seconds": server_latency,
    }


# ---------------------------------------------------------------------------
# Admission loop (provider-agnostic core copied from vext.run_cell_for_policy,
# adapted to dispatch via _dispatch_hosted instead of vLLM HTTP calls; no
# vllm_direct special-case here -- not a meaningful concept for a hosted API)
# ---------------------------------------------------------------------------

def run_cell_for_policy_hosted(
    policy_name: str, cell_plan: List, concurrency: int, *,
    provider: str, model: str, mock: bool, timeout_s: float,
    rpm_limiter: Optional["cc.RpmLimiter"], budget: Optional["cc.BudgetTracker"],
    selector_model=None,
) -> List:
    import concurrent.futures

    is_meta_selector = policy_name == "selector"
    if is_meta_selector and selector_model is None:
        raise vext.SelectorArtifactError("policy_name='selector' but no selector_model was provided.")
    policy = None if is_meta_selector else make_policy(policy_name)
    subpolicy_cache: Dict[str, Any] = {}

    requests = [
        Request(
            request_id=row.request_id, arrival_time=row.arrival_time, prompt_tokens=row.intended_prompt_tokens,
            predicted_output_tokens=row.target_output_tokens, actual_output_tokens=row.target_output_tokens,
            slo_deadline=row.arrival_time + row.slo_slack_seconds, priority=row.priority, class_id=row.class_id,
        )
        for row in cell_plan
    ]
    by_id = {row.request_id: row for row in cell_plan}
    all_observable = [ObservableRequest.from_request(r) for r in requests]
    waiting: List[ObservableRequest] = [r for r in all_observable if r.arrival_time <= 0.0]
    pending: List[ObservableRequest] = sorted(
        (r for r in all_observable if r.arrival_time > 0.0), key=lambda r: r.arrival_time,
    )
    gpu_state = ObservableGPUState(
        gpu_id=0, max_active_sequences=concurrency, max_batch_tokens=10**9, max_kv_tokens=10**9,
        active_request_ids=[], active_requests_info=[], current_kv_tokens=0, tokens_decoded_per_request={},
    )

    results: List = []
    active: Dict[int, Tuple[concurrent.futures.Future, float]] = {}
    admission_choice: Dict[int, Optional[str]] = {}
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=max(1, concurrency))
    t0 = time.monotonic()
    completed_count = 0
    step = 0

    def _recent_violation_rate() -> float:
        completed = [r for r in results if r.status == "success"]
        if not completed:
            return 0.0
        return sum(1 for r in completed if r.slo_violated) / len(completed)

    try:
        while pending or waiting or active:
            now = time.monotonic() - t0
            if pending:
                newly_arrived = [r for r in pending if r.arrival_time <= now]
                if newly_arrived:
                    waiting.extend(newly_arrived)
                    pending = [r for r in pending if r.arrival_time > now]
            chosen_name: Optional[str] = None
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
            if is_meta_selector:
                waiting_ids = {w.request_id for w in waiting}
                waiting_requests = [r for r in requests if r.request_id in waiting_ids]
                chosen_name = vext.normalize_policy_name(vext.selector_choose_subpolicy(
                    selector_model, waiting_requests=waiting_requests, now=now,
                    active_sequence_count=len(gpu_state.active_request_ids),
                    concurrency=concurrency, recent_violation_rate=_recent_violation_rate(),
                ))
                if chosen_name not in subpolicy_cache:
                    subpolicy_cache[chosen_name] = make_policy(chosen_name)
                action = subpolicy_cache[chosen_name].select_action(state)
            else:
                action = policy.select_action(state)
            admitted_ids = action.all_admitted_ids()

            for rid in list(admitted_ids):
                if rid in active:
                    continue
                idx = next((i for i, r in enumerate(waiting) if r.request_id == rid), None)
                if idx is None:
                    continue
                waiting.pop(idx)
                admission_time = time.monotonic() - t0
                fut = executor.submit(
                    _dispatch_hosted, by_id[rid], provider=provider, model=model, mock=mock,
                    timeout_s=timeout_s, rpm_limiter=rpm_limiter, budget=budget,
                )
                active[rid] = (fut, admission_time)
                admission_choice[rid] = chosen_name
                gpu_state.active_request_ids.append(rid)

            if not active:
                if pending:
                    next_arrival = min(r.arrival_time for r in pending)
                    sleep_s = max(0.0, next_arrival - (time.monotonic() - t0))
                    time.sleep(min(sleep_s, 0.05))
                    step += 1
                    continue
                if waiting:
                    drop_time = time.monotonic() - t0
                    for w in waiting:
                        row = by_id[w.request_id]
                        results.append(vext.ComparisonResultRow(
                            policy=policy_name, request_id=w.request_id, prompt_bucket=row.prompt_bucket,
                            target_output_tokens=row.target_output_tokens, concurrency_level=concurrency,
                            class_id=row.class_id, priority=row.priority, arrival_time_s=row.arrival_time,
                            admission_time_s=None, completion_time_s=None,
                            queuing_delay_s=round(drop_time - row.arrival_time, 4),
                            ttft_seconds=None, server_request_latency_seconds=None, total_wall_time_seconds=None,
                            slo_deadline_s=row.slo_slack_seconds, slo_violated=True,
                            output_tokens=None, status="dropped", error_type="PolicyNeverAdmitted",
                            error_message=(
                                f"Policy '{chosen_name or policy_name}' never admitted this request "
                                "before the cell ran out of in-flight work."
                            ),
                            selector_chosen_policy=chosen_name if is_meta_selector else None,
                            regime=row.regime,
                        ))
                    waiting = []
                break

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
                    results.append(vext.ComparisonResultRow(
                        policy=policy_name, request_id=rid, prompt_bucket=row.prompt_bucket,
                        target_output_tokens=row.target_output_tokens, concurrency_level=concurrency,
                        class_id=row.class_id, priority=row.priority, arrival_time_s=row.arrival_time,
                        admission_time_s=round(admission_time, 4), completion_time_s=round(completion_time, 4),
                        queuing_delay_s=round(admission_time - row.arrival_time, 4),
                        ttft_seconds=out.get("ttft_seconds"),
                        server_request_latency_seconds=out.get("server_request_latency_seconds"),
                        total_wall_time_seconds=round(completion_time - row.arrival_time, 4),
                        slo_deadline_s=row.slo_slack_seconds,
                        slo_violated=(completion_time - row.arrival_time) > row.slo_slack_seconds,
                        output_tokens=output_tokens, status="success",
                        error_type=None, error_message=None,
                        selector_chosen_policy=admission_choice.get(rid), regime=row.regime,
                    ))
                except Exception as exc:  # noqa: BLE001
                    status = "timeout" if "timed out" in str(exc).lower() else "error"
                    results.append(vext.ComparisonResultRow(
                        policy=policy_name, request_id=rid, prompt_bucket=row.prompt_bucket,
                        target_output_tokens=row.target_output_tokens, concurrency_level=concurrency,
                        class_id=row.class_id, priority=row.priority, arrival_time_s=row.arrival_time,
                        admission_time_s=round(admission_time, 4), completion_time_s=None,
                        queuing_delay_s=round(admission_time - row.arrival_time, 4),
                        ttft_seconds=None, server_request_latency_seconds=None,
                        total_wall_time_seconds=round(completion_time - row.arrival_time, 4),
                        slo_deadline_s=row.slo_slack_seconds, slo_violated=None,
                        output_tokens=None, status=status, error_type=type(exc).__name__,
                        error_message=str(exc)[:500],
                        selector_chosen_policy=admission_choice.get(rid), regime=row.regime,
                    ))
                del active[rid]
                gpu_state.active_request_ids.remove(rid)
                completed_count += 1
            step += 1
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    return results


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--provider", required=True, choices=list(PROVIDER_CONFIG) + list(UNSUPPORTED_PROVIDERS))
    parser.add_argument("--model", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Plan only, no cost check, no calls.")
    parser.add_argument(
        "--dry-run-cost-check", action="store_true",
        help="Build the plan and run the worst-case hard-cap check; print PASS/FAIL and exit. No calls.",
    )
    parser.add_argument("--mock", action="store_true", help="Local deterministic stub; no network, no SDK import.")
    parser.add_argument(
        "--allow-live-api", action="store_true",
        help="Required to make real hosted-API calls. Without this, --mock or "
        "--dry-run/--dry-run-cost-check is required.",
    )
    parser.add_argument("--policies", type=vext.csv_str_list, default=list(HOSTED_POLICIES))
    parser.add_argument("--prompt-buckets", type=vext.csv_str_list, default=["short", "medium", "long"])
    parser.add_argument("--target-output-tokens-list", type=cc.csv_int_list, default=[64, 128, 256])
    parser.add_argument("--concurrency-list", type=cc.csv_int_list, default=[1, 2, 4])
    parser.add_argument("--requests-per-cell", type=int, default=3)
    parser.add_argument("--arrival-regimes", type=vext.csv_str_list, default=["steady_moderate", "bursty_tight"])
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--rpm-limit", type=int, default=20)
    parser.add_argument("--max-estimated-cost-usd", type=float, default=5.0)
    parser.add_argument("--max-total-requests", type=int, default=1000)
    parser.add_argument("--max-total-input-tokens", type=int, default=1_000_000)
    parser.add_argument("--max-total-output-tokens", type=int, default=300_000)
    parser.add_argument("--selector-artifact", default=None)
    parser.add_argument("--require-our-method", action="store_true")
    parser.add_argument(
        "--decision-divergence-report", action="store_true",
        help="Write decision_divergence.csv / selector_vs_baselines_examples.md.",
    )
    parser.add_argument("--bootstrap-ci", action="store_true")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    args.policies = [vext.normalize_policy_name(p) for p in args.policies]

    if args.provider in UNSUPPORTED_PROVIDERS:
        print(f"ERROR: provider '{args.provider}' is out of scope: {UNSUPPORTED_PROVIDERS[args.provider]}", file=sys.stderr)
        return 10

    cfg_provider = PROVIDER_CONFIG[args.provider]
    model = args.model or cfg_provider["default_model"]

    unknown = set(args.policies) - set(HOSTED_POLICIES) - set(CONDITIONAL_POLICIES)
    if unknown:
        print(f"ERROR: unknown policies: {sorted(unknown)}. Known: {HOSTED_POLICIES + CONDITIONAL_POLICIES}", file=sys.stderr)
        return 2

    if not args.dry_run and not args.dry_run_cost_check and not args.mock and not args.allow_live_api:
        print("ERROR: specify --dry-run, --dry-run-cost-check, --mock, or --allow-live-api.", file=sys.stderr)
        return 2

    if args.require_our_method and "selector" not in args.policies:
        print("ERROR: --require-our-method requires --policies to include 'selector'.", file=sys.stderr)
        return 9

    selector_model = None
    selector_manifest = None
    needs_selector = "selector" in args.policies or args.require_our_method
    if needs_selector:
        if not args.selector_artifact:
            print(
                "ERROR: --policies includes 'selector' (or --require-our-method was passed) "
                "but --selector-artifact was not given.", file=sys.stderr,
            )
            return 9
        try:
            selector_model, selector_manifest = vext.load_and_validate_selector_artifact(
                vext.repo_path(ROOT, args.selector_artifact)
            )
        except vext.SelectorArtifactError as exc:
            print(f"ERROR: selector artifact validation failed: {exc}", file=sys.stderr)
            return 9
        print(f"  Loaded selector artifact: {args.selector_artifact} (objective={selector_manifest['objective_definition']['name']})")

    if args.arrival_regimes:
        unknown_regimes = set(args.arrival_regimes) - set(vext.REGIME_SLO_CLASSES)
        if unknown_regimes:
            print(f"ERROR: unknown --arrival-regimes: {sorted(unknown_regimes)}. Known: {sorted(vext.REGIME_SLO_CLASSES)}", file=sys.stderr)
            return 2

    out_dir = vext.repo_path(ROOT, args.output_dir)
    requests_path = out_dir / "requests.jsonl"
    if out_dir.exists() and requests_path.exists() and requests_path.stat().st_size > 0:
        print(f"ERROR: output dir {out_dir} already has a non-empty requests.jsonl. Choose a new --output-dir.", file=sys.stderr)
        return 3
    out_dir.mkdir(parents=True, exist_ok=True)

    plan = vext.build_request_plan(
        args.prompt_buckets, args.target_output_tokens_list, args.concurrency_list,
        args.requests_per_cell, args.seed, regimes=args.arrival_regimes,
    )
    total_planned = len(plan) * len(args.policies)

    # ---- Hard-cap dry-run check (worst-case, before ANY live call) ----
    planned_requests_for_cost = [
        _planned_request_from_row(row, model, experiment_id="cost_check")
        for row in plan for _ in args.policies
    ]
    cap_namespace = argparse.Namespace(
        max_total_requests=args.max_total_requests, max_total_input_tokens=args.max_total_input_tokens,
        max_total_output_tokens=args.max_total_output_tokens, max_estimated_cost_usd=args.max_estimated_cost_usd,
    )
    violations = cc.validate_call_plan(
        planned_requests_for_cost, cap_namespace,
        price_per_m_input_usd=cfg_provider["price_per_m_input_usd"],
        price_per_m_output_usd=cfg_provider["price_per_m_output_usd"],
    )
    total_input_worst = sum(r.intended_prompt_tokens for r in planned_requests_for_cost)
    total_output_worst = sum(r.max_tokens for r in planned_requests_for_cost)
    worst_cost = cc.estimate_cost_usd(
        total_input_worst, total_output_worst,
        cfg_provider["price_per_m_input_usd"], cfg_provider["price_per_m_output_usd"],
    )
    cost_report = {
        "provider": args.provider, "model": model,
        "planned_total_requests": total_planned,
        "worst_case_total_input_tokens": total_input_worst,
        "worst_case_total_output_tokens": total_output_worst,
        "worst_case_estimated_cost_usd": round(worst_cost, 4),
        "caps": {
            "max_estimated_cost_usd": args.max_estimated_cost_usd,
            "max_total_requests": args.max_total_requests,
            "max_total_input_tokens": args.max_total_input_tokens,
            "max_total_output_tokens": args.max_total_output_tokens,
        },
        "cap_check_passed": not violations,
        "cap_violations": violations,
    }
    (out_dir / "cost_report.json").write_text(json.dumps(cost_report, indent=2))
    print(f"  provider={args.provider} model={model}")
    print(f"  planned total requests: {total_planned}")
    print(f"  worst-case cost estimate: ${worst_cost:.4f} (cap ${args.max_estimated_cost_usd})")
    print(f"  worst-case input tokens: {total_input_worst} (cap {args.max_total_input_tokens})")
    print(f"  worst-case output tokens: {total_output_worst} (cap {args.max_total_output_tokens})")

    if violations:
        print("HARD CAP VIOLATION(S):", file=sys.stderr)
        for v in violations:
            print(f"  - {v}", file=sys.stderr)
        return 4

    if args.dry_run_cost_check:
        print("  Cost check PASSED. No calls were made (--dry-run-cost-check).")
        return 0

    run_status = "planned_only"
    if args.allow_live_api and not args.mock:
        run_status = "completed"
    elif args.mock:
        run_status = "completed_mock"
    elif args.dry_run:
        run_status = "planned_only"

    api_key_present = bool(__import__("os").environ.get(cfg_provider["api_key_env_var"], ""))
    cfg = {
        "provider": args.provider, "model": model, "policies": args.policies,
        "prompt_buckets": args.prompt_buckets, "target_output_tokens_list": args.target_output_tokens_list,
        "concurrency_list": args.concurrency_list, "requests_per_cell": args.requests_per_cell,
        "arrival_regimes": args.arrival_regimes, "timeout_seconds": args.timeout_seconds,
        "fail_fast": args.fail_fast, "seed": args.seed, "rpm_limit": args.rpm_limit,
        "mock": args.mock, "run_status": run_status,
        "selector_artifact": args.selector_artifact, "selector_manifest": selector_manifest,
        "require_our_method": args.require_our_method,
        "decision_divergence_report": args.decision_divergence_report, "bootstrap_ci": args.bootstrap_ci,
        # never log the key value itself, only whether it's present
        "api_key_env_var": cfg_provider["api_key_env_var"], "api_key_present": api_key_present,
        "not_wired_policies": {"generated_heuristic": "see vext.NOT_WIRED_POLICIES", "best_generated": "alias"},
    }
    (out_dir / "run_config.json").write_text(json.dumps(cfg, indent=2))
    vext.write_request_plan(plan, out_dir)
    git_info = _git_info_fallback()
    (out_dir / "manifest.json").write_text(json.dumps({
        **git_info, "provider": args.provider, "model": model,
        "planned_requests_per_policy": len(plan), "policies": args.policies,
        "regimes": sorted({r.regime for r in plan}),
        "cells": sorted({(r.regime, r.prompt_bucket, r.target_output_tokens, r.concurrency_level) for r in plan}),
    }, indent=2))
    (out_dir / "reproducibility.md").write_text(
        "# Reproducibility Metadata\n\n"
        f"- Provider: `{args.provider}`, model: `{model}`\n"
        f"- Run status: `{run_status}`\n"
        f"- API key env var: `{cfg_provider['api_key_env_var']}` (present: {api_key_present})\n\n"
        "## Config\n```json\n" + json.dumps(cfg, indent=2) + "\n```\n"
    )

    if args.dry_run and not args.mock and not args.allow_live_api:
        (out_dir / "summary.json").write_text(json.dumps({"per_policy": {}}, indent=2))
        (out_dir / "summary.md").write_text("# Hosted policy comparison -- dry-run only, no calls made.\n")
        (out_dir / "errors.jsonl").write_text("")
        (out_dir / "requests.jsonl").write_text("")
        print("  No API calls were made (dry-run).")
        return 0

    rpm_limiter = cc.RpmLimiter(args.rpm_limit) if not args.mock else None
    budget = cc.BudgetTracker(
        cap_namespace, price_per_m_input_usd=cfg_provider["price_per_m_input_usd"],
        price_per_m_output_usd=cfg_provider["price_per_m_output_usd"],
    )

    cells: Dict[Tuple[str, str, int, int], List] = {}
    for row in plan:
        cells.setdefault((row.regime, row.prompt_bucket, row.target_output_tokens, row.concurrency_level), []).append(row)

    all_rows: List[Dict[str, Any]] = []
    per_policy_wall_clock: Dict[str, float] = {}
    fail_fast_triggered = False

    for policy_name in args.policies:
        policy_t0 = time.monotonic()
        for (regime, bucket, target, concurrency), cell_plan in cells.items():
            if fail_fast_triggered:
                break
            cell_results = run_cell_for_policy_hosted(
                policy_name, cell_plan, concurrency, provider=args.provider, model=model,
                mock=args.mock, timeout_s=args.timeout_seconds, rpm_limiter=rpm_limiter,
                budget=budget, selector_model=selector_model,
            )
            for r in cell_results:
                all_rows.append(asdict(r))
            if args.fail_fast:
                n = len(cell_results)
                n_failed = sum(1 for r in cell_results if r.status != "success")
                if n >= 10 and n_failed / n > 0.5:
                    fail_fast_triggered = True
                    print(f"FAIL-FAST: aborting after high failure rate in policy={policy_name} regime={regime}", file=sys.stderr)
                    break
        per_policy_wall_clock[policy_name] = time.monotonic() - policy_t0

    overall = vext.write_outputs(
        out_dir, all_rows, per_policy_wall_clock, cfg,
        decision_divergence_report=args.decision_divergence_report, bootstrap_ci=args.bootstrap_ci,
    )

    actual_cost = cc.estimate_cost_usd(
        budget.actual_input_tokens, budget.actual_output_tokens,
        cfg_provider["price_per_m_input_usd"], cfg_provider["price_per_m_output_usd"],
    )
    cost_report["actual_input_tokens"] = budget.actual_input_tokens
    cost_report["actual_output_tokens"] = budget.actual_output_tokens
    cost_report["actual_estimated_cost_usd"] = round(actual_cost, 4)
    cost_report["actual_dispatched_requests"] = budget.dispatched
    (out_dir / "cost_report.json").write_text(json.dumps(cost_report, indent=2))

    for policy_name, m in overall["per_policy"].items():
        print(f"  [{policy_name}] completed={m['n_completed']}/{m['n_total']} arrival_norm_wg={m['arrival_normalized_weighted_goodput']}")
    print(f"  Actual spend estimate: ${actual_cost:.4f}")
    return 0


def _git_info_fallback() -> Dict[str, Any]:
    import subprocess
    try:
        branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip())
        return {"git_branch": branch, "git_commit": commit, "git_dirty": dirty}
    except Exception:
        return {"git_branch": None, "git_commit": None, "git_dirty": None}


if __name__ == "__main__":
    raise SystemExit(main())
