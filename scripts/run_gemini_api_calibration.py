#!/usr/bin/env python3
"""
Phase 2C.4 / Gemini API Calibration: dry-run and live infrastructure.

This script plans and optionally executes a minimal set of Gemini API calls to
calibrate simulator latency assumptions against a real hosted LLM endpoint.

DEFAULT MODE (no flags): refuse to run anything.
--dry-run:       plan calls, write manifest and summary, never contact any API.
--allow-live-api: actually issue calls under hard caps (requires credentials).
--mock:          replace API calls with a local stub (for tests only).

Usage:
    python scripts/run_gemini_api_calibration.py \\
        --config configs/api_calibration/gemini_minimal_v1.yaml \\
        --dry-run

Live (tiny pilot — DO NOT run without reviewing caps first):
    python scripts/run_gemini_api_calibration.py \\
        --config configs/api_calibration/gemini_minimal_v1.yaml \\
        --allow-live-api \\
        --max-calls 10
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: Path) -> dict:
    try:
        import yaml
    except ImportError:
        raise RuntimeError("PyYAML is required: pip install pyyaml")
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Call plan
# ---------------------------------------------------------------------------

@dataclass
class PlannedCall:
    call_id: str
    provider: str
    model: str
    prompt_bucket: str
    output_bucket: str
    concurrency_group: int
    repeat_index: int
    planned_prompt_tokens: int
    max_output_tokens: int
    prompt_text: str

    def to_manifest_dict(self) -> Dict[str, Any]:
        return {
            "call_id": self.call_id,
            "provider": self.provider,
            "model": self.model,
            "prompt_bucket": self.prompt_bucket,
            "output_bucket": self.output_bucket,
            "concurrency_group": self.concurrency_group,
            "repeat_index": self.repeat_index,
            "planned_prompt_tokens": self.planned_prompt_tokens,
            "max_output_tokens": self.max_output_tokens,
        }


def expand_call_plan(cfg: dict) -> List[PlannedCall]:
    provider = str(cfg.get("provider", "gemini_api"))
    model = str(cfg.get("model", "gemini-1.5-flash-latest"))
    prompt_buckets: dict = cfg.get("prompt_buckets", {})
    output_buckets: dict = cfg.get("output_buckets", {})
    repeats: int = int(cfg.get("repeats", 2))
    concurrency_groups: List[int] = [int(c) for c in cfg.get("concurrency_groups", [1])]

    calls: List[PlannedCall] = []
    idx = 0
    for pb_name, ob_name, concurrency, repeat_i in product(
        prompt_buckets.keys(),
        output_buckets.keys(),
        concurrency_groups,
        range(repeats),
    ):
        pb = prompt_buckets[pb_name]
        ob = output_buckets[ob_name]
        calls.append(
            PlannedCall(
                call_id=f"call_{idx:04d}",
                provider=provider,
                model=model,
                prompt_bucket=pb_name,
                output_bucket=ob_name,
                concurrency_group=concurrency,
                repeat_index=repeat_i,
                planned_prompt_tokens=int(pb.get("approx_tokens", 0)),
                max_output_tokens=int(ob.get("max_tokens", 64)),
                prompt_text=str(pb.get("template", "")),
            )
        )
        idx += 1
    return calls


def validate_call_plan(calls: List[PlannedCall], cfg: dict) -> List[str]:
    """Return list of violation messages (empty = OK)."""
    caps: dict = cfg.get("hard_caps", {})
    max_calls = int(caps.get("max_calls", 50))
    max_prompt_per_call = int(caps.get("max_prompt_tokens_per_call", 2048))
    max_output_per_call = int(caps.get("max_output_tokens_per_call", 512))
    max_total_prompt = int(caps.get("max_total_prompt_tokens", 40000))
    max_total_output = int(caps.get("max_total_output_tokens", 10000))

    violations: List[str] = []
    if len(calls) > max_calls:
        violations.append(
            f"Planned {len(calls)} calls exceeds hard cap max_calls={max_calls}"
        )
    for c in calls:
        if c.planned_prompt_tokens > max_prompt_per_call:
            violations.append(
                f"{c.call_id}: planned_prompt_tokens={c.planned_prompt_tokens} "
                f"> max_prompt_tokens_per_call={max_prompt_per_call}"
            )
        if c.max_output_tokens > max_output_per_call:
            violations.append(
                f"{c.call_id}: max_output_tokens={c.max_output_tokens} "
                f"> max_output_tokens_per_call={max_output_per_call}"
            )
    total_prompt = sum(c.planned_prompt_tokens for c in calls)
    total_output = sum(c.max_output_tokens for c in calls)
    if total_prompt > max_total_prompt:
        violations.append(
            f"Total planned prompt tokens {total_prompt} > cap {max_total_prompt}"
        )
    if total_output > max_total_output:
        violations.append(
            f"Total max output tokens {total_output} > cap {max_total_output}"
        )
    return violations


# ---------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------

# Gemini Flash pricing as of 2025-06 (approx, USD per 1M tokens).
# These are only used for dry-run cost estimates; no real billing happens here.
_PRICE_PER_M_INPUT = {"gemini_api": 0.075, "vertex": 0.075}
_PRICE_PER_M_OUTPUT = {"gemini_api": 0.30, "vertex": 0.30}


def estimate_cost_usd(calls: List[PlannedCall], provider: str) -> float:
    price_in = _PRICE_PER_M_INPUT.get(provider, 0.10)
    price_out = _PRICE_PER_M_OUTPUT.get(provider, 0.30)
    total_prompt = sum(c.planned_prompt_tokens for c in calls)
    total_output = sum(c.max_output_tokens for c in calls)
    return (total_prompt / 1_000_000) * price_in + (total_output / 1_000_000) * price_out


# ---------------------------------------------------------------------------
# Dry-run output
# ---------------------------------------------------------------------------

def write_manifest(out_dir: Path, calls: List[PlannedCall], cfg: dict, estimate: dict) -> Path:
    manifest = {
        "experiment": cfg.get("experiment", "unknown"),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "dry_run",
        "provider": cfg.get("provider", "gemini_api"),
        "model": cfg.get("model", "unknown"),
        "planned_calls": len(calls),
        "hard_caps": cfg.get("hard_caps", {}),
        "estimate": estimate,
        "calls": [c.to_manifest_dict() for c in calls],
    }
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path


def write_dry_run_summary(
    out_dir: Path,
    calls: List[PlannedCall],
    cfg: dict,
    estimate: dict,
    live_command: str,
) -> Path:
    caps = cfg.get("hard_caps", {})
    lines = [
        "# Gemini API Calibration — Dry-Run Summary",
        "",
        f"**Experiment:** `{cfg.get('experiment', 'unknown')}`",
        f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
        "## Planned Call Grid",
        f"- Total planned calls: **{len(calls)}** (hard cap: {caps.get('max_calls', 'n/a')})",
        f"- Prompt buckets: {', '.join(sorted({c.prompt_bucket for c in calls}))}",
        f"- Output buckets: {', '.join(sorted({c.output_bucket for c in calls}))}",
        f"- Concurrency groups: {sorted({c.concurrency_group for c in calls})}",
        f"- Repeats per combination: {cfg.get('repeats', 'n/a')}",
        "",
        "## Token Estimates (Worst-Case)",
        f"- Total prompt tokens (approx): **{estimate['total_planned_prompt_tokens']}**",
        f"  (cap: {caps.get('max_total_prompt_tokens', 'n/a')})",
        f"- Total max output tokens: **{estimate['total_max_output_tokens']}**",
        f"  (cap: {caps.get('max_total_output_tokens', 'n/a')})",
        f"- Max prompt tokens per call: {estimate['max_prompt_per_call']}",
        f"  (cap: {caps.get('max_prompt_tokens_per_call', 'n/a')})",
        f"- Max output tokens per call: {estimate['max_output_per_call']}",
        f"  (cap: {caps.get('max_output_tokens_per_call', 'n/a')})",
        "",
        "## Cost Estimate",
        f"- Worst-case estimated cost: **${estimate['estimated_cost_usd']:.5f} USD**",
        f"  (budget cap: ${caps.get('estimated_budget_usd', 'n/a')} USD)",
        "",
        "## Status",
        "- **No API calls were made.** This is a dry-run.",
        "- No credentials were accessed.",
        "- No SDK import was attempted.",
        "",
        "## To Run Live Pilot",
        "```bash",
        live_command,
        "```",
        "",
        "> **Warning:** Review caps in the config before issuing live calls.",
        "> Ensure `GOOGLE_API_KEY` or `GOOGLE_APPLICATION_CREDENTIALS` is set.",
    ]
    path = out_dir / "dry_run_summary.md"
    path.write_text("\n".join(lines) + "\n")
    return path


# ---------------------------------------------------------------------------
# Live mode stub
# ---------------------------------------------------------------------------

@dataclass
class CallResult:
    call_id: str
    provider: str
    model: str
    prompt_bucket: str
    output_bucket: str
    concurrency_group: int
    repeat_index: int
    planned_prompt_tokens: int
    max_output_tokens: int
    status: str
    prompt_tokens_actual: Optional[int]
    output_tokens_actual: Optional[int]
    latency_total_ms: Optional[float]
    time_to_first_token_ms: Optional[float]
    start_utc: str
    end_utc: str
    error_code: Optional[str]
    error_message: Optional[str]


def _mock_call(planned: PlannedCall, cfg: dict) -> CallResult:
    """Substitute for a real API call when --mock is passed (tests only)."""
    start = datetime.now(timezone.utc)
    time.sleep(0.001)
    end = datetime.now(timezone.utc)
    return CallResult(
        call_id=planned.call_id,
        provider=planned.provider,
        model=planned.model,
        prompt_bucket=planned.prompt_bucket,
        output_bucket=planned.output_bucket,
        concurrency_group=planned.concurrency_group,
        repeat_index=planned.repeat_index,
        planned_prompt_tokens=planned.planned_prompt_tokens,
        max_output_tokens=planned.max_output_tokens,
        status="mock_ok",
        prompt_tokens_actual=planned.planned_prompt_tokens,
        output_tokens_actual=min(planned.max_output_tokens, 32),
        latency_total_ms=42.0,
        time_to_first_token_ms=None,
        start_utc=start.isoformat(),
        end_utc=end.isoformat(),
        error_code=None,
        error_message=None,
    )


def _live_call_gemini_api(planned: PlannedCall, cfg: dict) -> CallResult:
    """Issue a real call via google-generativeai SDK."""
    try:
        import google.generativeai as genai  # type: ignore[import]
    except ImportError:
        raise RuntimeError(
            "google-generativeai is not installed. "
            "Install with: pip install google-generativeai"
        )
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY environment variable is not set. "
            "Set it before running live mode."
        )
    genai.configure(api_key=api_key)
    generation_config = genai.GenerationConfig(
        max_output_tokens=planned.max_output_tokens,
    )
    model = genai.GenerativeModel(planned.model, generation_config=generation_config)
    start = datetime.now(timezone.utc)
    start_ns = time.monotonic_ns()
    try:
        response = model.generate_content(planned.prompt_text)
        end_ns = time.monotonic_ns()
        end = datetime.now(timezone.utc)
        latency_ms = (end_ns - start_ns) / 1_000_000
        out_tokens = None
        in_tokens = None
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            in_tokens = getattr(response.usage_metadata, "prompt_token_count", None)
            out_tokens = getattr(response.usage_metadata, "candidates_token_count", None)
        return CallResult(
            call_id=planned.call_id,
            provider=planned.provider,
            model=planned.model,
            prompt_bucket=planned.prompt_bucket,
            output_bucket=planned.output_bucket,
            concurrency_group=planned.concurrency_group,
            repeat_index=planned.repeat_index,
            planned_prompt_tokens=planned.planned_prompt_tokens,
            max_output_tokens=planned.max_output_tokens,
            status="ok",
            prompt_tokens_actual=in_tokens,
            output_tokens_actual=out_tokens,
            latency_total_ms=round(latency_ms, 2),
            time_to_first_token_ms=None,
            start_utc=start.isoformat(),
            end_utc=end.isoformat(),
            error_code=None,
            error_message=None,
        )
    except Exception as exc:
        end = datetime.now(timezone.utc)
        return CallResult(
            call_id=planned.call_id,
            provider=planned.provider,
            model=planned.model,
            prompt_bucket=planned.prompt_bucket,
            output_bucket=planned.output_bucket,
            concurrency_group=planned.concurrency_group,
            repeat_index=planned.repeat_index,
            planned_prompt_tokens=planned.planned_prompt_tokens,
            max_output_tokens=planned.max_output_tokens,
            status="error",
            prompt_tokens_actual=None,
            output_tokens_actual=None,
            latency_total_ms=None,
            time_to_first_token_ms=None,
            start_utc=start.isoformat(),
            end_utc=end.isoformat(),
            error_code=type(exc).__name__,
            error_message=str(exc)[:500],
        )


def _live_call_vertex(planned: PlannedCall, cfg: dict) -> CallResult:
    """Issue a real call via google-cloud-aiplatform SDK (Vertex AI)."""
    try:
        import vertexai  # type: ignore[import]
        from vertexai.generative_models import GenerativeModel, GenerationConfig  # type: ignore[import]
    except ImportError:
        raise RuntimeError(
            "google-cloud-aiplatform is not installed. "
            "Install with: pip install google-cloud-aiplatform"
        )
    project = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
    if not project:
        raise RuntimeError(
            "GOOGLE_CLOUD_PROJECT environment variable is not set."
        )
    vertexai.init(project=project, location=location)
    gen_config = GenerationConfig(max_output_tokens=planned.max_output_tokens)
    model = GenerativeModel(planned.model, generation_config=gen_config)
    start = datetime.now(timezone.utc)
    start_ns = time.monotonic_ns()
    try:
        response = model.generate_content(planned.prompt_text)
        end_ns = time.monotonic_ns()
        end = datetime.now(timezone.utc)
        latency_ms = (end_ns - start_ns) / 1_000_000
        meta = getattr(response, "usage_metadata", None)
        return CallResult(
            call_id=planned.call_id,
            provider=planned.provider,
            model=planned.model,
            prompt_bucket=planned.prompt_bucket,
            output_bucket=planned.output_bucket,
            concurrency_group=planned.concurrency_group,
            repeat_index=planned.repeat_index,
            planned_prompt_tokens=planned.planned_prompt_tokens,
            max_output_tokens=planned.max_output_tokens,
            status="ok",
            prompt_tokens_actual=getattr(meta, "prompt_token_count", None) if meta else None,
            output_tokens_actual=getattr(meta, "candidates_token_count", None) if meta else None,
            latency_total_ms=round(latency_ms, 2),
            time_to_first_token_ms=None,
            start_utc=start.isoformat(),
            end_utc=end.isoformat(),
            error_code=None,
            error_message=None,
        )
    except Exception as exc:
        end = datetime.now(timezone.utc)
        return CallResult(
            call_id=planned.call_id,
            provider=planned.provider,
            model=planned.model,
            prompt_bucket=planned.prompt_bucket,
            output_bucket=planned.output_bucket,
            concurrency_group=planned.concurrency_group,
            repeat_index=planned.repeat_index,
            planned_prompt_tokens=planned.planned_prompt_tokens,
            max_output_tokens=planned.max_output_tokens,
            status="error",
            prompt_tokens_actual=None,
            output_tokens_actual=None,
            latency_total_ms=None,
            time_to_first_token_ms=None,
            start_utc=start.isoformat(),
            end_utc=end.isoformat(),
            error_code=type(exc).__name__,
            error_message=str(exc)[:500],
        )


def dispatch_call(planned: PlannedCall, cfg: dict, *, mock: bool) -> CallResult:
    if mock:
        return _mock_call(planned, cfg)
    provider = planned.provider
    if provider == "gemini_api":
        return _live_call_gemini_api(planned, cfg)
    elif provider == "vertex":
        return _live_call_vertex(planned, cfg)
    else:
        raise ValueError(f"Unknown provider: {provider!r}")


# ---------------------------------------------------------------------------
# Live runner
# ---------------------------------------------------------------------------

def run_live(
    calls: List[PlannedCall],
    cfg: dict,
    out_dir: Path,
    *,
    max_calls_override: Optional[int],
    mock: bool,
) -> List[CallResult]:
    caps = cfg.get("hard_caps", {})
    effective_max = min(
        int(caps.get("max_calls", 50)),
        max_calls_override if max_calls_override is not None else len(calls),
    )
    calls_to_run = calls[:effective_max]
    logging.info("Live mode: running %d calls (cap=%d)", len(calls_to_run), effective_max)

    results_dir = out_dir / "raw_responses"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_log = out_dir / "call_log.jsonl"
    results: List[CallResult] = []

    with open(results_log, "w") as log_fh:
        for planned in calls_to_run:
            logging.info("Dispatching %s (prompt_bucket=%s, output_bucket=%s, concurrency=%d)",
                         planned.call_id, planned.prompt_bucket, planned.output_bucket,
                         planned.concurrency_group)
            result = dispatch_call(planned, cfg, mock=mock)
            results.append(result)
            row = {
                **{k: v for k, v in asdict(result).items() if k != "prompt_text"},
            }
            log_fh.write(json.dumps(row) + "\n")
            log_fh.flush()
            logging.info(
                "%s status=%s latency_ms=%s prompt_tokens=%s output_tokens=%s",
                result.call_id, result.status, result.latency_total_ms,
                result.prompt_tokens_actual, result.output_tokens_actual,
            )
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gemini API calibration — dry-run and live infrastructure."
    )
    parser.add_argument(
        "--config",
        default="configs/api_calibration/gemini_minimal_v1.yaml",
        help="Path to calibration config YAML.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan calls and write manifest without contacting any API.",
    )
    parser.add_argument(
        "--allow-live-api",
        action="store_true",
        help="Actually issue API calls (requires credentials and explicit intent).",
    )
    parser.add_argument(
        "--max-calls",
        type=int,
        default=None,
        help="Override max_calls cap for this run (must be ≤ config hard cap).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory (default: from config, timestamped).",
    )
    parser.add_argument(
        "--provider",
        default=None,
        choices=["gemini_api", "vertex"],
        help="Override provider from config.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Replace live API calls with a local stub (for testing only).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    return parser.parse_args(argv)


def _repo_path(raw: str | Path) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else ROOT / p


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    cfg_path = _repo_path(args.config)
    if not cfg_path.exists():
        print(f"ERROR: config not found: {cfg_path}", file=sys.stderr)
        return 1
    cfg = load_config(cfg_path)

    if args.provider:
        cfg["provider"] = args.provider

    # Refuse to run anything unless exactly one mode is chosen.
    if not args.dry_run and not args.allow_live_api:
        print(
            "ERROR: specify --dry-run or --allow-live-api (or both to dry-run then live).\n"
            "Run with --dry-run first to see the planned call grid.",
            file=sys.stderr,
        )
        return 2

    # Expand the call grid.
    calls = expand_call_plan(cfg)
    violations = validate_call_plan(calls, cfg)
    if violations:
        print("HARD CAP VIOLATIONS — refusing to proceed:", file=sys.stderr)
        for v in violations:
            print(f"  {v}", file=sys.stderr)
        return 3

    # Apply --max-calls override (must not exceed config cap).
    max_calls_cap = int(cfg.get("hard_caps", {}).get("max_calls", 50))
    if args.max_calls is not None:
        if args.max_calls > max_calls_cap:
            print(
                f"ERROR: --max-calls {args.max_calls} exceeds config hard cap {max_calls_cap}.",
                file=sys.stderr,
            )
            return 3
        calls = calls[: args.max_calls]

    # Estimate cost.
    provider = cfg.get("provider", "gemini_api")
    estimate = {
        "total_planned_prompt_tokens": sum(c.planned_prompt_tokens for c in calls),
        "total_max_output_tokens": sum(c.max_output_tokens for c in calls),
        "max_prompt_per_call": max((c.planned_prompt_tokens for c in calls), default=0),
        "max_output_per_call": max((c.max_output_tokens for c in calls), default=0),
        "estimated_cost_usd": round(estimate_cost_usd(calls, provider), 6),
    }

    budget_cap = float(cfg.get("hard_caps", {}).get("estimated_budget_usd", 1.0))
    if estimate["estimated_cost_usd"] > budget_cap:
        print(
            f"ERROR: estimated cost ${estimate['estimated_cost_usd']:.5f} USD "
            f"exceeds budget cap ${budget_cap} USD.",
            file=sys.stderr,
        )
        return 3

    # Determine output directory.
    if args.output_dir:
        out_dir = _repo_path(args.output_dir)
    else:
        base_out = _repo_path(cfg.get("output_dir", "results/api_calibration/gemini_minimal_v1"))
        suffix = "dry_run" if args.dry_run and not args.allow_live_api else "live"
        out_dir = base_out / f"{suffix}_{_timestamp()}"

    out_dir.mkdir(parents=True, exist_ok=True)

    # Build the live command for reference.
    live_command = (
        f"python scripts/run_gemini_api_calibration.py \\\n"
        f"  --config {args.config} \\\n"
        f"  --allow-live-api \\\n"
        f"  --max-calls {min(10, len(calls))}"
    )

    if args.dry_run:
        manifest_path = write_manifest(out_dir, calls, cfg, estimate)
        summary_path = write_dry_run_summary(out_dir, calls, cfg, estimate, live_command)

        print(f"Dry-run complete.")
        print(f"  Planned calls:        {len(calls)}")
        print(f"  Prompt tokens (sum):  {estimate['total_planned_prompt_tokens']}")
        print(f"  Output tokens (max):  {estimate['total_max_output_tokens']}")
        print(f"  Estimated cost:       ${estimate['estimated_cost_usd']:.5f} USD")
        print(f"  Manifest:             {manifest_path}")
        print(f"  Summary:              {summary_path}")
        print(f"")
        print(f"  No API calls were made.")
        print(f"")
        print(f"  To run live pilot:")
        live_command_one_line = live_command.replace(chr(10), " ").replace("\\  ", " ")
        print(f"  {live_command_one_line}")

        if not args.allow_live_api:
            return 0

    if args.allow_live_api:
        logging.info("Live API mode enabled. Provider: %s", provider)
        if not args.mock:
            # Check credentials exist before wasting time on the plan.
            if provider == "gemini_api" and not os.environ.get("GOOGLE_API_KEY"):
                print(
                    "ERROR: GOOGLE_API_KEY is not set. "
                    "Export it before running live mode.",
                    file=sys.stderr,
                )
                return 4
            elif provider == "vertex" and not os.environ.get("GOOGLE_CLOUD_PROJECT"):
                print(
                    "ERROR: GOOGLE_CLOUD_PROJECT is not set. "
                    "Export it before running live mode.",
                    file=sys.stderr,
                )
                return 4

        results = run_live(
            calls, cfg, out_dir,
            max_calls_override=args.max_calls,
            mock=args.mock,
        )
        ok = sum(1 for r in results if r.status in ("ok", "mock_ok"))
        err = len(results) - ok
        logging.info("Live run complete: %d ok, %d errors", ok, err)
        print(f"Live run complete: {ok} ok, {err} errors → {out_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
