#!/usr/bin/env python3
"""
Fit a simple, interpretable TTFT/latency calibration model from one or more
completed real-LLM pilot directories (e.g. the Cohere and Gemini pilots in
experiments/real_llm/).

This does NOT make any API calls — it only reads each directory's existing
requests.jsonl. The goal is a reproducible calibration baseline that can
later inform the simulator's service-time assumptions, not a polished
predictive model.

Models fit per provider (and optionally pooled with a provider indicator):

- TTFT model: linear regression
      ttft_seconds ~ intercept + prompt_tokens + output_tokens + concurrency
- Latency model, one of two forms (--latency-model-form):
      ttft_plus_decode (default):
          latency_seconds ~ ttft_seconds + decode_intercept
                             + output_tokens / effective_decode_rate
          (effective_decode_rate fit by regressing (latency - ttft) on
          output_tokens)
      linear:
          latency_seconds ~ intercept + prompt_tokens + output_tokens
                             + concurrency

Robust summary statistics (p50/p95/p99, not just mean) are reported
alongside every fit.

Usage:
    python scripts/fit_real_llm_latency_model.py \\
        --experiment-dir experiments/real_llm/cohere_pilot_20260703T040421Z \\
        --experiment-dir experiments/real_llm/gemini_pilot_20260703T044905Z \\
        --exclude-rpm-wait-outliers \\
        --output-dir experiments/real_llm/latency_model_fit
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from llmserveopt.real_llm import calibration_common as cc  # noqa: E402


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-dir", dest="experiment_dirs", action="append", required=True,
        help="Pilot output directory containing requests.jsonl. Repeatable.",
    )
    parser.add_argument(
        "--label", dest="labels", action="append", default=None,
        help="Optional 'provider:model' override for the preceding "
        "--experiment-dir, in order given. If omitted for a directory, "
        "provider/model are read from that directory's run_config.json.",
    )
    parser.add_argument(
        "--exclude-rpm-wait-outliers", action="store_true",
        help="Drop requests heuristically flagged as local-RPM-wait-"
        "polluted (see calibration_common.flag_likely_rate_limiter_wait_"
        "outliers) before fitting. Recommended for pre-fix legacy logs.",
    )
    parser.add_argument(
        "--targets", choices=["ttft", "latency", "both"], default="both",
        help="Which model(s) to fit.",
    )
    parser.add_argument(
        "--latency-model-form", choices=["ttft_plus_decode", "linear"],
        default="ttft_plus_decode",
    )
    parser.add_argument("--pooled", dest="pooled", action="store_true", default=True)
    parser.add_argument("--no-pooled", dest="pooled", action="store_false")
    parser.add_argument("--plots", dest="plots", action="store_true", default=True)
    parser.add_argument("--no-plots", dest="plots", action="store_false")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _guess_provider_from_dir_name(exp_dir: Path) -> str:
    # Fallback for legacy run_config.json files predating the shared
    # calibration_common schema, which didn't record a "provider" key
    # (e.g. the original Cohere pilot, run before the refactor). Directory
    # names follow "<provider>_pilot_<timestamp>".
    name = exp_dir.name
    return name.split("_pilot")[0] if "_pilot" in name else name


def _resolve_label(exp_dir: Path, label: Optional[str]) -> tuple:
    if label:
        provider, _, model = label.partition(":")
        return provider or exp_dir.name, model or "unknown"
    cfg_path = exp_dir / "run_config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        provider = cfg.get("provider") or _guess_provider_from_dir_name(exp_dir)
        return provider, cfg.get("model", "unknown")
    return _guess_provider_from_dir_name(exp_dir), "unknown"


def load_dataset(
    experiment_dirs: List[str],
    labels: Optional[List[str]],
    *,
    root: Path,
    exclude_rpm_wait_outliers: bool,
) -> List[Dict[str, Any]]:
    labels = labels or [None] * len(experiment_dirs)
    if len(labels) < len(experiment_dirs):
        labels = labels + [None] * (len(experiment_dirs) - len(labels))

    records: List[Dict[str, Any]] = []
    for raw_dir, label in zip(experiment_dirs, labels):
        exp_dir = Path(raw_dir)
        if not exp_dir.is_absolute():
            exp_dir = root / exp_dir
        requests_path = exp_dir / "requests.jsonl"
        if not requests_path.exists():
            print(f"WARNING: {requests_path} does not exist, skipping.", file=sys.stderr)
            continue
        provider, model = _resolve_label(exp_dir, label)
        rows = list(cc.load_completed_request_ids(requests_path).values())

        flagged_ids = set()
        if exclude_rpm_wait_outliers:
            flagged_ids = set(cc.flag_likely_rate_limiter_wait_outliers(rows))

        for r in rows:
            if r.get("status") != "success":
                continue
            if r["request_id"] in flagged_ids:
                continue
            latency = cc.legacy_latency_seconds(r)
            ttft = r.get("ttft_seconds")
            output_tokens = r.get("output_tokens")
            prompt_tokens = r.get("actual_prompt_tokens") or r.get("intended_prompt_tokens")
            concurrency = r.get("concurrency_level")
            if prompt_tokens is None or concurrency is None:
                continue
            target_output_tokens = r.get("target_output_tokens")
            records.append({
                "provider": provider,
                "model": model,
                "experiment_dir": str(exp_dir),
                "request_id": r["request_id"],
                "prompt_bucket": r.get("prompt_bucket"),
                "prompt_tokens": float(prompt_tokens),
                "max_tokens": r.get("max_tokens"),
                "concurrency_level": float(concurrency),
                "output_tokens": float(output_tokens) if output_tokens is not None else None,
                "ttft_seconds": float(ttft) if ttft is not None else None,
                "latency_seconds": float(latency) if latency is not None else None,
                "rate_limiter_wait_seconds": r.get("rate_limiter_wait_seconds"),
                "total_wall_time_seconds": r.get("total_wall_time_seconds"),
                "was_rpm_wait_flagged": r["request_id"] in flagged_ids,
                # v2 length-targeted workload fields (None for v1/legacy rows).
                "target_output_tokens": float(target_output_tokens) if target_output_tokens is not None else None,
                "workload_version": r.get("workload_version"),
                "reached_target_output_range": r.get("reached_target_output_range"),
            })
    return records


# ---------------------------------------------------------------------------
# Stats and regression helpers
# ---------------------------------------------------------------------------

def _percentile(values: List[float], p: float) -> Optional[float]:
    return cc._percentile(sorted(values), p) if values else None


def robust_stats(values: List[float]) -> Dict[str, Optional[float]]:
    values = [v for v in values if v is not None]
    if not values:
        return {"n": 0, "mean": None, "p50": None, "p95": None, "p99": None, "min": None, "max": None}
    return {
        "n": len(values),
        "mean": sum(values) / len(values),
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "min": min(values),
        "max": max(values),
    }


def fit_ols(feature_rows: List[List[float]], y: List[float], feature_names: List[str]) -> Dict[str, Any]:
    """Simple OLS via numpy lstsq. Returns coefficients keyed by name plus
    intercept and R^2. Returns None if there are too few rows to fit."""
    import numpy as np

    n = len(y)
    if n < len(feature_names) + 2:
        return {"fit": False, "reason": f"too few rows ({n}) to fit {len(feature_names)} features"}

    X = np.array(feature_rows, dtype=float)
    y_arr = np.array(y, dtype=float)
    X1 = np.column_stack([np.ones(n), X])
    coef, _, _, _ = np.linalg.lstsq(X1, y_arr, rcond=None)
    y_hat = X1 @ coef
    ss_res = float(np.sum((y_arr - y_hat) ** 2))
    ss_tot = float(np.sum((y_arr - y_arr.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None

    result = {"fit": True, "n": n, "intercept": float(coef[0]), "r2": r2}
    for name, c in zip(feature_names, coef[1:]):
        result[f"coef_{name}"] = float(c)
    return result


def fit_ttft_model(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    rows = [r for r in records if r["ttft_seconds"] is not None and r["output_tokens"] is not None]
    features = [[r["prompt_tokens"], r["output_tokens"], r["concurrency_level"]] for r in rows]
    y = [r["ttft_seconds"] for r in rows]
    return fit_ols(features, y, ["prompt_tokens", "output_tokens", "concurrency_level"])


def fit_latency_model_linear(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    rows = [r for r in records if r["latency_seconds"] is not None and r["output_tokens"] is not None]
    features = [[r["prompt_tokens"], r["output_tokens"], r["concurrency_level"]] for r in rows]
    y = [r["latency_seconds"] for r in rows]
    fit = fit_ols(features, y, ["prompt_tokens", "output_tokens", "concurrency_level"])
    fit["form"] = "linear"
    return fit


def fit_latency_model_ttft_plus_decode(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """decode_seconds = latency - ttft ~ intercept + output_tokens / rate.
    Fit decode_seconds ~ a * output_tokens + b, so effective_decode_rate =
    1 / a (tokens/sec) and decode_intercept_s = b (fixed per-request
    overhead after first token, e.g. stream teardown)."""
    rows = [
        r for r in records
        if r["latency_seconds"] is not None and r["ttft_seconds"] is not None
        and r["output_tokens"] is not None
    ]
    decode_seconds = [r["latency_seconds"] - r["ttft_seconds"] for r in rows]
    features = [[r["output_tokens"]] for r in rows]
    fit = fit_ols(features, decode_seconds, ["output_tokens"])
    fit["form"] = "ttft_plus_decode"
    if fit.get("fit"):
        slope = fit["coef_output_tokens"]
        fit["decode_intercept_s"] = fit["intercept"]
        fit["effective_decode_rate_tokens_per_sec"] = (1.0 / slope) if slope > 0 else None
    return fit


def fit_latency_model(records: List[Dict[str, Any]], form: str) -> Dict[str, Any]:
    if form == "linear":
        return fit_latency_model_linear(records)
    return fit_latency_model_ttft_plus_decode(records)


# ---------------------------------------------------------------------------
# Per-provider report assembly
# ---------------------------------------------------------------------------

def build_provider_report(
    provider: str, records: List[Dict[str, Any]], *, targets: str, latency_form: str,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "provider": provider,
        "model": records[0]["model"] if records else None,
        "n_records": len(records),
        "n_rpm_wait_flagged_excluded": sum(1 for r in records if r["was_rpm_wait_flagged"]),
    }
    ttft_values = [r["ttft_seconds"] for r in records if r["ttft_seconds"] is not None]
    latency_values = [r["latency_seconds"] for r in records if r["latency_seconds"] is not None]
    report["ttft_stats"] = robust_stats(ttft_values)
    report["latency_stats"] = robust_stats(latency_values)

    if targets in ("ttft", "both"):
        report["ttft_model"] = fit_ttft_model(records)
    if targets in ("latency", "both"):
        report["latency_model"] = fit_latency_model(records, latency_form)
    return report


def build_pooled_report(
    records: List[Dict[str, Any]], *, targets: str, latency_form: str,
) -> Dict[str, Any]:
    providers = sorted({r["provider"] for r in records})
    # One-hot provider indicators, dropping the first (alphabetically) as baseline.
    baseline, *others = providers

    def _row_with_indicators(r: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(r)
        for p in others:
            out[f"is_{p}"] = 1.0 if r["provider"] == p else 0.0
        return out

    augmented = [_row_with_indicators(r) for r in records]

    report: Dict[str, Any] = {
        "providers": providers,
        "baseline_provider": baseline,
        "n_records": len(records),
    }
    ttft_values = [r["ttft_seconds"] for r in records if r["ttft_seconds"] is not None]
    latency_values = [r["latency_seconds"] for r in records if r["latency_seconds"] is not None]
    report["ttft_stats"] = robust_stats(ttft_values)
    report["latency_stats"] = robust_stats(latency_values)

    if targets in ("ttft", "both"):
        rows = [r for r in augmented if r["ttft_seconds"] is not None and r["output_tokens"] is not None]
        feature_names = ["prompt_tokens", "output_tokens", "concurrency_level"] + [f"is_{p}" for p in others]
        features = [[r[name] for name in feature_names] for r in rows]
        y = [r["ttft_seconds"] for r in rows]
        report["ttft_model"] = fit_ols(features, y, feature_names)

    if targets in ("latency", "both") and latency_form == "linear":
        rows = [r for r in augmented if r["latency_seconds"] is not None and r["output_tokens"] is not None]
        feature_names = ["prompt_tokens", "output_tokens", "concurrency_level"] + [f"is_{p}" for p in others]
        features = [[r[name] for name in feature_names] for r in rows]
        y = [r["latency_seconds"] for r in rows]
        fit = fit_ols(features, y, feature_names)
        fit["form"] = "linear"
        report["latency_model"] = fit
    elif targets in ("latency", "both"):
        # ttft_plus_decode's decode-rate fit is provider-agnostic by nature
        # (it already conditions on each request's own ttft), so pool it
        # directly rather than adding provider indicators to a 1-feature fit.
        report["latency_model"] = fit_latency_model_ttft_plus_decode(records)

    return report


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_csv(records: List[Dict[str, Any]], out_dir: Path) -> None:
    import pandas as pd

    df = pd.DataFrame(records)
    df.to_csv(out_dir / "latency_model_fit.csv", index=False)


def write_json(payload: Dict[str, Any], out_dir: Path) -> None:
    (out_dir / "latency_model_fit.json").write_text(json.dumps(payload, indent=2))


def _fmt_stats(stats: Dict[str, Optional[float]]) -> str:
    if not stats or stats.get("n") == 0:
        return "(no data)"
    return (
        f"n={stats['n']}, mean={stats['mean']:.4f}, p50={stats['p50']:.4f}, "
        f"p95={stats['p95']:.4f}, p99={stats['p99']:.4f}"
    )


def _fmt_model(model: Dict[str, Any]) -> List[str]:
    if not model.get("fit"):
        return [f"  (not fit: {model.get('reason')})"]
    lines = [f"  n={model['n']}, R^2={model.get('r2'):.4f}" if model.get('r2') is not None else f"  n={model['n']}"]
    lines.append(f"  intercept={model['intercept']:.6f}")
    for key, val in model.items():
        if key.startswith("coef_"):
            lines.append(f"  {key}={val:.6f}")
    if model.get("form") == "ttft_plus_decode":
        rate = model.get("effective_decode_rate_tokens_per_sec")
        lines.append(f"  effective_decode_rate_tokens_per_sec={rate:.4f}" if rate else "  effective_decode_rate_tokens_per_sec=None")
        lines.append(f"  decode_intercept_s={model.get('decode_intercept_s'):.6f}")
    return lines


def write_markdown(payload: Dict[str, Any], out_dir: Path) -> None:
    lines = [
        "# Real-LLM Latency Calibration Model Fit",
        "",
        f"Generated: {payload['generated_at_utc']}",
        f"Inputs: {', '.join(payload['inputs']['experiment_dirs'])}",
        f"RPM-wait outliers excluded: {payload['inputs']['exclude_rpm_wait_outliers']}",
        f"Targets fit: {payload['inputs']['targets']}",
        f"Latency model form: {payload['inputs']['latency_model_form']}",
        "",
        "This is a simple, interpretable OLS calibration baseline, not a",
        "production-quality predictive model. See docs/real_llm_cohere_gemini_"
        "comparison.md for the pilots' full caveats before using these",
        "coefficients to update simulator service-time assumptions.",
        "",
        "**Output-token-scaling caveat:** the source pilots' prompts elicited",
        "~22-35 output tokens regardless of `max_tokens` (see the max_tokens",
        "caveat in docs/real_llm_cohere_gemini_comparison.md), so",
        "`coef_output_tokens` / `effective_decode_rate_tokens_per_sec` below",
        "are fit over a very narrow output-length range and should be treated",
        "as low-confidence until refit on the proposed v2 workload (see",
        "docs/real_llm_v2_workload_proposal.md), which is designed to",
        "actually vary output length.",
        "",
    ]
    for provider, report in payload["providers"].items():
        lines += [
            f"## {provider} ({report.get('model')})",
            "",
            f"- n records used: {report['n_records']} "
            f"(RPM-wait-flagged excluded: {report['n_rpm_wait_flagged_excluded']})",
            f"- TTFT stats (s): {_fmt_stats(report['ttft_stats'])}",
            f"- Latency stats (s): {_fmt_stats(report['latency_stats'])}",
            "",
        ]
        if "ttft_model" in report:
            lines.append("### TTFT model: `ttft ~ intercept + prompt_tokens + output_tokens + concurrency`")
            lines += _fmt_model(report["ttft_model"])
            lines.append("")
        if "latency_model" in report:
            lines.append(f"### Latency model ({report['latency_model'].get('form')})")
            lines += _fmt_model(report["latency_model"])
            lines.append("")

    if "pooled" in payload:
        report = payload["pooled"]
        lines += [
            f"## Pooled model (baseline provider: {report['baseline_provider']})",
            "",
            f"- providers: {', '.join(report['providers'])}",
            f"- n records: {report['n_records']}",
            f"- TTFT stats (s): {_fmt_stats(report['ttft_stats'])}",
            f"- Latency stats (s): {_fmt_stats(report['latency_stats'])}",
            "",
        ]
        if "ttft_model" in report:
            lines.append("### Pooled TTFT model (provider indicator coefficients are the offset vs. baseline)")
            lines += _fmt_model(report["ttft_model"])
            lines.append("")
        if "latency_model" in report:
            lines.append(f"### Pooled latency model ({report['latency_model'].get('form')})")
            lines += _fmt_model(report["latency_model"])
            lines.append("")

    (out_dir / "latency_model_fit.md").write_text("\n".join(lines) + "\n")


def write_plots(records: List[Dict[str, Any]], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    providers = sorted({r["provider"] for r in records})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for provider in providers:
        sub = [r for r in records if r["provider"] == provider]
        ttfts = [r["ttft_seconds"] for r in sub if r["ttft_seconds"] is not None]
        axes[0].hist(ttfts, bins=20, alpha=0.5, label=provider)
    axes[0].set_xlabel("TTFT (s)")
    axes[0].set_ylabel("count")
    axes[0].set_title("TTFT distribution by provider")
    axes[0].legend()

    for provider in providers:
        sub = [r for r in records if r["provider"] == provider]
        xs = [r["output_tokens"] for r in sub if r["output_tokens"] is not None and r["latency_seconds"] is not None]
        ys = [r["latency_seconds"] for r in sub if r["output_tokens"] is not None and r["latency_seconds"] is not None]
        axes[1].scatter(xs, ys, alpha=0.5, label=provider, s=15)
    axes[1].set_xlabel("output tokens")
    axes[1].set_ylabel("latency (s)")
    axes[1].set_title("Latency vs. output tokens by provider")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_dir / "latency_model_fit.png", dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_dataset(
        args.experiment_dirs, args.labels, root=ROOT,
        exclude_rpm_wait_outliers=args.exclude_rpm_wait_outliers,
    )
    if not records:
        print("ERROR: no successful records loaded from any --experiment-dir.", file=sys.stderr)
        return 2

    providers = sorted({r["provider"] for r in records})
    payload: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "experiment_dirs": args.experiment_dirs,
            "labels": args.labels,
            "exclude_rpm_wait_outliers": args.exclude_rpm_wait_outliers,
            "targets": args.targets,
            "latency_model_form": args.latency_model_form,
            "pooled": args.pooled,
        },
        "providers": {},
    }
    for provider in providers:
        sub = [r for r in records if r["provider"] == provider]
        payload["providers"][provider] = build_provider_report(
            provider, sub, targets=args.targets, latency_form=args.latency_model_form,
        )

    if args.pooled and len(providers) > 1:
        payload["pooled"] = build_pooled_report(
            records, targets=args.targets, latency_form=args.latency_model_form,
        )

    write_csv(records, out_dir)
    write_json(payload, out_dir)
    write_markdown(payload, out_dir)
    if args.plots:
        try:
            write_plots(records, out_dir)
        except Exception as exc:  # noqa: BLE001 - plotting is best-effort
            print(f"WARNING: plot generation failed, skipping: {exc}", file=sys.stderr)

    print(f"Fit model(s) from {len(records)} records across {len(providers)} provider(s): {providers}")
    print(f"  wrote {out_dir / 'latency_model_fit.json'}")
    print(f"  wrote {out_dir / 'latency_model_fit.md'}")
    print(f"  wrote {out_dir / 'latency_model_fit.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
