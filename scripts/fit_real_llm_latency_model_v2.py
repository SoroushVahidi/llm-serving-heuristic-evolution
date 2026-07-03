#!/usr/bin/env python3
"""
Fit an updated TTFT / provider-latency / decode-rate calibration model from
completed v2 length-targeted real-LLM pilot directories (Cohere + Gemini so
far — see docs/real_llm_latency_model_v2.md).

This does NOT make any API calls — it only reads each directory's existing
requests.jsonl. Reuses load_dataset()/fit_ols()/robust_stats()/
fit_latency_model_ttft_plus_decode() from fit_real_llm_latency_model.py (the
v1 fit script) so both scripts compute numbers the same way.

Why this is a separate script from fit_real_llm_latency_model.py, not a
flag on it: v1's source pilots never varied output length with max_tokens,
so v1's TTFT model uses realized output_tokens as a feature (a stand-in for
"how long a response was requested", the only length signal available) and
its latency model regresses only decode_seconds on output_tokens. v2's
target_output_tokens is a real, varying, KNOWN-BEFORE-GENERATION independent
variable, which changes what the "right" model looks like:

  - TTFT model: ttft ~ intercept + target_output_tokens (falling back to
    output_tokens only for legacy rows without a target) + prompt_tokens
    + concurrency_level [+ provider indicators when pooled]. TTFT precedes
    decoding, so using the realized output_tokens (unknowable at TTFT time)
    as v1 did is a modeling smell v2 fixes now that a knowable target
    exists.
  - Provider latency model: latency ~ intercept + ttft_seconds
    + output_tokens + prompt_tokens + concurrency_level [+ provider
    indicators]. TTFT enters as an explicit feature (unlike v1's
    ttft_plus_decode form, which nets it out of the target instead).
  - Decode-rate table: effective_decode_rate_tokens_per_sec fit per
    (provider, target_output_tokens) group, not just one pooled-per-provider
    rate as in v1 — this is what makes the Cohere/Gemini crossover at
    target=64 (see docs/real_llm_latency_model_v2.md) directly visible
    instead of averaged away.

Usage:
    python scripts/fit_real_llm_latency_model_v2.py \\
        --experiment-dir experiments/real_llm/cohere_v2_length_targeted_20260703T134447Z \\
        --experiment-dir experiments/real_llm/gemini_v2_length_targeted_20260703T141723Z \\
        --output-dir experiments/real_llm/latency_model_fit_v2
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from llmserveopt.real_llm import calibration_common as cc  # noqa: E402
import fit_real_llm_latency_model as v1  # noqa: E402


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-dir", dest="experiment_dirs", action="append", required=True,
        help="v2 pilot output directory containing requests.jsonl. Repeatable.",
    )
    parser.add_argument(
        "--label", dest="labels", action="append", default=None,
        help="Optional 'provider:model' override for the preceding "
        "--experiment-dir, in order given.",
    )
    parser.add_argument(
        "--exclude-rpm-wait-outliers", action="store_true",
        help="Drop requests heuristically flagged as local-RPM-wait-"
        "polluted. Not needed for logs already using the corrected "
        "rate_limiter_wait_seconds/provider_request_latency_seconds split "
        "(all v2 pilots), kept for generality if mixed with legacy dirs.",
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Model-inputs manifest (reproducibility, no live calls, no diff snapshot)
# ---------------------------------------------------------------------------

def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    return json.loads(path.read_text()) if path.exists() else None


def _git_info(root: Path) -> Dict[str, Any]:
    branch = cc._run_git(root, ["rev-parse", "--abbrev-ref", "HEAD"])
    commit = cc._run_git(root, ["rev-parse", "HEAD"])
    dirty = bool(cc._run_git(root, ["status", "--porcelain"]))
    return {"git_branch": branch or None, "git_commit": commit or None, "git_dirty": dirty}


def build_model_inputs_manifest(
    experiment_dirs: List[str], labels: Optional[List[str]], *, root: Path, argv: List[str],
) -> Dict[str, Any]:
    labels = labels or [None] * len(experiment_dirs)
    if len(labels) < len(experiment_dirs):
        labels = labels + [None] * (len(experiment_dirs) - len(labels))

    dirs_info = []
    for raw_dir, label in zip(experiment_dirs, labels):
        exp_dir = Path(raw_dir)
        if not exp_dir.is_absolute():
            exp_dir = root / exp_dir
        provider, model = v1._resolve_label(exp_dir, label)
        summary = _read_json(exp_dir / "summary.json") or {}
        run_config = _read_json(exp_dir / "run_config.json") or {}
        dirs_info.append({
            "experiment_dir": str(exp_dir),
            "provider": provider,
            "model": model,
            "workload_version": run_config.get("workload_version"),
            "status_counts": summary.get("status_counts"),
            "total_records": summary.get("total_records"),
            "error_rate": _error_rate(summary.get("status_counts")),
            "frac_reached_target_output_range": summary.get("frac_reached_target_output_range"),
            "by_target_output_tokens": summary.get("by_target_output_tokens"),
        })

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command_line": " ".join(argv),
        **_git_info(root),
        "experiment_dirs": dirs_info,
    }


def _error_rate(status_counts: Optional[Dict[str, int]]) -> Optional[float]:
    if not status_counts:
        return None
    total = sum(status_counts.values())
    if total == 0:
        return None
    failed = sum(v for k, v in status_counts.items() if k in ("error", "timeout", "rate_limited"))
    return failed / total


# ---------------------------------------------------------------------------
# v2 model fitting (TTFT with target_output_tokens, latency with TTFT as a
# feature) — bundles keep {fit, rows, features, y, feature_names} together
# so residuals can be recomputed without refitting.
# ---------------------------------------------------------------------------

def _ttft_length_feature(r: Dict[str, Any]) -> Optional[float]:
    """Prefer target_output_tokens (known before TTFT is observed) over
    realized output_tokens (not knowable at TTFT time); falls back to
    output_tokens only for legacy rows with no target_output_tokens."""
    if r.get("target_output_tokens") is not None:
        return r["target_output_tokens"]
    return r.get("output_tokens")


def fit_ttft_model_v2(rows: List[Dict[str, Any]], indicator_names: Optional[List[str]] = None) -> Dict[str, Any]:
    indicator_names = indicator_names or []
    filtered = [
        r for r in rows
        if r.get("ttft_seconds") is not None
        and _ttft_length_feature(r) is not None
        and r.get("prompt_tokens") is not None
        and r.get("concurrency_level") is not None
    ]
    feature_names = ["target_or_output_tokens", "prompt_tokens", "concurrency_level"] + indicator_names
    features = [
        [_ttft_length_feature(r), r["prompt_tokens"], r["concurrency_level"]]
        + [r[name] for name in indicator_names]
        for r in filtered
    ]
    y = [r["ttft_seconds"] for r in filtered]
    fit = v1.fit_ols(features, y, feature_names)
    return {"fit": fit, "rows": filtered, "features": features, "y": y, "feature_names": feature_names}


def fit_latency_model_v2(rows: List[Dict[str, Any]], indicator_names: Optional[List[str]] = None) -> Dict[str, Any]:
    indicator_names = indicator_names or []
    filtered = [
        r for r in rows
        if r.get("latency_seconds") is not None
        and r.get("ttft_seconds") is not None
        and r.get("output_tokens") is not None
        and r.get("prompt_tokens") is not None
        and r.get("concurrency_level") is not None
    ]
    feature_names = ["ttft_seconds", "output_tokens", "prompt_tokens", "concurrency_level"] + indicator_names
    features = [
        [r["ttft_seconds"], r["output_tokens"], r["prompt_tokens"], r["concurrency_level"]]
        + [r[name] for name in indicator_names]
        for r in filtered
    ]
    y = [r["latency_seconds"] for r in filtered]
    fit = v1.fit_ols(features, y, feature_names)
    return {"fit": fit, "rows": filtered, "features": features, "y": y, "feature_names": feature_names}


def _add_provider_indicators(rows: List[Dict[str, Any]], others: List[str]) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        d = dict(r)
        for p in others:
            d[f"is_{p}"] = 1.0 if r["provider"] == p else 0.0
        out.append(d)
    return out


def residuals_from_bundle(bundle: Dict[str, Any], model_type: str) -> List[Dict[str, Any]]:
    fit = bundle["fit"]
    if not fit.get("fit"):
        return []
    out = []
    for r, feats, actual in zip(bundle["rows"], bundle["features"], bundle["y"]):
        pred = fit["intercept"] + sum(
            fit[f"coef_{name}"] * val for name, val in zip(bundle["feature_names"], feats)
        )
        out.append({
            "provider": r["provider"],
            "request_id": r.get("request_id"),
            "model_type": model_type,
            "target_output_tokens": r.get("target_output_tokens"),
            "actual": actual,
            "predicted": pred,
            "residual": actual - pred,
        })
    return out


def summarize_residuals(residual_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str], List[float]] = {}
    for row in residual_rows:
        groups.setdefault((row["provider"], row["model_type"]), []).append(row["residual"])
    out = []
    for (provider, model_type), residuals in sorted(groups.items()):
        n = len(residuals)
        mean_r = sum(residuals) / n
        std_r = st.pstdev(residuals) if n > 1 else 0.0
        rmse = (sum(r ** 2 for r in residuals) / n) ** 0.5
        out.append({
            "provider": provider, "model_type": model_type, "n": n,
            "mean_residual": mean_r, "std_residual": std_r, "rmse": rmse,
        })
    return out


# ---------------------------------------------------------------------------
# Decode-rate table: effective tokens/sec by (provider, target_output_tokens)
# ---------------------------------------------------------------------------

def build_decode_rate_table(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows_out: List[Dict[str, Any]] = []
    providers = sorted({r["provider"] for r in records})
    for provider in providers:
        prov_records = [r for r in records if r["provider"] == provider]
        overall_fit = v1.fit_latency_model_ttft_plus_decode(prov_records)
        rows_out.append({
            "provider": provider,
            "target_output_tokens": "overall",
            "fit": bool(overall_fit.get("fit", False)),
            "n": overall_fit.get("n"),
            "r2": overall_fit.get("r2"),
            "effective_decode_rate_tokens_per_sec": overall_fit.get("effective_decode_rate_tokens_per_sec"),
            "decode_intercept_s": overall_fit.get("decode_intercept_s"),
        })
        targets = sorted({
            r["target_output_tokens"] for r in prov_records if r["target_output_tokens"] is not None
        })
        for target in targets:
            sub = [r for r in prov_records if r["target_output_tokens"] == target]
            fit = v1.fit_latency_model_ttft_plus_decode(sub)
            rows_out.append({
                "provider": provider,
                "target_output_tokens": int(target),
                "fit": bool(fit.get("fit", False)),
                "n": fit.get("n"),
                "r2": fit.get("r2"),
                "effective_decode_rate_tokens_per_sec": fit.get("effective_decode_rate_tokens_per_sec"),
                "decode_intercept_s": fit.get("decode_intercept_s"),
            })
    return rows_out


# ---------------------------------------------------------------------------
# Raw (model-free) latency-by-target table — used to check whether the
# Cohere/Gemini crossover finding from the pilot comparison remains visible
# independent of any regression assumptions.
# ---------------------------------------------------------------------------

def build_raw_latency_by_target(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows_out: List[Dict[str, Any]] = []
    providers = sorted({r["provider"] for r in records})
    for provider in providers:
        prov = [r for r in records if r["provider"] == provider]
        targets = sorted({r["target_output_tokens"] for r in prov if r["target_output_tokens"] is not None})
        for target in targets:
            sub = [r for r in prov if r["target_output_tokens"] == target]
            lat = [r["latency_seconds"] for r in sub if r["latency_seconds"] is not None]
            ttft = [r["ttft_seconds"] for r in sub if r["ttft_seconds"] is not None]
            out_tok = [r["output_tokens"] for r in sub if r["output_tokens"] is not None]
            rows_out.append({
                "provider": provider,
                "target_output_tokens": int(target),
                "n": len(sub),
                "mean_latency_s": (sum(lat) / len(lat)) if lat else None,
                "mean_ttft_s": (sum(ttft) / len(ttft)) if ttft else None,
                "mean_output_tokens": (sum(out_tok) / len(out_tok)) if out_tok else None,
            })
    return rows_out


def describe_crossover(raw_by_target: List[Dict[str, Any]]) -> List[str]:
    """Generate a plain-language per-target 'who is faster' statement
    directly from the raw means (not from any fitted model), so this
    doesn't silently go stale if the underlying data changes."""
    by_target: Dict[int, Dict[str, float]] = {}
    for row in raw_by_target:
        by_target.setdefault(row["target_output_tokens"], {})[row["provider"]] = row["mean_latency_s"]

    lines = []
    for target in sorted(by_target):
        provs = by_target[target]
        if len(provs) < 2:
            continue
        fastest = min(provs, key=lambda p: provs[p])
        ordered = sorted(provs.items(), key=lambda kv: kv[1])
        gap = ordered[1][1] - ordered[0][1]
        lines.append(
            f"- target={target}: **{fastest}** faster "
            f"({', '.join(f'{p}={v:.3f}s' for p, v in ordered)}, gap={gap:.3f}s)"
        )
    return lines


# ---------------------------------------------------------------------------
# Per-provider / pooled report assembly
# ---------------------------------------------------------------------------

def build_provider_report_v2(provider: str, records: List[Dict[str, Any]]) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "provider": provider,
        "model": records[0]["model"] if records else None,
        "n_records": len(records),
    }
    ttft_values = [r["ttft_seconds"] for r in records if r["ttft_seconds"] is not None]
    latency_values = [r["latency_seconds"] for r in records if r["latency_seconds"] is not None]
    report["ttft_stats"] = v1.robust_stats(ttft_values)
    report["latency_stats"] = v1.robust_stats(latency_values)

    ttft_bundle = fit_ttft_model_v2(records)
    latency_bundle = fit_latency_model_v2(records)
    report["ttft_model"] = ttft_bundle["fit"]
    report["latency_model"] = latency_bundle["fit"]
    return report, ttft_bundle, latency_bundle


def build_pooled_report_v2(records: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    providers = sorted({r["provider"] for r in records})
    baseline, *others = providers
    augmented = _add_provider_indicators(records, others)
    indicator_names = [f"is_{p}" for p in others]

    report: Dict[str, Any] = {
        "providers": providers,
        "baseline_provider": baseline,
        "n_records": len(records),
    }
    ttft_values = [r["ttft_seconds"] for r in records if r["ttft_seconds"] is not None]
    latency_values = [r["latency_seconds"] for r in records if r["latency_seconds"] is not None]
    report["ttft_stats"] = v1.robust_stats(ttft_values)
    report["latency_stats"] = v1.robust_stats(latency_values)

    ttft_bundle = fit_ttft_model_v2(augmented, indicator_names)
    latency_bundle = fit_latency_model_v2(augmented, indicator_names)
    report["ttft_model"] = ttft_bundle["fit"]
    report["latency_model"] = latency_bundle["fit"]
    return report, ttft_bundle, latency_bundle


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_records_csv(records: List[Dict[str, Any]], out_dir: Path) -> None:
    import pandas as pd
    pd.DataFrame(records).to_csv(out_dir / "latency_model_fit_v2.csv", index=False)


def write_decode_rates_csv(rows: List[Dict[str, Any]], out_dir: Path) -> None:
    import pandas as pd
    pd.DataFrame(rows).to_csv(out_dir / "provider_decode_rates.csv", index=False)


def write_residuals_csv(rows: List[Dict[str, Any]], out_dir: Path) -> None:
    import pandas as pd
    cols = ["provider", "model_type", "n", "mean_residual", "std_residual", "rmse"]
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(out_dir / "residuals_by_provider.csv", index=False)


def write_json_v2(payload: Dict[str, Any], out_dir: Path) -> None:
    (out_dir / "latency_model_fit_v2.json").write_text(json.dumps(payload, indent=2))


def write_manifest(manifest: Dict[str, Any], out_dir: Path) -> None:
    (out_dir / "model_inputs_manifest.json").write_text(json.dumps(manifest, indent=2))


def _fmt_model_v2(model: Dict[str, Any]) -> List[str]:
    if not model.get("fit"):
        return [f"  (not fit: {model.get('reason')})"]
    lines = [f"  n={model['n']}, R^2={model.get('r2'):.4f}" if model.get("r2") is not None else f"  n={model['n']}"]
    lines.append(f"  intercept={model['intercept']:.6f}")
    for key, val in model.items():
        if key.startswith("coef_"):
            lines.append(f"  {key}={val:.6f}")
    return lines


def _fmt_decode_row(row: Dict[str, Any]) -> str:
    if not row["fit"]:
        return f"| {row['provider']} | {row['target_output_tokens']} | n/a (not enough data) | | |"
    rate = row.get("effective_decode_rate_tokens_per_sec")
    rate_s = f"{rate:.1f}" if rate is not None else "n/a"
    r2 = row.get("r2")
    r2_s = f"{r2:.3f}" if r2 is not None else "n/a"
    if row["target_output_tokens"] != "overall" and (r2 is None or r2 < 0.5):
        r2_s += " ⚠️ low, do not trust"
    return f"| {row['provider']} | {row['target_output_tokens']} | {row['n']} | {rate_s} | {r2_s} |"


def write_markdown_v2(
    payload: Dict[str, Any], manifest: Dict[str, Any], decode_rows: List[Dict[str, Any]],
    residual_summary: List[Dict[str, Any]], raw_by_target: List[Dict[str, Any]], out_dir: Path,
) -> None:
    lines = [
        "# Real-LLM Latency Calibration Model Fit — v2 (length-targeted)",
        "",
        f"Generated: {payload['generated_at_utc']}",
        f"Git commit at fit time: `{manifest.get('git_commit')}` (dirty: {manifest.get('git_dirty')})",
        "",
        "This is a simple, interpretable OLS calibration baseline computed",
        "entirely from existing pilot logs — **no live API calls were made**",
        "to produce this fit. See docs/real_llm_latency_model_v2.md for the",
        "full write-up, safe/unsafe claims, and simulator-calibration",
        "guidance before using any coefficient here.",
        "",
        "## Input experiment directories",
        "",
        "| Provider | Model | Workload | Total records | Error rate | Reached target range |",
        "|---|---|---|---|---|---|",
    ]
    for d in manifest["experiment_dirs"]:
        err = d["error_rate"]
        err_s = f"{err:.1%}" if err is not None else "n/a"
        reached = d["frac_reached_target_output_range"]
        reached_s = f"{reached:.1%}" if reached is not None else "n/a"
        lines.append(
            f"| {d['provider']} | `{d['model']}` | {d['workload_version']} | "
            f"{d['total_records']} | {err_s} | {reached_s} |"
        )
    lines.append("")

    lines += ["## Output-token distribution by target length", ""]
    for d in manifest["experiment_dirs"]:
        by_target = d.get("by_target_output_tokens") or []
        if not by_target:
            continue
        lines.append(f"**{d['provider']}** (`{d['model']}`)")
        lines.append("")
        lines.append("| target | n_success | mean_output_tokens | mean_ratio | frac_reached |")
        lines.append("|---|---|---|---|---|")
        for row in by_target:
            lines.append(
                f"| {row['target_output_tokens']} | {row['n_success']} | "
                f"{row['mean_output_tokens']:.2f} | {row['mean_output_token_ratio']:.3f} | "
                f"{row['frac_reached_target_range']:.1%} |"
            )
        lines.append("")

    for provider, report in payload["providers"].items():
        lines += [
            f"## {provider} ({report.get('model')})",
            "",
            f"- n records used in fit: {report['n_records']}",
            f"- TTFT stats (s): {v1._fmt_stats(report['ttft_stats'])}",
            f"- Provider latency stats (s, excludes rate_limiter_wait_seconds): "
            f"{v1._fmt_stats(report['latency_stats'])}",
            "",
            "### TTFT model: `ttft ~ intercept + target_output_tokens(or output_tokens) "
            "+ prompt_tokens + concurrency`",
            *_fmt_model_v2(report["ttft_model"]),
            "",
            "### Provider latency model: `latency ~ intercept + ttft_seconds "
            "+ output_tokens + prompt_tokens + concurrency`",
            *_fmt_model_v2(report["latency_model"]),
            "",
        ]

    if "pooled" in payload:
        report = payload["pooled"]
        lines += [
            f"## Pooled model (baseline provider: {report['baseline_provider']})",
            "",
            f"- providers: {', '.join(report['providers'])}",
            f"- n records: {report['n_records']}",
            "",
            "### Pooled TTFT model (provider-indicator coefficients are the offset vs. baseline)",
            *_fmt_model_v2(report["ttft_model"]),
            "",
            "### Pooled provider latency model",
            *_fmt_model_v2(report["latency_model"]),
            "",
        ]

    lines += [
        "## Whether the target=64 crossover finding remains visible",
        "",
        "Raw (model-free) mean provider latency by (provider, target_output_tokens),",
        "computed directly from the same records used to fit the models above —",
        "independent of any regression assumption:",
        "",
        "| provider | target | n | mean latency (s) | mean TTFT (s) | mean output tokens |",
        "|---|---|---|---|---|---|",
    ]
    for row in raw_by_target:
        lines.append(
            f"| {row['provider']} | {row['target_output_tokens']} | {row['n']} | "
            f"{row['mean_latency_s']:.4f} | {row['mean_ttft_s']:.4f} | {row['mean_output_tokens']:.2f} |"
        )
    lines.append("")
    crossover_lines = describe_crossover(raw_by_target)
    if crossover_lines:
        lines += crossover_lines + [""]

    lines += [
        "## Decode-rate estimates by provider and target length",
        "",
        "Fit as `decode_seconds = latency_seconds - ttft_seconds ~ a * output_tokens + b`",
        "(same form as v1's `ttft_plus_decode`), per (provider, target_output_tokens) group,",
        "so the rate is not averaged across target lengths.",
        "",
        "**Caveat:** each per-target row has only n=36 and a narrow within-group",
        "range of realized `output_tokens` (since actual output clusters near its",
        "own target), which makes the *slope* (tokens/sec) hard to identify from a",
        "single target group alone — low R^2 rows below (arbitrarily, R^2 < 0.5) should",
        "not be read as a real decode-rate measurement, only the `overall` "
        "per-provider row (n=108, pooled across all three targets) is reasonably",
        "well-identified. This is a real limitation of a 36-request-per-cell pilot,",
        "not a bug in the fit.",
        "",
        "| provider | target | n | tokens/sec | R^2 |",
        "|---|---|---|---|---|",
    ]
    for row in decode_rows:
        lines.append(_fmt_decode_row(row))
    lines.append("")

    lines += [
        "## Residual / caveat discussion",
        "",
        "Per-(provider, model) residual summary for both fitted models",
        "(`actual - predicted`, in seconds). A `rmse` well above the",
        "provider's own TTFT/latency p50 (see stats above) means the linear",
        "form is a poor fit for that provider/model and coefficients should",
        "not be trusted for extrapolation beyond the pilot's own grid",
        "(targets 64/128/256, concurrency 1-8, 3 prompt buckets).",
        "",
        "| provider | model_type | n | mean_residual (s) | std_residual (s) | rmse (s) |",
        "|---|---|---|---|---|---|",
    ]
    for row in residual_summary:
        lines.append(
            f"| {row['provider']} | {row['model_type']} | {row['n']} | "
            f"{row['mean_residual']:.4f} | {row['std_residual']:.4f} | {row['rmse']:.4f} |"
        )
    lines.append("")

    (out_dir / "latency_model_fit_v2.md").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    records = v1.load_dataset(
        args.experiment_dirs, args.labels, root=ROOT,
        exclude_rpm_wait_outliers=args.exclude_rpm_wait_outliers,
    )
    if not records:
        print("ERROR: no successful records loaded from any --experiment-dir.", file=sys.stderr)
        return 2

    manifest = build_model_inputs_manifest(
        args.experiment_dirs, args.labels, root=ROOT,
        argv=[sys.argv[0]] + (argv if argv is not None else sys.argv[1:]),
    )

    providers = sorted({r["provider"] for r in records})
    payload: Dict[str, Any] = {
        "generated_at_utc": manifest["generated_at_utc"],
        "inputs": {
            "experiment_dirs": args.experiment_dirs,
            "labels": args.labels,
            "exclude_rpm_wait_outliers": args.exclude_rpm_wait_outliers,
        },
        "providers": {},
    }

    all_residuals: List[Dict[str, Any]] = []
    for provider in providers:
        sub = [r for r in records if r["provider"] == provider]
        report, ttft_bundle, latency_bundle = build_provider_report_v2(provider, sub)
        payload["providers"][provider] = report
        all_residuals += residuals_from_bundle(ttft_bundle, "ttft")
        all_residuals += residuals_from_bundle(latency_bundle, "latency")

    if len(providers) > 1:
        pooled_report, pooled_ttft_bundle, pooled_latency_bundle = build_pooled_report_v2(records)
        payload["pooled"] = pooled_report

    decode_rows = build_decode_rate_table(records)
    residual_summary = summarize_residuals(all_residuals)
    raw_by_target = build_raw_latency_by_target(records)
    payload["decode_rate_table"] = decode_rows
    payload["residual_summary"] = residual_summary
    payload["raw_latency_by_target"] = raw_by_target

    write_records_csv(records, out_dir)
    write_decode_rates_csv(decode_rows, out_dir)
    write_residuals_csv(residual_summary, out_dir)
    write_json_v2(payload, out_dir)
    write_manifest(manifest, out_dir)
    write_markdown_v2(payload, manifest, decode_rows, residual_summary, raw_by_target, out_dir)

    print(f"Fit v2 model(s) from {len(records)} records across {len(providers)} provider(s): {providers}")
    for fname in (
        "latency_model_fit_v2.json", "latency_model_fit_v2.md", "latency_model_fit_v2.csv",
        "provider_decode_rates.csv", "residuals_by_provider.csv", "model_inputs_manifest.json",
    ):
        print(f"  wrote {out_dir / fname}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
