#!/usr/bin/env python3
"""
Offline sanity-check: compare the simulator's service-time assumptions
against the fitted hosted-API latency model from
docs/real_llm_latency_model_v2.md / experiments/real_llm/latency_model_fit_v2/.

This makes NO API calls and imports no provider SDK. It only reads:
  - configs/real_llm_latency/cohere_gemini_v2_fit.yaml (--fitted-config)
  - experiments/real_llm/latency_model_fit_v2/latency_model_fit_v2.json (--fitted-model-dir)
  - results/gpu_calibration/service_curves.json, if present (--calibration-file)
and instantiates the simulator's own ServiceModel / CalibratedServiceModel
classes directly (no simulation run, just their timing formulas).

Why this comparison, not a "which is right" verdict: hosted-API latency is
a black-box, client-observed measurement of someone else's managed service
(network + queueing + provider-side batching/scheduling, all invisible to
us) while the simulator's service model represents controllable, local
GPU-serving internals. They measure genuinely different things. See
docs/real_llm_simulator_integration_plan.md's "what cannot be calibrated
from hosted APIs" section before treating any number here as a target to
match exactly.

Usage:
    python scripts/compare_simulator_to_real_llm_latency.py \\
        --fitted-config configs/real_llm_latency/cohere_gemini_v2_fit.yaml \\
        --fitted-model-dir experiments/real_llm/latency_model_fit_v2 \\
        --output-dir experiments/real_llm/simulator_latency_sanity_check
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from llmserveopt.real_llm.calibration_common import PROMPT_BUCKET_TARGET_TOKENS  # noqa: E402
from llmserveopt.simulator.service_model import ServiceModel  # noqa: E402
from llmserveopt.simulator.calibrated_service_model import CalibratedServiceModel  # noqa: E402

TARGET_OUTPUT_TOKENS = (64, 128, 256)
REPRESENTATIVE_BATCH_SIZES = (1, 8)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fitted-config", default="configs/real_llm_latency/cohere_gemini_v2_fit.yaml",
        help="Data-only YAML with the 'overall' per-provider decode rate / TTFT / latency stats.",
    )
    parser.add_argument(
        "--fitted-model-dir", default="experiments/real_llm/latency_model_fit_v2",
        help="Directory containing latency_model_fit_v2.json (per-target raw latency table).",
    )
    parser.add_argument(
        "--calibration-file", default="results/gpu_calibration/service_curves.json",
        help="Optional GPU-calibrated service curves. If missing, the "
        "calibrated-simulator comparison is skipped (not an error).",
    )
    parser.add_argument("--step-size", type=float, default=0.001, help="Synthetic ServiceModel step_size (s).")
    parser.add_argument("--output-dir", default="experiments/real_llm/simulator_latency_sanity_check")
    return parser.parse_args(argv)


def _resolve(root: Path, raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else root / p


# ---------------------------------------------------------------------------
# Loading fitted hosted-API data (no live calls, local files only)
# ---------------------------------------------------------------------------

def load_fitted_yaml(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    import yaml
    return yaml.safe_load(path.read_text())


def load_fitted_model_json(fitted_model_dir: Path) -> Optional[Dict[str, Any]]:
    path = fitted_model_dir / "latency_model_fit_v2.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def raw_latency_by_target(fitted_payload: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not fitted_payload:
        return []
    return fitted_payload.get("raw_latency_by_target") or []


# ---------------------------------------------------------------------------
# Simulator service-model instantiation (no simulation run — just the
# timing formulas these classes already expose)
# ---------------------------------------------------------------------------

def build_synthetic_service_model(step_size: float) -> ServiceModel:
    return ServiceModel(step_size=step_size, enable_prefill_modeling=False)


def build_calibrated_service_model(calibration_file: Path) -> Optional[CalibratedServiceModel]:
    if not calibration_file.exists():
        return None
    try:
        return CalibratedServiceModel(calibration_file=calibration_file, enable_prefill_modeling=True)
    except Exception as exc:  # noqa: BLE001 - degrade gracefully, this is a sanity check not a hard dependency
        print(f"WARNING: could not load calibrated service model: {exc}", file=sys.stderr)
        return None


def synthetic_decode_rate_tokens_per_sec(model: ServiceModel) -> float:
    return 1.0 / model.step_size


def synthetic_prefill_seconds(model: ServiceModel, prompt_tokens: int) -> float:
    # enable_prefill_modeling=False (the actual simulator default) means
    # prefill is instantaneous — there is no TTFT-analogue at all.
    return model.prefill_steps(prompt_tokens) * model.step_size


def calibrated_decode_rate_tokens_per_sec(
    model: CalibratedServiceModel, *, batch_size: int, context_tokens: int,
) -> float:
    per_token_s = model.compute_decode_step_time(batch_size=batch_size, context_tokens=context_tokens)
    return 1.0 / per_token_s if per_token_s > 0 else float("inf")


def calibrated_prefill_seconds(model: CalibratedServiceModel, prompt_tokens: int) -> float:
    return model.compute_prefill_steps(prompt_tokens) * model.step_size


# ---------------------------------------------------------------------------
# Comparison tables
# ---------------------------------------------------------------------------

def build_comparison_by_target(
    hosted_raw: List[Dict[str, Any]],
    synthetic_model: ServiceModel,
    calibrated_model: Optional[CalibratedServiceModel],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    by_key = {(r["provider"], r["target_output_tokens"]): r for r in hosted_raw}
    providers = sorted({r["provider"] for r in hosted_raw})

    for target in TARGET_OUTPUT_TOKENS:
        synth_decode_s = target * synthetic_model.step_size
        synth_rate = synthetic_decode_rate_tokens_per_sec(synthetic_model)

        cal_rates: Dict[int, Optional[float]] = {}
        cal_decode_s: Dict[int, Optional[float]] = {}
        if calibrated_model is not None:
            for batch_size in REPRESENTATIVE_BATCH_SIZES:
                rate = calibrated_decode_rate_tokens_per_sec(
                    calibrated_model, batch_size=batch_size, context_tokens=target,
                )
                cal_rates[batch_size] = rate
                cal_decode_s[batch_size] = target / rate if rate else None

        row: Dict[str, Any] = {
            "target_output_tokens": target,
            "simulator_synthetic_decode_rate_tokens_per_sec": synth_rate,
            "simulator_synthetic_decode_time_s": synth_decode_s,
        }
        for batch_size in REPRESENTATIVE_BATCH_SIZES:
            row[f"simulator_calibrated_b{batch_size}_decode_rate_tokens_per_sec"] = cal_rates.get(batch_size)
            row[f"simulator_calibrated_b{batch_size}_decode_time_s"] = cal_decode_s.get(batch_size)

        for provider in providers:
            hosted = by_key.get((provider, target))
            if hosted is None:
                continue
            row[f"{provider}_mean_latency_s"] = hosted["mean_latency_s"]
            row[f"{provider}_mean_ttft_s"] = hosted["mean_ttft_s"]
            row[f"{provider}_mean_output_tokens"] = hosted["mean_output_tokens"]
            if hosted["mean_latency_s"]:
                row[f"simulator_synthetic_vs_{provider}_speed_ratio"] = (
                    hosted["mean_latency_s"] / synth_decode_s if synth_decode_s else None
                )
        rows.append(row)
    return rows


def build_comparison_by_provider(
    fitted_yaml: Optional[Dict[str, Any]],
    synthetic_model: ServiceModel,
    calibrated_model: Optional[CalibratedServiceModel],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    if fitted_yaml:
        for key, entry in (fitted_yaml.get("providers") or {}).items():
            decode = entry.get("decode_rate_overall") or {}
            rows.append({
                "entity": key,
                "source": "hosted_api",
                "model": entry.get("model"),
                "decode_rate_tokens_per_sec": decode.get("effective_decode_rate_tokens_per_sec"),
                "decode_rate_r2": decode.get("r2"),
                "ttft_or_prefill_analogue_s": (entry.get("ttft_seconds") or {}).get("mean"),
                "notes": "client-observed, includes network + provider scheduling",
            })

    # Simulator prefill/TTFT-analogue evaluated at the "medium" prompt
    # bucket (512 tokens) so it's comparable in scale to the hosted pilots'
    # own prompt-bucket sizing (calibration_common.PROMPT_BUCKET_TARGET_TOKENS).
    medium_prompt_tokens = PROMPT_BUCKET_TARGET_TOKENS["medium"]
    rows.append({
        "entity": "simulator_synthetic_default",
        "source": "simulator",
        "model": "ServiceModel (enable_prefill_modeling=False)",
        "decode_rate_tokens_per_sec": synthetic_decode_rate_tokens_per_sec(synthetic_model),
        "decode_rate_r2": None,
        "ttft_or_prefill_analogue_s": synthetic_prefill_seconds(synthetic_model, medium_prompt_tokens),
        "notes": "constant per-token decode time; prefill instantaneous by default (no TTFT analogue at all)",
    })

    if calibrated_model is not None:
        for batch_size in REPRESENTATIVE_BATCH_SIZES:
            rows.append({
                "entity": f"simulator_calibrated_rtx5060ti_b{batch_size}",
                "source": "simulator",
                "model": "CalibratedServiceModel (Qwen2.5-0.5B, RTX 5060 Ti)",
                "decode_rate_tokens_per_sec": calibrated_decode_rate_tokens_per_sec(
                    calibrated_model, batch_size=batch_size, context_tokens=medium_prompt_tokens,
                ),
                "decode_rate_r2": None,
                "ttft_or_prefill_analogue_s": calibrated_prefill_seconds(calibrated_model, medium_prompt_tokens),
                "notes": "local GPU prefill compute only; no network/provider-scheduling component",
            })

    return rows


# ---------------------------------------------------------------------------
# Narrative answers to the sanity-check questions
# ---------------------------------------------------------------------------

def build_findings(
    by_target: List[Dict[str, Any]], by_provider: List[Dict[str, Any]], calibrated_available: bool,
) -> Dict[str, Any]:
    synth_rate = next(
        (r["decode_rate_tokens_per_sec"] for r in by_provider if r["entity"] == "simulator_synthetic_default"), None,
    )
    hosted_rates = {
        r["entity"]: r["decode_rate_tokens_per_sec"] for r in by_provider if r["source"] == "hosted_api"
    }
    closest_provider = None
    if hosted_rates and synth_rate:
        closest_provider = min(hosted_rates, key=lambda p: abs(hosted_rates[p] - synth_rate))

    return {
        "simulator_synthetic_decode_rate_tokens_per_sec": synth_rate,
        "hosted_decode_rates_tokens_per_sec": hosted_rates,
        "simulator_synthetic_closest_hosted_provider": closest_provider,
        "simulator_synthetic_faster_than_all_hosted": (
            all(synth_rate > v for v in hosted_rates.values()) if synth_rate and hosted_rates else None
        ),
        "calibrated_model_available": calibrated_available,
        "simulator_decode_time_linear_in_output_tokens": True,  # true by construction in both variants
        "simulator_has_ttft_analogue_by_default": False,  # enable_prefill_modeling=False by default
    }


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_outputs(
    out_dir: Path,
    by_target: List[Dict[str, Any]],
    by_provider: List[Dict[str, Any]],
    findings: Dict[str, Any],
    inputs: Dict[str, Any],
) -> None:
    import pandas as pd

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(by_target).to_csv(out_dir / "comparison_by_target_output_tokens.csv", index=False)
    pd.DataFrame(by_provider).to_csv(out_dir / "comparison_by_provider.csv", index=False)

    summary = {
        "inputs": inputs,
        "findings": findings,
        "comparison_by_target_output_tokens": by_target,
        "comparison_by_provider": by_provider,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    write_markdown(out_dir, by_target, by_provider, findings, inputs)


def _fmt(v: Optional[float], digits: int = 2) -> str:
    return f"{v:.{digits}f}" if isinstance(v, (int, float)) else "n/a"


def write_markdown(
    out_dir: Path,
    by_target: List[Dict[str, Any]],
    by_provider: List[Dict[str, Any]],
    findings: Dict[str, Any],
    inputs: Dict[str, Any],
) -> None:
    lines = [
        "# Simulator vs. Real-LLM (Cohere/Gemini v2) Latency Sanity Check",
        "",
        "**Offline analysis only — no live API calls were made.** This compares",
        "the simulator's own service-time formulas (evaluated directly, not via",
        "a simulation run) against the fitted hosted-API latency model in",
        "`docs/real_llm_latency_model_v2.md`. See",
        "`docs/real_llm_simulator_integration_plan.md` for why these measure",
        "different things and should not be equated.",
        "",
        f"- Fitted config: `{inputs['fitted_config']}`",
        f"- Fitted model dir: `{inputs['fitted_model_dir']}`",
        f"- GPU calibration file: `{inputs['calibration_file']}` "
        f"({'found' if findings['calibrated_model_available'] else 'NOT FOUND — calibrated-simulator rows skipped'})",
        "",
        "## What decode tok/s does the simulator implicitly assume?",
        "",
    ]
    for row in by_provider:
        lines.append(
            f"- **{row['entity']}** ({row['source']}, `{row['model']}`): "
            f"{_fmt(row['decode_rate_tokens_per_sec'], 1)} tokens/sec"
            + (f" (R^2={_fmt(row['decode_rate_r2'], 3)})" if row.get("decode_rate_r2") is not None else "")
        )
    lines += [
        "",
        "## Is the simulator closer to Cohere v2, Gemini v2, or neither?",
        "",
    ]
    closest = findings.get("simulator_synthetic_closest_hosted_provider")
    if closest:
        lines.append(
            f"The default synthetic model's {_fmt(findings['simulator_synthetic_decode_rate_tokens_per_sec'], 1)} "
            f"tokens/sec is numerically closest to **{closest}**, but "
            f"{'faster than both providers' if findings.get('simulator_synthetic_faster_than_all_hosted') else 'not uniformly faster or slower than both'} "
            "— see the ratio columns in `comparison_by_target_output_tokens.csv`. "
            "This is a coincidental numeric proximity, not evidence the synthetic "
            "model represents either provider's actual behavior (it was hand-tuned "
            "for Phase 1, not calibrated against any provider)."
        )
    else:
        lines.append("Not enough data to determine (missing hosted or simulator rates).")
    lines += [
        "",
        "## Are simulator decode assumptions faster or slower than hosted measurements?",
        "",
    ]
    for row in by_provider:
        if row["source"] != "hosted_api":
            continue
        rate = row["decode_rate_tokens_per_sec"]
        synth_rate = findings["simulator_synthetic_decode_rate_tokens_per_sec"]
        if rate and synth_rate:
            factor = synth_rate / rate
            lines.append(
                f"- Synthetic default is **{factor:.1f}x {'faster' if factor > 1 else 'slower'}** "
                f"than {row['entity']} ({_fmt(synth_rate, 1)} vs. {_fmt(rate, 1)} tok/s)."
            )
    lines += [
        "",
        "## Does simulator latency scale linearly with output length?",
        "",
        "Yes, by construction in both simulator variants: `decode_time = "
        "output_tokens * per_token_time`, with zero intercept. Real hosted",
        "latency is *also* well-approximated as linear in output_tokens",
        "(R^2=0.89-0.92 per `docs/real_llm_latency_model_v2.md`) but with a",
        "materially nonzero intercept (~0.1-0.2s, comparable in scale to TTFT)",
        "that the simulator's zero-intercept formulas do not represent.",
        "",
        "## Does simulator TTFT/prefill have any analogue to hosted TTFT?",
        "",
    ]
    sim_default_row = next((r for r in by_provider if r["entity"] == "simulator_synthetic_default"), None)
    if sim_default_row:
        lines.append(
            f"The simulator's default (`enable_prefill_modeling=False`) has "
            f"**no TTFT analogue at all** — prefill is instantaneous "
            f"(computed value: {_fmt(sim_default_row['ttft_or_prefill_analogue_s'], 4)}s). "
        )
    cal_rows = [r for r in by_provider if "calibrated" in r["entity"]]
    if cal_rows:
        hosted_prefill_values = ", ".join(
            f"{r['entity']}={_fmt(r['ttft_or_prefill_analogue_s'], 3)}s"
            for r in by_provider
            if r["source"] == "hosted_api"
        )
        lines.append(
            f"The GPU-calibrated variant does model prefill "
            f"(~{_fmt(cal_rows[0]['ttft_or_prefill_analogue_s'], 4)}s at a 512-token "
            "prompt), but this is *local GPU compute time only* for a 0.5B model — "
            "one to two orders of magnitude smaller than hosted TTFT "
            f"({hosted_prefill_values}), "
            "because hosted TTFT also bundles network round-trip and provider-side "
            "admission/queueing that a local prefill-compute formula cannot represent."
        )
    else:
        lines.append("(GPU calibration file not found — calibrated-model prefill comparison skipped.)")

    lines += [
        "",
        "## Which real-LLM quantities can safely calibrate the simulator?",
        "",
        "- The `overall` per-provider decode rate (Cohere ~88.5 tok/s, Gemini",
        "  ~289 tok/s) as an order-of-magnitude sanity check against the",
        "  simulator's own decode-rate assumption — not a value to copy in",
        "  directly, since it reflects a specific hosted model/provider, not",
        "  a serving-engine property this simulator controls.",
        "- The qualitative finding that provider latency is linear in output",
        "  length with a nonzero intercept — worth checking the simulator's",
        "  own service model reflects *some* fixed per-request overhead if a",
        "  future task adds one, even though the specific hosted intercept",
        "  value should not be copied in directly (see below).",
        "",
        "## Which hosted measurements should NOT be used directly?",
        "",
        "- Any hosted TTFT value, as a stand-in for simulator prefill time —",
        "  it bundles network + provider-side admission/scheduling the",
        "  simulator has no equivalent for and should not fabricate one for.",
        "- Any per-target-length decode rate with R^2 < 0.5 (see",
        "  `docs/real_llm_latency_model_v2.md`) — several of these are fit",
        "  noise, not a real per-length effect.",
        "- Hosted concurrency/prompt-bucket latency trends — both pilots showed",
        "  no significant concurrency effect at 1-8 concurrent requests from a",
        "  single client, which says nothing about how either provider (or this",
        "  simulator) behaves under real multi-tenant load.",
        "",
        "## Comparison tables",
        "",
        "See `comparison_by_target_output_tokens.csv` and `comparison_by_provider.csv`",
        "for the full numeric detail behind every claim above.",
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    fitted_config_path = _resolve(ROOT, args.fitted_config)
    fitted_model_dir = _resolve(ROOT, args.fitted_model_dir)
    calibration_file = _resolve(ROOT, args.calibration_file)
    out_dir = _resolve(ROOT, args.output_dir)

    fitted_yaml = load_fitted_yaml(fitted_config_path)
    fitted_payload = load_fitted_model_json(fitted_model_dir)
    hosted_raw = raw_latency_by_target(fitted_payload)

    if fitted_yaml is None and not hosted_raw:
        print(
            f"ERROR: neither {fitted_config_path} nor "
            f"{fitted_model_dir}/latency_model_fit_v2.json could be read.",
            file=sys.stderr,
        )
        return 2

    synthetic_model = build_synthetic_service_model(args.step_size)
    calibrated_model = build_calibrated_service_model(calibration_file)

    by_target = build_comparison_by_target(hosted_raw, synthetic_model, calibrated_model)
    by_provider = build_comparison_by_provider(fitted_yaml, synthetic_model, calibrated_model)
    findings = build_findings(by_target, by_provider, calibrated_model is not None)

    inputs = {
        "fitted_config": str(fitted_config_path),
        "fitted_model_dir": str(fitted_model_dir),
        "calibration_file": str(calibration_file),
        "step_size": args.step_size,
    }

    write_outputs(out_dir, by_target, by_provider, findings, inputs)

    print(f"Wrote comparison to {out_dir}")
    for fname in ("summary.json", "summary.md", "comparison_by_target_output_tokens.csv", "comparison_by_provider.csv"):
        print(f"  {out_dir / fname}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
