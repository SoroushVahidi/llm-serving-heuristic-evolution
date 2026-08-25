#!/usr/bin/env python3
"""Generate the CC4b oracle-composition-dataset expansion config.

Programmatically replicates each of CC4's 10 synthetic regime templates (plus
its 2 real-trace regimes) across many seeded, lightly-jittered variants, so
CC5 has enough held-out windows for a statistically meaningful retry. Uses
the exact same candidate_search config as CC4 (same 34 candidates) for
direct comparability -- only the workload catalog is expanded. Writes the
resolved config to configs/cc4b_oracle_composition_expansion.yaml.
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

_SEED_RETRY_INCREMENT = 10_000
_MAX_SEED_RETRIES = 20


def _validate_and_fix_seeds(workloads: list[dict]) -> list[dict]:
    """The shared, frozen bursty-arrival generator (workloads/synthetic.py)
    can legitimately draw zero arrivals for some (seed, duration, burst
    params) combinations, especially at CC4b's short ~1-3s durations. Rather
    than hand-picking seeds around this, deterministically bump a failing
    entry's seed by a fixed increment and retry until it produces a
    non-empty trace, logging every adjustment for reproducibility."""
    from llmserveopt.experiments.cc1_composition_opportunity import CC1Error, _build_synthetic_requests

    fixed = []
    adjustments = []
    for w in workloads:
        if w["kind"] != "synthetic":
            fixed.append(w)
            continue
        original_seed = w["seed"]
        attempt = 0
        while True:
            try:
                _build_synthetic_requests(w, seed=w["seed"], max_requests=w.get("max_requests"))
                break
            except CC1Error:
                attempt += 1
                if attempt > _MAX_SEED_RETRIES:
                    raise CC1Error(f"{w['tag']}: could not find a non-empty seed after {_MAX_SEED_RETRIES} retries")
                w["seed"] = original_seed + attempt * _SEED_RETRY_INCREMENT
        if w["seed"] != original_seed:
            adjustments.append((w["tag"], original_seed, w["seed"]))
        fixed.append(w)
    if adjustments:
        print(f"Seed adjustments (zero-arrival retry) for {len(adjustments)} window(s):")
        for tag, old, new in adjustments:
            print(f"  {tag}: seed {old} -> {new}")
    return fixed

SLO_CLASSES = {
    "tight": [
        {"class_id": "tight", "slo_slack": 0.08, "priority": 3.0, "weight": 0.45},
        {"class_id": "medium", "slo_slack": 0.16, "priority": 2.0, "weight": 0.35},
        {"class_id": "loose", "slo_slack": 0.35, "priority": 1.0, "weight": 0.20},
    ],
    "mixed": [
        {"class_id": "tight", "slo_slack": 0.10, "priority": 3.0, "weight": 0.30},
        {"class_id": "medium", "slo_slack": 0.22, "priority": 2.0, "weight": 0.40},
        {"class_id": "loose", "slo_slack": 0.45, "priority": 1.0, "weight": 0.30},
    ],
    "priority": [
        {"class_id": "tight_hi", "slo_slack": 0.07, "priority": 5.0, "weight": 0.25},
        {"class_id": "tight_low", "slo_slack": 0.09, "priority": 1.0, "weight": 0.35},
        {"class_id": "medium_hi", "slo_slack": 0.20, "priority": 4.0, "weight": 0.20},
        {"class_id": "loose_low", "slo_slack": 0.40, "priority": 1.0, "weight": 0.20},
    ],
}

# One template per CC4 regime, taken verbatim from configs/cc4_oracle_composition_dataset.yaml.
TEMPLATES = {
    "underloaded": dict(base_seed=400, arrival_process="poisson", arrival_rate=8.0, duration=3.0,
                         prompt_mean=96.0, prompt_sigma=0.6, prompt_low=16, prompt_high=512,
                         output_mean=64.0, output_sigma=0.6, output_low=16, output_high=256,
                         prediction_noise_rel=0.1, max_requests=60, slo_classes="mixed"),
    "saturated": dict(base_seed=401, arrival_process="poisson", arrival_rate=80.0, duration=1.0,
                       prompt_mean=96.0, prompt_sigma=0.8, prompt_low=16, prompt_high=512,
                       output_mean=128.0, output_sigma=0.8, output_low=16, output_high=512,
                       prediction_noise_rel=0.2, max_requests=90, slo_classes="tight"),
    "mixed_slo": dict(base_seed=402, arrival_process="poisson", arrival_rate=40.0, duration=1.3,
                       prompt_mean=128.0, prompt_sigma=0.7, prompt_low=16, prompt_high=640,
                       output_mean=96.0, output_sigma=0.7, output_low=16, output_high=384,
                       prediction_noise_rel=0.15, max_requests=70, slo_classes="mixed"),
    "long_prompt": dict(base_seed=403, arrival_process="poisson", arrival_rate=50.0, duration=1.2,
                         prompt_mean=768.0, prompt_sigma=0.6, prompt_low=128, prompt_high=1500,
                         output_mean=96.0, output_sigma=0.6, output_low=16, output_high=256,
                         prediction_noise_rel=0.15, max_requests=70, slo_classes="tight"),
    "long_output": dict(base_seed=404, arrival_process="poisson", arrival_rate=45.0, duration=1.2,
                         prompt_mean=96.0, prompt_sigma=0.6, prompt_low=16, prompt_high=384,
                         output_mean=640.0, output_sigma=0.6, output_low=128, output_high=1400,
                         prediction_noise_rel=0.15, max_requests=65, slo_classes="tight"),
    "burst_transition": dict(base_seed=405, arrival_process="bursty", arrival_rate=30.0, duration=1.5,
                              burst_factor=6.0, burst_fraction=0.25,
                              prompt_mean=128.0, prompt_sigma=0.7, prompt_low=16, prompt_high=512,
                              output_mean=96.0, output_sigma=0.7, output_low=16, output_high=384,
                              prediction_noise_rel=0.15, max_requests=80, slo_classes="mixed"),
    "kv_pressure": dict(base_seed=406, arrival_process="poisson", arrival_rate=60.0, duration=1.2,
                         prompt_mean=256.0, prompt_sigma=0.7, prompt_low=32, prompt_high=768,
                         output_mean=192.0, output_sigma=0.7, output_low=32, output_high=512,
                         prediction_noise_rel=0.15, max_requests=85, slo_classes="tight"),
    "prediction_noise": dict(base_seed=407, arrival_process="poisson", arrival_rate=40.0, duration=1.3,
                              prompt_mean=128.0, prompt_sigma=0.7, prompt_low=16, prompt_high=512,
                              output_mean=128.0, output_sigma=0.7, output_low=16, output_high=512,
                              prediction_noise_rel=0.6, max_requests=70, slo_classes="mixed"),
    "priority_conflict": dict(base_seed=408, arrival_process="poisson", arrival_rate=45.0, duration=1.3,
                               prompt_mean=128.0, prompt_sigma=0.7, prompt_low=16, prompt_high=512,
                               output_mean=96.0, output_sigma=0.7, output_low=16, output_high=384,
                               prediction_noise_rel=0.15, max_requests=75, slo_classes="priority"),
    "selective_admission_trap": dict(base_seed=409, arrival_process="poisson", arrival_rate=90.0, duration=1.0,
                                      prompt_mean=160.0, prompt_sigma=0.8, prompt_low=16, prompt_high=640,
                                      output_mean=160.0, output_sigma=0.8, output_low=16, output_high=640,
                                      prediction_noise_rel=0.2, max_requests=90, slo_classes="priority"),
}

# Deterministic jitter multipliers applied to (arrival_rate, prompt_mean,
# output_mean) per variant index -- keeps each variant a genuinely different
# workload instance (not just a different RNG draw of the same intensity),
# while staying within the same regime "shape" (same slo_classes anchor,
# same arrival process family).
JITTER = [1.00, 0.90, 1.10, 0.82, 1.18, 0.95, 1.05, 0.88, 1.12, 0.97, 1.03, 0.85, 1.15]
# Wider jitter reserved for OOD_TEST synthetic variants -- these are meant
# to be genuinely shifted relative to the TRAIN/VALIDATION/ID_TEST range,
# not just more of the same.
OOD_JITTER = [1.35, 0.65]


def _make_workload(template_name: str, params: dict, *, split: str, seed: int, variant_idx: int, jitter: float) -> dict:
    p = dict(params)
    slo_key = p.pop("slo_classes")
    base_seed = p.pop("base_seed")
    entry = {
        "tag": f"cc4b_{template_name}_{split.lower()}_{variant_idx:02d}",
        "kind": "synthetic",
        "split": split,
        "regime": template_name,
        "seed": seed,
        **p,
        "slo_classes": copy.deepcopy(SLO_CLASSES[slo_key]),
    }
    if "arrival_rate" in entry:
        entry["arrival_rate"] = round(entry["arrival_rate"] * jitter, 4)
    if "prompt_mean" in entry:
        entry["prompt_mean"] = round(entry["prompt_mean"] * jitter, 4)
    if "output_mean" in entry:
        entry["output_mean"] = round(entry["output_mean"] * jitter, 4)
    return entry


def build_workloads() -> list[dict]:
    workloads: list[dict] = []
    for name, params in TEMPLATES.items():
        base_seed = params["base_seed"]

        # TRAIN: 2 variants
        for i in range(2):
            seed = base_seed * 100 + 1 + i
            workloads.append(_make_workload(name, params, split="TRAIN", seed=seed, variant_idx=i, jitter=JITTER[i]))

        # VALIDATION: 1 variant
        seed = base_seed * 100 + 20
        workloads.append(_make_workload(name, params, split="VALIDATION", seed=seed, variant_idx=0, jitter=JITTER[2]))

        # ID_TEST: 5 variants
        for i in range(5):
            seed = base_seed * 100 + 30 + i
            workloads.append(_make_workload(name, params, split="ID_TEST", seed=seed, variant_idx=i, jitter=JITTER[3 + i]))

        # OOD_TEST (synthetic, shifted intensity): 2 variants
        for i in range(2):
            seed = base_seed * 100 + 60 + i
            workloads.append(_make_workload(name, params, split="OOD_TEST", seed=seed, variant_idx=i, jitter=OOD_JITTER[i]))

    # Real-trace OOD windows: 3 request_transform intensity variants per trace file.
    real_trace_variants = [
        {"suffix": "baseline", "arrival_time_scale": 0.05, "slo_slack_scale": 0.02, "slo_slack_cap": 0.20, "slo_slack_floor": 0.02},
        {"suffix": "tight_slo", "arrival_time_scale": 0.05, "slo_slack_scale": 0.012, "slo_slack_cap": 0.12, "slo_slack_floor": 0.015},
        {"suffix": "loose_slo", "arrival_time_scale": 0.08, "slo_slack_scale": 0.035, "slo_slack_cap": 0.30, "slo_slack_floor": 0.03},
    ]
    for i, variant in enumerate(real_trace_variants):
        suffix = variant.pop("suffix")
        workloads.append({
            "tag": f"cc4b_azure_conversation_like_ood_test_{suffix}",
            "kind": "real_trace", "split": "OOD_TEST", "regime": "azure_conversation_like",
            "seed": 3102 + i, "max_requests": 80,
            "path": "data/processed/azure/azure_llm_2023_conv.jsonl",
            "request_transform": variant,
        })

    burstgpt_variants = [
        {"suffix": "baseline", "arrival_time_scale": 0.00012, "slo_slack_scale": 0.02, "slo_slack_cap": 0.20, "slo_slack_floor": 0.02},
        {"suffix": "tight_slo", "arrival_time_scale": 0.00012, "slo_slack_scale": 0.012, "slo_slack_cap": 0.12, "slo_slack_floor": 0.015},
        {"suffix": "loose_slo", "arrival_time_scale": 0.00018, "slo_slack_scale": 0.035, "slo_slack_cap": 0.30, "slo_slack_floor": 0.03},
    ]
    for i, variant in enumerate(burstgpt_variants):
        suffix = variant.pop("suffix")
        workloads.append({
            "tag": f"cc4b_burstgpt_derived_ood_test_{suffix}",
            "kind": "real_trace", "split": "OOD_TEST", "regime": "burstgpt_derived",
            "seed": 3200 + i, "max_requests": 80,
            "path": "data/processed/burstgpt/burstgpt_natural_10k.jsonl",
            "request_transform": variant,
        })

    return workloads


def build_config() -> dict:
    base = yaml.safe_load((ROOT / "configs" / "cc4_oracle_composition_dataset.yaml").read_text())
    config = {
        "schema_version": 1,
        "mode": "cc4",
        "seed": 20260803,
        "policy_subset": base["policy_subset"],
        "cc1b_borda_baseline": base["cc1b_borda_baseline"],
        "candidate_search": base["candidate_search"],  # identical to CC4 -- same 34 candidates
        "metrics": base["metrics"],
        "near_tie_primary_threshold": base["near_tie_primary_threshold"],
        "near_tie_thresholds": base["near_tie_thresholds"],
        "development_splits": base["development_splits"],
        "evaluation_splits": base["evaluation_splits"],
        "safeguards": {**base["safeguards"], "max_runs": 20000},
        "outputs": {"root": "results/cc4b_oracle_composition_expansion"},
        "service_model": base["service_model"],
        "simulator": base["simulator"],
        "gpus": base["gpus"],
        "workloads": _validate_and_fix_seeds(build_workloads()),
    }
    return config


def main() -> None:
    config = build_config()
    out_path = ROOT / "configs" / "cc4b_oracle_composition_expansion.yaml"
    out_path.write_text(yaml.safe_dump(config, sort_keys=False))
    windows = config["workloads"]
    by_split: dict[str, int] = {}
    for w in windows:
        by_split[w["split"]] = by_split.get(w["split"], 0) + 1
    print(f"Wrote {out_path} with {len(windows)} workload entries: {by_split}")


if __name__ == "__main__":
    main()
