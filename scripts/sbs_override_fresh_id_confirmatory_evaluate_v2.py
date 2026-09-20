#!/usr/bin/env python3
"""One-shot Stage-B evaluator for frozen SBS override V2 predictions.

This script is intentionally label-capable and must only be run in a later
explicitly authorized one-shot confirmation task.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "sbs_override_fresh_id_confirmatory_evaluate_v2.0.0"
SELECTOR_JSON_HASH = "1fdd2c0a19234bafd679a2678664119328fa37ae7d51c0178359640c2955de35"
SELECTOR_JOBLIB_HASH = "f882a54b808d9ab8f2c4ec139e19e3096ea5e3152cadf4c0badaa55c9644cfcd"
BOOTSTRAP_SEED = 20260919
BOOTSTRAP_REPLICATES = 2000
CI_LOW_Q = 0.025
CI_HIGH_Q = 0.975
PRIMARY_DENOMINATOR = 2862

DEFAULT_STAGE_A = (
    ROOT
    / "experiments"
    / "sbs_override_fresh_id_confirmatory_v2"
    / "stage_a"
    / "FROZEN_PREDICTIONS_V2.csv"
)
DEFAULT_STAGE_A_FREEZE = (
    ROOT
    / "experiments"
    / "sbs_override_fresh_id_confirmatory_v2"
    / "stage_a"
    / "STAGE_A_FREEZE.json"
)
DEFAULT_PROTOCOL = (
    ROOT / "experiments" / "sbs_override_fresh_id_confirmatory_v2" / "CONFIRMATORY_PROTOCOL_V2.json"
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_json(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=True, separators=(",", ": ")) + "\n"


def write_json(path: Path, obj: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(stable_json(obj))


def git(args: Sequence[str]) -> Optional[str]:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except Exception:
        return None


def package_versions() -> dict[str, str]:
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ["numpy", "pandas"]:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = "NOT_INSTALLED"
    return out


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text())
    primary = protocol["primary_endpoint"]
    bootstrap = protocol["bootstrap"]
    if primary["denominator"] != PRIMARY_DENOMINATOR:
        raise ValueError("protocol denominator mismatch")
    if bootstrap["replicates"] != BOOTSTRAP_REPLICATES:
        raise ValueError("protocol bootstrap replicate mismatch")
    if bootstrap["random_seed"] != BOOTSTRAP_SEED:
        raise ValueError("protocol bootstrap seed mismatch")
    if bootstrap["ci_quantiles"] != [CI_LOW_Q, CI_HIGH_Q]:
        raise ValueError("protocol CI quantiles mismatch")
    return protocol


def load_stage_a_predictions(path: Path, expected_hash: str) -> pd.DataFrame:
    observed = sha256_file(path)
    if observed != expected_hash:
        raise ValueError({"stage_a_prediction_hash_mismatch": {"observed": observed, "expected": expected_hash}})
    pred = pd.read_csv(path)
    if len(pred) != PRIMARY_DENOMINATOR or pred["state_id"].nunique() != PRIMARY_DENOMINATOR:
        raise ValueError("Stage-A predictions must have exactly 2862 unique states")
    if int(pred["override"].sum() + pred["abstain"].sum()) != PRIMARY_DENOMINATOR:
        raise ValueError("invalid Stage-A override/abstain indicators")
    if set(pred["selector_json_sha256"]) != {SELECTOR_JSON_HASH}:
        raise ValueError("selector JSON hash mismatch in Stage-A predictions")
    if set(pred["selector_joblib_sha256"]) != {SELECTOR_JOBLIB_HASH}:
        raise ValueError("selector joblib hash mismatch in Stage-A predictions")
    return pred


def load_labels(label_source: Path) -> pd.DataFrame:
    labels = pd.read_csv(label_source)
    required = {"state_id", "canonical_action_id", "a_sbs_anwg"}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError({"missing_label_columns": sorted(missing)})
    return labels


def join_predictions_to_labels(predictions: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    override = predictions[predictions["override"].astype(int) == 1].copy()
    abstain = predictions[predictions["override"].astype(int) == 0].copy()
    joined = override.merge(
        labels[["state_id", "canonical_action_id", "a_sbs_anwg"]],
        left_on=["state_id", "selected_canonical_action_id"],
        right_on=["state_id", "canonical_action_id"],
        how="left",
        validate="one_to_one",
    )
    if joined["a_sbs_anwg"].isna().any():
        raise ValueError("selected override missing SBS-relative label")
    joined["realized_gain"] = joined["a_sbs_anwg"].astype(float)
    abstain = abstain.copy()
    abstain["canonical_action_id"] = "SBS_ABSTAIN"
    abstain["a_sbs_anwg"] = 0.0
    abstain["realized_gain"] = 0.0
    out = pd.concat([joined, abstain], ignore_index=True, sort=False)
    if len(out) != PRIMARY_DENOMINATOR or out["state_id"].nunique() != PRIMARY_DENOMINATOR:
        raise ValueError("joined confirmation rows must preserve Stage-A denominator")
    return out


def primary_mean_gain(joined: pd.DataFrame) -> float:
    if len(joined) != PRIMARY_DENOMINATOR:
        raise ValueError("primary denominator drift")
    return float(joined["realized_gain"].sum() / PRIMARY_DENOMINATOR)


def scenario_cluster_bootstrap(
    joined: pd.DataFrame,
    *,
    seed: int = BOOTSTRAP_SEED,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    scenarios = np.asarray(sorted(joined["scenario_id"].astype(str).unique()))
    by_scenario = {sid: g.copy() for sid, g in joined.groupby("scenario_id")}
    rng = np.random.default_rng(seed)
    means = np.empty(int(replicates), dtype=float)
    for i in range(int(replicates)):
        sample = rng.choice(scenarios, size=len(scenarios), replace=True)
        total = 0.0
        n = 0
        for sid in sample:
            g = by_scenario[str(sid)]
            total += float(g["realized_gain"].sum())
            n += int(len(g))
        means[i] = total / float(n)
    return {
        "replicates": int(replicates),
        "random_seed": int(seed),
        "ci95_low": float(np.quantile(means, CI_LOW_Q)),
        "ci95_high": float(np.quantile(means, CI_HIGH_Q)),
        "bootstrap_mean": float(means.mean()),
    }


def primary_verdict(mean_gain: float, ci95_low: float) -> str:
    if mean_gain > 0.0 and ci95_low > 0.0:
        return "CONFIRMED_POSITIVE_GENERALIZATION"
    return "POSITIVE_GENERALIZATION_NOT_CONFIRMED"


def secondary_diagnostics(joined: pd.DataFrame) -> dict[str, Any]:
    overrides = joined[joined["override"].astype(int) == 1]
    scenario_gain = joined.groupby("scenario_id")["realized_gain"].sum()
    positive = scenario_gain[scenario_gain > 0]
    top = positive.sort_values(ascending=False)
    positive_total = float(positive.sum())
    return {
        "override_count": int(len(overrides)),
        "override_rate": float(len(overrides) / PRIMARY_DENOMINATOR),
        "beneficial_overrides": int((overrides["realized_gain"] > 0).sum()),
        "harmful_overrides": int((overrides["realized_gain"] < 0).sum()),
        "zero_effect_overrides": int((overrides["realized_gain"] == 0).sum()),
        "total_realized_gain": float(joined["realized_gain"].sum()),
        "negative_aggregate_gain_scenarios": int((scenario_gain < 0).sum()),
        "positive_aggregate_gain_scenarios": int((scenario_gain > 0).sum()),
        "zero_aggregate_gain_scenarios": int((scenario_gain == 0).sum()),
        "worst_override": float(overrides["realized_gain"].min()) if len(overrides) else 0.0,
        "worst_scenario": float(scenario_gain.min()) if len(scenario_gain) else 0.0,
        "top1_scenario_share_of_positive_gain": float(top.iloc[:1].sum() / positive_total) if positive_total > 0 else None,
        "top5_scenario_share_of_positive_gain": float(top.iloc[:5].sum() / positive_total) if positive_total > 0 else None,
    }


def evaluate_once(args: argparse.Namespace) -> dict[str, Any]:
    result_dir = Path(args.result_dir)
    if result_dir.exists():
        raise FileExistsError(f"refusing to reuse existing one-shot result directory: {result_dir}")
    result_dir.mkdir(parents=True)
    try:
        protocol_path = Path(args.protocol)
        protocol_hash = sha256_file(protocol_path)
        expected_protocol_hash = args.expected_protocol_sha256
        if expected_protocol_hash and protocol_hash != expected_protocol_hash:
            raise ValueError({"protocol_hash_mismatch": {"observed": protocol_hash, "expected": expected_protocol_hash}})
        load_protocol(protocol_path)
        stage_a_freeze = json.loads(Path(args.stage_a_freeze).read_text())
        predictions = load_stage_a_predictions(
            Path(args.stage_a_predictions),
            stage_a_freeze["prediction_sha256"],
        )
        if sha256_file(Path(args.stage_a_freeze)) != args.expected_stage_a_freeze_sha256:
            raise ValueError("Stage-A freeze hash mismatch")
        labels = load_labels(Path(args.label_source))
        joined = join_predictions_to_labels(predictions, labels)
        mean = primary_mean_gain(joined)
        boot = scenario_cluster_bootstrap(joined)
        verdict = primary_verdict(mean, boot["ci95_low"])
        result = {
            "schema_version": SCHEMA_VERSION,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "primary_mean_realized_gain": mean,
            "primary_denominator": PRIMARY_DENOMINATOR,
            "bootstrap": boot,
            "primary_verdict": verdict,
            "secondary_diagnostics": secondary_diagnostics(joined),
            "input_hashes": {
                "stage_a_predictions": sha256_file(Path(args.stage_a_predictions)),
                "stage_a_freeze": sha256_file(Path(args.stage_a_freeze)),
                "protocol": protocol_hash,
                "label_source": sha256_file(Path(args.label_source)),
            },
            "git": {"head": git(["rev-parse", "HEAD"]), "branch": git(["branch", "--show-current"])},
            "package_versions": package_versions(),
        }
        write_json(result_dir / "CONFIRMATORY_RESULT_V2.json", result)
        (result_dir / "COMPLETE").write_text(stable_json({"status": "complete", "primary_verdict": verdict}))
        return result
    except Exception as exc:
        write_json(
            result_dir / "FAILED_BEFORE_OR_DURING_ONE_SHOT.json",
            {
                "schema_version": SCHEMA_VERSION,
                "error": repr(exc),
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        )
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-a-predictions", default=str(DEFAULT_STAGE_A))
    parser.add_argument("--stage-a-freeze", default=str(DEFAULT_STAGE_A_FREEZE))
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--expected-stage-a-freeze-sha256", required=True)
    parser.add_argument("--label-source", required=True)
    parser.add_argument("--result-dir", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    result = evaluate_once(args)
    print(stable_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
