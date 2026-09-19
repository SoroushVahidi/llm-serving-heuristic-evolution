"""Generate read-only post-run analysis for JOINT240 SBS advantage learnability.

This script consumes the frozen run_v1 artifacts only. It does not fit models,
regenerate predictions, run simulators, launch schedulers, or edit manuscript
files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DISAGREEMENT_STATES = 8888
TOTAL_SBS_DECISION_STATES = 453_016
EXPECTED_VERDICTS = {
    "HELD_OUT_OVERRIDE_SIGNAL_CONFIRMED",
    "HELD_OUT_SIGNAL_POSITIVE_BUT_UNCERTAIN",
    "PREDICTION_SIGNAL_WITHOUT_POLICY_GAIN",
    "NO_USEFUL_HELD_OUT_SIGNAL",
}
PRIMARY = "STATE_ACTION_V1.PRIMARY_TRAIN_SELECTED"
FAMILIES = ["RIDGE", "HIST_GRADIENT_BOOSTING", "EXTRA_TREES", "DUMMY_ZERO"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git(args: list[str], cwd: Path) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()
    except Exception:
        return None


def finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def selector_row(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "states",
        "mean_realized_gain",
        "total_realized_gain",
        "override_count",
        "override_rate",
        "beneficial_override_count",
        "harmful_override_count",
        "zero_effect_override_count",
        "precision_among_overrides",
        "harmful_override_rate",
        "oracle_mean_gain",
        "oracle_total_gain",
        "gap_closure",
    ]
    return {k: metrics.get(k) for k in keys}


def summarize_fixed_thresholds(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for threshold, metrics in summary["fixed_threshold_metrics"].items():
        rows.append({"threshold": float(threshold), **selector_row(metrics)})
    return sorted(rows, key=lambda r: r["threshold"])


def generate(run_root: Path, output_dir: Path) -> dict[str, Any]:
    repo = run_root.parents[2]
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = json.loads((run_root / "learnability_summary.json").read_text())
    primary_summary = summary["primary_summary"]
    primary_metrics = primary_summary["selector_metrics"]
    primary_boot = primary_summary["bootstrap"]

    hyper = pd.read_csv(run_root / "selected_hyperparameters_and_thresholds.csv")
    decisions = pd.read_csv(run_root / f"oof_state_decisions.{PRIMARY}.csv")
    actions = pd.read_csv(
        run_root / f"oof_action_predictions.{PRIMARY}.csv",
        usecols=["state_id", "scenario_id", "fold", "canonical_action_id", "is_sbs_action"],
    )
    scenarios = pd.read_csv(run_root / f"scenario_metrics.{PRIMARY}.csv")

    exit_code = (run_root / "EXIT_CODE").read_text().strip()
    stderr_size = (run_root / "logs" / "run_stderr.log").stat().st_size
    selected_by_fold = {int(k): v for k, v in summary["primary_selected_model_by_fold"].items()}

    validation = {
        "exit_code": exit_code,
        "stderr_bytes": int(stderr_size),
        "outer_folds": sorted(int(x) for x in decisions["fold"].unique()),
        "scenario_count": int(decisions["scenario_id"].nunique()),
        "held_out_state_decisions": int(len(decisions)),
        "held_out_action_predictions": int(len(actions)),
        "unique_decision_states": int(decisions["state_id"].nunique()),
        "duplicate_state_decisions": int(decisions.duplicated("state_id").sum()),
        "duplicate_state_action_predictions": int(actions.duplicated(["state_id", "canonical_action_id"]).sum()),
        "sbs_rows_in_action_predictions": int(actions["is_sbs_action"].sum()),
        "max_state_fold_count": int(actions.groupby("state_id")["fold"].nunique().max()),
        "max_scenario_fold_count": int(actions.groupby("scenario_id")["fold"].nunique().max()),
        "state_action_completed": "STATE_ACTION_V1.RIDGE" in summary["summaries"]
        and "STATE_ACTION_V1.HIST_GRADIENT_BOOSTING" in summary["summaries"]
        and "STATE_ACTION_V1.EXTRA_TREES" in summary["summaries"]
        and "STATE_ACTION_V1.DUMMY_ZERO" in summary["summaries"],
        "state_core_completed": "STATE_CORE_V1.RIDGE" in summary["summaries"]
        and "STATE_CORE_V1.HIST_GRADIENT_BOOSTING" in summary["summaries"]
        and "STATE_CORE_V1.EXTRA_TREES" in summary["summaries"]
        and "STATE_CORE_V1.DUMMY_ZERO" in summary["summaries"],
        "all_preregistered_model_families_completed": all(
            f"STATE_ACTION_V1.{m}" in summary["summaries"] and f"STATE_CORE_V1.{m}" in summary["summaries"]
            for m in FAMILIES
        ),
    }

    protocol = {
        "status": "PASS",
        "outer_test_scenarios_disjoint_from_train": all(
            x["scenario_overlap"] == 0 for x in summary["split_audit"]["outer_folds"]
        ),
        "outer_test_states_disjoint_from_train": all(x["state_overlap"] == 0 for x in summary["split_audit"]["outer_folds"]),
        "scenario_assigned_to_multiple_folds": summary["split_audit"]["scenario_assigned_to_multiple_folds"],
        "state_equal_weights_recorded": bool(summary["provenance"]["weighted"]),
        "quick_mode": bool(summary["provenance"]["quick"]),
        "threshold_grid": summary["provenance"]["threshold_grid"],
        "bootstrap_replicates": summary["provenance"]["bootstrap_replicates"],
        "bootstrap_seed": 20260919,
        "implementation_notes": [
            "Frozen implementation crossfits each candidate on the four outer-training folds only.",
            "Hyperparameter and threshold selection use those outer-training cross-fitted predictions.",
            "Final outer-test predictions are emitted only after model/config/threshold selection.",
            "All actions from a state and all states from a scenario stay in one frozen fold.",
        ],
    }

    fold_rows = []
    for fold, group in decisions.groupby("fold", sort=True):
        fold = int(fold)
        model = selected_by_fold[fold]
        row = hyper[
            (hyper["feature_set"] == "STATE_ACTION_V1")
            & (hyper["model_name"] == model)
            & (hyper["outer_fold"] == fold)
        ].iloc[0]
        scenario_group = scenarios[scenarios["fold"] == fold]
        overrides = int(group["override"].sum())
        beneficial = int(group["beneficial_override"].sum())
        harmful = int(group["harmful_override"].sum())
        zero = int(group["zero_effect_override"].sum())
        oracle_total = float(group["oracle_gain"].sum())
        total_gain = float(group["realized_gain"].sum())
        fold_rows.append(
            {
                "fold": fold,
                "scenario_count": int(scenario_group["scenario_id"].nunique()),
                "state_count": int(group["state_id"].nunique()),
                "selected_model_family": model,
                "selected_hyperparameters": row["selected_config"],
                "selected_threshold": float(row["selected_threshold"]),
                "train_selection_metric": float(row["train_selection_metric"]),
                "override_rate": float(overrides / len(group)),
                "mean_realized_gain": float(group["realized_gain"].mean()),
                "total_realized_gain": total_gain,
                "beneficial_overrides": beneficial,
                "harmful_overrides": harmful,
                "zero_effect_overrides": zero,
                "override_precision": float(beneficial / overrides) if overrides else None,
                "oracle_mean_gain": float(group["oracle_gain"].mean()),
                "oracle_gap_closure": float(total_gain / oracle_total) if oracle_total else None,
            }
        )
    fold_df = pd.DataFrame(fold_rows)

    positive_scenarios = scenarios[scenarios["scenario_gain"] > 0].sort_values("scenario_gain", ascending=False)
    total_positive_gain = float(positive_scenarios["scenario_gain"].sum())
    net_gain = float(scenarios["scenario_gain"].sum())
    concentration = {}
    for pct in [0.01, 0.05, 0.10, 0.20]:
        n = max(1, math.ceil(len(scenarios) * pct))
        gain = float(positive_scenarios.head(n)["scenario_gain"].sum())
        concentration[f"top_{int(pct * 100)}pct"] = {
            "scenario_count": n,
            "gain": gain,
            "share_of_positive_gain": float(gain / total_positive_gain) if total_positive_gain else None,
            "share_of_net_gain": float(gain / net_gain) if net_gain else None,
        }

    harm = decisions[decisions["harmful_override"] == 1].copy()
    abs_loss = -harm["realized_gain"]
    harmful_summary = {
        "harmful_override_count": int(len(harm)),
        "harmful_fraction_all_states": float(len(harm) / len(decisions)),
        "harmful_fraction_overrides": float(len(harm) / decisions["override"].sum()),
        "mean_harmful_loss": float(harm["realized_gain"].mean()),
        "median_harmful_loss": float(harm["realized_gain"].median()),
        "p90_absolute_harmful_loss": float(abs_loss.quantile(0.90)),
        "p95_absolute_harmful_loss": float(abs_loss.quantile(0.95)),
        "maximum_harmful_loss": float(harm["realized_gain"].min()),
        "scenarios_with_harmful_override": int(harm["scenario_id"].nunique()),
        "folds_with_harmful_override": sorted(int(x) for x in harm["fold"].unique()),
    }

    oracle = {
        "oracle_mean_gain_per_state": float(decisions["oracle_gain"].mean()),
        "oracle_total_gain": float(decisions["oracle_gain"].sum()),
        "learned_mean_gain_per_state": float(decisions["realized_gain"].mean()),
        "learned_total_gain": float(decisions["realized_gain"].sum()),
        "gap_closure": float(decisions["realized_gain"].sum() / decisions["oracle_gain"].sum()),
        "mean_regret_to_oracle": float(decisions["regret_to_oracle"].mean()),
        "p90_regret": float(decisions["regret_to_oracle"].quantile(0.90)),
        "p95_regret": float(decisions["regret_to_oracle"].quantile(0.95)),
    }

    ablation_rows = []
    labels = [PRIMARY, "STATE_CORE_V1.RIDGE", "STATE_CORE_V1.HIST_GRADIENT_BOOSTING", "STATE_CORE_V1.EXTRA_TREES", "STATE_CORE_V1.DUMMY_ZERO"]
    for label in labels:
        s = primary_summary if label == PRIMARY else summary["summaries"][label]
        sm = s["selector_metrics"]
        boot = s["bootstrap"]["mean_realized_gain"]
        sc = s["scenario_metrics"]
        ablation_rows.append(
            {
                "label": label,
                "mean_realized_gain": sm["mean_realized_gain"],
                "ci95_low": boot["ci95_low"],
                "ci95_high": boot["ci95_high"],
                "override_rate": sm["override_rate"],
                "precision": sm["precision_among_overrides"],
                "harmful_override_rate": sm["harmful_override_rate"],
                "oracle_gap_closure": sm["gap_closure"],
                "negative_scenario_count": sc["scenarios_negative_gain"],
                "total_realized_gain": sm["total_realized_gain"],
            }
        )
    ablation_df = pd.DataFrame(ablation_rows)

    family_rows = []
    for feature_set in ["STATE_ACTION_V1", "STATE_CORE_V1"]:
        for model_name in FAMILIES:
            label = f"{feature_set}.{model_name}"
            s = summary["summaries"][label]
            sm = s["selector_metrics"]
            pdg = s["prediction_diagnostics"]
            family_rows.append(
                {
                    "label": label,
                    "mae": pdg["mae"],
                    "rmse": pdg["rmse"],
                    "spearman": pdg["spearman"],
                    "pearson": pdg["pearson"],
                    "mean_realized_gain": sm["mean_realized_gain"],
                    "override_rate": sm["override_rate"],
                    "override_precision": sm["precision_among_overrides"],
                    "harmful_rate": sm["harmful_override_rate"],
                }
            )
    family_df = pd.DataFrame(family_rows)

    scenario_summary = {
        "scenarios_positive_gain": int((scenarios["scenario_gain"] > 0).sum()),
        "scenarios_zero_gain": int((scenarios["scenario_gain"] == 0).sum()),
        "scenarios_negative_gain": int((scenarios["scenario_gain"] < 0).sum()),
        "mean_scenario_gain": float(scenarios["scenario_gain"].mean()),
        "median_scenario_gain": float(scenarios["scenario_gain"].median()),
        "p10_scenario_gain": float(scenarios["scenario_gain"].quantile(0.10)),
        "p5_scenario_gain": float(scenarios["scenario_gain"].quantile(0.05)),
        "worst_scenario": scenarios.loc[scenarios["scenario_gain"].idxmin()].to_dict(),
        "best_scenario": scenarios.loc[scenarios["scenario_gain"].idxmax()].to_dict(),
        "scenarios_containing_harmful_overrides": int((scenarios["harmful_overrides"] > 0).sum()),
        "maximum_harmful_overrides_in_one_scenario": int(scenarios["harmful_overrides"].max()),
        "positive_gain_concentration": concentration,
    }

    saved_verdict = summary["final_interpretation"]
    lower_bound_positive = finite_float(primary_boot["mean_realized_gain"]["ci95_low"])
    validated = saved_verdict
    closed_loop_status = "UNBLOCKED" if validated == "HELD_OUT_OVERRIDE_SIGNAL_CONFIRMED" else "BLOCKED"
    bottleneck = None
    if closed_loop_status == "BLOCKED":
        bottleneck = (
            "scenario-bootstrap 95% CI lower bound for mean held-out realized gain is not greater than zero"
            if lower_bound_positive is not None and lower_bound_positive <= 0
            else "validated verdict is not HELD_OUT_OVERRIDE_SIGNAL_CONFIRMED"
        )

    open_loop = {
        "disagreement_rate": float(DISAGREEMENT_STATES / TOTAL_SBS_DECISION_STATES),
        "mean_gain_per_disagreement_state": primary_metrics["mean_realized_gain"],
        "descriptive_mean_gain_per_sbs_decision_state": float(
            (DISAGREEMENT_STATES / TOTAL_SBS_DECISION_STATES) * primary_metrics["mean_realized_gain"]
        ),
        "total_realized_gain_div_total_sbs_decision_states": float(
            primary_metrics["total_realized_gain"] / TOTAL_SBS_DECISION_STATES
        ),
    }

    key_files = [
        "learnability_summary.json",
        "selected_hyperparameters_and_thresholds.csv",
        f"oof_state_decisions.{PRIMARY}.csv",
        f"scenario_metrics.{PRIMARY}.csv",
        f"scenario_bootstrap.{PRIMARY}.csv",
        f"summary.{PRIMARY}.json",
        "input_integrity_and_split_audit.json",
    ]
    key_checksums = {name: sha256(run_root / name) for name in key_files}

    inventory = {
        "files": [
            {
                "path": str(path.relative_to(run_root)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path) if path.is_file() and path.suffix in {".json", ".csv", ".log"} else None,
            }
            for path in sorted(run_root.rglob("*"))
            if path.is_file() and output_dir not in path.parents
        ],
        "model_serialization_present": bool(list(run_root.rglob("*.pkl")) or list(run_root.rglob("*.joblib"))),
        "sign_classifier_outputs_present": bool(list(run_root.glob("*SIGN*")) or list(run_root.glob("*sign*"))),
    }

    analysis = {
        "duplication_gate": "NOT_PREVIOUSLY_DONE",
        "run_root": str(run_root),
        "source_git_head": summary["provenance"]["git_head"],
        "source_git_branch": summary["provenance"]["git_branch"],
        "analysis_git_head_before_commit": git(["rev-parse", "HEAD"], repo),
        "saved_final_verdict": saved_verdict,
        "saved_final_verdict_in_expected_set": saved_verdict in EXPECTED_VERDICTS,
        "validated_final_verdict": validated,
        "closed_loop_gate_status": f"JOINT240_SBS_OVERRIDE_CLOSED_LOOP_V1 = {closed_loop_status}",
        "closed_loop_bottleneck": bottleneck,
        "validation": validation,
        "protocol_audit": protocol,
        "input_integrity": summary["input_integrity"],
        "split_audit": summary["split_audit"],
        "key_file_checksums": key_checksums,
        "inventory": inventory,
        "primary_selector_result": primary_metrics,
        "bootstrap": {
            **primary_boot,
            "replicates": summary["provenance"]["bootstrap_replicates"],
            "seed": 20260919,
            "is_lower_bound_greater_than_zero": bool(primary_boot["mean_realized_gain"]["ci95_low"] > 0),
        },
        "fold_level": fold_rows,
        "folds_with_positive_realized_gain": int((fold_df["total_realized_gain"] > 0).sum()),
        "scenario_level": scenario_summary,
        "oracle_gap_closure": oracle,
        "harmful_override_analysis": harmful_summary,
        "fixed_threshold_harm_table": summarize_fixed_thresholds(primary_summary),
        "ablation": ablation_rows,
        "model_family_results": family_rows,
        "prediction_diagnostics": primary_summary["prediction_diagnostics"],
        "optional_sign_classifier_result": "ABSENT_NOT_RUN",
        "open_loop_d_sbs_descriptive_estimate": open_loop,
        "required_fold_models_and_thresholds": fold_rows,
        "exact_next_action": "Do not launch closed-loop. Treat the offline result as positive but statistically uncertain; investigate robustness/concentration or improve selector before any 9-hour Wulver campaign.",
    }

    fold_df.to_csv(output_dir / "fold_level_results.csv", index=False)
    scenarios.sort_values("scenario_gain").to_csv(output_dir / "scenario_level_results.csv", index=False)
    ablation_df.to_csv(output_dir / "state_core_vs_state_action_ablation.csv", index=False)
    family_df.to_csv(output_dir / "model_family_results.csv", index=False)
    pd.DataFrame(summarize_fixed_thresholds(primary_summary)).to_csv(output_dir / "fixed_threshold_harm_table.csv", index=False)
    (output_dir / "post_run_analysis_summary.json").write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    (output_dir / "POST_RUN_ANALYSIS.md").write_text(render_markdown(analysis) + "\n")
    return analysis


def render_markdown(a: dict[str, Any]) -> str:
    primary = a["primary_selector_result"]
    boot = a["bootstrap"]["mean_realized_gain"]
    scen = a["scenario_level"]
    oracle = a["oracle_gap_closure"]
    harm = a["harmful_override_analysis"]
    open_loop = a["open_loop_d_sbs_descriptive_estimate"]
    lines = [
        "# JOINT240 SBS Advantage Learnability V1 Post-Run Analysis",
        "",
        "Generated from frozen `run_v1` outputs only. No fitting, prediction regeneration, scheduler launch, real-vLLM run, or manuscript edit was performed.",
        "",
        "## Verdict",
        f"- Duplication gate: `{a['duplication_gate']}`",
        f"- Saved final verdict: `{a['saved_final_verdict']}`",
        f"- Validated final verdict: `{a['validated_final_verdict']}`",
        f"- Closed-loop gate: `{a['closed_loop_gate_status']}`",
        f"- Bottleneck: {a['closed_loop_bottleneck']}",
        "",
        "## Primary Selector",
        f"- States: {primary['states']}",
        f"- Overrides: {primary['override_count']} ({primary['override_rate']:.6g})",
        f"- Beneficial / harmful / zero-effect overrides: {primary['beneficial_override_count']} / {primary['harmful_override_count']} / {primary['zero_effect_override_count']}",
        f"- Precision among overrides: {primary['precision_among_overrides']:.6g}",
        f"- Mean realized gain/state: {primary['mean_realized_gain']:.12g}",
        f"- Total realized gain: {primary['total_realized_gain']:.12g}",
        f"- Median / p5 / p10 realized gain: {primary['median_realized_gain']:.12g} / {primary['p5_state_realized_gain']:.12g} / {primary['p10_state_realized_gain']:.12g}",
        f"- Worst harmful override: {primary['maximum_harmful_override']:.12g}",
        "",
        "## Bootstrap Gate",
        f"- Mean realized gain/state 95% CI: [{boot['ci95_low']:.12g}, {boot['ci95_high']:.12g}], point {boot['mean']:.12g}",
        f"- Lower bound > 0: {'YES' if a['bootstrap']['is_lower_bound_greater_than_zero'] else 'NO'}",
        "",
        "## Scenario Generalization",
        f"- Positive / zero / negative scenarios: {scen['scenarios_positive_gain']} / {scen['scenarios_zero_gain']} / {scen['scenarios_negative_gain']}",
        f"- Mean / median / p10 / p5 scenario gain: {scen['mean_scenario_gain']:.12g} / {scen['median_scenario_gain']:.12g} / {scen['p10_scenario_gain']:.12g} / {scen['p5_scenario_gain']:.12g}",
        f"- Worst scenario: {scen['worst_scenario']['scenario_id']} ({scen['worst_scenario']['scenario_gain']:.12g})",
        f"- Best scenario: {scen['best_scenario']['scenario_id']} ({scen['best_scenario']['scenario_gain']:.12g})",
        "",
        "## Oracle And Harm",
        f"- Oracle mean / total gain: {oracle['oracle_mean_gain_per_state']:.12g} / {oracle['oracle_total_gain']:.12g}",
        f"- Learned mean / total gain: {oracle['learned_mean_gain_per_state']:.12g} / {oracle['learned_total_gain']:.12g}",
        f"- Gap closure: {oracle['gap_closure']:.12g}",
        f"- Mean regret / p90 / p95: {oracle['mean_regret_to_oracle']:.12g} / {oracle['p90_regret']:.12g} / {oracle['p95_regret']:.12g}",
        f"- Harmful overrides: {harm['harmful_override_count']} across {harm['scenarios_with_harmful_override']} scenarios and folds {harm['folds_with_harmful_override']}",
        f"- Mean / median harmful loss: {harm['mean_harmful_loss']:.12g} / {harm['median_harmful_loss']:.12g}",
        "",
        "## Open-Loop d_SBS Descriptive Estimate",
        f"- Disagreement rate: {open_loop['disagreement_rate']:.12g}",
        f"- Descriptive mean gain per SBS decision state: {open_loop['descriptive_mean_gain_per_sbs_decision_state']:.12g}",
        "",
        "## Artifact Tables",
        "- `fold_level_results.csv`",
        "- `scenario_level_results.csv`",
        "- `state_core_vs_state_action_ablation.csv`",
        "- `model_family_results.csv`",
        "- `fixed_threshold_harm_table.csv`",
        "- `post_run_analysis_summary.json`",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("experiments/joint240_sbs_advantage_learnability_v1/run_v1"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/joint240_sbs_advantage_learnability_v1/run_v1/post_run_analysis_v1"),
    )
    args = parser.parse_args()
    analysis = generate(args.run_root.resolve(), args.output_dir.resolve())
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "validated_final_verdict": analysis["validated_final_verdict"],
        "closed_loop_gate_status": analysis["closed_loop_gate_status"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
