"""CC5 uncertainty / regime-fallback refinement.

Tightly scoped follow-up to the CC4b/CC5 retry: attach model-agnostic
uncertainty to the already-selected predictor class, analyze held-out
performance by regime, and compare validation-tuned fallback variants
without redesigning the predictor pipeline or beginning CC6.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from llmserveopt.experiments.cc1_composition_opportunity import ROOT, display_path, git_state
from llmserveopt.experiments.cc5_contextual_predictor import (
    PRIMARY_COL,
    COMPLETION_COL,
    CC4Dataset,
    CC5Error,
    LookupBaseline,
    PredictorArtifact,
    UncertaintyOODGate,
    UNCERTAINTY_SCHEMA_VERSION,
    assert_uncertainty_calibrator_compatible,
    bootstrap_ci,
    build_regret_regressor_factories,
    build_candidate_matrix,
    determine_cc5_verdict,
    evaluate_selector,
    fit_best_fixed_policy,
    fit_best_global_composition,
    fit_existing_hard_selector,
    fit_model_agnostic_uncertainty,
    load_cc4_dataset,
    regret_vs_oracle_fixed,
    select_composition_with_fallback,
    validate_cc4_dataset,
    FeatureEncoder,
    _actual_metrics,
)


DEFAULT_DATASET = "results/cc4b_oracle_composition_expansion/20260803T182426Z"
DEFAULT_OUTPUT_ROOT = "results/cc5_uncertainty_regime_refinement"


@dataclass
class RefinementResult:
    output_dir: Path
    manifest: dict[str, Any]
    verdict: dict[str, Any]


def heartbeat(out_dir: Path, stage: str, **payload: Any) -> None:
    path = out_dir / "checkpoints" / "heartbeat.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "stage": stage,
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **payload,
    }, indent=2, sort_keys=True, default=str))


def resolve_output_dir(root: str | Path, *, timestamp: str | None) -> Path:
    stamp = timestamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return ROOT / str(root) / stamp


def _oracle_composition_anwg(ds: CC4Dataset, window_ids: Sequence[str]) -> dict[str, float]:
    rows = ds.oracle_labels[ds.oracle_labels["window_id"].isin(window_ids)]
    return dict(zip(rows["window_id"], rows["oracle_anwg"]))


def fit_regime_fallback_rules(
    *,
    artifact: PredictorArtifact,
    ds: CC4Dataset,
    val_ids: Sequence[str],
    best_global: LookupBaseline,
) -> dict[str, str]:
    """VALIDATION-only regime rule: fall back when predictor mean ANWG on a
    regime is strictly below best-global mean ANWG for that regime."""
    pred = evaluate_selector(
        lambda row: select_composition_with_fallback(artifact, ds, row, gate_mode="uncertainty_only"),
        ds, val_ids,
    )
    glob = evaluate_selector(lambda row: {"selected_candidate_id": best_global.select(row["regime"])}, ds, val_ids)
    rules: dict[str, str] = {}
    for regime in sorted(set(pred["regime"])):
        p = float(pred.loc[pred["regime"] == regime, PRIMARY_COL].mean())
        g = float(glob.loc[glob["regime"] == regime, PRIMARY_COL].mean())
        rules[regime] = "fallback" if p < g else "trust_predictor"
    return rules


def fit_completion_safe_fallback_rules(
    *,
    ds: CC4Dataset,
    val_ids: Sequence[str],
    best_fixed: LookupBaseline,
    best_global: LookupBaseline,
) -> dict[str, str]:
    """VALIDATION-only: prefer best-global fallback on a regime only when it
    beats best-fixed on ANWG without mean completion regression > 0.05."""
    fixed_val = evaluate_selector(lambda row: {"selected_candidate_id": best_fixed.select(row["regime"])}, ds, val_ids)
    glob_val = evaluate_selector(lambda row: {"selected_candidate_id": best_global.select(row["regime"])}, ds, val_ids)
    rules: dict[str, str] = {}
    for regime in sorted(set(fixed_val["regime"]) | set(glob_val["regime"])):
        f = fixed_val[fixed_val["regime"] == regime]
        g = glob_val[glob_val["regime"] == regime]
        if f.empty:
            rules[regime] = "global"
            continue
        if g.empty:
            rules[regime] = "fixed"
            continue
        if (
            float(g[PRIMARY_COL].mean()) >= float(f[PRIMARY_COL].mean())
            and float(g[COMPLETION_COL].mean()) >= float(f[COMPLETION_COL].mean()) - 0.05
        ):
            rules[regime] = "global"
        else:
            rules[regime] = "fixed"
    return rules


@dataclass
class HybridLookupBaseline:
    """Per-regime choice between two global lookup baselines (validation-tuned)."""

    name: str
    rules: dict[str, str]
    best_fixed: LookupBaseline
    best_global: LookupBaseline

    @property
    def selection(self) -> dict[str, str]:
        out = {}
        for regime, rule in self.rules.items():
            out[regime] = self.best_global.selection if rule == "global" else self.best_fixed.selection
        return out

    def select(self, regime: str) -> str:
        rule = self.rules.get(regime, "fixed")
        return self.best_global.select(regime) if rule == "global" else self.best_fixed.select(regime)


def build_deployable_artifact(
    ds: CC4Dataset,
    *,
    seed: int = 0,
    ood_z_threshold: float = 2.0,
    n_bootstrap: int = 12,
) -> tuple[PredictorArtifact, dict[str, Any], pd.DataFrame, LookupBaseline, LookupBaseline, LookupBaseline]:
    """Rebuild the CC5 point model with unchanged LOWO selection, then attach
    model-agnostic uncertainty calibrated on VALIDATION only."""
    audit = validate_cc4_dataset(ds)
    dev_ids = sorted(ds.causal_features[ds.causal_features["split"].isin(ds.development_splits)]["window_id"])
    encoder = FeatureEncoder.fit(ds.causal_features[ds.causal_features["window_id"].isin(dev_ids)])

    # Reuse the retry's selected model class without changing selection criteria:
    # re-run LOWO exactly as CC5 does.
    from llmserveopt.experiments.cc5_contextual_predictor import leave_one_window_out_cv, build_regret_training_table

    factories = build_regret_regressor_factories(seed=seed)
    cv = leave_one_window_out_cv(factories, ds, encoder, dev_ids)
    best_model_name = cv.groupby("model")[PRIMARY_COL].mean().sort_values(ascending=False).index[0]
    model = factories[best_model_name]()
    X_dev, y_dev, _ = build_regret_training_table(ds, encoder, dev_ids)
    model.fit(X_dev, y_dev)

    best_fixed = fit_best_fixed_policy(ds, dev_ids)
    best_global = fit_best_global_composition(ds, dev_ids)
    hard_selector = fit_existing_hard_selector(ds, dev_ids)

    calibrator, all_cals, threshold_grid = fit_model_agnostic_uncertainty(
        model_name=best_model_name,
        model_factory=factories[best_model_name],
        encoder=encoder,
        ds=ds,
        dev_window_ids=dev_ids,
        seed=seed,
        n_bootstrap=n_bootstrap,
        fallback_for_threshold=best_global,  # main candidate uses global-composition fallback
    )
    assert_uncertainty_calibrator_compatible(calibrator)
    gate = UncertaintyOODGate.fit(
        encoder, ds.causal_features, dev_ids,
        ood_z_threshold=ood_z_threshold,
        uncertainty_threshold=float(calibrator.uncertainty_threshold),
    )
    git = git_state()
    artifact = PredictorArtifact(
        model_name=best_model_name,
        model=model,
        encoder=encoder,
        gate=gate,
        fallback=best_global,
        supports_ensemble_uncertainty=(best_model_name == "random_forest"),
        dsl_schema_version=2,
        compiler_version="cc3.1",
        dataset_config_hash=ds.manifest.get("config_hash", ""),
        dataset_dir=display_path(ds.dataset_dir),
        git_sha=git["commit"],
        feature_schema=encoder.feature_names,
        target_definition="regret = window_oracle_anwg - candidate_anwg; argmin predicted regret",
        split_definition={
            "development_splits": list(ds.development_splits),
            "evaluation_splits": list(ds.evaluation_splits),
            "dev_windows": dev_ids,
        },
        hyperparameters={
            "ood_z_threshold": ood_z_threshold,
            "uncertainty_threshold": float(calibrator.uncertainty_threshold),
            "seed": seed,
            "uncertainty_schema_version": UNCERTAINTY_SCHEMA_VERSION,
            "n_bootstrap": n_bootstrap,
        },
        uncertainty_method=calibrator.method,
        ood_method="max_abs_zscore_vs_dev_causal_feature_distribution",
        objective_definition="arrival_normalized_weighted_goodput",
        training_timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        dependency_versions={},
        uncertainty_calibrator=calibrator,
        gate_mode="ood_or_uncertainty",
    )
    meta = {
        "dataset_audit": audit,
        "best_model_name": best_model_name,
        "cv_ranking": cv.groupby("model")[PRIMARY_COL].mean().sort_values(ascending=False).to_dict(),
        "calibrator": calibrator.calibration_manifest,
        "all_calibrators": {
            name: {
                "empirical_coverage": c.empirical_coverage,
                "calibration_error": c.calibration_error,
                "runtime_overhead_s": c.runtime_overhead_s,
                "uncertainty_threshold": c.uncertainty_threshold,
            }
            for name, c in all_cals.items()
        },
        "selected_uncertainty_method": calibrator.method,
    }
    return artifact, meta, threshold_grid, best_fixed, best_global, hard_selector


def evaluate_fallback_variants(
    artifact: PredictorArtifact,
    ds: CC4Dataset,
    eval_ids: Sequence[str],
    val_ids: Sequence[str],
    best_fixed: LookupBaseline,
    best_global: LookupBaseline,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, str], dict[str, str], str]:
    """Compare gating/fallback variants; thresholds/rules from VALIDATION only.

    Returns summary, per-variant evals, regime trust rules, completion-safe
    fallback rules, and the selected main deployable variant name.
    """
    artifact_fixed = PredictorArtifact(**{**artifact.__dict__, "fallback": best_fixed})
    regime_rules = fit_regime_fallback_rules(
        artifact=artifact_fixed, ds=ds, val_ids=val_ids, best_global=best_global,
    )
    completion_safe_rules = fit_completion_safe_fallback_rules(
        ds=ds, val_ids=val_ids, best_fixed=best_fixed, best_global=best_global,
    )
    hybrid = HybridLookupBaseline(
        name="completion_safe_global_or_fixed",
        rules=completion_safe_rules,
        best_fixed=best_fixed,
        best_global=best_global,
    )
    artifact_regime = PredictorArtifact(
        **{**artifact.__dict__, "fallback": hybrid, "regime_fallback_rules": regime_rules, "gate_mode": "regime_aware"}
    )

    variants: dict[str, Any] = {
        "current_ood_only_fixed_fallback": lambda row: select_composition_with_fallback(
            artifact_fixed, ds, row, gate_mode="ood_only", fallback_override=best_fixed,
        ),
        "uncertainty_only_fixed_fallback": lambda row: select_composition_with_fallback(
            artifact_fixed, ds, row, gate_mode="uncertainty_only", fallback_override=best_fixed,
        ),
        "ood_plus_uncertainty_fixed_fallback": lambda row: select_composition_with_fallback(
            artifact_fixed, ds, row, gate_mode="ood_or_uncertainty", fallback_override=best_fixed,
        ),
        "uncertainty_only_global_fallback": lambda row: select_composition_with_fallback(
            artifact, ds, row, gate_mode="uncertainty_only", fallback_override=best_global,
        ),
        "ood_plus_uncertainty_global_fallback": lambda row: select_composition_with_fallback(
            artifact, ds, row, gate_mode="ood_or_uncertainty", fallback_override=best_global,
        ),
        "regime_aware_hybrid_fallback": lambda row: select_composition_with_fallback(
            artifact_regime, ds, row, gate_mode="regime_aware", fallback_override=hybrid,
        ),
        "ood_plus_uncertainty_hybrid_fallback": lambda row: select_composition_with_fallback(
            artifact, ds, row, gate_mode="ood_or_uncertainty", fallback_override=hybrid,
        ),
        "never_abstain_predictor": lambda row: {
            "selected_candidate_id": select_composition_with_fallback(
                artifact, ds, row, gate_mode="uncertainty_only",
            )["model_recommended_candidate_id"],
            "abstained": False,
            "fallback_reason": None,
        },
    }

    fixed_eval = evaluate_selector(lambda row: {"selected_candidate_id": best_fixed.select(row["regime"])}, ds, eval_ids)
    fixed_comp = fixed_eval.set_index("window_id")[COMPLETION_COL]

    per_variant: dict[str, pd.DataFrame] = {}
    summary_rows = []
    for name, selector in variants.items():
        ev = evaluate_selector(selector, ds, eval_ids)
        per_variant[name] = ev
        ci = bootstrap_ci(ev[PRIMARY_COL].tolist())
        viol = int((ev[COMPLETION_COL].to_numpy() < fixed_comp.loc[ev["window_id"]].to_numpy() - 0.05).sum())
        summary_rows.append({
            "variant": name,
            "mean_anwg": ci["mean"],
            "ci_low": ci["ci_low"],
            "ci_high": ci["ci_high"],
            "n": ci["n"],
            "abstention_rate": float(ev["abstained"].mean()) if "abstained" in ev else 0.0,
            "mean_completion": float(ev[COMPLETION_COL].mean()),
            "mean_regret": float(ev["regret"].mean()),
            "completion_violations": viol,
        })
    summary = pd.DataFrame(summary_rows).sort_values("mean_anwg", ascending=False)

    # Main deployable: best ANWG among completion-safe variants. Prefer the
    # hybrid OOD+uncertainty path when tied within 1e-9.
    safe = summary[summary["completion_violations"] == 0]
    if safe.empty:
        main_name = "ood_plus_uncertainty_fixed_fallback"
    else:
        safe_sorted = safe.sort_values(["mean_anwg", "variant"], ascending=[False, True])
        preferred = "ood_plus_uncertainty_hybrid_fallback"
        if preferred in set(safe_sorted["variant"]) and float(
            safe_sorted.loc[safe_sorted["variant"] == preferred, "mean_anwg"].iloc[0]
        ) >= float(safe_sorted.iloc[0]["mean_anwg"]) - 1e-12:
            main_name = preferred
        else:
            main_name = str(safe_sorted.iloc[0]["variant"])
    return summary, per_variant, regime_rules, completion_safe_rules, main_name


def build_regime_table(
    *,
    predictor_eval: pd.DataFrame,
    best_fixed_eval: pd.DataFrame,
    best_global_eval: pd.DataFrame,
    hard_eval: pd.DataFrame,
    oracle_anwg: Mapping[str, float],
    ds: CC4Dataset,
) -> pd.DataFrame:
    fixed_by_w = best_fixed_eval.set_index("window_id")
    global_by_w = best_global_eval.set_index("window_id")
    hard_by_w = hard_eval.set_index("window_id")
    rows = []
    for regime, group in predictor_eval.groupby("regime"):
        wids = list(group["window_id"])
        pred_vals = group[PRIMARY_COL].tolist()
        fixed_vals = [float(fixed_by_w.loc[w, PRIMARY_COL]) for w in wids]
        global_vals = [float(global_by_w.loc[w, PRIMARY_COL]) for w in wids]
        hard_vals = [float(hard_by_w.loc[w, PRIMARY_COL]) for w in wids]
        oracle_vals = [float(oracle_anwg[w]) for w in wids]
        pred_ci = bootstrap_ci(pred_vals)
        g_mean = float(np.mean(global_vals))
        p_mean = float(np.mean(pred_vals))
        if p_mean > g_mean + 1e-6:
            winner = "predictor"
        elif g_mean > p_mean + 1e-6:
            winner = "global_composition"
        else:
            winner = "tied"
        # Uncertainty catch/miss: among windows where model recommendation
        # underperforms global, did we abstain?
        catch = miss = 0
        for _, row in group.iterrows():
            w = row["window_id"]
            if "model_recommended_candidate_id" in group.columns and pd.notna(row.get("model_recommended_candidate_id")):
                model_anwg = _actual_metrics(ds, w, row["model_recommended_candidate_id"])[PRIMARY_COL]
            else:
                selected_col = "predictor_selected_candidate_id" if "predictor_selected_candidate_id" in group.columns else "selected_candidate_id"
                model_anwg = float(row[PRIMARY_COL]) if selected_col not in group.columns else _actual_metrics(ds, w, row[selected_col])[PRIMARY_COL]
            global_anwg = float(global_by_w.loc[w, PRIMARY_COL])
            failed = model_anwg + 1e-9 < global_anwg
            if failed and row["abstained"]:
                catch += 1
            elif failed and not row["abstained"]:
                miss += 1
        rows.append({
            "regime": regime,
            "window_count": len(wids),
            "predictor_anwg": p_mean,
            "best_global_composition_anwg": g_mean,
            "hard_selector_anwg": float(np.mean(hard_vals)),
            "best_fixed_anwg": float(np.mean(fixed_vals)),
            "oracle_composition_anwg": float(np.mean(oracle_vals)),
            "regret_vs_oracle": float(np.mean(oracle_vals) - p_mean),
            "abstention_rate": float(group["abstained"].mean()),
            "fallback_rate": float(group["abstained"].mean()),
            "completion_fraction": float(group[COMPLETION_COL].mean()),
            "ci_low": pred_ci["ci_low"],
            "ci_high": pred_ci["ci_high"],
            "winner_vs_global": winner,
            "uncertainty_catch_count": catch,
            "uncertainty_miss_count": miss,
        })
    return pd.DataFrame(rows).sort_values("predictor_anwg")


def classify_uncertainty_behavior(regime_table: pd.DataFrame) -> dict[str, list[str]]:
    return {
        "predictor_win_regimes": sorted(regime_table.loc[regime_table["winner_vs_global"] == "predictor", "regime"]),
        "global_composition_win_regimes": sorted(regime_table.loc[regime_table["winner_vs_global"] == "global_composition", "regime"]),
        "tied_regimes": sorted(regime_table.loc[regime_table["winner_vs_global"] == "tied", "regime"]),
        "worst_regimes": sorted(regime_table.nsmallest(3, "predictor_anwg")["regime"]),
        "uncertainty_catch_regimes": sorted(
            regime_table.loc[regime_table["uncertainty_catch_count"] > 0, "regime"]
        ),
        "uncertainty_miss_regimes": sorted(
            regime_table.loc[regime_table["uncertainty_miss_count"] > 0, "regime"]
        ),
    }


def run_refinement(
    *,
    dataset_dir: str | Path = DEFAULT_DATASET,
    output_root: str = DEFAULT_OUTPUT_ROOT,
    timestamp: str | None = None,
    resume_dir: str | Path | None = None,
    seed: int = 0,
    ood_z_threshold: float = 2.0,
    n_bootstrap: int = 12,
) -> RefinementResult:
    ds = load_cc4_dataset(dataset_dir)
    output_dir = Path(resume_dir) if resume_dir is not None else resolve_output_dir(output_root, timestamp=timestamp)
    if (output_dir / "manifest.json").exists():
        manifest = json.loads((output_dir / "manifest.json").read_text())
        verdict = json.loads((output_dir / "verdict.json").read_text())
        return RefinementResult(output_dir=output_dir, manifest=manifest, verdict=verdict)

    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    heartbeat(output_dir, "start")

    artifact, meta, threshold_grid, best_fixed, best_global, hard_selector = build_deployable_artifact(
        ds, seed=seed, ood_z_threshold=ood_z_threshold, n_bootstrap=n_bootstrap,
    )
    heartbeat(output_dir, "artifact_ready", model=meta["best_model_name"], uncertainty=meta["selected_uncertainty_method"])

    dev_ids = sorted(ds.causal_features[ds.causal_features["split"].isin(ds.development_splits)]["window_id"])
    val_ids = sorted(ds.causal_features[ds.causal_features["split"] == "VALIDATION"]["window_id"])
    eval_ids = sorted(ds.causal_features[ds.causal_features["split"].isin(ds.evaluation_splits)]["window_id"])

    variant_summary, per_variant, regime_rules, completion_safe_rules, main_name = evaluate_fallback_variants(
        artifact, ds, eval_ids, val_ids, best_fixed, best_global,
    )
    # Main deployable candidate selected among completion-safe variants.
    predictor_eval = per_variant[main_name].rename(columns={"selected_candidate_id": "predictor_selected_candidate_id"})
    # Attach model recommendation for regime catch/miss analysis.
    gate_for_main = "regime_aware" if "regime_aware" in main_name else (
        "ood_only" if "ood_only" in main_name and "uncertainty" not in main_name else
        "uncertainty_only" if main_name.startswith("uncertainty_only") else
        "ood_or_uncertainty"
    )
    if "hybrid" in main_name:
        fallback_for_main: Any = HybridLookupBaseline(
            name="completion_safe_global_or_fixed",
            rules=completion_safe_rules,
            best_fixed=best_fixed,
            best_global=best_global,
        )
    elif "global" in main_name:
        fallback_for_main = best_global
    else:
        fallback_for_main = best_fixed
    rec_rows = []
    for wid in eval_ids:
        causal_row = ds.causal_features.set_index("window_id").loc[wid]
        if main_name == "never_abstain_predictor":
            decision = select_composition_with_fallback(
                artifact, ds, causal_row, gate_mode="uncertainty_only", fallback_override=fallback_for_main,
            )
            decision = {**decision, "selected_candidate_id": decision["model_recommended_candidate_id"], "abstained": False}
        else:
            decision = select_composition_with_fallback(
                artifact if "fixed" not in main_name or "hybrid" in main_name else PredictorArtifact(**{**artifact.__dict__, "fallback": best_fixed}),
                ds, causal_row, gate_mode=gate_for_main, fallback_override=fallback_for_main,
            )
        rec_rows.append({
            "window_id": wid,
            "model_recommended_candidate_id": decision["model_recommended_candidate_id"],
            "uncertainty": decision["uncertainty"],
            "ood_score": decision["ood_score"],
            "inference_overhead_s": decision["inference_overhead_s"],
        })
    rec_df = pd.DataFrame(rec_rows)
    predictor_eval = predictor_eval.merge(rec_df, on="window_id", how="left")

    best_fixed_eval = evaluate_selector(lambda row: {"selected_candidate_id": best_fixed.select(row["regime"])}, ds, eval_ids)
    best_global_eval = evaluate_selector(lambda row: {"selected_candidate_id": best_global.select(row["regime"])}, ds, eval_ids)
    hard_eval = evaluate_selector(lambda row: {"selected_candidate_id": hard_selector.select(row["regime"])}, ds, eval_ids)

    near_tie = set(ds.near_tie_flags[
        (ds.near_tie_flags["threshold"] == 0.005)
        & (ds.near_tie_flags["near_tie"] == True)  # noqa: E712
        & (ds.near_tie_flags["window_id"].isin(eval_ids))
    ]["window_id"])
    verdict = determine_cc5_verdict(predictor_eval, best_fixed_eval, best_global_eval, hard_eval, near_tie)

    if "oracle_anwg" in ds.oracle_labels.columns:
        oracle_anwg = _oracle_composition_anwg(ds, eval_ids)
    else:
        oracle_anwg = (
            ds.per_window_results[ds.per_window_results["window_id"].isin(eval_ids)]
            .groupby("window_id")[PRIMARY_COL].max().to_dict()
        )

    regime_table = build_regime_table(
        predictor_eval=predictor_eval,
        best_fixed_eval=best_fixed_eval,
        best_global_eval=best_global_eval,
        hard_eval=hard_eval,
        oracle_anwg=oracle_anwg,
        ds=ds,
    )
    regime_classes = classify_uncertainty_behavior(regime_table)
    predictor_with_regret = regret_vs_oracle_fixed(ds, predictor_eval)

    # Coverage / error tables from calibrator.
    cal = artifact.uncertainty_calibrator
    assert cal is not None
    coverage_error = pd.DataFrame([
        {
            "method": name,
            **vals,
            "selected": name == cal.method,
        }
        for name, vals in meta["all_calibrators"].items()
    ])

    # Inference overhead measurement (mean over eval windows).
    overhead = float(rec_df["inference_overhead_s"].mean()) if not rec_df.empty else 0.0

    # Write artifacts
    (output_dir / "calibration_manifest.json").write_text(
        json.dumps(cal.calibration_manifest, indent=2, sort_keys=True, default=str) + "\n"
    )
    coverage_error.to_csv(output_dir / "coverage_error_tables.csv", index=False)
    threshold_grid.to_csv(output_dir / "uncertainty_threshold_grid.csv", index=False)
    predictor_with_regret.to_csv(output_dir / "per_window_predictions.csv", index=False)
    regime_table.to_csv(output_dir / "per_regime_summaries.csv", index=False)
    variant_summary.to_csv(output_dir / "fallback_comparisons.csv", index=False)
    pd.DataFrame([{"window_id": w, **bootstrap_ci([float(predictor_eval.loc[predictor_eval.window_id == w, PRIMARY_COL].iloc[0])])}
                  for w in eval_ids]).to_csv(output_dir / "confidence_intervals.csv", index=False)
    # Proper CI table for methods:
    ci_table = pd.DataFrame([
        {"method": "predictor_main", **verdict["predictor_anwg"]},
        {"method": "best_fixed", **verdict["best_fixed_anwg"]},
        {"method": "best_global_composition", **verdict["best_global_composition_anwg"]},
        {"method": "hard_selector", **verdict["existing_hard_selector_anwg"]},
        {"method": "oracle_composition", **bootstrap_ci([oracle_anwg[w] for w in eval_ids])},
    ])
    ci_table.to_csv(output_dir / "confidence_intervals.csv", index=False)

    diagnostics = predictor_eval[["window_id", "regime", "split", "abstained", "fallback_reason", "uncertainty", "ood_score", PRIMARY_COL, "regret"]].copy()
    diagnostics.to_csv(output_dir / "uncertainty_diagnostics.csv", index=False)

    (output_dir / "regime_fallback_rules.json").write_text(json.dumps(regime_rules, indent=2, sort_keys=True) + "\n")
    (output_dir / "regime_analysis.json").write_text(json.dumps(regime_classes, indent=2, sort_keys=True) + "\n")

    for name, df in per_variant.items():
        df.to_csv(output_dir / f"variant_{name}.csv", index=False)

    manifest = {
        "schema_version": 1,
        "experiment": "cc5_uncertainty_regime_refinement",
        "git_sha": git_state()["commit"],
        "dataset_dir": display_path(ds.dataset_dir),
        "dataset_config_hash": ds.manifest.get("config_hash", ""),
        "model_type": meta["best_model_name"],
        "cv_model_ranking": meta["cv_ranking"],
        "uncertainty_methods_tested": sorted(meta["all_calibrators"]),
        "selected_uncertainty_method": meta["selected_uncertainty_method"],
        "uncertainty_schema_version": UNCERTAINTY_SCHEMA_VERSION,
        "uncertainty_calibration": cal.calibration_manifest,
        "calibration_coverage": cal.empirical_coverage,
        "calibration_error": cal.calibration_error,
        "inference_overhead_s_mean": overhead,
        "main_variant": main_name,
        "fallback_policy": getattr(fallback_for_main, "selection", None),
        "fallback_policy_kind": getattr(fallback_for_main, "name", type(fallback_for_main).__name__),
        "gate_mode": gate_for_main,
        "regime_fallback_rules": regime_rules,
        "completion_safe_fallback_rules": completion_safe_rules,
        "regime_analysis": regime_classes,
        "fallback_variant_summary": variant_summary.to_dict(orient="records"),
        "ood_z_threshold": ood_z_threshold,
        "seed": seed,
        "n_bootstrap": n_bootstrap,
        "n_dev_windows": len(dev_ids),
        "n_val_windows": len(val_ids),
        "n_eval_windows": len(eval_ids),
        "no_live_api": True,
        "no_gpu": True,
        "no_real_vllm": True,
        "runtime_s": round(time.perf_counter() - t0, 3),
        "verdict": verdict,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
    (output_dir / "verdict.json").write_text(json.dumps(verdict, indent=2, sort_keys=True, default=str) + "\n")

    model_card = render_refinement_model_card(manifest, verdict, regime_table, variant_summary)
    (output_dir / "model_card.md").write_text(model_card)
    (output_dir / "replay_commands.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\ncd \"$(git rev-parse --show-toplevel)\"\n"
        f"python scripts/run_cc5_uncertainty_regime_refinement.py "
        f"--dataset-dir {display_path(ds.dataset_dir)} --full-run "
        f"--resume-dir {display_path(output_dir)}\n"
    )
    heartbeat(output_dir, "complete", verdict=verdict["status"])
    return RefinementResult(output_dir=output_dir, manifest=manifest, verdict=verdict)


def render_refinement_model_card(
    manifest: Mapping[str, Any],
    verdict: Mapping[str, Any],
    regime_table: pd.DataFrame,
    variant_summary: pd.DataFrame,
) -> str:
    lines = [
        "# CC5 Uncertainty / Regime Refinement Model Card",
        "",
        f"Model type: `{manifest['model_type']}`",
        f"Selected uncertainty: `{manifest['selected_uncertainty_method']}`",
        f"Calibration coverage: {manifest['calibration_coverage']:.4f}",
        f"Calibration error: {manifest['calibration_error']:.4f}",
        f"Main variant: `{manifest['main_variant']}`",
        f"Git SHA: `{manifest['git_sha']}`",
        "",
        "## Verdict",
        "",
        f"- Status: `{verdict['status']}`",
        f"- Reason: {verdict['reason']}",
        f"- Predictor ANWG: {verdict['predictor_anwg']}",
        f"- Best global composition ANWG: {verdict['best_global_composition_anwg']}",
        f"- Best fixed ANWG: {verdict['best_fixed_anwg']}",
        f"- Hard selector ANWG: {verdict['existing_hard_selector_anwg']}",
        f"- Completion violations: {verdict['completion_violations']}",
        "",
        "## Fallback Variants",
        "",
    ]
    for _, row in variant_summary.iterrows():
        lines.append(
            f"- `{row['variant']}`: ANWG={row['mean_anwg']:.4f} "
            f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}], abstention={row['abstention_rate']:.2f}"
        )
    lines += ["", "## Per-Regime Summary", ""]
    for _, row in regime_table.iterrows():
        lines.append(
            f"- `{row['regime']}` (n={int(row['window_count'])}): predictor={row['predictor_anwg']:.4f}, "
            f"global={row['best_global_composition_anwg']:.4f}, winner={row['winner_vs_global']}, "
            f"abstention={row['abstention_rate']:.2f}"
        )
    lines += ["", "## Reproduction", "", "```bash", "bash replay_commands.sh", "```", ""]
    return "\n".join(lines)
