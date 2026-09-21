"""CC5 finalization: paired statistical analysis and a frozen, deterministic
regime-specific operating envelope.

Tightly scoped follow-up to the CC5 uncertainty/regime refinement. Does not
retrain or redesign the predictor pipeline: reuses
``cc5_contextual_predictor``'s model fitting/gating primitives and
``cc5_uncertainty_regime_refinement``'s deployable-artifact builder and
completion-safe fallback rules unchanged.

Two pieces of new analysis:

1. Paired statistical tests (bootstrap CI of the paired difference,
   paired sign-flip permutation test, Cohen's d, win/tie/loss) between the
   unrestricted contextual predictor and each baseline on the untouched
   held-out evaluation windows -- overall, non-near-tie-only, ID-only,
   OOD-only, and per-regime.

2. A frozen operating envelope: which regimes to trust the predictor in,
   decided using *only* development-split (TRAIN+VALIDATION) evidence via
   leave-one-window-out (LOWO) comparison against baselines refit on the
   other development windows for each fold. No evaluation-split window is
   ever touched while deciding envelope membership. The resulting gate is
   deterministic and versioned; it is evaluated on held-out data exactly
   once, with no further adjustment.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

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
    bootstrap_ci,
    build_regret_training_table,
    evaluate_selector,
    fit_best_fixed_policy,
    fit_best_global_composition,
    fit_existing_hard_selector,
    load_cc4_dataset,
    regret_vs_oracle_fixed,
    select_candidate_for_window,
    select_composition_with_fallback,
    validate_cc4_dataset,
    _actual_metrics,
)
from llmserveopt.experiments.cc5_uncertainty_regime_refinement import (
    HybridLookupBaseline,
    build_deployable_artifact,
    fit_completion_safe_fallback_rules,
    _oracle_composition_anwg,
)

ENVELOPE_SCHEMA_VERSION = 1
DEFAULT_DATASET = "results/cc4b_oracle_composition_expansion/20260803T182426Z"
DEFAULT_OUTPUT_ROOT = "results/cc5_final_operating_envelope"
NEAR_TIE_DIFF_EPS = 0.005  # matches CC4's own near_tie threshold column
MIN_ENVELOPE_DEV_WINDOWS = 2


# ---------------------------------------------------------------------------
# Paired statistical analysis
# ---------------------------------------------------------------------------


def paired_bootstrap_ci(a: Sequence[float], b: Sequence[float], *, n_boot: int = 5000, seed: int = 0) -> dict[str, float]:
    """Bootstrap CI of the paired mean difference a-b, resampling window
    indices jointly (never resampling a and b independently)."""
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    if len(a_arr) != len(b_arr):
        raise CC5Error("paired_bootstrap_ci requires equal-length paired sequences")
    diff = a_arr - b_arr
    n = len(diff)
    if n == 0:
        return {"mean_diff": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = diff[idx].mean(axis=1)
    return {
        "mean_diff": float(diff.mean()),
        "ci_low": float(np.percentile(boots, 2.5)),
        "ci_high": float(np.percentile(boots, 97.5)),
        "n": n,
    }


def paired_permutation_test(a: Sequence[float], b: Sequence[float], *, n_perm: int = 10000, seed: int = 0) -> dict[str, float]:
    """Two-sided paired sign-flip (randomization) test: under the null that
    predictor and baseline are exchangeable per window, the sign of each
    window's paired difference is randomized n_perm times."""
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    diff = a_arr - b_arr
    n = len(diff)
    if n == 0:
        return {"observed_mean_diff": 0.0, "p_value_two_sided": 1.0, "n_perm": n_perm, "n": 0}
    observed = float(diff.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, n))
    perm_means = (diff[None, :] * signs).mean(axis=1)
    count = int((np.abs(perm_means) >= abs(observed) - 1e-12).sum())
    p_value = count / n_perm
    return {"observed_mean_diff": observed, "p_value_two_sided": float(p_value), "n_perm": n_perm, "n": n}


def cohens_d_paired(a: Sequence[float], b: Sequence[float]) -> float:
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if len(diff) < 2:
        return 0.0
    sd = float(diff.std(ddof=1))
    if sd < 1e-12:
        return 0.0
    return float(diff.mean() / sd)


def win_tie_loss(a: Sequence[float], b: Sequence[float], *, tie_eps: float = NEAR_TIE_DIFF_EPS) -> dict[str, int]:
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    wins = int((diff > tie_eps).sum())
    losses = int((diff < -tie_eps).sum())
    ties = int(len(diff) - wins - losses)
    return {"wins": wins, "ties": ties, "losses": losses, "n": int(len(diff))}


def paired_comparison(
    comparison: str,
    subset: str,
    predictor_by_w: Mapping[str, float],
    baseline_by_w: Mapping[str, float],
    window_ids: Sequence[str],
    *,
    seed: int = 0,
) -> dict[str, Any]:
    """One row of the paired-statistics table for a given (comparison,
    subset) pair, e.g. (predictor_vs_best_global, non_near_tie)."""
    wids = [w for w in window_ids if w in predictor_by_w and w in baseline_by_w]
    a = [predictor_by_w[w] for w in wids]
    b = [baseline_by_w[w] for w in wids]
    boot = paired_bootstrap_ci(a, b, seed=seed)
    perm = paired_permutation_test(a, b, seed=seed)
    wtl = win_tie_loss(a, b)
    return {
        "comparison": comparison,
        "subset": subset,
        "n": len(wids),
        "mean_diff": boot["mean_diff"],
        "ci_low": boot["ci_low"],
        "ci_high": boot["ci_high"],
        "distinguishable_from_zero": bool(boot["ci_low"] > 0 or boot["ci_high"] < 0),
        "p_value_two_sided": perm["p_value_two_sided"],
        "cohens_d": cohens_d_paired(a, b),
        "wins": wtl["wins"],
        "ties": wtl["ties"],
        "losses": wtl["losses"],
    }


def run_paired_statistical_analysis(
    *,
    predictor_eval: pd.DataFrame,
    best_fixed_eval: pd.DataFrame,
    best_global_eval: pd.DataFrame,
    hard_eval: pd.DataFrame,
    near_tie_windows: set[str],
    eval_ids: Sequence[str],
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (overall/subset table, per-regime table). Every comparison is
    against the same untouched held-out evaluation windows; no threshold or
    rule is fit here -- this function only measures."""
    pred_by_w = predictor_eval.set_index("window_id")[PRIMARY_COL].to_dict()
    fixed_by_w = best_fixed_eval.set_index("window_id")[PRIMARY_COL].to_dict()
    global_by_w = best_global_eval.set_index("window_id")[PRIMARY_COL].to_dict()
    hard_by_w = hard_eval.set_index("window_id")[PRIMARY_COL].to_dict()
    split_by_w = predictor_eval.set_index("window_id")["split"].to_dict()
    regime_by_w = predictor_eval.set_index("window_id")["regime"].to_dict()

    baselines = {
        "predictor_vs_best_global_composition": global_by_w,
        "predictor_vs_best_fixed_policy": fixed_by_w,
        "predictor_vs_hard_selector": hard_by_w,
    }
    subsets = {
        "overall": list(eval_ids),
        "non_near_tie": [w for w in eval_ids if w not in near_tie_windows],
        "id_only": [w for w in eval_ids if split_by_w.get(w) == "ID_TEST"],
        "ood_only": [w for w in eval_ids if split_by_w.get(w) == "OOD_TEST"],
    }
    rows = []
    for comparison, baseline_by_w in baselines.items():
        for subset_name, wids in subsets.items():
            rows.append(paired_comparison(comparison, subset_name, pred_by_w, baseline_by_w, wids, seed=seed))
    overall_table = pd.DataFrame(rows)

    regime_rows = []
    for comparison, baseline_by_w in baselines.items():
        for regime in sorted(set(regime_by_w.values())):
            wids = [w for w in eval_ids if regime_by_w.get(w) == regime]
            regime_rows.append({"regime": regime, **paired_comparison(comparison, "per_regime", pred_by_w, baseline_by_w, wids, seed=seed)})
    regime_table = pd.DataFrame(regime_rows)
    return overall_table, regime_table


# ---------------------------------------------------------------------------
# Frozen operating envelope (development-split evidence only)
# ---------------------------------------------------------------------------


def compute_dev_lowo_table(
    ds: CC4Dataset,
    encoder: Any,
    dev_ids: Sequence[str],
    model_factory: Callable[[], Any],
) -> pd.DataFrame:
    """Leave-one-development-window-out comparison of the predictor against
    baselines that are themselves refit on the other development windows
    for that fold. Never touches an evaluation-split window -- this is the
    only evidence used to decide the frozen envelope."""
    causal_by_window = ds.causal_features.set_index("window_id")
    rows = []
    for held_out in dev_ids:
        train_windows = [w for w in dev_ids if w != held_out]
        X_train, y_train, _ = build_regret_training_table(ds, encoder, train_windows)
        model = model_factory()
        model.fit(X_train, y_train)
        pred_cand = select_candidate_for_window(ds, encoder, model, held_out)
        pred_anwg = _actual_metrics(ds, held_out, pred_cand)[PRIMARY_COL]

        regime = str(causal_by_window.loc[held_out, "regime"])
        fb_fixed = fit_best_fixed_policy(ds, train_windows)
        fb_global = fit_best_global_composition(ds, train_windows)
        fb_hard = fit_existing_hard_selector(ds, train_windows)
        rows.append({
            "window_id": held_out,
            "regime": regime,
            "predictor_lowo_anwg": pred_anwg,
            "best_fixed_lowo_anwg": _actual_metrics(ds, held_out, fb_fixed.select(regime))[PRIMARY_COL],
            "best_global_lowo_anwg": _actual_metrics(ds, held_out, fb_global.select(regime))[PRIMARY_COL],
            "hard_selector_lowo_anwg": _actual_metrics(ds, held_out, fb_hard.select(regime))[PRIMARY_COL],
        })
    return pd.DataFrame(rows)


def freeze_operating_envelope(
    dev_lowo: pd.DataFrame,
    all_regimes: Sequence[str],
    *,
    min_windows: int = MIN_ENVELOPE_DEV_WINDOWS,
) -> dict[str, Any]:
    """A regime enters the trusted envelope only if (a) it has at least
    `min_windows` development-split windows -- so the comparison is not a
    single-sample fluke -- and (b) the LOWO predictor beats the LOWO
    best-global-composition baseline on mean ANWG across those development
    windows. Regimes absent from development data (pure-OOD-only regimes)
    are excluded by construction: there is no development evidence to
    trust them on, so they default to the verified fallback."""
    rows = []
    grouped = {regime: g for regime, g in dev_lowo.groupby("regime")}
    for regime in sorted(all_regimes):
        g = grouped.get(regime)
        if g is None or g.empty:
            rows.append({
                "regime": regime, "n_dev_windows": 0,
                "predictor_lowo_anwg": None, "best_global_lowo_anwg": None, "best_fixed_lowo_anwg": None,
                "trust_predictor": False, "reason": "no_development_windows",
            })
            continue
        n = len(g)
        pred_mean = float(g["predictor_lowo_anwg"].mean())
        global_mean = float(g["best_global_lowo_anwg"].mean())
        fixed_mean = float(g["best_fixed_lowo_anwg"].mean())
        if n < min_windows:
            trust, reason = False, f"insufficient_development_windows(n={n}<{min_windows})"
        elif pred_mean >= global_mean:
            trust, reason = True, "lowo_predictor_beats_lowo_best_global_composition"
        else:
            trust, reason = False, "lowo_predictor_below_lowo_best_global_composition"
        rows.append({
            "regime": regime, "n_dev_windows": n,
            "predictor_lowo_anwg": pred_mean, "best_global_lowo_anwg": global_mean, "best_fixed_lowo_anwg": fixed_mean,
            "trust_predictor": trust, "reason": reason,
        })
    table = pd.DataFrame(rows).sort_values("regime").reset_index(drop=True)
    trusted = sorted(table.loc[table["trust_predictor"], "regime"])
    return {"table": table, "trusted_regimes": trusted}


@dataclass
class FrozenEnvelopeGate:
    """Deterministic, versioned regime-gating policy. Frozen at fit time
    from development-split evidence only; never adjusted using held-out
    results."""

    schema_version: int
    envelope_version: int
    trusted_regimes: tuple[str, ...]
    fitted_at: str
    dataset_config_hash: str
    dev_window_count: int
    decision_basis: str

    def in_envelope(self, regime: str) -> bool:
        return regime in self.trusted_regimes

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "envelope_version": self.envelope_version,
            "trusted_regimes": list(self.trusted_regimes),
            "fitted_at": self.fitted_at,
            "dataset_config_hash": self.dataset_config_hash,
            "dev_window_count": self.dev_window_count,
            "decision_basis": self.decision_basis,
        }


def assert_envelope_compatible(gate: FrozenEnvelopeGate | None) -> None:
    if gate is None:
        raise CC5Error("missing frozen operating-envelope gate -- stale artifact rejected")
    if gate.schema_version != ENVELOPE_SCHEMA_VERSION:
        raise CC5Error(
            f"stale operating-envelope schema {gate.schema_version} (expected {ENVELOPE_SCHEMA_VERSION})"
        )


def select_with_frozen_envelope(
    gate: FrozenEnvelopeGate,
    artifact: PredictorArtifact,
    ds: CC4Dataset,
    causal_row: Mapping[str, Any],
    *,
    fallback: LookupBaseline,
) -> dict[str, Any]:
    """Deployable policy:

    if workload context (regime) is inside the validated contextual
    operating envelope and uncertainty/OOD checks pass:
        use contextual composition predictor
    else:
        use verified fallback (validation-tuned completion-safe choice
        between best-global and best-fixed)
    """
    assert_envelope_compatible(gate)
    regime = str(causal_row["regime"] if hasattr(causal_row, "index") else causal_row.get("regime"))
    in_envelope = gate.in_envelope(regime)
    decision = select_composition_with_fallback(
        artifact, ds, causal_row, gate_mode="ood_or_uncertainty", fallback_override=fallback,
    )
    uncertainty_ood_ok = not decision["abstained"]
    used_predictor = in_envelope and uncertainty_ood_ok
    selected = decision["model_recommended_candidate_id"] if used_predictor else fallback.select(regime)

    reasons: list[str] = []
    if not in_envelope:
        reasons.append("regime_outside_envelope")
    if not uncertainty_ood_ok:
        reasons.append(decision["fallback_reason"] or "uncertainty_or_ood")

    return {
        "selected_candidate_id": selected,
        "model_recommended_candidate_id": decision["model_recommended_candidate_id"],
        "in_envelope": in_envelope,
        "uncertainty_ood_ok": uncertainty_ood_ok,
        "used_predictor": used_predictor,
        "abstained": not used_predictor,
        "fallback_reason": ",".join(reasons) if reasons else None,
        "envelope_version": gate.envelope_version,
        "uncertainty": decision["uncertainty"],
        "ood_score": decision["ood_score"],
        "inference_overhead_s": decision["inference_overhead_s"],
    }


# ---------------------------------------------------------------------------
# Final CC5 verdict
# ---------------------------------------------------------------------------

FINAL_VERDICTS = ("COMPLETE_FULL", "COMPLETE_REGIME_SPECIFIC", "STOP_OR_REDESIGN", "INCONCLUSIVE")


def determine_final_cc5_verdict(
    *,
    frozen_eval: pd.DataFrame,
    best_fixed_eval: pd.DataFrame,
    best_global_eval: pd.DataFrame,
    hard_eval: pd.DataFrame,
    trusted_regimes: Sequence[str],
    n_eval: int,
    paired_overall: pd.DataFrame,
) -> dict[str, Any]:
    """Compute (not presuppose) the final CC5 classification from the
    frozen system's held-out numbers.

    Superiority claims ("beats X") are decided from the PAIRED statistical
    test (paired bootstrap CI of the frozen-system-minus-X difference
    excludes zero), not from independent-CI point-estimate comparison --
    paired analysis is the correct, higher-powered test here because the
    two systems are evaluated on the identical set of windows. A point
    estimate in favor of the frozen system that the paired test cannot
    distinguish from zero is reported as a tie, not a win.
    """
    frozen_ci = bootstrap_ci(frozen_eval[PRIMARY_COL].tolist())
    fixed_ci = bootstrap_ci(best_fixed_eval[PRIMARY_COL].tolist())
    global_ci = bootstrap_ci(best_global_eval[PRIMARY_COL].tolist())
    selector_ci = bootstrap_ci(hard_eval[PRIMARY_COL].tolist())

    def _paired_row(comparison: str) -> pd.Series:
        rows = paired_overall[(paired_overall["comparison"] == comparison) & (paired_overall["subset"] == "overall")]
        if rows.empty:
            raise CC5Error(f"missing paired overall row for comparison={comparison!r}")
        return rows.iloc[0]

    vs_global = _paired_row("predictor_vs_best_global_composition")
    vs_fixed = _paired_row("predictor_vs_best_fixed_policy")
    vs_selector = _paired_row("predictor_vs_hard_selector")

    beats_fixed_significant = bool(vs_fixed["ci_low"] > 0)
    beats_global_significant = bool(vs_global["ci_low"] > 0)
    competitive_with_selector = bool(vs_selector["ci_low"] > 0 or vs_selector["ci_high"] >= -0.01)

    completion_violations = int((
        frozen_eval[COMPLETION_COL].to_numpy()
        < best_fixed_eval.set_index("window_id").loc[frozen_eval["window_id"], COMPLETION_COL].to_numpy() - 0.05
    ).sum())

    if completion_violations > 0:
        status = "STOP_OR_REDESIGN"
        reason = f"{completion_violations} evaluation window(s) show a completion-fraction regression >0.05 vs best fixed"
    elif not beats_fixed_significant:
        status = "STOP_OR_REDESIGN"
        reason = (
            "frozen regime-specific system's advantage over best fixed policy is not statistically "
            "distinguishable from zero on the paired held-out comparison"
        )
    elif not trusted_regimes:
        status = "INCONCLUSIVE"
        reason = "no regime has sufficient development-split evidence to trust the predictor -- envelope is empty"
    elif beats_global_significant and competitive_with_selector:
        status = "COMPLETE_FULL"
        reason = (
            "frozen system's advantage over best fixed AND best global composition is statistically "
            "distinguishable from zero on the paired held-out comparison, and it remains competitive "
            "with the hard selector"
        )
    elif beats_fixed_significant and competitive_with_selector:
        status = "COMPLETE_REGIME_SPECIFIC"
        reason = (
            "frozen system statistically beats best fixed policy and remains competitive with the hard "
            "selector, with a validated non-empty operating envelope and zero completion violations; its "
            f"point-estimate edge over best global composition (mean diff {vs_global['mean_diff']:+.4f}) is "
            f"NOT statistically distinguishable from zero (paired 95% CI [{vs_global['ci_low']:+.4f}, "
            f"{vs_global['ci_high']:+.4f}], p={vs_global['p_value_two_sided']:.4f}) -- full-context "
            "superiority over best global composition is not established"
        )
    else:
        status = "INCONCLUSIVE"
        reason = "frozen system's held-out evidence does not cleanly support any other classification"

    return {
        "status": status,
        "reason": reason,
        "n_evaluation_windows": n_eval,
        "trusted_regimes": list(trusted_regimes),
        "frozen_system_anwg": frozen_ci,
        "best_fixed_anwg": fixed_ci,
        "best_global_composition_anwg": global_ci,
        "existing_hard_selector_anwg": selector_ci,
        "completion_violations": completion_violations,
        "paired_vs_best_fixed": vs_fixed.to_dict(),
        "paired_vs_best_global_composition": vs_global.to_dict(),
        "paired_vs_hard_selector": vs_selector.to_dict(),
        "beats_fixed": bool(beats_fixed_significant),
        "beats_global_overall": bool(beats_global_significant),
        "competitive_with_hard_selector": bool(competitive_with_selector),
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


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


@dataclass
class FinalizationResult:
    output_dir: Path
    manifest: dict[str, Any]
    verdict: dict[str, Any]


def run_finalization(
    *,
    dataset_dir: str | Path = DEFAULT_DATASET,
    output_root: str = DEFAULT_OUTPUT_ROOT,
    timestamp: str | None = None,
    resume_dir: str | Path | None = None,
    seed: int = 0,
    ood_z_threshold: float = 2.0,
    n_bootstrap: int = 12,
    min_envelope_dev_windows: int = MIN_ENVELOPE_DEV_WINDOWS,
) -> FinalizationResult:
    ds = load_cc4_dataset(dataset_dir)
    validate_cc4_dataset(ds)

    output_dir = Path(resume_dir) if resume_dir is not None else resolve_output_dir(output_root, timestamp=timestamp)
    if (output_dir / "manifest.json").exists():
        manifest = json.loads((output_dir / "manifest.json").read_text())
        verdict = json.loads((output_dir / "verdict.json").read_text())
        return FinalizationResult(output_dir=output_dir, manifest=manifest, verdict=verdict)
    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    heartbeat(output_dir, "start")

    dev_ids = sorted(ds.causal_features[ds.causal_features["split"].isin(ds.development_splits)]["window_id"])
    val_ids = sorted(ds.causal_features[ds.causal_features["split"] == "VALIDATION"]["window_id"])
    eval_ids = sorted(ds.causal_features[ds.causal_features["split"].isin(ds.evaluation_splits)]["window_id"])
    all_regimes = sorted(ds.causal_features["regime"].unique())

    # --- Rebuild the deployed artifact deterministically (unchanged fitting
    # criteria; not a redesign, a reproducible re-derivation, seed fixed). ---
    artifact, meta, threshold_grid, best_fixed, best_global, hard_selector = build_deployable_artifact(
        ds, seed=seed, ood_z_threshold=ood_z_threshold, n_bootstrap=n_bootstrap,
    )
    heartbeat(output_dir, "artifact_ready", model=meta["best_model_name"])

    completion_safe_rules = fit_completion_safe_fallback_rules(
        ds=ds, val_ids=val_ids, best_fixed=best_fixed, best_global=best_global,
    )
    hybrid_fallback = HybridLookupBaseline(
        name="completion_safe_global_or_fixed", rules=completion_safe_rules,
        best_fixed=best_fixed, best_global=best_global,
    )

    # --- Unrestricted contextual predictor (existing OOD+uncertainty gate,
    # hybrid completion-safe fallback) evaluated once on held-out data. ---
    predictor_eval = evaluate_selector(
        lambda row: select_composition_with_fallback(artifact, ds, row, gate_mode="ood_or_uncertainty", fallback_override=hybrid_fallback),
        ds, eval_ids,
    )
    best_fixed_eval = evaluate_selector(lambda row: {"selected_candidate_id": best_fixed.select(row["regime"])}, ds, eval_ids)
    best_global_eval = evaluate_selector(lambda row: {"selected_candidate_id": best_global.select(row["regime"])}, ds, eval_ids)
    hard_eval = evaluate_selector(lambda row: {"selected_candidate_id": hard_selector.select(row["regime"])}, ds, eval_ids)

    if "oracle_anwg" in ds.oracle_labels.columns:
        oracle_anwg = _oracle_composition_anwg(ds, eval_ids)
    else:
        oracle_anwg = (
            ds.per_window_results[ds.per_window_results["window_id"].isin(eval_ids)]
            .groupby("window_id")[PRIMARY_COL].max().to_dict()
        )
    oracle_eval = pd.DataFrame([
        {"window_id": w, PRIMARY_COL: oracle_anwg[w]} for w in eval_ids
    ])

    near_tie_windows = set(ds.near_tie_flags[
        (ds.near_tie_flags["threshold"] == 0.005)
        & (ds.near_tie_flags["near_tie"] == True)  # noqa: E712
        & (ds.near_tie_flags["window_id"].isin(eval_ids))
    ]["window_id"])

    heartbeat(output_dir, "unrestricted_predictor_evaluated")

    # --- Step 2: paired statistical analysis on held-out data (measurement
    # only; no threshold is fit here). ---
    paired_overall, paired_regime = run_paired_statistical_analysis(
        predictor_eval=predictor_eval, best_fixed_eval=best_fixed_eval, best_global_eval=best_global_eval,
        hard_eval=hard_eval, near_tie_windows=near_tie_windows, eval_ids=eval_ids, seed=seed,
    )
    heartbeat(output_dir, "paired_statistics_computed")

    # --- Step 3: freeze the operating envelope using development-split
    # (never evaluation-split) LOWO evidence only. ---
    from llmserveopt.experiments.cc5_contextual_predictor import FeatureEncoder, build_regret_regressor_factories
    encoder = FeatureEncoder.fit(ds.causal_features[ds.causal_features["window_id"].isin(dev_ids)])
    factories = build_regret_regressor_factories(seed=seed)
    dev_lowo = compute_dev_lowo_table(ds, encoder, dev_ids, factories[meta["best_model_name"]])
    envelope = freeze_operating_envelope(dev_lowo, all_regimes, min_windows=min_envelope_dev_windows)
    gate = FrozenEnvelopeGate(
        schema_version=ENVELOPE_SCHEMA_VERSION,
        envelope_version=1,
        trusted_regimes=tuple(envelope["trusted_regimes"]),
        fitted_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        dataset_config_hash=ds.manifest.get("config_hash", ""),
        dev_window_count=len(dev_ids),
        decision_basis="development_lowo_evidence_only",
    )
    heartbeat(output_dir, "envelope_frozen", trusted_regimes=list(gate.trusted_regimes))

    # --- Step 4: evaluate the frozen system on held-out data exactly once,
    # with no further adjustment. ---
    frozen_eval = evaluate_selector(
        lambda row: select_with_frozen_envelope(gate, artifact, ds, row, fallback=hybrid_fallback),
        ds, eval_ids,
    )
    heartbeat(output_dir, "frozen_system_evaluated")

    frozen_paired_overall, frozen_paired_regime = run_paired_statistical_analysis(
        predictor_eval=frozen_eval, best_fixed_eval=best_fixed_eval, best_global_eval=best_global_eval,
        hard_eval=hard_eval, near_tie_windows=near_tie_windows, eval_ids=eval_ids, seed=seed,
    )
    # Final verdict uses the FROZEN system's own paired statistics (not the
    # unrestricted predictor's) -- it is the frozen system being classified.
    final_verdict = determine_final_cc5_verdict(
        frozen_eval=frozen_eval, best_fixed_eval=best_fixed_eval, best_global_eval=best_global_eval,
        hard_eval=hard_eval, trusted_regimes=gate.trusted_regimes, n_eval=len(eval_ids),
        paired_overall=frozen_paired_overall,
    )
    # Tag which analysis these paired rows describe (unrestricted predictor
    # vs frozen envelope system) before concatenating for a single artifact.
    paired_overall = paired_overall.assign(system="unrestricted_predictor")
    paired_regime = paired_regime.assign(system="unrestricted_predictor")
    frozen_paired_overall = frozen_paired_overall.assign(system="frozen_envelope_system")
    frozen_paired_regime = frozen_paired_regime.assign(system="frozen_envelope_system")
    paired_stats_table = pd.concat([paired_overall, frozen_paired_overall], ignore_index=True)
    paired_regime_table = pd.concat([paired_regime, frozen_paired_regime], ignore_index=True)

    split_by_w = frozen_eval.set_index("window_id")["split"].to_dict()
    id_eval = frozen_eval[frozen_eval["window_id"].map(split_by_w) == "ID_TEST"]
    ood_eval = frozen_eval[frozen_eval["window_id"].map(split_by_w) == "OOD_TEST"]
    non_near_tie_eval = frozen_eval[~frozen_eval["window_id"].isin(near_tie_windows)]
    final_verdict["frozen_system_id_anwg"] = bootstrap_ci(id_eval[PRIMARY_COL].tolist())
    final_verdict["frozen_system_ood_anwg"] = bootstrap_ci(ood_eval[PRIMARY_COL].tolist())
    final_verdict["frozen_system_non_near_tie_anwg"] = bootstrap_ci(non_near_tie_eval[PRIMARY_COL].tolist())
    final_verdict["oracle_composition_anwg"] = bootstrap_ci(oracle_eval[PRIMARY_COL].tolist())
    final_verdict["unrestricted_predictor_anwg"] = bootstrap_ci(predictor_eval[PRIMARY_COL].tolist())
    final_verdict["frozen_system_abstention_rate"] = float(frozen_eval["abstained"].mean())
    final_verdict["frozen_system_fallback_rate"] = float(frozen_eval["abstained"].mean())
    final_verdict["frozen_system_completion_fraction"] = float(frozen_eval[COMPLETION_COL].mean())
    heartbeat(output_dir, "final_verdict_determined", status=final_verdict["status"])

    # --- Comparison table across all six systems ---
    comparison = pd.DataFrame([
        {"system": "best_fixed_policy", **bootstrap_ci(best_fixed_eval[PRIMARY_COL].tolist())},
        {"system": "hard_selector", **bootstrap_ci(hard_eval[PRIMARY_COL].tolist())},
        {"system": "best_global_composition", **bootstrap_ci(best_global_eval[PRIMARY_COL].tolist())},
        {"system": "unrestricted_contextual_predictor", **bootstrap_ci(predictor_eval[PRIMARY_COL].tolist())},
        {"system": "frozen_regime_specific_system", **bootstrap_ci(frozen_eval[PRIMARY_COL].tolist())},
        {"system": "oracle_composition", **bootstrap_ci(oracle_eval[PRIMARY_COL].tolist())},
    ])

    per_regime_rows = []
    for regime, group in frozen_eval.groupby("regime"):
        wids = list(group["window_id"])
        per_regime_rows.append({
            "regime": regime,
            "window_count": len(wids),
            "in_envelope": gate.in_envelope(regime),
            "frozen_system_anwg": float(group[PRIMARY_COL].mean()),
            "unrestricted_predictor_anwg": float(predictor_eval.set_index("window_id").loc[wids, PRIMARY_COL].mean()),
            "best_global_anwg": float(best_global_eval.set_index("window_id").loc[wids, PRIMARY_COL].mean()),
            "best_fixed_anwg": float(best_fixed_eval.set_index("window_id").loc[wids, PRIMARY_COL].mean()),
            "hard_selector_anwg": float(hard_eval.set_index("window_id").loc[wids, PRIMARY_COL].mean()),
            "oracle_anwg": float(oracle_eval.set_index("window_id").loc[wids, PRIMARY_COL].mean()),
            "abstention_rate": float(group["abstained"].mean()),
            "fallback_rate": float(group["abstained"].mean()),
            "completion_fraction": float(group[COMPLETION_COL].mean()),
        })
    per_regime_table = pd.DataFrame(per_regime_rows).sort_values("regime").reset_index(drop=True)

    overhead_mean = float(frozen_eval.merge(
        pd.DataFrame([
            {"window_id": w, "inference_overhead_s": select_with_frozen_envelope(
                gate, artifact, ds, ds.causal_features.set_index("window_id").loc[w], fallback=hybrid_fallback,
            )["inference_overhead_s"]}
            for w in eval_ids
        ]),
        on="window_id",
    )["inference_overhead_s"].mean())

    # --- Write artifacts ---
    frozen_eval_with_regret = regret_vs_oracle_fixed(ds, frozen_eval)
    frozen_eval_with_regret.to_csv(output_dir / "per_window_predictions.csv", index=False)
    per_regime_table.to_csv(output_dir / "per_regime_summaries.csv", index=False)
    envelope["table"].to_csv(output_dir / "envelope_definition.csv", index=False)
    dev_lowo.to_csv(output_dir / "dev_lowo_table.csv", index=False)
    paired_stats_table.to_csv(output_dir / "paired_statistical_analysis.csv", index=False)
    paired_regime_table.to_csv(output_dir / "paired_regime_analysis.csv", index=False)
    comparison.to_csv(output_dir / "system_comparison.csv", index=False)
    (output_dir / "envelope_definition.json").write_text(json.dumps({
        **gate.as_dict(),
        "regime_table": envelope["table"].to_dict(orient="records"),
    }, indent=2, sort_keys=True, default=str) + "\n")
    (output_dir / "calibration_manifest.json").write_text(
        json.dumps(artifact.uncertainty_calibrator.calibration_manifest, indent=2, sort_keys=True, default=str) + "\n"
    )

    manifest = {
        "schema_version": 1,
        "experiment": "cc5_final_operating_envelope",
        "git_sha": git_state()["commit"],
        "dataset_dir": display_path(ds.dataset_dir),
        "dataset_config_hash": ds.manifest.get("config_hash", ""),
        "model_type": meta["best_model_name"],
        "uncertainty_method": meta["selected_uncertainty_method"],
        "envelope_schema_version": ENVELOPE_SCHEMA_VERSION,
        "envelope": gate.as_dict(),
        "min_envelope_dev_windows": min_envelope_dev_windows,
        "fallback_policy_kind": hybrid_fallback.name,
        "completion_safe_fallback_rules": completion_safe_rules,
        "n_dev_windows": len(dev_ids),
        "n_val_windows": len(val_ids),
        "n_eval_windows": len(eval_ids),
        "inference_overhead_s_mean": overhead_mean,
        "no_live_api": True,
        "no_gpu": True,
        "no_real_vllm": True,
        "runtime_s": round(time.perf_counter() - t0, 3),
        "verdict": final_verdict,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
    (output_dir / "verdict.json").write_text(json.dumps(final_verdict, indent=2, sort_keys=True, default=str) + "\n")
    model_card = render_finalization_model_card(manifest, final_verdict, per_regime_table, comparison)
    (output_dir / "model_card.md").write_text(model_card)
    (output_dir / "replay_commands.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\ncd \"$(git rev-parse --show-toplevel)\"\n"
        f"python scripts/run_cc5_final_operating_envelope.py "
        f"--dataset-dir {display_path(ds.dataset_dir)} --full-run "
        f"--resume-dir {display_path(output_dir)}\n"
    )
    heartbeat(output_dir, "complete", verdict=final_verdict["status"])
    return FinalizationResult(output_dir=output_dir, manifest=manifest, verdict=final_verdict)


def render_finalization_model_card(
    manifest: Mapping[str, Any],
    verdict: Mapping[str, Any],
    per_regime: pd.DataFrame,
    comparison: pd.DataFrame,
) -> str:
    lines = [
        "# CC5 Final Operating Envelope Model Card",
        "",
        f"Model type: `{manifest['model_type']}`",
        f"Uncertainty method: `{manifest['uncertainty_method']}`",
        f"Envelope version: {manifest['envelope']['envelope_version']} (schema {manifest['envelope_schema_version']})",
        f"Trusted regimes: {manifest['envelope']['trusted_regimes']}",
        f"Git SHA: `{manifest['git_sha']}`",
        "",
        "## Final Verdict",
        "",
        f"- Status: `{verdict['status']}`",
        f"- Reason: {verdict['reason']}",
        f"- Evaluation windows: {verdict['n_evaluation_windows']}",
        f"- Completion violations: {verdict['completion_violations']}",
        f"- Frozen system ANWG: {verdict['frozen_system_anwg']}",
        f"- Best global composition ANWG: {verdict['best_global_composition_anwg']}",
        f"- Best fixed ANWG: {verdict['best_fixed_anwg']}",
        f"- Hard selector ANWG: {verdict['existing_hard_selector_anwg']}",
        "",
        "## System Comparison",
        "",
    ]
    for _, row in comparison.iterrows():
        lines.append(f"- `{row['system']}`: ANWG={row['mean']:.4f} [{row['ci_low']:.4f}, {row['ci_high']:.4f}] (n={int(row['n'])})")
    lines += ["", "## Per-Regime Summary", ""]
    for _, row in per_regime.iterrows():
        lines.append(
            f"- `{row['regime']}` (n={int(row['window_count'])}, in_envelope={row['in_envelope']}): "
            f"frozen={row['frozen_system_anwg']:.4f}, global={row['best_global_anwg']:.4f}, "
            f"fallback_rate={row['fallback_rate']:.2f}"
        )
    lines += ["", "## Reproduction", "", "```bash", "bash replay_commands.sh", "```", ""]
    return "\n".join(lines)
