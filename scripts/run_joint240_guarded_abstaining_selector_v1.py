#!/usr/bin/env python3
"""Joint-240 guarded abstaining selector v1 — matrix-only OOF reanalysis."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# Keep CPU light while other jobs may run
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from llmserveopt.analysis.joint240_guarded_abstaining_selector_v1 import (
    BOOTSTRAP_SEED,
    CATASTROPHIC_EPS,
    JOINT240_EXP,
    MARGIN_GRID,
    MAXPROB_GRID,
    N_BOOTSTRAP,
    SBS_POLICY,
    SCHEMA_VERSION,
    UTIL_ADV_GRID,
    assign_verdicts,
    paired_bootstrap_mean,
    run_oof_experiment,
    summarize_method,
)
from llmserveopt.analysis.joint240_same_distribution_adaptive_v1 import (
    FEATURE_ALLOWLIST,
    P6,
    generator_feature_table,
    load_utility_matrix,
    rebuild_all_scenarios,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "joint240_guarded_abstaining_selector_v1"
DESIGN = ROOT / "docs/design/JOINT240_GUARDED_ABSTAINING_SELECTOR_V1.md"
JOINT_DIR = ROOT / "experiments" / "joint_multimechanism_generalization_v1"
REPORT = ROOT / "docs/current/joint240_guarded_abstaining_selector_v1_analysis_20260825.md"


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")


def calibration_table(oof: pd.DataFrame, conf_col: str, anwg_col: str, n_bins: int = 8) -> pd.DataFrame:
    df = oof.copy()
    try:
        df["bin"] = pd.qcut(df[conf_col], q=n_bins, duplicates="drop")
    except ValueError:
        df["bin"] = pd.cut(df[conf_col], bins=n_bins)
    rows = []
    for b, g in df.groupby("bin", observed=True):
        rows.append(
            {
                "bin": str(b),
                "n": int(len(g)),
                "mean_confidence": float(g[conf_col].mean()),
                "frac_correct_unguarded": float(g["pred_correct_unguarded"].mean()),
                "mean_regret_vs_vbs": float((g["vbs_anwg"] - g[anwg_col]).mean()),
                "mean_specialist_adv_vs_sbs": float((g[anwg_col] - g["sbs_anwg"]).mean()),
                "frac_catastrophic": float((g[anwg_col] < g["sbs_anwg"] - CATASTROPHIC_EPS).mean()),
            }
        )
    return pd.DataFrame(rows)


def pressure_strata(oof: pd.DataFrame, anwg_col: str, abstain_col: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for flag in [
        "high_fairness_pressure",
        "high_service_heterogeneity",
        "high_prefill_decode_pressure",
        "high_kv_pressure",
        "high_urgency_pressure",
        "high_burst_pressure",
    ]:
        if flag not in oof.columns:
            continue
        for val, g in oof.groupby(flag):
            key = f"{flag}={val}"
            out[key] = {
                "n": int(len(g)),
                "frac_abstain": float(g[abstain_col].mean()),
                "mean_gain_vs_sbs": float((g[anwg_col] - g["sbs_anwg"]).mean()),
                "n_catastrophic": int((g[anwg_col] < g["sbs_anwg"] - CATASTROPHIC_EPS).sum()),
            }
    elev = {}
    for label, mask in [
        ("0_1", oof["n_elevated_mechanisms"] <= 1),
        ("2", oof["n_elevated_mechanisms"] == 2),
        ("3", oof["n_elevated_mechanisms"] == 3),
        ("ge4", oof["n_elevated_mechanisms"] >= 4),
    ]:
        g = oof.loc[mask]
        if len(g) == 0:
            continue
        elev[label] = {
            "n": int(len(g)),
            "frac_abstain": float(g[abstain_col].mean()),
            "mean_gain_vs_sbs": float((g[anwg_col] - g["sbs_anwg"]).mean()),
            "n_catastrophic": int((g[anwg_col] < g["sbs_anwg"] - CATASTROPHIC_EPS).sum()),
        }
    out["n_elevated_bins"] = elev
    return out


def loss_decomposition(oof: pd.DataFrame) -> Dict[str, Any]:
    """Where Ascen picked non-SBS and lost badly; did guards abstain?"""
    non_sbs = oof["unguarded_policy"] != SBS_POLICY
    loss = oof["unguarded_anwg"] < oof["sbs_anwg"]
    bad = non_sbs & loss
    very_bad = non_sbs & (oof["unguarded_anwg"] < oof["sbs_anwg"] - CATASTROPHIC_EPS)
    return {
        "n_ascen_non_sbs": int(non_sbs.sum()),
        "n_ascen_non_sbs_and_below_sbs": int(bad.sum()),
        "n_ascen_non_sbs_catastrophic": int(very_bad.sum()),
        "frac_maxprob_abstain_on_bad": float(oof.loc[bad, "maxprob_abstain"].mean()) if bad.any() else None,
        "frac_margin_abstain_on_bad": float(oof.loc[bad, "margin_abstain"].mean()) if bad.any() else None,
        "frac_util_choose_sbs_on_bad": float((oof.loc[bad, "util_policy"] == SBS_POLICY).mean()) if bad.any() else None,
        "frac_maxprob_abstain_on_catastrophic": float(oof.loc[very_bad, "maxprob_abstain"].mean()) if very_bad.any() else None,
        "frac_margin_abstain_on_catastrophic": float(oof.loc[very_bad, "margin_abstain"].mean()) if very_bad.any() else None,
        "mean_maxprob_on_catastrophic": float(oof.loc[very_bad, "maxprob"].mean()) if very_bad.any() else None,
        "mean_margin_on_catastrophic": float(oof.loc[very_bad, "margin"].mean()) if very_bad.any() else None,
    }


def write_report(summary: Dict[str, Any], out_dir: Path) -> None:
    methods = summary["methods"]
    boot = summary["bootstrap"]
    verd = summary["verdict"]
    lines = [
        "# Joint-240 Guarding / Abstaining Selector v1 — Analysis",
        "",
        f"**Date:** 2026-08-25  ",
        f"**Schema:** `{SCHEMA_VERSION}`  ",
        f"**Experiment:** `experiments/joint240_guarded_abstaining_selector_v1/`  ",
        f"**Preregistration:** `docs/design/JOINT240_GUARDED_ABSTAINING_SELECTOR_V1.md`",
        "",
        "## Verdict",
        "",
        f"- Labels: `{verd.get('labels')}`",
        f"- Best guarded method: `{verd.get('best_guarded_method')}`",
        f"- Best guarded gain vs SBS: {verd.get('best_guarded_gain'):.6f} "
        f"(CI [{verd.get('best_guarded_gain_ci', {}).get('ci95_low'):.6f}, "
        f"{verd.get('best_guarded_gain_ci', {}).get('ci95_high'):.6f}])",
        f"- Catastrophic: Ascen={verd.get('ascen_n_catastrophic')}, "
        f"best guarded={verd.get('best_guarded_n_catastrophic')}",
        "",
        "## Main OOF results",
        "",
        "| Method | R_A | Gain vs SBS | Gap vs VBS | GapClosure | Frac abstain | N catastrophic |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    order = ["SBS_fixed", "unguarded_ascen", "maxprob_guard", "margin_guard", "util_advantage_guard", "VBS_oracle"]
    for name in order:
        if name not in methods:
            continue
        m = methods[name]
        lines.append(
            f"| {name} | {m['R_A']:.6f} | {m.get('realized_gain_vs_sbs', float('nan')):.6f} | "
            f"{m.get('exploitability_gap', float('nan')):.6f} | {m.get('gap_closure', float('nan')):.4f} | "
            f"{m.get('frac_abstain_sbs', float('nan')):.3f} | {m.get('n_catastrophic', '—')} |"
        )
    lines += [
        "",
        "## Bootstrap CIs (gain vs SBS; B=2000 scenario-paired)",
        "",
    ]
    for name, ci in boot.get("gain_vs_sbs", {}).items():
        lines.append(
            f"- `{name}`: mean={ci['mean']:.6f}, CI95=[{ci['ci95_low']:.6f}, {ci['ci95_high']:.6f}]"
        )
    lines += [
        "",
        "### vs unguarded Ascen",
        "",
    ]
    for name, ci in boot.get("delta_vs_unguarded", {}).items():
        lines.append(
            f"- `{name} - Ascen`: mean={ci['mean']:.6f}, CI95=[{ci['ci95_low']:.6f}, {ci['ci95_high']:.6f}]"
        )
    lines += [
        "",
        "## Leakage checks",
        "",
        "- Thresholds chosen on inner VAL of training folds only (no outer-test outcomes).",
        "- Features = 17 allowlisted generator parameters only.",
        "- SBS fallback fixed globally as `kv_constrained_online`.",
        "- No VBS label / held-out best policy as input feature.",
        "",
        "## Loss decomposition (Ascen non-SBS mistakes)",
        "",
        "```json",
        json.dumps(summary.get("loss_decomposition", {}), indent=2),
        "```",
        "",
        "## Notes",
        "",
        "- Alive (0.283967) is online routing; this study guards **Ascen-style scenario selection** only.",
        "- Manuscript not edited.",
        "",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n")


def main() -> int:
    t0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(exist_ok=True)
    log_path = OUT / "logs" / "run.log"

    def log(msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}"
        print(line, flush=True)
        with log_path.open("a") as f:
            f.write(line + "\n")

    log(f"start schema={SCHEMA_VERSION}")
    if DESIGN.exists():
        (OUT / "DESIGN_FROZEN.md").write_text(DESIGN.read_text())

    matrix = load_utility_matrix()
    scenarios = rebuild_all_scenarios()
    feats = generator_feature_table(scenarios)
    folds = pd.read_csv(JOINT240_EXP / "split_oof_folds.csv")
    manifest = pd.read_csv(JOINT_DIR / "scenario_manifest.csv")
    data = (
        matrix.merge(feats, on="scenario_id", how="inner")
        .merge(folds[["scenario_id", "fold"]], on="scenario_id", how="inner")
        .merge(
            manifest[
                [
                    "scenario_id",
                    "fairness_pressure",
                    "service_heterogeneity",
                    "prefill_decode_pressure",
                    "kv_pressure",
                    "urgency_pressure",
                    "burst_pressure",
                    "high_fairness_pressure",
                    "high_service_heterogeneity",
                    "high_prefill_decode_pressure",
                    "high_kv_pressure",
                    "high_urgency_pressure",
                    "high_burst_pressure",
                ]
            ],
            on="scenario_id",
            how="left",
        )
    )
    assert len(data) == 240

    # Verify frozen Ascen reference
    prior = pd.read_csv(JOINT240_EXP / "per_scenario_oof_results.csv")
    prior_ascen = float(prior["a_scen_anwg"].mean())
    prior_sbs = float(prior["kv_constrained_online"].mean())
    prior_vbs = float(prior["vbs_anwg"].mean())
    prior_live = float(prior["a_live_anwg"].mean())
    log(
        f"prior Table4 SBS={prior_sbs:.10f} VBS={prior_vbs:.10f} "
        f"Ascen={prior_ascen:.10f} Alive={prior_live:.10f}"
    )

    config = {
        "schema_version": SCHEMA_VERSION,
        "sbs_policy_fixed": SBS_POLICY,
        "feature_allowlist": list(FEATURE_ALLOWLIST),
        "maxprob_grid": list(MAXPROB_GRID),
        "margin_grid": list(MARGIN_GRID),
        "util_adv_grid": list(UTIL_ADV_GRID),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "n_bootstrap": N_BOOTSTRAP,
        "catastrophic_eps": CATASTROPHIC_EPS,
        "threshold_protocol": (
            "per outer fold: fit on inner train; choose tau on inner VAL mean ANWG; "
            "refit on train+val; apply once to outer test"
        ),
        "p6": list(P6),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
    }
    _write_json(OUT / "config.json", config)
    _write_json(OUT / "threshold_grids.json", {
        "maxprob": list(MAXPROB_GRID),
        "margin": list(MARGIN_GRID),
        "util_advantage": list(UTIL_ADV_GRID),
    })

    log("running OOF guarded experiment...")
    result = run_oof_experiment(data)
    oof = result["oof"]
    tau_log = pd.DataFrame(result["tau_log"])
    oof.to_csv(OUT / "per_scenario_oof_results.csv", index=False)
    tau_log.to_csv(OUT / "per_fold_chosen_thresholds.csv", index=False)

    # Reproduce Ascen check
    repro_ascen = float(oof["unguarded_anwg"].mean())
    ascen_match = abs(repro_ascen - prior_ascen) < 1e-9
    log(f"repro unguarded Ascen={repro_ascen:.10f} match_prior={ascen_match}")

    sbs = oof["sbs_anwg"].to_numpy(float)
    vbs = oof["vbs_anwg"].to_numpy(float)

    methods = {
        "SBS_fixed": summarize_method(
            sbs, sbs, vbs, [SBS_POLICY] * len(oof), np.ones(len(oof), dtype=bool)
        ),
        "VBS_oracle": summarize_method(
            vbs, sbs, vbs, oof["vbs_policy"].tolist(), np.zeros(len(oof), dtype=bool)
        ),
        "unguarded_ascen": summarize_method(
            oof["unguarded_anwg"].to_numpy(float),
            sbs,
            vbs,
            oof["unguarded_policy"].tolist(),
            (oof["unguarded_policy"] == SBS_POLICY).to_numpy(),
        ),
        "maxprob_guard": summarize_method(
            oof["maxprob_anwg"].to_numpy(float),
            sbs,
            vbs,
            oof["maxprob_policy"].tolist(),
            oof["maxprob_abstain"].to_numpy(bool),
        ),
        "margin_guard": summarize_method(
            oof["margin_anwg"].to_numpy(float),
            sbs,
            vbs,
            oof["margin_policy"].tolist(),
            oof["margin_abstain"].to_numpy(bool),
        ),
        "util_advantage_guard": summarize_method(
            oof["util_anwg"].to_numpy(float),
            sbs,
            vbs,
            oof["util_policy"].tolist(),
            (oof["util_policy"] == SBS_POLICY).to_numpy(),
        ),
    }
    # Also report prior Alive as reference number (not recomputed here)
    methods["Alive_prior_reference"] = {
        "R_A": prior_live,
        "R_SBS": prior_sbs,
        "R_VBS": prior_vbs,
        "realized_gain_vs_sbs": prior_live - prior_sbs,
        "exploitability_gap": prior_vbs - prior_live,
        "note": "from frozen joint240 OOF; not recomputed in this matrix study",
    }

    boot_gain = {
        "unguarded_ascen": paired_bootstrap_mean(oof["unguarded_anwg"].to_numpy(float) - sbs),
        "maxprob_guard": paired_bootstrap_mean(oof["maxprob_anwg"].to_numpy(float) - sbs),
        "margin_guard": paired_bootstrap_mean(oof["margin_anwg"].to_numpy(float) - sbs),
        "util_advantage_guard": paired_bootstrap_mean(oof["util_anwg"].to_numpy(float) - sbs),
    }
    boot_vs_ascen = {
        "maxprob_guard": paired_bootstrap_mean(
            oof["maxprob_anwg"].to_numpy(float) - oof["unguarded_anwg"].to_numpy(float)
        ),
        "margin_guard": paired_bootstrap_mean(
            oof["margin_anwg"].to_numpy(float) - oof["unguarded_anwg"].to_numpy(float)
        ),
        "util_advantage_guard": paired_bootstrap_mean(
            oof["util_anwg"].to_numpy(float) - oof["unguarded_anwg"].to_numpy(float)
        ),
    }
    # catastrophic rate differences vs Ascen
    ascen_cat = (oof["unguarded_anwg"].to_numpy(float) < sbs - CATASTROPHIC_EPS).astype(float)
    boot_cat_delta = {}
    for name, col in [
        ("maxprob_guard", "maxprob_anwg"),
        ("margin_guard", "margin_anwg"),
        ("util_advantage_guard", "util_anwg"),
    ]:
        cat = (oof[col].to_numpy(float) < sbs - CATASTROPHIC_EPS).astype(float)
        boot_cat_delta[name] = paired_bootstrap_mean(cat - ascen_cat)

    verdict = assign_verdicts(
        methods, boot_gain, int(methods["unguarded_ascen"]["n_catastrophic"])
    )

    leakage = {
        "held_out_outcomes_used_for_tau": False,
        "future_output_length_in_features": False,
        "vbs_label_as_input_feature": False,
        "sbs_fallback_fixed_global": True,
        "sbs_policy": SBS_POLICY,
        "threshold_from_train_only_inner_val": True,
        "feature_allowlist": list(FEATURE_ALLOWLIST),
        "unguarded_ascen_reproduces_prior": ascen_match,
        "unguarded_ascen_mean": repro_ascen,
        "prior_ascen_mean": prior_ascen,
    }

    calib_max = calibration_table(oof, "maxprob", "unguarded_anwg")
    calib_margin = calibration_table(oof, "margin", "unguarded_anwg")
    calib_max.to_csv(OUT / "calibration_maxprob.csv", index=False)
    calib_margin.to_csv(OUT / "calibration_margin.csv", index=False)

    diagnostics = {
        "calibration_maxprob": calib_max.to_dict(orient="records"),
        "calibration_margin": calib_margin.to_dict(orient="records"),
        "loss_decomposition": loss_decomposition(oof),
        "pressure_maxprob": pressure_strata(oof, "maxprob_anwg", "maxprob_abstain"),
        "pressure_margin": pressure_strata(oof, "margin_anwg", "margin_abstain"),
        "pressure_util": pressure_strata(oof, "util_anwg", "util_abstain"),
    }
    _write_json(OUT / "diagnostics.json", diagnostics)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "n_scenarios": int(len(oof)),
        "table4_prior": {
            "SBS": prior_sbs,
            "VBS": prior_vbs,
            "Ascen": prior_ascen,
            "Alive": prior_live,
        },
        "methods": methods,
        "bootstrap": {
            "n": N_BOOTSTRAP,
            "seed": BOOTSTRAP_SEED,
            "gain_vs_sbs": boot_gain,
            "delta_vs_unguarded": boot_vs_ascen,
            "catastrophic_rate_delta_vs_ascen": boot_cat_delta,
        },
        "tau_log": result["tau_log"],
        "leakage_checks": leakage,
        "loss_decomposition": diagnostics["loss_decomposition"],
        "verdict": verdict,
        "wall_s": float(time.perf_counter() - t0),
    }
    _write_json(OUT / "summary.json", summary)
    _write_json(OUT / "bootstrap.json", summary["bootstrap"])

    # summary CSV
    rows = []
    for name, m in methods.items():
        if "R_A" not in m:
            continue
        rows.append({"method": name, **{k: v for k, v in m.items() if not isinstance(v, dict)}})
    pd.DataFrame(rows).to_csv(OUT / "summary.csv", index=False)

    provenance = {
        "joint_utility_sha256": sha256_file(JOINT_DIR / "utility_matrix_wide.csv"),
        "joint_manifest_sha256": sha256_file(JOINT_DIR / "scenario_manifest.csv"),
        "split_oof_folds": str(JOINT240_EXP / "split_oof_folds.csv"),
        "design": str(DESIGN.relative_to(ROOT)),
    }
    _write_json(OUT / "provenance.json", provenance)
    write_report(summary, OUT)
    (OUT / "DONE").write_text(json.dumps({"ok": True, "wall_s": summary["wall_s"]}, indent=2) + "\n")
    log(f"DONE wall_s={summary['wall_s']:.2f} verdict={verdict.get('labels')}")
    log(
        "best={best} gain={gain:.6f} CI=[{lo:.6f},{hi:.6f}]".format(
            best=verdict.get("best_guarded_method"),
            gain=float(verdict.get("best_guarded_gain", float("nan"))),
            lo=float(verdict.get("best_guarded_gain_ci", {}).get("ci95_low", float("nan"))),
            hi=float(verdict.get("best_guarded_gain_ci", {}).get("ci95_high", float("nan"))),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
