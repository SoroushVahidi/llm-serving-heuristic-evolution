#!/usr/bin/env python3
"""Terminal-ANWG one-step decision criticality on joint-240 v1."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from llmserveopt.analysis import decision_criticality_terminal_anwg_joint240_v1 as jtan  # noqa: E402

DESIGN = REPO / "docs/design/DECISION_CRITICALITY_TERMINAL_ANWG_JOINT240_V1.md"
OUT = REPO / "experiments/decision_criticality_terminal_anwg_joint240_v1"
PRIOR_SUMMARY = REPO / "experiments/decision_criticality_terminal_anwg_v1/summary.json"
PRIOR_BRANCHES = REPO / "experiments/decision_criticality_terminal_anwg_v1/branches.csv"


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()


def _git_dirty() -> bool:
    return bool(
        subprocess.check_output(
            ["git", "-C", str(REPO), "status", "--short"], text=True
        ).strip()
    )


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _ci(vals: List[float]) -> Dict[str, Optional[float]]:
    arr = np.asarray([v for v in vals if v is not None and np.isfinite(v)], dtype=float)
    if len(arr) == 0:
        return {"mean": None, "ci95_low": None, "ci95_high": None}
    return {
        "mean": float(np.mean(arr)),
        "ci95_low": float(np.quantile(arr, 0.025)),
        "ci95_high": float(np.quantile(arr, 0.975)),
    }


def analyze(branches: pd.DataFrame) -> dict:
    n = len(branches)
    abs_d = branches["abs_delta_anwg"].to_numpy(dtype=float) if n else np.asarray([])
    signed = branches["delta_anwg"].to_numpy(dtype=float) if n else np.asarray([])
    pos = np.maximum(signed, 0.0) if n else np.asarray([])

    prev = {
        "n_states": n,
        "frac_exact_zero": float((abs_d <= jtan.ANWG_EQ_ATOL).mean()) if n else None,
        "frac_nonzero": float((abs_d > jtan.ANWG_EQ_ATOL).mean()) if n else None,
        "frac_positive": float((signed > jtan.ANWG_EQ_ATOL).mean()) if n else None,
        "frac_negative": float((signed < -jtan.ANWG_EQ_ATOL).mean()) if n else None,
        "frac_abs_ge_0.01": float((abs_d >= 0.01).mean()) if n else None,
        "mean_delta": float(signed.mean()) if n else None,
        "median_delta": float(np.median(signed)) if n else None,
        "mean_abs_delta": float(abs_d.mean()) if n else None,
        "median_abs_delta": float(np.median(abs_d)) if n else None,
        "quantiles_abs_delta": {
            str(q): float(np.quantile(abs_d, q)) for q in (0.5, 0.9, 0.95, 0.99)
        }
        if n
        else {},
        "thresholds": {},
    }
    for t in jtan.PRACTICAL_THRESHOLDS:
        prev["thresholds"][str(t)] = float((abs_d >= t).mean()) if n else None

    by_acq: Dict[str, Any] = {}
    for acq, g in branches.groupby("acquisition_type") if n else []:
        a = g["abs_delta_anwg"].to_numpy(dtype=float)
        by_acq[str(acq)] = {
            "n": int(len(g)),
            "mean_abs": float(a.mean()),
            "median_abs": float(np.median(a)),
            "frac_nonzero": float((a > jtan.ANWG_EQ_ATOL).mean()),
            "thresholds": {
                str(t): float((a >= t).mean()) for t in jtan.PRACTICAL_THRESHOLDS
            },
        }

    # Scenario prevalence
    scen_prev = {"n_scenarios": 0, "frac_with_ge1_nonzero": None, "mean_nonzero_per_scenario": None}
    if n:
        g = branches.groupby("scenario_id")
        nz_per = g["abs_delta_anwg"].apply(lambda s: float((s > jtan.ANWG_EQ_ATOL).sum()))
        scen_prev = {
            "n_scenarios": int(len(nz_per)),
            "frac_with_ge1_nonzero": float((nz_per > 0).mean()),
            "mean_nonzero_per_scenario": float(nz_per.mean()),
            "mean_states_per_scenario": float(g.size().mean()),
        }

    # Disagreement proxy
    disagree_proxy: Dict[str, Any] = {"available": False}
    if n and "acquisition_type" in branches.columns:
        y = (branches["abs_delta_anwg"] > jtan.ANWG_EQ_ATOL).astype(int).to_numpy()
        s = (branches["acquisition_type"] == "DISAGREEMENT").astype(float).to_numpy()
        base = float(y.mean())
        d_mask = s == 1.0
        a_mask = s == 0.0
        prev_d = float(y[d_mask].mean()) if d_mask.any() else None
        prev_a = float(y[a_mask].mean()) if a_mask.any() else None
        enrichment = (
            float(prev_d / prev_a)
            if prev_d is not None and prev_a is not None and prev_a > 0
            else None
        )
        auroc = jtan.auroc_binary_score(y, s)
        auprc = jtan.auprc_binary_score(y, s)
        disagree_proxy = {
            "available": True,
            "base_prevalence": base,
            "prevalence_disagreement": prev_d,
            "prevalence_agreement_control": prev_a,
            "enrichment_ratio": enrichment,
            "auroc_disagreement_for_nonzero_abs_delta": auroc,
            "auprc_disagreement_for_nonzero_abs_delta": auprc,
            "n_positive": int(y.sum()),
            "n_negative": int((1 - y).sum()),
        }

    # Pressure strata (frozen flags)
    by_pressure: Dict[str, Any] = {}
    if n:
        for flag in jtan.PRESSURE_FLAGS:
            if flag not in branches.columns:
                continue
            for val, g in branches.groupby(flag):
                a = g["abs_delta_anwg"].to_numpy(dtype=float)
                by_pressure[f"{flag}={val}"] = {
                    "n": int(len(g)),
                    "frac_nonzero": float((a > jtan.ANWG_EQ_ATOL).mean()),
                    "mean_abs": float(a.mean()),
                }
        elev_bins = [
            ("0_1", branches["n_elevated_mechanisms"] <= 1),
            ("2", branches["n_elevated_mechanisms"] == 2),
            ("3", branches["n_elevated_mechanisms"] == 3),
            ("ge4", branches["n_elevated_mechanisms"] >= 4),
        ]
        by_elev = {}
        for label, mask in elev_bins:
            sub = branches.loc[mask]
            if len(sub) == 0:
                continue
            a = sub["abs_delta_anwg"].to_numpy(dtype=float)
            by_elev[label] = {
                "n": int(len(sub)),
                "frac_nonzero": float((a > jtan.ANWG_EQ_ATOL).mean()),
                "mean_abs": float(a.mean()),
            }
        by_pressure["n_elevated_mechanisms_bins"] = by_elev

    # Scenario mass
    if n:
        sc = (
            branches.groupby("scenario_id")["abs_delta_anwg"]
            .sum()
            .sort_values(ascending=False)
        )
        sc_vals = sc.to_numpy(dtype=float)
    else:
        sc = pd.Series(dtype=float)
        sc_vals = np.asarray([], dtype=float)

    sc_conc = jtan.concentration_curve(sc_vals)
    top_scen = {
        "top_1": jtan.scenario_top_k_share(sc_vals, 1),
        "top_2": jtan.scenario_top_k_share(sc_vals, 2),
        "top_5": jtan.scenario_top_k_share(sc_vals, 5),
        "top_5pct_scenarios": sc_conc.get("0.05", {}).get("share"),
        "top_10pct_scenarios": sc_conc.get("0.1", {}).get("share"),
    }

    # Divergence
    divergence: Dict[str, Any] = {"available": False}
    if n and "subsequent_trajectory_diverged" in branches.columns:
        nz = branches["abs_delta_anwg"] > jtan.ANWG_EQ_ATOL
        z = ~nz
        divergence = {
            "available": True,
            "rate_among_nonzero": float(
                branches.loc[nz, "subsequent_trajectory_diverged"].mean()
            )
            if nz.any()
            else None,
            "rate_among_zero": float(
                branches.loc[z, "subsequent_trajectory_diverged"].mean()
            )
            if z.any()
            else None,
            "mean_cf_extra_steps_nonzero": float(
                branches.loc[nz, "cf_extra_steps"].mean()
            )
            if nz.any() and "cf_extra_steps" in branches.columns
            else None,
            "median_cf_extra_steps_nonzero": float(
                branches.loc[nz, "cf_extra_steps"].median()
            )
            if nz.any() and "cf_extra_steps" in branches.columns
            else None,
            "intervention_step_quantiles": {
                str(q): float(np.quantile(branches["step"], q))
                for q in (0.1, 0.5, 0.9)
            }
            if "step" in branches.columns
            else {},
        }

    # H10 join — expected unavailable on joint-240
    h10_join = {
        "available": False,
        "reason": (
            "H10 completed-count events are from the A/B/C TRAIN/VAL timescale "
            "experiment; they do not share scenario_id/step with joint-240 Alive "
            "trajectories. No new proxy invented."
        ),
    }

    # Scenario-grouped bootstrap
    rng = np.random.default_rng(jtan.BOOTSTRAP_SEED)
    boot: Dict[str, Any] = {
        "n": jtan.N_BOOTSTRAP,
        "seed": jtan.BOOTSTRAP_SEED,
        "zero_mass_rule": "concentration_share=0.0 when total abs mass is 0",
    }
    if n:
        scen_ids = sc.index.to_numpy()
        by_scen = {sid: g for sid, g in branches.groupby("scenario_id")}
        # Per-scenario absolute mass for scenario-level concentration bootstrap.
        # IMPORTANT: with-replacement draws must retain multiplicity. Collapsing via
        # groupby(scenario_id) after concatenating states biases concentration upward
        # and can yield CIs that exclude the unique-scenario point estimate.
        scen_abs_mass = {
            sid: float(g["abs_delta_anwg"].sum()) for sid, g in by_scen.items()
        }
        b_prev, b_mean_abs, b_t1, b_t5, b_t10, b_top5sc = [], [], [], [], [], []
        b_top10sc = []
        b_enr, b_auroc, b_auprc = [], [], []
        for _ in range(jtan.N_BOOTSTRAP):
            draw = scen_ids[rng.integers(0, len(scen_ids), size=len(scen_ids))]
            parts = [by_scen[sid] for sid in draw]
            sample = pd.concat(parts, ignore_index=True)
            a = sample["abs_delta_anwg"].to_numpy(dtype=float)
            b_prev.append(float((a > jtan.ANWG_EQ_ATOL).mean()))
            b_mean_abs.append(float(a.mean()))
            conc = jtan.concentration_curve(a, fracs=(0.01, 0.05, 0.10))
            b_t1.append(float(conc["0.01"]["share"]))
            b_t5.append(float(conc["0.05"]["share"]))
            b_t10.append(float(conc["0.1"]["share"]))
            sc_mass = np.asarray([scen_abs_mass[sid] for sid in draw], dtype=float)
            b_top5sc.append(jtan.scenario_top_k_share(sc_mass, 5))
            b_top10sc.append(
                float(jtan.concentration_curve(sc_mass, fracs=(0.10,))["0.1"]["share"])
            )

            y = (sample["abs_delta_anwg"] > jtan.ANWG_EQ_ATOL).astype(int).to_numpy()
            s_sc = (sample["acquisition_type"] == "DISAGREEMENT").astype(float).to_numpy()
            d_m = s_sc == 1.0
            a_m = s_sc == 0.0
            pd_ = float(y[d_m].mean()) if d_m.any() else np.nan
            pa_ = float(y[a_m].mean()) if a_m.any() else np.nan
            b_enr.append(pd_ / pa_ if pa_ > 0 else np.nan)
            ar = jtan.auroc_binary_score(y, s_sc)
            ap = jtan.auprc_binary_score(y, s_sc)
            b_auroc.append(ar if ar is not None else np.nan)
            b_auprc.append(ap if ap is not None else np.nan)

        boot.update(
            {
                "nonzero_prevalence": _ci(b_prev),
                "mean_abs_delta": _ci(b_mean_abs),
                "top1pct_state_mass_share": _ci(b_t1),
                "top5pct_state_mass_share": _ci(b_t5),
                "top10pct_state_mass_share": _ci(b_t10),
                "top5_scenario_mass_share": _ci(b_top5sc),
                "top10pct_scenario_mass_share": _ci(b_top10sc),
                "disagreement_enrichment": _ci(b_enr),
                "auroc": _ci(b_auroc),
                "auprc": _ci(b_auprc),
                "scenario_bootstrap_note": (
                    "scenario-level concentration uses with-replacement scenario "
                    "masses with multiplicity retained (not groupby-collapsed)"
                ),
            }
        )

    # REF replay
    ref_replay = {"n_checks": 0, "n_match": 0, "max_abs_mismatch": None}
    if n and "ref_replay_anwg" in branches.columns:
        rr = branches[branches["ref_replay_anwg"].notna()]
        ref_replay["n_checks"] = int(len(rr))
        if len(rr):
            mism = (rr["ref_replay_anwg"] - rr["reference_anwg"]).abs()
            if "ref_replay_matches_reference" in rr.columns:
                ref_replay["n_match"] = int(rr["ref_replay_matches_reference"].fillna(False).sum())
            else:
                ref_replay["n_match"] = int((mism <= jtan.ANWG_EQ_ATOL).sum())
            ref_replay["max_abs_mismatch"] = float(mism.max())

    summary = {
        "prevalence": prev,
        "scenario_prevalence": scen_prev,
        "by_acquisition": by_acq,
        "by_pressure": by_pressure,
        "concentration_abs_delta_all_states": jtan.concentration_curve(abs_d),
        "concentration_positive_gain_all_states": jtan.concentration_curve(pos),
        "concentration_abs_delta_disagreement_only": jtan.concentration_curve(
            branches.loc[
                branches["acquisition_type"] == "DISAGREEMENT", "abs_delta_anwg"
            ].to_numpy(float)
            if n
            else np.asarray([])
        ),
        "scenario_concentration_abs_mass": sc_conc,
        "scenario_top_concentration": top_scen,
        "bootstrap": boot,
        "h10_proxy_join": h10_join,
        "disagreement_as_criticality_proxy": disagree_proxy,
        "closed_loop_divergence": divergence,
        "ref_replay": ref_replay,
    }
    # Attach bootstrap into proxy usefulness check
    summary["verdicts"] = jtan.assign_verdicts(summary)
    return summary


def compare_with_prior(summary: dict) -> dict:
    """Side-by-side contrast table vs A/B/C terminal-ANWG v1 (not pooled)."""
    prior = {}
    if PRIOR_SUMMARY.exists():
        prior = json.loads(PRIOR_SUMMARY.read_text())
    # Prefer re-deriving prior prevalence from branches if summary prevalence is corrupted
    prior_prev = prior.get("prevalence")
    if not isinstance(prior_prev, dict) and PRIOR_BRANCHES.exists():
        b = pd.read_csv(PRIOR_BRANCHES)
        abs_d = b["abs_delta_anwg"].to_numpy(float)
        signed = b["delta_anwg"].to_numpy(float)
        prior_prev = {
            "n_states": int(len(b)),
            "frac_nonzero": float((abs_d > jtan.ANWG_EQ_ATOL).mean()),
            "frac_positive": float((signed > jtan.ANWG_EQ_ATOL).mean()),
            "frac_negative": float((signed < -jtan.ANWG_EQ_ATOL).mean()),
            "mean_abs_delta": float(abs_d.mean()),
        }
        prior_conc = jtan.concentration_curve(abs_d)
        y = (abs_d > jtan.ANWG_EQ_ATOL).astype(int)
        s = (b["acquisition_type"] == "DISAGREEMENT").astype(float).to_numpy()
        prior_proxy = {
            "auroc": jtan.auroc_binary_score(y, s),
            "auprc": jtan.auprc_binary_score(y, s),
        }
        prior_div = None
        if "subsequent_trajectory_diverged" in b.columns:
            nz = abs_d > jtan.ANWG_EQ_ATOL
            prior_div = float(b.loc[nz, "subsequent_trajectory_diverged"].mean()) if nz.any() else None
    else:
        prior_conc = prior.get("concentration_abs_delta_all_states") or {}
        prior_proxy_raw = prior.get("disagreement_as_criticality_proxy") or {}
        prior_proxy = {
            "auroc": prior_proxy_raw.get("auroc_disagreement_for_nonzero_abs_delta"),
            "auprc": prior_proxy_raw.get("auprc_disagreement_for_nonzero_abs_delta"),
        }
        prior_div = (prior.get("closed_loop_divergence") or {}).get("rate_among_nonzero")
        if not isinstance(prior_prev, dict):
            prior_prev = {}

    cur_prev = summary.get("prevalence") or {}
    cur_conc = summary.get("concentration_abs_delta_all_states") or {}
    cur_proxy = summary.get("disagreement_as_criticality_proxy") or {}
    cur_div = summary.get("closed_loop_divergence") or {}
    prior_h10 = prior.get("h10_proxy_join") or {}

    return {
        "note": "Contrast only; corpora are not pooled.",
        "rows": [
            {
                "corpus": "A/B/C TRAIN/VAL (terminal_anwg_v1)",
                "scenarios": prior.get("n_scenarios_succeeded", 144),
                "acquired_states": prior_prev.get("n_states"),
                "nonzero_pct": None
                if prior_prev.get("frac_nonzero") is None
                else 100 * float(prior_prev["frac_nonzero"]),
                "positive_pct": None
                if prior_prev.get("frac_positive") is None
                else 100 * float(prior_prev["frac_positive"]),
                "negative_pct": None
                if prior_prev.get("frac_negative") is None
                else 100 * float(prior_prev["frac_negative"]),
                "mean_abs_delta": prior_prev.get("mean_abs_delta"),
                "top1pct_mass": (prior_conc.get("0.01") or {}).get("share"),
                "top5pct_mass": (prior_conc.get("0.05") or {}).get("share"),
                "disagreement_auroc": prior_proxy.get("auroc"),
                "disagreement_auprc": prior_proxy.get("auprc"),
                "h10_spearman": prior_h10.get("spearman_abs_delta_vs_h10_completed"),
                "divergence_among_nonzero": prior_div,
            },
            {
                "corpus": "joint-240 (terminal_anwg_joint240_v1)",
                "scenarios": summary.get("n_scenarios_succeeded"),
                "acquired_states": cur_prev.get("n_states"),
                "nonzero_pct": None
                if cur_prev.get("frac_nonzero") is None
                else 100 * float(cur_prev["frac_nonzero"]),
                "positive_pct": None
                if cur_prev.get("frac_positive") is None
                else 100 * float(cur_prev["frac_positive"]),
                "negative_pct": None
                if cur_prev.get("frac_negative") is None
                else 100 * float(cur_prev["frac_negative"]),
                "mean_abs_delta": cur_prev.get("mean_abs_delta"),
                "top1pct_mass": (cur_conc.get("0.01") or {}).get("share"),
                "top5pct_mass": (cur_conc.get("0.05") or {}).get("share"),
                "disagreement_auroc": cur_proxy.get(
                    "auroc_disagreement_for_nonzero_abs_delta"
                ),
                "disagreement_auprc": cur_proxy.get(
                    "auprc_disagreement_for_nonzero_abs_delta"
                ),
                "h10_spearman": None,
                "divergence_among_nonzero": cur_div.get("rate_among_nonzero"),
            },
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=OUT)
    ap.add_argument("--limit-scenarios", type=int, default=0)
    ap.add_argument("--scenario-ids", nargs="*", default=[])
    ap.add_argument(
        "--max-disagreement", type=int, default=jtan.MAX_DISAGREEMENT_PER_SCENARIO
    )
    ap.add_argument(
        "--max-agreement", type=int, default=jtan.MAX_AGREEMENT_CONTROL_PER_SCENARIO
    )
    ap.add_argument("--skip-fit-cache", action="store_true")
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(exist_ok=True)
    log = out_dir / "logs" / "full_run.log"
    t0 = time.time()
    start = datetime.now(timezone.utc).isoformat()

    def logline(msg: str) -> None:
        line = f"[{datetime.now(timezone.utc).isoformat()}] {msg}"
        print(line, flush=True)
        with log.open("a") as f:
            f.write(line + "\n")

    if args.analyze_only:
        branches = pd.read_csv(out_dir / "branches.csv")
        summary = analyze(branches)
        summary["comparison_vs_abc"] = compare_with_prior(summary)
        summary["reanalyzed"] = True
        (out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        logline("analyze-only DONE")
        return 0

    logline(f"start {start}")
    ctx = jtan.load_frozen_joint240_context()
    matrix = ctx["matrix"]
    scenarios = ctx["scenarios"]
    folds = ctx["folds"]
    manifest = ctx["manifest"].set_index("scenario_id")

    table = folds.merge(
        matrix[["scenario_id", "vbs_policy", "vbs_anwg"]], on="scenario_id", how="inner"
    )
    if args.scenario_ids:
        table = table[table["scenario_id"].isin(args.scenario_ids)]
    if args.limit_scenarios > 0:
        table = table.sort_values("scenario_id").head(args.limit_scenarios)

    config = {
        "schema_version": jtan.SCHEMA_VERSION,
        "max_disagreement_per_scenario": args.max_disagreement,
        "max_agreement_control_per_scenario": args.max_agreement,
        "control_seed": jtan.CONTROL_SEED,
        "bootstrap_seed": jtan.BOOTSTRAP_SEED,
        "n_bootstrap": jtan.N_BOOTSTRAP,
        "design_doc": str(DESIGN.relative_to(REPO)),
        "design_sha256": _sha(DESIGN) if DESIGN.exists() else None,
        "n_scenarios": int(len(table)),
        "continuation_policy": "live_p6_dwell_router_v1 (OOF Alive)",
        "estimand": "continuation-policy-conditional one-step terminal Delta ANWG",
        "git_head": _git_head(),
        "git_dirty": _git_dirty(),
        "python": sys.executable,
        "python_version": platform.python_version(),
        "joint240_exp": str(jtan.JOINT240_EXP.relative_to(REPO)),
        "p6": list(jtan.P6),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    # Freeze design copy into experiment dir (do not overwrite prior experiment)
    if DESIGN.exists():
        (out_dir / "DESIGN_FROZEN.md").write_text(DESIGN.read_text())

    # Fit OOF Alive Stage-1 models (needed folds only; full train pools)
    needed_folds = sorted(table["fold"].astype(int).unique().tolist())
    logline(f"Fitting Alive Stage-1 for folds={needed_folds} ...")
    models = jtan.fit_oof_alive_stage1_models(
        scenarios, matrix, folds, fold_ids=needed_folds
    )
    logline(f"Fitted {len(models)} fold models")

    branch_path = out_dir / "branches.jsonl"
    if branch_path.exists():
        branch_path.unlink()
    scen_rows = []
    failures = []

    for i, (_, row) in enumerate(table.iterrows()):
        sid = str(row["scenario_id"])
        fold = int(row["fold"])
        logline(f"[{i+1}/{len(table)}] {sid} fold={fold}")
        try:
            if fold not in models:
                raise KeyError(f"missing Stage-1 for fold {fold}")
            prow = manifest.loc[sid].to_dict()
            seed = int(prow.get("seed", 20260824))
            res = jtan.run_scenario_terminal_anwg_joint240(
                scenarios[sid],
                stage1=models[fold],
                fold=fold,
                seed=seed,
                n_elevated_mechanisms=int(row["n_elevated_mechanisms"]),
                pressure_row=prow,
                max_disagreement=args.max_disagreement,
                max_agreement_control=args.max_agreement,
            )
            with branch_path.open("a") as f:
                for br in res["branch_rows"]:
                    f.write(json.dumps(br, sort_keys=True, default=str) + "\n")
            scen_rows.append({k: v for k, v in res.items() if k != "branch_rows"})
        except Exception as e:  # noqa: BLE001
            import traceback

            failures.append({"scenario_id": sid, "error": f"{type(e).__name__}: {e}"})
            logline(f"FAIL {sid}: {e}")
            logline(traceback.format_exc())

    scen_df = pd.DataFrame(scen_rows)
    scen_df.to_csv(out_dir / "scenario_summaries.csv", index=False)

    branches = (
        pd.DataFrame(
            [json.loads(l) for l in branch_path.read_text().splitlines() if l.strip()]
        )
        if branch_path.exists() and branch_path.stat().st_size
        else pd.DataFrame()
    )
    if len(branches):
        branches.to_csv(out_dir / "branches.csv", index=False)
        summary = analyze(branches)
    else:
        summary = {"prevalence": {"n_states": 0}, "verdicts": ["JOINT240_INSUFFICIENT_EFFECT_EVENTS"]}

    summary.update(
        {
            "started_utc": start,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_s": time.time() - t0,
            "n_scenarios_attempted": int(len(table)),
            "n_scenarios_succeeded": int(len(scen_rows)),
            "n_scenarios_failed": int(len(failures)),
            "failures": failures,
            "config": config,
            "comparison_vs_abc": compare_with_prior(summary),
        }
    )
    # Recompute verdicts after attaching bootstrap fully
    summary["verdicts"] = jtan.assign_verdicts(summary)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n"
    )

    if len(branches):
        abs_d = branches["abs_delta_anwg"].sort_values(ascending=False).to_numpy(float)
        total = abs_d.sum()
        cum = np.cumsum(abs_d)
        frac_states = (np.arange(1, len(abs_d) + 1)) / len(abs_d)
        frac_mass = cum / total if total > 0 else np.zeros_like(cum)
        pd.DataFrame(
            {"frac_states": frac_states, "frac_abs_anwg_mass": frac_mass}
        ).to_csv(out_dir / "concentration_curve.csv", index=False)

    # Integrity gate
    rr = summary.get("ref_replay") or {}
    if rr.get("n_checks", 0) > 0 and rr.get("n_match", 0) < rr.get("n_checks", 0):
        logline(
            f"REF_REPLAY_INTEGRITY_FAIL checks={rr.get('n_checks')} "
            f"match={rr.get('n_match')} max_abs={rr.get('max_abs_mismatch')}"
        )
        (out_dir / "INTEGRITY_FAIL").write_text(json.dumps(rr, indent=2) + "\n")
        return 2

    (out_dir / "DONE").write_text(
        json.dumps(
            {
                "ok": True,
                "elapsed_s": summary["elapsed_s"],
                "n_branches": int(len(branches)),
                "verdicts": summary.get("verdicts"),
            },
            indent=2,
        )
        + "\n"
    )
    logline(
        f"DONE elapsed={time.time()-t0:.1f}s branches={len(branches)} "
        f"fails={len(failures)} verdicts={summary.get('verdicts')}"
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
