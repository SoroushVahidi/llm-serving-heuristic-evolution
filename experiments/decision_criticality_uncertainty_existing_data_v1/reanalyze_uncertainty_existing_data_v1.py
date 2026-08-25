#!/usr/bin/env python3
"""Analysis-only uncertainty reanalysis of frozen terminal-ANWG v1 branches.

Does NOT re-run the simulator. Reads frozen artifacts under
experiments/decision_criticality_terminal_anwg_v1/ and writes derived
summaries into this directory only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "experiments/decision_criticality_terminal_anwg_v1"
OUT = Path(__file__).resolve().parent

ANWG_EQ_ATOL = 1e-12
BOOTSTRAP_SEED = 202608251
N_BOOTSTRAP = 10_000
CI_LO = 0.025
CI_HI = 0.975


def _mann_whitney_auroc(y: np.ndarray, s: np.ndarray) -> float:
    """AUROC via Mann–Whitney (tie-safe midrank), matching the frozen runner."""
    s_pos = s[y == 1]
    s_neg = s[y == 0]
    gt = float(np.mean(s_pos[:, None] > s_neg[None, :]))
    eq = float(np.mean(s_pos[:, None] == s_neg[None, :]))
    return gt + 0.5 * eq


def _top_frac_share(vals: np.ndarray, frac: float) -> tuple[int, float]:
    vals = np.asarray(vals, dtype=float)
    if len(vals) == 0:
        return 0, float("nan")
    total = float(vals.sum())
    if total <= 0:
        return max(1, int(np.ceil(frac * len(vals)))), 0.0
    order = np.argsort(-vals)
    sorted_v = vals[order]
    k = max(1, int(np.ceil(frac * len(sorted_v))))
    return k, float(sorted_v[:k].sum() / total)


def _ci(xs: list[float]) -> dict:
    arr = np.asarray(xs, dtype=float)
    return {
        "n": int(len(arr)),
        "mean": float(np.mean(arr)) if len(arr) else None,
        "ci95_low": float(np.quantile(arr, CI_LO)) if len(arr) else None,
        "ci95_high": float(np.quantile(arr, CI_HI)) if len(arr) else None,
    }


def main() -> int:
    branches = pd.read_csv(SRC / "branches.csv")
    scenarios = pd.read_csv(SRC / "scenario_summaries.csv")
    parent_ids = scenarios["canonical_scenario_id"].astype(str).tolist()
    assert len(parent_ids) == 144, len(parent_ids)

    # Precompute per-scenario arrays for fast clustered resampling.
    abs_by: list[np.ndarray] = []
    score_by: list[np.ndarray] = []
    mass_by: list[float] = []
    for sid in parent_ids:
        g = branches.loc[branches["canonical_scenario_id"] == sid]
        if len(g) == 0:
            abs_by.append(np.asarray([], dtype=float))
            score_by.append(np.asarray([], dtype=float))
            mass_by.append(0.0)
        else:
            a = g["abs_delta_anwg"].to_numpy(dtype=float)
            s = (g["acquisition_type"] == "DISAGREEMENT").astype(float).to_numpy()
            abs_by.append(a)
            score_by.append(s)
            mass_by.append(float(a.sum()))
    mass_by_arr = np.asarray(mass_by, dtype=float)

    abs_all = branches["abs_delta_anwg"].to_numpy(dtype=float)
    y_all = (abs_all > ANWG_EQ_ATOL).astype(int)
    s_all = (branches["acquisition_type"] == "DISAGREEMENT").astype(float).to_numpy()

    n_states = int(len(branches))
    n_nonzero = int(y_all.sum())
    prevalence = float(y_all.mean())
    top1_k, top1_share = _top_frac_share(abs_all, 0.01)

    scen_k, scen_share = _top_frac_share(mass_by_arr, 0.01)
    acquired_mask = np.asarray([len(a) > 0 for a in abs_by], dtype=bool)
    acq_k, acq_share = _top_frac_share(mass_by_arr[acquired_mask], 0.01)

    auroc_point = _mann_whitney_auroc(y_all, s_all)
    auroc_sklearn = float(roc_auc_score(y_all, s_all))
    auprc_point = float(average_precision_score(y_all, s_all))

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n_parent = len(parent_ids)

    boot_prev: list[float] = []
    boot_top1: list[float] = []
    boot_scen: list[float] = []
    boot_auroc: list[float] = []
    boot_auprc: list[float] = []
    n_skipped_single_class = 0
    n_skipped_empty = 0

    for _ in range(N_BOOTSTRAP):
        draw = rng.integers(0, n_parent, size=n_parent)
        # Concatenate acquired states for drawn scenarios (multiplicity preserved).
        parts_abs = [abs_by[i] for i in draw if len(abs_by[i])]
        if not parts_abs:
            n_skipped_empty += 1
            continue
        abs_d = np.concatenate(parts_abs)
        score = np.concatenate([score_by[i] for i in draw if len(score_by[i])])
        y = (abs_d > ANWG_EQ_ATOL).astype(int)

        boot_prev.append(float(y.mean()))
        _, tshare = _top_frac_share(abs_d, 0.01)
        boot_top1.append(tshare)

        unit_mass = mass_by_arr[draw]
        _, sshare = _top_frac_share(unit_mass, 0.01)
        boot_scen.append(sshare)

        if y.min() == y.max() or score.min() == score.max():
            n_skipped_single_class += 1
            continue
        boot_auroc.append(_mann_whitney_auroc(y, score))
        boot_auprc.append(float(average_precision_score(y, score)))

    summary = {
        "schema_version": "decision_criticality_uncertainty_existing_data_v1.0.0",
        "analysis_only": True,
        "source_experiment": "experiments/decision_criticality_terminal_anwg_v1",
        "source_files": {
            "branches_csv": str(SRC / "branches.csv"),
            "scenario_summaries_csv": str(SRC / "scenario_summaries.csv"),
            "summary_json": str(SRC / "summary.json"),
        },
        "equality_tolerance": ANWG_EQ_ATOL,
        "bootstrap": {
            "method": "scenario_clustered_with_replacement",
            "sampling_unit": "canonical_scenario_id",
            "parent_universe_n": 144,
            "acquired_scenarios_n": int(acquired_mask.sum()),
            "seed": BOOTSTRAP_SEED,
            "n_resamples": N_BOOTSTRAP,
            "n_empty_skipped": n_skipped_empty,
            "n_single_class_skipped_for_ranking_metrics": n_skipped_single_class,
            "n_auroc_auprc_retained": len(boot_auroc),
            "n_prevalence_top1_scen_retained": len(boot_prev),
        },
        "point_estimates": {
            "n_evaluated_states": n_states,
            "n_nonzero": n_nonzero,
            "nonzero_prevalence": prevalence,
            "top1pct_state_mass": {
                "k": top1_k,
                "share": top1_share,
                "denominator": "evaluated_states",
            },
            "scenario_concentration_parent144": {
                "k": scen_k,
                "share": scen_share,
                "denominator": "all_144_parent_TRAIN_VAL_scenarios",
                "manuscript_phrase": "2 of 144 scenarios carry 42.4%",
            },
            "scenario_concentration_acquired124": {
                "k": acq_k,
                "share": acq_share,
                "denominator": "124_scenarios_with_acquired_states",
                "note": (
                    "Same k=2 and share as parent-144 definition because "
                    "ceil(0.01*124)=2 and zero-mass scenarios do not enter the top-2."
                ),
            },
            "disagreement_auroc_mann_whitney": auroc_point,
            "disagreement_auroc_sklearn": auroc_sklearn,
            "disagreement_auprc_average_precision": auprc_point,
            "positive_prevalence_noskill_auprc": prevalence,
        },
        "bootstrap_ci95_percentile": {
            "nonzero_prevalence": _ci(boot_prev),
            "top1pct_state_mass_share": _ci(boot_top1),
            "scenario_concentration_top1pct_of_144_units": _ci(boot_scen),
            "disagreement_auroc": _ci(boot_auroc),
            "disagreement_auprc": _ci(boot_auprc),
        },
        "manuscript_rounding": {
            "nonzero_prevalence_pct": round(100.0 * prevalence, 1),
            "top1pct_state_mass_pct": round(100.0 * top1_share, 1),
            "scenario_concentration_pct": round(100.0 * scen_share, 1),
            "auroc": round(auroc_point, 3),
            "auprc": round(auprc_point, 3),
            "noskill_prevalence": round(prevalence, 3),
        },
    }

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    rows = [
        {
            "statistic": "nonzero_prevalence",
            "point": prevalence,
            **summary["bootstrap_ci95_percentile"]["nonzero_prevalence"],
        },
        {
            "statistic": "top1pct_state_mass_share",
            "point": top1_share,
            **summary["bootstrap_ci95_percentile"]["top1pct_state_mass_share"],
        },
        {
            "statistic": "scenario_concentration_top1pct_of_144_units",
            "point": scen_share,
            **summary["bootstrap_ci95_percentile"]["scenario_concentration_top1pct_of_144_units"],
        },
        {
            "statistic": "disagreement_auroc",
            "point": auroc_point,
            **summary["bootstrap_ci95_percentile"]["disagreement_auroc"],
        },
        {
            "statistic": "disagreement_auprc",
            "point": auprc_point,
            **summary["bootstrap_ci95_percentile"]["disagreement_auprc"],
        },
    ]
    pd.DataFrame(rows).to_csv(OUT / "bootstrap_ci_summary.csv", index=False)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
