#!/usr/bin/env python3
"""Build joint-240 Pext matrix and predeclared portfolio analysis (Pass 4).

Does NOT modify canonical P6 artifacts. Reads P6 wide matrix + external
cells.jsonl outputs and writes analysis/ artifacts only.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
JOINT = REPO / "experiments/joint_multimechanism_generalization_v1"
EXT = REPO / "experiments/external_baseline_comparison_v1"
OUT = EXT / "analysis"

P6 = [
    "full_prefill",
    "chunked_prefill_small",
    "estimated_service_time_first",
    "weighted_fair_share",
    "least_laxity_first",
    "kv_constrained_online",
]
EXT_POL = [
    "official_vtc_joint_token_budget_remap",
    "vllm_style_continuous_batching",
]
PEXT = P6 + EXT_POL
EPS = 0.01
N_BOOT = 1000
BOOT_SEED = 20260825

# Canonical manuscript targets
CANON_SBS = 0.314072
CANON_VBS = 0.333106
CANON_HR = 0.019034


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_cells(path: Path) -> pd.DataFrame:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return pd.DataFrame(rows)


def envelope_stats(mat: np.ndarray, policies: list[str]) -> dict:
    """mat shape (n, p); higher ANWG is better."""
    sbs_idx = int(np.argmax(mat.mean(axis=0)))
    sbs = float(mat[:, sbs_idx].mean())
    vbs_row = mat.max(axis=1)
    vbs = float(vbs_row.mean())
    return {
        "SBS_policy": policies[sbs_idx],
        "SBS": sbs,
        "VBS": vbs,
        "headroom": vbs - sbs,
        "mean_by_policy": {policies[i]: float(mat[:, i].mean()) for i in range(len(policies))},
    }


def winners(mat: np.ndarray, policies: list[str], eps: float) -> dict:
    n, p = mat.shape
    win = []
    eps_unique = {pol: 0 for pol in policies}
    for i in range(n):
        row = mat[i]
        best = float(row.max())
        winners_i = [policies[j] for j in range(p) if row[j] >= best - 1e-15]
        # primary winner: argmax with stable tie-break by policy order
        w = policies[int(np.argmax(row))]
        win.append(w)
        for j, pol in enumerate(policies):
            others = [row[k] for k in range(p) if k != j]
            if float(row[j]) >= max(others) + eps:
                eps_unique[pol] += 1
    counts = pd.Series(win).value_counts().reindex(policies, fill_value=0).astype(int).to_dict()
    return {"winner_counts": counts, "epsilon_unique_counts": eps_unique, "winners": win}


def dominates(a: np.ndarray, b: np.ndarray, tol: float = 1e-12) -> bool:
    return bool(np.all(a >= b - tol) and np.any(a > b + tol))


def epsilon_dominates(a: np.ndarray, b: np.ndarray, eps: float) -> bool:
    """A epsilon-dominates B if A >= B-eps everywhere and A >= B+eps somewhere."""
    return bool(np.all(a >= b - eps) and np.any(a >= b + eps))


def incremental_envelope(mat_p6: np.ndarray, col_b: np.ndarray) -> float:
    e_p6 = mat_p6.max(axis=1)
    e_aug = np.maximum(e_p6, col_b)
    return float((e_aug - e_p6).mean())


def bootstrap(mat_p6: np.ndarray, mat_pext: np.ndarray, policies_p6, policies_pext, seed: int, n: int):
    rng = np.random.default_rng(seed)
    n_sc = mat_p6.shape[0]
    stats = []
    for _ in range(n):
        idx = rng.integers(0, n_sc, size=n_sc)
        sp = envelope_stats(mat_p6[idx], policies_p6)
        se = envelope_stats(mat_pext[idx], policies_pext)
        stats.append(
            {
                "headroom_p6": sp["headroom"],
                "headroom_pext": se["headroom"],
                "vbs_delta": se["VBS"] - sp["VBS"],
                "sbs_delta": se["SBS"] - sp["SBS"],
                "headroom_delta": se["headroom"] - sp["headroom"],
            }
        )
    df = pd.DataFrame(stats)

    def ci(col):
        return {
            "mean": float(df[col].mean()),
            "ci95_low": float(df[col].quantile(0.025)),
            "ci95_high": float(df[col].quantile(0.975)),
        }

    return {
        "n_bootstrap": n,
        "seed": seed,
        "Headroom_P6": ci("headroom_p6"),
        "Headroom_Pext": ci("headroom_pext"),
        "VBS_Pext_minus_VBS_P6": ci("vbs_delta"),
        "SBS_Pext_minus_SBS_P6": ci("sbs_delta"),
        "Headroom_Pext_minus_Headroom_P6": ci("headroom_delta"),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    p6_path = JOINT / "utility_matrix_wide.csv"
    manifest = JOINT / "scenario_manifest.csv"
    vtc_path = EXT / "results/joint_vtc/cells.jsonl"
    vllm_path = EXT / "results/joint_vllm_style/cells.jsonl"

    p6 = pd.read_csv(p6_path)
    assert len(p6) == 240, len(p6)
    for pol in P6:
        assert f"anwg__{pol}" in p6.columns

    # Reproduce canonical
    mat_p6 = p6[[f"anwg__{p}" for p in P6]].to_numpy(dtype=float)
    env_p6 = envelope_stats(mat_p6, P6)
    # Manuscript SBS = best fixed = max mean policy (not mean of row-max of single)
    means = mat_p6.mean(axis=0)
    sbs_pol = P6[int(np.argmax(means))]
    sbs = float(means.max())
    vbs = float(mat_p6.max(axis=1).mean())
    hr = vbs - sbs
    canon_ok = (
        abs(sbs - CANON_SBS) < 5e-7
        and abs(vbs - CANON_VBS) < 5e-7
        and abs(hr - CANON_HR) < 5e-7
    )
    if not canon_ok:
        raise SystemExit(
            f"P6 CANONICAL MISMATCH: SBS={sbs} VBS={vbs} HR={hr} "
            f"expected {CANON_SBS}/{CANON_VBS}/{CANON_HR}"
        )

    vtc = load_cells(vtc_path)
    vllm = load_cells(vllm_path)
    for name, df in [("vtc", vtc), ("vllm", vllm)]:
        assert len(df) == 240, (name, len(df))
        assert (df["failure_status"] == "success").all()
        assert df["scenario_id"].nunique() == 240
        assert df["anwg"].notna().all()

    # Align by scenario_id order of P6 matrix
    sid = p6["scenario_id"].tolist()
    vtc_m = vtc.set_index("scenario_id").loc[sid]
    vllm_m = vllm.set_index("scenario_id").loc[sid]
    assert list(vtc_m.index) == sid
    assert list(vllm_m.index) == sid

    wide = p6[["scenario_id"] + [f"anwg__{p}" for p in P6]].copy()
    wide["anwg__official_vtc_joint_token_budget_remap"] = vtc_m["anwg"].to_numpy()
    wide["anwg__vllm_style_continuous_batching"] = vllm_m["anwg"].to_numpy()
    assert wide.shape == (240, 1 + 8)
    assert wide.isna().sum().sum() == 0

    matrix_path = OUT / "joint_240_pext_matrix.csv"
    wide.to_csv(matrix_path, index=False)

    mat_pext = wide[[f"anwg__{p}" for p in PEXT]].to_numpy(dtype=float)
    env_pext = envelope_stats(mat_pext, PEXT)
    # redefine SBS as best-fixed mean
    env_p6 = {
        "SBS_policy": sbs_pol,
        "SBS": sbs,
        "VBS": vbs,
        "headroom": hr,
        "mean_by_policy": {P6[i]: float(means[i]) for i in range(6)},
    }
    means_e = mat_pext.mean(axis=0)
    sbs_e_pol = PEXT[int(np.argmax(means_e))]
    sbs_e = float(means_e.max())
    vbs_e = float(mat_pext.max(axis=1).mean())
    env_pext = {
        "SBS_policy": sbs_e_pol,
        "SBS": sbs_e,
        "VBS": vbs_e,
        "headroom": vbs_e - sbs_e,
        "mean_by_policy": {PEXT[i]: float(means_e[i]) for i in range(len(PEXT))},
    }

    w_p6 = winners(mat_p6, P6, EPS)
    w_pext = winners(mat_pext, PEXT, EPS)

    # Incremental envelope from each external
    incr = {
        "official_vtc_joint_token_budget_remap": incremental_envelope(
            mat_p6, mat_pext[:, PEXT.index("official_vtc_joint_token_budget_remap")]
        ),
        "vllm_style_continuous_batching": incremental_envelope(
            mat_p6, mat_pext[:, PEXT.index("vllm_style_continuous_batching")]
        ),
        "both_externals_vs_P6": float(
            (mat_pext.max(axis=1) - mat_p6.max(axis=1)).mean()
        ),
    }

    # Dominance
    dom = {"strict": [], "epsilon_0.01": []}
    for i, a in enumerate(PEXT):
        for j, b in enumerate(PEXT):
            if i == j:
                continue
            if dominates(mat_pext[:, i], mat_pext[:, j]):
                dom["strict"].append({"dominator": a, "dominated": b})
            if epsilon_dominates(mat_pext[:, i], mat_pext[:, j], EPS):
                dom["epsilon_0.01"].append({"dominator": a, "dominated": b})

    # Survival: remove-one VBS drop
    def leave_one_out(mat, policies):
        out = {}
        full_vbs = float(mat.max(axis=1).mean())
        for i, pol in enumerate(policies):
            sub = np.delete(mat, i, axis=1)
            vbs_wo = float(sub.max(axis=1).mean())
            out[pol] = {
                "pext_wins": int(w_pext["winner_counts"].get(pol, 0)),
                "pext_epsilon_unique": int(w_pext["epsilon_unique_counts"].get(pol, 0)),
                "VBS_without": vbs_wo,
                "VBS_drop_if_removed": full_vbs - vbs_wo,
                "contributes_to_envelope": (full_vbs - vbs_wo) > 1e-15,
            }
        return out

    survival = leave_one_out(mat_pext, PEXT)

    boot = bootstrap(mat_p6, mat_pext, P6, PEXT, BOOT_SEED, N_BOOT)

    col_sources = {
        "p6_utility_matrix_wide": {
            "path": str(p6_path.relative_to(REPO)),
            "sha256": sha256_file(p6_path),
        },
        "scenario_manifest": {
            "path": str(manifest.relative_to(REPO)),
            "sha256": sha256_file(manifest),
        },
        "official_vtc_joint_token_budget_remap": {
            "path": str(vtc_path.relative_to(REPO)),
            "sha256": sha256_file(vtc_path),
            "n": 240,
        },
        "vllm_style_continuous_batching": {
            "path": str(vllm_path.relative_to(REPO)),
            "sha256": sha256_file(vllm_path),
            "n": 240,
        },
    }

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_scenarios": 240,
        "n_policies_pext": 8,
        "n_cells": 1920,
        "epsilon": EPS,
        "canonical_p6_reproduction": {
            "SBS": sbs,
            "VBS": vbs,
            "headroom": hr,
            "SBS_policy": sbs_pol,
            "matches_manuscript": True,
            "targets": {"SBS": CANON_SBS, "VBS": CANON_VBS, "headroom": CANON_HR},
        },
        "P6": env_p6,
        "Pext": env_pext,
        "deltas": {
            "SBS_Pext_minus_SBS_P6": sbs_e - sbs,
            "VBS_Pext_minus_VBS_P6": vbs_e - vbs,
            "Headroom_Pext_minus_Headroom_P6": (vbs_e - sbs_e) - hr,
        },
        "incremental_envelope_gain": incr,
        "column_sources": col_sources,
        "matrix_path": str(matrix_path.relative_to(REPO)),
        "key_questions": {},
    }

    # Key Q answers (factual, not forced)
    q = {
        "Q1_external_improve_SBS": bool(sbs_e > sbs + 1e-12),
        "Q2_external_improve_VBS": bool(vbs_e > vbs + 1e-12),
        "Q3_VBS_gt_SBS_pext_nontrivial": bool((vbs_e - sbs_e) >= 0.01),
        "Q3_headroom_pext": vbs_e - sbs_e,
        "Q4_multiple_winners": int(sum(1 for v in w_pext["winner_counts"].values() if v > 0)),
        "Q5_p6_retain_unique_envelope": {
            p: survival[p]["contributes_to_envelope"] for p in P6
        },
        "Q6_external_strict_dominates_any_p6": [
            d for d in dom["strict"] if d["dominator"] in EXT_POL and d["dominated"] in P6
        ],
        "Q7_single_external_collapses_opportunity": bool(
            max(
                survival["official_vtc_joint_token_budget_remap"]["pext_wins"],
                survival["vllm_style_continuous_batching"]["pext_wins"],
            )
            >= 200
            and (vbs_e - sbs_e) < 0.003
        ),
        "Q8_core_claim_preserved_qualitative": None,  # filled below
    }
    # Core claim: VBS>SBS nontrivial AND multiple complementary winners AND
    # some P6 still contribute OR externals don't fully collapse headroom
    p6_any_contrib = any(survival[p]["contributes_to_envelope"] for p in P6)
    q["Q8_core_claim_preserved_qualitative"] = bool(
        (vbs_e - sbs_e) >= 0.005
        and q["Q4_multiple_winners"] >= 3
        and (p6_any_contrib or (vbs_e - vbs) < (vbs_e - sbs_e))
    )
    summary["key_questions"] = q
    summary["survival"] = survival
    summary["dominance"] = dom
    summary["winner_counts_p6"] = w_p6["winner_counts"]
    summary["winner_counts_pext"] = w_pext["winner_counts"]
    summary["epsilon_unique_p6"] = w_p6["epsilon_unique_counts"]
    summary["epsilon_unique_pext"] = w_pext["epsilon_unique_counts"]

    (OUT / "joint_pext_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (OUT / "joint_pext_bootstrap.json").write_text(json.dumps(boot, indent=2, sort_keys=True) + "\n")

    pol_rows = []
    for i, pol in enumerate(PEXT):
        pol_rows.append(
            {
                "policy": pol,
                "in_P6": pol in P6,
                "mean_anwg": float(means_e[i]),
                "winner_count": w_pext["winner_counts"][pol],
                "epsilon_unique_count": w_pext["epsilon_unique_counts"][pol],
                "VBS_drop_if_removed": survival[pol]["VBS_drop_if_removed"],
                "contributes_to_envelope": survival[pol]["contributes_to_envelope"],
            }
        )
    pd.DataFrame(pol_rows).to_csv(OUT / "joint_pext_policy_summary.csv", index=False)

    win_rows = [{"scenario_id": sid[i], "winner_pext": w_pext["winners"][i], "winner_p6": w_p6["winners"][i]} for i in range(240)]
    pd.DataFrame(win_rows).to_csv(OUT / "joint_pext_winner_summary.csv", index=False)

    print(json.dumps({"P6": env_p6, "Pext": env_pext, "incr": incr, "Q": q}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
