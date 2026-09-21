#!/usr/bin/env python3
"""POST-HOC robustness / sensitivity analysis of the frozen fresh causal latency-headroom result.

Scope and status
----------------
* Input : the canonical, frozen artifacts in
  ``experiments/fresh_production_latency_headroom_confirmatory_v1/`` (read-only).
* Output: ``experiments/fresh_production_latency_headroom_confirmatory_v1_robustness/``.
* Nothing here regenerates, filters, relabels or modifies canonical data, and nothing here
  redefines the primary estimand, the regimes, or the disagreement-state population.
* Every analysis in this module is POST-HOC.  None of the practical thresholds, weightings,
  leave-one-out diagnostics, jackknife, BCa or extra-replicate bootstraps was frozen in
  ``PREREGISTRATION_V1.json`` / ``METRIC_PROTOCOL_V1.json``; do not describe them as
  preregistered or confirmatory.

Notation (all latencies in seconds unless a name ends in ``_ms``)
-----------------------------------------------------------------
* ``D``       : the 720 canonical disagreement states.
* ``A(s,a)``  : one-step latency advantage  L_SBS(s) - L_CF(s,a)  (positive = alternative better).
* ``H(s)``    : oracle latency headroom  max(0, max_a A(s,a))   (canonical ``oracle_headroom``).
* window      : faithful source window = ``(source_dataset, window_index)``  (36 clusters).
* regime      : ``(source_dataset, axis, condition_id)``                        (5 cells).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "experiments" / "fresh_production_latency_headroom_confirmatory_v1"
DEFAULT_OUT = ROOT / "experiments" / "fresh_production_latency_headroom_confirmatory_v1_robustness"

# ---- canonical constants (mirrors scripts/fresh_latency_causal_confirmatory_v1.py) -------------
CANON_SEED = 20260920
CANON_REPLICATES = 2000
CANON_QUANTILES = (0.025, 0.975)
STEP_S = 0.001  # simulator decode step (seconds); latency contrasts live on a step/N grid

# ---- post-hoc analysis constants --------------------------------------------------------------
THRESHOLDS_MS = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0)
EPS_S = 1e-12  # numerical tolerance (s); justified by the grid audit in tolerance_audit()
LARGE_BOOT_REPLICATES = 1_000_000
MID_BOOT_REPLICATES = 200_000
MC_SEEDS = 1000  # repeats of the canonical-size (2,000-rep) bootstrap to expose Monte-Carlo noise
BOOT_SEEDS_LARGE = (CANON_SEED, 20260921, 20260922)
MIN_SUPPORT_STATES = 30
MIN_SUPPORT_WINDOWS = 5

CANONICAL_INPUTS = (
    "FRESH_LATENCY_STATE_LEVEL_V1.csv",
    "FRESH_LATENCY_ACTION_LEVEL_V1.csv",
    "FRESH_LATENCY_WORKLOAD_REGIME_V1.csv",
    "FRESH_LATENCY_CAUSAL_RESULT_V1.json",
    "FRESH_LATENCY_BOOTSTRAP_V1.json",
    "FRESH_LATENCY_CONFIRMATORY_REPORT_V1.md",
    "FRESH_LATENCY_ARTIFACT_HASHES_V1.json",
    "FRESH_LATENCY_EXECUTION_PROVENANCE_V1.json",
    "PREREGISTRATION_V1.json",
    "METRIC_PROTOCOL_V1.json",
    "CAUSAL_PROTOCOL_V1.json",
    "FRESH_REGIME_SELECTION_V1.json",
    "BOOTSTRAP_CLUSTER_KEY_CORRECTION_20260920.md",
)

REGIME_LABELS = {
    "azure_2023_code|active_sequence_capacity|active_8": "Code, active cap 8",
    "azure_2023_code|active_sequence_capacity|active_4": "Code, active cap 4",
    "azure_2023_code|kv_capacity|kv_16000": "Code, KV 16,000",
    "azure_2023_conv|active_sequence_capacity|active_4": "Conv, active cap 4",
    "azure_2023_conv|kv_capacity|kv_16000": "Conv, KV 16,000",
}
REGIME_ORDER = list(REGIME_LABELS)
WORKLOAD_LABELS = {"azure_2023_code": "Azure code", "azure_2023_conv": "Azure conversation"}


# =============================================================================================
# generic helpers
# =============================================================================================
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git(*args: str) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # pragma: no cover - git absent
        return None


def ms(x):
    return np.asarray(x, dtype=float) * 1e3


def dump_json(obj, path: Path) -> None:
    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))

    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=default) + "\n")


# =============================================================================================
# loading
# =============================================================================================
def load_states(design: Path = DESIGN) -> pd.DataFrame:
    """Canonical 720-state table plus derived unit keys.  Read-only."""
    df = pd.read_csv(design / "FRESH_LATENCY_STATE_LEVEL_V1.csv")
    return add_units(df)


def add_units(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["window"] = df["source_dataset"].astype(str) + ":w" + df["window_index"].astype(int).astype(str)
    df["regime"] = df["source_dataset"].astype(str) + "|" + df["axis"].astype(str) + "|" + df["condition_id"].astype(str)
    df["H_ms"] = df["oracle_headroom"].astype(float) * 1e3
    return df


def load_actions(design: Path = DESIGN) -> pd.DataFrame:
    """Canonical 831 counterfactual branches with the recovered SBS reference latency.

    NOTE: the canonical *state-level* ``mean_ref_latency`` column is NOT the SBS reference (it is the
    first counterfactual branch's mean latency; see the report).  The true reference is recovered
    exactly as ``cf mean latency + a_lat`` (constant within a state to ~2e-16 s).
    """
    a = pd.read_csv(design / "FRESH_LATENCY_ACTION_LEVEL_V1.csv")
    a["window"] = a["source_dataset"].astype(str) + ":w" + a["window_index"].astype(int).astype(str)
    a["regime"] = a["source_dataset"].astype(str) + "|" + a["axis"].astype(str) + "|" + a["condition_id"].astype(str)
    a["ref_mean_latency"] = a["mean_latency"].astype(float) + a["a_lat"].astype(float)
    return a


# =============================================================================================
# canonical-primary reproduction
# =============================================================================================
def canonical_bootstrap_loop(df: pd.DataFrame, cluster_cols: Sequence[str] = ("source_dataset", "window_index"),
                             seed: int = CANON_SEED, reps: int = CANON_REPLICATES) -> dict:
    """Line-by-line re-implementation of ``bootstrap()`` in scripts/fresh_latency_causal_confirmatory_v1.py.

    ``cluster_cols=("window_index",)`` reproduces the *original* (pre-correction) 26-cluster key.
    """
    if len(cluster_cols) == 2:
        cluster = list(zip(df[cluster_cols[0]].astype(str), df[cluster_cols[1]].astype(int)))
    else:
        cluster = list(df[cluster_cols[0]].astype(int))
    windows = sorted(set(cluster))
    groups = {w: df[pd.Series(cluster, index=df.index) == w] for w in windows}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(reps):
        sample = pd.concat([groups[windows[i]] for i in rng.integers(0, len(windows), len(windows))], ignore_index=True)
        vals.append(float(sample["oracle_headroom"].mean()))
    return {"clusters": len(windows), "lo": float(np.quantile(vals, CANON_QUANTILES[0])),
            "hi": float(np.quantile(vals, CANON_QUANTILES[1]))}


def reproduction_check(states: pd.DataFrame, actions: pd.DataFrame, design: Path = DESIGN) -> dict:
    """Recompute the manuscript's primary aggregates from the canonical artifacts."""
    res = json.loads((design / "FRESH_LATENCY_CAUSAL_RESULT_V1.json").read_text())["primary"]
    boot_json = json.loads((design / "FRESH_LATENCY_BOOTSTRAP_V1.json").read_text())
    h = states["oracle_headroom"].to_numpy(float)
    pos = h[h > 0]
    checks = []

    def add(name, recomputed, reported, tol=0.0, unit=""):
        diff = abs(recomputed - reported)
        status = "EXACT" if diff == 0 else ("WITHIN_TOLERANCE" if diff <= tol else "FAILED")
        checks.append({"quantity": name, "recomputed": recomputed, "reported": reported, "abs_diff": diff,
                       "tolerance": tol, "unit": unit, "status": status})

    add("states", len(states), res["states"])
    add("beneficial_states (canonical rule max_a A > 0)", int((states["beneficial_opportunity"] == 1).sum()), res["beneficial_states"])
    add("beneficial_states recomputed from oracle_headroom > 0", int((h > 0).sum()), res["beneficial_states"])
    add("P(B_LAT|D)", float((h > 0).mean()), res["P_B_given_D"], 1e-15)
    add("mean oracle headroom (s)", float(h.mean()), res["mean_oracle_headroom"], 1e-15, "s")
    add("positive-headroom mean (s)", float(pos.mean()), res["positive_headroom_mean"], 1e-15, "s")
    add("positive-headroom median (s)", float(np.median(pos)), res["positive_headroom_median"], 1e-15, "s")
    add("all-harmful states", int(states["all_alternatives_harmful"].sum()), res["all_harmful_states"])
    add("all-zero states", int(states["all_alternatives_zero"].sum()), res["all_zero_states"])
    add("mixed states", int(states["mixed_beneficial_and_harmful"].sum()), res["mixed_states"])

    # independent state-level reconstruction from the ACTION-level artifact
    h_from_actions = actions.groupby("state_id")["a_lat"].max().clip(lower=0).reindex(states["state_id"])
    add("max |H_state - H recomputed from action-level a_lat| (s)", float(np.abs(states["oracle_headroom"].to_numpy() - h_from_actions.to_numpy()).max()), 0.0, 0.0, "s")
    add("action-level branches", len(actions), 831)
    add("distinct clusters (source_dataset, window_index)", states.groupby(["source_dataset", "window_index"]).ngroups, boot_json["clusters"])

    # bootstrap: the canonical loop, verbatim, plus the fast vectorised engine used later
    loop = canonical_bootstrap_loop(states)
    add("bootstrap clusters (loop)", loop["clusters"], boot_json["clusters"])
    add("bootstrap CI low (s) [canonical loop re-run]", loop["lo"], boot_json["mean_oracle_headroom_ci95_low"], 1e-15, "s")
    add("bootstrap CI high (s) [canonical loop re-run]", loop["hi"], boot_json["mean_oracle_headroom_ci95_high"], 1e-15, "s")
    cl = cluster_arrays(states)
    fast = boot_ratio(cl["S"][:, None], cl["N"], CANON_REPLICATES, CANON_SEED)[:, 0]
    fast_lo, fast_hi = np.quantile(fast, CANON_QUANTILES)
    add("bootstrap CI low (s) [vectorised engine]", float(fast_lo), boot_json["mean_oracle_headroom_ci95_low"], 1e-15, "s")
    add("bootstrap CI high (s) [vectorised engine]", float(fast_hi), boot_json["mean_oracle_headroom_ci95_high"], 1e-15, "s")

    # regime table (compare against the frozen regime CSV that feeds manuscript Table tab:fresh)
    reg_csv = pd.read_csv(design / "FRESH_LATENCY_WORKLOAD_REGIME_V1.csv")
    for _, r in reg_csv.iterrows():
        key = f"{r.source_dataset}|{r.axis}|{r.condition_id}"
        g = states[states["regime"] == key]
        add(f"regime {REGIME_LABELS.get(key, key)}: states", len(g), int(r.states))
        add(f"regime {REGIME_LABELS.get(key, key)}: P(B|D)", float(g["beneficial_opportunity"].mean()), float(r.P_B_given_D), 1e-15)
        add(f"regime {REGIME_LABELS.get(key, key)}: mean H (s)", float(g["oracle_headroom"].mean()), float(r.mean_oracle_headroom), 1e-15, "s")
        add(f"regime {REGIME_LABELS.get(key, key)}: contributing windows", int(g["window"].nunique()), int(r.contributing_windows))

    # manuscript-rounded values (paper/performance_evaluation/main.tex, Abstract + Sec. fresh)
    add("manuscript: mean headroom (ms, 4 dp)", round(float(h.mean()) * 1e3, 4), 1.9958, 5e-5, "ms")
    add("manuscript: CI low (ms, 4 dp)", round(loop["lo"] * 1e3, 4), 0.1909, 5e-5, "ms")
    add("manuscript: CI high (ms, 4 dp)", round(loop["hi"] * 1e3, 4), 3.5462, 5e-5, "ms")
    add("manuscript: P(B|D) (4 dp)", round(590 / 720, 4), 0.8194, 5e-5)

    # documented pre-correction check: original bare-window_index key -> 26 clusters
    orig = canonical_bootstrap_loop(states, cluster_cols=("window_index",))
    original_key = {"clusters": orig["clusters"], "lo_s": orig["lo"], "hi_s": orig["hi"],
                    "documented_lo_ms": 0.184589, "documented_hi_ms": 3.497418,
                    "matches_documented_to_1e-6_ms": abs(orig["lo"] * 1e3 - 0.184589) < 1e-6 and abs(orig["hi"] * 1e3 - 3.497418) < 1e-6 and orig["clusters"] == 26}

    statuses = {c["status"] for c in checks}
    overall = "FAILED" if "FAILED" in statuses else ("EXACT" if statuses == {"EXACT"} else "WITHIN_TOLERANCE")
    return {"overall": overall, "n_checks": len(checks), "n_exact": sum(c["status"] == "EXACT" for c in checks),
            "n_within_tolerance": sum(c["status"] == "WITHIN_TOLERANCE" for c in checks),
            "n_failed": sum(c["status"] == "FAILED" for c in checks), "checks": checks,
            "canonical_bootstrap_loop": loop, "original_pre_correction_key_check": original_key}


# =============================================================================================
# cluster machinery (vectorised, exactly the canonical resampling scheme)
# =============================================================================================
def cluster_arrays(df: pd.DataFrame, extra_counts: dict[str, np.ndarray] | None = None) -> dict:
    """Per-window sums.  Windows are sorted exactly as in the canonical ``bootstrap()``."""
    keys = sorted(set(zip(df["source_dataset"].astype(str), df["window_index"].astype(int))))
    pos = {k: i for i, k in enumerate(keys)}
    idx = np.array([pos[(a, int(b))] for a, b in zip(df["source_dataset"].astype(str), df["window_index"].astype(int))])
    K = len(keys)
    h = df["oracle_headroom"].to_numpy(float)
    out = {"keys": keys, "K": K, "idx": idx,
           "S": np.bincount(idx, weights=h, minlength=K),
           "N": np.bincount(idx, minlength=K).astype(float)}
    return out


def boot_ratio(num: np.ndarray, den: np.ndarray, reps: int, seed: int, chunk: int = 20000) -> np.ndarray:
    """Cluster bootstrap of ratio estimators  sum_k num[k]/sum_k den[k]  over sampled clusters.

    Sampling is with replacement at the cluster level; every state of a sampled cluster is retained with
    multiplicity (encoded in ``num``/``den`` cluster totals).  ``rng.integers(0, K, size=(B, K))`` consumes the
    generator identically to B successive ``rng.integers(0, K, K)`` calls (asserted in the tests), so the
    first 2,000 draws with ``seed=20260920`` are the canonical draws.
    """
    K, T = num.shape
    rng = np.random.default_rng(seed)
    out = np.empty((reps, T))
    done = 0
    while done < reps:
        c = min(chunk, reps - done)
        idx = rng.integers(0, K, size=(c, K))
        out[done:done + c] = num[idx].sum(axis=1) / den[idx].sum(axis=1)[:, None]
        done += c
    return out


def pct_ci(vals: np.ndarray, q: Sequence[float] = CANON_QUANTILES) -> tuple[float, float]:
    lo, hi = np.quantile(vals, q)
    return float(lo), float(hi)


def jackknife_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Delete-one-cluster estimates of  sum num / sum den  (recomputes the ratio, never averages ratios)."""
    return (num.sum() - num) / (den.sum() - den)


def jackknife_summary(theta_hat: float, theta_i: np.ndarray) -> dict:
    K = len(theta_i)
    mean_i = float(theta_i.mean())
    se = math.sqrt((K - 1) / K * float(((theta_i - mean_i) ** 2).sum()))
    tcrit = float(sps.t.ppf(0.975, K - 1))
    pseudo = K * theta_hat - (K - 1) * theta_i
    return {"K": K, "theta_hat": theta_hat, "jackknife_se": se, "t_crit_df_K_minus_1": tcrit,
            "ci95_t": [theta_hat - tcrit * se, theta_hat + tcrit * se],
            "ci95_normal": [theta_hat - 1.959963984540054 * se, theta_hat + 1.959963984540054 * se],
            "loo_min": float(theta_i.min()), "loo_max": float(theta_i.max()),
            "bias_corrected_estimate": float(pseudo.mean()),
            "n_loo_estimates_ci_low_positive": int((theta_i > 0).sum())}


def bca_ci(theta_hat: float, boot: np.ndarray, theta_i: np.ndarray, alpha: float = 0.05) -> dict:
    """Bias-corrected & accelerated percentile interval (acceleration from the delete-one-cluster values)."""
    frac = float((boot < theta_hat).mean())
    frac = min(max(frac, 1.0 / (len(boot) + 1)), 1 - 1.0 / (len(boot) + 1))
    z0 = float(sps.norm.ppf(frac))
    d = theta_i.mean() - theta_i
    denom = 6.0 * (float((d ** 2).sum()) ** 1.5)
    acc = float((d ** 3).sum() / denom) if denom > 0 else 0.0
    out = []
    for a in (alpha / 2, 1 - alpha / 2):
        z = float(sps.norm.ppf(a))
        adj = float(sps.norm.cdf(z0 + (z0 + z) / (1 - acc * (z0 + z))))
        out.append(float(np.quantile(boot, adj)))
    return {"z0": z0, "acceleration": acc, "ci95": out}


# =============================================================================================
# A. practical thresholds
# =============================================================================================
def _regimes(df: pd.DataFrame) -> list[str]:
    present = set(df["regime"].unique())
    return [r for r in REGIME_ORDER if r in present] + sorted(present - set(REGIME_ORDER))


def exceeds(h_s: np.ndarray, tau_ms: float, eps: float = EPS_S) -> np.ndarray:
    """H > tau with an absolute numerical guard: H - tau > eps  (strictly greater than, up to FP noise)."""
    return h_s > (tau_ms * 1e-3 + eps)


def threshold_table(states: pd.DataFrame, thresholds_ms: Iterable[float] = THRESHOLDS_MS) -> pd.DataFrame:
    rows = []
    scopes = [("global", "all", states)]
    scopes += [("workload", w, states[states["source_dataset"] == w]) for w in sorted(states["source_dataset"].unique())]
    scopes += [("regime", r, states[states["regime"] == r]) for r in _regimes(states)]
    for scope, sid, g in scopes:
        h = g["oracle_headroom"].to_numpy(float)
        n = len(g)
        nw = g["window"].nunique()
        for tau in thresholds_ms:
            m = exceeds(h, tau)
            strict = h > tau * 1e-3  # raw floating-point comparison (shown only to expose tie/noise effects)
            near = np.abs(h - tau * 1e-3) <= 1e-9
            rows.append({"scope": scope, "scope_id": sid, "scope_label": REGIME_LABELS.get(sid, WORKLOAD_LABELS.get(sid, sid)),
                         "threshold_ms": tau, "n_exceed": int(m.sum()), "n_states": n, "fraction": float(m.mean()),
                         "n_exceed_raw_fp_strict": int(strict.sum()), "n_ties_within_1e-9s": int(near.sum()),
                         "n_windows_with_exceedance": int(g.loc[m, "window"].nunique()), "n_windows": nw,
                         "low_support": bool(n < MIN_SUPPORT_STATES or nw < MIN_SUPPORT_WINDOWS)})
    return pd.DataFrame(rows)


def threshold_global_ci(states: pd.DataFrame, reps: int, seed: int) -> pd.DataFrame:
    """Window-cluster bootstrap CI for the global exceedance fractions (post-hoc, percentile)."""
    cl = cluster_arrays(states)
    h = states["oracle_headroom"].to_numpy(float)
    cols = np.stack([np.bincount(cl["idx"], weights=exceeds(h, t).astype(float), minlength=cl["K"]) for t in THRESHOLDS_MS], axis=1)
    b = boot_ratio(cols, cl["N"], reps, seed)
    return pd.DataFrame({"threshold_ms": THRESHOLDS_MS, "boot_ci95_low": np.quantile(b, 0.025, axis=0),
                         "boot_ci95_high": np.quantile(b, 0.975, axis=0), "boot_replicates": reps, "boot_seed": seed})


def tolerance_audit(states: pd.DataFrame, actions: pd.DataFrame) -> dict:
    """Is an epsilon tolerance scientifically warranted?  (Inspection only; canonical values unchanged.)"""
    a = actions.copy()
    a["k"] = a["a_lat"] * a["population_count"] / STEP_S
    a["grid_dev"] = (a["k"] - a["k"].round()).abs()
    nz = a.loc[a["k"].round() != 0]
    tiny = a[(a["a_lat"].abs() < 1e-9) & (a["a_lat"] != 0)]
    h = states["oracle_headroom"].to_numpy(float)
    ties = {}
    for t in THRESHOLDS_MS[1:]:
        near = states[np.abs(states["oracle_headroom"] - t * 1e-3) <= 1e-9]
        ties[f"{t}"] = {"n_ties": int(len(near)), "H_values_repr": [repr(float(v)) for v in near["oracle_headroom"]],
                        "raw_fp_strict_counts_as_exceeding": [bool(v > t * 1e-3) for v in near["oracle_headroom"]]}
    noise_pos = states[(states["oracle_headroom"] > 0) & (states["oracle_headroom"] <= EPS_S)]
    noise_neg = states[(states["max_a_lat"] < 0) & (states["max_a_lat"] >= -EPS_S)]
    return {
        "step_s": STEP_S, "epsilon_s": EPS_S,
        "population_count_range": [int(a["population_count"].min()), int(a["population_count"].max())],
        "grid_claim": "each A(s,a) equals integer * step / N_s  (N_s = state request population); confirmed if grid_dev ~ FP",
        "max_grid_deviation_in_grid_units": float(a["grid_dev"].max()),
        "n_actions_off_grid_by_more_than_1e-6_units": int((a["grid_dev"] > 1e-6).sum()),
        "smallest_genuine_nonzero_abs_A_s": float(nz["a_lat"].abs().min()),
        "smallest_genuine_nonzero_abs_A_ms": float(nz["a_lat"].abs().min() * 1e3),
        "largest_step_over_N_s": STEP_S / float(a["population_count"].max()),
        "sub_1e-9_nonzero_advantages": [{"state_id": r.state_id, "a_lat_s": float(r.a_lat), "population_count": int(r.population_count)} for r in tiny.itertuples()],
        "noise_floor_ratio_smallest_genuine_over_eps": float(nz["a_lat"].abs().min() / EPS_S),
        "eps_justification": ("all genuine |A| are >= step/N_max ~ 5.0e-6 s while FP residue is ~5.6e-17 s; any eps in "
                              "[1e-15, 1e-7] s yields identical classifications; eps=1e-12 s (1e-9 ms) is used"),
        "canonical_beneficial_states_strict_gt0": int((h > 0).sum()),
        "beneficial_states_with_eps_guard": int((h > EPS_S).sum()),
        "n_states_H_positive_but_le_eps": int(len(noise_pos)),
        "noise_positive_states": [{"state_id": r.state_id, "H_s": float(r.oracle_headroom)} for r in noise_pos.itertuples()],
        "noise_negative_max_states (counted all-harmful by canonical rule)": [{"state_id": r.state_id, "max_a_lat_s": float(r.max_a_lat)} for r in noise_neg.itertuples()],
        "threshold_ties_within_1e-9_s": ties,
        "recommendation": ("An epsilon guard is warranted for ANY threshold comparison (one canonical 'beneficial' state has "
                           "H=5.55e-17 s and one state at 0.5 ms exceeds the threshold by 1e-16 s). The canonical result is "
                           "NOT altered; guarded counts are reported as a clearly labelled sensitivity."),
    }


# =============================================================================================
# B. effect distribution
# =============================================================================================
def dist_stats(x_s: np.ndarray) -> dict:
    x = np.asarray(x_s, float)
    pos = x[x > EPS_S]
    q = lambda v, p: float(np.quantile(v, p)) if len(v) else float("nan")
    return {
        "N": int(len(x)), "mean_ms": float(x.mean() * 1e3), "median_ms": q(x, .5) * 1e3,
        "sd_ms": float(x.std(ddof=1) * 1e3) if len(x) > 1 else float("nan"),
        "p25_ms": q(x, .25) * 1e3, "p75_ms": q(x, .75) * 1e3, "p90_ms": q(x, .90) * 1e3, "p95_ms": q(x, .95) * 1e3,
        "max_ms": float(x.max() * 1e3),
        "n_exact_zero": int((x == 0).sum()), "frac_exact_zero": float((x == 0).mean()),
        "n_noise_level_positive(0<H<=eps)": int(((x > 0) & (x <= EPS_S)).sum()),
        "n_positive_above_eps": int(len(pos)), "frac_positive_above_eps": float(len(pos) / len(x)),
        "pos_mean_ms": float(pos.mean() * 1e3) if len(pos) else float("nan"),
        "pos_median_ms": q(pos, .5) * 1e3, "pos_p25_ms": q(pos, .25) * 1e3, "pos_p75_ms": q(pos, .75) * 1e3,
        "pos_p90_ms": q(pos, .90) * 1e3, "pos_p95_ms": q(pos, .95) * 1e3,
    }


def distribution_table(states: pd.DataFrame) -> pd.DataFrame:
    rows = [{"scope": "global", "scope_id": "all", "scope_label": "All 720 states", **dist_stats(states["oracle_headroom"].to_numpy())}]
    for w in sorted(states["source_dataset"].unique()):
        rows.append({"scope": "workload", "scope_id": w, "scope_label": WORKLOAD_LABELS[w], **dist_stats(states.loc[states["source_dataset"] == w, "oracle_headroom"].to_numpy())})
    for r in _regimes(states):
        g = states[states["regime"] == r]
        rows.append({"scope": "regime", "scope_id": r, "scope_label": REGIME_LABELS.get(r, r), **dist_stats(g["oracle_headroom"].to_numpy())})
    return pd.DataFrame(rows)


def signed_action_summary(actions: pd.DataFrame) -> pd.DataFrame:
    """Signed A(s,a) BEFORE the max/truncation.  Descriptive only -- not the primary estimand."""
    rows = []
    scopes = [("global", "all", actions)] + [("regime", r, actions[actions["regime"] == r]) for r in REGIME_ORDER]
    for scope, sid, g in scopes:
        a = g["a_lat"].to_numpy(float)
        rows.append({"scope": scope, "scope_id": sid, "scope_label": REGIME_LABELS.get(sid, "All 831 branches"), "n_actions": len(a),
                     "n_positive": int((a > EPS_S).sum()), "n_negative": int((a < -EPS_S).sum()), "n_zero_within_eps": int((np.abs(a) <= EPS_S).sum()),
                     "frac_positive": float((a > EPS_S).mean()), "frac_negative": float((a < -EPS_S).mean()),
                     "mean_ms": float(a.mean() * 1e3), "median_ms": float(np.median(a) * 1e3), "sd_ms": float(a.std(ddof=1) * 1e3),
                     "p05_ms": float(np.quantile(a, .05) * 1e3), "p25_ms": float(np.quantile(a, .25) * 1e3),
                     "p75_ms": float(np.quantile(a, .75) * 1e3), "p95_ms": float(np.quantile(a, .95) * 1e3),
                     "best_gain_ms": float(a.max() * 1e3), "worst_harm_ms": float(a.min() * 1e3)})
    return pd.DataFrame(rows)


def relative_headroom(states: pd.DataFrame, actions: pd.DataFrame) -> pd.DataFrame:
    """H_REL(s) = max(0, max_a A(s,a) / L_SBS(s)) with L_SBS recovered from the action-level artifact (supplementary)."""
    g = actions.groupby("state_id")
    ref = g["ref_mean_latency"].mean()
    best = g["a_lat"].max()
    rel = (best / ref).clip(lower=0).reindex(states["state_id"])
    out = states[["state_id", "regime", "window"]].copy()
    out["ref_mean_latency_s"] = ref.reindex(states["state_id"]).to_numpy()
    out["H_rel"] = rel.to_numpy()
    return out


# =============================================================================================
# C. weighting sensitivity
# =============================================================================================
def _unit_means(df: pd.DataFrame, unit_cols: Sequence[str]) -> pd.DataFrame:
    g = df.groupby(list(unit_cols), sort=True)
    return pd.DataFrame({"n": g.size(), "S": g["oracle_headroom"].sum(), "B": g["oracle_headroom"].apply(lambda v: float((v > EPS_S).sum())),
                         "B_canon": g["beneficial_opportunity"].sum().astype(float)})


def weighting_estimands(states: pd.DataFrame) -> pd.DataFrame:
    """Point estimates under alternative aggregation estimands.

    Let unit u partition D, D_u its states, n_u=|D_u|, U the number of units.
      state-weighted : theta = sum_s H(s) / |D|                              (canonical; ratio of totals)
      equal-unit     : theta_U = (1/U) sum_u [ (1/n_u) sum_{s in D_u} H(s) ]  (mean of per-unit means)
    Equal-unit weighting is the definition of that estimand, not an error; the pooled estimand is never computed as
    a mean of unit means.  Beneficial fraction P is aggregated identically.
    """
    rows = []

    def add(name, unit_cols, note):
        if unit_cols is None:
            n = len(states)
            m = states["oracle_headroom"].mean()
            p_c = states["beneficial_opportunity"].mean()
            p_g = (states["oracle_headroom"] > EPS_S).mean()
            rows.append({"estimand": name, "n_units": 1, "mean_H_ms": m * 1e3, "P_beneficial_canonical": p_c, "P_beneficial_eps_guarded": p_g, "note": note})
            return
        u = _unit_means(states, unit_cols)
        mh = (u["S"] / u["n"]).mean()
        rows.append({"estimand": name, "n_units": len(u), "mean_H_ms": mh * 1e3,
                     "P_beneficial_canonical": (u["B_canon"] / u["n"]).mean(), "P_beneficial_eps_guarded": (u["B"] / u["n"]).mean(), "note": note})

    add("state-weighted (canonical primary)", None, "ratio of totals; each of 720 states weight 1/720")
    add("equal-window", ["source_dataset", "window_index"], "each of 36 windows weight 1/36; window mean pools all its states across regimes")
    add("equal-regime", ["regime"], "each of 5 regimes weight 1/5")
    add("equal-workload (supplementary)", ["source_dataset"], "each of 2 workloads weight 1/2")
    add("equal window x regime cell (supplementary)", ["regime", "source_dataset", "window_index"], "each of 49 window-regime cells weight 1/49")
    return pd.DataFrame(rows)


def weighting_cis(states: pd.DataFrame, reps: int, seed: int) -> pd.DataFrame:
    """Window-cluster bootstrap CIs for each weighting (post-hoc; the canonical resampling scheme applied to each estimand).

    * state-weighted, equal-window : resample the 36 windows (all states retained).
    * equal-regime / equal-workload / equal-cell : regimes(workloads) are fixed design strata, so we resample window-regime
      (window-workload) cells *within* each stratum and average the stratum means.  This treats cells of different strata
      as independent (a window contributing to two regimes is resampled independently in each) -- an approximation that is
      flagged as descriptive.  The 1-window regime (Conv, KV 16,000) contributes zero resampling variance.
    """
    out = []
    cl = cluster_arrays(states)
    S, N = cl["S"], cl["N"]
    b = boot_ratio(S[:, None], N, reps, seed)[:, 0]
    out.append(("state-weighted (canonical primary)", *pct_ci(b)))
    m = S / N
    bw = boot_ratio(m[:, None], np.ones_like(N), reps, seed)[:, 0]  # mean of window means
    out.append(("equal-window", *pct_ci(bw)))

    def stratified(strat_col: str):
        rng = np.random.default_rng(seed)
        keys = list(dict.fromkeys([strat_col, "source_dataset", "window_index"]))
        cells = states.groupby(keys)["oracle_headroom"].agg(["sum", "count"]).reset_index()
        est = np.zeros(reps)
        strata = sorted(cells[strat_col].unique())
        for s in strata:
            c = cells[cells[strat_col] == s]
            Sc, Nc = c["sum"].to_numpy(float), c["count"].to_numpy(float)
            K = len(c)
            acc = np.empty(reps)
            done = 0
            while done < reps:
                k = min(20000, reps - done)
                idx = rng.integers(0, K, size=(k, K))
                acc[done:done + k] = Sc[idx].sum(1) / Nc[idx].sum(1)
                done += k
            est += acc
        return est / len(strata)

    out.append(("equal-regime", *pct_ci(stratified("regime"))))
    out.append(("equal-workload (supplementary)", *pct_ci(stratified("source_dataset"))))
    return pd.DataFrame([{"estimand": n, "boot_ci95_low_ms": lo * 1e3, "boot_ci95_high_ms": hi * 1e3, "boot_replicates": reps, "boot_seed": seed} for n, lo, hi in out])


# =============================================================================================
# D. leave-one-out
# =============================================================================================
def _agg(df: pd.DataFrame) -> dict:
    h = df["oracle_headroom"].to_numpy(float)
    return {"n_states": len(df), "n_windows": int(df.groupby(["source_dataset", "window_index"]).ngroups),
            "mean_H_ms": float(h.mean() * 1e3), "P_beneficial_canonical": float((df["beneficial_opportunity"] == 1).mean()),
            "P_beneficial_eps_guarded": float((h > EPS_S).mean()), "sum_H_s": float(h.sum())}


def leave_one_out(states: pd.DataFrame, unit_col: str, label_map: dict | None = None) -> pd.DataFrame:
    """Recompute the POOLED (state-weighted) mean and beneficial fraction after deleting each unit."""
    full = _agg(states)
    rows = []
    for u in sorted(states[unit_col].unique()):
        rest = states[states[unit_col] != u]
        a = _agg(rest)
        removed = states[states[unit_col] == u]
        rows.append({"omitted": u, "omitted_label": (label_map or {}).get(u, u), "n_states_omitted": len(removed),
                     "share_of_total_headroom_omitted": float(removed["oracle_headroom"].sum() / states["oracle_headroom"].sum()), **a,
                     "delta_mean_H_ms": a["mean_H_ms"] - full["mean_H_ms"],
                     "delta_P_canonical": a["P_beneficial_canonical"] - full["P_beneficial_canonical"]})
    return pd.DataFrame(rows)


def loo_extremes(loo: pd.DataFrame) -> dict:
    lo_i, hi_i = loo["mean_H_ms"].idxmin(), loo["mean_H_ms"].idxmax()
    plo, phi = loo["P_beneficial_canonical"].idxmin(), loo["P_beneficial_canonical"].idxmax()
    return {"mean_H_ms_min": float(loo.loc[lo_i, "mean_H_ms"]), "mean_H_ms_min_omitted": str(loo.loc[lo_i, "omitted"]),
            "mean_H_ms_max": float(loo.loc[hi_i, "mean_H_ms"]), "mean_H_ms_max_omitted": str(loo.loc[hi_i, "omitted"]),
            "P_min": float(loo.loc[plo, "P_beneficial_canonical"]), "P_min_omitted": str(loo.loc[plo, "omitted"]),
            "P_max": float(loo.loc[phi, "P_beneficial_canonical"]), "P_max_omitted": str(loo.loc[phi, "omitted"]),
            "all_loo_mean_H_positive": bool((loo["mean_H_ms"] > 0).all())}


def loo_remainder_ci(states: pd.DataFrame, col: str, reps: int, seed: int) -> pd.DataFrame:
    """Post-hoc descriptive: window-cluster bootstrap CI of the pooled mean on the remainder after omitting each unit."""
    rows = []
    for u in sorted(states[col].unique()):
        rest = states[states[col] != u]
        cl = cluster_arrays(rest)
        b = boot_ratio(cl["S"][:, None], cl["N"], reps, seed)[:, 0]
        lo, hi = pct_ci(b)
        rows.append({"omitted": u, "n_states": len(rest), "n_windows": cl["K"], "mean_H_ms": float(rest["oracle_headroom"].mean() * 1e3),
                     "boot_ci95_low_ms": lo * 1e3, "boot_ci95_high_ms": hi * 1e3, "ci_excludes_zero": bool(lo > 0),
                     "boot_replicates": reps, "boot_seed": seed})
    return pd.DataFrame(rows)


# =============================================================================================
# E. cluster-level robustness
# =============================================================================================
def cluster_robustness(states: pd.DataFrame, large_reps: int, mc_seeds: int) -> dict:
    cl = cluster_arrays(states)
    S, N, K = cl["S"], cl["N"], cl["K"]
    theta = float(S.sum() / N.sum())
    out: dict = {"design": {
        "cluster_unit": "faithful source window = (source_dataset, window_index)", "n_clusters": K,
        "clusters_by_workload": {w: int(sum(1 for k in cl["keys"] if k[0] == w)) for w in sorted(states["source_dataset"].unique())},
        "states_per_cluster": {"min": int(N.min()), "median": float(np.median(N)), "max": int(N.max())},
        "resampling": "windows sampled with replacement, all states of a sampled window retained with multiplicity",
        "estimator_in_each_replicate": "pooled state-weighted mean (sum H / sum n over sampled windows)",
        "seed": CANON_SEED, "replicates": CANON_REPLICATES, "ci": "percentile 2.5% / 97.5% (numpy default linear quantile)",
        "min_windows_rule": 5}}
    # canonical-size run with canonical seed (exact reproduction of the manuscript CI)
    canon = boot_ratio(S[:, None], N, CANON_REPLICATES, CANON_SEED)[:, 0]
    out["canonical_2000_seed20260920"] = {"ci95_s": pct_ci(canon), "ci95_ms": [v * 1e3 for v in pct_ci(canon)]}
    # Monte-Carlo noise of the canonical-size procedure across seeds
    lows, highs = [], []
    rng_seeds = 30_000_000 + np.arange(mc_seeds)
    for sd in rng_seeds:
        v = boot_ratio(S[:, None], N, CANON_REPLICATES, int(sd), chunk=CANON_REPLICATES)[:, 0]
        lo, hi = pct_ci(v)
        lows.append(lo); highs.append(hi)
    lows, highs = np.array(lows) * 1e3, np.array(highs) * 1e3
    out["mc_noise_canonical_size"] = {"n_seeds": mc_seeds, "replicates_each": CANON_REPLICATES,
                                      "ci_low_ms": {"mean": float(lows.mean()), "sd": float(lows.std(ddof=1)), "min": float(lows.min()), "max": float(lows.max()), "p2.5": float(np.quantile(lows, .025)), "p97.5": float(np.quantile(lows, .975))},
                                      "ci_high_ms": {"mean": float(highs.mean()), "sd": float(highs.std(ddof=1)), "min": float(highs.min()), "max": float(highs.max())},
                                      "frac_seeds_ci_low_positive": float((lows > 0).mean())}
    # large-replicate bootstrap with several seeds
    large = {}
    pooled = None
    for sd in BOOT_SEEDS_LARGE:
        v = boot_ratio(S[:, None], N, large_reps, sd)[:, 0]
        lo, hi = pct_ci(v)
        large[str(sd)] = {"replicates": large_reps, "ci95_ms": [lo * 1e3, hi * 1e3], "frac_boot_le_0": float((v <= 0).mean()),
                          "boot_mean_ms": float(v.mean() * 1e3), "boot_sd_ms": float(v.std(ddof=1) * 1e3)}
        if pooled is None:
            pooled = v
    out["large_bootstrap"] = large
    # BCa + jackknife using the first large run
    th_i = jackknife_ratio(S, N)
    out["jackknife_window"] = jackknife_summary(theta, th_i)
    out["jackknife_window"]["loo_estimates_ms"] = {f"{k[0]}:w{k[1]}": float(v * 1e3) for k, v in zip(cl["keys"], th_i)}
    out["bca_large_bootstrap"] = bca_ci(theta, pooled, th_i)
    out["bca_large_bootstrap"]["ci95_ms"] = [v * 1e3 for v in out["bca_large_bootstrap"]["ci95"]]
    # bimodality diagnostic: split bootstrap replicates by whether the dominant window is resampled at all
    top_k = int(np.argmax(S))
    rng = np.random.default_rng(CANON_SEED)
    reps_d = MID_BOOT_REPLICATES
    idx = rng.integers(0, K, size=(reps_d, K))
    vals = S[idx].sum(1) / N[idx].sum(1)
    has = (idx == top_k).any(axis=1)
    def sub(v):
        lo, hi = pct_ci(v)
        return {"n": int(len(v)), "mean_ms": float(v.mean() * 1e3), "ci95_ms": [lo * 1e3, hi * 1e3], "min_ms": float(v.min() * 1e3), "max_ms": float(v.max() * 1e3)}
    out["dominant_window_bootstrap_split"] = {
        "window": "%s:w%d" % cl["keys"][top_k], "replicates": reps_d,
        "frac_replicates_without_dominant_window": float((~has).mean()), "analytic_(1-1/K)^K": float((1 - 1 / K) ** K),
        "without_dominant_window": sub(vals[~has]), "with_dominant_window": sub(vals[has]),
        "note": "the bootstrap distribution is a mixture; the pooled percentile CI spans the valley between the two components"}
    # basic (reverse-percentile) interval and studentised-free diagnostics
    plo, phi = pct_ci(pooled)
    out["basic_interval_large_bootstrap_ms"] = [(2 * theta - phi) * 1e3, (2 * theta - plo) * 1e3]
    out["bootstrap_distribution_skew"] = float(sps.skew(pooled))
    out["theta_hat_ms"] = theta * 1e3
    # concentration diagnostics that bear on the effective number of clusters
    share = S / S.sum()
    out["effective_number_of_windows_by_headroom_mass"] = float(1.0 / (share ** 2).sum())
    out["effective_number_of_windows_by_states"] = float(1.0 / (((N / N.sum()) ** 2).sum()))
    out["top_window_share_of_total_headroom"] = float(share.max())
    return out


# =============================================================================================
# F. composition / concentration
# =============================================================================================
def composition(states: pd.DataFrame) -> dict:
    tot_H, n = float(states["oracle_headroom"].sum()), len(states)
    full_mean = tot_H / n
    def table(col, order=None, labels=None):
        g = states.groupby(col)
        t = pd.DataFrame({"n_states": g.size(), "sum_H_s": g["oracle_headroom"].sum(), "mean_H_ms": g["oracle_headroom"].mean() * 1e3,
                          "n_windows": g["window"].nunique(), "P_beneficial_canonical": g["beneficial_opportunity"].mean()})
        t["share_of_states"] = t["n_states"] / n
        t["share_of_total_headroom"] = t["sum_H_s"] / tot_H
        t["contribution_to_pooled_mean_ms"] = t["sum_H_s"] / n * 1e3  # = share_of_states * mean_H ; sums to pooled mean
        if order:  # canonical keys first (manuscript order), then any others
            t = t.reindex([o for o in order if o in t.index] + sorted(i for i in t.index if i not in order))
        t.insert(0, "label", [(labels or {}).get(i, i) for i in t.index])
        return t.reset_index().rename(columns={col: "id"})
    reg = table("regime", REGIME_ORDER, REGIME_LABELS)
    wl = table("source_dataset", None, WORKLOAD_LABELS)
    ax = table("axis")
    win = table("window")
    win = win.sort_values("sum_H_s", ascending=False).reset_index(drop=True)
    win["cum_share_of_total_headroom"] = win["share_of_total_headroom"].cumsum()
    win["rank"] = np.arange(1, len(win) + 1)
    # state-level concentration
    h = np.sort(states["oracle_headroom"].to_numpy(float))[::-1]
    cum = np.cumsum(h) / h.sum()
    def top_share(frac): k = int(math.ceil(frac * n)); return float(cum[k - 1])
    gini = float((2 * np.arange(1, n + 1) - n - 1).dot(np.sort(h)) / (n * h.sum()))
    share_w = win["share_of_total_headroom"].to_numpy()
    top = reg.loc[reg["sum_H_s"].idxmax()]
    rest = states[states["regime"] != top["id"]]
    top_w = str(win.loc[0, "id"])
    tw = states[states["window"] == top_w].copy()
    tw["step"] = pd.to_numeric(tw["state_id"].str.extract(r"step(\d+)$")[0], errors="coerce")
    prof = tw.groupby("regime").agg(n=("H_ms", "size"), sum_H_s=("oracle_headroom", "sum"), mean_H_ms=("H_ms", "mean"), min_step=("step", "min"), max_step=("step", "max"))
    dom_id = str(top["id"])
    dom_wo_top = states[(states["regime"] == dom_id) & (states["window"] != top_w)]
    high = {}
    for tau in (2.0, 5.0):
        m = states[exceeds(states["oracle_headroom"].to_numpy(float), tau)]
        cnt = m.groupby("window").size().sort_values(ascending=False)
        high[f"gt_{tau:g}_ms"] = {"n_states": int(len(m)), "n_windows": int(cnt.size), "by_window": {k: int(v) for k, v in cnt.items()},
                                   "share_from_top_window": float(cnt.iloc[0] / len(m)) if len(m) else float("nan")}
    conc = {
        "top_window_profile": {"window": top_w, "n_states": int(len(tw)), "by_regime": {k: {kk: (None if pd.isna(vv) else float(vv)) for kk, vv in r.items()} for k, r in prof.iterrows()},
                               "step_span": ([int(tw["step"].min()), int(tw["step"].max())] if tw["step"].notna().any() else None), "mean_H_ms": float(tw["H_ms"].mean()),
                               "P_beneficial_canonical": float(tw["beneficial_opportunity"].mean())},
        "dominant_regime_without_top_window": {"n_states": int(len(dom_wo_top)), "n_windows": int(dom_wo_top["window"].nunique()), "mean_H_ms": float(dom_wo_top["H_ms"].mean()),
                                               "sum_H_s": float(dom_wo_top["oracle_headroom"].sum())},
        "high_headroom_states_by_window": high,
        "pooled_mean_ms": full_mean * 1e3, "total_headroom_s": tot_H,
        "dominant_regime": top["id"], "dominant_regime_label": top["label"],
        "dominant_regime_share_of_states": float(top["share_of_states"]),
        "dominant_regime_share_of_total_headroom": float(top["share_of_total_headroom"]),
        "dominant_regime_contribution_to_pooled_mean_ms": float(top["contribution_to_pooled_mean_ms"]),
        "pooled_mean_without_dominant_regime_ms": float(rest["oracle_headroom"].mean() * 1e3),
        "states_without_dominant_regime": len(rest),
        "state_top_share": {"top_1pct": top_share(.01), "top_5pct": top_share(.05), "top_10pct": top_share(.10), "top_20pct": top_share(.20)},
        "n_states_for_50pct_of_headroom": int(np.searchsorted(cum, 0.5) + 1), "n_states_for_80pct_of_headroom": int(np.searchsorted(cum, 0.8) + 1),
        "state_gini": gini,
        "n_windows_for_50pct": int(np.searchsorted(win["cum_share_of_total_headroom"].to_numpy(), 0.5) + 1),
        "n_windows_for_80pct": int(np.searchsorted(win["cum_share_of_total_headroom"].to_numpy(), 0.8) + 1),
        "top_window": str(win.loc[0, "id"]), "top_window_share": float(share_w[0]),
        "top3_window_share": float(share_w[:3].sum()),
        "effective_number_of_windows_by_headroom": float(1.0 / (share_w ** 2).sum()),
        "effective_number_of_regimes_by_headroom": float(1.0 / (reg["share_of_total_headroom"] ** 2).sum()),
        "windows_total": len(win), "windows_in_dominant_regime": int(top["n_windows"]),
    }
    return {"regime": reg, "workload": wl, "axis": ax, "window": win, "summary": conc}


# =============================================================================================
# figures
# =============================================================================================
OKABE = {"azure_2023_code|active_sequence_capacity|active_8": "#E69F00", "azure_2023_code|active_sequence_capacity|active_4": "#56B4E9",
         "azure_2023_code|kv_capacity|kv_16000": "#D55E00", "azure_2023_conv|active_sequence_capacity|active_4": "#009E73",
         "azure_2023_conv|kv_capacity|kv_16000": "#0072B2"}
LINESTYLE = {"azure_2023_code|active_sequence_capacity|active_8": (0, (1, 1)), "azure_2023_code|active_sequence_capacity|active_4": "-.",
             "azure_2023_code|kv_capacity|kv_16000": "-", "azure_2023_conv|active_sequence_capacity|active_4": "--",
             "azure_2023_conv|kv_capacity|kv_16000": (0, (5, 1, 1, 1))}
WL_MARKER = {"azure_2023_code": "o", "azure_2023_conv": "s"}


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5, "legend.fontsize": 7, "xtick.labelsize": 7.5,
                         "ytick.labelsize": 7.5, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 150, "savefig.dpi": 300, "axes.grid": True, "grid.color": "0.88", "grid.linewidth": 0.5})
    return plt


def _save(fig, out: Path, name: str) -> None:
    (out / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "figures" / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / "figures" / f"{name}.png", bbox_inches="tight")


def fig_distribution(states: pd.DataFrame, out: Path) -> None:
    plt = _mpl()
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.2))
    lin = 0.01
    def ecdf(a, x, y, **kw):
        xs = np.sort(x)
        a.step(np.concatenate([[0], xs]), np.concatenate([[0], np.arange(1, len(xs) + 1) / len(xs)]), where="post", **kw)
    x_all = np.where(states["oracle_headroom"].to_numpy() <= EPS_S, 0.0, states["H_ms"].to_numpy())
    for r in REGIME_ORDER:
        g = states[states["regime"] == r]
        xr = np.where(g["oracle_headroom"].to_numpy() <= EPS_S, 0.0, g["H_ms"].to_numpy())
        ecdf(ax[0], xr, None, color=OKABE[r], ls=LINESTYLE[r], lw=1.3, label=f"{REGIME_LABELS[r]} (n={len(g)})")
    ecdf(ax[0], x_all, None, color="black", lw=2.0, label="All states (n=720)")
    for t in THRESHOLDS_MS[1:]:
        ax[0].axvline(t, color="0.55", lw=0.6, ls=":")
    ax[0].axvline(states["H_ms"].mean(), color="black", lw=0.9, ls="--")
    ax[0].text(states["H_ms"].mean() * 1.08, 0.42, f"pooled mean\n{states['H_ms'].mean():.2f} ms", fontsize=6.5, va="center")
    ax[0].set_xscale("symlog", linthresh=lin, linscale=0.4)
    ax[0].set_xlim(0, 15)
    ax[0].set_xticks([0, 0.01, 0.1, 1, 10]); ax[0].set_xticklabels(["0", "0.01", "0.1", "1", "10"])
    ax[0].set_xlabel("Oracle latency headroom $H_{LAT}(s)$ (ms, symlog)")
    ax[0].set_ylabel("Empirical CDF over disagreement states")
    ax[0].set_ylim(0, 1.0)
    ax[0].set_title("(a) Distribution of headroom by regime")
    ax[0].legend(loc="lower right", frameon=True, framealpha=0.9, fontsize=6.3)
    # (b) count-ECDF vs mass-weighted CDF
    xs = np.sort(x_all)
    hs = np.sort(states["oracle_headroom"].to_numpy())
    mass = np.cumsum(hs) / hs.sum()
    ax[1].step(np.concatenate([[0], xs]), np.concatenate([[0], np.arange(1, 721) / 720]), where="post", color="black", lw=1.8, label="share of states with $H \\leq x$")
    ax[1].step(np.concatenate([[0], np.sort(x_all)]), np.concatenate([[0], mass]), where="post", color="0.45", lw=1.8, ls="--", label="share of total headroom in those states")
    ax[1].set_xscale("symlog", linthresh=lin, linscale=0.4)
    ax[1].set_xlim(0, 15)
    ax[1].set_xticks([0, 0.01, 0.1, 1, 10]); ax[1].set_xticklabels(["0", "0.01", "0.1", "1", "10"])
    ax[1].set_xlabel("Oracle latency headroom $x$ (ms, symlog)")
    ax[1].set_ylabel("Cumulative share")
    ax[1].set_ylim(0, 1.0)
    ax[1].set_title("(b) Headroom mass is carried by few states")
    ax[1].legend(loc="upper left", frameon=True, framealpha=0.9)
    fig.tight_layout()
    _save(fig, out, "fig1_headroom_distribution")
    plt.close(fig)


def fig_loo(states: pd.DataFrame, loo_w: pd.DataFrame, loo_r: pd.DataFrame, out: Path) -> None:
    plt = _mpl()
    full_m = states["H_ms"].mean()
    full_p = (states["beneficial_opportunity"] == 1).mean()
    fig, ax = plt.subplots(2, 2, figsize=(7.4, 5.4), gridspec_kw={"width_ratios": [1, 1.5]})
    # (a) regime LOO mean
    r = loo_r.set_index("omitted").reindex(REGIME_ORDER)
    y = np.arange(len(r))[::-1]
    ax[0, 0].barh(y, r["mean_H_ms"], color="0.55", edgecolor="black", hatch=["", "", "//", "", ""], height=0.6)
    ax[0, 0].axvline(full_m, color="black", ls="--", lw=1, label=f"all 720 states ({full_m:.3f} ms)")
    ax[0, 0].legend(loc="center right", frameon=True, framealpha=0.9)
    ax[0, 0].set_yticks(y); ax[0, 0].set_yticklabels([f"omit: {REGIME_LABELS[k]}" for k in r.index])
    for yi, v in zip(y, r["mean_H_ms"]):
        ax[0, 0].text(v + 0.03, yi, f"{v:.2f}", va="center", fontsize=6.8)
    ax[0, 0].set_xlabel("Pooled mean headroom (ms)")
    ax[0, 0].set_title("(a) Leave-one-regime-out: mean")
    ax[0, 0].set_xlim(0, full_m * 1.35)
    # (b) window LOO mean
    w = loo_w.sort_values("mean_H_ms").reset_index(drop=True)
    for _, row in w.iterrows():
        pass
    for wl, mk in WL_MARKER.items():
        m = w["omitted"].str.startswith(wl)
        ax[0, 1].scatter(w.index[m], w.loc[m, "mean_H_ms"], marker=mk, s=22, facecolor="white" if wl.endswith("conv") else "black", edgecolor="black", linewidth=0.8, label=WORKLOAD_LABELS[wl], zorder=3)
    ax[0, 1].axhline(full_m, color="black", ls="--", lw=1, label=f"all 720 states ({full_m:.3f} ms)")
    lo, hi = w.iloc[0], w.iloc[-1]
    ax[0, 1].annotate(f"min: omit {lo['omitted']}\n{lo['mean_H_ms']:.3f} ms", (0, lo["mean_H_ms"]), xytext=(5, 0.85), fontsize=6.5, arrowprops=dict(arrowstyle="-", lw=0.5))
    ax[0, 1].annotate(f"max: omit {hi['omitted']}\n{hi['mean_H_ms']:.3f} ms", (len(w) - 1, hi["mean_H_ms"]), xytext=(14, 2.55), fontsize=6.5, arrowprops=dict(arrowstyle="-", lw=0.5))
    ax[0, 1].set_ylim(0, full_m * 1.45)
    ax[0, 1].set_xlabel("Omitted window (sorted by resulting estimate)")
    ax[0, 1].set_ylabel("Pooled mean headroom (ms)")
    ax[0, 1].set_title("(b) Leave-one-window-out: mean (36 windows)")
    ax[0, 1].legend(loc="center right", frameon=True, framealpha=0.9)
    # (c) regime LOO fraction
    ax[1, 0].barh(y, r["P_beneficial_canonical"], color="0.55", edgecolor="black", hatch=["", "", "//", "", ""], height=0.6)
    ax[1, 0].axvline(full_p, color="black", ls="--", lw=1)
    ax[1, 0].set_yticks(y); ax[1, 0].set_yticklabels([f"omit: {REGIME_LABELS[k]}" for k in r.index])
    for yi, v in zip(y, r["P_beneficial_canonical"]):
        ax[1, 0].text(v - 0.012, yi, f"{v:.3f}", va="center", ha="right", fontsize=6.8, bbox=dict(facecolor="white", edgecolor="none", pad=1.0, alpha=0.9))
    ax[1, 0].set_xlim(0, 1.0)
    ax[1, 0].set_xlabel("Beneficial fraction $P(B_{LAT}|D)$")
    ax[1, 0].set_title("(c) Leave-one-regime-out: beneficial fraction")
    # (d) window LOO fraction
    w2 = loo_w.sort_values("P_beneficial_canonical").reset_index(drop=True)
    for wl, mk in WL_MARKER.items():
        m = w2["omitted"].str.startswith(wl)
        ax[1, 1].scatter(w2.index[m], w2.loc[m, "P_beneficial_canonical"], marker=mk, s=22, facecolor="white" if wl.endswith("conv") else "black", edgecolor="black", linewidth=0.8, zorder=3, label=WORKLOAD_LABELS[wl])
    ax[1, 1].axhline(full_p, color="black", ls="--", lw=1)
    ax[1, 1].set_ylim(0.78, 0.86)
    ax[1, 1].set_xlabel("Omitted window (sorted by resulting estimate)")
    ax[1, 1].set_ylabel("Beneficial fraction")
    ax[1, 1].set_title("(d) Leave-one-window-out: beneficial fraction")
    fig.tight_layout()
    _save(fig, out, "fig2_leave_one_out")
    plt.close(fig)


def fig_composition(comp: dict, out: Path) -> None:
    plt = _mpl()
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.1), gridspec_kw={"width_ratios": [1.35, 1]})
    reg = comp["regime"]
    y = np.arange(len(reg))[::-1]
    h = 0.36
    ax[0].barh(y + h / 2, reg["share_of_states"], height=h, color="white", edgecolor="black", hatch="//", label="share of the 720 states")
    ax[0].barh(y - h / 2, reg["share_of_total_headroom"], height=h, color="0.25", edgecolor="black", label="share of total summed headroom")
    for yi, a, b in zip(y, reg["share_of_states"], reg["share_of_total_headroom"]):
        ax[0].text(a + 0.01, yi + h / 2, f"{a:.1%}", va="center", fontsize=6.5)
        ax[0].text(b + 0.01, yi - h / 2, f"{b:.1%}", va="center", fontsize=6.5)
    ax[0].set_yticks(y); ax[0].set_yticklabels(reg["label"])
    ax[0].set_xlim(0, 1.05)
    ax[0].set_xlabel("Share")
    ax[0].set_title("(a) Regime composition")
    ax[0].legend(loc="lower right", frameon=True, framealpha=0.9)
    win = comp["window"]
    ax[1].plot(np.concatenate([[0], win["rank"]]), np.concatenate([[0], win["cum_share_of_total_headroom"]]), color="black", lw=1.6, marker="o", ms=2.5, label="cumulative share of headroom")
    ax[1].plot([0, len(win)], [0, 1], color="0.5", ls="--", lw=1, label="equal contribution")
    ax[1].set_xlabel("Windows ranked by total headroom")
    ax[1].set_ylabel("Cumulative share of total headroom")
    ax[1].set_title("(b) Window concentration")
    ax[1].set_xlim(0, len(win)); ax[1].set_ylim(0, 1.02)
    ax[1].legend(loc="lower right", frameon=True, framealpha=0.9)
    fig.tight_layout()
    _save(fig, out, "fig3_composition")
    plt.close(fig)


def fig_robustness(weights: pd.DataFrame, wci: pd.DataFrame, clus: dict, states: pd.DataFrame, out: Path) -> None:
    plt = _mpl()
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.1), gridspec_kw={"width_ratios": [1.15, 1]})
    m = weights.merge(wci, on="estimand", how="left")
    m = m[m["estimand"].isin(["state-weighted (canonical primary)", "equal-window", "equal-regime", "equal-workload (supplementary)"])]
    y = np.arange(len(m))[::-1]
    ax[0].errorbar(m["mean_H_ms"], y, xerr=[m["mean_H_ms"] - m["boot_ci95_low_ms"], m["boot_ci95_high_ms"] - m["mean_H_ms"]], fmt="s", color="black", ecolor="black", capsize=3, ms=4.5, lw=1.1)
    for yi, v, hi_ in zip(y, m["mean_H_ms"], m["boot_ci95_high_ms"]):
        ax[0].text(hi_ + 0.06, yi, f"{v:.3f} ms", ha="left", va="center", fontsize=6.8)
    ax[0].set_xlim(-0.1, 4.6)
    ax[0].axvline(0, color="0.3", lw=0.8)
    ax[0].set_yticks(y); ax[0].set_yticklabels([e.replace(" (canonical primary)", "\n(canonical)").replace(" (supplementary)", "\n(suppl.)") for e in m["estimand"]])
    ax[0].set_xlabel("Mean oracle headroom (ms), 95% window-cluster bootstrap CI")
    ax[0].set_title("(a) Weighting sensitivity")
    cl = cluster_arrays(states)
    b = boot_ratio(cl["S"][:, None], cl["N"], 200_000, CANON_SEED)[:, 0] * 1e3
    ax[1].hist(b, bins=80, color="0.7", edgecolor="0.4", linewidth=0.3)
    c = clus["canonical_2000_seed20260920"]["ci95_ms"]; L = clus["large_bootstrap"][str(CANON_SEED)]["ci95_ms"]
    jk = clus["jackknife_window"]["ci95_t"]; bca = clus["bca_large_bootstrap"]["ci95_ms"]
    top = ax[1].get_ylim()[1]
    jkc = [v * 1e3 for v in clus["jackknife_window"]["ci95_t"]]
    items = [("canonical percentile, B=2,000", c, "-"), ("percentile, B=1,000,000", L, "--"), ("BCa, B=1,000,000", bca, "-."), ("delete-one-window jackknife (t)", jkc, ":")]
    ax[1].set_ylim(0, top * 1.85)
    for j, (lab, ci, ls) in enumerate(items):
        yy = top * (1.12 + 0.15 * j)
        ax[1].hlines(yy, ci[0], ci[1], color="black", ls=ls, lw=1.4)
        ax[1].plot(ci, [yy, yy], "|", color="black", ms=6)
        ax[1].text(-1.55, yy + top * 0.045, f"{lab}: [{ci[0]:.2f}, {ci[1]:.2f}] ms", fontsize=6, va="bottom", ha="left")
    ax[1].axvline(states["H_ms"].mean(), color="black", lw=0.6)
    ax[1].set_xlabel("Bootstrap pooled mean headroom (ms)")
    ax[1].set_ylabel("Replicates (of 200,000 shown)")
    ax[1].set_title("(b) Cluster-bootstrap distribution")
    ax[1].set_xlim(-1.6, 7.2)
    fig.tight_layout()
    _save(fig, out, "fig4_weighting_and_cluster_robustness")
    plt.close(fig)


# =============================================================================================
# markdown tables for the report
# =============================================================================================
def md_table(df: pd.DataFrame, fmt: dict | None = None) -> str:
    fmt = fmt or {}
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---:" if pd.api.types.is_numeric_dtype(df[c]) else "---" for c in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            f = fmt.get(c)
            cells.append(f(v) if f else (f"{v:.4g}" if isinstance(v, (float, np.floating)) else str(v)))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_tables(res: dict) -> str:
    out = []
    thr = res["thresholds"]
    g = thr[thr["scope"] == "global"].merge(res["threshold_ci"], on="threshold_ms")
    t1 = pd.DataFrame({"tau (ms)": g["threshold_ms"], "states > tau": g["n_exceed"].astype(str) + " / " + g["n_states"].astype(str),
                       "fraction": g["fraction"].map(lambda v: f"{v:.4f}"),
                       "95% window-cluster CI": [f"[{lo:.3f}, {hi:.3f}]" for lo, hi in zip(g["boot_ci95_low"], g["boot_ci95_high"])],
                       "windows with >=1 exceeding state": g["n_windows_with_exceedance"].astype(str) + " / " + g["n_windows"].astype(str),
                       "raw-FP strict count": g["n_exceed_raw_fp_strict"], "ties (|H-tau|<=1e-9 s)": g["n_ties_within_1e-9s"]})
    out += ["<<TABLE thresholds_global>>", md_table(t1), ""]
    piv = []
    for scope in ("workload", "regime"):
        s = thr[thr["scope"] == scope]
        for sid in s["scope_id"].unique():
            r = s[s["scope_id"] == sid]
            row = {"scope": r["scope_label"].iloc[0] + (" (low support)" if r["low_support"].iloc[0] else ""), "n": int(r["n_states"].iloc[0]), "windows": int(r["n_windows"].iloc[0])}
            for _, x in r.iterrows():
                row[f">{x.threshold_ms:g} ms"] = f"{x.n_exceed}/{x.n_states} ({x.fraction:.3f})"
            piv.append(row)
    out += ["<<TABLE thresholds_by_scope>>", md_table(pd.DataFrame(piv)), ""]
    d = res["distribution"]
    cols = ["scope_label", "N", "mean_ms", "median_ms", "sd_ms", "p25_ms", "p75_ms", "p90_ms", "p95_ms", "max_ms", "n_exact_zero", "frac_exact_zero"]
    out += ["<<TABLE distribution>>", md_table(d[cols].rename(columns={"scope_label": "scope"})), ""]
    cols2 = ["scope_label", "n_positive_above_eps", "pos_mean_ms", "pos_median_ms", "pos_p25_ms", "pos_p75_ms", "pos_p90_ms", "pos_p95_ms"]
    out += ["<<TABLE distribution_positive>>", md_table(d[cols2].rename(columns={"scope_label": "scope"})), ""]
    sa = res["signed_actions"]
    out += ["<<TABLE signed_actions>>", md_table(sa[["scope_label", "n_actions", "n_positive", "n_negative", "n_zero_within_eps", "mean_ms", "median_ms", "p05_ms", "p95_ms", "best_gain_ms", "worst_harm_ms"]].rename(columns={"scope_label": "scope"})), ""]
    w = res["weighting"].merge(res["weighting_ci"], on="estimand", how="left")
    w2 = pd.DataFrame({"estimand": w["estimand"], "units": w["n_units"], "mean H (ms)": w["mean_H_ms"].map(lambda v: f"{v:.4f}"),
                       "95% CI (ms)": [("n/a" if pd.isna(a) else f"[{a:.3f}, {b:.3f}]") for a, b in zip(w["boot_ci95_low_ms"], w["boot_ci95_high_ms"])],
                       "P(beneficial), canonical": w["P_beneficial_canonical"].map(lambda v: f"{v:.4f}"),
                       "P(beneficial), eps-guarded": w["P_beneficial_eps_guarded"].map(lambda v: f"{v:.4f}")})
    out += ["<<TABLE weighting>>", md_table(w2), ""]
    for key, nm in (("loo_regime", "loo_regime"), ("loo_group", "loo_group"), ("loo_window", "loo_window")):
        d = res[key].copy()
        cols = ["omitted_label", "n_states_omitted", "share_of_total_headroom_omitted", "n_states", "n_windows", "mean_H_ms", "delta_mean_H_ms", "P_beneficial_canonical", "P_beneficial_eps_guarded"]
        d = d[cols]
        if key == "loo_window":
            d = d.sort_values("mean_H_ms").reset_index(drop=True)
            d = pd.concat([d.head(5), d.tail(5)]) if len(d) > 10 else d
        out += [f"<<TABLE {nm}>>", md_table(d.rename(columns={"omitted_label": "omitted"}), {"share_of_total_headroom_omitted": lambda v: f"{v:.3f}"}), ""]
    lr = res["loo_regime_ci"].copy()
    lr["omitted"] = lr["omitted"].map(lambda k: REGIME_LABELS.get(k, k))
    out += ["<<TABLE loo_regime_ci>>", md_table(lr[["omitted", "n_states", "n_windows", "mean_H_ms", "boot_ci95_low_ms", "boot_ci95_high_ms", "ci_excludes_zero"]]), ""]
    lw = res["loo_window_ci"].sort_values("mean_H_ms").head(5)
    out += ["<<TABLE loo_window_ci_lowest5>>", md_table(lw[["omitted", "n_states", "n_windows", "mean_H_ms", "boot_ci95_low_ms", "boot_ci95_high_ms", "ci_excludes_zero"]]), ""]
    comp = res["composition"]
    cr = comp["regime"][["label", "n_states", "share_of_states", "n_windows", "sum_H_s", "share_of_total_headroom", "mean_H_ms", "contribution_to_pooled_mean_ms", "P_beneficial_canonical"]]
    out += ["<<TABLE composition_regime>>", md_table(cr.rename(columns={"label": "regime", "contribution_to_pooled_mean_ms": "contrib. to pooled mean (ms)"}),
                                                   {"share_of_states": lambda v: f"{v:.3f}", "share_of_total_headroom": lambda v: f"{v:.3f}", "sum_H_s": lambda v: f"{v:.4f}"}), ""]
    cw = comp["window"].head(10)[["rank", "id", "n_states", "share_of_states", "sum_H_s", "share_of_total_headroom", "cum_share_of_total_headroom", "mean_H_ms"]]
    out += ["<<TABLE composition_window_top10>>", md_table(cw, {"share_of_states": lambda v: f"{v:.3f}", "share_of_total_headroom": lambda v: f"{v:.3f}", "cum_share_of_total_headroom": lambda v: f"{v:.3f}", "sum_H_s": lambda v: f"{v:.4f}"}), ""]
    return "\n".join(out)


# =============================================================================================
# driver
# =============================================================================================
def run(out: Path, design: Path = DESIGN, large_reps: int = LARGE_BOOT_REPLICATES, mid_reps: int = MID_BOOT_REPLICATES, mc_seeds: int = MC_SEEDS) -> dict:
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    timings: dict = {}
    def tick(name):
        timings[name] = round(time.time() - t0, 2)

    states, actions = load_states(design), load_actions(design)
    hashes_before = {f: sha256_file(design / f) for f in CANONICAL_INPUTS if (design / f).exists()}
    canon_hash_ok = json.loads((design / "FRESH_LATENCY_ARTIFACT_HASHES_V1.json").read_text())["compact_artifacts"]
    verify = {f: (sha256_file(design / f) == h) for f, h in canon_hash_ok.items()}

    rep = reproduction_check(states, actions, design); tick("reproduction")
    dump_json(rep, out / "reproduction_check.json")
    if rep["overall"] == "FAILED":
        raise SystemExit("primary reproduction FAILED; refusing to run sensitivity analyses (see reproduction_check.json)")

    tol = tolerance_audit(states, actions); dump_json(tol, out / "numerical_tolerance_audit.json")
    thr = threshold_table(states); thr.to_csv(out / "practical_thresholds.csv", index=False)
    thr_ci = threshold_global_ci(states, mid_reps, CANON_SEED); thr_ci.to_csv(out / "practical_thresholds_global_ci.csv", index=False); tick("thresholds")

    dist = distribution_table(states); dist.to_csv(out / "distribution_summary.csv", index=False)
    signed = signed_action_summary(actions); signed.to_csv(out / "signed_action_advantage_summary.csv", index=False)
    rel = relative_headroom(states, actions)
    rel_summary = {"mean": float(rel["H_rel"].mean()), "median": float(rel["H_rel"].median()), "p90": float(rel["H_rel"].quantile(.9)),
                   "frac_gt_1pct": float((rel["H_rel"] > 0.01 + 1e-12).mean()), "frac_gt_5pct": float((rel["H_rel"] > 0.05 + 1e-12).mean()),
                   "mean_ref_latency_ms_state_weighted": float(rel["ref_mean_latency_s"].mean() * 1e3),
                   "by_regime_mean_rel": {REGIME_LABELS[r]: float(rel.loc[rel["regime"] == r, "H_rel"].mean()) for r in REGIME_ORDER},
                   "pooled_mean_H_over_pooled_mean_ref_latency": float(states["oracle_headroom"].mean() / rel["ref_mean_latency_s"].mean())}
    dump_json(rel_summary, out / "relative_headroom_summary.json"); tick("distribution")

    weights = weighting_estimands(states); weights.to_csv(out / "weighting_sensitivity.csv", index=False)
    wci = weighting_cis(states, mid_reps, CANON_SEED); wci.to_csv(out / "weighting_sensitivity_ci.csv", index=False); tick("weighting")

    loo_w = leave_one_out(states, "window"); loo_w.to_csv(out / "leave_one_window_out.csv", index=False)
    loo_r = leave_one_out(states, "regime", REGIME_LABELS); loo_r.to_csv(out / "leave_one_regime_out.csv", index=False)
    states_g = states.copy(); states_g["group"] = np.nan
    loo_g = pd.concat([leave_one_out(states, "source_dataset", WORKLOAD_LABELS).assign(kind="workload"), leave_one_out(states, "axis").assign(kind="axis")], ignore_index=True)
    loo_g.to_csv(out / "leave_one_group_out_supplementary.csv", index=False)
    loo_r_ci = loo_remainder_ci(states, "regime", mid_reps, CANON_SEED); loo_r_ci.to_csv(out / "leave_one_regime_out_remainder_ci.csv", index=False)
    loo_w_ci = loo_remainder_ci(states, "window", mid_reps, CANON_SEED); loo_w_ci.to_csv(out / "leave_one_window_out_remainder_ci.csv", index=False)
    loo_summary = {"window": loo_extremes(loo_w), "regime": loo_extremes(loo_r),
                   "window_remainder_ci": {"n_omissions_ci_excludes_zero": int(loo_w_ci["ci_excludes_zero"].sum()), "n_omissions": int(len(loo_w_ci)),
                                           "lowest_ci_low_ms": float(loo_w_ci["boot_ci95_low_ms"].min()),
                                           "lowest_ci_low_omitted": str(loo_w_ci.loc[loo_w_ci["boot_ci95_low_ms"].idxmin(), "omitted"]),
                                           "omit_top_window": loo_w_ci.sort_values("mean_H_ms").iloc[0].to_dict()}}
    dump_json(loo_summary, out / "leave_one_out_summary.json"); tick("loo")

    clus = cluster_robustness(states, large_reps, mc_seeds); dump_json(clus, out / "cluster_robustness.json"); tick("cluster")

    comp = composition(states)
    comp["regime"].to_csv(out / "composition_regime.csv", index=False)
    comp["workload"].to_csv(out / "composition_workload.csv", index=False)
    comp["axis"].to_csv(out / "composition_axis.csv", index=False)
    comp["window"].to_csv(out / "composition_window.csv", index=False)
    dump_json(comp["summary"], out / "composition_summary.json"); tick("composition")

    fig_distribution(states, out); fig_loo(states, loo_w, loo_r, out); fig_composition(comp, out); fig_robustness(weights, wci, clus, states, out); tick("figures")

    res = {"thresholds": thr, "threshold_ci": thr_ci, "distribution": dist, "signed_actions": signed, "weighting": weights, "weighting_ci": wci,
           "loo_window": loo_w, "loo_regime": loo_r, "loo_group": loo_g, "loo_regime_ci": loo_r_ci, "loo_window_ci": loo_w_ci, "composition": comp}
    (out / "report_tables.md").write_text(build_tables(res))

    hashes_after = {f: sha256_file(design / f) for f in hashes_before}
    prov = {
        "analysis_status": "POST_HOC_ROBUSTNESS_SENSITIVITY (not preregistered, not confirmatory)",
        "git_head": git("rev-parse", "HEAD"), "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_status_of_canonical_dir": git("status", "--porcelain", "--", str(design.relative_to(ROOT))),
        "script": "scripts/fresh_causal_robustness_v1.py", "script_sha256": sha256_file(Path(__file__)),
        "canonical_inputs_sha256": hashes_before, "canonical_inputs_unchanged_by_run": hashes_before == hashes_after,
        "canonical_compact_artifacts_match_frozen_hash_file": verify,
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": __import__("scipy").__version__,
        "constants": {"EPS_S": EPS_S, "THRESHOLDS_MS": list(THRESHOLDS_MS), "LARGE_BOOT_REPLICATES": large_reps, "MID_BOOT_REPLICATES": mid_reps,
                      "MC_SEEDS": mc_seeds, "BOOT_SEEDS_LARGE": list(BOOT_SEEDS_LARGE), "CANON_SEED": CANON_SEED, "CANON_REPLICATES": CANON_REPLICATES},
        "wall_seconds_cumulative": timings, "cpu_count": os.cpu_count(),
    }
    dump_json(prov, out / "provenance.json")
    return {"reproduction": rep["overall"], "timings": timings, "canonical_unchanged": prov["canonical_inputs_unchanged_by_run"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--large-reps", type=int, default=LARGE_BOOT_REPLICATES)
    ap.add_argument("--mid-reps", type=int, default=MID_BOOT_REPLICATES)
    ap.add_argument("--mc-seeds", type=int, default=MC_SEEDS)
    args = ap.parse_args()
    print(json.dumps(run(args.out, DESIGN, args.large_reps, args.mid_reps, args.mc_seeds), indent=2))


if __name__ == "__main__":
    main()
