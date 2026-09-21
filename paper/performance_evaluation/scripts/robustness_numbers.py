"""Verified numbers for the post hoc robustness section of the Performance Evaluation manuscript.

Every quantity is recomputed from the FROZEN confirmatory artifacts (state-level CSV parsed with Python
``float()``, i.e. correctly rounded) and cross-checked against the post hoc robustness outputs.  The module
raises ``AssertionError`` on any disagreement, so tables, figure and tests share one verified source.

Only ``oracle_headroom``, ``beneficial_opportunity`` and the window/regime keys of the state-level CSV are used;
the two mislabeled descriptive columns (``mean_ref_latency``, ``p95_ref_latency``) are never read.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FROZEN = ROOT / "experiments" / "fresh_production_latency_headroom_confirmatory_v1"
ROBUST = ROOT / "experiments" / "fresh_production_latency_headroom_confirmatory_v1_robustness"
CORRECTED = ROOT / "experiments" / "fresh_production_latency_headroom_confirmatory_v1_corrected"

EPS_S = 1e-12
THRESHOLDS_MS = (0.1, 0.25, 0.5, 1.0, 2.0, 5.0)
CODE_KV = "azure_2023_code|kv_capacity|kv_16000"
W11 = "azure_2023_code:w11"


def _read(path: Path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _quantile(xs, q):  # numpy 'linear' method
    s = sorted(xs)
    pos = q * (len(s) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def load_states():
    rows = _read(FROZEN / "FRESH_LATENCY_STATE_LEVEL_V1.csv")
    out = []
    for r in rows:
        out.append({
            "window": f"{r['source_dataset']}:w{int(r['window_index'])}",
            "workload": r["source_dataset"],
            "regime": f"{r['source_dataset']}|{r['axis']}|{r['condition_id']}",
            "H": float(r["oracle_headroom"]),
            "beneficial": int(r["beneficial_opportunity"]),
        })
    return out


def _mean(xs):
    return math.fsum(xs) / len(xs)


def compute():
    S = load_states()
    n = len(S)
    H = [s["H"] for s in S]
    total = math.fsum(H)
    N = {}

    # ---------------------------------------------------------------- primary (preregistered)
    prim = json.loads((FROZEN / "FRESH_LATENCY_CAUSAL_RESULT_V1.json").read_text())["primary"]
    boot = json.loads((FROZEN / "FRESH_LATENCY_BOOTSTRAP_V1.json").read_text())
    assert n == prim["states"] == 720
    assert sum(s["beneficial"] for s in S) == prim["beneficial_states"] == 590 == sum(1 for h in H if h > 0)
    assert _mean(H) == prim["mean_oracle_headroom"]
    N["primary"] = {"mean_ms": 1e3 * _mean(H), "ci_ms": (1e3 * boot["mean_oracle_headroom_ci95_low"], 1e3 * boot["mean_oracle_headroom_ci95_high"]),
                    "beneficial": 590, "guarded_beneficial": sum(1 for h in H if h > EPS_S), "n": n,
                    "replicates": boot["bootstrap_replicates"], "seed": boot["bootstrap_seed"], "clusters": boot["clusters"]}
    assert N["primary"]["guarded_beneficial"] == 589

    # ---------------------------------------------------------------- practical thresholds (recomputed + cross-check)
    thr_csv = {(r["scope"], r["scope_id"], float(r["threshold_ms"])): r for r in _read(ROBUST / "practical_thresholds.csv")}
    windows = sorted({s["window"] for s in S})
    rows = []

    def row(label, pred):
        idx = [s for s in S if pred(s["H"])]
        return {"label": label, "n": len(idx), "share": len(idx) / n, "windows": len({s["window"] for s in idx}),
                "in_code_kv": sum(1 for s in idx if s["regime"] == CODE_KV)}

    rows.append(row("strict>0", lambda h: h > 0))
    rows.append(row("guard", lambda h: h > EPS_S))
    for t in THRESHOLDS_MS:
        r = row(f">{t}", lambda h, t=t: h > t * 1e-3 + EPS_S)
        g = thr_csv[("global", "all", t)]
        assert r["n"] == int(g["n_exceed"]) and r["windows"] == int(g["n_windows_with_exceedance"])
        assert r["in_code_kv"] == int(thr_csv[("regime", CODE_KV, t)]["n_exceed"])
        r["tau_ms"] = t
        rows.append(r)
    N["thresholds"] = rows
    resid = [h for h in H if 0 < h <= EPS_S]
    assert len(resid) == 1
    tol = json.loads((ROBUST / "numerical_tolerance_audit.json").read_text())
    rc = json.loads((CORRECTED / "ROBUSTNESS_AND_590_589_RECHECK_V1.json").read_text())["count_590_vs_589"]
    assert tol["smallest_genuine_nonzero_abs_A_s"] == rc["smallest_genuine_nonzero_abs_A_s"]
    N["residue"] = {"advantage_s": resid[0], "ulps": rc["advantage_in_ulps"], "ref_latency_s": rc["ref_mean_latency_s"],
                    "smallest_genuine_s": tol["smallest_genuine_nonzero_abs_A_s"], "identical_completed_requests": rc["sbs_and_cf_completed_request_id_hash_identical"],
                    "mean_effect_s": resid[0] / n}
    assert resid[0] == rc["residue_advantage_s"] and rc["advantage_in_ulps"] == 1.0 and rc["sbs_and_cf_completed_request_id_hash_identical"]

    # ---------------------------------------------------------------- distribution
    dist = _read(ROBUST / "distribution_summary.csv")[0]
    q = {k: _quantile(H, v) for k, v in {"p25": .25, "median": .5, "p75": .75, "p90": .9, "p95": .95}.items()}
    for k in q:
        assert abs(1e3 * q[k] - float(dist[f"{k}_ms" if k != "median" else "median_ms"])) < 1e-9
    zeros = sum(1 for h in H if h == 0)
    pos = [h for h in H if h > EPS_S]
    N["dist"] = {"mean_ms": 1e3 * _mean(H), "median_ms": 1e3 * q["median"], "p25_ms": 1e3 * q["p25"], "p75_ms": 1e3 * q["p75"], "p90_ms": 1e3 * q["p90"],
                 "p95_ms": 1e3 * q["p95"], "max_ms": 1e3 * max(H), "n_zero": zeros, "frac_zero": zeros / n,
                 "pos_mean_ms": 1e3 * _mean(pos), "pos_median_ms": 1e3 * _quantile(pos, .5), "n_pos": len(pos),
                 "frac_le_mean": sum(1 for h in H if h <= _mean(H)) / n}
    assert zeros == int(dist["n_exact_zero"]) == 130

    # ---------------------------------------------------------------- weighting
    w = {r["estimand"]: r for r in _read(ROBUST / "weighting_sensitivity.csv")}
    wc = {r["estimand"]: r for r in _read(ROBUST / "weighting_sensitivity_ci.csv")}

    def unit_mean(key):
        g = {}
        for s in S:
            g.setdefault(s[key], []).append(s)
        return g

    def equal(key):
        g = unit_mean(key)
        return len(g), _mean([_mean([s["H"] for s in v]) for v in g.values()]), _mean([_mean([s["beneficial"] for s in v]) for v in g.values()])

    weighting = []
    for name, key, units in (("state-weighted (canonical primary)", None, 720), ("equal-workload (supplementary)", "workload", 2),
                             ("equal-regime", "regime", 5), ("equal-window", "window", 36)):
        if key is None:
            mean, p = _mean(H), sum(s["beneficial"] for s in S) / n
        else:
            u, mean, p = equal(key)
            assert u == units
        assert abs(1e3 * mean - float(w[name]["mean_H_ms"])) < 1e-9 and abs(p - float(w[name]["P_beneficial_canonical"])) < 1e-12
        ci = (float(wc[name]["boot_ci95_low_ms"]), float(wc[name]["boot_ci95_high_ms"]))
        weighting.append({"name": name, "units": units, "mean_ms": 1e3 * mean, "P": p, "ci_ms": ci, "replicates": int(wc[name]["boot_replicates"])})
    weighting[0]["ci_ms"] = N["primary"]["ci_ms"]  # preregistered interval for the canonical estimand
    weighting[0]["replicates"] = N["primary"]["replicates"]
    N["weighting"] = weighting

    # ---------------------------------------------------------------- leave-one-out
    def loo(key):
        out = {}
        for u in sorted({s[key] for s in S}):
            rest = [s for s in S if s[key] != u]
            out[u] = {"n": len(rest), "mean_ms": 1e3 * _mean([s["H"] for s in rest]), "P": sum(s["beneficial"] for s in rest) / len(rest),
                      "share_omitted": math.fsum(s["H"] for s in S if s[key] == u) / total}
        return out

    lw, lr = loo("window"), loo("regime")
    lwc = {r["omitted"]: r for r in _read(ROBUST / "leave_one_window_out_remainder_ci.csv")}
    lrc = {r["omitted"]: r for r in _read(ROBUST / "leave_one_regime_out_remainder_ci.csv")}
    for u, v in lw.items():
        assert abs(v["mean_ms"] - float(lwc[u]["mean_H_ms"])) < 1e-9
        v["ci_ms"] = (float(lwc[u]["boot_ci95_low_ms"]), float(lwc[u]["boot_ci95_high_ms"]))
    for u, v in lr.items():
        assert abs(v["mean_ms"] - float(lrc[u]["mean_H_ms"])) < 1e-9
        v["ci_ms"] = (float(lrc[u]["boot_ci95_low_ms"]), float(lrc[u]["boot_ci95_high_ms"]))
    assert all(v["mean_ms"] > 0 and v["ci_ms"][0] > 0 for v in list(lw.values()) + list(lr.values()))
    others_w = [v["mean_ms"] for u, v in lw.items() if u != W11]
    others_r = [v["mean_ms"] for u, v in lr.items() if u != CODE_KV]
    N["loo"] = {"window": lw, "regime": lr, "n_windows": len(lw), "n_regimes": len(lr),
                "window_range_ms": (min(v["mean_ms"] for v in lw.values()), max(v["mean_ms"] for v in lw.values())),
                "regime_range_ms": (min(v["mean_ms"] for v in lr.values()), max(v["mean_ms"] for v in lr.values())),
                "window_P_range": (min(v["P"] for v in lw.values()), max(v["P"] for v in lw.values())),
                "regime_P_range": (min(v["P"] for v in lr.values()), max(v["P"] for v in lr.values())),
                "other_windows_range_ms": (min(others_w), max(others_w)), "other_regimes_range_ms": (min(others_r), max(others_r)),
                "all_ci_exclude_zero": True}

    # ---------------------------------------------------------------- concentration
    comp = json.loads((ROBUST / "composition_summary.json").read_text())
    cw = {r["id"]: r for r in _read(ROBUST / "composition_window.csv")}
    cr = {r["id"]: r for r in _read(ROBUST / "composition_regime.csv")}
    share_w = {u: v["share_omitted"] for u, v in lw.items()}
    assert abs(share_w[W11] - float(cw[W11]["share_of_total_headroom"])) < 1e-12
    sw = sorted(((math.fsum(s["H"] for s in S if s["window"] == u) / total, u) for u in windows), reverse=True)
    eff = 1.0 / sum(p * p for p, _ in sw)
    assert abs(eff - comp["effective_number_of_windows_by_headroom"]) < 1e-9
    w11 = [s for s in S if s["window"] == W11]
    hi5 = [s for s in S if s["H"] > 5e-3 + EPS_S]
    hi2 = [s for s in S if s["H"] > 2e-3 + EPS_S]
    N["conc"] = {"w11_states": len(w11), "w11_state_share": len(w11) / n, "w11_headroom_share": sw[0][0], "top3_share": sum(p for p, _ in sw[:3]),
                 "eff_windows": eff, "eff_windows_by_states": 1.0 / sum((sum(1 for s in S if s['window'] == u) / n) ** 2 for u in windows),
                 "code_kv_states": sum(1 for s in S if s["regime"] == CODE_KV), "code_kv_state_share": sum(1 for s in S if s["regime"] == CODE_KV) / n,
                 "code_kv_headroom_share": math.fsum(s["H"] for s in S if s["regime"] == CODE_KV) / total,
                 "code_kv_contrib_ms": 1e3 * math.fsum(s["H"] for s in S if s["regime"] == CODE_KV) / n,
                 "n_gt5": len(hi5), "n_gt5_in_w11": sum(1 for s in hi5 if s["window"] == W11),
                 "n_gt2": len(hi2), "n_gt2_in_w11": sum(1 for s in hi2 if s["window"] == W11),
                 "w11_code_kv_states": sum(1 for s in w11 if s["regime"] == CODE_KV),
                 "w11_code_kv_mean_ms": 1e3 * _mean([s["H"] for s in w11 if s["regime"] == CODE_KV]),
                 "top_windows": sw[:3], "windows_total": len(windows)}
    assert abs(N["conc"]["w11_headroom_share"] - comp["top_window_share"]) < 1e-12
    assert abs(N["conc"]["code_kv_headroom_share"] - comp["dominant_regime_share_of_total_headroom"]) < 1e-12
    assert comp["top_window"] == W11 and comp["dominant_regime"] == CODE_KV
    assert N["conc"]["w11_states"] == 263 and N["conc"]["n_gt5_in_w11"] == 128 and N["conc"]["n_gt5"] == 129

    # ---------------------------------------------------------------- interval diagnostics
    cl = json.loads((ROBUST / "cluster_robustness.json").read_text())
    big = cl["large_bootstrap"][str(20260920)]
    split = cl["dominant_window_bootstrap_split"]
    N["diag"] = {"large_reps": big["replicates"], "large_ci_ms": tuple(big["ci95_ms"]), "bca_ci_ms": tuple(cl["bca_large_bootstrap"]["ci95_ms"]),
                 "jack_ci_ms": tuple(1e3 * x for x in cl["jackknife_window"]["ci95_t"]), "jack_se_ms": 1e3 * cl["jackknife_window"]["jackknife_se"],
                 "frac_without_w11": split["frac_replicates_without_dominant_window"], "mean_without_w11_ms": split["without_dominant_window"]["mean_ms"],
                 "mean_with_w11_ms": split["with_dominant_window"]["mean_ms"], "mc_low_sd_ms": cl["mc_noise_canonical_size"]["ci_low_ms"]["sd"],
                 "mc_seeds": cl["mc_noise_canonical_size"]["n_seeds"], "frac_boot_le_0": big["frac_boot_le_0"]}
    return N


def figure_data():
    S = load_states()
    return S, _read(ROBUST / "composition_window.csv")


if __name__ == "__main__":
    import pprint
    pprint.pprint(compute(), width=140)
