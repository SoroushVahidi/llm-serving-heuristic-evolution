"""Verified numbers for the reference-policy, load-range, overlay and secondary-outcome statements of the manuscript.

Everything is recomputed from COMPLETED artifacts (frozen protocol, fresh support/causal CSV/JSON, the corrected
state-level derivative, Phase A/B summaries and the whole-window replay checkpoint).  Nothing is simulated.

Evidence hierarchy: preregistration/metric protocol and machine-readable artifacts first, code second; generated
Markdown reports are not used.  The reference latency L_SBS(s) is taken from the CORRECTED state-level derivative
(SBS reference branch), never from the mislabeled columns of the frozen state-level file; it is cross-checked
against the action-level rows (L_SBS = mean_latency_CF + A_LAT) and against the pre-specified per-regime
``mean_relative_headroom`` stored in the frozen result JSON.
"""
from __future__ import annotations

import csv
import datetime as _dt
import json
import math
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments"
FROZEN = EXP / "fresh_production_latency_headroom_confirmatory_v1"
CORRECTED = EXP / "fresh_production_latency_headroom_confirmatory_v1_corrected"
PHASE_A = EXP / "industry_realism_action_opportunity_phase_a_v1"
PHASE_B = EXP / "industry_realism_action_opportunity_phase_b_v2"
REPLAY = EXP / "public_trace_replay_v1" / "layer3_checkpoint.jsonl"

POLICIES = ("full_prefill", "chunked_prefill_small", "estimated_service_time_first", "weighted_fair_share", "least_laxity_first")


def _rows(path: Path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _q(xs, q):
    s = sorted(xs)
    pos = q * (len(s) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


# protocol freeze and result commits (fallback epochs are the commit timestamps recorded in the repository history)
PROTOCOL_COMMIT, RESULT_COMMIT = "b4e6c60", "b196c3e"
_FALLBACK = {PROTOCOL_COMMIT: "2026-09-19T22:12:26-04:00", RESULT_COMMIT: "2026-09-20T00:26:14-04:00"}


def _commit_time(commit: str) -> _dt.datetime:
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "log", "-1", "--format=%cI", commit], capture_output=True, text=True, check=True).stdout.strip()
        if out:
            return _dt.datetime.fromisoformat(out)
    except Exception:
        pass
    return _dt.datetime.fromisoformat(_FALLBACK[commit])


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs)


def compute() -> dict:
    out: dict = {}

    # ---------------------------------------------------------------- fresh support map (BurstGPT, arrival scaling)
    sup = _rows(FROZEN / "FRESH_SUPPORT_WINDOW_CONDITIONS_V1.csv")
    valid = [r for r in sup if r["validity_class"].startswith("VALID")]
    burst = [r for r in sup if r["source_dataset"] == "burstgpt"]
    invalid = [r for r in sup if r["validity_class"].startswith("INVALID")]
    out["burst"] = {
        "conditions": len(burst),
        "valid": sum(r["validity_class"].startswith("VALID") for r in burst),
        "windows": len({r["fresh_window_id"] for r in burst}),
        "disagreement_states": sum(int(r["true_canonical_disagreement_states"]) for r in burst),
        "binding_states": sum(int(r["active_sequence_capacity_binding_states"]) + int(r["kv_capacity_binding_or_over_requested_states"]) for r in burst),
        "max_queue_length": max(int(r["max_queue_length"]) for r in burst),
    }
    out["invalid_fresh"] = {
        "total": len(invalid),
        "by_workload": dict(Counter(r["source_dataset"] for r in invalid)),
        "all_kv_8000": all(r["axis"] == "kv_capacity" and float(r["axis_value"]) == 8000.0 for r in invalid),
    }
    fa = [r for r in valid if r["axis"] == "arrival_pressure"]
    by = defaultdict(list)
    for r in fa:
        by[(r["source_dataset"], r["arrival_multiplier"])].append(float(r["mean_queue_length"]))
    out["arrival_fresh"] = {
        "max_multiplier": max(float(r["arrival_multiplier"]) for r in fa),
        "peak_kv_utilization": max(float(r["max_kv_utilization"]) for r in fa),
        "max_workload_mean_queue": max(_mean(v) for v in by.values()),
        "max_active_sequences": max(int(r["max_active_sequences"]) for r in fa),
        "default_active_cap": max(int(r["max_active_sequences_param"]) for r in fa),
        "default_kv_tokens": max(int(float(r["max_kv_tokens_param"])) for r in fa),
        "binding_states": sum(int(r["active_sequence_capacity_binding_states"]) + int(r["kv_capacity_binding_or_over_requested_states"]) for r in fa),
        "disagreement_states": sum(int(r["true_canonical_disagreement_states"]) for r in fa),
    }
    # original windows (Phase B): workload x multiplier summary rows
    oa = [r for r in _rows(PHASE_B / "PHASE_B_V2_WORKLOAD_AXIS_SUMMARY.csv") if "arrival" in r["axis"]]
    out["arrival_original"] = {
        "max_multiplier": max(float(r["arrival_multiplier"]) for r in oa),
        "peak_kv_utilization": max(float(r["max_kv_utilization"]) for r in oa),
        "max_workload_mean_queue": max(float(r["mean_queue_length"]) for r in oa),
        "max_active_sequences": max(int(float(r["max_active_sequences"])) for r in oa),
        "binding_states": sum(int(r["active_sequence_capacity_binding_states"]) + int(r["kv_capacity_binding_or_over_requested_states"]) for r in oa),
        "disagreement_states": sum(int(r["disagreement_states"]) for r in oa),
    }
    wl = {r["source_dataset"]: r for r in _rows(PHASE_A / "PHASE_A_WORKLOAD_SUMMARY_V1.csv")}
    out["native_mean_queue"] = {k: float(v["mean_queue_length"]) for k, v in wl.items()}

    # ---------------------------------------------------------------- overlays and capacity (machine-readable freeze)
    fv = json.loads((PHASE_A / "PHASE_A_REPLAY_SEMANTICS_V1.json").read_text())["faithful_view"]
    out["overlay"] = {k: fv[k] for k in ("slo_deadline", "priority", "class_id", "predicted_output_tokens", "capacity_assignment", "service_rate_assumptions")}
    assert fv["slo_deadline"].startswith("arrival + 1000.0") and fv["priority"] == "uniform 1.0" and fv["class_id"] == "default"
    assert fv["predicted_output_tokens"].startswith("set equal to actual_output_tokens")

    # ---------------------------------------------------------------- state-level causal results
    st = {r["state_id"]: r for r in _rows(FROZEN / "FRESH_ELIGIBLE_DISAGREEMENT_STATES_V1.csv")}
    cor = {r["state_id"]: r for r in _rows(CORRECTED / "FRESH_LATENCY_STATE_LEVEL_V1_CORRECTED.csv")}
    act = _rows(FROZEN / "FRESH_LATENCY_ACTION_LEVEL_V1.csv")
    assert len(st) == len(cor) == 720

    H = {s: float(r["oracle_headroom"]) for s, r in cor.items()}
    total = sum(H.values())
    npol = {s: len(r["p6_policies_with_non_sbs_action"].split(",")) for s, r in st.items()}
    five = [s for s, n in npol.items() if n == 5]
    kv = [s for s, r in st.items() if r["condition_id"] == "kv_16000"]
    kv_nobind = [s for s in kv if int(st[s]["kv_binding"]) == 0]
    kv_H = sum(H[s] for s in kv)
    out["reference"] = {
        "states": 720,
        "all_five_differ": len(five),
        "all_five_headroom_share": sum(H[s] for s in five) / total,
        "all_five_in_kv_regimes": sum(1 for s in five if st[s]["condition_id"] == "kv_16000"),
        "all_five_single_alternative": sum(1 for s in five if int(st[s]["n_unique_non_sbs_actions"]) == 1),
        "kv_states": len(kv),
        "kv_without_kv_binding": len(kv_nobind),
        "kv_nobind_headroom_share_of_kv": sum(H[s] for s in kv_nobind) / kv_H,
        "mean_distinct_alternatives": _mean(int(r["n_unique_non_sbs_actions"]) for r in st.values()),
        "single_alternative_states": sum(1 for r in st.values() if int(r["n_unique_non_sbs_actions"]) == 1),
        "differs": {p: sum(1 for r in st.values() if p in r["p6_policies_with_non_sbs_action"].split(",")) for p in POLICIES},
    }
    assert out["reference"]["differs"]["estimated_service_time_first"] == out["reference"]["differs"]["weighted_fair_share"]
    assert out["reference"]["differs"]["full_prefill"] == out["reference"]["differs"]["chunked_prefill_small"]

    # ---------------------------------------------------------------- preregistered secondary outcomes
    by_state = defaultdict(list)
    for r in act:
        by_state[r["state_id"]].append((float(r["a_lat"]), float(r["mean_latency"])))
    assert len(by_state) == 720
    all_harm = sum(1 for v in by_state.values() if max(a for a, _ in v) < 0)
    all_zero = sum(1 for v in by_state.values() if max(a for a, _ in v) == 0 and min(a for a, _ in v) == 0)
    benef = sum(1 for v in by_state.values() if max(a for a, _ in v) > 0)
    mixed = sum(1 for v in by_state.values() if max(a for a, _ in v) > 0 and min(a for a, _ in v) < 0)
    flat = [a for v in by_state.values() for a, _ in v]
    out["structure"] = {
        "beneficial": benef, "mixed_beneficial_and_harmful": mixed, "all_harmful": all_harm, "all_zero": all_zero,
        "zero_headroom_other": 720 - benef - all_harm,
        "branches": len(flat), "positive": sum(a > 0 for a in flat), "negative": sum(a < 0 for a in flat), "zero": sum(a == 0 for a in flat),
    }
    assert benef == 590 and benef + all_harm + (720 - benef - all_harm) == 720

    # reference latency from the corrected derivative, verified against the action-level rows
    ref = {s: float(r["mean_ref_latency"]) for s, r in cor.items()}
    for s, v in by_state.items():
        lsbs = [m + a for a, m in v]
        assert max(lsbs) - min(lsbs) < 1e-12 and abs(ref[s] - lsbs[0]) < 1e-12, s
    rel = {s: H[s] / ref[s] for s in H}
    rv = sorted(rel.values())
    kv_code = [s for s in rel if cor[s]["source_dataset"] == "azure_2023_code" and cor[s]["condition_id"] == "kv_16000"]
    out["relative"] = {
        "median": _q(rv, 0.5), "p90": _q(rv, 0.9), "mean": _mean(rv),
        "gt1": sum(x > 0.01 for x in rv), "gt5": sum(x > 0.05 for x in rv), "gt10": sum(x > 0.10 for x in rv),
        "kv_code_states": len(kv_code), "kv_code_p90": _q([rel[s] for s in kv_code], 0.9), "kv_code_gt10": sum(rel[s] > 0.10 for s in kv_code),
        "max_mean_reference_latency_s": max(ref.values()),
    }
    # pre-specified per-regime relative headroom stored in the frozen result JSON must equal our recomputation
    res = json.loads((FROZEN / "FRESH_LATENCY_CAUSAL_RESULT_V1.json").read_text())["workload_regime_results"]
    checked = 0
    for x in res:
        ss = [s for s in rel if cor[s]["condition_id"] == x["condition_id"] and cor[s]["source_dataset"] == x["source_dataset"]]
        assert len(ss) == x["states"], x["condition_id"]
        assert abs(_mean(rel[s] for s in ss) - x["mean_relative_headroom"]) < 1e-9, x["condition_id"]
        checked += 1
    assert checked == 5
    out["relative"]["kv_code_mean_prereg"] = next(x["mean_relative_headroom"] for x in res if x["condition_id"] == "kv_16000" and x["source_dataset"] == "azure_2023_code")

    # ---------------------------------------------------------------- pre-specification timeline and cluster-key correction
    t0, t1 = _commit_time(PROTOCOL_COMMIT), _commit_time(RESULT_COMMIT)
    minutes = (t1 - t0).total_seconds() / 60.0
    win = {(r["source_dataset"], r["window_index"]) for r in cor.values()}
    idx = {r["window_index"] for r in cor.values()}
    out["prespec"] = {"protocol_commit": PROTOCOL_COMMIT, "protocol_date": t0.date().isoformat(), "minutes_to_result_commit": minutes,
                      "clusters_source_window": len(win), "clusters_bare_index": len(idx)}
    assert out["prespec"]["clusters_source_window"] == 36 and out["prespec"]["clusters_bare_index"] == 26 and 133 <= minutes <= 135

    # ---------------------------------------------------------------- whole-window replay on the same 60 native windows
    rp = [json.loads(l) for l in REPLAY.read_text().splitlines()]
    fa_rows = [r for r in rp if r["canonical_scenario_id"].endswith("::faithful")]
    piv = defaultdict(dict)
    for r in fa_rows:
        piv[r["canonical_scenario_id"]][r["canonical_policy_id"]] = r["mean_latency"]
    diffs = sorted((v["chunked_prefill_small"] - v["full_prefill"]) / v["full_prefill"] for v in piv.values())
    out["whole_window"] = {
        "windows": len(piv), "policies_in_faithful_view": sorted({r["canonical_policy_id"] for r in fa_rows}),
        "chunked_slower_windows": sum(d > 0 for d in diffs), "median": _q(diffs, 0.5), "min": diffs[0], "max": diffs[-1],
    }
    return out


if __name__ == "__main__":
    print(json.dumps(compute(), indent=1, default=str))
