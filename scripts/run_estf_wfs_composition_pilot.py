#!/usr/bin/env python3
"""Minimal ESTF↔WFS composition falsification pilot on Family A v2.

Reuses parent ANWG from Job 1182377 CSV. Runs new sims only for composed
children and selector policies. No LLM APIs. Ranking-only composition.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llmserveopt.composition.estf_wfs_features import (  # noqa: E402
    FORBIDDEN_FEATURE_KEYS,
    assert_no_hidden_leakage,
    scenario_observable_features,
)
from llmserveopt.composition.estf_wfs_metrics import (  # noqa: E402
    envelope_gain,
    paired_bootstrap_ci,
    regret_to_oracle,
)
from llmserveopt.composition.estf_wfs_models import (  # noqa: E402
    save_model_meta,
    select_model_on_val,
)
from llmserveopt.composition.estf_wfs_policies import (  # noqa: E402
    EstfWfsContextualAlphaPolicy,
    EstfWfsHardConditionalPolicy,
    EstfWfsTop1SelectorPolicy,
    alpha_collapse_stats,
    make_static_estf_wfs_blend,
)
from llmserveopt.composition.estf_wfs_splits import (  # noqa: E402
    assert_no_split_leakage,
    assign_family_a_v2_splits,
)
from llmserveopt.policy_separation.templates_fairness_starvation_v2 import (  # noqa: E402
    case_fairness_vs_size_v2,
)
from llmserveopt.simulator.simulator import Simulator, SimulatorConfig  # noqa: E402

PRIMARY = "arrival_normalized_weighted_goodput"
ESTF = "estimated_service_time_first"
WFS = "weighted_fair_share"
STATIC_ALPHAS = (0.25, 0.50, 0.75)


def _log(run_dir: Path, msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(run_dir / "run.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_parent_pivot(csv_path: Path) -> Dict[str, Dict[str, float]]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out: Dict[str, Dict[str, float]] = {}
    for r in rows:
        if r["policy_name"] not in {ESTF, WFS}:
            continue
        out.setdefault(r["scenario_id"], {})[r["policy_name"]] = float(r[PRIMARY])
    return out


def load_scenario_features_csv(path: Path) -> List[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_scenario(
    meta: Mapping[str, Any], *, allow_synthetic: bool, datasets_root: Optional[Path]
) -> Any:
    return case_fairness_vs_size_v2(
        target_utilization=float(meta["target_utilization"]),
        tenant_weight_skew=float(meta["tenant_weight_skew"]),
        favored_tenant_size=str(meta["favored_tenant_size"]),
        prediction_noise_sigma=float(meta["prediction_noise_sigma"]),
        seed=int(meta["seed"]),
        n_total_jobs=int(float(meta.get("n_total_jobs", 120))),
        max_active_sequences=int(float(meta.get("max_active_sequences", 1))),
        favored_slo_slack_s=float(meta.get("favored_slo_slack_s", 1.0)),
        other_slo_slack_s=float(meta.get("other_slo_slack_s", 8.0)),
        allow_synthetic_tokens=allow_synthetic,
        datasets_root=datasets_root,
    )



def run_method_on_scenario(
    scenario_id: str,
    method: str,
    meta: Mapping[str, Any],
    *,
    allow_synthetic: bool,
    datasets_root: Optional[Path],
    selector=None,
    alpha_model=None,
) -> dict:
    scenario = build_scenario(
        meta, allow_synthetic=allow_synthetic, datasets_root=datasets_root
    )
    feats = scenario_observable_features(scenario.requests)
    assert_no_hidden_leakage(feats)
    for k in feats:
        if k in FORBIDDEN_FEATURE_KEYS:
            raise RuntimeError(f"forbidden feature {k}")

    if method.startswith("static_alpha_"):
        alpha = float(method.replace("static_alpha_", ""))
        policy = make_static_estf_wfs_blend(alpha)
    elif method == "contextual_top1":
        assert selector is not None
        policy = EstfWfsTop1SelectorPolicy(selector, feats)
    elif method == "contextual_alpha":
        assert alpha_model is not None
        policy = EstfWfsContextualAlphaPolicy(alpha_model, feats)
    elif method == "hard_conditional":
        policy = EstfWfsHardConditionalPolicy(feats)
    else:
        raise ValueError(method)

    sim = Simulator(SimulatorConfig(gpu_configs=list(scenario.gpu_configs)))
    sim.load_trace(list(scenario.requests))
    metrics = sim.run(policy, workload_tag=f"{scenario_id}:{method}")
    completed = sim._completed  # noqa: SLF001
    fav = [c for c in completed if c.request.class_id == "tenant_favored"]
    oth = [c for c in completed if c.request.class_id == "tenant_other"]
    fav_v = sum(1 for c in fav if c.completion_time > c.request.slo_deadline)
    oth_v = sum(1 for c in oth if c.completion_time > c.request.slo_deadline)
    total_v = fav_v + oth_v
    unweighted = (len(completed) - total_v) / max(1, len(scenario.requests))
    ttfts = [c.ttft for c in completed if c.first_token_time >= 0]
    mean_ttft = sum(ttfts) / len(ttfts) if ttfts else 0.0
    g1 = (len(fav) - fav_v) / max(1, len(fav))
    g2 = (len(oth) - oth_v) / max(1, len(oth))
    jfi = ((g1 + g2) ** 2) / max(1e-12, 2 * (g1 * g1 + g2 * g2))

    row = {
        "scenario_id": scenario_id,
        "method": method,
        PRIMARY: float(metrics.arrival_normalized_weighted_goodput),
        "unweighted_slo_success_rate": float(unweighted),
        "completion_fraction": float(len(completed) / max(1, len(scenario.requests))),
        "favored_violations": fav_v,
        "favored_total": len(fav),
        "other_violations": oth_v,
        "other_total": len(oth),
        "jains_fairness_index": float(jfi),
        "mean_ttft": mean_ttft,
        "status": "success",
        "predicted_alpha_or_choice": "",
        "switch_count": 0,
        "frac_alpha_intermediate": "",
    }
    if isinstance(policy, EstfWfsContextualAlphaPolicy):
        stats = alpha_collapse_stats(policy.alpha_history)
        row["predicted_alpha_or_choice"] = (
            f"mean_alpha={stats['mean_alpha']:.3f}"
        )
        row["switch_count"] = int(policy.switch_count)
        row["frac_alpha_intermediate"] = stats["frac_intermediate"]
        row["frac_alpha_near_0"] = stats["frac_near_0"]
        row["frac_alpha_near_1"] = stats["frac_near_1"]
        row["mean_alpha"] = stats["mean_alpha"]
    elif isinstance(policy, EstfWfsTop1SelectorPolicy):
        row["predicted_alpha_or_choice"] = selector.predict_parent(feats)
        row["switch_count"] = int(policy.switch_count)
    elif isinstance(policy, EstfWfsHardConditionalPolicy):
        row["predicted_alpha_or_choice"] = policy.choice
    return row


def summarize_split(
    method_scores: Dict[str, Dict[str, float]],
    parent: Dict[str, Dict[str, float]],
    sids: List[str],
    method: str,
) -> dict:
    child = method_scores[method]
    estf = {sid: parent[sid][ESTF] for sid in sids}
    wfs = {sid: parent[sid][WFS] for sid in sids}
    gains = []
    for sid in sids:
        env = max(estf[sid], wfs[sid])
        gains.append(max(child[sid], env) - env)
    mean_g, lo, hi = paired_bootstrap_ci(gains)
    eg = envelope_gain(child, estf, wfs, sids)
    rg = regret_to_oracle(child, estf, wfs, sids)
    vals = [child[sid] for sid in sids]
    return {
        "method": method,
        "n": len(sids),
        "mean_anwg": sum(vals) / len(vals),
        "median_anwg": sorted(vals)[len(vals) // 2],
        **eg,
        "envelope_gain_mean": mean_g,
        "envelope_gain_ci95_lo": lo,
        "envelope_gain_ci95_hi": hi,
        **rg,
        "mean_estf": sum(estf[s] for s in sids) / len(sids),
        "mean_wfs": sum(wfs[s] for s in sids) / len(sids),
        "mean_oracle": sum(max(estf[s], wfs[s]) for s in sids) / len(sids),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--family-a-dir",
        type=Path,
        default=ROOT
        / "experiments/policy_separation_fairness_starvation_pilot_v2_20260816T220113Z_1182377",
    )
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--allow-synthetic-tokens", action="store_true")
    ap.add_argument("--require-burstgpt", action="store_true", default=True)
    ap.add_argument("--no-require-burstgpt", action="store_false", dest="require_burstgpt")
    ap.add_argument("--max-scenarios", type=int, default=0, help="0=all; smoke use small n")
    ap.add_argument(
        "--datasets-root",
        type=Path,
        default=None,
        help="Root containing burstgpt_v2/raw/",
    )
    args = ap.parse_args()

    run_dir = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    allow_synthetic = bool(args.allow_synthetic_tokens) and not args.require_burstgpt
    datasets_root = args.datasets_root

    feat_rows = load_scenario_features_csv(args.family_a_dir / "scenario_features.csv")
    parent = load_parent_pivot(args.family_a_dir / "per_policy_results.csv")
    meta_by_id = {r["scenario_id"]: r for r in feat_rows}

    # Enrich meta with fixed defaults used by generator
    for m in feat_rows:
        m.setdefault("n_total_jobs", "120")
        m.setdefault("max_active_sequences", "1")
        m.setdefault("favored_slo_slack_s", "1.0")
        m.setdefault("other_slo_slack_s", "8.0")

    splits = assign_family_a_v2_splits(feat_rows)
    assert_no_split_leakage(splits)
    _log(run_dir, f"Split sizes train={len(splits.train)} val={len(splits.val)} "
         f"test={len(splits.test)} ood={len(splits.ood)}")
    _log(run_dir, f"Split logic: {splits.logic}")

    # Observable features from regenerated scenarios (must match Family A ids)
    obs_features: Dict[str, dict] = {}
    for sid, meta in meta_by_id.items():
        scen = build_scenario(meta, allow_synthetic=allow_synthetic, datasets_root=datasets_root)
        feats = scenario_observable_features(scen.requests)
        assert_no_hidden_leakage(feats)
        obs_features[sid] = feats
        # provenance check
        if meta.get("token_length_source") and meta["token_length_source"] != "burstgpt_staged":
            if args.require_burstgpt and not allow_synthetic:
                raise RuntimeError(f"{sid} not burstgpt: {meta.get('token_length_source')}")

    def _pack(sids: List[str]):
        return (
            [obs_features[s] for s in sids],
            [parent[s][ESTF] for s in sids],
            [parent[s][WFS] for s in sids],
        )

    f_tr, e_tr, w_tr = _pack(splits.train)
    f_va, e_va, w_va = _pack(splits.val)
    selector, alpha_model, fit_meta = select_model_on_val(
        f_tr, e_tr, w_tr, f_va, e_va, w_va
    )
    save_model_meta(run_dir / "model_selection.json", fit_meta)
    _log(run_dir, f"Selected selector={fit_meta['selector_model_type']} "
         f"acc={fit_meta['selector_val_accuracy']:.3f}; "
         f"alpha={fit_meta['alpha_model_type']} "
         f"proxy_acc={fit_meta['alpha_val_proxy_accuracy']:.3f}")

    methods = [f"static_alpha_{a:.2f}" for a in STATIC_ALPHAS] + [
        "contextual_top1",
        "contextual_alpha",
        "hard_conditional",
    ]

    eval_sids = splits.val + splits.test + splits.ood
    if args.max_scenarios > 0:
        eval_sids = eval_sids[: args.max_scenarios]
        _log(run_dir, f"Smoke truncation: evaluating {len(eval_sids)} scenarios")

    result_rows: List[dict] = []
    # Also record parents for convenience
    for sid in eval_sids:
        for pol in (ESTF, WFS):
            result_rows.append(
                {
                    "scenario_id": sid,
                    "method": pol,
                    PRIMARY: parent[sid][pol],
                    "unweighted_slo_success_rate": "",
                    "completion_fraction": "",
                    "favored_violations": "",
                    "favored_total": "",
                    "other_violations": "",
                    "other_total": "",
                    "jains_fairness_index": "",
                    "mean_ttft": "",
                    "status": "from_family_a_v2_csv",
                    "predicted_alpha_or_choice": "",
                    "switch_count": 0,
                    "frac_alpha_intermediate": "",
                }
            )

    total = len(eval_sids) * len(methods)
    done = 0
    t0 = time.time()
    for sid in eval_sids:
        meta = meta_by_id[sid]
        for method in methods:
            row = run_method_on_scenario(
                sid,
                method,
                meta,
                allow_synthetic=allow_synthetic,
                datasets_root=datasets_root,
                selector=selector,
                alpha_model=alpha_model,
            )
            # attach split label
            if sid in splits.train:
                row["split"] = "train"
            elif sid in splits.val:
                row["split"] = "val"
            elif sid in splits.test:
                row["split"] = "test"
            else:
                row["split"] = "ood"
            result_rows.append(row)
            done += 1
            if done % 20 == 0 or done == total:
                _log(run_dir, f"Completed {done}/{total} child evaluations")

    # Write results CSV
    fieldnames = [
        "scenario_id",
        "split",
        "method",
        PRIMARY,
        "unweighted_slo_success_rate",
        "completion_fraction",
        "favored_violations",
        "favored_total",
        "other_violations",
        "other_total",
        "jains_fairness_index",
        "mean_ttft",
        "status",
        "predicted_alpha_or_choice",
        "switch_count",
        "frac_alpha_intermediate",
        "frac_alpha_near_0",
        "frac_alpha_near_1",
        "mean_alpha",
    ]
    out_csv = run_dir / "composition_results.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in result_rows:
            w.writerow(r)

    # Build method score pivots for analysis splits
    method_scores: Dict[str, Dict[str, float]] = {}
    for r in result_rows:
        if r.get("status") not in {"success", "from_family_a_v2_csv"}:
            continue
        method_scores.setdefault(r["method"], {})[r["scenario_id"]] = float(r[PRIMARY])

    summary = {
        "split_logic": splits.logic,
        "split_sizes": {
            "train": len(splits.train),
            "val": len(splits.val),
            "test": len(splits.test),
            "ood": len(splits.ood),
        },
        "model_selection": fit_meta,
        "methods": methods,
        "test": {},
        "ood": {},
        "val": {},
    }
    for split_name, sids in (
        ("val", [s for s in splits.val if s in eval_sids]),
        ("test", [s for s in splits.test if s in eval_sids]),
        ("ood", [s for s in splits.ood if s in eval_sids]),
    ):
        if not sids:
            continue
        # parent baselines on this split
        summary[split_name]["parents"] = {
            "mean_estf": sum(parent[s][ESTF] for s in sids) / len(sids),
            "mean_wfs": sum(parent[s][WFS] for s in sids) / len(sids),
            "mean_oracle": sum(max(parent[s][ESTF], parent[s][WFS]) for s in sids) / len(sids),
            "estf_beats_wfs": sum(
                1 for s in sids if parent[s][ESTF] > parent[s][WFS] + 0.01
            ),
            "wfs_beats_estf": sum(
                1 for s in sids if parent[s][WFS] > parent[s][ESTF] + 0.01
            ),
        }
        summary[split_name]["methods"] = {}
        for method in methods:
            summary[split_name]["methods"][method] = summarize_split(
                method_scores, parent, sids, method
            )

    # Decisive comparison on TEST
    if summary["test"].get("methods"):
        top1 = summary["test"]["methods"]["contextual_top1"]
        comp = summary["test"]["methods"]["contextual_alpha"]
        delta = comp["mean_anwg"] - top1["mean_anwg"]
        env_delta = comp["envelope_gain_mean"] - top1["envelope_gain_mean"]
        # Alpha collapse across test contextual_alpha rows
        alpha_rows = [
            r
            for r in result_rows
            if r["method"] == "contextual_alpha"
            and r.get("split") == "test"
            and r.get("status") == "success"
        ]
        mean_alphas = [float(r["mean_alpha"]) for r in alpha_rows if r.get("mean_alpha") != ""]
        collapse = alpha_collapse_stats(mean_alphas) if mean_alphas else {}
        # Verdict
        eps = 0.01
        ci_lo = comp["envelope_gain_ci95_lo"]
        beats_selector = (
            comp["mean_anwg"] > top1["mean_anwg"] + eps
            and comp["envelope_gain_mean"] > top1["envelope_gain_mean"] + 1e-6
            and ci_lo > 0
        )
        selector_wins = top1["mean_anwg"] + eps >= comp["mean_anwg"] and (
            top1["envelope_gain_mean"] >= comp["envelope_gain_mean"] - 1e-6
        )
        if beats_selector and collapse.get("frac_intermediate", 0) >= 0.2:
            verdict = "COMPOSITION_GO"
        elif selector_wins or collapse.get("frac_intermediate", 1) < 0.1:
            verdict = "SELECTION_SUFFICIENT_FOR_THIS_PAIR"
        else:
            verdict = "INCONCLUSIVE"
        summary["decisive_test"] = {
            "top1_mean_anwg": top1["mean_anwg"],
            "composition_mean_anwg": comp["mean_anwg"],
            "delta_composition_minus_top1": delta,
            "top1_envelope_gain": top1["envelope_gain_mean"],
            "composition_envelope_gain": comp["envelope_gain_mean"],
            "composition_envelope_ci95": [
                comp["envelope_gain_ci95_lo"],
                comp["envelope_gain_ci95_hi"],
            ],
            "env_delta_composition_minus_top1": env_delta,
            "alpha_collapse": collapse,
            "verdict": verdict,
        }
        _log(run_dir, f"VERDICT={verdict} test ΔANWG(comp-top1)={delta:.4f} "
             f"comp_env_gain={comp['envelope_gain_mean']:.4f} "
             f"CI=[{comp['envelope_gain_ci95_lo']:.4f},{comp['envelope_gain_ci95_hi']:.4f}]")

    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (run_dir / "splits.json").write_text(
        json.dumps(
            {
                "logic": splits.logic,
                "train": splits.train,
                "val": splits.val,
                "test": splits.test,
                "ood": splits.ood,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    # Sanity: alpha=1 vs ESTF rank identity on one scenario
    sid0 = splits.train[0]
    scen0 = build_scenario(meta_by_id[sid0], allow_synthetic=allow_synthetic, datasets_root=datasets_root)
    from llmserveopt.core.types import ObservableGPUState, ObservableRequest, ObservableState

    # skip deep identity here; covered by unit tests
    _log(run_dir, f"Wrote {out_csv} elapsed={time.time()-t0:.1f}s")
    print(json.dumps(summary.get("decisive_test", summary), indent=2))


if __name__ == "__main__":
    main()
