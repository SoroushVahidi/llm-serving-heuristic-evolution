"""Deterministic stratified window selection for the repaired load pilot.

This module is intentionally free of Slurm job IDs and absolute run roots so it
can be unit-tested with temporary catalogs. Selection is inventory-driven only:
it never inspects policy outcomes.
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

DATASETS = [
    "burstgpt_v2",
    "azure_llm_2023",
    "azure_llm_2024",
    "bailian_qwen",
    "mooncake",
]

QUOTA_NATURAL = 8
QUOTA_BUSY = 10
QUOTA_SCALED_PER_FACTOR = 8
SCALED_FACTORS = (2, 4, 8)
QUOTA_SYNTHETIC = 8
DEFAULT_SAMPLING_SEED = 20260725

EXACT_TIE_EPS = 1e-12
NEAR_TIE_MARGIN = 0.01
SATURATION_COMPLETION = 0.999

# Outcome-signature fields used for "behavioral disagreement" diagnostics.
# These are NOT true scheduler action traces.
OUTCOME_SIGNATURE_FIELDS = (
    "num_completed",
    "num_dropped",
    "anwg_rounded_6",
    "slo_violation_rate_rounded_6",
    "mean_active_batch_size_rounded_4",
)


def diversified_sample(
    pool: Sequence[Mapping[str, Any]],
    k: int,
    seed: int,
    stratum_tag: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Deterministic coverage sample across split / family / rate quantiles."""
    meta = {
        "stratum": stratum_tag,
        "available": len(pool),
        "quota": k,
        "deficit": max(0, k - len(pool)),
        "fallback_used": False,
    }
    if not pool:
        return [], meta
    if len(pool) <= k:
        out = [dict(w) for w in sorted(pool, key=lambda w: w["window_id"])]
        return out, meta

    stratum_salt = int(hashlib.md5(stratum_tag.encode()).hexdigest()[:8], 16)
    rng = random.Random((seed ^ stratum_salt) & 0xFFFFFFFF)

    def rate(w: Mapping[str, Any]) -> float:
        fp = w.get("fingerprint") or {}
        return float(fp.get("total_token_arrival_rate") or fp.get("request_arrival_rate") or 0.0)

    rates = sorted(rate(w) for w in pool)

    def qbin(w: Mapping[str, Any]) -> int:
        r = rate(w)
        qs = [rates[int((len(rates) - 1) * t)] for t in (0.25, 0.5, 0.75)]
        for i, thr in enumerate(qs):
            if r <= thr:
                return i
        return 3

    buckets: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for w in pool:
        key = (
            w.get("chronological_split", "?"),
            w.get("source_family", "?"),
            qbin(w),
            int((w.get("fingerprint") or {}).get("n_requests") or 0) // 250,
        )
        buckets[key].append(dict(w))
    for rows in buckets.values():
        rows.sort(key=lambda x: x["window_id"])
        rng.shuffle(rows)

    keys = sorted(buckets.keys())
    selected: List[Dict[str, Any]] = []
    while len(selected) < k:
        progressed = False
        for key in keys:
            if len(selected) >= k:
                break
            if buckets[key]:
                selected.append(buckets[key].pop())
                progressed = True
        if not progressed:
            break
    selected.sort(key=lambda w: w["window_id"])
    return selected, meta


def normalize_catalog_row(ds: str, w: Mapping[str, Any]) -> Dict[str, Any]:
    row = dict(w)
    row["dataset"] = ds
    if ds == "mooncake":
        row["evaluation_role"] = "internal_ood_only"
        row["redistribution"] = "prohibited_until_license_clarified"
    else:
        row.setdefault("evaluation_role", w.get("evaluation_role", "primary_or_supporting"))
        row.setdefault("redistribution", w.get("redistribution", "dataset_license_dependent"))
    return row


def select_from_inventory(
    inventory: Mapping[str, Mapping[str, Any]],
    *,
    seed: int = DEFAULT_SAMPLING_SEED,
    datasets: Optional[Sequence[str]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Select stratified windows from an in-memory inventory.

    Expected inventory[ds] keys:
      windows_by_origin: {origin: [window_dicts]}
      synthetic_windows: [window_dicts]
      validation_ok: bool
    """
    selected: List[Dict[str, Any]] = []
    deficits: List[Dict[str, Any]] = []
    inv_report: Dict[str, Any] = {}
    counts: Dict[str, Any] = {
        "by_dataset": {},
        "by_origin": Counter(),
        "by_scale": Counter(),
        "total": 0,
    }
    ds_list = list(datasets or DATASETS)

    for ds in ds_list:
        if ds not in inventory:
            raise FileNotFoundError(f"missing inventory for dataset: {ds}")
        entry = inventory[ds]
        if not entry.get("validation_ok", False):
            raise RuntimeError(f"validation not ok for {ds}")
        by_origin_raw = entry.get("windows_by_origin") or {}
        by_origin: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for origin, rows in by_origin_raw.items():
            for w in rows:
                by_origin[origin].append(normalize_catalog_row(ds, w))
        syn_rows = [
            normalize_catalog_row(ds, w) for w in (entry.get("synthetic_windows") or [])
        ]

        inv_ds = {
            "natural_replay": len(by_origin.get("natural_replay", [])),
            "natural_busy_period": len(by_origin.get("natural_busy_period", [])),
            "scaled_by_factor": {
                str(f): len(
                    [
                        w
                        for w in by_origin.get("trace_derived_time_scaled", [])
                        if int(w.get("load_factor") or 1) == f
                    ]
                )
                for f in SCALED_FACTORS
            },
            "trace_calibrated_synthetic": len(syn_rows),
            "validation_ok": True,
        }
        inv_report[ds] = inv_ds
        if ds == "mooncake" and (
            inv_ds["natural_replay"] == 0
            and inv_ds["natural_busy_period"] == 0
            and sum(inv_ds["scaled_by_factor"].values()) == 0
            and inv_ds["trace_calibrated_synthetic"] == 0
        ):
            raise RuntimeError("Mooncake has no valid candidate windows")

        ds_selected: List[Dict[str, Any]] = []
        rows, meta = diversified_sample(
            by_origin.get("natural_replay", []), QUOTA_NATURAL, seed + 1, f"{ds}:natural"
        )
        if meta["deficit"]:
            deficits.append({"dataset": ds, **meta})
        ds_selected.extend(rows)

        rows, meta = diversified_sample(
            by_origin.get("natural_busy_period", []), QUOTA_BUSY, seed + 2, f"{ds}:busy"
        )
        if meta["deficit"]:
            deficits.append({"dataset": ds, **meta})
        ds_selected.extend(rows)

        scaled = by_origin.get("trace_derived_time_scaled", [])
        for f in SCALED_FACTORS:
            pool = [w for w in scaled if int(w.get("load_factor") or 1) == f]
            rows, meta = diversified_sample(
                pool, QUOTA_SCALED_PER_FACTOR, seed + 10 + f, f"{ds}:scaled_{f}x"
            )
            if meta["deficit"]:
                deficits.append({"dataset": ds, **meta})
            ds_selected.extend(rows)

        rows, meta = diversified_sample(
            syn_rows, QUOTA_SYNTHETIC, seed + 99, f"{ds}:synthetic"
        )
        if meta["deficit"]:
            deficits.append({"dataset": ds, **meta})
        ds_selected.extend(rows)

        origins_present = {w["window_origin"] for w in ds_selected}
        for origin, pool in list(by_origin.items()) + [
            ("trace_calibrated_synthetic", syn_rows)
        ]:
            if pool and origin not in origins_present:
                pick = sorted(pool, key=lambda w: w["window_id"])[0]
                ds_selected.append(dict(pick))
                deficits.append(
                    {
                        "dataset": ds,
                        "stratum": f"{ds}:ensure_{origin}",
                        "note": "force-included one window from available origin",
                    }
                )

        counts["by_dataset"][ds] = len(ds_selected)
        for w in ds_selected:
            counts["by_origin"][w["window_origin"]] += 1
            lf = int(w.get("load_factor") or 1)
            if w["window_origin"] == "trace_derived_time_scaled":
                counts["by_scale"][f"{lf}x"] += 1
            else:
                counts["by_scale"]["1x_or_na"] += 1
        selected.extend(ds_selected)

    counts["by_origin"] = dict(counts["by_origin"])
    counts["by_scale"] = dict(counts["by_scale"])
    counts["total"] = len(selected)
    counts["mooncake_included"] = counts["by_dataset"].get("mooncake", 0) > 0
    counts["intended_quotas"] = {
        "per_dataset": {
            "natural_replay": QUOTA_NATURAL,
            "natural_busy_period": QUOTA_BUSY,
            "trace_derived_time_scaled_2x": QUOTA_SCALED_PER_FACTOR,
            "trace_derived_time_scaled_4x": QUOTA_SCALED_PER_FACTOR,
            "trace_derived_time_scaled_8x": QUOTA_SCALED_PER_FACTOR,
            "trace_calibrated_synthetic": QUOTA_SYNTHETIC,
            "dataset_total": (
                QUOTA_NATURAL
                + QUOTA_BUSY
                + QUOTA_SCALED_PER_FACTOR * len(SCALED_FACTORS)
                + QUOTA_SYNTHETIC
            ),
        },
        "datasets": list(ds_list),
        "grand_total_target": 50 * len(ds_list),
    }
    selection_meta = {
        "seed": seed,
        "inventory": inv_report,
        "deficits": deficits,
        "counts": counts,
        "sampling": "diversified_round_robin_by_split_family_rate_nreq",
        "no_silent_origin_replacement": True,
        "outcome_based_sampling": False,
        "diagnostic_note": (
            "behavioral disagreement uses outcome signatures "
            f"{list(OUTCOME_SIGNATURE_FIELDS)}; not true action traces"
        ),
    }
    if not counts["mooncake_included"]:
        raise RuntimeError("preflight failed: zero Mooncake windows selected")
    return selected, selection_meta


def load_inventory_from_run_root(run_root: Path) -> Dict[str, Dict[str, Any]]:
    """Build selection inventory from a validated real-window run root."""
    inventory: Dict[str, Dict[str, Any]] = {}
    for ds in DATASETS:
        ds_dir = Path(run_root) / "windows" / ds
        cat_path = ds_dir / "window_catalog.json"
        val_path = ds_dir / "validation_report.json"
        if not cat_path.exists():
            raise FileNotFoundError(f"missing catalog: {cat_path}")
        val = json.loads(val_path.read_text())
        cat = json.loads(cat_path.read_text())
        by_origin: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for w in cat["windows"]:
            by_origin[w["window_origin"]].append(dict(w))
        syn_rows: List[Dict[str, Any]] = []
        syn_report = ds_dir / "synthetic_calibration_report.json"
        if syn_report.exists():
            for w in json.loads(syn_report.read_text()).get("windows", []):
                syn_rows.append(
                    {
                        "window_id": w["window_id"],
                        "path": w["path"],
                        "window_origin": "trace_calibrated_synthetic",
                        "chronological_split": w.get("fingerprint", {}).get(
                            "chronological_split", "train_fit_only"
                        ),
                        "source_family": w.get("fingerprint", {}).get(
                            "source_family", "synthetic"
                        ),
                        "load_factor": int(w.get("load_factor") or 1),
                        "fingerprint": w.get("fingerprint") or {},
                        "evaluation_role": "supporting_synthetic",
                    }
                )
        inventory[ds] = {
            "windows_by_origin": dict(by_origin),
            "synthetic_windows": syn_rows,
            "validation_ok": bool(val.get("ok", False)),
        }
    return inventory


def select_windows_stratified(
    run_root: Path, seed: int = DEFAULT_SAMPLING_SEED
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    inventory = load_inventory_from_run_root(run_root)
    return select_from_inventory(inventory, seed=seed)


def outcome_signature(policy_metrics: Mapping[str, Any]) -> Tuple[Any, ...]:
    """Build an outcome signature (not an action trace)."""
    anwg = policy_metrics.get("anwg")
    slo = policy_metrics.get("slo_violation_rate")
    batch = policy_metrics.get("mean_active_batch_size")
    return (
        int(policy_metrics.get("num_completed") or 0),
        int(policy_metrics.get("num_dropped") or 0),
        None if anwg is None or (isinstance(anwg, float) and anwg != anwg) else round(float(anwg), 6),
        None if slo is None or (isinstance(slo, float) and slo != slo) else round(float(slo), 6),
        None
        if batch is None or (isinstance(batch, float) and batch != batch)
        else round(float(batch), 4),
    )


def exact_tie(best: float, second: float, eps: float = EXACT_TIE_EPS) -> bool:
    return abs(best - second) <= eps


def near_tie(margin: float, threshold: float = NEAR_TIE_MARGIN) -> bool:
    return margin <= threshold
