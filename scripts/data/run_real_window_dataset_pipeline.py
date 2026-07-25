#!/usr/bin/env python3
"""
Per-dataset overnight pipeline:
  preflight → construct → validate → characterize → synthetic fit → report

Streams large JSONL traces; does not load complete multi-million-row files.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from llmserveopt.core.types import Request
from llmserveopt.workloads.real_window_construction import (
    BUSY_SELECTION_RULE,
    DEFAULT_LOAD_FACTORS,
    WINDOW_ORIGIN_BUSY,
    WINDOW_ORIGIN_NATURAL,
    WINDOW_ORIGIN_SCALED,
    WINDOW_ORIGIN_SYNTHETIC,
    WindowCatalogEntry,
    apply_load_factor,
    atomic_write_json,
    build_catalog_streaming,
    extract_window_requests,
    fingerprint_requests,
    fit_and_sample_synthetic,
    load_window_jsonl,
    mark_busy_windows,
    mark_scaled_windows,
    validate_window_requests,
    write_marker,
    write_window_jsonl,
)

DATASET_FILES = {
    "burstgpt_v2": {
        "processed_dir": "/mmfs1/project/ikoutis/sv96/llmserveopt-data/datasets/burstgpt_v2/processed",
        "files": [
            ("burstgpt_without_fails_1.jsonl", "without_fails_1"),
            ("burstgpt_without_fails_2.jsonl", "without_fails_2"),
            ("burstgpt_without_fails_3.jsonl", "without_fails_3"),
        ],
        "max_natural_per_file": 40,
        "request_window_size": 1000,
        "azure2024_checks": False,
        "mooncake_restricted": False,
    },
    "azure_llm_2023": {
        "processed_dir": "/mmfs1/project/ikoutis/sv96/llmserveopt-data/datasets/azure_llm_2023/processed",
        "files": [
            ("azure_llm_2023_code.jsonl", "code"),
            ("azure_llm_2023_conv.jsonl", "conversation"),
        ],
        "max_natural_per_file": 60,
        "request_window_size": 400,
        "azure2024_checks": False,
        "mooncake_restricted": False,
    },
    "azure_llm_2024": {
        "processed_dir": "/mmfs1/project/ikoutis/sv96/llmserveopt-data/datasets/azure_llm_2024/processed",
        "files": [
            ("azure_llm_2024_code.jsonl", "code"),
            ("azure_llm_2024_conv.jsonl", "conversation"),
        ],
        "max_natural_per_file": 48,
        "request_window_size": 2000,
        "azure2024_checks": True,
        "mooncake_restricted": False,
        "expected_conv_inversions": 164,
    },
    "bailian_qwen": {
        "processed_dir": "/mmfs1/project/ikoutis/sv96/llmserveopt-data/datasets/bailian_qwen/processed",
        "files": [
            ("bailian_to_c_traceA.jsonl", "to_c_traceA"),
            ("bailian_to_b_traceB.jsonl", "to_b_traceB"),
            ("bailian_thinking.jsonl", "thinking"),
            ("bailian_coder.jsonl", "coder"),
        ],
        "max_natural_per_file": 40,
        "request_window_size": 500,
        "azure2024_checks": False,
        "mooncake_restricted": False,
    },
    "mooncake": {
        "processed_dir": "/mmfs1/project/ikoutis/sv96/llmserveopt-data/datasets/mooncake/processed_real",
        "files": [
            ("mooncake_conversation_trace.jsonl", "conversation_trace"),
            ("mooncake_toolagent_trace.jsonl", "toolagent_trace"),
        ],
        "max_natural_per_file": 40,
        "request_window_size": 400,
        "azure2024_checks": False,
        "mooncake_restricted": True,
        "forbidden_substrings": ["raw_synthetic_quarantine", "synthetic_trace"],
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stage_preflight(dataset: str, cfg: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    files = []
    for name, family in cfg["files"]:
        p = Path(cfg["processed_dir"]) / name
        if not p.exists():
            raise FileNotFoundError(f"missing processed file: {p}")
        if cfg.get("mooncake_restricted"):
            for bad in cfg.get("forbidden_substrings", []):
                if bad in str(p):
                    raise RuntimeError(f"mooncake real-only violation: {p}")
        files.append(
            {
                "path": str(p),
                "family": family,
                "bytes": p.stat().st_size,
            }
        )
    report = {
        "dataset": dataset,
        "utc": utc_now(),
        "files": files,
        "mooncake_restricted": bool(cfg.get("mooncake_restricted")),
    }
    atomic_write_json(out_dir / "preflight.json", report)
    write_marker(out_dir / "markers" / "preflight.ok", "ok", utc=utc_now())
    return report


def materialize_entry(
    entry: WindowCatalogEntry,
    windows_dir: Path,
    *,
    redistribution: str,
    evaluation_role: str,
) -> Tuple[Path, Dict[str, Any], List[str]]:
    raw = extract_window_requests(Path(entry.source_file), entry.start_index, entry.end_index)
    reqs = apply_load_factor(raw, entry.load_factor)
    issues = validate_window_requests(reqs)
    fp = fingerprint_requests(
        reqs,
        window_origin=entry.window_origin,
        load_factor=entry.load_factor,
        chronological_split=entry.chronological_split,
        source_family=entry.source_family,
        extra={
            "busy_selection_rule": entry.busy_selection_rule,
            "parent_window_id": entry.parent_window_id,
            "source_start_index": entry.start_index,
            "source_end_index": entry.end_index,
            "source_file": entry.source_file,
            "redistribution": redistribution,
            "evaluation_role": evaluation_role,
        },
    )
    meta = {
        "window_id": entry.window_id,
        "window_origin": entry.window_origin,
        "load_factor": entry.load_factor,
        "time_scale": entry.time_scale,
        "chronological_split": entry.chronological_split,
        "source_family": entry.source_family,
        "source_file": entry.source_file,
        "source_start_index": entry.start_index,
        "source_end_index": entry.end_index,
        "parent_window_id": entry.parent_window_id,
        "busy_selection_rule": entry.busy_selection_rule,
        "redistribution": redistribution,
        "evaluation_role": evaluation_role,
        "fingerprint": fp,
        "validation_issues": issues,
    }
    out = windows_dir / f"{entry.window_id}.jsonl"
    write_window_jsonl(reqs, out, meta=meta)
    return out, meta, issues


def stage_construct(dataset: str, cfg: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    windows_dir = out_dir / "windows"
    windows_dir.mkdir(parents=True, exist_ok=True)
    catalog: List[WindowCatalogEntry] = []
    file_reports = []
    redistribution = (
        "prohibited_until_license_clarified"
        if cfg.get("mooncake_restricted")
        else "allowed_with_dataset_license"
    )
    evaluation_role = "internal_ood_only" if cfg.get("mooncake_restricted") else "train_val_test_candidate"

    for name, family in cfg["files"]:
        path = Path(cfg["processed_dir"]) / name
        entries, rep = build_catalog_streaming(
            path,
            source_family=family,
            request_window_size=int(cfg["request_window_size"]),
            max_natural_windows=int(cfg["max_natural_per_file"]),
        )
        if cfg.get("azure2024_checks"):
            if rep["negative_arrivals"] != 0:
                raise RuntimeError(f"Azure2024 negative arrivals in {path}: {rep}")
            if not rep["nondecreasing_arrivals"]:
                raise RuntimeError(f"Azure2024 nondecreasing failure in {path}: {rep}")
        for e in entries:
            e.redistribution = redistribution
            e.evaluation_role = evaluation_role
        busy = mark_busy_windows(entries)
        scaled = mark_scaled_windows(entries, load_factors=DEFAULT_LOAD_FACTORS, max_per_factor=8)
        catalog.extend(entries)
        catalog.extend(busy)
        catalog.extend(scaled)
        file_reports.append(rep)

    # Cap busy/scaled materialization for overnight bound while keeping natural set.
    natural = [e for e in catalog if e.window_origin == WINDOW_ORIGIN_NATURAL]
    busy = [e for e in catalog if e.window_origin == WINDOW_ORIGIN_BUSY][:24]
    scaled = [e for e in catalog if e.window_origin == WINDOW_ORIGIN_SCALED][:24]
    to_materialize = natural + busy + scaled

    manifests = []
    critical_failures = []
    for entry in to_materialize:
        path, meta, issues = materialize_entry(
            entry,
            windows_dir,
            redistribution=redistribution,
            evaluation_role=evaluation_role,
        )
        manifests.append({"path": str(path), **{k: meta[k] for k in meta if k != "fingerprint"}, "fingerprint": meta["fingerprint"]})
        if issues:
            critical_failures.append({"window_id": entry.window_id, "issues": issues})

    if critical_failures:
        atomic_write_json(out_dir / "construct_failures.json", critical_failures)
        raise RuntimeError(f"window validation failed for {len(critical_failures)} windows")

    # Azure 2024 inversion accounting from staged conversion report (no login rescans).
    azure_note = None
    if cfg.get("azure2024_checks"):
        conv_report = Path(cfg["processed_dir"]) / "azure_llm_2024_conv.report.json"
        if conv_report.exists():
            rep = json.loads(conv_report.read_text())
            inv = int(rep.get("file_order_inversions", -1))
            expected = int(cfg.get("expected_conv_inversions", 164))
            if inv != expected:
                raise RuntimeError(
                    f"Azure2024 conv inversions expected {expected}, found {inv} in conversion report"
                )
            if not rep.get("sorted_by_wall_clock_timestamp", False):
                raise RuntimeError("Azure2024 conv missing sorted_by_wall_clock_timestamp disclosure")
            azure_note = {
                "file_order_inversions": inv,
                "sorted_by_wall_clock_timestamp": True,
                "rows_retained": rep.get("rows_retained"),
                "rows_read": rep.get("rows_read"),
                "row_loss": int(rep.get("rows_read", 0)) - int(rep.get("rows_retained", 0)),
            }

    report = {
        "dataset": dataset,
        "utc": utc_now(),
        "n_materialized": len(manifests),
        "n_natural": len(natural),
        "n_busy": len(busy),
        "n_scaled": len(scaled),
        "busy_selection_rule": BUSY_SELECTION_RULE,
        "load_factors": list(DEFAULT_LOAD_FACTORS),
        "file_reports": file_reports,
        "azure2024_inversion_accounting": azure_note,
        "windows": manifests,
    }
    atomic_write_json(out_dir / "window_catalog.json", report)
    write_marker(out_dir / "markers" / "construct.ok", "ok", utc=utc_now(), n=len(manifests))
    return report


def stage_validate(out_dir: Path) -> Dict[str, Any]:
    catalog = json.loads((out_dir / "window_catalog.json").read_text())
    issues = []
    ids = []
    source_spans = []
    for w in catalog["windows"]:
        wid = w["window_id"]
        ids.append(wid)
        path = Path(w["path"])
        if not path.exists():
            issues.append({"window_id": wid, "issue": "missing_file"})
            continue
        meta, reqs = load_window_jsonl(path)
        local = validate_window_requests(reqs)
        if local:
            issues.append({"window_id": wid, "issue": local})
        if meta.get("window_origin") == WINDOW_ORIGIN_NATURAL and float(meta.get("load_factor", 1)) != 1:
            issues.append({"window_id": wid, "issue": "natural_with_nonunit_load_factor"})
        if meta.get("window_origin") == WINDOW_ORIGIN_SCALED and int(meta.get("load_factor", 1)) <= 1:
            issues.append({"window_id": wid, "issue": "scaled_without_factor"})
        source_spans.append(
            (
                meta.get("source_file"),
                int(meta.get("source_start_index", -1)),
                int(meta.get("source_end_index", -1)),
                meta.get("chronological_split"),
                meta.get("window_origin"),
            )
        )
    if len(ids) != len(set(ids)):
        issues.append({"issue": "duplicate_window_ids"})

    # Overlap check among natural windows only (busy/scaled share parent spans by design).
    natural_spans = [s for s in source_spans if s[4] == WINDOW_ORIGIN_NATURAL]
    natural_spans.sort()
    for i in range(1, len(natural_spans)):
        a = natural_spans[i - 1]
        b = natural_spans[i]
        if a[0] == b[0] and a[2] > b[1]:
            issues.append(
                {
                    "issue": "natural_source_overlap",
                    "a": a,
                    "b": b,
                }
            )

    report = {
        "utc": utc_now(),
        "n_windows": len(catalog["windows"]),
        "ok": len(issues) == 0,
        "issues": issues,
    }
    atomic_write_json(out_dir / "validation_report.json", report)
    md = [
        f"# Validation report",
        f"",
        f"- utc: {report['utc']}",
        f"- n_windows: {report['n_windows']}",
        f"- ok: {report['ok']}",
        f"- n_issues: {len(issues)}",
    ]
    (out_dir / "validation_report.md").write_text("\n".join(md) + "\n")
    if issues:
        raise RuntimeError(f"validation failed: {len(issues)} issues")
    write_marker(out_dir / "markers" / "validate.ok", "ok", utc=utc_now())
    return report


def stage_characterize(out_dir: Path) -> Dict[str, Any]:
    catalog = json.loads((out_dir / "window_catalog.json").read_text())
    rows = []
    for w in catalog["windows"]:
        rows.append(
            {
                "window_id": w["window_id"],
                "window_origin": w["window_origin"],
                "chronological_split": w["chronological_split"],
                "source_family": w["source_family"],
                "load_factor": w["load_factor"],
                **w.get("fingerprint", {}),
            }
        )
    summary = {
        "utc": utc_now(),
        "n_windows": len(rows),
        "by_origin": {},
        "by_split": {},
        "rows": rows,
    }
    for r in rows:
        summary["by_origin"].setdefault(r["window_origin"], 0)
        summary["by_origin"][r["window_origin"]] += 1
        summary["by_split"].setdefault(r["chronological_split"], 0)
        summary["by_split"][r["chronological_split"]] += 1
    atomic_write_json(out_dir / "characterization" / "window_fingerprints.json", summary)
    write_marker(out_dir / "markers" / "characterize.ok", "ok", utc=utc_now())
    return summary


def stage_synthetic(dataset: str, out_dir: Path, n_synth: int = 120) -> Dict[str, Any]:
    catalog = json.loads((out_dir / "window_catalog.json").read_text())
    train_pairs: List[Tuple[WindowCatalogEntry, List[Request]]] = []
    for w in catalog["windows"]:
        if w["window_origin"] != WINDOW_ORIGIN_NATURAL or w["chronological_split"] != "train":
            continue
        meta, reqs = load_window_jsonl(Path(w["path"]))
        entry = WindowCatalogEntry(
            window_id=w["window_id"],
            source_file=w["source_file"],
            source_family=w["source_family"],
            start_index=int(w["source_start_index"]),
            end_index=int(w["source_end_index"]),
            start_arrival=0.0,
            end_arrival=0.0,
            n_requests=len(reqs),
            prompt_sum=0.0,
            output_sum=0.0,
            chronological_split="train",
        )
        train_pairs.append((entry, reqs))
        if len(train_pairs) >= 20:
            break
    synth_dir = out_dir / "synthetic"
    synth_dir.mkdir(parents=True, exist_ok=True)
    generated = fit_and_sample_synthetic(
        train_pairs,
        n_windows=n_synth,
        window_size=400,
        seed=17,
        source_fit_dataset=dataset,
    )
    rows = []
    for meta, reqs in generated:
        if out_dir.name.startswith("mooncake") or "mooncake" in dataset:
            meta["redistribution"] = "prohibited_until_license_clarified"
            meta["evaluation_role"] = "internal_ood_only"
        fp = fingerprint_requests(
            reqs,
            window_origin=WINDOW_ORIGIN_SYNTHETIC,
            load_factor=1,
            chronological_split="train_fit_only",
            source_family="synthetic",
        )
        meta["fingerprint"] = fp
        path = synth_dir / f"{meta['window_id']}.jsonl"
        write_window_jsonl(reqs, path, meta=meta)
        rows.append({"path": str(path), **meta})
    # Simple GoF: compare mean prompt/gap of synth vs train.
    train_prompts = [r.prompt_tokens for _, reqs in train_pairs for r in reqs]
    synth_prompts = [r.prompt_tokens for _, reqs in generated for r in reqs]
    import numpy as np

    gof = {
        "train_prompt_mean": float(np.mean(train_prompts)) if train_prompts else None,
        "synth_prompt_mean": float(np.mean(synth_prompts)) if synth_prompts else None,
        "train_n_windows_fit": len(train_pairs),
        "synth_n_windows": len(generated),
    }
    report = {"utc": utc_now(), "gof": gof, "windows": rows}
    atomic_write_json(out_dir / "synthetic_calibration_report.json", report)
    write_marker(out_dir / "markers" / "synthetic.ok", "ok", utc=utc_now(), n=len(rows))
    return report


def stage_final_report(dataset: str, out_dir: Path) -> None:
    parts = {}
    for name in [
        "preflight.json",
        "window_catalog.json",
        "validation_report.json",
        "synthetic_calibration_report.json",
    ]:
        p = out_dir / name
        if p.exists():
            parts[name] = json.loads(p.read_text())
    report = {
        "dataset": dataset,
        "utc": utc_now(),
        "status": "COMPLETE",
        "parts_present": sorted(parts.keys()),
        "n_windows": parts.get("window_catalog.json", {}).get("n_materialized"),
        "validation_ok": parts.get("validation_report.json", {}).get("ok"),
        "n_synthetic": parts.get("synthetic_calibration_report.json", {}).get("gof", {}).get("synth_n_windows"),
    }
    atomic_write_json(out_dir / "FINAL_DATASET_REPORT.json", report)
    md = [
        f"# Final dataset report: {dataset}",
        "",
        f"- utc: {report['utc']}",
        f"- status: {report['status']}",
        f"- n_windows: {report['n_windows']}",
        f"- validation_ok: {report['validation_ok']}",
        f"- n_synthetic: {report['n_synthetic']}",
    ]
    (out_dir / "FINAL_DATASET_REPORT.md").write_text("\n".join(md) + "\n")
    write_marker(out_dir / "markers" / "final.ok", "ok", utc=utc_now())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_FILES))
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--git-sha", required=True)
    args = parser.parse_args()
    cfg = DATASET_FILES[args.dataset]
    out_dir = Path(args.run_root) / "windows" / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "markers").mkdir(exist_ok=True)
    (out_dir / "characterization").mkdir(exist_ok=True)

    meta = {
        "dataset": args.dataset,
        "git_sha": args.git_sha,
        "start_utc": utc_now(),
        "python": sys.version,
    }
    atomic_write_json(out_dir / "job_meta_start.json", meta)
    try:
        stage_preflight(args.dataset, cfg, out_dir)
        stage_construct(args.dataset, cfg, out_dir)
        stage_validate(out_dir)
        stage_characterize(out_dir)
        stage_synthetic(args.dataset, out_dir)
        stage_final_report(args.dataset, out_dir)
        atomic_write_json(
            out_dir / "job_meta_finish.json",
            {**meta, "finish_utc": utc_now(), "exit_status": 0},
        )
        print(f"PIPELINE_OK dataset={args.dataset}")
    except Exception as exc:
        atomic_write_json(
            out_dir / "job_meta_finish.json",
            {
                **meta,
                "finish_utc": utc_now(),
                "exit_status": 1,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        print(f"PIPELINE_FAIL dataset={args.dataset}: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
