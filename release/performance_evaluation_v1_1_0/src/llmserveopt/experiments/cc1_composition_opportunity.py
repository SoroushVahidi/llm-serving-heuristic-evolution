"""CC1 true simulator-executed composition-opportunity experiment.

This module intentionally keeps composition as an ordinary simulator policy:
fixed policies and weighted Borda rank mixtures are all executed through
``run_policy``. Stored metric rows are used only after execution to summarize
fixed-policy, hard-selector, and oracle upper-bound comparisons.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import yaml

from llmserveopt.core.metrics import RunMetrics, metrics_to_dict
from llmserveopt.core.types import GPUConfig, Request
from llmserveopt.evaluation.run_policy import run_policy
from llmserveopt.policies.capabilities import RANK_CAPABLE_EXPERTS
from llmserveopt.policies.composition import RankExpertSpec, StaticRankEnsemblePolicy
from llmserveopt.policies.instrumentation import DecisionTraceSink, InstrumentedPolicy
from llmserveopt.policies.registry import make_policy
from llmserveopt.simulator.service_model import ServiceModel
from llmserveopt.workloads.synthetic import SLOClass, WorkloadConfig, generate_workload
from llmserveopt.workloads.trace_io_extended import load_extended_jsonl


ROOT = Path(__file__).resolve().parents[3]
PRIMARY = "arrival_normalized_weighted_goodput"
PRIMARY_COL = f"metric_{PRIMARY}"
COMPLETION_COL = "metric_completion_fraction"
DEV_SPLITS = {"TRAIN", "VALIDATION", "ROBUST_DEV"}
EVAL_SPLITS = {"ID_TEST", "OOD_TEST", "TEMPORAL_OOD", "CROSS_SOURCE_OOD", "FINAL_OOD"}
VERDICTS = {"PROCEED", "REGIME_SPECIFIC_ONLY", "STOP_OR_REDESIGN", "INCONCLUSIVE"}


class CC1Error(ValueError):
    """Raised when the CC1 experiment config or runtime state is invalid."""


@dataclass(frozen=True)
class WorkloadWindow:
    window_id: str
    split: str
    regime: str
    source: str
    seed: int
    requests: tuple[Request, ...]
    skipped_reason: str | None = None


@dataclass(frozen=True)
class MixtureSpec:
    mixture_id: str
    weights: dict[str, float]
    top_k: int | str


@dataclass(frozen=True)
class PlannedRun:
    window_id: str
    treatment_id: str
    treatment_kind: str


@dataclass
class ExperimentResult:
    output_dir: Path
    manifest: dict[str, Any]
    verdict: str
    summaries: dict[str, Any] = field(default_factory=dict)


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise CC1Error("config must be a YAML mapping")
    return data


def normalize_weights(weights: Mapping[str, float]) -> dict[str, float]:
    clean: dict[str, float] = {}
    for name, value in weights.items():
        val = float(value)
        if not math.isfinite(val):
            raise CC1Error(f"non-finite weight for {name!r}: {value!r}")
        if val < 0.0:
            raise CC1Error(f"negative weight for {name!r}: {value!r}")
        if val > 0.0:
            clean[str(name)] = val
    total = sum(clean.values())
    if total <= 0.0:
        raise CC1Error("at least one positive weight is required")
    return {name: val / total for name, val in sorted(clean.items())}


def simplex_weight_grid(
    experts: Sequence[str],
    *,
    step: float,
    top_k: int | str = "all",
) -> list[MixtureSpec]:
    if not experts:
        raise CC1Error("at least one expert is required")
    if not math.isfinite(step) or step <= 0.0 or step > 1.0:
        raise CC1Error(f"weight_grid_step must be in (0, 1], got {step!r}")
    units_f = 1.0 / step
    units = int(round(units_f))
    if abs(units_f - units) > 1e-9:
        raise CC1Error("weight_grid_step must divide 1.0 exactly")
    max_active = len(experts) if top_k == "all" else int(top_k)
    if max_active <= 0:
        raise CC1Error("top_k must be positive or 'all'")

    out: list[MixtureSpec] = []
    seen: set[tuple[tuple[str, float], ...]] = set()
    for counts in itertools.product(range(units + 1), repeat=len(experts)):
        if sum(counts) != units:
            continue
        active = sum(1 for c in counts if c > 0)
        if active == 0 or active > max_active:
            continue
        raw = {name: count * step for name, count in zip(experts, counts) if count > 0}
        weights = normalize_weights(raw)
        key = tuple(weights.items())
        if key in seen:
            continue
        seen.add(key)
        suffix = "__".join(f"{name}-{weights[name]:.3f}".replace(".", "p") for name in weights)
        out.append(MixtureSpec(f"mix__{suffix}", weights, top_k))
    return sorted(out, key=lambda m: (len(m.weights), m.mixture_id))


def validate_config(config: Mapping[str, Any], *, require_full_flag: bool = False) -> None:
    if config.get("schema_version") != 1:
        raise CC1Error("schema_version must be 1")
    mode = config.get("mode")
    if mode not in {"smoke", "full", "cc1b"}:
        raise CC1Error("mode must be 'smoke', 'full', or 'cc1b'")
    policy_subset = list(config.get("policy_subset", []))
    if not policy_subset:
        raise CC1Error("policy_subset must not be empty")
    incompatible = [name for name in policy_subset if name not in RANK_CAPABLE_EXPERTS]
    if incompatible:
        raise CC1Error(f"incompatible rank experts: {incompatible}")
    for required in ("composition", "metrics", "safeguards", "outputs", "gpus", "workloads"):
        if required not in config:
            raise CC1Error(f"missing required config section: {required}")
    comp = config["composition"]
    if comp.get("implementation") != "StaticRankEnsemblePolicy":
        raise CC1Error("CC1 requires StaticRankEnsemblePolicy")
    if comp.get("method") != "borda":
        raise CC1Error("CC1 requires method: borda")
    simplex_weight_grid(
        policy_subset,
        step=float(comp.get("weight_grid_step", 0.0)),
        top_k=comp.get("top_k", "all"),
    )
    if config["metrics"].get("primary") != PRIMARY:
        raise CC1Error(f"primary metric must be {PRIMARY!r}")
    if mode in {"full", "cc1b"} and require_full_flag:
        raise CC1Error(f"{mode} mode requires explicit --full-run")


def planned_runs(config: Mapping[str, Any]) -> list[PlannedRun]:
    windows, _skipped = build_workload_windows(config)
    mixtures = simplex_weight_grid(
        list(config["policy_subset"]),
        step=float(config["composition"]["weight_grid_step"]),
        top_k=config["composition"].get("top_k", "all"),
    )
    runs: list[PlannedRun] = []
    for window in windows:
        for policy in config["policy_subset"]:
            runs.append(PlannedRun(window.window_id, f"fixed__{policy}", "fixed_policy"))
        for mixture in mixtures:
            runs.append(PlannedRun(window.window_id, mixture.mixture_id, "weighted_borda_mixture"))
    return runs


def build_workload_windows(config: Mapping[str, Any]) -> tuple[list[WorkloadWindow], list[dict[str, str]]]:
    windows: list[WorkloadWindow] = []
    skipped: list[dict[str, str]] = []
    for raw in config.get("workloads", []):
        kind = str(raw.get("kind", "synthetic"))
        split = str(raw["split"])
        tag = str(raw["tag"])
        regime = str(raw.get("regime", tag))
        seed = int(raw.get("seed", config.get("seed", 0)))
        max_requests = raw.get("max_requests")
        if kind == "synthetic":
            requests = _build_synthetic_requests(raw, seed=seed, max_requests=max_requests)
            windows.append(WorkloadWindow(tag, split, regime, "synthetic", seed, tuple(requests)))
        elif kind == "real_trace":
            path = ROOT / str(raw["path"])
            if not path.exists():
                skipped.append({"tag": tag, "path": str(path.relative_to(ROOT)), "reason": "missing local trace data"})
                continue
            requests, _metadata = load_extended_jsonl(path)
            requests = _apply_request_transform(
                _slice_and_rebase_requests(requests, max_requests=max_requests),
                raw.get("request_transform", {}),
            )
            windows.append(WorkloadWindow(tag, split, regime, "real_trace", seed, tuple(requests)))
        else:
            raise CC1Error(f"unknown workload kind {kind!r}")
    if not windows:
        raise CC1Error("no runnable workload windows were available")
    groups: dict[str, set[str]] = {}
    for window in windows:
        groups.setdefault(window.window_id, set()).add(window.split)
    leaked = {group: splits for group, splits in groups.items() if len(splits) > 1}
    if leaked:
        raise CC1Error(f"split-group leakage detected: {leaked}")
    return windows, skipped


def _build_synthetic_requests(raw: Mapping[str, Any], *, seed: int, max_requests: Any) -> list[Request]:
    allowed = set(WorkloadConfig.__dataclass_fields__)
    payload = {key: value for key, value in raw.items() if key in allowed}
    if "slo_classes" in payload:
        payload["slo_classes"] = parse_slo_classes(payload["slo_classes"])
    payload.setdefault("tag", str(raw["tag"]))
    cfg = WorkloadConfig(**payload)
    return _apply_request_transform(
        _slice_and_rebase_requests(generate_workload(cfg, seed=seed), max_requests=max_requests),
        raw.get("request_transform", {}),
    )


def parse_slo_classes(raw_classes: Any) -> list[SLOClass]:
    if raw_classes is None:
        raise CC1Error("slo_classes must not be null")
    out: list[SLOClass] = []
    for idx, raw in enumerate(raw_classes):
        if isinstance(raw, SLOClass):
            out.append(raw)
            continue
        if not isinstance(raw, Mapping):
            raise CC1Error(f"slo_classes[{idx}] must be a mapping")
        out.append(SLOClass(
            class_id=str(raw["class_id"]),
            slo_slack=float(raw["slo_slack"]),
            priority=float(raw["priority"]),
            weight=float(raw["weight"]),
        ))
    if not out:
        raise CC1Error("slo_classes must not be empty")
    if any(cls.slo_slack <= 0.0 for cls in out):
        raise CC1Error("all slo_classes must have positive slo_slack")
    if any(cls.priority <= 0.0 for cls in out):
        raise CC1Error("all slo_classes must have positive priority")
    if sum(cls.weight for cls in out) <= 0.0:
        raise CC1Error("slo_classes weights must sum to a positive value")
    return out


def _slice_and_rebase_requests(requests: Sequence[Request], *, max_requests: Any) -> list[Request]:
    reqs = list(requests[: int(max_requests)]) if max_requests is not None else list(requests)
    if not reqs:
        raise CC1Error("workload produced zero requests")
    first_arrival = reqs[0].arrival_time
    out: list[Request] = []
    for idx, req in enumerate(reqs):
        arrival = max(0.0, req.arrival_time - first_arrival)
        deadline_delta = max(req.slo_deadline - req.arrival_time, 0.001)
        out.append(Request(
            request_id=idx,
            arrival_time=arrival,
            prompt_tokens=req.prompt_tokens,
            predicted_output_tokens=req.predicted_output_tokens,
            actual_output_tokens=req.actual_output_tokens,
            slo_deadline=arrival + deadline_delta,
            priority=req.priority,
            class_id=req.class_id,
        ))
    return out


def _apply_request_transform(requests: Sequence[Request], raw_transform: Any) -> list[Request]:
    if raw_transform in (None, {}):
        return list(requests)
    if not isinstance(raw_transform, Mapping):
        raise CC1Error("request_transform must be a mapping")

    arrival_time_scale = float(raw_transform.get("arrival_time_scale", 1.0))
    slo_slack_scale = float(raw_transform.get("slo_slack_scale", 1.0))
    slo_slack_cap = raw_transform.get("slo_slack_cap")
    slo_slack_floor = float(raw_transform.get("slo_slack_floor", 0.001))
    if arrival_time_scale <= 0.0:
        raise CC1Error("request_transform.arrival_time_scale must be positive")
    if slo_slack_scale <= 0.0:
        raise CC1Error("request_transform.slo_slack_scale must be positive")
    if slo_slack_floor <= 0.0:
        raise CC1Error("request_transform.slo_slack_floor must be positive")
    cap = float(slo_slack_cap) if slo_slack_cap is not None else None
    if cap is not None and cap <= 0.0:
        raise CC1Error("request_transform.slo_slack_cap must be positive when provided")

    out: list[Request] = []
    first_arrival = requests[0].arrival_time
    previous_raw = first_arrival
    previous_scaled = 0.0
    for idx, req in enumerate(requests):
        if idx == 0:
            arrival = 0.0
        else:
            arrival = previous_scaled + max(0.0, req.arrival_time - previous_raw) * arrival_time_scale
        slack = max(req.slo_deadline - req.arrival_time, slo_slack_floor) * slo_slack_scale
        if cap is not None:
            slack = min(slack, cap)
        slack = max(slack, slo_slack_floor)
        out.append(Request(
            request_id=idx,
            arrival_time=arrival,
            prompt_tokens=req.prompt_tokens,
            predicted_output_tokens=req.predicted_output_tokens,
            actual_output_tokens=req.actual_output_tokens,
            slo_deadline=arrival + slack,
            priority=req.priority,
            class_id=req.class_id,
        ))
        previous_raw = req.arrival_time
        previous_scaled = arrival
    return out


def build_gpu_configs(config: Mapping[str, Any]) -> list[GPUConfig]:
    return [GPUConfig(**raw) for raw in config["gpus"]]


def build_service_model(config: Mapping[str, Any]) -> ServiceModel:
    raw = dict(config.get("service_model", {}))
    return ServiceModel(**raw)


def run_experiment(
    config: Mapping[str, Any],
    *,
    config_path: str | Path,
    dry_run: bool = False,
    full_run: bool = False,
    max_runs: int | None = None,
    allow_dirty: bool = False,
    timestamp: str | None = None,
    runner: Callable[..., RunMetrics] = run_policy,
) -> ExperimentResult:
    needs_full_flag = config.get("mode") in {"full", "cc1b"} and not full_run and not dry_run
    validate_config(config, require_full_flag=needs_full_flag)
    git = git_state()
    windows, skipped_traces = build_workload_windows(config)
    mixtures = simplex_weight_grid(
        list(config["policy_subset"]),
        step=float(config["composition"]["weight_grid_step"]),
        top_k=config["composition"].get("top_k", "all"),
    )
    planned = planned_runs(config)
    cap = int(max_runs if max_runs is not None else config["safeguards"].get("max_runs", 0))
    if cap <= 0:
        raise CC1Error("max_runs must be positive")
    if len(planned) > cap:
        raise CC1Error(f"planned run count {len(planned)} exceeds max_runs {cap}")
    if dry_run:
        return ExperimentResult(
            output_dir=Path(""),
            manifest={
                "dry_run": True,
                "planned_run_count": len(planned),
                "window_count": len(windows),
                "mixture_count": len(mixtures),
                "skipped_real_traces": skipped_traces,
                "fixed_policy_spread_requirement": config.get("discriminativeness"),
            },
            verdict="INCONCLUSIVE",
        )
    dirty = bool(git["dirty"])
    if dirty and not allow_dirty:
        raise CC1Error("non-dry CC1 runs require a clean git worktree unless --allow-dirty is explicitly set")

    output_dir = resolve_output_dir(config, timestamp=timestamp)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "decision_traces").mkdir(exist_ok=True)
    resolved_config = json.loads(json.dumps(config))
    write_yaml(output_dir / "config.yaml", resolved_config)

    t0 = time.perf_counter()
    gpu_configs = build_gpu_configs(config)
    service_model = build_service_model(config)
    execution_rows: list[dict[str, Any]] = []
    trace_enabled = bool(config.get("outputs", {}).get("decision_traces", False))

    for window in windows:
        for policy_name in config["policy_subset"]:
            policy = make_policy(policy_name, seed=window.seed)
            row = execute_policy_row(
                runner,
                policy,
                window,
                gpu_configs,
                service_model,
                treatment_id=f"fixed__{policy_name}",
                treatment_kind="fixed_policy",
                expert_weights={policy_name: 1.0},
                drain_steps=int(config.get("simulator", {}).get("drain_steps", 5000)),
            )
            execution_rows.append(row)

    fixed_spread_rows = fixed_policy_spread_rows(execution_rows, config)
    assert_cc1b_discriminative(fixed_spread_rows, config)

    for window in windows:
        for mixture in mixtures:
            experts = [RankExpertSpec(name, weight) for name, weight in mixture.weights.items()]
            base_policy = StaticRankEnsemblePolicy(
                experts,
                method="borda",
                top_k=None if mixture.top_k == "all" else int(mixture.top_k),
            )
            sink = DecisionTraceSink(enabled=trace_enabled, scenario_id=window.window_id)
            policy = InstrumentedPolicy(base_policy, sink) if trace_enabled else base_policy
            row = execute_policy_row(
                runner,
                policy,
                window,
                gpu_configs,
                service_model,
                treatment_id=mixture.mixture_id,
                treatment_kind="weighted_borda_mixture",
                expert_weights=mixture.weights,
                drain_steps=int(config.get("simulator", {}).get("drain_steps", 5000)),
            )
            execution_rows.append(row)
            if trace_enabled:
                sink.write_jsonl(output_dir / "decision_traces" / f"{window.window_id}__{mixture.mixture_id}.jsonl")

    per_window = summarize_per_window(execution_rows, config)
    method_rows = summarize_methods(execution_rows, per_window, config)
    weights_rows = composition_weight_rows(mixtures, method_rows)
    near_tie_rows = near_tie_sensitivity_rows(per_window, config)
    subset_rows = subset_analysis_rows(per_window, config)
    verdict_payload = determine_verdict(per_window, method_rows, subset_rows, config)

    write_csv(output_dir / "policy_execution_rows.csv", execution_rows)
    write_csv(output_dir / "fixed_policy_spread.csv", fixed_spread_rows)
    write_csv(output_dir / "per_window_summary.csv", per_window)
    write_csv(output_dir / "method_comparison.csv", method_rows)
    write_csv(output_dir / "composition_weights.csv", weights_rows)
    write_csv(output_dir / "near_tie_sensitivity.csv", near_tie_rows)
    write_csv(output_dir / "subset_analysis.csv", subset_rows)
    (output_dir / "verdict.json").write_text(json.dumps(verdict_payload, indent=2, sort_keys=True) + "\n")

    manifest = build_manifest(
        config,
        config_path=config_path,
        output_dir=output_dir,
        git=git,
        planned_count=len(planned),
        windows=windows,
        mixtures=mixtures,
        skipped_traces=skipped_traces,
        runtime_s=time.perf_counter() - t0,
        verdict_payload=verdict_payload,
    )
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (output_dir / "cc1_report.md").write_text(render_report(manifest, method_rows, subset_rows, verdict_payload))
    return ExperimentResult(output_dir=output_dir, manifest=manifest, verdict=verdict_payload["verdict"], summaries=verdict_payload)


def execute_policy_row(
    runner: Callable[..., RunMetrics],
    policy: Any,
    window: WorkloadWindow,
    gpu_configs: list[GPUConfig],
    service_model: ServiceModel,
    *,
    treatment_id: str,
    treatment_kind: str,
    expert_weights: Mapping[str, float],
    drain_steps: int,
) -> dict[str, Any]:
    metrics = runner(
        policy=policy,
        requests=list(window.requests),
        gpu_configs=gpu_configs,
        service_model=service_model,
        workload_tag=window.regime,
        seed=window.seed,
        drain_steps=drain_steps,
    )
    row = {
        "window_id": window.window_id,
        "split": window.split,
        "regime": window.regime,
        "source": window.source,
        "treatment_id": treatment_id,
        "treatment_kind": treatment_kind,
        "expert_weights_json": json.dumps(dict(expert_weights), sort_keys=True),
        "true_simulator_executed": True,
        "reward_vector_interpolated": False,
    }
    md = metrics_to_dict(metrics)
    for key, value in md.items():
        if key == "policy":
            row["policy_name"] = value
        elif key == "workload":
            row["workload_tag"] = value
        else:
            row[f"metric_{key}"] = value
    return row


def fixed_policy_spread_rows(
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    eval_splits = set(config.get("evaluation_splits", sorted(EVAL_SPLITS)))
    by_window: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        if row["treatment_kind"] == "fixed_policy":
            by_window.setdefault(str(row["window_id"]), []).append(row)

    out: list[dict[str, Any]] = []
    for window_id, window_rows in sorted(by_window.items()):
        values = sorted((metric_value(r, PRIMARY_COL) for r in window_rows), reverse=True)
        if not values:
            continue
        base = window_rows[0]
        out.append({
            "window_id": window_id,
            "split": base["split"],
            "regime": base["regime"],
            "source": base["source"],
            "is_evaluation": base["split"] in eval_splits,
            "fixed_policy_spread": values[0] - values[-1],
            "fixed_top2_margin": values[0] - values[1] if len(values) >= 2 else 0.0,
            "best_fixed_treatment_id": max(window_rows, key=lambda r: metric_value(r, PRIMARY_COL))["treatment_id"],
            "worst_fixed_treatment_id": min(window_rows, key=lambda r: metric_value(r, PRIMARY_COL))["treatment_id"],
            "best_fixed_anwg": values[0],
            "worst_fixed_anwg": values[-1],
        })
    return out


def assert_cc1b_discriminative(
    fixed_spread: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> None:
    if config.get("mode") != "cc1b":
        return
    raw = config.get("discriminativeness", {})
    if not raw:
        raise CC1Error("cc1b mode requires a discriminativeness section")
    min_spread = float(raw.get("min_fixed_policy_spread", 0.0))
    min_eval_windows = int(raw.get("min_evaluation_windows_with_spread", 1))
    min_top2_margin = float(raw.get("min_fixed_top2_margin", 0.0))
    eval_rows = [r for r in fixed_spread if bool(r["is_evaluation"])]
    informative = [
        r for r in eval_rows
        if float(r["fixed_policy_spread"]) >= min_spread
        and float(r["fixed_top2_margin"]) >= min_top2_margin
    ]
    if len(informative) < min_eval_windows:
        raise CC1Error(
            "cc1b fixed-policy spread gate failed before mixture evaluation: "
            f"required {min_eval_windows} evaluation windows with spread >= {min_spread} "
            f"and top2 margin >= {min_top2_margin}, found {len(informative)}"
        )


def summarize_per_window(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> list[dict[str, Any]]:
    by_window: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_window.setdefault(str(row["window_id"]), []).append(row)
    out: list[dict[str, Any]] = []
    near_tie_threshold = float(config.get("near_tie_primary_threshold", 0.005))
    for window_id, window_rows in sorted(by_window.items()):
        fixed = [r for r in window_rows if r["treatment_kind"] == "fixed_policy"]
        mixtures = [r for r in window_rows if r["treatment_kind"] == "weighted_borda_mixture"]
        oracle_fixed = max(fixed, key=lambda r: metric_value(r, PRIMARY_COL))
        oracle_mixture = max(mixtures, key=lambda r: metric_value(r, PRIMARY_COL))
        fixed_values = sorted((metric_value(r, PRIMARY_COL) for r in fixed), reverse=True)
        margin = fixed_values[0] - fixed_values[1] if len(fixed_values) >= 2 else 0.0
        spread = fixed_values[0] - fixed_values[-1] if fixed_values else 0.0
        base = fixed[0]
        out.append({
            "window_id": window_id,
            "split": base["split"],
            "regime": base["regime"],
            "source": base["source"],
            "oracle_fixed_treatment_id": oracle_fixed["treatment_id"],
            "oracle_fixed_anwg": metric_value(oracle_fixed, PRIMARY_COL),
            "oracle_fixed_completion_fraction": metric_value(oracle_fixed, COMPLETION_COL),
            "oracle_mixture_treatment_id": oracle_mixture["treatment_id"],
            "oracle_mixture_anwg": metric_value(oracle_mixture, PRIMARY_COL),
            "oracle_mixture_completion_fraction": metric_value(oracle_mixture, COMPLETION_COL),
            "composition_opportunity_gap": metric_value(oracle_mixture, PRIMARY_COL) - metric_value(oracle_fixed, PRIMARY_COL),
            "fixed_policy_spread": spread,
            "fixed_top2_margin": margin,
            "near_tie": margin < near_tie_threshold,
            "meaningful_window": margin >= near_tie_threshold,
            "num_arrivals": int(base.get("metric_num_total") or 0),
        })
    return out


def summarize_methods(
    execution_rows: Sequence[Mapping[str, Any]],
    per_window: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    eval_splits = set(config.get("evaluation_splits", sorted(EVAL_SPLITS)))
    dev_splits = set(config.get("development_splits", sorted(DEV_SPLITS)))
    eval_windows = {r["window_id"] for r in per_window if r["split"] in eval_splits}
    dev_rows = [r for r in execution_rows if r["split"] in dev_splits]
    eval_rows = [r for r in execution_rows if r["window_id"] in eval_windows]
    method_rows: list[dict[str, Any]] = []

    for treatment_id in sorted({str(r["treatment_id"]) for r in execution_rows}):
        treatment_eval = [r for r in eval_rows if r["treatment_id"] == treatment_id]
        if treatment_eval:
            method_rows.append(method_summary(treatment_id, treatment_eval, selection_scope="fixed_or_mixture"))

    best_fixed = choose_best_treatment(dev_rows, kind="fixed_policy")
    best_global_mixture = choose_best_treatment(dev_rows, kind="weighted_borda_mixture")
    if best_fixed is not None:
        method_rows.append(method_summary(
            "best_fixed_policy",
            [r for r in eval_rows if r["treatment_id"] == best_fixed],
            selected_treatment_id=best_fixed,
            selection_scope="development",
        ))
    if best_global_mixture is not None:
        method_rows.append(method_summary(
            "best_global_mixture",
            [r for r in eval_rows if r["treatment_id"] == best_global_mixture],
            selected_treatment_id=best_global_mixture,
            selection_scope="development",
        ))
    hard_selector = hard_selector_rows(dev_rows, eval_rows, fallback=best_fixed)
    if hard_selector:
        method_rows.append(method_summary(
            "learned_hard_selector_regime_lookup",
            hard_selector,
            selected_treatment_id="per_regime_development_lookup",
            selection_scope="development",
        ))

    eval_per_window = [r for r in per_window if r["window_id"] in eval_windows]
    method_rows.append(oracle_summary("oracle_best_fixed_per_window", eval_per_window, "oracle_fixed"))
    method_rows.append(oracle_summary("oracle_best_mixture_per_window", eval_per_window, "oracle_mixture"))
    return method_rows


def method_summary(
    method_id: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    selected_treatment_id: str | None = None,
    selection_scope: str,
) -> dict[str, Any]:
    return {
        "method_id": method_id,
        "selected_treatment_id": selected_treatment_id or method_id,
        "selection_scope": selection_scope,
        "n_windows": len(rows),
        "mean_anwg": mean(metric_value(r, PRIMARY_COL) for r in rows),
        "mean_completion_fraction": mean(metric_value(r, COMPLETION_COL) for r in rows),
        "mean_completed_request_quality": mean(metric_value(r, "metric_weighted_goodput") for r in rows),
        "mean_num_arrivals": mean(float(r.get("metric_num_total") or 0) for r in rows),
    }


def oracle_summary(method_id: str, per_window: Sequence[Mapping[str, Any]], prefix: str) -> dict[str, Any]:
    return {
        "method_id": method_id,
        "selected_treatment_id": "per_window_hindsight_upper_bound",
        "selection_scope": "oracle_hindsight",
        "n_windows": len(per_window),
        "mean_anwg": mean(float(r[f"{prefix}_anwg"]) for r in per_window),
        "mean_completion_fraction": mean(float(r[f"{prefix}_completion_fraction"]) for r in per_window),
        "mean_completed_request_quality": None,
        "mean_num_arrivals": mean(float(r.get("num_arrivals") or 0) for r in per_window),
    }


def choose_best_treatment(rows: Sequence[Mapping[str, Any]], *, kind: str) -> str | None:
    candidates = sorted({str(r["treatment_id"]) for r in rows if r["treatment_kind"] == kind})
    if not candidates:
        return None
    means = {
        treatment: mean(metric_value(r, PRIMARY_COL) for r in rows if r["treatment_id"] == treatment)
        for treatment in candidates
    }
    return max(means.items(), key=lambda item: (item[1], item[0]))[0]


def hard_selector_rows(
    dev_rows: Sequence[Mapping[str, Any]],
    eval_rows: Sequence[Mapping[str, Any]],
    *,
    fallback: str | None,
) -> list[Mapping[str, Any]]:
    fixed_dev = [r for r in dev_rows if r["treatment_kind"] == "fixed_policy"]
    if not fixed_dev or fallback is None:
        return []
    by_regime: dict[str, list[Mapping[str, Any]]] = {}
    for row in fixed_dev:
        by_regime.setdefault(str(row["regime"]), []).append(row)
    learned = {
        regime: choose_best_treatment(rows, kind="fixed_policy") or fallback
        for regime, rows in by_regime.items()
    }
    out: list[Mapping[str, Any]] = []
    for window_id in sorted({str(r["window_id"]) for r in eval_rows}):
        rows = [r for r in eval_rows if r["window_id"] == window_id and r["treatment_kind"] == "fixed_policy"]
        if not rows:
            continue
        regime = str(rows[0]["regime"])
        selected = learned.get(regime, fallback)
        match = [r for r in rows if r["treatment_id"] == selected]
        out.append(match[0] if match else rows[0])
    return out


def composition_weight_rows(mixtures: Sequence[MixtureSpec], method_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    selected_global = next((r["selected_treatment_id"] for r in method_rows if r["method_id"] == "best_global_mixture"), None)
    out = []
    for mixture in mixtures:
        out.append({
            "mixture_id": mixture.mixture_id,
            "weights_json": json.dumps(mixture.weights, sort_keys=True),
            "top_k": mixture.top_k,
            "selected_as_best_global": mixture.mixture_id == selected_global,
        })
    return out


def near_tie_sensitivity_rows(per_window: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> list[dict[str, Any]]:
    eval_splits = set(config.get("evaluation_splits", sorted(EVAL_SPLITS)))
    eval_rows = [r for r in per_window if r["split"] in eval_splits]
    out = []
    for threshold in config.get("near_tie_thresholds", [0.001, 0.005, 0.01]):
        thr = float(threshold)
        non_tie = [r for r in eval_rows if float(r["fixed_top2_margin"]) >= thr]
        out.append({
            "threshold": thr,
            "n_windows": len(eval_rows),
            "near_tie_count": len(eval_rows) - len(non_tie),
            "near_tie_fraction": (len(eval_rows) - len(non_tie)) / len(eval_rows) if eval_rows else None,
            "non_near_tie_gap": mean(float(r["composition_opportunity_gap"]) for r in non_tie),
            "non_near_tie_count": len(non_tie),
        })
    return out


def subset_analysis_rows(per_window: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> list[dict[str, Any]]:
    eval_splits = set(config.get("evaluation_splits", sorted(EVAL_SPLITS)))
    eval_rows = [r for r in per_window if r["split"] in eval_splits]
    out: list[dict[str, Any]] = []
    for regime in sorted({str(r["regime"]) for r in eval_rows}):
        rows = [r for r in eval_rows if r["regime"] == regime]
        out.append({
            "subset": f"regime:{regime}",
            "n_windows": len(rows),
            "mean_opportunity_gap": mean(float(r["composition_opportunity_gap"]) for r in rows),
            "mean_oracle_fixed_anwg": mean(float(r["oracle_fixed_anwg"]) for r in rows),
            "mean_oracle_mixture_anwg": mean(float(r["oracle_mixture_anwg"]) for r in rows),
            "mean_oracle_fixed_completion": mean(float(r["oracle_fixed_completion_fraction"]) for r in rows),
            "mean_oracle_mixture_completion": mean(float(r["oracle_mixture_completion_fraction"]) for r in rows),
            "near_tie_fraction": mean(1.0 if bool(r["near_tie"]) else 0.0 for r in rows),
        })
    non_tie = [r for r in eval_rows if not bool(r["near_tie"])]
    out.append({
        "subset": "non_near_tie",
        "n_windows": len(non_tie),
        "mean_opportunity_gap": mean(float(r["composition_opportunity_gap"]) for r in non_tie),
        "mean_oracle_fixed_anwg": mean(float(r["oracle_fixed_anwg"]) for r in non_tie),
        "mean_oracle_mixture_anwg": mean(float(r["oracle_mixture_anwg"]) for r in non_tie),
        "mean_oracle_fixed_completion": mean(float(r["oracle_fixed_completion_fraction"]) for r in non_tie),
        "mean_oracle_mixture_completion": mean(float(r["oracle_mixture_completion_fraction"]) for r in non_tie),
        "near_tie_fraction": 0.0 if non_tie else None,
    })
    return out


def determine_verdict(
    per_window: Sequence[Mapping[str, Any]],
    method_rows: Sequence[Mapping[str, Any]],
    subset_rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    eval_splits = set(config.get("evaluation_splits", sorted(EVAL_SPLITS)))
    eval_rows = [r for r in per_window if r["split"] in eval_splits]
    thresholds = config.get("thresholds", {})
    proceed_gap = float(thresholds.get("aggregate_anwg_gain", 0.005))
    regime_gap_threshold = float(thresholds.get("regime_specific_gain", 0.01))
    stop_gap = float(thresholds.get("stop_non_near_tie_gap", 0.002))
    completion_tolerance = float(config["metrics"].get("completion_fraction_tolerance", 0.005))

    opportunity_gap = mean(float(r["composition_opportunity_gap"]) for r in eval_rows)
    non_near_tie = [r for r in eval_rows if not bool(r["near_tie"])]
    non_near_tie_gap = mean(float(r["composition_opportunity_gap"]) for r in non_near_tie)
    best_fixed = next((r for r in method_rows if r["method_id"] == "best_fixed_policy"), None)
    oracle_mix = next((r for r in method_rows if r["method_id"] == "oracle_best_mixture_per_window"), None)
    oracle_fixed = next((r for r in method_rows if r["method_id"] == "oracle_best_fixed_per_window"), None)
    completion_impact = (
        float(oracle_mix["mean_completion_fraction"]) - float(best_fixed["mean_completion_fraction"])
        if oracle_mix and best_fixed and oracle_mix["mean_completion_fraction"] is not None and best_fixed["mean_completion_fraction"] is not None
        else None
    )
    completion_ok = completion_impact is not None and completion_impact >= -completion_tolerance
    regime_rows = [r for r in subset_rows if str(r["subset"]).startswith("regime:")]
    best_regime_gain = max((float(r["mean_opportunity_gap"]) for r in regime_rows if r["mean_opportunity_gap"] is not None), default=float("nan"))

    if not eval_rows or oracle_mix is None or oracle_fixed is None:
        verdict = "INCONCLUSIVE"
        reason = "missing evaluation rows or oracle summaries"
    elif not completion_ok:
        verdict = "STOP_OR_REDESIGN"
        reason = "completion-fraction constraint failed"
    elif non_near_tie and non_near_tie_gap >= proceed_gap:
        verdict = "PROCEED"
        reason = "aggregate non-near-tie opportunity gap passed"
    elif math.isfinite(best_regime_gain) and best_regime_gain >= regime_gap_threshold:
        verdict = "REGIME_SPECIFIC_ONLY"
        reason = "only regime-specific opportunity passed"
    elif non_near_tie and non_near_tie_gap < stop_gap:
        verdict = "STOP_OR_REDESIGN"
        reason = "non-near-tie opportunity gap below stop threshold"
    elif opportunity_gap <= 0.0:
        verdict = "STOP_OR_REDESIGN"
        reason = "oracle mixture did not beat oracle fixed on average"
    else:
        verdict = "INCONCLUSIVE"
        reason = "opportunity signal did not satisfy proceed or stop thresholds"

    return {
        "verdict": verdict,
        "reason": reason,
        "composition_opportunity_gap": opportunity_gap,
        "non_near_tie_gap": non_near_tie_gap,
        "non_near_tie_count": len(non_near_tie),
        "completion_impact_vs_best_fixed": completion_impact,
        "completion_ok": completion_ok,
        "best_regime_gain": best_regime_gain if math.isfinite(best_regime_gain) else None,
        "thresholds": {
            "aggregate_anwg_gain": proceed_gap,
            "regime_specific_gain": regime_gap_threshold,
            "stop_non_near_tie_gap": stop_gap,
            "completion_fraction_tolerance": completion_tolerance,
        },
    }


def metric_value(row: Mapping[str, Any], key: str) -> float:
    value = row.get(key)
    if value is None or value == "":
        return float("nan")
    return float(value)


def mean(values: Iterable[float]) -> float | None:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return sum(vals) / len(vals) if vals else None


def resolve_output_dir(config: Mapping[str, Any], *, timestamp: str | None) -> Path:
    root = ROOT / str(config["outputs"].get("root", "results/cc1_composition_opportunity"))
    stamp = timestamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return root / stamp


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_yaml(path: Path, data: Mapping[str, Any]) -> None:
    path.write_text(yaml.safe_dump(dict(data), sort_keys=False))


def build_manifest(
    config: Mapping[str, Any],
    *,
    config_path: str | Path,
    output_dir: Path,
    git: Mapping[str, Any],
    planned_count: int,
    windows: Sequence[WorkloadWindow],
    mixtures: Sequence[MixtureSpec],
    skipped_traces: Sequence[Mapping[str, str]],
    runtime_s: float,
    verdict_payload: Mapping[str, Any],
) -> dict[str, Any]:
    config_text = yaml.safe_dump(dict(config), sort_keys=True)
    return {
        "schema_version": 1,
        "experiment": "cc1_composition_opportunity",
        "mode": config["mode"],
        "config_path": str(config_path),
        "config_hash": hashlib.sha256(config_text.encode("utf-8")).hexdigest(),
        "output_dir": display_path(output_dir),
        "git": dict(git),
        "seed": config.get("seed"),
        "policy_subset": list(config["policy_subset"]),
        "composition": dict(config["composition"]),
        "discriminativeness": dict(config.get("discriminativeness", {})),
        "planned_run_count": planned_count,
        "window_count": len(windows),
        "mixture_count": len(mixtures),
        "windows": [
            {"window_id": w.window_id, "split": w.split, "regime": w.regime, "source": w.source, "seed": w.seed, "num_requests": len(w.requests)}
            for w in windows
        ],
        "skipped_real_traces": list(skipped_traces),
        "no_live_api": True,
        "no_gpu": True,
        "no_real_vllm": True,
        "runtime_s": round(runtime_s, 3),
        "verdict": dict(verdict_payload),
    }


def render_report(
    manifest: Mapping[str, Any],
    method_rows: Sequence[Mapping[str, Any]],
    subset_rows: Sequence[Mapping[str, Any]],
    verdict: Mapping[str, Any],
) -> str:
    best_global = next((r for r in method_rows if r["method_id"] == "best_global_mixture"), {})
    best_fixed = next((r for r in method_rows if r["method_id"] == "best_fixed_policy"), {})
    lines = [
        "# CC1 Composition Opportunity Report",
        "",
        f"Verdict: `{verdict['verdict']}`",
        f"Reason: {verdict['reason']}",
        "",
        "## Run",
        "",
        f"- Commit: `{manifest['git']['commit']}`",
        f"- Mode: `{manifest['mode']}`",
        f"- Planned simulator executions: `{manifest['planned_run_count']}`",
        f"- Windows: `{manifest['window_count']}`",
        f"- Mixtures: `{manifest['mixture_count']}`",
        "",
        "## Key Results",
        "",
        f"- Best fixed policy: `{best_fixed.get('selected_treatment_id')}`",
        f"- Best global mixture: `{best_global.get('selected_treatment_id')}`",
        f"- Composition opportunity gap: `{verdict['composition_opportunity_gap']}`",
        f"- Non-near-tie gap: `{verdict['non_near_tie_gap']}`",
        f"- Non-near-tie count: `{verdict['non_near_tie_count']}`",
        f"- Completion impact vs best fixed: `{verdict['completion_impact_vs_best_fixed']}`",
        f"- Best regime gain: `{verdict['best_regime_gain']}`",
        "",
        "## Regime Summary",
        "",
    ]
    for row in subset_rows:
        lines.append(
            f"- `{row['subset']}`: n={row['n_windows']}, "
            f"gap={row['mean_opportunity_gap']}, near_tie_fraction={row['near_tie_fraction']}"
        )
    return "\n".join(lines) + "\n"


def git_state() -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()

    status = git("status", "--short")
    return {
        "branch": git("branch", "--show-current"),
        "commit": git("rev-parse", "HEAD"),
        "upstream": git("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"),
        "ahead_behind": git("rev-list", "--left-right", "--count", "HEAD...@{u}"),
        "dirty": bool(status),
    }


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)
