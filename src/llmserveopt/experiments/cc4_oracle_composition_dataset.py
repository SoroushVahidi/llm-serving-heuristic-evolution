"""CC4 true simulator-executed oracle composition dataset.

Builds a reproducible, resumable dataset of (workload window x composition
candidate) simulator executions, then derives oracle labels, a regret
matrix, near-tie flags, and completion constraints from the executed rows --
matching the roadmap's CC4 required outputs. Candidates are compiled,
verified CC3 DSL heuristics (weighted-primitive mixtures, sparse top-k
mixtures, admission-gate and placement variants) plus fixed baselines and
the CC1b weighted-Borda reference; every candidate is executed through the
exact same ``run_policy`` entry point CC1 and the selector datasets use --
no new simulator-invocation code, no reward-vector interpolation.

Heavy reuse from ``cc1_composition_opportunity``: workload-window
construction (with its split-leakage check), GPU/service-model construction,
git-state capture, and CSV writing are imported, not re-derived.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import pandas as pd
import yaml

from llmserveopt.core.metrics import RunMetrics, metrics_to_dict
from llmserveopt.core.types import GPUConfig
from llmserveopt.evaluation.run_policy import run_policy
from llmserveopt.experiments.cc1_composition_opportunity import (
    ROOT,
    WorkloadWindow,
    build_gpu_configs,
    build_service_model,
    build_workload_windows,
    display_path,
    git_state,
    mean,
    metric_value,
    simplex_weight_grid,
)
from llmserveopt.heuristics.compiler import CompilationError, compile_heuristic
from llmserveopt.heuristics.dsl_schema import COMPILER_VERSION, DSL_SCHEMA_VERSION, heuristic_hash
from llmserveopt.heuristics.policy import HeuristicPolicy
from llmserveopt.heuristics.verifier import verify_heuristic
from llmserveopt.policies.composition import RankExpertSpec, StaticRankEnsemblePolicy
from llmserveopt.policies.registry import make_policy
from llmserveopt.simulator.service_model import ServiceModel

PRIMARY = "arrival_normalized_weighted_goodput"
PRIMARY_COL = f"metric_{PRIMARY}"
COMPLETION_COL = "metric_completion_fraction"


class CC4Error(ValueError):
    """Raised when the CC4 config or runtime state is invalid."""


# ---------------------------------------------------------------------------
# Config loading (identical shape to CC1's load_config; kept local so CC4
# does not depend on CC1's validate_config, which enforces CC1-only fields)
# ---------------------------------------------------------------------------


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise CC4Error("config must be a YAML mapping")
    return data


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise CC4Error("schema_version must be 1")
    if config.get("mode") != "cc4":
        raise CC4Error("mode must be 'cc4'")
    for required in ("policy_subset", "candidate_search", "metrics", "safeguards", "outputs", "gpus", "workloads"):
        if required not in config:
            raise CC4Error(f"missing required config section: {required}")
    if config["metrics"].get("primary") != PRIMARY:
        raise CC4Error(f"primary metric must be {PRIMARY!r}")


# ---------------------------------------------------------------------------
# Candidate representation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    family: str  # fixed_policy | cc1b_borda_baseline | weighted_primitive_mixture |
    #              sparse_topk_mixture | admission_gate_variant | placement_variant
    heuristic_doc: dict[str, Any] | None = None
    policy_name: str | None = None
    borda_weights: dict[str, float] | None = None
    primitive_weights: dict[str, float] = field(default_factory=dict)
    extra_params: dict[str, Any] = field(default_factory=dict)


def _primitive_leaf(name: str, higher_is_preferred: bool) -> dict[str, Any]:
    leaf: dict[str, Any] = {"primitive": name}
    return leaf if higher_is_preferred else {"op": "neg", "args": [leaf]}


def build_fixed_candidates(policy_subset: Sequence[str]) -> list[Candidate]:
    return [Candidate(f"fixed__{name}", "fixed_policy", policy_name=name) for name in policy_subset]


def build_cc1b_borda_candidate(config: Mapping[str, Any]) -> Candidate | None:
    cfg = config.get("cc1b_borda_baseline")
    if not cfg:
        return None
    weights = {str(k): float(v) for k, v in cfg["weights"].items()}
    return Candidate("cc1b_weighted_borda_baseline", "cc1b_borda_baseline", borda_weights=weights)


def build_weighted_mixture_candidates(search_cfg: Mapping[str, Any]) -> list[Candidate]:
    pool = search_cfg["primitive_pool"]
    orientation = {p["name"]: bool(p["higher_is_preferred"]) for p in pool}
    names = [p["name"] for p in pool]
    mixtures = simplex_weight_grid(
        names, step=float(search_cfg["weight_grid_step"]), top_k=int(search_cfg["weight_grid_top_k"])
    )
    out = []
    for mix in mixtures:
        candidate_id = f"wmix__{mix.mixture_id[len('mix__'):]}"
        terms = [[_primitive_leaf(name, orientation[name]), weight] for name, weight in mix.weights.items()]
        doc = {
            "name": candidate_id,
            "tie_breaker": "earliest_deadline",
            "default": {"request_score": {"op": "weighted_sum", "terms": terms}},
        }
        out.append(Candidate(
            candidate_id, "weighted_primitive_mixture", heuristic_doc=doc,
            primitive_weights=dict(mix.weights),
        ))
    return out


def build_topk_candidates(search_cfg: Mapping[str, Any]) -> list[Candidate]:
    pool = search_cfg["primitive_pool"]
    orientation = {p["name"]: bool(p["higher_is_preferred"]) for p in pool}
    names = [p["name"] for p in pool]
    out = []
    for k in search_cfg["topk_mixture_k_values"]:
        k = int(k)
        candidate_id = f"topk{k}__pool"
        terms = [[_primitive_leaf(name, orientation[name]), 1.0] for name in names]
        doc = {
            "name": candidate_id,
            "tie_breaker": "earliest_deadline",
            "default": {"request_score": {"op": "topk_mixture", "k": k, "terms": terms}},
        }
        out.append(Candidate(
            candidate_id, "sparse_topk_mixture", heuristic_doc=doc,
            primitive_weights={n: 1.0 for n in names}, extra_params={"k": k},
        ))
    return out


def build_admission_gate_candidates(search_cfg: Mapping[str, Any]) -> list[Candidate]:
    out = []
    for threshold in search_cfg["admission_gate_laxity_thresholds"]:
        threshold = float(threshold)
        suffix = f"{threshold:.3f}".replace(".", "p")
        candidate_id = f"admgate__thr{suffix}"
        doc = {
            "name": candidate_id,
            "tie_breaker": "earliest_deadline",
            "fallback": {"policy": "fifo_like"},
            "on_no_admits": "safe_fallback",
            "default": {
                "request_score": {"primitive": "laxity_urgency"},
                "admission_condition": {
                    "primitive_gate": "laxity_gate",
                    "params": {"laxity_threshold": threshold},
                },
            },
        }
        out.append(Candidate(
            candidate_id, "admission_gate_variant", heuristic_doc=doc,
            primitive_weights={"laxity_urgency": 1.0}, extra_params={"laxity_threshold": threshold},
        ))
    return out


def build_placement_candidates(search_cfg: Mapping[str, Any]) -> list[Candidate]:
    out = []
    for keys in search_cfg["placement_key_variants"]:
        if not keys:
            continue  # the empty variant is exactly the fixed/weighted-mixture default behavior -- not a new candidate
        candidate_id = f"place__{'_'.join(keys)}"
        doc = {
            "name": candidate_id,
            "tie_breaker": "earliest_deadline",
            "placement": {"keys": [{"name": k} for k in keys]},
            "default": {"request_score": {"primitive": "laxity_urgency"}},
        }
        out.append(Candidate(
            candidate_id, "placement_variant", heuristic_doc=doc,
            primitive_weights={"laxity_urgency": 1.0}, extra_params={"placement_keys": list(keys)},
        ))
    return out


def maybe_generate_cloudrift_candidates(config: Mapping[str, Any]) -> tuple[list[Candidate], dict[str, Any]]:
    """Optional, additive CloudRift-proposed candidate templates.

    Clean no-op unless CLOUDRIFT_API_KEY is set (opt-in) AND
    config['cloudrift']['enabled'] is true. Every returned candidate would
    still be verified by verify_heuristic() before execution like any other
    candidate -- CloudRift output is never treated as ground truth. No live
    API call is made in this implementation; the cache/dedup skeleton exists
    for a future integration to fill in without changing this function's
    contract.
    """
    cr_cfg = config.get("cloudrift") or {}
    api_key = os.environ.get("CLOUDRIFT_API_KEY")
    info: dict[str, Any] = {
        "requested": bool(cr_cfg.get("enabled", False)),
        "api_key_present": bool(api_key),
        "used": False,
        "provider": "cloudrift",
        "model": cr_cfg.get("model"),
        "calls": 0,
        "cost_usd": 0.0,
        "cache_hits": 0,
        "skip_reason": None,
    }
    if not cr_cfg.get("enabled", False):
        info["skip_reason"] = "cloudrift.enabled is false in config"
        return [], info
    if not api_key:
        info["skip_reason"] = "CLOUDRIFT_API_KEY not set"
        return [], info
    # Live API integration intentionally not implemented in this query
    # (no opt-in credentials were supplied); skip cleanly rather than guess.
    info["skip_reason"] = "live CloudRift integration not implemented in this run"
    return [], info


def generate_all_candidates(config: Mapping[str, Any]) -> list[Candidate]:
    candidates = build_fixed_candidates(config["policy_subset"])
    borda = build_cc1b_borda_candidate(config)
    if borda is not None:
        candidates.append(borda)
    search_cfg = config["candidate_search"]
    candidates += build_weighted_mixture_candidates(search_cfg)
    candidates += build_topk_candidates(search_cfg)
    candidates += build_admission_gate_candidates(search_cfg)
    candidates += build_placement_candidates(search_cfg)
    cloudrift_candidates, _ = maybe_generate_cloudrift_candidates(config)
    candidates += cloudrift_candidates
    seen: set[str] = set()
    deduped: list[Candidate] = []
    for c in candidates:
        if c.candidate_id in seen:
            raise CC4Error(f"duplicate candidate_id: {c.candidate_id}")
        seen.add(c.candidate_id)
        deduped.append(c)
    return deduped


# ---------------------------------------------------------------------------
# Verification + policy construction
# ---------------------------------------------------------------------------


def verify_candidate(candidate: Candidate) -> tuple[bool, list[str], str]:
    """Return (ok, error_codes, dsl_hash). Non-DSL candidates (fixed
    policies, the CC1b Borda baseline) are always ok with an empty hash."""
    if candidate.heuristic_doc is None:
        return True, [], ""
    result = verify_heuristic(candidate.heuristic_doc)
    dsl_hash = heuristic_hash(candidate.heuristic_doc) if result.valid else ""
    return result.valid, [code for code, _ in result.errors], dsl_hash


def build_policy_for_candidate(candidate: Candidate, *, seed: int):
    if candidate.family == "fixed_policy":
        return make_policy(candidate.policy_name, seed=seed)
    if candidate.family == "cc1b_borda_baseline":
        experts = [RankExpertSpec(name, weight) for name, weight in candidate.borda_weights.items()]
        return StaticRankEnsemblePolicy(experts, method="borda", top_k=2)
    compiled = compile_heuristic(candidate.heuristic_doc)
    return HeuristicPolicy(compiled)


# ---------------------------------------------------------------------------
# Resumable trial store
# ---------------------------------------------------------------------------


class CC4TrialStore:
    """Append-only JSONL of completed (window_id, candidate_id) execution
    rows. Loaded at construction so a re-run skips already-completed work --
    the same trial-id-keyed append/skip pattern as
    scripts/run_module_credit_overnight.py's TrialStore, adapted to a
    composite string key instead of an integer trial counter."""

    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        self.jsonl_path = out_dir / "checkpoints" / "trial_results.jsonl"
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self.completed_keys: set[str] = self._read_completed()

    @staticmethod
    def trial_key(window_id: str, candidate_id: str) -> str:
        return f"{window_id}::{candidate_id}"

    def _read_completed(self) -> set[str]:
        if not self.jsonl_path.exists():
            return set()
        keys: set[str] = set()
        with self.jsonl_path.open() as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                keys.add(self.trial_key(row["window_id"], row["candidate_id"]))
        return keys

    def append(self, row: Mapping[str, Any]) -> None:
        with self.jsonl_path.open("a") as handle:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        self.completed_keys.add(self.trial_key(row["window_id"], row["candidate_id"]))

    def load_all_rows(self) -> list[dict[str, Any]]:
        if not self.jsonl_path.exists():
            return []
        rows = []
        with self.jsonl_path.open() as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows


def heartbeat(out_dir: Path, stage: str, **payload: Any) -> None:
    path = out_dir / "checkpoints" / "heartbeat.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "stage": stage,
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pid": os.getpid(),
        **payload,
    }, indent=2, sort_keys=True, default=str))


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def execute_candidate_row(
    runner: Callable[..., RunMetrics],
    candidate: Candidate,
    window: WorkloadWindow,
    gpu_configs: list[GPUConfig],
    service_model: ServiceModel,
    *,
    drain_steps: int,
    git_sha: str,
) -> dict[str, Any]:
    ok, error_codes, dsl_hash = verify_candidate(candidate)
    row: dict[str, Any] = {
        "window_id": window.window_id,
        "split": window.split,
        "regime": window.regime,
        "source": window.source,
        "seed": window.seed,
        "candidate_id": candidate.candidate_id,
        "family": candidate.family,
        "policy_name": candidate.policy_name or "",
        "git_sha": git_sha,
        "composition_hash": dsl_hash,
        "dsl_schema_version": DSL_SCHEMA_VERSION if candidate.heuristic_doc is not None else None,
        "compiler_version": COMPILER_VERSION if candidate.heuristic_doc is not None else None,
        "primitive_weights_json": json.dumps(candidate.primitive_weights, sort_keys=True),
        "extra_params_json": json.dumps(candidate.extra_params, sort_keys=True),
        "verification_outcome": "valid" if ok else "invalid:" + ",".join(error_codes),
        "true_simulator_executed": False,
        "reward_vector_interpolated": False,
        "fallback_activated_last_step": False,
        "runtime_s": 0.0,
    }
    if not ok:
        return row
    t0 = time.perf_counter()
    try:
        policy = build_policy_for_candidate(candidate, seed=window.seed)
    except CompilationError as exc:
        row["verification_outcome"] = f"compile_error:{exc}"
        return row
    metrics = runner(
        policy=policy,
        requests=list(window.requests),
        gpu_configs=gpu_configs,
        service_model=service_model,
        workload_tag=window.regime,
        seed=window.seed,
        drain_steps=drain_steps,
    )
    row["true_simulator_executed"] = True
    row["runtime_s"] = round(time.perf_counter() - t0, 4)
    if isinstance(policy, HeuristicPolicy) and policy.last_trace:
        row["fallback_activated_last_step"] = bool(policy.last_trace.get("fallback_activated", False))
    md = metrics_to_dict(metrics)
    for key, value in md.items():
        if key in ("policy", "workload"):
            continue
        row[f"metric_{key}"] = value
    return row


# ---------------------------------------------------------------------------
# Post-processing: oracle labels, regret, near-tie, completion constraints
# ---------------------------------------------------------------------------


def _executed_only(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["true_simulator_executed"] == True].copy()  # noqa: E712


def compute_oracle_labels(rows_df: pd.DataFrame) -> pd.DataFrame:
    executed = _executed_only(rows_df)
    out = []
    for window_id, group in executed.groupby("window_id"):
        ranked = group.sort_values(PRIMARY_COL, ascending=False)
        best = ranked.iloc[0]
        second = ranked.iloc[1] if len(ranked) > 1 else None
        margin = float(best[PRIMARY_COL]) - float(second[PRIMARY_COL]) if second is not None else float("nan")
        out.append({
            "window_id": window_id,
            "split": best["split"],
            "regime": best["regime"],
            "source": best["source"],
            "oracle_candidate_id": best["candidate_id"],
            "oracle_family": best["family"],
            "oracle_anwg": float(best[PRIMARY_COL]),
            "oracle_completion_fraction": float(best[COMPLETION_COL]),
            "second_best_candidate_id": second["candidate_id"] if second is not None else None,
            "top2_margin": margin,
            "n_candidates": len(ranked),
        })
    return pd.DataFrame(out).sort_values("window_id").reset_index(drop=True)


def compute_regret_matrix(rows_df: pd.DataFrame, oracle_df: pd.DataFrame) -> pd.DataFrame:
    executed = _executed_only(rows_df)
    oracle_by_window = oracle_df.set_index("window_id")["oracle_anwg"].to_dict()
    out = []
    for _, row in executed.iterrows():
        oracle_anwg = oracle_by_window.get(row["window_id"])
        if oracle_anwg is None:
            continue
        out.append({
            "window_id": row["window_id"],
            "candidate_id": row["candidate_id"],
            "family": row["family"],
            "anwg": float(row[PRIMARY_COL]),
            "oracle_anwg": float(oracle_anwg),
            "regret": float(oracle_anwg) - float(row[PRIMARY_COL]),
        })
    return pd.DataFrame(out).sort_values(["window_id", "regret"]).reset_index(drop=True)


def compute_near_tie_flags(oracle_df: pd.DataFrame, thresholds: Sequence[float]) -> pd.DataFrame:
    out = []
    for _, row in oracle_df.iterrows():
        for threshold in thresholds:
            margin = row["top2_margin"]
            near_tie = bool(math.isnan(margin) or margin < float(threshold))
            out.append({
                "window_id": row["window_id"],
                "threshold": float(threshold),
                "top2_margin": margin,
                "near_tie": near_tie,
            })
    return pd.DataFrame(out)


def compute_completion_constraints(rows_df: pd.DataFrame, oracle_df: pd.DataFrame, tolerance: float) -> pd.DataFrame:
    executed = _executed_only(rows_df)
    fixed = executed[executed["family"] == "fixed_policy"]
    out = []
    for _, orow in oracle_df.iterrows():
        window_id = orow["window_id"]
        window_fixed = fixed[fixed["window_id"] == window_id]
        if window_fixed.empty:
            continue
        best_fixed = window_fixed.sort_values(PRIMARY_COL, ascending=False).iloc[0]
        impact = float(orow["oracle_completion_fraction"]) - float(best_fixed[COMPLETION_COL])
        out.append({
            "window_id": window_id,
            "split": orow["split"],
            "regime": orow["regime"],
            "best_fixed_candidate_id": best_fixed["candidate_id"],
            "best_fixed_completion_fraction": float(best_fixed[COMPLETION_COL]),
            "oracle_completion_fraction": float(orow["oracle_completion_fraction"]),
            "completion_impact": impact,
            "completion_ok": impact >= -float(tolerance),
        })
    return pd.DataFrame(out)


def compute_primitive_usage_statistics(candidates: Sequence[Candidate], oracle_df: pd.DataFrame) -> pd.DataFrame:
    all_names: set[str] = set()
    for c in candidates:
        all_names.update(c.primitive_weights.keys())
    searched_counts = {name: 0 for name in all_names}
    for c in candidates:
        for name in c.primitive_weights:
            searched_counts[name] += 1
    oracle_ids = set(oracle_df["oracle_candidate_id"]) if not oracle_df.empty else set()
    by_id = {c.candidate_id: c for c in candidates}
    oracle_counts = {name: 0 for name in all_names}
    for cid in oracle_ids:
        cand = by_id.get(cid)
        if cand is None:
            continue
        for name in cand.primitive_weights:
            oracle_counts[name] += 1
    return pd.DataFrame([
        {
            "primitive_name": name,
            "n_candidates_referencing": searched_counts[name],
            "n_windows_oracle_selected": oracle_counts[name],
        }
        for name in sorted(all_names)
    ])


def compute_search_summary(
    candidates: Sequence[Candidate],
    rejected: Sequence[dict[str, Any]],
    rows_df: pd.DataFrame,
    windows: Sequence[WorkloadWindow],
    skipped_traces: Sequence[Mapping[str, str]],
) -> pd.DataFrame:
    by_family: dict[str, int] = {}
    for c in candidates:
        by_family[c.family] = by_family.get(c.family, 0) + 1
    executed = _executed_only(rows_df)
    unique_hashes = executed["composition_hash"].replace("", pd.NA).dropna().nunique()
    summary = {
        "n_windows": len(windows),
        "n_windows_skipped_real_trace": len(skipped_traces),
        "n_candidates_total": len(candidates),
        "n_candidates_rejected": len(rejected),
        "n_simulator_executions": len(executed),
        "n_rows_total": len(rows_df),
        "n_unique_verified_compositions": int(unique_hashes),
    }
    for family, count in sorted(by_family.items()):
        summary[f"n_candidates__{family}"] = count
    return pd.DataFrame([summary])


def build_causal_features(windows: Sequence[WorkloadWindow]) -> pd.DataFrame:
    rows = []
    for w in windows:
        reqs = w.requests
        n = len(reqs)
        prompt_tokens = [r.prompt_tokens for r in reqs]
        output_tokens = [r.predicted_output_tokens for r in reqs]
        slacks = [max(r.slo_deadline - r.arrival_time, 1e-6) for r in reqs]
        duration = max((r.arrival_time for r in reqs), default=0.0)
        class_counts: dict[str, int] = {}
        for r in reqs:
            class_counts[r.class_id] = class_counts.get(r.class_id, 0) + 1
        rows.append({
            "window_id": w.window_id,
            "split": w.split,
            "regime": w.regime,
            "source": w.source,
            "num_requests": n,
            "mean_prompt_tokens": mean(prompt_tokens) if n else None,
            "mean_predicted_output_tokens": mean(output_tokens) if n else None,
            "mean_slo_slack": mean(slacks) if n else None,
            "arrival_span_s": duration,
            "arrival_rate_est": n / duration if duration > 0 else None,
            "num_slo_classes": len(class_counts),
            "class_distribution_json": json.dumps(class_counts, sort_keys=True),
        })
    return pd.DataFrame(rows)


def build_workload_windows_table(windows: Sequence[WorkloadWindow]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "window_id": w.window_id, "split": w.split, "regime": w.regime,
            "source": w.source, "seed": w.seed, "num_requests": len(w.requests),
        }
        for w in windows
    ])


def build_candidate_compositions_table(candidates: Sequence[Candidate], rejected: Sequence[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for c in candidates:
        ok, error_codes, dsl_hash = verify_candidate(c)
        rows.append({
            "candidate_id": c.candidate_id,
            "family": c.family,
            "policy_name": c.policy_name or "",
            "heuristic_doc_json": json.dumps(c.heuristic_doc, sort_keys=True) if c.heuristic_doc else "",
            "composition_hash": dsl_hash,
            "verification_outcome": "valid" if ok else "invalid:" + ",".join(error_codes),
        })
    return pd.DataFrame(rows)


def build_composition_parameters_table(candidates: Sequence[Candidate]) -> pd.DataFrame:
    rows = []
    for c in candidates:
        for primitive_name, weight in c.primitive_weights.items():
            rows.append({
                "candidate_id": c.candidate_id,
                "family": c.family,
                "primitive_name": primitive_name,
                "weight": weight,
                "extra_params_json": json.dumps(c.extra_params, sort_keys=True),
            })
        if not c.primitive_weights:
            rows.append({
                "candidate_id": c.candidate_id, "family": c.family,
                "primitive_name": None, "weight": None,
                "extra_params_json": json.dumps(c.extra_params, sort_keys=True),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Manifest / dataset card / replay commands
# ---------------------------------------------------------------------------


def resolve_output_dir(config: Mapping[str, Any], *, timestamp: str | None) -> Path:
    root = ROOT / str(config["outputs"].get("root", "results/cc4_oracle_composition_dataset"))
    stamp = timestamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return root / stamp


def config_hash(config: Mapping[str, Any]) -> str:
    text = yaml.safe_dump(dict(config), sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_manifest(
    config: Mapping[str, Any],
    *,
    config_path: str | Path,
    output_dir: Path,
    git: Mapping[str, Any],
    windows: Sequence[WorkloadWindow],
    skipped_traces: Sequence[Mapping[str, str]],
    candidates: Sequence[Candidate],
    rejected: Sequence[dict[str, Any]],
    n_executed: int,
    runtime_s: float,
    resumed: bool,
    cloudrift_info: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment": "cc4_oracle_composition_dataset",
        "mode": config["mode"],
        "config_path": str(config_path),
        "config_hash": config_hash(config),
        "output_dir": display_path(output_dir),
        "git": dict(git),
        "seed": config.get("seed"),
        "policy_subset": list(config["policy_subset"]),
        "window_count": len(windows),
        "skipped_real_traces": list(skipped_traces),
        "candidate_count": len(candidates),
        "candidate_families": sorted({c.family for c in candidates}),
        "rejected_candidate_count": len(rejected),
        "simulator_executions": n_executed,
        "resumed_from_checkpoint": resumed,
        "no_live_api": True,
        "no_gpu": True,
        "no_real_vllm": True,
        "cloudrift": dict(cloudrift_info),
        "runtime_s": round(runtime_s, 3),
        "dsl_schema_version": DSL_SCHEMA_VERSION,
        "compiler_version": COMPILER_VERSION,
    }


def render_dataset_card(manifest: Mapping[str, Any], search_summary: pd.DataFrame, verdict: Mapping[str, Any]) -> str:
    s = search_summary.iloc[0].to_dict() if not search_summary.empty else {}
    lines = [
        "# CC4 Oracle Composition Dataset Card",
        "",
        f"Generated: {manifest['git']['commit']}",
        f"Config hash: `{manifest['config_hash']}`",
        "",
        "## Coverage",
        "",
        f"- Windows: {s.get('n_windows')} ({s.get('n_windows_skipped_real_trace')} real-trace windows skipped -- local data missing)",
        f"- Candidates searched: {s.get('n_candidates_total')} ({s.get('n_candidates_rejected')} rejected by the verifier)",
        f"- Simulator executions: {s.get('n_simulator_executions')}",
        f"- Unique verified compositions: {s.get('n_unique_verified_compositions')}",
        "",
        "## Verdict",
        "",
        f"- Status: `{verdict['status']}`",
        f"- Reason: {verdict['reason']}",
        f"- Oracle composition gain over oracle fixed: `{verdict['oracle_composition_gain']}`",
        f"- Near-tie fraction (primary threshold): `{verdict['near_tie_fraction']}`",
        "",
        "## Files",
        "",
        "See `manifest.json` for the full file list and schema pointers; "
        "`docs/audits/contextual_composition_cc4_oracle_dataset_report_20260803.md` "
        "documents the search design, splits, and limitations in full.",
        "",
        "## Reproduction",
        "",
        "```bash",
        "bash replay_commands.sh",
        "```",
    ]
    return "\n".join(lines) + "\n"


def render_replay_commands(config_path: str | Path, output_dir: Path) -> str:
    rel_config = display_path(Path(config_path)) if Path(config_path).is_absolute() else str(config_path)
    return (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "cd \"$(git rev-parse --show-toplevel)\"\n"
        f"python scripts/run_cc4_oracle_composition_dataset.py --config {rel_config} "
        f"--full-run --resume-dir {display_path(output_dir)}\n"
    )


def determine_dataset_verdict(
    oracle_df: pd.DataFrame,
    near_tie_df: pd.DataFrame,
    completion_df: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Compute the dataset-level verdict restricted to evaluation-split
    windows only (development-split windows may have been used to pick
    representative candidates/regimes and must not also certify the
    held-out signal -- the same dev/eval separation CC1 applies to its own
    verdict). oracle_labels.parquet/regret_matrix.parquet/etc. themselves
    still cover *all* windows (dev+eval) for CC5 to train on."""
    default_eval_splits = ["ID_TEST", "OOD_TEST", "TEMPORAL_OOD", "CROSS_SOURCE_OOD", "FINAL_OOD"]
    eval_splits = set(config.get("evaluation_splits", default_eval_splits))
    oracle_eval = oracle_df[oracle_df["split"].isin(eval_splits)] if not oracle_df.empty else oracle_df
    if oracle_eval.empty:
        return {"status": "INCONCLUSIVE", "reason": "no executed evaluation-split rows", "oracle_composition_gain": None, "near_tie_fraction": None}

    eval_window_ids = set(oracle_eval["window_id"])
    near_tie_eval = near_tie_df[near_tie_df["window_id"].isin(eval_window_ids)] if not near_tie_df.empty else near_tie_df
    completion_eval = completion_df[completion_df["window_id"].isin(eval_window_ids)] if not completion_df.empty else completion_df

    primary_threshold = float(config.get("near_tie_primary_threshold", 0.005))
    primary_rows = near_tie_eval[near_tie_eval["threshold"] == primary_threshold] if not near_tie_eval.empty else near_tie_eval
    near_tie_fraction = float(primary_rows["near_tie"].mean()) if not primary_rows.empty else None
    completion_ok = bool(completion_eval["completion_ok"].all()) if not completion_eval.empty else False
    non_composition_families = {"fixed_policy", "cc1b_borda_baseline"}
    composition_oracle = oracle_eval[~oracle_eval["oracle_family"].isin(non_composition_families)]
    gain_windows = len(composition_oracle)
    if not completion_ok:
        status = "IN_PROGRESS"
        reason = "completion-fraction constraint failed on at least one evaluation-split window"
    elif near_tie_fraction is not None and near_tie_fraction < 1.0 and gain_windows > 0:
        status = "COMPLETE"
        reason = "reproducible, resumable dataset with sufficient non-near-tie evaluation-split signal for CC5"
    else:
        status = "IN_PROGRESS"
        reason = "insufficient non-near-tie composition-selected evaluation-split windows for a confident CC5 entry"
    return {
        "status": status,
        "reason": reason,
        "oracle_composition_gain": gain_windows / len(oracle_eval) if len(oracle_eval) else None,
        "near_tie_fraction": near_tie_fraction,
        "n_evaluation_windows": len(oracle_eval),
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


@dataclass
class CC4Result:
    output_dir: Path
    manifest: dict[str, Any]
    verdict: dict[str, Any]


def run_search(
    config: Mapping[str, Any],
    *,
    config_path: str | Path,
    dry_run: bool = False,
    full_run: bool = False,
    max_runs: int | None = None,
    allow_dirty: bool = False,
    timestamp: str | None = None,
    resume_dir: str | Path | None = None,
    runner: Callable[..., RunMetrics] = run_policy,
    heartbeat_every: int = 10,
) -> CC4Result:
    validate_config(config)
    windows, skipped_traces = build_workload_windows(config)
    candidates = generate_all_candidates(config)
    _, cloudrift_info = maybe_generate_cloudrift_candidates(config)

    planned = len(windows) * len(candidates)
    cap = int(max_runs if max_runs is not None else config["safeguards"].get("max_runs", 0))
    if cap <= 0:
        raise CC4Error("max_runs must be positive")
    if planned > cap:
        raise CC4Error(f"planned run count {planned} exceeds max_runs {cap}")

    if dry_run:
        return CC4Result(
            output_dir=Path(""),
            manifest={
                "dry_run": True,
                "window_count": len(windows),
                "candidate_count": len(candidates),
                "planned_run_count": planned,
                "skipped_real_traces": skipped_traces,
            },
            verdict={"status": "INCONCLUSIVE", "reason": "dry run"},
        )

    git = git_state()
    dirty = bool(git["dirty"])
    needs_full_flag = not full_run
    if needs_full_flag:
        raise CC4Error("cc4 full runs require explicit --full-run")
    if dirty and not allow_dirty:
        raise CC4Error("non-dry CC4 runs require a clean git worktree unless --allow-dirty is explicitly set")

    resumed = resume_dir is not None
    output_dir = Path(resume_dir) if resume_dir is not None else resolve_output_dir(config, timestamp=timestamp)
    if resume_dir is None:
        output_dir.mkdir(parents=True, exist_ok=False)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
    resolved_config = json.loads(json.dumps(config))
    (output_dir / "resolved_config.yaml").write_text(yaml.safe_dump(resolved_config, sort_keys=False))

    store = CC4TrialStore(output_dir)
    t0 = time.perf_counter()
    gpu_configs = build_gpu_configs(config)
    service_model = build_service_model(config)
    drain_steps = int(config.get("simulator", {}).get("drain_steps", 5000))

    n_new = 0
    total = len(windows) * len(candidates)
    done = 0
    for window in windows:
        for candidate in candidates:
            key = CC4TrialStore.trial_key(window.window_id, candidate.candidate_id)
            done += 1
            if key in store.completed_keys:
                continue
            row = execute_candidate_row(
                runner, candidate, window, gpu_configs, service_model,
                drain_steps=drain_steps, git_sha=git["commit"],
            )
            store.append(row)
            n_new += 1
            if n_new % heartbeat_every == 0:
                heartbeat(output_dir, "searching", progress=f"{done}/{total}", n_new_this_run=n_new)

    heartbeat(output_dir, "post_processing", progress=f"{total}/{total}", n_new_this_run=n_new)
    all_rows = store.load_all_rows()
    rows_df = pd.DataFrame(all_rows)
    rejected = [r for r in all_rows if not str(r["verification_outcome"]).startswith("valid") and str(r["verification_outcome"]) != "not_applicable_non_dsl" and not r["true_simulator_executed"]]

    oracle_df = compute_oracle_labels(rows_df)
    regret_df = compute_regret_matrix(rows_df, oracle_df)
    near_tie_df = compute_near_tie_flags(oracle_df, config.get("near_tie_thresholds", [0.001, 0.005, 0.01]))
    completion_df = compute_completion_constraints(
        rows_df, oracle_df, float(config["metrics"].get("completion_fraction_tolerance", 0.005))
    )
    primitive_stats_df = compute_primitive_usage_statistics(candidates, oracle_df)
    search_summary_df = compute_search_summary(candidates, rejected, rows_df, windows, skipped_traces)
    causal_features_df = build_causal_features(windows)
    windows_df = build_workload_windows_table(windows)
    compositions_df = build_candidate_compositions_table(candidates, rejected)
    parameters_df = build_composition_parameters_table(candidates)

    verdict = determine_dataset_verdict(oracle_df, near_tie_df, completion_df, config)

    for name, df in (
        ("workload_windows", windows_df),
        ("causal_features", causal_features_df),
        ("candidate_compositions", compositions_df),
        ("per_window_results", rows_df),
        ("oracle_labels", oracle_df),
        ("regret_matrix", regret_df),
        ("composition_parameters", parameters_df),
        ("near_tie_flags", near_tie_df),
        ("completion_constraints", completion_df),
    ):
        df.to_parquet(output_dir / f"{name}.parquet", index=False)
    primitive_stats_df.to_csv(output_dir / "primitive_usage_statistics.csv", index=False)
    search_summary_df.to_csv(output_dir / "search_summary.csv", index=False)

    manifest = build_manifest(
        config, config_path=config_path, output_dir=output_dir, git=git, windows=windows,
        skipped_traces=skipped_traces, candidates=candidates, rejected=rejected,
        n_executed=int(_executed_only(rows_df).shape[0]), runtime_s=time.perf_counter() - t0,
        resumed=resumed, cloudrift_info=cloudrift_info,
    )
    manifest["verdict"] = verdict
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
    (output_dir / "dataset_card.md").write_text(render_dataset_card(manifest, search_summary_df, verdict))
    (output_dir / "replay_commands.sh").write_text(render_replay_commands(config_path, output_dir))
    heartbeat(output_dir, "complete", progress=f"{total}/{total}", n_new_this_run=n_new)

    return CC4Result(output_dir=output_dir, manifest=manifest, verdict=verdict)
