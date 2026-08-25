#!/usr/bin/env python3
"""Apt-Serve Strategy C micro-trace differential (Phase A, step 8).

Only runs if apt_serve_import_probe.py's "patched_scheduler_construct_synthetic_config"
phase reported OK -- this script assumes a working, Apt-Serve-patched
`vllm.core.scheduler.Scheduler` is importable and constructible with
synthetic config objects, and goes one step further: feeding synthetic
Sequence/SequenceGroup objects through it to record real scheduling
decisions (selected batch, cache-type assignment, deprioritized/rejected
requests, queue state) as a compact, reproducible JSON trace.

Contains no Apt-Serve source (see apt_serve_import_probe.py's module
docstring for the license/no-vendoring rationale -- identical here).

This script is DEFENSIVE BY DESIGN: vLLM 0.5.0.post1's exact
`Sequence`/`SequenceGroup` constructor signatures were not verified
against a local install before this script was written (no working
vLLM install exists on the local workstation -- see
docs/audits/apt_serve_official_artifact_audit_20260805.md section 9).
Every construction attempt is wrapped so a wrong assumption produces a
diagnostic entry in the report rather than a silent crash that loses
everything gathered so far. If a scenario fails, the introspected
constructor signature is recorded so a human can adjust the scenario
inputs without re-running the whole probe from scratch.

Usage (run inside the pinned/patched environment on Wulver, only after
the import probe's patched-construction phase succeeds):
    python apt_serve_micro_trace.py --output /path/to/micro_trace_report.json \
        --commit c953217988274a761da35cf06c01033b18dadf68 --seed 20260806
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import subprocess
import sys
from datetime import datetime, timezone


def _introspect(obj) -> dict:
    try:
        sig = str(inspect.signature(obj))
    except (TypeError, ValueError) as e:
        sig = f"<introspection failed: {e}>"
    return {"qualname": getattr(obj, "__qualname__", repr(obj)), "signature": sig}


def _environment_hash() -> str:
    """Cheap, reproducible fingerprint of the installed package versions
    relevant to this probe -- NOT a full lockfile, just enough to detect
    'this trace was produced under a different environment than expected'
    on a later re-run."""
    parts = []
    for mod_name in ("torch", "vllm", "xformers"):
        try:
            mod = __import__(mod_name)
            parts.append(f"{mod_name}={getattr(mod, '__version__', 'unknown')}")
        except Exception as e:  # noqa: BLE001
            parts.append(f"{mod_name}=IMPORT_FAILED:{e}")
    fingerprint = "|".join(parts)
    return hashlib.sha256(fingerprint.encode()).hexdigest()[:16] + " (" + fingerprint + ")"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--commit", required=True, help="Pinned Apt-Serve commit SHA this trace is against")
    parser.add_argument("--seed", type=int, default=20260806)
    args = parser.parse_args()

    report = {
        "probe": "apt_serve_strategy_c_micro_trace",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "apt_serve_commit": args.commit,
        "environment_hash": _environment_hash(),
        "seed": args.seed,
        "command": " ".join(sys.argv),
        "hostname": subprocess.run(["hostname"], capture_output=True, text=True).stdout.strip(),
        "introspection": {},
        "scenarios": [],
    }

    # --- Introspect constructor signatures first (always safe, always useful) ---
    try:
        from vllm.sequence import Sequence, SequenceGroup, SequenceData
        report["introspection"]["Sequence.__init__"] = _introspect(Sequence.__init__)
        report["introspection"]["SequenceGroup.__init__"] = _introspect(SequenceGroup.__init__)
        report["introspection"]["SequenceData.__init__"] = _introspect(SequenceData.__init__)
    except Exception as e:  # noqa: BLE001
        report["introspection"]["error"] = f"{type(e).__name__}: {e}"
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print("FATAL: could not import vllm.sequence classes for introspection; "
              "wrote partial report and stopping (nothing else in this script "
              "can proceed without these classes).")
        return 1

    from vllm.config import CacheConfig, SchedulerConfig
    from vllm.core.scheduler import Scheduler

    def _build_scheduler(max_num_seqs=16, max_num_batched_tokens=2048, num_gpu_blocks=64):
        sched_cfg = SchedulerConfig(
            max_num_batched_tokens=max_num_batched_tokens, max_num_seqs=max_num_seqs,
            max_model_len=2048,
        )
        cache_cfg = CacheConfig(block_size=16, gpu_memory_utilization=0.9,
                                 swap_space=4, cache_dtype="auto")
        cache_cfg.num_gpu_blocks = num_gpu_blocks
        cache_cfg.num_cpu_blocks = num_gpu_blocks
        return Scheduler(sched_cfg, cache_cfg, lora_config=None)

    def _try_build_seq_group(request_id: str, prompt_len: int, block_size: int = 16):
        """Synthetic SequenceGroup construction, corrected against job
        1163782's real introspection output (--commit c953217988's pinned
        vLLM 0.5.0.post1 build): `Sequence.__init__` takes an `inputs:
        LLMInputs` TypedDict (`{"prompt_token_ids": [...], "prompt": ...}`),
        not separate `prompt`/`prompt_token_ids` kwargs -- the original
        best-effort guess (a plausible but wrong assumption about an
        earlier vLLM minor version's shape) failed with `TypeError:
        Sequence.__init__() got an unexpected keyword argument 'prompt'`
        on the first run; this is the exact fix, not a re-guess."""
        from vllm.inputs import LLMInputs
        from vllm.sampling_params import SamplingParams
        from vllm.sequence import Sequence, SequenceGroup
        prompt_token_ids = list(range(prompt_len))
        inputs = LLMInputs(prompt_token_ids=prompt_token_ids, prompt="x" * prompt_len)
        seq = Sequence(seq_id=hash(request_id) & 0xFFFFFFF, inputs=inputs, block_size=block_size)
        seq_group = SequenceGroup(
            request_id=request_id, seqs=[seq], arrival_time=0.0,
            sampling_params=SamplingParams(max_tokens=16),
        )
        return seq_group

    def _run_scenario(name: str, requests: list) -> dict:
        """requests: list of (request_id, prompt_len) tuples, all admitted
        at once via add_seq_group, then a single schedule() call recorded."""
        scenario = {"name": name, "requests": [{"request_id": r[0], "prompt_len": r[1]} for r in requests]}
        try:
            sched = _build_scheduler()
            for rid, plen in requests:
                seq_group = _try_build_seq_group(rid, plen)
                sched.add_seq_group(seq_group)
            seq_group_metas, outputs = sched.schedule()
            scenario["status"] = "OK"
            scenario["scheduled_request_ids"] = [m.request_id for m in seq_group_metas]
            scenario["num_scheduled"] = len(seq_group_metas)
            scenario["waiting_after"] = len(sched.waiting)
            scenario["running_after"] = len(sched.running)
        except Exception as e:  # noqa: BLE001
            scenario["status"] = "FAILED"
            scenario["exception_type"] = type(e).__name__
            scenario["exception_message"] = str(e)
        return scenario

    # --- Hand-constructed micro-scenarios (task step 8's own list) ---
    scenarios_to_run = [
        ("three_requests_two_fit_memory_budget",
         [("req-a", 200), ("req-b", 200), ("req-c", 4000)]),
        ("homogeneous_low_contention",
         [(f"req-{i}", 100) for i in range(4)]),
        ("single_oversized_request_extreme_case",
         [("req-small-1", 50), ("req-small-2", 50), ("req-huge", 8000)]),
    ]
    for name, requests in scenarios_to_run:
        report["scenarios"].append(_run_scenario(name, requests))

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2, default=str)

    n_ok = sum(1 for s in report["scenarios"] if s["status"] == "OK")
    print(f"micro-trace complete: {n_ok}/{len(report['scenarios'])} scenarios OK. Report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
