#!/usr/bin/env python3
"""Apt-Serve Strategy C import/separability probe (Phase A).

Runs entirely on Wulver, against a pinned Apt-Serve checkout applied to
an isolated vLLM 0.5.0.post1 install -- never against this project's
own source tree. Contains no Apt-Serve source of its own (the upstream
repo has no LICENSE file; nothing from it is vendored or embedded here
-- see docs/audits/apt_serve_official_artifact_audit_20260805.md section
1's license disclosure).

Answers, in order, the exact questions
docs/audits/apt_serve_strategy_c_wulver_probe_20260806.md section 5
requires:
  1. Does `vllm.core.scheduler` (vanilla, pre-replacement) import?
  2. After applying ONLY the scheduler-critical file subset (not the
     full 13-file insert_designs.sh set -- see rationale in the audit
     doc's "minimum official source subset" discussion), do the
     Apt-Serve scheduler/block-manager/sequence/block/interfaces
     modules import?
  3. Is CUDA touched merely by importing (not constructing/running)
     these modules?
  4. Does `Scheduler.__init__` succeed with synthetic, GPU-free
     SchedulerConfig/CacheConfig/LoRAConfig objects and a synthetic (or
     real, if it doesn't itself need a GPU) AptServeBlockManager?
  5. Do model weights need to be loaded for any of the above?
  6. Is Ray required?
  7. Are custom CUDA kernels (mixed_cache_ops) required for scheduler
     construction/scheduling decisions specifically (as opposed to
     actual token generation, which is out of scope for this probe)?

Every phase is wrapped so a failure in a later phase does not prevent
earlier phases' results from being recorded. Writes a single JSON
report to the path given by --output.

Usage (run inside the pinned/patched environment on Wulver):
    python apt_serve_import_probe.py --output /path/to/import_probe_report.json
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime, timezone


def _try(label: str, fn) -> dict:
    """Run fn(), catching everything, recording success/failure/exception
    without letting one phase's failure abort the rest of the probe."""
    entry = {"label": label}
    try:
        result = fn()
        entry["status"] = "OK"
        if result is not None:
            entry["detail"] = result
    except Exception as e:  # noqa: BLE001 -- deliberately broad, this IS the probe
        entry["status"] = "FAILED"
        entry["exception_type"] = type(e).__name__
        entry["exception_message"] = str(e)
        entry["traceback"] = traceback.format_exc()
    return entry


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--phase", choices=["vanilla", "patched", "both"], default="both",
                         help="'vanilla' = before Apt-Serve file replacement, "
                              "'patched' = after -- run as two separate job steps "
                              "in different Python processes so a hard crash in "
                              "one phase cannot corrupt the other's already-recorded results.")
    args = parser.parse_args()

    report = {
        "probe": "apt_serve_strategy_c_import_probe",
        "phase_requested": args.phase,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "results": [],
    }

    def record(label, fn):
        report["results"].append(_try(label, fn))

    # --- Phase 0: environment fingerprint (always safe) ---
    def _env_fingerprint():
        import torch
        return {
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
    record("env_fingerprint_torch", _env_fingerprint)

    # --- Phase 1: vanilla vLLM import (before any Apt-Serve file replacement) ---
    if args.phase in ("vanilla", "both"):
        record("import_vllm_bare", lambda: __import__("vllm").__version__)
        record("import_vllm_core_scheduler", lambda: bool(__import__(
            "vllm.core.scheduler", fromlist=["Scheduler"]).Scheduler))
        record("import_vllm_sequence", lambda: bool(__import__(
            "vllm.sequence", fromlist=["SequenceGroup"]).SequenceGroup))
        record("import_vllm_core_interfaces", lambda: bool(__import__(
            "vllm.core.interfaces", fromlist=["BlockSpaceManager"]).BlockSpaceManager))
        record("import_vllm_config", lambda: bool(__import__(
            "vllm.config", fromlist=["SchedulerConfig"]).SchedulerConfig))
        record("import_vllm_lora_request", lambda: bool(__import__(
            "vllm.lora.request", fromlist=["LoRARequest"]).LoRARequest))
        record("import_vllm_core_policy", lambda: bool(__import__(
            "vllm.core.policy", fromlist=["Policy"]).Policy))

        def _vanilla_scheduler_construct():
            from vllm.config import CacheConfig, SchedulerConfig
            from vllm.core.scheduler import Scheduler
            sched_cfg = SchedulerConfig(
                max_num_batched_tokens=2048, max_num_seqs=16, max_model_len=2048,
            )
            cache_cfg = CacheConfig(block_size=16, gpu_memory_utilization=0.9,
                                     swap_space=4, cache_dtype="auto")
            cache_cfg.num_gpu_blocks = 100
            cache_cfg.num_cpu_blocks = 100
            sched = Scheduler(sched_cfg, cache_cfg, lora_config=None)
            return {"scheduler_repr": repr(type(sched))}
        record("vanilla_scheduler_construct_synthetic_config", _vanilla_scheduler_construct)

    # --- Phase 2: Apt-Serve-patched import (after minimal-subset file replacement --
    #     see the SBATCH driver script for exactly which files were copied over
    #     which vLLM package files before this process was launched) ---
    if args.phase in ("patched", "both"):
        record("import_ray", lambda: __import__("ray").__version__)

        def _patched_scheduler_import():
            import importlib
            import vllm.core.scheduler as sched_mod
            importlib.reload(sched_mod)  # ensure we're not seeing a cached vanilla import
            return {"scheduler_module_file": sched_mod.__file__,
                    "has_greedy_selection_prefill": hasattr(sched_mod.Scheduler, "greedy_selection_prefill"),
                    "has_greedy_selection_decode": hasattr(sched_mod.Scheduler, "greedy_selection_decode"),
                    "has_dynamic_priority": hasattr(sched_mod.Scheduler, "_dynamic_priority")}
        record("import_patched_scheduler_module", _patched_scheduler_import)

        def _patched_block_manager_import():
            import vllm.core.block_manager_v1 as bm_mod
            return {"block_manager_module_file": bm_mod.__file__}
        record("import_patched_block_manager", _patched_block_manager_import)

        def _patched_sequence_import():
            import vllm.sequence as seq_mod
            return {"sequence_module_file": seq_mod.__file__,
                     "sequence_group_has_use_hidden": hasattr(seq_mod.SequenceGroup, "set_use_hidden")
                     or hasattr(seq_mod.SequenceGroup, "use_hidden")}
        record("import_patched_sequence", _patched_sequence_import)

        def _patched_block_import():
            import vllm.block as block_mod
            return {"block_module_file": block_mod.__file__}
        record("import_patched_block", _patched_block_import)

        # Construction with SYNTHETIC objects only -- no real model, no real GPU
        # memory allocation, no engine lifecycle. This directly answers task
        # step 6's "scheduler initialization" / "does construction succeed
        # with synthetic objects" question.
        def _patched_scheduler_construct():
            from vllm.config import CacheConfig, SchedulerConfig
            from vllm.core.scheduler import Scheduler
            sched_cfg = SchedulerConfig(
                max_num_batched_tokens=2048, max_num_seqs=16, max_model_len=2048,
            )
            cache_cfg = CacheConfig(block_size=16, gpu_memory_utilization=0.9,
                                     swap_space=4, cache_dtype="auto")
            cache_cfg.num_gpu_blocks = 100
            cache_cfg.num_cpu_blocks = 100
            sched = Scheduler(sched_cfg, cache_cfg, lora_config=None)
            return {"scheduler_repr": repr(type(sched)),
                    "block_manager_type": repr(type(sched.block_manager))}
        record("patched_scheduler_construct_synthetic_config", _patched_scheduler_construct)

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2, default=str)

    n_ok = sum(1 for r in report["results"] if r["status"] == "OK")
    n_failed = sum(1 for r in report["results"] if r["status"] == "FAILED")
    print(f"probe complete: {n_ok} OK, {n_failed} FAILED. Report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
