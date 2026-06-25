#!/usr/bin/env python3
"""
Inspect and save the GPU/software environment.

By default, writes results/gpu_calibration/environment.json and
docs/gpu_environment.md (override with --json-output/--md-output, or
skip writing entirely with --dry-run). Does NOT expose environment
variables, tokens, or secrets.

Usage:
    python scripts/inspect_gpu_environment.py
    python scripts/inspect_gpu_environment.py --dry-run
    python scripts/inspect_gpu_environment.py --json-output /tmp/env.json --md-output /tmp/env.md
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
DEFAULT_JSON_OUTPUT = ROOT / "results" / "gpu_calibration" / "environment.json"
DEFAULT_MD_OUTPUT = ROOT / "docs" / "gpu_environment.md"


def _run(cmd: list[str], timeout: int = 15) -> str:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return result.stdout.strip()
    except Exception as e:
        return f"(error: {e})"


def _gpu_info() -> list[dict]:
    try:
        import torch

        gpus = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpus.append(
                {
                    "index": i,
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / (1024**3), 2),
                    "multi_processor_count": props.multi_processor_count,
                    "major": props.major,
                    "minor": props.minor,
                }
            )
        return gpus
    except Exception:
        return []


def _cuda_version() -> str:
    try:
        import torch
        return torch.version.cuda or "unavailable"
    except Exception:
        return "unavailable"


def _torch_version() -> str:
    try:
        import torch
        return torch.__version__
    except Exception:
        return "unavailable"


def _transformers_version() -> str:
    try:
        import transformers
        return transformers.__version__
    except Exception:
        return "unavailable"


def _accelerate_version() -> str:
    try:
        import accelerate
        return accelerate.__version__
    except Exception:
        return "unavailable"


def _vllm_version() -> str | None:
    try:
        import vllm
        return vllm.__version__
    except Exception:
        return None


def _driver_version() -> str:
    raw = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    if raw.startswith("(error"):
        return "unavailable"
    return raw.strip().split("\n")[0]


def _disk_free_gb() -> float:
    try:
        usage = shutil.disk_usage("/")
        return round(usage.free / (1024**3), 1)
    except Exception:
        return -1.0


def _ram_stats() -> tuple[float, float]:
    try:
        import psutil
        vm = psutil.virtual_memory()
        return round(vm.total / (1024**3), 1), round(vm.available / (1024**3), 1)
    except ImportError:
        try:
            with open("/proc/meminfo") as f:
                lines = f.read()
            total_kb = int([l for l in lines.splitlines() if l.startswith("MemTotal")][0].split()[1])
            avail_kb = int([l for l in lines.splitlines() if l.startswith("MemAvailable")][0].split()[1])
            return round(total_kb / (1024**2), 1), round(avail_kb / (1024**2), 1)
        except Exception:
            return -1.0, -1.0


def _hf_cache_info() -> tuple[str, list[str]]:
    hf_home = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    cache_dir = Path(hf_home) / "hub"
    models = []
    if cache_dir.exists():
        for child in cache_dir.iterdir():
            if child.is_dir():
                models.append(child.name)
    return hf_home, sorted(models)


def collect_environment() -> dict:
    gpus = _gpu_info()
    ram_total, ram_free = _ram_stats()
    hf_cache_dir, hf_models = _hf_cache_info()
    return {
        "gpu": gpus,
        "nvidia_smi": _run(["nvidia-smi"]),
        "cuda_version": _cuda_version(),
        "driver_version": _driver_version(),
        "torch_version": _torch_version(),
        "python_version": platform.python_version(),
        "transformers_version": _transformers_version(),
        "accelerate_version": _accelerate_version(),
        "vllm_version": _vllm_version(),
        "disk_free_gb": _disk_free_gb(),
        "ram_total_gb": ram_total,
        "ram_free_gb": ram_free,
        "hf_cache_dir": hf_cache_dir,
        "hf_cached_models": hf_models,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def render_markdown(env: dict) -> str:
    lines = [
        "# GPU Environment",
        "",
        f"Generated: {env['timestamp']}",
        "",
        "## Hardware",
        "",
    ]
    for g in env["gpu"]:
        lines.append(
            f"- GPU {g['index']}: **{g['name']}** — {g['total_memory_gb']} GB VRAM, "
            f"{g['multi_processor_count']} SMs, compute {g['major']}.{g['minor']}"
        )
    lines += [
        "",
        f"- Driver version: {env['driver_version']}",
        f"- CUDA version: {env['cuda_version']}",
        f"- RAM total: {env['ram_total_gb']} GB, free: {env['ram_free_gb']} GB",
        f"- Disk free: {env['disk_free_gb']} GB",
        "",
        "## Software",
        "",
        f"- Python: {env['python_version']}",
        f"- PyTorch: {env['torch_version']}",
        f"- Transformers: {env['transformers_version']}",
        f"- Accelerate: {env['accelerate_version']}",
        f"- vLLM: {env['vllm_version'] or 'not installed'}",
        "",
        "## HuggingFace Cache",
        "",
        f"- Cache dir: `{env['hf_cache_dir']}`",
        f"- Cached models ({len(env['hf_cached_models'])}):",
    ]
    for m in env["hf_cached_models"]:
        lines.append(f"  - {m}")
    lines += ["", "## nvidia-smi", "", "```", env["nvidia_smi"], "```", ""]
    return "\n".join(lines)


def print_summary(env: dict) -> None:
    print("\n=== GPU Environment Summary ===")
    for g in env["gpu"]:
        print(f"  GPU {g['index']}: {g['name']} ({g['total_memory_gb']} GB)")
    print(f"  CUDA: {env['cuda_version']}, Driver: {env['driver_version']}")
    print(f"  PyTorch: {env['torch_version']}")
    print(f"  Transformers: {env['transformers_version']}")
    print(f"  vLLM: {env['vllm_version'] or 'NOT installed'}")
    print(f"  Disk free: {env['disk_free_gb']} GB, RAM free: {env['ram_free_gb']} GB")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect and save the GPU/software environment. "
            "Writes a JSON report and a markdown doc by default; "
            "use --dry-run to only print, without writing any file."
        )
    )
    parser.add_argument(
        "--json-output", type=Path, default=DEFAULT_JSON_OUTPUT,
        help=f"Path to write the environment JSON report (default: {DEFAULT_JSON_OUTPUT}).",
    )
    parser.add_argument(
        "--md-output", type=Path, default=DEFAULT_MD_OUTPUT,
        help=f"Path to write the environment markdown doc (default: {DEFAULT_MD_OUTPUT}).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Collect and print environment info without writing any file.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    env = collect_environment()

    if args.dry_run:
        print(json.dumps(env, indent=2))
        print_summary(env)
        print(f"\n[dry-run] no files written (would have written {args.json_output} and {args.md_output})")
        return 0

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.json_output, "w") as f:
        json.dump(env, f, indent=2)
    print(f"Saved: {args.json_output}")

    args.md_output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.md_output, "w") as f:
        f.write(render_markdown(env))
    print(f"Saved: {args.md_output}")

    print_summary(env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
