"""Loader for the official, unmodified VTC scheduling classes.

Does NOT vendor the official repository's source into this project. The
``VTCReqQueue``, ``ReqQueue``, ``Req``, ``Batch``, and ``SamplingParams``
classes are dynamically imported, at runtime, directly from the pinned
local clone (see ``provenance.DEFAULT_OFFICIAL_CLONE_PATH``) -- the exact
same class definitions the official S-LoRA-based server uses, never
copy-pasted into this file.

The official files use ordinary intra-package relative imports (e.g.
``from ..io_struct import Batch, Req`` inside ``vtc_req_queue.py``), so a
minimal synthetic package hierarchy (``slora``, ``slora.server``,
``slora.server.router``, ``slora.utils``) is registered in
``sys.modules`` pointing at the clone's real directories before each
submodule is exec'd -- this lets Python's relative-import machinery
resolve correctly without needing the clone's compiled CUDA extension
(``slora._kernels``) or any package files this adapter doesn't need.
Verified to work with zero GPU/CUDA/Triton dependency: see PROVENANCE.md's
"Why this blocker does not block a faithful integration" section.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import types
from dataclasses import dataclass
from typing import Optional

from . import provenance
from .errors import MissingOfficialCloneError, StaleCloneCommitError


def resolve_official_clone_path() -> str:
    return os.environ.get("VTC_OFFICIAL_CLONE_PATH", provenance.DEFAULT_OFFICIAL_CLONE_PATH)


def verify_official_clone(clone_path: Optional[str] = None) -> str:
    """Locate the pinned local clone and verify its HEAD commit matches
    ``provenance.PINNED_COMMIT``. Raises rather than silently using an
    unpinned/drifted copy of the official code."""
    path = clone_path or resolve_official_clone_path()
    if not os.path.isdir(os.path.join(path, ".git")):
        raise MissingOfficialCloneError(
            f"No git clone of {provenance.OFFICIAL_REPOSITORY} found at {path!r}. "
            f"Clone it first: git clone {provenance.OFFICIAL_REPOSITORY} {path}"
        )
    head = subprocess.run(
        ["git", "-C", path, "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    if head != provenance.PINNED_COMMIT:
        raise StaleCloneCommitError(
            f"Local clone at {path!r} is at commit {head!r}, expected the "
            f"pinned commit {provenance.PINNED_COMMIT!r}. Re-clone or "
            f"`git checkout {provenance.PINNED_COMMIT}` before using this adapter."
        )
    for rel in provenance.CORE_POLICY_FILES:
        if not os.path.exists(os.path.join(path, rel)):
            raise MissingOfficialCloneError(
                f"Expected official file not found: {os.path.join(path, rel)}. "
                "Clone may be incomplete or the official repository's layout has changed."
            )
    return path


def _register_stub_package(name: str, directory: str) -> None:
    if name in sys.modules:
        return
    pkg = types.ModuleType(name)
    pkg.__path__ = [directory]
    sys.modules[name] = pkg


def _exec_module(module_name: str, file_path: str):
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class VTCOfficialClasses:
    VTCReqQueue: type
    ReqQueue: type
    Req: type
    Batch: type
    SamplingParams: type
    clone_path: str


def load_vtc_official_classes(clone_path: Optional[str] = None) -> VTCOfficialClasses:
    """Import the real, unmodified VTC scheduling classes from the pinned
    clone. Safe to call repeatedly (subsequent calls reuse sys.modules)."""
    path = verify_official_clone(clone_path)

    _register_stub_package("slora", os.path.join(path, "slora"))
    _register_stub_package("slora.server", os.path.join(path, "slora", "server"))
    _register_stub_package("slora.server.router", os.path.join(path, "slora", "server", "router"))
    _register_stub_package("slora.utils", os.path.join(path, "slora", "utils"))

    sampling_params_mod = _exec_module(
        "slora.server.sampling_params", os.path.join(path, "slora", "server", "sampling_params.py")
    )
    io_struct_mod = _exec_module(
        "slora.server.io_struct", os.path.join(path, "slora", "server", "io_struct.py")
    )
    _exec_module("slora.utils.infer_utils", os.path.join(path, "slora", "utils", "infer_utils.py"))
    req_queue_mod = _exec_module(
        "slora.server.router.req_queue", os.path.join(path, "slora", "server", "router", "req_queue.py")
    )
    vtc_req_queue_mod = _exec_module(
        "slora.server.router.vtc_req_queue",
        os.path.join(path, "slora", "server", "router", "vtc_req_queue.py"),
    )

    return VTCOfficialClasses(
        VTCReqQueue=vtc_req_queue_mod.VTCReqQueue,
        ReqQueue=req_queue_mod.ReqQueue,
        Req=io_struct_mod.Req,
        Batch=io_struct_mod.Batch,
        SamplingParams=sampling_params_mod.SamplingParams,
        clone_path=path,
    )
