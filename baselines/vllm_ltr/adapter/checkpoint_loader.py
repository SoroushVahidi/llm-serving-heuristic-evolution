"""Loader for the official vLLM-LTR OPT predictor checkpoint.

Key finding this loader relies on (verified 2026-08-04 by diffing the pinned
commit's ``vllm/model_executor/models/opt.py::OPTForSequenceClassification``
against HuggingFace ``transformers``' own ``OPTForSequenceClassification``,
v4.30.0): the two classes are field-for-field identical --
``self.model = OPTModel(config)`` + ``self.score =
nn.Linear(config.word_embed_proj_dim, config.num_labels, bias=False)``, no
pooling/MLP/dropout added. The vLLM fork's custom class exists only to plug
into vLLM's paged-attention/logits-processor plumbing for serving
efficiency; it is not architecturally distinct from stock HF
``OPTForSequenceClassification``.

CONFIRMED 2026-08-04 against the real checkpoint (see
``docs/audits/vllm_ltr_baseline_audit_20260804.md`` and
``CHECKPOINT_PROVENANCE.md``): the official checkpoint's own
``config.json`` records ``"architectures": ["OPTForSequenceClassification"]``
and loads cleanly through plain ``transformers.AutoModelForSequenceClassification``
with no missing/unexpected state-dict keys.

torch/transformers are optional dependencies here (see the ``vllm_ltr``
extra in ``pyproject.toml``); the core simulator has no hard dependency on
either.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from . import provenance
from .errors import (
    MissingCheckpointError,
    MissingDependencyError,
    StaleArtifactError,
    VersionMismatchError,
)

PROVENANCE_SIDECAR_FILENAME = "vllm_ltr_provenance.json"


def _require_torch_and_transformers():
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise MissingDependencyError(
            "torch is required to load the vLLM-LTR OPT predictor but is not "
            "installed. Install the optional extra: pip install -e "
            "'.[vllm_ltr]' (or: pip install torch transformers)."
        ) from exc
    try:
        import transformers  # noqa: F401
    except ImportError as exc:
        raise MissingDependencyError(
            "transformers is required to load the vLLM-LTR OPT predictor but "
            "is not installed. Install the optional extra: pip install -e "
            "'.[vllm_ltr]' (or: pip install torch transformers)."
        ) from exc
    return torch, transformers


def _major_minor(version_str: str) -> str:
    parts = version_str.split(".")
    return ".".join(parts[:2])


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_provenance_sidecar(checkpoint_dir: str) -> dict:
    sidecar_path = os.path.join(checkpoint_dir, PROVENANCE_SIDECAR_FILENAME)
    if not os.path.isfile(sidecar_path):
        raise StaleArtifactError(
            f"Checkpoint directory {checkpoint_dir!r} has no "
            f"{PROVENANCE_SIDECAR_FILENAME} sidecar recording which pinned "
            "vllm-ltr commit and library versions it was verified against. "
            "Refusing to load an unprovenanced artifact -- write one with "
            "write_local_provenance_sidecar() first."
        )
    with open(sidecar_path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_local_provenance_sidecar(
    checkpoint_dir: str,
    *,
    pinned_commit: str,
    verified_environments: Sequence[Dict[str, str]],
    checkpoint_repo_id: Optional[str] = None,
    checkpoint_revision: Optional[str] = None,
    checkpoint_subfolder: Optional[str] = None,
    file_hashes: Optional[Dict[str, str]] = None,
) -> str:
    """Write a ``vllm_ltr_provenance.json`` sidecar into ``checkpoint_dir``.

    ``verified_environments`` is a list of ``{"torch_version": ...,
    "transformers_version": ...}`` dicts -- every environment (major.minor)
    this checkpoint has actually been confirmed to load and score correctly
    under. A checkpoint typically has at least two: the original export
    environment (recorded in its own HF ``config.json``'s
    ``transformers_version`` field) and whatever environment re-verified it
    here. The loader accepts the currently-installed torch/transformers if
    their major.minor matches ANY entry -- see ``_validate_provenance_sidecar``.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    sidecar_path = os.path.join(checkpoint_dir, PROVENANCE_SIDECAR_FILENAME)
    payload = {
        "pinned_commit": pinned_commit,
        "verified_environments": list(verified_environments),
        "checkpoint_repo_id": checkpoint_repo_id,
        "checkpoint_revision": checkpoint_revision,
        "checkpoint_subfolder": checkpoint_subfolder,
        "file_hashes": file_hashes or {},
    }
    with open(sidecar_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return sidecar_path


def _validate_provenance_sidecar(sidecar: dict, torch_mod, transformers_mod) -> None:
    recorded_commit = sidecar.get("pinned_commit")
    if recorded_commit != provenance.PINNED_COMMIT:
        raise StaleArtifactError(
            f"Checkpoint sidecar records pinned_commit={recorded_commit!r}, "
            f"but this adapter is pinned to {provenance.PINNED_COMMIT!r}. "
            "Refusing to load a checkpoint verified against a different "
            "vllm-ltr commit."
        )
    verified_environments = sidecar.get("verified_environments") or []
    if not verified_environments:
        raise StaleArtifactError(
            "Checkpoint sidecar has an empty verified_environments list -- "
            "refusing to load a checkpoint with no recorded verification."
        )
    installed_torch = torch_mod.__version__
    installed_transformers = transformers_mod.__version__
    installed_torch_mm = _major_minor(installed_torch)
    installed_transformers_mm = _major_minor(installed_transformers)
    for env in verified_environments:
        if (
            _major_minor(env.get("torch_version", "")) == installed_torch_mm
            and _major_minor(env.get("transformers_version", "")) == installed_transformers_mm
        ):
            return
    raise VersionMismatchError(
        f"Installed torch=={installed_torch!r}, transformers=={installed_transformers!r} "
        f"matches none of this checkpoint's verified_environments: "
        f"{verified_environments!r}. Refusing to load: numeric/pooling "
        "behavior is not guaranteed identical outside a verified environment. "
        "Re-run the fidelity verification in this environment and add it to "
        "verified_environments if you have confirmed it loads and scores "
        "correctly here."
    )


@dataclass
class OPTPredictorHandle:
    """A loaded, ready-to-score official OPT predictor.

    ``score()``/``score_batch()`` are deterministic: the underlying model is
    put in ``eval()`` mode and every forward pass runs under
    ``torch.no_grad()`` -- no dropout, no sampling.
    """

    model: object
    tokenizer: object
    num_labels: int

    def _reduce(self, logits) -> List[float]:
        if self.num_labels > 1:
            return [float(v) for v in logits.argmax(dim=-1).tolist()]
        return [float(v[0]) for v in logits.tolist()]

    def score(self, prompt_text: str) -> float:
        return self.score_batch([prompt_text])[0]

    def score_batch(self, prompts: Sequence[str], batch_size: int = 8) -> List[float]:
        torch, _ = _require_torch_and_transformers()
        device = next(self.model.parameters()).device
        scores: List[float] = []
        for start in range(0, len(prompts), batch_size):
            chunk = list(prompts[start : start + batch_size])
            inputs = self.tokenizer(chunk, return_tensors="pt", truncation=True, padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
            scores.extend(self._reduce(outputs.logits))
        return scores


def load_opt_predictor_from_local(checkpoint_dir: str) -> OPTPredictorHandle:
    """Load the official predictor from a local checkpoint directory.

    Requires a ``vllm_ltr_provenance.json`` sidecar in ``checkpoint_dir``
    (see ``write_local_provenance_sidecar``) recording the pinned commit and
    library versions the checkpoint was verified against; rejects the
    checkpoint outright if that sidecar is missing, stale, or records no
    verified environment matching what's currently installed. See
    ``errors.py`` for the exact exception raised in each case.
    """
    if not os.path.isdir(checkpoint_dir):
        raise MissingCheckpointError(
            f"No checkpoint directory found at {checkpoint_dir!r}. Download "
            f"the official checkpoint from "
            f"https://huggingface.co/{provenance.CHECKPOINT_HF_REPO} first."
        )
    torch_mod, transformers_mod = _require_torch_and_transformers()
    sidecar = _read_provenance_sidecar(checkpoint_dir)
    _validate_provenance_sidecar(sidecar, torch_mod, transformers_mod)

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    # Read the RAW config.json before from_pretrained() runs: transformers
    # overwrites config._name_or_path to the local checkpoint_dir path once
    # loaded, destroying the original base-model reference we need below.
    with open(os.path.join(checkpoint_dir, "config.json"), "r", encoding="utf-8") as f:
        raw_config = json.load(f)
    original_base_model_name = raw_config.get("_name_or_path")

    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
    model.eval()

    # CONFIRMED 2026-08-04: the official checkpoint repo ships only
    # config.json/model.safetensors/usage_config.json -- no tokenizer files.
    # AutoTokenizer.from_pretrained(checkpoint_dir) silently falls back to a
    # generic GPT-2-style tokenizer (wrong vocab/pad id) instead of erroring.
    # The correct tokenizer is the base pretrained model recorded in the
    # checkpoint's own (pre-load) config.json._name_or_path (e.g.
    # "facebook/opt-125m"), whose pad_token_id (1) matches this checkpoint's
    # config.pad_token_id.
    if not original_base_model_name:
        raise StaleArtifactError(
            f"Checkpoint config.json at {checkpoint_dir!r} has no "
            "_name_or_path recording its base pretrained model -- cannot "
            "determine which tokenizer to load (the checkpoint repo itself "
            "ships no tokenizer files)."
        )
    tokenizer = AutoTokenizer.from_pretrained(original_base_model_name)
    if tokenizer.pad_token_id != model.config.pad_token_id:
        raise StaleArtifactError(
            f"Tokenizer {base_model_name!r} pad_token_id={tokenizer.pad_token_id} "
            f"does not match checkpoint config.pad_token_id={model.config.pad_token_id}. "
            "Refusing to score with a mismatched tokenizer."
        )
    num_labels = int(model.config.num_labels)
    return OPTPredictorHandle(model=model, tokenizer=tokenizer, num_labels=num_labels)


def download_and_provision_checkpoint(
    *,
    repo_id: str,
    subfolder: str,
    revision: str,
    local_dir: str,
    verified_environments: Sequence[Dict[str, str]],
) -> str:
    """Download one checkpoint variant's ``finetuned/`` weights from the Hub
    into ``local_dir`` (never inside the git repo -- this is a large binary
    artifact) and write the provenance sidecar this loader requires,
    including a sha256 of every downloaded file. Never modifies the
    downloaded weights.
    """
    _require_torch_and_transformers()
    from huggingface_hub import snapshot_download

    snapshot_path = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        allow_patterns=[f"{subfolder}/*"],
    )
    finetuned_dir = os.path.join(snapshot_path, subfolder, "finetuned")
    if not os.path.isdir(finetuned_dir):
        raise MissingCheckpointError(
            f"Expected {finetuned_dir!r} after snapshot_download; not found."
        )

    file_hashes = {}
    for fname in sorted(os.listdir(finetuned_dir)):
        if fname == PROVENANCE_SIDECAR_FILENAME:
            continue  # idempotency: never hash our own previously-written sidecar
        fpath = os.path.join(finetuned_dir, fname)
        if os.path.isfile(fpath):
            file_hashes[fname] = sha256_file(fpath)

    write_local_provenance_sidecar(
        finetuned_dir,
        pinned_commit=provenance.PINNED_COMMIT,
        verified_environments=verified_environments,
        checkpoint_repo_id=repo_id,
        checkpoint_revision=revision,
        checkpoint_subfolder=subfolder,
        file_hashes=file_hashes,
    )
    return finetuned_dir
