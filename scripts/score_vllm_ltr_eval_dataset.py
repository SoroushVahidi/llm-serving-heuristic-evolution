#!/usr/bin/env python3
"""Offline vLLM-LTR scoring of the ingested WildChat evaluation prompts.

Uses the official, hash-verified checkpoint
(``baselines/vllm_ltr/adapter/checkpoint_loader.py``,
``docs/audits/vllm_ltr_baseline_audit_20260804.md``) to score every prompt
in ``wildchat_eval_prompts_by_id.json`` exactly once, offline, before any
simulator run. Regression variant by default -- the audit's own
recommendation (retains ranking signal the classification variant's argmax
collapses on short prompts).

Resumable: if ``--cache-path`` already exists, only missing request_ids are
scored; existing entries are kept as-is (after verifying their stored
prompt hash still matches the current prompt text -- ``StaleScoreCacheError``
otherwise, via ``offline_scoring.scores_only``). Rejects reuse across a
different checkpoint or tokenizer: a sibling ``<cache-path>.provenance.json``
records the checkpoint's file hashes + tokenizer identity; a mismatch
against the currently-loaded handle raises before any scoring happens.

Run inside tmux for long dataset sizes:
    tmux new -s vllm_ltr_eval_dataset
    python scripts/score_vllm_ltr_eval_dataset.py ...
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines.vllm_ltr.adapter import provenance
from baselines.vllm_ltr.adapter.checkpoint_loader import (
    download_and_provision_checkpoint,
    load_opt_predictor_from_local,
)
from baselines.vllm_ltr.adapter.errors import VLLMLTRAdapterError
from baselines.vllm_ltr.adapter.offline_scoring import (
    load_score_cache,
    save_score_cache,
    score_prompts_offline,
    scores_only,
)


class ScoreCacheProvenanceMismatchError(VLLMLTRAdapterError):
    """Raised when an existing score cache's recorded checkpoint/tokenizer
    provenance does not match the handle currently being used to score."""


def _provenance_path(cache_path: str) -> str:
    return cache_path + ".provenance.json"


def _current_provenance(ckpt_dir: str, subfolder: str, device: str, batch_size: int, dtype: str) -> dict:
    sidecar_path = os.path.join(ckpt_dir, "vllm_ltr_provenance.json")
    with open(sidecar_path, "r", encoding="utf-8") as f:
        sidecar = json.load(f)
    return {
        "checkpoint_repo_id": sidecar.get("checkpoint_repo_id"),
        "checkpoint_revision": sidecar.get("checkpoint_revision"),
        "checkpoint_subfolder": sidecar.get("checkpoint_subfolder"),
        "checkpoint_file_hashes": sidecar.get("file_hashes"),
        "tokenizer_name": "facebook/opt-125m",
        "device": device,
        "batch_size": batch_size,
        "dtype": dtype,
    }


def _check_provenance_compatible(recorded: dict, current: dict) -> None:
    keys = ["checkpoint_repo_id", "checkpoint_revision", "checkpoint_subfolder", "checkpoint_file_hashes"]
    mismatches = {k: (recorded.get(k), current.get(k)) for k in keys if recorded.get(k) != current.get(k)}
    if mismatches:
        raise ScoreCacheProvenanceMismatchError(
            f"Existing score cache provenance does not match the current "
            f"checkpoint handle. Mismatched fields: {mismatches}. Refusing "
            "to reuse/extend a cache scored under a different checkpoint. "
            "Use a fresh --cache-path if you intend to score with a "
            "different checkpoint variant."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts-path", default="data/processed/wildchat/wildchat_eval_prompts_by_id.json")
    parser.add_argument("--cache-path", default="data/processed/wildchat/vllm_ltr_score_cache.json")
    parser.add_argument(
        "--variant",
        choices=["regression", "classification"],
        default="regression",
        help="regression = raw logit, retains ranking signal (audit-recommended); "
             "classification = argmax bin, may saturate on short prompts.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    with open(args.prompts_path, "r", encoding="utf-8") as f:
        raw_prompts = json.load(f)
    id_to_prompt: Dict[int, str] = {int(k): v for k, v in raw_prompts.items()}
    print(f"Loaded {len(id_to_prompt)} prompts from {args.prompts_path}")

    subfolder = (
        provenance.CHECKPOINT_VARIANT_REGRESSION
        if args.variant == "regression"
        else provenance.CHECKPOINT_VARIANT_CLASSIFICATION
    )

    import torch
    import transformers

    verified_environments = [
        {"torch_version": "2.2.1", "transformers_version": "4.45.2"},
        {"torch_version": torch.__version__, "transformers_version": transformers.__version__},
    ]
    print(f"Downloading/provisioning checkpoint variant: {subfolder} ...")
    ckpt_dir = download_and_provision_checkpoint(
        repo_id=provenance.CHECKPOINT_HF_REPO,
        subfolder=subfolder,
        revision=provenance.CHECKPOINT_HF_REVISION,
        local_dir="unused",
        verified_environments=verified_environments,
    )
    handle = load_opt_predictor_from_local(ckpt_dir)
    device = str(next(handle.model.parameters()).device)
    dtype = str(next(handle.model.parameters()).dtype)
    current_prov = _current_provenance(ckpt_dir, subfolder, device, args.batch_size, dtype)

    existing_cache: Dict[int, dict] = {}
    if os.path.exists(args.cache_path):
        existing_cache = load_score_cache(args.cache_path)
        prov_path = _provenance_path(args.cache_path)
        if not os.path.exists(prov_path):
            raise ScoreCacheProvenanceMismatchError(
                f"Score cache {args.cache_path} exists but has no sibling "
                f"provenance file {prov_path}. Refusing to resume/extend an "
                "unprovenanced cache."
            )
        with open(prov_path, "r", encoding="utf-8") as f:
            recorded_prov = json.load(f)
        _check_provenance_compatible(recorded_prov, current_prov)
        # Reject stale entries (prompt text changed for a reused id).
        scores_only(existing_cache, id_to_prompt={
            rid: p for rid, p in id_to_prompt.items() if rid in existing_cache
        })
        print(f"Resuming: {len(existing_cache)} scores already cached.")

    missing_ids = {rid: p for rid, p in id_to_prompt.items() if rid not in existing_cache}
    print(f"Scoring {len(missing_ids)} missing prompts (batch_size={args.batch_size}) ...")
    t0 = time.perf_counter()
    new_scores = score_prompts_offline(handle, missing_ids, batch_size=args.batch_size) if missing_ids else {}
    elapsed = time.perf_counter() - t0

    merged = {**existing_cache, **new_scores}
    save_score_cache(merged, args.cache_path)

    current_prov["num_scored_this_run"] = len(new_scores)
    current_prov["num_total_cached"] = len(merged)
    current_prov["scoring_wall_clock_s_this_run"] = elapsed
    current_prov["num_prompts_truncated_this_run"] = handle.num_prompts_truncated
    current_prov["command"] = " ".join(sys.argv)
    current_prov["timestamp_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(_provenance_path(args.cache_path), "w", encoding="utf-8") as f:
        json.dump(current_prov, f, indent=2)

    print(f"Scored {len(new_scores)} new prompts in {elapsed:.2f}s.")
    print(f"Wrote {args.cache_path} ({len(merged)} total entries)")
    print(f"Wrote {_provenance_path(args.cache_path)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
