#!/usr/bin/env python3
"""Deterministic ingestion of a real-prompt evaluation sample from
WildChat-1M (official, ODC-BY, ungated — see
``external/datasets/wildchat.md`` for the full dataset-selection audit).

Pipeline
--------
1. Stream the pinned dataset revision (never ``revision="main"``), scanning
   at most ``--scan-row-cap`` rows in the dataset's own on-disk order (a
   deterministic prefix of a pinned parquet snapshot).
2. Filter to single-turn (``turn == 1``), English, non-toxic, non-redacted
   conversations, reading the filter fields from the FIRST turn
   (``conversation[0]``) specifically, since that is the turn whose text
   becomes the evaluation prompt.
3. Deterministically sample ``--sample-size`` conversations: sort the
   filtered candidate pool by its own content-addressed
   ``conversation_hash`` (removes any dependence on stream iteration
   order), then take a seeded random sample. Same revision + scan cap +
   sample size + seed => byte-identical sample, always.
4. Tokenize each selected prompt/response with the exact vLLM-LTR
   checkpoint tokenizer (``facebook/opt-125m`` — see
   ``baselines/vllm_ltr/adapter/checkpoint_loader.py``, which sources this
   same tokenizer from the checkpoint's own recorded base model).
5. Assign stable integer ``request_id``s (0..n-1, in whatever order
   ``deterministic_sample()`` actually returns) and compute a sha256 prompt
   hash for each. CORRECTION (2026-08-04): an earlier version of this
   docstring claimed request_ids follow "sorted-hash order from step 3" --
   that is false. Step 3's sort-by-``conversation_hash`` only controls
   *which* conversations are selected deterministically (so
   ``random.Random(seed).sample()`` draws from an order-independent pool);
   ``random.Random.sample()`` does **not** preserve the input sequence's
   order in its return value, so the sample handed to this step is a
   seeded-random permutation, not a hash-sorted list. The actual guarantee
   is: same revision + scan cap + sample size + seed => the exact same
   permutation, and therefore the exact same request_id assignment, every
   rerun -- determinism, not sortedness. See
   docs/audits/vllm_ltr_comparative_evaluation_recovery_20260804.md.
6. Write three artifacts (all under ``--output-dir``, none committed to
   git per this repo's existing ``data/raw/*`` / ``data/processed/*``
   .gitignore convention):
   - ``wildchat_eval_sharegpt_shaped.json`` — ShareGPT-compatible shape,
     directly consumable by the existing, unmodified
     ``llmserveopt.workloads.sharegpt.convert_sharegpt_to_requests``.
   - ``wildchat_eval_prompts_by_id.json`` — ``{request_id: prompt_text}``,
     for offline vLLM-LTR scoring.
   - ``wildchat_eval_manifest.json`` — full reproducibility record,
     including duplicate-prompt accounting (WildChat itself contains a
     small number of organically duplicated prompts across distinct
     conversations; this pipeline does not deduplicate them -- see
     ``write_outputs()``).

No future information leaks into this artifact set: only prompt text and
the real first-response text (kept solely for the same
real-actual-output-tokens role ``workloads/sharegpt.py`` already gives
ShareGPT responses — i.e. the oracle/prediction-noise source, never
exposed to deployable policies) are extracted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
import unicodedata
from typing import Dict, List, Optional, Tuple

HF_REPO = "allenai/WildChat-1M"
PINNED_REVISION = "7d6490e462285cf85d91eabea0f9a954fbddcd1f"
TOKENIZER_NAME = "facebook/opt-125m"

DEFAULT_SCAN_ROW_CAP = 100_000
DEFAULT_SAMPLE_SIZE = 300
DEFAULT_SEED = 20260804


def normalize_text(text: str) -> str:
    """Deterministic text normalization: Unicode NFC + stripped whitespace."""
    return unicodedata.normalize("NFC", text).strip()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def first_turn_eligible(row: dict) -> bool:
    """True iff this row's evaluation-relevant fields pass the filter.

    Reads ``conversation[0]`` (the turn that becomes the prompt) directly,
    rather than the row's top-level ``language``/``toxic``/``redacted``
    (which, per manual inspection of real rows, can reflect a later turn
    in multi-turn conversations)."""
    if row.get("turn") != 1:
        return False
    conv = row.get("conversation") or []
    if len(conv) < 2:
        return False
    user_turn, assistant_turn = conv[0], conv[1]
    if user_turn.get("role") != "user" or assistant_turn.get("role") != "assistant":
        return False
    if user_turn.get("language") != "English":
        return False
    if user_turn.get("toxic", True) is not False:
        return False
    if user_turn.get("redacted", True) is not False:
        return False
    prompt = user_turn.get("content")
    response = assistant_turn.get("content")
    if not prompt or not isinstance(prompt, str) or not prompt.strip():
        return False
    if not response or not isinstance(response, str) or not response.strip():
        return False
    conversation_hash = row.get("conversation_hash")
    if not conversation_hash or not isinstance(conversation_hash, str):
        return False
    return True


def extract_pair(row: dict) -> Tuple[str, str, str]:
    """Returns (prompt_text, response_text, conversation_hash), normalized."""
    conv = row["conversation"]
    prompt = normalize_text(conv[0]["content"])
    response = normalize_text(conv[1]["content"])
    return prompt, response, row["conversation_hash"]


def scan_candidates(scan_row_cap: int, revision: str = PINNED_REVISION) -> List[Tuple[str, str, str]]:
    """Stream up to scan_row_cap rows from the pinned revision and return
    every (prompt, response, conversation_hash) tuple that passes the
    filter. Deterministic: same revision + same cap => same candidate
    pool, regardless of when/where this is run."""
    from datasets import load_dataset

    ds = load_dataset(
        HF_REPO, split="train", streaming=True, revision=revision
    )
    candidates: List[Tuple[str, str, str]] = []
    scanned = 0
    for row in ds:
        if scanned >= scan_row_cap:
            break
        scanned += 1
        if first_turn_eligible(row):
            candidates.append(extract_pair(row))
    return candidates


def deterministic_sample(
    candidates: List[Tuple[str, str, str]],
    sample_size: int,
    seed: int,
) -> List[Tuple[str, str, str]]:
    """Sort by conversation_hash (content-addressed, order-independent) so
    *which* items are eligible to be drawn never depends on stream/scan
    order, then take a seeded random sample. The returned list's order is
    NOT the sorted order -- ``random.Random(seed).sample()`` returns items
    in a seeded-random permutation, not in the order of its input sequence
    (verified: sampling from an already-sorted list does not yield a sorted
    result). Determinism holds regardless: same input pool + sample_size +
    seed => the exact same output list, byte-for-byte, every rerun. Raises
    if the pool is too small."""
    if len(candidates) < sample_size:
        raise ValueError(
            f"Candidate pool ({len(candidates)}) smaller than requested "
            f"sample_size ({sample_size}); increase --scan-row-cap."
        )
    pool = sorted(candidates, key=lambda t: t[2])  # sort by conversation_hash
    rng = random.Random(seed)
    return rng.sample(pool, sample_size)


def tokenize_and_hash(
    sample: List[Tuple[str, str, str]],
    tokenizer_name: str,
) -> Tuple[List[dict], str]:
    """Returns (rows, tokenizer_revision). request_id = index in whatever
    order `sample` is given in -- NOT necessarily conversation_hash order.
    When `sample` comes from `deterministic_sample()`, that order is a
    seeded-random permutation of the hash-sorted candidate pool (see that
    function's docstring), not the sorted order itself. This function is
    order-preserving w.r.t. its input either way: same `sample` order in
    => same request_id assignment out, always."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer_revision = getattr(tokenizer, "_commit_hash", None) or "unknown"

    rows = []
    for request_id, (prompt, response, conv_hash) in enumerate(sample):
        prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
        response_tokens = len(tokenizer.encode(response, add_special_tokens=False))
        rows.append({
            "request_id": request_id,
            "conversation_hash": conv_hash,
            "prompt_text": prompt,
            "response_text": response,
            "prompt_sha256": sha256_text(prompt),
            "prompt_tokens": prompt_tokens,
            "response_tokens": response_tokens,
        })
    return rows, tokenizer_revision


def duplicate_prompt_summary(rows: List[dict]) -> dict:
    """Account for organically duplicated prompt text across distinct
    WildChat conversations (different users can submit byte-identical
    prompts; this pipeline does not deduplicate them -- see write_outputs()
    docstring for why). Returns total/unique counts plus, for each
    duplicated prompt hash, the request_ids sharing it -- so a rerun can
    see at a glance whether the sample's effective prompt diversity is
    lower than sample_size implies, without re-deriving it by hand."""
    from collections import defaultdict

    by_hash: Dict[str, List[int]] = defaultdict(list)
    for r in rows:
        by_hash[r["prompt_sha256"]].append(r["request_id"])
    duplicate_groups = [
        {"prompt_sha256": h, "request_ids": sorted(ids)}
        for h, ids in sorted(by_hash.items())
        if len(ids) > 1
    ]
    return {
        "total_sampled_rows": len(rows),
        "unique_prompt_hashes": len(by_hash),
        "duplicate_prompt_count": len(rows) - len(by_hash),
        "duplicate_groups": duplicate_groups,
    }


def write_outputs(rows: List[dict], manifest_extra: dict, output_dir: str) -> dict:
    """Write the three pipeline artifacts. Deliberately does NOT deduplicate
    ``rows`` by prompt text: two different WildChat conversations
    (different ``conversation_hash``, different submitter/timestamp per the
    raw dataset) can carry byte-identical prompt text -- a real, organic
    property of the source data, not an ingestion artifact. Silently
    dropping one would (a) make ``--sample-size`` stop meaning "this many
    conversations sampled" and (b) require an explicit dedup policy this
    evaluation's spec never asked for. Instead, ``duplicate_prompt_summary()``
    records exactly how many such duplicates exist so anyone reading the
    manifest sees whether the sample's *unique*-prompt diversity is lower
    than sample_size implies."""
    os.makedirs(output_dir, exist_ok=True)

    sharegpt_shaped = [
        {
            "conversations": [
                {"from": "human", "value": r["prompt_text"]},
                {"from": "gpt", "value": r["response_text"]},
            ]
        }
        for r in rows
    ]
    pairs_path = os.path.join(output_dir, "wildchat_eval_sharegpt_shaped.json")
    with open(pairs_path, "w", encoding="utf-8") as f:
        json.dump(sharegpt_shaped, f, ensure_ascii=False)

    prompts_by_id = {str(r["request_id"]): r["prompt_text"] for r in rows}
    prompts_path = os.path.join(output_dir, "wildchat_eval_prompts_by_id.json")
    with open(prompts_path, "w", encoding="utf-8") as f:
        json.dump(prompts_by_id, f, ensure_ascii=False, indent=2)

    prompt_lengths = sorted(r["prompt_tokens"] for r in rows)
    n = len(prompt_lengths)
    manifest = {
        "dataset": "WildChat-1M",
        "hf_repo": HF_REPO,
        "pinned_revision": PINNED_REVISION,
        "license": "ODC-BY",
        "official_source": "https://huggingface.co/datasets/allenai/WildChat-1M",
        "sample_size": n,
        "row_hashes": {str(r["request_id"]): r["prompt_sha256"] for r in rows},
        "conversation_hashes": {str(r["request_id"]): r["conversation_hash"] for r in rows},
        "duplicate_prompt_accounting": duplicate_prompt_summary(rows),
        "prompt_tokens_stats": {
            "min": prompt_lengths[0] if n else None,
            "p50": prompt_lengths[n // 2] if n else None,
            "p95": prompt_lengths[int(n * 0.95)] if n else None,
            "max": prompt_lengths[-1] if n else None,
        },
        "outputs": {
            "sharegpt_shaped_path": pairs_path,
            "prompts_by_id_path": prompts_path,
        },
        "access_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **manifest_extra,
    }
    manifest_path = os.path.join(output_dir, "wildchat_eval_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return {"pairs_path": pairs_path, "prompts_path": prompts_path, "manifest_path": manifest_path}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-row-cap", type=int, default=DEFAULT_SCAN_ROW_CAP)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--tokenizer", type=str, default=TOKENIZER_NAME)
    parser.add_argument("--revision", type=str, default=PINNED_REVISION)
    parser.add_argument(
        "--output-dir", type=str, default="data/processed/wildchat"
    )
    args = parser.parse_args()

    print(f"Scanning up to {args.scan_row_cap} rows of {HF_REPO}@{args.revision} ...")
    candidates = scan_candidates(args.scan_row_cap, revision=args.revision)
    print(f"  {len(candidates)} rows passed the filter.")

    sample = deterministic_sample(candidates, args.sample_size, args.seed)
    print(f"  Deterministically sampled {len(sample)} conversations (seed={args.seed}).")

    rows, tokenizer_revision = tokenize_and_hash(sample, args.tokenizer)

    manifest_extra = {
        "scan_row_cap": args.scan_row_cap,
        "candidate_pool_size": len(candidates),
        "sampling_seed": args.seed,
        "tokenizer_name": args.tokenizer,
        "tokenizer_revision": tokenizer_revision,
        "command": " ".join(sys.argv),
        "filters": {
            "turn": 1,
            "language": "English",
            "toxic": False,
            "redacted": False,
        },
    }
    paths = write_outputs(rows, manifest_extra, args.output_dir)
    print(f"Wrote {paths['pairs_path']}")
    print(f"Wrote {paths['prompts_path']}")
    print(f"Wrote {paths['manifest_path']}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    else:
        # Work-around: the `datasets` streaming backend (fsspec/HTTP) can
        # leave a background thread in a state that crashes normal Python
        # interpreter finalization (PyGILState_Release on an already-freed
        # thread state) even after all real work -- including every file
        # write above -- has completed successfully. All output files are
        # already flushed and closed via `with open(...)` by this point, so
        # skipping the rest of normal shutdown is safe here.
        sys.stdout.flush()
        os._exit(0)
