"""Importable provenance constants for the VTC baseline.

Mirrors ``baselines/vtc/PROVENANCE.md``. Kept as plain constants (not
parsed out of the markdown) so tests can assert code and doc agree without
a markdown parser; if you update one, update the other. Follows
``baselines/pars/adapter/provenance.py``'s established convention.
"""
from __future__ import annotations

import os as _os

OFFICIAL_REPOSITORY = "https://github.com/Ying1123/VTC-artifact"
PINNED_COMMIT = "192c2e2014c69c8c6c699d7113c3822e4db632e6"
PINNED_COMMIT_DATE = "2024-06-07"
LICENSE = "Apache-2.0"

PAPER_TITLE = "Fairness in Serving Large Language Models"
PAPER_VENUE = "OSDI 2024 (18th USENIX Symposium on Operating Systems Design and Implementation)"
PAPER_ARXIV_ID = "2401.00588"
PAPER_AUTHORS = (
    "Sheng, Ying", "Cao, Shiyi", "Li, Dacheng", "Zhu, Banghua",
    "Li, Zhuohan", "Zhuo, Danyang", "Gonzalez, Joseph E.", "Stoica, Ion",
)
PAPER_YEAR = 2024

#: Local, non-committed clone path (see PROVENANCE.md -- the official repo
#: is never vendored into this project's git history, even though its
#: Apache-2.0 license would permit it, for consistency with the other
#: external baselines). Overridable via the VTC_OFFICIAL_CLONE_PATH
#: environment variable.
DEFAULT_OFFICIAL_CLONE_PATH = _os.path.expanduser("~/.cache/external_baselines/VTC")

#: Relative paths (within the clone) of the official, unmodified source
#: files this adapter dynamically imports. Never copy-pasted into this
#: project -- see official_loader.py.
CORE_POLICY_FILES = (
    "slora/server/router/vtc_req_queue.py",
    "slora/server/router/req_queue.py",
    "slora/server/io_struct.py",
    "slora/server/sampling_params.py",
    "slora/utils/infer_utils.py",
)

#: Default cost-accounting parameters, matching VTCReqQueue's own
#: constructor defaults exactly (input_price=1, output_price=2).
DEFAULT_INPUT_PRICE = 1
DEFAULT_OUTPUT_PRICE = 2
#: "profile" requires a hardware-calibrated regression fit to the official
#: authors' own A10G + Llama-2-7B setup (see PROVENANCE.md) and is not
#: portable to this simulator's synthetic timing model -- "linear" is the
#: only cost function this adapter supports.
SUPPORTED_COST_FUNC = "linear"

FIDELITY_LABEL = (
    "official policy reused with simulator adapter (unmodified VTCReqQueue "
    "executed verbatim; GPU serving-engine layer not run -- see "
    "PROVENANCE.md hardware blocker)"
)
SELECTOR_CANDIDATE = False
HISTORICAL = False
EVALUATION_ONLY = True
