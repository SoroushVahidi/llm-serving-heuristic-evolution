"""Regression test for a ranking-agreement bug found while recovering the
vLLM-LTR comparative-evaluation run (see
docs/audits/vllm_ltr_comparative_evaluation_recovery_20260804.md):
``compute_ranking_agreement_record`` (formerly inlined in ``main()``) used
to compute both ``est_order`` and ``sof_order`` with the identical formula
(``-predicted_output_tokens``), so "agreement with EST" and "agreement with
SOF" were always numerically identical -- silently hiding the fact that
``estimated_service_time_first`` actually ranks by
``alpha*prompt_tokens + beta*predicted_output_tokens``
(``EstimatedServiceTimeFirstPolicy._sort_key``), not by predicted output
length alone.

This test constructs requests where prompt length and predicted output
length are anti-correlated, so the true EST and SOF orderings differ, then
sets synthetic LTR scores that perfectly track the true EST ordering. If
the bug were still present, both correlations would be identical (whatever
they came out to be); with the fix, agreement-with-EST is a perfect 1.0
while agreement-with-SOF is strictly lower.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from llmserveopt.core.types import Request

_SPEC = importlib.util.spec_from_file_location(
    "run_vllm_ltr_first_comparative_evaluation",
    Path(__file__).parent.parent / "scripts" / "run_vllm_ltr_first_comparative_evaluation.py",
)
run_eval = importlib.util.module_from_spec(_SPEC)
sys.modules["run_vllm_ltr_first_comparative_evaluation"] = run_eval
_SPEC.loader.exec_module(run_eval)


def _req(request_id: int, prompt_tokens: int, predicted_output_tokens: int) -> Request:
    return Request(
        request_id=request_id,
        arrival_time=float(request_id),
        prompt_tokens=prompt_tokens,
        predicted_output_tokens=predicted_output_tokens,
        actual_output_tokens=predicted_output_tokens,
        slo_deadline=float(request_id) + 100.0,
        priority=1.0,
        class_id="medium",
    )


# prompt_tokens and predicted_output_tokens are deliberately anti-correlated
# so the true EST ranking (alpha*prompt + beta*predicted_output) differs
# from the SOF ranking (predicted_output_tokens alone).
_REQUESTS = [
    _req(0, prompt_tokens=1000, predicted_output_tokens=100),
    _req(1, prompt_tokens=100, predicted_output_tokens=400),
    _req(2, prompt_tokens=700, predicted_output_tokens=200),
    _req(3, prompt_tokens=50, predicted_output_tokens=700),
]


def test_est_and_sof_orderings_actually_differ_for_this_fixture():
    """Sanity-check the fixture itself: EST rank order must differ from SOF
    rank order, otherwise this test can't distinguish the bug from the fix."""
    est_by_id = {
        r.request_id: run_eval.predicted_service_proxy(
            r, alpha=run_eval.DEFAULT_ALPHA, beta=run_eval.DEFAULT_BETA
        )
        for r in _REQUESTS
    }
    sof_by_id = {r.request_id: r.predicted_output_tokens for r in _REQUESTS}

    est_rank_order = sorted(est_by_id, key=lambda i: est_by_id[i])
    sof_rank_order = sorted(sof_by_id, key=lambda i: sof_by_id[i])
    assert est_rank_order != sof_rank_order


def test_ranking_agreement_est_and_sof_are_not_conflated():
    est_by_id = {
        r.request_id: run_eval.predicted_service_proxy(
            r, alpha=run_eval.DEFAULT_ALPHA, beta=run_eval.DEFAULT_BETA
        )
        for r in _REQUESTS
    }
    # LTR "scores" (higher = higher priority) constructed to perfectly track
    # the true EST ordering exactly.
    ltr_scores = {rid: -est for rid, est in est_by_id.items()}

    record = run_eval.compute_ranking_agreement_record(seed=0, requests=_REQUESTS, ltr_scores=ltr_scores)

    est_corr = record["spearman_ltr_vs_estimated_service_time_first"]
    sof_corr = record["spearman_ltr_vs_shortest_output_first"]

    assert est_corr == pytest.approx(1.0)
    assert sof_corr != pytest.approx(est_corr)


def test_est_order_uses_prompt_tokens_not_just_predicted_output():
    """Direct regression guard for the exact bug: est_order must depend on
    prompt_tokens, not be a duplicate of predicted_output_tokens-only
    ordering."""
    # Two requests with identical predicted_output_tokens but very
    # different prompt_tokens must get different EST scores (they'd tie
    # under the buggy predicted_output_tokens-only formula).
    a = _req(10, prompt_tokens=10, predicted_output_tokens=500)
    b = _req(11, prompt_tokens=2000, predicted_output_tokens=500)
    est_a = run_eval.predicted_service_proxy(a, alpha=run_eval.DEFAULT_ALPHA, beta=run_eval.DEFAULT_BETA)
    est_b = run_eval.predicted_service_proxy(b, alpha=run_eval.DEFAULT_ALPHA, beta=run_eval.DEFAULT_BETA)
    assert est_a != est_b
