# Official reference: scheduler ranking excerpt

**Not executable. Not imported by any adapter code.** This is a literal
citation of the pinned-commit source used to verify the ranking/ordering
semantics reproduced in `adapter/ranking_adapter.py`.

- **Source:** `vllm/core/scheduler.py`
- **Repository:** https://github.com/hao-ai-lab/vllm-ltr
- **Pinned commit:** `13bbf6ff3dab661791d41362551b089e5f77c91c`
- **License:** Apache License 2.0 (inherited from the source repository)
- **Lines:** 1040–1050 (method body; surrounding class/imports elided)

```python
def _get_ltr_ordered_requests(self):

    need_aux_scores = []
    for r in self.waiting:
        if r.need_aux_model_score():
            need_aux_scores.append(r)

    if need_aux_scores:
        self.aux_model.obtain_aux_scores(need_aux_scores)

    return list(sorted(list(self.waiting) + list(self.running) + list(self.swapped), key=lambda req: -req.aux_model_score))
```

## What this establishes

1. **Ordering key:** `-req.aux_model_score` — i.e. **descending** score is
   highest priority (larger `aux_model_score` schedules first).
2. **Scope:** the sort is applied across `waiting + running + swapped`
   combined, not `waiting` alone — this baseline reorders in-flight requests
   too, not just admission order. `adapter/ranking_adapter.py` reproduces the
   *ordering rule* faithfully; whether a simulator policy wrapper can apply it
   to in-flight requests depends on the calling simulator's action schema
   (see `docs/audits/vllm_ltr_baseline_audit_20260804.md` for how this project
   handles that).
3. **Tie-breaking:** Python's `sorted()` is stable, so requests with equal
   `aux_model_score` keep their relative order from
   `list(self.waiting) + list(self.running) + list(self.swapped)` — i.e.
   ties break by (queue-membership order, then within-queue insertion order).
   No secondary key (e.g. request ID or arrival time) is applied explicitly.
4. **Score source:** `aux_model.obtain_aux_scores(...)` is called lazily,
   only for requests that don't already have a cached score
   (`need_aux_model_score()` returns `True` iff `aux_model_score is None`,
   see `vllm/sequence.py:461-465` in the pinned commit) — scores are computed
   once per request and cached, not recomputed every scheduling step.
