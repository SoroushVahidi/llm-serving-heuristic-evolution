
import json
import os
import time

TRACE_PATH = '/home/soroush/llm-serving-heuristic-evolution/experiments/real_vllm_mechanism_validation_v1/native_vllm_chunk_budget_semantics_probe_v1/scheduler_trace.jsonl'
CONTEXT_PATH = '/home/soroush/llm-serving-heuristic-evolution/experiments/real_vllm_mechanism_validation_v1/native_vllm_chunk_budget_semantics_probe_v1/trace_context.json'

try:
    from vllm.v1.core.sched.scheduler import Scheduler
    _ORIGINAL_SCHEDULE = Scheduler.schedule

    def _load_context():
        try:
            with open(CONTEXT_PATH, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return {}

    def _req_snapshot(req):
        return {
            "request_id": getattr(req, "request_id", None),
            "num_computed_tokens": getattr(req, "num_computed_tokens", None),
            "num_prompt_tokens": getattr(req, "num_prompt_tokens", None),
            "num_tokens": getattr(req, "num_tokens", None),
            "is_prefill_chunk": bool(getattr(req, "is_prefill_chunk", False)),
            "status": str(getattr(req, "status", None)),
        }

    def _traced_schedule(self, *args, **kwargs):
        before = {}
        try:
            for req_id, req in getattr(self, "requests", {}).items():
                before[req_id] = _req_snapshot(req)
        except Exception:
            before = {}
        out = _ORIGINAL_SCHEDULE(self, *args, **kwargs)
        try:
            scheduled = []
            prefill_tokens = 0
            decode_tokens = 0
            partial_prefill_items = 0
            for req_id, n_tokens in getattr(out, "num_scheduled_tokens", {}).items():
                snap = before.get(req_id, {})
                req = getattr(self, "requests", {}).get(req_id)
                prompt = snap.get("num_prompt_tokens")
                computed = snap.get("num_computed_tokens")
                if prompt is None and req is not None:
                    prompt = getattr(req, "num_prompt_tokens", None)
                if computed is None and req is not None:
                    computed = getattr(req, "num_computed_tokens", None)
                if prompt is not None and computed is not None:
                    p = max(0, min(int(n_tokens), int(prompt) - int(computed)))
                    d = max(0, int(n_tokens) - p)
                    if p > 0 and int(computed) + p < int(prompt):
                        partial_prefill_items += 1
                else:
                    p = None
                    d = None
                if p is not None:
                    prefill_tokens += p
                if d is not None:
                    decode_tokens += d
                scheduled.append({
                    "request_id": req_id,
                    "num_scheduled_tokens": int(n_tokens),
                    "computed_before": computed,
                    "prompt_tokens": prompt,
                    "prefill_tokens": p,
                    "decode_tokens": d,
                    "was_prefill_chunk_before": snap.get("is_prefill_chunk"),
                })
            row = {
                **_load_context(),
                "time": time.time(),
                "pid": os.getpid(),
                "current_step": getattr(self, "current_step", None),
                "max_num_scheduled_tokens": getattr(self, "max_num_scheduled_tokens", None),
                "running_len_after_schedule": len(getattr(self, "running", [])),
                "waiting_len_after_schedule": len(getattr(self, "waiting", [])),
                "scheduled_req_count": len(scheduled),
                "total_num_scheduled_tokens": getattr(out, "total_num_scheduled_tokens", None),
                "prefill_tokens": prefill_tokens,
                "decode_tokens": decode_tokens,
                "has_prefill_and_decode": prefill_tokens > 0 and decode_tokens > 0,
                "partial_prefill_items": partial_prefill_items,
                "scheduled": scheduled,
            }
            with open(TRACE_PATH, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, sort_keys=True) + "\n")
        except Exception as exc:
            with open(TRACE_PATH, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"time": time.time(), "pid": os.getpid(), "trace_error": repr(exc)}, sort_keys=True) + "\n")
        return out

    Scheduler.schedule = _traced_schedule
except Exception as exc:
    with open(TRACE_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"time": time.time(), "pid": os.getpid(), "patch_error": repr(exc)}, sort_keys=True) + "\n")
