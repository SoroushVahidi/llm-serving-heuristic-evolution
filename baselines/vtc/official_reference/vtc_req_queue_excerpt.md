# Official reference: VTC scheduling loop excerpt

**Not executable. Not imported by any adapter code.** This is a literal
citation of the pinned-commit source `adapter/official_loader.py` actually
imports and runs at runtime (dynamically, from the local clone) -- included
here only so the algorithm is readable without needing the clone checked
out. If this excerpt and the live clone ever disagree, the live clone (at
`adapter.provenance.PINNED_COMMIT`) is authoritative.

- **Source:** `slora/server/router/vtc_req_queue.py`
- **Repository:** https://github.com/Ying1123/VTC-artifact
- **Pinned commit:** `192c2e2014c69c8c6c699d7113c3822e4db632e6`
- **License:** Apache License 2.0

```python
class VTCReqQueue(ReqQueue):

    def __init__(self, max_total_tokens, batch_max_tokens, running_max_req_size,
                 adapter_dirs, fair_weights, cost_func,
                 input_price=1, output_price=2) -> None:
        super().__init__(max_total_tokens, batch_max_tokens, running_max_req_size)
        self.input_price = input_price
        self.output_price = output_price
        self.served = {}
        self.user_req_list = {}
        self.adapter_dirs = adapter_dirs
        self.fair_weights = fair_weights
        self.cost_func = cost_func
        self.fairw = {}
        for i in range(len(adapter_dirs)):
            if i < len(fair_weights):
                self.fairw[adapter_dirs[i]] = fair_weights[i]
            else:
                self.fairw[adapter_dirs[i]] = 1

    def append(self, req):
        self.waiting_req_list.append(req)
        if req.adapter_dir not in self.user_req_list:
            self.user_req_list[req.adapter_dir] = deque([req])
            self.served[req.adapter_dir] = 0
        else:
            self.user_req_list[req.adapter_dir].append(req)
        # waiting queue was empty before
        if len(self.user_req_list[req.adapter_dir]) == 1:
            # lift counter
            cnts = [v for k, v in self.served.items()
                      if (len(self.user_req_list[k]) > 0 and k != req.adapter_dir)]
            if len(cnts) > 0:
                self.served[req.adapter_dir] = max(self.served[req.adapter_dir], min(cnts))

    def generate_new_batch(self, current_batch, lora_ranks):
        # ... (memory-feasibility setup elided; see req_queue.py)
        active_served = {k: v for k, v in self.served.items()}
        while True:
            if len(active_served) == 0:
                break
            adapter_dir = min(active_served, key=active_served.get)
            if len(self.user_req_list[adapter_dir]) > 0:
                req = self.user_req_list[adapter_dir][0]
                if req.aborted:
                    # dropped for free, no cost charged
                    ...
                if (self._can_add_new_req(req, lora_ranks) and
                    new_batch_total_tokens + req.input_len <= self.batch_max_tokens):
                    can_run_list.append(req)
                    new_batch_total_tokens += req.input_len
                    self.user_req_list[adapter_dir].popleft()
                    if self.cost_func == "linear":
                        self.served[adapter_dir] += req.input_len * self.input_price / self.fairw[adapter_dir]
                        active_served[adapter_dir] += req.input_len * self.input_price / self.fairw[adapter_dir]
                    # ... "profile" branch elided (not used by this project's adapter)
                else:
                    break
            else:
                del active_served[adapter_dir]
        # ...

    def update_counter(self, current_batch):
        for req in current_batch.reqs:
            if self.cost_func == "linear":
                self.served[req.adapter_dir] += 1 * self.output_price / self.fairw[req.adapter_dir]
            # ... "profile" branch elided
```

## What this establishes

1. **Scheduling rule:** `adapter_dir = min(active_served, key=active_served.get)`
   -- greedy min-served-first, evaluated fresh (against a *local copy*,
   `active_served`) for every request admitted within a single
   `generate_new_batch` call, so later picks in the same call already see
   earlier picks' cost updates.
2. **Tie-breaking:** Python `min()` returns the first key achieving the
   minimum in dict iteration order = insertion order of `self.served`,
   which is the order each tenant's queue first became non-empty (set
   inside `append()`). Not a documented rule in the paper -- an
   implementation artifact this adapter inherits automatically by calling
   the real object, never re-derives independently.
3. **"Lift counter" rule (`append()`):** when a tenant's queue transitions
   from empty to non-empty (a previously-inactive tenant returns), its
   `served` counter is raised to at least the minimum counter among other
   *currently non-empty-queue* tenants. This is the mechanism the paper's
   2×-tight service-difference bound theorem depends on for returning
   clients -- without it, a long-idle tenant could return with an
   arbitrarily large service deficit and monopolize the scheduler.
4. **Aborted requests cost nothing:** dropped from the head of a tenant's
   queue with zero `served` charge -- distinct from admitted-then-charged
   requests.
5. **Accounting is per-call-count, not per-token-inspection:** `update_counter`
   under `cost_func="linear"` adds a flat `1 * output_price / fairw` for
   *every request present in the batch passed to it*, once per call --
   it does not read `len(req.output_ids)` at all in this branch. This is
   why this project's adapter can faithfully reconstruct decode cost from
   per-step `tokens_decoded_per_request` deltas by calling the real method
   once per delta token, rather than needing a native per-decode-tick hook.
