# Problem Formulation

## Setting

We study **online LLM inference serving** with multiple GPUs and dynamic batching.

### Request model

Each request $r$ is characterized by observable features available at arrival time:

| Symbol | Description |
|---|---|
| $i$ | Request ID |
| $a_i$ | Arrival time |
| $p_i$ | Prompt length (tokens) |
| $\hat{o}_i$ | Predicted output length (tokens) — noisy estimate |
| $d_i$ | SLO deadline: $a_i + \text{slack}$ |
| $\pi_i$ | Priority class weight |
| $c_i$ | SLO class string |

Ground truth (hidden from online policies):

| Symbol | Description |
|---|---|
| $o_i$ | Actual output length (known only after generation completes) |

### GPU model

A pool of $G$ identical GPUs. Each GPU $g$ has:

| Parameter | Description |
|---|---|
| $S_g$ | Maximum concurrent active sequences |
| $B_g$ | Maximum batch tokens per decode step |
| $K_g$ | Maximum KV-cache capacity (tokens) |

### State

At any time $t$, the serving state consists of:
- **Waiting queue** $\mathcal{W}(t)$: arrived but not yet admitted requests
- **Active sets** $\mathcal{A}_g(t)$ for each GPU $g$: requests currently being decoded
- **Completed set** $\mathcal{C}(t)$: finished requests

### Action space

At each scheduling step, a policy outputs:

$$\pi(s_t) = \{(g, r) : r \text{ is admitted to GPU } g\}$$

subject to feasibility constraints for each GPU $g$:

$$|\mathcal{A}_g(t)| + |\text{admitted}_g| \leq S_g$$

$$\text{KV}(\mathcal{A}_g(t)) + \sum_{r \in \text{admitted}_g} p_r \leq K_g$$

$$|\mathcal{A}_g(t)| + |\text{admitted}_g| \leq B_g$$

where $\text{KV}(\mathcal{A}_g(t)) = \sum_{r \in \mathcal{A}_g(t)} (p_r + \text{decoded}_r)$.

### Dynamics

In Phase 1, each admitted request $r$ takes exactly $o_r$ decode steps to complete.
At each step, each active request generates 1 output token.
Requests complete when $\text{decoded}_r \geq o_r$.

### Objectives

We evaluate policies on:

1. **Priority-weighted SLO goodput** (primary objective, paper-facing name):
   $$\text{priority\_weighted\_slo\_goodput} = \frac{\sum_r w_r \cdot \mathbf{1}[c_r \leq d_r]}{\sum_r w_r}$$
   where $w_r = \text{priority}_r$ (default 1.0 when unset).
   Internal alias: `weighted_goodput`. Both names appear in experiment CSVs.
2. **Mean / P95 / P99 latency** — end-to-end response time ($c_r - a_r$)
3. **SLO violation rate** — fraction where $c_r > d_r$
4. **Request throughput** — completed requests per second
5. **Token throughput** — output tokens per second
6. **GPU utilization** — mean fraction of $S_g$ in use

### Optimal policy (theoretical)

The problem is NP-hard in general (reduction from makespan scheduling).
Offline SRTF (Shortest Remaining Time First) minimizes mean completion time in the
single-machine setting; the oracle policy in this codebase implements a multi-GPU
greedy approximation using true $o_r$ values.

## Phase 1 simplifications

- No preemption after admission
- No speculative decoding
- Prefill cost not separately modeled (treated as zero)
- All GPUs are identical
- Token generation rate is uniform (1 token/step regardless of batch size)
