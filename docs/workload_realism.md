# Workload Realism in LLM Serving Simulation

## Why Real Arrival Patterns Matter

Synthetic Poisson arrivals assume memoryless, independent inter-request gaps.
Real LLM serving traces exhibit:

- **Temporal correlation**: bursts of requests arrive together due to shared
  usage patterns (e.g., office hours, viral content, batch jobs)
- **Non-stationarity**: mean arrival rate shifts over time
- **Diurnal patterns**: request rate varies by time of day

Schedulers that perform well under Poisson arrivals may degrade under real
bursty patterns because burst management requires either queueing capacity or
load-shedding policies not stressed by Poisson workloads.

## Why Real Token Distributions Matter

Synthetic token length distributions (lognormal, Pareto) are parameterized
approximations. Real distributions differ in:

- **Heavy tails**: a small fraction of requests consume disproportionate
  KV-cache space, stressing memory management
- **Prompt-output correlation**: long prompts often produce long outputs;
  this correlation affects batch efficiency predictions
- **Multi-modal structure**: some workloads have distinct clusters (e.g.,
  short chat vs. long document analysis)

Ignoring these properties can overstate or understate the benefits of
output-length-aware scheduling policies.

## Why Synthetic SLOs Are Necessary

Neither BurstGPT nor ShareGPT contains SLO annotations. Real production
systems assign SLOs based on business logic not present in public traces.

The synthetic SLO augmentation in this project:
- Assigns request classes (interactive, standard, batch) with configurable
  weights
- Sets SLO deadlines as `arrival_time + slo_slack` per class
- Labels all synthetic fields explicitly in trace metadata

**Research implication**: Results that depend on SLO-aware scheduling (EDF,
SLO slack score, etc.) reflect the synthetic SLO distribution, not any real
production SLO. Claims about SLO violation rates must acknowledge this.

## Types of Workloads

### Real Trace Replay
Replays an actual recorded trace. Arrival times and token counts reflect
real system behavior. SLOs are synthetic. This is the most faithful
evaluation of arrival and token distribution effects.

### Scaled Trace Replay
Compresses or expands interarrival times by a factor. Useful for stress
testing at higher or lower loads without collecting new traces. Assumes
request content is independent of arrival rate (often true in practice).

### Synthetic Workloads
Generated from parameterized distributions (Poisson, lognormal, Pareto).
Fully controlled. No real data required. Good for ablation studies and
reproducibility, but cannot capture real distributional structure.

## Current Limitations

- **Single GPU**: The simulator models one GPU. Multi-GPU load balancing
  effects are not captured.
- **No preemption**: Requests run to completion once admitted. Preemptive
  scheduling (e.g., pause-and-resume for KV-cache eviction) is not modeled.
- **No context switching**: Partial decode state is not tracked.
- **Synthetic SLOs**: All deadline/priority fields are simulated.
- **No network latency**: Input/output transfer times are not modeled.
- **Token generation is deterministic**: Actual output length is fixed at
  request creation time (no speculative generation, no rejection sampling).

## Research-Safe vs. Unsafe Claims

### Safe claims
- "Under this synthetic workload and single-GPU simulator, policy X achieves
  lower mean latency than policy Y"
- "The BurstGPT arrival pattern produces higher SLO violation rates than
  Poisson arrivals at the same mean rate under the simulator"

### Unsafe claims
- "Policy X will reduce latency in production by X%"
- "Results generalize to multi-GPU deployments"
- "SLO violation rates match real production systems"
- "Token distribution statistics from BurstGPT are representative of all
  LLM production workloads"
