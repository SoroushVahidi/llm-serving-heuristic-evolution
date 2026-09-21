# Finalization Query 1 Literature, Contribution, and Positioning Audit

Date: 2026-09-21  
Branch: `release/peva-v1-20260920`  
Starting HEAD: `d2b35475c4b34980a172627a2a7ac4103050c4e0`  
Scope: literature positioning only; no new experiments, no packaging, no push.

## Searches Performed

Searches covered the requested families using multiple formulations:

- LLM request scheduling, inference scheduling, continuous batching, SLO-aware scheduling, deadline-aware scheduling, service-time-aware scheduling.
- KV-cache-aware scheduling, KV cache admission/management, prefix/cache reuse, sequence capacity, memory-aware serving.
- Chunked prefill, prefill/decode disaggregation, prefill scheduling, decode scheduling, head-of-line blocking.
- Adaptive/distributed serving, request routing, load balancing, migration, heterogeneous serving, distributed serving, autoscaling.
- LLM serving simulators, trace-driven replay, workload characterization, workload generation, counterfactual performance evaluation, oracle/VBS gaps, scheduler portfolios, algorithm selection, opportunity regions.
- Recent Performance Evaluation papers on scheduling, dispatching, prediction, resource allocation, GPU/model serving, queueing, and service-system evaluation.

Authoritative sources inspected included USENIX OSDI/NSDI pages and PDFs, ACM/EuroSys metadata, MLSys/OpenReview/arXiv pages where conference metadata was inaccessible, ScienceDirect/DOI metadata for Performance Evaluation papers, and official project/repository pages only as secondary evidence.

## Manuscript Changes Made

- Added two recent, source-verified SLO-aware serving systems to the manuscript's positioning:
  - JITServe: USENIX NSDI 2026, SLO-aware LLM serving with imprecise request information.
  - AdaServe: EuroSys 2026, multi-SLO serving with SLO-customized speculative decoding.
- Revised Table 1:
  - Replaced the less central OpenTela table row with an SLO-aware/state-aware row covering SOLA, JITServe, and AdaServe.
  - Split Llumnix from SOLA because they operate at different decision levels.
  - Kept the table descriptive, not scorecard-like.
- No new scientific result or experimental comparison was added.

## Closest-Work Matrix

| Paper | Year/Venue | Primary problem | Decision level | Mechanism | Workload/evaluation | Adaptive? | Measures action disagreement? | One-step causal intervention? | Reference-conditioned headroom? | Trace replay? | System eval? | Relation to this paper | Experimental comparability |
|---|---:|---|---|---|---|---|---|---|---|---|---|---|---|
| Llumnix | 2024 OSDI | Dynamic LLM serving under heterogeneous requests | Across instances | Runtime request migration/rescheduling | Multi-instance serving evaluation | Yes | Not established from inspected source | Not established | Not established | Uses serving workloads | Yes | Mechanism whose opportunity could be measured | Conceptually close, different unit |
| SOLA | 2025 MLSys | SLO attainment via state-aware scheduling | Iteration/request within engine | Request/system-state-aware scheduling | LLM serving SLO evaluation | Yes | Not established | Not established | Not established | Uses workload-driven evaluation | Yes | SLO-aware mechanism; paper asks prior opportunity question | Conceptually close |
| Preble | 2025 ICLR | Distributed prompt scheduling with KV reuse/load balance | Across instances | KV-reuse-aware routing | Distributed LLM serving workloads | Yes | Not established | Not established | Not established | Workload evaluation | Yes | Routing mechanism; this paper measures per-state executable opportunity | Different decision level |
| LMetric | 2026 OSDI | KV-aware and load-aware request routing | Across instances | Multiplicative score from KV reuse and load indicators | Real-world chat/API/coding-agent workloads | Yes | Not established | Not established | Not established | Yes | Yes | Closest routing-score work; not per-action headroom | Conceptually close, not direct |
| JITServe | 2026 NSDI | SLO-aware serving under imprecise request information | Request/batch | Just-in-time bandwidth allocation and batch composition | Diverse realistic workloads | Yes | Not established | Not established | Not established | Yes | Yes | Strong recent SLO-aware scheduler; manuscript uses loose uniform SLOs | Different objective and SLO setting |
| AdaServe | 2026 EuroSys | Multi-SLO LLM serving | Request/token tree | SLO-customized speculative decoding | Diverse workloads | Yes | Not established | Not established | Not established | Yes | Yes | SLO-aware serving system; not an opportunity measurement | Different mechanism/objective |
| Sarathi-Serve | 2024 OSDI | Throughput-latency tradeoff | Within engine | Chunked prefill, stall-free scheduling | LLM inference system evaluation | Yes/algorithmic | Not established | Not established | Not established | Workload evaluation | Yes | Mechanism represented only abstractly by chunking axis | Not direct |
| DistServe | 2024 OSDI | Goodput-optimized serving | Prefill/decode components | Prefill/decode disaggregation | LLM serving evaluation | Yes | Not established | Not established | Not established | Workload evaluation | Yes | Related architecture; not per-action opportunity | Not direct |
| FastServe | 2026 NSDI | Low-latency inference under head-of-line blocking | Iteration/token | Preemptive MLFQ-style scheduling and memory offload | Compared with vLLM | Yes | Not established | Not established | Not established | Workload evaluation | Yes | Scheduling mechanism; no reference-conditioned opportunity | Conceptually close |
| Libra | 2026 NSDI | Unbalanced/dynamic workloads | Request partitioning | Micro-request partitioning and scheduling | LLM serving workloads | Yes | Not established | Not established | Not established | Workload evaluation | Yes | Mechanism; this paper measures opportunity before mechanism choice | Not direct |
| Mooncake | 2025 FAST | KV-cache-centric serving architecture | KV placement/system architecture | Storage-for-computation KV architecture | Chatbot serving evaluation | Yes | Not established | Not established | Not established | Workload evaluation | Yes | KV pressure relevance | Not direct |
| MorphServe | 2025 arXiv | Workload-aware serving under resource pressure | Runtime resource/cache | Quantized layer swapping and KV resizing | Workload-aware serving evaluation | Yes | Not established | Not established | Not established | Workload evaluation | Not fully source-verified as systems venue | KV capacity relevance | Not direct |
| Jaillet et al. | 2025 arXiv/author manuscript | Online scheduling with KV-cache constraints | Within engine | Algorithms and hindsight optimal benchmark | Public LLM inference dataset, theoretical analysis | Online algorithms | Not in this paper's sense | Not established | Hindsight benchmark, not reference-conditioned one-step headroom | Empirical component | Analytical/empirical | Closest analytical counterpart | Different estimand |
| Vidur | 2024 MLSys | LLM inference simulation | System simulation | Operator-profile latency model | Simulator validation | No scheduler contribution | No | No | No | Simulation framework | Validation | Simulator/fidelity context | Not a baseline |
| ServeGen | 2026 NSDI | Workload characterization/generation | Workload model | Production trace characterization and generator | Production LLM workloads | No | No | No | No | Generates workloads | Case studies | Workload-methodology complement | Not a scheduler |

Conservative interpretation: "Not established" means the inspected authoritative source did not establish that the work performs that measurement; it is not a claim of absence beyond the inspected material.

## Four to Six Closest Papers

1. **Jaillet et al., Online Scheduling for LLM Inference with KV Cache Constraints.**  
   Solves an analytical online-scheduling problem under KV-cache memory limits. Closest because it formalizes KV-constrained scheduling and uses a hindsight benchmark. Difference: this paper is empirical, replay-based, reference-conditioned, one-step, and asks whether the visited states expose useful alternative actions.

2. **SOLA.**  
   Solves SLO-attainment scheduling with request/system state. Difference: SOLA proposes a scheduler; this paper measures whether scheduler choice has actionable headroom before proposing a selector. Complementary, not competing.

3. **Llumnix.**  
   Solves cross-instance dynamic scheduling through request/state migration. Difference: Llumnix changes the serving system; this paper evaluates per-step opportunity under a fixed reference and simple policy set.

4. **Preble/LMetric.**  
   Both concern KV-aware/load-aware routing across instances. Difference: they design routing scores; this paper asks whether alternative executable actions appear and have one-step causal value.

5. **JITServe/AdaServe.**  
   Recent SLO-aware systems solve practical serving under heterogeneous or imprecise SLOs. Difference: the present manuscript intentionally uses loose uniform SLOs, states that those rules receive no informative signal, and does not claim SLO-system gains.

6. **ServeGen/FineServe/Nixon et al. workload papers.**  
   These works expand production workload characterization. Difference: the manuscript uses production-derived traces for replay but contributes an action-opportunity measurement, not a workload dataset.

## Important Works Added or Repositioned

Added to manuscript:

- JITServe, NSDI 2026, because it is a recent and authoritative SLO-aware LLM serving system.
- AdaServe, EuroSys 2026, because it is recent, accepted, and directly relevant to multi-SLO serving.

Repositioned:

- Table 1 now distinguishes cross-instance rescheduling, routing, SLO-aware/state-aware serving, workload generation, KV architecture, and analytical KV scheduling.

## Considered but Not Added

- Kairos / "Taming Request Imbalance": relevant SLO-aware disaggregated scheduling, but only arXiv was found; JITServe/AdaServe/SOLA cover the SLO-aware point without overloading the manuscript.
- FineServe: important 2026 workload characterization, but arXiv only and not needed because ServeGen plus Nixon et al. already cover production workload evolution/generation.
- UNAS: relevant urgency/fairness-aware scheduling, but less central and would add another scheduler citation without changing the contribution distinction.
- PKAS and locality/fairness-aware scheduling papers: relevant to KV/cache-aware scheduling, but not closer than Preble/LMetric/Jaillet for the manuscript's core question.
- ReaLLM/LLMServingSim: simulator-related, but Vidur is already the closest high-visibility simulator reference used for calibration/fidelity context.
- POSUM and broader portfolio scheduling: conceptually related, but the existing Rice/Gomes/SATzilla/AutoFolio citations are sufficient for SBS/VBS/algorithm-selection framing.

## Claim-by-Claim Citation Audit

| Location | Claim | Citation(s) | Source support | Action |
|---|---|---|---|---|
| Intro | Recent systems adapt scheduling/routing to request, cache, load state | Llumnix, Preble, LMetric | Supported by USENIX/ICLR/OSDI sources | Split into focused citation group |
| Intro | Other systems optimize serving under request and SLO state | SOLA, JITServe, AdaServe | Supported by MLSys/OpenReview/arXiv and USENIX/EuroSys metadata | Added JITServe/AdaServe |
| Related Work | Orca/vLLM underlie iteration scheduling and paged KV cache | Orca, vLLM | Supported | No change |
| Related Work | Sarathi/DistServe/Splitwise/FastServe change batching, separation, or preemption | OSDI/ISCA/NSDI metadata | Supported | No change |
| Related Work | Llumnix migrates/reschedules requests | Llumnix | Supported by USENIX page | No change |
| Related Work | Preble/LMetric route using KV-reuse/load indicators | Preble, LMetric | Supported | No change |
| Related Work | SOLA/JITServe/AdaServe are SLO/state-aware | SOLA, JITServe, AdaServe | Supported | Added and placed close to claim |
| Related Work | Mooncake/Strata/MorphServe address KV capacity/cache | FAST/arXiv sources | Supported sufficiently for high-level relation | No change |
| Related Work | ServeGen characterizes/generates production workloads | USENIX source | Supported | No change |
| Related Work | Jaillet provides analytical online KV-cache scheduling and hindsight benchmark | arXiv/author manuscript | Supported | No change |
| Related Work | PEVA neighboring papers cover assignment, dispatching, prediction, serverless training | DOI/ScienceDirect metadata | Supported | No change |
| Algorithm-selection paragraph | SBS/VBS terminology from algorithm portfolios/selection | Rice, Gomes, SATzilla, AutoFolio | Supported | No change |

## Contribution Audit

The manuscript's novelty claims are internally consistent and describe one core contribution: a staged performance-evaluation methodology for measuring whether a fixed scheduler reference encounters executable alternatives, whether those alternatives causally improve latency under one-step intervention, and how opportunity depends on reference, workload, and resource regime.

Genuinely new or claimed new:

- The staged action-opportunity evaluation in this LLM-serving setting.
- Distinguishing executable action disagreement from whole-trace policy-score diversity.
- One-step counterfactual measurement on fresh disagreement states.
- Reference-conditioned distribution/concentration analysis of headroom.

Existing methods used:

- Trace replay.
- Simple scheduling heuristics.
- KV and active-sequence capacity interventions.
- SBS/VBS/oracle concepts from algorithm selection.
- Bootstrap/window-clustered uncertainty.
- Simulator and vLLM correspondence probe.

Empirical contributions:

- Native no-action-opportunity behavior with abundant resources.
- Capacity-induced disagreement and light-load arrival null result.
- Fresh-window causal opportunity on Azure traces and BurstGPT fresh-window null.
- Small median headroom despite positive state-weighted mean.
- High concentration in one window.
- Reference and portfolio dependence.

System contributions:

- None in the sense of a new deployed scheduler or serving system. The manuscript now states this explicitly and preserves the limitation.

## Hard Novelty Question

Why is this not merely "take several known schedulers, find when they disagree, try their actions, and report the result"?

The non-trivial part is the staged estimand and its controls:

- It separates policy-score diversity from executable action diversity; prior portfolio/SBS/VBS framing operates at whole-instance or whole-trace score level.
- It separates action disagreement from causal benefit; most scheduler papers evaluate end-to-end mechanisms rather than the opportunity surface visited by a fixed reference.
- It conditions every quantity on the reference, workload, and resource regime, avoiding a universal scheduler claim.
- It uses fresh windows and pre-specified one-step intervention after support discovery, reducing the risk of treating exploratory support discovery as confirmatory outcome evidence.
- It reports distribution, concentration, and harmful alternatives rather than only an oracle mean.

The search did not find a prior LLM-serving paper that performs the same staged action-disagreement -> one-step causal headroom -> concentration analysis. The closest conceptual ancestors are algorithm selection/VBS work and Jaillet et al.'s hindsight benchmark, but neither establishes this paper's visited-state, reference-conditioned, one-step replay estimand.

## Existing Experimental Comparisons

Candidate existing comparisons in the repository:

- Six-policy SBS/VBS and selector studies from earlier manuscripts: **do not use** in the current main paper, because they answer a different whole-scenario/selector question and could blur the Query 13 contribution.
- Small-chunk versus full-prefill faithful-window comparison: **use only as context**, already present in Discussion to show score diversity need not imply per-state action diversity.
- vLLM bounded full-versus-chunked pressure probe: **use in main paper**, already present as bounded correspondence evidence, not latency validation.
- External baseline/selector documents in `docs/`: **do not use**, because they belong to older or superseded experiments and would introduce new claims outside this manuscript's population.

No new baseline experiment was run or recommended.

## Recency Distribution

After adding JITServe and AdaServe:

- <=2022: 5
- 2023: 2
- 2024: 5
- 2025: 11
- 2026: 11

The bibliography is recent enough for submission. No major 2025-2026 work appears missing in a way that would make the manuscript outdated; several additional 2026 arXiv works were considered but not added to avoid turning Related Work into a catalogue.

## Performance Evaluation Positioning

The manuscript naturally aligns with Performance Evaluation through:

- workload replay;
- scheduling/resource evaluation;
- causal measurement of scheduler actions;
- window-clustered uncertainty;
- distribution and concentration of effects;
- bounded interpretation of a simulator against a real-engine probe.

No journal-name praise was inserted.

## Title Assessment

The title remains accurate. "Action Opportunity" describes executable alternative support; "Causal Headroom" is defined in the paper and does not conflict with inspected LLM-serving terminology. No title change is recommended.

## Validation Plan

After edits:

- Rebuild PDF.
- Regenerate/check claim manifest if the source hash changes.
- Run manuscript tests and robustness/correction tests.
- Check abstract word count, keyword count, figure/table count, page count.
- Visually inspect affected Related Work/Table 1 pages.

