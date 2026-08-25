# LLM 2026 Manuscript Plan

Date: 2026-08-24

Scope: manuscript-production scaffold using completed evidence only. This file
does not authorize new experiments, selector training, GP search, DEV/TEST/FINAL
use, Wulver/Vulver, GPU/vLLM runs, external APIs, or threshold changes.

## Frozen Thesis

LLM-serving scheduler portfolios exhibit real workload-dependent
complementarity, including under jointly varying multi-mechanism workloads.
However, translating that oracle complementarity into robust adaptive gain is
substantially harder than detecting regimes or exposing parent mechanisms. Under
frozen gates, contextual selection, shared-feature selection, hierarchical
routing, target-free support expansion, guarded ESTF/WFS composition, and
portfolio-guided structural crossover fail to convert the six-policy envelope
into transferable practical gain. Real-vLLM validation adds a systems-methodology
lesson: simulator mechanisms must be checked against native serving-engine
semantics, because a direct mapping can fail even when the real engine exhibits a
distinct reproducible scheduler-control tradeoff.

Safe compressed thesis:

> Scheduler complementarity is real in controlled and jointly varying LLM-serving
> workloads, but the tested adaptive exploitation mechanisms expose a persistent
> exploitability gap between oracle headroom and deployable scheduling gain.

Unsafe stronger thesis:

> Adaptive LLM-serving schedulers are impossible or useless in production.

## Contributions

1. **Complementarity.** A controlled six-policy LLM-serving portfolio showing
   workload-dependent winner diversity and measurable six-policy oracle headroom
   across both controlled A/B/C mechanism families and the new
   `joint_multimechanism_generalization_v1` workload.

2. **Exploitability gap.** A systematic evaluation showing that detectable
   regime structure and oracle complementarity do not automatically translate
   into robust adaptive scheduling gain. Main evidence: pooled selector,
   shared-feature selector, mechanism-target no-go, and hierarchical router
   no-go.

3. **Constructive falsification.** A stronger sequence testing target-free
   support expansion, guarded ESTF/WFS composition, exact parent representation,
   and typed structural crossover under frozen gates. These tests prevent
   post-hoc conversion of TRAIN novelty into a held-out or deployable claim.

4. **Real-system semantic validation.** A local vLLM validation showing that the
   simulator Family-B abstraction does not directly map to native vLLM 0.27.1,
   while native vLLM's `max_num_batched_tokens` control yields a distinct,
   trace-explained scheduling tradeoff.

## Recommended Title

Top recommendation:

**The Exploitability Gap in LLM-Serving Scheduler Portfolios**

Other strong options:

1. **When Scheduler Complementarity Is Not Enough for Adaptive LLM Serving**
2. **From Oracle Complementarity to Adaptive Scheduling: A Falsification Study in LLM Serving**
3. **Measuring and Falsifying Adaptive Scheduler Portfolios for LLM Serving**

Additional title options:

- **Oracle Headroom Is Not Deployable Gain: Lessons from LLM-Serving Scheduler Portfolios**
- **Complementarity, Transfer, and Semantics in LLM-Serving Schedulers**
- **Why Selecting Among Complementary LLM-Serving Schedulers Is Hard**
- **A Systems Study of the Exploitability Gap in LLM-Serving Scheduling**
- **Scheduler Portfolios for LLM Serving: Complementarity Without Robust Exploitation**
- **Frozen-Gate Evidence for the Limits of Contextual Scheduler Selection**
- **Mechanism-Aware Scheduling Portfolios for LLM Serving: Evidence and No-Go Results**
- **From Simulator Mechanisms to Serving-Engine Semantics in LLM Scheduling**

## Abstract Draft

LLM-serving systems expose multiple scheduling mechanisms: prefill/decode
control, fairness, deadline urgency, service-time ordering, and KV-cache
pressure. This makes policy portfolios attractive: if different schedulers win
under different workload conditions, an adaptive system might hope to select or
combine them. We study this premise using a six-policy LLM-serving portfolio,
mechanism-focused workloads, a broader 240-scenario joint workload distribution,
and a sequence of frozen attempts to exploit the resulting oracle headroom.
Complementarity is real: in the joint workload, 92.9% of scenarios activate at
least two elevated mechanism pressures, the epsilon-0.01 unique-winner fraction
is 59.6%, and the six-policy oracle improves mean arrival-normalized weighted
goodput from 0.314072 for the best fixed policy to 0.333106, with bootstrap 95%
CI [0.015988, 0.022433] for oracle headroom. However, contextual selection,
shared-feature selection, hierarchical routing, target-free support expansion,
guarded ESTF/WFS composition, and exact-parent typed structural synthesis all
fail their frozen transfer or practicality gates. A real-vLLM validation further
shows that simulator mechanism abstractions must be checked against native
serving-engine semantics: a direct simulator-to-vLLM prefill/decode mapping
failed, while native vLLM's batch-token budget produced a distinct reproducible
tradeoff. Together, these results identify an exploitability gap:
workload-dependent scheduler complementarity is necessary but not sufficient for
robust adaptive scheduling gain.

## Main-Text Structure

1. **Introduction**
   - Paradox: no scheduler is universally best, but oracle headroom is not
     deployable headroom.
   - End with four contributions.

2. **Problem Setting and Methodology**
   - Six-policy envelope, ANWG, best fixed versus oracle, frozen gates, split
     discipline, and claim-safety policy.

3. **Scheduler Portfolio and Workloads**
   - Six policies and mechanism matrix.
   - Public trace saturation as motivation for stress workloads.
   - Controlled A/B/C families and joint multi-mechanism workload.

4. **Complementarity and Oracle Headroom**
   - Controlled mechanism-family evidence.
   - Joint multi-mechanism generalization as the breadth result.

5. **Why Complementarity Is Hard to Exploit**
   - Contextual selection and shared-feature no-go.
   - Regime detection versus hierarchical router no-go.
   - Support expansion, guarded composition, and typed structural synthesis.

6. **Real-Serving Validation**
   - Failed direct simulator-to-vLLM transfer.
   - Semantic diagnosis.
   - Native vLLM scheduler-budget tradeoff.

7. **Implications for Adaptive LLM Serving**
   - What future adaptive systems must demonstrate.

8. **Related Work**
   - TODO verified citations only.

9. **Limitations**
   - Synthetic workloads, 0.5B real run, one vLLM version/GPU, no impossibility
     theorem, oracle headroom not deployable gain.

10. **Conclusion**

## Main-Text Science

- Public trace saturation: use to motivate mechanism-focused workloads.
- Six-policy/MF-PSD/unified 176x6 matrix: core complementarity setup.
- Joint multi-mechanism result: prominent response to workload-breadth concern.
- Selector/shared-feature/mechanism-target chain: condensed exploitability-gap
  evidence.
- Hierarchical router: main example of detectable regimes not yielding practical
  gain.
- One compact constructive-falsification table: Wulver support, guarded rule,
  typed GP.
- Real-vLLM three-stage validation: main systems-methodology result.

## Appendix Science

- Full Family-A pi0/D1/Wulver/DEV support chain.
- Wulver engineering and shard integrity details.
- Full support metrics and top-gap tables.
- Guarded composite-rule candidate ledger.
- Typed GP grammar, exact parent reproduction, smoke timing, candidate ledgers.
- Random candidate branch activation and ablations.
- Full real-vLLM server configs, prompt manifests, scheduler traces.

## Related-Work Structure

Do not invent citations. Organize verified references into:

- LLM-serving schedulers and engines: vLLM, Sarathi, Llumnix, PARS, Apt-Serve,
  batching/chunking/KV-management systems as applicable.
- Adaptive/contextual scheduling and dynamic algorithm selection.
- Algorithm portfolios and hyper-heuristics.
- GP/program-synthesis scheduling, grammar-guided GP, MAP-Elites/QDGP.
- Self-evolving/LLM-guided serving systems such as Autopoiesis if supported by
  existing notes.

Missing references are TODOs in `paper/llm2026/main.tex`.

## Limitations Draft

The workload suites are synthetic and mechanism-focused; the joint workload
increases breadth but does not represent all production traffic. The real-vLLM
experiment uses Qwen2.5-0.5B-Instruct, vLLM 0.27.1, and one RTX 5060 Ti
workstation, so it validates semantics for one concrete serving stack rather
than all engines or model scales. The no-go results falsify tested selectors,
routers, support-expansion gates, composition rules, and the first typed-GP
screen; they do not establish a universal impossibility theorem for adaptive
scheduling. Finally, oracle headroom is an analytical upper bound and should not
be read as deployable adaptive gain.

## Manuscript Scaffold

Created:

- `paper/llm2026/main.tex`
- `paper/llm2026/references.bib`
- `paper/llm2026/README.md`

Compilation target:

```bash
cd paper/llm2026
tectonic -X compile main.tex
```

Compilation status: succeeded locally with Tectonic on 2026-08-24, producing
`paper/llm2026/main.pdf`. Tectonic emitted non-fatal warnings for the empty
bibliography placeholder and one overfull line in the scaffold.

## Next Writing Task

Turn Section 4, "Complementarity and Oracle Headroom," into a full prose draft
with Figure 2 and Table 1 integrated. This is the highest-value first writing
task because it establishes the positive result before the no-go chain.
