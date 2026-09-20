# FGCS Manuscript Claim Map V1

Audit date: 2026-09-20

Canonical manuscript: `paper/llm2026/main.tex`

Canonical rewrite base: `ba739597a076e7468b957ccd1080054a31b15cd8`

## Central thesis

Adaptive LLM-serving value is regime dependent. Faithful production-derived
replay can be scheduler-action null under abundant resources; KV or
active-sequence pressure can create canonical executable alternatives; and
fresh one-step replay can show modest SBS-relative latency headroom inside a
subset of those disagreements. Action opportunity, causal value, and later
predictability are separate gates.

## Evidence ledger

| Claim | Evidence | Scope | Manuscript treatment |
| --- | --- | --- | --- |
| Native replay is action-null | Phase A result summary | Azure code 0/99,992; Azure conversation 0/461,985; BurstGPT 0/440,461 | Main result |
| Pressure creates disagreement | Phase B V2 result and transition map | 1,080 conditions; KV/active caps bind; arrival scaling through 8x remains null | Main result |
| Fresh support transfers | Fresh support artifacts and non-overlap audit | Untouched windows; selected causal universe excludes BurstGPT | Main result and limitation |
| Fresh latency headroom exists | `FRESH_LATENCY_CAUSAL_RESULT_V1.json` | 720 states; 590 beneficial; 1.996 ms mean; clustered CI positive | Principal causal result |
| ANWG can saturate | Phase-D V1 objective audit and sensitivity tests | Terminal ANWG=1 and zero ANWG advantage despite latency changes | Objective-sensitivity result |
| Real pressure/action mechanism partially transfers | Real-vLLM validation package | R1/R2 support; no R3; direct simulator reversal no-go | Bounded systems result |
| Learned predictability is established | None | No selector claim | Explicitly not claimed |

## Supporting commits and artifacts

- Phase A: result artifact records execution head `f394d73119d5d4039ee2e1dc38482071db2bd493`.
- Phase B V2: result artifact records execution head `4caed7fee92e052ec1de13d983ea3f8c03b523f1`.
- Fresh latency: result artifact records execution head `38e02bcdaeb8f9cec2289b05dea9a1406318d14f` and frozen canonical result lineage is `b196c3e27d1e5a0d43fd0664e24e51090ec8e442`.
- Real-vLLM package: commit `ba739597a076e7468b957ccd1080054a31b15cd8`.
- Latest literature audit: `docs/current/LATEST_LITERATURE_DIFFERENTIATION_AUDIT_V1.md`, updated in this rewrite to distinguish ServeGen, LMetric, OpenTela, longitudinal trace work, and modern adaptive serving systems.

## Claim boundaries

The manuscript does not claim: production deployment improvement, universal
modern-scheduler coverage, real-system causal latency headroom, learned
selector generalization, or acceptance probability. P6 is an explicit
default-relative portfolio. Fresh $P(B_{\mathrm{LAT}}\mid D)$ is conditional
on disagreement and must not be reported as a rate over all decisions.
