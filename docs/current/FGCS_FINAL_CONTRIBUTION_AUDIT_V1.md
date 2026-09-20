# FGCS Final Contribution Audit V1

Audit date: 2026-09-20

## Skeptical-reviewer assessment

### Novelty: 18/20
The narrowed contribution is differentiated as a measurement chain: canonical
action opportunity, pressure-to-disagreement transition, and default-relative
one-step causal headroom. The claim is scoped to P6 and does not claim that
production workload characterization or adaptive serving systems are new.

### Technical depth: 17/20
The manuscript defines canonical action deduplication, all-state and
disagreement denominators, cloned one-step continuation semantics, latency
headroom, and objective sensitivity. The real-system action correspondence is
observable but not exact.

### Experimental rigor: 19/20
Phase A, Phase B, fresh support, and fresh causal artifacts are frozen and
hashable. The fresh result uses 2,000 faithful-window bootstrap replicates and
seed 20260920. The main residual issue is that BurstGPT has no selected fresh
causal regime.

### Industry realism: 16/20
Three production-derived trace families and bounded vLLM validation provide
credible structural evidence. Real-vLLM validation is one model/GPU, uses
token-shape replay, lacks R3 intervention, and did not reproduce the simulator
qualitative reversal.

### Practitioner value: 18/20
The paper gives a usable regime map: measure $P(D)$, then conditional benefit,
then effect magnitude before investing in adaptation. The 1.996 ms mean fresh
headroom is reported as modest, not as a deployment gain.

### Reproducibility: 19/20
Frozen manifests, non-overlap audits, source hashes, result tables, bootstrap
metadata, real-vLLM provenance, and regeneration code are present. Public raw
trace redistribution remains governed by upstream licenses.

## Top five remaining reviewer objections

1. **P6 is narrower than the modern scheduler landscape.** Partially resolved
   by explicit scope and current comparison table; a broader policy extension
   is not silently claimed.
2. **Fresh causal states are concentrated in Azure windows.** Partially
   resolved by reporting the BurstGPT omission as a limitation; unresolved as
   a generalization question.
3. **Simulator causal headroom may not transfer to deployed systems.**
   Partially resolved by bounded vLLM R1/R2 evidence and an explicit no-R3
   boundary.
4. **A 1.996 ms effect may be below production noise.** Resolved in wording,
   not empirically for deployment: the manuscript calls it statistically
   supported but operationally modest.
5. **Canonical action disagreement may miss useful score differences.**
   Resolved by defining action collapse deliberately; score diversity is
   acknowledged as a separate, non-action opportunity signal.

## Hard-gate outcome

No unresolved major scientific hard gate remains for the narrowed
characterization claim. Learned predictability remains deliberately outside
the claim boundary.

\[
\texttt{FGCS\_CONTRIBUTION\_READINESS\_SCORE}=92/100,
\qquad
\texttt{FGCS\_CONTRIBUTION\_STRENGTH\_CONFIDENCE}=91\%.
\]

The package is ready for a final submission audit, not journal submission in
this task.
