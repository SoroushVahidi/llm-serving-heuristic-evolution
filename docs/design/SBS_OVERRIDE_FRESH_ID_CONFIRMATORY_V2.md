# SBS Override Fresh ID Confirmatory V2

This document freezes the outcome-blind readiness hardening for the frozen
`FINAL_CONFIRMATORY_SELECTOR_V2`.

The authoritative V2 preregistration is the git object at:

`daf093acc552932f7ac7a7e74263956ed4af5c74`

It is preserved verbatim at:

`experiments/sbs_override_conservative_selector_dev_v2/run_v2/PREREGISTERED_SEARCH_DESIGN_V2_CANONICAL_DAF093A.json`

The current `PREREGISTERED_SEARCH_DESIGN_V2.json` is runtime provenance from
the development run and is not the authoritative preregistration record.

Stage A is outcome-blind. It emits exactly one frozen decision per clean fresh
disagreement state using the frozen V2 selector and structural manifests only.
It does not read terminal labels, `Q_SBS`, `A_SBS`, realized gains, or outcome
signs.

Stage B is the later one-shot evaluator. It must load the frozen Stage-A
prediction hash and protocol hash before reading fresh labels. The primary
endpoint is the state-weighted mean realized gain over the canonical 2,862-state
population. The denominator is always 2,862, with abstentions assigned zero
gain.

Primary confirmation is declared only when the observed mean is positive and
the lower endpoint of the frozen two-sided 95% scenario-clustered percentile
bootstrap confidence interval is strictly greater than zero. Secondary metrics
cannot override this primary verdict.
