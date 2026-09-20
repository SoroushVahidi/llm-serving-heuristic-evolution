# INDUSTRY_REALISM_PHASE_D_CAUSAL_HEADROOM_REPORT

Execution git head: `c2eda156bfebe721cc970c2c73c347244425889b`

Phase-D outcomes accessed: YES, only after frozen-design/hash and completeness gates.

## Campaign Completeness

- SBS references: 11328 / 11328
- Non-SBS branches: 12169 / 12169
- Total continuations: 23497 / 23497
- Integrity passed: True

## Overall State-Level Result

- P(B|D): 0
- Mean oracle headroom: 0
- Beneficial states: 0 / 11328
- All-harmful states: 0
- All-zero states: 11328
- Mixed beneficial/harmful states: 0

## Workload-Regime Results

| workload | axis | regime | P(D) | P(B|D) | mean H | positive mean | windows | CI |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| azure_2023_code | active_sequence_capacity | active_4 | 0.000998732 | 0 | 0 | 0 | 16 | yes |
| azure_2023_code | active_sequence_capacity | active_8 | 1.00008e-05 | 0 | 0 | 0 | 1 | no |
| azure_2023_code | kv_capacity | kv_16000 | 0.00595315 | 0 | 0 | 0 | 13 | yes |
| azure_2023_conv | active_sequence_capacity | active_4 | 6.69907e-05 | 0 | 0 | 0 | 11 | yes |
| azure_2023_conv | kv_capacity | kv_16000 | 3.89623e-05 | 0 | 0 | 0 | 1 | no |
| azure_2023_conv | kv_capacity | kv_8000 | 0.0218267 | 0 | 0 | 0 | 17 | yes |
| burstgpt | active_sequence_capacity | active_16 | 3.17683e-05 | 0 | 0 | 0 | 1 | no |
| burstgpt | active_sequence_capacity | active_4 | 0.000535496 | 0 | 0 | 0 | 8 | yes |
| burstgpt | active_sequence_capacity | active_8 | 0.000170991 | 0 | 0 | 0 | 2 | no |
| burstgpt | kv_capacity | kv_16000 | 2.7244e-05 | 0 | 0 | 0 | 4 | no |
| burstgpt | kv_capacity | kv_8000 | 0.000263343 | 0 | 0 | 0 | 6 | yes |

## Action-Level Diagnostics

- Positive branch fraction: 0
- Negative branch fraction: 0
- Exact-zero branch fraction: 1
- Mean branch advantage: 0
- Best branch gain: 0
- Worst branch harm: 0

## Interpretation Boundaries

These are local oracle opportunity quantities under modeled replay. They are not learned-policy gains, closed-loop gains, production latency improvements, or real-vLLM effects.

PREDICTABILITY_FOLLOWUP = NOT_CURRENTLY_JUSTIFIED
CAUSAL_HEADROOM_GATE = FAIL
