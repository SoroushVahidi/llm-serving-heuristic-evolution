# Loss-Case Summary — Selector vs. Fixed Baselines

**Total loss cases:** 149 (request x baseline pairs where the selector lost)

## Losses by baseline

| Value | Count |
|---|---|
| fifo | 31 |
| shortest_output_first | 31 |
| vllm_direct | 30 |
| estimated_service_time_first | 20 |
| edf | 19 |
| least_laxity_first | 18 |

## Losses by regime

| Value | Count |
|---|---|
| overloaded_mixed_priority | 62 |
| steady_moderate | 51 |
| bursty_tight | 36 |

## Losses by prompt bucket

| Value | Count |
|---|---|
| short | 74 |
| medium | 42 |
| long | 33 |

## Losses by target output length

| Value | Count |
|---|---|
| 64 | 63 |
| 128 | 59 |
| 256 | 27 |

## Losses by concurrency

| Value | Count |
|---|---|
| 4 | 45 |
| 1 | 36 |
| 8 | 36 |
| 2 | 32 |

## Losses by priority

| Value | Count |
|---|---|
| 2.0 | 91 |
| 1.0 | 57 |
| 3.0 | 1 |

## Losses by selector-chosen sub-policy

| Value | Count |
|---|---|
| edf | 48 |
| weighted_shortest_processing | 34 |
| fifo | 30 |
| scorpio_style_slo_guard | 19 |
| admission_control | 18 |

## Losses by reason category

| Value | Count |
|---|---|
| long-output underestimation | 146 |
| different ordering | 2 |
| high-priority request missed | 1 |

## Top 5 recurring loss reasons

- **long-output underestimation**: 146 cases (98.0%)
- **different ordering**: 2 cases (1.3%)
- **high-priority request missed**: 1 cases (0.7%)

Reason categories are assigned by a deterministic heuristic over recorded fields (see `_classify_loss_reason`), not a causal analysis; treat as a triage label, not ground truth.
