# LLM 2026 Revision Pass 2 — Figure + Visual Integration Audit

Date: 2026-08-24  
Scope: presentation-only pass. No experiments, simulations, scientific numbers, thresholds, or conclusions changed. No commit/push.

## A. Figure 1 Source Artifacts

| Role | Path |
|---|---|
| Winner counts | `experiments/joint_multimechanism_generalization_v1/winner_summary.json` |
| Per-scenario VBS gain | `experiments/joint_multimechanism_generalization_v1/utility_matrix_wide.csv` (`oracle_gain_over_best_fixed`) |
| Mean / distribution checks | `experiments/joint_multimechanism_generalization_v1/oracle_summary.json` |
| Elevated-pressure counts | `experiments/joint_multimechanism_generalization_v1/coverage_summary.json` (`mechanism_pressure_counts`) |
| Plotting script | `paper/llm2026/scripts/plot_joint_complementarity.py` |
| Outputs | `paper/llm2026/figures/joint_complementarity.pdf`, `.png` |

Local verification (unchanged): winners 46/5/45/35/59/50; SBS mean 0.314072; VBS mean 0.333106; headroom 0.019034; CI [0.015988, 0.022433]; median 0.010622; p90 0.058040; positive 60.4%; ≥0.01 51.3%; ≥2 pressures 223/240; ≥3 175/240; gain shares 93.1% / 63.3%.

## B. Figure 1 Layout Decision

Three compact candidates were rendered from the same frozen data (`--preview-all`):

| Candidate | Layout | Preview PNG height (px) | Notes |
|---|---|---:|---|
| A | 1×3 horizontal | 757 | Equal narrow panels; readable but cramped primary panels |
| B | 2-row: (a)(b) top, shallow (c) bottom | 736 at figheight 2.35 | **Selected** — more width for (a)(b), shallow supportive (c), ≤ prior height |
| C | 2 panels only | 796 | Drops required mixed-pressure panel |

**Decision: Candidate B (compact two-row, figsize 6.75×2.35 in).**  
Candidate B at 2.35 in is slightly shorter than A while giving primary panels more horizontal space. Initial B at 3.05 in increased page count to 16; compact B restored 15 pages.

## C. Figure 1 Elements Removed

- Figure-wide internal title
- Bar-top winner counts
- In-plot SBS/VBS means, CI, median, p90, positive/≥0.01 percentages
- Panel (c) three-line percentage annotation block
- Color-dependent six-policy palette (replaced with monochrome bars)
- Legend in panel (b) (replaced with direct `mean` / `0.01` labels)

## D. Figure 1 Elements Retained

- Six policy bars with concise labels (Full, Small-chunk, ESTF, WFS, LLF, KV-constrained)
- Y-axis scenario counts with coarse ticks 0/20/40/60
- Histogram of per-scenario VBS gain over SBS
- Solid mean line and dashed 0.01 reference line
- Elevated-pressure-count distribution with subtle ≥2 shaded region
- Panel labels (a–c)

## E. Figure 1 Caption / Prose Reconciliation

- Caption rewritten to describe panels and reference Section 2 for elevated pressures; no numeric dump.
- Section 3.3 retains exact winner counts, headroom/CI, median/p90, fractions, and mixed-pressure gain shares.
- Removed redundant bundled-intervention sentence from Section 6.3 (Figure 2 domain); Figure 1 prose unchanged except caption.

## F. Figure 1 Final Effective Font Sizes

Matplotlib sizes at export; estimated at LNCS `\textwidth` inclusion (scale ≈ 0.68):

| Element | Matplotlib (pt) | Est. rendered (pt) |
|---|---:|---:|
| Tick labels | 11 | ~7.5 |
| Axis labels | 12 | ~8.1 |
| Panel labels | 13 | ~8.8 |
| Direct markers (`mean`, `0.01`, `≥2`) | 10 | ~6.8 |

Axis and panel labels meet the 7.5–9 pt target band; direct marker text is minimal and adjacent to large graphical elements.

## G. Figure 2 Source Artifacts

| Role | Path |
|---|---|
| Native budget effects + CIs | `experiments/real_vllm_mechanism_validation_v1/native_vllm_chunk_budget_semantics_probe_v1/statistical_summary.json` |
| Semantic context (prose/caption only) | `docs/current/real_vllm_prefill_decode_fidelity_diagnosis_v1_20260824.md` |
| Plotting script | `paper/llm2026/scripts/plot_vllm_semantic_validation.py` |
| Outputs | `paper/llm2026/figures/vllm_semantic_validation.pdf`, `.png` |

Verified T4096−T512 (ms): low-late TTFT −30.6 [−38.0, −22.8]; high-late TTFT −2.5 [−7.7, 1.8]; low-late E2E +16.3 [9.2, 23.5]; high-late E2E +23.3 [15.6, 31.8].

## H. Figure 2 Layout Decision

Three-panel single row with width ratios `[1.0, 1.05, 1.55]`: compact schematics (a–b) plus wider effect panel (c). Figsize 6.75×2.55 in; included at `0.92\textwidth`.

## I. Figure 2 Schematic Simplification

- **(a) Simulator:** two treatment boxes (FULL chunk large / SMALL chunk 64), shared `budget: 512 (fixed)`, arrow `chunk size varies`.
- **(b) Initial vLLM analogue:** 2×2 row matrix (chunking off/on; budget 4096/512) with `bundled intervention` tag; removed mixed-step and partial-chunk diagnostics from graphic.

## J. Figure 2 Dot-and-Whisker Implementation

Replaced bar chart with horizontal point + 95% CI whiskers on `Latency difference, T4096 − T512 (ms)`; zero reference line dashed; circles = TTFT, squares = E2E; no filled bars; no red/green encoding.

## K. Figure 2 Elements Moved to Caption / Prose

- Figure-wide title removed
- Mixed-step counts (6 vs 16), partial chunks (0 vs 21), scheduler-order prose removed from graphic
- Sign interpretation moved to caption; trace step/token statistics remain in Section 6.4 prose

## L. Figure 2 Final Effective Font Sizes

| Element | Matplotlib (pt) | Est. rendered (pt) |
|---|---:|---:|
| Schematic cell text | 9.5–10 | ~6.5–6.8 |
| Schematic panel headings | 12 | ~8.1 |
| Effect-panel ticks / axis | 11–12 | ~7.5–8.1 |
| Category labels (y-axis) | 11 | ~7.5 |

Schematic text is short (2–3 words per cell); effect panel meets axis-label readability target.

## M. Accessibility / Grayscale Result

- Figure 1: monochrome bars, gray histogram, solid/dashed reference lines, no color-only encoding.
- Figure 2: gray boxes/outlines, shape-encoded TTFT (circle) vs E2E (square), dashed zero line.
- Both figures remain interpretable in grayscale print.

## N. Figure / Caption / Prose Redundancy Result

| Fact | Primary location |
|---|---|
| Winner counts, headroom, CI, gain fractions | Section 3.3 prose |
| Panel meanings / elevated-pressure pointer | Figure 1 caption |
| Mixed-step / partial-chunk diagnostics | Section 6.3 prose |
| Bundled native intervention | Figure 2 (b) + caption |
| T4096−T512 sign rules | Figure 2 caption + Section 6.4 prose |
| Exact latency CIs | Section 6.4 prose |

Removed duplicate bundled-intervention sentence from Section 6.3; tightened Section 7 real-vLLM implication and conclusion to recover page budget.

## O. Page-Budget Changes

- Compact Figure 1 layout B (2.35 in height)
- Shortened both figure captions
- Trimmed Section 6.3 (bundled-intervention sentence → caption/figure)
- Trimmed Section 7 implication + conclusion (~6 lines)

No font, margin, line-spacing, or `\vspace` hacks.

## P. Final Page Count

**15 pages** (`pdfinfo paper/llm2026/main.pdf`).

## Q. Compile Result

```bash
cd paper/llm2026 && tectonic --keep-logs main.tex
```

**Success.** Warnings only (underfull/overfull boxes in table and bibliography); no errors.

## R. Final Visual-Quality Assessment

- Figure 1 on page 6: panels readable at 100% PDF zoom; no clipped labels; primary panels (a)(b) dominate; (c) supportive and annotation-light.
- Figure 2 on page 12: three-stage intervention structure visible without multiline boxes; effect panel shows sign, magnitude, and CI zero crossings clearly.
- No color-only dependencies observed.
- Captions concise and non-enumerative.

## S. Remaining Visual Issues

Minor (non-blocking):

- Section 6.3/6.4 and Table 2 retain pre-existing overfull hbox warnings.
- Schematic cell text in Figure 2(a–b) is slightly below the 7.5 pt ideal at final size but limited to very short strings.
- Layout preview PNGs (`joint_complementarity_layout_{A,B,C}.png`, `joint_complementarity_layout_B_h*.png`) remain in `figures/` as audit artifacts; optional cleanup before release.

## Pass-1 Integration Note

Standalone `docs/current/llm2026_revision_pass1_20260824.md` is not present in the workspace; Pass 1 terminology/definitions are integrated in `paper/llm2026/main.tex` and the Pass 1 addenda in `docs/current/llm2026_number_source_of_truth_20260824.md` and `docs/current/llm2026_claim_evidence_ledger_20260824.md`.
