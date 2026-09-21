# QUERY_12 — Final coherence, fairness, and compression pass (record)

Start: branch `release/peva-v1-20260920`, HEAD `5887cb4`, `main.tex` SHA-256 `7d2ee7ff…`, 34 pages, clean tree.
No experiment, simulation, vLLM run, or new baseline was executed. All numbers are recomputed from completed artifacts.

## 1. "Preregistered" — what actually existed

| Question | Evidence | Finding |
|---|---|---|
| File | `experiments/fresh_production_latency_headroom_confirmatory_v1/PREREGISTRATION_V1.json` (+ `METRIC_PROTOCOL_V1`, `CAUSAL_PROTOCOL_V1`, `SUPPORT_PROTOCOL_V1`) | One commit each, unchanged since |
| Commit | `b4e6c60` "Freeze fresh latency headroom confirmation design", 2026-09-19 22:12:26 −0400 | Support map frozen `38e02bc` 23:00; results `b196c3e` 2026-09-20 00:26:14 (2 h 14 min after the protocol) |
| Registry | grep for OSF / AsPredicted / registered-report: none in docs, protocol JSON or manuscript | **No external registration** |
| Visibility | GitHub API: repository is public **now**; earlier session notes say it was private when created; visibility history is not exposed, and the events API no longer returns the push of `b4e6c60` | Push time before outcomes **cannot be verified independently**; commit timestamps are author-controlled |
| Could the author change it? | Yes (author-controlled repo). Content hashes pin the artifacts (`ARTIFACT_HASHES_V1.json`) | Integrity is checkable by hash, not by third-party time-stamp |
| Post-outcome change | `BOOTSTRAP_CLUSTER_KEY_CORRECTION_20260920.md` (commit `a8fd735`, 17:38): first bootstrap keyed on bare `window_index` (26 clusters), corrected to `(source_dataset, window_index)` (36) | **Was not disclosed in the manuscript**; point estimate/decision unchanged (CI [0.185, 3.497] -> [0.191, 3.546] ms) |

Decision: the manuscript says **pre-specified** (protocol committed before outcomes, no external registry, author-controlled
repository); "preregistered" survives only in the sentence that explains the choice. The cluster-key correction is now
disclosed in Methods 3.4 and registered as manifest claim `prespec.protocol_timeline_and_cluster_correction`.
Historical file names keep "PREREGISTRATION"; the reproducibility guide records the terminology decision.

## 2. Policy information audit (from `src/llmserveopt/policies/*`, `core/types.py`)

Every policy receives an `ObservableState`: per waiting request `arrival_time, prompt_tokens, predicted_output_tokens, slo_deadline, priority, class_id`
(`actual_output_tokens` is absent by construction) and per GPU capacities and KV occupancy. In this study `predicted_output_tokens == actual output length`
(`PHASE_A_REPLAY_SEMANTICS_V1.json`), `slo_deadline = arrival + 1000 s`, `priority = 1.0`, `class_id = "default"`.

| Policy | Information used | Uses (true) output length? | Deadline? | Priority? | KV estimate? | Oracle information? | Disclosed now? |
|---|---|---|---|---|---|---|---|
| `full_prefill` | arrival order; prompt tokens (feasibility) | no | no | no | prompt-only feasibility, no future growth | none | yes (Methods 3.2) |
| `chunked_prefill_small` | same rule as full; 64-token execution chunk | no | no | no | same | none | yes |
| `estimated_service_time_first` (ESTF) | α·prompt + β·predicted output; ties by deadline, priority, id | **yes, exact** | tie-break only | tie-break only | no | exact service estimate | yes |
| `least_laxity_first` | deadline − now − service estimate | **yes, exact** | yes (1000 s, uninformative) | tie-break | no | exact service estimate | yes |
| `weighted_fair_share` | class shares × priority ÷ estimated steps | **yes, exact** | no | yes (constant) | no | exact service estimate | yes |
| `kv_constrained_online` (SBS) | KV footprint = prompt + 0.25·predicted output; post-admission KV utilisation ≤ 0.82 unless slack < 0.25 s | **yes, exact (ranking only)** | urgency exemption never fires | yes (constant) | yes (current KV + prompt; no future growth) | exact length in ranking | yes |

Name: kept "estimated-service-first"; Methods now says the estimates are exact, that four rules use them, and that the rules are
oracle-informed heuristics, not deployable schedulers. Limitations repeats it once.

## 3. Fairness of the six-policy comparison

| Aspect | Verified from | Result |
|---|---|---|
| Same state | `scripts/industry_realism_action_opportunity_phase_a_v1.py` (`actions = {pid: shadow[pid].select_action(deepcopy(state))}`) | all six queried on the same live SBS state |
| Same future arrivals / hardware / simulator | protocol `intervention` block (`preserve_future_arrivals`, `preserve_random_state`, `preserve_pressure_configuration`) | yes |
| Same candidates and canonicalisation | `canonical_action()` applied to every policy | yes |
| Same continuation / horizon | protocol: SBS continuation, same request population `R_s`, invalid otherwise | yes |
| Asymmetric information | code | none between policies **given the state**; asymmetry is in *use*: the two prefill variants ignore lengths that four others use. Relative to deployment, four policies are oracle-informed |
| Realism | code | not equally realistic: FIFO variants are deployable; the other four assume perfect length prediction |

## 4. SBS wording

Old: "best single policy on an earlier, unrelated benchmark". The joint-240 benchmark is the same project, simulator and six policies
(`experiments/joint_multimechanism_generalization_v1`, `joint240_*`, completed 2026-08-25), so "unrelated" overstated independence.
New: "highest mean goodput among the six policies on an earlier benchmark of 240 multi-mechanism scenarios from the same project, run with the same
simulator and policy implementations on scenarios other than these trace windows; fixed before the fresh evaluation, not selected, tuned, or
changed using fresh outcomes; best only among these six on that benchmark and not claimed optimal or strong." (manifest claim `reference.sbs_selection` recomputes the argmax.)

## 5. Cross-section consistency matrix (final state)

| Topic | Canonical statement | Abstract | Intro/RQ/Contrib | Methods | Results | Discussion/Limitations | Conclusion | Captions |
|---|---|---|---|---|---|---|---|---|
| A BurstGPT | 360/360 valid, 0 disagreement, 0 binding, max queue 3; never queued enough | yes | yes | trace-order windows | 5.1 mechanism | yes | (Azure only) | Table 2 = original windows |
| B arrival scaling | lightly loaded, negative result | yes | yes | — | 4.2, 5.1 numbers | yes | yes | — |
| C physical binding | waiting prompt > free KV / no free slot | — | — | defined | Table 2 "first binding", 6.4 | 8.2, 8.3 | — | Table 2 |
| D SBS reserve | 0.82 hard admission cap (urgency never fires) | "reference-conditional" | "conditional" | defined | 6.4 | yes | yes | — |
| E beneficial states | 590/720, sign of effect | yes | yes | Eq. 3 | 5.2 | yes | yes | Table 3 |
| F magnitude | median 0.20 ms, mean 2.00 ms | yes | yes | — | 5.2, 6.1 | yes | yes | Fig 4 |
| G relative | median 0.21 %, 12.8 % > 10 % | (none) | — | Eq. defined | 5.3 Table 4 | 8.1 | — | Table 4 |
| H concentration | w11 88.4 %, eff. windows 1.27 | 88 % | 88 % | — | 6.3 | yes | yes | Fig 4 |
| I VBS | illustration only; six-policy gap not measured | — | Sec. 1 motivation | — | — | 8.2 | — | — |
| J diversity | 1.15 alternatives; ESTF≡WFS, full≡chunked | — | — | 3.2 | — | Limitations | — | — |
| K causal | simulated counterfactual | — | — | 3.3 "local counterfactual" | — | 8.2 | — | — |
| L vLLM | bounded probe, no magnitude validation | yes | yes | — | 7 | 8.3 | — | — |
| M realism | production-derived traces + controlled caps | yes | yes | 3.1 | — | Limitations | — | — |
| N pre-specification | protocol committed 2 h 14 min before results; no registry; one post-outcome fix | "pre-specified" | "pre-specified" | 3.4 | 5.2 | — | "pre-specified" | Table 4/5 captions |

Issues found and resolved this pass: **MAJOR** "preregistered" without external registration (N); **MAJOR** undisclosed post-outcome bootstrap key correction (N);
**MAJOR** "unrelated benchmark" (D); **MINOR** `Q_SBS` defined but never used (removed; `L_CF`, `L_SBS` defined at first use); **MINOR** one unnumbered
display equation for `P(B|D)` (now inline); **MINOR** "reference-dependent"/"reference-conditional"/"reference dependence" drift (kept as adjective/noun pair); no CRITICAL contradictions remained after QUERY_11.

## 6. Structure, compression, appendix

| Before (5887cb4) | After |
|---|---|
| Intro; Related Work; Methods (6 subsections, retrospective ANWG/secondary/post hoc order); Results: Native; Results: Pressure; Results: Fresh; Results: Robustness; Results: vLLM; Discussion (6 subsections); Conclusion (3 dense paragraphs) | Intro; Related Work; Methods (Workloads and simulator; Scheduling policies and reference; Action opportunity and one-step headroom; Endpoints and pre-specification; Statistical analysis); Where Disagreement Appears; Causal Headroom on Untouched Windows; Concentration, Sensitivity, and Reference Dependence; A Bounded vLLM System Correspondence Probe; Discussion (5 subsections); Conclusion; Appendix A |

Pages 34 -> 31. Moves/removals: interval diagnostics -> Appendix A; Table 7 (vLLM correspondence) folded into two sentences; Figure 3 (six-point
transition line, redundant with Table 2/Fig 2) and Figure 4 (per-regime headroom, redundant with Table 3/regime map) removed; pipeline figure moved to the Introduction next to Eq. 1;
Discussion no longer re-quotes Section 6 numbers; captions shortened; limitations shortened; "Directions" paragraph and roadmap trimmed. The class's `\appendix` forces a
page break, so the appendix uses an unnumbered "Appendix A." heading placed after the Conclusion. Font, margins, row spacing and figure sizes are unchanged.

Tests changed deliberately (structure/wording): section titles and slices; Discussion subsection list and number pins (interpretive numbers kept, table-duplicated
ones dropped; still checked in Section 6 tests and the manifest); table count 7 -> 6 and figure count 6 -> 4; "fixed" instead of "frozen"; "pre-specified"; quartiles no longer pinned in prose.
Removed test: `test_figure_3_axis_direction...` (figure removed). Manifest 61 -> 60 claims (dropped `figure3.*`, added `prespec.*`).

## 7. Literature terminology and citation checks

Terminology (vLLM `--max-num-seqs` = "maximum number of sequences processed in a single iteration"; `--max-num-batched-tokens`; `--enable-chunked-prefill`;
Sarathi-Serve "chunked-prefills", "stall-free scheduling"; Rice/Gomes-Selman "algorithm selection/portfolio", "virtual best solver"): the manuscript's
"prefill", "chunked prefill", "KV cache", "active-sequence cap", "admission", "virtual best solver" match. "Single best scheduler" adapts "single best solver".

Edited literature sentences re-checked against sources: Llumnix (arXiv 2406.03243 abstract), SOLA (MLSys'25 page), Preble (ICLR'25 page), Jaillet et al. (arXiv 2502.07115),
Mitzenmacher-Shahout (arXiv 2503.07545; Stochastic Systems 15(3):195-219), Vidur (MLSys 2024), Mooncake (arXiv 2407.00079; Kimi traces), MorphServe, Strata, LMetric §5.2,
ServeGen (8 authors, pp. 1845-1859), Libra (pp. 1243-1258), OpenTela (releases and analyses a production trace), PEVA papers (Crossref: title, authors, volume, article number; **no abstracts**, so title-level descriptions).
Citation clusters: maximum 4 keys per `\cite`.

## 8. Journal compliance (Performance Evaluation)

| Item | Status |
|---|---|
| Abstract <= 250 words | 237 |
| Keywords 1-7 | 6 |
| Numbered sections; numbered display equations | 9 sections; Eqs. (1)-(5), none unnumbered |
| Citation style | numbered (`elsarticle-num`) |
| Tables without vertical rules; figure fonts/size | tested (`test_manuscript_presentation`, `test_manuscript_figures`) |
| AI declaration / competing interest / acknowledgements before references | present |
| Data statement | present (Zenodo wording untouched pending release) |
| CRediT | **only in `submission/credit_authorship_statement.txt`, not in `main.tex`**; Elsevier expects it for each author — add before submission |
| Highlights | 5 bullets <= 85 chars, but see packaging list |
| Page limit | **NO EXPLICIT PAGE LIMIT FOUND IN THE CURRENT AUTHOR GUIDANCE CHECKED.** (The official guide URL returned HTTP 403 to the fetcher; search excerpts of it confirm the 250-word abstract and 3-5 highlights of <= 85 characters and show no length limit.) |

## 9. Packaging to-do (not done here: `submission/` and `release/` are frozen for this pass)

Stale text in the submission copies that must change when packaging is refreshed:
- `submission/performance_evaluation_highlights.txt` line 2 "KV and active-sequence caps, not load scaling, expose scheduler disagreement" (overclaims). Suggested: "Capacity caps, not lightly loaded arrival scaling, expose scheduler disagreement" (80 chars); line 3 -> "Fresh one-step interventions lower latency in most Azure disagreement states" (76); line 4 -> "Headroom is small (median 0.2 ms), concentrated, and reference-dependent" (72).
- `submission/cover_letter.txt` says "frozen preregistered artifacts" and describes a "portfolio ... can do better than the best single policy" premise; replace with "pre-specified" and the reference-conditional framing.
- Regenerate `submission/main.tex`, `main.bbl`, PDF, source zip, checksums; add CRediT section to the manuscript; Data and Code Availability still says "until the v1.1.0 version of the archive is published".
- Tracked `paper/when_does_llm_serving_scheduler_adaptation_matter.pdf` is stale.

## 10. Remaining weaknesses

Simulator uncalibrated (1 ms step); four of six policies use exact lengths; SBS untuned and its KV reserve partly explains the headroom (reserve vs ordering not separated);
Azure-only causal evidence with trace-order windows; the "pre-specified" claim rests on commit timestamps; PEVA descriptions are title-level; 31 pages is still on the long side.
