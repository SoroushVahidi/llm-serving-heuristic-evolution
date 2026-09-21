# Declarations (verbatim from the manuscript; for the Elsevier declarations tool and the submission portal)

## Declaration of competing interest
The author declares that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Funding
This work received in-kind computational support through the Google Cloud Research Credits Program (USD 1,000 in cloud credits) and computational/tooling support from CloudRift Inc. Neither Google nor CloudRift Inc. had any role in the study design, data collection, analysis, interpretation, manuscript preparation decisions, or the decision to submit the article for publication. No monetary research grant was received from these organizations.

## Declaration of generative AI and AI-assisted technologies in the manuscript preparation process
During the preparation of this work, the author used ChatGPT (OpenAI), Codex (OpenAI), Gemini (Google), and Claude (Anthropic) in order to assist with organization, language editing, literature-search planning, and drafting analysis, figure, and test code. After using these tools, the author reviewed and edited the content as needed and takes full responsibility for the content of the publication.

## CRediT authorship contribution statement
Soroush Vahidi: Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data curation, Visualization, Writing - original draft, Writing - review & editing.
(Candidate statement for the sole author; the author must confirm it. See `credit_authorship_statement.txt`.)

## Acknowledgements
The author gratefully acknowledges Professor Ioannis Koutis for research guidance and scientific feedback. The author thanks the NJIT Wulver HPC for providing computational resources for the fresh confirmatory computation, and Anders Borum (Secure ShellFish) for providing tools that enabled remote research access. Finally, the author thanks his mother for her continued support.

## Data availability
Code, derived research artifacts, experiment configurations, and reproducibility materials for this study are available in the GitHub repository https://github.com/SoroushVahidi/llm-serving-heuristic-evolution (release performance-evaluation-v1.1.0) and are permanently archived on Zenodo [28]. The archive is a snapshot of that release: the published v1.0.0 record predates the post hoc robustness analysis, and the materials it lacks (the robustness outputs, the corrected derivative, the continuation shards, the figure code, and the claim manifest) are available in the GitHub repository only until the v1.1.0 version of the archive is published. The v1.1.0 archive contains the frozen confirmatory artifacts (native replay, pressure mapping, and fresh causal latency headroom), the post hoc robustness outputs (Section 7), a corrected derivative of two descriptive state-level columns that leaves the original frozen file unchanged, the continuation shards from which that derivative is computed, the simulator and analysis code, the figure and table scripts, and the claim manifest that checks the quoted numbers. Raw third-party workload traces are not redistributed and are obtained from their original upstream sources.

Archive: Zenodo record 22865294 (published v1.0.0: https://doi.org/10.5281/zenodo.22865294; concept DOI 10.5281/zenodo.22865293).
The v1.1.0 version DOI is assigned by Zenodo when the v1.1.0 version is published; see the release report (QUERY_8).
