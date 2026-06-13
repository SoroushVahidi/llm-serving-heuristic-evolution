# ShareGPT Dataset Provenance

## Source

**Dataset**: ShareGPT public dataset (collected from sharegpt.com)
**Common distribution**: `ShareGPT_V3_unfiltered_cleaned_split.json`

## License

CC-BY-4.0 (note: verify current license status before use — the dataset's
license has been subject to discussion; check the source repository for the
current terms).

## Dataset Description

ShareGPT contains human-assistant conversation pairs collected from the
ShareGPT website, where users shared their ChatGPT conversations. The dataset
is widely used for LLM fine-tuning and workload analysis.

## Fields

The raw dataset contains conversation records with alternating human/assistant
turns:

```json
{
  "conversations": [
    {"from": "human", "value": "..."},
    {"from": "gpt", "value": "..."},
    ...
  ]
}
```

Role names vary: `"human"/"gpt"` or `"user"/"assistant"`.

## Derived Fields

| Field | Derivation |
|---|---|
| `prompt_tokens` | Token count of first human turn |
| `actual_output_tokens` | Token count of first assistant/gpt response |

**Tokenizer**: Uses Hugging Face `AutoTokenizer` if available (specify via
`--tokenizer` or `tokenizer_name` in config). Falls back to whitespace
tokenization if `transformers` is not installed and `fallback_whitespace=True`.

## Synthetic Fields (NOT from dataset)

The following fields are fully synthetic:

| Field | Generation Method |
|---|---|
| `arrival_time` | Generated via Poisson, bursty, or MMPP arrival process |
| `predicted_output_tokens` | lognormal noise applied to `actual_output_tokens` |
| `class_id` | randomly sampled from SLO class distribution |
| `priority` | derived from assigned `class_id` |
| `slo_deadline` | `arrival_time + slo_slack` for the assigned class |

## Manual Download

ShareGPT is not available via an automated download script due to licensing
considerations. Obtain the dataset from one of these sources:

```bash
# Common distribution (check licensing before use)
# Search for: ShareGPT_V3_unfiltered_cleaned_split.json
# Place at: data/raw/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json
```

## Download Date

<!-- Update this when you download the dataset -->
Downloaded: _not yet downloaded_
