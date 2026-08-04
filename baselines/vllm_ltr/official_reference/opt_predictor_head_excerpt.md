# Official reference: OPT predictor head excerpt

**Not executable. Not imported by any adapter code.** This is a literal
citation of the pinned-commit source used to verify the ranking-model
architecture reproduced (independently of vLLM internals) in
`adapter/checkpoint_loader.py`.

- **Source:** `vllm/model_executor/models/opt.py`
- **Repository:** https://github.com/hao-ai-lab/vllm-ltr
- **Pinned commit:** `13bbf6ff3dab661791d41362551b089e5f77c91c`
- **License:** Apache License 2.0 (inherited from the source repository;
  file header additionally credits HuggingFace/Fairseq/vLLM-team as prior
  authors of the base OPT implementation this class is adapted from)
- **Lines:** 361–408 (class body; imports/surrounding file elided)

```python
class OPTForSequenceClassification(nn.Module):

    def __init__(
        self,
        config,
        linear_method=None,
    ):
        super().__init__()
        self.config = config
        self.linear_method = linear_method
        self.model = OPTModel(config, linear_method)
        self.num_labels = config.num_labels
        self.score = nn.Linear(config.word_embed_proj_dim, self.num_labels, bias=False)
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.sampler = Sampler()

    def forward(self, input_ids, positions, kv_caches, attn_metadata):
        hidden_states, pred_scores = self.model(input_ids, positions, kv_caches, attn_metadata)
        return hidden_states, pred_scores

    def compute_logits(self, hidden_states, sampling_metadata):
        logits = self.logits_processor(self.score.weight, hidden_states, sampling_metadata)
        if self.num_labels > 1 and logits is not None:
            logits = logits[:, :self.num_labels].argmax(dim=-1, keepdim=True).float()
        return logits

    def sample(self, logits, sampling_metadata, pred_scores):
        assert logits.dim() == 2
        next_tokens = self.sampler(logits, sampling_metadata, pred_scores=pred_scores,
                                    aux_model_scores=logits[:, 0].tolist())
        return next_tokens
```

## What this establishes

1. **Backbone:** `OPTModel` — the *standard* HuggingFace-compatible OPT
   transformer (same as used for ordinary causal-LM OPT checkpoints); no
   ranking-specific architectural change to the backbone itself.
2. **Head:** a single `nn.Linear(word_embed_proj_dim, num_labels, bias=False)`
   applied to the backbone's final hidden state — no pooling layer, no MLP,
   no dropout at inference. This is loadable and runnable with **plain
   HuggingFace `transformers`**, without any vLLM-specific machinery
   (`Attention`, `ColumnParallelLinear`, paged KV cache, `LogitsProcessor`,
   `Sampler` are all vLLM *inference-engine* plumbing for serving efficiency,
   not part of the ranking computation itself — a standard
   `transformers.OPTModel(...).forward(...)` forward pass over the same
   `input_ids` produces the same last-hidden-state given the same weights).
3. **Score semantics:** when `num_labels > 1` (the classification config
   variant, `train/configs/config_prefill_opt_classify.txt`), the head's
   `num_labels` logits are **not** used as a probability distribution — they
   are reduced via `argmax` to a single bin index, cast to `float`. That bin
   index (not a softmax probability, not a raw regression value) is what
   flows into `aux_model_score` and therefore into
   `_get_ltr_ordered_requests`'s sort key. For the regression config variant
   (`config_prefill_opt.txt`, `num_labels == 1`), `compute_logits` skips the
   argmax branch entirely and the single logit is used directly.
4. **Weight compatibility:** `load_weights` (elided above) uses the same
   `stacked_params_mapping` / `default_weight_loader` pattern as vLLM's other
   HF-compatible model loaders — i.e. checkpoints on
   `LLM-ltr/OPT-Predictors` are expected to be ordinary HF-format OPT
   checkpoints plus a `score.weight` tensor, not a vLLM-specific serialization
   format. This is what makes independent (non-vLLM) loading via
   `adapter/checkpoint_loader.py` plausible in principle — see that module's
   docstring for the exact untested assumptions this relies on.
