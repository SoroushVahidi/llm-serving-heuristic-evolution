# API Provider Setup

This project uses external LLM APIs in Phase 4 (heuristic generation and evolution).
Phases 1–1.7C (simulator, calibration, real-trace replay) do not call any LLM APIs.

---

## Credential Policy

- Store all API keys in `.env` (gitignored).
- Copy `.env.example` → `.env` and fill in real values.
- **Never** commit `.env`, credentials files, or printed token values.
- Check login status with non-printing commands (see below).

---

## Providers

### Hugging Face (`HF_TOKEN`)

Used to download BurstGPT from the Hub and to load calibration models
(e.g., `Qwen/Qwen2.5-0.5B`).

```bash
# Check login without printing the token
python -c "from huggingface_hub import whoami; print(whoami()['name'])"
```

Set in environment:
```bash
export HF_TOKEN=hf_...  # or put in .env
```

### CloudRift (`CLOUDRIFT_API_KEY`, `CLOUDRIFT_BASE_URL`)

CloudRift is an **OpenAI-compatible** inference endpoint that is **not** OpenAI.com.
When using the `openai` Python SDK with CloudRift, always pass `base_url` explicitly:

```python
from openai import OpenAI
client = OpenAI(
    api_key=os.environ["CLOUDRIFT_API_KEY"],
    base_url=os.environ["CLOUDRIFT_BASE_URL"],
)
```

Do not rely on `OPENAI_API_KEY` pointing to CloudRift — the variable name and the
routing destination must be kept separate to avoid accidentally hitting OpenAI.com.

### Cohere (`COHERE_API_KEY`)

Used for heuristic generation in Phase 4. The `cohere` SDK reads `COHERE_API_KEY`
from the environment automatically.

```bash
pip install cohere
python -c "import cohere; c = cohere.Client(); print('ok')"
```

### Mistral (`MISTRAL_API_KEY`)

```bash
pip install mistralai
```

### Cerebras, Fireworks, Together, Groq, xAI, OpenRouter

All are OpenAI-compatible. Use the `openai` SDK with the appropriate `base_url`
and `api_key`.

### Google Cloud / Vertex AI (`GOOGLE_CLOUD_PROJECT`, `GOOGLE_APPLICATION_CREDENTIALS`)

For Vertex AI Gemini access. Authenticate via Application Default Credentials:

```bash
gcloud auth application-default login
```

---

## Phase-by-phase API usage

| Phase | APIs used |
|---|---|
| 1–1.7C (simulator, calibration, real-trace) | None |
| 2A (selector training) | None — uses local simulator data |
| 2B (LLM heuristic DSL) | CloudRift, Cohere, Mistral (design phase) |
| 4 (evolution loop) | CloudRift, Cohere (generation); local GPU (evaluation) |

---

## Safe practices checklist

- [ ] `.env` is in `.gitignore` (already configured)
- [ ] `.env.example` has placeholder names only, no real values
- [ ] No `print(os.environ["HF_TOKEN"])` or equivalent in scripts
- [ ] CloudRift `base_url` is always passed explicitly — not assumed from env
- [ ] Model weights loaded from HuggingFace are stored locally and not re-uploaded
