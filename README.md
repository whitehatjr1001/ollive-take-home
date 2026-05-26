# Ollie Assistants

Lean take-home implementation for comparing two personal assistants:

- OSS assistant using a Hugging Face/local-compatible provider.
- Frontier assistant using OpenAI directly.

The app uses a lightweight ports-and-adapters design: core assistant/eval logic depends on typed interfaces, factories select concrete providers, and facades expose simple chat/eval workflows to FastAPI, Gradio, and Modal.

## Quick Start

```bash
uv sync
cp .env.example .env
uv run ollie-api
```

Open:

- API: `http://127.0.0.1:8000`
- Docs: `http://127.0.0.1:8000/docs`

Run Gradio:

```bash
uv run ollie-gradio
```

Run evals:

```bash
uv run ollie-eval
```

## Deployment Split

The main app is lightweight and can deploy to a free app host such as Render, Railway, Fly.io, or a Hugging Face Space. The OSS model runs separately on Modal and exposes an OpenAI-compatible `/v1/chat/completions` endpoint.

## Hugging Face Spaces Deployment

The repo includes a Docker deployment that runs Streamlit on port `7860` and an
internal FastAPI server on `127.0.0.1:8000`.

Use a Hugging Face **Docker Space** and set these secrets:

```text
OPENAI_API_KEY
OSS_BASE_URL=https://your-modal-endpoint/v1
OSS_BEARER_TOKEN
```

Set these variables:

```text
API_BASE_URL=http://127.0.0.1:8000
FRONTIER_MODEL=gpt-4.1
OSS_MODEL=oss-assistant
OSS_SERVED_MODEL_NAME=oss-assistant
MODAL_L4_USD_PER_HOUR=0.80
OPENAI_INPUT_USD_PER_1M_TOKENS=2.00
OPENAI_OUTPUT_USD_PER_1M_TOKENS=8.00
```

The container entrypoint is `scripts/start_app.sh`.

Deploy OSS model endpoint:

```bash
OSS_BEARER_TOKEN=$(openssl rand -hex 32)
modal secret create ollie-oss-secrets OSS_BEARER_TOKEN="$OSS_BEARER_TOKEN"
uv run modal deploy src/ollie_assistants/deploy/modal_oss.py
```

Then set the generated Modal endpoint URL:

```bash
OSS_PROVIDER=openai_compatible
OSS_BASE_URL=https://your-modal-endpoint/v1
OSS_BEARER_TOKEN=your-shared-secret-token
OSS_MODEL=oss-assistant
OPENAI_INPUT_USD_PER_1M_TOKENS=0.40
OPENAI_OUTPUT_USD_PER_1M_TOKENS=1.60
```

Deploy app somewhere free/cheap:

```bash
uv run ollie-api
```

The app host needs `OSS_BASE_URL` for OSS inference and `OPENAI_API_KEY` for the frontier assistant. Secrets should be configured in the host provider, not committed.

## Architecture

```text
FastAPI / Gradio / CLI
        |
AssistantFacade
        |
Guardrails -> Memory -> Tools -> LLMProvider
        |
AnalyticsRecorder

AssistantComparisonService
        |
EvalRunner -> JudgeFactory -> Metric judges
        |
ReportBuilder
```

## Tradeoffs

- Direct OpenAI provider instead of LiteLLM keeps dependencies and debugging smaller.
- OSS provider uses an OpenAI-compatible endpoint so Modal, vLLM, TGI, Ollama-compatible gateways, or other hosted OSS runtimes can be swapped without changing app code.
- JSONL analytics are enough for a take-home; a database would be added for production.
- Eval cases are custom and small by default, but the runner supports adding benchmark-backed case repositories.
