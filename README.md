# Ollive Assistants

Compare two AI personal assistants with the same chat experience:

- **OSS assistant** served from a public Modal vLLM OpenAI-compatible endpoint.
- **Frontier assistant** served through OpenAI.

The app includes multi-turn chat, short-term context, memory/tool use, guardrails,
observability, evals, benchmark reports, and a Docker deployment for Hugging Face
Spaces.

Demo: https://www.loom.com/share/5abb9ed62d5a448c8fe84b3f7c6abf6c

## Take-Home Checklist

| Requirement | Status | Implementation |
| --- | --- | --- |
| Deploy OSS model publicly | Done | Modal vLLM OpenAI-compatible endpoint with bearer-token proxy |
| Cost + latency table for OSS deployment | Done | Benchmark runner + `reports/cost_latency_report.md` |
| Observability/evals | Done | SQLite run store, Streamlit observability tab, eval history, markdown reports |
| Guardrails/safety layers | Done | Input/output safety checks, unsafe refusal handling, tool policy |
| Memory/tool use | Done | Agent loop with memory search/write decisions and current-time tool |
| Compare OSS vs frontier | Done | Shared assistant interface and side-by-side compare mode |

## Features

- Multi-turn conversations with bounded short-term memory.
- OSS and frontier assistants behind the same `AssistantFacade`.
- OpenAI-compatible OSS provider for Modal/vLLM.
- Frontier OpenAI provider using configured token pricing.
- Agent loop with:
  - safety check
  - dynamic memory search decision
  - local tool execution
  - model response
  - output safety check
  - dynamic memory-write decision
- Tools:
  - `current_time`
  - `search_memory`
  - `remember_user_fact`
- Guardrails:
  - jailbreak/refusal checks
  - self-harm safe-completion behavior
  - harmful tool-write blocking
- Observability:
  - chat run traces in SQLite
  - latency, tokens, cost, safety action, tool calls, event timeline
  - Streamlit observability explorer
- Evals:
  - factual prompts
  - adversarial/jailbreak prompts
  - sensitive/bias prompts
  - SimpleQA-style sample prompts
  - optional LLM-as-judge verification
- Reports:
  - OSS deployment report
  - cost + latency report
  - assistant evaluation report

## Architecture

```text
Streamlit UI
    |
FastAPI
    |
AssistantFacade
    |
AgentLoop
    |-- GuardrailService
    |-- ToolRegistry
    |-- MemoryTools
    |-- LLMProvider
          |-- OpenAIProvider
          |-- OpenAICompatibleProvider

Eval API / CLI
    |
AssistantComparisonService
    |
EvalRunner + JudgeFactory + BenchmarkRunner
    |
Reports + SQLite eval history
```

The code uses a ports-and-adapters style. Provider-specific code stays behind
`LLMProvider`; app workflows use facades and factories so OSS and frontier models
can be swapped without changing the UI or eval runner.

## Project Structure

```text
src/ollie_assistants/
  api/                 FastAPI routers
  assistant/           assistant facade, agent loop, prompts, tools
  analytics/           latency and pricing helpers
  deploy/              Modal OSS model deployment and benchmark runner
  evals/               cases, runner, judges, metrics, LLM judge, CLI
  interface/           Streamlit UI
  llm/                 provider interfaces and factories
  memory/              in-memory session memory store
  observability/       trace types, formatting, SQLite recorder
  reports/             report rendering and loading
  safety/              guardrails and tool policy
```

## Setup

Install dependencies:

```bash
uv sync
```

Create env file:

```bash
cp .env.example .env
```

Required env values:

```text
OPENAI_API_KEY=...
FRONTIER_MODEL=gpt-4.1

OSS_PROVIDER=openai_compatible
OSS_BASE_URL=https://your-modal-endpoint/v1
OSS_BEARER_TOKEN=...
OSS_MODEL=oss-assistant
OSS_SERVED_MODEL_NAME=oss-assistant

MODAL_L4_USD_PER_HOUR=0.80
OPENAI_INPUT_USD_PER_1M_TOKENS=2.00
OPENAI_OUTPUT_USD_PER_1M_TOKENS=8.00
```

## Run Locally

Start FastAPI:

```bash
uv run ollie-api
```

Start Streamlit:

```bash
uv run streamlit run src/ollie_assistants/interface/streamlit_app.py \
  --server.address 127.0.0.1 \
  --server.port 8501 \
  --browser.gatherUsageStats false
```

API docs:

```text
http://127.0.0.1:8000/docs
```

## Run Evals

CLI:

```bash
uv run ollie-eval --include-benchmark --use-llm-judge
```

API:

```bash
curl -s http://127.0.0.1:8000/evals/take-home \
  -H "Content-Type: application/json" \
  -d '{"include_benchmark": true, "use_llm_judge": true}'
```

The Streamlit Reports tab can also start evals in the background, refresh eval
history, select any previous eval run, and show the stored metrics/report.

Generated reports:

```text
reports/oss_deployment_report.md
reports/cost_latency_report.md
reports/assistant_evaluation_report.md
reports/evaluation_report.pdf
```

Generate the PDF submission artifact:

```bash
uv run ollie-report-pdf
```

## Public OSS Model Deployment

The OSS model is deployed separately on Modal using vLLM in OpenAI-compatible
mode. The raw Modal endpoint is protected with a bearer token; reviewers use the
public app UI, not the raw model endpoint.

Deploy the Modal endpoint:

```bash
OSS_BEARER_TOKEN=$(openssl rand -hex 32)
modal secret create ollie-oss-secrets OSS_BEARER_TOKEN="$OSS_BEARER_TOKEN"
uv run modal deploy src/ollie_assistants/deploy/modal_oss.py
```

Test the endpoint:

```bash
curl -s https://your-modal-endpoint/health \
  -H "Authorization: Bearer $OSS_BEARER_TOKEN"
```

## Hugging Face Spaces Deployment

Use a Hugging Face **Docker Space**. The container runs:

- FastAPI on `127.0.0.1:8000`
- Streamlit on port `7860`

The Docker entrypoint is:

```text
scripts/start_app.sh
```

Set Space secrets:

```text
OPENAI_API_KEY
OSS_BASE_URL
OSS_BEARER_TOKEN
```

Set Space variables:

```text
API_BASE_URL=http://127.0.0.1:8000
FRONTIER_MODEL=gpt-4.1
OSS_MODEL=oss-assistant
OSS_SERVED_MODEL_NAME=oss-assistant
MODAL_L4_USD_PER_HOUR=0.80
OPENAI_INPUT_USD_PER_1M_TOKENS=2.00
OPENAI_OUTPUT_USD_PER_1M_TOKENS=8.00
```

Optional CLI setup:

```bash
hf spaces secrets add RohithMidigudla/assitant-comparision --secrets-file .hf-secrets.env
hf spaces variables add RohithMidigudla/assitant-comparision --env-file .hf-vars.env
```

Do not commit `.hf-secrets.env`.

## Observability

Runtime traces are stored in SQLite at:

```text
.runs/observability.sqlite
```

Each chat run records:

- run id
- session id
- assistant/provider
- latency
- input/output tokens
- estimated cost
- safety action
- tool calls
- event timeline

Eval runs are stored separately from chat runs, so the Observability tab stays
focused on real assistant interactions while the Reports tab tracks eval history.

## Cost Model

Frontier cost:

```text
input_tokens * input_price + output_tokens * output_price
```

OSS Modal cost:

```text
request_latency_seconds * MODAL_L4_USD_PER_HOUR / 3600
```

Benchmark reports aggregate cost per request and cost per 1M tokens.

## Tradeoffs

- Used a public Modal OSS endpoint instead of running the model inside the app
  container, keeping the UI deploy small and fast.
- Used OpenAI-compatible OSS serving so the provider can be swapped later.
- Used SQLite for observability because it is simple, inspectable, and enough for
  a take-home demo.
- Used lightweight heuristic judges plus optional LLM-as-judge verification
  rather than a heavy benchmark harness.
- Kept guardrails local and transparent; production would use stronger policy
  models and red-team test sets.

## Improvements With More Time

- Add true token streaming from the OSS and frontier providers instead of
  UI-level chunking after full responses.
- Replace in-memory user memory with persistent user-scoped memory.
- Add stronger public benchmark adapters such as full SimpleQA or BBQ slices.
- Add auth and rate limiting for a production public demo.
- Add hosted persistent storage for observability on deployed environments.

## Verification

```bash
uv run ruff check .
uv run pytest
```
