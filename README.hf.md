---
title: Ollive Assistants
emoji: 🧪
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Ollive Assistants

Docker Space for comparing an OSS Modal-vLLM assistant with a frontier OpenAI
assistant. The container runs Streamlit on port `7860` and an internal FastAPI
server on `127.0.0.1:8000`.

Set these Space secrets before starting:

- `OPENAI_API_KEY`
- `OSS_BASE_URL`
- `OSS_BEARER_TOKEN`

Set these Space variables if you want to override defaults:

- `FRONTIER_MODEL=gpt-4.1`
- `OSS_MODEL=oss-assistant`
- `OSS_SERVED_MODEL_NAME=oss-assistant`
- `MODAL_L4_USD_PER_HOUR=0.80`
- `OPENAI_INPUT_USD_PER_1M_TOKENS=2.00`
- `OPENAI_OUTPUT_USD_PER_1M_TOKENS=8.00`
