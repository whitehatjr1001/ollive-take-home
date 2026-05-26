#!/usr/bin/env bash
set -euo pipefail

uv run uvicorn ollie_assistants.main:app --host 127.0.0.1 --port 8000 &

uv run streamlit run src/ollie_assistants/interface/streamlit_app.py \
  --server.address 0.0.0.0 \
  --server.port "${PORT:-7860}" \
  --browser.gatherUsageStats false
