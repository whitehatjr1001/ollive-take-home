FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src
ENV API_BASE_URL=http://127.0.0.1:8000
ENV PORT=7860

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev

COPY . .

RUN chmod +x scripts/start_app.sh

EXPOSE 7860

CMD ["./scripts/start_app.sh"]
