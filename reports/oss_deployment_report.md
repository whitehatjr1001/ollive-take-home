# OSS Deployment Report

| Field | Value |
| --- | --- |
| OSS model | `oss-assistant` |
| Served model name | `oss-assistant` |
| Runtime | Modal + vLLM |
| GPU | `L4` |
| OpenAI-compatible base URL | `https://rohithsaimidigudla--ollie-oss-vllm-serve.modal.run/v1` |
| Auth | Bearer token between app host and Modal endpoint |
| Scaledown window | 600s |
| Max concurrent inputs | 20 |
| Max model length | 4096 tokens |

## Deploy Command

```bash
OSS_BEARER_TOKEN=$(openssl rand -hex 32)
modal secret create ollie-oss-secrets OSS_BEARER_TOKEN="$OSS_BEARER_TOKEN"
uv run modal deploy src/ollie_assistants/deploy/modal_oss.py
```