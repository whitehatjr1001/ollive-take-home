import json
import os
import subprocess
import time

import httpx
import modal
from fastapi import FastAPI, Header, Request
from fastapi.responses import Response

from ollie_assistants.deploy.auth import BearerTokenValidator

MODEL_NAME = os.environ.get("OSS_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
SERVED_MODEL_NAME = os.environ.get("OSS_SERVED_MODEL_NAME", "oss-assistant")
VLLM_PORT = 8000
MINUTES = 60
APP_NAME = os.environ.get("OSS_MODAL_APP_NAME", "ollie-oss-vllm")
GPU = os.environ.get("OSS_MODAL_GPU", "L4")
SCALEDOWN_SECONDS = int(os.environ.get("OSS_MODAL_SCALEDOWN_SECONDS", "600"))
MAX_INPUTS = int(os.environ.get("OSS_MODAL_MAX_INPUTS", "20"))
MAX_MODEL_LEN = os.environ.get("OSS_MODAL_MAX_MODEL_LEN", "4096")
SECRET_NAME = os.environ.get("OSS_MODAL_SECRET_NAME", "ollie-oss-secrets")

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.21.0", "fastapi>=0.115.0", "httpx>=0.28.0")
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "VLLM_LOG_STATS_INTERVAL": "1"})
    .add_local_python_source("ollie_assistants")
)

hf_cache_vol = modal.Volume.from_name("ollie-huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("ollie-vllm-cache", create_if_missing=True)

app = modal.App(APP_NAME)


def build_proxy_app() -> FastAPI:
    proxy = FastAPI(title="Ollie OSS vLLM Proxy")
    validator = BearerTokenValidator(os.environ.get("OSS_BEARER_TOKEN"))
    vllm_base_url = f"http://127.0.0.1:{VLLM_PORT}"

    @proxy.on_event("startup")
    async def startup() -> None:
        command = [
            "vllm",
            "serve",
            MODEL_NAME,
            "--served-model-name",
            SERVED_MODEL_NAME,
            "--host",
            "0.0.0.0",
            "--port",
            str(VLLM_PORT),
            "--uvicorn-log-level",
            "info",
            "--max-model-len",
            MAX_MODEL_LEN,
        ]
        subprocess.Popen(command)
        await wait_for_vllm(vllm_base_url)

    @proxy.get("/health")
    async def health(authorization: str | None = Header(default=None)) -> dict[str, str]:
        validator.validate(authorization)
        async with httpx.AsyncClient(base_url=vllm_base_url, timeout=10) as client:
            response = await client.get("/health")
            response.raise_for_status()
        return {"status": "ok"}

    @proxy.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
    async def forward_v1(
        path: str,
        request: Request,
        authorization: str | None = Header(default=None),
    ) -> Response:
        validator.validate(authorization)
        body = await request.body()
        headers = {
            key: value
            for key, value in request.headers.items()
            if key.lower() not in {"host", "content-length", "authorization"}
        }
        async with httpx.AsyncClient(base_url=vllm_base_url, timeout=120) as client:
            response = await client.request(
                request.method,
                f"/v1/{path}",
                content=body,
                headers=headers,
                params=request.query_params,
            )
        return Response(
            content=response.content,
            status_code=response.status_code,
            media_type=response.headers.get("content-type"),
        )

    return proxy


async def wait_for_vllm(base_url: str, timeout_s: int = 10 * MINUTES) -> None:
    deadline = time.monotonic() + timeout_s
    async with httpx.AsyncClient(base_url=base_url, timeout=5) as client:
        while time.monotonic() < deadline:
            try:
                response = await client.get("/health")
                if response.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(2)
    raise TimeoutError("vLLM did not become healthy before startup timeout")


@app.function(
    image=vllm_image,
    gpu=GPU,
    scaledown_window=SCALEDOWN_SECONDS,
    timeout=10 * MINUTES,
    secrets=[modal.Secret.from_name(SECRET_NAME)],
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
@modal.concurrent(max_inputs=MAX_INPUTS)
@modal.asgi_app()
def serve():
    return build_proxy_app()


@app.local_entrypoint()
async def test(content: str = "Say hello in one short sentence.") -> None:
    url = await serve.get_web_url.aio()
    token = os.environ["OSS_BEARER_TOKEN"]
    payload = {
        "model": SERVED_MODEL_NAME,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 64,
    }
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(base_url=url, timeout=120) as client:
        health = await client.get("/health", headers=headers)
        health.raise_for_status()
        response = await client.post("/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
    print(json.dumps(response.json(), indent=2))
