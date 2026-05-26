import time
from collections.abc import Sequence

import httpx

from ollie_assistants.deploy.types import DeploymentHealth, DeploymentSmokeResult


async def check_health(base_url: str, bearer_token: str, timeout_s: float = 30) -> DeploymentHealth:
    started = time.perf_counter()
    async with httpx.AsyncClient(base_url=base_url, timeout=timeout_s) as client:
        response = await client.get("/health", headers=_headers(bearer_token))
    latency_ms = (time.perf_counter() - started) * 1000
    return DeploymentHealth(
        ok=response.status_code == 200,
        latency_ms=latency_ms,
        detail=response.text,
    )


async def smoke_chat(
    base_url: str,
    bearer_token: str,
    model: str,
    messages: Sequence[dict[str, str]],
    timeout_s: float = 120,
) -> DeploymentSmokeResult:
    started = time.perf_counter()
    try:
        async with httpx.AsyncClient(base_url=base_url, timeout=timeout_s) as client:
            response = await client.post(
                "/v1/chat/completions",
                headers=_headers(bearer_token),
                json={"model": model, "messages": list(messages), "max_tokens": 64},
            )
            response.raise_for_status()
        payload = response.json()
        text = payload["choices"][0]["message"]["content"]
        return DeploymentSmokeResult(
            ok=True,
            latency_ms=(time.perf_counter() - started) * 1000,
            text=text,
        )
    except Exception as err:
        return DeploymentSmokeResult(
            ok=False,
            latency_ms=(time.perf_counter() - started) * 1000,
            text="",
            error=str(err),
        )


def _headers(bearer_token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {bearer_token}"}
