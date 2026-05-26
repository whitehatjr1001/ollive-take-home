import asyncio
import time
from collections.abc import Sequence
from dataclasses import dataclass

from ollie_assistants.analytics.latency import p50, p95
from ollie_assistants.analytics.pricing import ModalGpuPricing
from ollie_assistants.assistant.service import AssistantFacade, AssistantResponse


@dataclass(frozen=True)
class BenchmarkScenario:
    name: str
    concurrency: int
    requests: int
    prompt: str


@dataclass(frozen=True)
class BenchmarkRow:
    provider: str
    model: str
    scenario: str
    concurrency: int
    requests: int
    p50_latency_ms: float
    p95_latency_ms: float
    output_tokens_per_second: float
    cost_per_request_usd: float
    cost_per_1m_tokens_usd: float
    error_rate: float


class AssistantBenchmarkRunner:
    def __init__(self, modal_pricing: ModalGpuPricing) -> None:
        self.modal_pricing = modal_pricing

    async def run(
        self,
        provider_label: str,
        model: str,
        assistant: AssistantFacade,
        scenarios: Sequence[BenchmarkScenario],
        cost_mode: str,
    ) -> tuple[BenchmarkRow, ...]:
        rows: list[BenchmarkRow] = []
        for scenario in scenarios:
            rows.append(
                await self._run_scenario(provider_label, model, assistant, scenario, cost_mode)
            )
        return tuple(rows)

    async def _run_scenario(
        self,
        provider_label: str,
        model: str,
        assistant: AssistantFacade,
        scenario: BenchmarkScenario,
        cost_mode: str,
    ) -> BenchmarkRow:
        semaphore = asyncio.Semaphore(scenario.concurrency)
        started = time.perf_counter()
        results = await asyncio.gather(
            *[
                self._run_one(assistant, scenario.prompt, index, semaphore)
                for index in range(scenario.requests)
            ],
            return_exceptions=True,
        )
        duration_s = time.perf_counter() - started
        responses = [result for result in results if isinstance(result, AssistantResponse)]
        errors = scenario.requests - len(responses)
        latencies = [response.latency_ms for response in responses]
        total_tokens = sum(
            (response.input_tokens or 0) + (response.output_tokens or 0)
            for response in responses
        )
        output_tokens = sum(response.output_tokens or 0 for response in responses)
        total_cost = self._total_cost(cost_mode, duration_s, responses)
        completed = max(len(responses), 1)
        return BenchmarkRow(
            provider=provider_label,
            model=model,
            scenario=scenario.name,
            concurrency=scenario.concurrency,
            requests=scenario.requests,
            p50_latency_ms=p50(latencies),
            p95_latency_ms=p95(latencies),
            output_tokens_per_second=output_tokens / duration_s if duration_s > 0 else 0.0,
            cost_per_request_usd=total_cost / completed,
            cost_per_1m_tokens_usd=(total_cost / total_tokens * 1_000_000) if total_tokens else 0.0,
            error_rate=errors / scenario.requests,
        )

    async def _run_one(
        self,
        assistant: AssistantFacade,
        prompt: str,
        index: int,
        semaphore: asyncio.Semaphore,
    ) -> AssistantResponse:
        async with semaphore:
            return await assistant.chat(
                f"bench-{time.time_ns()}-{index}",
                prompt,
                record_trace=False,
            )

    def _total_cost(
        self,
        cost_mode: str,
        duration_s: float,
        responses: Sequence[AssistantResponse],
    ) -> float:
        if cost_mode == "modal_gpu_seconds":
            return self.modal_pricing.estimate(duration_s)
        if cost_mode == "provider_reported":
            return sum(response.estimated_cost_usd for response in responses)
        raise ValueError(f"Unknown benchmark cost mode: {cost_mode}")


def default_benchmark_scenarios() -> tuple[BenchmarkScenario, ...]:
    prompt = "Give three practical tips for planning a productive workday."
    return (
        BenchmarkScenario("warm", concurrency=1, requests=5, prompt=prompt),
        BenchmarkScenario("batch", concurrency=4, requests=16, prompt=prompt),
    )
