import asyncio
import time
from collections.abc import Sequence
from typing import Protocol

from openai import AsyncOpenAI

from ollie_assistants.analytics.pricing import (
    ModalGpuPricing,
    OpenAITokenPricing,
    PricingMethod,
)
from ollie_assistants.llm.types import ChatConfig, ChatMessage, ChatResult


class LLMProvider(Protocol):
    provider_id: str

    async def chat(self, messages: Sequence[ChatMessage], config: ChatConfig) -> ChatResult:
        ...


class OpenAIProvider:
    provider_id = "openai"

    def __init__(self, api_key: str | None, pricing: OpenAITokenPricing) -> None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAIProvider")
        self.client = AsyncOpenAI(api_key=api_key)
        self.pricing = pricing

    async def chat(self, messages: Sequence[ChatMessage], config: ChatConfig) -> ChatResult:
        response = await self.client.chat.completions.create(
            model=config.model,
            messages=[{"role": msg.role.value, "content": msg.content} for msg in messages],
            temperature=config.temperature,
            max_tokens=config.max_new_tokens,
        )
        usage = response.usage
        return ChatResult(
            text=response.choices[0].message.content or "",
            input_tokens=usage.prompt_tokens if usage else None,
            output_tokens=usage.completion_tokens if usage else None,
            estimated_cost_usd=(
                self.pricing.estimate(
                    usage.prompt_tokens,
                    usage.completion_tokens,
                )
                if usage
                else 0.0
            ),
            pricing_method=PricingMethod.TOKEN_PRICING if usage else PricingMethod.NOT_AVAILABLE,
        )


class OpenAICompatibleProvider:
    provider_id = "openai_compatible"

    def __init__(
        self,
        base_url: str | None,
        api_key: str | None,
        pricing: ModalGpuPricing | None = None,
    ) -> None:
        if not base_url:
            raise ValueError("OSS_BASE_URL is required for OpenAICompatibleProvider")
        self.client = AsyncOpenAI(api_key=api_key or "not-needed", base_url=base_url)
        self.pricing = pricing

    async def chat(self, messages: Sequence[ChatMessage], config: ChatConfig) -> ChatResult:
        started = time.perf_counter()
        response = await self.client.chat.completions.create(
            model=config.model,
            messages=[{"role": msg.role.value, "content": msg.content} for msg in messages],
            temperature=config.temperature,
            max_tokens=config.max_new_tokens,
        )
        duration_s = time.perf_counter() - started
        usage = response.usage
        return ChatResult(
            text=response.choices[0].message.content or "",
            input_tokens=usage.prompt_tokens if usage else None,
            output_tokens=usage.completion_tokens if usage else None,
            estimated_cost_usd=(
                self.pricing.estimate(duration_s) if self.pricing is not None else 0.0
            ),
            pricing_method=PricingMethod.MODAL_GPU_SECONDS
            if self.pricing is not None
            else PricingMethod.NOT_AVAILABLE,
        )


class TransformersProvider:
    provider_id = "transformers"

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._pipeline = None

    def _load_pipeline(self):
        if self._pipeline is None:
            from transformers import pipeline

            self._pipeline = pipeline("text-generation", model=self.model_name, device_map="auto")
        return self._pipeline

    async def chat(self, messages: Sequence[ChatMessage], config: ChatConfig) -> ChatResult:
        prompt = "\n".join(f"{msg.role.value}: {msg.content}" for msg in messages) + "\nassistant:"

        def generate() -> str:
            pipe = self._load_pipeline()
            result = pipe(
                prompt,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                do_sample=config.temperature > 0,
                return_full_text=False,
            )
            return result[0]["generated_text"]

        return ChatResult(text=await asyncio.to_thread(generate))
