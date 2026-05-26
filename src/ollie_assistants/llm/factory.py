from collections.abc import Callable

from ollie_assistants.analytics.pricing import ModalGpuPricing, OpenAITokenPricing
from ollie_assistants.llm.providers import (
    LLMProvider,
    OpenAICompatibleProvider,
    OpenAIProvider,
    TransformersProvider,
)
from ollie_assistants.settings import Settings

ProviderBuilder = Callable[[Settings], LLMProvider]


class LLMProviderFactory:
    def __init__(self) -> None:
        self._builders: dict[str, ProviderBuilder] = {}

    def register(self, provider_id: str, builder: ProviderBuilder) -> None:
        self._builders[provider_id] = builder

    def create(self, provider_id: str, settings: Settings) -> LLMProvider:
        builder = self._builders.get(provider_id)
        if builder is None:
            known = ", ".join(sorted(self._builders))
            raise ValueError(f"Unknown LLM provider '{provider_id}'. Known providers: {known}")
        return builder(settings)


def default_llm_factory() -> LLMProviderFactory:
    factory = LLMProviderFactory()
    factory.register(
        "openai",
        lambda settings: OpenAIProvider(
            settings.openai_api_key,
            OpenAITokenPricing(
                input_usd_per_1m_tokens=settings.openai_input_usd_per_1m_tokens,
                output_usd_per_1m_tokens=settings.openai_output_usd_per_1m_tokens,
            ),
        ),
    )
    factory.register(
        "openai_compatible",
        lambda settings: OpenAICompatibleProvider(
            settings.oss_base_url,
            settings.oss_bearer_token,
            ModalGpuPricing(gpu_usd_per_hour=settings.modal_l4_usd_per_hour),
        ),
    )
    factory.register("transformers", lambda settings: TransformersProvider(settings.oss_model))
    return factory
