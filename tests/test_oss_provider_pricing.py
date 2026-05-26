import pytest

from ollie_assistants.analytics.pricing import ModalGpuPricing, PricingMethod
from ollie_assistants.llm.providers import OpenAICompatibleProvider
from ollie_assistants.llm.types import ChatConfig, ChatMessage, Role


class FakeChatCompletions:
    async def create(self, **kwargs):
        return FakeResponse()


class FakeChat:
    def __init__(self) -> None:
        self.completions = FakeChatCompletions()


class FakeClient:
    def __init__(self) -> None:
        self.chat = FakeChat()


class FakeUsage:
    prompt_tokens = 10
    completion_tokens = 5


class FakeMessage:
    content = "ok"


class FakeChoice:
    message = FakeMessage()


class FakeResponse:
    usage = FakeUsage()
    choices = [FakeChoice()]


@pytest.mark.asyncio
async def test_openai_compatible_provider_estimates_modal_gpu_cost() -> None:
    provider = OpenAICompatibleProvider(
        "https://example.test/v1",
        "token",
        ModalGpuPricing(gpu_usd_per_hour=0.80),
    )
    provider.client = FakeClient()

    result = await provider.chat(
        [ChatMessage(Role.USER, "hi")],
        ChatConfig(model="oss-assistant"),
    )

    assert result.estimated_cost_usd > 0
    assert result.pricing_method == PricingMethod.MODAL_GPU_SECONDS
