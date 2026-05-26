from dataclasses import dataclass
from enum import StrEnum


class PricingMethod(StrEnum):
    PROVIDER_REPORTED = "provider_reported"
    TOKEN_PRICING = "token_pricing"
    MODAL_GPU_SECONDS = "modal_gpu_seconds"
    NOT_AVAILABLE = "not_available"


@dataclass(frozen=True)
class OpenAITokenPricing:
    input_usd_per_1m_tokens: float
    output_usd_per_1m_tokens: float

    def estimate(self, input_tokens: int, output_tokens: int) -> float:
        input_cost = input_tokens * self.input_usd_per_1m_tokens / 1_000_000
        output_cost = output_tokens * self.output_usd_per_1m_tokens / 1_000_000
        return input_cost + output_cost


@dataclass(frozen=True)
class ModalGpuPricing:
    gpu_usd_per_hour: float

    def estimate(self, duration_s: float) -> float:
        return duration_s * self.gpu_usd_per_hour / 3600


def default_openai_pricing(model: str) -> OpenAITokenPricing:
    normalized = model.lower()
    if "gpt-4.1-mini" in normalized:
        return OpenAITokenPricing(input_usd_per_1m_tokens=0.40, output_usd_per_1m_tokens=1.60)
    return OpenAITokenPricing(input_usd_per_1m_tokens=0.0, output_usd_per_1m_tokens=0.0)
