from dataclasses import dataclass
from enum import StrEnum

from ollie_assistants.analytics.pricing import PricingMethod


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str


@dataclass(frozen=True)
class ChatConfig:
    model: str
    max_new_tokens: int = 256
    temperature: float = 0.2


@dataclass(frozen=True)
class ChatResult:
    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    estimated_cost_usd: float = 0.0
    pricing_method: PricingMethod = PricingMethod.NOT_AVAILABLE
