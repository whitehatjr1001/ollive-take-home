from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from ollie_assistants.analytics.pricing import PricingMethod


class TraceEventType(StrEnum):
    SAFETY = "safety"
    TOOL = "tool"
    LLM = "llm"
    RESPONSE = "response"


@dataclass(frozen=True)
class TraceEvent:
    event_type: TraceEventType
    name: str
    latency_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ConversationTrace:
    run_id: str
    session_id: str
    assistant_id: str
    provider_id: str
    total_latency_ms: float
    input_tokens: int | None
    output_tokens: int | None
    estimated_cost_usd: float
    pricing_method: PricingMethod
    events: tuple[TraceEvent, ...]
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
