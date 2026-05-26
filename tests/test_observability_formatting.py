from ollie_assistants.analytics.pricing import PricingMethod
from ollie_assistants.observability.formatting import format_trace
from ollie_assistants.observability.types import ConversationTrace, TraceEvent, TraceEventType


def test_format_trace_includes_run_cost_and_events() -> None:
    trace = ConversationTrace(
        run_id="run-1",
        session_id="session-1",
        assistant_id="oss",
        provider_id="openai_compatible",
        total_latency_ms=123.4,
        input_tokens=10,
        output_tokens=20,
        estimated_cost_usd=0.001,
        pricing_method=PricingMethod.TOKEN_PRICING,
        events=(
            TraceEvent(
                event_type=TraceEventType.LLM,
                name="openai_compatible",
                latency_ms=100,
                metadata={"model": "oss-assistant"},
            ),
        ),
    )

    formatted = format_trace(trace)

    assert "run_id: run-1" in formatted
    assert "cost: $0.001000" in formatted
    assert "pricing: token_pricing" in formatted
    assert "llm.openai_compatible" in formatted
