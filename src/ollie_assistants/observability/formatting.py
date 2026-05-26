from ollie_assistants.observability.types import ConversationTrace


def format_trace(trace: ConversationTrace) -> str:
    lines = [
        f"run_id: {trace.run_id}",
        f"session_id: {trace.session_id}",
        f"assistant: {trace.assistant_id}",
        f"provider: {trace.provider_id}",
        f"latency: {trace.total_latency_ms:.0f} ms",
        f"tokens: input={trace.input_tokens or 0} output={trace.output_tokens or 0}",
        f"cost: ${trace.estimated_cost_usd:.6f}",
        f"pricing: {trace.pricing_method.value}",
        "events:",
    ]
    for event in trace.events:
        lines.append(f"- {event.event_type.value}.{event.name}: {event.latency_ms:.0f} ms")
        for key, value in event.metadata.items():
            lines.append(f"  {key}: {value}")
    return "\n".join(lines)
