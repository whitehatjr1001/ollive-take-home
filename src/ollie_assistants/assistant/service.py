import time
from dataclasses import dataclass
from uuid import uuid4

from ollie_assistants.analytics.pricing import PricingMethod
from ollie_assistants.assistant.agent_loop import AgentLoop
from ollie_assistants.assistant.memory import ConversationMemory
from ollie_assistants.assistant.prompts import SYSTEM_PROMPT
from ollie_assistants.assistant.tools import ToolRegistry
from ollie_assistants.llm.providers import LLMProvider
from ollie_assistants.llm.types import ChatMessage, Role
from ollie_assistants.observability.formatting import format_trace
from ollie_assistants.observability.recorder import TraceRecorder
from ollie_assistants.observability.types import ConversationTrace, TraceEvent, TraceEventType
from ollie_assistants.safety.guardrails import GuardrailService, SafetyAction


@dataclass(frozen=True)
class AssistantResponse:
    run_id: str
    session_id: str
    text: str
    provider_id: str
    latency_ms: float
    estimated_cost_usd: float
    pricing_method: str
    safety_action: str
    input_tokens: int | None
    output_tokens: int | None
    trace: str | None
    tool_calls: tuple[str, ...]


class AssistantFacade:
    def __init__(
        self,
        provider: LLMProvider,
        model: str,
        memory: ConversationMemory,
        tools: ToolRegistry,
        guardrails: GuardrailService,
        traces: TraceRecorder | None,
        max_new_tokens: int,
        assistant_id: str,
    ) -> None:
        self.provider = provider
        self.model = model
        self.memory = memory
        self.tools = tools
        self.guardrails = guardrails
        self.traces = traces
        self.max_new_tokens = max_new_tokens
        self.assistant_id = assistant_id
        self.agent_loop = AgentLoop(provider, tools, guardrails)

    async def chat(
        self,
        session_id: str,
        message: str,
        record_trace: bool = True,
    ) -> AssistantResponse:
        run_id = str(uuid4())
        started = time.perf_counter()
        events: list[TraceEvent] = []
        safety_started = time.perf_counter()
        safety = self.guardrails.check_input(message)
        events.append(
            TraceEvent(
                event_type=TraceEventType.SAFETY,
                name="input_check",
                latency_ms=(time.perf_counter() - safety_started) * 1000,
                metadata={"action": safety.action.value, "reason": safety.reason},
            )
        )
        if safety.action == SafetyAction.REFUSE:
            return self._response(
                run_id,
                session_id,
                started,
                self.guardrails.safe_response_text(safety),
                safety.action.value,
                0.0,
                PricingMethod.NOT_AVAILABLE,
                None,
                None,
                events,
                (),
                record_trace,
            )

        messages = [ChatMessage(Role.SYSTEM, SYSTEM_PROMPT), *self.memory.get(session_id)]
        messages.append(ChatMessage(Role.USER, message))

        loop_result = await self.agent_loop.run(
            run_id,
            session_id,
            self.model,
            self.max_new_tokens,
            messages,
            message,
            safety,
        )
        events.extend(loop_result.events)
        result = loop_result.result
        self.memory.append(session_id, ChatMessage(Role.USER, message))
        self.memory.append(session_id, ChatMessage(Role.ASSISTANT, result.text))
        return self._response(
            run_id,
            session_id,
            started,
            result.text,
            safety.action.value,
            result.estimated_cost_usd,
            result.pricing_method,
            result.input_tokens,
            result.output_tokens,
            events,
            tuple(tool.name for tool in loop_result.tool_results),
            record_trace,
        )

    def _response(
        self,
        run_id: str,
        session_id: str,
        started: float,
        text: str,
        safety_action: str,
        estimated_cost_usd: float,
        pricing_method: PricingMethod,
        input_tokens: int | None,
        output_tokens: int | None,
        events: list[TraceEvent],
        tool_calls: tuple[str, ...] = (),
        record_trace: bool = True,
    ) -> AssistantResponse:
        latency_ms = (time.perf_counter() - started) * 1000
        events.append(
            TraceEvent(
                event_type=TraceEventType.RESPONSE,
                name="final",
                latency_ms=latency_ms,
                metadata={"chars": len(text)},
            )
        )
        trace = ConversationTrace(
            run_id=run_id,
            session_id=session_id,
            assistant_id=self.assistant_id,
            provider_id=self.provider.provider_id,
            total_latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=estimated_cost_usd,
            pricing_method=pricing_method,
            events=tuple(events),
        )
        if record_trace and self.traces is not None:
            self.traces.record(trace)
        return AssistantResponse(
            run_id=run_id,
            session_id=session_id,
            text=text,
            provider_id=self.provider.provider_id,
            latency_ms=latency_ms,
            estimated_cost_usd=estimated_cost_usd,
            pricing_method=pricing_method.value,
            safety_action=safety_action,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            trace=format_trace(trace),
            tool_calls=tool_calls,
        )
