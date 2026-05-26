import time
from dataclasses import dataclass

from ollie_assistants.assistant.memory_decisions import MemoryDecisionService
from ollie_assistants.assistant.tools import ToolRegistry, ToolResult
from ollie_assistants.llm.providers import LLMProvider
from ollie_assistants.llm.types import ChatConfig, ChatMessage, ChatResult, Role
from ollie_assistants.observability.types import TraceEvent, TraceEventType
from ollie_assistants.safety.guardrails import GuardrailService
from ollie_assistants.safety.policy import SafetyAction, SafetyDecision


@dataclass(frozen=True)
class AgentLoopResult:
    result: ChatResult
    safety_decision: SafetyDecision
    tool_results: tuple[ToolResult, ...]
    events: tuple[TraceEvent, ...]


class AgentLoop:
    def __init__(
        self,
        provider: LLMProvider,
        tools: ToolRegistry,
        guardrails: GuardrailService,
        max_tool_rounds: int = 3,
    ) -> None:
        self.provider = provider
        self.tools = tools
        self.guardrails = guardrails
        self.max_tool_rounds = max_tool_rounds
        self.memory_decisions: MemoryDecisionService | None = None

    async def run(
        self,
        run_id: str,
        session_id: str,
        model: str,
        max_new_tokens: int,
        messages: list[ChatMessage],
        user_message: str,
        safety_decision: SafetyDecision,
    ) -> AgentLoopResult:
        self.memory_decisions = self.memory_decisions or MemoryDecisionService(self.provider, model)
        events: list[TraceEvent] = []
        tool_results: list[ToolResult] = []
        memory_started = time.perf_counter()
        search_decision = await self.memory_decisions.decide_search(user_message)
        events.append(
            TraceEvent(
                event_type=TraceEventType.LLM,
                name="memory_search_decision",
                latency_ms=(time.perf_counter() - memory_started) * 1000,
                metadata={
                    "should_search": search_decision.should_search,
                    "query": search_decision.query,
                    "reason": search_decision.reason,
                },
            )
        )
        if search_decision.should_search:
            memory_result = self.tools.search_memory_context(
                session_id,
                search_decision.query,
                safety_decision,
            )
            tool_results.append(memory_result)
            messages.append(
                ChatMessage(
                    Role.SYSTEM,
                    f"Relevant memory for this user: {memory_result.content}",
                )
            )
            events.append(
                TraceEvent(
                    event_type=TraceEventType.TOOL,
                    name=memory_result.name,
                    latency_ms=0,
                    metadata={"content": memory_result.content},
                )
            )
        for _ in range(self.max_tool_rounds):
            tool_started = time.perf_counter()
            tool_result = self.tools.maybe_run(
                session_id,
                user_message,
                safety_decision,
                run_id,
            )
            if tool_result is None:
                break
            tool_results.append(tool_result)
            messages.append(
                ChatMessage(
                    Role.SYSTEM,
                    f"Local tool result from {tool_result.name}: {tool_result.content}",
                )
            )
            events.append(
                TraceEvent(
                    event_type=TraceEventType.TOOL,
                    name=tool_result.name,
                    latency_ms=(time.perf_counter() - tool_started) * 1000,
                    metadata={"content": tool_result.content},
                )
            )
            break

        llm_started = time.perf_counter()
        result = await self.provider.chat(
            messages,
            ChatConfig(model=model, max_new_tokens=max_new_tokens),
        )
        events.append(
            TraceEvent(
                event_type=TraceEventType.LLM,
                name=self.provider.provider_id,
                latency_ms=(time.perf_counter() - llm_started) * 1000,
                metadata={"model": model},
            )
        )
        output_decision = self.guardrails.check_output(result.text)
        if output_decision.action != SafetyAction.ALLOW:
            result = ChatResult(text=self.guardrails.safe_response_text(output_decision))
            events.append(
                TraceEvent(
                    event_type=TraceEventType.SAFETY,
                    name="output_check",
                    latency_ms=0,
                    metadata={
                        "action": output_decision.action.value,
                        "category": output_decision.category.value,
                        "reason": output_decision.reason,
                    },
                )
            )
            return AgentLoopResult(result, output_decision, tuple(tool_results), tuple(events))
        write_started = time.perf_counter()
        write_decision = await self.memory_decisions.decide_write(user_message, result.text)
        events.append(
            TraceEvent(
                event_type=TraceEventType.LLM,
                name="memory_write_decision",
                latency_ms=(time.perf_counter() - write_started) * 1000,
                metadata={
                    "should_write": write_decision.should_write,
                    "content": write_decision.content,
                    "kind": write_decision.kind,
                    "reason": write_decision.reason,
                },
            )
        )
        if write_decision.should_write and write_decision.content:
            memory_write = self.tools.remember_user_fact(
                session_id,
                write_decision.content,
                write_decision.kind,
                safety_decision,
                run_id,
            )
            tool_results.append(memory_write)
            events.append(
                TraceEvent(
                    event_type=TraceEventType.TOOL,
                    name=memory_write.name,
                    latency_ms=0,
                    metadata={"content": memory_write.content},
                )
            )
        events.append(
            TraceEvent(
                event_type=TraceEventType.SAFETY,
                name="output_check",
                latency_ms=0,
                metadata={
                    "action": output_decision.action.value,
                    "category": output_decision.category.value,
                    "reason": output_decision.reason,
                },
            )
        )
        return AgentLoopResult(result, safety_decision, tuple(tool_results), tuple(events))
