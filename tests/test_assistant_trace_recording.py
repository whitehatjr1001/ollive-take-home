import pytest

from ollie_assistants.assistant.memory import ConversationMemory
from ollie_assistants.assistant.service import AssistantFacade
from ollie_assistants.assistant.tools import default_tool_registry
from ollie_assistants.llm.types import ChatConfig, ChatMessage, ChatResult, Role
from ollie_assistants.memory.store import InMemoryStore
from ollie_assistants.memory.tools import MemoryTools
from ollie_assistants.safety.guardrails import GuardrailService


class StaticProvider:
    provider_id = "static"

    async def chat(self, messages: list[ChatMessage], config: ChatConfig) -> ChatResult:
        system_text = messages[0].content if messages and messages[0].role == Role.SYSTEM else ""
        if "should_search" in system_text:
            return ChatResult(text='{"should_search": false, "query": "", "reason": "none"}')
        if "should_write" in system_text:
            return ChatResult(
                text='{"should_write": false, "content": "", "kind": "", "reason": "none"}'
            )
        return ChatResult(text="ok")


class CapturingRecorder:
    def __init__(self) -> None:
        self.count = 0

    def record(self, trace) -> None:
        self.count += 1


def make_assistant(recorder: CapturingRecorder) -> AssistantFacade:
    memory_tools = MemoryTools(InMemoryStore())
    return AssistantFacade(
        provider=StaticProvider(),
        model="test",
        memory=ConversationMemory(max_turns=2),
        tools=default_tool_registry(memory_tools),
        guardrails=GuardrailService(),
        traces=recorder,
        max_new_tokens=10,
        assistant_id="oss",
    )


@pytest.mark.asyncio
async def test_assistant_chat_can_skip_trace_recording() -> None:
    recorder = CapturingRecorder()
    assistant = make_assistant(recorder)

    await assistant.chat("s1", "hi", record_trace=False)

    assert recorder.count == 0


@pytest.mark.asyncio
async def test_assistant_chat_records_trace_by_default() -> None:
    recorder = CapturingRecorder()
    assistant = make_assistant(recorder)

    await assistant.chat("s1", "hi")

    assert recorder.count == 1
