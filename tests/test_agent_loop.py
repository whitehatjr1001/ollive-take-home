import json

import pytest

from ollie_assistants.assistant.agent_loop import AgentLoop
from ollie_assistants.assistant.tools import default_tool_registry
from ollie_assistants.llm.types import ChatConfig, ChatMessage, ChatResult, Role
from ollie_assistants.memory.store import InMemoryStore
from ollie_assistants.memory.tools import MemoryTools
from ollie_assistants.safety.guardrails import GuardrailService


class CapturingProvider:
    provider_id = "test"

    def __init__(self) -> None:
        self.messages = []
        self.calls = []
        self.main_messages = []

    async def chat(self, messages, config: ChatConfig) -> ChatResult:
        self.calls.append(list(messages))
        self.messages = list(messages)
        system_text = messages[0].content if messages and messages[0].role == Role.SYSTEM else ""
        if "should_search" in system_text:
            return ChatResult(
                text=json.dumps(
                    {
                        "should_search": True,
                        "query": "answer style preference",
                        "reason": "user asks remembered preference",
                    }
                )
            )
        if "should_write" in system_text:
            return ChatResult(
                text=json.dumps(
                    {
                        "should_write": True,
                        "content": "I prefer concise answers",
                        "kind": "preference",
                        "reason": "stable preference",
                    }
                )
            )
        self.main_messages = list(messages)
        return ChatResult(text="ok")


@pytest.mark.asyncio
async def test_agent_loop_executes_memory_tool_before_llm() -> None:
    guardrails = GuardrailService()
    provider = CapturingProvider()
    memory_tools = MemoryTools(InMemoryStore())
    memory_tools.remember_user_fact("s1", "I prefer concise answers", "setup", "preference")
    loop = AgentLoop(
        provider,
        default_tool_registry(memory_tools),
        guardrails,
    )
    safety = guardrails.check_input("what is my answer style preference?")

    result = await loop.run(
        run_id="run-1",
        session_id="s1",
        model="test",
        max_new_tokens=10,
        messages=[ChatMessage(Role.USER, "what is my answer style preference?")],
        user_message="what is my answer style preference?",
        safety_decision=safety,
    )

    tool_names = [tool.name for tool in result.tool_results]
    assert "search_memory" in tool_names
    assert "remember_user_fact" in tool_names
    assert all(message.role != Role.TOOL for message in provider.main_messages)
    assert any("Relevant memory" in message.content for message in provider.main_messages)
    assert any(event.name == "memory_search_decision" for event in result.events)
    assert any(event.name == "memory_write_decision" for event in result.events)
