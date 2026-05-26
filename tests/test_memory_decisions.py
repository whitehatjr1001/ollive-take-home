import pytest

from ollie_assistants.assistant.memory_decisions import MemoryDecisionService
from ollie_assistants.llm.types import ChatConfig, ChatMessage, ChatResult


class DecisionProvider:
    provider_id = "decision"

    def __init__(self, text: str | None = None) -> None:
        self.text = text

    async def chat(self, messages: list[ChatMessage], config: ChatConfig) -> ChatResult:
        if self.text is not None:
            return ChatResult(text=self.text)
        if "should_search" in messages[0].content:
            return ChatResult(
                text='{"should_search": true, "query": "name", "reason": "asks name"}'
            )
        return ChatResult(
            text='{"should_write": true, "content": "my name is Rohith", '
            '"kind": "name", "reason": "stable user fact"}'
        )


@pytest.mark.asyncio
async def test_memory_decision_service_parses_search_decision() -> None:
    decision = await MemoryDecisionService(DecisionProvider(), "m").decide_search(
        "what is my name?"
    )

    assert decision.should_search
    assert decision.query == "name"


@pytest.mark.asyncio
async def test_memory_decision_service_parses_write_decision() -> None:
    decision = await MemoryDecisionService(DecisionProvider(), "m").decide_write(
        "my name is Rohith",
        "Nice to meet you.",
    )

    assert decision.should_write
    assert decision.kind == "name"


@pytest.mark.asyncio
async def test_memory_decision_service_treats_primitive_json_as_empty() -> None:
    decision = await MemoryDecisionService(DecisionProvider("false"), "m").decide_search(
        "what is my name?"
    )

    assert not decision.should_search
    assert decision.query == "what is my name?"
