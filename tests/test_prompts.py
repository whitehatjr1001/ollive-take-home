from ollie_assistants.assistant.prompts import SYSTEM_PROMPT


def test_system_prompt_describes_tool_protocol_and_examples() -> None:
    assert "Tool protocol" in SYSTEM_PROMPT
    assert "search_memory" in SYSTEM_PROMPT
    assert "Examples:" in SYSTEM_PROMPT
    assert "Never guess personal facts" in SYSTEM_PROMPT
    assert 'explicitly says "remember that ..."' not in SYSTEM_PROMPT
