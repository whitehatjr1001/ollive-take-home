from ollie_assistants.api.schemas import ChatResponse, CompareChatResponse


def test_compare_chat_response_schema_contains_both_assistants() -> None:
    response = ChatResponse(
        run_id="run",
        session_id="s",
        text="hi",
        provider_id="provider",
        latency_ms=1,
        estimated_cost_usd=0,
        pricing_method="not_available",
        safety_action="allow",
        input_tokens=None,
        output_tokens=None,
        trace=None,
        tool_calls=(),
    )

    compare = CompareChatResponse(message="hi", oss=response, frontier=response)

    assert compare.oss.text == "hi"
    assert compare.frontier.text == "hi"


def test_compare_chat_response_schema_allows_partial_errors() -> None:
    compare = CompareChatResponse(message="hi", oss=None, frontier=None, errors={"oss": "bad"})

    assert compare.errors["oss"] == "bad"
