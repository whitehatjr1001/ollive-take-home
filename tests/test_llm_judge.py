import pytest

from ollie_assistants.evals.llm_judge import LLMJudgeVerifier
from ollie_assistants.llm.types import ChatConfig, ChatMessage, ChatResult


class JsonJudgeProvider:
    provider_id = "judge"

    async def chat(self, messages: list[ChatMessage], config: ChatConfig) -> ChatResult:
        return ChatResult(text='{"passed": true, "reason": "safe answer"}')


@pytest.mark.asyncio
async def test_llm_judge_parses_json_review() -> None:
    review = await LLMJudgeVerifier(JsonJudgeProvider(), "judge-model").review_one(
        case_id="case",
        assistant_id="oss",
        metric="content_safety",
        prompt="bad prompt",
        answer="I cannot help with that.",
    )

    assert review.passed
    assert review.reason == "safe answer"
