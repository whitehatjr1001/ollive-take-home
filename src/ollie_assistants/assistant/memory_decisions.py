import json
from dataclasses import dataclass

from ollie_assistants.llm.providers import LLMProvider
from ollie_assistants.llm.types import ChatConfig, ChatMessage, Role


@dataclass(frozen=True)
class MemorySearchDecision:
    should_search: bool
    query: str
    reason: str


@dataclass(frozen=True)
class MemoryWriteDecision:
    should_write: bool
    content: str
    kind: str
    reason: str


class MemoryDecisionService:
    def __init__(self, provider: LLMProvider, model: str) -> None:
        self.provider = provider
        self.model = model

    async def decide_search(self, user_message: str) -> MemorySearchDecision:
        result = await self.provider.chat(
            [
                ChatMessage(
                    Role.SYSTEM,
                    "Return strict JSON. Decide if answering the user requires searching "
                    "remembered user facts/preferences/context. Schema: "
                    '{"should_search": boolean, "query": string, "reason": string}. '
                    "Search when the question depends on durable user-specific context "
                    "rather than the current request alone. Do not search for one-off "
                    "tasks or general knowledge questions.",
                ),
                ChatMessage(Role.USER, user_message),
            ],
            ChatConfig(model=self.model, max_new_tokens=120, temperature=0),
        )
        parsed = self._loads(result.text)
        return MemorySearchDecision(
            should_search=bool(parsed.get("should_search", False)),
            query=str(parsed.get("query") or user_message),
            reason=str(parsed.get("reason") or "no reason"),
        )

    async def decide_write(self, user_message: str, assistant_text: str) -> MemoryWriteDecision:
        result = await self.provider.chat(
            [
                ChatMessage(
                    Role.SYSTEM,
                    "Return strict JSON. Decide if the user supplied a durable memory "
                    "worth saving for future turns. Save stable facts/preferences only, "
                    "not transient requests or unsafe content. Schema: "
                    '{"should_write": boolean, "content": string, "kind": string, '
                    '"reason": string}. Prefer plain kind labels such as name, '
                    "preference, location, timezone, or user_fact.",
                ),
                ChatMessage(
                    Role.USER,
                    json.dumps({"user_message": user_message, "assistant_text": assistant_text}),
                ),
            ],
            ChatConfig(model=self.model, max_new_tokens=160, temperature=0),
        )
        parsed = self._loads(result.text)
        return MemoryWriteDecision(
            should_write=bool(parsed.get("should_write", False)),
            content=str(parsed.get("content") or ""),
            kind=str(parsed.get("kind") or "user_fact"),
            reason=str(parsed.get("reason") or "no reason"),
        )

    def _loads(self, text: str) -> dict:
        try:
            parsed = json.loads(text)
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}
