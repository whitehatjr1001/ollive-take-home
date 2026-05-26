from collections import defaultdict, deque
from collections.abc import Sequence

from ollie_assistants.llm.types import ChatMessage


class ConversationMemory:
    def __init__(self, max_turns: int) -> None:
        self.max_messages = max_turns * 2
        self._messages: dict[str, deque[ChatMessage]] = defaultdict(
            lambda: deque(maxlen=self.max_messages)
        )

    def get(self, session_id: str) -> Sequence[ChatMessage]:
        return tuple(self._messages[session_id])

    def append(self, session_id: str, message: ChatMessage) -> None:
        self._messages[session_id].append(message)
