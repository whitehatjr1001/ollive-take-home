from ollie_assistants.memory.store import InMemoryStore
from ollie_assistants.memory.types import MemoryRecord


class MemoryTools:
    def __init__(self, store: InMemoryStore) -> None:
        self.store = store

    def remember_user_fact(
        self,
        session_id: str,
        content: str,
        source_run_id: str | None = None,
        kind: str = "user_fact",
    ) -> MemoryRecord:
        return self.store.add(
            MemoryRecord(
                session_id=session_id,
                content=content,
                kind=kind,
                source_run_id=source_run_id,
            )
        )

    def search_memory(
        self,
        session_id: str,
        query: str,
        limit: int = 3,
    ) -> tuple[MemoryRecord, ...]:
        return self.store.search(session_id=session_id, query=query, limit=limit)
