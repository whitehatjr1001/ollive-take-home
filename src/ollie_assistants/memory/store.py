import re

from ollie_assistants.memory.types import MemoryRecord


class InMemoryStore:
    def __init__(self) -> None:
        self._records: list[MemoryRecord] = []

    def add(self, record: MemoryRecord) -> MemoryRecord:
        self._records.append(record)
        return record

    def search(self, session_id: str, query: str, limit: int = 3) -> tuple[MemoryRecord, ...]:
        query_terms = _terms(query)
        scored: list[tuple[int, MemoryRecord]] = []
        session_records: list[MemoryRecord] = []
        for record in self._records:
            if record.session_id != session_id:
                continue
            session_records.append(record)
            content_terms = _terms(record.content)
            score = len(query_terms & content_terms)
            if score > 0:
                scored.append((score, record))
        scored.sort(key=lambda item: item[0], reverse=True)
        if scored:
            return tuple(record for _, record in scored[:limit])
        return tuple(reversed(session_records[-limit:]))


def _terms(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))
