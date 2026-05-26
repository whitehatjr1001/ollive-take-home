from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import uuid4


@dataclass(frozen=True)
class MemoryRecord:
    session_id: str
    content: str
    kind: str = "user_fact"
    source_run_id: str | None = None
    memory_id: str = field(default_factory=lambda: str(uuid4()))
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
