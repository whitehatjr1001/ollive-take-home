from ollie_assistants.assistant.memory import ConversationMemory
from ollie_assistants.assistant.service import AssistantFacade
from ollie_assistants.assistant.tools import default_tool_registry
from ollie_assistants.llm.factory import default_llm_factory
from ollie_assistants.memory.store import InMemoryStore
from ollie_assistants.memory.tools import MemoryTools
from ollie_assistants.observability.recorder import (
    CompositeTraceRecorder,
    JsonlTraceRecorder,
    SqliteTraceRecorder,
)
from ollie_assistants.safety.guardrails import GuardrailService
from ollie_assistants.settings import Settings


class AssistantFactory:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.llm_factory = default_llm_factory()
        self.memory_store = InMemoryStore()
        self.trace_store = SqliteTraceRecorder(settings.observability_db_path)

    def create_oss(self) -> AssistantFacade:
        return self._create("oss", self.settings.oss_provider, self.settings.oss_model)

    def create_frontier(self) -> AssistantFacade:
        return self._create(
            "frontier",
            self.settings.frontier_provider,
            self.settings.frontier_model,
        )

    def _create(self, assistant_id: str, provider_id: str, model: str) -> AssistantFacade:
        traces = (
            CompositeTraceRecorder(
                (
                    JsonlTraceRecorder(self.settings.traces_path),
                    self.trace_store,
                )
            )
            if self.settings.analytics_enabled
            else None
        )
        return AssistantFacade(
            provider=self.llm_factory.create(provider_id, self.settings),
            model=model,
            memory=ConversationMemory(self.settings.memory_turns),
            tools=default_tool_registry(MemoryTools(self.memory_store)),
            guardrails=GuardrailService(),
            traces=traces,
            max_new_tokens=self.settings.max_new_tokens,
            assistant_id=assistant_id,
        )
