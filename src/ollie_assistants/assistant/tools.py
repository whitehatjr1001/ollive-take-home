from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from ollie_assistants.memory.tools import MemoryTools
from ollie_assistants.safety.policy import SafetyDecision
from ollie_assistants.safety.tool_policy import ToolPolicy


@dataclass(frozen=True)
class ToolResult:
    name: str
    content: str


class Tool(Protocol):
    name: str
    description: str

    def run(self, input_text: str) -> ToolResult:
        ...


class CurrentTimeTool:
    name = "current_time"
    description = "Returns the current UTC time."

    def run(self, input_text: str) -> ToolResult:
        return ToolResult(name=self.name, content=datetime.now(UTC).isoformat())


class ToolRegistry:
    def __init__(
        self,
        tools: tuple[Tool, ...],
        memory_tools: MemoryTools,
        tool_policy: ToolPolicy,
    ) -> None:
        self._tools = {tool.name: tool for tool in tools}
        self.memory_tools = memory_tools
        self.tool_policy = tool_policy

    def maybe_run(
        self,
        session_id: str,
        input_text: str,
        safety_decision: SafetyDecision,
        run_id: str,
    ) -> ToolResult | None:
        lowered = input_text.lower()
        if "time" in lowered or "date" in lowered:
            self.tool_policy.authorize("current_time", safety_decision)
            return self._tools["current_time"].run(input_text)
        return None

    def remember_user_fact(
        self,
        session_id: str,
        content: str,
        kind: str,
        safety_decision: SafetyDecision,
        run_id: str,
    ) -> ToolResult:
        decision = self.tool_policy.authorize("remember_user_fact", safety_decision)
        if decision.action.value != "allow":
            return ToolResult(name="remember_user_fact", content=decision.reason)
        record = self.memory_tools.remember_user_fact(session_id, content, run_id, kind)
        return ToolResult(name="remember_user_fact", content=f"remembered: {record.content}")

    def search_memory_context(
        self,
        session_id: str,
        input_text: str,
        safety_decision: SafetyDecision,
    ) -> ToolResult:
        self.tool_policy.authorize("search_memory", safety_decision)
        records = self.memory_tools.search_memory(session_id, input_text)
        content = "\n".join(record.content for record in records) or "no relevant memory"
        return ToolResult(name="search_memory", content=content)


def default_tool_registry(memory_tools: MemoryTools) -> ToolRegistry:
    return ToolRegistry((CurrentTimeTool(),), memory_tools, ToolPolicy())
