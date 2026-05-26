from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4_000)
    session_id: str = "default"
    assistant: str = "oss"
    include_trace: bool = False


class ChatResponse(BaseModel):
    run_id: str
    session_id: str
    text: str
    provider_id: str
    latency_ms: float
    estimated_cost_usd: float
    pricing_method: str
    safety_action: str
    input_tokens: int | None
    output_tokens: int | None
    trace: str | None
    tool_calls: tuple[str, ...]


class CompareChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4_000)
    session_id: str = "default"
    include_trace: bool = False


class CompareChatResponse(BaseModel):
    message: str
    oss: ChatResponse | None = None
    frontier: ChatResponse | None = None
    errors: dict[str, str] = {}
