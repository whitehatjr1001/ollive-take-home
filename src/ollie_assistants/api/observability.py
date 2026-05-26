from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

router = APIRouter(prefix="/observability", tags=["observability"])


class RunSummary(BaseModel):
    run_id: str
    session_id: str
    assistant_id: str
    provider_id: str
    created_at: str
    total_latency_ms: float
    input_tokens: int | None
    output_tokens: int | None
    estimated_cost_usd: float
    pricing_method: str
    tool_calls: str
    safety_action: str | None
    event_count: int


class RunListResponse(BaseModel):
    runs: list[RunSummary]


class RunDetailResponse(BaseModel):
    run: dict


@router.get("/runs", response_model=RunListResponse)
def list_runs(
    request: Request,
    limit: int = Query(default=50, ge=1, le=500),
    q: str | None = None,
) -> RunListResponse:
    store = request.app.state.assistant_factory.trace_store
    return RunListResponse(runs=[RunSummary(**row) for row in store.list_runs(limit, q)])


@router.get("/runs/{run_id}", response_model=RunDetailResponse)
def get_run(run_id: str, request: Request) -> RunDetailResponse:
    store = request.app.state.assistant_factory.trace_store
    run = store.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    return RunDetailResponse(run=run)
