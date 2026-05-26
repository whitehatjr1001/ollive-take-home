from datetime import UTC, datetime
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Request
from pydantic import BaseModel

from ollie_assistants.evals.facade import AssistantComparisonService

router = APIRouter(prefix="/evals", tags=["evals"])


class ComparisonRunResponse(BaseModel):
    report_markdown: str
    eval_run_id: str | None = None
    status: str = "completed"


class ComparisonRunRequest(BaseModel):
    include_benchmark: bool = False
    use_llm_judge: bool = False


class EvalRunListResponse(BaseModel):
    runs: list[dict]


class EvalRunDetailResponse(BaseModel):
    run: dict


@router.post("/take-home", response_model=ComparisonRunResponse)
async def run_take_home_comparison(
    payload: ComparisonRunRequest,
    request: Request,
    background_tasks: BackgroundTasks,
) -> ComparisonRunResponse:
    eval_run_id = str(uuid4())
    factory = request.app.state.assistant_factory
    factory.trace_store.start_eval_run(
        eval_run_id=eval_run_id,
        created_at=datetime.now(UTC).isoformat(),
        include_benchmark=payload.include_benchmark,
        use_llm_judge=payload.use_llm_judge,
    )
    background_tasks.add_task(
        run_eval_background,
        factory,
        eval_run_id,
        payload.include_benchmark,
        payload.use_llm_judge,
    )
    return ComparisonRunResponse(report_markdown="", eval_run_id=eval_run_id, status="running")


async def run_eval_background(
    assistant_factory,
    eval_run_id: str,
    include_benchmark: bool,
    use_llm_judge: bool,
) -> None:
    try:
        await AssistantComparisonService(assistant_factory).run_comparison(
            include_benchmark=include_benchmark,
            use_llm_judge=use_llm_judge,
            eval_run_id=eval_run_id,
        )
    except Exception as err:
        assistant_factory.trace_store.update_eval_status(eval_run_id, "failed", str(err))


@router.get("/runs", response_model=EvalRunListResponse)
def list_eval_runs(
    request: Request,
    limit: int = Query(default=20, ge=1, le=200),
) -> EvalRunListResponse:
    store = request.app.state.assistant_factory.trace_store
    return EvalRunListResponse(runs=store.list_eval_runs(limit))


@router.get("/runs/{eval_run_id}", response_model=EvalRunDetailResponse)
def get_eval_run(eval_run_id: str, request: Request) -> EvalRunDetailResponse:
    store = request.app.state.assistant_factory.trace_store
    run = store.get_eval_run(eval_run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="eval run not found")
    return EvalRunDetailResponse(run=run)
