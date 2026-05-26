import asyncio
import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from ollie_assistants.api.schemas import (
    ChatRequest,
    ChatResponse,
    CompareChatRequest,
    CompareChatResponse,
)

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    assistants = request.app.state.assistants
    assistant = assistants.get(payload.assistant)
    if assistant is None:
        raise HTTPException(status_code=400, detail="assistant must be 'oss' or 'frontier'")
    try:
        response = await assistant.chat(payload.session_id, payload.message)
    except Exception as err:
        raise HTTPException(status_code=502, detail=str(err)) from err
    data = response.__dict__ | {"trace": response.trace if payload.include_trace else None}
    return ChatResponse(**data)


@router.post("/compare", response_model=CompareChatResponse)
async def compare_chat(payload: CompareChatRequest, request: Request) -> CompareChatResponse:
    assistants = request.app.state.assistants
    oss_result, frontier_result = await asyncio.gather(
        assistants["oss"].chat(payload.session_id, payload.message),
        assistants["frontier"].chat(payload.session_id, payload.message),
        return_exceptions=True,
    )
    errors: dict[str, str] = {}
    oss = _chat_response_or_error("oss", oss_result, payload.include_trace, errors)
    frontier = _chat_response_or_error(
        "frontier",
        frontier_result,
        payload.include_trace,
        errors,
    )
    return CompareChatResponse(
        message=payload.message,
        oss=oss,
        frontier=frontier,
        errors=errors,
    )


def _chat_response_or_error(
    assistant_id: str,
    result,
    include_trace: bool,
    errors: dict[str, str],
) -> ChatResponse | None:
    if isinstance(result, Exception):
        errors[assistant_id] = str(result)
        return None
    return ChatResponse(
        **(result.__dict__ | {"trace": result.trace if include_trace else None})
    )


@router.post("/stream")
async def chat_stream(payload: ChatRequest, request: Request) -> StreamingResponse:
    async def events() -> AsyncIterator[str]:
        async for event in _stream_one(payload.assistant, payload, request):
            yield _jsonl(event)

    return StreamingResponse(events(), media_type="application/x-ndjson")


@router.post("/compare/stream")
async def compare_chat_stream(
    payload: CompareChatRequest,
    request: Request,
) -> StreamingResponse:
    async def events() -> AsyncIterator[str]:
        oss_payload = ChatRequest(
            assistant="oss",
            session_id=payload.session_id,
            message=payload.message,
            include_trace=payload.include_trace,
        )
        frontier_payload = ChatRequest(
            assistant="frontier",
            session_id=payload.session_id,
            message=payload.message,
            include_trace=payload.include_trace,
        )
        async for event in _stream_many(
            (
                ("oss", oss_payload),
                ("frontier", frontier_payload),
            ),
            request,
        ):
            yield _jsonl(event)

    return StreamingResponse(events(), media_type="application/x-ndjson")


async def _stream_many(
    payloads: tuple[tuple[str, ChatRequest], ...],
    request: Request,
) -> AsyncIterator[dict]:
    queue: asyncio.Queue[dict | None] = asyncio.Queue()
    tasks = [
        asyncio.create_task(_pump_stream(assistant_id, payload, request, queue))
        for assistant_id, payload in payloads
    ]
    remaining = len(tasks)
    try:
        while remaining:
            event = await queue.get()
            if event is None:
                remaining -= 1
                continue
            yield event
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def _pump_stream(
    assistant_id: str,
    payload: ChatRequest,
    request: Request,
    queue: asyncio.Queue[dict | None],
) -> None:
    try:
        async for event in _stream_one(assistant_id, payload, request):
            await queue.put(event)
    finally:
        await queue.put(None)


async def _stream_one(
    assistant_id: str,
    payload: ChatRequest,
    request: Request,
) -> AsyncIterator[dict]:
    assistants = request.app.state.assistants
    assistant = assistants.get(assistant_id)
    if assistant is None:
        yield {"type": "error", "assistant": assistant_id, "text": "unknown assistant"}
        return
    yield {"type": "status", "assistant": assistant_id, "text": "checking safety and tools"}
    try:
        response = await assistant.chat(payload.session_id, payload.message)
    except Exception as err:
        yield {"type": "error", "assistant": assistant_id, "text": str(err)}
        return
    for tool_call in response.tool_calls:
        yield {"type": "tool", "assistant": assistant_id, "name": tool_call}
    for chunk in _word_chunks(response.text):
        yield {"type": "token", "assistant": assistant_id, "text": chunk}
    yield {
        "type": "final",
        "assistant": assistant_id,
        "run_id": response.run_id,
        "latency_ms": response.latency_ms,
        "estimated_cost_usd": response.estimated_cost_usd,
        "pricing_method": response.pricing_method,
        "tool_calls": response.tool_calls,
        "trace": response.trace if payload.include_trace else None,
    }


def _word_chunks(text: str) -> list[str]:
    words = text.split(" ")
    return [f"{word} " for word in words[:-1]] + ([words[-1]] if words else [])


def _jsonl(event: dict) -> str:
    return json.dumps(event, ensure_ascii=False) + "\n"
