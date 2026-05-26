from ollie_assistants.analytics.pricing import PricingMethod
from ollie_assistants.observability.recorder import SqliteTraceRecorder
from ollie_assistants.observability.types import ConversationTrace, TraceEvent, TraceEventType


def test_sqlite_trace_recorder_records_lists_and_fetches_run(tmp_path) -> None:
    recorder = SqliteTraceRecorder(tmp_path / "runs.sqlite")
    trace = ConversationTrace(
        run_id="run-1",
        session_id="session-1",
        assistant_id="oss",
        provider_id="openai_compatible",
        total_latency_ms=123.4,
        input_tokens=10,
        output_tokens=5,
        estimated_cost_usd=0.001,
        pricing_method=PricingMethod.TOKEN_PRICING,
        events=(
            TraceEvent(
                event_type=TraceEventType.SAFETY,
                name="input_check",
                latency_ms=1.0,
                metadata={"action": "allow"},
            ),
            TraceEvent(
                event_type=TraceEventType.TOOL,
                name="search_memory",
                latency_ms=2.0,
                metadata={"content": "memory hit"},
            ),
        ),
    )

    recorder.record(trace)

    runs = recorder.list_runs()
    detail = recorder.get_run("run-1")

    assert runs[0]["run_id"] == "run-1"
    assert runs[0]["tool_calls"] == "search_memory"
    assert detail is not None
    assert detail["trace"]["run_id"] == "run-1"
    assert recorder.list_runs(query="memory hit")[0]["run_id"] == "run-1"


def test_sqlite_trace_recorder_records_eval_runs(tmp_path) -> None:
    recorder = SqliteTraceRecorder(tmp_path / "runs.sqlite")
    summary = {
        "case_count": 3,
        "required_metrics": {
            "hallucination_rate": {"oss_failure_rate": 0.0, "frontier_failure_rate": 0.0},
            "bias_harmful_output_rate": {
                "oss_failure_rate": 0.5,
                "frontier_failure_rate": 0.0,
            },
            "content_safety_failure_rate": {
                "oss_failure_rate": 0.25,
                "frontier_failure_rate": 0.0,
            },
        },
        "assistants": {
            "oss": {
                "average_latency_ms": 100.0,
                "estimated_cost_usd": 0.01,
                "input_tokens": 10,
                "output_tokens": 5,
            },
            "frontier": {
                "average_latency_ms": 200.0,
                "estimated_cost_usd": 0.02,
                "input_tokens": 11,
                "output_tokens": 6,
            },
        },
        "benchmark_rows": [{"provider": "OSS Modal"}],
    }

    recorder.record_eval_run(
        eval_run_id="eval-1",
        created_at="2026-05-26T00:00:00+00:00",
        include_benchmark=True,
        use_llm_judge=False,
        report_markdown="# Report",
        summary=summary,
    )

    runs = recorder.list_eval_runs()
    detail = recorder.get_eval_run("eval-1")

    assert runs[0]["eval_run_id"] == "eval-1"
    assert runs[0]["status"] == "completed"
    assert runs[0]["benchmark_count"] == 1
    assert detail is not None
    assert detail["summary"]["case_count"] == 3
    assert detail["report_markdown"] == "# Report"


def test_sqlite_trace_recorder_tracks_running_eval_run(tmp_path) -> None:
    recorder = SqliteTraceRecorder(tmp_path / "runs.sqlite")

    recorder.start_eval_run(
        eval_run_id="eval-running",
        created_at="2026-05-26T00:00:00+00:00",
        include_benchmark=True,
        use_llm_judge=True,
    )

    run = recorder.get_eval_run("eval-running")

    assert run is not None
    assert run["status"] == "running"
    assert run["case_count"] == 0
