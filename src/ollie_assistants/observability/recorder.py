import json
import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

from ollie_assistants.observability.types import ConversationTrace


class TraceRecorder(Protocol):
    def record(self, trace: ConversationTrace) -> None:
        ...


class JsonlTraceRecorder:
    def __init__(self, path: Path) -> None:
        self.path = path

    def record(self, trace: ConversationTrace) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(asdict(trace), ensure_ascii=False) + "\n")


class CompositeTraceRecorder:
    def __init__(self, recorders: tuple[TraceRecorder, ...]) -> None:
        self.recorders = recorders

    def record(self, trace: ConversationTrace) -> None:
        for recorder in self.recorders:
            recorder.record(trace)


class SqliteTraceRecorder:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def record(self, trace: ConversationTrace) -> None:
        row = trace_to_row(trace)
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                """
                insert or replace into runs (
                    run_id, session_id, assistant_id, provider_id, created_at,
                    total_latency_ms, input_tokens, output_tokens,
                    estimated_cost_usd, pricing_method, tool_calls, safety_action,
                    event_count, trace_json
                )
                values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                row,
            )

    def list_runs(self, limit: int = 100, query: str | None = None) -> list[dict]:
        sql = """
            select run_id, session_id, assistant_id, provider_id, created_at,
                   total_latency_ms, input_tokens, output_tokens,
                   estimated_cost_usd, pricing_method, tool_calls, safety_action,
                   event_count
            from runs
        """
        params: list[str | int] = []
        if query:
            sql += """
                where session_id like ?
                   or assistant_id like ?
                   or provider_id like ?
                   or tool_calls like ?
                   or trace_json like ?
            """
            pattern = f"%{query}%"
            params.extend([pattern, pattern, pattern, pattern, pattern])
        sql += " order by created_at desc limit ?"
        params.append(limit)
        with sqlite3.connect(self.path) as connection:
            connection.row_factory = sqlite3.Row
            return [dict(row) for row in connection.execute(sql, params).fetchall()]

    def get_run(self, run_id: str) -> dict | None:
        with sqlite3.connect(self.path) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                "select * from runs where run_id = ?",
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        data = dict(row)
        data["trace"] = json.loads(data.pop("trace_json"))
        return data

    def record_eval_run(
        self,
        eval_run_id: str,
        created_at: str,
        include_benchmark: bool,
        use_llm_judge: bool,
        report_markdown: str,
        summary: dict,
    ) -> None:
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                """
                insert or replace into eval_runs (
                    eval_run_id, created_at, include_benchmark, use_llm_judge,
                    case_count, oss_hallucination_rate, frontier_hallucination_rate,
                    oss_bias_harm_rate, frontier_bias_harm_rate,
                    oss_safety_failure_rate, frontier_safety_failure_rate,
                    oss_avg_latency_ms, frontier_avg_latency_ms,
                    oss_estimated_cost_usd, frontier_estimated_cost_usd,
                    benchmark_count, report_markdown, summary_json
                )
                values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                eval_run_to_row(
                    eval_run_id,
                    created_at,
                    include_benchmark,
                    use_llm_judge,
                    report_markdown,
                    summary,
                ),
            )

    def start_eval_run(
        self,
        eval_run_id: str,
        created_at: str,
        include_benchmark: bool,
        use_llm_judge: bool,
    ) -> None:
        self.record_eval_run(
            eval_run_id=eval_run_id,
            created_at=created_at,
            include_benchmark=include_benchmark,
            use_llm_judge=use_llm_judge,
            report_markdown="",
            summary=empty_eval_summary(),
        )
        self.update_eval_status(eval_run_id, "running")

    def update_eval_status(
        self,
        eval_run_id: str,
        status: str,
        error_message: str | None = None,
    ) -> None:
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                """
                update eval_runs
                set status = ?, error_message = ?, completed_at = datetime('now')
                where eval_run_id = ?
                """,
                (status, error_message, eval_run_id),
            )

    def list_eval_runs(self, limit: int = 50) -> list[dict]:
        with sqlite3.connect(self.path) as connection:
            connection.row_factory = sqlite3.Row
            return [
                dict(row)
                for row in connection.execute(
                    """
                    select eval_run_id, created_at, completed_at, status, error_message,
                           include_benchmark, use_llm_judge,
                           case_count, oss_hallucination_rate,
                           frontier_hallucination_rate, oss_bias_harm_rate,
                           frontier_bias_harm_rate, oss_safety_failure_rate,
                           frontier_safety_failure_rate, oss_avg_latency_ms,
                           frontier_avg_latency_ms, oss_estimated_cost_usd,
                           frontier_estimated_cost_usd, benchmark_count
                    from eval_runs
                    order by created_at desc
                    limit ?
                    """,
                    (limit,),
                ).fetchall()
            ]

    def get_eval_run(self, eval_run_id: str) -> dict | None:
        with sqlite3.connect(self.path) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                "select * from eval_runs where eval_run_id = ?",
                (eval_run_id,),
            ).fetchone()
        if row is None:
            return None
        data = dict(row)
        data["summary"] = json.loads(data.pop("summary_json"))
        return data

    def _init(self) -> None:
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                """
                create table if not exists runs (
                    run_id text primary key,
                    session_id text not null,
                    assistant_id text not null,
                    provider_id text not null,
                    created_at text not null,
                    total_latency_ms real not null,
                    input_tokens integer,
                    output_tokens integer,
                    estimated_cost_usd real not null,
                    pricing_method text not null,
                    tool_calls text not null,
                    safety_action text,
                    event_count integer not null,
                    trace_json text not null
                )
                """
            )
            connection.execute(
                "create index if not exists idx_runs_created_at on runs(created_at)"
            )
            connection.execute(
                "create index if not exists idx_runs_session_id on runs(session_id)"
            )
            connection.execute(
                """
                create table if not exists eval_runs (
                    eval_run_id text primary key,
                    created_at text not null,
                    completed_at text,
                    status text not null default 'completed',
                    error_message text,
                    include_benchmark integer not null,
                    use_llm_judge integer not null,
                    case_count integer not null,
                    oss_hallucination_rate real not null,
                    frontier_hallucination_rate real not null,
                    oss_bias_harm_rate real not null,
                    frontier_bias_harm_rate real not null,
                    oss_safety_failure_rate real not null,
                    frontier_safety_failure_rate real not null,
                    oss_avg_latency_ms real not null,
                    frontier_avg_latency_ms real not null,
                    oss_estimated_cost_usd real not null,
                    frontier_estimated_cost_usd real not null,
                    benchmark_count integer not null,
                    report_markdown text not null,
                    summary_json text not null
                )
                """
            )
            connection.execute(
                "create index if not exists idx_eval_runs_created_at on eval_runs(created_at)"
            )
            _ensure_column(connection, "eval_runs", "completed_at", "text")
            _ensure_column(
                connection,
                "eval_runs",
                "status",
                "text not null default 'completed'",
            )
            _ensure_column(connection, "eval_runs", "error_message", "text")


def trace_to_row(trace: ConversationTrace) -> tuple:
    trace_dict = asdict(trace)
    tool_calls = [
        event["name"]
        for event in trace_dict["events"]
        if event["event_type"] == "tool"
    ]
    safety_action = next(
        (
            event["metadata"].get("action")
            for event in trace_dict["events"]
            if event["event_type"] == "safety" and event["name"] == "input_check"
        ),
        None,
    )
    return (
        trace.run_id,
        trace.session_id,
        trace.assistant_id,
        trace.provider_id,
        trace.created_at,
        trace.total_latency_ms,
        trace.input_tokens,
        trace.output_tokens,
        trace.estimated_cost_usd,
        trace.pricing_method.value,
        ", ".join(tool_calls),
        safety_action,
        len(trace.events),
        json.dumps(trace_dict, ensure_ascii=False),
    )


def eval_run_to_row(
    eval_run_id: str,
    created_at: str,
    include_benchmark: bool,
    use_llm_judge: bool,
    report_markdown: str,
    summary: dict,
) -> tuple:
    metrics = summary["required_metrics"]
    assistants = summary["assistants"]
    return (
        eval_run_id,
        created_at,
        int(include_benchmark),
        int(use_llm_judge),
        summary["case_count"],
        metrics["hallucination_rate"]["oss_failure_rate"],
        metrics["hallucination_rate"]["frontier_failure_rate"],
        metrics["bias_harmful_output_rate"]["oss_failure_rate"],
        metrics["bias_harmful_output_rate"]["frontier_failure_rate"],
        metrics["content_safety_failure_rate"]["oss_failure_rate"],
        metrics["content_safety_failure_rate"]["frontier_failure_rate"],
        assistants["oss"]["average_latency_ms"],
        assistants["frontier"]["average_latency_ms"],
        assistants["oss"]["estimated_cost_usd"],
        assistants["frontier"]["estimated_cost_usd"],
        len(summary.get("benchmark_rows", [])),
        report_markdown,
        json.dumps(summary, ensure_ascii=False),
    )


def empty_eval_summary() -> dict:
    return {
        "case_count": 0,
        "required_metrics": {
            "hallucination_rate": {"oss_failure_rate": 0.0, "frontier_failure_rate": 0.0},
            "bias_harmful_output_rate": {
                "oss_failure_rate": 0.0,
                "frontier_failure_rate": 0.0,
            },
            "content_safety_failure_rate": {
                "oss_failure_rate": 0.0,
                "frontier_failure_rate": 0.0,
            },
        },
        "assistants": {
            "oss": {
                "average_latency_ms": 0.0,
                "estimated_cost_usd": 0.0,
                "input_tokens": 0,
                "output_tokens": 0,
            },
            "frontier": {
                "average_latency_ms": 0.0,
                "estimated_cost_usd": 0.0,
                "input_tokens": 0,
                "output_tokens": 0,
            },
        },
        "benchmark_rows": [],
    }


def _ensure_column(connection: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
    existing = {
        row[1]
        for row in connection.execute(f"pragma table_info({table})").fetchall()
    }
    if column not in existing:
        connection.execute(f"alter table {table} add column {column} {ddl}")
