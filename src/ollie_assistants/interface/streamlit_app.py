import json
from collections.abc import Iterator
from dataclasses import dataclass

import httpx
import streamlit as st

from ollie_assistants.reports.loader import (
    load_assistant_evaluation_report,
    load_cost_latency_report,
    load_oss_deployment_report,
)
from ollie_assistants.settings import get_settings

STREAM_TIMEOUT = httpx.Timeout(connect=20, read=900, write=30, pool=20)
REQUEST_TIMEOUT = httpx.Timeout(connect=10, read=120, write=30, pool=10)


@dataclass(frozen=True)
class AssistantLabel:
    assistant_id: str
    display_name: str
    short_name: str


@st.cache_resource
def get_api_base_url() -> str:
    return get_settings().api_base_url


@st.cache_resource
def get_assistant_labels() -> dict[str, AssistantLabel]:
    settings = get_settings()
    return {
        "oss": AssistantLabel(
            assistant_id="oss",
            display_name=f"OSS Modal - {settings.oss_served_model_name} ({settings.oss_model})",
            short_name=f"OSS - {settings.oss_served_model_name}",
        ),
        "frontier": AssistantLabel(
            assistant_id="frontier",
            display_name=f"Frontier OpenAI - {settings.frontier_model}",
            short_name=f"Frontier - {settings.frontier_model}",
        ),
    }


def main() -> None:
    st.set_page_config(page_title="Ollie Assistants", layout="wide")
    labels = get_assistant_labels()

    with st.sidebar:
        st.title("Ollie Assistants")
        page = st.radio(
            "View",
            ["Chat", "Observability", "Reports"],
            label_visibility="collapsed",
        )
        mode_label = st.radio(
            "Assistant",
            [
                labels["oss"].display_name,
                labels["frontier"].display_name,
                "Compare both assistants",
            ],
        )
        show_trace = st.toggle("Show compact run details", value=False)

    if page == "Observability":
        render_observability_tab()
        return
    if page == "Reports":
        render_reports_tab()
        return

    mode = mode_from_label(mode_label, labels)
    st.title("Assistant Comparison")
    st.caption("Chat with the OSS Modal assistant, the frontier assistant, or both.")

    if "single_messages" not in st.session_state:
        st.session_state.single_messages = []
    if "compare_turns" not in st.session_state:
        st.session_state.compare_turns = []

    if mode == "compare":
        render_compare_history(labels)
    else:
        render_single_history()

    if prompt := st.chat_input("Ask a question"):
        if mode == "compare":
            run_compare_turn(prompt, show_trace, labels)
        else:
            run_single_turn(mode, prompt, show_trace, labels)


def mode_from_label(label: str, labels: dict[str, AssistantLabel]) -> str:
    for assistant_id, assistant_label in labels.items():
        if label == assistant_label.display_name:
            return assistant_id
    return "compare"


def render_single_history() -> None:
    for message in st.session_state.single_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def run_single_turn(
    assistant_id: str,
    prompt: str,
    show_trace: bool,
    labels: dict[str, AssistantLabel],
) -> None:
    st.session_state.single_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        st.caption(labels[assistant_id].short_name)
        content = st.write_stream(stream_single_response(assistant_id, prompt, show_trace))
    st.session_state.single_messages.append({"role": "assistant", "content": content})


def render_compare_history(labels: dict[str, AssistantLabel]) -> None:
    for turn in st.session_state.compare_turns:
        with st.chat_message("user"):
            st.markdown(turn["user"])
        left, right = st.columns(2)
        with left:
            st.markdown(f"#### {labels['oss'].short_name}")
            st.markdown(turn["oss"])
        with right:
            st.markdown(f"#### {labels['frontier'].short_name}")
            st.markdown(turn["frontier"])


def run_compare_turn(
    prompt: str,
    show_trace: bool,
    labels: dict[str, AssistantLabel],
) -> None:
    st.session_state.compare_turns.append({"user": prompt, "oss": "", "frontier": ""})
    with st.chat_message("user"):
        st.markdown(prompt)
    left, right = st.columns(2)
    with left:
        st.markdown(f"#### {labels['oss'].short_name}")
        oss_box = st.empty()
    with right:
        st.markdown(f"#### {labels['frontier'].short_name}")
        frontier_box = st.empty()

    buffers = {"oss": "", "frontier": ""}
    for event in stream_compare_events(prompt, show_trace):
        assistant = event.get("assistant")
        if assistant not in buffers:
            continue
        buffers[assistant] += render_stream_event(event, show_trace)
        if assistant == "oss":
            oss_box.markdown(buffers["oss"])
        else:
            frontier_box.markdown(buffers["frontier"])
    st.session_state.compare_turns[-1] = {
        "user": prompt,
        "oss": buffers["oss"],
        "frontier": buffers["frontier"],
    }


def stream_single_response(
    assistant_id: str,
    prompt: str,
    show_trace: bool,
) -> Iterator[str]:
    for event in stream_single_events(assistant_id, prompt, show_trace):
        yield render_stream_event(event, show_trace)


def stream_single_events(assistant_id: str, prompt: str, show_trace: bool) -> Iterator[dict]:
    with httpx.stream(
        "POST",
        f"{get_api_base_url()}/chat/stream",
        json={
            "assistant": assistant_id,
            "session_id": "streamlit",
            "message": prompt,
            "include_trace": show_trace,
        },
        timeout=STREAM_TIMEOUT,
    ) as response:
        response.raise_for_status()
        yield from iter_jsonl(response)


def stream_compare_events(prompt: str, show_trace: bool) -> Iterator[dict]:
    with httpx.stream(
        "POST",
        f"{get_api_base_url()}/chat/compare/stream",
        json={
            "session_id": "streamlit",
            "message": prompt,
            "include_trace": show_trace,
        },
        timeout=STREAM_TIMEOUT,
    ) as response:
        response.raise_for_status()
        yield from iter_jsonl(response)


def iter_jsonl(response: httpx.Response) -> Iterator[dict]:
    for line in response.iter_lines():
        if line:
            yield json.loads(line)


def render_stream_event(event: dict, show_trace: bool) -> str:
    event_type = event.get("type")
    if event_type == "status":
        return ""
    if event_type == "tool":
        return f"\n\n_Used `{event['name']}`._\n\n"
    if event_type == "token":
        return event["text"]
    if event_type == "error":
        return f"\n\nError: {event['text']}"
    if event_type == "final" and show_trace:
        return compact_run_details(event)
    return ""


def compact_run_details(event: dict) -> str:
    latency = int(event.get("latency_ms") or 0)
    cost = float(event.get("estimated_cost_usd") or 0)
    tools = event.get("tool_calls") or []
    tool_text = ", ".join(str(tool) for tool in tools) if tools else "none"
    return (
        "\n\n---\n"
        f"`run_id={event.get('run_id')}`  \n"
        f"`latency={latency}ms` `cost=${cost:.6f}` `tools={tool_text}`"
    )


def render_reports_tab() -> None:
    st.markdown("# Reports")
    st.caption("Run evals from the app, persist each eval run, and review report artifacts.")
    left, middle, right = st.columns([1, 1, 2])
    with left:
        include_benchmark = st.toggle("Include benchmark", value=True)
    with middle:
        use_llm_judge = st.toggle("Use LLM judge", value=False)
    with right:
        if st.button("Run evals", type="primary", width="stretch"):
            try:
                response = run_eval_from_api(include_benchmark, use_llm_judge)
                st.success(f"Eval run started: {response['eval_run_id']}")
            except httpx.HTTPError as err:
                st.error(f"Eval run failed to start: {err}")

    selected_eval_run = render_eval_dashboard()
    deployment_tab, cost_tab, evaluation_tab = st.tabs(
        ["OSS Deployment", "Cost + Latency", "Assistant Evaluation"]
    )
    with deployment_tab:
        st.markdown(load_oss_deployment_report())
    with cost_tab:
        st.markdown(load_cost_latency_report())
    with evaluation_tab:
        if selected_eval_run and selected_eval_run.get("report_markdown"):
            st.markdown(selected_eval_run["report_markdown"])
        else:
            st.markdown(load_assistant_evaluation_report())


def run_eval_from_api(include_benchmark: bool, use_llm_judge: bool) -> dict:
    response = httpx.post(
        f"{get_api_base_url()}/evals/take-home",
        json={
            "include_benchmark": include_benchmark,
            "use_llm_judge": use_llm_judge,
        },
        timeout=STREAM_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def render_eval_dashboard() -> dict | None:
    if st.button("Refresh eval runs"):
        st.rerun()
    try:
        runs = fetch_eval_runs(limit=20)
    except httpx.HTTPError as err:
        st.warning(f"Could not load eval history: {err}")
        return None
    if not runs:
        st.info("No eval runs recorded yet.")
        return None

    selected_eval_run_id = st.selectbox(
        "Eval run history",
        [run["eval_run_id"] for run in runs],
        format_func=lambda run_id: eval_run_label(run_id, runs),
    )
    try:
        selected = fetch_eval_run(selected_eval_run_id)
    except httpx.HTTPError as err:
        st.warning(f"Could not load selected eval run: {err}")
        return None

    if selected["status"] == "running":
        st.info("Eval is running in the background. Refresh to load results when it completes.")
    if selected["status"] == "failed":
        st.error(selected.get("error_message") or "Eval run failed.")

    metric_cols = st.columns(5)
    metric_cols[0].metric("Status", selected["status"])
    metric_cols[1].metric("Cases", selected["case_count"])
    metric_cols[2].metric(
        "OSS hallucination",
        format_percent(selected["oss_hallucination_rate"]),
    )
    metric_cols[4].metric(
        "Frontier hallucination",
        format_percent(selected["frontier_hallucination_rate"]),
    )

    chart_rows = [
        {
            "metric": "Hallucination",
            "OSS": selected["oss_hallucination_rate"],
            "Frontier": selected["frontier_hallucination_rate"],
        },
        {
            "metric": "Bias/Harm",
            "OSS": selected["oss_bias_harm_rate"],
            "Frontier": selected["frontier_bias_harm_rate"],
        },
        {
            "metric": "Safety",
            "OSS": selected["oss_safety_failure_rate"],
            "Frontier": selected["frontier_safety_failure_rate"],
        },
    ]
    st.markdown("#### Eval Failure Rates")
    st.bar_chart(chart_rows, x="metric", y=["OSS", "Frontier"])

    perf_rows = [
        {
            "assistant": "OSS",
            "avg_latency_ms": int(selected["oss_avg_latency_ms"]),
            "estimated_cost_usd": round(float(selected["oss_estimated_cost_usd"]), 6),
        },
        {
            "assistant": "Frontier",
            "avg_latency_ms": int(selected["frontier_avg_latency_ms"]),
            "estimated_cost_usd": round(float(selected["frontier_estimated_cost_usd"]), 6),
        },
    ]
    st.markdown("#### Cost + Latency")
    st.dataframe(perf_rows, hide_index=True, width="stretch")

    with st.expander("Eval run history", expanded=False):
        st.dataframe(runs, hide_index=True, width="stretch")
    return selected


def fetch_eval_runs(limit: int) -> list[dict]:
    response = httpx.get(
        f"{get_api_base_url()}/evals/runs",
        params={"limit": limit},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()["runs"]


def fetch_eval_run(eval_run_id: str) -> dict:
    response = httpx.get(
        f"{get_api_base_url()}/evals/runs/{eval_run_id}",
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()["run"]


def eval_run_label(eval_run_id: str, runs: list[dict]) -> str:
    run = next((item for item in runs if item["eval_run_id"] == eval_run_id), None)
    if run is None:
        return eval_run_id
    return f"{run['created_at']} | {run['status']} | {eval_run_id[:8]}"


def format_percent(value: float) -> str:
    return f"{float(value):.0%}"


def render_observability_tab() -> None:
    st.title("Observability")
    st.caption("Recent assistant runs, tool calls, safety checks, latency, token use, and cost.")
    left, right = st.columns([3, 1])
    with left:
        query = st.text_input("Search runs", placeholder="session, assistant, tool, trace text")
    with right:
        limit = st.number_input("Limit", min_value=10, max_value=500, value=50, step=10)

    try:
        runs = fetch_runs(query=query, limit=int(limit))
    except httpx.HTTPError as err:
        st.error(f"Could not load runs from API: {err}")
        return

    if not runs:
        st.info("No runs recorded yet. Send a chat message first.")
        return

    rows = [
        {
            "created_at": run["created_at"],
            "assistant": run["assistant_id"],
            "session": run["session_id"],
            "latency_ms": int(run["total_latency_ms"]),
            "tokens": (run.get("input_tokens") or 0) + (run.get("output_tokens") or 0),
            "cost": round(float(run["estimated_cost_usd"]), 6),
            "tools": run["tool_calls"] or "none",
            "run_id": run["run_id"],
        }
        for run in runs
    ]
    st.dataframe(rows, hide_index=True, width="stretch")

    selected_run_id = st.selectbox(
        "Inspect run",
        [run["run_id"] for run in runs],
        format_func=lambda run_id: run_label(run_id, runs),
    )
    try:
        detail = fetch_run(selected_run_id)
    except httpx.HTTPError as err:
        st.error(f"Could not load run detail: {err}")
        return
    render_run_detail(detail)


def fetch_runs(query: str, limit: int) -> list[dict]:
    params: dict[str, str | int] = {"limit": limit}
    if query:
        params["q"] = query
    response = httpx.get(
        f"{get_api_base_url()}/observability/runs",
        params=params,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()["runs"]


def fetch_run(run_id: str) -> dict:
    response = httpx.get(
        f"{get_api_base_url()}/observability/runs/{run_id}",
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()["run"]


def run_label(run_id: str, runs: list[dict]) -> str:
    run = next((item for item in runs if item["run_id"] == run_id), None)
    if run is None:
        return run_id
    return f"{run['created_at']} | {run['assistant_id']} | {run_id[:8]}"


def render_run_detail(run: dict) -> None:
    trace = run["trace"]
    metrics = st.columns(5)
    metrics[0].metric("Assistant", run["assistant_id"])
    metrics[1].metric("Latency", f"{int(run['total_latency_ms'])} ms")
    metrics[2].metric(
        "Tokens",
        str((run.get("input_tokens") or 0) + (run.get("output_tokens") or 0)),
    )
    metrics[3].metric("Cost", f"${float(run['estimated_cost_usd']):.6f}")
    metrics[4].metric("Tools", run["tool_calls"] or "none")

    st.markdown("#### Events")
    for event in trace["events"]:
        metadata = event.get("metadata") or {}
        with st.expander(
            f"{event['event_type']}.{event['name']} - {int(event['latency_ms'])} ms",
            expanded=False,
        ):
            st.json(metadata)

    with st.expander("Raw trace JSON", expanded=False):
        st.json(trace)


if __name__ == "__main__":
    main()
