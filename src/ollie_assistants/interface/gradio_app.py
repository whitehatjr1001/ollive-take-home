import gradio as gr

from ollie_assistants.assistant.factory import AssistantFactory
from ollie_assistants.evals.facade import AssistantComparisonService
from ollie_assistants.reports.loader import (
    load_assistant_evaluation_report,
    load_cost_latency_report,
    load_oss_deployment_report,
)
from ollie_assistants.settings import get_settings

settings = get_settings()
factory = AssistantFactory(settings)
assistants = {"oss": factory.create_oss(), "frontier": factory.create_frontier()}
comparison_service = AssistantComparisonService(factory)


async def response_steps(
    message: str,
    assistant_kind: str,
    show_trace: bool,
) -> list[str]:
    steps = [f"[{assistant_kind}] checking safety and tools..."]
    response = await assistants[assistant_kind].chat("gradio", message)
    if response.tool_calls:
        steps.append(f"[{assistant_kind}] tool calls: {', '.join(response.tool_calls)}")
    final = (
        f"{response.text}\n\n--- trace ---\n{response.trace}"
        if show_trace and response.trace
        else response.text
    )
    steps.append(final)
    return steps


async def send_message(
    message: str,
    history: list[tuple[str, str]],
    assistant_selection: str,
    show_trace: bool,
):
    if not message.strip():
        yield history, ""
        return

    history = [*history, (message, "")]
    yield history, ""

    if assistant_selection != "both":
        assistant_text = ""
        for step in await response_steps(message, assistant_selection, show_trace):
            assistant_text = step if not assistant_text else f"{assistant_text}\n\n{step}"
            history[-1] = (message, assistant_text)
            yield history, ""
        return

    assistant_text = ""
    for assistant_kind in ("oss", "frontier"):
        section = ""
        for step in await response_steps(message, assistant_kind, show_trace):
            section = step if not section else f"{section}\n\n{step}"
            combined = f"{assistant_text}\n\n## {assistant_kind}\n{section}".strip()
            history[-1] = (message, combined)
            yield history, ""
        assistant_text = f"{assistant_text}\n\n## {assistant_kind}\n{section}".strip()


async def respond_single(
    message: str,
    assistant_kind: str,
    show_trace: bool,
):
    yield f"[{assistant_kind}] checking safety and tools..."
    response = await assistants[assistant_kind].chat("gradio", message)
    if response.tool_calls:
        yield f"[{assistant_kind}] tool calls: {', '.join(response.tool_calls)}"
    if show_trace and response.trace:
        yield f"{response.text}\n\n--- trace ---\n{response.trace}"
        return
    yield response.text


async def respond(
    message: str,
    history: list[dict[str, str]],
    assistant_selection: str,
    show_trace: bool,
):
    if assistant_selection != "both":
        async for chunk in respond_single(message, assistant_selection, show_trace):
            yield chunk
        return

    outputs: list[str] = []
    for assistant_kind in ("oss", "frontier"):
        async for chunk in respond_single(message, assistant_kind, show_trace):
            outputs.append(f"## {assistant_kind}\n{chunk}")
            yield "\n\n".join(outputs)


async def run_reports(include_benchmark: bool, use_llm_judge: bool) -> str:
    return await comparison_service.run_comparison(
        include_benchmark=include_benchmark,
        use_llm_judge=use_llm_judge,
    )


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Ollie Assistants") as demo:
        with gr.Tab("Chat"):
            gr.Markdown("# Ollie Assistants")
            assistant_kind = gr.Radio(["oss", "frontier", "both"], value="oss", label="Assistant")
            show_trace = gr.Checkbox(value=False, label="Show trace")
            chatbot = gr.Chatbot()
            message = gr.Textbox(placeholder="Ask either assistant...", label="Message")
            message.submit(
                send_message,
                inputs=[message, chatbot, assistant_kind, show_trace],
                outputs=[chatbot, message],
            )
        with gr.Tab("Reports"):
            gr.Markdown("# Reports")
            include_benchmark = gr.Checkbox(value=True, label="Include benchmark")
            use_llm_judge = gr.Checkbox(value=False, label="Use LLM-as-judge")
            run_status = gr.Markdown("Reports load from disk until you run them.")
            gr.Button("Run Reports").click(
                run_reports,
                inputs=[include_benchmark, use_llm_judge],
                outputs=run_status,
            )
            with gr.Tab("OSS Deployment"):
                oss_report = gr.Markdown(load_oss_deployment_report())
                gr.Button("Reload OSS Deployment Report").click(
                    load_oss_deployment_report,
                    outputs=oss_report,
                )
            with gr.Tab("Cost + Latency"):
                cost_report = gr.Markdown(load_cost_latency_report())
                gr.Button("Reload Cost + Latency Report").click(
                    load_cost_latency_report,
                    outputs=cost_report,
                )
            with gr.Tab("Assistant Evaluation"):
                eval_report = gr.Markdown(load_assistant_evaluation_report())
                gr.Button("Reload Assistant Evaluation Report").click(
                    load_assistant_evaluation_report,
                    outputs=eval_report,
                )
    return demo


def main() -> None:
    build_demo().launch()


demo = build_demo()
