from ollie_assistants.deploy.types import CostLatencyRow


def modal_cost_latency_rows(warm_latency_ms: float | None = None) -> tuple[CostLatencyRow, ...]:
    return (
        CostLatencyRow(
            target="Modal vLLM",
            model="Qwen/Qwen2.5-0.5B-Instruct",
            hardware="L4",
            cold_start_note="model download is cached in Modal Volumes after first deploy",
            warm_latency_ms=warm_latency_ms,
            cost_note="GPU-seconds while replica is active; scaledown limits idle spend",
        ),
    )


def modal_cost_latency_table() -> list[dict[str, str | float | None]]:
    return [row.__dict__ for row in modal_cost_latency_rows()]
