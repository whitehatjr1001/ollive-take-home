from collections.abc import Sequence

from ollie_assistants.deploy.benchmark import BenchmarkRow
from ollie_assistants.evals.metrics import EvalMetricsAggregator
from ollie_assistants.evals.types import EvalCaseResult, JudgeReview


def render_assistant_evaluation_report(
    results: Sequence[EvalCaseResult],
    modal_gpu_usd_per_hour: float | None = None,
    judge_reviews: Sequence[JudgeReview] = (),
) -> str:
    aggregator = EvalMetricsAggregator()
    required_metrics = aggregator.summarize_required_metrics(results)
    oss_summary = aggregator.summarize_assistant("oss", [result.oss_output for result in results])
    frontier_summary = aggregator.summarize_assistant(
        "frontier",
        [result.frontier_output for result in results],
    )
    oss_eval_cost = oss_summary.total_estimated_cost_usd
    if oss_eval_cost == 0 and modal_gpu_usd_per_hour is not None:
        oss_eval_cost = _estimate_gpu_cost(
            total_latency_ms=sum(result.oss_output.latency_ms for result in results),
            gpu_usd_per_hour=modal_gpu_usd_per_hour,
        )
    lines = [
        "# Assistant Evaluation Report",
        "",
        "## Evaluation Sources",
        "",
        *_source_rows(results),
        "",
        "## Required Evaluation Metrics",
        "",
        "| Required metric | OSS failure rate | Frontier failure rate |",
        "| --- | ---: | ---: |",
    ]
    for summary in required_metrics:
        lines.append(
            "| "
            f"{summary.metric.value} | "
            f"{summary.oss_failure_rate:.0%} | "
            f"{summary.frontier_failure_rate:.0%} |"
        )
    lines.extend(
        [
            "",
            "## Cost and Latency During Evals",
            "",
            "| Assistant | Avg latency | Est. cost | Input tokens | Output tokens |",
            "| --- | ---: | ---: | ---: | ---: |",
            _assistant_summary_row(oss_summary, estimated_cost_usd=oss_eval_cost),
            _assistant_summary_row(frontier_summary),
            "",
            "Cost note: OSS eval cost is estimated from measured request latency and "
            "Modal L4 GPU-second pricing. Frontier cost is estimated from configured "
            "OpenAI input/output token pricing.",
            "",
            "Judge note: these are lightweight heuristic judges for a take-home demo, "
            "not a certified safety benchmark.",
            "",
            *_llm_judge_rows(judge_reviews),
            "",
            "## Recommendation",
            "",
            "Prefer the assistant with lower safety and hallucination failure rates after "
            "reviewing qualitative failures.",
        ]
    )
    return "\n".join(lines)


def _llm_judge_rows(reviews: Sequence[JudgeReview]) -> list[str]:
    if not reviews:
        return [
            "## LLM-as-Judge Verification",
            "",
            "Not run. Use `uv run ollie-eval --use-llm-judge` to add verifier results.",
        ]
    lines = [
        "## LLM-as-Judge Verification",
        "",
        "| Case | Assistant | Metric | Passed | Reason |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for review in reviews:
        lines.append(
            f"| {review.case_id} | {review.assistant_id} | {review.metric} | "
            f"{'yes' if review.passed else 'no'} | {review.reason} |"
        )
    return lines


def render_cost_latency_report(rows: Sequence[BenchmarkRow]) -> str:
    lines = [
        "# Cost and Latency Report",
        "",
        "Latency and throughput are measured from the benchmark run and may vary by "
        "cold starts, provider load, network, prompt length, and concurrency.",
        "",
        "| Provider | Model | Scenario | Concurrency | Requests | p50 latency | "
        "p95 latency | Output tok/s | Cost/request | Cost/1M tokens |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    lines.extend(_benchmark_row(row) for row in rows)
    return "\n".join(lines)


def _source_rows(results: Sequence[EvalCaseResult]) -> list[str]:
    counts: dict[str, int] = {}
    for result in results:
        source = result.case.source.value
        counts[source] = counts.get(source, 0) + 1
    return [f"- {source}: {count} cases" for source, count in sorted(counts.items())]


def _assistant_summary_row(summary, estimated_cost_usd: float | None = None) -> str:
    cost = summary.total_estimated_cost_usd if estimated_cost_usd is None else estimated_cost_usd
    return (
        f"| {summary.assistant_id} | "
        f"{summary.average_latency_ms:.0f} ms | "
        f"${cost:.4f} | "
        f"{summary.total_input_tokens} | "
        f"{summary.total_output_tokens} |"
    )


def _benchmark_row(row: BenchmarkRow) -> str:
    return (
        f"| {row.provider} | {row.model} | {row.scenario} | "
        f"{row.concurrency} | {row.requests} | "
        f"{row.p50_latency_ms:.0f} ms | {row.p95_latency_ms:.0f} ms | "
        f"{row.output_tokens_per_second:.1f} | "
        f"${row.cost_per_request_usd:.6f} | "
        f"${row.cost_per_1m_tokens_usd:.2f} |"
    )


def _estimate_gpu_cost(total_latency_ms: float, gpu_usd_per_hour: float) -> float:
    return total_latency_ms / 1000 * gpu_usd_per_hour / 3600
