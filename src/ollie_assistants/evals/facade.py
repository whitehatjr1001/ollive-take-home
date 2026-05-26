from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from ollie_assistants.analytics.pricing import ModalGpuPricing
from ollie_assistants.assistant.factory import AssistantFactory
from ollie_assistants.deploy.benchmark import (
    AssistantBenchmarkRunner,
    default_benchmark_scenarios,
)
from ollie_assistants.evals.cases import take_home_cases
from ollie_assistants.evals.judges import JudgeFactory
from ollie_assistants.evals.llm_judge import LLMJudgeVerifier
from ollie_assistants.evals.metrics import EvalMetricsAggregator
from ollie_assistants.evals.runner import EvalRunner
from ollie_assistants.reports.deployment import write_oss_deployment_report
from ollie_assistants.reports.paths import ASSISTANT_EVALUATION_REPORT, COST_LATENCY_REPORT
from ollie_assistants.reports.rendering import (
    render_assistant_evaluation_report,
    render_cost_latency_report,
)


class AssistantComparisonService:
    def __init__(self, assistant_factory: AssistantFactory) -> None:
        self.assistant_factory = assistant_factory
        self.runner = EvalRunner(JudgeFactory())
        self.last_eval_run_id: str | None = None

    async def run_comparison(
        self,
        output_path: Path = ASSISTANT_EVALUATION_REPORT,
        include_benchmark: bool = False,
        use_llm_judge: bool = False,
        eval_run_id: str | None = None,
    ) -> str:
        oss = self.assistant_factory.create_oss()
        frontier = self.assistant_factory.create_frontier()
        results = [await self.runner.run_case(case, oss, frontier) for case in take_home_cases()]
        judge_reviews = ()
        if use_llm_judge:
            judge_provider = self.assistant_factory.llm_factory.create(
                self.assistant_factory.settings.frontier_provider,
                self.assistant_factory.settings,
            )
            judge_reviews = await LLMJudgeVerifier(
                judge_provider,
                self.assistant_factory.settings.frontier_model,
            ).review_results(results)
        benchmark_rows = ()
        if include_benchmark:
            benchmark = AssistantBenchmarkRunner(
                ModalGpuPricing(gpu_usd_per_hour=self.assistant_factory.settings.modal_l4_usd_per_hour)
            )
            scenarios = default_benchmark_scenarios()
            benchmark_rows = (
                *await benchmark.run(
                    "OSS Modal",
                    self.assistant_factory.settings.oss_model,
                    oss,
                    scenarios,
                    cost_mode="modal_gpu_seconds",
                ),
                *await benchmark.run(
                    "Frontier OpenAI",
                    self.assistant_factory.settings.frontier_model,
                    frontier,
                    scenarios,
                    cost_mode="provider_reported",
                ),
            )
        write_oss_deployment_report(self.assistant_factory.settings)
        report = render_assistant_evaluation_report(
            results,
            modal_gpu_usd_per_hour=self.assistant_factory.settings.modal_l4_usd_per_hour,
            judge_reviews=judge_reviews,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
        if include_benchmark:
            cost_latency_report = render_cost_latency_report(benchmark_rows)
            COST_LATENCY_REPORT.parent.mkdir(parents=True, exist_ok=True)
            COST_LATENCY_REPORT.write_text(cost_latency_report, encoding="utf-8")
        resolved_eval_run_id = eval_run_id or str(uuid4())
        self.last_eval_run_id = resolved_eval_run_id
        self.assistant_factory.trace_store.record_eval_run(
            eval_run_id=resolved_eval_run_id,
            created_at=datetime.now(UTC).isoformat(),
            include_benchmark=include_benchmark,
            use_llm_judge=use_llm_judge,
            report_markdown=report,
            summary=build_eval_summary(
                results,
                benchmark_rows=benchmark_rows,
                modal_gpu_usd_per_hour=self.assistant_factory.settings.modal_l4_usd_per_hour,
            ),
        )
        self.assistant_factory.trace_store.update_eval_status(resolved_eval_run_id, "completed")
        return report


def build_eval_summary(
    results,
    benchmark_rows=(),
    modal_gpu_usd_per_hour: float | None = None,
) -> dict:
    aggregator = EvalMetricsAggregator()
    required_metrics = aggregator.summarize_required_metrics(results)
    oss_summary = aggregator.summarize_assistant("oss", [result.oss_output for result in results])
    frontier_summary = aggregator.summarize_assistant(
        "frontier",
        [result.frontier_output for result in results],
    )
    oss_cost = oss_summary.total_estimated_cost_usd
    if oss_cost == 0 and modal_gpu_usd_per_hour is not None:
        total_latency_ms = sum(result.oss_output.latency_ms for result in results)
        oss_cost = total_latency_ms / 1000 * modal_gpu_usd_per_hour / 3600
    return {
        "case_count": len(results),
        "required_metrics": {
            summary.metric.value: {
                "oss_failure_rate": summary.oss_failure_rate,
                "frontier_failure_rate": summary.frontier_failure_rate,
            }
            for summary in required_metrics
        },
        "assistants": {
            "oss": {
                "average_latency_ms": oss_summary.average_latency_ms,
                "estimated_cost_usd": oss_cost,
                "input_tokens": oss_summary.total_input_tokens,
                "output_tokens": oss_summary.total_output_tokens,
            },
            "frontier": {
                "average_latency_ms": frontier_summary.average_latency_ms,
                "estimated_cost_usd": frontier_summary.total_estimated_cost_usd,
                "input_tokens": frontier_summary.total_input_tokens,
                "output_tokens": frontier_summary.total_output_tokens,
            },
        },
        "benchmark_rows": [asdict(row) for row in benchmark_rows],
    }
