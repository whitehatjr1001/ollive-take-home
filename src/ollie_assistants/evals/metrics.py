from collections.abc import Sequence

from ollie_assistants.evals.types import (
    AssistantEvalSummary,
    AssistantOutput,
    EvalCaseResult,
    EvalMetric,
    JudgeScore,
    MetricSummary,
)

JUDGE_TO_REQUIRED_METRIC = {
    "hallucination": EvalMetric.HALLUCINATION_RATE,
    "bias_harm": EvalMetric.BIAS_HARMFUL_OUTPUT_RATE,
    "content_safety": EvalMetric.CONTENT_SAFETY_FAILURE_RATE,
}


class EvalMetricsAggregator:
    def summarize_required_metrics(
        self,
        results: Sequence[EvalCaseResult],
    ) -> tuple[MetricSummary, ...]:
        summaries: list[MetricSummary] = []
        for judge_metric, required_metric in JUDGE_TO_REQUIRED_METRIC.items():
            summaries.append(
                MetricSummary(
                    metric=required_metric,
                    oss_failure_rate=self._failure_rate(
                        score
                        for result in results
                        for score in result.oss_scores
                        if score.metric == judge_metric
                    ),
                    frontier_failure_rate=self._failure_rate(
                        score
                        for result in results
                        for score in result.frontier_scores
                        if score.metric == judge_metric
                    ),
                )
            )
        return tuple(summaries)

    def summarize_assistant(
        self,
        assistant_id: str,
        outputs: Sequence[AssistantOutput],
    ) -> AssistantEvalSummary:
        return AssistantEvalSummary(
            assistant_id=assistant_id,
            average_latency_ms=self._average(output.latency_ms for output in outputs),
            total_estimated_cost_usd=sum(output.estimated_cost_usd for output in outputs),
            total_input_tokens=sum(output.input_tokens or 0 for output in outputs),
            total_output_tokens=sum(output.output_tokens or 0 for output in outputs),
        )

    def _failure_rate(self, scores: Sequence[JudgeScore]) -> float:
        scores = tuple(scores)
        if not scores:
            return 0.0
        return sum(1 for score in scores if not score.passed) / len(scores)

    def _average(self, values: Sequence[float]) -> float:
        values = tuple(values)
        if not values:
            return 0.0
        return sum(values) / len(values)
