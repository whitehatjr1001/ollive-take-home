from ollie_assistants.assistant.service import AssistantFacade
from ollie_assistants.evals.judges import JudgeFactory
from ollie_assistants.evals.types import AssistantOutput, EvalCase, EvalCaseResult


class EvalRunner:
    def __init__(self, judge_factory: JudgeFactory) -> None:
        self.judge_factory = judge_factory

    async def run_case(
        self,
        case: EvalCase,
        oss: AssistantFacade,
        frontier: AssistantFacade,
    ) -> EvalCaseResult:
        oss_response = await oss.chat(
            f"eval-{case.id}-oss",
            case.prompt,
            record_trace=False,
        )
        frontier_response = await frontier.chat(
            f"eval-{case.id}-frontier",
            case.prompt,
            record_trace=False,
        )
        judges = self.judge_factory.create_for(case.category)
        oss_output = AssistantOutput(
            "oss",
            oss_response.text,
            oss_response.latency_ms,
            oss_response.estimated_cost_usd,
            oss_response.pricing_method,
            oss_response.input_tokens,
            oss_response.output_tokens,
        )
        frontier_output = AssistantOutput(
            "frontier",
            frontier_response.text,
            frontier_response.latency_ms,
            frontier_response.estimated_cost_usd,
            frontier_response.pricing_method,
            frontier_response.input_tokens,
            frontier_response.output_tokens,
        )
        return EvalCaseResult(
            case=case,
            oss_output=oss_output,
            frontier_output=frontier_output,
            oss_scores=tuple(judge.score(case, oss_output.text) for judge in judges),
            frontier_scores=tuple(judge.score(case, frontier_output.text) for judge in judges),
        )
