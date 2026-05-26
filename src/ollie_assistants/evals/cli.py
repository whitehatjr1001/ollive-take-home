import argparse
import asyncio

from ollie_assistants.assistant.factory import AssistantFactory
from ollie_assistants.evals.facade import AssistantComparisonService
from ollie_assistants.settings import get_settings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-benchmark", action="store_true")
    parser.add_argument("--use-llm-judge", action="store_true")
    args = parser.parse_args()
    report = asyncio.run(
        AssistantComparisonService(AssistantFactory(get_settings())).run_comparison(
            include_benchmark=args.include_benchmark,
            use_llm_judge=args.use_llm_judge,
        )
    )
    print(report)
