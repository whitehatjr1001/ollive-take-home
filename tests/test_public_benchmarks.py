from ollie_assistants.evals.public_benchmarks import load_simpleqa_sample_cases
from ollie_assistants.evals.types import EvalCaseSource, EvalCategory


def test_load_simpleqa_sample_cases_returns_factual_cases_with_source() -> None:
    cases = load_simpleqa_sample_cases(limit=2)

    assert len(cases) == 2
    assert all(case.category == EvalCategory.FACTUAL for case in cases)
    assert all(case.source == EvalCaseSource.SIMPLEQA_SAMPLE for case in cases)
    assert all(case.expected for case in cases)
