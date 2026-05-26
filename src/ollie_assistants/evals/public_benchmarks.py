from ollie_assistants.evals.types import EvalCase, EvalCaseSource, EvalCategory

SIMPLEQA_REFERENCE_URL = "https://openai.com/index/introducing-simpleqa/"


def load_simpleqa_sample_cases(limit: int = 5) -> tuple[EvalCase, ...]:
    cases = (
        EvalCase(
            id="simpleqa_sample_001",
            category=EvalCategory.FACTUAL,
            prompt="Who developed the theory of general relativity?",
            expected="Albert Einstein",
            source=EvalCaseSource.SIMPLEQA_SAMPLE,
        ),
        EvalCase(
            id="simpleqa_sample_002",
            category=EvalCategory.FACTUAL,
            prompt="What is the largest planet in the Solar System?",
            expected="Jupiter",
            source=EvalCaseSource.SIMPLEQA_SAMPLE,
        ),
        EvalCase(
            id="simpleqa_sample_003",
            category=EvalCategory.FACTUAL,
            prompt="Which element has the chemical symbol Au?",
            expected="Gold",
            source=EvalCaseSource.SIMPLEQA_SAMPLE,
        ),
        EvalCase(
            id="simpleqa_sample_004",
            category=EvalCategory.FACTUAL,
            prompt="Who painted the Mona Lisa?",
            expected="Leonardo da Vinci",
            source=EvalCaseSource.SIMPLEQA_SAMPLE,
        ),
        EvalCase(
            id="simpleqa_sample_005",
            category=EvalCategory.FACTUAL,
            prompt="What is the smallest prime number?",
            expected="2",
            source=EvalCaseSource.SIMPLEQA_SAMPLE,
        ),
    )
    return cases[:limit]
