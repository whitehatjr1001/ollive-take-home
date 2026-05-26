from dataclasses import dataclass
from enum import StrEnum


class EvalCategory(StrEnum):
    FACTUAL = "factual"
    JAILBREAK = "jailbreak"
    BIAS = "bias"
    SAFETY = "safety"
    ASSISTANT_BEHAVIOR = "assistant_behavior"


class EvalMetric(StrEnum):
    HALLUCINATION_RATE = "hallucination_rate"
    BIAS_HARMFUL_OUTPUT_RATE = "bias_harmful_output_rate"
    CONTENT_SAFETY_FAILURE_RATE = "content_safety_failure_rate"


class EvalCaseSource(StrEnum):
    CUSTOM = "custom"
    SIMPLEQA_SAMPLE = "simpleqa_sample"


@dataclass(frozen=True)
class EvalCase:
    id: str
    category: EvalCategory
    prompt: str
    expected: str | None = None
    unsafe_intent: bool = False
    source: EvalCaseSource = EvalCaseSource.CUSTOM


@dataclass(frozen=True)
class AssistantOutput:
    assistant_id: str
    text: str
    latency_ms: float
    estimated_cost_usd: float
    pricing_method: str
    input_tokens: int | None
    output_tokens: int | None


@dataclass(frozen=True)
class JudgeScore:
    metric: str
    score: float
    passed: bool
    reason: str


@dataclass(frozen=True)
class EvalCaseResult:
    case: EvalCase
    oss_output: AssistantOutput
    frontier_output: AssistantOutput
    oss_scores: tuple[JudgeScore, ...]
    frontier_scores: tuple[JudgeScore, ...]


@dataclass(frozen=True)
class MetricSummary:
    metric: EvalMetric
    oss_failure_rate: float
    frontier_failure_rate: float


@dataclass(frozen=True)
class AssistantEvalSummary:
    assistant_id: str
    average_latency_ms: float
    total_estimated_cost_usd: float
    total_input_tokens: int
    total_output_tokens: int


@dataclass(frozen=True)
class JudgeReview:
    case_id: str
    assistant_id: str
    metric: str
    passed: bool
    reason: str
