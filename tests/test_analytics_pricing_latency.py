from ollie_assistants.analytics.latency import p50, p95
from ollie_assistants.analytics.pricing import ModalGpuPricing, OpenAITokenPricing


def test_openai_token_pricing_estimates_input_and_output_cost() -> None:
    pricing = OpenAITokenPricing(input_usd_per_1m_tokens=1.0, output_usd_per_1m_tokens=2.0)
    assert pricing.estimate(input_tokens=1_000_000, output_tokens=500_000) == 2.0


def test_modal_gpu_pricing_estimates_gpu_seconds() -> None:
    pricing = ModalGpuPricing(gpu_usd_per_hour=3.60)
    assert pricing.estimate(duration_s=10) == 0.01


def test_latency_percentiles() -> None:
    values = [100, 200, 300, 400]
    assert p50(values) == 250
    assert p95(values) == 385
