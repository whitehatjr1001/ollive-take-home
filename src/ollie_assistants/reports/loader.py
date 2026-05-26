from pathlib import Path

from ollie_assistants.reports.paths import (
    ASSISTANT_EVALUATION_REPORT,
    COST_LATENCY_REPORT,
    OSS_DEPLOYMENT_REPORT,
)


def load_report(path: Path) -> str:
    if not path.exists():
        return f"Report not generated yet: `{path}`"
    return path.read_text(encoding="utf-8")


def load_oss_deployment_report() -> str:
    return load_report(OSS_DEPLOYMENT_REPORT)


def load_cost_latency_report() -> str:
    return load_report(COST_LATENCY_REPORT)


def load_assistant_evaluation_report() -> str:
    return load_report(ASSISTANT_EVALUATION_REPORT)
