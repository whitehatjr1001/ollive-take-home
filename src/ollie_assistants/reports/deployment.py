from ollie_assistants.reports.paths import OSS_DEPLOYMENT_REPORT
from ollie_assistants.settings import Settings, get_settings


def build_oss_deployment_report(settings: Settings | None = None) -> str:
    resolved = settings or get_settings()
    lines = [
        "# OSS Deployment Report",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| OSS model | `{resolved.oss_model}` |",
        f"| Served model name | `{resolved.oss_served_model_name}` |",
        "| Runtime | Modal + vLLM |",
        f"| GPU | `{resolved.oss_modal_gpu}` |",
        f"| OpenAI-compatible base URL | `{resolved.oss_base_url or 'not configured'}` |",
        "| Auth | Bearer token between app host and Modal endpoint |",
        f"| Scaledown window | {resolved.oss_modal_scaledown_seconds}s |",
        f"| Max concurrent inputs | {resolved.oss_modal_max_inputs} |",
        f"| Max model length | {resolved.oss_modal_max_model_len} tokens |",
        "",
        "## Deploy Command",
        "",
        "```bash",
        'OSS_BEARER_TOKEN=$(openssl rand -hex 32)',
        'modal secret create ollie-oss-secrets OSS_BEARER_TOKEN="$OSS_BEARER_TOKEN"',
        "uv run modal deploy src/ollie_assistants/deploy/modal_oss.py",
        "```",
    ]
    return "\n".join(lines)


def write_oss_deployment_report(settings: Settings | None = None) -> str:
    report = build_oss_deployment_report(settings)
    OSS_DEPLOYMENT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OSS_DEPLOYMENT_REPORT.write_text(report, encoding="utf-8")
    return report
