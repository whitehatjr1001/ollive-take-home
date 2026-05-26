import re
from dataclasses import dataclass
from pathlib import Path
from textwrap import wrap

from ollie_assistants.reports.loader import (
    load_assistant_evaluation_report,
    load_cost_latency_report,
    load_oss_deployment_report,
)
from ollie_assistants.reports.paths import REPORTS_DIR

EVALUATION_PDF = REPORTS_DIR / "evaluation_report.pdf"


@dataclass(frozen=True)
class EvalPdfStats:
    case_count: int
    oss_latency_ms: int
    frontier_latency_ms: int
    oss_cost: float
    frontier_cost: float
    metric_rows: tuple[tuple[str, float, float], ...]
    benchmark_rows: tuple[tuple[str, str, str, str, str], ...]


def write_evaluation_pdf(output_path: Path = EVALUATION_PDF) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation_report = load_assistant_evaluation_report()
    cost_latency_report = load_cost_latency_report()
    deployment_report = load_oss_deployment_report()
    stats = parse_report_stats(evaluation_report, cost_latency_report)
    appendix_lines = markdown_to_lines(
        "\n\n".join(
            ["# Appendix", evaluation_report, cost_latency_report, deployment_report]
        )
    )
    output_path.write_bytes(render_evaluation_pdf(stats, appendix_lines))
    return output_path


def parse_report_stats(evaluation_report: str, cost_latency_report: str) -> EvalPdfStats:
    metric_rows: list[tuple[str, float, float]] = []
    for line in evaluation_report.splitlines():
        match = re.match(r"\| ([a-z_]+) \| ([0-9.]+)% \| ([0-9.]+)% \|", line)
        if match:
            metric_rows.append(
                (
                    metric_label(match.group(1)),
                    float(match.group(2)) / 100,
                    float(match.group(3)) / 100,
                )
            )
    assistant_rows = {}
    for line in evaluation_report.splitlines():
        match = re.match(
            r"\| (oss|frontier) \| ([0-9.]+) ms \| \$([0-9.]+) \|",
            line,
        )
        if match:
            assistant_rows[match.group(1)] = (int(float(match.group(2))), float(match.group(3)))
    case_count = 0
    for line in evaluation_report.splitlines():
        source_match = re.match(r"- .*: ([0-9]+) cases", line)
        if source_match:
            case_count += int(source_match.group(1))
    benchmark_rows: list[tuple[str, str, str, str, str]] = []
    for line in cost_latency_report.splitlines():
        parts = [part.strip() for part in line.strip("|").split("|")]
        if len(parts) == 10 and parts[0] not in {"Provider", "---"}:
            benchmark_rows.append((parts[0], parts[2], parts[5], parts[6], parts[8]))
    return EvalPdfStats(
        case_count=case_count,
        oss_latency_ms=assistant_rows.get("oss", (0, 0.0))[0],
        frontier_latency_ms=assistant_rows.get("frontier", (0, 0.0))[0],
        oss_cost=assistant_rows.get("oss", (0, 0.0))[1],
        frontier_cost=assistant_rows.get("frontier", (0, 0.0))[1],
        metric_rows=tuple(metric_rows),
        benchmark_rows=tuple(benchmark_rows),
    )


def metric_label(metric: str) -> str:
    return {
        "hallucination_rate": "Hallucination",
        "bias_harmful_output_rate": "Bias/Harm",
        "content_safety_failure_rate": "Safety Failure",
    }.get(metric, metric)


def markdown_to_lines(markdown: str) -> list[str]:
    lines: list[str] = []
    for raw_line in markdown.splitlines():
        line = raw_line.strip()
        if not line:
            lines.append("")
            continue
        line = line.replace("|", "  ")
        line = line.replace("---", "")
        line = line.removeprefix("# ").removeprefix("## ").removeprefix("### ")
        for wrapped in wrap(line, width=100) or [""]:
            lines.append(wrapped)
    return lines


def render_pdf(lines: list[str]) -> bytes:
    pages = paginate(lines)
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        page_tree_object(len(pages)),
    ]
    for index, page_lines in enumerate(pages):
        page_obj = 3 + index * 2
        content_obj = page_obj + 1
        objects.append(page_object(content_obj))
        objects.append(content_object(page_lines))
    return build_pdf(objects)


def render_evaluation_pdf(stats: EvalPdfStats, appendix_lines: list[str]) -> bytes:
    appendix_pages = paginate(appendix_lines, lines_per_page=58)
    page_streams = [
        infographic_page(stats),
        benchmark_page(stats),
        *[text_page(page_lines) for page_lines in appendix_pages],
    ]
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        page_tree_object(len(page_streams)),
    ]
    for index, stream in enumerate(page_streams):
        page_obj = 3 + index * 2
        content_obj = page_obj + 1
        objects.append(page_object(content_obj))
        objects.append(stream_object(stream))
    return build_pdf(objects)


def infographic_page(stats: EvalPdfStats) -> bytes:
    commands = [
        "0.08 0.10 0.14 rg 0 0 612 792 re f",
        text("Ollive Assistants", 48, 730, 24, "1 1 1"),
        text("OSS Modal vLLM vs Frontier OpenAI", 48, 705, 11, "0.75 0.82 0.92"),
        text("Evaluation Summary", 48, 660, 18, "1 1 1"),
        card(48, 585, 120, 55, "Cases", str(stats.case_count)),
        card(184, 585, 120, 55, "OSS avg latency", f"{stats.oss_latency_ms} ms"),
        card(320, 585, 120, 55, "Frontier avg latency", f"{stats.frontier_latency_ms} ms"),
        card(456, 585, 108, 55, "OSS eval cost", f"${stats.oss_cost:.4f}"),
        text("Failure Rates", 48, 540, 16, "1 1 1"),
    ]
    y = 490
    for label, oss_rate, frontier_rate in stats.metric_rows:
        commands.extend(bar_pair(label, oss_rate, frontier_rate, 48, y))
        y -= 75
    commands.extend(
        [
            text("Recommendation", 48, 255, 16, "1 1 1"),
            text(
                "Use the model with lower safety and hallucination failure rates.",
                48,
                230,
                10,
                "0.86 0.90 0.96",
            ),
            text(
                "Keep both assistants under continuous evals.",
                48,
                216,
                10,
                "0.86 0.90 0.96",
            ),
            text(
                "Cold starts, prompt changes, and provider behavior can shift results.",
                48,
                202,
                10,
                "0.86 0.90 0.96",
            ),
        ]
    )
    return "\n".join(commands).encode("latin-1", errors="replace")


def benchmark_page(stats: EvalPdfStats) -> bytes:
    commands = [
        "1 1 1 rg 0 0 612 792 re f",
        text("Cost + Latency Benchmark", 48, 735, 20, "0.08 0.10 0.14"),
        text(
            "Measured through the same assistant interface used by the demo app.",
            48,
            710,
            10,
            "0.30 0.34 0.40",
        ),
        table_header(48, 660),
    ]
    y = 632
    for row in stats.benchmark_rows[:12]:
        commands.extend(table_row(row, 48, y))
        y -= 30
    commands.extend(
        [
            text("Cost model", 48, 165, 16, "0.08 0.10 0.14"),
            text(
                "Frontier: input tokens * input price + output tokens * output price",
                48,
                140,
                10,
            ),
            text("OSS Modal: request latency seconds * Modal L4 hourly price / 3600", 48, 124, 10),
        ]
    )
    return "\n".join(commands).encode("latin-1", errors="replace")


def text_page(lines: list[str]) -> bytes:
    commands = ["1 1 1 rg 0 0 612 792 re f", "BT", "/F1 9 Tf", "50 760 Td", "12 TL"]
    for line in lines:
        commands.append(f"({escape_pdf_text(line)}) Tj")
        commands.append("T*")
    commands.append("ET")
    return "\n".join(commands).encode("latin-1", errors="replace")


def paginate(lines: list[str], lines_per_page: int = 58) -> list[list[str]]:
    return [lines[index : index + lines_per_page] for index in range(0, len(lines), lines_per_page)]


def page_tree_object(page_count: int) -> bytes:
    kids = " ".join(f"{3 + index * 2} 0 R" for index in range(page_count))
    return f"<< /Type /Pages /Kids [{kids}] /Count {page_count} >>".encode()


def page_object(content_obj: int) -> bytes:
    return (
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        f"/Resources << /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >> "
        f"/Contents {content_obj} 0 R >>"
    ).encode()


def content_object(lines: list[str]) -> bytes:
    commands = ["BT", "/F1 9 Tf", "50 760 Td", "12 TL"]
    for line in lines:
        commands.append(f"({escape_pdf_text(line)}) Tj")
        commands.append("T*")
    commands.append("ET")
    stream = "\n".join(commands).encode("latin-1", errors="replace")
    return b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream"


def stream_object(stream: bytes) -> bytes:
    return b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream"


def text(
    value: str,
    x: float,
    y: float,
    size: int,
    color: str = "0.08 0.10 0.14",
) -> str:
    return f"{color} rg BT /F1 {size} Tf {x} {y} Td ({escape_pdf_text(value)}) Tj ET"


def card(x: float, y: float, width: float, height: float, label: str, value: str) -> str:
    return "\n".join(
        [
            "0.14 0.18 0.25 rg",
            f"{x} {y} {width} {height} re f",
            text(label, x + 12, y + height - 18, 8, "0.70 0.78 0.90"),
            text(value, x + 12, y + 14, 16, "1 1 1"),
        ]
    )


def bar_pair(label: str, oss_rate: float, frontier_rate: float, x: float, y: float) -> list[str]:
    max_width = 250
    return [
        text(label, x, y + 24, 11, "0.92 0.95 1"),
        text("OSS", x, y, 9, "0.70 0.78 0.90"),
        "0.13 0.17 0.24 rg",
        f"{x + 65} {y - 2} {max_width} 12 re f",
        "0.14 0.55 0.92 rg",
        f"{x + 65} {y - 2} {max_width * oss_rate:.1f} 12 re f",
        text(f"{oss_rate:.0%}", x + 330, y, 9, "0.92 0.95 1"),
        text("Frontier", x, y - 24, 9, "0.70 0.78 0.90"),
        "0.13 0.17 0.24 rg",
        f"{x + 65} {y - 26} {max_width} 12 re f",
        "0.25 0.80 0.55 rg",
        f"{x + 65} {y - 26} {max_width * frontier_rate:.1f} 12 re f",
        text(f"{frontier_rate:.0%}", x + 330, y - 24, 9, "0.92 0.95 1"),
    ]


def table_header(x: float, y: float) -> str:
    return "\n".join(
        [
            "0.90 0.93 0.98 rg",
            f"{x} {y - 8} 510 24 re f",
            text("Provider", x + 8, y, 9),
            text("Scenario", x + 130, y, 9),
            text("p50", x + 225, y, 9),
            text("p95", x + 305, y, 9),
            text("Cost/request", x + 390, y, 9),
        ]
    )


def table_row(row: tuple[str, str, str, str, str], x: float, y: float) -> list[str]:
    provider, scenario, p50_latency, p95_latency, cost_request = row
    return [
        "0.98 0.99 1 rg",
        f"{x} {y - 8} 510 24 re f",
        text(provider[:22], x + 8, y, 8),
        text(scenario[:18], x + 130, y, 8),
        text(p50_latency, x + 225, y, 8),
        text(p95_latency, x + 305, y, 8),
        text(cost_request, x + 390, y, 8),
    ]


def escape_pdf_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def build_pdf(objects: list[bytes]) -> bytes:
    output = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{index} 0 obj\n".encode())
        output.extend(obj)
        output.extend(b"\nendobj\n")
    xref_offset = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode())
    output.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode()
    )
    return bytes(output)
