from ollie_assistants.reports.pdf import markdown_to_lines, parse_report_stats, render_pdf


def test_render_pdf_writes_valid_pdf_bytes() -> None:
    lines = markdown_to_lines("# Report\n\n| A | B |\n| --- | --- |\n| x | y |")
    pdf = render_pdf(lines)

    assert pdf.startswith(b"%PDF-1.4")
    assert b"%%EOF" in pdf


def test_parse_report_stats_extracts_metrics() -> None:
    stats = parse_report_stats(
        """
| hallucination_rate | 0% | 5% |
| bias_harmful_output_rate | 10% | 0% |
| oss | 100 ms | $0.0010 | 1 | 2 |
| frontier | 200 ms | $0.0020 | 3 | 4 |
- custom: 2 cases
""",
        """
| OSS Modal | model | warm | 1 | 5 | 100 ms | 200 ms | 1.0 | $0.01 | $1.00 |
""",
    )

    assert stats.case_count == 2
    assert stats.metric_rows[0] == ("Hallucination", 0.0, 0.05)
    assert stats.oss_latency_ms == 100
    assert stats.benchmark_rows[0][0] == "OSS Modal"
