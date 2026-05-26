from ollie_assistants.reports.loader import load_report


def test_load_report_returns_missing_message(tmp_path) -> None:
    missing = tmp_path / "missing.md"

    assert "Report not generated yet" in load_report(missing)
