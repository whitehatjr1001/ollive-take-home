import argparse
from pathlib import Path

from ollie_assistants.reports.pdf import EVALUATION_PDF, write_evaluation_pdf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=EVALUATION_PDF)
    args = parser.parse_args()
    path = write_evaluation_pdf(args.output)
    print(path)
