#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from manga_sft.reporting import create_reports, merge_prediction_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge baseline predictions and build reports")
    parser.add_argument("predictions", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/evaluation"))
    args = parser.parse_args()
    rows = merge_prediction_files(args.predictions)
    if not rows:
        raise ValueError("No predictions found")
    print(json.dumps(create_reports(rows, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

