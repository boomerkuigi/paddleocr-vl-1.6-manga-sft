#!/usr/bin/env python3
"""Create a Hub-safe benchmark bundle without Manga109-s gold text or images."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path


def sanitize(prediction_dir: Path, evaluation_dir: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    prediction_count = 0
    destination = output / "raw_model_predictions.jsonl"
    with destination.open("w", encoding="utf-8", newline="\n") as writer:
        for path in sorted(prediction_dir.glob("*.jsonl")):
            with path.open("r", encoding="utf-8") as reader:
                for line in reader:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    safe = {
                        key: item[key]
                        for key in (
                            "sample_id",
                            "model_alias",
                            "model_id",
                            "model_revision",
                            "prediction",
                        )
                        if key in item
                    }
                    writer.write(json.dumps(safe, ensure_ascii=False, sort_keys=True) + "\n")
                    prediction_count += 1

    shutil.copy2(evaluation_dir / "metrics.json", output / "metrics.json")
    category_counts: Counter[str] = Counter()
    with (evaluation_dir / "disagreements.csv").open(encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            category_counts.update(filter(None, row["categories"].split(";")))
    summary = {
        "prediction_rows": prediction_count,
        "disagreement_category_counts": dict(sorted(category_counts.items())),
        "privacy": (
            "Manga109-s gold text, image paths, crops, and annotations are excluded. "
            "Raw model outputs are retained as experimental results."
        ),
    }
    (output / "disagreement_counts.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            sanitize(args.prediction_dir, args.evaluation_dir, args.output),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
