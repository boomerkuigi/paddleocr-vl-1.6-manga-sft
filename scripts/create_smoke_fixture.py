#!/usr/bin/env python3
"""Create synthetic *format-only* fixtures, never quality-training data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

from manga_sft.dataset import image_sha256, stable_sample_id, validate_no_leakage, write_jsonl


GOLDS = (
    "これは煙ではない",
    "行くぞ！",
    "えっ…",
    "まって〜",
    "ドキドキ♡",
    "アナタ専用",
    "小さい声",
    "ありがとう。",
)


def build(output: Path) -> dict:
    images = output / "images"
    manifests = output / "manifests"
    split_plan = {"train": range(0, 5), "validation": range(5, 7), "test": range(7, 8)}
    all_rows: dict[str, list[dict]] = {}
    for split, indexes in split_plan.items():
        rows = []
        for index in indexes:
            image_path = images / f"fixture-{index}.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image = Image.new("RGB", (160 + index * 3, 64), "white")
            draw = ImageDraw.Draw(image)
            draw.rectangle((8, 8, image.width - 8, image.height - 8), outline="black", width=2)
            draw.line((20, 32, image.width - 20, 32), fill="gray", width=1)
            image.save(image_path)
            book = f"fixture_book_{split}"
            sample_id = stable_sample_id(book, str(index), [0, 0, image.width, image.height], GOLDS[index])
            rows.append(
                {
                    "sample_id": sample_id,
                    "image_path": str(Path("..") / "images" / image_path.name),
                    "gold": GOLDS[index],
                    "book": book,
                    "page": str(index),
                    "bbox": [0, 0, image.width, image.height],
                    "split": split,
                    "source": "format-only synthetic smoke fixture",
                    "image_sha256": image_sha256(image_path),
                }
            )
        all_rows[split] = rows
    validate_no_leakage(all_rows)
    sizes = {split: write_jsonl(manifests / f"{split}.jsonl", rows) for split, rows in all_rows.items()}
    summary = {
        "warning": "Formatting smoke fixture only; never use to assess OCR quality.",
        "sizes": sizes,
        "seed": 42,
    }
    (manifests / "split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/prepared/smoke"))
    args = parser.parse_args()
    print(json.dumps(build(args.output.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

