#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from manga_sft.dataset import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    for row in read_jsonl(args.manifest, verify_images=True)[: args.limit]:
        with Image.open(row["image_path"]) as image:
            payload = {
                "sample_id": row["sample_id"],
                "size": image.size,
                "mode": image.mode,
                "gold": row["gold"],
                "book": row.get("book"),
                "split": row.get("split"),
            }
        print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()

