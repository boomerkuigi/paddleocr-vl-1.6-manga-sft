#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from manga_sft.dataset import read_jsonl
from manga_sft.inference import MODEL_IDS, MangaOCRAdapter, PaddleOCRVLAdapter
from manga_sft.metrics import aggregate_metrics
from manga_sft.reporting import write_prediction_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one OCR model at a time")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--model",
        choices=("manga_ocr", "paddle_manga", "paddle_1_6", "new_model"),
        required=True,
    )
    parser.add_argument("--model-id")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    rows = read_jsonl(args.manifest, verify_images=True)
    if args.limit is not None:
        rows = rows[: args.limit]
    model_id = args.model_id or MODEL_IDS.get(args.model) or os.environ.get("HF_MODEL_REPO")
    if not model_id:
        raise ValueError("--model-id or HF_MODEL_REPO is required for new_model")
    adapter = (
        MangaOCRAdapter(model_id, args.device)
        if args.model == "manga_ocr"
        else PaddleOCRVLAdapter(model_id=model_id, device=args.device)
    )
    predictions = []
    for index, row in enumerate(rows, start=1):
        prediction = adapter.predict(row["image_path"])
        predictions.append(
            {
                "sample_id": row["sample_id"],
                "image_path": row["image_path"],
                "gold": row["gold"],
                "model_alias": args.model,
                "model_id": model_id,
                "prediction": prediction,
            }
        )
        print(json.dumps({"completed": index, "total": len(rows), "sample_id": row["sample_id"]}))
    write_prediction_jsonl(args.output, predictions)
    pairs = [(item["gold"], item["prediction"]) for item in predictions]
    print(json.dumps(aggregate_metrics(pairs), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

