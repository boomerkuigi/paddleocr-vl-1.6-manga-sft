#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from manga_sft.inference import MODEL_IDS, MODEL_REVISIONS
from manga_sft.metrics import aggregate_metrics
from manga_sft.reporting import read_prediction_jsonl


def model_specs() -> list[tuple[str, str, str]]:
    new_repo = os.environ["HF_MODEL_REPO"]
    new_revision = os.environ["HF_MODEL_REVISION"]
    return [
        ("manga_ocr", MODEL_IDS["manga_ocr"], MODEL_REVISIONS["manga_ocr"]),
        ("paddle_manga", MODEL_IDS["paddle_manga"], MODEL_REVISIONS["paddle_manga"]),
        ("paddle_1_6", MODEL_IDS["paddle_1_6"], MODEL_REVISIONS["paddle_1_6"]),
        ("new_model", new_repo, new_revision),
    ]


def tail(path: Path, lines: int = 40) -> str:
    if not path.is_file():
        return ""
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])


def run(manifest: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    prediction_dir = output / "predictions"
    log_dir = output / "logs"
    prediction_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)
    results = {}
    safe_predictions = output / "raw_model_predictions.jsonl"
    with safe_predictions.open("w", encoding="utf-8", newline="\n") as safe_handle:
        for alias, model_id, revision in model_specs():
            prediction_path = prediction_dir / f"{alias}.jsonl"
            log_path = log_dir / f"{alias}.log"
            command = [
                sys.executable,
                "scripts/evaluate_baselines.py",
                "--manifest",
                str(manifest),
                "--model",
                alias,
                "--model-id",
                model_id,
                "--revision",
                revision,
                "--output",
                str(prediction_path),
            ]
            started = time.perf_counter()
            with log_path.open("w", encoding="utf-8", newline="\n") as log_handle:
                completed = subprocess.run(
                    command,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
            runtime = time.perf_counter() - started
            print(tail(log_path), flush=True)
            if completed.returncode != 0:
                results[alias] = {
                    "status": "failed",
                    "model_id": model_id,
                    "model_revision": revision,
                    "runtime_seconds": runtime,
                    "error_tail": tail(log_path),
                }
                continue

            rows = read_prediction_jsonl(prediction_path)
            pairs = [(row["gold"], row["prediction"]) for row in rows]
            metrics = aggregate_metrics(pairs)
            results[alias] = {
                "status": "passed",
                "model_id": model_id,
                "model_revision": revision,
                "runtime_seconds": runtime,
                "metrics": metrics,
            }
            for row in rows:
                safe_handle.write(
                    json.dumps(
                        {
                            "sample_id": row["sample_id"],
                            "model_alias": row["model_alias"],
                            "model_id": row["model_id"],
                            "model_revision": row["model_revision"],
                            "prediction": row["prediction"],
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )

    summary = {
        "samples": 100,
        "all_models_passed": all(item["status"] == "passed" for item in results.values())
        and len(results) == 4,
        "models": results,
    }
    (output / "smoke_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("BENCHMARK_SMOKE_SUMMARY=" + json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.manifest, args.output)


if __name__ == "__main__":
    main()
