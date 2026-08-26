from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Iterable

from .metrics import aggregate_metrics, score_pair


MODEL_COLUMNS = ("manga_ocr", "paddle_manga", "paddle_1_6", "new_model")


def write_prediction_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_prediction_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def merge_prediction_files(paths: Iterable[Path]) -> list[dict]:
    merged: dict[str, dict] = {}
    for path in paths:
        for item in read_prediction_jsonl(path):
            sample_id = str(item["sample_id"])
            row = merged.setdefault(
                sample_id,
                {
                    "sample_id": sample_id,
                    "image_path": item.get("image_path", ""),
                    "gold": item["gold"],
                },
            )
            if row["gold"] != item["gold"]:
                raise ValueError(f"Conflicting gold text for {sample_id}")
            alias = item["model_alias"]
            row[alias] = item["prediction"]
    return [merged[key] for key in sorted(merged)]


def disagreement_categories(row: dict) -> list[str]:
    gold = row["gold"]
    correct = {column: row.get(column) == gold for column in MODEL_COLUMNS}
    categories: list[str] = []
    if "manga_ocr" in row and "paddle_manga" in row:
        if correct["manga_ocr"] and correct["paddle_manga"]:
            categories.append("both_older_correct")
        elif correct["manga_ocr"] and not correct["paddle_manga"]:
            categories.append("manga_ocr_correct_paddle_manga_wrong")
        elif correct["paddle_manga"] and not correct["manga_ocr"]:
            categories.append("paddle_manga_correct_manga_ocr_wrong")
        else:
            categories.append("both_older_wrong")
    if "new_model" in row and correct["new_model"]:
        if not correct["manga_ocr"] and not correct["paddle_manga"]:
            categories.append("new_model_correct_both_older_fail")
        if "paddle_manga" in row and not correct["paddle_manga"]:
            categories.append("new_model_correct_existing_paddle_wrong")
    if "new_model" in row and "paddle_manga" in row:
        if correct["paddle_manga"] and not correct["new_model"]:
            categories.append("existing_paddle_correct_new_model_wrong")
    return categories or ["unclassified"]


def create_reports(rows: list[dict], output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "predictions.csv"
    fieldnames = ["sample_id", "image_path", "gold", *MODEL_COLUMNS]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    disagreement_path = output_dir / "disagreements.csv"
    with disagreement_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*fieldnames, "categories"], extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "categories": ";".join(disagreement_categories(row))})

    metrics: dict[str, dict] = {}
    for model in MODEL_COLUMNS:
        pairs = [(row["gold"], row[model]) for row in rows if model in row]
        if not pairs:
            continue
        metrics[model] = {
            "raw": aggregate_metrics(pairs, "preserve"),
            "line_endings": aggregate_metrics(pairs, "line_endings"),
            "nfkc_whitespace_diagnostic": aggregate_metrics(pairs, "nfkc_whitespace"),
        }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report_path = output_dir / "report.html"
    report_path.write_text(_render_html(rows), encoding="utf-8")
    return {
        "predictions_csv": str(csv_path),
        "disagreements_csv": str(disagreement_path),
        "metrics_json": str(metrics_path),
        "html": str(report_path),
    }


def _render_html(rows: list[dict]) -> str:
    blocks: list[str] = []
    for row in rows:
        predictions = []
        for model in MODEL_COLUMNS:
            if model not in row:
                continue
            scored = score_pair(row["gold"], row[model])
            predictions.append(
                "<tr><th>{}</th><td>{}</td><td>{}</td><td>{:.4f}</td></tr>".format(
                    html.escape(model),
                    html.escape(row[model]),
                    scored.edit_distance,
                    scored.cer,
                )
            )
        blocks.append(
            "<article><h2>{}</h2><img loading='lazy' src='{}' alt='{}'>"
            "<p><strong>Gold:</strong> {}</p><table><thead><tr><th>Model</th>"
            "<th>Prediction</th><th>Edit distance</th><th>CER</th></tr></thead>"
            "<tbody>{}</tbody></table><p>{}</p></article>".format(
                html.escape(row["sample_id"]),
                html.escape(row.get("image_path", ""), quote=True),
                html.escape(row["sample_id"], quote=True),
                html.escape(row["gold"]),
                "".join(predictions),
                html.escape(", ".join(disagreement_categories(row))),
            )
        )
    return """<!doctype html><html lang="en"><meta charset="utf-8">
<title>Manga OCR comparison</title><style>
body{font-family:system-ui,sans-serif;max-width:1100px;margin:auto;padding:1rem}
article{border-bottom:1px solid #ccc;padding:1rem 0}img{max-width:420px;max-height:240px}
table{border-collapse:collapse;width:100%}th,td{border:1px solid #ccc;padding:.4rem;text-align:left}
</style><h1>Manga OCR comparison</h1>""" + "".join(blocks) + "</html>\n"

