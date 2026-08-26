import csv
import json
from pathlib import Path

from manga_sft.reporting import create_reports, merge_prediction_files, write_prediction_jsonl
from scripts.sanitize_results_for_hub import sanitize


def test_result_serialization_and_disagreements(tmp_path: Path):
    first = tmp_path / "manga.jsonl"
    second = tmp_path / "paddle.jsonl"
    common = {"sample_id": "case", "image_path": "crop.png", "gold": "正解"}
    write_prediction_jsonl(
        first, [{**common, "model_alias": "manga_ocr", "prediction": "正解"}]
    )
    write_prediction_jsonl(
        second, [{**common, "model_alias": "paddle_manga", "prediction": "誤り"}]
    )
    rows = merge_prediction_files([first, second])
    paths = create_reports(rows, tmp_path / "report")
    metrics = json.loads(Path(paths["metrics_json"]).read_text(encoding="utf-8"))
    assert metrics["manga_ocr"]["raw"]["exact_accuracy"] == 1.0
    with Path(paths["disagreements_csv"]).open(encoding="utf-8-sig") as handle:
        row = next(csv.DictReader(handle))
    assert "manga_ocr_correct_paddle_manga_wrong" in row["categories"]
    assert Path(paths["html"]).is_file()
    safe = sanitize(tmp_path, tmp_path / "report", tmp_path / "safe")
    assert safe["prediction_rows"] == 2
    content = (tmp_path / "safe" / "raw_model_predictions.jsonl").read_text(encoding="utf-8")
    assert '"gold"' not in content
    assert "crop.png" not in content
