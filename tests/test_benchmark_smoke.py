import json
from types import SimpleNamespace

from scripts import run_benchmark_smoke


def test_smoke_runner_records_each_model_and_exact_metrics(tmp_path, monkeypatch):
    manifest = tmp_path / "smoke.jsonl"
    manifest.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "output"
    specs = [
        ("manga_ocr", "repo/a", "rev-a"),
        ("paddle_manga", "repo/b", "rev-b"),
        ("paddle_1_6", "repo/c", "rev-c"),
        ("new_model", "repo/d", "rev-d"),
    ]
    monkeypatch.setattr(run_benchmark_smoke, "model_specs", lambda: specs)

    def fake_run(command, stdout, **kwargs):
        alias = command[command.index("--model") + 1]
        prediction_path = command[command.index("--output") + 1]
        rows = [
            {
                "sample_id": f"sample-{index}",
                "gold": "正解",
                "prediction": "正解" if index < 75 else "誤り",
                "model_alias": alias,
                "model_id": command[command.index("--model-id") + 1],
                "model_revision": command[command.index("--revision") + 1],
            }
            for index in range(100)
        ]
        with open(prediction_path, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        stdout.write("completed\n")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_benchmark_smoke.subprocess, "run", fake_run)
    summary = run_benchmark_smoke.run(manifest, output)

    assert summary["all_models_passed"] is True
    assert set(summary["models"]) == {item[0] for item in specs}
    for model in summary["models"].values():
        assert model["status"] == "passed"
        assert model["metrics"]["samples"] == 100
        assert model["metrics"]["exact_accuracy"] == 0.75
    safe_rows = (output / "raw_model_predictions.jsonl").read_text(encoding="utf-8")
    assert '"gold"' not in safe_rows
