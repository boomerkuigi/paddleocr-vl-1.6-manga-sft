import json
import sys

import pytest

from scripts import evaluate_baselines


class DummyAdapter:
    def __init__(self, model_id, device, revision):
        self.arguments = (model_id, device, revision)

    def predict(self, image_path):
        return "予測"


def test_new_model_revision_is_forwarded_and_recorded(tmp_path, monkeypatch):
    output = tmp_path / "predictions.jsonl"
    adapters = []

    def create_adapter(model_id, device, revision):
        adapter = DummyAdapter(model_id, device, revision)
        adapters.append(adapter)
        return adapter

    monkeypatch.setattr(evaluate_baselines, "PaddleOCRVLAdapter", create_adapter)
    monkeypatch.setattr(
        evaluate_baselines,
        "read_jsonl",
        lambda *args, **kwargs: [
            {"sample_id": "sample-1", "image_path": "crop.png", "gold": "正解"}
        ],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_baselines.py",
            "--manifest",
            "test.jsonl",
            "--model",
            "new_model",
            "--model-id",
            "owner/model",
            "--revision",
            "abc123",
            "--output",
            str(output),
        ],
    )

    evaluate_baselines.main()

    assert adapters[0].arguments == ("owner/model", "auto", "abc123")
    prediction = json.loads(output.read_text(encoding="utf-8"))
    assert prediction["model_id"] == "owner/model"
    assert prediction["model_revision"] == "abc123"


def test_new_model_rejects_a_moving_revision(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_MODEL_REVISION", raising=False)
    monkeypatch.setattr(evaluate_baselines, "read_jsonl", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_baselines.py",
            "--manifest",
            "test.jsonl",
            "--model",
            "new_model",
            "--model-id",
            "owner/model",
            "--output",
            str(tmp_path / "predictions.jsonl"),
        ],
    )

    with pytest.raises(ValueError, match="revision"):
        evaluate_baselines.main()
