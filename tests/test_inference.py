import sys
from types import SimpleNamespace

import pytest

from manga_sft.inference import (
    PaddleOCRVLAdapter,
    load_paddleocr_vl_processor,
    normalize_paddleocr_vl_config,
)


def test_normalize_paddle_config_requires_native_text_config():
    with pytest.raises(RuntimeError, match="trust_remote_code=False"):
        normalize_paddleocr_vl_config(SimpleNamespace())


def test_normalize_paddle_config_preserves_untied_checkpoint_contract():
    config = SimpleNamespace(
        tie_word_embeddings=True,
        text_config=SimpleNamespace(tie_word_embeddings=True),
    )
    assert normalize_paddleocr_vl_config(config) is config
    assert config.tie_word_embeddings is False
    assert config.text_config.tie_word_embeddings is False


def test_paddle_adapter_uses_native_converted_config_and_exact_load(monkeypatch):
    calls = {}
    config = SimpleNamespace(
        tie_word_embeddings=True,
        text_config=SimpleNamespace(tie_word_embeddings=True),
    )
    processor = object()

    class FakeModel:
        def eval(self):
            calls["eval"] = True

    class FakeAutoConfig:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls["config"] = (model_id, kwargs)
            return config

    class FakeAutoModel:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls["model"] = (model_id, kwargs)
            return FakeModel(), {
                "missing_keys": [],
                "unexpected_keys": [],
                "mismatched_keys": [],
                "error_msgs": [],
            }

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(bfloat16="bf16"))
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoConfig=FakeAutoConfig,
            AutoModelForImageTextToText=FakeAutoModel,
        ),
    )
    monkeypatch.setattr(
        "manga_sft.inference.load_paddleocr_vl_processor",
        lambda model_id, revision=None: calls.setdefault(
            "processor", (model_id, revision, processor)
        )[2],
    )

    adapter = PaddleOCRVLAdapter("owner/model", revision="abc123")

    common = {"trust_remote_code": False, "revision": "abc123"}
    assert calls["processor"][:2] == ("owner/model", "abc123")
    assert calls["config"] == ("owner/model", common)
    model_kwargs = calls["model"][1]
    assert model_kwargs["config"] is config
    assert model_kwargs["dtype"] == "bf16"
    assert "torch_dtype" not in model_kwargs
    assert model_kwargs["trust_remote_code"] is False
    assert config.tie_word_embeddings is False
    assert config.text_config.tie_word_embeddings is False
    assert calls["eval"] is True
    assert not any(adapter.loading_info.values())


def test_paddle_adapter_rejects_partial_checkpoint_load(monkeypatch):
    config = SimpleNamespace(text_config=SimpleNamespace())

    class Loader:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return config

    class Model:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return SimpleNamespace(eval=lambda: None), {
                "missing_keys": ["model.language_model.weight"],
                "unexpected_keys": [],
                "mismatched_keys": [],
                "error_msgs": [],
            }

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(bfloat16="bf16"))
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoConfig=Loader,
            AutoModelForImageTextToText=Model,
        ),
    )
    monkeypatch.setattr(
        "manga_sft.inference.load_paddleocr_vl_processor",
        lambda *args, **kwargs: object(),
    )

    with pytest.raises(RuntimeError, match="did not load exactly"):
        PaddleOCRVLAdapter("owner/model", revision="abc123")


def test_native_processor_loader_uses_auto_processor_for_modern_metadata(monkeypatch):
    calls = {}

    class AutoProcessor:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls["auto"] = (model_id, kwargs)
            return "modern"

    monkeypatch.setattr("manga_sft.inference._legacy_processor_metadata", lambda *_: None)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoProcessor=AutoProcessor, AutoTokenizer=object()),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers.models.paddleocr_vl.image_processing_paddleocr_vl",
        SimpleNamespace(PaddleOCRVLImageProcessor=object()),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers.models.paddleocr_vl.processing_paddleocr_vl",
        SimpleNamespace(PaddleOCRVLProcessor=object()),
    )

    assert load_paddleocr_vl_processor("owner/model", "rev") == "modern"
    assert calls["auto"][1] == {"trust_remote_code": False, "revision": "rev"}


def test_native_processor_loader_repairs_legacy_metadata(monkeypatch, tmp_path):
    calls = {}
    template = tmp_path / "chat_template.jinja"
    template.write_text("template", encoding="utf-8")

    class ImageProcessor:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls["image"] = (model_id, kwargs)
            return "image-processor"

    class Tokenizer:
        image_token = None

        def convert_tokens_to_ids(self, token):
            calls["token"] = token
            return 100295

    class AutoTokenizer:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls["tokenizer"] = (model_id, kwargs)
            return Tokenizer()

    class Processor:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(
        "manga_sft.inference._legacy_processor_metadata",
        lambda *_: {
            "size": {"min_pixels": 147384, "max_pixels": 2822400},
            "min_pixels": 147384,
            "max_pixels": 2822400,
        },
    )
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(hf_hub_download=lambda *args, **kwargs: str(template)),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoProcessor=object(), AutoTokenizer=AutoTokenizer),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers.models.paddleocr_vl.image_processing_paddleocr_vl",
        SimpleNamespace(PaddleOCRVLImageProcessor=ImageProcessor),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers.models.paddleocr_vl.processing_paddleocr_vl",
        SimpleNamespace(PaddleOCRVLProcessor=Processor),
    )

    processor = load_paddleocr_vl_processor("legacy/model", "rev")
    assert calls["image"][1]["size"] is None
    assert calls["image"][1]["min_pixels"] == 147384
    assert calls["image"][1]["max_pixels"] == 2822400
    assert calls["image"][1]["trust_remote_code"] is False
    assert calls["token"] == "<|IMAGE_PLACEHOLDER|>"
    assert processor.kwargs["tokenizer"].image_token_id == 100295
    assert processor.kwargs["chat_template"] == "template"
