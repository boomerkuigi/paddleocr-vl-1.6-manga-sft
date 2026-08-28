from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image


MODEL_IDS = {
    "manga_ocr": "kha-white/manga-ocr-base",
    "paddle_manga": "jzhang533/PaddleOCR-VL-For-Manga",
    "paddle_1_6": "PaddlePaddle/PaddleOCR-VL-1.6",
}

MODEL_REVISIONS = {
    "manga_ocr": "aa6573bd10b0d446cbf622e29c3e084914df9741",
    "paddle_manga": "1e8aa5f1dd90cc86fe9137c9c0b26ebde613cfe8",
    "paddle_1_6": "c5630abae1d940eafe0697512a0325494b02ab42",
}


def normalize_paddleocr_vl_config(config):
    """Validate Transformers 5's native PaddleOCR-VL conversion and preserve untied weights."""
    text_config = getattr(config, "text_config", None)
    if text_config is None:
        raise RuntimeError(
            "Native Transformers PaddleOCR-VL config conversion did not create text_config; "
            "ensure trust_remote_code=False and use the pinned Transformers 5 version"
        )
    # All three benchmark Paddle checkpoints declare untied embeddings in their
    # source configs. Transformers 5's flat-config converter otherwise restores
    # the class default (True), even though input/output tensors differ.
    config.tie_word_embeddings = False
    text_config.tie_word_embeddings = False
    return config


def _legacy_processor_metadata(model_id: str, revision: str | None) -> dict[str, Any] | None:
    """Return legacy Paddle processor metadata without executing repository code."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import RemoteEntryNotFoundError

    try:
        path = hf_hub_download(
            repo_id=model_id,
            filename="preprocessor_config.json",
            revision=revision,
        )
    except RemoteEntryNotFoundError:
        return None
    metadata = json.loads(Path(path).read_text(encoding="utf-8"))
    size = metadata.get("size")
    if not isinstance(size, dict) or not ({"min_pixels", "max_pixels"} & size.keys()):
        return None
    return metadata


def load_paddleocr_vl_processor(model_id: str, revision: str | None = None):
    """Load a native Transformers 5 processor, repairing legacy flat metadata.

    Older PaddleOCR-VL repositories identify their image processor as SigLIP and
    store ``size={min_pixels,max_pixels}``. Transformers 5's native processor
    requires PaddleOCRVLImageProcessor and shortest/longest-edge size keys. The
    same repositories also omit the tokenizer's image-token attributes.
    """
    from transformers import AutoProcessor, AutoTokenizer
    from transformers.models.paddleocr_vl.image_processing_paddleocr_vl import (
        PaddleOCRVLImageProcessor,
    )
    from transformers.models.paddleocr_vl.processing_paddleocr_vl import (
        PaddleOCRVLProcessor,
    )

    common: dict[str, Any] = {"trust_remote_code": False, "revision": revision}
    metadata = _legacy_processor_metadata(model_id, revision)
    if metadata is None:
        return AutoProcessor.from_pretrained(model_id, **common)

    from huggingface_hub import hf_hub_download

    legacy_size = metadata["size"]
    min_pixels = metadata.get("min_pixels", legacy_size.get("min_pixels"))
    max_pixels = metadata.get("max_pixels", legacy_size.get("max_pixels"))
    if min_pixels is None or max_pixels is None:
        raise RuntimeError("Legacy PaddleOCR-VL processor metadata lacks pixel limits")

    image_processor = PaddleOCRVLImageProcessor.from_pretrained(
        model_id,
        size=None,
        min_pixels=int(min_pixels),
        max_pixels=int(max_pixels),
        **common,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, **common)
    image_token = getattr(tokenizer, "image_token", None) or "<|IMAGE_PLACEHOLDER|>"
    tokenizer.image_token = image_token
    tokenizer.image_token_id = tokenizer.convert_tokens_to_ids(image_token)
    if tokenizer.image_token_id is None:
        raise RuntimeError(f"PaddleOCR-VL tokenizer does not contain {image_token!r}")
    chat_template = Path(
        hf_hub_download(model_id, "chat_template.jinja", revision=revision)
    ).read_text(encoding="utf-8")
    return PaddleOCRVLProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        chat_template=chat_template,
    )


class MangaOCRAdapter:
    def __init__(
        self,
        model_id: str = MODEL_IDS["manga_ocr"],
        device: str | None = None,
        revision: str | None = None,
    ):
        try:
            from manga_ocr import MangaOcr
        except ImportError as exc:
            raise RuntimeError("Install the baseline extra: pip install -r requirements-baselines.txt") from exc
        if revision:
            from huggingface_hub import snapshot_download

            model_id = snapshot_download(repo_id=model_id, revision=revision)
        kwargs: dict[str, Any] = {"pretrained_model_name_or_path": model_id}
        if device:
            kwargs["force_cpu"] = device == "cpu"
        self.model = MangaOcr(**kwargs)

    def predict(self, image_path: str | Path) -> str:
        with Image.open(image_path) as image:
            return str(self.model(image.convert("L")))


class PaddleOCRVLAdapter:
    def __init__(
        self,
        model_id: str,
        prompt: str = "OCR:",
        device: str = "auto",
        dtype: str = "bfloat16",
        revision: str | None = None,
    ):
        import torch
        from transformers import AutoConfig, AutoModelForImageTextToText

        torch_dtype = getattr(torch, dtype)
        common: dict[str, Any] = {
            "trust_remote_code": False,
            "revision": revision,
        }
        self.processor = load_paddleocr_vl_processor(model_id, revision=revision)
        config = normalize_paddleocr_vl_config(
            AutoConfig.from_pretrained(model_id, **common)
        )
        self.model, loading_info = AutoModelForImageTextToText.from_pretrained(
            model_id,
            config=config,
            dtype=torch_dtype,
            device_map=device,
            attn_implementation="sdpa",
            output_loading_info=True,
            **common,
        )
        problems = {
            key: sorted(loading_info.get(key, []))
            for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
        }
        if any(problems.values()):
            raise RuntimeError(f"PaddleOCR-VL checkpoint did not load exactly: {problems}")
        self.model.eval()
        self.prompt = prompt
        self.loading_info = problems

    def predict(self, image_path: str | Path) -> str:
        import torch

        with Image.open(image_path) as source:
            image = source.convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[text], images=[image], return_tensors="pt")
        target_device = next(self.model.parameters()).device
        inputs = {key: value.to(target_device) for key, value in inputs.items()}
        with torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=512, do_sample=False)
        prompt_tokens = inputs["input_ids"].shape[1]
        return self.processor.batch_decode(
            generated[:, prompt_tokens:], skip_special_tokens=True
        )[0]
