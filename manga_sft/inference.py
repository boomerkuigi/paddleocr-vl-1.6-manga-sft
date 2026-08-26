from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image


MODEL_IDS = {
    "manga_ocr": "kha-white/manga-ocr-base",
    "paddle_manga": "jzhang533/PaddleOCR-VL-For-Manga",
    "paddle_1_6": "PaddlePaddle/PaddleOCR-VL-1.6",
}


class MangaOCRAdapter:
    def __init__(self, model_id: str = MODEL_IDS["manga_ocr"], device: str | None = None):
        try:
            from manga_ocr import MangaOcr
        except ImportError as exc:
            raise RuntimeError("Install the baseline extra: pip install -r requirements-baselines.txt") from exc
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
        from transformers import AutoModelForImageTextToText, AutoProcessor

        torch_dtype = getattr(torch, dtype)
        load_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "torch_dtype": torch_dtype,
            "device_map": device,
        }
        if revision:
            load_kwargs["revision"] = revision
        self.processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True, revision=revision
        )
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, **load_kwargs)
        self.model.eval()
        self.prompt = prompt

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

