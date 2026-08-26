from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image


ASSISTANT_MARKER = "Assistant:\n"


def find_subsequence(sequence: list[int], subsequence: list[int]) -> int:
    if not subsequence:
        raise ValueError("response marker token sequence is empty")
    for start in range(len(sequence) - len(subsequence) + 1):
        if sequence[start : start + len(subsequence)] == subsequence:
            return start
    return -1


@dataclass
class PaddleOCRVLCollator:
    processor: Any
    prompt: str = "OCR:"
    max_length: int = 2048

    def __post_init__(self) -> None:
        marker = self.processor.tokenizer.encode(ASSISTANT_MARKER, add_special_tokens=False)
        if not marker:
            raise ValueError("Tokenizer produced no assistant marker tokens")
        self.response_marker_ids = marker

    def __call__(self, examples: list[dict]) -> dict:
        texts: list[str] = []
        images: list[Image.Image] = []
        for example in examples:
            image = Image.open(example["image_path"]).convert("RGB")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": self.prompt},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": example["gold"]}],
                },
            ]
            texts.append(
                self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
            )
            images.append(image)

        batch = self.processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt",
            truncation=False,
        )
        if batch["input_ids"].shape[1] > self.max_length:
            raise ValueError(
                f"Batch sequence length {batch['input_ids'].shape[1]} exceeds "
                f"model.max_length={self.max_length}; do not silently truncate OCR targets"
            )

        labels = batch["input_ids"].clone()
        pad_id = self.processor.tokenizer.pad_token_id
        for row_index, row_tensor in enumerate(batch["input_ids"]):
            row = row_tensor.tolist()
            marker_start = find_subsequence(row, self.response_marker_ids)
            if marker_start < 0:
                raise ValueError(
                    "PaddleOCR-VL 1.6 assistant marker was not found. "
                    "The model chat template may have changed."
                )
            response_start = marker_start + len(self.response_marker_ids)
            labels[row_index, :response_start] = -100
            if pad_id is not None:
                labels[row_index, row_tensor == pad_id] = -100
        batch["labels"] = labels
        return batch
