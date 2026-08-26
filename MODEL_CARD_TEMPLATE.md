---
license: apache-2.0
base_model: PaddlePaddle/PaddleOCR-VL-1.6
pipeline_tag: image-text-to-text
language:
- ja
tags:
- ocr
- manga
- paddleocr-vl
---

# PaddleOCR-VL-1.6-For-Manga (experimental)

Private experimental checkpoint fine-tuned from
`PaddlePaddle/PaddleOCR-VL-1.6` at revision
`c5630abae1d940eafe0697512a0325494b02ab42` for Japanese manga text-region
recognition.

## Intended use

OCR of already detected manga text crops. It is not a full-page detector and
must not be assumed to determine page reading order.

## Training data

Manga109-s [VERSION], prepared locally into text-region crops. Manga109-s images
and annotations are not redistributed. Add the required Manga109 citations and
the exact train/validation/test counts from `split_summary.json` before release.

## Training

Attach `configs/pilot.yaml`, exact package lock/export, hardware, wall time,
revision, seed, and final Trainer state.

## Evaluation

Attach held-out metrics and raw/disagreement reports for Manga-OCR, existing
Paddle manga, vanilla PaddleOCR-VL 1.6, and this checkpoint. State whether any
reported set overlaps upstream training data.

## Limitations

Document failure modes by vertical/horizontal layout, furigana, decorative text,
low resolution, unusual symbols, no-text crops, and long/multi-line regions.

## Licenses and attribution

Apache-2.0 base. Manga109-s use must be disclosed and cited according to its
terms. Complete all placeholders before making the repository public.

