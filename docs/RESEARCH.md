# Model and reference review

## Core models

### PaddleOCR-VL 1.6

- About 1B parameters, BF16, Apache-2.0.
- Current official loading uses Transformers 5+,
  `AutoProcessor`, and `AutoModelForImageTextToText` with remote model code.
- The task prompt is `OCR:`. The chat template's assistant boundary is
  `Assistant:\n`, which matters for label masking.
- Intended as a general document OCR/element-recognition VLM. For this project it
  receives already detected text-region crops, not raw manga pages.
- The forward method accepts labels and computes next-token cross-entropy, so a
  standard PyTorch full fine-tune is viable.

### PaddleOCR-VL-For-Manga

- About 1B parameters, BF16; model card declares Apache-2.0.
- Based on the older PaddleOCR-VL generation.
- Trained on roughly 0.1M Manga109-s crops plus about 1.5M synthetic samples.
- Model card reports roughly 70% sentence accuracy versus 27% for its base.
- It is a crop recognizer; it does not replace page text detection/order logic.
- Its published training loader uses crop-random splits, and its synthetic-data
  path can make in-training evaluation provenance hard to interpret. This port
  uses book-level splits and real data only for the first experiment.

### Manga-OCR

- Roughly 110M-parameter VisionEncoderDecoder; Apache-2.0.
- Manga-specialized for vertical/horizontal text, furigana, overlapping text,
  varied fonts, and multi-line bubbles.
- The upstream documentation notes hallucination on no-text images and reduced
  reliability for long text. It remains an important complementary baseline.
- Transformers 5 removed a pipeline behavior used by older examples; this
  project's adapter uses the maintained `manga_ocr.MangaOcr` package.

## Newer relevant work

[`genshiai-daichi/baberu-ocr`](https://huggingface.co/genshiai-daichi/baberu-ocr)
is a newer, much smaller crop OCR model (DINOv2 encoder plus a decoder) with
published Manga109-v2026 comparisons. Its authors exclude
PaddleOCR-VL-For-Manga from that benchmark because it trained on Manga109, a
useful reminder that benchmark provenance matters. Baberu is a sensible future
out-of-domain baseline but is not one of the four required pilot models.

A search also found `gigass03/PaddleOCR-VL-1.6-Manga-Finetuned`, but it was
private or otherwise inaccessible during review, so no claims or design choices
depend on it.

## Evaluation definition

Headline metrics are corpus/micro CER (sum of character edit distances divided
by sum of gold characters) and exact sentence accuracy. Macro/sample-mean CER,
perfect/failure counts, total edit distance, and a secondary normalization
diagnostic are also saved. Every generated string is preserved before metric
normalization.

