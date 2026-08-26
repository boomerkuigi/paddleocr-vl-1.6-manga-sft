# License and redistribution assessment

This is a technical assessment, not legal advice. Re-check upstream terms before
public release because repository cards and dataset terms can change.

| Asset | Observed terms | Practical consequence |
|---|---|---|
| PaddleOCR-VL 1.6 | Apache-2.0 | Fine-tuning and redistribution are generally permitted with notices. |
| `jzhang533/PaddleOCR-VL-For-Manga` weights | HF card declares Apache-2.0 | Baseline use is allowed under that card; its GitHub code has no license file, so none is copied here. |
| `kha-white/manga-ocr-base` / code | Apache-2.0 | Baseline evaluation and redistribution under Apache terms. |
| Manga109-s | custom gated terms | Third-party redistribution is forbidden; do not upload pages, annotations, or derived crops. Published trained models are expressly permitted if Manga109-s use is clearly disclosed. |
| OpenVINO-book manga SFT code | MIT | May be reused with notice, but this repository uses original code. |
| `megemini/PaddleOCR-VL-KIE` code | no license file found | Cite findings only; do not copy code. |

The official Manga109-s terms also permit machine-learning/image-processing
experiments and commercial use of results, subject to their conditions. Direct
or modified manga images cannot be sold as products, and publication of whole
pages is limited by the terms. This project takes the stricter operational
position: no Manga109-s image or derived crop leaves the user's private runtime.

Therefore:

- no Hugging Face dataset repository is created for Manga109-s derivatives;
- Git ignores all local raw/prepared data;
- a Hugging Face Job mounts the official gated dataset read-only;
- reports containing crop thumbnails remain local/ignored;
- the experimental model repository remains private;
- a later public model release appears permitted if the model card identifies
  Manga109-s use, provides required attribution/citations, includes base-model
  notices, and contains no training images.

Current Manga109-s is the v2026 release (87 books). A user holding the v2023
layout may also prepare it locally, but dataset version must be recorded because
split sizes and comparability differ.

