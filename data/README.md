# Data directory

No Manga109-s images, annotations, crops, or labels are committed here.
Manga109-s forbids redistribution to third parties. Obtain access from the
official gated Hugging Face dataset, accept its terms, and make the mounted or
downloaded data available as:

```text
Manga109-s/
├── annotations/
│   └── BOOK.xml
└── images/
    └── BOOK/
        └── 000.jpg
```

Then run:

```bash
python scripts/prepare_dataset.py \
  --manga109-root /absolute/path/to/Manga109-s \
  --output data/prepared \
  --seed 42
```

The official gated Hub repository currently exposes
`Manga109s_released_2026_05_21.zip`. The Hugging Face Job entrypoint safely
extracts that single mounted archive into ephemeral Job storage before invoking
the same preparation script. Nested archive roots are detected automatically.

The generated crops and manifests remain local and are ignored by Git. The
default split assigns entire books, not individual crops, to deterministic
80/10/10 train/validation/test partitions. `split_summary.json` records the
seed, assigned books, sizes, and filtering. The test manifest is loaded only for
leakage validation during training and is never passed to `Trainer`.

Training targets use the annotation string as supplied. No NFKC conversion,
whitespace deletion, punctuation folding, kana folding, or symbol stripping is
performed.

For a format-only dry run (not meaningful OCR training data):

```bash
python scripts/create_smoke_fixture.py
python scripts/validate_environment.py --config configs/smoke.yaml
```
