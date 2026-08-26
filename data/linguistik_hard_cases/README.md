# Linguistik hard cases (evaluation only)

Put real, user-authorized crop images in the ignored `images/` directory and
create `manifest.jsonl` with one JSON object per line:

```json
{"sample_id":"case-001","image_path":"images/case-001.png","gold":"正しい日本語","manga_ocr":null,"paddle_manga":null,"notes":"Why this crop is difficult"}
```

Fields:

- `sample_id`: stable unique ID.
- `image_path`: crop path relative to this manifest.
- `gold`: human-verified Japanese transcription.
- `manga_ocr`: optional saved Manga-OCR prediction.
- `paddle_manga`: optional saved Paddle manga prediction.
- `notes`: optional provenance/difficulty notes.

The known reading placeholder is in `manifest.example.jsonl`, but its
`image_path` is null. It cannot be evaluated until the real crop is supplied.
Hard cases are evaluation-only and must never be merged into a training
manifest.

