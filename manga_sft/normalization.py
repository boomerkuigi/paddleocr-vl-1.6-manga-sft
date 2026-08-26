from __future__ import annotations

import re
import unicodedata


NORMALIZATION_MODES = ("preserve", "line_endings", "nfkc_whitespace")


def normalize_text(text: str, mode: str = "preserve") -> str:
    """Normalize text using an explicitly named, auditable policy.

    `preserve` is the headline policy: punctuation, width, kana, symbols,
    whitespace, and ordering are untouched. `line_endings` only canonicalizes
    platform line endings. `nfkc_whitespace` is a secondary diagnostic and must
    never replace the raw metrics.
    """
    if mode == "preserve":
        return text
    if mode == "line_endings":
        return text.replace("\r\n", "\n").replace("\r", "\n")
    if mode == "nfkc_whitespace":
        value = unicodedata.normalize("NFKC", text)
        return re.sub(r"\s+", "", value)
    raise ValueError(f"Unknown normalization mode: {mode}")


def training_target(text: str) -> str:
    """Return the XML transcription without benchmark-inflating normalization."""
    if "\x00" in text:
        raise ValueError("NUL is not valid in a training target")
    return text

