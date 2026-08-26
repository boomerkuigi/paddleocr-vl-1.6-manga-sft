from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .normalization import normalize_text


def edit_distance(reference: str, prediction: str) -> int:
    if len(reference) < len(prediction):
        reference, prediction = prediction, reference
    previous = list(range(len(prediction) + 1))
    for row, ref_char in enumerate(reference, start=1):
        current = [row]
        for col, pred_char in enumerate(prediction, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[col] + 1,
                    previous[col - 1] + (ref_char != pred_char),
                )
            )
        previous = current
    return previous[-1]


@dataclass(frozen=True)
class PairMetric:
    edit_distance: int
    reference_chars: int
    cer: float
    exact: bool


def score_pair(gold: str, prediction: str, normalization: str = "preserve") -> PairMetric:
    ref = normalize_text(gold, normalization)
    hyp = normalize_text(prediction, normalization)
    distance = edit_distance(ref, hyp)
    denominator = max(1, len(ref))
    return PairMetric(distance, len(ref), distance / denominator, ref == hyp)


def aggregate_metrics(
    pairs: Iterable[tuple[str, str]], normalization: str = "preserve"
) -> dict[str, float | int | str]:
    scored = [score_pair(gold, prediction, normalization) for gold, prediction in pairs]
    total_distance = sum(item.edit_distance for item in scored)
    total_chars = sum(item.reference_chars for item in scored)
    perfect = sum(item.exact for item in scored)
    count = len(scored)
    return {
        "normalization": normalization,
        "samples": count,
        "exact_count": perfect,
        "failure_count": count - perfect,
        "exact_accuracy": perfect / count if count else 0.0,
        "cer_micro": total_distance / max(1, total_chars),
        "cer_macro": sum(item.cer for item in scored) / count if count else 0.0,
        "total_edit_distance": total_distance,
        "total_reference_chars": total_chars,
    }

