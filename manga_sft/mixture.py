from __future__ import annotations

import hashlib
import random
from collections import Counter
from dataclasses import dataclass
from typing import Iterable


# This is deliberately a small, auditable text-only/geometry heuristic.  It is
# used to choose additional *training exposure*, not to relabel a crop.
LONG_MARK_FORMS = frozenset("ーｰ―─−")
ELLIPSIS_AND_MIDDLE_DOT_FORMS = frozenset("…‥・･")
ASCII_PUNCTUATION = frozenset(r'''!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~''')
FULLWIDTH_PUNCTUATION = frozenset("，．！？：；（）［］｛｝「」『』【】〈〉《》")

TARGET_GROUPS = (
    "repeated_or_long_mark",
    "likely_sfx",
    "punctuation_form",
    "visual_or_unusual_unicode",
)
DEFAULT_TARGET_WEIGHTS = {
    "repeated_or_long_mark": 0.40,
    "likely_sfx": 0.25,
    "punctuation_form": 0.20,
    "visual_or_unusual_unicode": 0.15,
}


def has_repeated_character(text: str) -> bool:
    return any(left == right for left, right in zip(text, text[1:]))


def _katakana_fraction(text: str) -> float:
    meaningful = [char for char in text if not char.isspace()]
    if not meaningful:
        return 0.0
    katakana = sum("\u30a0" <= char <= "\u30ff" for char in meaningful)
    return katakana / len(meaningful)


def _punctuation_fraction(text: str) -> float:
    meaningful = [char for char in text if not char.isspace()]
    if not meaningful:
        return 0.0
    punct = sum(
        char in ASCII_PUNCTUATION
        or char in FULLWIDTH_PUNCTUATION
        or char in ELLIPSIS_AND_MIDDLE_DOT_FORMS
        or char in LONG_MARK_FORMS
        for char in meaningful
    )
    return punct / len(meaningful)


def likely_sfx(text: str) -> bool:
    """Conservative SFX/onomatopoeia proxy used in the V2 audit."""
    return (
        has_repeated_character(text)
        or (len(text) <= 8 and _katakana_fraction(text) >= 0.40)
        or _punctuation_fraction(text) >= 0.20
    )


def _is_unusual_unicode(char: str) -> bool:
    if char.isspace() or char.isascii() or char in FULLWIDTH_PUNCTUATION:
        return False
    codepoint = ord(char)
    return not (
        0x3040 <= codepoint <= 0x30ff  # Hiragana + Katakana
        or 0x3400 <= codepoint <= 0x9fff  # CJK unified ideographs
        or 0xff00 <= codepoint <= 0xffef  # Full-width forms
        or 0x3000 <= codepoint <= 0x303f  # CJK punctuation
    )


def _visual_or_unusual_unicode(row: dict) -> bool:
    text = str(row["gold"])
    bbox = row.get("bbox") or []
    if len(bbox) == 4:
        width = max(1, int(bbox[2]) - int(bbox[0]))
        height = max(1, int(bbox[3]) - int(bbox[1]))
        # Vertical text, unusually compact crops, and symbols outside the
        # ordinary Japanese OCR character ranges are intentionally pooled.
        if height / width >= 1.5 or width * height <= 1600:
            return True
    return any(_is_unusual_unicode(char) for char in text)


def feature_groups(row: dict) -> frozenset[str]:
    """Return all eligible V2 targeting groups for one manifest row."""
    text = str(row["gold"])
    groups: set[str] = set()
    if has_repeated_character(text) or any(char in LONG_MARK_FORMS for char in text):
        groups.add("repeated_or_long_mark")
    if likely_sfx(text):
        groups.add("likely_sfx")
    if (
        any(char in ASCII_PUNCTUATION or char in FULLWIDTH_PUNCTUATION for char in text)
        or any(char in ELLIPSIS_AND_MIDDLE_DOT_FORMS for char in text)
    ):
        groups.add("punctuation_form")
    if _visual_or_unusual_unicode(row):
        groups.add("visual_or_unusual_unicode")
    return frozenset(groups)


def _quota_counts(extra_draws: int, weights: dict[str, float]) -> dict[str, int]:
    if extra_draws <= 0:
        raise ValueError("extra_draws must be positive")
    if set(weights) != set(TARGET_GROUPS):
        raise ValueError(f"target weights must name exactly {TARGET_GROUPS}")
    if abs(sum(weights.values()) - 1.0) > 1e-9 or any(weight < 0 for weight in weights.values()):
        raise ValueError("target weights must be non-negative and sum to one")
    counts = {group: int(extra_draws * weights[group]) for group in TARGET_GROUPS}
    remainder = extra_draws - sum(counts.values())
    # The requested V2 proportions divide 25,000 exactly; retain a stable rule
    # for future values that do not.
    fractions = sorted(
        ((extra_draws * weights[group] - counts[group], group) for group in TARGET_GROUPS),
        reverse=True,
    )
    for _, group in fractions[:remainder]:
        counts[group] += 1
    return counts


def _rank(seed: int, group: str, sample_id: str) -> bytes:
    return hashlib.sha256(f"{seed}:{group}:{sample_id}".encode("utf-8")).digest()


@dataclass(frozen=True)
class MixturePlan:
    seed: int
    base_count: int
    extra_draws: int
    target_counts: dict[str, int]
    selected_indices: tuple[int, ...]
    selected_groups: tuple[str, ...]
    sampler_entries: tuple[tuple[int, str], ...]

    @property
    def total_draws(self) -> int:
        return self.base_count + self.extra_draws

    def summary(self, rows: list[dict]) -> dict:
        selected_ids = [str(rows[index]["sample_id"]) for index in self.selected_indices]
        return {
            "seed": self.seed,
            "ordinary_train_exposures": self.base_count,
            "additional_targeted_exposures": self.extra_draws,
            "total_exposures_per_mixture_epoch": self.total_draws,
            "targeted_exposure_counts": dict(self.target_counts),
            "unique_targeted_samples": len(set(self.selected_indices)),
            "targeted_sample_id_sha256": hashlib.sha256(
                "\n".join(selected_ids).encode("utf-8")
            ).hexdigest(),
        }


def build_mixture_plan(
    rows: list[dict],
    *,
    seed: int,
    extra_draws: int = 25_000,
    target_weights: dict[str, float] | None = None,
) -> MixturePlan:
    """Build one deterministic, no-accidental-duplication V2 mixture epoch.

    Every train row is exposed once.  The 25,000 added slots are assigned to a
    shuffled sequence of exact group quotas.  Each group ranks its eligible
    rows by a salted SHA-256 score; a row already selected for another slot is
    skipped.  Consequently an overlapping row is deliberately doubled at most
    once (ordinary + one targeted exposure), never once per matching feature.
    """
    if len({str(row["sample_id"]) for row in rows}) != len(rows):
        raise ValueError("train manifest must contain unique sample_id values")
    weights = target_weights or DEFAULT_TARGET_WEIGHTS
    quotas = _quota_counts(extra_draws, weights)
    eligible = {
        group: sorted(
            (index for index, row in enumerate(rows) if group in feature_groups(row)),
            key=lambda index: _rank(seed, group, str(rows[index]["sample_id"])),
        )
        for group in TARGET_GROUPS
    }
    for group, count in quotas.items():
        if len(eligible[group]) < count:
            raise ValueError(
                f"Not enough {group} examples for {count} deterministic extra draws; "
                f"only {len(eligible[group])} are eligible"
            )

    slots = [group for group, count in quotas.items() for _ in range(count)]
    random.Random(seed).shuffle(slots)
    cursors = Counter()
    selected: set[int] = set()
    selected_indices: list[int] = []
    selected_groups: list[str] = []
    for group in slots:
        candidates = eligible[group]
        while cursors[group] < len(candidates) and candidates[cursors[group]] in selected:
            cursors[group] += 1
        if cursors[group] >= len(candidates):
            raise ValueError(
                f"Overlap left too few unique rows to fulfill the {group} quota of {quotas[group]}"
            )
        index = candidates[cursors[group]]
        cursors[group] += 1
        selected.add(index)
        selected_indices.append(index)
        selected_groups.append(group)

    # The final permutation is fixed from the seed.  Restarting from a Trainer
    # checkpoint therefore replays the same stream and its skipped batches.
    sampler_entries = [(index, "ordinary") for index in range(len(rows))] + list(
        zip(selected_indices, selected_groups, strict=True)
    )
    random.Random(seed).shuffle(sampler_entries)
    return MixturePlan(
        seed=seed,
        base_count=len(rows),
        extra_draws=extra_draws,
        target_counts=dict(quotas),
        selected_indices=tuple(selected_indices),
        selected_groups=tuple(selected_groups),
        sampler_entries=tuple(sampler_entries),
    )


class DeterministicMixtureSampler:
    """Minimal index sampler with a stable order across restart/resume."""

    def __init__(self, plan: MixturePlan):
        self.plan = plan

    def __iter__(self) -> Iterable[tuple[int, str]]:
        # The Dataset discards the provenance token before collating.  Keeping
        # it in the sampler is what makes partial-pilot exposure accounting
        # exact rather than inferring category membership from a duplicated ID.
        return iter(self.plan.sampler_entries)

    def __len__(self) -> int:
        return len(self.plan.sampler_entries)

    def set_epoch(self, epoch: int) -> None:
        # Trainer/Accelerate may call this method.  This pilot deliberately has
        # a fixed epoch order so checkpoint resume remains reproducible.
        del epoch
