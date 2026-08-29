from collections import Counter

from manga_sft.mixture import (
    DEFAULT_TARGET_WEIGHTS,
    DeterministicMixtureSampler,
    build_mixture_plan,
    feature_groups,
)


def _row(index: int, text: str, bbox: list[int]) -> dict:
    return {"sample_id": f"sample-{index:05d}", "gold": text, "bbox": bbox}


def _rows() -> list[dict]:
    rows = []
    # Enough deliberately overlapping candidates make the uniqueness rule real
    # rather than relying on mutually exclusive synthetic classes.
    for index in range(500):
        rows.append(_row(index, "ああ", [0, 0, 100, 100]))
    for index in range(500, 1000):
        rows.append(_row(index, "カタカナ", [0, 0, 100, 100]))
    for index in range(1000, 1500):
        rows.append(_row(index, "...", [0, 0, 100, 100]))
    for index in range(1500, 2000):
        rows.append(_row(index, "漢字", [0, 0, 20, 80]))
    return rows


def test_feature_groups_cover_requested_v2_categories():
    assert "repeated_or_long_mark" in feature_groups(_row(1, "ああー", [0, 0, 100, 100]))
    assert "likely_sfx" in feature_groups(_row(2, "ガタン", [0, 0, 100, 100]))
    assert "punctuation_form" in feature_groups(_row(3, "…・", [0, 0, 100, 100]))
    assert "visual_or_unusual_unicode" in feature_groups(_row(4, "漢字", [0, 0, 20, 80]))


def test_mixture_is_deterministic_with_exact_quotas_and_no_target_duplication():
    rows = _rows()
    first = build_mixture_plan(rows, seed=42, extra_draws=100, target_weights=DEFAULT_TARGET_WEIGHTS)
    second = build_mixture_plan(rows, seed=42, extra_draws=100, target_weights=DEFAULT_TARGET_WEIGHTS)
    assert first == second
    assert first.target_counts == {
        "repeated_or_long_mark": 40,
        "likely_sfx": 25,
        "punctuation_form": 20,
        "visual_or_unusual_unicode": 15,
    }
    assert Counter(first.selected_groups) == Counter(first.target_counts)
    assert len(set(first.selected_indices)) == 100
    counts = Counter(index for index, _ in DeterministicMixtureSampler(first))
    assert len(counts) == len(rows)
    assert set(counts.values()) <= {1, 2}
    assert sum(count == 2 for count in counts.values()) == 100
    assert len(first.sampler_entries) == len(rows) + 100
    assert Counter(group for _, group in first.sampler_entries)["ordinary"] == len(rows)


def test_mixture_rejects_duplicate_train_ids():
    rows = _rows()
    rows.append(dict(rows[0]))
    try:
        build_mixture_plan(rows, seed=42, extra_draws=100)
    except ValueError as error:
        assert "unique sample_id" in str(error)
    else:
        raise AssertionError("duplicate manifest IDs must be rejected")
