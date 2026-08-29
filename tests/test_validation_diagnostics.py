from manga_sft.validation import diagnostic_metrics, validation_group_indices


def _row(sample_id: str, gold: str, bbox: list[int]) -> dict:
    return {"sample_id": sample_id, "gold": gold, "bbox": bbox}


def test_validation_groups_are_views_of_only_validation_rows():
    rows = [
        _row("v1", "ああ", [0, 0, 100, 100]),
        _row("v2", "…・", [0, 0, 100, 100]),
        _row("v3", "漢字", [0, 0, 20, 80]),
    ]
    groups = validation_group_indices(rows)
    assert groups["all_validation"] == [0, 1, 2]
    assert 0 in groups["repeated_or_long_mark"]
    assert 1 in groups["punctuation_form"]
    assert 2 in groups["visual_or_unusual_unicode"]


def test_validation_diagnostics_report_exact_cer_and_edit_bins():
    rows = [
        _row("v1", "ああ", [0, 0, 100, 100]),
        _row("v2", "…・", [0, 0, 100, 100]),
        _row("v3", "漢字", [0, 0, 20, 80]),
    ]
    metrics = diagnostic_metrics(rows, ["ああ", "…", "漢"])
    primary = metrics["all_validation"]
    assert primary["exact_count"] == 1
    assert primary["edit_distance_distribution"] == {"0": 1, "1": 2, "2": 0, "3_plus": 0}
