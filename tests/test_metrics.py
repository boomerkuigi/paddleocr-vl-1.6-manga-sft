from manga_sft.metrics import aggregate_metrics, edit_distance, score_pair


def test_edit_distance_and_cer():
    assert edit_distance("猫", "犬") == 1
    assert edit_distance("アナタ", "アナタ") == 0
    result = score_pair("かな", "か")
    assert result.edit_distance == 1
    assert result.cer == 0.5
    assert not result.exact


def test_exact_and_micro_cer():
    result = aggregate_metrics([("猫", "猫"), ("かな", "か")])
    assert result["exact_count"] == 1
    assert result["failure_count"] == 1
    assert result["exact_accuracy"] == 0.5
    assert result["cer_micro"] == 1 / 3


def test_normalized_metric_is_explicitly_secondary():
    assert not score_pair("Ａ　Ｂ", "AB", "preserve").exact
    assert score_pair("Ａ　Ｂ", "AB", "nfkc_whitespace").exact

