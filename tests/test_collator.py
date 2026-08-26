from manga_sft.collator import find_subsequence


def test_find_subsequence():
    assert find_subsequence([1, 2, 3, 4, 2, 3], [2, 3]) == 1
    assert find_subsequence([1, 2], [3]) == -1

