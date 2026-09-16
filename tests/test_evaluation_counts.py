"""Hand-computed engineering text-view examples, not benchmark results."""

import pytest

from active_ocr.evaluation import (
    character_error_rate,
    edit_count,
    page_text_metrics,
    page_text_view,
)
from active_ocr.models import PredictionStatus as Status


@pytest.mark.parametrize(
    "reference,hypothesis,edits",
    [("", "abc", 3), ("abc", "", 3), ("cat", "cut", 1), ("ab", "cabd", 2), ("abc", "ac", 1)],
)
def test_raw_edits(reference, hypothesis, edits):
    assert edit_count(reference, hypothesis) == edits
    assert character_error_rate("", "abc") == 1  # Legacy shortcut stays compatible.


def test_text_view_preserves_order_blanks_case_punctuation_and_literal_text():
    lines = ["e\u0301\r\nA", "", " [ILLEGIBLE]!? ", "z"]
    assert page_text_view(lines) == "é\nA\n\n [ILLEGIBLE]!? \nz"
    assert lines == ["e\u0301\r\nA", "", " [ILLEGIBLE]!? ", "z"]
    same = page_text_metrics([["e\u0301", "a\r\nb"]], [["é", "a\nb"]], [Status.OK])
    assert same["char_edits"] == same["word_edits"] == 0
    ordered = page_text_metrics([["a", "b"]], [["b", "a"]], [Status.OK])
    assert ordered["char_edits"] == ordered["word_edits"] == 2
    blanks = page_text_metrics([["a", "", "b"]], [["a", "b"]], [Status.OK])
    assert blanks["char_edits"] == 1 and blanks["word_edits"] == 0


def test_micro_counts_include_empty_reference_insertions_and_can_exceed_one():
    result = page_text_metrics(
        [["a"], ["abcdefghi"], []], [["b"], ["abcdefghi"], ["xx yy"]], [Status.OK] * 3
    )
    assert result["pages"] == 3
    assert result["char_edits"] == 6 and result["reference_chars"] == 10
    assert result["word_edits"] == 3 and result["reference_words"] == 2
    assert result["cer"] == 0.6 and result["wer"] == 1.5
    assert all(type(v) is int for k, v in result.items() if k not in ("cer", "wer"))
    # Concatenating across pages would incorrectly erase these two edits.
    per_page = page_text_metrics([["a"], []], [[], ["a"]], [Status.OK] * 2)
    assert per_page["char_edits"] == 2 and per_page["cer"] == 2


def test_undefined_rates_and_unicode_whitespace():
    empty = page_text_metrics([[], [""]], [["abc"], []], [Status.OK] * 2)
    assert empty["char_edits"] == 3 and empty["word_edits"] == 1
    assert empty["cer_defined"] == empty["wer_defined"] == 0
    assert "cer" not in empty and "wer" not in empty
    space = page_text_metrics([[" \t\u2003"]], [[]], [Status.OK])
    assert space["cer_defined"] == 1 and space["cer"] == 1
    assert space["wer_defined"] == 0 and "wer" not in space
    tokens = page_text_metrics([["a\u2003b"]], [["a b"]], [Status.OK])
    assert tokens["word_edits"] == 0 and tokens["char_edits"] == 1
    with pytest.raises(ValueError):
        page_text_metrics([], [], [])


def test_failures_are_empty_hypotheses_counted_once_even_with_partial_text():
    result = page_text_metrics(
        [["a"], ["bc"], []],
        [["a"], ["bc"], ["ignored"]],
        [Status.INVALID_OUTPUT, Status.TRUNCATED, Status.REFUSAL],
    )
    assert result["pages"] == result["failed_pages"] == 3
    assert result["char_edits"] == result["reference_chars"] == 3
    assert result["word_edits"] == result["reference_words"] == 2
    assert (
        result["invalid_output_pages"] == result["truncated_pages"] == result["refusal_pages"] == 1
    )
    assert result["cer"] == result["wer"] == 1
