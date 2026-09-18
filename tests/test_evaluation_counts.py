"""Hand-computed engineering text-view examples, not benchmark results."""

import pytest

from active_ocr.evaluation import (
    character_error_rate,
    edit_count,
    match_boxes,
    page_localization_metrics,
    page_text_metrics,
    page_text_view,
)
from active_ocr.models import Box, Region
from active_ocr.models import PredictionStatus as Status


def region(identifier, x, y, text="line", size=10):
    return Region(id=identifier, text=text, box=Box(x=x, y=y, width=size, height=size))


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


def test_greedy_matching_is_one_to_one_with_stable_index_ties():
    references = [Box(x=0, y=0, width=10, height=10), Box(x=100, y=0, width=10, height=10)]
    # Two identical predictions can only consume one reference each.
    duplicates = [Box(x=0, y=0, width=10, height=10)] * 2
    assert match_boxes(references, duplicates) == ((0, 0),)
    # Equal IoU resolves by reference index, then prediction index.
    tied = [Box(x=0, y=0, width=10, height=10), Box(x=0, y=0, width=10, height=10)]
    assert match_boxes([Box(x=0, y=0, width=10, height=10)], tied) == ((0, 0),)
    assert match_boxes(references, list(reversed(references))) == ((0, 1), (1, 0))
    # A 0.5 IoU is inside the threshold; anything below is unmatched.
    unit = [Box(x=0, y=0, width=10, height=10)]
    assert match_boxes(unit, [Box(x=0, y=0, width=10, height=20)]) == ((0, 0),)
    assert match_boxes(unit, [Box(x=9, y=0, width=10, height=10)]) == ()


def test_localization_counts_perfect_disjoint_and_duplicate_predictions():
    references = [(region("r1", 0, 0, "alpha"), region("r2", 100, 0, "beta"))]
    perfect = page_localization_metrics(references, references, [Status.OK])
    assert perfect["box_matched_boxes"] == 2
    assert perfect["box_precision"] == perfect["box_recall"] == perfect["box_f1"] == 1
    assert perfect["matched_line_coverage"] == 1 and perfect["matched_line_cer"] == 0
    disjoint = page_localization_metrics(
        references, [(region("p1", 500, 500), region("p2", 700, 700))], [Status.OK]
    )
    # Unmatched predictions and unmatched references are both errors.
    assert disjoint["box_matched_boxes"] == 0
    assert disjoint["box_predicted_boxes"] == disjoint["box_reference_boxes"] == 2
    assert disjoint["box_precision"] == disjoint["box_recall"] == disjoint["box_f1"] == 0
    assert "matched_line_cer" not in disjoint and disjoint["matched_line_coverage"] == 0
    duplicated = page_localization_metrics(
        references,
        [(region("p1", 0, 0, "alpha"), region("p2", 0, 0, "alpha"), region("p3", 0, 0, "alpha"))],
        [Status.OK],
    )
    assert duplicated["box_matched_boxes"] == 1 and duplicated["box_predicted_boxes"] == 3
    assert duplicated["box_precision"] == 1 / 3 and duplicated["box_recall"] == 0.5
    assert duplicated["box_f1"] == pytest.approx(0.4)


def test_localization_penalizes_failed_pages_and_reports_coverage_with_matched_cer():
    references = [(region("r1", 0, 0, "alpha"), region("r2", 100, 0, "beta"))]
    failed = page_localization_metrics(
        references, [(region("p1", 0, 0, "alpha"),)], [Status.TRUNCATED]
    )
    # A failed page contributes no accepted boxes, so its reference lines stay unmatched.
    assert failed["box_predicted_boxes"] == 0 and failed["box_matched_boxes"] == 0
    assert failed["box_precision_defined"] == 0 and "box_precision" not in failed
    assert failed["box_recall"] == 0
    partial = page_localization_metrics(
        references, [(region("p1", 0, 0, "alpho"),)], [Status.OK]
    )
    # Half the lines were found and the matched one has one wrong character.
    assert partial["matched_line_coverage"] == 0.5
    assert partial["matched_line_char_edits"] == 1
    assert partial["matched_line_reference_chars"] == 5
    assert partial["matched_line_cer"] == 0.2
    with pytest.raises(ValueError):
        page_localization_metrics([], [], [])
