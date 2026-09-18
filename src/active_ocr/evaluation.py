"""Evaluation metrics, deliberately separate from sample selection."""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence

from rapidfuzz.distance import Levenshtein

from active_ocr.models import Box, PredictionStatus, Region

TEXT_EVALUATOR_ID = "page-text-nfc-v1"
# Engineering localization reporting, versioned separately from the unchanged text metrics.
LOCALIZATION_EVALUATOR_ID = "page-boxes-iou50-v1"
IOU_THRESHOLD = 0.5


def intersection_over_union(left: Box, right: Box) -> float:
    """Return axis-aligned bounding-box intersection over union."""

    x1 = max(left.x, right.x)
    y1 = max(left.y, right.y)
    x2 = min(left.x + left.width, right.x + right.width)
    y2 = min(left.y + left.height, right.y + right.height)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = left.width * left.height + right.width * right.height - intersection
    return intersection / union if union else 0.0


def character_error_rate(reference: str, prediction: str) -> float:
    """Calculate Unicode code-point CER with Levenshtein distance."""

    if not reference:
        return 0.0 if not prediction else 1.0
    return edit_count(reference, prediction) / len(reference)


def area_under_learning_curve(points: Sequence[tuple[int, float]]) -> float:
    """Integrate performance by labelled-page count using trapezoids."""

    ordered = sorted(points)
    if len({x for x, _ in ordered}) != len(ordered):
        raise ValueError("learning-curve x values must be unique")
    return sum(
        (right_x - left_x) * (left_y + right_y) / 2
        for (left_x, left_y), (right_x, right_y) in zip(ordered, ordered[1:], strict=False)
    )


def edit_count(reference: Sequence[str], prediction: Sequence[str]) -> int:
    """Raw Levenshtein edits for code points or tokens, including empty references."""
    return Levenshtein.distance(reference, prediction)


def page_text_view(lines: Sequence[str]) -> str:
    """page-text-nfc-v1: preserve declared order/blank lines; normalize a copy only."""
    return unicodedata.normalize("NFC", "\n".join(lines).replace("\r\n", "\n"))


def page_text_metrics(
    references: Sequence[Sequence[str]],
    hypotheses: Sequence[Sequence[str]],
    statuses: Sequence[PredictionStatus],
) -> dict[str, int | float]:
    """Corpus text counts and independently defined rates; no layout or test policy."""
    if not references or not (len(references) == len(hypotheses) == len(statuses)):
        raise ValueError("text evaluation requires nonempty, matching page coverage")
    counts = dict.fromkeys(
        (
            "char_edits",
            "reference_chars",
            "word_edits",
            "reference_words",
            "invalid_output_pages",
            "truncated_pages",
            "refusal_pages",
            "failed_pages",
        ),
        0,
    )
    failure_keys = {
        PredictionStatus.INVALID_OUTPUT: "invalid_output_pages",
        PredictionStatus.TRUNCATED: "truncated_pages",
        PredictionStatus.REFUSAL: "refusal_pages",
    }
    for reference_lines, hypothesis_lines, raw_status in zip(
        references, hypotheses, statuses, strict=True
    ):
        status = PredictionStatus(raw_status)
        reference = page_text_view(reference_lines)
        hypothesis = page_text_view(hypothesis_lines) if status is PredictionStatus.OK else ""
        if status is not PredictionStatus.OK:
            counts[failure_keys[status]] += 1
            counts["failed_pages"] += 1
        counts["char_edits"] += edit_count(reference, hypothesis)
        counts["reference_chars"] += len(reference)
        counts["word_edits"] += edit_count(reference.split(), hypothesis.split())
        counts["reference_words"] += len(reference.split())
    metrics: dict[str, int | float] = {
        "pages": len(references),
        **counts,
        "cer_defined": int(counts["reference_chars"] > 0),
        "wer_defined": int(counts["reference_words"] > 0),
    }
    if counts["reference_chars"]:
        metrics["cer"] = counts["char_edits"] / counts["reference_chars"]
    if counts["reference_words"]:
        metrics["wer"] = counts["word_edits"] / counts["reference_words"]
    return metrics


def match_boxes(
    references: Sequence[Box], predictions: Sequence[Box], threshold: float = IOU_THRESHOLD
) -> tuple[tuple[int, int], ...]:
    """Greedy one-to-one matching in descending IoU; ties break by reference then prediction."""
    candidates = sorted(
        (
            (-intersection_over_union(reference, prediction), i, j)
            for i, reference in enumerate(references)
            for j, prediction in enumerate(predictions)
            if intersection_over_union(reference, prediction) >= threshold
        )
    )
    matched_references: set[int] = set()
    matched_predictions: set[int] = set()
    pairs = []
    for _, i, j in candidates:
        if i in matched_references or j in matched_predictions:
            continue
        matched_references.add(i)
        matched_predictions.add(j)
        pairs.append((i, j))
    return tuple(pairs)


def page_localization_metrics(
    references: Sequence[Sequence[Region]],
    hypotheses: Sequence[Sequence[Region]],
    statuses: Sequence[PredictionStatus],
) -> dict[str, int | float]:
    """page-boxes-iou50-v1 counts. Unmatched predictions and references are both errors."""
    if not references or not (len(references) == len(hypotheses) == len(statuses)):
        raise ValueError("localization evaluation requires nonempty, matching page coverage")
    counts = dict.fromkeys(
        (
            "box_reference_boxes",
            "box_predicted_boxes",
            "box_matched_boxes",
            "matched_line_char_edits",
            "matched_line_reference_chars",
        ),
        0,
    )
    for reference_regions, hypothesis_regions, raw_status in zip(
        references, hypotheses, statuses, strict=True
    ):
        # A failed page contributes zero predictions, so its reference lines stay unmatched.
        ok = PredictionStatus(raw_status) is PredictionStatus.OK
        predicted = tuple(hypothesis_regions) if ok else ()
        counts["box_reference_boxes"] += len(reference_regions)
        counts["box_predicted_boxes"] += len(predicted)
        for i, j in match_boxes([r.box for r in reference_regions], [p.box for p in predicted]):
            expected = page_text_view([reference_regions[i].text])
            actual = page_text_view([predicted[j].text])
            counts["box_matched_boxes"] += 1
            counts["matched_line_char_edits"] += edit_count(expected, actual)
            counts["matched_line_reference_chars"] += len(expected)
    matched = counts["box_matched_boxes"]
    metrics: dict[str, int | float] = {
        "localization_pages": len(references),
        **counts,
        "box_precision_defined": int(counts["box_predicted_boxes"] > 0),
        "box_recall_defined": int(counts["box_reference_boxes"] > 0),
    }
    precision = matched / counts["box_predicted_boxes"] if counts["box_predicted_boxes"] else None
    recall = matched / counts["box_reference_boxes"] if counts["box_reference_boxes"] else None
    if precision is not None:
        metrics["box_precision"] = precision
    if recall is not None:
        metrics["box_recall"] = recall
        # Coverage states how much of the page was matched at all, so a high matched-line
        # CER cannot be read as page quality when most reference lines were never found.
        metrics["matched_line_coverage"] = recall
    if precision is not None and recall is not None and precision + recall:
        metrics["box_f1"] = 2 * precision * recall / (precision + recall)
    elif counts["box_reference_boxes"] or counts["box_predicted_boxes"]:
        metrics["box_f1"] = 0.0
    if counts["matched_line_reference_chars"]:
        metrics["matched_line_cer"] = (
            counts["matched_line_char_edits"] / counts["matched_line_reference_chars"]
        )
    return metrics
