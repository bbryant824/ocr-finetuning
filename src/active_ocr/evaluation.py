"""Evaluation metrics, deliberately separate from sample selection."""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence

from active_ocr.models import Box, PredictionStatus


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
    previous = list(range(len(prediction) + 1))
    for row, expected in enumerate(reference, start=1):
        current = [row]
        for column, actual in enumerate(prediction, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (expected != actual),
                )
            )
        previous = current
    return previous[-1]


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
