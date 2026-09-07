"""Evaluation metrics, deliberately separate from sample selection."""

from __future__ import annotations

from collections.abc import Sequence

from active_ocr.models import Annotation, Box


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
    return previous[-1] / len(reference)


def total_annotation_seconds(annotations: Sequence[Annotation]) -> float:
    """Sum available annotation durations."""

    return sum(annotation.seconds or 0 for annotation in annotations)


def area_under_learning_curve(points: Sequence[tuple[int, float]]) -> float:
    """Integrate performance by labelled-page count using trapezoids."""

    ordered = sorted(points)
    if len({x for x, _ in ordered}) != len(ordered):
        raise ValueError("learning-curve x values must be unique")
    return sum(
        (right_x - left_x) * (left_y + right_y) / 2
        for (left_x, left_y), (right_x, right_y) in zip(ordered, ordered[1:], strict=False)
    )
