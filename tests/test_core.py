import pytest

from active_ocr.active_learning import select_pages
from active_ocr.evaluation import (
    area_under_learning_curve,
    character_error_rate,
    intersection_over_union,
)
from active_ocr.models import Box, Prediction, Strategy


def test_box_validation_and_iou() -> None:
    with pytest.raises(ValueError):
        Box(x=0, y=0, width=0, height=10)
    left = Box(x=0, y=0, width=10, height=10)
    right = Box(x=5, y=5, width=10, height=10)
    assert intersection_over_union(left, right) == pytest.approx(25 / 175)


def test_random_selection_is_reproducible_and_excludes_pages() -> None:
    first = select_pages(["a", "b", "c", "d"], Strategy.RANDOM, 2, 42, excluded={"b"})
    second = select_pages(["d", "c", "b", "a"], Strategy.RANDOM, 2, 42, excluded={"b"})
    assert first == second
    assert "b" not in first


def test_uncertainty_strategies_rank_highest_first() -> None:
    predictions = [
        Prediction(
            page_id="a",
            experiment_id="exp",
            round_number=1,
            model_id="m",
            confidence=0.9,
            entropy=0.1,
        ),
        Prediction(
            page_id="b",
            experiment_id="exp",
            round_number=1,
            model_id="m",
            confidence=0.3,
            entropy=0.8,
        ),
    ]
    assert select_pages(["a", "b"], Strategy.LEAST_CONFIDENCE, 2, 1, predictions=predictions) == (
        "b",
        "a",
    )
    assert select_pages(["a", "b"], Strategy.ENTROPY, 2, 1, predictions=predictions) == (
        "b",
        "a",
    )


def test_character_error_rate() -> None:
    assert character_error_rate("text", "text") == 0
    assert character_error_rate("cat", "cut") == pytest.approx(1 / 3)


def test_area_under_learning_curve() -> None:
    assert area_under_learning_curve([(0, 0.2), (10, 0.6), (20, 0.8)]) == pytest.approx(11)
    with pytest.raises(ValueError, match="unique"):
        area_under_learning_curve([(10, 0.5), (10, 0.6)])
