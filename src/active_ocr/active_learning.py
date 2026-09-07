"""Active-learning query strategies, kept independent for future research work."""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence

from active_ocr.models import Prediction, Strategy


def select_pages(
    page_ids: Iterable[str],
    strategy: Strategy,
    batch_size: int,
    seed: int,
    *,
    excluded: Iterable[str] = (),
    predictions: Sequence[Prediction] = (),
) -> tuple[str, ...]:
    """Select unique pages using a deterministic strategy.

    The initial implementation uses page-level confidence and normalized entropy.
    This function is the intended extension point for future active-learning
    algorithms; pipeline and integration code should not contain query logic.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    candidates = sorted(set(page_ids) - set(excluded))
    if strategy is Strategy.RANDOM:
        random.Random(seed).shuffle(candidates)
        return tuple(candidates[:batch_size])

    by_page = {prediction.page_id: prediction for prediction in predictions}
    if len(by_page) != len(predictions):
        raise ValueError("only one prediction per page is allowed")

    def uncertainty(page_id: str) -> float:
        prediction = by_page.get(page_id)
        if prediction is None:
            raise ValueError(f"missing prediction for page {page_id}")
        if strategy is Strategy.LEAST_CONFIDENCE:
            if prediction.confidence is None:
                raise ValueError(f"missing confidence for page {page_id}")
            return 1 - prediction.confidence
        if strategy is Strategy.ENTROPY:
            if prediction.entropy is None:
                raise ValueError(f"missing entropy for page {page_id}")
            return prediction.entropy
        raise ValueError(f"unsupported strategy: {strategy}")

    ranked = sorted(candidates, key=lambda page_id: (-uncertainty(page_id), page_id))
    return tuple(ranked[:batch_size])
