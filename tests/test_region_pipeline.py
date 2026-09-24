"""Known PAGE boxes stay ordered and unlabelled outside selected TRAIN pages."""

from types import SimpleNamespace

import pytest
from PIL import Image

from active_ocr.region_pipeline import _check_records, _crop, _record


def test_known_boxes_crop_and_keep_source_order():
    image = Image.new("RGB", (100, 80), "white")
    assert _crop(image, {"x": 10, "y": 20, "width": 30, "height": 10}, 3).size == (36, 16)
    assert _crop(image, {"x": 0, "y": 0, "width": 5, "height": 5}, 3).size == (8, 8)
    page = SimpleNamespace(id="read:train:1", image_sha256="a" * 64, width=100, height=80)
    geometry = [
        {"id": "second", "box": {"x": 10, "y": 20, "width": 30, "height": 10}},
        {"id": "first", "box": {"x": 0, "y": 0, "width": 5, "height": 5}},
    ]
    blind = _record(page, geometry)
    assert [r["id"] for r in blind["regions"]] == ["second", "first"]
    _check_records([blind], labelled=False)
    with pytest.raises(ValueError, match="leakage"):
        _check_records([blind | {"regions": [blind["regions"][0] | {"target": "secret"}, blind["regions"][1]]}], labelled=False)
    labelled = _record(page, geometry, {"second": "B", "first": "A"})
    assert [r["target"] for r in labelled["regions"]] == ["B", "A"]
    _check_records([labelled], labelled=True)


def test_region_predictions_reconstruct_page_in_declared_order():
    from active_ocr.region_pipeline import _metrics

    page_id = "read:validation:1"
    geometry = [
        {"id": "line-b", "box": {"x": 10, "y": 20, "width": 30, "height": 10}},
        {"id": "line-a", "box": {"x": 0, "y": 0, "width": 5, "height": 5}},
    ]
    receipt = {
        "stage": 0, "model_repository": "qwen", "model_revision": "pinned",
        "predictions": [{"page_id": page_id, "regions": [
            {"id": "line-b", "text": "second", "truncated": False},
            {"id": "line-a", "text": "incomplete", "truncated": True},
        ]}],
    }

    class Oracle:
        def evaluate_validation(self, predictions, evaluator, ids):
            assert ids == (page_id,)
            assert [r.id for r in predictions[0].regions] == ["line-b", "line-a"]
            assert [r.text for r in predictions[0].regions] == ["second", ""]
            return {"cer": 0.5, "wer": 1.0}

    result = _metrics(receipt, {page_id: SimpleNamespace(id=page_id)}, {page_id: geometry}, Oracle(), [page_id], "exp")
    assert (result["regions"], result["truncated_regions"]) == (2, 1)
    receipt["predictions"][0]["regions"].reverse()
    with pytest.raises(ValueError, match="order"):
        _metrics(receipt, {page_id: SimpleNamespace(id=page_id)}, {page_id: geometry}, Oracle(), [page_id], "exp")
