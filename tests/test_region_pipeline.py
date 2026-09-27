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
        _check_records(
            [
                blind
                | {"regions": [blind["regions"][0] | {"target": "secret"}, blind["regions"][1]]}
            ],
            labelled=False,
        )
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
        "stage": 0,
        "model_repository": "qwen",
        "model_revision": "pinned",
        "predictions": [
            {
                "page_id": page_id,
                "regions": [
                    {"id": "line-b", "text": "second", "truncated": False},
                    {"id": "line-a", "text": "incomplete", "truncated": True},
                ],
            }
        ],
    }

    class Oracle:
        def evaluate_validation(self, predictions, evaluator, ids):
            assert ids == (page_id,)
            assert [r.id for r in predictions[0].regions] == ["line-b", "line-a"]
            assert [r.text for r in predictions[0].regions] == ["second", ""]
            return {"cer": 0.5, "wer": 1.0}

    result = _metrics(
        receipt,
        {page_id: SimpleNamespace(id=page_id)},
        {page_id: geometry},
        Oracle(),
        [page_id],
        "exp",
    )
    assert (result["regions"], result["truncated_regions"]) == (2, 1)
    receipt["predictions"][0]["regions"].reverse()
    with pytest.raises(ValueError, match="order"):
        _metrics(
            receipt,
            {page_id: SimpleNamespace(id=page_id)},
            {page_id: geometry},
            Oracle(),
            [page_id],
            "exp",
        )


def test_baseline_reuse_checks_identity_and_keeps_original_receipt(tmp_path):
    import json

    from active_ocr.recognition_pipeline import sha_json
    from active_ocr.region_pipeline import _baseline_receipt

    original = {
        "config": {
            "selection": {"pages_per_round": 64},
            "prompt": "text",
            "models": {"qwen": {"repository": "qwen", "revision": "pinned"}},
        },
        "source_run": "source",
        "input_bundle": "bundle",
        "modal_image": "image",
        "validation_ids": ["val"],
    }
    receipt = {
        "stage": 0,
        "selected_ids": [],
        "adapter_sha256": None,
        "source_run": "source",
        "recipe_sha256": sha_json(original),
        "model_repository": "qwen",
        "model_revision": "pinned",
        "validation_ids": ["val"],
        "gpu": "L4",
    }
    (tmp_path / "recipe.json").write_text(json.dumps(original))
    (tmp_path / "round-0.json").write_text(json.dumps({"receipt": receipt, "metrics": {}}))
    current = {
        **original,
        "config": {
            **original["config"],
            "selection": {"pages_per_round": 24, "initial_batch": "strategy"},
        },
    }
    saved = _baseline_receipt(tmp_path, current)
    assert saved["receipt"] == receipt
    assert saved["reused_from"]["recipe_sha256"] == sha_json(original)
    current["config"]["prompt"] = "changed"
    with pytest.raises(ValueError, match="baseline"):
        _baseline_receipt(tmp_path, current)


def test_score_cache_checks_coverage_sample_ids_and_finite_scores(tmp_path):
    from active_ocr.region_pipeline import _check_scores, _save

    records = [{"page_id": "train", "regions": [{"id": str(i)} for i in range(4)]}]
    row = {
        "page_id": "train",
        "sampled_region_ids": ["1", "2", "3"],
        "entropy": 0.5,
        "confidence": 0.2,
        "truncated_regions": 0,
    }
    _check_scores([row], records, "uncertainty")
    path = tmp_path / "cache.json"
    assert _save(path, row) == _save(path, row)
    with pytest.raises(ValueError, match="differs"):
        _save(path, {**row, "confidence": 0.3})
    with pytest.raises(ValueError, match="score"):
        _check_scores([{**row, "entropy": float("nan")}], records, "uncertainty")
    with pytest.raises(ValueError, match="score"):
        _check_scores([{**row, "sampled_region_ids": ["0"]}], records, "uncertainty")
