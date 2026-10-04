"""The content controls alter demonstrations while preserving the query and labels."""

import json

import pytest
from PIL import Image

from active_ocr.icl_controls import raw_metrics, require_gate
from active_ocr.region_pipeline import _inputs


@pytest.mark.parametrize("mode", ["paired", "text-only", "mismatched"])
def test_control_changes_only_demo_images(mode):
    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["add_generation_prompt"] is True
            return messages

    box = {"x": 0, "y": 0, "width": 8, "height": 4}
    query = {"id": "query", "box": box}
    config = {"crop_padding_px": 0, "training": {"image_longest_edge": 100}, "prompt": "OCR"}
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    examples = [
        (Image.new("RGB", (8, 4), color), {"id": str(i), "box": box, "target": f"label-{i}"})
        for i, color in enumerate(colors)
    ]
    messages = _inputs(
        Processor(),
        Image.new("RGB", (8, 4), "white"),
        query,
        config,
        examples=examples,
        demonstration_mode=mode,
    )
    assert [m["content"][0]["text"] for m in messages[1:-1:2]] == ["label-0", "label-1", "label-2"]
    assert all(m["content"][-1] == {"type": "text", "text": "OCR"} for m in messages[::2])
    images = [
        item["image"].getpixel((0, 0))
        for m in messages
        for item in m["content"]
        if item["type"] == "image"
    ]
    expected = (
        colors if mode == "paired" else [] if mode == "text-only" else colors[1:] + colors[:1]
    )
    assert images == expected + [(255, 255, 255)]
    assert "target" not in query
    assert [example["target"] for _, example in examples] == ["label-0", "label-1", "label-2"]
    with pytest.raises(ValueError, match="mode"):
        _inputs(Processor(), examples[0][0], query, config, demonstration_mode="invalid")
    if mode != "paired":
        with pytest.raises(ValueError, match="mode"):
            _inputs(Processor(), examples[0][0], query, config, train=True, demonstration_mode=mode)


def test_raw_line_metric_retains_caps_and_prevents_cross_line_matching():
    truth = {"val": {"a": "ab", "b": "cd"}}
    receipt = {
        "predictions": [
            {
                "page_id": "val",
                "regions": [
                    {"id": "a", "text": "abcd", "truncated": True},
                    {"id": "b", "text": "", "truncated": False},
                ],
            }
        ]
    }
    result = raw_metrics(receipt, truth)
    assert result["lines"] == 2 and result["reference_chars"] == 4
    assert result["char_edits"] == 4 and result["cer"] == 1
    assert result["word_edits"] == 2 and result["wer"] == 1
    assert receipt["predictions"][0]["regions"][0]["text"] == "abcd"


def test_gates_reject_missing_changed_or_incomplete_evidence(tmp_path):
    path = tmp_path / "gate.json"
    with pytest.raises(ValueError, match="gate"):
        require_gate(path, "recipe", 56)
    path.write_text(json.dumps({"passed": True, "recipe_sha256": "recipe", "outputs": 56}))
    require_gate(path, "recipe", 56)
    for digest, outputs in [("other", 56), ("recipe", 55)]:
        with pytest.raises(ValueError, match="gate"):
            require_gate(path, digest, outputs)


def test_lighton_capacity_and_receipt_keep_model_identity():
    from types import SimpleNamespace

    import torch

    from active_ocr.icl import _receipt, check_input_capacity

    processor = SimpleNamespace(
        patch_size=14,
        spatial_merge_size=2,
        tokenizer=SimpleNamespace(model_max_length=16384),
    )
    model = SimpleNamespace(
        config=SimpleNamespace(
            image_token_id=99,
            text_config=SimpleNamespace(max_position_embeddings=16384),
        )
    )
    batch = {
        "input_ids": torch.tensor([[99, 99, 1, 99, 99, 2]]),
        "image_sizes": torch.tensor([[28, 56], [28, 56]]),
    }
    assert check_input_capacity(batch, processor, model, 2, 64) == 4
    with pytest.raises(ValueError, match="image count"):
        check_input_capacity(batch, processor, model, 1, 64)
    with pytest.raises(ValueError, match="lost"):
        check_input_capacity(
            batch | {"input_ids": torch.tensor([[99, 1]])}, processor, model, 2, 64
        )
    with pytest.raises(ValueError, match="token limit"):
        check_input_capacity(batch, processor, model, 2, 16384)
    config = {"models": {"lighton": {"repository": "lighton", "revision": "pinned"}}}
    receipt = _receipt(
        [{"page_id": "val", "id": "line"}],
        [{"page_id": "val", "regions": [{"id": "line"}]}],
        config,
        "source",
        "lighton",
    )
    assert (receipt["model_repository"], receipt["model_revision"]) == ("lighton", "pinned")
