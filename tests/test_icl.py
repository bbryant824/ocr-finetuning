"""Focused ICL checks: context sampling, input pairing and resumable evidence."""

import json

import pytest
from PIL import Image

from active_ocr.region_pipeline import _inputs


def test_icl_inputs_pair_each_crop_and_transcript_and_keep_zero_shot():
    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            return {"messages": messages, "kwargs": kwargs}

    processor = Processor()
    config = {"crop_padding_px": 0, "training": {"image_longest_edge": 100}, "prompt": "OCR"}
    query = {"id": "q", "box": {"x": 0, "y": 0, "width": 30, "height": 10}}
    page = Image.new("RGB", (100, 80), "white")
    demo = {**query, "id": "demo", "target": "correct training label"}
    demo_page = Image.new("RGB", (100, 80), "black")
    original = _inputs(processor, page, query, config)
    assert _inputs(processor, page, query, config, examples=[]) == original
    result = _inputs(processor, page, query, config, examples=[(demo_page, demo)])
    assert [m["role"] for m in result["messages"]] == ["user", "assistant", "user"]
    assert result["messages"][1]["content"] == [{"type": "text", "text": "correct training label"}]
    assert all("correct training label" not in str(m["content"]) for m in result["messages"][::2])
    assert result["messages"][-1]["content"][0]["image"].size == (30, 10)
    assert result["messages"][0]["content"][0]["image"].getpixel((0, 0)) == (0, 0, 0)
    assert result["messages"][-1]["content"][0]["image"].getpixel((0, 0)) == (255, 255, 255)
    assert result["kwargs"]["add_generation_prompt"] is True
    with pytest.raises(ValueError, match="training"):
        _inputs(processor, page, query, config, train=True, examples=[(demo_page, demo)])


def test_random_contexts_are_distinct_nested_replayable_and_train_only():
    from active_ocr.icl import sample_context

    bank = [f"read:train:p/line-{i}" for i in range(30)]
    full = sample_context(bank, "val/q", 24, 824, "bank-hash")
    assert len(full) == len(set(full)) == 24
    assert sample_context(bank, "val/q", 8, 824, "bank-hash") == full[:8]
    assert sample_context(list(reversed(bank)), "val/q", 24, 824, "bank-hash") == full
    assert sample_context(bank, "val/other", 24, 824, "bank-hash") != full
    assert sample_context(bank, "val/q", 0, 824, "bank-hash") == []
    with pytest.raises(ValueError, match="bank"):
        sample_context(bank + [bank[0]], "val/q", 24, 824, "bank-hash")
    with pytest.raises(ValueError, match="TRAIN"):
        sample_context(bank + ["read:validation:p/leaked"], "val/q", 24, 824, "bank-hash")


def test_cached_predictions_reject_changed_identity_or_corruption(tmp_path):
    from active_ocr.icl import read_cache
    from active_ocr.recognition_pipeline import sha_json

    path = tmp_path / "query.json"
    row = {"text": "saved", "demo_ids": ["train/demo"]}
    assert read_cache(path, "matching") is None
    value = {"identity": "matching", "row": row, "row_sha256": sha_json(row)}
    path.write_text(json.dumps(value))
    assert read_cache(path, "matching") == row
    with pytest.raises(ValueError, match="identity"):
        read_cache(path, "changed")
    value["row"]["text"] = "corrupted"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="checksum"):
        read_cache(path, "matching")


def test_reused_fit_follows_and_hash_checks_its_original_receipt(tmp_path):
    import hashlib

    from active_ocr.icl import reuse_reference
    from active_ocr.recognition_pipeline import sha_json

    recipe = {
        "config": {
            "selection": {"seed": 824},
            "models": {"qwen": {"repository": "qwen", "revision": "pinned"}},
        },
        "source_run": "source",
        "strategy": "random",
        "input_bundle": "bundle",
        "modal_image": "image",
        "validation_ids": ["val"],
    }
    receipt = {
        "stage": 1,
        "selected_ids": ["read:train:p"],
        "adapter_sha256": "weights",
        "source_run": "source",
        "recipe_sha256": sha_json(recipe),
        "validation_ids": ["val"],
        "model_repository": "qwen",
        "model_revision": "pinned",
        "gpu": "L4",
    }
    original, copied = tmp_path / "original", tmp_path / "copied"
    original.mkdir()
    copied.mkdir()
    path = original / "round-1.json"
    saved = {"receipt": receipt, "metrics": {"cer": 0.5}}
    path.write_text(json.dumps(saved))
    (original / "recipe.json").write_text(json.dumps(recipe))
    (copied / "recipe.json").write_text(json.dumps(recipe | {"runner_sha256": "new"}))
    (copied / "round-1.json").write_text(
        json.dumps(
            saved
            | {
                "reused_from": {
                    "path": str(path),
                    "receipt_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "recipe_sha256": sha_json(recipe),
                }
            }
        )
    )
    result = reuse_reference(copied, recipe, 1)
    assert result["receipt"] == receipt
    assert result["reused_from"]["path"] == str(path)
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="provenance"):
        reuse_reference(copied, recipe, 1)


def test_full_evaluation_requires_a_matching_complete_readiness(tmp_path):
    from active_ocr.icl import require_readiness

    path = tmp_path / "readiness.json"
    with pytest.raises(ValueError, match="readiness"):
        require_readiness(path, "recipe")
    value = {"passed": True, "recipe_sha256": "recipe", "outputs": 24}
    path.write_text(json.dumps(value))
    require_readiness(path, "recipe")
    with pytest.raises(ValueError, match="readiness"):
        require_readiness(path, "different")
    value["outputs"] = 23
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="readiness"):
        require_readiness(path, "recipe")


@pytest.mark.parametrize("flags", [[], ["--icl-stage", "check", "--prepare-only"]])
def test_icl_cli_requires_an_explicit_unambiguous_stage(tmp_path, flags):
    import os
    import subprocess
    import sys

    config = tmp_path / "config.json"
    config.write_text(json.dumps({"icl": {}}))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "active_ocr.region_pipeline",
            str(tmp_path / "source"),
            str(tmp_path / "output"),
            "--config",
            str(config),
            "--strategy",
            "random",
            *flags,
        ],
        env={**os.environ, "PYTHONPATH": "src"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2 and "ICL" in result.stderr
    assert not (tmp_path / "output").exists()
