"""Boundary checks for the LLaMA-Factory adapter. No OCR quality claim lives here.

Most cases are dependency free. `ACTIVE_OCR_PROCESSOR_DIR` opts into the pinned small-asset
`test_actual_*` checks that exercise the real toolkit template and tokenizer on CPU; no 4B
weights are ever loaded. Those cases are the image build gate, so they must never skip there.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
from pathlib import Path

import pytest

from active_ocr.integrations import artifacts as a
from active_ocr.integrations import factory_model as f
from active_ocr.models import (
    Box,
    ExpectedIdentity,
    RealOCRConfig,
    RevealedExample,
    SimulationPage,
    SourceRegion,
    Split,
)

PAGE_WIDTH, PAGE_HEIGHT = 320, 480


def real_config() -> RealOCRConfig:
    """Self-contained recipe fixture: only this file is copied into the build image."""
    identity = ExpectedIdentity(
        source_sha="c" * 40,
        dependency_sha256="d" * 64,
        model_revision=f.REVISION,
        model_manifest_sha256=a.digest(f.asset_manifest(f.BASE_FILES)),
        processor_revision=f.REVISION,
        processor_manifest_sha256=a.digest(f.asset_manifest(f.PROCESSOR_FILES)),
        recipe_version=f.recipe_field("recipe_version"),
        evaluator_id=f.recipe_field("evaluator_id"),
        code_bundle_sha256="e" * 64,
        remote_dependency_sha256="f" * 64,
    )
    return RealOCRConfig(
        backend=f.BACKEND,
        recipe_version=identity.recipe_version,
        model_repository=f.REPOSITORY,
        model_revision=f.REVISION,
        processor_repository=f.REPOSITORY,
        processor_revision=f.REVISION,
        training_policy_id=f.recipe_field("training_policy_id"),
        decode_policy_id=f.recipe_field("decode_policy_id"),
        evaluator_id=identity.evaluator_id,
        expected_identity=identity,
    )


def page(tmp_path: Path, split: Split = Split.TRAIN, name: str = "page-1") -> SimulationPage:
    image = tmp_path / f"{name}.png"
    if not image.exists():
        from PIL import Image

        Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), "white").save(image)
    data = image.read_bytes()
    return SimulationPage(
        id=name,
        document_id="document-1",
        image_uri=str(image),
        source_image=str(image),
        width=PAGE_WIDTH,
        height=PAGE_HEIGHT,
        split=split,
        image_sha256=hashlib.sha256(data).hexdigest(),
    )


def example(tmp_path: Path, *texts: str, split: Split = Split.TRAIN) -> RevealedExample:
    regions = tuple(
        SourceRegion(
            id=f"line-{i}",
            text=text,
            box=Box(x=0, y=20 * i, width=PAGE_WIDTH, height=18),
        )
        for i, text in enumerate(texts or ("a line",))
    )
    return RevealedExample(page=page(tmp_path, split), regions=regions)


def safetensors_bytes(tensors: dict[str, bytes]) -> bytes:
    """Minimal valid safetensors framing, so adapter_summary parses real bytes."""
    header, offset = {}, 0
    for name in sorted(tensors):
        body = tensors[name]
        header[name] = {
            "dtype": "F32",
            "shape": [len(body) // 4],
            "data_offsets": [offset, offset + len(body)],
        }
        offset += len(body)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    return (
        struct.pack("<Q", len(encoded))
        + encoded
        + b"".join(tensors[name] for name in sorted(tensors))
    )


# ------------------------------------------------------------------ the recipe


def test_recipe_is_the_single_validated_source_of_settings():
    recipe = f.recipe()
    assert recipe["toolkit"]["version"] == "0.9.5"
    assert recipe["toolkit"]["template"] == "qwen3_vl_nothink" == f.TEMPLATE
    assert recipe["toolkit"]["infer_backend"] == "huggingface"
    assert f.REVISION == "ebb281ec70b05090aa6165b016eac8ec08e71b17"
    assert (f.MAX_PROMPT_TOKENS, f.MAX_OUTPUT_TOKENS, f.MAX_SEQUENCE_TOKENS) == (3072, 4096, 7168)
    assert f.MAX_PROMPT_TOKENS + f.MAX_OUTPUT_TOKENS == f.MAX_SEQUENCE_TOKENS
    assert set(f.BASE_FILES) | set(f.PROCESSOR_FILES) == set(f.ASSETS)
    training = recipe["training"]
    assert (training["lora_rank"], training["lora_alpha"], training["lora_dropout"]) == (8, 16, 0.0)
    assert training["lora_target"] == "all"
    assert training["freeze_vision_tower"] is True
    assert training["packing"] is False and training["val_size"] == 0.0
    assert recipe["decode"]["do_sample"] is False


def test_rendered_yaml_matches_the_recipe_and_overrides_the_demo_cutoff(tmp_path):
    config = f.training_yaml(tmp_path / "model", tmp_path / "data", tmp_path / "out", 824)
    training = f.recipe_field("training")
    # The upstream demo truncates at 2,048 tokens; silently losing a target is unacceptable.
    assert config["cutoff_len"] == f.MAX_SEQUENCE_TOKENS == 7168
    assert config["template"] == f.TEMPLATE
    assert config["dataset"] == "selected_train"
    assert config["dataset_dir"] == str(tmp_path / "data")
    assert config["overwrite_output_dir"] is False
    assert config["packing"] is False and config["val_size"] == 0.0
    assert config["seed"] == config["data_seed"] == 824
    for key in ("lora_rank", "lora_alpha", "lora_dropout", "lora_target", "freeze_vision_tower"):
        assert config[key] == training[key]
    assert config["learning_rate"] == 1e-4 and config["lr_scheduler_type"] == "cosine"
    assert config["gradient_accumulation_steps"] == 4
    assert config["per_device_train_batch_size"] == 1
    assert config["gradient_checkpointing"] is True and config["bf16"] is True
    # No upstream demo dataset may enter training.
    assert "mllm_demo" not in config["dataset"] and "alpaca" not in config["dataset"]


def test_binding_is_pure_and_carries_the_whole_recipe():
    real = real_config()
    assert f.binding(real) == f.binding(real)
    assert f.binding(real)["settings"] == f.recipe()
    assert f.binding(real)["recipe"] == real.model_dump(mode="json")


# ----------------------------------------------------------- dataset boundary


def test_conversation_rows_carry_one_image_and_only_revealed_targets(tmp_path):
    revealed = example(tmp_path, "first line", "second line")
    target = f.serialize_target(revealed)
    row = f.conversation_row(Path(revealed.page.image_uri), target)
    assert [c["from"] for c in row["conversations"]] == ["human", "gpt"]
    assert row["conversations"][0]["value"].count("<image>") == len(row["images"]) == 1
    assert row["conversations"][0]["value"] == "<image>" + f.PROMPT
    assert row["conversations"][1]["value"] == target
    # An inference row never carries a response.
    assert [c["from"] for c in f.conversation_row(Path("/x.png"), None)["conversations"]] == [
        "human"
    ]


def test_dataset_export_is_self_contained_and_excludes_demo_data(tmp_path):
    revealed = example(tmp_path, "line")
    rows = [f.conversation_row(Path(revealed.page.image_uri), f.serialize_target(revealed))]
    written = f.write_dataset(tmp_path / "data", rows)
    assert written == tmp_path / "data/train.jsonl"
    assert [json.loads(line) for line in written.read_text().splitlines()] == rows
    info = json.loads((tmp_path / "data/dataset_info.json").read_text())
    assert set(info) == {"selected_train"}
    assert info["selected_train"]["file_name"] == "train.jsonl"
    assert info["selected_train"]["formatting"] == "sharegpt"
    assert info["selected_train"]["columns"] == {"messages": "conversations", "images": "images"}
    assert info["selected_train"]["tags"]["assistant_tag"] == "gpt"
    with pytest.raises(ValueError, match="at least one"):
        f.write_dataset(tmp_path / "empty", [])


@pytest.mark.parametrize("text", [' ä\n"\\  ', "", "<|im_end|><|image_pad|>", "a < b"])
def test_source_box_and_text_round_trip_through_the_target_format(tmp_path, text):
    revealed = example(tmp_path, text)
    target = f.serialize_target(revealed)
    # Escaping `<` keeps the toolkit's <image> placeholder out of the supervised response.
    assert "<" not in target
    regions = f.parse_regions(target, PAGE_WIDTH, PAGE_HEIGHT)
    assert len(regions) == 1
    assert regions[0].text == text
    original = revealed.regions[0].box
    for observed, expected in (
        (regions[0].box.x, original.x),
        (regions[0].box.y, original.y),
        (regions[0].box.width, original.width),
        (regions[0].box.height, original.height),
    ):
        assert observed == pytest.approx(expected, abs=0.5)


def test_validation_targets_are_never_serialized(tmp_path):
    with pytest.raises(ValueError, match="TRAIN"):
        f.serialize_target(example(tmp_path, "x", split=Split.VALIDATION))


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "not json",
        '{"regions":[],"extra":1}',
        '{"regions":{}}',
        '{"regions":[{"text":"a"}]}',
        '{"regions":[{"text":1,"bbox":[0,0,1,1]}]}',
        '{"regions":[{"text":"a","bbox":[0,0,0,1]}]}',
        '{"regions":[{"text":"a","bbox":[0,0,1001,1]}]}',
        '{"regions":[{"text":"a","bbox":[0,0,1]}]}',
        '{"regions":[{"text":"a","bbox":[0,0,1,1],"extra":2}]}',
    ],
)
def test_malformed_output_is_rejected_rather_than_repaired(raw):
    with pytest.raises(ValueError):
        f.parse_regions(raw, PAGE_WIDTH, PAGE_HEIGHT)


# ------------------------------------------------------- checkpoints and probes


def training_record(ids, **overrides):
    record = {
        "toolkit_version": "0.9.5",
        "examples": len(ids),
        "epochs": 3.0,
        "optimizer_steps": 3,
        "losses": [1.5, 1.4, 1.3],
        "train_runtime_seconds": 12.5,
        "preflight": {
            i: {"prompt_tokens": 800, "target_tokens": 900, "total_tokens": 1700} for i in ids
        },
        "changed_lora_B_tensors": 2,
    }
    record.update(overrides)
    return record


def adapter_checkpoint(ids=("page-1",), **overrides):
    fields = {
        "kind": "adapter",
        "binding": f.binding(real_config()),
        "experiment_id": "e" * 32,
        "round_number": 1,
        "selected": tuple((i, "a" * 64) for i in ids),
        "target_sha256": "b" * 64,
        "seed": 824,
        "training": training_record(ids),
    }
    fields.update(overrides)
    return f.Checkpoint(**fields)


def test_training_record_accepts_the_recorded_upstream_summary():
    f.check_training_record(adapter_checkpoint())


@pytest.mark.parametrize(
    "overrides",
    [
        {"toolkit_version": "0.9.4"},
        {"examples": 2},
        {"epochs": 1.0},
        {"optimizer_steps": 0},
        {"losses": []},
        {"losses": [-1.0]},
        {"changed_lora_B_tensors": 0},
        {"preflight": {"other-page": {"prompt_tokens": 1, "target_tokens": 1, "total_tokens": 2}}},
        {"preflight": {"page-1": {"prompt_tokens": 1, "target_tokens": 1}}},
    ],
)
def test_training_record_rejects_inconsistent_summaries(overrides):
    record = training_record(("page-1",), **overrides)
    with pytest.raises(ValueError):
        f.check_training_record(adapter_checkpoint(training=record))


@pytest.mark.parametrize(
    "preflight",
    [
        {"prompt_tokens": 3073, "target_tokens": 10, "total_tokens": 3083},
        {"prompt_tokens": 10, "target_tokens": 4097, "total_tokens": 4107},
        {"prompt_tokens": 3072, "target_tokens": 4096, "total_tokens": 7168 + 1},
        {"prompt_tokens": 10, "target_tokens": 10, "total_tokens": 21},
    ],
)
def test_training_record_rejects_overlong_or_inconsistent_preflight(preflight):
    record = training_record(("page-1",), preflight={"page-1": preflight})
    with pytest.raises(ValueError):
        f.check_training_record(adapter_checkpoint(training=record))


def test_adapter_summary_requires_learned_B_not_random_A(tmp_path):
    from active_ocr.integrations import modal_model as m

    path = tmp_path / "adapter.safetensors"
    tensors = {
        "model.lora_A.weight": struct.pack("<f", 0.5),
        "model.lora_B.weight": struct.pack("<f", 0.0),
    }
    path.write_bytes(safetensors_bytes(tensors))
    assert f.adapter_summary(path)["nonzero_tensors"] == 1
    with pytest.raises(ValueError, match="zero"):
        m.verify_adapter_file(path)
    tensors["model.lora_B.weight"] = struct.pack("<f", 0.01)
    path.write_bytes(safetensors_bytes(tensors))
    assert m.verify_adapter_file(path)["changed_lora_B_tensors"] == 1
    tensors["model.lora_B.weight"] = struct.pack("<f", float("nan"))
    path.write_bytes(safetensors_bytes(tensors))
    with pytest.raises(ValueError, match="nonfinite"):
        m.verify_adapter_file(path)


@pytest.mark.parametrize("damage", ["truncated", "short", "trailing"])
def test_adapter_summary_rejects_damaged_files(tmp_path, damage):
    raw = safetensors_bytes({"lora_A": b"\x01\x00\x00\x00"})
    path = tmp_path / "adapter.safetensors"
    path.write_bytes(
        {"truncated": raw[:-2], "short": b"\x00" * 4, "trailing": raw + b"extra"}[damage]
    )
    with pytest.raises(ValueError):
        f.adapter_summary(path)


def test_published_checkpoints_are_content_addressed_and_tamper_evident(tmp_path):
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "adapter_config.json").write_bytes(b"{}")
    (staged / "adapter_model.safetensors").write_bytes(
        safetensors_bytes({"lora_A": b"\x00\x00\x80\x3f"})
    )
    files = tuple(a.FileEntry(**e) for e in a.inventory(staged, f.ADAPTER_FILES))
    checkpoint = adapter_checkpoint(files=files)
    root = tmp_path / "out"
    root.mkdir()
    model_id = f.publish_checkpoint(root, checkpoint, staged)
    assert model_id == "checkpoint:sha256:" + a.digest(checkpoint.model_dump(mode="json"))
    # Republishing the same content is idempotent, not a second write.
    assert f.publish_checkpoint(root, checkpoint, staged) == model_id
    directory = root / "checkpoints" / model_id.rsplit(":", 1)[1]
    assert {p.name for p in directory.iterdir()} == {*f.ADAPTER_FILES, "manifest.json"}
    (directory / "adapter_config.json").write_bytes(b'{"tampered":true}')
    with pytest.raises(ValueError):
        f.verify_checkpoint_files(
            directory, checkpoint, a.canonical(checkpoint.model_dump(mode="json"))
        )


def probe_record(**overrides):
    record = {
        "schema_version": 1,
        "model_id": "checkpoint:sha256:" + "0" * 64,
        "page_id": "page-1",
        "image_sha256": "a" * 64,
        "original_width": PAGE_WIDTH,
        "original_height": PAGE_HEIGHT,
        "process_id": 11,
        "status": "ok",
        "finish_reason": "stop",
        "response_tokens": 42,
        "response_sha256": "c" * 64,
        "regions": [],
    }
    record.update(overrides)
    return record


def test_reload_probe_requires_identical_output_and_a_distinct_process():
    before = probe_record()
    f.compare_probes(before, probe_record(process_id=22), distinct_process=True)
    # A same-process comparison is a cheap self-check, not fresh-process evidence.
    f.compare_probes(before, probe_record())
    with pytest.raises(RuntimeError, match="distinct"):
        f.compare_probes(before, probe_record(), distinct_process=True)
    for change in ("response_sha256", "finish_reason", "status", "response_tokens"):
        altered = {
            "response_sha256": "d" * 64,
            "finish_reason": "length",
            "status": "truncated",
            "response_tokens": 41,
        }[change]
        with pytest.raises(RuntimeError, match=change):
            f.compare_probes(before, probe_record(process_id=22, **{change: altered}))


# --------------------------------------------------------------- adapter guards


def worker_roots(tmp_path):
    inputs, outputs = tmp_path / "inputs", tmp_path / "outputs"
    (inputs / "model").mkdir(parents=True)
    (inputs / "images").mkdir(parents=True)
    outputs.mkdir()
    manifest = tmp_path / "worker.json"
    manifest.write_bytes(
        a.canonical(
            a.WorkerManifest(
                schema_version=1,
                source_sha="c" * 40,
                files=(),
                environment=a.Packages(python="3.11.14", packages=()),
            ).model_dump(mode="json")
        )
    )
    return inputs, outputs, manifest


def test_adapter_rejects_an_unsupported_recipe_and_overlapping_roots(tmp_path):
    inputs, outputs, manifest = worker_roots(tmp_path)
    real = real_config()
    f.FactoryModel(inputs, outputs, real, runtime_manifest=manifest)
    other = real.model_copy(
        update={
            "backend": "other-v1",
            "expected_identity": real.expected_identity.model_copy(
                update={"recipe_version": real.recipe_version}
            ),
        }
    )
    with pytest.raises(ValueError, match="recipe"):
        f.FactoryModel(inputs, outputs, other, runtime_manifest=manifest)
    with pytest.raises(ValueError, match="overlap"):
        f.FactoryModel(inputs, inputs, real, runtime_manifest=manifest)
    with pytest.raises(ValueError, match="deadline"):
        f.FactoryModel(inputs, outputs, real, runtime_manifest=manifest, deadline_unix_seconds=-1)


@pytest.mark.parametrize(
    "experiment_id,round_number", [("", 0), ("ok", -1), ("ok", 1.0), (None, 0), ("../x", 0)]
)
def test_adapter_rejects_invalid_ownership(experiment_id, round_number):
    with pytest.raises(ValueError, match="ownership"):
        f.FactoryModel._ownership(experiment_id, round_number)


# ------------------------------------------------- actual toolkit CPU checks


@pytest.fixture
def toolkit(tmp_path_factory):
    """Real tokenizer/template from the pinned small processor assets. No weights load."""
    value = os.environ.get("ACTIVE_OCR_PROCESSOR_DIR")
    if not value:
        pytest.skip("UNVERIFIED: pinned toolkit template checks await the reviewed Linux build")
    from importlib.metadata import PackageNotFoundError, version

    try:
        installed = version("llamafactory")
    except PackageNotFoundError:
        pytest.skip("UNVERIFIED: optional llamafactory dependency is absent")
    assert installed == f.recipe_field("toolkit", "version")
    root = Path(value)
    # Small assets only; the 4B weight shards are never opened here.
    assert a.inventory(root, f.PROCESSOR_FILES) == f.asset_manifest(f.PROCESSOR_FILES)["files"]
    from llamafactory.data import get_template_and_fix_tokenizer
    from llamafactory.hparams import get_infer_args
    from llamafactory.model import load_tokenizer

    model_args, data_args, _, _ = get_infer_args(
        {
            "model_name_or_path": str(root),
            "template": f.TEMPLATE,
            "infer_backend": "huggingface",
            "trust_remote_code": False,
            "cutoff_len": f.MAX_SEQUENCE_TOKENS,
            "image_max_pixels": f.recipe_field("image", "image_max_pixels"),
            "image_min_pixels": f.recipe_field("image", "image_min_pixels"),
        }
    )
    module = load_tokenizer(model_args)
    template = get_template_and_fix_tokenizer(module["tokenizer"], data_args)
    return module["tokenizer"], module["processor"], template, data_args


def test_actual_template_is_the_pinned_nothink_variant(toolkit):
    from llamafactory.data.template import TEMPLATES

    tokenizer, _, template, _ = toolkit
    assert f.TEMPLATE in TEMPLATES
    assert template.stop_words == ["<|im_end|>"]
    assert tokenizer.convert_tokens_to_ids("<|im_end|>") in template.get_stop_token_ids(tokenizer)


def test_actual_template_supervises_the_whole_target_and_masks_the_prompt(toolkit, tmp_path):
    from llamafactory.data.processor.supervised import SupervisedDatasetProcessor
    from llamafactory.extras.constants import IGNORE_INDEX

    tokenizer, processor, template, data_args = toolkit
    revealed = example(tmp_path, ' ä"\\  ', "", "second")
    target = f.serialize_target(revealed)
    image_path = str(revealed.page.image_uri)
    encoder = SupervisedDatasetProcessor(
        template=template, tokenizer=tokenizer, processor=processor, data_args=data_args
    )
    input_ids, labels = encoder._encode_data_example(
        prompt=[{"role": "user", "content": "<image>" + f.PROMPT}],
        response=[{"role": "assistant", "content": target}],
        system=None,
        tools=None,
        images=[image_path],
        videos=[],
        audios=[],
    )
    assert len(input_ids) == len(labels)
    supervised = [i for i, label in enumerate(labels) if label != IGNORE_INDEX]
    # Exactly one contiguous supervised span at the end: the assistant target plus its stop token.
    assert supervised == list(range(supervised[0], len(labels)))
    assert labels[supervised[0] :] == input_ids[supervised[0] :]
    assert input_ids[-2] == tokenizer.convert_tokens_to_ids("<|im_end|>")
    assert tokenizer.decode(input_ids[-1:]) == "\n"
    image_token = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    assert input_ids.count(image_token) > 1
    assert all(labels[i] == IGNORE_INDEX for i, t in enumerate(input_ids) if t == image_token)
    decoded = tokenizer.decode(
        [t for t in labels[supervised[0] : -2] if t != IGNORE_INDEX],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    assert json.loads(decoded) == json.loads(target)
    assert len(input_ids) <= f.MAX_SEQUENCE_TOKENS


def test_actual_encoded_lengths_are_measured_and_overlong_targets_fail(toolkit, tmp_path):
    tokenizer, processor, template, _ = toolkit
    revealed = example(tmp_path, "a short line")
    messages = template.mm_plugin.process_messages(
        [
            {"role": "user", "content": "<image>" + f.PROMPT},
            {"role": "assistant", "content": f.serialize_target(revealed)},
        ],
        [str(revealed.page.image_uri)],
        [],
        [],
        processor,
    )
    prompt_ids, response_ids = template.encode_oneturn(tokenizer, messages, None, None)
    assert 1 <= len(prompt_ids) <= f.MAX_PROMPT_TOKENS
    assert 1 <= len(response_ids) <= f.MAX_OUTPUT_TOKENS
    assert len(prompt_ids) + len(response_ids) <= f.MAX_SEQUENCE_TOKENS
    huge = example(tmp_path, "long target " * 5000)
    long_messages = template.mm_plugin.process_messages(
        [
            {"role": "user", "content": "<image>" + f.PROMPT},
            {"role": "assistant", "content": f.serialize_target(huge)},
        ],
        [str(huge.page.image_uri)],
        [],
        [],
        processor,
    )
    _, long_response = template.encode_oneturn(tokenizer, long_messages, None, None)
    # Preflight must reject this rather than let the upstream processor truncate it.
    assert len(long_response) > f.MAX_OUTPUT_TOKENS
    worker = object.__new__(f.FactoryModel)
    worker.model_root = Path(os.environ["ACTIVE_OCR_PROCESSOR_DIR"])
    paths = {revealed.page.id: Path(revealed.page.image_uri)}
    measured = worker._preflight(
        (revealed,), paths, {revealed.page.id: f.serialize_target(revealed)}
    )
    assert measured[revealed.page.id]["total_tokens"] == len(prompt_ids) + len(response_ids)
    with pytest.raises(ValueError, match="target overflow"):
        worker._preflight((huge,), paths, {huge.page.id: f.serialize_target(huge)})


def test_actual_native_lora_starts_with_zero_B_and_changes_after_training(toolkit, tmp_path):
    """Real PEFT default-init/update/save evidence on a tiny CPU layer, not a 4B GPU fit."""
    import torch
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(824)
    base = torch.nn.Sequential(torch.nn.Linear(4, 4, bias=False))
    model = get_peft_model(
        base, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, target_modules=["0"])
    )
    model.save_pretrained(tmp_path / "before", safe_serialization=True)
    before = f.adapter_summary(tmp_path / "before/adapter_model.safetensors")
    assert before["nonzero_tensors"] > 0 and before["changed_lora_B_tensors"] == 0
    frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-5)
    model(torch.ones(1, 4)).square().mean().backward()
    optimizer.step()
    model.save_pretrained(tmp_path / "after", safe_serialization=True)
    after = f.verify_adapter(tmp_path / "after")
    assert after["changed_lora_B_tensors"] > 0
    assert all(torch.equal(p, frozen[n]) for n, p in model.named_parameters() if n in frozen)


def test_actual_native_cli_trains_and_fresh_process_reloads(toolkit, tmp_path, monkeypatch):
    """Actual toolkit SFT/serialization on a tiny random CPU Qwen3-VL, not the pinned 4B."""
    import shutil
    import subprocess
    import sys
    import time

    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    assets = Path(os.environ["ACTIVE_OCR_PROCESSOR_DIR"])
    model_root = tmp_path / "model"
    model_root.mkdir()
    for name in f.PROCESSOR_FILES:
        if name != "config.json":
            shutil.copyfile(assets / name, model_root / name)
    config = Qwen3VLConfig.from_pretrained(assets, local_files_only=True)
    text = config.text_config
    text.hidden_size, text.intermediate_size, text.num_hidden_layers = 64, 128, 1
    text.num_attention_heads, text.num_key_value_heads, text.head_dim = 2, 1, 32
    text.rope_parameters["mrope_section"] = [4, 6, 6]
    vision = config.vision_config
    vision.hidden_size, vision.intermediate_size, vision.depth = 32, 64, 1
    vision.num_heads, vision.out_hidden_size, vision.deepstack_visual_indexes = 2, 64, []
    Qwen3VLForConditionalGeneration(config).save_pretrained(model_root)
    revealed = example(tmp_path, "synthetic CPU line")
    f.write_dataset(
        tmp_path / "data",
        [f.conversation_row(Path(revealed.page.image_uri), f.serialize_target(revealed))],
    )
    resolved = f.training_yaml(model_root, tmp_path / "data", tmp_path / "adapter", 824)
    # Deliberately small CPU integration check; production BF16/3-epoch settings stay intact.
    resolved.update(
        use_cpu=True,
        bf16=False,
        max_steps=1,
        warmup_ratio=0.0,
        gradient_accumulation_steps=1,
        gradient_checkpointing=False,
        image_max_pixels=65536,
    )
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    worker = object.__new__(f.FactoryModel)
    worker.deadline_unix_seconds = time.time() + 120
    worker._train(resolved, tmp_path)
    steps, losses, runtime = worker._trainer_summary(tmp_path / "adapter")
    assert steps == 1 and losses and runtime > 0
    assert f.verify_adapter(tmp_path / "adapter")["changed_lora_B_tensors"] == 7
    program = """
import sys, torch
from pathlib import Path
from transformers import Qwen3VLForConditionalGeneration
from peft import PeftModel, get_peft_model_state_dict
from safetensors.torch import load_file
root = Path(sys.argv[1])
base = Qwen3VLForConditionalGeneration.from_pretrained(root / 'model', device_map='cpu')
model = PeftModel.from_pretrained(base, root / 'adapter')
actual = get_peft_model_state_dict(model)
saved = load_file(root / 'adapter/adapter_model.safetensors')
assert set(actual) == set(saved)
assert all(torch.equal(actual[k].cpu(), saved[k]) for k in saved)
"""
    reloaded = subprocess.run(
        [sys.executable, "-c", program, str(tmp_path)], capture_output=True, text=True, timeout=60
    )
    assert reloaded.returncode == 0, reloaded.stderr
