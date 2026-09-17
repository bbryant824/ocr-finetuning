"""Offline boundary checks and separately gated actual-ML synthetic checks.

QWEN_PROCESSOR_DIR opts into pinned small-asset checks; no 4B weights are loaded.
Absent optional dependencies/assets are explicitly UNVERIFIED skips.
"""

from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from active_ocr.integrations import qwen as q
from active_ocr.models import (
    Box,
    ExpectedIdentity,
    Prediction,
    PredictionPurpose,
    PredictionStatus,
    RealOCRConfig,
    RevealedExample,
    RunKind,
    SimulationPage,
    SourceRegion,
    Split,
)


def recipe(**identity_overrides):
    identity = dict(
        source_sha="a" * 40,
        dependency_sha256="b" * 64,
        model_revision=q.REVISION,
        processor_revision=q.REVISION,
        model_manifest_sha256=q.digest(q.asset_manifest(q.BASE_FILES)),
        processor_manifest_sha256=q.digest(q.asset_manifest(q.PROCESSOR_FILES)),
        recipe_version=q.RECIPE,
        evaluator_id="page-text-nfc-v1",
        code_bundle_sha256="c" * 64,
        remote_dependency_sha256="d" * 64,
    )
    identity.update(identity_overrides)
    return RealOCRConfig(
        backend="qwen3-vl-v1",
        recipe_version=q.RECIPE,
        model_repository=q.REPOSITORY,
        processor_repository=q.REPOSITORY,
        model_revision=q.REVISION,
        processor_revision=q.REVISION,
        training_policy_id=q.TRAIN_POLICY,
        decode_policy_id=q.DECODE_POLICY,
        expected_identity=ExpectedIdentity(**identity),
    )


def page(tmp_path, **overrides):
    root = tmp_path / "input" / "images"
    root.mkdir(parents=True, exist_ok=True)
    path = root / "p.png"
    if not path.exists():
        Image.new("RGB", (320, 480), "white").save(path)
    fields = dict(
        id="p",
        document_id="unused-document",
        image_uri=str(path),
        source_image="never-read-ground-truth.xml",
        width=320,
        height=480,
        image_sha256=q.file_hash(path),
        split=Split.TRAIN,
    )
    fields.update(overrides)
    return SimulationPage(**fields)


@pytest.fixture
def worker(tmp_path):
    page(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    manifest = tmp_path / "worker.json"
    manifest.write_bytes(
        q.canonical(
            dict(
                schema_version=1,
                source_sha="a" * 40,
                files=[],
                environment={"python": "3.11.14", "packages": []},
            )
        )
    )
    return q.QwenModel(tmp_path / "input", output, recipe(), runtime_manifest=manifest)


def test_import_never_imports_ml():
    script = (
        "import sys; import active_ocr.integrations.qwen; "
        "assert not {'torch','transformers','peft','torchvision'} & sys.modules.keys()"
    )
    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        env={**os.environ, "PYTHONPATH": str(Path(q.__file__).parents[2])},
    )


def test_load_processor_accepts_canonical_tuple_normalization_and_rejects_drift(
    tmp_path, monkeypatch
):
    # Transformers 5.16.1 BaseImageProcessor._standardize_kwargs converts both JSON
    # lists to tuples. Stub only the unavailable ML boundary; run our real loader/guard.
    class TorchvisionBackend:
        pass

    class Qwen2VLImageProcessor(TorchvisionBackend):
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return image

    class Qwen3VLProcessor:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    image = Qwen2VLImageProcessor()
    image.__dict__.update(
        patch_size=16,
        temporal_patch_size=2,
        merge_size=2,
        do_resize=True,
        do_rescale=True,
        do_normalize=True,
        rescale_factor=1 / 255,
        resample=Image.Resampling.BICUBIC,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
    )
    tokenizer = SimpleNamespace(
        is_fast=True,
        convert_tokens_to_ids={
            "<|im_end|>": 151645,
            "<|endoftext|>": 151643,
            "<|image_pad|>": 151655,
        }.__getitem__,
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: tokenizer),
            Qwen3VLVideoProcessor=SimpleNamespace(from_pretrained=lambda *a, **k: object()),
            Qwen3VLProcessor=Qwen3VLProcessor,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers.image_processing_backends",
        SimpleNamespace(TorchvisionBackend=TorchvisionBackend),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers.models.qwen2_vl.image_processing_qwen2_vl",
        SimpleNamespace(Qwen2VLImageProcessor=Qwen2VLImageProcessor),
    )
    (tmp_path / "chat_template.json").write_text('{"chat_template":"synthetic"}')
    loaded = q.load_processor(tmp_path)
    assert loaded.image_processor.image_mean == (0.5, 0.5, 0.5)
    assert loaded.image_processor.image_std == (0.5, 0.5, 0.5)
    for attribute in ("image_mean", "image_std"):
        setattr(image, attribute, (0.5, 0.5, 0.4))
        with pytest.raises(ValueError, match=attribute):
            q.load_processor(tmp_path)
        setattr(image, attribute, (0.5, 0.5, 0.5))


def test_public_assets_match_embedded_provenance():
    record = json.loads(
        (Path(__file__).parents[1] / "experiments/assets/read2016-qwen3vl4b.json").read_text()
    )
    assert {
        f["filename"]: (f["bytes"], f["sha256"])
        for f in record["model"]["files"]
        if f["filename"] in q.ASSETS
    } == q.ASSETS
    assert len(q.ASSETS) == 12


@pytest.mark.parametrize(
    "h,w,expected",
    [
        (3511, 2480, (1024, 704)),
        (480, 320, (480, 320)),
        (2480, 3511, (704, 1024)),
        (64, 64, (256, 256)),
        (1000, 1000, (992, 992)),
        (1025, 1025, (1024, 1024)),
    ],
)
def test_resize_golden(h, w, expected):
    assert q.resize_geometry(h, w)[:2] == expected


@pytest.mark.parametrize("h,w", [(0, 4), (True, 4), (100000, 32), (1, 17)])
def test_resize_rejects_infeasible(h, w):
    with pytest.raises(ValueError):
        q.resize_geometry(h, w)


def test_target_preserves_text_and_original_box(tmp_path):
    text = '  ä\n"\\ <|im_end|><tag>  '
    example = RevealedExample(
        page=page(tmp_path),
        regions=(
            SourceRegion(id="original", text=text, box=Box(x=32, y=48, width=160, height=240)),
            SourceRegion(id="blank", text="", box=Box(x=0, y=0, width=1, height=1)),
        ),
    )
    target = q.serialize_target(example)
    assert "<" not in target and r"\u003c|im_end|>" in target
    assert json.loads(target)["regions"][0] == {"text": text, "bbox": [100, 100, 600, 600]}
    assert json.loads(target)["regions"][1]["text"] == ""
    assert "original" not in target and "illegible" not in target
    assert q.serialize_target(RevealedExample(page=page(tmp_path), regions=())) == '{"regions":[]}'


@pytest.mark.parametrize("split", [Split.VALIDATION, Split.TEST])
def test_target_rejects_held_out(tmp_path, split):
    with pytest.raises(ValueError, match="selected TRAIN"):
        q.serialize_target(RevealedExample(page=page(tmp_path, split=split), regions=()))


@pytest.mark.parametrize(
    "raw",
    [
        "{}",
        "[]",
        '{"regions":[],"extra":0}',
        '{"regions":[],"regions":[]}',
        '```json\n{"regions":[]}\n```',
        '{"regions":[]} trailing',
        '{"regions":[{"text":"a","bbox":[false,0,100,100]}]}',
        '{"regions":[{"text":"a","bbox":["0",0,100,100]}]}',
        '{"regions":[{"text":"a","bbox":[0,0,NaN,100]}]}',
        '{"regions":[{"text":"a","bbox":[0,0,1e999,100]}]}',
        '{"regions":[{"text":"a","bbox":[-1,0,100,100]}]}',
        '{"regions":[{"text":"a","bbox":[0,0,1001,100]}]}',
        '{"regions":[{"text":"a","bbox":[0,0,0,100]}]}',
        '{"regions":[{"text":null,"bbox":[0,0,100,100]}]}',
        '{"regions":[{"text":"a","bbox":[0,0,100,100],"id":"hidden"}]}',
        '{"regions":[{"text":"\\ud800","bbox":[0,0,100,100]}]}',
    ],
)
def test_strict_output(raw):
    with pytest.raises((ValueError, UnicodeError)):
        q.parse_regions(raw, 101, 1237)


@pytest.mark.parametrize("width,x", [(101, 144.3408), (1237, 86.8629)])
def test_right_edge_survives_existing_pipeline_validator(tmp_path, width, x):
    from active_ocr.pipeline import Pipeline

    p = page(tmp_path, width=width, height=1237)
    regions = q.parse_regions(
        json.dumps({"regions": [{"text": "", "bbox": [x, x, 1000, 1000]}]}), width, 1237
    )
    prediction = Prediction(
        page_id=p.id,
        experiment_id="run",
        round_number=1,
        model_id="checkpoint:sha256:" + "a" * 64,
        regions=regions,
    )
    run = SimpleNamespace(id="run", kind=RunKind.REAL, dataset=SimpleNamespace(pages=(p,)))
    assert Pipeline._validate_simulation_predictions(
        (prediction,), (p.id,), run, 1, prediction.model_id, require_scores=False
    ) == (prediction,)
    assert regions[0].text == "" and not regions[0].illegible


def test_loss_span_exact_boundaries():
    ids, labels = q.training_span(
        [10, q.IMAGE, 20], [10, q.IMAGE, 20, 30, 31, q.EOS, 40], [30, 31], [40], {q.IMAGE, q.EOS}
    )
    assert ids == [10, q.IMAGE, 20, 30, 31, q.EOS]
    assert labels == [-100, -100, -100, 30, 31, q.EOS]
    # Causal model shifts internally: logits at position2 predict first target at3.
    assert [(i, labels[i + 1]) for i in range(len(ids) - 1) if labels[i + 1] != -100] == [
        (2, 30),
        (3, 31),
        (4, q.EOS),
    ]


@pytest.mark.parametrize("which", ["prefix", "eos", "trailing", "control", "prompt", "target"])
def test_span_rejects_unsafe_or_overflow(which):
    prompt, target, newline = [1, q.IMAGE], [2, 3], [4]
    if which == "control":
        target = [q.EOS]
    if which == "prompt":
        prompt = [1] * 2049
    if which == "target":
        target = [2] * 4096
    full = prompt + target + [q.EOS] + newline
    if which == "prefix":
        full[0] = 9
    if which == "eos":
        full[-2] = 8
    if which == "trailing":
        full.append(9)
    with pytest.raises(ValueError):
        q.training_span(prompt, full, target, newline, {q.EOS})


class DecodeTokenizer:
    all_special_ids = [q.EOS, q.IMAGE]

    def decode(self, ids, **kwargs):
        assert kwargs == dict(skip_special_tokens=False, clean_up_tokenization_spaces=False)
        return '{"regions":[]}' if q.EOS not in ids else '{"regions":[]}<|im_end|>'


def test_decode_cap_and_control_evidence():
    assert q.decode_result([7] * 2048, DecodeTokenizer(), 1, 1)[:2] == (
        PredictionStatus.TRUNCATED,
        (),
    )
    assert q.decode_result([7, q.EOS], DecodeTokenizer(), 1, 1)[0] is PredictionStatus.OK
    status, regions, raw, _ = q.decode_result([q.IMAGE, q.EOS], DecodeTokenizer(), 1, 1)
    assert status is PredictionStatus.INVALID_OUTPUT and not regions and raw
    with pytest.raises(RuntimeError):
        q.decode_result([7], DecodeTokenizer(), 1, 1)


@pytest.mark.parametrize("n,updates", [(2, 3), (10, 9), (20, 15)])
def test_cumulative_epoch_coverage_and_updates(n, updates):
    ids = tuple(str(i) for i in range(n))
    epochs = [q.epoch_groups(ids, 824, e) for e in range(3)]
    assert sum(map(len, epochs)) == updates
    for groups in epochs:
        assert sorted(p for g in groups for p in g) == sorted(ids)
        assert all(1 <= len(group) <= 4 for group in groups)
    assert q.epoch_groups(ids, 824, 0) == q.epoch_groups(tuple(reversed(ids)), 824, 0)


@pytest.mark.parametrize("path", ["../outside", "/outside", "images/../../outside"])
def test_path_traversal(tmp_path, path):
    with pytest.raises(ValueError):
        q.safe_path(tmp_path, path)


def test_symlink_and_image_mutation(worker, tmp_path):
    p = page(tmp_path)
    assert worker._images((p,), Split.TRAIN)[p.id].size == (320, 480)
    link = tmp_path / "input" / "images" / "alias"
    link.symlink_to(p.image_uri)
    with pytest.raises(ValueError, match="symlink"):
        worker._images((p.model_copy(update={"image_uri": str(link)}),), Split.TRAIN)
    Path(p.image_uri).write_bytes(b"changed")
    with pytest.raises(ValueError, match="bytes changed"):
        worker._images((p,), Split.TRAIN)


@pytest.mark.parametrize("case", ["test", "validation", "duplicate", "bad_box", "seed", "round"])
def test_fit_failure_before_processor_or_model(worker, tmp_path, monkeypatch, case):
    monkeypatch.setattr(worker, "_verify_runtime", lambda: None)
    monkeypatch.setattr(worker, "_processor_ready", lambda: pytest.fail("processor reached"))
    p = page(
        tmp_path, split={"test": Split.TEST, "validation": Split.VALIDATION}.get(case, Split.TRAIN)
    )
    regions = (
        (SourceRegion(id="bad", text="s", box=Box(x=0, y=0, width=999, height=1)),)
        if case == "bad_box"
        else ()
    )
    example = RevealedExample(page=p, regions=regions)
    with pytest.raises(ValueError):
        worker.fit(
            (example, example) if case == "duplicate" else (example,),
            seed=-1 if case == "seed" else 824,
            experiment_id="run",
            round_number=0 if case == "round" else 1,
        )
    assert not list(worker.output_root.iterdir())


def test_wrong_recipe_constructor_and_worker_before_load(worker, monkeypatch):
    monkeypatch.setattr(worker, "_load", lambda: pytest.fail("model reached"))
    with pytest.raises(ValueError, match="executing module"):
        worker.load_base(experiment_id="run")
    worker.real_config = worker.real_config.model_copy(update={"training_policy_id": "wrong"})
    with pytest.raises(ValueError, match="recipe"):
        worker.load_base(experiment_id="run")


@pytest.mark.parametrize("rank", [16, 8])
def test_fit_exports_full_targets_and_preserves_strict_config_guard(worker, monkeypatch, rank):
    # Real fit/save/bind/config checks with unavailable ML execution stubbed.
    expected = dict(
        base_model_name_or_path=q.REPOSITORY,
        revision=q.REVISION,
        target_modules=list(q.TARGETS),
        r=16,
        inference_mode=True,
    )
    model = SimpleNamespace(state={"parameter": "initial"})
    checked = []

    def check(candidate):
        assert candidate is model
        checked.append(True)
        return []

    class BeforeTensorValidation(Exception):
        pass

    def save(stage, **kwargs):
        assert checked == [True]
        # PEFT's >=20-target injection optimization, measured on the pinned build.
        config = {**expected, "target_modules": ["q_proj", "v_proj"], "r": rank}
        (stage / "adapter_config.json").write_text(json.dumps(config))
        (stage / "adapter_model.safetensors").write_bytes(b"synthetic adapter")

    def train(*args):
        model.state = {"parameter": "trained"}
        return {"updates": 3}

    def safe_open(*args, **kwargs):
        raise BeforeTensorValidation

    model.save_pretrained = save
    monkeypatch.setattr(worker, "_verify_runtime", lambda: None)
    monkeypatch.setattr(worker, "_processor_ready", lambda: object())
    monkeypatch.setattr(worker, "_load", lambda: model)
    monkeypatch.setattr(worker, "_drop", lambda: None)
    monkeypatch.setattr(q, "reset_seed", lambda seed: None)
    monkeypatch.setattr(q, "encode_page", lambda *args: ({}, {"target_tokens": 4}))
    monkeypatch.setattr(q, "lora_config", lambda: SimpleNamespace(to_dict=lambda: dict(expected)))
    monkeypatch.setattr(q, "check_trainables", check)
    monkeypatch.setattr(q, "frozen_hashes", lambda model: {"frozen": "unchanged"})
    monkeypatch.setattr(q, "tensor_hash", lambda value: value)
    monkeypatch.setattr(q, "train_epochs", train)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "safetensors", SimpleNamespace(safe_open=safe_open))
    monkeypatch.setitem(
        sys.modules,
        "peft",
        SimpleNamespace(
            get_peft_model=lambda *args, **kwargs: model,
            get_peft_model_state_dict=lambda model: model.state,
        ),
    )
    error = BeforeTensorValidation if rank == 16 else ValueError
    with pytest.raises(error, match=None if rank == 16 else "configuration mismatch"):
        worker.fit(
            (RevealedExample(page=page(worker.input_root.parent), regions=()),),
            seed=824,
            experiment_id="run",
            round_number=1,
        )


@pytest.mark.parametrize("image_first", [True, False])
def test_installed_packages_records_effective_search_path_versions(
    tmp_path, monkeypatch, image_first
):
    # Actual metadata resolution, without installing or importing any ML package.
    image, vendor = tmp_path / "image", tmp_path / "vendor"
    for root, name, version in (
        (image, "Typing_Extensions", "4.16.0"),
        (vendor, "typing-extensions", "4.13.2"),
        (image, "hyperframe", "6.1.0"),
        (vendor, "hyperframe", "6.1.0"),
        (image, "torch", q.PINS["torch"]),
    ):
        metadata = root / f"{name.lower().replace('-', '_')}-{version}.dist-info"
        metadata.mkdir(parents=True)
        (metadata / "METADATA").write_text(f"Name: {name}\nVersion: {version}\n")
    paths = [str(image), str(vendor)] if image_first else [str(vendor), str(image)]
    monkeypatch.setattr(sys, "path", paths)
    observed = q.installed_packages()
    assert observed.packages == (
        ("hyperframe", "6.1.0"),
        ("torch", q.PINS["torch"]),
        ("typing-extensions", "4.16.0" if image_first else "4.13.2"),
    )


def test_worker_manifest_no_git_checks_actual_code_and_packages(worker, monkeypatch):
    import active_ocr
    import active_ocr.models

    root = worker.runtime_manifest.parent
    names = ["code/__init__.py", "code/models.py", "code/qwen.py", "code/z_integrations.py"]
    (root / "code").mkdir()
    for name in names:
        (root / name).write_text("# frozen synthetic bundle\n")
    monkeypatch.setattr(active_ocr, "__file__", str(root / names[0]))
    monkeypatch.setattr(active_ocr.models, "__file__", str(root / names[1]))
    monkeypatch.setattr(q, "__file__", str(root / names[2]))
    monkeypatch.setattr(active_ocr.integrations, "__file__", str(root / names[3]))
    packages = q.Packages(python="3.11.14", packages=tuple(sorted(q.PINS.items())))
    files = q.inventory(root, names)
    worker.real_config = recipe(
        code_bundle_sha256=q.digest({"source_sha": "a" * 40, "files": files}),
        remote_dependency_sha256=q.digest(packages.model_dump(mode="json")),
    )
    worker.identity = worker.real_config.expected_identity
    worker.runtime_manifest.write_bytes(
        q.canonical(
            dict(
                schema_version=1,
                source_sha="a" * 40,
                files=files,
                environment=packages.model_dump(),
            )
        )
    )
    monkeypatch.setattr(q, "installed_packages", lambda: packages)
    monkeypatch.setattr(q.platform, "system", lambda: "Linux")
    monkeypatch.setattr(q.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(worker, "_verify_assets", lambda: None)
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: pytest.fail("worker invoked Git/process")
    )
    worker._verify_runtime()
    monkeypatch.setattr(
        q, "installed_packages", lambda: packages.model_copy(update={"python": "3.12"})
    )
    with pytest.raises(ValueError, match="package"):
        worker._verify_runtime()
    monkeypatch.setattr(q, "installed_packages", lambda: packages)
    (root / names[2]).write_text("# corrupted\n")
    with pytest.raises(ValueError, match="code identity"):
        worker._verify_runtime()


def test_base_checkpoint_content_address_and_ownership(worker, monkeypatch):
    checkpoint = q.Checkpoint(kind="base", binding=worker._binding())
    reference = q.publish_checkpoint(worker.output_root, checkpoint)
    assert reference == "checkpoint:sha256:" + q.digest(checkpoint.model_dump(mode="json"))
    assert q.publish_checkpoint(worker.output_root, checkpoint) == reference
    worker._checkpoint(reference, "run", 0)
    with pytest.raises(ValueError, match="positive round"):
        worker._checkpoint(reference, "run", 1)
    monkeypatch.setattr(worker, "_verify_runtime", lambda: None)
    monkeypatch.setattr(worker, "_load", lambda *a: pytest.fail("empty batch loaded model"))
    assert (
        worker.predict(
            (),
            experiment_id="run",
            round_number=0,
            model_id=reference,
            purpose=PredictionPurpose.BASELINE_VALIDATION,
        )
        == ()
    )


@pytest.mark.parametrize("corruption", ["extra", "manifest", "symlink", "missing"])
def test_checkpoint_corruption(worker, corruption):
    checkpoint = q.Checkpoint(kind="base", binding=worker._binding())
    reference = q.publish_checkpoint(worker.output_root, checkpoint)
    root = worker.output_root / "checkpoints" / reference.rsplit(":", 1)[1]
    if corruption == "extra":
        (root / "extra").write_bytes(b"x")
    elif corruption == "manifest":
        (root / "manifest.json").write_bytes(b"{}")
    elif corruption == "symlink":
        original = (root / "manifest.json").read_bytes()
        (worker.output_root / "elsewhere").write_bytes(original)
        (root / "manifest.json").unlink()
        (root / "manifest.json").symlink_to(worker.output_root / "elsewhere")
    else:
        (root / "manifest.json").unlink()
    with pytest.raises((ValueError, OSError)):
        worker._checkpoint(reference, "run", 0)


def test_adapter_byte_inventory_rejects_missing_corrupt_and_extra(worker, tmp_path):
    stage = tmp_path / "stage"
    stage.mkdir()
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        (stage / name).write_bytes(b"synthetic boundary bytes, not model tensors")
    checkpoint = q.Checkpoint(
        kind="adapter",
        binding=worker._binding(),
        experiment_id="run",
        round_number=1,
        selected=(("p", "a" * 64),),
        target_sha256="b" * 64,
        seed=824,
        training={},
        files=tuple(
            q.FileEntry(**f)
            for f in q.inventory(stage, ("adapter_config.json", "adapter_model.safetensors"))
        ),
    )
    reference = q.publish_checkpoint(worker.output_root, checkpoint, stage)
    root = worker.output_root / "checkpoints" / reference.rsplit(":", 1)[1]
    with pytest.raises(ValueError, match="ownership"):
        worker._checkpoint(reference, "other-run", 1)
    (root / "adapter_model.safetensors").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="bytes changed"):
        worker._checkpoint(reference, "run", 1)


def test_actual_staged_headers_when_available():
    value = os.environ.get("QWEN_HEADER_DIR")
    if not value:
        pytest.skip("UNVERIFIED here: QWEN_HEADER_DIR not set; header-only, never loads weights")
    root = Path(value)
    headers = {}
    for name in q.BASE_FILES:
        if name.endswith(".safetensors"):
            with (root / name).open("rb") as stream:
                size = struct.unpack("<Q", stream.read(8))[0]
                headers.update(json.loads(stream.read(size)))
    for target in q.TARGETS:
        assert headers[target + ".weight"]["shape"] == [
            4096 if target.endswith("q_proj") else 1024,
            2560,
        ]
    assert len(q.adapter_shapes()) == 144
    assert sum(a * b for a, b in q.adapter_shapes().values()) == 5898240


@pytest.fixture
def processor():
    value = os.environ.get("QWEN_PROCESSOR_DIR")
    if not value:
        pytest.skip("UNVERIFIED: pinned processor/tiny CPU checks await reviewed Linux build")
    from importlib.metadata import PackageNotFoundError, version

    for name, expected in q.PINS.items():
        try:
            observed = version(name)
        except PackageNotFoundError:
            pytest.skip(f"UNVERIFIED: optional pinned dependency {name} is absent")
        assert observed == expected
    root = Path(value)
    # Small assets only; no model weights are opened by this fixture.
    assert q.inventory(root, q.PROCESSOR_FILES) == q.asset_manifest(q.PROCESSOR_FILES)["files"]
    return q.load_processor(root)


@pytest.mark.parametrize("size", [(64, 64), (2480, 3511), (3511, 2480), (1000, 1000)])
def test_actual_processor_grid_and_template(processor, size):
    batch, receipt = q.encode_page(processor, Image.new("RGB", size, "white"))
    h, w, _ = q.resize_geometry(size[1], size[0])
    assert batch["image_grid_thw"].tolist() == [[1, h // 16, w // 16]]
    assert int((batch["input_ids"] == q.IMAGE).sum()) == h * w // 1024
    assert receipt["prompt_tokens"] == batch["input_ids"].shape[1]
    assert batch["mm_token_type_ids"].shape == batch["input_ids"].shape


@pytest.mark.parametrize("text", [' ä\n"\\  ', "", "<|im_end|><|image_pad|>"])
def test_actual_tokenization_and_loss_mask(processor, tmp_path, text):
    example = RevealedExample(
        page=page(tmp_path),
        regions=(SourceRegion(id="line", text=text, box=Box(x=0, y=0, width=320, height=480)),),
    )
    target = q.serialize_target(example)
    batch, receipt = q.encode_page(processor, Image.new("RGB", (320, 480)), target)
    p = receipt["prompt_tokens"]
    assert batch["labels"][0, :p].tolist() == [-100] * p
    assert batch["labels"][0, -1].item() == q.EOS
    decoded = processor.tokenizer.decode(
        batch["labels"][0, p:-1].tolist(), clean_up_tokenization_spaces=False
    )
    assert json.loads(decoded)["regions"][0]["text"] == text
    assert batch["labels"].shape == batch["input_ids"].shape
    empty, _ = q.encode_page(processor, Image.new("RGB", (320, 480)), '{"regions":[]}')
    assert empty["labels"][0, -1].item() == q.EOS


def tiny_model(text_layers=2):
    """Test-only FP32/math-SDPA model. Never a production fallback."""
    import torch
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    torch.set_num_threads(2)
    config = Qwen3VLConfig(
        text_config=dict(
            vocab_size=151936,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=text_layers,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            max_position_embeddings=8192,
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 5000000,
                "mrope_section": [2, 3, 3],
                "mrope_interleaved": True,
            },
        ),
        vision_config=dict(
            depth=2,
            hidden_size=32,
            intermediate_size=64,
            num_heads=4,
            patch_size=16,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=64,
            num_position_embeddings=64,
            deepstack_visual_indexes=[0, 1],
        ),
        image_token_id=q.IMAGE,
        video_token_id=151656,
        vision_start_token_id=151652,
        vision_end_token_id=151653,
        tie_word_embeddings=True,
    )
    config._attn_implementation = "sdpa"
    return Qwen3VLForConditionalGeneration(config).float()


def tiny_adapter(base):
    from peft import LoraConfig, get_peft_model

    targets = [
        f"model.language_model.layers.{i}.self_attn.{p}_proj" for i in range(2) for p in ("q", "v")
    ]
    base.requires_grad_(False)
    return get_peft_model(
        base,
        LoraConfig(
            r=2,
            lora_alpha=4,
            lora_dropout=0,
            task_type="CAUSAL_LM",
            target_modules=targets,
            bias="none",
        ),
        autocast_adapter_dtype=True,
    )


def tiny_probe_setup():
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from transformers import GenerationConfig

    # Explicit test-sized generation; exercise slicing, tensor/logit equality and strict status.
    # Production generation_config is separately asserted below and has no forced EOS.
    q.cuda_context = lambda: sdpa_kernel(SDPBackend.MATH)
    q.generation_config = lambda: GenerationConfig(
        do_sample=False,
        num_beams=1,
        max_new_tokens=2,
        forced_eos_token_id=q.EOS,
        eos_token_id=q.EOS,
        pad_token_id=q.PAD,
        use_cache=True,
    )


def tiny_reload_child(root: Path):
    import torch
    from peft import PeftModel

    tiny_probe_setup()
    q.reset_seed(824)
    model = PeftModel.from_pretrained(
        tiny_model(),
        root / "adapter",
        is_trainable=False,
        autocast_adapter_dtype=True,
        local_files_only=True,
    )
    processor = q.load_processor(Path(os.environ["QWEN_PROCESSOR_DIR"]))
    inputs = torch.load(root / "inputs.pt", weights_only=True)
    result = q.reload_probe(
        model,
        inputs,
        processor.tokenizer,
        original_size=(64, 64),
        fixed_ids=json.loads((root / "before-ids.json").read_text()),
    )
    torch.save(result, root / "after.pt")


def test_actual_tiny_forward_reset_frozen_and_fresh_process(processor, tmp_path):
    import torch
    import torch.nn.functional as F
    from peft import get_peft_model_state_dict
    from torch.nn.attention import SDPBackend, sdpa_kernel

    q.reset_seed(824)
    model = tiny_adapter(tiny_model())
    state = get_peft_model_state_dict(model)
    initial = {k: v.detach().clone() for k, v in state.items()}
    assert len(initial) == 8
    assert all(torch.count_nonzero(v) == 0 for k, v in initial.items() if "lora_B" in k)
    assert all(v.dtype == torch.float32 for v in initial.values())
    frozen = q.frozen_hashes(model)
    batch, receipt = q.encode_page(processor, Image.new("RGB", (64, 64)), '{"regions":[]}')
    model.eval()
    with sdpa_kernel(SDPBackend.MATH):
        out = model(**batch, use_cache=False)
    manual = F.cross_entropy(
        out.logits[:, :-1].float().reshape(-1, 151936),
        batch["labels"][:, 1:].reshape(-1),
        ignore_index=-100,
    )
    assert torch.allclose(out.loss, manual)
    # Hand-selected first/last supervised predictions confirm causal shift and EOS supervision.
    p = receipt["prompt_tokens"]
    positions = list(range(p - 1, batch["input_ids"].shape[1] - 1))
    individual = [
        -torch.log_softmax(out.logits[0, i].float(), -1)[batch["labels"][0, i + 1]]
        for i in positions
    ]
    assert torch.allclose(out.loss, torch.stack(individual).mean())
    parameters = [v for v in model.parameters() if v.requires_grad]
    training = q.train_epochs(
        model, {"selected": batch}, 824, parameters, lambda: sdpa_kernel(SDPBackend.MATH)
    )
    assert training["updates"] == 3
    assert frozen == q.frozen_hashes(model)
    updated = get_peft_model_state_dict(model)
    assert any(not torch.equal(initial[k], updated[k]) for k in initial)
    assert all(torch.isfinite(v).all() for v in updated.values())
    # Intervening work cannot change a new fit's base/initial A and zero B state.
    q.reset_seed(824)
    reset = tiny_adapter(tiny_model())
    assert all(torch.equal(initial[k], v) for k, v in get_peft_model_state_dict(reset).items())
    del reset, out
    model.save_pretrained(
        tmp_path / "adapter", safe_serialization=True, save_embedding_layers=False
    )
    inputs = {k: v for k, v in batch.items() if k != "labels"}
    for key in ("input_ids", "attention_mask", "mm_token_type_ids"):
        inputs[key] = inputs[key][:, :p]
    torch.save(inputs, tmp_path / "inputs.pt")
    original_context, original_generation = q.cuda_context, q.generation_config
    tiny_probe_setup()
    try:
        before = q.reload_probe(model, inputs, processor.tokenizer, original_size=(64, 64))
        (tmp_path / "before-ids.json").write_text(json.dumps(before["ids"]))
        del model, parameters, state, updated
        code = (
            "import runpy; from pathlib import Path; "
            f"ns=runpy.run_path({str(Path(__file__).absolute())!r}); "
            f"ns['tiny_reload_child'](Path({str(tmp_path)!r}))"
        )
        subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            timeout=120,
            env={
                **os.environ,
                "PYTHONPATH": str(Path(q.__file__).parents[2]),
                "HF_HUB_OFFLINE": "1",
            },
        )
        after = torch.load(tmp_path / "after.pt", weights_only=True)
        assert q.compare_probes(before, after) <= 1e-2
    finally:
        q.cuda_context, q.generation_config = original_context, original_generation


def test_actual_partial_accumulation_matches_manual_adam(processor):
    from contextlib import nullcontext

    import torch

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.tensor([0.25, -0.5]))
            self.config = SimpleNamespace(use_cache=True)

        def gradient_checkpointing_enable(self, **kwargs):
            assert kwargs == {"gradient_checkpointing_kwargs": {"use_reentrant": False}}

        def forward(self, x, target, use_cache=False):
            return SimpleNamespace(loss=((self.w * x - target) ** 2).mean())

    batches = {
        str(i): {"x": torch.tensor([i + 1.0, 2.0]), "target": torch.tensor([1.0, -1.0])}
        for i in range(6)
    }
    model, manual = Toy(), Toy()
    result = q.train_epochs(model, batches, 824, list(model.parameters()), nullcontext)
    optimizer = torch.optim.AdamW(
        manual.parameters(),
        lr=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        foreach=False,
        fused=False,
    )
    for order in result["epoch_orders"]:
        for start in (0, 4):
            group = order[start : start + 4]
            optimizer.zero_grad(set_to_none=True)
            mean = torch.stack([manual(**batches[p]).loss for p in group]).mean()
            mean.backward()
            torch.nn.utils.clip_grad_norm_(manual.parameters(), 1.0)
            optimizer.step()
    assert result["updates"] == 6 and torch.equal(model.w, manual.w)


def test_actual_generation_defaults(processor):
    config = q.generation_config()
    assert config.do_sample is False and config.num_beams == config.num_return_sequences == 1
    assert config.max_new_tokens == 2048 and config.eos_token_id == q.EOS
    assert config.pad_token_id == q.PAD and config.repetition_penalty == 1
    assert config.no_repeat_ngram_size == 0
    assert config.forced_eos_token_id is None and config.forced_bos_token_id is None
    assert config.stop_strings is None


def test_actual_adapter_tensor_and_config_validation(processor, tmp_path, worker):
    import torch
    from peft import get_peft_model
    from safetensors.torch import save_file

    # Cross PEFT's 20-target compaction threshold through real injection/export.
    # Tiny dimensions remain test-only; all 72 pinned target paths are present.
    base = tiny_model(text_layers=36)
    base.requires_grad_(False)
    model = get_peft_model(base, q.lora_config(), autocast_adapter_dtype=True)
    assert model.peft_config["default"].target_modules == {"q_proj", "v_proj"}
    assert {name for name, p in model.named_parameters() if p.requires_grad} == {
        name.replace(".weight", ".default.weight") for name in q.adapter_shapes()
    }
    model.save_pretrained(tmp_path, safe_serialization=True, save_embedding_layers=False)
    config_path = tmp_path / "adapter_config.json"
    config = json.loads(config_path.read_text())
    config.update(base_model_name_or_path=q.REPOSITORY, revision=q.REVISION)
    config_path.write_bytes(q.canonical(config))
    # Compact suffixes are still refused by the strict checkpoint reader.
    with pytest.raises(ValueError, match="configuration mismatch"):
        q.check_adapter(tmp_path)
    config["target_modules"] = list(q.TARGETS)
    config_path.write_bytes(q.canonical(config))
    del model, base
    tensors = {
        key: torch.zeros(shape, dtype=torch.float32) for key, shape in q.adapter_shapes().items()
    }
    path = tmp_path / "adapter_model.safetensors"
    save_file(tensors, path)
    assert set(q.check_adapter(tmp_path)) == set(tensors)
    files = tuple(
        q.FileEntry(**f)
        for f in q.inventory(tmp_path, ("adapter_config.json", "adapter_model.safetensors"))
    )
    worker._bind_adapter(tmp_path, files)
    key = next(iter(tensors))
    assert worker._adapter_identity[2][key] == q.tensor_hash(tensors[key])
    tensors[key][0, 0] = 1.0
    save_file(tensors, path)
    # Actual finite, correctly shaped replacement still fails the frozen inventory before CUDA.
    with pytest.raises(ValueError, match="checkpoint bytes changed"):
        worker._load(tmp_path)
    tensors[key][0, 0] = float("nan")
    save_file(tensors, path)
    with pytest.raises(ValueError, match="tensor"):
        q.check_adapter(tmp_path)
    tensors[key][0, 0] = 0
    tensors[key] = tensors[key][:-1]
    save_file(tensors, path)
    with pytest.raises(ValueError, match="tensor"):
        q.check_adapter(tmp_path)
    del tensors[key]
    save_file(tensors, path)
    with pytest.raises(ValueError, match="keys"):
        q.check_adapter(tmp_path)
    config_path = tmp_path / "adapter_config.json"
    changed = json.loads(config_path.read_text())
    changed["base_model_name_or_path"] = "unreviewed/base"
    config_path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="configuration"):
        q.check_adapter(tmp_path)


@pytest.mark.parametrize("name", ["/absolute", "../escape", "a/../b", "a//b", "a\\b", "."])
def test_inventory_names_are_portable_relative_paths(name):
    with pytest.raises(ValueError):
        q.FileEntry(filename=name, bytes=1, sha256="a" * 64)


def test_lock_pins_and_linux_wheels():
    import tomllib

    lock = tomllib.loads((Path(__file__).parents[1] / "uv.lock").read_text())
    packages = {p["name"]: p for p in lock["package"]}
    for name, version in q.PINS.items():
        assert packages[name]["version"] == version
    for name in ("torch", "torchvision", "tokenizers", "safetensors"):
        assert any(
            "manylinux" in w["url"] and "x86_64" in w["url"] for w in packages[name]["wheels"]
        )


def test_selected_record_validation_and_seed_orders():
    record = {
        "height": 256,
        "width": 256,
        "grid": [1, 16, 16],
        "prompt_tokens": 100,
        "target_tokens": 12,
        "target_exceeds_decode_budget": False,
    }
    training = dict(
        updates=3,
        epoch_orders=[["p"]] * 3,
        losses=[1.0] * 3,
        gradient_norms=[0.1] * 3,
        processing={"p": record},
        supervised_tokens=36,
        changed_tensors=72,
        frozen_sha256="b" * 64,
    )
    checkpoint = q.Checkpoint(
        kind="adapter",
        binding={},
        experiment_id="run",
        round_number=1,
        selected=(("p", "a" * 64),),
        seed=824,
        training=training,
    )
    q.check_training_record(checkpoint)
    for field, value in (
        ("updates", 4),
        ("supervised_tokens", 12),
        ("changed_tensors", 0),
        ("epoch_orders", [["held-out"]] * 3),
        ("losses", [float("inf")] * 3),
    ):
        with pytest.raises(ValueError):
            q.check_training_record(
                checkpoint.model_copy(update={"training": {**training, field: value}})
            )


def test_predict_failure_does_not_load_or_publish(worker, tmp_path, monkeypatch):
    base = q.publish_checkpoint(
        worker.output_root, q.Checkpoint(kind="base", binding=worker._binding())
    )
    monkeypatch.setattr(worker, "_verify_runtime", lambda: None)
    monkeypatch.setattr(worker, "_load", lambda *a: pytest.fail("model reached"))
    for purpose, round_number, split in (
        (PredictionPurpose.POOL, 0, Split.TRAIN),
        (PredictionPurpose.VALIDATION, 0, Split.VALIDATION),
        (PredictionPurpose.BASELINE_VALIDATION, 0, Split.TEST),
    ):
        with pytest.raises(ValueError):
            worker.predict(
                (page(tmp_path, split=split),),
                experiment_id="run",
                round_number=round_number,
                model_id=base,
                purpose=purpose,
            )
    assert not (worker.output_root / "receipts").exists()


def mutable_adapter(worker, tmp_path):
    stage = tmp_path / "mutable-adapter"
    stage.mkdir()
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        (stage / name).write_bytes(b"original")
    processing = dict(
        height=256,
        width=256,
        grid=[1, 16, 16],
        prompt_tokens=100,
        target_tokens=12,
        target_exceeds_decode_budget=False,
    )
    checkpoint = q.Checkpoint(
        kind="adapter",
        binding=worker._binding(),
        experiment_id="run",
        round_number=1,
        selected=(("p", "a" * 64),),
        target_sha256="b" * 64,
        seed=824,
        training=dict(
            updates=3,
            epoch_orders=[["p"]] * 3,
            losses=[1.0] * 3,
            gradient_norms=[0.1] * 3,
            processing={"p": processing},
            supervised_tokens=36,
            changed_tensors=72,
            frozen_sha256="c" * 64,
        ),
        files=tuple(
            q.FileEntry(**f)
            for f in q.inventory(stage, ("adapter_config.json", "adapter_model.safetensors"))
        ),
    )
    ref = q.publish_checkpoint(worker.output_root, checkpoint, stage)
    directory = worker.output_root / "checkpoints" / ref.rsplit(":", 1)[1]
    return ref, directory


class TokenRows:
    def __init__(self, values):
        self.values = values
        self.shape = (1, len(values))

    def to(self, device):
        return self

    def __getitem__(self, key):
        if isinstance(key, tuple):
            return TokenRows(self.values[key[1]])
        return self

    def tolist(self):
        return self.values


@pytest.mark.parametrize("when", ["load", "generate", "observe"])
@pytest.mark.parametrize("file", ["adapter_model.safetensors", "manifest.json"])
def test_checkpoint_drift_before_forward_or_publication(worker, tmp_path, monkeypatch, when, file):
    from contextlib import nullcontext

    from active_ocr.models import ExecutionTelemetry

    p = page(tmp_path, split=Split.VALIDATION)
    ref, directory = mutable_adapter(worker, tmp_path)
    events = []

    def mutate():
        (directory / file).write_bytes(b"replacement")

    def generate(**kwargs):
        events.append("generate")
        if when == "generate":
            mutate()
        return TokenRows([10, 20, 7, q.EOS])

    def load(path):
        assert path == directory
        if when == "load":
            mutate()
        return SimpleNamespace(eval=lambda: None, generate=generate)

    def observe(started):
        worker.telemetry = ExecutionTelemetry()
        if when == "observe":
            mutate()

    monkeypatch.setattr(worker, "_verify_runtime", lambda: None)
    monkeypatch.setattr(q, "check_adapter", lambda path: {})
    monkeypatch.setattr(
        worker, "_processor_ready", lambda: SimpleNamespace(tokenizer=DecodeTokenizer())
    )
    monkeypatch.setattr(q, "encode_page", lambda *a: ({"input_ids": TokenRows([10, 20])}, {}))
    monkeypatch.setattr(worker, "_load", load)
    monkeypatch.setattr(worker, "_observe", observe)
    monkeypatch.setattr(q, "cuda_context", nullcontext)
    monkeypatch.setattr(q, "generation_config", lambda: None)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(inference_mode=nullcontext, equal=lambda a, b: a.values == b.values),
    )
    with pytest.raises(ValueError, match="checkpoint.*changed"):
        worker.predict(
            (p,),
            experiment_id="run",
            round_number=1,
            model_id=ref,
            purpose=PredictionPurpose.VALIDATION,
        )
    assert events == ([] if when == "load" else ["generate"])
    assert not (worker.output_root / "receipts").exists()


@pytest.mark.parametrize("when", ["before_load", "base_load", "adapter_load", "replace_restore"])
def test_real_load_keeps_bound_tensor_identity(worker, tmp_path, monkeypatch, when):
    _, directory = mutable_adapter(worker, tmp_path)
    weights = directory / "adapter_model.safetensors"
    calls = []
    model = SimpleNamespace(parameters=lambda: (), requires_grad_=lambda value: None)
    # Exercise the actual _load method with library spies, not GPU/tensor execution.
    monkeypatch.setattr(q, "check_adapter", lambda path: {"tensor": weights.read_bytes()})
    files = tuple(
        q.FileEntry(**f)
        for f in q.inventory(directory, ("adapter_config.json", "adapter_model.safetensors"))
    )
    worker._bind_adapter(directory, files)
    assert worker._adapter_identity[2] == {"tensor": b"original"}

    def base_load(*args, **kwargs):
        calls.append("base")
        if when == "base_load":
            weights.write_bytes(b"replaced")
        return model

    def adapter_load(base, path, **kwargs):
        calls.append("adapter")
        if when in ("adapter_load", "replace_restore"):
            weights.write_bytes(b"replaced")
        model.state = {"tensor": weights.read_bytes()}
        if when == "replace_restore":
            weights.write_bytes(b"original")
        return model

    monkeypatch.setattr(q, "check_base_modules", lambda model: None)
    monkeypatch.setattr(q, "tensor_hash", lambda tensor: tensor)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: True, is_bf16_supported=lambda **kw: True),
            backends=SimpleNamespace(
                cuda=SimpleNamespace(matmul=SimpleNamespace()), cudnn=SimpleNamespace()
            ),
            bfloat16="BF16",
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(Qwen3VLForConditionalGeneration=SimpleNamespace(from_pretrained=base_load)),
    )
    monkeypatch.setitem(
        sys.modules,
        "peft",
        SimpleNamespace(
            PeftModel=SimpleNamespace(from_pretrained=adapter_load),
            get_peft_model_state_dict=lambda model: model.state,
        ),
    )
    if when == "before_load":
        weights.write_bytes(b"replaced")
    with pytest.raises(ValueError, match="checkpoint bytes changed|loaded adapter values differ"):
        worker._load(directory)
    assert calls == ([] if when == "before_load" else ["base", "adapter"])
    assert worker._model is None


def test_generated_parser_depth_retains_raw_but_tokenizer_recursion_escapes():
    raw = '{"regions":' + "[" * 1000 + "]" * 1000 + "}"

    class NestedTokenizer:
        all_special_ids = [q.EOS]

        def decode(self, ids, **kwargs):
            return raw

    status, regions, evidence, reason = q.decode_result([7, q.EOS], NestedTokenizer(), 1, 1)
    assert status is PredictionStatus.INVALID_OUTPUT and regions == ()
    assert evidence == raw and reason == "eos"

    class BrokenTokenizer(NestedTokenizer):
        def decode(self, ids, **kwargs):
            if q.EOS not in ids:
                raise RecursionError("tokenizer infrastructure failed")
            return raw

    with pytest.raises(RecursionError, match="tokenizer infrastructure"):
        q.decode_result([7, q.EOS], BrokenTokenizer(), 1, 1)


def test_absolute_deadline_stops_before_load_or_fit(worker, tmp_path, monkeypatch):
    worker.deadline_unix_seconds = 1
    monkeypatch.setattr(q.time, "time", lambda: 2)
    with pytest.raises(TimeoutError, match="absolute run deadline"):
        worker._load()
    with pytest.raises(TimeoutError, match="absolute run deadline"):
        worker.fit(
            (RevealedExample(page=page(tmp_path), regions=()),),
            seed=824,
            experiment_id="run",
            round_number=1,
            external_reload=True,
        )


def test_probe_files_are_exclusive_full_float32_and_qwen_relative(worker, tmp_path):
    from active_ocr.integrations.modal_model import ProbeMetadata

    class Logits:
        shape = (1, 151936)

        def reshape(self, _):
            return self

        def tolist(self):
            return [0.25] * 151936

    probe = dict(
        ids=[q.EOS],
        status="invalid_output",
        regions=[],
        finish_reason="eos",
        tensors={key: "a" * 64 for key in q.adapter_shapes()},
        logits=Logits(),
    )
    model_id = "checkpoint:sha256:" + "b" * 64
    metadata = worker._write_probe(model_id, page(tmp_path), probe, "before")
    parsed = ProbeMetadata.model_validate(metadata)
    assert parsed.logits.key == "probes/" + "b" * 64 + "/before.f32le"
    data = (worker.output_root / parsed.logits.key).read_bytes()
    assert data == struct.pack("<f", 0.25) * 151936
    assert q.file_hash(worker.output_root / parsed.logits.key) == parsed.logits.sha256
    with pytest.raises(FileExistsError):
        worker._write_probe(model_id, page(tmp_path), probe, "before")
