"""Independent offline review; spies do not establish Torch/processor/CUDA behavior."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from active_ocr.integrations import qwen as q
from active_ocr.models import (
    Box,
    ExecutionTelemetry,
    ExpectedIdentity,
    Prediction,
    PredictionPurpose,
    PredictionStatus,
    RealOCRConfig,
    RevealedExample,
    RunKind,
    SimulationPage,
    SourcePolicy,
    SourceRegion,
    Split,
)
from active_ocr.pipeline import Pipeline


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(data):
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


@pytest.fixture
def worker(tmp_path):
    source, output = tmp_path / "input", tmp_path / "output"
    (source / "images").mkdir(parents=True)
    output.mkdir()
    manifest = tmp_path / "runtime.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_sha": "1" * 40,
                "files": [],
                "environment": {"python": "3.11.14", "packages": []},
            }
        )
    )
    config = RealOCRConfig(
        backend="qwen3-vl-v1",
        recipe_version="qwen3-vl-read-engineering-v1",
        model_repository="Qwen/Qwen3-VL-4B-Instruct",
        processor_repository="Qwen/Qwen3-VL-4B-Instruct",
        model_revision="ebb281ec70b05090aa6165b016eac8ec08e71b17",
        processor_revision="ebb281ec70b05090aa6165b016eac8ec08e71b17",
        training_policy_id="qwen3-vl-page-lora-v1",
        decode_policy_id="qwen3-vl-page-greedy-v1",
        expected_identity=ExpectedIdentity(
            source_sha="1" * 40,
            dependency_sha256="2" * 64,
            model_revision=q.REVISION,
            processor_revision=q.REVISION,
            model_manifest_sha256=sha(canonical(q.asset_manifest(q.BASE_FILES))),
            processor_manifest_sha256=sha(canonical(q.asset_manifest(q.PROCESSOR_FILES))),
            recipe_version=q.RECIPE,
            evaluator_id="page-text-nfc-v1",
            code_bundle_sha256="3" * 64,
            remote_dependency_sha256="4" * 64,
        ),
    )
    return q.QwenModel(source, output, config, runtime_manifest=manifest)


def make_page(worker, name="one", split=Split.TRAIN, size=(101, 1237)):
    path = worker.images_root / (name + ".png")
    Image.new("RGB", size, (11, 22, 33)).save(path)
    return SimulationPage(
        id=name,
        document_id=None,
        source_policy=SourcePolicy.READ2016,
        image_uri=str(path),
        image_sha256=sha(path.read_bytes()),
        source_image="FORBIDDEN_SOURCE_XML",
        width=size[0],
        height=size[1],
        split=split,
    )


def test_import_has_no_optional_ml_dependency():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import active_ocr.integrations.qwen; "
            "assert not {'torch','peft','transformers','torchvision','safetensors'} "
            "& sys.modules.keys()",
        ],
        check=True,
        env={**os.environ, "PYTHONPATH": str(Path(q.__file__).parents[2])},
    )


def test_literal_target_excludes_oracle_metadata(worker):
    p = make_page(worker, size=(200, 400))
    texts = [' e\u0301\r\n\t"\\ <|im_end|> ', "", "\x00", "not a refusal"]
    example = RevealedExample(
        page=p,
        regions=tuple(
            SourceRegion(
                id=f"PRIVATE_LINE_{i}",
                text=text,
                illegible=bool(i % 2),
                box=Box(x=20, y=40, width=80, height=200),
            )
            for i, text in enumerate(texts)
        ),
    )
    target = q.serialize_target(example)
    assert "<" not in target and "PRIVATE" not in target and "FORBIDDEN" not in target
    assert json.loads(target) == {
        "regions": [{"text": text, "bbox": [100, 100, 500, 600]} for text in texts]
    }
    assert [r.text for r in q.parse_regions(target, 200, 400)] == texts
    assert all(not r.illegible for r in q.parse_regions(target, 200, 400))


@pytest.mark.parametrize("width,height", [(101, 1237), (1237, 101), (3511, 2480)])
def test_normalized_edges_pass_strict_pipeline(worker, width, height):
    p = make_page(worker, size=(width, height))
    randomizer = random.Random(824)
    coords = [144.3408, 86.8629, math.nextafter(1000, 0), 0.0]
    coords.extend(randomizer.uniform(0, 1000) for _ in range(300))
    run = SimpleNamespace(id="run", kind=RunKind.REAL, dataset=SimpleNamespace(pages=(p,)))
    for x in coords:
        regions = q.parse_regions(
            json.dumps(
                {
                    "regions": [
                        {"text": "", "bbox": [x, x, 1000, 1000]},
                    ]
                }
            ),
            width,
            height,
        )
        prediction = Prediction(
            page_id=p.id, experiment_id="run", round_number=1, model_id="adapter", regions=regions
        )
        assert Pipeline._validate_simulation_predictions(
            (prediction,), (p.id,), run, 1, "adapter", require_scores=False
        ) == (prediction,)
        assert regions[0].box.x == x * width / 1000
        assert regions[0].box.y == x * height / 1000


class CharacterTokenizer:
    all_special_ids = [151643, 151645, 151655]

    def decode(self, ids, **kwargs):
        assert kwargs == {"skip_special_tokens": False, "clean_up_tokenization_spaces": False}
        return "".join("<|im_end|>" if i == 151645 else chr(i) for i in ids)


def tokens(raw):
    return [ord(c) for c in raw] + [151645]


@pytest.mark.parametrize(
    "raw",
    [
        '{"regions":[{"text":"a","text":"b","bbox":[0,0,1,1]}]}',
        '{"regions":[{"text":"a","bbox":[0,0,1e9999,1]}]}',
        '{"regions":[{"text":"\\udfff","bbox":[0,0,1,1]}]}',
        '{"regions":[]}{"regions":[]}',
    ],
)
def test_invalid_decode_retains_raw_without_repairs(raw):
    status, regions, evidence, reason = q.decode_result(
        tokens(raw), CharacterTokenizer(), 101, 1237
    )
    assert status is PredictionStatus.INVALID_OUTPUT and regions == ()
    assert evidence == raw + "<|im_end|>" and reason == "eos"


def test_nested_invalid_json_is_page_failure_not_operation_error():
    # A syntactically nested but schema-invalid response, within the 2048-token cap.
    raw = '{"regions":' + "[" * 1000 + "]" * 1000 + "}"
    ids = tokens(raw)
    assert len(ids) < 2048
    status, regions, evidence, reason = q.decode_result(ids, CharacterTokenizer(), 101, 1237)
    assert status is PredictionStatus.INVALID_OUTPUT and not regions
    assert evidence.endswith("<|im_end|>") and reason == "eos"


def test_mask_limits_and_causal_endpoints():
    prompt, target = [151655] * 2048, [72] * 4095
    full = prompt + target + [151645, 10]
    ids, labels = q.training_span(prompt, full, target, [10], {151655, 151645})
    assert len(ids) == 6144 and labels[:2048] == [-100] * 2048
    assert labels[2048:] == target + [151645]
    assert [(2047, labels[2048]), (6142, labels[6143])] == [(2047, 72), (6142, 151645)]
    for p, t in ((prompt + [4], target), (prompt, target + [4])):
        with pytest.raises(ValueError, match="overflow"):
            q.training_span(p, p + t + [151645, 10], t, [10], {151655, 151645})
    with pytest.raises(ValueError, match="prefix"):
        q.training_span([1, 2], [2, 1, 3, 151645, 10], [3], [10], {151645})


def test_real_identity_failure_precedes_optional_model_import(worker, monkeypatch):
    monkeypatch.setattr(worker, "_processor_ready", lambda: pytest.fail("processor reached"))
    monkeypatch.setattr(worker, "_load", lambda *a: pytest.fail("model reached"))
    with pytest.raises(ValueError, match="executing module"):
        worker.load_base(experiment_id="run")
    assert list(worker.output_root.iterdir()) == []


@pytest.mark.parametrize("split", [Split.VALIDATION, Split.TEST])
def test_held_out_targets_stop_before_preparation(worker, monkeypatch, split):
    p = make_page(worker)
    if split is Split.TEST:
        # Strict known-group metadata permits a TEST page; runtime must still exclude it.
        p = p.model_copy(
            update={
                "split": split,
                "source_policy": SourcePolicy.KNOWN_DOCUMENT,
                "document_id": "held-out",
            }
        )
    else:
        p = p.model_copy(update={"split": split})
    monkeypatch.setattr(worker, "_verify_runtime", lambda: None)
    monkeypatch.setattr(worker, "_processor_ready", lambda: pytest.fail("processor reached"))
    with pytest.raises(ValueError, match="split"):
        worker.fit(
            (RevealedExample(page=p, regions=()),), seed=824, experiment_id="run", round_number=1
        )
    assert list(worker.output_root.iterdir()) == []


def test_every_selected_target_preflighted_before_load(worker, monkeypatch):
    examples = tuple(
        RevealedExample(
            page=make_page(worker, name, size=(320, 480)),
            regions=(
                SourceRegion(id="secret-id", text=name, box=Box(x=0, y=0, width=1, height=1)),
            ),
        )
        for name in ("selected-A", "selected-B")
    )
    observed = []
    monkeypatch.setattr(worker, "_verify_runtime", lambda: None)
    monkeypatch.setattr(worker, "_processor_ready", lambda: object())
    monkeypatch.setattr(worker, "_load", lambda: pytest.fail("load before full preflight"))

    def encode(processor, image, target):
        observed.append(json.loads(target)["regions"][0]["text"])
        if len(observed) == 2:
            raise ValueError("independent overflow sentinel")
        return {}, {}

    monkeypatch.setattr(q, "encode_page", encode)
    with pytest.raises(ValueError, match="overflow sentinel"):
        worker.fit(examples, seed=824, experiment_id="run", round_number=1)
    assert observed == ["selected-A", "selected-B"]
    assert not list(worker.output_root.iterdir())


class Matrix:
    """Only the two-dimensional token operations used by predict; no ML simulation."""

    def __init__(self, rows):
        self.rows = rows
        self.shape = (len(rows), len(rows[0]))

    def to(self, device):
        assert device == "cuda:0"
        return self

    def __getitem__(self, key):
        row, col = key
        if isinstance(row, slice):
            return Matrix([r[col] for r in self.rows[row]])
        return SimpleNamespace(tolist=lambda: self.rows[row][col])


def adapter_record(worker, tmp_path):
    stage = tmp_path / "adapter-stage"
    stage.mkdir()
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        (stage / name).write_bytes(b"synthetic tensor boundary; never loaded as ML")
    processing = dict(
        height=256,
        width=256,
        grid=[1, 16, 16],
        prompt_tokens=100,
        target_tokens=4,
        target_exceeds_decode_budget=False,
    )
    checkpoint = q.Checkpoint(
        kind="adapter",
        binding=worker._binding(),
        experiment_id="run",
        round_number=1,
        selected=(("selected-A", "a" * 64),),
        target_sha256="b" * 64,
        seed=824,
        training=dict(
            updates=3,
            epoch_orders=[["selected-A"]] * 3,
            losses=[1.0] * 3,
            gradient_norms=[0.1] * 3,
            processing={"selected-A": processing},
            supervised_tokens=12,
            changed_tensors=72,
            frozen_sha256="c" * 64,
        ),
        files=tuple(
            q.FileEntry(**entry)
            for entry in q.inventory(stage, ["adapter_config.json", "adapter_model.safetensors"])
        ),
    )
    reference = q.publish_checkpoint(worker.output_root, checkpoint, stage)
    return reference, worker.output_root / "checkpoints" / reference.rsplit(":", 1)[1]


def generation_spy(worker, monkeypatch, before_load=None, during_generate=None):
    events = []
    tokenizer = CharacterTokenizer()
    monkeypatch.setattr(worker, "_verify_runtime", lambda: events.append("identity"))
    monkeypatch.setattr(worker, "_processor_ready", lambda: SimpleNamespace(tokenizer=tokenizer))
    monkeypatch.setattr(q, "check_adapter", lambda path: {})  # Tensor execution remains gated.
    monkeypatch.setattr(q, "cuda_context", nullcontext)
    monkeypatch.setattr(q, "generation_config", lambda: "explicit-config")
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            inference_mode=nullcontext,
            equal=lambda a, b: a.rows == b.rows,
        ),
    )

    def encode(processor, image):
        events.append(("pixels", image.size))
        if before_load:
            before_load()
        return {"input_ids": Matrix([[10, 20]])}, {"source": "synthetic-processing"}

    def load(path=None):
        events.append(("load", path))

        def generate(**kwargs):
            if during_generate:
                during_generate()
            assert kwargs["generation_config"] == "explicit-config"
            return Matrix([[10, 20] + tokens('{"regions":[]}')])

        return SimpleNamespace(eval=lambda: None, generate=generate)

    def observe(started):
        worker.telemetry = ExecutionTelemetry(device="test spy, not CUDA")

    monkeypatch.setattr(q, "encode_page", encode)
    monkeypatch.setattr(worker, "_load", load)
    monkeypatch.setattr(worker, "_observe", observe)
    return events


def test_predict_restores_requested_base_and_reads_no_oracle_fields(worker, monkeypatch, tmp_path):
    p = make_page(worker, split=Split.VALIDATION)
    reference, directory = adapter_record(worker, tmp_path)
    base = q.publish_checkpoint(
        worker.output_root, q.Checkpoint(kind="base", binding=worker._binding())
    )
    events = generation_spy(worker, monkeypatch)
    for model_id, number, purpose in (
        (reference, 1, PredictionPurpose.VALIDATION),
        (base, 0, PredictionPurpose.BASELINE_VALIDATION),
    ):
        result = worker.predict(
            (p,), experiment_id="run", round_number=number, model_id=model_id, purpose=purpose
        )
        assert len(result) == 1 and result[0].regions == ()
        assert result[0].purpose is purpose and result[0].confidence is None
        evidence = Path(result[0].raw_output_artifact).read_text()
        assert "FORBIDDEN" not in evidence and "document_id" not in evidence
    assert [event for event in events if isinstance(event, tuple) and event[0] == "load"] == [
        ("load", directory),
        ("load", None),
    ]


def test_image_mutation_during_generation_publishes_no_receipt(worker, monkeypatch):
    p = make_page(worker, split=Split.VALIDATION)
    base = q.publish_checkpoint(
        worker.output_root, q.Checkpoint(kind="base", binding=worker._binding())
    )
    generation_spy(
        worker, monkeypatch, during_generate=lambda: Path(p.image_uri).write_bytes(b"changed")
    )
    with pytest.raises(ValueError, match="image bytes changed"):
        worker.predict(
            (p,),
            experiment_id="run",
            round_number=0,
            model_id=base,
            purpose=PredictionPurpose.BASELINE_VALIDATION,
        )
    assert not (worker.output_root / "receipts").exists()


def test_adapter_mutation_between_validation_and_load_is_rejected(worker, monkeypatch, tmp_path):
    p = make_page(worker, split=Split.VALIDATION)
    reference, directory = adapter_record(worker, tmp_path)
    weights = directory / "adapter_model.safetensors"
    events = generation_spy(
        worker, monkeypatch, before_load=lambda: weights.write_bytes(b"different adapter")
    )
    # Actual byte/manifest validation stays active; only ML tensor checks/load are spies.
    # A different valid tensor payload would pass check_adapter, which checks shape/config/finite
    # values but does not bind its new hashes to the earlier manifest inventory.
    with pytest.raises(ValueError, match="checkpoint bytes changed"):
        worker.predict(
            (p,),
            experiment_id="run",
            round_number=1,
            model_id=reference,
            purpose=PredictionPurpose.VALIDATION,
        )
    assert not any(isinstance(e, tuple) and e[0] == "load" for e in events)
    assert not (worker.output_root / "receipts").exists()


@pytest.mark.parametrize("change", ["extra", "deleted", "modified", "symlink"])
def test_real_checkpoint_inventory_rejects_drift(worker, tmp_path, monkeypatch, change):
    reference, directory = adapter_record(worker, tmp_path)
    monkeypatch.setattr(q, "check_adapter", lambda path: pytest.fail("tensor load reached"))
    weights = directory / "adapter_model.safetensors"
    if change == "extra":
        (directory / "unreviewed").write_bytes(b"x")
    elif change == "deleted":
        weights.unlink()
    elif change == "modified":
        weights.write_bytes(b"x")
    else:
        copy = tmp_path / "other"
        copy.write_bytes(weights.read_bytes())
        weights.unlink()
        weights.symlink_to(copy)
    with pytest.raises((ValueError, OSError)):
        worker._checkpoint(reference, "run", 1)


def test_adapter_owner_and_round_rejected_before_tensor_load(worker, tmp_path, monkeypatch):
    reference, _ = adapter_record(worker, tmp_path)
    monkeypatch.setattr(q, "check_adapter", lambda path: pytest.fail("tensor load reached"))
    for owner, number in (("other", 1), ("run", 0), ("run", 2)):
        with pytest.raises(ValueError, match="ownership"):
            worker._checkpoint(reference, owner, number)


def test_training_loop_partial_group_scaling_inside_backward_context(monkeypatch):
    # Lightweight arithmetic/call spy only; actual autograd/Adam numerical checks stay gated.
    active, groups, current, options = [], [], [], {}

    class Loss:
        def __init__(self, value, denominator=1):
            self.value, self.denominator = value, denominator

        def detach(self):
            return self.value

        def __truediv__(self, denominator):
            return Loss(self.value, denominator)

        def backward(self):
            assert active == [True]
            current.append(self.denominator)

    class Context:
        def __enter__(self):
            active.append(True)

        def __exit__(self, *args):
            active.pop()

    class Optimizer:
        def __init__(self, parameters, **kwargs):
            options.update(kwargs)
            assert parameters == [parameter]

        def zero_grad(self, *, set_to_none):
            assert set_to_none
            current.clear()

        def step(self):
            groups.append(tuple(current))

    class Model:
        config = SimpleNamespace(use_cache=True)

        def train(self):
            pass

        def gradient_checkpointing_enable(self, **kwargs):
            assert kwargs == {"gradient_checkpointing_kwargs": {"use_reentrant": False}}

        def __call__(self, **kwargs):
            assert active == [True] and kwargs["use_cache"] is False
            return SimpleNamespace(loss=Loss(float(kwargs["value"])))

    finite = SimpleNamespace(item=lambda: True)
    finite.all = lambda: finite
    parameter = SimpleNamespace(grad=1)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            optim=SimpleNamespace(AdamW=Optimizer),
            isfinite=lambda _: finite,
            nn=SimpleNamespace(utils=SimpleNamespace(clip_grad_norm_=lambda *a, **k: 0.5)),
        ),
    )
    model = Model()
    result = q.train_epochs(
        model, {str(i): {"value": i} for i in range(6)}, 824, [parameter], Context
    )
    assert groups == [(4, 4, 4, 4), (2, 2)] * 3
    assert result["updates"] == 6 and len(result["losses"]) == 18
    assert model.config.use_cache is False
    assert options == dict(
        lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, foreach=False, fused=False
    )


def test_original_size_is_used_by_reload_probe(worker, monkeypatch):
    # Exercise the actual probe's call path with token/logit spies, without model packages.
    events = generation_spy(worker, monkeypatch)
    del events
    raw = '{"regions":[{"text":"x","bbox":[100,200,1000,1000]}]}'
    prompt = Matrix([[10, 20]])
    generated = Matrix([[10, 20] + tokens(raw)])
    logits = SimpleNamespace()
    logits.float = lambda: logits
    logits.cpu = lambda: logits

    class Logits:
        def __getitem__(self, key):
            assert key == (0, slice(1, None))
            return logits

    class ProbeModel:
        def eval(self):
            pass

        def generate(self, **kwargs):
            return generated

        def __call__(self, **kwargs):
            assert kwargs["input_ids"].shape == (1, 9)  # P + min(8,T) - 1
            assert kwargs["attention_mask"].shape == kwargs["input_ids"].shape
            assert kwargs["mm_token_type_ids"].shape == kwargs["input_ids"].shape
            return SimpleNamespace(logits=Logits())

    monkeypatch.setattr(
        Matrix, "new_full", lambda self, shape, fill: Matrix([[fill] * shape[1]]), raising=False
    )
    torch = sys.modules["torch"]
    torch.cat = lambda matrices, dim: Matrix([sum((m.rows[0] for m in matrices), [])])
    torch.isfinite = lambda value: SimpleNamespace(all=lambda: SimpleNamespace(item=lambda: True))
    monkeypatch.setitem(
        sys.modules, "peft", SimpleNamespace(get_peft_model_state_dict=lambda m: {})
    )
    result = q.reload_probe(
        ProbeModel(),
        {
            "input_ids": prompt,
            "attention_mask": Matrix([[1, 1]]),
            "mm_token_type_ids": Matrix([[0, 1]]),
        },
        CharacterTokenizer(),
        original_size=(101, 1237),
    )
    assert result["regions"][0]["box"] == {
        "x": 10.1,
        "y": 247.4,
        "width": 90.9,
        "height": 989.6,
    }
    assert result["ids"] == tokens(raw) and result["status"] == "ok"


def test_checkpoint_interruption_never_becomes_complete(worker, tmp_path, monkeypatch):
    reference, directory = adapter_record(worker, tmp_path)
    manifest = directory / "manifest.json"
    checkpoint = q.Checkpoint.model_validate_json(manifest.read_bytes())
    manifest.unlink()  # Simulate interruption before manifest-last publication.
    with pytest.raises((OSError, ValueError)):
        q.publish_checkpoint(worker.output_root, checkpoint, tmp_path / "adapter-stage")
    assert not manifest.exists()
    assert (directory / "adapter_model.safetensors").is_file()
    with pytest.raises((OSError, ValueError)):
        worker._checkpoint(reference, "run", 1)


@pytest.mark.parametrize("point", ["load", "generate", "observe"])
@pytest.mark.parametrize("filename", ["manifest.json", "adapter_model.safetensors"])
def test_phase2_prediction_drift_stops_publication(worker, tmp_path, monkeypatch, point, filename):
    p = make_page(worker, split=Split.VALIDATION)
    reference, directory = adapter_record(worker, tmp_path)
    changed = directory / filename
    events = generation_spy(
        worker,
        monkeypatch,
        during_generate=(lambda: changed.write_bytes(b"drift")) if point == "generate" else None,
    )
    if point == "load":
        load = worker._load

        def changed_load(path):
            model = load(path)
            changed.write_bytes(b"drift")
            model.generate = lambda **kwargs: pytest.fail("forward after checkpoint drift")
            return model

        monkeypatch.setattr(worker, "_load", changed_load)
    if point == "observe":
        observe = worker._observe

        def changed_observe(start):
            observe(start)
            changed.write_bytes(b"drift")

        monkeypatch.setattr(worker, "_observe", changed_observe)
    with pytest.raises(ValueError, match="checkpoint.*changed"):
        worker.predict(
            (p,),
            experiment_id="run",
            round_number=1,
            model_id=reference,
            purpose=PredictionPurpose.VALIDATION,
        )
    assert not (worker.output_root / "receipts").exists()
    assert ("load", directory) in events


@pytest.mark.parametrize("mode", ["unchanged", "replace_restore", "change_after_tensor_hash"])
def test_phase2_actual_loader_uses_retained_tensor_identity(worker, tmp_path, monkeypatch, mode):
    # Actual _bind_adapter/_load and file guards; packages and tensor values are explicit spies.
    _, directory = adapter_record(worker, tmp_path)
    weights = directory / "adapter_model.safetensors"
    original = weights.read_bytes()
    model = SimpleNamespace(
        parameters=lambda: (SimpleNamespace(device="cuda:0", dtype="bf16"),),
        requires_grad_=lambda flag: None,
        named_modules=lambda: (
            (
                name,
                SimpleNamespace(
                    weight=SimpleNamespace(shape=(4096 if name.endswith("q_proj") else 1024, 2560))
                ),
            )
            for name in q.TARGETS
        ),
    )
    reads = []

    def validate(path):
        reads.append(path)
        return {"parameter": sha((path / "adapter_model.safetensors").read_bytes())}

    monkeypatch.setattr(q, "check_adapter", validate)
    files = tuple(
        q.FileEntry(**item)
        for item in q.inventory(directory, ("adapter_config.json", "adapter_model.safetensors"))
    )
    worker._bind_adapter(directory, files)
    expected = dict(worker._adapter_identity[2])

    def base_loader(path, **kwargs):
        assert path == worker.model_root
        assert kwargs["local_files_only"] and not kwargs["trust_remote_code"]
        assert kwargs["device_map"] == {"": "cuda:0"}
        return model

    def adapter_loader(base, path, **kwargs):
        assert base is model and path == directory
        assert kwargs["is_trainable"] is False and kwargs["local_files_only"]
        if mode == "replace_restore":
            weights.write_bytes(b"different finite tensor stand-in")
        model.state = {"parameter": weights.read_bytes()}
        weights.write_bytes(original)
        return model

    def tensor_digest(value):
        if mode == "change_after_tensor_hash":
            weights.write_bytes(b"drift after loaded tensor capture")
        return sha(value)

    monkeypatch.setattr(q, "tensor_hash", tensor_digest)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: True, is_bf16_supported=lambda **k: True),
            backends=SimpleNamespace(
                cuda=SimpleNamespace(matmul=SimpleNamespace()), cudnn=SimpleNamespace()
            ),
            bfloat16="bf16",
            device=lambda value: value,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            Qwen3VLForConditionalGeneration=SimpleNamespace(from_pretrained=base_loader)
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "peft",
        SimpleNamespace(
            PeftModel=SimpleNamespace(from_pretrained=adapter_loader),
            get_peft_model_state_dict=lambda m: m.state,
        ),
    )
    if mode == "unchanged":
        assert worker._load(directory) is model
        # A real base load clears the earlier adapter binding instead of warm-starting it.
        monkeypatch.setattr(worker, "_drop", lambda: setattr(worker, "_model", None))
        assert worker._load() is model and worker._adapter_identity is None
    else:
        with pytest.raises(
            ValueError, match="loaded adapter values differ|checkpoint bytes changed"
        ):
            worker._load(directory)
        assert worker._model is None
        assert worker._adapter_identity[2] == expected
    assert reads == [directory]  # Load must not replace the frozen tensor expectation.


def test_phase2_failed_binding_cannot_reuse_previous_identity(worker, tmp_path, monkeypatch):
    _, directory = adapter_record(worker, tmp_path)
    weights = directory / "adapter_model.safetensors"
    files = tuple(
        q.FileEntry(**item)
        for item in q.inventory(directory, ("adapter_config.json", "adapter_model.safetensors"))
    )
    monkeypatch.setattr(q, "check_adapter", lambda path: {"parameter": sha(weights.read_bytes())})
    worker._bind_adapter(directory, files)

    def changing_validator(path):
        weights.write_bytes(b"changed during tensor verification")
        return {"parameter": sha(weights.read_bytes())}

    monkeypatch.setattr(q, "check_adapter", changing_validator)
    with pytest.raises(ValueError, match="checkpoint bytes changed"):
        worker._bind_adapter(directory, files)
    assert worker._adapter_identity is None


@pytest.mark.parametrize("stage", ["raw_decode", "content_decode"])
def test_phase2_tokenizer_recursion_still_fails_operation(stage):
    class BrokenTokenizer(CharacterTokenizer):
        def decode(self, ids, **kwargs):
            if stage == "raw_decode" or 151645 not in ids:
                raise RecursionError("tokenizer operation failure")
            return super().decode(ids, **kwargs)

    with pytest.raises(RecursionError, match="tokenizer operation failure"):
        q.decode_result(tokens('{"regions":[]}'), BrokenTokenizer(), 101, 1237)


@pytest.mark.parametrize("drift", [None, "after_reload", "receipt"])
def test_phase2_fit_retains_staged_inventory_until_publication(worker, monkeypatch, drift):
    # Exercise actual fit/save/bind/publish control flow, not training or tensor numerics.
    p = make_page(worker, size=(320, 480))
    example = RevealedExample(page=p, regions=())
    saved = []
    model = SimpleNamespace(state={"parameter": b"initial"})
    processing = dict(
        height=480,
        width=320,
        grid=[1, 30, 20],
        prompt_tokens=200,
        target_tokens=4,
        target_exceeds_decode_budget=False,
    )

    def save(stage, **kwargs):
        saved.append(stage)
        (stage / "adapter_config.json").write_text("{}")
        (stage / "adapter_model.safetensors").write_bytes(model.state["parameter"])

    model.save_pretrained = save

    def load(stage=None):
        if stage is None:
            return model
        assert worker._adapter_identity[0] == stage
        q.verify_adapter_files(stage, worker._adapter_identity[1])
        return SimpleNamespace(
            state={"parameter": (stage / "adapter_model.safetensors").read_bytes()}
        )

    def train(*args):
        model.state = {"parameter": b"trained"}
        return dict(
            updates=3, epoch_orders=[[p.id]] * 3, losses=[1.0] * 3, gradient_norms=[0.25] * 3
        )

    def compare(before, after):
        assert before == after
        if drift == "after_reload":
            (saved[0] / "adapter_model.safetensors").write_bytes(b"drift after reload")
        return 0.0

    def receipt(data):
        if drift == "receipt":
            (saved[0] / "adapter_model.safetensors").write_bytes(b"drift during receipt")
        return "synthetic-receipt"

    monkeypatch.setattr(worker, "_verify_runtime", lambda: None)
    monkeypatch.setattr(worker, "_processor_ready", lambda: SimpleNamespace(tokenizer=object()))
    monkeypatch.setattr(worker, "_drop", lambda: setattr(worker, "_model", None))
    monkeypatch.setattr(worker, "_load", load)
    monkeypatch.setattr(
        worker, "_observe", lambda start: setattr(worker, "telemetry", ExecutionTelemetry())
    )
    monkeypatch.setattr(worker, "_receipt", receipt)
    monkeypatch.setattr(q, "reset_seed", lambda seed: None)
    monkeypatch.setattr(q, "encode_page", lambda *args: ({}, processing))
    monkeypatch.setattr(q, "check_trainables", lambda m: [])
    monkeypatch.setattr(q, "frozen_hashes", lambda m: {"frozen": "unchanged"})
    monkeypatch.setattr(q, "tensor_hash", sha)
    monkeypatch.setattr(q, "train_epochs", train)
    monkeypatch.setattr(
        q,
        "check_adapter",
        lambda stage: {"parameter": sha((stage / "adapter_model.safetensors").read_bytes())},
    )
    monkeypatch.setattr(q, "reload_probe", lambda m, *args, **kwargs: dict(m.state))
    monkeypatch.setattr(q, "compare_probes", compare)
    monkeypatch.setitem(
        sys.modules,
        "peft",
        SimpleNamespace(
            get_peft_model=lambda base, *args, **kwargs: base,
            get_peft_model_state_dict=lambda m: m.state,
        ),
    )
    monkeypatch.setattr(q, "lora_config", lambda: object())
    if drift is None:
        reference = worker.fit((example,), seed=824, experiment_id="run", round_number=1)
        directory = worker.output_root / "checkpoints" / reference.rsplit(":", 1)[1]
        checkpoint = json.loads((directory / "manifest.json").read_text())
        assert checkpoint["files"] == [entry.model_dump() for entry in worker._adapter_identity[1]]
        assert checkpoint["selected"] == [[p.id, p.image_sha256]]
    else:
        with pytest.raises(ValueError, match="checkpoint bytes changed|staged adapter changed"):
            worker.fit((example,), seed=824, experiment_id="run", round_number=1)
        assert not list(worker.output_root.glob("checkpoints/*/manifest.json"))
    assert not saved[0].exists()  # Private staging cleaned, no completed corrupt checkpoint.
