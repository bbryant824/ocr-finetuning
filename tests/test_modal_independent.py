"""Independent API/recovery checks. Synthetic records are not provider/runtime evidence."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import struct
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from pydantic import ValidationError

from active_ocr.integrations import artifacts as ar
from active_ocr.integrations import factory_model as fm
from active_ocr.integrations import modal_model as m

CODE_FILES = (
    "active_ocr/__init__.py",
    "active_ocr/entrypoints/__init__.py",
    "active_ocr/entrypoints/modal_app.py",
    "active_ocr/integrations/__init__.py",
    "active_ocr/integrations/artifacts.py",
    "active_ocr/integrations/factory_model.py",
    "active_ocr/integrations/modal_model.py",
    "active_ocr/models.py",
)
# Independent expectations: this suite states the pinned names itself instead of importing them.
RECIPE_FILE = "experiments/recipes/read2016-llamafactory-joint.json"
CPU_TEST_FILE = "tests/test_factory_model.py"
NAMESPACE = "factory"
BACKEND = "llamafactory-qwen3vl-v1"
TOOLKIT_VERSION = "0.9.5"
EPOCHS = 3.0
PACKAGES = {"llamafactory": TOOLKIT_VERSION, "modal": "1.5.5"}
EVALUATOR = "page-joint-nfc-iou50-v1"
RUN, ATTEMPT, EXECUTION = "1" * 32, "2" * 32, "3" * 32
CHECKPOINT = "checkpoint:sha256:" + "4" * 64
SECRET = 'SELECTED_ONLY e\u0301\r\n"\\ <|im_end|>'
# A probe records one raw response digest, never the response text or any label.
PROBE_RESPONSE = '{"regions":[{"text":"partial'


def encoded(value):
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def artifact(key, sha="a" * 64, size=19):
    return dict(key=key, bytes=size, sha256=sha)


def safetensors_adapter(*, nonzero=True):
    """Two small LoRA tensors in real safetensors framing, written with stdlib only.

    This is byte-level closure evidence for the adapter reader; it is not a trained adapter
    and makes no claim about LoRA topology, which belongs to the toolkit's own gated checks.
    """
    header, blocks, offset = {}, [], 0
    for name, shape, value in (
        ("base_model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight", [8, 16], 0.0),
        (
            "base_model.model.language_model.layers.0.self_attn.q_proj.lora_B.weight",
            [16, 8],
            0.125 if nonzero else 0.0,
        ),
    ):
        raw = struct.pack("<f", value) * (shape[0] * shape[1])
        header[name] = dict(dtype="F32", shape=shape, data_offsets=[offset, offset + len(raw)])
        blocks.append(raw)
        offset += len(raw)
    body = encoded(header)
    return len(body).to_bytes(8, "little") + body + b"".join(blocks)


@pytest.fixture(scope="session")
def adapter_bytes():
    return safetensors_adapter()


@pytest.fixture
def settings():
    # Hash-only metadata; this fixture makes no claim that these files were built or uploaded.
    bundle = m.InputBundle(
        files=sorted(
            [
                dict(filename="model/" + name, bytes=size, sha256=sha)
                for name, (size, sha) in fm.ASSETS.items()
            ]
            + [dict(filename="images/" + ch * 64, bytes=200, sha256=ch * 64) for ch in "ab"],
            key=lambda entry: entry["filename"],
        )
    )
    spec = m.BuildSpec(
        source_sha="c" * 40,
        code_files=[dict(filename=name, bytes=10, sha256="d" * 64) for name in CODE_FILES],
        lock_sha256="e" * 64,
        requirements_file=dict(filename="requirements-linux.txt", bytes=10, sha256="e" * 64),
        cpu_test_file=dict(filename=CPU_TEST_FILE, bytes=10, sha256="f" * 64),
        recipe_file=dict(filename=RECIPE_FILE, bytes=10, sha256="1" * 64),
        processor_manifest_sha256=digest(fm.asset_manifest(fm.PROCESSOR_FILES)),
    )
    environment = dict(python="3.11.14", packages=sorted(PACKAGES.items()))
    build = m.BuildReceipt(
        build_spec_sha256=spec.sha256,
        source_sha=spec.source_sha,
        code_files=spec.code_files,
        environment=environment,
        cpu_report=artifact("cpu-report.json"),
        cpu_passed=7,
        cpu_skipped=0,
    )
    identity = dict(
        source_sha=spec.source_sha,
        dependency_sha256="f" * 64,
        model_revision=fm.REVISION,
        processor_revision=fm.REVISION,
        model_manifest_sha256=digest(fm.asset_manifest(fm.BASE_FILES)),
        processor_manifest_sha256=spec.processor_manifest_sha256,
        recipe_version=fm.recipe_field("recipe_version"),
        evaluator_id=EVALUATOR,
        code_bundle_sha256=digest(
            dict(
                source_sha=spec.source_sha,
                files=[f.model_dump(mode="json") for f in spec.code_files],
            )
        ),
        remote_dependency_sha256=digest(environment),
        build_spec_sha256=spec.sha256,
        deployment_reference="modal:test/review/dispatch@im-synthetic",
    )
    real = dict(
        backend=BACKEND,
        recipe_version=fm.recipe_field("recipe_version"),
        model_repository=fm.REPOSITORY,
        processor_repository=fm.REPOSITORY,
        model_revision=fm.REVISION,
        processor_revision=fm.REVISION,
        training_policy_id=fm.recipe_field("training_policy_id"),
        decode_policy_id=fm.recipe_field("decode_policy_id"),
        evaluator_id=EVALUATOR,
        expected_identity=identity,
    )
    return m.RuntimeSettings(
        context=dict(
            experiment_id=RUN,
            source_policy="read2016-official-unknown-engineering-v1",
            bundle_sha256=bundle.sha256,
            real_config=real,
        ),
        environment_name="test",
        deployment_name="review",
        image_id="im-synthetic",
        input_volume_name="read-input",
        output_volume_name="write-output",
        input_volume_id="vo-input",
        output_volume_id="vo-output",
        build_spec=spec,
        build=build,
        bundle=bundle,
    )


def page(name="first", split="train", letter="a"):
    return dict(
        id=name,
        document_id=None,
        split=split,
        width=100,
        height=200,
        image_sha256=letter * 64,
        image_key="images/" + letter * 64,
    )


def fit(settings):
    return m.FitRequest(
        context=settings.context,
        round_number=1,
        input_checkpoint=CHECKPOINT,
        seed=824,
        examples=[
            dict(
                page=page(),
                regions=[
                    dict(
                        id="line",
                        text=SECRET,
                        illegible=False,
                        box=dict(x=0, y=10, width=100, height=190),
                    )
                ],
            )
        ],
    )


def predict(settings):
    return m.PredictRequest(
        context=settings.context,
        round_number=0,
        purpose="baseline_validation",
        input_checkpoint=CHECKPOINT,
        pages=[page(split="validation"), page("second", "validation", "b")],
    )


def invoke(request, **changes):
    return m.Invocation.model_validate(
        dict(
            request=request.model_dump(mode="json"),
            attempt_id=ATTEMPT,
            first_submission_unix_seconds=1000,
            deadline_unix_seconds=3400,
            **changes,
        )
    )


def completed(call):
    request = call.request
    worker = artifact("operations/worker.json")
    raw = artifact(f"{NAMESPACE}/receipts/raw.json", "b" * 64)
    manifest = artifact(f"{NAMESPACE}/checkpoints/" + "4" * 64 + "/manifest.json", "4" * 64)
    return m.OperationResult(
        operation_id=call.operation_id,
        attempt_id=call.attempt_id,
        execution_id=EXECUTION,
        provider_call_id="fc-synthetic",
        model_id=CHECKPOINT,
        artifacts=[worker, raw, manifest],
        worker_manifest=worker,
        predictions=[
            dict(
                page_id=p.id,
                experiment_id=RUN,
                round_number=request.round_number,
                purpose=request.purpose,
                model_id=CHECKPOINT,
                raw_output_artifact=raw["key"],
            )
            for p in request.pages
        ],
    )


def test_independent_semantic_digest_and_redaction(settings):
    request = fit(settings)
    wire = request.model_dump(mode="json")
    expected = {key: value for key, value in wire.items() if key != "examples"}
    expected["pages"] = [wire["examples"][0]["page"]]
    expected["target_sha256"] = digest(
        [dict(page_id="first", regions=wire["examples"][0]["regions"])]
    )
    assert request.semantic_record() == expected
    assert m.operation_id(request) == digest(dict(schema_version=1, request=expected))
    call = invoke(request)
    assert m.Invocation.model_validate_json(call.model_dump_json()) == call
    assert SECRET not in repr(call)
    assert "SELECTED_ONLY" not in json.dumps(expected) and "regions" not in expected
    assert (
        json.loads(call.model_dump_json())["request"]["examples"][0]["regions"][0]["text"] == SECRET
    )
    changed = call.model_dump(mode="json")
    changed.update(
        attempt_id="5" * 32, first_submission_unix_seconds=2000, deadline_unix_seconds=4300
    )
    assert m.Invocation.model_validate(changed).operation_id == call.operation_id


@pytest.mark.parametrize("field", ["text", "box", "order", "seed", "checkpoint", "policy", "image"])
def test_semantics_bind_actual_inputs(settings, field):
    original = fit(settings)
    data = original.model_dump(mode="json")
    if field == "text":
        data["examples"][0]["regions"][0]["text"] = "changed"
    elif field == "box":
        data["examples"][0]["regions"][0]["box"]["x"] = 1
        data["examples"][0]["regions"][0]["box"]["width"] = 99
    elif field == "order":
        extra = copy.deepcopy(data["examples"][0])
        extra["page"] = page("second", letter="b")
        original = m.FitRequest.model_validate({**data, "examples": data["examples"] + [extra]})
        data["examples"] = [extra] + data["examples"]
    elif field == "seed":
        data["seed"] = 825
    elif field == "checkpoint":
        data["input_checkpoint"] = "checkpoint:sha256:" + "7" * 64
    elif field == "policy":
        data["context"]["source_policy"] = "known-document-v1"
        data["examples"][0]["page"]["document_id"] = "known"
    else:
        data["examples"][0]["page"].update(image_sha256="b" * 64, image_key="images/" + "b" * 64)
    assert m.operation_id(m.FitRequest.model_validate(data)) != m.operation_id(original)


@pytest.mark.parametrize(
    "case",
    [
        "omitted_document",
        "omitted_policy",
        "fake_document",
        "strict_null",
        "duplicate",
        "validation",
        "test",
        "overflow",
    ],
)
def test_request_policy_and_selected_geometry(settings, case):
    data = fit(settings).model_dump(mode="json")
    remote = data["examples"][0]["page"]
    if case == "omitted_document":
        del remote["document_id"]
    elif case == "omitted_policy":
        del data["context"]["source_policy"]
    elif case == "fake_document":
        remote["document_id"] = "invented"
    elif case == "strict_null":
        data["context"]["source_policy"] = "known-document-v1"
    elif case == "duplicate":
        data["examples"].append(copy.deepcopy(data["examples"][0]))
    elif case in ("test", "validation"):
        remote["split"] = case
    else:
        data["examples"][0]["regions"][0]["box"]["height"] = 191
    with pytest.raises(ValidationError):
        m.REQUEST_ADAPTER.validate_python(data)


@pytest.mark.parametrize("operation", ["load_base", "predict"])
def test_labels_cannot_enter_other_requests(settings, operation):
    data = (
        m.BaseRequest(context=settings.context) if operation == "load_base" else predict(settings)
    ).model_dump(mode="json")
    data["examples"] = fit(settings).model_dump(mode="json")["examples"]
    with pytest.raises(ValidationError):
        m.REQUEST_ADAPTER.validate_python(data)


@pytest.mark.parametrize("field", ["source_image", "provenance", "regions", "image_uri", "text"])
def test_remote_page_does_not_admit_oracle_fields(field):
    with pytest.raises(ValidationError):
        m.RemotePage.model_validate({**page(), field: SECRET})


def test_bootstrap_hashes_and_paths_without_runtime_claim(settings):
    assert set(CODE_FILES) == m.RUNTIME_CODE_FILES
    assert m.MODEL_NAMESPACE == NAMESPACE
    assert settings.input_sub_path == "/bundles/" + digest(settings.bundle.model_dump(mode="json"))
    assert settings.output_sub_path == "/runs/" + RUN
    assert settings.build.code_bundle_sha256 == digest(
        dict(
            source_sha=settings.build.source_sha,
            files=[f.model_dump(mode="json") for f in settings.build.code_files],
        )
    )
    assert settings.build.remote_dependency_sha256 == digest(
        settings.build.environment.model_dump(mode="json")
    )
    settings.check_invocation(invoke(fit(settings)))
    # Same byte address under an unlisted image cannot be admitted solely by request metadata.
    changed = predict(settings).model_dump(mode="json")
    changed["pages"][0].update(image_key="images/" + "8" * 64, image_sha256="8" * 64)
    with pytest.raises(ValueError, match="input bundle"):
        settings.check_invocation(invoke(m.PredictRequest.model_validate(changed)))


@pytest.mark.parametrize(
    "change",
    [
        "image",
        "volumes",
        "code",
        "source",
        "package",
        "report_skip",
        "report_zero",
        "lock",
        "requirements",
        "cpu_test",
        "recipe",
        "processor",
        "allowlist",
    ],
)
def test_bootstrap_rejects_inconsistent_inputs(settings, change):
    data = settings.model_dump(mode="json")
    if change == "image":
        data["image_id"] = "im-other"
    elif change == "volumes":
        data["output_volume_id"] = data["input_volume_id"]
    elif change == "code":
        data["build"]["code_files"][0]["sha256"] = "0" * 64
    elif change == "source":
        data["build"]["source_sha"] = "0" * 40
    elif change == "package":
        data["build"]["environment"]["packages"][0][1] = "changed"
    elif change == "report_skip":
        data["build"]["cpu_skipped"] = 1
    elif change == "report_zero":
        # A positive case count replaced the fixed eleven; zero passing cases is still no gate.
        data["build"]["cpu_passed"] = 0
    elif change == "lock":
        data["build_spec"]["lock_sha256"] = "0" * 64
    elif change == "requirements":
        data["build_spec"]["requirements_file"]["filename"] = "pyproject.toml"
    elif change == "cpu_test":
        data["build_spec"]["cpu_test_file"]["filename"] = "tests/test_qwen.py"
    elif change == "recipe":
        data["build_spec"]["recipe_file"]["filename"] = "experiments/recipes/other.json"
    elif change == "processor":
        data["build_spec"]["processor_manifest_sha256"] = "0" * 64
    else:
        data["build_spec"]["code_files"].append(
            dict(filename="source.jsonl", bytes=1, sha256="0" * 64)
        )
    with pytest.raises(ValidationError):
        m.RuntimeSettings.model_validate(data)


@pytest.mark.parametrize(
    "path",
    ["source.jsonl", "source.xml", "model/private.json", CPU_TEST_FILE, RECIPE_FILE, "images/o"],
)
def test_bundle_has_no_extra_file_channel(settings, path):
    data = settings.bundle.model_dump(mode="json")
    data["files"].append(dict(filename=path, bytes=12, sha256="9" * 64))
    data["files"].sort(key=lambda entry: entry["filename"])
    with pytest.raises(ValidationError):
        m.InputBundle.model_validate(data)


@pytest.mark.parametrize("duration", [0, -1, 7201])
def test_deadlines_are_bounded_records(settings, duration):
    data = invoke(fit(settings)).model_dump(mode="json")
    data["deadline_unix_seconds"] = 1000 + duration
    with pytest.raises(ValidationError):
        m.Invocation.model_validate(data)


def test_deadline_window_is_not_a_clock_or_run_control_check(settings):
    call = invoke(fit(settings))
    assert call.deadline_unix_seconds - call.first_submission_unix_seconds == 2400
    # Deliberate API limit: a shifted envelope is structurally valid; later durable run-control
    # and actual clock checks must reject extension/expiry. No worker is exercised here.
    data = call.model_dump(mode="json")
    data.update(first_submission_unix_seconds=2000, deadline_unix_seconds=4400)
    shifted = m.Invocation.model_validate(data)
    assert shifted.operation_id == call.operation_id
    settings.check_invocation(shifted)


@pytest.mark.parametrize(
    "change", ["attempt", "page_order", "round", "run", "model", "raw", "score"]
)
def test_result_ownership_and_reference_structure(settings, change):
    call = invoke(predict(settings))
    result = completed(call)
    result.check_request(call)
    data = result.model_dump(mode="json")
    if change == "attempt":
        data["attempt_id"] = "9" * 32
    elif change == "page_order":
        data["predictions"].reverse()
    elif change == "round":
        data["predictions"][0].update(round_number=1, purpose="validation")
    elif change == "run":
        data["predictions"][0]["experiment_id"] = "9" * 32
    elif change == "model":
        data["predictions"][0]["model_id"] = "checkpoint:sha256:" + "9" * 64
    elif change == "raw":
        data["predictions"][0]["raw_output_artifact"] = f"{NAMESPACE}/receipts/unlisted.json"
    else:
        data["predictions"][0]["entropy"] = 0.5
    with pytest.raises((ValueError, ValidationError)):
        m.OperationResult.model_validate(data).check_request(call)


def test_completion_has_independent_canonical_digest(settings):
    call = invoke(predict(settings))
    result = completed(call)
    raw = encoded(result.model_dump(mode="json"))
    key = f"operations/{call.operation_id}/attempts/{ATTEMPT}/executions/{EXECUTION}/complete.json"
    completion = artifact(key, hashlib.sha256(raw).hexdigest(), len(raw))
    assert m.DispatchResponse(result=result, completion=completion).result == result
    for field, value in [
        ("key", key.replace(EXECUTION, "9" * 32)),
        ("bytes", len(raw) + 1),
        ("sha256", "9" * 64),
    ]:
        with pytest.raises(ValidationError):
            m.DispatchResponse(result=result, completion={**completion, field: value})


def test_probe_and_reload_records_reject_retired_and_inconsistent_fields():
    sha = "4" * 64
    record = dict(
        model_id=CHECKPOINT,
        page_id="first",
        image_sha256="a" * 64,
        original_width=100,
        original_height=200,
        process_id=71,
        status="ok",
        finish_reason="stop",
        response_tokens=1200,
        response_sha256=hashlib.sha256(PROBE_RESPONSE.encode()).hexdigest(),
        regions=[
            dict(id="line-0001", text="x", illegible=False, box=dict(x=1, y=2, width=3, height=4))
        ],
    )
    probe = m.ProbeMetadata.model_validate(record)
    assert probe.schema_version == 1 and probe.response_tokens == 1200
    # Probe keys are relative to the run output root even though the adapter writes them
    # under its own nested subtree. Consumers must require the exact root and key.
    assert m.probe_key(CHECKPOINT, "before") == f"{NAMESPACE}/probes/{sha}/before.json"
    assert str(Path(m.OUTPUT_ROOT) / m.probe_key(CHECKPOINT, "after")) == (
        f"/outputs/{NAMESPACE}/probes/{sha}/after.json"
    )
    for change in (
        {"status": "truncated"},  # A failed probe cannot also report parsed regions.
        {"status": "refusal"},
        {"finish_reason": "eos"},  # Retired token-level reason; the toolkit reports stop/length.
        {"response_tokens": 4097},  # Beyond the declared output capacity.
        {"response_tokens": -1},
        {"process_id": 0},
        {"model_id": "checkpoint:" + sha},
        {"response_sha256": "not-a-digest"},
        {"generated_ids": [151645]},  # Removed full-vocabulary/token probe fields stay removed.
        {"adapter_tensors": {}},
        {"logits": artifact(f"probes/{sha}/before.f32le")},
        {"logits_shape": [1, 151936]},
    ):
        with pytest.raises(ValidationError):
            m.ProbeMetadata.model_validate({**record, **change})
    probe_sha = digest(probe.model_dump(mode="json"))
    before = artifact(f"{NAMESPACE}/probes/{sha}/before.json", probe_sha)
    after = artifact(f"{NAMESPACE}/probes/{sha}/after.json", "c" * 64)
    evidence = m.ReloadEvidence(
        before=before,
        after=after,
        train_process_id=71,
        reload_process_id=72,
        identical_output=True,
    )
    assert (evidence.before.key, evidence.after.key) == (before["key"], after["key"])
    for change in (
        {"reload_process_id": 71},  # One interpreter is not fresh-process evidence.
        {"train_process_id": 0},
        {"identical_output": False},
        {"max_absolute_difference": 0.0},  # Retired numeric tolerance stays removed.
        {"exact_tensors_ids_status_regions": True},
    ):
        with pytest.raises(ValidationError):
            m.ReloadEvidence.model_validate({**evidence.model_dump(mode="json"), **change})


def test_adapter_verification_reads_framing_and_rejects_zero_or_tampered(tmp_path, adapter_bytes):
    path = tmp_path / "adapter_model.safetensors"
    path.write_bytes(adapter_bytes)
    summary = m.verify_adapter_file(path)
    assert len(summary["tensors"]) == 2 and summary["nonzero_tensors"] == 1
    path.write_bytes(safetensors_adapter(nonzero=False))
    with pytest.raises(ValueError, match="LoRA B is zero"):
        m.verify_adapter_file(path)
    for damaged in (
        adapter_bytes[:-4],  # Truncated tensor body.
        adapter_bytes + b"\0" * 4,  # Unreferenced trailing bytes.
        (0).to_bytes(8, "little") + adapter_bytes[8:],  # Unusable header length.
    ):
        path.write_bytes(damaged)
        with pytest.raises(ValueError):
            m.verify_adapter_file(path)


@pytest.mark.parametrize("extra", ["request", "targets", "traceback", "terminal", "cost"])
def test_failure_receipt_rejects_payload_and_provider_claims(extra):
    data = dict(
        operation_id="a" * 64,
        attempt_id=ATTEMPT,
        execution_id=EXECUTION,
        provider_call_id="fc-synthetic",
        code="model",
    )
    receipt = m.FailureReceipt(**data)
    assert SECRET not in receipt.model_dump_json()
    with pytest.raises(ValidationError):
        m.FailureReceipt.model_validate({**data, extra: SECRET})


def test_shared_api_import_requires_no_provider_or_ml():
    code = "import sys; import active_ocr.integrations.modal_model; "
    code += "assert not {'modal','torch','transformers','peft','llamafactory'} & sys.modules.keys()"
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        env={**os.environ, "PYTHONPATH": str(Path(fm.__file__).parents[2])},
    )


def test_fit_completion_requires_reload_and_direct_reference_inventory(settings):
    call = invoke(fit(settings))
    template = completed(invoke(predict(settings))).model_dump(mode="json")
    template.update(operation_id=call.operation_id, predictions=[])
    with pytest.raises(ValueError, match="completed fit"):
        m.OperationResult.model_validate(template).check_request(call)
    before = artifact(f"{NAMESPACE}/probes/" + "4" * 64 + "/before.json", "5" * 64)
    after = artifact(f"{NAMESPACE}/probes/" + "4" * 64 + "/after.json", "6" * 64)
    template["reload"] = dict(
        before=before,
        after=after,
        train_process_id=11,
        reload_process_id=12,
        identical_output=True,
    )
    with pytest.raises(ValidationError, match="reload evidence"):
        m.OperationResult.model_validate(template)
    template["artifacts"].extend([before, after])
    result = m.OperationResult.model_validate(template)
    result.check_request(call)
    # No bytes were read: this verifies direct references, not checkpoint kind/owner, probe
    # metadata contents, adapter bytes or a real second process. Those are phase B gates.
    assert {a.key for a in result.artifacts}.issuperset([before["key"], after["key"]])
    for change in (
        {"artifacts": result.artifacts + (result.artifacts[0],)},
        {"reload": None, "predictions": completed(invoke(predict(settings))).predictions},
    ):
        with pytest.raises((ValueError, ValidationError)):
            m.OperationResult.model_validate({**result.model_dump(), **change}).check_request(call)


@pytest.mark.parametrize(
    "key",
    [
        "../escape",
        "/outputs/absolute",
        f"{NAMESPACE}/../other",
        f"{NAMESPACE}//double",
        f"{NAMESPACE}\\windows",
    ],
)
def test_artifact_paths_reject_escape_and_ambiguous_separators(key):
    with pytest.raises(ValidationError):
        m.ArtifactRef.model_validate(artifact(key))


@pytest.mark.parametrize(
    "case", ["train_baseline", "test_validation", "zero_pool", "positive_baseline", "bool_round"]
)
def test_predict_purpose_round_and_split(settings, case):
    data = predict(settings).model_dump(mode="json")
    if case == "train_baseline":
        data["pages"][0]["split"] = "train"
    elif case == "test_validation":
        data["pages"][0]["split"] = "test"
    elif case == "zero_pool":
        data["purpose"] = "pool"
        data["pages"] = [page()]
    elif case == "positive_baseline":
        data["round_number"] = 1
    else:
        data["round_number"] = False
    with pytest.raises(ValidationError):
        m.PredictRequest.model_validate(data)


class ByteTransport:
    """Independent byte-producing base/prediction peer; no ML or provider imports."""

    def __init__(self, settings):
        self.settings = settings
        self.files = {}
        self.calls = []
        self.responses = {}
        self.cancelled = []
        self.lost_ack = False
        self.read_error = False
        self.preflight_hook = lambda: None

    def preflight(self):
        self.preflight_hook()

    def put(self, key, data):
        self.files[key] = data
        return m.ArtifactRef(key=key, bytes=len(data), sha256=hashlib.sha256(data).hexdigest())

    def spawn(self, payload):
        self.calls.append(payload)
        call_id = "fc-independent-" + str(len(self.calls))
        request = m.Invocation.model_validate(payload)
        checkpoint = fm.Checkpoint(
            kind="base", binding=fm.binding(self.settings.context.real_config)
        )
        raw = encoded(checkpoint.model_dump(mode="json"))
        sha = hashlib.sha256(raw).hexdigest()
        model_id = "checkpoint:sha256:" + sha
        refs = [self.put(f"{NAMESPACE}/checkpoints/{sha}/manifest.json", raw)]
        worker = ar.WorkerManifest(
            schema_version=1,
            source_sha=self.settings.build.source_sha,
            files=self.settings.build.code_files,
            environment=self.settings.build.environment,
            build_spec_sha256=self.settings.build_spec.sha256,
            deployment_reference=self.settings.deployment_reference,
        )
        worker_ref = self.put("worker.json", encoded(worker.model_dump(mode="json")))
        refs.append(worker_ref)
        predictions = []
        if isinstance(request.request, m.PredictRequest):
            req = request.request
            assert req.round_number == 0
            receipt = dict(
                operation="predict",
                experiment_id=req.context.experiment_id,
                round_number=0,
                purpose=req.purpose,
                model_id=model_id,
                pages=[
                    dict(
                        page_id=p.id,
                        image_sha256=p.image_sha256,
                        status="ok",
                        finish_reason="stop",
                        text='{"regions":[]}',
                    )
                    for p in req.pages
                ],
            )
            raw_ref = self.put(
                f"{NAMESPACE}/receipts/" + digest(receipt) + ".json", encoded(receipt)
            )
            refs.append(raw_ref)
            predictions = [
                dict(
                    experiment_id=req.context.experiment_id,
                    round_number=0,
                    purpose=req.purpose,
                    model_id=model_id,
                    page_id=p.id,
                    status="ok",
                    finish_reason="stop",
                    regions=[],
                    raw_output_artifact=raw_ref.key,
                )
                for p in req.pages
            ]
        else:
            assert isinstance(request.request, m.BaseRequest)
        result = m.OperationResult(
            operation_id=request.operation_id,
            attempt_id=request.attempt_id,
            execution_id=EXECUTION,
            provider_call_id=call_id,
            model_id=model_id,
            predictions=predictions,
            artifacts=refs,
            worker_manifest=worker_ref,
        )
        completion = self.put(
            f"operations/{request.operation_id}/attempts/{request.attempt_id}/"
            f"executions/{EXECUTION}/complete.json",
            encoded(result.model_dump(mode="json")),
        )
        self.responses[call_id] = m.DispatchResponse(result=result, completion=completion)
        if self.lost_ack:
            raise ConnectionError("synthetic acknowledgement loss")
        return call_id

    def get(self, call_id, timeout):
        if self.read_error:
            raise ConnectionError("synthetic read loss")
        return self.responses[call_id].model_dump(mode="json")

    def read(self, key):
        data = self.files[key]
        # Multiple fragments exercise streaming hash/size closure.
        yield data[:3]
        yield data[3:]

    def cancel(self, call_id):
        self.cancelled.append(call_id)


def local_model(tmp_path, settings, transport=None, clock=None):
    from active_ocr.integrations.storage import SQLiteStore

    store = SQLiteStore(tmp_path / "journal.db", tmp_path / "artifacts")
    store.initialize()
    return m.ModalModel(
        settings, store, transport or ByteTransport(settings), clock=clock or (lambda: 1000)
    )


def test_independent_byte_closure_survives_reconstruction(tmp_path, settings):
    first = local_model(tmp_path, settings)
    request = m.BaseRequest(context=settings.context)
    result = first.execute(request)
    recovered = local_model(tmp_path, settings, first.transport, clock=lambda: 3401)
    assert recovered.execute(request) == result
    assert len(first.transport.calls) == 1
    control = recovered.store.load("model-control", RUN, m.RunControl)
    assert (control.first_submission_unix_seconds, control.deadline_unix_seconds) == (1000, 3400)
    first.transport.files["worker.json"] += b"!"
    with pytest.raises(ValueError, match="exceeds declared size"):
        recovered.execute(request)
    assert len(first.transport.calls) == 1


@pytest.mark.parametrize("lost_at", ["spawn", "get"])
def test_independent_unknown_requires_reconciliation_without_resubmit(tmp_path, settings, lost_at):
    model = local_model(tmp_path, settings)
    model.transport.lost_ack = lost_at == "spawn"
    model.transport.read_error = lost_at == "get"
    request = m.BaseRequest(context=settings.context)
    with pytest.raises(RuntimeError):
        model.execute(request)
    restarted = local_model(tmp_path, settings, model.transport)
    with pytest.raises(RuntimeError, match="UNKNOWN"):
        restarted.execute(request)
    assert len(model.transport.calls) == 1
    record = restarted.store.load("model-operation", m.operation_id(request), m.ModelOperation)
    assert record.state == "UNKNOWN"
    assert record.provider_call_id == (None if lost_at == "spawn" else "fc-independent-1")
    model.transport.read_error = False
    restarted.reconcile(request, call_id="fc-independent-1")
    restarted.execute(request)
    assert len(model.transport.calls) == 1


def test_independent_two_sqlite_connections_cannot_both_submit(tmp_path, settings):
    entered, release = threading.Event(), threading.Event()
    first = local_model(tmp_path, settings)
    second = local_model(tmp_path, settings, first.transport)
    first._control()  # Reserve the run clock before racing only the operation reservation.
    count = 0

    def preflight():
        nonlocal count
        count += 1
        if count == 1:
            entered.set()
            assert release.wait(10)

    first.transport.preflight_hook = preflight
    request = m.BaseRequest(context=settings.context)
    with ThreadPoolExecutor(max_workers=1) as executor:
        paused = executor.submit(first.execute, request)
        try:
            assert entered.wait(10)
            second.execute(request)
        finally:
            release.set()
        with pytest.raises(RuntimeError):
            paused.result(timeout=10)
    assert len(first.transport.calls) == 1
    assert (
        second.store.load("model-operation", m.operation_id(request), m.ModelOperation).state
        == "COMPLETED"
    )


def test_independent_preflight_expiry_must_not_spawn(tmp_path, settings):
    """Regression: network preflight is not permission to submit after the fixed deadline."""
    now = [1000]
    model = local_model(tmp_path, settings, clock=lambda: now[0])
    model._control()
    now[0] = 3399
    model.transport.preflight_hook = lambda: now.__setitem__(0, 3401)
    with pytest.raises(TimeoutError):
        model.execute(m.BaseRequest(context=settings.context))
    assert model.transport.calls == [], "No paid submission is permitted after deadline 3400"


@pytest.fixture
def baseline_fixture(tmp_path, settings, monkeypatch):
    from active_ocr.integrations import simulation
    from active_ocr.models import SimulationConfig
    from active_ocr.pipeline import Pipeline

    # Only the machine identity boundary is synthetic; source freeze, evaluator,
    # real adapter checks, SQLite and artifact verification execute normally.
    identity = settings.context.real_config.expected_identity
    monkeypatch.setattr(
        simulation,
        "local_contract_identity",
        lambda: (
            identity.source_sha,
            identity.dependency_sha256,
        ),
    )
    manifest = simulation.create_fixture(tmp_path / "source", train_pages=4)
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    rows[3]["split"] = "validation"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    config = SimulationConfig(
        real=settings.context.real_config,
        backend=BACKEND,
        evaluator_id=EVALUATOR,
        max_rounds=1,
        batch_size=1,
        validation_page_ids=("page-4",),
    )
    return Pipeline.for_simulation(tmp_path / "run"), manifest, config, settings


def attach_baseline(pipeline, run, settings):
    files = [f.model_dump() for f in settings.bundle.files if f.filename.startswith("model/")]
    files += [
        dict(
            filename="images/" + p.image_sha256,
            sha256=p.image_sha256,
            bytes=Path(p.image_uri).stat().st_size,
        )
        for p in run.dataset.pages
    ]
    bundle = m.InputBundle(files=sorted(files, key=lambda f: f["filename"]))
    data = settings.model_dump()
    data["bundle"] = bundle.model_dump()
    data["context"].update(
        experiment_id=run.id, source_policy=run.config.source_policy, bundle_sha256=bundle.sha256
    )
    runtime = m.RuntimeSettings.model_validate(data)
    transport = ByteTransport(runtime)
    return m.ModalModel(runtime, pipeline.store, transport, clock=lambda: 1000)


@pytest.mark.parametrize(
    "subset,expected", [(("page-4",), ["page-4"]), (None, ["page-3", "page-4"])]
)
def test_independent_real_baseline_and_selected_fit_boundary(
    baseline_fixture, tmp_path, subset, expected
):
    from active_ocr.models import RunKind

    pipeline, manifest, config, settings = baseline_fixture
    config = config.model_copy(update={"validation_page_ids": subset})
    run = pipeline.create_simulation(manifest, config, kind=RunKind.REAL)
    model = attach_baseline(pipeline, run, settings)
    baseline = pipeline.step_simulation(run.id, model)
    assert baseline.baseline.labelled_count == 0
    assert [p.page_id for p in baseline.baseline.validation_predictions] == expected
    assert baseline.rounds == ()
    request = model.transport.calls[1]["request"]
    assert [p["id"] for p in request["pages"]] == expected
    assert "Synthetic page" not in json.dumps(model.transport.calls)
    output = tmp_path / "export"
    pipeline.export_simulation(run.id, output)
    assert json.loads((output / "validation.json").read_text()) == {
        "page_ids": expected,
        "page_count": len(expected),
    }
    with pytest.raises(ValueError, match="frozen run config"):
        pipeline.step_simulation(run.id, model, config=config.model_copy(update={"seed": 910}))
    # This byte peer deliberately implements no fit. Inspect the first selection
    # request, then assert its unverified output cannot commit an acquisition round.
    with pytest.raises(RuntimeError, match="submission outcome unknown"):
        pipeline.step_simulation(run.id, model)
    fit_request = model.transport.calls[-1]["request"]
    assert fit_request["operation"] == "fit"
    assert len(fit_request["examples"]) == 1
    selected = fit_request["examples"][0]
    assert selected["page"]["id"] in {"page-0", "page-1", "page-2"}
    assert selected["page"]["split"] == "train"
    assert selected["regions"][0]["text"] == "Synthetic page " + selected["page"]["id"][-1]
    assert pipeline.get_simulation(run.id).rounds == ()
    journal = model.store.load(
        "model-operation",
        m.operation_id(m.FitRequest.model_validate(fit_request)),
        m.ModelOperation,
    )
    assert "Synthetic page" not in journal.model_dump_json()


@pytest.mark.parametrize(
    "subset", [("absent",), ("page-0",), ("page-5",), ("page-4", "page-4"), ()]
)
def test_independent_bad_subset_rejected_at_real_create(baseline_fixture, subset):
    from active_ocr.models import RunKind

    pipeline, manifest, config, _ = baseline_fixture
    with pytest.raises(ValueError):
        pipeline.create_simulation(
            manifest, config.model_copy(update={"validation_page_ids": subset}), kind=RunKind.REAL
        )


def test_independent_frozen_image_mutation_blocks_resume(baseline_fixture):
    from active_ocr.models import RunKind

    pipeline, manifest, config, settings = baseline_fixture
    run = pipeline.create_simulation(manifest, config, kind=RunKind.REAL)
    model = attach_baseline(pipeline, run, settings)
    pipeline.step_simulation(run.id, model)
    frozen = Path(run.dataset.pages[0].image_uri)
    frozen.write_bytes(frozen.read_bytes() + b"changed")
    with pytest.raises(ValueError):
        pipeline.step_simulation(run.id, model)
    assert len(model.transport.calls) == 2


class FullByteTransport(ByteTransport):
    """Successful byte protocol peer, not a trained model or measured build."""

    def __init__(self, settings, adapter):
        super().__init__(settings)
        self.adapter = adapter
        self.training_override = {}
        self.intercept = lambda invocation, response: response

    def spawn(self, payload):
        request = m.Invocation.model_validate(payload)
        if isinstance(request.request, m.BaseRequest) or request.request.round_number == 0:
            return super().spawn(payload)
        self.calls.append(payload)
        call_id = "fc-independent-" + str(len(self.calls))
        response = self.make_result(request, call_id)
        self.responses[call_id] = self.intercept(request, response)
        return call_id

    def target(self, example):
        """Reproduce the literal training target independently of the adapter helper."""
        regions = []
        for region in example.regions:
            b, p = region.box, example.page
            regions.append(
                dict(
                    text=region.text,
                    bbox=[
                        1000 * b.x / p.width,
                        1000 * b.y / p.height,
                        1000 * (b.x + b.width) / p.width,
                        1000 * (b.y + b.height) / p.height,
                    ],
                )
            )
        return json.dumps(dict(regions=regions), ensure_ascii=False, separators=(",", ":")).replace(
            "<", r"\u003c"
        )

    def make_result(self, invocation, call_id):
        request = invocation.request
        worker = self.put(
            "worker.json",
            encoded(
                ar.WorkerManifest(
                    schema_version=1,
                    source_sha=self.settings.build.source_sha,
                    files=self.settings.build.code_files,
                    environment=self.settings.build.environment,
                    build_spec_sha256=self.settings.build_spec.sha256,
                    deployment_reference=self.settings.deployment_reference,
                ).model_dump(mode="json")
            ),
        )
        refs, predictions, reload = [worker], [], None
        if isinstance(request, m.FitRequest):
            pages = [e.page for e in request.examples]
            targets = [(e.page.id, self.target(e)) for e in request.examples]
            # Synthetic config bytes are closure evidence only. Actual toolkit adapter
            # compatibility remains in the separately gated LLaMA-Factory checks.
            config = encoded(
                dict(
                    peft_type="LORA",
                    r=8,
                    lora_alpha=16,
                    lora_dropout=0.0,
                    init_lora_weights=True,
                    use_dora=False,
                    use_rslora=False,
                    bias="none",
                    modules_to_save=None,
                )
            )
            payloads = {"adapter_config.json": config, "adapter_model.safetensors": self.adapter}
            training = dict(
                toolkit_version=TOOLKIT_VERSION,
                examples=len(pages),
                epochs=EPOCHS,
                optimizer_steps=3 * len(pages),
                losses=[0.25] * (3 * len(pages)),
                train_runtime_seconds=12.5,
                preflight={
                    p.id: dict(prompt_tokens=1024, target_tokens=256, total_tokens=1280)
                    for p in pages
                },
                changed_lora_B_tensors=1,
            )
            training.update(self.training_override)
            checkpoint = fm.Checkpoint(
                kind="adapter",
                binding=fm.binding(self.settings.context.real_config),
                experiment_id=request.context.experiment_id,
                round_number=request.round_number,
                selected=[(e.page.id, e.page.image_sha256) for e in request.examples],
                seed=request.seed,
                target_sha256=digest(targets),
                files=[
                    dict(filename=k, bytes=len(v), sha256=hashlib.sha256(v).hexdigest())
                    for k, v in payloads.items()
                ],
                training=training,
            )
            sha = digest(checkpoint.model_dump(mode="json"))
            model_id = "checkpoint:sha256:" + sha
            prefix = f"{NAMESPACE}/checkpoints/{sha}/"
            refs += [
                self.put(prefix + "manifest.json", encoded(checkpoint.model_dump(mode="json")))
            ]
            refs += [self.put(prefix + k, v) for k, v in payloads.items()]
            selected = min(pages, key=lambda p: p.id)
            probes = []
            for stage, pid in [("before", 1701), ("after", 1702)]:
                metadata = m.ProbeMetadata(
                    model_id=model_id,
                    page_id=selected.id,
                    image_sha256=selected.image_sha256,
                    original_width=selected.width,
                    original_height=selected.height,
                    process_id=pid,
                    status="invalid_output",
                    finish_reason="stop",
                    response_tokens=17,
                    response_sha256=hashlib.sha256(PROBE_RESPONSE.encode()).hexdigest(),
                    regions=[],
                )
                probes.append(
                    self.put(
                        f"{NAMESPACE}/probes/{sha}/{stage}.json",
                        encoded(metadata.model_dump(mode="json")),
                    )
                )
            refs += probes
            reload = m.ReloadEvidence(
                before=probes[0],
                after=probes[1],
                train_process_id=1701,
                reload_process_id=1702,
                identical_output=True,
            )
        else:
            model_id = request.input_checkpoint
            prefix = f"{NAMESPACE}/checkpoints/" + model_id.rsplit(":", 1)[1] + "/"
            refs += [self.put(k, v) for k, v in list(self.files.items()) if k.startswith(prefix)]
            receipt = dict(
                operation="predict",
                experiment_id=request.context.experiment_id,
                round_number=request.round_number,
                purpose=request.purpose,
                model_id=model_id,
                pages=[
                    dict(
                        page_id=p.id,
                        image_sha256=p.image_sha256,
                        status="ok",
                        finish_reason="stop",
                        text='{"regions":[]}',
                    )
                    for p in request.pages
                ],
            )
            raw = self.put(f"{NAMESPACE}/receipts/" + digest(receipt) + ".json", encoded(receipt))
            refs.append(raw)
            predictions = [
                dict(
                    experiment_id=request.context.experiment_id,
                    round_number=request.round_number,
                    purpose=request.purpose,
                    model_id=model_id,
                    page_id=p.id,
                    regions=[],
                    status="ok",
                    finish_reason="stop",
                    raw_output_artifact=raw.key,
                )
                for p in request.pages
            ]
        result = m.OperationResult(
            operation_id=invocation.operation_id,
            attempt_id=invocation.attempt_id,
            execution_id=EXECUTION,
            provider_call_id=call_id,
            model_id=model_id,
            predictions=predictions,
            artifacts=refs,
            worker_manifest=worker,
            reload=reload,
        )
        return self.envelope(result)

    def envelope(self, result):
        completion = self.put(
            f"operations/{result.operation_id}/attempts/{result.attempt_id}/"
            f"executions/{result.execution_id}/complete.json",
            encoded(result.model_dump(mode="json")),
        )
        return m.DispatchResponse(result=result, completion=completion)


def test_corrected_expiry_preserves_safe_journal(tmp_path, settings, monkeypatch):
    now = [1000]
    model = local_model(tmp_path, settings, clock=lambda: now[0])
    swap = model.store.compare_and_swap

    def delayed(kind, key, expected, value):
        swap(kind, key, expected, value)
        if kind == "model-operation" and value.state == "SUBMITTING":
            now[0] = 3400

    monkeypatch.setattr(model.store, "compare_and_swap", delayed)
    request = m.BaseRequest(context=settings.context)
    with pytest.raises(TimeoutError):
        model.execute(request)
    record = model.store.load("model-operation", m.operation_id(request), m.ModelOperation)
    assert record.state == "RESERVED" and record.provider_call_id is None
    assert model.transport.calls == model.transport.cancelled == []
    with pytest.raises(TimeoutError):
        model.execute(request)
    assert model.store.load("model-operation", record.operation_id, m.ModelOperation) == record
    assert model.transport.calls == []


def test_full_round_recovery_and_cumulative_reset_fit(
    baseline_fixture, adapter_bytes, monkeypatch, tmp_path
):
    from active_ocr.models import RunKind
    from active_ocr.pipeline import Pipeline

    pipeline, manifest, config, settings = baseline_fixture
    config = config.model_copy(update={"max_rounds": 2})
    run = pipeline.create_simulation(manifest, config, kind=RunKind.REAL)
    model = attach_baseline(pipeline, run, settings)
    transport = FullByteTransport(model.settings, adapter_bytes)
    model.transport = transport
    pipeline.step_simulation(run.id, model)
    swap = pipeline.store.compare_and_swap
    lost = [False]

    def lose_response(kind, key, expected, value):
        swap(kind, key, expected, value)
        if kind == "simulation" and len(value.rounds) == 1 and not lost[0]:
            lost[0] = True
            raise ConnectionError("committed round response was lost")

    monkeypatch.setattr(pipeline.store, "compare_and_swap", lose_response)
    with pytest.raises(ConnectionError):
        pipeline.step_simulation(run.id, model)
    assert len(pipeline.get_simulation(run.id).rounds) == 1
    reopened = Pipeline.for_simulation(tmp_path / "run")
    model = m.ModalModel(model.settings, reopened.store, transport, clock=lambda: 1001)
    complete = reopened.run_simulation(run.id, model)
    fits = [c["request"] for c in transport.calls if c["request"]["operation"] == "fit"]
    assert [len(f["examples"]) for f in fits] == [1, 2]
    assert fits[0]["input_checkpoint"] == fits[1]["input_checkpoint"]
    assert fits[0]["examples"][0] == fits[1]["examples"][0]
    assert len(complete.rounds) == 2 and complete.complete
    assert [r.labelled_count for r in complete.rounds] == [1, 2]
    assert all(c["request"]["purpose"] != "pool" for c in transport.calls)
    assert len(transport.calls) == 6
    assert reopened.step_simulation(run.id, model) == complete
    assert len(transport.calls) == 6
    reopened.export_simulation(run.id, tmp_path / "finished")
    assert json.loads((tmp_path / "finished/results.json").read_text())["complete"] is True


def test_independent_initial_fit_precedes_acquisition_rounds(
    baseline_fixture, adapter_bytes, tmp_path
):
    """A configured seed set is labelled work: one extra fit before round one, no extra round."""
    from active_ocr.models import RunKind

    pipeline, manifest, config, settings = baseline_fixture
    config = config.model_copy(update={"initial_batch_size": 2})
    run = pipeline.create_simulation(manifest, config, kind=RunKind.REAL)
    model = attach_baseline(pipeline, run, settings)
    model.transport = FullByteTransport(model.settings, adapter_bytes)
    complete = pipeline.run_simulation(run.id, model)
    assert complete.initial_fit is not None and complete.initial_fit.labelled_count == 2
    assert len(complete.rounds) == 1 and complete.rounds[0].labelled_count == 3
    fits = [c["request"] for c in model.transport.calls if c["request"]["operation"] == "fit"]
    assert [len(f["examples"]) for f in fits] == [2, 3]
    assert [f["round_number"] for f in fits] == [1, 2]
    revealed = set(complete.initial_fit.revealed_ids)
    assert revealed.issubset(set(complete.rounds[0].revealed_ids))


def test_completed_fit_reused_after_downstream_failure(
    baseline_fixture, adapter_bytes, monkeypatch
):
    from active_ocr.models import RunKind

    pipeline, manifest, config, settings = baseline_fixture
    run = pipeline.create_simulation(manifest, config, kind=RunKind.REAL)
    model = attach_baseline(pipeline, run, settings)
    transport = FullByteTransport(model.settings, adapter_bytes)
    model.transport = transport
    pipeline.step_simulation(run.id, model)
    original = transport.get
    fail_once = [True]

    def get(call_id, timeout):
        response = transport.responses[call_id]
        if (
            response.result.predictions
            and response.result.predictions[0].round_number == 1
            and fail_once[0]
        ):
            fail_once[0] = False
            raise ConnectionError("downstream response loss")
        return original(call_id, timeout)

    monkeypatch.setattr(transport, "get", get)
    with pytest.raises(RuntimeError, match="outcome unresolved"):
        pipeline.step_simulation(run.id, model)
    assert pipeline.get_simulation(run.id).rounds == ()
    assert len(transport.calls) == 4
    pending = m.Invocation.model_validate(transport.calls[-1]).request
    model.reconcile(pending)
    complete = pipeline.step_simulation(run.id, model)
    assert complete.complete and len(transport.calls) == 4
    assert len([c for c in transport.calls if c["request"]["operation"] == "fit"]) == 1


@pytest.mark.parametrize(
    "damage",
    [
        "probe_output",
        "probe_page",
        "probe_process",
        "noncanonical_probe",
        "probe_namespace",
        "zero_adapter",
        "training_record",
        "extra_checkpoint_file",
        "wrong_worker",
        "foreign_attempt",
    ],
)
def test_independent_fit_rejects_consistent_but_invalid_evidence(
    tmp_path, settings, adapter_bytes, damage
):
    transport = FullByteTransport(settings, adapter_bytes)
    # These two damages are internally consistent from the first byte: the published
    # checkpoint really does contain them, so only content verification can reject it.
    if damage == "zero_adapter":
        transport.adapter = safetensors_adapter(nonzero=False)
    elif damage == "training_record":
        transport.training_override = {"epochs": 1.0}
    model = local_model(tmp_path, settings, transport)
    base = model.load_base(experiment_id=RUN)
    request = fit(settings).model_copy(update={"input_checkpoint": base})

    def damage_response(invocation, response):
        result = response.result
        refs = {a.key: a for a in result.artifacts}
        reload = result.reload
        if damage in {"probe_output", "probe_page", "probe_process", "noncanonical_probe"}:
            which = "before" if damage in {"probe_process", "noncanonical_probe"} else "after"
            ref = getattr(reload, which)
            meta = json.loads(transport.files[ref.key])
            if damage == "probe_output":
                meta["response_sha256"] = "0" * 64
            elif damage == "probe_page":
                meta["page_id"] = "unselected"
            elif damage == "probe_process":
                meta["process_id"] = 1703
            raw = (
                json.dumps(meta, sort_keys=True, indent=2).encode()
                if damage == "noncanonical_probe"
                else encoded(meta)
            )
            refs[ref.key] = transport.put(ref.key, raw)
            reload = reload.model_copy(update={which: refs[ref.key]})
        elif damage == "probe_namespace":
            key = reload.after.key.replace("/after.json", "/reload.json")
            refs[key] = transport.put(key, transport.files[reload.after.key])
            reload = reload.model_copy(update={"after": refs[key]})
        elif damage == "extra_checkpoint_file":
            sha = result.model_id.rsplit(":", 1)[1]
            key = f"{NAMESPACE}/checkpoints/{sha}/unexpected.json"
            refs[key] = transport.put(key, b"{}")
        elif damage == "wrong_worker":
            worker = json.loads(transport.files[result.worker_manifest.key])
            worker["source_sha"] = "0" * 40
            ref = transport.put(result.worker_manifest.key, encoded(worker))
            refs[ref.key] = ref
            result = result.model_copy(update={"worker_manifest": ref})
        elif damage == "foreign_attempt":
            result = result.model_copy(update={"attempt_id": "0" * 32})
        return transport.envelope(
            result.model_copy(update={"artifacts": tuple(refs.values()), "reload": reload})
        )

    transport.intercept = damage_response
    with pytest.raises(ValueError):
        model.execute(request)
    record = model.store.load("model-operation", m.operation_id(request), m.ModelOperation)
    assert record.state == "UNKNOWN" and record.response is None
    assert len(transport.calls) == 2
    with pytest.raises(RuntimeError, match="UNKNOWN"):
        model.execute(request)
    assert len(transport.calls) == 2


def test_cancellation_keeps_call_and_deadline_unresolved(tmp_path, settings):
    now = [1000]
    model = local_model(tmp_path, settings, clock=lambda: now[0])
    request = m.BaseRequest(context=settings.context)

    def timeout(call_id, timeout):
        now[0] = 3400
        raise TimeoutError

    model.transport.get = timeout
    with pytest.raises(TimeoutError, match="terminal status unverified"):
        model.execute(request)
    record = model.store.load("model-operation", m.operation_id(request), m.ModelOperation)
    assert record.state == "CANCEL_REQUESTED"
    assert record.provider_call_id == model.transport.cancelled[0]
    with pytest.raises(RuntimeError, match="CANCEL_REQUESTED"):
        model.execute(request)
    assert len(model.transport.calls) == 1
    control = model.store.load("model-control", RUN, m.RunControl)
    assert control.deadline_unix_seconds == 3400 and control.active_operation == record.operation_id


@pytest.fixture
def worker_fixture(settings, tmp_path, monkeypatch, adapter_bytes):
    """Real dispatcher on temporary synthetic files; no asset/identity verifier disabled."""
    from types import SimpleNamespace

    from active_ocr.entrypoints import modal_app as app

    inputs, outputs, code = [tmp_path / n for n in ("inputs", "outputs", "code")]
    for root in (inputs, outputs, code):
        root.mkdir()
    assets, entries = {}, []
    for key in settings.bundle.files:
        name = key.filename
        path = inputs / name
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = ("synthetic file " + name).encode()
        path.write_bytes(raw)
        sha = hashlib.sha256(raw).hexdigest()
        if name.startswith("model/"):
            assets[name[6:]] = (len(raw), sha)
            entries.append(dict(filename=name, bytes=len(raw), sha256=sha))
        else:
            # Images used by this file-only worker peer are declared bytes, not
            # decoded pages. Actual page-header rejection is exercised separately.
            path.unlink()
            (inputs / "images" / sha).write_bytes(raw)
            entries.append(dict(filename="images/" + sha, bytes=len(raw), sha256=sha))
    monkeypatch.setattr(fm, "ASSETS", assets)
    monkeypatch.setattr(m, "ASSETS", assets)
    bundle = m.InputBundle(files=sorted(entries, key=lambda e: e["filename"]))
    files = []
    for name in CODE_FILES:
        raw = (Path(fm.__file__).parents[2] / name).read_bytes()
        path = code / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        files.append(dict(filename=name, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
    spec_data = settings.build_spec.model_dump()
    spec_data.update(
        code_files=files, processor_manifest_sha256=digest(fm.asset_manifest(fm.PROCESSOR_FILES))
    )
    spec = m.BuildSpec.model_validate(spec_data)
    build_data = settings.build.model_dump()
    build_data.update(code_files=files, build_spec_sha256=spec.sha256)
    build = m.BuildReceipt.model_validate(build_data)
    data = settings.model_dump()
    data.update(bundle=bundle.model_dump(), build_spec=spec.model_dump(), build=build.model_dump())
    context = data["context"]
    context["bundle_sha256"] = bundle.sha256
    expected = context["real_config"]["expected_identity"]
    expected.update(
        model_manifest_sha256=digest(fm.asset_manifest(fm.BASE_FILES)),
        processor_manifest_sha256=spec.processor_manifest_sha256,
        code_bundle_sha256=build.code_bundle_sha256,
        build_spec_sha256=spec.sha256,
    )
    runtime = m.RuntimeSettings.model_validate(data)
    (code / "build-receipt.json").write_bytes(encoded(build.model_dump(mode="json")))
    monkeypatch.setattr(ar, "installed_packages", lambda: build.environment)
    monkeypatch.setattr(app.time, "time", lambda: 1000)
    transport = FullByteTransport(runtime, adapter_bytes)
    calls, commits = [], []
    torch_was_loaded = "torch" in sys.modules
    deferred = {}

    def child(payload, deadline):
        calls.append(payload)
        assert deadline == 3390
        assert ("torch" in sys.modules) == torch_was_loaded
        if payload["stage"] == "reload":
            assert (
                "request" not in payload
                and "examples" not in payload
                and "regions" not in payload["page"]
            )
            for key, raw in deferred.items():
                (outputs / key).write_bytes(raw)
            return dict(status="ok", process_id=1702, model_id=payload["model_id"])
        request = m.REQUEST_ADAPTER.validate_python(payload["request"])
        inv = invoke(request)
        call_id = transport.spawn(inv.model_dump(mode="json"))
        result = transport.responses[call_id].result
        for key, raw in transport.files.items():
            if not key.startswith(NAMESPACE + "/"):
                continue
            if isinstance(request, m.FitRequest) and "/after." in key:
                deferred[key] = raw
                continue
            dest = outputs / key
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(raw)
        return dict(
            status="ok",
            process_id=1701,
            model_id=result.model_id,
            predictions=[
                p.model_copy(
                    update={
                        "raw_output_artifact": p.raw_output_artifact.removeprefix(NAMESPACE + "/")
                    }
                ).model_dump(mode="json")
                for p in result.predictions
            ],
        )

    def dispatch(request, **kw):
        invocation = invoke(request)
        return app.dispatch_operation(
            runtime,
            invocation.model_dump(mode="json"),
            provider_call_id="fc-worker",
            input_root=inputs,
            output_root=outputs,
            code_root=code,
            child_runner=child,
            commit=lambda: commits.append(True),
            **kw,
        )

    return SimpleNamespace(
        settings=runtime,
        inputs=inputs,
        outputs=outputs,
        code=code,
        calls=calls,
        commits=commits,
        dispatch=dispatch,
        child=child,
        transport=transport,
    )


def test_worker_dispatch_complete_fit_two_stages_and_reuse(worker_fixture):
    w = worker_fixture
    base = m.DispatchResponse.model_validate(w.dispatch(m.BaseRequest(context=w.settings.context)))
    image = next(f for f in w.settings.bundle.files if f.filename.startswith("images/"))
    request = m.FitRequest(
        context=w.settings.context,
        round_number=1,
        input_checkpoint=base.result.model_id,
        seed=7,
        examples=[dict(page=page(letter="a"), regions=fit(w.settings).examples[0].regions)],
    )
    data = request.model_dump()
    data["examples"][0]["page"].update(image_key=image.filename, image_sha256=image.sha256)
    request = m.FitRequest.model_validate(data)
    response = m.DispatchResponse.model_validate(w.dispatch(request))
    assert [p["stage"] for p in w.calls] == ["operation", "operation", "reload"]
    assert len(w.commits) == 6
    assert response.result.reload.train_process_id != response.result.reload.reload_process_id
    assert w.dispatch(request) == response.model_dump(mode="json")
    assert len(w.calls) == 3
    # Reuse must compare the immutable run clock, not merely the operation ID.
    from active_ocr.entrypoints import modal_app as app

    shifted = invoke(request).model_copy(
        update={"first_submission_unix_seconds": 999, "deadline_unix_seconds": 3399}
    )
    with pytest.raises(app.WorkerError):
        app.dispatch_operation(
            w.settings,
            shifted.model_dump(mode="json"),
            provider_call_id="fc-worker",
            input_root=w.inputs,
            output_root=w.outputs,
            code_root=w.code,
            child_runner=w.child,
            commit=lambda: None,
        )
    assert len(w.calls) == 3


@pytest.mark.parametrize(
    "damage",
    [
        "extra_input",
        "changed_input",
        "code",
        "build_receipt",
        "symlink",
        "partial",
        "wrong_deadline",
    ],
)
def test_worker_preconditions_never_start_child(worker_fixture, damage):
    from active_ocr.entrypoints import modal_app as app

    w = worker_fixture
    request = m.BaseRequest(context=w.settings.context)
    if damage == "extra_input":
        (w.inputs / "labels.jsonl").write_text(SECRET)
    elif damage == "changed_input":
        path = next((w.inputs / "model").iterdir())
        path.write_bytes(b"changed")
    elif damage == "code":
        (w.code / CODE_FILES[0]).write_text("# changed")
    elif damage == "build_receipt":
        (w.code / "build-receipt.json").write_text("{}")
    elif damage == "symlink":
        (w.inputs / "link").symlink_to(w.code)
    elif damage == "partial":
        root = w.outputs / "operations" / m.operation_id(request)
        root.mkdir(parents=True)
        (root / "partial").write_bytes(b"")
    else:
        (w.outputs / "run-control.json").write_bytes(
            encoded(
                dict(
                    settings_sha256=digest(w.settings.model_dump(mode="json")),
                    first_submission_unix_seconds=999,
                    deadline_unix_seconds=3399,
                )
            )
        )
    with pytest.raises(app.WorkerError):
        w.dispatch(request)
    assert w.calls == []
    assert not list(w.outputs.rglob("complete.json"))


def test_worker_watchdog_reaps_real_local_child(monkeypatch):
    import time

    from active_ocr.entrypoints import modal_app as app

    original = subprocess.Popen
    children = []

    def benign_child(args, **kwargs):
        assert kwargs["start_new_session"] is True
        child = original([sys.executable, "-c", "import time; time.sleep(20)"], **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(app.subprocess, "Popen", benign_child)
    try:
        with pytest.raises(app.WorkerError, match="deadline"):
            app.run_child({"test_only": True}, time.time() + 0.25)
        assert len(children) == 1 and children[0].poll() is not None
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
            child.communicate(timeout=5)


@pytest.mark.parametrize("change", ["skip", "failure", "duplicate", "empty", "exit", "success"])
@pytest.mark.parametrize("count", [3, 9])
def test_cpu_report_zero_skip_gate(tmp_path, change, count):
    from active_ocr.entrypoints import modal_app as app

    names = [f"case_{i}" for i in range(count)]
    if change == "duplicate":
        names[-1] = names[0]
    elif change == "empty":
        names = []
    body = "<skipped/>" if change == "skip" else "<failure/>" if change == "failure" else ""
    report = tmp_path / "report.xml"
    report.write_text(
        "<testsuite>"
        + "".join(
            f'<testcase classname="synthetic" name="{n}">{body if i == 0 else ""}</testcase>'
            for i, n in enumerate(names)
        )
        + "</testsuite>"
    )
    if change == "success":
        # Any positive number of selected cases is admissible; zero skips still are not.
        observed = app._cpu_report(report, 0)
        assert (observed["selected"], observed["passed"], observed["skipped"]) == (count, count, 0)
        assert observed["failed"] == 0 and observed["tests"] == sorted(
            "synthetic::" + n for n in names
        )
    else:
        with pytest.raises(app.WorkerError):
            app._cpu_report(report, 1 if change == "exit" else 0)


def test_runtime_file_worker_import_isolation(tmp_path):
    import shutil

    root = Path(fm.__file__).parents[3]
    for name in CODE_FILES:
        dest = tmp_path / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / "src" / name, dest)
    (tmp_path / "experiments/recipes").mkdir(parents=True)
    shutil.copyfile(root / RECIPE_FILE, tmp_path / RECIPE_FILE)
    program = """
import sys, pathlib, importlib.util
import active_ocr.entrypoints.modal_app as worker
import active_ocr.integrations as integration
assert pathlib.Path(worker.__file__).resolve().is_relative_to(pathlib.Path(sys.argv[1]).resolve())
removed = ('GPUClient','LabelStudioClient','load_image_pages')
assert not any(hasattr(integration, name) for name in removed)
assert importlib.util.find_spec('active_ocr.integrations.qwen') is None
assert not any(k.split('.')[0] in {'modal','torch','transformers','peft','llamafactory'}
    for k in sys.modules)
legacy = ('storage', 'label_studio', 'gpu_client', 'local_data')
assert not any('active_ocr.integrations.'+k in sys.modules for k in legacy)
print('runtime-file isolation verified')
"""
    result = subprocess.run(
        [sys.executable, "-c", program, str(tmp_path)],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "runtime-file isolation verified"


def test_cpu_build_selector_collects_only_actual_ml_cases(settings):
    """Run the worker's own selector in collection-only mode: no ML test executes."""
    import inspect
    import re

    from active_ocr.entrypoints import modal_app as app

    source = inspect.getsource(app.cpu_build_gate)
    selector = re.search(r'"-k",\s*"([^"]+)"', source).group(1)
    target = "tests/" + re.search(r'"tests/(test_\w+\.py)"', source).group(1)
    # The gate must execute exactly the CPU test file pinned in the reviewed build inputs.
    assert target == settings.build_spec.cpu_test_file.filename == CPU_TEST_FILE
    root = Path(fm.__file__).parents[3]
    if not (root / target).is_file():
        pytest.skip(f"{target} is not present in this checkout yet")
    program = """
import json, pytest, sys
class Nodes:
    def pytest_collection_finish(self,session):
        print('COLLECTED='+json.dumps([item.nodeid for item in session.items]))
args = [sys.argv[2],'--collect-only','-q','-k',sys.argv[1]]
raise SystemExit(pytest.main(args,plugins=[Nodes()]))
"""
    result = subprocess.run(
        [sys.executable, "-c", program, selector, target],
        cwd=root,
        env={**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    nodes = json.loads(
        next(
            line.removeprefix("COLLECTED=")
            for line in result.stdout.splitlines()
            if line.startswith("COLLECTED=")
        )
    )
    assert nodes, "the CPU build gate must select at least one real ML case"
    assert all(node.split("::")[-1].startswith("test_actual_") for node in nodes)


@pytest.mark.parametrize(
    "damage",
    [None, "code", "lock", "processor", "test", "recipe", "export", "dirty", "revision"],
)
def test_bootstrap_allowlist_and_preconstruction_rejections(
    worker_fixture, monkeypatch, tmp_path, damage
):
    import shutil
    from types import SimpleNamespace

    from active_ocr.entrypoints import modal_app as app

    w = worker_fixture
    checkout = tmp_path / "checkout"
    shutil.copytree(w.code / "active_ocr", checkout / "src/active_ocr")
    (checkout / "tests").mkdir()
    (checkout / CPU_TEST_FILE).write_bytes(b"# synthetic build-test input\n")
    (checkout / "experiments/recipes").mkdir(parents=True)
    (checkout / RECIPE_FILE).write_bytes(b'{"schema_version": 1}\n')
    (checkout / "uv.lock").write_bytes(b"version = 1\n")
    requirements = tmp_path / "requirements-linux.txt"
    export = b"example==1 --hash=sha256:" + b"0" * 64 + b"\n"
    requirements.write_bytes(export)
    # Unlisted source, labels and model weight files must never enter this image.
    (checkout / "private-labels.jsonl").write_text(SECRET)
    processor = w.inputs / "model"

    def entry(path, name):
        raw = path.read_bytes()
        return dict(filename=name, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())

    spec_data = w.settings.build_spec.model_dump()
    spec_data.update(
        lock_sha256=hashlib.sha256((checkout / "uv.lock").read_bytes()).hexdigest(),
        requirements_file=entry(requirements, requirements.name),
        cpu_test_file=entry(checkout / CPU_TEST_FILE, CPU_TEST_FILE),
        recipe_file=entry(checkout / RECIPE_FILE, RECIPE_FILE),
    )
    spec = m.BuildSpec.model_validate(spec_data)
    commands = []

    def run(args, **kwargs):
        commands.append(args)
        if args[:2] == ["uv", "export"]:
            assert "--offline" in args and "--frozen" in args and "--no-emit-project" in args
            output = b"wrong export" if damage == "export" else export
        elif args[1] == "rev-parse":
            output = ((("0" * 40) if damage == "revision" else spec.source_sha) + "\n").encode()
        else:
            assert args[1] == "status"
            output = b" M tracked.py" if damage == "dirty" else b""
        return SimpleNamespace(returncode=0, stdout=output)

    monkeypatch.setattr(app.subprocess, "run", run)  # Explicit Git/export boundary fake only.
    calls = []

    class Image:
        @classmethod
        def debian_slim(cls, **kwargs):
            calls.append(("base", kwargs))
            return cls()

        def pip_install_from_requirements(self, *args, **kwargs):
            calls.append(("requirements", args, kwargs))
            return self

        def add_local_file(self, *args, **kwargs):
            calls.append(("copy", args, kwargs))
            return self

        def env(self, value):
            calls.append(("env", value))
            return self

        def run_function(self, function, **kwargs):
            assert function is app.cpu_build_gate
            calls.append(("cpu", kwargs))
            return self

    volume = SimpleNamespace(object_id="vo-synthetic", with_mount_options=lambda **kw: kw)
    sdk = SimpleNamespace(__name__="independent_fake", Image=Image)
    damaged = {
        "code": checkout / "src" / CODE_FILES[0],
        "lock": checkout / "uv.lock",
        "processor": processor / "tokenizer.json",
        "test": checkout / CPU_TEST_FILE,
        "recipe": checkout / RECIPE_FILE,
    }
    if damage in damaged:
        damaged[damage].write_bytes(b"changed")
    kwargs = dict(
        checkout=checkout,
        requirements=requirements,
        processor_root=processor,
        output_volume=volume,
        sdk=sdk,
    )
    if damage:
        with pytest.raises(app.WorkerError):
            app.prepare_image(spec, **kwargs)
        assert calls == []
        return
    app.prepare_image(spec, **kwargs)
    copies = {args[1] for kind, *rest in calls if kind == "copy" for args in [rest[0]]}
    expected = {"/opt/ocr/" + name for name in CODE_FILES}
    expected |= {
        "/opt/ocr/" + CPU_TEST_FILE,
        "/opt/ocr/" + RECIPE_FILE,
        "/opt/ocr/uv.lock",
        "/opt/ocr/requirements-linux.txt",
    }
    expected |= {"/opt/ocr/processor/" + name for name in fm.PROCESSOR_FILES}
    assert copies == expected and len(copies) == len(CODE_FILES) + 4 + len(fm.PROCESSOR_FILES)
    assert all(rest[1] == {"copy": True} for kind, *rest in calls if kind == "copy")
    cpu = calls[-1][1]
    assert cpu["gpu"] is None and cpu["cpu"] == 2 and cpu["memory"] == 32768
    assert cpu["timeout"] == 900 and cpu["include_source"] is False
    assert cpu["volumes"] == {
        "/build-evidence": {"sub_path": "/builds/" + spec.sha256, "read_only": False}
    }
    assert calls[1][2] == {"extra_options": "--require-hashes"}
    assert len(commands) == 3


def test_real_cli_with_unmocked_clean_local_identity(tmp_path):
    """CLI in a clean checkout; only the provider transport has a fake implementation."""
    import shutil

    root = Path(fm.__file__).parents[3]
    checkout = tmp_path / "checkout"
    subprocess.run(["git", "clone", "--quiet", "--no-local", str(root), str(checkout)], check=True)
    # Snapshot the reviewed tree in a throwaway clone for clean CLI identity. An already
    # committed candidate needs no content changes, so an empty snapshot commit is allowed.
    for relative in ("src", "tests", "experiments/recipes"):
        shutil.rmtree(checkout / relative, ignore_errors=True)
        shutil.copytree(
            root / relative,
            checkout / relative,
            ignore=shutil.ignore_patterns("__pycache__"),
        )
    subprocess.run(["git", "-C", str(checkout), "add", "--all"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(checkout),
            "-c",
            "user.email=independent@review.invalid",
            "-c",
            "user.name=Independent Review",
            "commit",
            "--quiet",
            "--allow-empty",
            "--message=reviewed working tree",
        ],
        check=True,
    )
    program = """
import json, pathlib, runpy, sys
from typer.testing import CliRunner
from active_ocr.entrypoints import cli
from active_ocr.integrations import factory_model as fm
from active_ocr.integrations import modal_model as m
from active_ocr.integrations.simulation import local_contract_identity, create_fixture
from active_ocr.models import SimulationConfig
from active_ocr.pipeline import Pipeline
helper=runpy.run_path(sys.argv[1])
work=pathlib.Path(sys.argv[2])
checkout=pathlib.Path.cwd()
assert pathlib.Path(m.__file__).resolve().is_relative_to(checkout)
revision,dependencies=local_contract_identity()
assert len(revision)==40 and len(dependencies)==64
runtime=helper['settings'].__wrapped__()
data=runtime.model_dump()
data['build_spec']['source_sha']=revision
spec=m.BuildSpec.model_validate(data['build_spec'])
data['build'].update(source_sha=revision,build_spec_sha256=spec.sha256)
build=m.BuildReceipt.model_validate(data['build'])
identity=data['context']['real_config']['expected_identity']
identity.update(source_sha=revision,dependency_sha256=dependencies,
    code_bundle_sha256=build.code_bundle_sha256,build_spec_sha256=spec.sha256)
runtime=m.RuntimeSettings.model_validate(data)
manifest=create_fixture(work/'data',train_pages=3)
config=SimulationConfig(real=runtime.context.real_config,backend=fm.BACKEND,
    evaluator_id=helper['EVALUATOR'],max_rounds=1,batch_size=1,validation_page_ids=('page-3',))
configuration=work/'config.json'
configuration.write_text(config.model_dump_json())
directory=work/'run'
runner=CliRunner()
def invoke(*args,ok=True):
    result=runner.invoke(cli.app,['real',*map(str,args)])
    assert (result.exit_code==0)==ok, str(result.exception)
    return result
run_id=invoke('create',manifest,directory,configuration).output.strip()
pipeline=Pipeline.for_simulation(directory)
run=pipeline.get_simulation(run_id)
model=helper['attach_baseline'](pipeline,run,runtime)
runtime=model.settings
settings_file=work/'runtime.json'
settings_file.write_text(runtime.model_dump_json())
deployment=work/'deployment.json'
deployment.write_text(m.DeploymentObservation(settings_sha256=helper['digest'](runtime.model_dump(mode='json')),
    function_id='fu-synthetic',app_id='ap-synthetic').model_dump_json())
transport=helper['FullByteTransport'](runtime,helper['adapter_bytes'].__wrapped__())
m.SDKTransport=lambda settings,observed:transport
assert json.loads(invoke('preflight',directory,run_id,settings_file).output)['local']=='verified'
assert transport.calls==[]
# Wrong measured local dependency identity rejects before any model submission.
bad=config.model_dump()
bad['real']['expected_identity']['dependency_sha256']='0'*64
configuration.write_text(json.dumps(bad))
rejected=invoke('create',manifest,work/'bad-run',configuration,ok=False)
assert 'local code/dependency identity' in str(rejected.exception)
# A tracked local edit invalidates resume/preflight even though the saved recipe is unchanged.
readme=checkout/'README.md'
original=readme.read_bytes()
try:
    readme.write_bytes(original+b'\\nidentity test mutation\\n')
    rejected=invoke('preflight',directory,run_id,settings_file,ok=False)
    assert 'local code/dependency identity' in str(rejected.exception)
    assert transport.calls==[]
finally:
    readme.write_bytes(original)
baseline=json.loads(invoke('resume',directory,run_id,settings_file,deployment,'--one-round').output)
assert baseline['baseline']['labelled_count']==0 and baseline['rounds']==[]
done=json.loads(invoke('run',directory,run_id,settings_file,deployment).output)
assert done['complete'] and len(done['rounds'])==1
assert len(transport.calls)==4
invoke('resume',directory,run_id,settings_file,deployment)
assert len(transport.calls)==4
status=invoke('status',directory,run_id).output
assert 'Synthetic page' not in status
assert all(r['state']=='COMPLETED' for r in json.loads(status)['operations'])
invoke('export',directory,run_id,work/'export')
assert json.loads((work/'export/results.json').read_text())['complete'] is True
membership=json.loads((work/'export/validation.json').read_text())
assert membership==dict(page_ids=['page-3'],page_count=1)
assert local_contract_identity()==(revision,dependencies)
print('clean identity, rejection, baseline, fit, resume, status and export verified')
"""
    result = subprocess.run(
        [sys.executable, "-c", program, str(Path(__file__).resolve()), str(tmp_path)],
        cwd=checkout,
        env={**os.environ, "PYTHONPATH": str(checkout / "src")},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert (
        "clean identity, rejection, baseline, fit, resume, status and export verified"
        in result.stdout
    )
    assert subprocess.check_output(["git", "status", "--porcelain"], cwd=checkout) == b""


@pytest.mark.parametrize(
    "damage",
    [None, "missing_receipt", "corrupt_report", "skipped_report", "source", "noncanonical"],
)
def test_build_receipt_retrieval_never_rebuilds_missing_or_bad_evidence(
    settings, monkeypatch, damage
):
    from types import SimpleNamespace

    from active_ocr.entrypoints import modal_app as app

    spec = settings.build_spec
    report = dict(
        selected=5,
        passed=5,
        skipped=0,
        failed=0,
        tests=sorted(
            [
                "test_factory_model::test_actual_template_is_the_pinned_nothink_variant",
                "test_factory_model::test_actual_template_supervises_the_whole_target_and_masks_the_prompt",
                "test_factory_model::test_actual_encoded_lengths_are_measured_and_overlong_targets_fail",
                "test_factory_model::test_actual_native_lora_starts_with_zero_B_and_changes_after_training",
                "test_factory_model::test_actual_native_cli_trains_and_fresh_process_reloads",
            ]
        ),
    )
    if damage == "skipped_report":
        report.update(passed=10, skipped=1)
    raw = encoded(report)
    receipt = settings.build.model_copy(
        update={
            "cpu_report": m.ArtifactRef(
                key="cpu-report.json", bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()
            ),
            "cpu_passed": report["passed"],
        }
    )
    if damage == "source":
        receipt = receipt.model_copy(update={"source_sha": "0" * 40})
    prefix = "/builds/" + spec.sha256 + "/"
    files = {
        prefix + "build-receipt.json": encoded(receipt.model_dump(mode="json")),
        prefix + "cpu-report.json": raw,
    }
    if damage == "missing_receipt":
        del files[prefix + "build-receipt.json"]
    elif damage == "corrupt_report":
        files[prefix + "cpu-report.json"] += b"!"
    elif damage == "noncanonical":
        files[prefix + "build-receipt.json"] = receipt.model_dump_json(indent=2).encode()
    calls = []

    class Image:
        def build(self, app):
            calls.append("build")
            self.object_id = "im-observed-synthetic"

    def read(key):
        assert calls == ["build"]
        calls.append(key)
        try:
            return [files[key]]
        finally:
            calls.pop()

    monkeypatch.setattr(app, "prepare_image", lambda *args, **kw: Image())
    kwargs = dict(
        app=object(),
        checkout=Path("."),
        requirements=Path("."),
        processor_root=Path("."),
        output_volume=SimpleNamespace(read_file=read),
    )
    if damage:
        with pytest.raises((app.WorkerError, KeyError)):
            app.build_image(spec, **kwargs)
    else:
        image, observed, observed_report = app.build_image(spec, **kwargs)
        assert (
            image == "im-observed-synthetic" and observed == receipt and observed_report == report
        )
    assert calls == ["build"]


@pytest.mark.parametrize("damage", [None, "image", "model", "symlink"])
def test_input_upload_plan_excludes_labels_and_validates_before_batch(worker_fixture, damage):
    from contextlib import contextmanager
    from types import SimpleNamespace

    w = worker_fixture
    pages = [
        SimpleNamespace(image_uri=str(w.inputs / f.filename), image_sha256=f.sha256)
        for f in w.settings.bundle.files
        if f.filename.startswith("images/")
    ]
    (w.inputs / "model/private-labels.xml").write_text(SECRET)
    if damage == "image":
        Path(pages[0].image_uri).write_bytes(b"changed")
    elif damage == "model":
        (w.inputs / "model/tokenizer.json").write_bytes(b"changed")
    elif damage == "symlink":
        path = w.inputs / "model/tokenizer.json"
        raw = path.read_bytes()
        path.unlink()
        outside = w.outputs / "tokenizer.json"
        outside.write_bytes(raw)
        path.symlink_to(outside)
    uploaded = []
    batches = []

    @contextmanager
    def batch_upload(*, force):
        assert force is False
        batches.append(True)
        yield SimpleNamespace(put_file=lambda path, key: uploaded.append((path, key)))

    volume = SimpleNamespace(batch_upload=batch_upload)
    if damage:
        with pytest.raises(ValueError):
            m.upload_input_bundle(volume, w.settings.bundle, pages, w.inputs / "model")
        assert batches == uploaded == []
    else:
        m.upload_input_bundle(volume, w.settings.bundle, pages, w.inputs / "model")
        assert len(batches) == 1
        prefix = "/bundles/" + w.settings.bundle.sha256 + "/"
        assert {key for _, key in uploaded} == {
            prefix + f.filename for f in w.settings.bundle.files
        }
        assert all("private-labels" not in str(path) for path, _ in uploaded)
