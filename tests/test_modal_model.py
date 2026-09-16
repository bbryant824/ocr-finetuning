"""Shared API checks only; no Modal SDK, provider operation or model execution."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from active_ocr.integrations import modal_model as m
from active_ocr.integrations import qwen as q
from active_ocr.models import (
    Box,
    ExpectedIdentity,
    Prediction,
    RealOCRConfig,
    SourcePolicy,
    SourceRegion,
    Split,
)


def assets():
    return m.InputBundle(
        files=tuple(
            sorted(
                [
                    q.FileEntry(filename="model/" + n, bytes=size, sha256=sha)
                    for n, (size, sha) in q.ASSETS.items()
                ]
                + [
                    q.FileEntry(filename="images/" + "a" * 64, bytes=100, sha256="a" * 64),
                    q.FileEntry(filename="images/" + "b" * 64, bytes=100, sha256="b" * 64),
                ],
                key=lambda f: f.filename,
            )
        )
    )


def make_settings():
    bundle = assets()
    spec = m.BuildSpec(
        source_sha="c" * 40,
        code_files=tuple(
            q.FileEntry(filename=n, bytes=1, sha256="d" * 64) for n in sorted(m.RUNTIME_CODE_FILES)
        ),
        lock_sha256="e" * 64,
        requirements_file=q.FileEntry(filename="requirements-linux.txt", bytes=1, sha256="e" * 64),
        cpu_test_file=q.FileEntry(filename="tests/test_qwen.py", bytes=1, sha256="f" * 64),
        processor_manifest_sha256=q.digest(q.asset_manifest(q.PROCESSOR_FILES)),
    )
    build = m.BuildReceipt(
        build_spec_sha256=spec.sha256,
        source_sha=spec.source_sha,
        code_files=spec.code_files,
        environment=q.Packages(python="3.11.14", packages=tuple(sorted(q.PINS.items()))),
        cpu_report=m.ArtifactRef(key="cpu-report.json", bytes=10, sha256="a" * 64),
        cpu_passed=11,
        cpu_skipped=0,
    )
    expected = ExpectedIdentity(
        source_sha=spec.source_sha,
        dependency_sha256="f" * 64,
        model_revision=q.REVISION,
        processor_revision=q.REVISION,
        model_manifest_sha256=q.digest(q.asset_manifest(q.BASE_FILES)),
        processor_manifest_sha256=spec.processor_manifest_sha256,
        recipe_version=q.RECIPE,
        evaluator_id="page-text-nfc-v1",
        code_bundle_sha256=build.code_bundle_sha256,
        remote_dependency_sha256=build.remote_dependency_sha256,
        build_spec_sha256=spec.sha256,
        deployment_reference="modal:main/synthetic/dispatch@im-test",
    )
    real = RealOCRConfig(
        backend="qwen3-vl-v1",
        recipe_version=q.RECIPE,
        model_repository=q.REPOSITORY,
        processor_repository=q.REPOSITORY,
        model_revision=q.REVISION,
        processor_revision=q.REVISION,
        training_policy_id=q.TRAIN_POLICY,
        decode_policy_id=q.DECODE_POLICY,
        expected_identity=expected,
    )
    context = m.RunContext(
        experiment_id="1" * 32,
        source_policy=SourcePolicy.READ2016,
        bundle_sha256=bundle.sha256,
        real_config=real,
    )
    return m.RuntimeSettings(
        context=context,
        environment_name="main",
        deployment_name="synthetic",
        image_id="im-test",
        input_volume_name="input",
        output_volume_name="output",
        input_volume_id="vo-input",
        output_volume_id="vo-output",
        build_spec=spec,
        build=build,
        bundle=bundle,
    )


def remote_page(*, split=Split.TRAIN, id="one", sha="a" * 64, document_id=None):
    return m.RemotePage(
        id=id,
        document_id=document_id,
        split=split,
        width=100,
        height=200,
        image_sha256=sha,
        image_key="images/" + sha,
    )


def fit_request(settings=None, text="SELECTED SECRET"):
    context = (settings or make_settings()).context
    example = m.RemoteExample(
        page=remote_page(),
        regions=(SourceRegion(id="line", text=text, box=Box(x=0, y=0, width=10, height=20)),),
    )
    return m.FitRequest(
        context=context,
        round_number=1,
        input_checkpoint="checkpoint:sha256:" + "0" * 64,
        seed=824,
        examples=(example,),
    )


def invocation(request):
    return m.Invocation(
        request=request,
        attempt_id="2" * 32,
        first_submission_unix_seconds=1000,
        deadline_unix_seconds=3400,
    )


def test_shared_module_import_is_local_and_lightweight():
    code = (
        "import sys; import active_ocr.integrations.modal_model; "
        "assert not {'modal','torch','peft','transformers'} & sys.modules.keys()"
    )
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        env={"PYTHONPATH": str(Path(q.__file__).parents[2])},
    )


def test_rpc_roundtrip_and_semantic_target_redaction():
    request = fit_request()
    call = invocation(request)
    restored = m.Invocation.model_validate_json(call.model_dump_json())
    assert restored == call
    assert "SELECTED SECRET" in call.model_dump_json()  # Needed only in ephemeral fit RPC.
    assert "SELECTED SECRET" not in json.dumps(request.semantic_record())
    assert "SELECTED SECRET" not in repr(call)
    assert m.operation_id(request) != m.operation_id(fit_request(text="different selected text"))
    changed = call.model_copy(update={"attempt_id": "3" * 32, "deadline_unix_seconds": 3200})
    assert changed.operation_id == call.operation_id
    assert m.operation_id(request.model_copy(update={"seed": 825})) != call.operation_id


@pytest.mark.parametrize("field", ["source_image", "regions", "image_uri", "provenance", "targets"])
def test_remote_page_excludes_source_and_label_fields(field):
    data = remote_page().model_dump(mode="json")
    data[field] = "FORBIDDEN"
    with pytest.raises(ValidationError):
        m.RemotePage.model_validate(data)


@pytest.mark.parametrize(
    "key", ["../escape", "/host/file", "images/../secret", "images//bad", "images/abc"]
)
def test_remote_image_key_is_exact_content_address(key):
    data = remote_page().model_dump()
    data["image_key"] = key
    with pytest.raises(ValidationError):
        m.RemotePage.model_validate(data)


def test_unknown_document_policy_is_explicit():
    request = fit_request()
    data = request.model_dump(mode="json")
    del data["context"]["source_policy"]
    with pytest.raises(ValidationError):
        m.FitRequest.model_validate(data)
    data["context"]["source_policy"] = SourcePolicy.KNOWN_DOCUMENT.value
    with pytest.raises(ValidationError):
        m.FitRequest.model_validate(data)
    assert request.context.source_policy is SourcePolicy.READ2016


@pytest.mark.parametrize("split", [Split.VALIDATION, Split.TEST])
def test_held_out_fit_targets_rejected(split):
    with pytest.raises(ValidationError):
        m.RemoteExample(page=remote_page(split=split), regions=())


@pytest.mark.parametrize("kind", ["load_base", "predict"])
def test_label_free_operations_reject_targets(kind):
    context = make_settings().context
    request = (
        m.BaseRequest(context=context)
        if kind == "load_base"
        else m.PredictRequest(
            context=context,
            round_number=0,
            purpose="baseline_validation",
            input_checkpoint="checkpoint:sha256:" + "0" * 64,
            pages=(remote_page(split=Split.VALIDATION),),
        )
    )
    data = request.model_dump(mode="json")
    data["examples"] = fit_request().model_dump(mode="json")["examples"]
    with pytest.raises(ValidationError):
        m.REQUEST_ADAPTER.validate_python(data)


def test_semantic_hash_binds_order_purpose_checkpoint_and_context():
    request = m.PredictRequest(
        context=make_settings().context,
        round_number=1,
        purpose="validation",
        input_checkpoint="checkpoint:sha256:" + "0" * 64,
        pages=(
            remote_page(split=Split.VALIDATION),
            remote_page(id="two", sha="b" * 64, split=Split.VALIDATION),
        ),
    )
    original = m.operation_id(request)
    for change in (
        {"pages": tuple(reversed(request.pages))},
        {"purpose": "pool"},
        {"input_checkpoint": "checkpoint:sha256:" + "9" * 64},
        {"context": request.context.model_copy(update={"experiment_id": "5" * 32})},
    ):
        assert m.operation_id(request.model_copy(update=change)) != original


def test_deployment_mounts_and_bound_context():
    settings = make_settings()
    assert settings.input_sub_path == "/bundles/" + settings.bundle.sha256
    assert settings.output_sub_path == "/runs/" + "1" * 32
    settings.check_invocation(invocation(fit_request(settings)))
    for field, value in (("image_id", "different-image"), ("input_volume_id", "vo-output")):
        with pytest.raises(ValidationError):
            m.RuntimeSettings.model_validate({**settings.model_dump(), field: value})
    bad = invocation(fit_request(settings)).model_copy(
        update={
            "request": fit_request().model_copy(
                update={"context": settings.context.model_copy(update={"experiment_id": "5" * 32})}
            )
        }
    )
    with pytest.raises(ValueError, match="deployed run"):
        settings.check_invocation(bad)


def test_bundle_rejects_unlisted_source_and_changed_model():
    data = assets().model_dump(mode="json")
    data["files"].append(dict(filename="source.xml", bytes=1, sha256="d" * 64))
    data["files"].sort(key=lambda f: f["filename"])
    with pytest.raises(ValidationError):
        m.InputBundle.model_validate(data)
    data = assets().model_dump(mode="json")
    data["files"][-1]["sha256"] = "e" * 64
    with pytest.raises(ValidationError):
        m.InputBundle.model_validate(data)


def test_cpu_receipt_cannot_pass_with_skips_or_different_build():
    settings = make_settings()
    with pytest.raises(ValidationError):
        m.BuildReceipt.model_validate({**settings.build.model_dump(), "cpu_skipped": 1})
    data = settings.model_dump()
    data["build"]["build_spec_sha256"] = "f" * 64
    with pytest.raises(ValidationError):
        m.RuntimeSettings.model_validate(data)


@pytest.mark.parametrize("duration", [0, -1, 2401])
def test_deadline_envelope(duration):
    data = invocation(fit_request()).model_dump()
    data["deadline_unix_seconds"] = data["first_submission_unix_seconds"] + duration
    with pytest.raises(ValidationError):
        m.Invocation.model_validate(data)


def test_completion_checksum_owner_and_raw_evidence():
    settings = make_settings()
    request = m.PredictRequest(
        context=settings.context,
        round_number=0,
        purpose="baseline_validation",
        input_checkpoint="checkpoint:sha256:" + "0" * 64,
        pages=(remote_page(split=Split.VALIDATION),),
    )
    call = invocation(request)
    worker = m.ArtifactRef(key="worker.json", bytes=1, sha256="d" * 64)
    raw = m.ArtifactRef(key="qwen/receipts/raw.json", bytes=1, sha256="e" * 64)
    manifest = m.ArtifactRef(
        key="qwen/checkpoints/" + "0" * 64 + "/manifest.json", bytes=1, sha256="0" * 64
    )
    result = m.OperationResult(
        operation_id=call.operation_id,
        attempt_id=call.attempt_id,
        execution_id="3" * 32,
        provider_call_id="fc-synthetic",
        model_id=request.input_checkpoint,
        artifacts=(worker, raw, manifest),
        worker_manifest=worker,
        predictions=(
            Prediction(
                page_id="one",
                experiment_id=settings.context.experiment_id,
                round_number=0,
                purpose="baseline_validation",
                model_id=request.input_checkpoint,
                raw_output_artifact=raw.key,
            ),
        ),
    )
    result.check_request(call)
    key = (
        f"operations/{result.operation_id}/attempts/{result.attempt_id}/"
        f"executions/{result.execution_id}/complete.json"
    )
    encoded = q.canonical(result.model_dump(mode="json"))
    completion = m.ArtifactRef(
        key=key, bytes=len(encoded), sha256=q.digest(result.model_dump(mode="json"))
    )
    m.DispatchResponse(result=result, completion=completion)
    with pytest.raises(ValidationError):
        m.DispatchResponse(
            result=result, completion=completion.model_copy(update={"sha256": "f" * 64})
        )
    with pytest.raises(ValueError, match="ownership"):
        result.check_request(call.model_copy(update={"attempt_id": "4" * 32}))
    with pytest.raises(ValidationError):
        m.OperationResult.model_validate(
            {**result.model_dump(), "artifacts": [worker.model_dump()]}
        )


def test_reload_proof_requires_fresh_process_and_full_logits():
    ref = m.ArtifactRef(key="before.json", bytes=1, sha256="a" * 64)
    with pytest.raises(ValidationError):
        m.ReloadEvidence(
            before=ref,
            after=ref,
            train_process_id=7,
            reload_process_id=7,
            max_absolute_difference=0,
            exact_tensors_ids_status_regions=True,
        )
    data = dict(
        model_id="checkpoint:sha256:" + "0" * 64,
        page_id="selected",
        image_sha256="a" * 64,
        original_width=100,
        original_height=200,
        process_id=7,
        generated_ids=(q.EOS,),
        status="invalid_output",
        regions=(),
        finish_reason="eos",
        adapter_tensors={k: "b" * 64 for k in q.adapter_shapes()},
        logits=m.ArtifactRef(key="before.f32le", bytes=151936 * 4, sha256="c" * 64),
        logits_shape=(1, 151936),
    )
    m.ProbeMetadata(**data)
    with pytest.raises(ValidationError):
        m.ProbeMetadata(**{**data, "logits_shape": (2, 151936)})
    with pytest.raises(ValidationError):
        m.ProbeMetadata(**{**data, "adapter_tensors": {}})


class FakeTransport:
    def __init__(self, settings):
        self.settings = settings
        self.payloads = []
        self.files = {}
        self.responses = {}
        self.cancelled = []
        self.lose_ack = False
        self.get_error = None
        self.on_spawn = None
        self.preflights = 0

    def preflight(self):
        self.preflights += 1

    def artifact(self, key, data):
        import hashlib

        self.files[key] = data
        return m.ArtifactRef(key=key, bytes=len(data), sha256=hashlib.sha256(data).hexdigest())

    def base_response(self, invocation, call_id):
        model = object.__new__(q.QwenModel)
        model.real_config = self.settings.context.real_config
        checkpoint = q.Checkpoint(kind="base", binding=model._binding())
        raw = q.canonical(checkpoint.model_dump(mode="json"))
        model_sha = q.digest(checkpoint.model_dump(mode="json"))
        manifest = self.artifact(f"qwen/checkpoints/{model_sha}/manifest.json", raw)
        worker = q.WorkerManifest(
            schema_version=1,
            source_sha=self.settings.build.source_sha,
            files=self.settings.build.code_files,
            environment=self.settings.build.environment,
            build_spec_sha256=self.settings.build_spec.sha256,
            deployment_reference=self.settings.deployment_reference,
        )
        worker_ref = self.artifact("worker.json", q.canonical(worker.model_dump(mode="json")))
        result = m.OperationResult(
            operation_id=invocation.operation_id,
            attempt_id=invocation.attempt_id,
            execution_id="8" * 32,
            provider_call_id=call_id,
            model_id="checkpoint:sha256:" + model_sha,
            artifacts=(manifest, worker_ref),
            worker_manifest=worker_ref,
        )
        completion = self.artifact(
            f"operations/{invocation.operation_id}/attempts/{invocation.attempt_id}/"
            f"executions/{'8' * 32}/complete.json",
            q.canonical(result.model_dump(mode="json")),
        )
        return m.DispatchResponse(result=result, completion=completion).model_dump(mode="json")

    def spawn(self, payload):
        self.payloads.append(payload)
        call_id = "fc-" + str(len(self.payloads))
        if self.on_spawn:
            self.on_spawn(payload)
        invocation = m.Invocation.model_validate(payload)
        self.responses[call_id] = self.base_response(invocation, call_id)
        if self.lose_ack:
            raise ConnectionError("lost acknowledgement")
        return call_id

    def get(self, call_id, timeout):
        if self.get_error:
            raise self.get_error
        return self.responses[call_id]

    def read(self, key):
        yield self.files[key]

    def cancel(self, call_id):
        self.cancelled.append(call_id)


def coordinator(tmp_path):
    from active_ocr.integrations.storage import SQLiteStore

    settings = make_settings()
    store = SQLiteStore(tmp_path / "state.db", tmp_path / "artifacts")
    store.initialize()
    transport = FakeTransport(settings)
    model = m.ModalModel(settings, store, transport, clock=lambda: 1000)
    return model, transport, m.BaseRequest(context=settings.context)


def test_completed_operation_is_verified_and_reused_after_response_loss(tmp_path):
    model, transport, request = coordinator(tmp_path)
    first = model.execute(request)
    resumed = m.ModalModel(model.settings, model.store, transport, clock=lambda: 9000)
    assert resumed.execute(request) == first  # Reuse is valid even after the deadline.
    assert len(transport.payloads) == 1
    record = model.store.load("model-operation", m.operation_id(request), m.ModelOperation)
    assert record.state == "COMPLETED" and record.provider_call_id == "fc-1"


def test_lost_spawn_ack_never_resubmits_and_explicit_reconcile_preserves_producer(tmp_path):
    model, transport, request = coordinator(tmp_path)
    transport.lose_ack = True
    with pytest.raises(RuntimeError, match="submission outcome unknown"):
        model.execute(request)
    with pytest.raises(RuntimeError, match="UNKNOWN"):
        model.execute(request)
    assert len(transport.payloads) == 1
    result = model.reconcile(request, call_id="fc-1")
    assert result.provider_call_id == "fc-1"
    assert result.attempt_id == transport.payloads[0]["attempt_id"]
    assert model.execute(request) == result


def test_known_call_reconciliation_after_get_error(tmp_path):
    model, transport, request = coordinator(tmp_path)
    transport.get_error = ConnectionError("disconnected")
    with pytest.raises(RuntimeError, match="known call retained"):
        model.execute(request)
    record = model.store.load("model-operation", m.operation_id(request), m.ModelOperation)
    assert record.provider_call_id == "fc-1" and record.state == "UNKNOWN"
    transport.get_error = None
    assert model.reconcile(request).model_id.startswith("checkpoint:")
    assert len(transport.payloads) == 1


def test_reservation_precedes_spawn_and_concurrent_submission_is_blocked(tmp_path):
    model, transport, request = coordinator(tmp_path)

    def nested(payload):
        record = model.store.load("model-operation", m.operation_id(request), m.ModelOperation)
        assert record.state == "SUBMITTING"
        competitor = m.ModalModel(model.settings, model.store, transport, clock=lambda: 1000)
        with pytest.raises(RuntimeError, match="interrupted submission"):
            competitor.execute(request)

    transport.on_spawn = nested
    with pytest.raises(RuntimeError, match="submission outcome unknown"):
        model.execute(request)
    assert len(transport.payloads) == 1
    # The winner's late acknowledgement cannot overwrite the competitor's UNKNOWN checkpoint.
    assert model.reconcile(request, call_id="fc-1").provider_call_id == "fc-1"


def test_known_call_is_durable_before_wait_and_cancellation_is_not_terminal(tmp_path):
    model, transport, request = coordinator(tmp_path)

    def get(call_id, timeout):
        record = model.store.load("model-operation", m.operation_id(request), m.ModelOperation)
        assert record.state == "RUNNING" and record.provider_call_id == call_id
        model.clock = lambda: 3401
        raise TimeoutError()

    transport.get = get
    with pytest.raises(TimeoutError, match="terminal status unverified"):
        model.execute(request)
    record = model.store.load("model-operation", m.operation_id(request), m.ModelOperation)
    assert record.state == "CANCEL_REQUESTED" and transport.cancelled == ["fc-1"]
    with pytest.raises(RuntimeError, match="CANCEL_REQUESTED"):
        model.execute(request)
    assert len(transport.payloads) == 1


@pytest.mark.parametrize("corruption", ["bytes", "worker", "call", "attempt", "recipe"])
def test_unverified_receipt_never_completes(tmp_path, corruption):
    model, transport, request = coordinator(tmp_path)
    original = transport.base_response

    def response(invocation, call_id):
        result = original(invocation, call_id)
        if corruption == "bytes":
            transport.files[result["result"]["worker_manifest"]["key"]] = b"bad"
        elif corruption == "worker":
            key = result["result"]["worker_manifest"]["key"]
            worker = json.loads(transport.files[key])
            worker["source_sha"] = "0" * 40
            ref = transport.artifact(key, q.canonical(worker))
            result["result"]["worker_manifest"] = ref.model_dump()
            result["result"]["artifacts"][1] = ref.model_dump()
        elif corruption == "call":
            result["result"]["provider_call_id"] = "fc-wrong"
        elif corruption == "attempt":
            result["result"]["attempt_id"] = "f" * 32
        else:
            key = result["result"]["artifacts"][0]["key"]
            transport.files[key] = transport.files[key].replace(
                b"ordered-regions", b"changed-regions"
            )
        if corruption in ("worker", "call", "attempt"):
            r = m.OperationResult.model_validate(result["result"])
            completion = transport.artifact(
                f"operations/{r.operation_id}/attempts/{r.attempt_id}/executions/{r.execution_id}/complete.json",
                q.canonical(r.model_dump(mode="json")),
            )
            result = m.DispatchResponse(result=r, completion=completion).model_dump(mode="json")
        return result

    transport.base_response = response
    with pytest.raises(ValueError):
        model.execute(request)
    assert (
        model.store.load("model-operation", m.operation_id(request), m.ModelOperation).state
        == "UNKNOWN"
    )


def test_completed_evidence_corruption_is_not_silently_repaired(tmp_path):
    model, transport, request = coordinator(tmp_path)
    result = model.execute(request)
    (model.store.artifact_root / "model-evidence" / result.worker_manifest.sha256).write_bytes(
        b"bad"
    )
    with pytest.raises(ValueError, match="local model evidence corrupted"):
        model.execute(request)
    assert len(transport.payloads) == 1


def test_failure_stays_terminal_without_retry(tmp_path):
    model, transport, request = coordinator(tmp_path)

    def failure(invocation, call_id):
        return m.FailureReceipt(
            operation_id=invocation.operation_id,
            attempt_id=invocation.attempt_id,
            execution_id="4" * 32,
            provider_call_id=call_id,
            code="model",
        ).model_dump(mode="json")

    transport.base_response = failure
    with pytest.raises(RuntimeError, match="worker failed"):
        model.execute(request)
    with pytest.raises(RuntimeError, match="FAILED"):
        model.execute(request)
    assert len(transport.payloads) == 1


def test_fit_reservation_never_persists_targets(tmp_path):
    model, transport, _ = coordinator(tmp_path)
    base = model.execute(m.BaseRequest(context=model.settings.context))
    request = fit_request(model.settings, text="UNIQUE_SECRET_SELECTED_TARGET").model_copy(
        update={"input_checkpoint": base.model_id}
    )

    def lost(payload):
        assert (
            payload["request"]["examples"][0]["regions"][0]["text"]
            == "UNIQUE_SECRET_SELECTED_TARGET"
        )
        raise ConnectionError()

    transport.spawn = lost
    with pytest.raises(RuntimeError):
        model.execute(request)
    assert b"UNIQUE_SECRET_SELECTED_TARGET" not in model.store.database_path.read_bytes()
    record = model.store.load("model-operation", m.operation_id(request), m.ModelOperation)
    assert "target_sha256" in record.semantic and "examples" not in record.semantic


class CompleteTransport(FakeTransport):
    """Synthetic checkpoint/logits bytes exercising the real coordinator verifier, no ML."""

    def base_response(self, invocation, call_id):
        import hashlib
        import math
        import struct

        request = invocation.request
        if isinstance(request, m.BaseRequest):
            response = super().base_response(invocation, call_id)
            self.base = response
            return response
        base = m.DispatchResponse.model_validate(self.base).result
        worker_ref = base.worker_manifest
        artifacts = [worker_ref]
        reload = None
        predictions = ()
        if isinstance(request, m.FitRequest):
            model = object.__new__(q.QwenModel)
            model.real_config = self.settings.context.real_config
            shapes = q.adapter_shapes()
            header, content, tensors = {}, bytearray(), {}
            for key, shape in shapes.items():
                data = bytes(math.prod(shape) * 4)
                header[key] = dict(
                    dtype="F32",
                    shape=list(shape),
                    data_offsets=[len(content), len(content) + len(data)],
                )
                tensors[key] = hashlib.sha256(
                    q.canonical(dict(shape=list(shape), dtype="torch.float32")) + data
                ).hexdigest()
                content.extend(data)
            encoded = q.canonical(header)
            adapter = struct.pack("<Q", len(encoded)) + encoded + content
            cfg = q.canonical(dict(base_model_name_or_path=q.REPOSITORY, revision=q.REVISION))
            files = (
                q.FileEntry(
                    filename="adapter_config.json",
                    bytes=len(cfg),
                    sha256=hashlib.sha256(cfg).hexdigest(),
                ),
                q.FileEntry(
                    filename="adapter_model.safetensors",
                    bytes=len(adapter),
                    sha256=hashlib.sha256(adapter).hexdigest(),
                ),
            )
            ids = tuple(e.page.id for e in request.examples)
            processing = {
                i: dict(
                    height=256,
                    width=256,
                    grid=[1, 16, 16],
                    prompt_tokens=1,
                    target_tokens=1,
                    target_exceeds_decode_budget=False,
                )
                for i in ids
            }
            updates = 3 * math.ceil(len(ids) / 4)
            training = dict(
                updates=updates,
                epoch_orders=[
                    [p for group in q.epoch_groups(ids, request.seed, e) for p in group]
                    for e in range(3)
                ],
                losses=[0.1] * (3 * len(ids)),
                gradient_norms=[0.1] * updates,
                processing=processing,
                supervised_tokens=3 * len(ids),
                changed_tensors=1,
                frozen_sha256="c" * 64,
            )
            checkpoint = q.Checkpoint(
                kind="adapter",
                binding=model._binding(),
                experiment_id=request.context.experiment_id,
                round_number=request.round_number,
                selected=tuple((e.page.id, e.page.image_sha256) for e in request.examples),
                target_sha256=q.digest(
                    [(e.page.id, q.serialize_target(e)) for e in request.examples]
                ),
                seed=request.seed,
                training=training,
                files=files,
            )
            model_sha = q.digest(checkpoint.model_dump(mode="json"))
            model_id = "checkpoint:sha256:" + model_sha
            prefix = f"qwen/checkpoints/{model_sha}/"
            artifacts += [
                self.artifact(
                    prefix + "manifest.json", q.canonical(checkpoint.model_dump(mode="json"))
                ),
                self.artifact(prefix + "adapter_config.json", cfg),
                self.artifact(prefix + "adapter_model.safetensors", adapter),
            ]
            probes = []
            page = min((e.page for e in request.examples), key=lambda p: p.id)
            for name, pid in (("before", 123), ("after", 124)):
                key = f"probes/{model_sha}/{name}"
                logits = self.artifact("qwen/" + key + ".f32le", bytes(151936 * 4))
                probe = m.ProbeMetadata(
                    model_id=model_id,
                    page_id=page.id,
                    image_sha256=page.image_sha256,
                    original_width=page.width,
                    original_height=page.height,
                    process_id=pid,
                    generated_ids=(q.EOS,),
                    status="invalid_output",
                    regions=(),
                    finish_reason="eos",
                    adapter_tensors=tensors,
                    logits=logits.model_copy(update={"key": key + ".f32le"}),
                    logits_shape=(1, 151936),
                )
                ref = self.artifact(
                    "qwen/" + key + ".json", q.canonical(probe.model_dump(mode="json"))
                )
                artifacts += [logits, ref]
                probes.append(ref)
            reload = m.ReloadEvidence(
                before=probes[0],
                after=probes[1],
                train_process_id=123,
                reload_process_id=124,
                max_absolute_difference=0,
                exact_tensors_ids_status_regions=True,
            )
            self.fitted_artifacts = tuple(artifacts)
        else:
            model_id = request.input_checkpoint
            model_sha = model_id.rsplit(":", 1)[1]
            previous = base.artifacts if not request.round_number else self.fitted_artifacts
            artifacts += [
                ref for ref in previous if ref.key.startswith(f"qwen/checkpoints/{model_sha}/")
            ]
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
                        ids=[q.EOS],
                        text='{"regions":[]}',
                        finish_reason="eos",
                        status="ok",
                    )
                    for p in request.pages
                ],
            )
            raw = self.artifact(
                "qwen/receipts/" + q.digest(receipt) + ".json", q.canonical(receipt)
            )
            artifacts.append(raw)
            predictions = tuple(
                Prediction(
                    page_id=p.id,
                    experiment_id=request.context.experiment_id,
                    round_number=request.round_number,
                    purpose=request.purpose,
                    model_id=model_id,
                    finish_reason="eos",
                    raw_output_artifact=raw.key,
                )
                for p in request.pages
            )
        result = m.OperationResult(
            operation_id=invocation.operation_id,
            attempt_id=invocation.attempt_id,
            execution_id="8" * 32,
            provider_call_id=call_id,
            model_id=model_id,
            artifacts=tuple(artifacts),
            worker_manifest=worker_ref,
            reload=reload,
            predictions=predictions,
        )
        completion = self.artifact(
            f"operations/{invocation.operation_id}/attempts/{invocation.attempt_id}/"
            f"executions/{'8' * 32}/complete.json",
            q.canonical(result.model_dump(mode="json")),
        )
        return m.DispatchResponse(result=result, completion=completion).model_dump(mode="json")


def test_completed_fit_reused_after_downstream_failure_with_full_evidence(tmp_path):
    model, _, request = coordinator(tmp_path)
    transport = CompleteTransport(model.settings)
    model.transport = transport
    base = model.execute(request)
    fit = fit_request(model.settings).model_copy(update={"input_checkpoint": base.model_id})
    fitted = model.execute(fit)
    assert fitted.reload is not None
    predict = m.PredictRequest(
        context=model.settings.context,
        round_number=1,
        purpose="validation",
        input_checkpoint=fitted.model_id,
        pages=(remote_page(id="validation", split=Split.VALIDATION, sha="b" * 64),),
    )
    transport.get_error = ConnectionError()
    with pytest.raises(RuntimeError, match="known call retained"):
        model.execute(predict)
    resumed = m.ModalModel(model.settings, model.store, transport, clock=lambda: 1001)
    assert resumed.execute(fit) == fitted
    assert [p["request"]["operation"] for p in transport.payloads].count("fit") == 1
    transport.get_error = None
    result = resumed.reconcile(predict)
    assert len(result.predictions) == 1
    assert "SELECTED SECRET" not in model.store.database_path.read_bytes().decode(
        "utf8", errors="ignore"
    )


@pytest.mark.parametrize("change", ["namespace", "nonfinite", "tensors", "logits", "pid"])
def test_reload_evidence_rejects_mismatches(tmp_path, change):
    model, _, request = coordinator(tmp_path)
    transport = CompleteTransport(model.settings)
    model.transport = transport
    base = model.execute(request)
    fit = fit_request(model.settings).model_copy(update={"input_checkpoint": base.model_id})
    invocation = globals()["invocation"](fit)
    wire = transport.base_response(invocation, "fc-fit")
    result = m.DispatchResponse.model_validate(wire).result
    paths = {}
    for i, ref in enumerate(result.artifacts):
        path = tmp_path / str(i)
        path.write_bytes(transport.files[ref.key])
        paths[ref.key] = path
    probe_path = paths[result.reload.after.key]
    probe = json.loads(probe_path.read_bytes())
    if change == "namespace":
        probe["logits"]["key"] = "qwen/" + probe["logits"]["key"]
    elif change == "pid":
        probe["process_id"] = 123
    elif change == "tensors":
        probe["adapter_tensors"][next(iter(probe["adapter_tensors"]))] = "f" * 64
    else:
        import struct

        key = "qwen/" + probe["logits"]["key"]
        data = bytearray(paths[key].read_bytes())
        data[:4] = struct.pack("<f", float("nan") if change == "nonfinite" else 1.0)
        paths[key].write_bytes(data)
    probe_path.write_bytes(q.canonical(probe))
    with pytest.raises(ValueError):
        m.verify_reload_evidence(result, paths, fit)


def test_worker_allowlist_imports_without_legacy_integrations(tmp_path):
    import shutil

    source = Path(q.__file__).parents[2]
    # Platform entrypoint is independently owned; its import is tested in the combined candidate.
    for name in m.RUNTIME_CODE_FILES:
        if name.endswith("modal_app.py"):
            continue
        dest = tmp_path / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, dest)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from active_ocr.integrations import qwen, modal_model; "
            "assert not any(n in sys.modules for n in ('torch','modal',"
            "'active_ocr.integrations.storage','active_ocr.integrations.local_data'))",
        ],
        env={"PYTHONPATH": str(tmp_path)},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_legacy_lazy_exports_preserve_public_objects():
    from active_ocr.integrations import GPUClient, LabelStudioClient, SQLiteStore, load_image_pages
    from active_ocr.integrations.gpu_client import GPUClient as ExpectedGPU
    from active_ocr.integrations.storage import SQLiteStore as ExpectedStore

    assert GPUClient is ExpectedGPU and SQLiteStore is ExpectedStore
    assert callable(load_image_pages) and LabelStudioClient.__name__ == "LabelStudioClient"


def test_resume_reattaches_persisted_running_call_without_new_spawn(tmp_path):
    model, transport, request = coordinator(tmp_path)
    control = model._control()
    record = m.ModelOperation(
        operation_id=m.operation_id(request),
        semantic=request.semantic_record(),
        attempt_id="a" * 32,
        state="RUNNING",
        provider_call_id="fc-existing",
    )
    model.store.compare_and_swap("model-operation", record.operation_id, None, record)
    transport.responses["fc-existing"] = transport.base_response(
        model._invocation(request, record, control), "fc-existing"
    )
    assert model.execute(request).provider_call_id == "fc-existing"
    assert transport.payloads == []


def test_expired_new_operation_never_gets_fresh_deadline(tmp_path):
    model, transport, request = coordinator(tmp_path)
    base = model.execute(request)
    model.clock = lambda: 3401
    predict = m.PredictRequest(
        context=model.settings.context,
        round_number=0,
        purpose="baseline_validation",
        input_checkpoint=base.model_id,
        pages=(),
    )
    with pytest.raises(TimeoutError, match="before submission"):
        model.execute(predict)
    assert len(transport.payloads) == 1
    control = model.store.load("model-control", model.settings.context.experiment_id, m.RunControl)
    assert control.first_submission_unix_seconds == 1000 and control.deadline_unix_seconds == 3400


def test_real_pipeline_recovers_after_round_commit_failure_without_retraining(
    tmp_path, monkeypatch
):
    from active_ocr.integrations import simulation
    from active_ocr.models import RunKind, SimulationConfig
    from active_ocr.pipeline import Pipeline

    settings = make_settings()
    monkeypatch.setattr(
        simulation,
        "local_contract_identity",
        lambda: (
            settings.context.real_config.expected_identity.source_sha,
            settings.context.real_config.expected_identity.dependency_sha256,
        ),
    )
    manifest = simulation.create_fixture(tmp_path / "data", train_pages=3)
    pipeline = Pipeline.for_simulation(tmp_path / "runs")
    config = SimulationConfig(
        real=settings.context.real_config,
        backend="qwen3-vl-v1",
        evaluator_id="page-text-nfc-v1",
        max_rounds=1,
    )
    run = pipeline.create_simulation(manifest, config, kind=RunKind.REAL)
    bundle = m.InputBundle(
        files=tuple(
            sorted(
                [f for f in settings.bundle.files if f.filename.startswith("model/")]
                + [
                    q.FileEntry(
                        filename="images/" + p.image_sha256,
                        sha256=p.image_sha256,
                        bytes=Path(p.image_uri).stat().st_size,
                    )
                    for p in run.dataset.pages
                ],
                key=lambda f: f.filename,
            )
        )
    )
    settings = m.RuntimeSettings.model_validate(
        {
            **settings.model_dump(),
            "bundle": bundle.model_dump(),
            "context": {
                **settings.context.model_dump(),
                "source_policy": config.source_policy,
                "experiment_id": run.id,
                "bundle_sha256": bundle.sha256,
            },
        }
    )
    transport = CompleteTransport(settings)
    model = m.ModalModel(settings, pipeline.store, transport, clock=lambda: 1000)
    baseline = pipeline.step_simulation(run.id, model)
    assert baseline.baseline.labelled_count == 0
    original = pipeline.store.compare_and_swap

    def fail_round(kind, key, expected, value):
        if kind == "simulation" and value.rounds:
            raise RuntimeError("simulated round commit failure")
        original(kind, key, expected, value)

    monkeypatch.setattr(pipeline.store, "compare_and_swap", fail_round)
    with pytest.raises(RuntimeError, match="round commit failure"):
        pipeline.step_simulation(run.id, model)
    assert pipeline.get_simulation(run.id).rounds == ()
    monkeypatch.setattr(pipeline.store, "compare_and_swap", original)
    count = len(transport.payloads)
    done = pipeline.step_simulation(run.id, model)
    assert done.complete and done.rounds[0].labelled_count == 2
    assert len(transport.payloads) == count == 4
    assert pipeline.step_simulation(run.id, model) == done  # Lost successful round response.
    assert len(transport.payloads) == count


def test_upload_plan_contains_only_exact_images_and_pinned_model(tmp_path, monkeypatch):
    import hashlib
    from types import SimpleNamespace

    model_root = tmp_path / "model"
    model_root.mkdir()
    image = tmp_path / "image.png"
    image.write_bytes(b"synthetic-image")
    image_sha = q.file_hash(image)
    expected = {}
    for name in q.ASSETS:
        data = name.encode()
        (model_root / name).write_bytes(data)
        expected[name] = (len(data), hashlib.sha256(data).hexdigest())
    monkeypatch.setattr(m, "ASSETS", expected)  # Synthetic pins, never staged 4B asset access.
    (model_root / "oracle.json").write_text("DO NOT UPLOAD")
    files = [
        q.FileEntry(filename="model/" + name, bytes=size, sha256=sha)
        for name, (size, sha) in expected.items()
    ]
    files.append(
        q.FileEntry(filename="images/" + image_sha, bytes=image.stat().st_size, sha256=image_sha)
    )
    bundle = m.InputBundle(files=tuple(sorted(files, key=lambda f: f.filename)))
    page = SimpleNamespace(
        image_uri=str(image), image_sha256=image_sha, source_image="secret.xml", regions="secret"
    )
    plan = m.input_upload_files(bundle, (page,), model_root)
    assert {key for _, key in plan} == {f.filename for f in files}
    assert all("oracle" not in str(path) for path, _ in plan)
    image.write_bytes(b"changed")
    with pytest.raises(ValueError, match="image bytes changed"):
        m.input_upload_files(bundle, (page,), model_root)


def test_previous_adapter_cannot_be_used_to_warm_start_fit(tmp_path):
    model, transport, _ = coordinator(tmp_path)
    with pytest.raises(ValueError, match="pinned base checkpoint"):
        model.execute(fit_request(model.settings))
    assert transport.payloads == []


def test_sdk_boundary_checks_observed_ids_and_never_creates_resources(monkeypatch):
    import importlib.metadata
    from types import SimpleNamespace

    settings = make_settings()
    calls = []

    class Function:
        object_id = "fu-observed"

        def hydrate(self):
            calls.append("hydrate-function")

        def spawn(self, payload):
            calls.append(payload)
            return SimpleNamespace(object_id="fc-observed")

    class Volume:
        def __init__(self, id):
            self.object_id = id

        def hydrate(self):
            calls.append("hydrate-volume")

        def read_file(self, key):
            calls.append(key)
            return iter([b"verified"])

    def volume(name, *, environment_name):
        assert environment_name == settings.environment_name
        return Volume(
            settings.input_volume_id
            if name == settings.input_volume_name
            else settings.output_volume_id
        )

    def function(name, method, *, environment_name):
        assert (name, method, environment_name) == (
            settings.deployment_name,
            "dispatch",
            settings.environment_name,
        )
        return Function()

    monkeypatch.setattr(importlib.metadata, "version", lambda _: "1.5.5")
    monkeypatch.setitem(
        sys.modules,
        "modal",
        SimpleNamespace(
            Function=SimpleNamespace(from_name=function), Volume=SimpleNamespace(from_name=volume)
        ),
    )
    observed = m.DeploymentObservation(
        settings_sha256=q.digest(settings.model_dump(mode="json")),
        function_id="fu-observed",
        app_id="ap-observed",
    )
    transport = m.SDKTransport(settings, observed)
    assert calls == []
    transport.preflight()
    assert transport.spawn({"strict": "payload"}) == "fc-observed"
    assert list(transport.read("qwen/receipts/x.json")) == [b"verified"]
    assert calls[-1] == "/runs/" + settings.context.experiment_id + "/qwen/receipts/x.json"
    Function.object_id = "fu-replaced"
    with pytest.raises(ValueError, match="Function identity changed"):
        transport.preflight()
