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
