"""Independent shared-API checks. Synthetic records are not provider/runtime evidence."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from active_ocr.integrations import modal_model as m
from active_ocr.integrations import qwen as q

CODE_FILES = (
    "active_ocr/__init__.py",
    "active_ocr/entrypoints/__init__.py",
    "active_ocr/entrypoints/modal_app.py",
    "active_ocr/integrations/__init__.py",
    "active_ocr/integrations/modal_model.py",
    "active_ocr/integrations/qwen.py",
    "active_ocr/models.py",
)
RUN, ATTEMPT, EXECUTION = "1" * 32, "2" * 32, "3" * 32
CHECKPOINT = "checkpoint:sha256:" + "4" * 64
SECRET = 'SELECTED_ONLY e\u0301\r\n"\\ <|im_end|>'


def encoded(value):
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def artifact(key, sha="a" * 64, size=19):
    return dict(key=key, bytes=size, sha256=sha)


@pytest.fixture
def settings():
    # Hash-only metadata; this fixture makes no claim that these files were built or uploaded.
    bundle = m.InputBundle(
        files=sorted(
            [
                dict(filename="model/" + name, bytes=size, sha256=sha)
                for name, (size, sha) in q.ASSETS.items()
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
        cpu_test_file=dict(filename="tests/test_qwen.py", bytes=10, sha256="f" * 64),
        processor_manifest_sha256=digest(q.asset_manifest(q.PROCESSOR_FILES)),
    )
    environment = dict(python="3.11.14", packages=sorted(q.PINS.items()))
    build = m.BuildReceipt(
        build_spec_sha256=spec.sha256,
        source_sha=spec.source_sha,
        code_files=spec.code_files,
        environment=environment,
        cpu_report=artifact("cpu-report.json"),
        cpu_passed=11,
        cpu_skipped=0,
    )
    identity = dict(
        source_sha=spec.source_sha,
        dependency_sha256="f" * 64,
        model_revision=q.REVISION,
        processor_revision=q.REVISION,
        model_manifest_sha256=digest(q.asset_manifest(q.BASE_FILES)),
        processor_manifest_sha256=spec.processor_manifest_sha256,
        recipe_version=q.RECIPE,
        evaluator_id="page-text-nfc-v1",
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
        backend="qwen3-vl-v1",
        recipe_version=q.RECIPE,
        model_repository=q.REPOSITORY,
        processor_repository=q.REPOSITORY,
        model_revision=q.REVISION,
        processor_revision=q.REVISION,
        training_policy_id=q.TRAIN_POLICY,
        decode_policy_id=q.DECODE_POLICY,
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
    raw = artifact("qwen/receipts/raw.json", "b" * 64)
    manifest = artifact("qwen/checkpoints/" + "4" * 64 + "/manifest.json", "4" * 64)
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
        "lock",
        "requirements",
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
    elif change == "lock":
        data["build_spec"]["lock_sha256"] = "0" * 64
    elif change == "requirements":
        data["build_spec"]["requirements_file"]["filename"] = "pyproject.toml"
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
    ["source.jsonl", "source.xml", "model/private.json", "tests/test_qwen.py", "images/other"],
)
def test_bundle_has_no_extra_file_channel(settings, path):
    data = settings.bundle.model_dump(mode="json")
    data["files"].append(dict(filename=path, bytes=12, sha256="9" * 64))
    data["files"].sort(key=lambda entry: entry["filename"])
    with pytest.raises(ValidationError):
        m.InputBundle.model_validate(data)


@pytest.mark.parametrize("duration", [0, -1, 2401])
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
        data["predictions"][0]["raw_output_artifact"] = "qwen/receipts/unlisted.json"
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


def test_probe_roots_full_logits_and_closure_limits(settings):
    sha = "4" * 64
    probe = m.ProbeMetadata(
        model_id=CHECKPOINT,
        page_id="first",
        image_sha256="a" * 64,
        original_width=100,
        original_height=200,
        process_id=71,
        generated_ids=[q.EOS],
        status="invalid_output",
        regions=[],
        finish_reason="eos",
        adapter_tensors={name: "a" * 64 for name in q.adapter_shapes()},
        logits=artifact(f"probes/{sha}/before.f32le", "b" * 64, 151936 * 4),
        logits_shape=[1, 151936],
    )
    # DEC013: nested logits relative to Qwen root; enclosing metadata references to run root.
    assert (
        str(Path(m.QWEN_OUTPUT_ROOT) / probe.logits.key)
        == f"/outputs/qwen/probes/{sha}/before.f32le"
    )
    before = artifact(f"qwen/probes/{sha}/before.json", digest(probe.model_dump(mode="json")))
    after = artifact(f"qwen/probes/{sha}/after.json", "c" * 64)
    reload = m.ReloadEvidence(
        before=before,
        after=after,
        train_process_id=71,
        reload_process_id=72,
        max_absolute_difference=0.001,
        exact_tensors_ids_status_regions=True,
    )
    for change in (
        {"logits_shape": [2, 151936]},
        {"adapter_tensors": {}},
        {"logits": artifact(probe.logits.key, "b" * 64, 151936 * 4 - 1)},
    ):
        with pytest.raises(ValidationError):
            m.ProbeMetadata.model_validate({**probe.model_dump(mode="json"), **change})
    with pytest.raises(ValidationError):
        m.ReloadEvidence.model_validate({**reload.model_dump(mode="json"), "reload_process_id": 71})
    # API-only boundary: parsing cannot verify nested file bytes or enforce the approved root.
    # Future consumers must require the exact root/key rather than relying on ArtifactRef alone.
    other = probe.model_dump(mode="json")
    other["logits"]["key"] = f"qwen/probes/{sha}/before.f32le"
    assert m.ProbeMetadata.model_validate(other).logits.key != probe.logits.key


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
    code += "assert not {'modal','torch','transformers','peft'} & sys.modules.keys()"
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        env={**os.environ, "PYTHONPATH": str(Path(q.__file__).parents[2])},
    )


def test_fit_completion_requires_reload_and_direct_reference_inventory(settings):
    call = invoke(fit(settings))
    template = completed(invoke(predict(settings))).model_dump(mode="json")
    template.update(operation_id=call.operation_id, predictions=[])
    with pytest.raises(ValueError, match="completed fit"):
        m.OperationResult.model_validate(template).check_request(call)
    before = artifact("qwen/probes/" + "4" * 64 + "/before.json", "5" * 64)
    after = artifact("qwen/probes/" + "4" * 64 + "/after.json", "6" * 64)
    template["reload"] = dict(
        before=before,
        after=after,
        train_process_id=11,
        reload_process_id=12,
        max_absolute_difference=0,
        exact_tensors_ids_status_regions=True,
    )
    with pytest.raises(ValidationError, match="reload evidence"):
        m.OperationResult.model_validate(template)
    template["artifacts"].extend([before, after])
    result = m.OperationResult.model_validate(template)
    result.check_request(call)
    # No bytes were read: this verifies direct references, not checkpoint kind/owner, metadata
    # contents, nested logits, tensor equality or a real second process. Those are phase B gates.
    assert {a.key for a in result.artifacts}.issuperset([before["key"], after["key"]])
    for change in (
        {"artifacts": result.artifacts + (result.artifacts[0],)},
        {"reload": None, "predictions": completed(invoke(predict(settings))).predictions},
    ):
        with pytest.raises((ValueError, ValidationError)):
            m.OperationResult.model_validate({**result.model_dump(), **change}).check_request(call)


@pytest.mark.parametrize(
    "key", ["../escape", "/outputs/absolute", "qwen/../other", "qwen//double", "qwen\\windows"]
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
