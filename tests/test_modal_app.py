"""Offline Platform boundary verification with explicit SDK/model/process fakes."""

from __future__ import annotations

import array
import hashlib
import math
import os
import struct
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from active_ocr.entrypoints import modal_app as a
from active_ocr.integrations import modal_model as m
from active_ocr.integrations import qwen as q
from active_ocr.models import (
    Box,
    ExpectedIdentity,
    RealOCRConfig,
    SourcePolicy,
    SourceRegion,
    Split,
)


def entry(root, name):
    path = root / name
    return q.FileEntry(filename=name, bytes=path.stat().st_size, sha256=q.file_hash(path))


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    inputs, outputs, code = (tmp_path / n for n in ("input", "output", "code"))
    for root in (inputs, outputs, code):
        root.mkdir()
    (inputs / "model").mkdir()
    (inputs / "images").mkdir()
    assets = {}
    for name in q.ASSETS:
        raw = name.encode()
        (inputs / "model" / name).write_bytes(raw)
        assets[name] = (len(raw), hashlib.sha256(raw).hexdigest())
    monkeypatch.setattr(q, "ASSETS", assets)
    monkeypatch.setattr(m, "ASSETS", assets)
    image = inputs / "images" / "staging.png"
    Image.new("RGB", (320, 480)).save(image)
    sha = q.file_hash(image)
    image.rename(inputs / "images" / sha)
    files = tuple(
        entry(inputs, name) for name in sorted(["model/" + n for n in assets] + ["images/" + sha])
    )
    bundle = m.InputBundle(files=files)
    for name in m.RUNTIME_CODE_FILES:
        path = code / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fake\n")
    for name in ("uv.lock", "requirements-linux.txt", "tests/test_qwen.py"):
        path = code / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fake\n")
    spec = m.BuildSpec(
        source_sha="a" * 40,
        code_files=tuple(entry(code, n) for n in sorted(m.RUNTIME_CODE_FILES)),
        lock_sha256=q.file_hash(code / "uv.lock"),
        requirements_file=entry(code, "requirements-linux.txt"),
        cpu_test_file=entry(code, "tests/test_qwen.py"),
        processor_manifest_sha256=q.digest(q.asset_manifest(q.PROCESSOR_FILES)),
    )
    packages = q.Packages(
        python="3.11.14", packages=tuple(sorted({**q.PINS, "modal": "1.5.5"}.items()))
    )
    report = {"selected": 11, "passed": 11, "skipped": 0, "failed": 0}
    receipt = m.BuildReceipt(
        build_spec_sha256=spec.sha256,
        source_sha=spec.source_sha,
        code_files=spec.code_files,
        environment=packages,
        cpu_report=a._publish(code, "cpu-report.json", q.canonical(report)),
        cpu_passed=11,
        cpu_skipped=0,
    )
    a._publish(code, "build-receipt.json", q.canonical(receipt.model_dump(mode="json")))
    monkeypatch.setattr(q, "installed_packages", lambda: packages)
    expected = ExpectedIdentity(
        source_sha=spec.source_sha,
        dependency_sha256="b" * 64,
        model_revision=q.REVISION,
        processor_revision=q.REVISION,
        model_manifest_sha256=q.digest(q.asset_manifest(q.BASE_FILES)),
        processor_manifest_sha256=spec.processor_manifest_sha256,
        recipe_version=q.RECIPE,
        evaluator_id="page-text-nfc-v1",
        code_bundle_sha256=receipt.code_bundle_sha256,
        remote_dependency_sha256=receipt.remote_dependency_sha256,
        build_spec_sha256=spec.sha256,
        deployment_reference="modal:main/synthetic/dispatch@im-synthetic",
    )
    config = RealOCRConfig(
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
        real_config=config,
    )
    settings = m.RuntimeSettings(
        context=context,
        environment_name="main",
        deployment_name="synthetic",
        image_id="im-synthetic",
        input_volume_name="input",
        output_volume_name="output",
        input_volume_id="vo-input",
        output_volume_id="vo-output",
        build_spec=spec,
        build=receipt,
        bundle=bundle,
    )
    page = m.RemotePage(
        id="selected",
        document_id=None,
        split=Split.TRAIN,
        width=320,
        height=480,
        image_sha256=sha,
        image_key="images/" + sha,
    )
    return SimpleNamespace(
        settings=settings,
        inputs=inputs,
        outputs=outputs,
        code=code,
        page=page,
        commits=[],
        payloads=[],
    )


def invoke(runtime, request, **overrides):
    now = int(time.time())
    return m.Invocation(
        request=request,
        attempt_id="2" * 32,
        first_submission_unix_seconds=now,
        deadline_unix_seconds=now + 100,
        **overrides,
    )


def base(runtime):
    binding = q.QwenModel._binding(
        SimpleNamespace(real_config=runtime.settings.context.real_config)
    )
    return q.publish_checkpoint(
        runtime.outputs / "qwen", q.Checkpoint(kind="base", binding=binding)
    )


def fake_child(runtime):
    def run(payload, deadline):
        runtime.payloads.append(payload)
        model_id = base(runtime)
        return {
            "status": "ok",
            "model_id": model_id,
            "process_id": 101,
            "predictions": [],
            "telemetry": None,
        }

    return run


def dispatch(runtime, call, runner=None, commit=None):
    return a.dispatch_operation(
        runtime.settings,
        call.model_dump(mode="json"),
        provider_call_id="fc-observed",
        input_root=runtime.inputs,
        output_root=runtime.outputs,
        code_root=runtime.code,
        child_runner=runner or fake_child(runtime),
        commit=commit or (lambda: runtime.commits.append(True)),
    )


def test_import_has_no_modal_or_ml_side_effects():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import active_ocr.entrypoints.modal_app; "
            "assert not {'modal','torch','transformers','peft'} & sys.modules.keys()",
        ],
        env={**os.environ, "PYTHONPATH": str(Path(q.__file__).parents[2])},
        check=True,
    )


def test_base_completion_and_same_attempt_reuse(runtime):
    call = invoke(runtime, m.BaseRequest(context=runtime.settings.context))
    first = dispatch(runtime, call)
    assert len(runtime.payloads) == 1
    assert dispatch(runtime, call) == first
    assert len(runtime.payloads) == 1
    response = m.DispatchResponse.model_validate(first)
    assert a._read(runtime.outputs, response.completion) == q.canonical(
        response.result.model_dump(mode="json")
    )
    assert len(runtime.commits) >= 3
    assert response.result.provider_call_id == "fc-observed"


def test_changed_attempt_or_deadline_never_reuses_or_retrains(runtime):
    call = invoke(runtime, m.BaseRequest(context=runtime.settings.context))
    dispatch(runtime, call)
    for changed in (
        call.model_copy(update={"attempt_id": "3" * 32}),
        call.model_copy(update={"deadline_unix_seconds": call.deadline_unix_seconds + 1}),
    ):
        with pytest.raises(a.WorkerError):
            dispatch(runtime, changed)
    assert len(runtime.payloads) == 1


@pytest.mark.parametrize("kind", ["byte", "extra", "symlink", "package", "code"])
def test_identity_failure_before_any_child(runtime, monkeypatch, kind):
    if kind == "byte":
        (runtime.inputs / runtime.page.image_key).write_bytes(b"changed")
    elif kind == "extra":
        (runtime.inputs / "oracle.json").write_text("PRIVATE LABEL")
    elif kind == "symlink":
        path = runtime.inputs / runtime.page.image_key
        path.unlink()
        path.symlink_to(runtime.code / "uv.lock")
    elif kind == "package":
        monkeypatch.setattr(
            q, "installed_packages", lambda: q.Packages(python="3.11.14", packages=())
        )
    else:
        (runtime.code / "active_ocr/models.py").write_text("# changed")
    with pytest.raises(a.WorkerError):
        dispatch(runtime, invoke(runtime, m.BaseRequest(context=runtime.settings.context)))
    assert not runtime.payloads


def test_expired_or_missing_call_id_is_not_execution(runtime):
    call = invoke(runtime, m.BaseRequest(context=runtime.settings.context))
    with pytest.raises(a.WorkerError, match="deadline"):
        dispatch(
            runtime,
            call.model_copy(
                update={"first_submission_unix_seconds": 1, "deadline_unix_seconds": 2}
            ),
        )
    with pytest.raises(a.WorkerError, match="identity"):
        a.dispatch_operation(
            runtime.settings,
            call.model_dump(mode="json"),
            provider_call_id=None,
            commit=lambda: None,
        )
    assert not runtime.payloads


def test_failure_is_redacted_and_partial_attempt_cannot_repeat(runtime):
    def failing(payload, deadline):
        raise ValueError("PRIVATE SELECTED LABEL")

    call = invoke(runtime, m.BaseRequest(context=runtime.settings.context))
    with pytest.raises(a.WorkerError, match="^model$"):
        dispatch(runtime, call, failing)
    files = list(runtime.outputs.rglob("failure.json"))
    assert len(files) == 1 and b"PRIVATE" not in files[0].read_bytes()
    assert not list(runtime.outputs.rglob("complete.json"))
    with pytest.raises(a.WorkerError, match="identity|interrupted"):
        dispatch(runtime, call)
    assert not runtime.payloads


def test_completed_artifact_corruption_is_not_reused(runtime):
    call = invoke(runtime, m.BaseRequest(context=runtime.settings.context))
    result = dispatch(runtime, call)
    path = runtime.outputs / result["result"]["worker_manifest"]["key"]
    path.write_text("{}")
    with pytest.raises(a.WorkerError):
        dispatch(runtime, call)
    assert len(runtime.payloads) == 1


def test_atomic_publication_never_replaces(runtime):
    a._publish(runtime.outputs, "evidence.json", b"original")
    with pytest.raises(a.WorkerError):
        a._publish(runtime.outputs, "evidence.json", b"replacement")
    assert (runtime.outputs / "evidence.json").read_bytes() == b"original"
    assert not list(runtime.outputs.glob(".publish-*"))


def training_checkpoint(runtime, request):
    directory = runtime.outputs / "staging"
    directory.mkdir()
    (directory / "adapter_config.json").write_text("{}")
    header, chunks, hashes, offset = {}, [], {}, 0
    for name, shape in q.adapter_shapes().items():
        raw = bytes(math.prod(shape) * 4)
        header[name] = dict(
            dtype="F32", shape=list(shape), data_offsets=[offset, offset + len(raw)]
        )
        hashes[name] = hashlib.sha256(
            q.canonical(dict(shape=list(shape), dtype="torch.float32")) + raw
        ).hexdigest()
        chunks.append(raw)
        offset += len(raw)
    encoded = q.canonical(header)
    (directory / "adapter_model.safetensors").write_bytes(
        struct.pack("<Q", len(encoded)) + encoded + b"".join(chunks)
    )
    runtime.tensor_hashes = hashes
    ids = tuple(e.page.id for e in request.examples)
    training = dict(
        updates=3,
        epoch_orders=[
            [p for g in q.epoch_groups(ids, request.seed, n) for p in g] for n in range(3)
        ],
        losses=[1.0] * (3 * len(ids)),
        gradient_norms=[1.0] * 3,
        processing={
            p: dict(
                height=480,
                width=320,
                grid=[1, 30, 20],
                prompt_tokens=5,
                target_tokens=6,
                target_exceeds_decode_budget=False,
            )
            for p in ids
        },
        supervised_tokens=18 * len(ids),
        changed_tensors=1,
        frozen_sha256="a" * 64,
    )
    examples = tuple(
        a.RevealedExample(page=a._page(e.page, request.context, runtime.inputs), regions=e.regions)
        for e in request.examples
    )
    checkpoint = q.Checkpoint(
        kind="adapter",
        binding=q.QwenModel._binding(SimpleNamespace(real_config=request.context.real_config)),
        experiment_id=request.context.experiment_id,
        round_number=request.round_number,
        seed=request.seed,
        selected=tuple((e.page.id, e.page.image_sha256) for e in request.examples),
        target_sha256=q.digest([(e.page.id, q.serialize_target(e)) for e in examples]),
        training=training,
        files=tuple(
            entry(directory, n) for n in ("adapter_config.json", "adapter_model.safetensors")
        ),
    )
    return q.publish_checkpoint(runtime.outputs / "qwen", checkpoint, directory)


def probe(runtime, model_id, stage, pid, value=0.0):
    sha = model_id.rsplit(":", 1)[1]
    values = array.array("f", [value]) * 151936
    if sys.byteorder != "little":
        values.byteswap()
    logits = a._publish(runtime.outputs / "qwen", f"probes/{sha}/{stage}.f32le", values.tobytes())
    p = runtime.page
    meta = m.ProbeMetadata(
        model_id=model_id,
        page_id=p.id,
        image_sha256=p.image_sha256,
        original_width=p.width,
        original_height=p.height,
        process_id=pid,
        generated_ids=(151645,),
        status="ok",
        regions=(),
        finish_reason="eos",
        adapter_tensors=runtime.tensor_hashes,
        logits=logits,
        logits_shape=(1, 151936),
    )
    a._publish(
        runtime.outputs / "qwen",
        f"probes/{sha}/{stage}.json",
        q.canonical(meta.model_dump(mode="json")),
    )


@pytest.mark.parametrize(
    "after_value,second_pid,success",
    [(0.0, 202, True), (0.1, 202, False), (float("nan"), 202, False), (0.0, 101, False)],
)
def test_fit_uses_two_sequential_stages_and_checks_full_logits(
    runtime, after_value, second_pid, success
):
    request = m.FitRequest(
        context=runtime.settings.context,
        round_number=1,
        input_checkpoint=base(runtime),
        seed=824,
        examples=(
            m.RemoteExample(
                page=runtime.page,
                regions=(
                    SourceRegion(
                        id="line", text="SELECTED SECRET", box=Box(x=0, y=0, width=20, height=20)
                    ),
                ),
            ),
        ),
    )

    def runner(payload, deadline):
        runtime.payloads.append(payload)
        if payload["stage"] == "operation":
            model_id = training_checkpoint(runtime, request)
            probe(runtime, model_id, "before", 101)
            return dict(model_id=model_id, process_id=101, predictions=[], telemetry=None)
        assert "SELECTED SECRET" not in q.canonical(payload).decode()
        assert "request" not in payload and "examples" not in payload
        probe(runtime, payload["model_id"], "after", second_pid, after_value)
        return dict(
            model_id=payload["model_id"], process_id=second_pid, predictions=[], telemetry=None
        )

    if success:
        result = dispatch(runtime, invoke(runtime, request), runner)
        assert result["result"]["reload"]["reload_process_id"] == 202
        assert result["result"]["reload"]["before"]["key"].startswith("qwen/probes/")
    else:
        with pytest.raises(a.WorkerError):
            dispatch(runtime, invoke(runtime, request), runner)
        assert not list(runtime.outputs.rglob("complete.json"))
    assert [p["stage"] for p in runtime.payloads] == ["operation", "reload"]
    for path in runtime.outputs.rglob("*.json"):
        assert b"SELECTED SECRET" not in path.read_bytes()


def test_watchdog_kills_and_reaps_process_group(monkeypatch):
    events = []

    class Child:
        pid = 123

        def communicate(self, *args, **kwargs):
            raise subprocess.TimeoutExpired("child", 1)

        def poll(self):
            return None

        def wait(self, timeout):
            events.append(("wait", timeout))
            if timeout == 3:
                raise subprocess.TimeoutExpired("child", timeout)

    monkeypatch.setattr(a.subprocess, "Popen", lambda *args, **kwargs: Child())
    monkeypatch.setattr(a.os, "killpg", lambda *args: events.append(args))
    with pytest.raises(a.WorkerError):
        a.run_child({"labels": "ephemeral"}, int(time.time()) + 10)
    assert events == [(123, a.signal.SIGTERM), ("wait", 3), (123, a.signal.SIGKILL), ("wait", 5)]


@pytest.mark.parametrize(
    "count,child,exitcode",
    [(10, "", 0), (11, "<skipped/>", 0), (11, "<failure/>", 1), (11, "", 1), (12, "", 0)],
)
def test_cpu_gate_rejects_skips_failures_wrong_count(tmp_path, count, child, exitcode):
    report = tmp_path / "report.xml"
    report.write_text(
        "<testsuites><testsuite>"
        + "".join(f'<testcase classname="test" name="{i}">{child}</testcase>' for i in range(count))
        + "</testsuite></testsuites>"
    )
    with pytest.raises(a.WorkerError):
        a._cpu_report(report, exitcode)


class FakeSDK:
    __name__ = "fake_modal"

    def __init__(self):
        self.calls = []
        self.Image = self
        self.Volume = self
        self.object_id = "im-observed"

    def from_id(self, id):
        self.calls.append(("from_id", id))
        return self

    def with_mount_options(self, **kwargs):
        self.calls.append(("mount", kwargs))
        return kwargs

    def App(self, *args, **kwargs):
        self.calls.append(("app", args, kwargs))
        return self

    def function(self, **kwargs):
        self.calls.append(("function", kwargs))
        return lambda f: f

    def debian_slim(self, **kwargs):
        self.calls.append(("base", kwargs))
        return self

    def pip_install_from_requirements(self, *args, **kwargs):
        self.calls.append(("pip", args, kwargs))
        return self

    def add_local_file(self, *args, **kwargs):
        self.calls.append(("file", args, kwargs))
        return self

    def env(self, env):
        self.calls.append(("env", env))
        return self

    def run_function(self, *args, **kwargs):
        self.calls.append(("buildstep", args, kwargs))
        return self

    def commit(self):
        raise AssertionError("construction must not commit")


def test_sdk_runtime_construction_has_exact_limits_and_two_mounts(runtime):
    sdk = FakeSDK()
    a._publish(runtime.code, "settings.json", q.canonical(runtime.settings.model_dump(mode="json")))
    a.create_app(runtime.settings, settings_file=runtime.code / "settings.json", sdk=sdk)
    kwargs = next(c[1] for c in sdk.calls if c[0] == "function")
    assert (
        kwargs["gpu"] == "L40S" and kwargs["cpu"] == (2, 2) and kwargs["memory"] == (32768, 32768)
    )
    for k, v in dict(
        max_containers=1,
        min_containers=0,
        buffer_containers=0,
        retries=0,
        timeout=600,
        startup_timeout=300,
        scaledown_window=2,
        include_source=False,
    ).items():
        assert kwargs[k] == v
    assert kwargs["volumes"]["/inputs"] == dict(
        sub_path=runtime.settings.input_sub_path, read_only=True
    )
    assert kwargs["volumes"]["/outputs"] == dict(
        sub_path=runtime.settings.output_sub_path, read_only=False
    )
    assert ("from_id", "vo-input") in sdk.calls and ("from_id", "vo-output") in sdk.calls


def test_image_construction_only_copies_allowlists_and_cpu_step(runtime, tmp_path, monkeypatch):
    monkeypatch.setattr(a, "_verify_export", lambda *args: None)  # Explicit fake checkout.
    checkout = tmp_path / "checkout"
    (checkout / "src").mkdir(parents=True)
    import shutil

    shutil.copytree(runtime.code / "active_ocr", checkout / "src/active_ocr")
    shutil.copytree(runtime.code / "tests", checkout / "tests")
    shutil.copy(runtime.code / "uv.lock", checkout / "uv.lock")
    sdk = FakeSDK()
    a.prepare_image(
        runtime.settings.build_spec,
        checkout=checkout,
        requirements=runtime.code / "requirements-linux.txt",
        processor_root=runtime.inputs / "model",
        output_volume=sdk,
        sdk=sdk,
    )
    copies = [c for c in sdk.calls if c[0] == "file"]
    assert len(copies) == len(m.RUNTIME_CODE_FILES) + 3 + len(q.PROCESSOR_FILES)
    assert all(c[2] == {"copy": True} for c in copies)
    assert not any("safetensors" in c[1][0] for c in copies)
    step = next(c[2] for c in sdk.calls if c[0] == "buildstep")
    assert (
        step["gpu"] is None
        and step["cpu"] == 2
        and step["memory"] == 32768
        and step["timeout"] == 900
    )
    assert step["include_source"] is False
    assert set(step["volumes"]) == {"/build-evidence"}
    assert step["kwargs"]["build_spec_sha256"] == runtime.settings.build_spec.sha256


def test_commit_failure_cannot_turn_into_successful_reuse(runtime):
    call = invoke(runtime, m.BaseRequest(context=runtime.settings.context))
    count = 0

    def commit():
        nonlocal count
        count += 1
        if count >= 3:
            raise RuntimeError("provider unavailable")

    with pytest.raises(a.WorkerError):
        dispatch(runtime, call, commit=commit)
    with pytest.raises(a.WorkerError):
        dispatch(runtime, call)
    assert len(runtime.payloads) == 1


def test_cpu_report_accepts_exact_success_without_transcript(tmp_path):
    path = tmp_path / "report.xml"
    path.write_text(
        "<testsuite>"
        + "".join(f'<testcase classname="test_qwen" name="test_actual_{i}"/>' for i in range(11))
        + "</testsuite>"
    )
    report = a._cpu_report(path, 0)
    assert report["passed"] == 11 and report["skipped"] == 0
    assert len(report["tests"]) == 11


@pytest.mark.parametrize("mount_symlink", [False, True])
def test_cpu_build_writes_measured_receipt_and_commits_only_output(
    runtime, monkeypatch, mount_symlink
):
    import shutil

    evidence = runtime.outputs / "build-evidence"
    evidence.mkdir()
    shutil.copytree(runtime.inputs / "model", runtime.code / "processor")
    (runtime.code / "build-receipt.json").unlink()  # Fake build starts before its receipt exists.
    monkeypatch.setattr(m, "CODE_ROOT", str(runtime.code))
    mount = runtime.outputs / "mounted-evidence"
    if mount_symlink:
        mount.symlink_to(evidence, target_is_directory=True)
    monkeypatch.setattr(a, "BUILD_EVIDENCE_ROOT", str(mount if mount_symlink else evidence))

    class Child:
        returncode = 0

        def wait(self, timeout):
            assert timeout == 850

    def popen(args, **kwargs):
        assert kwargs["start_new_session"] is True
        assert kwargs["stdout"] == subprocess.DEVNULL
        assert kwargs["stderr"] == subprocess.DEVNULL
        path = Path(next(v.split("=", 1)[1] for v in args if v.startswith("--junitxml=")))
        path.write_text(
            "<testsuite>"
            + "".join(
                f'<testcase classname="test_qwen" name="test_actual_{i}"/>' for i in range(11)
            )
            + "</testsuite>"
        )
        return Child()

    monkeypatch.setattr(a.subprocess, "Popen", popen)
    observed = []
    volume = SimpleNamespace(commit=lambda: observed.append("commit"))
    monkeypatch.setattr(
        a,
        "_sdk",
        lambda: SimpleNamespace(
            Volume=SimpleNamespace(
                from_id=lambda identifier: volume if identifier == "vo-output" else None
            )
        ),
    )
    monkeypatch.setattr(a.platform, "system", lambda: "Linux")
    monkeypatch.setattr(a.platform, "machine", lambda: "x86_64")
    spec = runtime.settings.build_spec
    a.cpu_build_gate(spec.model_dump(mode="json"), spec.sha256, "vo-output")
    assert observed == ["commit"]
    receipt = m.BuildReceipt.model_validate_json((evidence / "build-receipt.json").read_bytes())
    assert receipt.environment == q.installed_packages()
    assert (runtime.code / "build-receipt.json").read_bytes() == (
        evidence / "build-receipt.json"
    ).read_bytes()
    assert a._read(evidence, receipt.cpu_report)


@pytest.mark.parametrize("damage", [None, "missing", "report", "receipt"])
def test_eager_build_requires_observed_image_and_durable_cpu_evidence(runtime, monkeypatch, damage):
    receipt = runtime.settings.build
    raw = q.canonical(receipt.model_dump(mode="json"))
    report = (runtime.code / receipt.cpu_report.key).read_bytes()
    events = []

    class Built:
        @property
        def object_id(self):
            assert events == ["build"]
            return "im-observed-after-build"

        def build(self, app):
            events.append("build")

    def read_file(key):
        if damage == "missing":
            raise FileNotFoundError()
        if key.endswith("build-receipt.json"):
            return iter([b"{}" if damage == "receipt" else raw])
        return iter([b"{}" if damage == "report" else report])

    monkeypatch.setattr(a, "prepare_image", lambda *args, **kwargs: Built())
    kwargs = dict(
        app=object(),
        checkout=runtime.code,
        requirements=runtime.code / "requirements-linux.txt",
        processor_root=runtime.inputs / "model",
        output_volume=SimpleNamespace(read_file=read_file),
    )
    if damage:
        with pytest.raises((a.WorkerError, ValueError, FileNotFoundError)):
            a.build_image(runtime.settings.build_spec, **kwargs)
    else:
        image_id, got, cpu = a.build_image(runtime.settings.build_spec, **kwargs)
        assert image_id == "im-observed-after-build" and got == receipt and cpu["passed"] == 11
    assert events == ["build"]  # No blind rebuild when cache evidence is missing.


def test_real_child_error_does_not_echo_request(capsys):
    with pytest.raises(a.WorkerError, match="model"):
        a.run_child({"unexpected": "NEVER PRINT THIS SECRET"}, int(time.time()) + 10)
    captured = capsys.readouterr()
    assert "SECRET" not in captured.out + captured.err


def test_child_fit_and_reload_keep_qwen_api_and_labels_separate(runtime, monkeypatch):
    calls = []

    class Model:
        telemetry = None

        def __init__(self, *args, **kwargs):
            calls.append(("construct", kwargs))

        def fit(self, examples, **kwargs):
            calls.append(("fit", kwargs, examples))
            return "checkpoint:sha256:" + "a" * 64

        def verify_reload(self, page, **kwargs):
            calls.append(("reload", kwargs, page))

    monkeypatch.setattr(q, "QwenModel", Model)
    common = dict(
        context=runtime.settings.context.model_dump(mode="json"),
        input_root=str(runtime.inputs),
        output_root=str(runtime.outputs),
        runtime_manifest=str(runtime.code / "worker.json"),
        deadline=int(time.time()) + 100,
    )
    req = m.FitRequest(
        context=runtime.settings.context,
        round_number=1,
        input_checkpoint="checkpoint:sha256:" + "b" * 64,
        seed=824,
        examples=(m.RemoteExample(page=runtime.page, regions=()),),
    )
    first = a._child({**common, "stage": "operation", "request": req.model_dump(mode="json")})
    assert calls[-1][0] == "fit" and calls[-1][1]["external_reload"] is True
    before = dict(key="qwen/probes/" + "a" * 64 + "/before.json", bytes=1, sha256="c" * 64)
    logits = dict(key="qwen/probes/" + "a" * 64 + "/before.f32le", bytes=1, sha256="d" * 64)
    a._child(
        {
            **common,
            "stage": "reload",
            "page": runtime.page.model_dump(mode="json"),
            "round_number": 1,
            "model_id": first["model_id"],
            "before": before,
            "before_logits": logits,
        }
    )
    assert calls[-1][0] == "reload" and "examples" not in calls[-1][1]
    assert calls[-1][1]["before_metadata"] == runtime.outputs / before["key"]


@pytest.mark.parametrize("damage", [None, "geometry", "raw_owner", "raw_text", "failed_regions"])
def test_prediction_completion_checks_geometry_and_raw_closure(runtime, damage):
    from active_ocr.models import Prediction, Region

    page = runtime.page.model_copy(update={"split": Split.VALIDATION})
    model_id = base(runtime)
    req = m.PredictRequest(
        context=runtime.settings.context,
        input_checkpoint=model_id,
        round_number=0,
        purpose="baseline_validation",
        pages=(page,),
    )

    def runner(payload, deadline):
        runtime.payloads.append(payload)
        text = '{"regions":[]}'
        regions = ()
        status = "ok"
        if damage in {"geometry", "failed_regions"}:
            regions = (Region(id="one", text="derived", box=Box(x=0, y=0, width=321, height=10)),)
        if damage == "failed_regions":
            status = "invalid_output"
        raw = dict(
            operation="predict",
            experiment_id=req.context.experiment_id,
            round_number=0,
            purpose=req.purpose,
            model_id=model_id,
            pages=[
                dict(
                    page_id=page.id,
                    image_sha256=page.image_sha256,
                    ids=[151645],
                    text=text,
                    finish_reason="eos",
                    status=status,
                    processing={},
                )
            ],
            telemetry={},
        )
        if damage == "raw_owner":
            raw["experiment_id"] = "9" * 32
        if damage == "raw_text":
            raw["pages"][0]["text"] = '{"regions":[{"text":"extra","bbox":[0,0,10,10]}]}'
        raw_key = "qwen/receipts/" + q.digest(raw) + ".json"
        a._publish(runtime.outputs, raw_key, q.canonical(raw))
        prediction = Prediction(
            page_id=page.id,
            experiment_id=req.context.experiment_id,
            round_number=0,
            model_id=model_id,
            purpose=req.purpose,
            regions=regions,
            status=status,
            finish_reason="eos",
            raw_output_artifact=str(runtime.outputs / raw_key),
        )
        return dict(
            model_id=model_id,
            process_id=123,
            predictions=[prediction.model_dump(mode="json")],
            telemetry=None,
        )

    call = invoke(runtime, req)
    if damage:
        with pytest.raises(a.WorkerError):
            dispatch(runtime, call, runner)
        assert not list(runtime.outputs.rglob("complete.json"))
    else:
        got = dispatch(runtime, call, runner)
        assert got["result"]["predictions"][0]["raw_output_artifact"].startswith("qwen/receipts/")


def test_cpu_gate_timeout_terminates_tree_without_receipt(runtime, monkeypatch):
    import shutil

    evidence = runtime.outputs / "build-evidence"
    evidence.mkdir()
    shutil.copytree(runtime.inputs / "model", runtime.code / "processor")
    (runtime.code / "build-receipt.json").unlink()
    monkeypatch.setattr(m, "CODE_ROOT", str(runtime.code))
    monkeypatch.setattr(a, "BUILD_EVIDENCE_ROOT", str(evidence))

    class Child:
        pid = 345
        returncode = -15

        def wait(self, timeout):
            if timeout == 850:
                raise subprocess.TimeoutExpired("pytest", 850)

    monkeypatch.setattr(a.subprocess, "Popen", lambda *args, **kwargs: Child())
    monkeypatch.setattr(
        a,
        "_sdk",
        lambda: SimpleNamespace(
            Volume=SimpleNamespace(from_id=lambda _: SimpleNamespace(commit=lambda: None))
        ),
    )
    killed = []
    monkeypatch.setattr(a.os, "killpg", lambda *args: killed.append(args))
    spec = runtime.settings.build_spec
    with pytest.raises(a.WorkerError):
        a.cpu_build_gate(spec.model_dump(mode="json"), spec.sha256, "vo-output")
    assert killed == [(345, a.signal.SIGTERM)]
    assert len(list((evidence / "failures").glob("*.json"))) == 1
    assert not (evidence / "build-receipt.json").exists()


@pytest.mark.parametrize("damage", [None, "head", "dirty", "export", "command"])
def test_build_export_binds_clean_commit_lock_and_requirements(runtime, monkeypatch, damage):
    seen = []

    def run(args, **kwargs):
        seen.append(args)
        if args[1:] == ["rev-parse", "HEAD"]:
            out = (
                "f" * 40 if damage == "head" else runtime.settings.build_spec.source_sha
            ).encode()
        elif args[1] == "status":
            out = b"M tracked.py" if damage == "dirty" else b""
        else:
            out = b"torch==WRONG" if damage == "export" else b"# fake\n"
        return SimpleNamespace(returncode=1 if damage == "command" else 0, stdout=out)

    monkeypatch.setattr(a.subprocess, "run", run)
    args = (runtime.code, runtime.settings.build_spec, runtime.code / "requirements-linux.txt")
    if damage:
        with pytest.raises(a.WorkerError):
            a._verify_export(*args)
    else:
        a._verify_export(*args)
        assert "--offline" in seen[-1] and "--frozen" in seen[-1]
        assert "--no-emit-project" in seen[-1] and "--no-hashes" not in seen[-1]


def test_cpu_failure_retains_bounded_diagnostic_not_success(runtime, monkeypatch, capsys):
    evidence = runtime.outputs / "build"
    evidence.mkdir()
    junit = runtime.outputs / "junit.xml"
    junit.write_text(
        '<testsuite><testcase classname="test_qwen" name="test_actual_tiny">'
        '<failure message="shape mismatch; password=supersecretvalue"/></testcase>'
        '<testcase classname="test_qwen" name="test_actual_mask">'
        '<skipped message="dependency absent"/>'
        "</testcase></testsuite>"
    )
    monkeypatch.setenv("EXAMPLE_PASSWORD", "supersecretvalue")
    committed = []
    monkeypatch.setattr(
        a,
        "_sdk",
        lambda: SimpleNamespace(
            Volume=SimpleNamespace(
                from_id=lambda _: SimpleNamespace(commit=lambda: committed.append(True))
            )
        ),
    )
    a._cpu_failure(junit, 1, "cpu_checks", runtime.settings.build_spec, evidence, "vo-fake")
    raw = next((evidence / "failures").glob("*.json")).read_bytes()
    record = q.strict_json(raw)
    assert record["returncode"] == 1 and record["reported_cases"] == 2
    assert [r["status"] for r in record["tests"]] == ["failure", "skipped"]
    assert "shape mismatch" in record["tests"][0]["excerpt"]
    assert b"supersecretvalue" not in raw and "supersecretvalue" not in capsys.readouterr().out
    assert committed == [True] and not (evidence / "build-receipt.json").exists()


@pytest.mark.parametrize("damage", ["tensor_byte", "tensor_header", "trailing", "nan"])
def test_probe_hashes_close_over_actual_adapter_bytes(runtime, damage):
    req = m.FitRequest(
        context=runtime.settings.context,
        round_number=1,
        input_checkpoint=base(runtime),
        seed=824,
        examples=(m.RemoteExample(page=runtime.page, regions=()),),
    )
    model_id = training_checkpoint(runtime, req)
    probe(runtime, model_id, "before", 101)
    probe(runtime, model_id, "after", 202)
    sha = model_id.rsplit(":", 1)[1]
    path = runtime.outputs / f"qwen/checkpoints/{sha}/adapter_model.safetensors"
    data = bytearray(path.read_bytes())
    offset = 8 + struct.unpack("<Q", data[:8])[0]
    if damage == "tensor_byte":
        data[offset : offset + 4] = struct.pack("<f", 1.0)
    elif damage == "nan":
        data[offset : offset + 4] = struct.pack("<f", float("nan"))
    elif damage == "tensor_header":
        data[8] = 0
    else:
        data.extend(b"trailing")
    path.write_bytes(data)
    with pytest.raises((a.WorkerError, ValueError)):
        a._reload(runtime.outputs, model_id, runtime.page)


def test_publication_without_hardlinks_stays_atomic_and_exclusive(tmp_path, monkeypatch):
    def unavailable(*args, **kwargs):
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(a.os, "link", unavailable)
    target = tmp_path / "evidence.json"
    rename = a.os.rename
    observed = []

    def check_rename(source, destination):
        assert not target.exists()
        assert Path(source).read_bytes() == b"original"
        # A competing publisher cannot steal the reserved destination.
        with pytest.raises(a.WorkerError):
            a._publish(tmp_path, "evidence.json", b"competitor")
        observed.append(True)
        rename(source, destination)

    monkeypatch.setattr(a.os, "rename", check_rename)
    ref = a._publish(tmp_path, "evidence.json", b"original")
    assert a._read(tmp_path, ref) == b"original" and observed == [True]
    assert a._publish(tmp_path, "evidence.json", b"original") == ref
    with pytest.raises(a.WorkerError):
        a._publish(tmp_path, "evidence.json", b"replacement")
    assert target.read_bytes() == b"original"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["evidence.json"]


@pytest.mark.parametrize("stage", ["publish", "commit"])
def test_cpu_failure_logs_persistence_stage_without_exception_secrets(
    runtime, monkeypatch, capsys, stage
):
    def fail(*args):
        raise PermissionError(1, "secret=private-value")

    if stage == "publish":
        monkeypatch.setattr(a, "_publish", fail)
    else:
        monkeypatch.setattr(a, "_sdk", lambda: SimpleNamespace(
            Volume=SimpleNamespace(from_id=lambda _: SimpleNamespace(commit=fail)),
        ))
    a._cpu_failure(runtime.code / "absent.xml", 1, "cpu_checks", runtime.settings.build_spec,
                   runtime.outputs, "vo-fake")
    records = [q.strict_json(line.encode()) for line in capsys.readouterr().out.splitlines()]
    assert records[-1] == {"code": "cpu_failure_persistence", "stage": stage,
                           "type": "PermissionError", "errno": 1}
    assert "private-value" not in str(records)


def test_failed_fsync_never_exposes_partial_publication(tmp_path, monkeypatch):
    def fail(fd):
        raise OSError(5, "synthetic I/O error")

    monkeypatch.setattr(a.os, "fsync", fail)
    with pytest.raises(OSError):
        a._publish(tmp_path, "complete.json", b'{"complete":true}')
    assert list(tmp_path.iterdir()) == []


def test_dispatch_resolves_provider_mounts_but_rejects_descendant_symlinks(runtime, monkeypatch):
    input_mount = runtime.code.parent / "input-mount"
    output_mount = runtime.code.parent / "output-mount"
    input_mount.symlink_to(runtime.inputs, target_is_directory=True)
    output_mount.symlink_to(runtime.outputs, target_is_directory=True)
    monkeypatch.setattr(m, "INPUT_ROOT", str(input_mount))
    monkeypatch.setattr(m, "OUTPUT_ROOT", str(output_mount))
    monkeypatch.setattr(m, "CODE_ROOT", str(runtime.code))
    monkeypatch.setattr(a, "__file__", str(runtime.code / "active_ocr/entrypoints/modal_app.py"))
    monkeypatch.setattr(m, "__file__", str(runtime.code / "active_ocr/integrations/modal_model.py"))
    (runtime.code / "runtime-settings.json").write_bytes(
        q.canonical(runtime.settings.model_dump(mode="json"))
    )
    monkeypatch.setattr(a, "_sdk", lambda: SimpleNamespace(
        current_function_call_id=lambda: "fc-synthetic",
        Volume=SimpleNamespace(from_id=lambda _: SimpleNamespace(commit=lambda: None)),
    ))
    def observe(settings, payload, **kwargs):
        assert kwargs["input_root"] == runtime.inputs.resolve()
        assert kwargs["output_root"] == runtime.outputs.resolve()
        (runtime.outputs / "escape").symlink_to(runtime.code, target_is_directory=True)
        with pytest.raises(ValueError, match="symlink"):
            a._publish(kwargs["output_root"], "escape/forbidden.json", b"forbidden")
        return {"boundary_verified": True}
    monkeypatch.setattr(a, "dispatch_operation", observe)
    assert a.dispatch({}) == {"boundary_verified": True}
