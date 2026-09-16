"""Explicit Modal bootstrap and one bounded dispatcher; import never contacts Modal.

Only the child interpreter constructs QwenModel. Build/deploy helpers are operator
entrypoints, not import-time actions. See docs/modal-deployment.md before execution.
"""

from __future__ import annotations

import array
import hashlib
import math
import os
import platform
import re
import signal
import subprocess
import sys
import tempfile
import time
import uuid
import xml.etree.ElementTree as ET
from contextlib import suppress
from pathlib import Path

from PIL import Image

from active_ocr.integrations import modal_model as m
from active_ocr.integrations import qwen as q
from active_ocr.models import ExecutionTelemetry, Prediction, RevealedExample, SimulationPage

BUILD_EVIDENCE_ROOT = "/build-evidence"


class WorkerError(RuntimeError):
    """Only bounded codes may cross the provider/log boundary."""


def _read(root: Path, ref: m.ArtifactRef | q.FileEntry) -> bytes:
    key = ref.key if isinstance(ref, m.ArtifactRef) else ref.filename
    path = q.safe_path(root, key)
    if not path.is_file() or path.stat().st_size != ref.bytes or q.file_hash(path) != ref.sha256:
        raise WorkerError("identity")
    return path.read_bytes()


def _ref(root: Path, key: str) -> m.ArtifactRef:
    path = q.safe_path(root, m.relative_key(key))
    return m.ArtifactRef(key=key, bytes=path.stat().st_size, sha256=q.file_hash(path))


def _json(root: Path, key: str, model):
    raw = q.safe_path(root, key).read_bytes()
    value = model.model_validate(q.strict_json(raw))
    if raw != q.canonical(value.model_dump(mode="json")):
        raise WorkerError("identity")
    return value


def _publish(root: Path, key: str, data: bytes) -> m.ArtifactRef:
    """Atomic, no-clobber publication; a previous different value is never replaced."""
    path = q.safe_path(root, m.relative_key(key))
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".publish-", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
            try:
                os.link(temporary, path)
            except FileExistsError:
                if path.read_bytes() != data:
                    raise WorkerError("identity") from None
        finally:
            temporary.unlink(missing_ok=True)
    return _ref(root, key)


def _deadline(deadline: int) -> float:
    remaining = deadline - time.time()
    if remaining <= 0:
        raise WorkerError("deadline")
    return remaining


def _terminate(child) -> None:
    """Best-effort group stop, but never report success until the child is reaped."""
    with suppress(ProcessLookupError):
        os.killpg(child.pid, signal.SIGTERM)
    try:
        child.wait(timeout=3)
    except subprocess.TimeoutExpired:
        with suppress(ProcessLookupError):
            os.killpg(child.pid, signal.SIGKILL)
        child.wait(timeout=5)


def _files(root: Path, entries: tuple[q.FileEntry, ...]) -> None:
    for entry in entries:
        path = q.safe_path(root, entry.filename)
        if (
            not path.is_file()
            or path.stat().st_size != entry.bytes
            or q.file_hash(path) != entry.sha256
        ):
            raise WorkerError("identity")


def _sdk(sdk=None):
    if sdk is None:
        import modal

        sdk = modal
    from importlib.metadata import version

    if sdk.__name__ == "modal" and version("modal") != "1.5.5":
        raise WorkerError("identity")
    return sdk


def _verify_export(checkout: Path, spec: m.BuildSpec, requirements: Path) -> None:
    """Reproduce the lock export locally, offline, before any image construction."""

    def output(args):
        result = subprocess.run(args, cwd=checkout, capture_output=True, check=False, timeout=30)
        if result.returncode:
            raise WorkerError("identity")
        return result.stdout

    if output(["git", "rev-parse", "HEAD"]).decode().strip() != spec.source_sha:
        raise WorkerError("identity")
    if output(["git", "status", "--porcelain", "--untracked-files=no"]).strip():
        raise WorkerError("identity")
    exported = output(
        [
            "uv",
            "export",
            "--offline",
            "--frozen",
            "--format",
            "requirements-txt",
            "--no-emit-project",
            "--no-header",
            "--no-annotate",
            "--extra",
            "dev",
            "--extra",
            "gpu",
            "--extra",
            "modal",
        ]
    )

    def body(raw):
        return [
            line.strip()
            for line in raw.splitlines()
            if line.strip() and not line.lstrip().startswith(b"#")
        ]

    if body(exported) != body(requirements.read_bytes()):
        raise WorkerError("identity")


def prepare_image(
    spec: m.BuildSpec,
    *,
    checkout: Path,
    requirements: Path,
    processor_root: Path,
    output_volume,
    sdk=None,
):
    """Construct only; caller supplies an already observed output Volume handle."""
    sdk = _sdk(sdk)
    checkout = q.safe_path(checkout, checkout.absolute())
    _files(checkout / "src", spec.code_files)
    _files(checkout, (spec.cpu_test_file,))
    if q.file_hash(q.safe_path(checkout, "uv.lock")) != spec.lock_sha256:
        raise WorkerError("identity")
    _files(requirements.parent, (spec.requirements_file,))
    if requirements.name != spec.requirements_file.filename:
        raise WorkerError("identity")
    _verify_export(checkout, spec, requirements)
    processor_entries = tuple(
        q.FileEntry(**e) for e in q.asset_manifest(q.PROCESSOR_FILES)["files"]
    )
    _files(processor_root, processor_entries)
    image = sdk.Image.debian_slim(python_version="3.11").pip_install_from_requirements(
        str(requirements), extra_options="--require-hashes"
    )
    copies = [(checkout / "src" / e.filename, "/opt/ocr/" + e.filename) for e in spec.code_files]
    copies += [
        (checkout / spec.cpu_test_file.filename, "/opt/ocr/tests/test_qwen.py"),
        (checkout / "uv.lock", "/opt/ocr/uv.lock"),
        (requirements, "/opt/ocr/requirements-linux.txt"),
    ]
    copies += [
        (processor_root / e.filename, "/opt/ocr/processor/" + e.filename) for e in processor_entries
    ]
    for local, remote in copies:
        image = image.add_local_file(str(local), remote, copy=True)
    image = image.env(
        {
            "PYTHONPATH": "/opt/ocr",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    return image.run_function(
        cpu_build_gate,
        kwargs={
            "build_spec": spec.model_dump(mode="json"),
            "build_spec_sha256": spec.sha256,
            "output_volume_id": output_volume.object_id,
        },
        include_source=False,
        cpu=2,
        memory=32768,
        timeout=900,
        gpu=None,
        volumes={
            "/build-evidence": output_volume.with_mount_options(
                sub_path="/builds/" + spec.sha256, read_only=False
            )
        },
    )


def _cpu_report(junit: Path, returncode: int) -> dict:
    cases = list(ET.parse(junit).getroot().iter("testcase"))
    names = [case.attrib["classname"] + "::" + case.attrib["name"] for case in cases]
    failures = sum(bool(list(c)) for c in cases)
    if returncode != 0 or len(cases) != 11 or failures or len(set(names)) != 11:
        raise WorkerError("model")
    return {
        "schema_version": 1,
        "selected": 11,
        "passed": 11,
        "skipped": 0,
        "failed": 0,
        "tests": sorted(names),
    }


def _cpu_failure(
    junit: Path, returncode: int | None, code: str, spec: m.BuildSpec, output: Path, volume_id: str
) -> None:
    """Bounded diagnostics for synthetic tests only; never a successful receipt."""
    cases = []
    with suppress(OSError, ET.ParseError):
        cases = list(ET.parse(junit).getroot().iter("testcase"))
    records = []
    for case in cases[:32]:
        failure = next((c for c in case if c.tag in {"failure", "error", "skipped"}), None)
        excerpt = "" if failure is None else failure.attrib.get("message", "")[:400]
        for key, value in os.environ.items():
            if value and re.search("token|password|secret|credential|api.?key", key, re.I):
                excerpt = excerpt.replace(value, "[redacted]")
        excerpt = re.sub(
            r"(?i)(token|password|secret|credential|api.?key)\s*[:=]\s*\S+",
            r"\1=[redacted]",
            excerpt,
        )
        records.append(
            {
                "name": (case.get("classname", "") + "::" + case.get("name", ""))[:256],
                "status": failure.tag if failure is not None else "passed",
                "excerpt": excerpt,
            }
        )
    diagnostic = {
        "schema_version": 1,
        "status": "failed",
        "code": code,
        "build_spec_sha256": spec.sha256,
        "returncode": returncode,
        "reported_cases": len(cases) if cases else None,
        "records_truncated": len(cases) > 32,
        "tests": records,
    }
    raw = q.canonical(diagnostic)
    # Safe log is fallback evidence if the Volume itself is unavailable.
    print(raw.decode(), flush=True)
    try:
        _publish(output, "failures/" + uuid.uuid4().hex + ".json", raw)
        _sdk().Volume.from_id(volume_id).commit()
    except Exception:
        pass


def cpu_build_gate(build_spec: dict, build_spec_sha256: str, output_volume_id: str) -> None:
    """Remote CPU image-build step. Never invoked by module import or local tests."""
    spec = m.BuildSpec.model_validate(build_spec)
    root, output = Path(m.CODE_ROOT), Path(BUILD_EVIDENCE_ROOT)
    if spec.sha256 != build_spec_sha256:
        raise WorkerError("identity")
    _files(root, spec.code_files + (spec.cpu_test_file, spec.requirements_file))
    if q.file_hash(root / "uv.lock") != spec.lock_sha256:
        raise WorkerError("identity")
    _files(
        root / "processor",
        tuple(q.FileEntry(**e) for e in q.asset_manifest(q.PROCESSOR_FILES)["files"]),
    )
    with tempfile.TemporaryDirectory(prefix="ocr-cpu-") as temporary:
        junit = Path(temporary) / "report.xml"
        child = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "pytest",
                str(root / "tests/test_qwen.py"),
                "-k",
                "actual and not staged_headers",
                "--junitxml=" + str(junit),
                "--basetemp=" + str(Path(temporary) / "tests"),
                "-q",
            ],
            cwd=root,
            env={
                **os.environ,
                "QWEN_PROCESSOR_DIR": str(root / "processor"),
                "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
                "PYTHONPATH": str(root),
            },
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        try:
            child.wait(timeout=850)
        except BaseException:
            _terminate(child)
            _cpu_failure(
                junit, child.returncode, "timeout_or_interruption", spec, output, output_volume_id
            )
            raise WorkerError("model") from None
        try:
            report = _cpu_report(junit, child.returncode)
        except (WorkerError, OSError, ET.ParseError):
            _cpu_failure(junit, child.returncode, "cpu_checks", spec, output, output_volume_id)
            raise WorkerError("model") from None
    environment = q.installed_packages()
    if (
        platform.system() != "Linux"
        or platform.machine() != "x86_64"
        or not environment.python.startswith("3.11.")
    ) or any(
        dict(environment.packages).get(k) != v for k, v in {**q.PINS, "modal": "1.5.5"}.items()
    ):
        raise WorkerError("identity")
    receipt = m.BuildReceipt(
        build_spec_sha256=spec.sha256,
        source_sha=spec.source_sha,
        code_files=spec.code_files,
        environment=environment,
        cpu_report=_publish(output, "cpu-report.json", q.canonical(report)),
        cpu_passed=11,
        cpu_skipped=0,
    )
    raw = q.canonical(receipt.model_dump(mode="json"))
    _publish(output, "build-receipt.json", raw)
    _publish(root, "build-receipt.json", raw)
    _sdk().Volume.from_id(output_volume_id).commit()


def build_image(
    spec: m.BuildSpec,
    *,
    app,
    checkout: Path,
    requirements: Path,
    processor_root: Path,
    output_volume,
    sdk=None,
) -> tuple[str, m.BuildReceipt, dict]:
    """Explicit provider action. Missing cached evidence fails without rebuilding."""
    image = prepare_image(
        spec,
        checkout=checkout,
        requirements=requirements,
        processor_root=processor_root,
        output_volume=output_volume,
        sdk=sdk,
    )
    image.build(app)
    image_id = image.object_id  # Observed only after successful eager build.
    prefix = "/builds/" + spec.sha256 + "/"
    raw = b"".join(output_volume.read_file(prefix + "build-receipt.json"))
    receipt = m.BuildReceipt.model_validate(q.strict_json(raw))
    if raw != q.canonical(receipt.model_dump(mode="json")) or (
        receipt.build_spec_sha256 != spec.sha256
        or receipt.source_sha != spec.source_sha
        or receipt.code_files != spec.code_files
        or receipt.cpu_report.key != "cpu-report.json"
    ):
        raise WorkerError("identity")
    report_raw = b"".join(output_volume.read_file(prefix + receipt.cpu_report.key))
    if (
        len(report_raw) != receipt.cpu_report.bytes
        or hashlib.sha256(report_raw).hexdigest() != receipt.cpu_report.sha256
    ):
        raise WorkerError("identity")
    report = q.strict_json(report_raw)
    if report_raw != q.canonical(report) or any(
        report.get(k) != v
        for k, v in {"selected": 11, "passed": 11, "skipped": 0, "failed": 0}.items()
    ):
        raise WorkerError("identity")
    return image_id, receipt, report


def create_app(settings: m.RuntimeSettings, *, settings_file: Path, sdk=None):
    """Construct one Function; explicit operator deploy occurs outside this helper."""
    sdk = _sdk(sdk)
    if q.safe_path(settings_file.parent, settings_file.absolute()).read_bytes() != q.canonical(
        settings.model_dump(mode="json")
    ):
        raise WorkerError("identity")
    input_volume = sdk.Volume.from_id(settings.input_volume_id)
    output_volume = sdk.Volume.from_id(settings.output_volume_id)
    app = sdk.App(settings.deployment_name, include_source=False)

    app.function(
        name="dispatch",
        image=sdk.Image.from_id(settings.image_id).add_local_file(
            str(settings_file), "/opt/ocr/runtime-settings.json", copy=False
        ),
        include_source=False,
        volumes={
            m.INPUT_ROOT: input_volume.with_mount_options(
                sub_path=settings.input_sub_path, read_only=True
            ),
            m.OUTPUT_ROOT: output_volume.with_mount_options(
                sub_path=settings.output_sub_path, read_only=False
            ),
        },
        gpu="L40S",
        cpu=(2, 2),
        memory=(32768, 32768),
        max_containers=1,
        min_containers=0,
        buffer_containers=0,
        scaledown_window=2,
        retries=0,
        timeout=600,
        startup_timeout=300,
    )(dispatch)

    return app


def dispatch(payload: dict) -> dict:
    """Importable provider Function, using only independently frozen settings."""
    try:
        q.safe_path(Path(m.CODE_ROOT), Path(__file__).absolute())
        q.safe_path(Path(m.CODE_ROOT), Path(m.__file__).absolute())
        settings = _json(Path(m.CODE_ROOT), "runtime-settings.json", m.RuntimeSettings)
        sdk = _sdk()
        return dispatch_operation(
            settings,
            payload,
            provider_call_id=sdk.current_function_call_id(),
            commit=sdk.Volume.from_id(settings.output_volume_id).commit,
        )
    except BaseException:
        raise WorkerError("dispatch_failed") from None


def _page(page: m.RemotePage, context: m.RunContext, input_root: Path) -> SimulationPage:
    path = q.safe_path(input_root, page.image_key)
    if q.file_hash(path) != page.image_sha256:
        raise WorkerError("identity")
    with Image.open(path) as image:
        if (
            image.size != (page.width, page.height)
            or image.mode != "RGB"
            or image.getexif().get(274, 1) != 1
        ):
            raise WorkerError("identity")
        image.load()
    return SimulationPage(
        **page.model_dump(exclude={"image_key"}),
        image_uri=str(path),
        source_image=page.image_key,
        source_policy=context.source_policy,
    )


def _worker(settings: m.RuntimeSettings, code_root: Path) -> q.WorkerManifest:
    build = _json(code_root, "build-receipt.json", m.BuildReceipt)
    if build != settings.build or q.installed_packages() != build.environment:
        raise WorkerError("identity")
    _files(code_root, build.code_files)
    worker = q.WorkerManifest(
        schema_version=1,
        source_sha=build.source_sha,
        files=build.code_files,
        environment=build.environment,
        build_spec_sha256=build.build_spec_sha256,
        deployment_reference=settings.deployment_reference,
    )
    _publish(code_root, "worker.json", q.canonical(worker.model_dump(mode="json")))
    return worker


def _checkpoint(
    root: Path, model_id: str, request, *, base: bool = False
) -> tuple[q.Checkpoint, list[m.ArtifactRef]]:
    sha = model_id.rsplit(":", 1)[1]
    key = f"qwen/checkpoints/{sha}/manifest.json"
    checkpoint = _json(root, key, q.Checkpoint)
    # This pure helper reads only real_config; no model instance/state is created.
    expected_binding = q.QwenModel._binding(request.context)
    if _ref(root, key).sha256 != sha or checkpoint.binding != expected_binding:
        raise WorkerError("identity")
    directory = q.safe_path(root, f"qwen/checkpoints/{sha}")
    q.verify_checkpoint_files(
        directory, checkpoint, q.canonical(checkpoint.model_dump(mode="json"))
    )
    if (
        base
        or isinstance(request, m.BaseRequest)
        or (isinstance(request, m.PredictRequest) and request.round_number == 0)
    ):
        if checkpoint != q.Checkpoint(kind="base", binding=checkpoint.binding):
            raise WorkerError("identity")
    elif (
        checkpoint.kind != "adapter"
        or checkpoint.experiment_id != request.context.experiment_id
        or checkpoint.round_number != request.round_number
    ):
        raise WorkerError("identity")
    else:
        q.check_training_record(checkpoint)
        if [f.filename for f in checkpoint.files] != [
            "adapter_config.json",
            "adapter_model.safetensors",
        ]:
            raise WorkerError("identity")
        if isinstance(request, m.FitRequest):
            pages = tuple((e.page.id, e.page.image_sha256) for e in request.examples)
            if checkpoint.selected != pages or checkpoint.seed != request.seed:
                raise WorkerError("identity")
            if checkpoint.target_sha256 != q.digest(
                [(e.page.id, q.serialize_target(e)) for e in request.examples]
            ):
                raise WorkerError("identity")
    return checkpoint, [_ref(root, key)] + [
        _ref(root, f"qwen/checkpoints/{sha}/{e.filename}") for e in checkpoint.files
    ]


def _probe(
    root: Path, model_id: str, stage: str, page: m.RemotePage
) -> tuple[m.ProbeMetadata, m.ArtifactRef, m.ArtifactRef]:
    sha = model_id.rsplit(":", 1)[1]
    key = f"qwen/probes/{sha}/{stage}.json"
    metadata = _json(root, key, m.ProbeMetadata)
    if (
        metadata.model_id,
        metadata.page_id,
        metadata.image_sha256,
        metadata.original_width,
        metadata.original_height,
    ) != (model_id, page.id, page.image_sha256, page.width, page.height):
        raise WorkerError("identity")
    if metadata.logits.key != f"probes/{sha}/{stage}.f32le":
        raise WorkerError("identity")
    outer = metadata.logits.model_copy(update={"key": "qwen/" + metadata.logits.key})
    _read(root, outer)
    return metadata, _ref(root, key), outer


def _reload(
    root: Path, model_id: str, page: m.RemotePage, pids=None
) -> tuple[m.ReloadEvidence, list[m.ArtifactRef]]:
    before, before_ref, before_logits = _probe(root, model_id, "before", page)
    after, after_ref, after_logits = _probe(root, model_id, "after", page)
    if pids is not None and (before.process_id, after.process_id) != tuple(pids):
        raise WorkerError("identity")
    sha = model_id.rsplit(":", 1)[1]
    m._verify_adapter_tensors(
        q.safe_path(root, f"qwen/checkpoints/{sha}/adapter_model.safetensors"),
        before.adapter_tensors,
    )
    if before.model_dump(exclude={"process_id", "logits"}) != after.model_dump(
        exclude={"process_id", "logits"}
    ):
        raise WorkerError("model")
    a, b = array.array("f"), array.array("f")
    a.frombytes(_read(root, before_logits))
    b.frombytes(_read(root, after_logits))
    if sys.byteorder != "little":
        a.byteswap()
        b.byteswap()
    maximum = 0.0
    for x, y in zip(a, b, strict=True):
        if not math.isfinite(x) or not math.isfinite(y) or abs(y - x) > 0.01 + 0.001 * abs(y):
            raise WorkerError("model")
        maximum = max(maximum, abs(y - x))
    evidence = m.ReloadEvidence(
        before=before_ref,
        after=after_ref,
        train_process_id=before.process_id,
        reload_process_id=after.process_id,
        max_absolute_difference=maximum,
        exact_tensors_ids_status_regions=True,
    )
    return evidence, [before_ref, after_ref, before_logits, after_logits]


def run_child(payload: dict, deadline: int) -> dict:
    """Transient pipe input; kill/reap the entire child group on timeout/interruption."""
    remaining = _deadline(deadline)
    child = subprocess.Popen(
        [sys.executable, "-m", "active_ocr.entrypoints.modal_app", "--child"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    try:
        stdout, _ = child.communicate(q.canonical(payload), timeout=remaining)
        if child.returncode != 0:
            raise WorkerError("model")
        result = q.strict_json(stdout)
        if result.get("process_id") != child.pid or result.get("status") != "ok":
            raise WorkerError("model")
        _deadline(deadline)
        return result
    except BaseException:
        _terminate(child)
        raise WorkerError("deadline" if time.time() >= deadline else "model") from None


def _child(payload: dict) -> dict:
    """Executed only in the isolated interpreter; second stage has no examples/labels."""
    context = m.RunContext.model_validate(payload["context"])
    input_root, output_root = Path(payload["input_root"]), Path(payload["output_root"])
    model = q.QwenModel(
        input_root,
        output_root / "qwen",
        context.real_config,
        runtime_manifest=Path(payload["runtime_manifest"]),
        deadline_unix_seconds=payload["deadline"],
    )
    if payload["stage"] == "reload":
        page = _page(m.RemotePage.model_validate(payload["page"]), context, input_root)
        before = m.ArtifactRef.model_validate(payload["before"])
        logits = m.ArtifactRef.model_validate(payload["before_logits"])
        model.verify_reload(
            page,
            experiment_id=context.experiment_id,
            round_number=payload["round_number"],
            model_id=payload["model_id"],
            before_metadata=q.safe_path(output_root, before.key),
            before_metadata_sha256=before.sha256,
            before_logits=q.safe_path(output_root, logits.key),
            before_logits_sha256=logits.sha256,
        )
        model_id, predictions = payload["model_id"], ()
    else:
        request = m.REQUEST_ADAPTER.validate_python(payload["request"])
        if request.context != context:
            raise WorkerError("identity")
        predictions = ()
        if isinstance(request, m.BaseRequest):
            model_id = model.load_base(experiment_id=context.experiment_id)
        elif isinstance(request, m.FitRequest):
            model_id = model.fit(
                tuple(
                    RevealedExample(page=_page(e.page, context, input_root), regions=e.regions)
                    for e in request.examples
                ),
                seed=request.seed,
                experiment_id=context.experiment_id,
                round_number=request.round_number,
                external_reload=True,
            )
        else:
            model_id = request.input_checkpoint
            predictions = model.predict(
                tuple(_page(p, context, input_root) for p in request.pages),
                experiment_id=context.experiment_id,
                round_number=request.round_number,
                model_id=model_id,
                purpose=request.purpose,
            )
    return {
        "status": "ok",
        "process_id": os.getpid(),
        "model_id": model_id,
        "predictions": [p.model_dump(mode="json") for p in predictions],
        "telemetry": model.telemetry.model_dump(mode="json") if model.telemetry else None,
    }


def _validate_result(root: Path, result: m.OperationResult, invocation: m.Invocation) -> None:
    result.check_request(invocation)
    for ref in result.artifacts:
        _read(root, ref)
    _, required = _checkpoint(root, result.model_id, invocation.request)
    refs = {r.key: r for r in result.artifacts}
    if any(refs.get(r.key) != r for r in required):
        raise WorkerError("identity")
    if isinstance(invocation.request, m.PredictRequest):
        pages = {p.id: p for p in invocation.request.pages}
        for prediction in result.predictions:
            page = pages[prediction.page_id]
            if prediction.status.value != "ok" and prediction.regions:
                raise WorkerError("model")
            if len({r.id for r in prediction.regions}) != len(prediction.regions):
                raise WorkerError("model")
            for region in prediction.regions:
                box = region.box
                if (
                    not all(math.isfinite(x) for x in (box.x, box.y, box.width, box.height))
                    or box.x + box.width > page.width
                    or box.y + box.height > page.height
                ):
                    raise WorkerError("model")
            raw_key = prediction.raw_output_artifact
            raw = _read(root, refs[raw_key])
            record = q.strict_json(raw)
            if (
                raw != q.canonical(record)
                or raw_key != "qwen/receipts/" + hashlib.sha256(raw).hexdigest() + ".json"
                or record.get("operation") != "predict"
                or record.get("experiment_id") != invocation.request.context.experiment_id
                or record.get("round_number") != invocation.request.round_number
                or record.get("purpose") != invocation.request.purpose
                or record.get("model_id") != result.model_id
            ):
                raise WorkerError("identity")
            raw_pages = record.get("pages", [])
            if [p.get("page_id") for p in raw_pages] != [p.id for p in invocation.request.pages]:
                raise WorkerError("identity")
            observed = next(p for p in raw_pages if p["page_id"] == page.id)
            if (
                observed.get("image_sha256") != page.image_sha256
                or observed.get("status") != prediction.status.value
                or observed.get("finish_reason") != prediction.finish_reason
            ):
                raise WorkerError("identity")
            if (
                prediction.status.value == "ok"
                and q.parse_regions(observed["text"], page.width, page.height) != prediction.regions
            ):
                raise WorkerError("model")
    if isinstance(invocation.request, m.FitRequest):
        page = min((e.page for e in invocation.request.examples), key=lambda p: p.id)
        reload, required = _reload(root, result.model_id, page)
        if result.reload != reload or any(refs.get(r.key) != r for r in required):
            raise WorkerError("identity")


def dispatch_operation(
    settings: m.RuntimeSettings,
    payload: dict,
    *,
    provider_call_id: str,
    commit,
    input_root: Path = Path(m.INPUT_ROOT),
    output_root: Path = Path(m.OUTPUT_ROOT),
    code_root: Path = Path(m.CODE_ROOT),
    child_runner=run_child,
) -> dict:
    """Verify all preconditions before spawning; complete.json is published last."""
    invocation, prefix = None, None
    try:
        invocation = m.Invocation.model_validate(payload)
        settings.check_invocation(invocation)
        if not provider_call_id or not isinstance(provider_call_id, str):
            raise WorkerError("identity")
        _deadline(invocation.deadline_unix_seconds)
        execution_deadline = min(
            invocation.deadline_unix_seconds, int(time.time()) + settings.timeout_seconds - 10
        )
        if invocation.first_submission_unix_seconds > time.time():
            raise WorkerError("identity")
        for root in (input_root, output_root, code_root):
            if not q.safe_path(root, root.absolute()).is_dir():
                raise WorkerError("identity")
        roots = [r.absolute() for r in (input_root, output_root, code_root)]
        if any(
            x.is_relative_to(y) for i, x in enumerate(roots) for j, y in enumerate(roots) if i != j
        ):
            raise WorkerError("identity")
        _files(input_root, settings.bundle.files)
        if any(p.is_symlink() for p in input_root.rglob("*")):
            raise WorkerError("identity")
        actual = {
            p.relative_to(input_root).as_posix() for p in input_root.rglob("*") if p.is_file()
        }
        if actual != {f.filename for f in settings.bundle.files}:
            raise WorkerError("identity")
        worker = _worker(settings, code_root)
        control = {
            "settings_sha256": q.digest(settings.model_dump(mode="json")),
            "first_submission_unix_seconds": invocation.first_submission_unix_seconds,
            "deadline_unix_seconds": invocation.deadline_unix_seconds,
        }
        _publish(output_root, "run-control.json", q.canonical(control))
        commit()
        operation_root = q.safe_path(output_root, "operations/" + invocation.operation_id)
        completions = sorted(operation_root.glob("attempts/*/executions/*/complete.json"))
        if len(completions) > 1 or list(
            operation_root.glob("attempts/*/executions/*/failure.json")
        ):
            raise WorkerError("identity")
        if completions:
            key = completions[0].relative_to(output_root).as_posix()
            result = _json(output_root, key, m.OperationResult)
            _validate_result(output_root, result, invocation)
            if _read(output_root, result.worker_manifest) != q.canonical(
                worker.model_dump(mode="json")
            ):
                raise WorkerError("identity")
            return m.DispatchResponse(result=result, completion=_ref(output_root, key)).model_dump(
                mode="json"
            )
        # A failed/partial execution is not a license to repeat paid work.
        if operation_root.exists() and any(operation_root.iterdir()):
            raise WorkerError("interrupted")
        execution_id = uuid.uuid4().hex
        prefix = (
            f"operations/{invocation.operation_id}/attempts/{invocation.attempt_id}/"
            f"executions/{execution_id}/"
        )
        q.safe_path(output_root, prefix + "worker.json").parent.mkdir(parents=True, exist_ok=False)
        worker_ref = _publish(
            output_root, prefix + "worker.json", q.canonical(worker.model_dump(mode="json"))
        )
        commit()  # Make the started execution durable before model work.
        request = invocation.request
        if isinstance(request, (m.FitRequest, m.PredictRequest)):
            _checkpoint(
                output_root,
                request.input_checkpoint,
                request,
                base=isinstance(request, m.FitRequest),
            )
        (output_root / "qwen").mkdir(exist_ok=True)
        common = {
            "context": request.context.model_dump(mode="json"),
            "input_root": str(input_root),
            "output_root": str(output_root),
            "runtime_manifest": str(code_root / "worker.json"),
            "deadline": execution_deadline,
        }
        first = child_runner(
            {**common, "stage": "operation", "request": request.model_dump(mode="json")},
            execution_deadline,
        )
        model_id = first["model_id"]
        _, artifacts = _checkpoint(output_root, model_id, request)
        artifacts.append(worker_ref)
        reload = None
        if isinstance(request, m.FitRequest):
            page = min((e.page for e in request.examples), key=lambda p: p.id)
            before, before_ref, before_logits = _probe(output_root, model_id, "before", page)
            if before.process_id != first["process_id"]:
                raise WorkerError("identity")
            _deadline(execution_deadline)
            second = child_runner(
                {
                    **common,
                    "stage": "reload",
                    "page": page.model_dump(mode="json"),
                    "round_number": request.round_number,
                    "model_id": model_id,
                    "before": before_ref.model_dump(mode="json"),
                    "before_logits": before_logits.model_dump(mode="json"),
                },
                execution_deadline,
            )
            reload, probe_refs = _reload(
                output_root, model_id, page, (first["process_id"], second["process_id"])
            )
            artifacts.extend(probe_refs)
        predictions = []
        for value in first.get("predictions", []):
            prediction = Prediction.model_validate(value)
            path = q.safe_path(output_root / "qwen", prediction.raw_output_artifact)
            key = path.relative_to(output_root).as_posix()
            artifacts.append(_ref(output_root, key))
            predictions.append(prediction.model_copy(update={"raw_output_artifact": key}))
        result = m.OperationResult(
            operation_id=invocation.operation_id,
            attempt_id=invocation.attempt_id,
            execution_id=execution_id,
            provider_call_id=provider_call_id,
            model_id=model_id,
            predictions=tuple(predictions),
            artifacts=tuple({r.key: r for r in artifacts}.values()),
            worker_manifest=worker_ref,
            reload=reload,
            telemetry=ExecutionTelemetry.model_validate(first["telemetry"])
            if first.get("telemetry")
            else None,
        )
        _validate_result(output_root, result, invocation)
        _deadline(execution_deadline)
        completion = _publish(
            output_root, prefix + "complete.json", q.canonical(result.model_dump(mode="json"))
        )
        commit()
        return m.DispatchResponse(result=result, completion=completion).model_dump(mode="json")
    except BaseException as error:
        code = str(error) if isinstance(error, WorkerError) else "model"
        if code not in {"identity", "model", "deadline", "interrupted", "invalid_request"}:
            code = "model"
        if invocation is None:
            code = "invalid_request"
        if prefix and invocation is not None:
            failure = m.FailureReceipt(
                operation_id=invocation.operation_id,
                attempt_id=invocation.attempt_id,
                execution_id=execution_id,
                provider_call_id=provider_call_id,
                code=code,
            )
            try:
                _publish(
                    output_root,
                    prefix + "failure.json",
                    q.canonical(failure.model_dump(mode="json")),
                )
                commit()
            except Exception:
                pass  # Failure persistence is best effort; never assert provider termination.
        raise WorkerError(code) from None


if __name__ == "__main__":
    # Reserve the result pipe before silencing Python and native-library output.
    result_fd = os.dup(sys.stdout.fileno())
    with open(os.devnull, "wb") as sink:
        os.dup2(sink.fileno(), 1)
        os.dup2(sink.fileno(), 2)
    try:
        if sys.argv[1:] != ["--child"]:
            raise WorkerError("invalid_request")
        result = _child(q.strict_json(sys.stdin.buffer.read()))
        with os.fdopen(os.dup(result_fd), "wb") as stream:
            stream.write(q.canonical(result))
    except BaseException:
        sys.exit(1)
    finally:
        os.close(result_fd)
