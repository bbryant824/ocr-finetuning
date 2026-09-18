"""Wire contract and synchronous durable Modal coordinator.

Only fit carries selected labels. Journal semantic_record(), never the wire fit payload.
Platform owns the one dispatcher; see docs/PROJECT_GUIDE.md for bootstrap and ownership.
"""

from __future__ import annotations

import math
from pathlib import PurePosixPath
from typing import Annotated, Literal

from pydantic import Field, TypeAdapter, field_validator, model_validator

from active_ocr.integrations.qwen import (
    ASSETS,
    MAX_OUTPUT_TOKENS,
    PROCESSOR_FILES,
    FileEntry,
    Packages,
    adapter_shapes,
    asset_manifest,
    digest,
)
from active_ocr.models import (
    SHA256,
    CheckpointReference,
    CommitSHA,
    ExecutionTelemetry,
    Model,
    Prediction,
    RealOCRConfig,
    Region,
    SourcePolicy,
    SourceRegion,
    Split,
    validate_source_policy,
)

Identifier = Annotated[str, Field(strict=True, pattern=r"^[0-9a-f]{32}$")]
Name = Annotated[str, Field(strict=True, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,127}$")]
PositiveInt = Annotated[int, Field(strict=True, gt=0)]
INPUT_ROOT = "/inputs"
OUTPUT_ROOT = "/outputs"
CODE_ROOT = "/opt/ocr"
QWEN_OUTPUT_ROOT = "/outputs/qwen"


def relative_key(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in value
        or path.as_posix() != value
        or value == "."
    ):
        raise ValueError("expected a normalized relative POSIX artifact key")
    return value


class ArtifactRef(Model):
    """Bytes relative to the assigned output subtree; never a host/provider URL."""

    key: str = Field(strict=True)
    bytes: int = Field(strict=True, ge=0)
    sha256: SHA256

    _safe_key = field_validator("key")(relative_key)


class InputBundle(Model):
    schema_version: Literal[1] = 1
    files: tuple[FileEntry, ...]

    @model_validator(mode="after")
    def allowlist(self) -> InputBundle:
        names = [f.filename for f in self.files]
        if names != sorted(set(names)):
            raise ValueError("bundle inventory must be sorted and unique")
        required = {"model/" + name for name in ASSETS}
        if not required.issubset(names):
            raise ValueError("bundle missing pinned model/processor assets")
        for entry in self.files:
            if entry.filename in required:
                size, sha = ASSETS[entry.filename.removeprefix("model/")]
                if (entry.bytes, entry.sha256) != (size, sha):
                    raise ValueError("bundle model identity differs from approved assets")
            elif entry.filename != "images/" + entry.sha256 or entry.bytes == 0:
                raise ValueError(
                    "bundle accepts only content-addressed images and pinned model files"
                )
        return self

    @property
    def sha256(self) -> str:
        return digest(self.model_dump(mode="json"))


RUNTIME_CODE_FILES = {
    "active_ocr/__init__.py",
    "active_ocr/models.py",
    "active_ocr/integrations/__init__.py",
    "active_ocr/integrations/qwen.py",
    "active_ocr/integrations/modal_model.py",
    "active_ocr/entrypoints/__init__.py",
    "active_ocr/entrypoints/modal_app.py",
}


class BuildSpec(Model):
    schema_version: Literal[1] = 1
    source_sha: CommitSHA
    code_files: tuple[FileEntry, ...]
    lock_sha256: SHA256
    requirements_file: FileEntry
    cpu_test_file: FileEntry
    processor_manifest_sha256: SHA256
    modal_version: Literal["1.5.5"] = "1.5.5"
    base: Literal["debian_slim:python3.11"] = "debian_slim:python3.11"
    cpu: Literal[2] = 2
    memory_mib: Literal[32768] = 32768
    timeout_seconds: Literal[900] = 900
    gpu: None = None
    extras: tuple[Literal["dev"], Literal["gpu"], Literal["modal"]] = ("dev", "gpu", "modal")

    @model_validator(mode="after")
    def exact_build_inputs(self) -> BuildSpec:
        if [f.filename for f in self.code_files] != sorted(RUNTIME_CODE_FILES):
            raise ValueError("build requires the exact reviewed runtime code allowlist")
        if (
            self.requirements_file.filename != "requirements-linux.txt"
            or self.cpu_test_file.filename != "tests/test_qwen.py"
            or self.processor_manifest_sha256 != digest(asset_manifest(PROCESSOR_FILES))
        ):
            raise ValueError("build accepts only the pinned synthetic test/processor inputs")
        return self

    @property
    def sha256(self) -> str:
        return digest(self.model_dump(mode="json"))


class BuildReceipt(Model):
    """CPU build writes this before its provider image ID is available."""

    schema_version: Literal[1] = 1
    build_spec_sha256: SHA256
    source_sha: CommitSHA
    code_files: tuple[FileEntry, ...]
    environment: Packages
    cpu_report: ArtifactRef  # Relative to /builds/<build_spec_sha256>, not a run.
    cpu_passed: Literal[11]
    cpu_skipped: Literal[0]

    @model_validator(mode="after")
    def inventories(self) -> BuildReceipt:
        names = [f.filename for f in self.code_files]
        if names != sorted(set(names)) or not names or any(not n.endswith(".py") for n in names):
            raise ValueError("build code allowlist must be sorted unique Python files")
        packages = self.environment.packages
        if packages != tuple(sorted(set(packages))) or len(dict(packages)) != len(packages):
            raise ValueError("build package inventory must be sorted with unique names")
        return self

    @property
    def code_bundle_sha256(self) -> str:
        return digest(
            {
                "source_sha": self.source_sha,
                "files": [f.model_dump(mode="json") for f in self.code_files],
            }
        )

    @property
    def remote_dependency_sha256(self) -> str:
        return digest(self.environment.model_dump(mode="json"))


class RemotePage(Model):
    id: str = Field(strict=True, min_length=1)
    document_id: str | None = Field(min_length=1)  # Required null is not implicit admission.
    split: Split
    width: PositiveInt
    height: PositiveInt
    image_sha256: SHA256
    image_key: str = Field(strict=True)

    @model_validator(mode="after")
    def image_address(self) -> RemotePage:
        if self.image_key != "images/" + self.image_sha256:
            raise ValueError("image key must be the content address inside images/")
        return self


class RunContext(Model):
    experiment_id: Identifier
    source_policy: SourcePolicy  # Explicit on every operation, never inferred from null doc IDs.
    bundle_sha256: SHA256
    real_config: RealOCRConfig

    def check_pages(self, pages: tuple[RemotePage, ...], split: Split) -> None:
        if len({p.id for p in pages}) != len(pages):
            raise ValueError("duplicate remote page IDs")
        for page in pages:
            validate_source_policy(page.document_id, page.split, self.source_policy)
            if page.split is not split:
                raise ValueError("wrong operation page split")


class BaseRequest(Model):
    operation: Literal["load_base"] = "load_base"
    context: RunContext
    round_number: Literal[0] = 0
    purpose: Literal["baseline_validation"] = "baseline_validation"
    input_checkpoint: None = None

    def semantic_record(self) -> dict:
        return self.model_dump(mode="json")


class RemoteExample(Model):
    page: RemotePage
    regions: tuple[SourceRegion, ...] = Field(repr=False)

    @model_validator(mode="after")
    def selected_geometry(self) -> RemoteExample:
        if self.page.split is not Split.TRAIN:
            raise ValueError("fit targets must be selected TRAIN")
        if len({r.id for r in self.regions}) != len(self.regions):
            raise ValueError("duplicate selected region IDs")
        for region in self.regions:
            b = region.box
            if (
                not all(math.isfinite(v) for v in (b.x, b.y, b.width, b.height))
                or b.x + b.width > self.page.width
                or b.y + b.height > self.page.height
            ):
                raise ValueError("selected target box outside original image")
            region.text.encode("utf-8", errors="strict")
        return self


class FitRequest(Model):
    operation: Literal["fit"] = "fit"
    context: RunContext
    round_number: PositiveInt
    purpose: Literal["selected_train"] = "selected_train"
    input_checkpoint: (
        CheckpointReference  # Must resolve to the verified base, not previous adapter.
    )
    seed: int = Field(strict=True, ge=0, lt=2**32)
    examples: tuple[RemoteExample, ...] = Field(min_length=1, repr=False)

    @model_validator(mode="after")
    def selected_only(self) -> FitRequest:
        self.context.check_pages(tuple(e.page for e in self.examples), Split.TRAIN)
        return self

    def semantic_record(self) -> dict:
        record = self.model_dump(mode="json", exclude={"examples"})
        record["pages"] = [e.page.model_dump(mode="json") for e in self.examples]
        record["target_sha256"] = digest(
            [
                {"page_id": e.page.id, "regions": [r.model_dump(mode="json") for r in e.regions]}
                for e in self.examples
            ]
        )
        return record


class PredictRequest(Model):
    operation: Literal["predict"] = "predict"
    context: RunContext
    round_number: int = Field(strict=True, ge=0)
    purpose: Literal["pool", "validation", "baseline_validation"]
    input_checkpoint: CheckpointReference
    pages: tuple[RemotePage, ...]

    @model_validator(mode="after")
    def prediction_split(self) -> PredictRequest:
        if (self.round_number == 0) != (self.purpose == "baseline_validation"):
            raise ValueError("round zero is reserved for baseline_validation")
        self.context.check_pages(
            self.pages, Split.TRAIN if self.purpose == "pool" else Split.VALIDATION
        )
        return self

    def semantic_record(self) -> dict:
        return self.model_dump(mode="json")


Request = Annotated[BaseRequest | FitRequest | PredictRequest, Field(discriminator="operation")]
REQUEST_ADAPTER = TypeAdapter(Request)


def operation_id(request: BaseRequest | FitRequest | PredictRequest) -> str:
    return digest({"schema_version": 1, "request": request.semantic_record()})


class Invocation(Model):
    schema_version: Literal[1] = 1
    request: Request
    attempt_id: Identifier  # Client-generated submission identity, never an RNG seed.
    first_submission_unix_seconds: PositiveInt
    deadline_unix_seconds: PositiveInt  # Frozen run-wide deadline, never reset on resume.

    @model_validator(mode="after")
    def bounded_deadline(self) -> Invocation:
        if not 0 < self.deadline_unix_seconds - self.first_submission_unix_seconds <= 2400:
            raise ValueError("run-wide wall-clock envelope must not exceed 2400 seconds")
        return self

    @property
    def operation_id(self) -> str:
        return operation_id(self.request)


class RuntimeSettings(Model):
    """Per-run deployment configuration; generated only after local run creation."""

    schema_version: Literal[1] = 1
    context: RunContext
    environment_name: Name
    deployment_name: Name
    function_name: Literal["dispatch"] = "dispatch"
    image_id: str = Field(strict=True, min_length=1)  # Observed after image.build(app).
    input_volume_name: Name
    output_volume_name: Name
    input_volume_id: str = Field(strict=True, min_length=1)
    output_volume_id: str = Field(strict=True, min_length=1)
    build_spec: BuildSpec
    build: BuildReceipt
    bundle: InputBundle
    gpu: Literal["L40S"] = "L40S"
    cpu: Literal[2] = 2
    memory_mib: Literal[32768] = 32768
    timeout_seconds: Literal[600] = 600
    startup_timeout_seconds: Literal[300] = 300
    max_containers: Literal[1] = 1
    min_containers: Literal[0] = 0
    buffer_containers: Literal[0] = 0
    retries: Literal[0] = 0
    scaledown_seconds: Literal[2] = 2
    aggregate_gpu_seconds: Literal[2400] = 2400
    cancellation_tail_seconds: Literal[300] = 300

    @property
    def deployment_reference(self) -> str:
        return f"modal:{self.environment_name}/{self.deployment_name}/dispatch@{self.image_id}"

    @property
    def input_sub_path(self) -> str:
        return "/bundles/" + self.bundle.sha256

    @property
    def output_sub_path(self) -> str:
        return "/runs/" + self.context.experiment_id

    @model_validator(mode="after")
    def frozen_identity(self) -> RuntimeSettings:
        expected = self.context.real_config.expected_identity
        if (
            self.input_volume_id == self.output_volume_id
            or self.input_volume_name == self.output_volume_name
        ):
            raise ValueError("input/output require two distinct Volumes")
        if (
            self.build.build_spec_sha256 != self.build_spec.sha256
            or self.build.source_sha != self.build_spec.source_sha
            or self.build.code_files != self.build_spec.code_files
        ):
            raise ValueError("CPU build receipt differs from reviewed build inputs")
        if (
            self.context.bundle_sha256 != self.bundle.sha256
            or expected.source_sha != self.build.source_sha
            or expected.code_bundle_sha256 != self.build.code_bundle_sha256
            or expected.remote_dependency_sha256 != self.build.remote_dependency_sha256
            or expected.build_spec_sha256 != self.build.build_spec_sha256
            or expected.deployment_reference != self.deployment_reference
        ):
            raise ValueError("runtime settings differ from frozen identities")
        return self

    def check_invocation(self, invocation: Invocation) -> None:
        if invocation.request.context != self.context:
            raise ValueError("request does not belong to deployed run")
        request = invocation.request
        pages = (
            tuple(e.page for e in request.examples)
            if isinstance(request, FitRequest)
            else request.pages
            if isinstance(request, PredictRequest)
            else ()
        )
        entries = {f.filename: f for f in self.bundle.files}
        if any(
            p.image_key not in entries or entries[p.image_key].sha256 != p.image_sha256
            for p in pages
        ):
            raise ValueError("request image not in verified input bundle")


class ProbeMetadata(Model):
    """Derived TRAIN diagnostic; logits are finite FP32 little-endian rows in a separate file."""

    schema_version: Literal[1] = 1
    model_id: CheckpointReference
    page_id: str = Field(strict=True, min_length=1)
    image_sha256: SHA256
    original_width: PositiveInt
    original_height: PositiveInt
    process_id: PositiveInt
    generated_ids: tuple[Annotated[int, Field(strict=True, ge=0)], ...] = Field(
        min_length=1, max_length=MAX_OUTPUT_TOKENS
    )
    status: Literal["ok", "invalid_output", "truncated"]
    regions: tuple[Region, ...]
    finish_reason: Literal["eos", "length"]
    adapter_tensors: dict[str, SHA256]
    logits: ArtifactRef
    logits_shape: tuple[PositiveInt, Literal[151936]]

    @model_validator(mode="after")
    def logit_shape(self) -> ProbeMetadata:
        rows, columns = self.logits_shape
        if set(self.adapter_tensors) != set(adapter_shapes()):
            raise ValueError("probe requires all 144 adapter tensor hashes")
        if rows != min(8, len(self.generated_ids)) or self.logits.bytes != rows * columns * 4:
            raise ValueError("probe must retain full-vocabulary first-min(8,length) FP32 logits")
        if self.status != "ok" and self.regions:
            raise ValueError("failed probe output cannot contain regions")
        return self


class ReloadEvidence(Model):
    before: ArtifactRef  # Canonical ProbeMetadata JSON.
    after: ArtifactRef
    train_process_id: PositiveInt
    reload_process_id: PositiveInt
    rtol: Literal[0.001] = 0.001
    atol: Literal[0.01] = 0.01
    max_absolute_difference: float = Field(ge=0, allow_inf_nan=False)
    exact_tensors_ids_status_regions: Literal[True]

    @model_validator(mode="after")
    def fresh_interpreter(self) -> ReloadEvidence:
        if self.train_process_id == self.reload_process_id:
            raise ValueError("same-process reload does not satisfy fresh-process evidence")
        return self


class OperationResult(Model):
    schema_version: Literal[1] = 1
    status: Literal["completed"] = "completed"
    operation_id: SHA256
    attempt_id: Identifier
    execution_id: Identifier  # Worker boot/execution UUID; rescheduling must not overwrite outputs.
    provider_call_id: str = Field(strict=True, min_length=1)
    model_id: CheckpointReference
    predictions: tuple[Prediction, ...] = ()
    artifacts: tuple[ArtifactRef, ...]
    worker_manifest: (
        ArtifactRef  # Actual Qwen WorkerManifest, including observed installed packages.
    )
    telemetry: ExecutionTelemetry | None = None
    reload: ReloadEvidence | None = None

    @model_validator(mode="after")
    def references(self) -> OperationResult:
        refs = {a.key: a for a in self.artifacts}
        if len(refs) != len(self.artifacts):
            raise ValueError("duplicate output artifact key")
        checkpoint_sha = self.model_id.rsplit(":", 1)[1]
        manifest = refs.get(f"qwen/checkpoints/{checkpoint_sha}/manifest.json")
        if manifest is None or manifest.sha256 != checkpoint_sha:
            raise ValueError("result must inventory its content-addressed checkpoint manifest")
        if refs.get(self.worker_manifest.key) != self.worker_manifest:
            raise ValueError("worker manifest absent from artifact inventory")
        for prediction in self.predictions:
            if (
                prediction.confidence is not None
                or prediction.entropy is not None
                or prediction.raw_output_artifact not in refs
            ):
                raise ValueError("prediction needs raw evidence and no acquisition scores")
        if self.reload and any(
            refs.get(a.key) != a for a in (self.reload.before, self.reload.after)
        ):
            raise ValueError("reload evidence absent from artifact inventory")
        return self

    def check_request(self, invocation: Invocation) -> None:
        request = invocation.request
        if self.operation_id != invocation.operation_id or self.attempt_id != invocation.attempt_id:
            raise ValueError("result semantic/submission ownership mismatch")
        if isinstance(request, PredictRequest):
            if self.model_id != request.input_checkpoint or self.reload is not None:
                raise ValueError("prediction checkpoint/result kind mismatch")
            if tuple(p.page_id for p in self.predictions) != tuple(p.id for p in request.pages):
                raise ValueError("predictions must preserve exact requested page coverage/order")
            for prediction in self.predictions:
                if (
                    prediction.experiment_id != request.context.experiment_id
                    or prediction.round_number != request.round_number
                    or prediction.purpose.value != request.purpose
                    or prediction.model_id != self.model_id
                ):
                    raise ValueError("prediction ownership/purpose mismatch")
        elif self.predictions or (isinstance(request, FitRequest) != (self.reload is not None)):
            raise ValueError("only completed fit requires fresh-process reload evidence")


class DispatchResponse(Model):
    """Completion artifact hashes canonical result JSON; no self-referential checksum."""

    result: OperationResult
    completion: ArtifactRef

    @model_validator(mode="after")
    def completion_identity(self) -> DispatchResponse:
        from active_ocr.integrations.qwen import canonical

        r = self.result
        key = (
            f"operations/{r.operation_id}/attempts/{r.attempt_id}/"
            f"executions/{r.execution_id}/complete.json"
        )
        data = canonical(r.model_dump(mode="json"))
        if (
            self.completion.key != key
            or self.completion.bytes != len(data)
            or self.completion.sha256 != digest(r.model_dump(mode="json"))
        ):
            raise ValueError("completion must hash the exact canonical result")
        return self


class FailureReceipt(Model):
    schema_version: Literal[1] = 1
    status: Literal["failed"] = "failed"
    operation_id: SHA256
    attempt_id: Identifier
    execution_id: Identifier
    provider_call_id: str = Field(strict=True, min_length=1)
    code: Literal["invalid_request", "identity", "model", "deadline", "interrupted"]
    # No request, target, raw exception text, traceback or asserted provider-terminal state.


# Local coordinator records. No fit payload or source label object is persisted here.
class RunControl(Model):
    settings_sha256: SHA256
    first_submission_unix_seconds: PositiveInt
    deadline_unix_seconds: PositiveInt
    active_operation: SHA256 | None = None


class ModelOperation(Model):
    operation_id: SHA256
    semantic: dict
    attempt_id: Identifier
    state: Literal[
        "RESERVED", "SUBMITTING", "RUNNING", "UNKNOWN", "CANCEL_REQUESTED", "FAILED", "COMPLETED"
    ]
    provider_call_id: Annotated[str, Field(strict=True, min_length=1)] | None = None
    response: DispatchResponse | None = None
    failure: FailureReceipt | None = None

    @model_validator(mode="after")
    def journal_identity(self):
        if self.operation_id != digest({"schema_version": 1, "request": self.semantic}):
            raise ValueError("journal semantic digest mismatch")
        if (
            self.state in ("RUNNING", "COMPLETED", "CANCEL_REQUESTED", "FAILED")
            and not self.provider_call_id
        ):
            raise ValueError("journal state requires observed call identity")
        if (self.state == "COMPLETED") != (self.response is not None):
            raise ValueError("journal completion/state mismatch")
        return self


class DeploymentObservation(Model):
    settings_sha256: SHA256
    function_id: str = Field(min_length=1, strict=True)
    app_id: str = Field(min_length=1, strict=True)


class SDKTransport:
    """Only this boundary imports Modal. Construction performs no provider operations."""

    def __init__(self, settings: RuntimeSettings, deployment: DeploymentObservation):
        from importlib.metadata import PackageNotFoundError, version

        try:
            installed = version("modal")
        except PackageNotFoundError:
            raise ValueError("install the optional modal==1.5.5 client") from None
        if installed != "1.5.5":
            raise ValueError("install the exact optional modal==1.5.5 client")
        import modal

        if deployment.settings_sha256 != digest(settings.model_dump(mode="json")):
            raise ValueError("deployment observation/settings mismatch")
        self.sdk, self.settings, self.deployment = modal, settings, deployment
        self.function = None
        self.output = None

    def preflight(self):
        s = self.settings
        function = self.sdk.Function.from_name(
            s.deployment_name, s.function_name, environment_name=s.environment_name
        )
        function.hydrate()
        if function.object_id != self.deployment.function_id:
            raise ValueError("deployed Function identity changed")
        volumes = []
        for name, expected in (
            (s.input_volume_name, s.input_volume_id),
            (s.output_volume_name, s.output_volume_id),
        ):
            volume = self.sdk.Volume.from_name(name, environment_name=s.environment_name)
            volume.hydrate()
            if volume.object_id != expected:
                raise ValueError("deployed Volume identity changed")
            volumes.append(volume)
        self.function, self.output = function, volumes[1]

    def spawn(self, payload):
        if self.function is None:
            self.preflight()
        return self.function.spawn(payload).object_id

    def get(self, call_id, timeout):
        return self.sdk.FunctionCall.from_id(call_id).get(timeout=timeout)

    def cancel(self, call_id):
        self.sdk.FunctionCall.from_id(call_id).cancel(terminate_containers=True)

    def read(self, key):
        relative_key(key)
        if self.output is None:
            self.preflight()
        return self.output.read_file(self.settings.output_sub_path + "/" + key)


def remote_page(page) -> RemotePage:
    return RemotePage(
        id=page.id,
        document_id=page.document_id,
        split=page.split,
        width=page.width,
        height=page.height,
        image_sha256=page.image_sha256,
        image_key="images/" + page.image_sha256,
    )


def _probe_values(data):
    import sys
    from array import array

    values = array("f")
    values.frombytes(data)
    if sys.byteorder != "little":
        values.byteswap()
    if any(not math.isfinite(v) for v in values):
        raise ValueError("nonfinite probe logits")
    return values


def verify_reload_evidence(result, paths, request):
    """Compare all fixed-prefix FP32 entries without installing Torch on the coordinator."""
    from active_ocr.integrations.qwen import canonical, strict_json

    evidence = result.reload
    sha = result.model_id.rsplit(":", 1)[1]
    page = min((e.page for e in request.examples), key=lambda p: p.id)
    probes, values = [], []
    for name, ref in (("before", evidence.before), ("after", evidence.after)):
        prefix = f"probes/{sha}/{name}"
        if ref.key != "qwen/" + prefix + ".json":
            raise ValueError("probe metadata namespace mismatch")
        raw = paths[ref.key].read_bytes()
        probe = ProbeMetadata.model_validate(strict_json(raw))
        if raw != canonical(probe.model_dump(mode="json")):
            raise ValueError("noncanonical probe metadata")
        if (
            probe.model_id != result.model_id
            or probe.page_id != page.id
            or probe.image_sha256 != page.image_sha256
            or (probe.original_width, probe.original_height) != (page.width, page.height)
            or probe.logits.key != prefix + ".f32le"
        ):
            raise ValueError("probe page/model/logits namespace mismatch")
        # Only this fixed prefix translates the nested Qwen-root key into a run-root key.
        outer = probe.logits.model_copy(update={"key": "qwen/" + probe.logits.key})
        if outer not in result.artifacts:
            raise ValueError("probe logits missing from artifact inventory")
        probes.append(probe)
        values.append(_probe_values(paths[outer.key].read_bytes()))
    before, after = probes
    if (before.process_id, after.process_id) != (
        evidence.train_process_id,
        evidence.reload_process_id,
    ):
        raise ValueError("probe process identity mismatch")
    for key in (
        "generated_ids",
        "status",
        "regions",
        "finish_reason",
        "adapter_tensors",
        "logits_shape",
    ):
        if getattr(before, key) != getattr(after, key):
            raise ValueError("fresh reload exact evidence mismatch")
    maximum = 0.0
    for a, b in zip(*values, strict=True):
        delta = abs(a - b)
        if delta > 1e-2 + 1e-3 * abs(b):
            raise ValueError("fresh reload logits exceed fixed tolerance")
        maximum = max(maximum, delta)
    if not math.isclose(maximum, evidence.max_absolute_difference, rel_tol=1e-6, abs_tol=1e-7):
        raise ValueError("reported probe difference mismatch")
    return before.adapter_tensors


def _verify_adapter_tensors(path, hashes):
    """Read safe tensor framing and FP32 bytes; never deserialize executable model objects."""
    import hashlib
    import struct

    from active_ocr.integrations.qwen import canonical, strict_json

    raw = path.read_bytes()
    if len(raw) < 8:
        raise ValueError("truncated adapter")
    size = struct.unpack("<Q", raw[:8])[0]
    if not 2 <= size <= 1024 * 1024 or size + 8 > len(raw):
        raise ValueError("invalid adapter header")
    header = strict_json(raw[8 : 8 + size])
    header.pop("__metadata__", None)
    shapes = adapter_shapes()
    if set(header) != set(shapes):
        raise ValueError("adapter tensor inventory mismatch")
    intervals = []
    for name, shape in shapes.items():
        entry = header[name]
        start, end = entry["data_offsets"]
        if (
            entry["dtype"] != "F32"
            or entry["shape"] != list(shape)
            or type(start) is not int
            or type(end) is not int
            or start < 0
            or end - start != math.prod(shape) * 4
            or end > len(raw) - size - 8
        ):
            raise ValueError("invalid adapter tensor framing")
        data = raw[8 + size + start : 8 + size + end]
        _probe_values(data)
        actual = hashlib.sha256(
            canonical(dict(shape=list(shape), dtype="torch.float32")) + data
        ).hexdigest()
        if actual != hashes[name]:
            raise ValueError("adapter tensor differs from reload probe")
        intervals.append((start, end))
    cursor = 0
    for start, end in sorted(intervals):
        if start != cursor:
            raise ValueError("adapter tensor overlap/gap")
        cursor = end
    if cursor != len(raw) - size - 8:
        raise ValueError("unreferenced adapter bytes")


class ModalModel:
    """Synchronous adapter with durable reservation, known-call recovery and verified reuse."""

    from active_ocr.models import RunKind

    kind = RunKind.REAL
    backend = "qwen3-vl-v1"
    fit_policy = "reset-fit-cumulative-v1"

    def __init__(self, settings: RuntimeSettings, store, transport, *, clock=None):
        import time

        self.settings = RuntimeSettings.model_validate(settings.model_dump())
        self.real_config = self.settings.context.real_config
        self.identity = self.real_config.expected_identity
        self.store, self.transport = store, transport
        self.clock = time.time if clock is None else clock
        self.telemetry = None

    def check_run(self, run, store):
        context = self.settings.context
        if (
            run.id != context.experiment_id
            or run.config.real != context.real_config
            or run.config.source_policy != context.source_policy
            or store.database_path.absolute() != self.store.database_path.absolute()
        ):
            raise ValueError("Modal adapter does not own this frozen run/store")
        bundle = {f.filename: f for f in self.settings.bundle.files}
        expected_images = {"images/" + p.image_sha256 for p in run.dataset.pages}
        if {n for n in bundle if n.startswith("images/")} != expected_images:
            raise ValueError("input bundle must contain exactly the frozen source images")
        from pathlib import Path

        for page in run.dataset.pages:
            entry = bundle["images/" + page.image_sha256]
            if (
                entry.sha256 != page.image_sha256
                or entry.bytes != Path(page.image_uri).stat().st_size
            ):
                raise ValueError("input bundle image identity mismatch")
        control = self.store.load("model-control", run.id, RunControl)
        if control is not None and control.settings_sha256 != digest(
            self.settings.model_dump(mode="json")
        ):
            raise ValueError("frozen Modal runtime settings changed")

    def _check_input_checkpoint(self, request):
        from active_ocr.integrations import qwen as q

        if isinstance(request, BaseRequest):
            return
        model = object.__new__(q.QwenModel)
        model.real_config = self.real_config
        base_id = "checkpoint:sha256:" + digest(
            q.Checkpoint(kind="base", binding=model._binding()).model_dump(mode="json")
        )
        if isinstance(request, FitRequest) or request.round_number == 0:
            if request.input_checkpoint != base_id:
                raise ValueError("operation requires this recipe's pinned base checkpoint")
        elif not any(
            r.state == "COMPLETED"
            and r.semantic["operation"] == "fit"
            and r.semantic["context"] == request.context.model_dump(mode="json")
            and r.semantic["round_number"] == request.round_number
            and r.response.result.model_id == request.input_checkpoint
            for r in self.store.list("model-operation", ModelOperation)
        ):
            raise ValueError("prediction requires a previously verified completed fit")

    def _control(self):
        key = self.settings.context.experiment_id
        expected = digest(self.settings.model_dump(mode="json"))
        control = self.store.load("model-control", key, RunControl)
        if control is None:
            now = max(1, int(self.clock()))
            value = RunControl(
                settings_sha256=expected,
                first_submission_unix_seconds=now,
                deadline_unix_seconds=now + self.settings.aggregate_gpu_seconds,
            )
            self.store.compare_and_swap("model-control", key, None, value)
            control = value
        if (
            control.settings_sha256 != expected
            or control.deadline_unix_seconds
            != control.first_submission_unix_seconds + self.settings.aggregate_gpu_seconds
            or self.clock() < control.first_submission_unix_seconds
        ):
            raise ValueError("frozen Modal runtime settings changed")
        return control

    def _change(self, record, **changes):
        value = ModelOperation.model_validate({**record.model_dump(), **changes})
        self.store.compare_and_swap("model-operation", record.operation_id, record, value)
        return value

    def _invocation(self, request, record, control):
        invocation = Invocation(
            request=request,
            attempt_id=record.attempt_id,
            first_submission_unix_seconds=control.first_submission_unix_seconds,
            deadline_unix_seconds=control.deadline_unix_seconds,
        )
        self.settings.check_invocation(invocation)
        return invocation

    def _unlock(self, operation_id):
        key = self.settings.context.experiment_id
        control = self.store.load("model-control", key, RunControl)
        if control.active_operation == operation_id:
            self.store.compare_and_swap(
                "model-control", key, control, control.model_copy(update={"active_operation": None})
            )

    def execute(self, request):
        from uuid import uuid4

        # Revalidate before reservation/provider calls, rejecting model_copy bypasses.
        request = REQUEST_ADAPTER.validate_python(request.model_dump())
        self._check_input_checkpoint(request)
        op_id = operation_id(request)
        control = self._control()
        record = self.store.load("model-operation", op_id, ModelOperation)
        if record is None:
            record = ModelOperation(
                operation_id=op_id,
                semantic=request.semantic_record(),
                attempt_id=uuid4().hex,
                state="RESERVED",
            )
            self.settings.check_invocation(self._invocation(request, record, control))
            self.store.compare_and_swap("model-operation", op_id, None, record)
        if record.semantic != request.semantic_record():
            raise ValueError("operation semantic identity changed")
        invocation = self._invocation(request, record, control)
        if record.state == "COMPLETED":
            result = self.verify_response(record.response, invocation, record.provider_call_id)
            self._unlock(op_id)
            self.telemetry = result.telemetry
            return result
        if control.active_operation not in (None, op_id):
            raise RuntimeError("another operation is unresolved; reconcile it first")
        if control.active_operation is None:
            updated = control.model_copy(update={"active_operation": op_id})
            self.store.compare_and_swap(
                "model-control", request.context.experiment_id, control, updated
            )
            control = updated
        if record.state == "RESERVED":
            if self.clock() >= control.deadline_unix_seconds:
                raise TimeoutError("absolute run deadline reached before submission")
            # Preflight fails safely before SUBMITTING; no spawn has happened yet.
            self.transport.preflight()
            payload = invocation.model_dump(mode="json")
            if self.clock() >= control.deadline_unix_seconds:
                raise TimeoutError("absolute run deadline reached before submission")
            record = self._change(record, state="SUBMITTING")
            # The journal write can also wait. An expired, unsubmitted claim is safe to release.
            if self.clock() >= control.deadline_unix_seconds:
                self._change(record, state="RESERVED")
                raise TimeoutError("absolute run deadline reached before submission")
            try:
                call_id = self.transport.spawn(payload)
                if not isinstance(call_id, str) or not call_id:
                    raise ValueError("missing provider call identity")
                record = self._change(record, state="RUNNING", provider_call_id=call_id)
            except BaseException:
                # A response or the subsequent durable write may have been lost. Never respawn.
                current = self.store.load("model-operation", op_id, ModelOperation)
                if current.state == "SUBMITTING":
                    self._change(current, state="UNKNOWN")
                raise RuntimeError(
                    "submission outcome unknown; explicit reconciliation required"
                ) from None
        elif record.state == "SUBMITTING":
            self._change(record, state="UNKNOWN")
            raise RuntimeError("interrupted submission; explicit reconciliation required")
        if record.state != "RUNNING":
            raise RuntimeError(f"operation is {record.state}; explicit reconciliation required")
        return self._wait(record, invocation, control)

    def _wait(self, record, invocation, control):
        while True:
            remaining = control.deadline_unix_seconds - self.clock()
            if remaining <= 0:
                self.cancel(record.operation_id)
                raise TimeoutError(
                    "deadline reached; cancellation requested, terminal status unverified"
                )
            try:
                response = self.transport.get(record.provider_call_id, timeout=min(30, remaining))
            except TimeoutError:
                continue
            except BaseException:
                self._change(record, state="UNKNOWN")
                raise RuntimeError(
                    "provider outcome unresolved; known call retained for reconciliation"
                ) from None
            if isinstance(response, dict) and response.get("status") == "failed":
                failure = FailureReceipt.model_validate(response)
                if (
                    failure.operation_id != record.operation_id
                    or failure.attempt_id != record.attempt_id
                    or failure.provider_call_id != record.provider_call_id
                ):
                    raise ValueError("failure receipt ownership mismatch")
                self._change(record, state="FAILED", failure=failure)
                # Keep the run locked: failure requires a reviewed decision, not automatic retries.
                raise RuntimeError("worker failed; partial evidence preserved; no automatic retry")
            try:
                parsed = DispatchResponse.model_validate(response)
                result = self.verify_response(parsed, invocation, record.provider_call_id)
            except BaseException:
                self._change(record, state="UNKNOWN")
                raise
            self._change(record, state="COMPLETED", response=parsed)
            self._unlock(record.operation_id)
            self.telemetry = result.telemetry
            return result

    def cancel(self, operation_id):
        record = self.store.load("model-operation", operation_id, ModelOperation)
        if record is None or record.provider_call_id is None:
            raise ValueError("cannot cancel without a known provider call")
        if record.state == "COMPLETED":
            raise ValueError("operation is already completed")
        record = self._change(record, state="CANCEL_REQUESTED")
        self.transport.cancel(record.provider_call_id)
        # The acknowledgement is not terminal evidence. Reconciliation must read a real result.
        return record

    def reconcile(self, request, *, call_id=None, response=None):
        """Explicitly attach an observed call or verify preserved completion; never resubmit."""
        request = REQUEST_ADAPTER.validate_python(request.model_dump())
        record = self.store.load("model-operation", operation_id(request), ModelOperation)
        if record is None or record.semantic != request.semantic_record():
            raise ValueError("no matching reserved operation")
        if record.state == "COMPLETED":
            return self.execute(request)
        if call_id is not None:
            if record.provider_call_id not in (None, call_id):
                raise ValueError("cannot replace recorded provider call identity")
            record = self._change(record, provider_call_id=call_id)
        if not record.provider_call_id:
            raise ValueError("reconciliation needs an observed provider call ID")
        control = self._control()
        invocation = self._invocation(request, record, control)
        if response is None:
            # Poll, including after the deadline. Never wait indefinitely or launch new work.
            response = self.transport.get(record.provider_call_id, timeout=0)
        parsed = DispatchResponse.model_validate(response)
        result = self.verify_response(parsed, invocation, record.provider_call_id)
        self._change(record, state="COMPLETED", response=parsed)
        self._unlock(record.operation_id)
        return result

    def _artifact(self, ref):
        import hashlib
        import os
        import tempfile
        from pathlib import Path

        from active_ocr.integrations.qwen import file_hash, safe_path

        root = self.store.artifact_root / "model-evidence"
        root.mkdir(parents=True, exist_ok=True)
        target = safe_path(root, ref.sha256)
        if target.exists() and (
            target.stat().st_size != ref.bytes or file_hash(target) != ref.sha256
        ):
            raise ValueError("local model evidence corrupted")
        sha, size = hashlib.sha256(), 0
        with tempfile.NamedTemporaryFile(dir=root, delete=False) as stream:
            temporary = Path(stream.name)
            try:
                for chunk in self.transport.read(ref.key):
                    size += len(chunk)
                    if size > ref.bytes:
                        raise ValueError("remote artifact exceeds declared size")
                    sha.update(chunk)
                    stream.write(chunk)
                stream.flush()
                os.fsync(stream.fileno())
                if size != ref.bytes or sha.hexdigest() != ref.sha256:
                    raise ValueError("remote artifact bytes/digest mismatch")
                os.link(temporary, target)
            except FileExistsError:
                if target.stat().st_size != ref.bytes or file_hash(target) != ref.sha256:
                    raise ValueError("concurrent evidence collision") from None
            finally:
                temporary.unlink(missing_ok=True)
        return target

    def verify_response(self, response, invocation, provider_call_id):
        from active_ocr.integrations import qwen as q

        response = DispatchResponse.model_validate(response.model_dump())
        result, request = response.result, invocation.request
        result.check_request(invocation)
        if result.provider_call_id != provider_call_id:
            raise ValueError("provider call identity mismatch")
        complete = self._artifact(response.completion).read_bytes()
        if complete != q.canonical(result.model_dump(mode="json")):
            raise ValueError("completion content mismatch")
        paths = {ref.key: self._artifact(ref) for ref in result.artifacts}
        manifest_raw = paths[result.worker_manifest.key].read_bytes()
        manifest = q.WorkerManifest.model_validate(q.strict_json(manifest_raw))
        if (
            manifest_raw != q.canonical(manifest.model_dump(mode="json"))
            or manifest.source_sha != self.settings.build.source_sha
            or manifest.files != self.settings.build.code_files
            or manifest.environment != self.settings.build.environment
            or manifest.build_spec_sha256 != self.identity.build_spec_sha256
            or manifest.deployment_reference != self.identity.deployment_reference
        ):
            raise ValueError("observed worker identity mismatch")
        sha = result.model_id.rsplit(":", 1)[1]
        prefix = f"qwen/checkpoints/{sha}/"
        raw = paths[prefix + "manifest.json"].read_bytes()
        checkpoint = q.Checkpoint.model_validate(q.strict_json(raw))
        model = object.__new__(q.QwenModel)
        model.real_config = self.real_config
        if (
            raw != q.canonical(checkpoint.model_dump(mode="json"))
            or checkpoint.binding != model._binding()
        ):
            raise ValueError("checkpoint recipe mismatch")
        refs = {ref.key: ref for ref in result.artifacts}
        required = {prefix + "manifest.json"}
        for f in checkpoint.files:
            key = prefix + f.filename
            if refs.get(key) != ArtifactRef(key=key, bytes=f.bytes, sha256=f.sha256):
                raise ValueError("checkpoint file closure mismatch")
            required.add(key)
        if {key for key in refs if key.startswith(prefix)} != required:
            raise ValueError("extra checkpoint artifacts")
        if checkpoint.kind == "base":
            if (
                checkpoint != q.Checkpoint(kind="base", binding=model._binding())
                or isinstance(request, FitRequest)
                or request.round_number
            ):
                raise ValueError("unexpected base checkpoint")
        else:
            if (
                checkpoint.experiment_id != request.context.experiment_id
                or checkpoint.round_number != request.round_number
                or [f.filename for f in checkpoint.files]
                != ["adapter_config.json", "adapter_model.safetensors"]
            ):
                raise ValueError("checkpoint ownership mismatch")
            q.check_training_record(checkpoint)
        if isinstance(request, FitRequest):
            # Serialize only the selected TRAIN view; no oracle/source access at this boundary.
            if (
                checkpoint.selected
                != tuple((e.page.id, e.page.image_sha256) for e in request.examples)
                or checkpoint.seed != request.seed
                or checkpoint.target_sha256
                != digest([(e.page.id, q.serialize_target(e)) for e in request.examples])
            ):
                raise ValueError("fit checkpoint selected/seed/target mismatch")
            hashes = verify_reload_evidence(result, paths, request)
            _verify_adapter_tensors(paths[prefix + "adapter_model.safetensors"], hashes)
        if isinstance(request, PredictRequest):
            for prediction, page in zip(result.predictions, request.pages, strict=True):
                if prediction.status.value != "ok" and prediction.regions:
                    raise ValueError("failed prediction has regions")
                for region in prediction.regions:
                    b = region.box
                    if (
                        not all(math.isfinite(v) for v in (b.x, b.y, b.width, b.height))
                        or b.x + b.width > page.width
                        or b.y + b.height > page.height
                    ):
                        raise ValueError("prediction geometry outside original page")
                raw = paths[prediction.raw_output_artifact].read_bytes()
                receipt = q.strict_json(raw)
                if (
                    raw != q.canonical(receipt)
                    or prediction.raw_output_artifact
                    != "qwen/receipts/" + q.digest(receipt) + ".json"
                    or len({r.id for r in prediction.regions}) != len(prediction.regions)
                ):
                    raise ValueError("raw prediction canonical identity/region IDs mismatch")
                if (
                    receipt.get("operation") != "predict"
                    or receipt.get("experiment_id") != request.context.experiment_id
                    or receipt.get("round_number") != request.round_number
                    or receipt.get("purpose") != request.purpose
                    or receipt.get("model_id") != result.model_id
                    or [p["page_id"] for p in receipt.get("pages", [])]
                    != [p.id for p in request.pages]
                ):
                    raise ValueError("raw prediction evidence ownership mismatch")
                evidence = next(p for p in receipt["pages"] if p["page_id"] == page.id)
                if (
                    evidence["image_sha256"] != page.image_sha256
                    or evidence["status"] != prediction.status.value
                    or evidence["finish_reason"] != prediction.finish_reason
                ):
                    raise ValueError("raw prediction evidence result mismatch")
                if (
                    prediction.status.value == "ok"
                    and q.parse_regions(evidence["text"], page.width, page.height)
                    != prediction.regions
                ):
                    raise ValueError("raw text and parsed prediction regions disagree")
        return result

    def load_base(self, *, experiment_id):
        if experiment_id != self.settings.context.experiment_id:
            raise ValueError("wrong run")
        return self.execute(BaseRequest(context=self.settings.context)).model_id

    def fit(self, examples, *, seed, experiment_id, round_number):
        if experiment_id != self.settings.context.experiment_id:
            raise ValueError("wrong run")
        base = self.load_base(experiment_id=experiment_id)
        return self.execute(
            FitRequest(
                context=self.settings.context,
                round_number=round_number,
                input_checkpoint=base,
                seed=seed,
                examples=tuple(
                    RemoteExample(page=remote_page(e.page), regions=e.regions) for e in examples
                ),
            )
        ).model_id

    def predict(self, pages, *, experiment_id, round_number, model_id, purpose):
        if experiment_id != self.settings.context.experiment_id:
            raise ValueError("wrong run")
        return self.execute(
            PredictRequest(
                context=self.settings.context,
                round_number=round_number,
                input_checkpoint=model_id,
                purpose=purpose,
                pages=tuple(remote_page(p) for p in pages),
            )
        ).predictions


def input_upload_files(bundle: InputBundle, pages, model_root):
    """Explicit verified image/model allowlist. No directory traversal or source/oracle copying."""
    from pathlib import Path

    from active_ocr.integrations.qwen import file_hash, safe_path

    bundle = InputBundle.model_validate(bundle.model_dump())
    images = {}
    for page in pages:
        path = Path(page.image_uri)
        path = safe_path(path.parent, path.absolute())
        if file_hash(path) != page.image_sha256:
            raise ValueError("frozen image bytes changed")
        images["images/" + page.image_sha256] = path
    entries = {f.filename: f for f in bundle.files}
    if set(images) != {key for key in entries if key.startswith("images/")}:
        raise ValueError("bundle must contain exactly the frozen images")
    files = []
    for entry in bundle.files:
        path = (
            images[entry.filename]
            if entry.filename.startswith("images/")
            else safe_path(model_root, entry.filename.removeprefix("model/"))
        )
        if path.stat().st_size != entry.bytes or file_hash(path) != entry.sha256:
            raise ValueError("input upload bytes differ from reviewed bundle")
        files.append((path, entry.filename))
    return tuple(files)


def upload_input_bundle(volume, bundle: InputBundle, pages, model_root):
    """Explicit provider mutation for a later authorized bootstrap; never invoked by preflight."""
    files = input_upload_files(bundle, pages, model_root)
    with volume.batch_upload(force=False) as batch:
        for path, key in files:
            batch.put_file(path, "/bundles/" + bundle.sha256 + "/" + key)
