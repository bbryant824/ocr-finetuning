"""Candidate-3 shared wire contract. No Modal calls or transport implementation yet.

Only fit carries selected labels. Journal semantic_record(), never the wire fit payload.
Platform owns the one dispatcher; see docs/modal-runtime.md for bootstrap and ownership.
"""

from __future__ import annotations

import math
from pathlib import PurePosixPath
from typing import Annotated, Literal

from pydantic import Field, TypeAdapter, field_validator, model_validator

from active_ocr.integrations.qwen import (
    ASSETS,
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
        min_length=1, max_length=2048
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
