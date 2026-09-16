"""Small set of data structures shared by the research pipeline."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Model(BaseModel):
    """Validated, immutable base model with JSON serialization."""

    model_config = ConfigDict(extra="forbid", frozen=True)


SHA256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class Split(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class Strategy(StrEnum):
    RANDOM = "random"
    LEAST_CONFIDENCE = "least_confidence"
    ENTROPY = "entropy"


class Stage(StrEnum):
    READY = "ready"
    ANNOTATING = "annotating"
    TRAINING = "training"
    SCORING = "scoring"
    COMPLETE = "complete"
    ERROR = "error"


class Box(Model):
    """Axis-aligned line box in image pixels."""

    x: float = Field(ge=0)
    y: float = Field(ge=0)
    width: float = Field(gt=0)
    height: float = Field(gt=0)


class Region(Model):
    """One detected or annotated text line."""

    id: str = Field(min_length=1)
    box: Box
    text: str = ""
    illegible: bool = False

    @model_validator(mode="after")
    def normalize_illegible_text(self) -> Region:
        if self.illegible and not self.text:
            return self.model_copy(update={"text": "[ILLEGIBLE]"})
        return self


class Page(Model):
    """Rendered page image, which is the active-learning query unit."""

    id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    image_uri: str = Field(min_length=1)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    split: Split = Split.TRAIN


class Annotation(Model):
    """Validated human transcription and boxes for one page."""

    page_id: str = Field(min_length=1)
    regions: tuple[Region, ...]
    seconds: float | None = Field(default=None, ge=0)


class PredictionPurpose(StrEnum):
    POOL = "pool"
    VALIDATION = "validation"
    BASELINE_VALIDATION = "baseline_validation"


class PredictionStatus(StrEnum):
    OK = "ok"
    INVALID_OUTPUT = "invalid_output"
    TRUNCATED = "truncated"
    REFUSAL = "refusal"


class Prediction(Model):
    """OCR output plus page-level values used by query strategies."""

    page_id: str = Field(min_length=1)
    experiment_id: str = Field(min_length=1)
    round_number: int = Field(ge=0)
    model_id: str = Field(min_length=1)
    regions: tuple[Region, ...] = ()
    confidence: float | None = Field(default=None, ge=0, le=1)
    entropy: float | None = Field(default=None, ge=0, le=1)

    purpose: PredictionPurpose = PredictionPurpose.POOL
    status: PredictionStatus = PredictionStatus.OK
    raw_output_artifact: str | None = Field(default=None, min_length=1)
    finish_reason: str | None = None

    @model_validator(mode="after")
    def validate_purpose(self) -> Prediction:
        if (self.round_number == 0) != (self.purpose is PredictionPurpose.BASELINE_VALIDATION):
            raise ValueError("round zero is reserved for baseline_validation purpose")
        return self

    @property
    def storage_key(self) -> str:
        return f"{self.experiment_id}:{self.round_number}:{self.page_id}"


class Experiment(Model):
    """State for one reproducible strategy run."""

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    strategy: Strategy
    batch_size: int = Field(gt=0)
    max_rounds: int = Field(gt=0)
    seed: int
    round_number: int = Field(default=0, ge=0)
    stage: Stage = Stage.READY
    labelled_page_ids: tuple[str, ...] = ()
    current_batch: tuple[str, ...] = ()
    external_batch_id: str | None = None
    job_id: str | None = None
    model_id: str | None = None
    error: str | None = None

    @model_validator(mode="after")
    def validate_state(self) -> Experiment:
        if self.round_number > self.max_rounds:
            raise ValueError("round_number cannot exceed max_rounds")
        if len(set(self.labelled_page_ids)) != len(self.labelled_page_ids):
            raise ValueError("labelled page IDs must be unique")
        if len(set(self.current_batch)) != len(self.current_batch):
            raise ValueError("current batch page IDs must be unique")
        if self.stage is Stage.ERROR and not self.error:
            raise ValueError("an error stage requires an error message")
        return self


class SourceRegion(Model):
    """Oracle ground truth, preserved verbatim (including empty illegible text)."""

    id: str = Field(min_length=1)
    box: Box
    text: str
    illegible: bool = False


class SourcePolicy(StrEnum):
    KNOWN_DOCUMENT = "known-document-v1"
    READ2016 = "read2016-official-unknown-engineering-v1"


class SourceArtifact(Model):
    uri: str = Field(min_length=1)
    sha256: SHA256


def validate_source_policy(document_id: str | None, split: Split, policy: SourcePolicy) -> None:
    if policy is SourcePolicy.KNOWN_DOCUMENT:
        if not document_id:
            raise ValueError("known-document policy requires a document ID")
    elif document_id is not None or split is Split.TEST:
        raise ValueError("READ engineering policy requires null documents and excludes TEST")


class SimulationPage(Page):
    """Frozen image metadata; no labels are exposed to model prediction."""

    document_id: Annotated[str, Field(min_length=1)] | None
    source_policy: SourcePolicy = SourcePolicy.KNOWN_DOCUMENT
    image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_image: str

    @model_validator(mode="after")
    def validate_policy(self) -> SimulationPage:
        validate_source_policy(self.document_id, self.split, self.source_policy)
        return self


class RevealedExample(Model):
    page: SimulationPage
    regions: tuple[SourceRegion, ...]
    seconds: None = None


class DatasetSnapshot(Model):
    source_manifest: str
    frozen_manifest: str
    manifest_sha256: str
    ground_truth_sha256: str
    pages: tuple[SimulationPage, ...]
    source_policy: SourcePolicy = SourcePolicy.KNOWN_DOCUMENT
    source_provenance: SourceArtifact | None = None

    @model_validator(mode="after")
    def validate_policy(self) -> DatasetSnapshot:
        if any(page.source_policy is not self.source_policy for page in self.pages):
            raise ValueError("snapshot/page source policy mismatch")
        if (self.source_policy is SourcePolicy.READ2016) != (self.source_provenance is not None):
            raise ValueError("source provenance must be present exactly for READ policy")
        return self


CommitSHA = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
CheckpointReference = Annotated[str, Field(pattern=r"^checkpoint:sha256:[0-9a-f]{64}$")]


class RunKind(StrEnum):
    FIXTURE = "fixture-simulation-not-ocr-evidence"
    CONTRACT_TEST = "adapter-contract-test-not-ocr-evidence"
    REAL = "real-ocr-simulated-annotation-v1"


class ExpectedIdentity(Model):
    """Immutable expectations, known without loading a model or discovering a GPU."""

    source_sha: CommitSHA
    dependency_sha256: SHA256
    model_revision: CommitSHA
    model_manifest_sha256: SHA256
    processor_revision: CommitSHA
    processor_manifest_sha256: SHA256
    recipe_version: str = Field(min_length=1)
    evaluator_id: str = Field(min_length=1)
    code_bundle_sha256: SHA256 | None = None
    remote_dependency_sha256: SHA256 | None = None
    build_spec_sha256: SHA256 | None = None
    deployment_reference: str | None = Field(default=None, min_length=1)


class RealOCRConfig(Model):
    """Represent one pinned recipe; candidate 1A supplies no production real adapter."""

    backend: str = Field(min_length=1)
    recipe_version: str = Field(min_length=1)
    model_repository: str = Field(min_length=1)
    model_revision: CommitSHA
    processor_repository: str = Field(min_length=1)
    processor_revision: CommitSHA
    training_policy_id: str = Field(min_length=1)
    decode_policy_id: str = Field(min_length=1)
    evaluator_id: Literal["page-text-nfc-v1"] = "page-text-nfc-v1"
    expected_identity: ExpectedIdentity

    @model_validator(mode="after")
    def validate_recipe(self) -> RealOCRConfig:
        if self.backend == "deterministic-fixture-v1":
            raise ValueError("a real recipe cannot use the fixture backend")
        for field in ("model_revision", "processor_revision", "recipe_version", "evaluator_id"):
            if getattr(self, field) != getattr(self.expected_identity, field):
                raise ValueError(f"recipe and expected identity disagree: {field}")
        return self


class ExecutionTelemetry(Model):
    """Optional observations, never identity expectations or creation prerequisites."""

    device: str | None = None
    cuda_version: str | None = None
    driver_version: str | None = None
    elapsed_seconds: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    free_memory_bytes: int | None = Field(default=None, ge=0)
    peak_memory_bytes: int | None = Field(default=None, ge=0)
    call_id: str | None = None
    billed_cost: float | None = Field(default=None, ge=0, allow_inf_nan=False)


class SimulationConfig(Model):
    source_policy: SourcePolicy = SourcePolicy.KNOWN_DOCUMENT
    strategy: Strategy = Strategy.RANDOM
    batch_size: int = Field(default=2, gt=0)
    page_budget: int = Field(default=6, ge=0)
    max_rounds: int = Field(default=3, ge=0)
    seed: int = 824
    report_predictions: bool = False
    backend: str = "deterministic-fixture-v1"
    fit_policy: str = "reset-fit-cumulative-v1"
    evaluator_id: str | None = Field(default=None, min_length=1)
    real: RealOCRConfig | None = None

    @model_validator(mode="after")
    def validate_real_recipe(self) -> SimulationConfig:
        if self.real is not None:
            if self.backend != self.real.backend or self.evaluator_id != self.real.evaluator_id:
                raise ValueError("run backend/evaluator must match real recipe")
            if self.strategy is not Strategy.RANDOM or self.report_predictions:
                raise ValueError("real contracts currently support random without pool reporting")
            if self.fit_policy != "reset-fit-cumulative-v1":
                raise ValueError("real contracts require reset-fit-cumulative-v1")
        return self


class SimulationBaseline(Model):
    model_id: CheckpointReference
    labelled_count: Literal[0] = 0
    validation_predictions: tuple[Prediction, ...]
    validation_metrics: dict[str, int | float]
    telemetry: ExecutionTelemetry | None = None


class SimulationRound(Model):
    number: int = Field(ge=1)
    selected_ids: tuple[str, ...]
    revealed_ids: tuple[str, ...]
    labelled_count: int
    remaining_count: int
    model_id: str
    predictions: tuple[Prediction, ...]
    validation_predictions: tuple[Prediction, ...] = ()
    validation_metrics: dict[str, int | float] | None = None
    annotation_seconds: None = None
    telemetry: ExecutionTelemetry | None = None


class SimulationRun(Model):
    """One atomic SQLite record contains frozen inputs and all committed rounds."""

    id: str
    kind: RunKind = RunKind.FIXTURE
    config: SimulationConfig
    code_revision: str
    software_versions: tuple[tuple[str, str], ...]
    dataset: DatasetSnapshot
    baseline: SimulationBaseline | None = None
    rounds: tuple[SimulationRound, ...] = ()
    complete: bool = False
    stop_reason: str | None = None

    @model_validator(mode="after")
    def validate_kind(self) -> SimulationRun:
        if self.config.source_policy is not self.dataset.source_policy:
            raise ValueError("config/snapshot source policy mismatch")
        if self.kind is RunKind.FIXTURE:
            if self.config.real is not None or self.config.backend != "deterministic-fixture-v1":
                raise ValueError("fixture kind requires the fixture backend and no real recipe")
            if self.baseline is not None:
                raise ValueError("fixture runs do not have a real-contract baseline")
        elif self.config.real is None:
            raise ValueError("real/contract-test kind requires an explicit real recipe")
        return self
