"""Small set of data structures shared by the research pipeline."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Model(BaseModel):
    """Validated, immutable base model with JSON serialization."""

    model_config = ConfigDict(extra="forbid", frozen=True)


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


class Prediction(Model):
    """OCR output plus page-level values used by query strategies."""

    page_id: str = Field(min_length=1)
    experiment_id: str = Field(min_length=1)
    round_number: int = Field(ge=1)
    model_id: str = Field(min_length=1)
    regions: tuple[Region, ...] = ()
    confidence: float | None = Field(default=None, ge=0, le=1)
    entropy: float | None = Field(default=None, ge=0, le=1)

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


class SimulationPage(Page):
    """Frozen image metadata; no labels are exposed to model prediction."""

    image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_image: str


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


class SimulationConfig(Model):
    strategy: Strategy = Strategy.RANDOM
    batch_size: int = Field(default=2, gt=0)
    page_budget: int = Field(default=6, ge=0)
    max_rounds: int = Field(default=3, ge=0)
    seed: int = 824
    report_predictions: bool = False
    backend: str = "deterministic-fixture-v1"
    fit_policy: str = "reset-fit-cumulative-v1"
    evaluator_id: str | None = Field(default=None, min_length=1)


class SimulationRound(Model):
    number: int
    selected_ids: tuple[str, ...]
    revealed_ids: tuple[str, ...]
    labelled_count: int
    remaining_count: int
    model_id: str
    predictions: tuple[Prediction, ...]
    validation_predictions: tuple[Prediction, ...] = ()
    validation_metrics: dict[str, float] | None = None
    annotation_seconds: None = None


class SimulationRun(Model):
    """One atomic SQLite record contains frozen inputs and all committed rounds."""

    id: str
    kind: str = "fixture-simulation-not-ocr-evidence"
    config: SimulationConfig
    code_revision: str
    software_versions: tuple[tuple[str, str], ...]
    dataset: DatasetSnapshot
    rounds: tuple[SimulationRound, ...] = ()
    complete: bool = False
    stop_reason: str | None = None
