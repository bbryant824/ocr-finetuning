"""Local true-label oracle and deterministic fixture model. No OCR quality claims."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import re
import subprocess
from importlib.metadata import distributions, version
from pathlib import Path
from typing import Annotated, Protocol

from PIL import Image, ImageDraw
from pydantic import Field, model_validator

from active_ocr.evaluation import page_text_metrics
from active_ocr.integrations.storage import SQLiteStore
from active_ocr.models import (
    DatasetSnapshot,
    ExpectedIdentity,
    Page,
    Prediction,
    PredictionPurpose,
    RealOCRConfig,
    RevealedExample,
    RunKind,
    SimulationPage,
    SourceArtifact,
    SourcePolicy,
    SourceRegion,
    Split,
    validate_source_policy,
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class SourcePage(Page):
    """One JSONL row; image_uri is a local path relative to the manifest."""

    document_id: Annotated[str, Field(min_length=1)] | None
    source_policy: SourcePolicy = SourcePolicy.KNOWN_DOCUMENT
    source_provenance: SourceArtifact | None = None
    split: Split  # Explicit; never infer or reassign a source split.
    image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    regions: tuple[SourceRegion, ...] | None = None

    @model_validator(mode="after")
    def validate_policy(self) -> SourcePage:
        validate_source_policy(self.document_id, self.split, self.source_policy)
        if (self.source_policy is SourcePolicy.READ2016) != (self.source_provenance is not None):
            raise ValueError("source provenance must be present exactly for READ policy")
        return self


def _read_source(
    data: bytes, directory: Path, *, source_policy: SourcePolicy = SourcePolicy.KNOWN_DOCUMENT
) -> tuple[SourcePage, ...]:
    from active_ocr.integrations.public_dataset import (
        source_path,
        validate_image,
        validate_provenance,
    )

    source_policy = SourcePolicy(source_policy)
    pages = tuple(
        SourcePage.model_validate_json(line) for line in data.splitlines() if line.strip()
    )
    if any(p.source_policy is not source_policy for p in pages):
        raise ValueError("caller/row source policy mismatch")
    if source_policy is SourcePolicy.READ2016:
        if not pages or any(p.source_provenance != pages[0].source_provenance for p in pages):
            raise ValueError("READ requires one consistent provenance reference")
        validate_provenance(directory, pages[0].source_provenance, pages)
    ids: set[str] = set()
    documents: dict[str, Split] = {}
    images: dict[str, Split] = {}
    for page in pages:
        if page.id in ids:
            raise ValueError(f"duplicate page ID: {page.id}")
        ids.add(page.id)
        if page.document_id is not None:
            if page.document_id in documents and documents[page.document_id] != page.split:
                raise ValueError("document or duplicate image content crosses splits")
            documents[page.document_id] = page.split
        if page.image_sha256 in images and (
            images[page.image_sha256] != page.split or source_policy is SourcePolicy.READ2016
        ):
            raise ValueError("document or duplicate image content crosses splits or READ pages")
        images[page.image_sha256] = page.split
        path = (
            source_path(directory, page.image_uri)
            if source_policy is SourcePolicy.READ2016
            else (directory / page.image_uri).resolve()
        )
        image_bytes = path.read_bytes()
        if sha256(image_bytes) != page.image_sha256:
            raise ValueError(f"image checksum changed: {page.id}")
        if source_policy is SourcePolicy.READ2016:
            validate_image(image_bytes, page.width, page.height)
        else:
            with Image.open(path) as image:
                image.load()
                if image.size != (page.width, page.height):
                    raise ValueError(f"image dimensions do not match: {page.id}")
        region_ids = set()
        for region in page.regions or ():
            if region.id in region_ids:
                raise ValueError(f"duplicate region ID: {page.id}/{region.id}")
            region_ids.add(region.id)
            box = region.box
            if (
                not all(math.isfinite(v) for v in (box.x, box.y, box.width, box.height))
                or box.x + box.width > page.width
                or box.y + box.height > page.height
            ):
                raise ValueError(f"ground-truth box out of bounds: {page.id}")
    return pages


class ValidationEvaluator(Protocol):
    """Caller-defined validation policy, versioned in the frozen run config."""

    identifier: str

    def __call__(
        self, examples: tuple[RevealedExample, ...], predictions: tuple[Prediction, ...]
    ) -> dict[str, int | float]: ...


class LocalOracle:
    """Own source labels. Only reveal explicitly requested TRAIN pages.

    This is an API isolation boundary, not a sandbox against a malicious adapter.
    Source and frozen bytes must still match on every restart/round.
    """

    def __init__(self, snapshot: DatasetSnapshot) -> None:
        self.snapshot = snapshot
        data = Path(snapshot.source_manifest).read_bytes()
        frozen = Path(snapshot.frozen_manifest).read_bytes()
        if sha256(data) != snapshot.manifest_sha256 or frozen != data:
            raise ValueError("dataset manifest/ground truth changed")
        source = _read_source(
            data, Path(snapshot.source_manifest).parent, source_policy=snapshot.source_policy
        )
        if any(p.source_provenance != snapshot.source_provenance for p in source):
            raise ValueError("snapshot/source provenance mismatch")
        self._source = {page.id: page for page in source}
        self._pages = {page.id: page for page in snapshot.pages}
        if len(self._pages) != len(snapshot.pages) or set(self._source) != set(self._pages):
            raise ValueError("dataset membership changed")
        for page in snapshot.pages:
            metadata = {
                "id",
                "document_id",
                "split",
                "width",
                "height",
                "image_sha256",
                "source_policy",
            }
            if page.model_dump(include=metadata) != self._source[page.id].model_dump(
                include=metadata
            ):
                raise ValueError("source/frozen page metadata mismatch")
            if sha256(Path(page.image_uri).read_bytes()) != page.image_sha256:
                raise ValueError(f"frozen image changed: {page.id}")

    @staticmethod
    def freeze(
        manifest: Path,
        store: SQLiteStore,
        *,
        source_policy: SourcePolicy = SourcePolicy.KNOWN_DOCUMENT,
    ) -> DatasetSnapshot:
        source_policy = SourcePolicy(source_policy)
        if source_policy is SourcePolicy.READ2016 and (
            manifest.is_symlink() or manifest.parent.is_symlink()
        ):
            raise ValueError("symlinked READ manifest/root")
        manifest = manifest.resolve()
        data = manifest.read_bytes()
        source = _read_source(data, manifest.parent, source_policy=source_policy)
        pages = []
        for page in source:
            original = (manifest.parent / page.image_uri).resolve()
            image_bytes = original.read_bytes()
            if sha256(image_bytes) != page.image_sha256:
                raise ValueError("source image changed during freeze")
            image = store.save_artifact("simulation-images", image_bytes)
            if sha256(image.read_bytes()) != page.image_sha256:
                raise ValueError("persisted image changed during freeze")
            pages.append(
                SimulationPage(
                    **page.model_dump(exclude={"regions", "image_uri", "source_provenance"}),
                    image_uri=str(image.resolve()),
                    source_image=str(original),
                )
            )
        labels = [
            (p.id, None if p.regions is None else [r.model_dump() for r in p.regions])
            for p in source
        ]
        snapshot = DatasetSnapshot(
            source_policy=source_policy,
            source_provenance=source[0].source_provenance if source else None,
            source_manifest=str(manifest),
            frozen_manifest=str(store.save_artifact("simulation-source", data, ".jsonl").resolve()),
            manifest_sha256=sha256(data),
            ground_truth_sha256=sha256(json.dumps(labels, ensure_ascii=False).encode()),
            pages=tuple(pages),
        )
        LocalOracle(snapshot)  # Verify the bytes actually persisted, plus any source mutation.
        return snapshot

    def reveal(self, page_ids: tuple[str, ...]) -> tuple[RevealedExample, ...]:
        if len(set(page_ids)) != len(page_ids):
            raise ValueError("duplicate oracle request")
        examples = []
        for page_id in page_ids:
            page = self._source.get(page_id)
            if page is None or page.split is not Split.TRAIN:
                raise ValueError(f"oracle may only reveal selected train pages: {page_id}")
            if page.regions is None:
                raise ValueError(f"missing ground truth for selected page: {page_id}")
            examples.append(RevealedExample(page=self._pages[page_id], regions=page.regions))
        return tuple(examples)

    def evaluate_validation(
        self,
        predictions: tuple[Prediction, ...],
        evaluator: ValidationEvaluator,
        page_ids: tuple[str, ...] | None = None,
    ) -> dict[str, int | float]:
        """Give only validation truth to the explicit evaluator, never to fit/predict."""
        examples = []
        ids = (
            tuple(p.id for p in self._source.values() if p.split is Split.VALIDATION)
            if page_ids is None
            else page_ids
        )
        if len(set(ids)) != len(ids) or any(
            i not in self._source or self._source[i].split is not Split.VALIDATION for i in ids
        ):
            raise ValueError("invalid validation membership")
        for page in (self._source[i] for i in ids):
            if page.split is not Split.VALIDATION:
                continue
            if page.regions is None:
                raise ValueError(f"missing validation ground truth: {page.id}")
            examples.append(RevealedExample(page=self._pages[page.id], regions=page.regions))
        metrics = evaluator(tuple(examples), predictions)
        if not isinstance(metrics, dict) or any(
            not isinstance(k, str)
            or not k
            or not isinstance(v, (int, float))
            or not math.isfinite(v)
            for k, v in metrics.items()
        ):
            raise ValueError("validation metrics must be named finite numbers")
        return metrics


class SimulationModel(Protocol):
    """A reset-fit synchronous adapter. Predict receives images/metadata, never labels."""

    backend: str
    fit_policy: str

    def fit(
        self,
        examples: tuple[RevealedExample, ...],
        *,
        seed: int,
        experiment_id: str,
        round_number: int,
    ) -> str: ...

    def predict(
        self,
        pages: tuple[SimulationPage, ...],
        *,
        experiment_id: str,
        round_number: int,
        model_id: str,
        purpose: PredictionPurpose = PredictionPurpose.POOL,
    ) -> tuple[Prediction, ...]: ...


class FixtureModel:
    """Deterministic synthetic scores, empty OCR output, no learned recognition.

    fit resets state using only selected examples. Scores are hash-derived test values,
    not probabilities or a proposed generative uncertainty definition.
    """

    backend = "deterministic-fixture-v1"
    fit_policy = "reset-fit-cumulative-v1"

    def fit(
        self,
        examples: tuple[RevealedExample, ...],
        *,
        seed: int,
        experiment_id: str | None = None,
        round_number: int | None = None,
    ) -> str:
        content = [(e.page.id, [r.model_dump() for r in e.regions]) for e in examples]
        return f"fixture-{sha256(json.dumps([seed, content], ensure_ascii=False).encode())}"

    def predict(
        self,
        pages: tuple[SimulationPage, ...],
        *,
        experiment_id: str,
        round_number: int,
        model_id: str,
        purpose: PredictionPurpose = PredictionPurpose.POOL,
    ) -> tuple[Prediction, ...]:
        return tuple(
            Prediction(
                page_id=p.id,
                experiment_id=experiment_id,
                round_number=round_number,
                model_id=model_id,
                purpose=purpose,
                confidence=int(sha256(f"{model_id}:{p.id}".encode())[:8], 16) / 0xFFFFFFFF,
                entropy=int(sha256(f"entropy:{model_id}:{p.id}".encode())[:8], 16) / 0xFFFFFFFF,
            )
            for p in pages
        )


def create_fixture(directory: Path, train_pages: int = 7) -> Path:
    """Generate distinct synthetic images and true labels in a NEW directory."""
    if train_pages < 0:
        raise ValueError("train_pages must be nonnegative")
    directory.mkdir(parents=True, exist_ok=False)
    rows = []
    for i in range(train_pages + 2):
        split = (
            Split.TRAIN
            if i < train_pages
            else (Split.VALIDATION if i == train_pages else Split.TEST)
        )
        image = Image.new("RGB", (160, 40), (255, 255, 255 - i % 256))
        text = f"Synthetic page {i}"
        ImageDraw.Draw(image).text((2, 2), text, fill="black")
        path = directory / f"page-{i}.png"
        image.save(path)
        rows.append(
            {
                "id": f"page-{i}",
                "document_id": f"document-{i}",
                "image_uri": path.name,
                "width": 160,
                "height": 40,
                "split": split.value,
                "image_sha256": sha256(path.read_bytes()),
                "regions": [
                    {
                        "id": "line-1",
                        "box": {"x": 0, "y": 0, "width": 160, "height": 30},
                        "text": text,
                    }
                ],
            }
        )
    manifest = directory / "pages.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return manifest


def runtime_identity() -> tuple[str, tuple[tuple[str, str], ...]]:
    """Record installed versions and source revision when running from a Git checkout."""
    revision = "unknown (installed package or unavailable Git)"
    root = Path(__file__).resolve().parents[3]
    if (root / ".git").exists():
        try:
            revision = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
            dirty = subprocess.run(
                ["git", "-C", str(root), "status", "--porcelain", "--", "src", "pyproject.toml"],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
            if dirty:
                revision += "-dirty"
        except (OSError, subprocess.SubprocessError):
            revision = "unknown (Git inspection failed)"
    versions = (("python", platform.python_version()),) + tuple(
        (package, version(package)) for package in ("pydantic", "pillow")
    )
    return revision, versions


class ContractModel(SimulationModel, Protocol):
    """Explicit metadata shared by real and synthetic contract adapters."""

    kind: RunKind
    real_config: RealOCRConfig
    identity: ExpectedIdentity

    def load_base(self, *, experiment_id: str) -> str: ...


def local_contract_identity() -> tuple[str, str]:
    """Clean code SHA and canonical installed local versions, without GPU inspection."""
    revision, _ = runtime_identity()
    root = Path(__file__).resolve().parents[3]
    if re.fullmatch(r"[0-9a-f]{40}", revision):
        try:
            dirty = subprocess.run(
                ["git", "-C", str(root), "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
            if dirty:
                revision += "-dirty"
        except (OSError, subprocess.SubprocessError):
            revision = "unknown (Git inspection failed)"
    packages = sorted(
        (re.sub(r"[-_.]+", "-", d.metadata["Name"].lower()), d.version)
        for d in distributions()
        if d.metadata["Name"]
    )
    data = {"python": platform.python_version(), "packages": packages}
    return revision, sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode())


def check_local_identity(expected: ExpectedIdentity) -> None:
    revision, dependencies = local_contract_identity()
    if revision != expected.source_sha or dependencies != expected.dependency_sha256:
        raise ValueError("local code/dependency identity differs from frozen expectations")


def check_contract_identity(recipe: RealOCRConfig, model: ContractModel, kind: RunKind) -> None:
    """Only immutable identities/recipes participate; telemetry is deliberately excluded."""
    check_local_identity(recipe.expected_identity)
    if (
        getattr(model, "real_config", None) != recipe
        or getattr(model, "identity", None) != recipe.expected_identity
        or getattr(model, "kind", None) != kind
        or getattr(model, "backend", None) != recipe.backend
        or getattr(model, "fit_policy", None) != "reset-fit-cumulative-v1"
    ):
        raise ValueError("adapter recipe/expected identity mismatch")


def _aligned_validation(
    examples: tuple[RevealedExample, ...], predictions: tuple[Prediction, ...], name: str
) -> list[Prediction]:
    reference_ids = [e.page.id for e in examples]
    ids = [p.page_id for p in predictions]
    if (
        len(set(reference_ids)) != len(reference_ids)
        or len(set(ids)) != len(ids)
        or set(ids) != set(reference_ids)
    ):
        raise ValueError("evaluator page coverage mismatch")
    if any(e.page.split is not Split.VALIDATION for e in examples):
        raise ValueError(f"{name} accepts only validation examples")
    by_id = {p.page_id: p for p in predictions}
    return [by_id[e.page.id] for e in examples]


class PageTextEvaluatorV1:
    """Fixed local engineering text view; never an official benchmark/test policy."""

    identifier = "page-text-nfc-v1"

    def __call__(
        self, examples: tuple[RevealedExample, ...], predictions: tuple[Prediction, ...]
    ) -> dict[str, int | float]:
        ordered = _aligned_validation(examples, predictions, "PageTextEvaluatorV1")
        return page_text_metrics(
            [tuple(r.text for r in e.regions) for e in examples],
            [tuple(r.text for r in p.regions) for p in ordered],
            [p.status for p in ordered],
        )


EVALUATORS = {PageTextEvaluatorV1.identifier: PageTextEvaluatorV1}
