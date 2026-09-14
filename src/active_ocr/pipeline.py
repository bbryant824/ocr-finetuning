"""The single coordinator for the iterative OCR research pipeline."""

from __future__ import annotations

import csv
import io
import json
import math
import re
from pathlib import Path
from uuid import uuid4

from pydantic import RootModel

from active_ocr.active_learning import select_pages
from active_ocr.config import Settings
from active_ocr.integrations.gpu_client import GPUClient
from active_ocr.integrations.label_studio import LabelStudioClient
from active_ocr.integrations.local_data import load_image_pages
from active_ocr.integrations.simulation import (
    FixtureModel,
    LocalOracle,
    PageTextEvaluatorV1,
    SimulationModel,
    ValidationEvaluator,
    check_contract_identity,
    check_local_identity,
    runtime_identity,
)
from active_ocr.integrations.storage import SQLiteStore
from active_ocr.models import (
    Annotation,
    Experiment,
    Page,
    Prediction,
    PredictionPurpose,
    PredictionStatus,
    RunKind,
    SimulationBaseline,
    SimulationConfig,
    SimulationRound,
    SimulationRun,
    Split,
    Stage,
    Strategy,
)


class Pipeline:
    """Coordinate data, annotation, training, scoring, and selection.

    Integrations are ordinary constructor arguments so tests can replace them
    with small fakes. No abstract port hierarchy is needed at this stage.
    """

    def __init__(
        self,
        settings: Settings,
        store: SQLiteStore,
        label_studio: LabelStudioClient | None,
        gpu: GPUClient | None,
    ) -> None:
        self.settings = settings
        self.store = store
        self.label_studio = label_studio
        self.gpu = gpu

    @classmethod
    def from_settings(cls, settings: Settings) -> Pipeline:
        """Construct the default local pipeline without contacting services."""

        return cls(
            settings,
            SQLiteStore(settings.database, settings.artifacts),
            LabelStudioClient(
                settings.label_studio.url,
                settings.label_studio.token,
                settings.label_studio.project_id,
            ),
            GPUClient(settings.gpu.url, settings.gpu.token),
        )

    def setup(self) -> None:
        self.store.initialize()

    def import_pages(self, manifest: Path) -> tuple[Page, ...]:
        """Import a page manifest and persist its document-level splits."""

        ratios = self.settings.experiment
        pages = load_image_pages(
            manifest,
            document_root=self.settings.documents,
            train_ratio=ratios.train_ratio,
            validation_ratio=ratios.validation_ratio,
            test_ratio=ratios.test_ratio,
        )
        for page in pages:
            self.store.save("page", page.id, page)
        return pages

    def create_experiment(
        self,
        name: str,
        strategy: Strategy,
        *,
        batch_size: int | None = None,
        rounds: int | None = None,
        seed: int | None = None,
    ) -> Experiment:
        """Create one strategy run; repeat this for each baseline or algorithm."""

        defaults = self.settings.experiment
        experiment = Experiment(
            id=uuid4().hex,
            name=name,
            strategy=strategy,
            batch_size=defaults.batch_size if batch_size is None else batch_size,
            max_rounds=defaults.rounds if rounds is None else rounds,
            seed=defaults.seed if seed is None else seed,
        )
        self.store.save("experiment", experiment.id, experiment)
        return experiment

    def get_experiment(self, experiment_id: str) -> Experiment:
        experiment = self.store.load("experiment", experiment_id, Experiment)
        if experiment is None:
            raise LookupError(f"unknown experiment: {experiment_id}")
        return experiment

    def prepare_batch(self, experiment_id: str) -> Experiment:
        """Select the next train-pool batch and move the run to annotation."""

        experiment = self.get_experiment(experiment_id)
        if experiment.stage is not Stage.READY:
            raise ValueError(f"experiment is not ready: {experiment.stage}")
        pages = self.store.list("page", Page)
        train_ids = [page.id for page in pages if page.split is Split.TRAIN]
        predictions = self._predictions_for(experiment)

        # Every strategy gets the same deterministic random first batch.
        strategy = Strategy.RANDOM if experiment.round_number == 0 else experiment.strategy
        selected = select_pages(
            train_ids,
            strategy,
            experiment.batch_size,
            experiment.seed + experiment.round_number,
            excluded=experiment.labelled_page_ids,
            predictions=predictions,
        )
        if not selected:
            return self._save(experiment.model_copy(update={"stage": Stage.COMPLETE}))
        return self._save(
            experiment.model_copy(update={"stage": Stage.ANNOTATING, "current_batch": selected})
        )

    def publish_batch(self, experiment_id: str) -> Experiment:
        """Send the prepared batch and optional OCR pre-annotations to Label Studio."""

        experiment = self.get_experiment(experiment_id)
        if experiment.stage is not Stage.ANNOTATING or not experiment.current_batch:
            raise ValueError("experiment has no prepared annotation batch")
        pages = {page.id: page for page in self.store.list("page", Page)}
        selected_pages = tuple(pages[page_id] for page_id in experiment.current_batch)
        selected_ids = set(experiment.current_batch)
        predictions = tuple(
            prediction
            for prediction in self._predictions_for(experiment)
            if prediction.page_id in selected_ids
        )
        batch_id = f"{experiment.id}-round-{experiment.round_number + 1}"
        external_id = self.label_studio.publish(batch_id, selected_pages, predictions)
        return self._save(experiment.model_copy(update={"external_batch_id": external_id}))

    def collect_annotations(self, experiment_id: str) -> Experiment:
        """Store a completed Label Studio batch and make it available for training."""

        experiment = self.get_experiment(experiment_id)
        if not experiment.external_batch_id:
            raise ValueError("annotation batch has not been published")
        if not self.label_studio.is_complete(experiment.external_batch_id):
            return experiment
        pages = {page.id: page for page in self.store.list("page", Page)}
        annotations = self.label_studio.fetch(experiment.external_batch_id, pages)
        annotation_ids = [annotation.page_id for annotation in annotations]
        expected_ids = set(experiment.current_batch)
        if len(set(annotation_ids)) != len(annotation_ids):
            raise ValueError("Label Studio returned duplicate page annotations")
        if set(annotation_ids) != expected_ids:
            missing = sorted(expected_ids - set(annotation_ids))
            unexpected = sorted(set(annotation_ids) - expected_ids)
            raise ValueError(
                "annotation batch does not match selection; "
                f"missing={missing}, unexpected={unexpected}"
            )
        for annotation in annotations:
            self.store.save("annotation", annotation.page_id, annotation)
        labelled = tuple(dict.fromkeys((*experiment.labelled_page_ids, *experiment.current_batch)))
        return self._save(
            experiment.model_copy(
                update={"labelled_page_ids": labelled, "stage": Stage.TRAINING, "job_id": None}
            )
        )

    def submit_training(self, experiment_id: str) -> Experiment:
        """Submit the current labelled set to the remote training endpoint."""

        experiment = self.get_experiment(experiment_id)
        if experiment.stage is not Stage.TRAINING or experiment.job_id:
            raise ValueError("experiment is not ready to submit training")
        annotations_by_page = {
            annotation.page_id: annotation
            for annotation in self.store.list("annotation", Annotation)
        }
        missing = [
            page_id
            for page_id in experiment.labelled_page_ids
            if page_id not in annotations_by_page
        ]
        if missing:
            raise ValueError(f"missing training annotations for pages: {missing}")
        annotations = [annotations_by_page[page_id] for page_id in experiment.labelled_page_ids]
        response = self.gpu.submit(
            "train",
            {
                "experiment_id": experiment.id,
                "round_number": experiment.round_number + 1,
                "model": self.settings.gpu.model,
                "annotations": [item.model_dump(mode="json") for item in annotations],
                "config": self.settings.training,
            },
        )
        return self._save(experiment.model_copy(update={"job_id": str(response["job_id"])}))

    def submit_scoring(self, experiment_id: str) -> Experiment:
        """Request predictions for the remaining unlabelled training pool."""

        experiment = self.get_experiment(experiment_id)
        if experiment.stage is not Stage.SCORING or experiment.job_id:
            raise ValueError("experiment is not ready to submit scoring")
        if not experiment.model_id:
            raise ValueError("experiment has no trained model to score with")
        pages = self.store.list("page", Page)
        page_ids = [
            page.id
            for page in pages
            if page.split is Split.TRAIN and page.id not in experiment.labelled_page_ids
        ]
        response = self.gpu.submit(
            "predict",
            {
                "experiment_id": experiment.id,
                "round_number": experiment.round_number + 1,
                "model_id": experiment.model_id,
                "pages": page_ids,
            },
        )
        return self._save(experiment.model_copy(update={"job_id": str(response["job_id"])}))

    def poll_job(self, experiment_id: str) -> Experiment:
        """Apply a finished training or scoring result; otherwise leave state unchanged."""

        experiment = self.get_experiment(experiment_id)
        if not experiment.job_id or experiment.stage not in {Stage.TRAINING, Stage.SCORING}:
            raise ValueError("experiment has no active model job")
        job = self.gpu.job(experiment.job_id)
        status = job.get("status")
        if status in {"queued", "running"}:
            return experiment
        if status == "failed":
            return self._save(
                experiment.model_copy(
                    update={"stage": Stage.ERROR, "error": str(job.get("error", "GPU job failed"))}
                )
            )
        if status != "succeeded":
            raise ValueError(f"unknown GPU job status: {status}")

        result = job.get("result", {})
        if not isinstance(result, dict):
            raise ValueError("GPU job result must be an object")
        if experiment.stage is Stage.TRAINING:
            model_id = result.get("model_id")
            if not isinstance(model_id, str) or not model_id:
                raise ValueError("training result is missing model_id")
            return self._save(
                experiment.model_copy(
                    update={
                        "stage": Stage.SCORING,
                        "job_id": None,
                        "model_id": model_id,
                    }
                )
            )

        target_round = experiment.round_number + 1
        predictions = tuple(
            Prediction.model_validate(raw_prediction)
            for raw_prediction in result.get("predictions", [])
        )
        expected_page_ids = {
            page.id
            for page in self.store.list("page", Page)
            if page.split is Split.TRAIN and page.id not in experiment.labelled_page_ids
        }
        prediction_page_ids = [prediction.page_id for prediction in predictions]
        if len(set(prediction_page_ids)) != len(prediction_page_ids):
            raise ValueError("GPU returned duplicate predictions for a page")
        if set(prediction_page_ids) != expected_page_ids:
            raise ValueError("GPU predictions do not cover the current unlabelled training pool")
        if any(
            prediction.experiment_id != experiment.id
            or prediction.round_number != target_round
            or prediction.model_id != experiment.model_id
            for prediction in predictions
        ):
            raise ValueError("GPU prediction ownership does not match the current experiment")
        for prediction in predictions:
            self.store.save("prediction", prediction.storage_key, prediction)
        next_round = target_round
        next_stage = Stage.COMPLETE if next_round >= experiment.max_rounds else Stage.READY
        return self._save(
            experiment.model_copy(
                update={
                    "round_number": next_round,
                    "stage": next_stage,
                    "current_batch": (),
                    "external_batch_id": None,
                    "job_id": None,
                }
            )
        )

    def tick(self, experiment_id: str) -> Experiment:
        """Advance an experiment by at most one safe external or state-changing step."""

        experiment = self.get_experiment(experiment_id)
        if experiment.stage is Stage.READY:
            experiment = self.prepare_batch(experiment_id)
            return (
                experiment
                if experiment.stage is Stage.COMPLETE
                else self.publish_batch(experiment_id)
            )
        if experiment.stage is Stage.ANNOTATING:
            return self.collect_annotations(experiment_id)
        if experiment.stage is Stage.TRAINING:
            return (
                self.poll_job(experiment_id)
                if experiment.job_id
                else self.submit_training(experiment_id)
            )
        if experiment.stage is Stage.SCORING:
            return (
                self.poll_job(experiment_id)
                if experiment.job_id
                else self.submit_scoring(experiment_id)
            )
        return experiment

    def _save(self, experiment: Experiment) -> Experiment:
        self.store.save("experiment", experiment.id, experiment)
        return experiment

    def _predictions_for(self, experiment: Experiment) -> tuple[Prediction, ...]:
        """Return only predictions created by this run's current model."""

        if experiment.round_number == 0 or not experiment.model_id:
            return ()
        return tuple(
            prediction
            for prediction in self.store.list("prediction", Prediction)
            if prediction.experiment_id == experiment.id
            and prediction.round_number == experiment.round_number
            and prediction.model_id == experiment.model_id
        )

    @classmethod
    def for_simulation(cls, directory: Path) -> Pipeline:
        """Use a separate local database; construct no annotation or GPU clients."""
        settings = Settings(
            database=directory / "simulation.sqlite3", artifacts=directory / "artifacts"
        )
        pipeline = cls(settings, SQLiteStore(settings.database, settings.artifacts), None, None)
        pipeline.setup()
        return pipeline

    def create_simulation(
        self, manifest: Path, config: SimulationConfig, *, kind: RunKind = RunKind.FIXTURE
    ) -> SimulationRun:
        """Freeze source membership, labels, image bytes and settings for a new UUID run."""
        config = SimulationConfig.model_validate(config.model_dump())
        kind = RunKind(kind)
        if kind is RunKind.FIXTURE:
            if config.real is not None or config.backend != FixtureModel.backend:
                raise ValueError("fixture kind requires fixture backend and no real recipe")
        elif config.real is None:
            raise ValueError("real/contract-test kind requires an explicit real recipe")
        if config.real is not None:
            check_local_identity(config.real.expected_identity)
        snapshot = LocalOracle.freeze(manifest, self.store)
        if config.real is not None and not any(p.split is Split.VALIDATION for p in snapshot.pages):
            raise ValueError("real contracts require nonempty validation pages")
        LocalOracle(snapshot)
        revision, versions = runtime_identity()
        run = SimulationRun(
            id=uuid4().hex,
            kind=kind,
            config=config,
            dataset=snapshot,
            code_revision=revision,
            software_versions=versions,
        )
        reason = self._simulation_stop(run)
        if reason:
            run = run.model_copy(update={"complete": True, "stop_reason": reason})
        self.store.compare_and_swap("simulation", run.id, None, run)
        return run

    def get_simulation(self, run_id: str) -> SimulationRun:
        run = self.store.load("simulation", run_id, SimulationRun)
        if run is None:
            raise LookupError(f"unknown simulation: {run_id}")
        return run

    @staticmethod
    def _simulation_stop(run: SimulationRun) -> str | None:
        if run.kind is not RunKind.FIXTURE and run.baseline is None:
            return None
        revealed = run.rounds[-1].revealed_ids if run.rounds else ()
        if len(revealed) >= run.config.page_budget:
            return "page_budget"
        if len(run.rounds) >= run.config.max_rounds:
            return "round_limit"
        if not any(p.split is Split.TRAIN and p.id not in revealed for p in run.dataset.pages):
            return "pool_exhausted"
        return None

    @staticmethod
    def _validate_simulation_model_id(run: SimulationRun, model_id: str) -> None:
        if not isinstance(model_id, str) or not model_id:
            raise ValueError("model must return a nonempty model identifier")
        if run.kind is not RunKind.FIXTURE and not re.fullmatch(
            r"checkpoint:sha256:[0-9a-f]{64}", model_id
        ):
            raise ValueError("real contract requires an immutable checkpoint reference")

    @staticmethod
    def _validate_simulation_predictions(
        predictions: tuple[Prediction, ...],
        page_ids: tuple[str, ...],
        run: SimulationRun,
        number: int,
        model_id: str,
        *,
        require_scores: bool = True,
        purpose: PredictionPurpose = PredictionPurpose.POOL,
    ) -> tuple[Prediction, ...]:
        predictions = tuple(
            Prediction.model_validate(p.model_dump() if isinstance(p, Prediction) else p)
            for p in predictions
        )
        ids = [p.page_id for p in predictions]
        if len(set(ids)) != len(ids) or set(ids) != set(page_ids):
            raise ValueError("predictions must cover requested pages exactly, without duplicates")
        if any(
            p.experiment_id != run.id
            or p.round_number != number
            or p.model_id != model_id
            or p.purpose is not purpose
            for p in predictions
        ):
            raise ValueError("prediction ownership does not match run/round/model/purpose")
        pages = {p.id: p for p in run.dataset.pages}
        for prediction in predictions:
            if run.kind is not RunKind.FIXTURE and prediction.status is PredictionStatus.OK:
                page = pages[prediction.page_id]
                region_ids = [r.id for r in prediction.regions]
                if len(set(region_ids)) != len(region_ids):
                    raise ValueError("duplicate predicted region ID")
                for region in prediction.regions:
                    box = region.box
                    if (
                        not all(math.isfinite(v) for v in (box.x, box.y, box.width, box.height))
                        or box.x + box.width > page.width
                        or box.y + box.height > page.height
                    ):
                        raise ValueError("predicted region out of original pixel bounds")
            if require_scores:
                if run.config.strategy is Strategy.ENTROPY and prediction.entropy is None:
                    raise ValueError("missing entropy")
                if (
                    run.config.strategy is Strategy.LEAST_CONFIDENCE
                    and prediction.confidence is None
                ):
                    raise ValueError("missing confidence")
        by_id = {p.page_id: p for p in predictions}
        return tuple(by_id[page_id] for page_id in page_ids)

    def step_simulation(
        self,
        run_id: str,
        model: SimulationModel | None = None,
        *,
        config: SimulationConfig | None = None,
        evaluator: ValidationEvaluator | None = None,
    ) -> SimulationRun:
        """Commit one whole round or nothing. Failed fits/predictions can be retried.

        Adapters must reset on fit. Concurrent attempts may compute, but only one
        can commit; the loser reloads. No external side-effect exactly-once promise.
        """
        run = self.get_simulation(run_id)
        if config is not None and config != run.config:
            raise ValueError("cannot change frozen run config")
        oracle = LocalOracle(run.dataset)  # Also verifies inputs on completed-run resume.
        if run.kind is RunKind.FIXTURE:
            model = FixtureModel() if model is None else model
        else:
            if model is None:
                raise ValueError("real contracts require an explicit adapter")
            if getattr(model, "kind", None) != run.kind:
                raise ValueError("adapter kind differs from frozen run kind")
            check_contract_identity(run.config.real, model, run.kind)
            evaluator = PageTextEvaluatorV1() if evaluator is None else evaluator
            if run.kind is RunKind.REAL:
                raise NotImplementedError("production real execution is not available")
        if model.backend != run.config.backend or model.fit_policy != run.config.fit_policy:
            raise ValueError("model backend/fit policy differs from frozen config")
        if (evaluator.identifier if evaluator is not None else None) != run.config.evaluator_id:
            raise ValueError("validation evaluator differs from frozen config")
        if run.complete:
            return run
        validation = tuple(p for p in run.dataset.pages if p.split is Split.VALIDATION)
        if run.kind is not RunKind.FIXTURE and run.baseline is None:
            model_id = model.load_base(experiment_id=run.id)
            self._validate_simulation_model_id(run, model_id)
            check_contract_identity(run.config.real, model, run.kind)
            predictions = self._validate_simulation_predictions(
                tuple(
                    model.predict(
                        validation,
                        experiment_id=run.id,
                        round_number=0,
                        model_id=model_id,
                        purpose=PredictionPurpose.BASELINE_VALIDATION,
                    )
                ),
                tuple(p.id for p in validation),
                run,
                0,
                model_id,
                require_scores=False,
                purpose=PredictionPurpose.BASELINE_VALIDATION,
            )
            baseline = SimulationBaseline(
                model_id=model_id,
                validation_predictions=predictions,
                validation_metrics=oracle.evaluate_validation(predictions, evaluator),
                telemetry=getattr(model, "telemetry", None),
            )
            updated = run.model_copy(update={"baseline": baseline})
            return self._commit_simulation_step(run, updated, model)
        previous = run.rounds[-1] if run.rounds else None
        revealed = previous.revealed_ids if previous else ()
        train = {p.id: p for p in run.dataset.pages if p.split is Split.TRAIN}
        remaining = tuple(sorted(set(train) - set(revealed)))
        strategy = run.config.strategy if previous else Strategy.RANDOM
        if previous and strategy is not Strategy.RANDOM:
            self._validate_simulation_predictions(
                previous.predictions, remaining, run, previous.number, previous.model_id
            )
        selected = select_pages(
            remaining,
            strategy,
            min(run.config.batch_size, run.config.page_budget - len(revealed)),
            run.config.seed + len(run.rounds),
            predictions=previous.predictions if previous else (),
        )
        cumulative = (*revealed, *selected)
        examples = oracle.reveal(cumulative)
        number = len(run.rounds) + 1
        model_id = model.fit(
            examples, seed=run.config.seed, experiment_id=run.id, round_number=number
        )
        self._validate_simulation_model_id(run, model_id)
        if run.config.real is not None:
            check_contract_identity(run.config.real, model, run.kind)
        pool = tuple(train[p] for p in remaining if p not in selected)
        predictions = ()
        if pool and (run.config.strategy is not Strategy.RANDOM or run.config.report_predictions):
            predictions = tuple(
                model.predict(
                    pool,
                    experiment_id=run.id,
                    round_number=number,
                    model_id=model_id,
                    purpose=PredictionPurpose.POOL,
                )
            )
            predictions = self._validate_simulation_predictions(
                predictions, tuple(p.id for p in pool), run, number, model_id
            )
        validation_predictions = ()
        metrics = None
        if evaluator is not None:
            validation_predictions = tuple(
                model.predict(
                    validation,
                    experiment_id=run.id,
                    round_number=number,
                    model_id=model_id,
                    purpose=PredictionPurpose.VALIDATION,
                )
            )
            validation_predictions = self._validate_simulation_predictions(
                validation_predictions,
                tuple(p.id for p in validation),
                run,
                number,
                model_id,
                require_scores=False,
                purpose=PredictionPurpose.VALIDATION,
            )
            metrics = oracle.evaluate_validation(validation_predictions, evaluator)
        record = SimulationRound(
            number=number,
            selected_ids=selected,
            revealed_ids=cumulative,
            labelled_count=len(cumulative),
            remaining_count=len(pool),
            model_id=model_id,
            predictions=predictions,
            validation_predictions=validation_predictions,
            validation_metrics=metrics,
            telemetry=getattr(model, "telemetry", None) if run.config.real is not None else None,
        )
        updated = run.model_copy(update={"rounds": (*run.rounds, record)})
        return self._commit_simulation_step(run, updated, model)

    def _commit_simulation_step(
        self, run: SimulationRun, updated: SimulationRun, model: SimulationModel
    ) -> SimulationRun:
        reason = self._simulation_stop(updated)
        if reason:
            updated = updated.model_copy(update={"complete": True, "stop_reason": reason})
        # Catch mutations during synchronous model work before making labels/budget durable.
        LocalOracle(run.dataset)
        if run.config.real is not None:
            check_contract_identity(run.config.real, model, run.kind)
        # Old rows lack newer defaults. Compare the originally present fields, not
        # a reserialization that adds purpose/status/baseline to historical JSON.
        expected = RootModel(run.model_dump(mode="json", exclude_unset=True))
        self.store.compare_and_swap("simulation", run.id, expected, updated)
        return updated

    def run_simulation(
        self,
        run_id: str,
        model: SimulationModel | None = None,
        *,
        evaluator: ValidationEvaluator | None = None,
    ) -> SimulationRun:
        """Run or resume sequential rounds; the adapter resets on each cumulative fit."""
        run = self.step_simulation(run_id, model, evaluator=evaluator)
        while not run.complete:
            run = self.step_simulation(run_id, model, evaluator=evaluator)
        return run

    def export_simulation(self, run_id: str, directory: Path) -> None:
        """Export one consistent committed snapshot; refuse to overwrite prior results."""
        run = self.get_simulation(run_id)
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "results.json").write_text(run.model_dump_json(indent=2), encoding="utf-8")
        output = io.StringIO()
        writer = csv.writer(output)
        metric_columns = (
            "pages",
            "char_edits",
            "reference_chars",
            "word_edits",
            "reference_words",
            "cer_defined",
            "wer_defined",
            "invalid_output_pages",
            "truncated_pages",
            "refusal_pages",
            "failed_pages",
            "cer",
            "wer",
        )
        writer.writerow(
            [
                "run_id",
                "kind",
                "strategy",
                "seed",
                "round",
                "selected_ids",
                "revealed_ids",
                "labelled_pages",
                "remaining_pages",
                "annotation_seconds",
                "model_id",
                "predictions",
                "config",
                "manifest_sha256",
                "ground_truth_sha256",
                "backend",
                "fit_policy",
                "code_revision",
                "validation_predictions",
                "validation_metrics",
                "record_type",
                "purpose",
                "validation_statuses",
                *metric_columns,
            ]
        )
        records = ([run.baseline] if run.baseline is not None else []) + list(run.rounds)
        for record in records:
            is_baseline = isinstance(record, SimulationBaseline)
            metrics = record.validation_metrics or {}
            writer.writerow(
                [
                    run.id,
                    run.kind,
                    run.config.strategy,
                    run.config.seed,
                    0 if is_baseline else record.number,
                    json.dumps(()) if is_baseline else json.dumps(record.selected_ids),
                    json.dumps(()) if is_baseline else json.dumps(record.revealed_ids),
                    record.labelled_count,
                    sum(p.split is Split.TRAIN for p in run.dataset.pages)
                    if is_baseline
                    else record.remaining_count,
                    "",
                    record.model_id,
                    json.dumps(
                        []
                        if is_baseline
                        else [p.model_dump(mode="json") for p in record.predictions]
                    ),
                    run.config.model_dump_json(),
                    run.dataset.manifest_sha256,
                    run.dataset.ground_truth_sha256,
                    run.config.backend,
                    run.config.fit_policy,
                    run.code_revision,
                    json.dumps([p.model_dump(mode="json") for p in record.validation_predictions]),
                    json.dumps(record.validation_metrics),
                    "baseline" if is_baseline else "round",
                    PredictionPurpose.BASELINE_VALIDATION
                    if is_baseline
                    else (PredictionPurpose.VALIDATION if record.validation_predictions else ""),
                    json.dumps({p.page_id: p.status for p in record.validation_predictions}),
                    *(metrics.get(key, "") for key in metric_columns),
                ]
            )
        (directory / "rounds.csv").write_text(output.getvalue(), encoding="utf-8")
