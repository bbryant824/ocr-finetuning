"""The single coordinator for the iterative OCR research pipeline."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from active_ocr.active_learning import select_pages
from active_ocr.config import Settings
from active_ocr.integrations.gpu_client import GPUClient
from active_ocr.integrations.label_studio import LabelStudioClient
from active_ocr.integrations.local_data import load_image_pages
from active_ocr.integrations.storage import SQLiteStore
from active_ocr.models import Annotation, Experiment, Page, Prediction, Split, Stage, Strategy


class Pipeline:
    """Coordinate data, annotation, training, scoring, and selection.

    Integrations are ordinary constructor arguments so tests can replace them
    with small fakes. No abstract port hierarchy is needed at this stage.
    """

    def __init__(
        self,
        settings: Settings,
        store: SQLiteStore,
        label_studio: LabelStudioClient,
        gpu: GPUClient,
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
