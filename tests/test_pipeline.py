from pathlib import Path
from typing import Any

import pytest

from active_ocr.config import Settings
from active_ocr.integrations.storage import SQLiteStore
from active_ocr.models import Annotation, Box, Page, Prediction, Region, Split, Stage, Strategy
from active_ocr.pipeline import Pipeline


class FakeLabelStudio:
    def publish(self, batch_id: str, pages: object, predictions: object) -> str:
        return batch_id

    def is_complete(self, batch_id: str) -> bool:
        return True

    def fetch(self, batch_id: str, pages: dict[str, Page]) -> tuple[Annotation, ...]:
        return tuple(
            Annotation(
                page_id=page_id,
                regions=(Region(id="line", box=Box(x=0, y=0, width=10, height=10), text="x"),),
            )
            for page_id in pages
            if page_id in self.selected
        )


class FakeGPU:
    def __init__(self) -> None:
        self.payloads: dict[str, dict[str, Any]] = {}

    def submit(self, kind: str, payload: dict[str, Any]) -> dict[str, str]:
        self.payloads[kind] = payload
        return {"job_id": kind}

    def job(self, job_id: str) -> dict[str, object]:
        if job_id == "train":
            return {"status": "succeeded", "result": {"model_id": "model-1"}}
        payload = self.payloads["predict"]
        predictions = [
            Prediction(
                page_id=page_id,
                experiment_id=payload["experiment_id"],
                round_number=payload["round_number"],
                model_id=payload["model_id"],
                confidence=0.5,
                entropy=0.5,
            ).model_dump()
            for page_id in payload["pages"]
        ]
        return {"status": "succeeded", "result": {"predictions": predictions}}


def make_pipeline(tmp_path: Path) -> tuple[Pipeline, FakeLabelStudio, FakeGPU]:
    settings = Settings(database=tmp_path / "state.sqlite3", artifacts=tmp_path / "artifacts")
    store = SQLiteStore(settings.database, settings.artifacts)
    label_studio = FakeLabelStudio()
    gpu = FakeGPU()
    pipeline = Pipeline(settings, store, label_studio, gpu)
    pipeline.setup()
    for number in range(1, 4):
        page = Page(
            id=f"page-{number}",
            document_id=f"document-{number}",
            image_uri=f"file:///page-{number}.png",
            width=100,
            height=100,
            split=Split.TRAIN,
        )
        store.save("page", page.id, page)
    return pipeline, label_studio, gpu


def test_pipeline_runs_one_minimal_round(tmp_path: Path) -> None:
    pipeline, label_studio, _ = make_pipeline(tmp_path)
    experiment = pipeline.create_experiment("test", Strategy.ENTROPY, batch_size=2, rounds=1)

    experiment = pipeline.tick(experiment.id)
    assert experiment.stage is Stage.ANNOTATING
    label_studio.selected = set(experiment.current_batch)
    assert experiment.external_batch_id

    assert pipeline.tick(experiment.id).stage is Stage.TRAINING
    assert pipeline.tick(experiment.id).job_id == "train"
    assert pipeline.tick(experiment.id).stage is Stage.SCORING
    assert pipeline.tick(experiment.id).job_id == "predict"
    assert pipeline.tick(experiment.id).stage is Stage.COMPLETE
    predictions = pipeline.store.list("prediction", Prediction)
    assert predictions
    for prediction in predictions:
        assert pipeline.store.load("prediction", prediction.storage_key, Prediction) == prediction


def test_experiments_share_the_same_random_first_batch(tmp_path: Path) -> None:
    pipeline, _, _ = make_pipeline(tmp_path)
    random_run = pipeline.create_experiment("random", Strategy.RANDOM, batch_size=2, seed=7)
    entropy_run = pipeline.create_experiment("entropy", Strategy.ENTROPY, batch_size=2, seed=7)
    assert (
        pipeline.prepare_batch(random_run.id).current_batch
        == pipeline.prepare_batch(entropy_run.id).current_batch
    )


def test_experiment_rejects_zero_overrides(tmp_path: Path) -> None:
    pipeline, _, _ = make_pipeline(tmp_path)
    with pytest.raises(ValueError):
        pipeline.create_experiment("invalid", Strategy.RANDOM, batch_size=0)


def test_predictions_are_isolated_between_experiments(tmp_path: Path) -> None:
    pipeline, _, _ = make_pipeline(tmp_path)
    first = pipeline.create_experiment("first", Strategy.ENTROPY, batch_size=1)
    second = pipeline.create_experiment("second", Strategy.ENTROPY, batch_size=1)
    first = first.model_copy(update={"round_number": 1, "model_id": "model-a"})
    second = second.model_copy(update={"round_number": 1, "model_id": "model-b"})
    pipeline.store.save("experiment", first.id, first)
    pipeline.store.save("experiment", second.id, second)

    for page_number in range(1, 4):
        page_id = f"page-{page_number}"
        first_prediction = Prediction(
            page_id=page_id,
            experiment_id=first.id,
            round_number=1,
            model_id="model-a",
            entropy=0.9 if page_number == 1 else 0.1,
        )
        second_prediction = Prediction(
            page_id=page_id,
            experiment_id=second.id,
            round_number=1,
            model_id="model-b",
            entropy=0.9 if page_number == 2 else 0.1,
        )
        pipeline.store.save("prediction", f"{first.id}:1:{page_id}", first_prediction)
        pipeline.store.save("prediction", f"{second.id}:1:{page_id}", second_prediction)

    wrong_model = Prediction(
        page_id="page-2",
        experiment_id=first.id,
        round_number=1,
        model_id="old-model",
        entropy=1.0,
    )
    pipeline.store.save("prediction", f"{first.id}:1:page-2:old", wrong_model)

    assert pipeline.prepare_batch(first.id).current_batch == ("page-1",)
    assert pipeline.prepare_batch(second.id).current_batch == ("page-2",)


def test_training_uses_only_experiment_labelled_pages(tmp_path: Path) -> None:
    pipeline, _, gpu = make_pipeline(tmp_path)
    experiment = pipeline.create_experiment("first", Strategy.RANDOM)
    experiment = experiment.model_copy(
        update={"stage": Stage.TRAINING, "labelled_page_ids": ("page-1",)}
    )
    pipeline.store.save("experiment", experiment.id, experiment)
    for page_id in ("page-1", "page-2"):
        pipeline.store.save(
            "annotation",
            page_id,
            Annotation(
                page_id=page_id,
                regions=(Region(id="line", box=Box(x=0, y=0, width=10, height=10), text=page_id),),
            ),
        )

    pipeline.submit_training(experiment.id)
    sent_page_ids = {item["page_id"] for item in gpu.payloads["train"]["annotations"]}
    assert sent_page_ids == {"page-1"}


def test_annotation_batch_must_match_selected_pages(tmp_path: Path) -> None:
    pipeline, label_studio, _ = make_pipeline(tmp_path)
    experiment = pipeline.create_experiment("test", Strategy.RANDOM, batch_size=2)
    experiment = pipeline.tick(experiment.id)
    label_studio.selected = {experiment.current_batch[0]}
    with pytest.raises(ValueError, match="does not match selection"):
        pipeline.tick(experiment.id)


def test_scoring_rejects_prediction_owned_by_another_experiment(tmp_path: Path) -> None:
    pipeline, _, gpu = make_pipeline(tmp_path)
    experiment = pipeline.create_experiment("test", Strategy.RANDOM)
    experiment = experiment.model_copy(
        update={"stage": Stage.SCORING, "job_id": "predict", "model_id": "model-1"}
    )
    pipeline.store.save("experiment", experiment.id, experiment)
    gpu.payloads["predict"] = {
        "experiment_id": "another-experiment",
        "round_number": 1,
        "model_id": "model-1",
        "pages": ["page-1", "page-2", "page-3"],
    }

    with pytest.raises(ValueError, match="ownership"):
        pipeline.poll_job(experiment.id)


def test_scoring_requires_a_trained_model(tmp_path: Path) -> None:
    pipeline, _, _ = make_pipeline(tmp_path)
    experiment = pipeline.create_experiment("test", Strategy.RANDOM)
    experiment = experiment.model_copy(update={"stage": Stage.SCORING})
    pipeline.store.save("experiment", experiment.id, experiment)

    with pytest.raises(ValueError, match="no trained model"):
        pipeline.submit_scoring(experiment.id)
