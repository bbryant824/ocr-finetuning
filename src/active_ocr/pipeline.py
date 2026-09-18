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
    Prediction,
    PredictionPurpose,
    PredictionStatus,
    RunKind,
    SimulationBaseline,
    SimulationConfig,
    SimulationRound,
    SimulationRun,
    SourcePolicy,
    Split,
    Strategy,
)


class Pipeline:
    """Coordinate frozen source-label simulation with fixture or real OCR models."""

    def __init__(self, store: SQLiteStore) -> None:
        self.store = store

    @classmethod
    def for_simulation(cls, directory: Path) -> Pipeline:
        """Open the local simulation store without constructing model/provider clients."""
        store = SQLiteStore(directory / "simulation.sqlite3", directory / "artifacts")
        store.initialize()
        return cls(store)

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
        snapshot = LocalOracle.freeze(manifest, self.store, source_policy=config.source_policy)
        if config.real is not None and not any(p.split is Split.VALIDATION for p in snapshot.pages):
            raise ValueError("real contracts require nonempty validation pages")
        self._validation_pages(config, snapshot.pages)
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
            validate_layout = (
                run.kind is not RunKind.FIXTURE and prediction.status is PredictionStatus.OK
            )
            if validate_layout:
                region_ids = [r.id for r in prediction.regions]
                if len(set(region_ids)) != len(region_ids):
                    raise ValueError("duplicate predicted region ID")
            for region in prediction.regions:
                box = region.box
                if not all(math.isfinite(v) for v in (box.x, box.y, box.width, box.height)):
                    raise ValueError("predicted geometry must be finite for every status")
                if validate_layout:
                    page = pages[prediction.page_id]
                    if box.x + box.width > page.width or box.y + box.height > page.height:
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
                from active_ocr.integrations.modal_model import ModalModel

                if type(model) is not ModalModel:
                    raise ValueError("production real execution requires the Modal adapter")
                model.check_run(run, self.store)
        if model.backend != run.config.backend or model.fit_policy != run.config.fit_policy:
            raise ValueError("model backend/fit policy differs from frozen config")
        if (evaluator.identifier if evaluator is not None else None) != run.config.evaluator_id:
            raise ValueError("validation evaluator differs from frozen config")
        validation = self._validation_pages(run.config, run.dataset.pages)
        if run.complete:
            return run
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
                validation_metrics=oracle.evaluate_validation(
                    predictions, evaluator, tuple(p.id for p in validation)
                ),
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
            metrics = oracle.evaluate_validation(
                validation_predictions, evaluator, tuple(p.id for p in validation)
            )
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

    @staticmethod
    def _validation_pages(config, pages):
        validation = {p.id: p for p in pages if p.split is Split.VALIDATION}
        ids = config.validation_page_ids
        if ids is None:
            return tuple(validation.values())
        if not ids or len(set(ids)) != len(ids) or any(i not in validation for i in ids):
            raise ValueError("validation subset must contain only frozen validation pages")
        return tuple(validation[i] for i in ids)

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
        validation_ids = [p.id for p in self._validation_pages(run.config, run.dataset.pages)]
        (directory / "results.json").write_text(run.model_dump_json(indent=2), encoding="utf-8")
        (directory / "validation.json").write_text(
            json.dumps(dict(page_ids=validation_ids, page_count=len(validation_ids)), indent=2),
            encoding="utf-8",
        )
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
                "validation_page_ids",
                "validation_page_count",
                "source_policy",
                "document_grouping",
                "engineering_only",
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
                    json.dumps(validation_ids),
                    len(validation_ids),
                    run.config.source_policy,
                    "unknown" if run.config.source_policy is SourcePolicy.READ2016 else "known",
                    "true" if run.config.source_policy is SourcePolicy.READ2016 else "false",
                    *(metrics.get(key, "") for key in metric_columns),
                ]
            )
        (directory / "rounds.csv").write_text(output.getvalue(), encoding="utf-8")
