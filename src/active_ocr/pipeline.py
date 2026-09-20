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
    EVALUATORS,
    FixtureModel,
    LocalOracle,
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
    SimulationInitialFit,
    SimulationRound,
    SimulationRun,
    SimulationSelection,
    SourcePolicy,
    Split,
    Strategy,
)


class Pipeline:
    """Coordinate frozen source-label simulation and fixture runs."""

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
    def _revealed(run: SimulationRun) -> tuple[str, ...]:
        """Cumulative labelled pages, including the initial fit before any acquisition."""
        if run.rounds:
            return run.rounds[-1].revealed_ids
        if run.initial_fit is not None:
            return run.initial_fit.revealed_ids
        return ()

    @staticmethod
    def _stage_number(run: SimulationRun) -> int:
        """Round zero is the baseline; the initial fit and each acquired round follow in order."""
        return len(run.rounds) + (run.initial_fit is not None) + 1

    @staticmethod
    def _round_seed(run: SimulationRun) -> int:
        return run.config.seed + len(run.rounds) + (run.initial_fit is not None)

    @staticmethod
    def _initial_pending(run: SimulationRun) -> bool:
        return bool(run.config.initial_batch_size) and run.initial_fit is None

    @staticmethod
    def _simulation_stop(run: SimulationRun) -> str | None:
        if (
            run.kind is not RunKind.FIXTURE
            and not run.config.initial_batch_size
            and run.baseline is None
        ):
            return None
        if run.pending_selection is not None:
            return None
        revealed = Pipeline._revealed(run)
        train = [p for p in run.dataset.pages if p.split is Split.TRAIN]
        if Pipeline._initial_pending(run) and run.config.page_budget and train:
            return None  # An unfinished initial fit is not a completed run.
        if len(revealed) >= run.config.page_budget:
            return "page_budget"
        if len(run.rounds) >= run.config.max_rounds:
            return "round_limit"
        if not any(p.id not in revealed for p in train):
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
            if evaluator is None:
                if run.config.evaluator_id not in EVALUATORS:
                    raise ValueError(f"unsupported evaluator: {run.config.evaluator_id}")
                evaluator = EVALUATORS[run.config.evaluator_id]()
            if run.kind is RunKind.REAL:
                raise ValueError("archived real runs use the page-text recognition pipeline")
        if model.backend != run.config.backend or model.fit_policy != run.config.fit_policy:
            raise ValueError("model backend/fit policy differs from frozen config")
        if (evaluator.identifier if evaluator is not None else None) != run.config.evaluator_id:
            raise ValueError("validation evaluator differs from frozen config")
        validation = self._validation_pages(run.config, run.dataset.pages)
        if run.complete:
            return run
        if (
            run.kind is not RunKind.FIXTURE
            and not run.config.initial_batch_size
            and run.baseline is None
        ):
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
        train = {p.id: p for p in run.dataset.pages if p.split is Split.TRAIN}
        if self._initial_pending(run):
            return self._initial_fit_step(run, model, evaluator, oracle, train, validation)
        previous = run.rounds[-1] if run.rounds else None
        revealed = self._revealed(run)
        remaining = tuple(sorted(set(train) - set(revealed)))
        strategy = run.config.strategy if previous else Strategy.RANDOM
        if previous and strategy is not Strategy.RANDOM:
            self._validate_simulation_predictions(
                previous.predictions, remaining, run, previous.number, previous.model_id
            )
        selected = (
            run.pending_selection.selected_ids
            if run.pending_selection
            else select_pages(
                remaining,
                strategy,
                min(run.config.batch_size, run.config.page_budget - len(revealed)),
                self._round_seed(run),
                predictions=previous.predictions if previous else (),
            )
        )
        run, selected, examples = self._reveal_selection(run, selected, revealed, oracle)
        cumulative = (*revealed, *selected)
        number = self._stage_number(run)
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
        updated = run.model_copy(
            update={"rounds": (*run.rounds, record), "pending_selection": None}
        )
        return self._commit_simulation_step(run, updated, model)

    def _initial_fit_step(
        self, run: SimulationRun, model, evaluator, oracle, train, validation
    ) -> SimulationRun:
        """Select, reveal and fit the seed set once; this is labelled work, not a baseline."""
        size = min(run.config.initial_batch_size, run.config.page_budget, len(train))
        if size <= 0:
            raise ValueError("initial fit requires budget and available TRAIN pages")
        selected = (
            run.pending_selection.selected_ids
            if run.pending_selection
            else select_pages(tuple(sorted(train)), Strategy.RANDOM, size, self._round_seed(run))
        )
        run, selected, examples = self._reveal_selection(run, selected, (), oracle)
        number = self._stage_number(run)
        model_id = model.fit(
            examples, seed=run.config.seed, experiment_id=run.id, round_number=number
        )
        self._validate_simulation_model_id(run, model_id)
        if run.config.real is not None:
            check_contract_identity(run.config.real, model, run.kind)
        validation_predictions = ()
        metrics = None
        if evaluator is not None:
            validation_predictions = self._validate_simulation_predictions(
                tuple(
                    model.predict(
                        validation,
                        experiment_id=run.id,
                        round_number=number,
                        model_id=model_id,
                        purpose=PredictionPurpose.VALIDATION,
                    )
                ),
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
        record = SimulationInitialFit(
            number=number,
            selected_ids=selected,
            revealed_ids=selected,
            labelled_count=len(selected),
            model_id=model_id,
            validation_predictions=validation_predictions,
            validation_metrics=metrics,
            telemetry=getattr(model, "telemetry", None) if run.config.real is not None else None,
        )
        updated = run.model_copy(update={"initial_fit": record, "pending_selection": None})
        return self._commit_simulation_step(run, updated, model)

    def _reveal_selection(self, run, selected, previous, oracle):
        # Preserve historical atomic fixture semantics. New initial-fit runs reserve their
        # selection before revealing, and retain consumed labels even if preflight/fit fails.
        if not run.config.initial_batch_size:
            return run, selected, oracle.reveal((*previous, *selected))
        pending = run.pending_selection
        if pending is None:
            pending = SimulationSelection(number=self._stage_number(run), selected_ids=selected)
            updated = run.model_copy(update={"pending_selection": pending})
            self.store.compare_and_swap(
                "simulation",
                run.id,
                RootModel(run.model_dump(mode="json", exclude_unset=True)),
                updated,
            )
            run = self.get_simulation(run.id)
        selected = pending.selected_ids
        cumulative = (*previous, *selected)
        if pending.number != self._stage_number(run) or (
            pending.revealed_ids and pending.revealed_ids != cumulative
        ):
            raise ValueError("pending selection differs from the current stage")
        examples = oracle.reveal(cumulative)
        if not pending.revealed_ids:
            updated = run.model_copy(
                update={
                    "pending_selection": pending.model_copy(update={"revealed_ids": cumulative})
                }
            )
            self.store.compare_and_swap(
                "simulation",
                run.id,
                RootModel(run.model_dump(mode="json", exclude_unset=True)),
                updated,
            )
            run = self.get_simulation(run.id)
        return run, selected, examples

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
            "localization_pages",
            "box_reference_boxes",
            "box_predicted_boxes",
            "box_matched_boxes",
            "box_precision_defined",
            "box_recall_defined",
            "box_precision",
            "box_recall",
            "box_f1",
            "matched_line_char_edits",
            "matched_line_reference_chars",
            "matched_line_cer",
            "matched_line_coverage",
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
        records = (
            ([run.baseline] if run.baseline is not None else [])
            + ([run.initial_fit] if run.initial_fit is not None else [])
            + list(run.rounds)
        )
        train_pages = sum(p.split is Split.TRAIN for p in run.dataset.pages)
        for record in records:
            is_baseline = isinstance(record, SimulationBaseline)
            is_round = isinstance(record, SimulationRound)
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
                    record.remaining_count
                    if is_round
                    else train_pages - (0 if is_baseline else record.labelled_count),
                    "",
                    record.model_id,
                    json.dumps(
                        [p.model_dump(mode="json") for p in record.predictions] if is_round else []
                    ),
                    run.config.model_dump_json(),
                    run.dataset.manifest_sha256,
                    run.dataset.ground_truth_sha256,
                    run.config.backend,
                    run.config.fit_policy,
                    run.code_revision,
                    json.dumps([p.model_dump(mode="json") for p in record.validation_predictions]),
                    json.dumps(record.validation_metrics),
                    "baseline" if is_baseline else ("round" if is_round else "initial_fit"),
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
